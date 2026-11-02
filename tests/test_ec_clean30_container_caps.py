"""CPU-only checks for fail-closed delegated cgroup containment."""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from ec_clean30 import container_caps as caps


@pytest.fixture
def kernel(tmp_path, monkeypatch):
    """An explicit kernel-interface fake; this is not provider qualification."""
    root = tmp_path / "delegated"
    root.mkdir()
    (root / "cgroup.type").write_text("domain\n")
    (root / "cgroup.subtree_control").write_text("cpu memory\n")
    monkeypatch.setattr(caps, "_require_cgroup_mount", lambda path: None)
    mkdir, rmdir = Path.mkdir, Path.rmdir
    created, children, killed = [], [], []

    def make(path, *args, **kwargs):
        result = mkdir(path, *args, **kwargs)
        if path.parent == root and path.name.startswith("deepmzyme-g2-"):
            created.append(path)
            for name, value in {"cgroup.type": "domain", "cgroup.procs": "", "cgroup.kill": "",
                                "cgroup.events": "populated 0\nfrozen 0", "cpu.max": "max 100000",
                                "cpu.max.burst": "0", "memory.max": "max", "memory.swap.max": "max",
                                "memory.oom.group": "0", "memory.swap.current": "0", "memory.current": "0",
                                "memory.events": "oom 0\noom_kill 0", "cpu.stat": "nr_throttled 0\nthrottled_usec 0"}.items():
                (path / name).write_text(value + "\n")
        return result

    def remove(path):
        if path in created:
            for item in path.iterdir():
                if item.is_file():
                    item.unlink()
        return rmdir(path)

    def write(path, value):
        assert path.exists(), "A controller must exist before a write."
        path.write_text(value + "\n")
        if path.name == "cgroup.kill":
            killed.append(path.parent)
            (path.parent / "cgroup.procs").write_text("")
            (path.parent / "cgroup.events").write_text("populated 0\nfrozen 0\n")

    class Child:
        def __init__(self, argv, *, preexec_fn, **kwargs):
            self.scope = preexec_fn.__self__
            self.argv, self.returncode = argv, 0
            assert (self.scope.path / "memory.swap.max").read_text().strip() == "0"
            assert (self.scope.path / "memory.max").read_text().strip() != "max"
            assert (self.scope.path / "cpu.max").read_text().strip() != "max 100000"
            preexec_fn()
            children.append(self)

        def communicate(self, timeout=None):
            if "bytearray" in self.argv[-1]:
                (self.scope.path / "memory.events").write_text("oom 1\noom_kill 1\n")
                self.returncode = -signal.SIGKILL
            elif "while time.monotonic()" in self.argv[-1]:
                (self.scope.path / "cpu.stat").write_text("nr_throttled 20\nthrottled_usec 1400000\n")
            return f"0::/{self.scope.path.name}\n", ""

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(Path, "mkdir", make)
    monkeypatch.setattr(Path, "rmdir", remove)
    monkeypatch.setattr(caps, "_write", write)
    monkeypatch.setattr(caps.subprocess, "Popen", Child)
    return root, created, children, killed


def test_canonical_prefix_and_declared_profiles():
    assert caps.cap_prefix("/env/bin/python", "/repo", "/sys/fs/cgroup/jobs", cpu_cores=2, memory_bytes=8*caps.GIB) == [
        "/env/bin/python", "/repo/src/ec_clean30/container_caps.py", "run", "--cgroup-root", "/sys/fs/cgroup/jobs",
        "--cpu-cores", "2", "--memory-bytes", str(8*caps.GIB), "--"]
    with pytest.raises(caps.CapError):
        caps.cap_prefix("python", "/repo", "/caps", cpu_cores=2, memory_bytes=8*caps.GIB)
    with pytest.raises(caps.CapError):
        caps.cap_prefix("/python", "/repo", "/caps", cpu_cores=True, memory_bytes=8*caps.GIB)


def test_ordinary_files_cannot_impersonate_kernel_cgroups(tmp_path):
    (tmp_path / "cgroup.type").write_text("domain")
    (tmp_path / "cgroup.subtree_control").write_text("cpu memory")
    with pytest.raises(caps.CapError, match="writable delegated cgroup v2"):
        caps.validate_root(tmp_path)
    assert not list(tmp_path.glob("deepmzyme-g2-*"))


def test_missing_delegated_controller_refuses_before_creation(kernel):
    root, created, children, _ = kernel
    (root / "cgroup.subtree_control").write_text("cpu")
    with pytest.raises(caps.CapError, match="delegate cpu and memory"):
        caps._run(root, 2*caps.PERIOD_US, 8*caps.GIB, [sys.executable, "-c", "pass"])
    assert created == children == []


def test_no_swap_control_is_mandatory_before_child_spawn(kernel, monkeypatch):
    root, created, children, _ = kernel
    write = caps._write

    def refuse_swap(path, value):
        if path.name == "memory.swap.max":
            raise PermissionError("not delegated")
        write(path, value)

    monkeypatch.setattr(caps, "_write", refuse_swap)
    with pytest.raises(PermissionError, match="not delegated"):
        caps._run(root, 2*caps.PERIOD_US, 8*caps.GIB, [sys.executable, "-c", "pass"])
    assert children == []
    assert len(created) == 1 and not created[0].exists()


def test_missing_recursive_kill_support_refuses_and_cleans_empty_scope(kernel, monkeypatch):
    root, created, children, _ = kernel
    write = caps._write

    def refuse_kill(path, value):
        if path.name == "cgroup.kill":
            path.unlink()
            raise FileNotFoundError("cgroup.kill unavailable")
        write(path, value)

    monkeypatch.setattr(caps, "_write", refuse_kill)
    with pytest.raises(FileNotFoundError, match="cgroup.kill unavailable"):
        caps._run(root, 2*caps.PERIOD_US, 8*caps.GIB, [sys.executable, "-c", "pass"])
    assert children == [] and len(created) == 1 and not created[0].exists()


def test_three_lane_children_get_independent_fit_caps(kernel):
    root, created, children, killed = kernel
    for _ in range(3):
        result = caps._run(root, 2*caps.PERIOD_US, 8*caps.GIB, [sys.executable, "-c", "pass"])
        assert result["membership_verified"] and result["cleanup_verified"]
        assert result["limits"]["cpu.max"] == "200000 100000"
        assert result["limits"]["memory.max"] == str(8*caps.GIB)
        assert result["limits"]["memory.swap.max"] == "0"
    assert len(set(created)) == len(children) == 3
    assert all(path in killed and not path.exists() for path in created)


def test_timeout_recursively_kills_descendants_and_removes_scope(kernel, monkeypatch):
    root, created, children, killed = kernel

    def timeout(self, timeout=None):
        (self.scope.path / "cgroup.events").write_text("populated 1\nfrozen 0\n")
        (self.scope.path / "cgroup.procs").write_text("123\n456\n")
        raise subprocess.TimeoutExpired(self.argv, 1)

    monkeypatch.setattr(caps.subprocess.Popen, "communicate", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        caps._run(root, 2*caps.PERIOD_US, 8*caps.GIB, [sys.executable, "-c", "pass"], timeout=1)
    assert len(children) == 1 and killed.count(created[0]) >= 2
    assert not created[0].exists()


def test_oom_and_tampered_controls_cannot_pass(kernel, monkeypatch):
    root, created, _, _ = kernel
    with pytest.raises(caps.CapError, match="OOM"):
        caps._run(root, caps.PERIOD_US, 64*1024**2, [sys.executable, "-c", "bytearray(128*1024**2)"])
    assert all(not path.exists() for path in created)

    def tamper(self, timeout=None):
        (self.scope.path / "memory.swap.max").write_text("max\n")
        return "", ""

    monkeypatch.setattr(caps.subprocess.Popen, "communicate", tamper)
    with pytest.raises(caps.CapError, match="differs from the requested"):
        caps._run(root, 2*caps.PERIOD_US, 8*caps.GIB, [sys.executable, "-c", "pass"])
    assert all(not path.exists() for path in created)


def test_probe_requires_actual_enforcement_and_exact_profiles(kernel):
    root, created, children, _ = kernel
    evidence = caps.probe(root)
    assert caps.validate_probe_evidence(evidence) == evidence
    assert len(created) == len(children) == 4 and not any(path.exists() for path in created)
    assert evidence["profiles"]["esm1b"]["limits"]["memory.max"] == str(24*caps.GIB)
    assert evidence["memory_oom_test"]["returncode"] == -signal.SIGKILL
    assert evidence["cpu_throttle_test"]["after"]["cpu_stat"]["nr_throttled"] > 0


@pytest.mark.parametrize("mutation", [
    lambda item: item["profiles"]["fit"]["limits"].update({"memory.max": "max"}),
    lambda item: item["profiles"]["esm1b"].update({"cpu_cores": 2}),
    lambda item: item["profiles"]["fit"].update({"membership_verified": False}),
    lambda item: item["profiles"]["fit"].update({"cleanup_verified": False}),
    lambda item: item["profiles"]["fit"]["after"].update({"swap_current_bytes": 1}),
    lambda item: item["cpu_throttle_test"]["after"]["cpu_stat"].update({"nr_throttled": 0}),
    lambda item: item["memory_oom_test"]["after"]["memory_events"].update({"oom_kill": 0}),
    lambda item: item["memory_oom_test"].update({"returncode": 0}),
    lambda item: item.update({"helper_sha256": "unknown"}),
    lambda item: item.pop("cpu_throttle_test"),
])
def test_forged_or_incomplete_probe_evidence_refused(kernel, mutation):
    evidence = deepcopy(caps.probe(kernel[0]))
    mutation(evidence)
    with pytest.raises(caps.CapError):
        caps.validate_probe_evidence(evidence)


def test_live_kernel_delegation_when_explicitly_supplied():
    root = os.environ.get("DEEPMZYME_TEST_CGROUP_ROOT")
    if not root:
        pytest.skip("Set DEEPMZYME_TEST_CGROUP_ROOT to an owned, delegated cgroup for live CPU qualification.")
    evidence = caps.probe(Path(root))
    assert caps.validate_probe_evidence(evidence)["status"] == "passed"
    assert not list(Path(root).glob("deepmzyme-g2-*"))


def test_cli_missing_delegation_does_not_execute_child(tmp_path):
    marker = tmp_path / "uncapped-child-must-not-run"
    result = subprocess.run([sys.executable, str(Path(caps.__file__)), "run", "--cgroup-root", str(tmp_path),
                             "--cpu-cores", "2", "--memory-bytes", str(8*caps.GIB), "--", sys.executable,
                             "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 125 and "container caps refused" in result.stderr
    assert not marker.exists()
