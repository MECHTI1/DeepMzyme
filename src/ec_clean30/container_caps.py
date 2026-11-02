"""Direct cgroup-v2 caps for the G2 container route; no advisory fallback.

The provider must delegate a writable cgroup with cpu and memory controllers
already enabled for its children. This module neither modifies ancestor limits
nor attempts to mount a cgroup filesystem. Each command gets its own leaf, using
the kernel controls underlying systemd CPUQuota, MemoryMax and MemorySwapMax.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Iterator, Sequence

GIB = 1024**3
PERIOD_US = 100_000
PROBE_FORMAT = "deepmzyme_ec_clean30_container_caps_probe_v1"


class CapError(RuntimeError):
    """The requested kernel containment could not be established or preserved."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CapError(message)


def _read(path: Path) -> str:
    return path.read_text().strip()


def _write(path: Path, value: str) -> None:
    # An existing kernel control is required; never create a similarly named file.
    descriptor = os.open(path, os.O_WRONLY | os.O_CLOEXEC)
    try:
        payload = (value + "\n").encode()
        _require(os.write(descriptor, payload) == len(payload), "A kernel resource-control write was incomplete.")
    finally:
        os.close(descriptor)


def _fields(path: Path) -> dict[str, int]:
    return {key: int(value) for key, value in (line.split() for line in _read(path).splitlines())}


def _mount_unescape(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def _require_cgroup_mount(root: Path) -> None:
    candidates = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        fields, details = left.split(), right.split()
        mount = Path(_mount_unescape(fields[4]))
        if root == mount or mount in root.parents:
            candidates.append((len(mount.parts), fields, details))
    _require(bool(candidates), "No mounted filesystem contains the cgroup root.")
    _, fields, details = max(candidates, key=lambda item: item[0])
    _require(details[0] == "cgroup2" and "rw" in fields[5].split(",")
             and not os.statvfs(root).f_flag & os.ST_RDONLY,
             "A writable delegated cgroup v2 mount is required; no resource-cap fallback is allowed.")


def validate_root(root: Path) -> Path:
    root = Path(root)
    _require(root.is_absolute() and root.is_dir(), "The delegated cgroup root must be an existing absolute directory.")
    _require(root == root.resolve(), "The delegated cgroup root must be canonical and contain no symlinks.")
    _require_cgroup_mount(root)
    _require(_read(root / "cgroup.type") == "domain", "A valid domain cgroup is required.")
    _require({"cpu", "memory"}.issubset(_read(root / "cgroup.subtree_control").split()),
             "The provider must delegate cpu and memory controllers enabled for child cgroups.")
    return root


def cap_prefix(python: str, repo: str | Path, cgroup_root: str | Path, *,
               cpu_cores: int, memory_bytes: int) -> list[str]:
    _require(type(cpu_cores) is int and cpu_cores > 0, "CPU cores must be a positive integer.")
    _require(type(memory_bytes) is int and memory_bytes > 0, "Memory bytes must be a positive integer.")
    _require(all(Path(value).is_absolute() for value in (python, repo, cgroup_root)),
             "The interpreter, checkout and delegated cgroup root must be absolute paths.")
    return [str(python), str(Path(repo) / "src/ec_clean30/container_caps.py"), "run", "--cgroup-root",
            str(cgroup_root), "--cpu-cores", str(cpu_cores), "--memory-bytes", str(memory_bytes), "--"]


class _Scope:
    def __init__(self, root: Path, cpu_quota_us: int, memory_bytes: int):
        self.path = root / f"deepmzyme-g2-{os.getpid()}-{secrets.token_hex(8)}"
        self.created = False
        self.limits = {"cpu.max": f"{cpu_quota_us} {PERIOD_US}", "memory.max": str(memory_bytes),
                       "memory.swap.max": "0", "memory.oom.group": "1"}

    def establish(self) -> None:
        self.path.mkdir(mode=0o700)
        self.created = True
        _require(_read(self.path / "cgroup.type") == "domain", "The child cgroup is not a valid domain.")
        _require(not _read(self.path / "cgroup.procs"), "The new child cgroup unexpectedly contains processes.")
        for name, value in self.limits.items():
            _write(self.path / name, value)
        if (self.path / "cpu.max.burst").exists():
            self.limits["cpu.max.burst"] = "0"
            _write(self.path / "cpu.max.burst", "0")
        # Verify recursive kill support while the new scope is still empty.
        _write(self.path / "cgroup.kill", "1")
        self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        limits = {name: _read(self.path / name) for name in self.limits}
        _require(limits == self.limits, "A child CPU, RAM or zero-swap limit differs from the requested kernel setting.")
        swap = int(_read(self.path / "memory.swap.current"))
        _require(swap == 0, "The child cgroup has nonzero swap usage.")
        return {"limits": limits, "cpu_stat": _fields(self.path / "cpu.stat"),
                "memory_events": _fields(self.path / "memory.events"), "swap_current_bytes": swap,
                "memory_current_bytes": int(_read(self.path / "memory.current"))}

    def attach(self) -> None:
        # Popen waits for preexec_fn to finish before returning; exec and every
        # subsequent fork therefore happen inside the already configured leaf.
        _write(self.path / "cgroup.procs", str(os.getpid()))
        _require(str(os.getpid()) in _read(self.path / "cgroup.procs").split(),
                 "The child did not join its capped cgroup before exec.")
        _require(os.sched_getscheduler(0) == os.SCHED_OTHER,
                 "CPU bandwidth containment requires the ordinary fair scheduler.")
        self.snapshot()

    def cleanup(self) -> None:
        if not self.created:
            return
        if (self.path / "cgroup.kill").exists():
            _write(self.path / "cgroup.kill", "1")
        else:
            _require(_fields(self.path / "cgroup.events").get("populated") == 0,
                     "Recursive cgroup termination is unavailable while child processes remain.")
        deadline = time.monotonic() + 5
        while _fields(self.path / "cgroup.events").get("populated") != 0:
            _require(time.monotonic() < deadline, "The capped cgroup still contains processes after recursive kill.")
            time.sleep(0.02)
        # Only remove our own unique subtree, including empty descendant groups.
        children = (path for path in self.path.rglob("*") if path.is_dir())
        for child in sorted(children, key=lambda path: len(path.parts), reverse=True):
            child.rmdir()
        self.path.rmdir()


@contextmanager
def _scope(root: Path, cpu_quota_us: int, memory_bytes: int) -> Iterator[_Scope]:
    _require(type(cpu_quota_us) is int and cpu_quota_us > 0, "A positive CPU quota is required.")
    _require(type(memory_bytes) is int and memory_bytes > 0, "A positive memory limit is required.")
    scope = _Scope(validate_root(root), cpu_quota_us, memory_bytes)
    try:
        scope.establish()
        yield scope
    finally:
        scope.cleanup()


def _run(root: Path, cpu_quota_us: int, memory_bytes: int, argv: Sequence[str], *,
         timeout: float | None = None, capture: bool = False, allow_oom: bool = False,
         emit: bool = False) -> dict[str, Any]:
    _require(bool(argv) and all(isinstance(value, str) and value for value in argv), "A child command is required.")
    with _scope(root, cpu_quota_us, memory_bytes) as scope:
        before = scope.snapshot()
        evidence = {"cgroup": str(scope.path), "limits": before["limits"], "before": before,
                    "memory_limit_bytes": memory_bytes}
        if emit:
            print(json.dumps({"container_caps_started": evidence}, sort_keys=True), file=sys.stderr, flush=True)
        child = subprocess.Popen(list(argv), preexec_fn=scope.attach,
                                 stdout=subprocess.PIPE if capture else None,
                                 stderr=subprocess.PIPE if capture else None, text=capture)
        try:
            stdout, stderr = child.communicate(timeout=timeout)
        except BaseException:
            _write(scope.path / "cgroup.kill", "1")
            child.wait(timeout=5)
            raise
        after = scope.snapshot()
        evidence.update({"after": after, "returncode": child.returncode, "membership_verified": True})
        if capture:
            evidence.update({"stdout": stdout, "stderr": stderr})
        if emit:
            print(json.dumps({"container_caps_finished": evidence}, sort_keys=True), file=sys.stderr, flush=True)
        _require(allow_oom or after["memory_events"].get("oom", 0) == 0,
                 "The command encountered a cgroup OOM; its outputs cannot qualify.")
        _require(allow_oom or after["memory_events"].get("oom_kill", 0) == 0,
                 "A process in the capped command was OOM-killed; its outputs cannot qualify.")
    evidence["cleanup_verified"] = True
    return evidence


def probe(root: Path) -> dict[str, Any]:
    """Exercise kernel enforcement in bounded disposable scopes, then both profiles."""
    root = validate_root(root)
    cpu = _run(root, PERIOD_US // 4, 64 * 1024**2,
               [sys.executable, "-c", "import time; end=time.monotonic()+2\nwhile time.monotonic()<end: pass"],
               timeout=15, capture=True)
    _require(cpu["returncode"] == 0 and cpu["after"]["cpu_stat"].get("nr_throttled", 0) > 0
             and cpu["after"]["cpu_stat"].get("throttled_usec", 0) > 0,
             "The contained CPU probe did not demonstrate kernel throttling.")
    cpu["status"] = "passed"
    memory = _run(root, PERIOD_US, 64 * 1024**2,
                  [sys.executable, "-c", "payload=bytearray(128*1024*1024); print(len(payload))"],
                  timeout=15, capture=True, allow_oom=True)
    _require(memory["returncode"] == -signal.SIGKILL
             and memory["after"]["memory_events"].get("oom_kill", 0) > 0,
             "The contained memory probe did not demonstrate kernel OOM termination.")
    memory["status"] = "passed"
    profiles = {}
    for name, cores, gib in (("fit", 2, 8), ("esm1b", 4, 24)):
        evidence = _run(root, cores * PERIOD_US, gib * GIB,
                        [sys.executable, "-c", "from pathlib import Path; print(Path('/proc/self/cgroup').read_text())"],
                        timeout=15, capture=True)
        _require(evidence["returncode"] == 0 and Path(evidence["cgroup"]).name in evidence["stdout"],
                 "The profile probe did not confirm its actual cgroup membership.")
        evidence.update({"cpu_cores": cores, "status": "passed"})
        profiles[name] = evidence
    return {"format": PROBE_FORMAT, "status": "passed", "cgroup_root": str(root),
            "hostname": socket.gethostname(), "boot_id": _read(Path("/proc/sys/kernel/random/boot_id")),
            "created_unix": time.time(), "python_executable": sys.executable,
            "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "profiles": profiles, "cpu_throttle_test": cpu, "memory_oom_test": memory}


def validate_probe_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Validate a recorded probe; execution still re-establishes every live cap."""
    try:
        _require(evidence["format"] == PROBE_FORMAT and evidence["status"] == "passed",
                 "The container-cap probe did not pass.")
        root = Path(evidence["cgroup_root"])
        _require(root.is_absolute() and ".." not in root.parts, "The cap probe root is not absolute and canonical.")
        _require(bool(evidence["hostname"]) and bool(re.fullmatch(r"[0-9a-f-]{36}", evidence["boot_id"]))
                 and bool(re.fullmatch(r"[0-9a-f]{64}", evidence["helper_sha256"]))
                 and Path(evidence["python_executable"]).is_absolute()
                 and math.isfinite(evidence["created_unix"]) and evidence["created_unix"] > 0,
                 "The cap probe lacks host, interpreter, source or time identity.")

        def check(record: dict[str, Any], quota: int, memory: int, *, oom: bool = False) -> None:
            expected = {"cpu.max": f"{quota} {PERIOD_US}", "memory.max": str(memory),
                        "memory.swap.max": "0", "memory.oom.group": "1"}
            if "cpu.max.burst" in record["limits"]:
                expected["cpu.max.burst"] = "0"
            _require(record["status"] == "passed" and record["limits"] == expected
                     and record["memory_limit_bytes"] == memory and record["membership_verified"] is True
                     and record["cleanup_verified"] is True and Path(record["cgroup"]).parent == root
                     and Path(record["cgroup"]).name.startswith("deepmzyme-g2-"),
                     "The cap probe lacks exact kernel limits, membership or cleanup evidence.")
            for stage in ("before", "after"):
                snapshot = record[stage]
                _require(snapshot["limits"] == expected and snapshot["swap_current_bytes"] == 0
                         and snapshot["memory_current_bytes"] >= 0
                         and all(type(value) is int and value >= 0 for value in snapshot["memory_events"].values())
                         and all(type(value) is int and value >= 0 for value in snapshot["cpu_stat"].values()),
                         "The cap probe contains inconsistent kernel readbacks.")
                if not oom or stage == "before":
                    _require(snapshot["memory_events"]["oom"] == 0 and snapshot["memory_events"]["oom_kill"] == 0,
                             "The cap probe encountered an unexpected OOM.")

        _require(set(evidence["profiles"]) == {"fit", "esm1b"}, "Both exact G2 resource profiles are required.")
        for name, cores, gib in (("fit", 2, 8), ("esm1b", 4, 24)):
            record = evidence["profiles"][name]
            check(record, cores * PERIOD_US, gib * GIB)
            _require(record["cpu_cores"] == cores and record["returncode"] == 0
                     and Path(record["cgroup"]).name in record["stdout"],
                     "The G2 cap profile lacks successful in-child membership proof.")
        cpu = evidence["cpu_throttle_test"]
        check(cpu, PERIOD_US // 4, 64 * 1024**2)
        _require(cpu["returncode"] == 0
                 and cpu["after"]["cpu_stat"]["nr_throttled"] > cpu["before"]["cpu_stat"]["nr_throttled"]
                 and cpu["after"]["cpu_stat"]["throttled_usec"] > cpu["before"]["cpu_stat"]["throttled_usec"],
                 "The cap probe did not establish actual kernel CPU throttling.")
        memory = evidence["memory_oom_test"]
        check(memory, PERIOD_US, 64 * 1024**2, oom=True)
        _require(memory["returncode"] == -signal.SIGKILL and memory["after"]["memory_events"]["oom_kill"] > 0,
                 "The cap probe did not establish contained kernel OOM termination.")
    except (KeyError, TypeError, ValueError) as exc:
        raise CapError(f"Malformed container-cap probe evidence: {exc}") from exc
    return evidence


def _interrupted(signum: int, _frame: Any) -> None:
    raise CapError(f"Resource-cap wrapper interrupted by signal {signum}.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("probe", "run"):
        command = commands.add_parser(name)
        command.add_argument("--cgroup-root", type=Path, required=True)
        if name == "run":
            command.add_argument("--cpu-cores", type=int, required=True)
            command.add_argument("--memory-bytes", type=int, required=True)
            command.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, _interrupted)
    try:
        if args.action == "probe":
            print(json.dumps(probe(args.cgroup_root), sort_keys=True))
            return 0
        _require(args.cpu_cores > 0 and args.memory_bytes > 0, "Positive CPU and RAM limits are required.")
        _require(args.command and args.command[0] == "--", "The command must follow an explicit -- separator.")
        result = _run(args.cgroup_root, args.cpu_cores * PERIOD_US, args.memory_bytes, args.command[1:], emit=True)
        return result["returncode"] if result["returncode"] >= 0 else 128 - result["returncode"]
    except (CapError, OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"container caps refused: {exc}", file=sys.stderr)
        return 125


if __name__ == "__main__":
    raise SystemExit(main())
