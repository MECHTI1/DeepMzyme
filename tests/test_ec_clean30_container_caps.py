"""CPU-only checks for fail-closed delegated cgroup containment."""
from __future__ import annotations

from copy import deepcopy
import errno
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ec_clean30 import container_caps as caps


@pytest.fixture
def preparation_kernel(tmp_path, monkeypatch):
    """Private proc/cgroup interface simulation; no host controls are changed."""
    root = tmp_path / 'private_cgroup'
    root.mkdir()
    for name, value in {'cgroup.type': 'domain', 'cgroup.controllers': 'cpu memory',
                        'cgroup.subtree_control': '', 'cpu.max': '1020000 100000', 'cpu.weight': '100',
                        'memory.max': '31000000000', 'memory.high': 'max', 'memory.swap.max': '0',
                        'pids.max': 'max', 'cgroup.procs': ''}.items():
        (root / name).write_text(value + '\n')
    owner = 'owned-pod'
    current = os.getpid()
    memberships = {1: '0::/', current: '0::/', 222: '0::/'}
    evidence = {'init': 'docker-init', 'namespace': 'pid:[987]', 'self_namespace': 'pid:[987]',
                'mounts': f'1 0 0:1 / /proc rw - proc proc rw\n2 0 0:2 / {root} rw - cgroup2 cgroup rw,nsdelegate\n',
                'foreign_pid': None, 'changed_pid': None}
    reads, writes, managers = {}, [], []
    original_read, original_mkdir, original_rmdir, original_link = Path.read_text, Path.mkdir, Path.rmdir, os.readlink

    def read(path, *args, **kwargs):
        value = str(path)
        if value == '/proc/self/mountinfo': return evidence['mounts']
        if value == '/proc/1/comm': return evidence['init'] + '\n'
        parts = path.parts
        if len(parts) >= 4 and parts[:2] == ('/', 'proc') and parts[2].isdigit():
            pid = int(parts[2]); name = parts[3]
            if name == 'cgroup': return memberships[pid] + '\n'
            if name == 'status':
                return ''.join(f'{key}:\t{pid}'+(' 999' if key == 'NSpid' and evidence['foreign_pid'] == pid else '')+'\n'
                               for key in ('Pid', 'Tgid', 'NSpid', 'NStgid'))
            if name == 'stat':
                if evidence.get('malformed_pid') == pid: return 'incomplete stat record\n'
                reads[pid] = reads.get(pid, 0) + 1
                stamp = 1000 + pid + (reads[pid] if evidence['changed_pid'] == pid else 0)
                return f'{pid} (owned process) ' + ' '.join(['S'] + ['0'] * 18 + [str(stamp)]) + '\n'
        if path.name == 'cgroup.procs' and (path.parent == root or path.parent in managers):
            membership = '0::/' if path.parent == root else f'0::/{path.parent.name}'
            return ''.join(f'{pid}\n' for pid, group in memberships.items() if group == membership)
        return original_read(path, *args, **kwargs)

    def link(path, *args, **kwargs):
        if path == '/proc/self/ns/pid': return evidence['self_namespace']
        if path.startswith('/proc/') and path.endswith('/ns/pid'): return evidence['namespace']
        if path in ('/proc/self/ns/cgroup', '/proc/1/ns/cgroup'): return 'cgroup:[654]'
        return original_link(path, *args, **kwargs)

    def mkdir(path, *args, **kwargs):
        result = original_mkdir(path, *args, **kwargs)
        if path.parent == root and path.name.startswith('deepmzyme-control-'):
            managers.append(path)
            for name, value in {'cgroup.type': 'domain', 'cgroup.subtree_control': '', 'cgroup.procs': ''}.items():
                (path / name).write_text(value + '\n')
        return result

    def rmdir(path):
        if path in managers:
            assert not read(path / 'cgroup.procs').strip()
            for entry in path.iterdir(): entry.unlink()
        return original_rmdir(path)

    def write(path, value):
        writes.append((path, value))
        assert path.is_file()
        if path.name == 'cgroup.procs':
            memberships[int(value)] = '0::/' if path.parent == root else f'0::/{path.parent.name}'
        elif path.name == 'cgroup.subtree_control':
            controllers = set(read(path).split())
            for token in value.split():
                if token.startswith('+'): controllers.add(token[1:])
                elif token.startswith('-'): controllers.discard(token[1:])
                else: raise AssertionError(value)
            path.write_text(' '.join(sorted(controllers)) + '\n')
        else: path.write_text(value + '\n')

    monkeypatch.setattr(Path, 'read_text', read)
    monkeypatch.setattr(Path, 'mkdir', mkdir)
    monkeypatch.setattr(Path, 'rmdir', rmdir)
    monkeypatch.setattr(caps.os, 'readlink', link)
    monkeypatch.setattr(caps, '_write', write)
    return SimpleNamespace(root=root, owner=owner, evidence=evidence, writes=writes, members=memberships,
                           managers=managers, read=read, write=write)


def test_prepare_enables_only_delegated_child_controllers_and_is_idempotent(preparation_kernel):
    k = preparation_kernel
    result = caps.prepare(k.root, owner_id=k.owner)
    assert result['status'] == 'prepared' and result['parent_limits_unchanged'] and result['probe_required']
    assert result['parent_limits_before'] == result['parent_limits_after']
    assert set(result['moved_container_pids']) == set(k.members)
    assert all(group == f'0::/deepmzyme-control-{k.owner}' for group in k.members.values())
    writes = list(k.writes)
    again = caps.prepare(k.root, owner_id=k.owner)
    assert again['status'] == 'already_delegated' and k.writes == writes
    assert all(path.name in {'cgroup.procs', 'cgroup.subtree_control'} for path, _ in writes)


@pytest.mark.parametrize('case', ['foreign_init', 'foreign_namespace', 'foreign_pid', 'changed_pid',
                                  'obscured_proc', 'obscured_self', 'obscured_thread_self', 'malformed_proc',
                                  'unbounded_root', 'zero_cpu', 'zero_memory', 'empty_cpu',
                                  'partial_controllers', 'unknown_child'])
def test_prepare_refuses_unproven_namespace_without_mutating_controls(preparation_kernel, case):
    k = preparation_kernel
    if case == 'foreign_init': k.evidence['init'] = 'systemd'
    elif case == 'foreign_namespace': k.evidence['self_namespace'] = 'pid:[123]'
    elif case == 'foreign_pid': k.evidence['foreign_pid'] = 1
    elif case == 'changed_pid': k.evidence['changed_pid'] = 1
    elif case == 'obscured_proc': k.evidence['mounts'] += '3 1 0:3 / /proc/1/status rw - tmpfs tmpfs rw\n'
    elif case == 'obscured_self': k.evidence['mounts'] += '3 1 0:3 / /proc/self/ns/pid rw - tmpfs tmpfs rw\n'
    elif case == 'obscured_thread_self': k.evidence['mounts'] += '3 1 0:3 / /proc/thread-self rw - tmpfs tmpfs rw\n'
    elif case == 'malformed_proc': k.evidence['malformed_pid'] = 1
    elif case == 'unbounded_root': (k.root / 'memory.max').write_text('max\n')
    elif case == 'zero_cpu': (k.root / 'cpu.max').write_text('0 100000\n')
    elif case == 'zero_memory': (k.root / 'memory.max').write_text('0\n')
    elif case == 'empty_cpu': (k.root / 'cpu.max').write_text('\n')
    elif case == 'partial_controllers': (k.root / 'cgroup.subtree_control').write_text('cpu\n')
    else: (k.root / 'foreign').mkdir()
    with pytest.raises(caps.CapError): caps.prepare(k.root, owner_id=k.owner)
    assert k.writes == []


def test_prepare_reconciles_an_owned_partial_manager(preparation_kernel):
    k = preparation_kernel
    manager = k.root / f'deepmzyme-control-{k.owner}'
    manager.mkdir()
    k.members[1] = k.members[os.getpid()] = f'0::/{manager.name}'
    result = caps.prepare(k.root, owner_id=k.owner)
    assert result['status'] == 'prepared'
    assert result['moved_container_pids'] == [222]


def _deny_namespace_link(monkeypatch, path, *, number=errno.EACCES, before_denial=None):
    original = caps.os.readlink

    def readlink(candidate, *args, **kwargs):
        if str(candidate) == path:
            if before_denial is not None:
                before_denial()
            raise OSError(number, 'synthetic namespace access failure')
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr(caps.os, 'readlink', readlink)


def test_prepare_accepts_denied_nonanchor_namespace_only_with_private_procfs_proof(preparation_kernel, monkeypatch):
    k = preparation_kernel
    _deny_namespace_link(monkeypatch, '/proc/222/ns/pid')
    result = caps.prepare(k.root, owner_id=k.owner)
    assert result['parent_limits_before'] == result['parent_limits_after']
    assert result['probe_required'] and 222 in result['moved_container_pids']
    assert all(group == f'0::/deepmzyme-control-{k.owner}' for group in k.members.values())
    assert all(path.name in {'cgroup.procs', 'cgroup.subtree_control'} for path, _ in k.writes)


def test_denied_namespace_fallback_preserves_complete_rollback(preparation_kernel, monkeypatch):
    k = preparation_kernel
    initial = dict(k.members)
    _deny_namespace_link(monkeypatch, '/proc/222/ns/pid')

    def refuse_enable(path, value):
        if path == k.root / 'cgroup.subtree_control' and value.startswith('+'):
            raise PermissionError('delegation refused after process migration')
        k.write(path, value)

    monkeypatch.setattr(caps, '_write', refuse_enable)
    with pytest.raises(PermissionError, match='after process migration'):
        caps.prepare(k.root, owner_id=k.owner)
    assert k.members == initial and not k.managers[0].exists()
    assert not k.read(k.root / 'cgroup.subtree_control').strip()


@pytest.mark.parametrize('problem', ['nested_pid', 'changed_start', 'obscured_proc', 'changed_anchor'])
def test_late_denied_namespace_with_unproven_identity_rolls_back(preparation_kernel, monkeypatch, problem):
    k = preparation_kernel
    initial = dict(k.members)
    if problem == 'nested_pid': k.evidence['foreign_pid'] = 222
    elif problem == 'changed_start': k.evidence['changed_pid'] = 222

    def invalidate_anchor():
        if problem == 'obscured_proc':
            k.evidence['mounts'] += '3 1 0:3 / /proc/222/status rw - tmpfs tmpfs rw\n'
        elif problem == 'changed_anchor':
            k.evidence['self_namespace'] = 'pid:[other]'

    _deny_namespace_link(monkeypatch, '/proc/222/ns/pid', before_denial=invalidate_anchor)
    with pytest.raises(caps.CapError):
        caps.prepare(k.root, owner_id=k.owner)
    assert k.members == initial and not k.managers[0].exists()
    assert not k.read(k.root / 'cgroup.subtree_control').strip()


def test_readable_namespace_mismatch_is_never_replaced_by_procfs_fallback(preparation_kernel, monkeypatch):
    k = preparation_kernel
    initial = dict(k.members)
    original = caps.os.readlink
    monkeypatch.setattr(caps.os, 'readlink', lambda path, *a, **kw:
        'pid:[foreign]' if str(path) == '/proc/222/ns/pid' else original(path, *a, **kw))
    with pytest.raises(caps.CapError, match='outside the private'):
        caps.prepare(k.root, owner_id=k.owner)
    assert k.members == initial and not k.managers[0].exists()


@pytest.mark.parametrize('path', ['/proc/1/ns/pid', '/proc/self/ns/pid', '/proc/self/ns/cgroup',
                                  '/proc/1/ns/cgroup'])
def test_denied_procfs_anchor_refuses_before_mutation(preparation_kernel, monkeypatch, path):
    k = preparation_kernel
    _deny_namespace_link(monkeypatch, path)
    with pytest.raises(PermissionError): caps.prepare(k.root, owner_id=k.owner)
    assert not k.writes and not k.managers


@pytest.mark.parametrize('number', [errno.EPERM, errno.EIO])
def test_non_eacces_namespace_errors_are_not_accepted(preparation_kernel, monkeypatch, number):
    k = preparation_kernel
    initial = dict(k.members)
    _deny_namespace_link(monkeypatch, '/proc/222/ns/pid', number=number)
    with pytest.raises(OSError) as caught: caps.prepare(k.root, owner_id=k.owner)
    assert caught.value.errno == number
    assert k.members == initial and not k.managers[0].exists()


@pytest.mark.parametrize('problem', ['missing_status', 'foreign_cgroup', 'numeric_self_denied'])
def test_denied_namespace_does_not_relax_remaining_identity_guards(preparation_kernel, monkeypatch, problem):
    k = preparation_kernel
    _deny_namespace_link(monkeypatch, f'/proc/{os.getpid()}/ns/pid' if problem == 'numeric_self_denied'
                         else '/proc/222/ns/pid')
    original = Path.read_text

    def read(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if str(path) == '/proc/222/status' and problem == 'missing_status':
            return '\n'.join(line for line in result.splitlines() if not line.startswith('NSpid:'))
        if str(path) == '/proc/222/cgroup' and problem == 'foreign_cgroup': return '0::/foreign\n'
        return result

    monkeypatch.setattr(Path, 'read_text', read)
    initial = dict(k.members)
    with pytest.raises((caps.CapError, PermissionError)): caps.prepare(k.root, owner_id=k.owner)
    assert k.members == initial
    assert not any(manager.exists() for manager in k.managers)


def test_prepare_rolls_back_its_membership_changes_on_permission_failure(preparation_kernel, monkeypatch):
    k = preparation_kernel
    initial = dict(k.members)

    def refuse(path, value):
        if path == k.root / 'cgroup.subtree_control' and value.startswith('+'):
            raise PermissionError('delegation refused')
        k.write(path, value)

    monkeypatch.setattr(caps, '_write', refuse)
    with pytest.raises(PermissionError, match='delegation refused'):
        caps.prepare(k.root, owner_id=k.owner)
    assert k.members == initial and not k.managers[0].exists()


def test_prepare_reports_failed_rollback_without_claiming_success(preparation_kernel, monkeypatch):
    k = preparation_kernel

    def fail(path, value):
        if path == k.root / 'cgroup.subtree_control': raise PermissionError('cannot enable')
        if path == k.root / 'cgroup.procs': raise PermissionError('cannot restore')
        k.write(path, value)

    monkeypatch.setattr(caps, '_write', fail)
    with pytest.raises(caps.CapError, match='rollback is unconfirmed'):
        caps.prepare(k.root, owner_id=k.owner)


@pytest.mark.parametrize('owner', ['', '../foreign', 'pod/foreign', 'pod\nother', True])
def test_prepare_refuses_unsafe_owner_before_inspection(tmp_path, owner):
    with pytest.raises(caps.CapError, match='owner ID'):
        caps.prepare(tmp_path, owner_id=owner)


def test_prepare_cli_requires_explicit_owner_and_emits_structured_receipt(tmp_path, monkeypatch, capsys):
    expected = {'status': 'already_delegated', 'probe_required': True}
    called = []
    monkeypatch.setattr(caps.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(caps, 'prepare', lambda root, *, owner_id: called.append((root, owner_id)) or expected)
    assert caps.main(['prepare', '--cgroup-root', str(tmp_path), '--owner-id', 'named-pod']) == 0
    import json
    assert json.loads(capsys.readouterr().out) == expected
    assert called == [(tmp_path, 'named-pod')]


def test_prepare_inspects_existing_v1_delegation_without_changing_it(kernel_v1):
    root, created, children, _ = kernel_v1
    result = caps.prepare(root, owner_id='owned-pod')
    assert result == {'status': 'already_delegated', 'backend': 'cgroup_v1', 'owner_id': 'owned-pod',
                      'cgroup_root': str(root), 'parent_limits_unchanged': True, 'probe_required': True}
    assert created == children == []


def test_prepare_refuses_unwritable_v1_before_creating_any_child(kernel_v1, monkeypatch):
    root, created, children, _ = kernel_v1
    monkeypatch.setattr(caps.os, 'access', lambda *args: False)
    with pytest.raises(caps.CapError, match='not writable'):
        caps.prepare(root, owner_id='owned-pod')
    assert created == children == []


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


@pytest.fixture
def kernel_v1(tmp_path, monkeypatch):
    """A split-controller kernel fake, never proof of a provider host."""
    root = tmp_path / 'v1'
    cpu, memory = root / 'cpu,cpuacct', root / 'memory'
    cpu.mkdir(parents=True)
    memory.mkdir()
    monkeypatch.setattr(caps, '_v1_roots', lambda path: {'cpu': cpu, 'memory': memory})
    basis = {'swap_total_kib': 0, 'swap_devices': [], 'basis': 'host_has_no_swap_backing', 'pswpin': 5, 'pswpout': 6,
             'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    monkeypatch.setattr(caps, '_no_host_swap', lambda: dict(basis))
    mkdir, rmdir = Path.mkdir, Path.rmdir
    created, children = [], []

    def make(path, *args, **kwargs):
        result = mkdir(path, *args, **kwargs)
        if path.parent in (cpu, memory):
            created.append(path)
            controls = {'cgroup.procs': ''}
            if path.parent == cpu:
                controls.update({'cpu.cfs_quota_us': '-1', 'cpu.cfs_period_us': '100000', 'cpu.cfs_burst_us': '1000',
                                 'cpu.stat': 'nr_throttled 0\nthrottled_time 0'})
            else:
                controls.update({'memory.limit_in_bytes': str(2**63-1), 'memory.memsw.limit_in_bytes': str(2**63-1),
                    'memory.swappiness': '60', 'memory.use_hierarchy': '0',
                    'memory.oom_control': 'oom_kill_disable 0\nunder_oom 0\noom_kill 0', 'cgroup.event_control': '',
                    'memory.failcnt': '0', 'memory.usage_in_bytes': '0', 'memory.stat': 'swap 0\ntotal_swap 0'})
            for name, value in controls.items():
                (path / name).write_text(value + '\n')
        return result

    def remove(path):
        if path in created:
            for f in path.iterdir():
                if f.is_file():
                    f.unlink()
        return rmdir(path)

    def write(path, value):
        assert path.exists()
        if path.name != 'memory.oom_control':
            path.write_text(value + '\n')
        else:
            assert value == '0'

    class Child:
        def __init__(self, argv, *, preexec_fn, **kwargs):
            self.scope, self.argv, self.returncode = preexec_fn.__self__, argv, 0
            preexec_fn()
            assert all(str(os.getpid()) in caps._read(path / 'cgroup.procs') for path in self.scope.paths.values())
            children.append(self)

        def communicate(self, timeout=None):
            if 'bytearray' in self.argv[-1]:
                (self.scope.paths['memory'] / 'memory.oom_control').write_text('oom_kill_disable 0\nunder_oom 0\noom_kill 1\n')
                (self.scope.paths['memory'] / 'memory.failcnt').write_text('1\n')
                os.eventfd_write(self.scope.event_descriptor, 1)
                self.returncode = -signal.SIGKILL
            elif 'while time.monotonic()' in self.argv[-1]:
                (self.scope.paths['cpu'] / 'cpu.stat').write_text('nr_throttled 20\nthrottled_time 1400000000\n')
            for path in self.scope.paths.values():
                (path / 'cgroup.procs').write_text('')
            suffix = ''
            if 'descendant_membership_verified=' in self.argv[-1]:
                self.scope.kill_count += 1  # Explicit simulated kernel cleanup evidence.
                suffix = 'descendant_membership_verified=12345\n'
            return f'3:cpu,cpuacct:/docker/owned/{self.scope.path.name}\n5:memory:/docker/owned/{self.scope.path.name}\n'+suffix, ''

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(Path, 'mkdir', make)
    monkeypatch.setattr(Path, 'rmdir', remove)
    monkeypatch.setattr(caps, '_write', write)
    monkeypatch.setattr(caps.subprocess, 'Popen', Child)
    return root, created, children, basis


def test_v1_profiles_require_split_membership_exact_raw_limits_and_no_swap_host(kernel_v1):
    root, created, children, _ = kernel_v1
    evidence = caps.probe(root)
    assert caps.validate_probe_evidence(evidence) == evidence
    assert evidence['backend'] == 'cgroup_v1'
    assert len(created) == 8 and len(children) == 4 and not any(p.exists() for p in created)
    for name, gib in [('fit', 8), ('esm1b', 24)]:
        record = evidence['profiles'][name]
        assert record['limits']['memory']['memory.limit_in_bytes'] == str(gib*caps.GIB)
        assert record['limits']['memory']['memory.memsw.limit_in_bytes'] == str(gib*caps.GIB)
        assert record['limits']['memory']['memory.swappiness'] == '0'
        assert record['limits']['cpu']['cpu.cfs_burst_us'] == '0'
        assert record['before']['zero_swap_basis']['basis'] == 'host_has_no_swap_backing'
        assert 'memory.swap.max' not in record['limits']


def test_v1_unavailable_memsw_refuses_before_child_spawn(kernel_v1, monkeypatch):
    root, created, children, _ = kernel_v1
    original = caps._write

    def unavailable(path, value):
        if path.name == 'memory.memsw.limit_in_bytes':
            raise PermissionError('memsw not delegated')
        original(path, value)

    monkeypatch.setattr(caps, '_write', unavailable)
    with pytest.raises(PermissionError, match='memsw'):
        caps._run(root, caps.PERIOD_US, caps.GIB, [sys.executable, '-c', 'pass'])
    assert not children and not any(p.exists() for p in created)


@pytest.mark.parametrize('key', ['memory.swappiness', 'memory.memsw.limit_in_bytes', 'memory.use_hierarchy'])
def test_v1_control_tampering_refuses(kernel_v1, monkeypatch, key):
    root, created, _, _ = kernel_v1
    communicate = caps.subprocess.Popen.communicate

    def tamper(child, timeout=None):
        result = communicate(child, timeout)
        (child.scope.paths['memory'] / key).write_text('999\n')
        return result

    monkeypatch.setattr(caps.subprocess.Popen, 'communicate', tamper)
    with pytest.raises(caps.CapError, match='control differs'):
        caps._run(root, caps.PERIOD_US, caps.GIB, [sys.executable, '-c', 'pass'])
    assert not any(p.exists() for p in created)


def test_v1_host_swap_io_change_invalidates_even_after_swap_disabled_again(kernel_v1, monkeypatch):
    root, created, _, basis = kernel_v1
    communicate = caps.subprocess.Popen.communicate

    def changed(child, timeout=None):
        result = communicate(child, timeout)
        basis['pswpout'] += 1
        return result

    monkeypatch.setattr(caps.subprocess.Popen, 'communicate', changed)
    with pytest.raises(caps.CapError, match='swap I/O counters changed'):
        caps._run(root, caps.PERIOD_US, caps.GIB, [sys.executable, '-c', 'pass'])
    assert not any(p.exists() for p in created)


@pytest.mark.parametrize('mutation', [
    lambda e: e['profiles']['fit']['before']['zero_swap_basis'].update({'swap_total_kib': 100}),
    lambda e: e['profiles']['fit']['after']['zero_swap_basis'].update({'pswpout': 999}),
    lambda e: e['profiles']['fit']['limits']['memory'].update({'memory.swappiness': '60'}),
    lambda e: e['profiles']['fit']['controller_cgroups'].update({'cpu': '/host/other'}),
    lambda e: e['memory_oom_test']['after'].update({'memory_failcnt': 0}),
    lambda e: e['memory_oom_test']['after']['memory_events'].update({'oom': 0}),
])
def test_v1_probe_rejects_weakened_or_inconsistent_evidence(kernel_v1, mutation):
    evidence = deepcopy(caps.probe(kernel_v1[0]))
    mutation(evidence)
    with pytest.raises(caps.CapError):
        caps.validate_probe_evidence(evidence)


@pytest.mark.parametrize('problem', [None, 'swap_total', 'swap_free', 'swap_device', 'empty_swaps', 'bound_proc_file', 'ambiguous_proc'])
def test_v1_host_no_swap_evidence_must_be_genuine_and_complete(monkeypatch, problem):
    original = Path.read_text
    values = {'/proc/self/mountinfo': '1 0 0:1 / /proc rw - proc proc rw\n',
              '/proc/meminfo': 'SwapTotal: 0 kB\nSwapFree: 0 kB\n',
              '/proc/swaps': 'Filename Type Size Used Priority\n',
              '/proc/vmstat': 'pswpin 11\npswpout 12\n',
              '/proc/sys/kernel/random/boot_id': '00000000-0000-0000-0000-000000000000\n'}
    if problem == 'swap_total': values['/proc/meminfo'] = 'SwapTotal: 100 kB\nSwapFree: 0 kB\n'
    if problem == 'swap_free': values['/proc/meminfo'] = 'SwapTotal: 0 kB\nSwapFree: 100 kB\n'
    if problem == 'swap_device': values['/proc/swaps'] += '/swap file 100 0 -2\n'
    if problem == 'empty_swaps': values['/proc/swaps'] = ''
    if problem == 'bound_proc_file': values['/proc/self/mountinfo'] += '2 1 0:1 /swaps /proc/swaps rw - proc proc rw\n'
    if problem == 'ambiguous_proc': values['/proc/self/mountinfo'] += '2 1 0:2 / /proc rw - fuse.lxcfs lxcfs rw\n'
    monkeypatch.setattr(Path, 'read_text', lambda path, *a, **kw: values.get(str(path), '') if str(path) in values else original(path, *a, **kw))
    if problem:
        with pytest.raises(caps.CapError): caps._no_host_swap()
    else:
        assert caps._no_host_swap()['pswpout'] == 12


@pytest.mark.parametrize('problem', [None, 'host_wide_mount', 'foreign_membership', 'missing_membership'])
def test_v1_controller_mounts_must_be_container_scoped(tmp_path, monkeypatch, problem):
    root = tmp_path / 'controllers'
    cpu, mem = root / 'cpu,cpuacct', root / 'memory'
    cpu.mkdir(parents=True)
    mem.mkdir()
    for group in (cpu, mem): (group / 'cgroup.procs').write_text(str(os.getpid())+'\n')
    if problem == 'missing_membership': (cpu / 'cgroup.procs').write_text('')
    container = '/' if problem == 'host_wide_mount' else '/docker/owned'
    member = '/docker/foreign' if problem == 'foreign_membership' else '/docker/owned'
    mountinfo = (f'1 0 0:1 {container} {cpu} rw - cgroup cgroup rw,cpu,cpuacct\n'
                 f'2 0 0:2 {container} {mem} rw - cgroup cgroup rw,memory\n')
    original = Path.read_text

    def read(path, *a, **kw):
        if str(path) == '/proc/self/mountinfo': return mountinfo
        if str(path) == '/proc/self/cgroup': return f'3:cpu,cpuacct:{member}\n5:memory:{member}\n'
        return original(path, *a, **kw)

    monkeypatch.setattr(Path, 'read_text', read)
    monkeypatch.setattr(caps, '_no_host_swap', lambda: {})
    if problem:
        with pytest.raises(caps.CapError): caps._v1_roots(root)
    else:
        assert caps._v1_roots(root) == {'cpu': cpu, 'memory': mem}


def test_pidfd_uses_runtime_libc_when_conda_omits_python_wrapper():
    descriptor = caps._pidfd_open(os.getpid())
    try:
        signal.pidfd_send_signal(descriptor, 0)
    finally:
        os.close(descriptor)


def test_v1_swap_accounting_can_be_absent_only_with_explicit_host_no_swap_basis(kernel_v1, monkeypatch):
    root = kernel_v1[0]
    create = Path.mkdir

    def no_accounting(path, *args, **kwargs):
        result = create(path, *args, **kwargs)
        if path.parent.name == 'memory':
            (path / 'memory.memsw.limit_in_bytes').unlink()
            (path / 'memory.stat').write_text('rss 0\n')
        return result

    monkeypatch.setattr(Path, 'mkdir', no_accounting)
    evidence = caps.probe(root)
    assert caps.validate_probe_evidence(evidence) == evidence
    record = evidence['profiles']['fit']
    assert record['memsw_accounting_available'] is False
    assert 'memory.memsw.limit_in_bytes' not in record['limits']['memory']
    assert record['after']['swap_stat_bytes'] is None
    assert record['after']['swap_current_source'] == 'host_no_swap_invariant'


@pytest.mark.parametrize('foreign', [False, True])
def test_v1_cleanup_binds_pidfd_and_rechecks_membership_before_signal(kernel_v1, monkeypatch, foreign):
    root = kernel_v1[0]
    signaled, descriptors, visits = [], [], []
    original_read = Path.read_text

    def open_pid(pid):
        assert pid == 12345
        descriptor = os.open('/dev/null', os.O_RDONLY)
        descriptors.append(descriptor)
        return descriptor

    def read(path, *args, **kwargs):
        if str(path) == '/proc/12345/cgroup':
            return '5:memory:/docker/owned/'+('other' if foreign else scope.path.name)+'\n'
        return original_read(path, *args, **kwargs)

    def members():
        visits.append(True)
        return {12345} if len(visits) == 1 else set()

    monkeypatch.setattr(caps, '_pidfd_open', open_pid)
    monkeypatch.setattr(caps.signal, 'pidfd_send_signal', lambda fd, sig: signaled.append((fd, sig)))
    monkeypatch.setattr(Path, 'read_text', read)
    with caps._scope(root, caps.PERIOD_US, caps.GIB) as scope:
        monkeypatch.setattr(scope, '_members', members)
        scope.kill()
    assert len(descriptors) == 1
    assert signaled == ([] if foreign else [(descriptors[0], signal.SIGKILL)])
    with pytest.raises(OSError): os.fstat(descriptors[0])


def test_v1_swap_counter_change_while_child_runs_forces_termination(kernel_v1, monkeypatch):
    root, created, _, basis = kernel_v1
    killed = []

    def running(child, timeout=None):
        basis['pswpout'] += 1
        raise subprocess.TimeoutExpired(child.argv, timeout)

    def kill(scope):
        killed.append(scope.path)
        for path in scope.paths.values(): (path / 'cgroup.procs').write_text('')

    monkeypatch.setattr(caps.subprocess.Popen, 'communicate', running)
    monkeypatch.setattr(caps._V1Scope, 'kill', kill)
    with pytest.raises(caps.CapError, match='swap I/O counters changed'):
        caps._run(root, caps.PERIOD_US, caps.GIB, [sys.executable, '-c', 'pass'])
    assert killed and not any(path.exists() for path in created)
