"""CPU-only checks for fail-closed delegated cgroup containment."""
from __future__ import annotations

import ast
import contextlib
from copy import deepcopy
import errno
import io
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
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


def test_prepare_selects_supervision_for_unwritable_v1_without_creating_any_child(kernel_v1, monkeypatch):
    root, created, children, _ = kernel_v1
    monkeypatch.setattr(caps.os, 'access', lambda *args: False)
    monkeypatch.setattr(caps, '_subreaper', lambda: True)
    assert caps.prepare(root, owner_id='owned-pod') == {
        'status': 'supervised_without_delegation', 'backend': 'cgroup_v1_readonly', 'owner_id': 'owned-pod',
        'cgroup_root': str(root), 'parent_limits_unchanged': True, 'probe_required': True}
    assert created == children == []
    monkeypatch.setattr(caps, '_subreaper', lambda: False)
    with pytest.raises(caps.CapError, match='subreaper'):
        caps.prepare(root, owner_id='owned-pod')


READONLY_DRIVER = r"""
import json, sys
from pathlib import Path
from ec_clean30 import container_caps as caps
root = Path(sys.argv[1]); (root / 'cpu').mkdir(parents=True, exist_ok=True); (root / 'memory').mkdir(exist_ok=True)
basis = {'swap_total_kib': 0, 'swap_devices': [], 'basis': 'host_has_no_swap_backing', 'pswpin': 5, 'pswpout': 6,
         'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
caps._v1_roots = lambda path, **kw: {'cpu': root / 'cpu', 'memory': root / 'memory'}
caps._v1_backend = lambda path: caps.READONLY_BACKEND
caps._no_host_swap = lambda: dict(basis)
caps.CPU_CLAIM_ROOT = root / 'claims'
caps._helper_command = lambda: [sys.executable, str(root.parent / 'caps_helper.py')]
"""

# The helper started as a command of its own, as ``cap_prefix`` starts it, with the same substitutions. It finds
# the package by itself, because a step may start it with a rebuilt environment that carries no PYTHONPATH.
READONLY_HELPER = r"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, {src!r})
arguments, sys.argv[1:] = sys.argv[1:], [{root!r}]
""" + READONLY_DRIVER.replace('{', '{{').replace('}', '}}') + r"""
{extra}
raise SystemExit(caps.main(arguments))
"""

# What scripts/run_clean_split30_comparator.py keeps when it builds the environment of a CLEAN step.
REBUILT_ENVIRONMENT = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "USER", "LOGNAME")


def _need_cores(count):
    """Only for a case that also runs in a two-core form, so a small runner still tests the behaviour."""
    if len(os.sched_getaffinity(0)) < count:
        pytest.skip(f"needs {count} usable CPU cores; the two-core form of this case runs instead")


def _helper(tmp_path, extra=''):
    path = tmp_path / 'caps_helper.py'
    path.write_text(READONLY_HELPER.format(src=str(Path(caps.__file__).resolve().parents[1]),
                                           root=str(tmp_path / 'ro'), extra=extra))
    return path


def _prefix(tmp_path, cores, memory):
    """The cap prefix of the read-only stand-in, in the form ``cap_prefix`` gives."""
    return [sys.executable, str(tmp_path / 'caps_helper.py'), 'run', '--cgroup-root', str(tmp_path / 'ro'),
            '--cpu-cores', str(cores), '--memory-bytes', str(memory), '--']


def _records(text):
    """The start and end records a wrapper printed, and its last other line."""
    rows = [json.loads(line) for line in text.splitlines() if line.startswith('{"container_caps_')]
    other = [line for line in text.splitlines() if line.strip() and not line.startswith('{"container_caps_')]
    return {'started': next((row['container_caps_started'] for row in rows if 'container_caps_started' in row), None),
            'finished': next((row['container_caps_finished'] for row in rows if 'container_caps_finished' in row), None),
            'last': other[-1] if other else ''}


def _unscoped_environment():
    return {key: value for key, value in os.environ.items() if key != caps.SCOPE_ENV}


def _supervised(tmp_path, body, timeout=120, helper=''):
    """The supervisor owns its whole process tree, so each case runs in a wrapper process of its own."""
    _helper(tmp_path, helper)
    result = subprocess.run([sys.executable, '-c', READONLY_DRIVER + body, str(tmp_path / 'ro')],
                            capture_output=True, text=True, timeout=timeout,
                            env={**os.environ, 'PYTHONPATH': str(Path(caps.__file__).resolve().parents[1])})
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_readonly_backend_probe_passes_end_to_end_and_its_evidence_validates(tmp_path):
    evidence = _supervised(tmp_path, "print(json.dumps(caps.probe(root)))")
    assert evidence['backend'] == 'cgroup_v1_readonly'
    caps.validate_probe_evidence(evidence)
    assert evidence['cpu_throttle_test']['affinity_proof']['affinity_cpus'] == 1
    assert evidence['memory_oom_test']['returncode'] == -signal.SIGKILL
    assert [len(evidence['profiles'][name]['cpu_affinity']) for name in ('fit', 'esm1b')] == [2, 4]
    # The nested command ran with the rebuilt environment, on its parent's cores, under its own 8 GiB limit.
    outer = evidence['nested_test']
    inner = outer['nested']
    holder = int(Path(outer['cgroup']).name.split('-')[2])
    assert outer['status'] == 'passed' and outer['memory_limit_bytes'] == 24 * caps.GIB and len(outer['cpu_affinity']) == 4
    assert outer['before']['enclosing_scope'] is None and 'enclosing_claim' not in outer['before']
    assert set(outer['environment']) <= set(REBUILT_ENVIRONMENT) and caps.SCOPE_ENV not in outer['environment']
    assert inner['helper_returncode'] == 0 and inner['memory_limit_bytes'] == 8 * caps.GIB
    assert inner['limits']['memory']['limit_bytes'] == 8 * caps.GIB and inner['cpu_affinity'] == outer['cpu_affinity']
    assert inner['before']['enclosing_scope'] == Path(outer['cgroup']).name
    assert inner['before']['enclosing_claim']['pid'] == holder
    assert Path(inner['cgroup']).name in inner['stdout'] and 'affinity_cpus=4\n' in inner['stdout']
    assert caps.NESTED_ENVIRONMENT == REBUILT_ENVIRONMENT
    for change in ('cpu', 'cores', 'swap', 'descendant', 'nested_missing', 'nested_refused', 'nested_limit',
                   'nested_cores', 'nested_parent', 'nested_no_claim', 'nested_other_holder', 'nested_same_scope',
                   'nested_environment', 'nested_membership'):
        bad = deepcopy(evidence)
        if change == 'cpu':
            bad['cpu_throttle_test']['affinity_proof']['cpu_seconds'] = 4 * bad['cpu_throttle_test']['affinity_proof']['wall_seconds']
        elif change == 'cores':
            bad['profiles']['esm1b']['cpu_affinity'] = bad['profiles']['esm1b']['cpu_affinity'][:3]
        elif change == 'swap':
            bad['profiles']['fit']['after']['zero_swap_basis']['pswpout'] += 1
        elif change == 'descendant':
            bad['profiles']['fit']['descendant_cleanup_count'] = 0
        elif change == 'nested_missing':
            del bad['nested_test']
        elif change == 'nested_refused':
            bad['nested_test']['nested']['helper_returncode'] = 125
        elif change == 'nested_limit':
            bad['nested_test']['nested']['limits']['memory']['limit_bytes'] = 24 * caps.GIB
        elif change == 'nested_cores':
            bad['nested_test']['nested']['cpu_affinity'] = bad['nested_test']['nested']['cpu_affinity'][::-1]
        elif change == 'nested_parent':
            bad['nested_test']['nested']['before']['enclosing_scope'] = None
        elif change == 'nested_no_claim':
            del bad['nested_test']['nested']['before']['enclosing_claim']
        elif change == 'nested_other_holder':
            bad['nested_test']['nested']['before']['enclosing_claim']['pid'] = holder + 1
        elif change == 'nested_same_scope':
            bad['nested_test']['nested']['cgroup'] = bad['nested_test']['cgroup']
        elif change == 'nested_environment':
            bad['nested_test']['environment'].append(caps.SCOPE_ENV)
        else:
            bad['nested_test']['nested']['stdout'] = 'affinity_cpus=4\n'
        with pytest.raises(caps.CapError):
            caps.validate_probe_evidence(bad)


NOT_RUN = 'A capped command started inside another one with a rebuilt environment did not run: '
# The inner command reports its records without the holder that its claim records named.
NO_HOLDER = """
_snapshot = caps._ReadonlyScope.snapshot
def _without_holder(self):
    record = _snapshot(self)
    record.pop('enclosing_claim', None)
    return record
caps._ReadonlyScope.snapshot = _without_holder
"""


@pytest.mark.parametrize('fault,refusal', [
    ("caps._live_ancestor = lambda pid, start: False", NOT_RUN + "Too few unclaimed CPU cores remain for this capped command."),
    ("raise SystemExit(3)", NOT_RUN + "exit status 3"),
    (NO_HOLDER, "The nested supervised command did not stay on its parent's cores."),
], ids=['parent_not_recognised', 'helper_fails', 'parent_not_from_claim_records'])
def test_readonly_probe_refuses_a_host_whose_nested_command_does_not_run(tmp_path, fault, refusal):
    """The probe exits as a containment refusal, in the short form that host admission reads as a host rejection."""
    result = _supervised(tmp_path, """
import contextlib, io
errors = io.StringIO()
with contextlib.redirect_stderr(errors):
    code = caps.main(['probe', '--cgroup-root', str(root)])
print(json.dumps({'code': code, 'stderr': errors.getvalue()}))
""", helper=fault)
    assert result['code'] == 125
    assert result['stderr'].startswith('container caps refused: ' + refusal)
    # workstation.py reports "Remote command failed (125): " plus the last 2,000 characters of stderr and treats
    # the host as rejected only when that text starts with the refusal prefix.
    assert len(result['stderr']) < 2000 and result['stderr'].count('\n') == 1


def test_readonly_backend_kills_the_whole_tree_above_its_memory_cap(tmp_path):
    program = "payload=bytearray(160*1024*1024); import time; time.sleep(30)"
    result = _supervised(tmp_path, f"""
try:
    caps._run(root, 2 * caps.PERIOD_US, 96 * 1024**2, [sys.executable, '-c', {program!r}], timeout=20)
    print(json.dumps({{'refused': False}}))
except caps.CapError as exc:
    print(json.dumps({{'refused': True, 'error': str(exc), 'left': len(caps._descendants(__import__('os').getpid()))}}))
""")
    assert result['refused'] and 'OOM' in result['error'] and result['left'] == 0


def test_readonly_backend_refuses_a_child_that_widens_its_cpu_affinity(tmp_path):
    program = "import os,time; os.sched_setaffinity(0, range(os.cpu_count())); time.sleep(30)"
    result = _supervised(tmp_path, f"""
try:
    caps._run(root, caps.PERIOD_US, 1024**3, [sys.executable, '-c', {program!r}], timeout=20)
    print(json.dumps({{'refused': False}}))
except caps.CapError as exc:
    print(json.dumps({{'refused': True, 'error': str(exc)}}))
""")
    assert result['refused'] and 'affinity' in result['error']


def test_readonly_backend_gives_concurrent_commands_distinct_cores_and_nests_on_the_parents_cores(tmp_path):
    # Leave two CPUs for the outer command and its nested child on four-core runners.
    peer_cores = 2 if len(os.sched_getaffinity(0)) >= 6 else 1
    result = _supervised(tmp_path, f"peer_quota = {peer_cores} * caps.PERIOD_US\n" + """
import os
first = caps._ReadonlyScope(root, peer_quota, 1024**3); first.establish()
second = caps._ReadonlyScope(root, peer_quota, 1024**3); second.establish()
env = first.environment()
out = {'first': list(first.cpus), 'second': list(second.cpus), 'threads': env.get('OMP_NUM_THREADS') == os.environ.get('OMP_NUM_THREADS'),
       'scope': env[caps.SCOPE_ENV] == first.path.name}
nested = "import json,os,sys; exec(open(sys.argv[2]).read()); " \
         "print(json.dumps(caps._run(root, 2*caps.PERIOD_US, 512*1024**2, [sys.executable,'-c','import os;print(sorted(os.sched_getaffinity(0)))'], capture=True)))"
driver = root / 'driver.py'; driver.write_text(DRIVER)
inner = caps._run(root, 2 * caps.PERIOD_US, 1024**3, [sys.executable, '-c', nested, str(root), str(driver)], capture=True)
out['outer'] = inner['cpu_affinity']; record = json.loads(inner['stdout'].strip().splitlines()[-1])
out['inner'] = record['cpu_affinity']; out['inner_seen'] = json.loads(record['stdout']); out['enclosing'] = record['before']['enclosing_scope'] == Path(inner['cgroup']).name
first.cleanup(); second.cleanup()
third = caps._ReadonlyScope(root, peer_quota, 1024**3); third.establish(); out['third'] = list(third.cpus); third.cleanup()
print(json.dumps(out))
""".replace('DRIVER', repr(READONLY_DRIVER)))
    assert len(result['first']) == len(result['second']) == peer_cores and not set(result['first']) & set(result['second'])
    assert result['third'] == result['first'] and result['threads'] and result['scope']
    assert len(result['outer']) == 2 and not set(result['outer']) & (set(result['first']) | set(result['second']))
    assert result['inner'] == result['inner_seen'] == result['outer'] and result['enclosing']


# The child of an outer capped command: it starts argv[1:], with an environment rebuilt as a CLEAN step rebuilds it
# or, for comparison, with its own environment handed on unchanged.
def _report_program(rebuilt=True):
    env = (f"{{name:os.environ[name] for name in {REBUILT_ENVIRONMENT!r} if name in os.environ}}" if rebuilt
           else "dict(os.environ)")
    return ("import json,os,subprocess,sys; "
            f"env={env}; "
            "done=subprocess.run(sys.argv[1:],env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True); "
            "print('REPORT '+json.dumps({'scope':os.environ.get('DEEPMZYME_CAPS_SCOPE'),"
            "'affinity':sorted(os.sched_getaffinity(0)),'child_environment':sorted(env),"
            "'returncode':done.returncode,'stdout':done.stdout,'stderr':done.stderr}))")


NESTED_REPORT = _report_program()
SEEN = "import json,os; print('SEEN '+json.dumps(sorted(os.sched_getaffinity(0))))"
TOO_FEW = 'Too few unclaimed CPU cores remain for this capped command.'


def _child_report(text):
    line = next((line[7:] for line in text.splitlines() if line.startswith('REPORT ')), None)
    return json.loads(line) if line else None


def _nested(tmp_path, outer, inner, command, *, launcher=False, rebuilt=True, helper='', timeout=120):
    """An outer capped command whose child rebuilds the environment and starts an inner capped command.

    Each wrapper is a helper process of its own, as under ``cap_prefix``. Returns the outer result, the outer
    wrapper's records, the child's report and the inner wrapper's records.
    """
    _helper(tmp_path, helper)
    nested = [*_prefix(tmp_path, *inner), *command]
    if launcher:
        # The adapter phase4._clean_fit writes for the training and export steps.
        path = tmp_path / 'capped_python'
        path.write_text('#!/bin/sh\nexec ' + shlex.join([*_prefix(tmp_path, *inner), command[0]]) + ' "$@"\n')
        path.chmod(0o755)
        nested = [str(path), *command[1:]]
    result = subprocess.run([*_prefix(tmp_path, *outer), sys.executable, '-c', _report_program(rebuilt), *nested],
                            capture_output=True, text=True, timeout=timeout, env=_unscoped_environment())
    report = _child_report(result.stdout)
    return result, _records(result.stderr), report, _records(report['stderr']) if report else None


@pytest.mark.parametrize('outer_cores,inner_cores,launcher', [(4, 4, False), (4, 4, True), (4, 2, False),
                                                              (2, 2, False), (2, 2, True)])
def test_readonly_nested_command_with_a_rebuilt_environment_stays_on_its_parents_cores(
        tmp_path, outer_cores, inner_cores, launcher):
    """A capped unit starts its own capped step with an environment that no longer names the unit's scope."""
    _need_cores(outer_cores)
    result, outer, report, inner = _nested(tmp_path, (outer_cores, 24 * caps.GIB), (inner_cores, 8 * caps.GIB),
                                           [sys.executable, '-c', SEEN], launcher=launcher)
    scope, cores = Path(outer['started']['cgroup']).name, outer['started']['cpu_affinity']
    assert len(cores) == outer_cores and outer['started']['before']['enclosing_scope'] is None
    assert 'enclosing_claim' not in outer['started']['before']
    assert report['scope'] == scope and report['affinity'] == cores
    assert caps.SCOPE_ENV not in report['child_environment']
    assert report['returncode'] == 0, inner['last']
    started = inner['started']
    assert started['backend'] == caps.READONLY_BACKEND and started['memory_limit_bytes'] == 8 * caps.GIB
    assert started['limits']['memory'] == {'policy': 'supervised_process_tree', 'limit_bytes': 8 * caps.GIB,
                                           'poll_seconds': caps.POLL_SECONDS}
    assert started['cpu_affinity'] == cores[:inner_cores]
    assert started['before']['enclosing_scope'] == scope
    assert started['before']['enclosing_claim']['pid'] == int(scope.split('-')[2])
    assert f'SEEN {json.dumps(cores[:inner_cores])}' in report['stdout']
    assert inner['finished']['returncode'] == 0 and result.returncode == 0 and outer['finished']['returncode'] == 0
    # Free cores elsewhere on the machine change nothing: the nested command only ever looks at its parent's cores.
    # No claim file exists for any other core, so none of them was reserved or even examined; the parent's own
    # files name no holder once it has ended.
    claims = sorted((tmp_path / 'ro' / 'claims').iterdir())
    assert [path.name for path in claims] == sorted(f'cpu{cpu}' for cpu in cores)
    assert all(path.read_bytes() == b'' for path in claims)
    if len(os.sched_getaffinity(0)) > outer_cores:
        assert set(os.sched_getaffinity(0)) - set(cores)


def test_readonly_nested_command_that_keeps_the_environment_is_recognised_from_it_as_before(tmp_path):
    result, outer, report, inner = _nested(tmp_path, (2, caps.GIB), (2, caps.GIB), [sys.executable, '-c', SEEN],
                                           rebuilt=False)
    scope = Path(outer['started']['cgroup']).name
    assert caps.SCOPE_ENV in report['child_environment'] and report['returncode'] == 0 and result.returncode == 0
    assert inner['started']['before']['enclosing_scope'] == scope
    assert inner['started']['cpu_affinity'] == outer['started']['cpu_affinity']
    # The claim records are not consulted for a command whose environment names its parent.
    assert 'enclosing_claim' not in inner['started']['before'] and 'enclosing_claim' not in inner['finished']['after']


def test_readonly_command_inside_a_nested_command_names_the_holder_of_its_cores(tmp_path):
    """Three levels, each with a rebuilt environment: only the outermost wrapper holds cores, and both others find it."""
    third = [*_prefix(tmp_path, 2, 256 * 1024**2), sys.executable, '-c', SEEN]
    result, outer, report, inner = _nested(tmp_path, (2, caps.GIB), (2, 512 * 1024**2),
                                           [sys.executable, '-c', NESTED_REPORT, *third])
    scope, cores = Path(outer['started']['cgroup']).name, outer['started']['cpu_affinity']
    assert result.returncode == 0 and report['returncode'] == 0, inner['last']
    middle = _child_report(report['stdout'])
    innermost = _records(middle['stderr'])
    assert middle['returncode'] == 0, innermost['last']
    # The middle command's child was told the middle scope; the innermost wrapper never saw that name.
    assert middle['scope'] == Path(inner['started']['cgroup']).name and caps.SCOPE_ENV not in middle['child_environment']
    for record, limit in ((inner['started'], 512 * 1024**2), (innermost['started'], 256 * 1024**2)):
        assert record['before']['enclosing_scope'] == scope and record['cpu_affinity'] == cores
        assert record['before']['enclosing_claim']['pid'] == int(scope.split('-')[2])
        assert record['memory_limit_bytes'] == limit
    assert f'SEEN {json.dumps(cores)}' in middle['stdout'] and innermost['finished']['returncode'] == 0


@pytest.mark.parametrize('fault', ["def _fail(*args):\n    raise OSError(28, 'No space left on device')\ncaps.os.pwrite = _fail",
                                   "_pwrite = caps.os.pwrite\n"
                                   "caps.os.pwrite = lambda descriptor, data, offset: _pwrite(descriptor, data[:9], offset)"],
                         ids=['write_error', 'short_write'])
def test_readonly_command_whose_claim_record_cannot_be_written_still_runs_and_nothing_nests_inside_it(tmp_path, fault):
    """The lock reserves the cores as before. Without a complete record a command inside finds no parent."""
    program = ("import json,os,sys; from pathlib import Path; "
               "print('HELD '+json.dumps([Path(sys.argv[1], f'cpu{cpu}').read_text() "
               "for cpu in sorted(os.sched_getaffinity(0))]))")
    _helper(tmp_path, fault)
    alone = subprocess.run([*_prefix(tmp_path, 2, caps.GIB), sys.executable, '-c', program, str(tmp_path / 'ro' / 'claims')],
                           capture_output=True, text=True, timeout=60, env=_unscoped_environment())
    assert alone.returncode == 0 and 'HELD ["", ""]' in alone.stdout, alone.stderr[-300:]
    assert len(_records(alone.stderr)['started']['cpu_affinity']) == 2
    result, outer, report, inner = _nested(tmp_path, (2, caps.GIB), (2, caps.GIB), [sys.executable, '-c', SEEN],
                                           helper=fault)
    assert result.returncode == 0 and outer['finished']['returncode'] == 0
    assert report['returncode'] == 125 and inner['started'] is None
    assert inner['last'] == 'container caps refused: ' + TOO_FEW and 'SEEN' not in report['stdout']


def test_readonly_nested_command_is_killed_at_its_own_memory_limit_and_its_parent_goes_on(tmp_path):
    program = "payload=bytearray(160*1024*1024); import time; time.sleep(30)"
    result, outer, report, inner = _nested(tmp_path, (2, caps.GIB), (2, 96 * 1024**2), [sys.executable, '-c', program])
    assert inner['started'] is not None, inner['last']
    assert inner['started']['before']['enclosing_scope'] == Path(outer['started']['cgroup']).name
    assert inner['started']['memory_limit_bytes'] == 96 * 1024**2
    assert report['returncode'] == 125 and 'OOM' in inner['last']
    after = inner['finished']['after']
    assert inner['finished']['returncode'] == -signal.SIGKILL and after['memory_events']['oom_kill'] > 0
    assert 96 * 1024**2 < after['memory_peak_bytes'] < caps.GIB
    # The unit that started the step keeps running under its own, larger limit and sees no memory event.
    assert result.returncode == 0 and outer['finished']['returncode'] == 0
    assert outer['finished']['after']['memory_events'] == {'oom': 0, 'oom_kill': 0}


def test_readonly_parent_memory_limit_also_counts_its_nested_command(tmp_path):
    marker = tmp_path / 'nested.pid'
    program = (f"import os,pathlib,time; pathlib.Path({str(marker)!r}).write_text(str(os.getpid())); "
               "payload=bytearray(160*1024*1024); time.sleep(30)")
    result, outer, report, _ = _nested(tmp_path, (2, 96 * 1024**2), (2, 512 * 1024**2), [sys.executable, '-c', program])
    # The nested command is below its own 512 MiB limit; the parent's 96 MiB limit ends the whole tree.
    assert result.returncode == 125 and 'OOM' in outer['last'] and report is None
    assert outer['finished']['after']['memory_events']['oom_kill'] > 0
    status = Path(f'/proc/{int(marker.read_text())}/stat')
    until = time.monotonic() + 5
    while status.exists() and status.read_text().rsplit(')', 1)[1].split()[0] != 'Z' and time.monotonic() < until:
        time.sleep(.05)
    assert not status.exists() or status.read_text().rsplit(')', 1)[1].split()[0] == 'Z'


PINNED = ("import json,os,subprocess,sys; os.sched_setaffinity(0, json.loads(sys.argv[1])); "
          f"env={{name:os.environ[name] for name in {REBUILT_ENVIRONMENT!r} if name in os.environ}}; "
          "raise SystemExit(subprocess.call(sys.argv[2:], env=env))")


@pytest.mark.parametrize('ending', ['terminated', 'killed'])
def test_readonly_command_outside_a_capped_command_cannot_take_its_cores(tmp_path, ending):
    """Same cores and the same rebuilt environment, but no ancestry: refused as before, until the holder ends."""
    _helper(tmp_path)
    env = _unscoped_environment()
    holder = subprocess.Popen([*_prefix(tmp_path, 2, caps.GIB), sys.executable, '-c', 'import time; time.sleep(60)'],
                              stderr=subprocess.PIPE, text=True, env=env, start_new_session=True)
    try:
        cores = json.loads(holder.stderr.readline())['container_caps_started']['cpu_affinity']
        stranger = [sys.executable, '-c', PINNED, json.dumps(cores), *_prefix(tmp_path, 2, caps.GIB),
                    sys.executable, '-c', SEEN]
        refused = subprocess.run(stranger, capture_output=True, text=True, timeout=60, env=env)
        assert refused.returncode == 125 and _records(refused.stderr)['last'] == 'container caps refused: ' + TOO_FEW
        assert 'SEEN' not in refused.stdout
        if len(os.sched_getaffinity(0)) >= 4:
            # A command elsewhere on the machine still gets cores of its own.
            other = subprocess.run([*_prefix(tmp_path, 2, caps.GIB), sys.executable, '-c', SEEN],
                                   capture_output=True, text=True, timeout=60, env=env)
            assert other.returncode == 0
            assert not set(_records(other.stderr)['started']['cpu_affinity']) & set(cores)
    finally:
        if ending == 'killed':
            os.killpg(holder.pid, signal.SIGKILL)  # No cleanup: whatever the claim files held stays behind.
        else:
            holder.terminate()
        holder.wait(timeout=20)
        holder.stderr.close()
    again = subprocess.run(stranger, capture_output=True, text=True, timeout=60, env=env)
    assert again.returncode == 0, again.stderr[-300:]
    assert _records(again.stderr)['started']['cpu_affinity'] == cores
    assert _records(again.stderr)['started']['before']['enclosing_scope'] is None


def test_readonly_claim_records_name_their_holder_and_are_cleared_at_release(tmp_path):
    result = _supervised(tmp_path, """
import os
scope = caps._ReadonlyScope(root, 2 * caps.PERIOD_US, 1024**3); scope.establish()
files = [caps.CPU_CLAIM_ROOT / f'cpu{cpu}' for cpu in scope.cpus]
start = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]
out = {'held': [path.read_text() for path in files], 'modes': [path.stat().st_mode & 0o777 for path in files],
       'expected': json.dumps({'pid': os.getpid(), 'scope': scope.path.name, 'start': start}, sort_keys=True) + '\\n',
       'nested': 'enclosing_claim' in scope.snapshot()}
scope.cleanup()
out['released'] = [path.read_text() for path in files]
print(json.dumps(out))
""")
    assert result['held'] == [result['expected']] * 2 and result['modes'] == [0o600] * 2
    assert result['released'] == ['', ''] and result['nested'] is False


# A plain process holds the claim locks of a candidate's two cores and writes what the case names into them. The
# candidate is its child, so the holder is a real ancestor; every other fact of a case is forged in the records.
CLAIM_CASE = r"""
import fcntl, os, subprocess
case = CASE
def started(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
def record(pid, start, scope=None, **extra):
    fields = {'pid': pid, 'scope': scope or f'deepmzyme-g2-{pid}-0123456789abcdef', 'start': start, **extra}
    return (json.dumps(fields, sort_keys=True) + '\n').encode()
cores = sorted(os.sched_getaffinity(0))[:2]
other = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
gone = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
gone_start = started(gone.pid); gone.kill(); gone.wait()
me = os.getpid()
own, foreign = record(me, started(me)), record(other.pid, started(other.pid))
content = {'ancestor': [own, own], 'needs_more_cores': [own, own], 'not_private': [own, own],
           'one_core_free': [own, None], 'one_core_free_and_enough': [own, None],
           'not_an_ancestor': [foreign, foreign], 'holders_differ': [own, foreign],
           'restarted_number': [record(me, str(int(started(me)) + 1))] * 2,
           'holder_gone': [record(gone.pid, gone_start)] * 2,
           'record_missing': [own, b''], 'record_damaged': [own[:-3]] * 2, 'record_padded': [own + b' '] * 2,
           'record_extra_field': [record(me, started(me), host='other')] * 2,
           'scope_of_another_process': [record(me, started(me), scope=f'deepmzyme-g2-{other.pid}-0123456789abcdef')] * 2}[case]
caps.CPU_CLAIM_ROOT.mkdir(mode=0o700, exist_ok=True)
held = []
for cpu, data in zip(cores, content):
    if data is not None:
        descriptor = os.open(caps.CPU_CLAIM_ROOT / f'cpu{cpu}', os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.ftruncate(descriptor, 0); os.pwrite(descriptor, data, 0)
        held.append(descriptor)
if case == 'not_private':
    os.chmod(caps.CPU_CLAIM_ROOT / f'cpu{cores[1]}', 0o644)
wanted = {'needs_more_cores': 3, 'one_core_free_and_enough': 1}.get(case, 2)
candidate = '''
import json, os, sys
exec(open(sys.argv[2]).read())
os.sched_setaffinity(0, json.loads(sys.argv[3]))
scope = caps._ReadonlyScope(root, int(sys.argv[4]) * caps.PERIOD_US, 1024**3)
try:
    scope.establish()
    out = {'cpus': list(scope.cpus), 'enclosing': scope.snapshot()['enclosing_scope']}
except caps.CapError as exc:
    out = {'refused': str(exc)}
finally:
    scope.cleanup()
print(json.dumps(out))
'''
driver = root / 'driver.py'; driver.write_text(DRIVER)
env = {key: value for key, value in os.environ.items() if key != caps.SCOPE_ENV}
done = subprocess.run([sys.executable, '-c', candidate, str(root), str(driver), json.dumps(cores), str(wanted)],
                      capture_output=True, text=True, env=env)
other.kill(); other.wait()
out = json.loads(done.stdout.strip().splitlines()[-1]) if done.returncode == 0 else {'failed': done.stderr[-800:]}
out.update(holder=f'deepmzyme-g2-{me}-0123456789abcdef', cores=cores)
print(json.dumps(out))
"""


@pytest.mark.parametrize('case', ['ancestor', 'needs_more_cores', 'one_core_free_and_enough', 'one_core_free',
                                  'not_an_ancestor', 'holders_differ', 'restarted_number', 'holder_gone',
                                  'record_missing', 'record_damaged', 'record_padded', 'record_extra_field',
                                  'scope_of_another_process', 'not_private'])
def test_readonly_parent_is_recognised_only_from_one_live_ancestor_holding_every_inherited_core(tmp_path, case):
    result = _supervised(tmp_path, CLAIM_CASE.replace('CASE', repr(case)).replace('DRIVER', repr(READONLY_DRIVER)))
    assert 'failed' not in result, result
    if case == 'ancestor':
        assert result['cpus'] == result['cores'] and result['enclosing'] == result['holder']
    elif case == 'needs_more_cores':
        assert result['refused'] == 'The enclosing capped command has fewer cores than this one needs.'
    elif case == 'one_core_free_and_enough':
        # A free core is claimed as before; the held one beside it does not make this a nested command.
        assert result['cpus'] == result['cores'][1:] and result['enclosing'] is None
    else:
        assert result['refused'] == TOO_FEW


def test_live_ancestor_counts_only_stable_links_to_the_same_process(monkeypatch):
    """A synthetic process table: every way a number can stop meaning the recorded process ends in a refusal."""
    me = os.getpid()
    table = {me: (90, 'S', '300'), 90: (80, 'S', '200'), 80: (1, 'S', '100'), 70: (1, 'S', '150'), 1: (0, 'S', '1')}
    events = []

    def identity(pid):
        events.append(pid)
        for step in list(changes):
            if len(events) == step:
                table.update(changes.pop(step))
        return table.get(pid)

    monkeypatch.setattr(caps, '_process_identity', identity)
    changes = {}
    assert caps._live_ancestor(90, '200') and caps._live_ancestor(80, '100') and caps._live_ancestor(1, '1')
    assert not caps._live_ancestor(80, '101')          # another process took over the number
    assert not caps._live_ancestor(70, '150')          # alive, but not above this process
    assert not caps._live_ancestor(me, '300')          # a process is not its own ancestor
    assert not caps._live_ancestor(60, '1')            # no such process
    table[80] = (1, 'Z', '100')
    assert not caps._live_ancestor(80, '100')          # the holder has exited
    table[80] = (1, 'S', '100')
    # This process is handed to another parent while the first link is being read.
    events.clear(); changes.update({2: {me: (70, 'S', '300')}})
    assert not caps._live_ancestor(90, '200')
    table[me] = (90, 'S', '300')
    # The first parent exits after its link was read, and its number then names a different process.
    events.clear(); changes.update({3: {90: (80, 'S', '250')}})
    assert not caps._live_ancestor(80, '100')
    table[90] = (80, 'S', '200')
    # A parent chain that never reaches the holder ends without one, even if procfs shows a loop.
    table[80] = (me, 'S', '100')
    assert not caps._live_ancestor(70, '150')


MIB = 1024**2


def _cores(count):
    if len(os.sched_getaffinity(0)) < count:
        pytest.skip(f"needs {count} usable CPU cores")


def _slow_helper(tmp_path, seconds):
    """A second stand-in helper whose supervision poll is slow, so its child is not ended before it has acted."""
    path = tmp_path / 'caps_helper_slow.py'
    path.write_text(READONLY_HELPER.format(src=str(Path(caps.__file__).resolve().parents[1]),
                                           root=str(tmp_path / 'ro'), extra=f'caps.POLL_SECONDS = {seconds}'))
    return path


def test_readonly_descendant_of_another_capped_command_cannot_join_the_holder_of_the_cores(tmp_path):
    """Two real wrappers. A step of the second pins itself onto the first one's cores and starts a capped command."""
    _cores(4)
    _helper(tmp_path)
    slow = _slow_helper(tmp_path, 3.0)
    env = _unscoped_environment()
    holder = subprocess.Popen([*_prefix(tmp_path, 2, caps.GIB), sys.executable, '-c', 'import time; time.sleep(60)'],
                              stderr=subprocess.PIPE, text=True, env=env, start_new_session=True)
    try:
        cores = json.loads(holder.stderr.readline())['container_caps_started']['cpu_affinity']
        inner = [*_prefix(tmp_path, 2, caps.GIB), sys.executable, '-c', SEEN]
        intruder = subprocess.run([sys.executable, str(slow), 'run', '--cgroup-root', str(tmp_path / 'ro'),
                                   '--cpu-cores', '2', '--memory-bytes', str(caps.GIB), '--',
                                   sys.executable, '-c', PINNED, json.dumps(cores), *inner],
                                  capture_output=True, text=True, timeout=60, env=env)
    finally:
        os.killpg(holder.pid, signal.SIGKILL)
        holder.wait(timeout=20)
        holder.stderr.close()
    # Only the second wrapper started, on cores of its own; the command its step started was refused.
    rows = [json.loads(line)['container_caps_started'] for line in intruder.stderr.splitlines()
            if line.startswith('{"container_caps_started"')]
    assert len(rows) == 1 and not set(rows[0]['cpu_affinity']) & set(cores), intruder.stderr[-600:]
    assert 'container caps refused: ' + TOO_FEW in intruder.stderr.splitlines()
    assert intruder.returncode == 125 and 'SEEN' not in intruder.stdout


TWO_AT_ONCE = ("import json,os,subprocess,sys; "
               f"env={{name:os.environ[name] for name in {REBUILT_ENVIRONMENT!r} if name in os.environ}}; "
               "jobs=[subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,"
               "stderr=subprocess.PIPE,text=True) for command in json.loads(sys.argv[1])]; "
               "ends=[job.communicate() for job in jobs]; "
               "print('REPORT '+json.dumps([{'returncode':job.returncode,'stdout':end[0],'stderr':end[1]} "
               "for job,end in zip(jobs,ends)]))")


def test_readonly_nested_commands_started_at_once_share_the_parents_cores_under_separate_memory_limits(tmp_path):
    """Two steps of one unit at the same time: both find the parent, neither reserves a core, each has its own limit."""
    _cores(2)
    _helper(tmp_path)
    greedy = [*_prefix(tmp_path, 2, 96 * MIB), sys.executable, '-c',
              "payload=bytearray(160*1024*1024); import time; time.sleep(30)"]
    modest = [*_prefix(tmp_path, 2, 512 * MIB), sys.executable, '-c',
              "import json,os,time; time.sleep(3); print('SEEN '+json.dumps(sorted(os.sched_getaffinity(0))))"]
    result = subprocess.run([*_prefix(tmp_path, 2, caps.GIB), sys.executable, '-c', TWO_AT_ONCE,
                             json.dumps([greedy, modest])],
                            capture_output=True, text=True, timeout=120, env=_unscoped_environment())
    outer = _records(result.stderr)
    reports = _child_report(result.stdout)
    assert reports is not None, result.stderr[-800:]
    scope, cores = Path(outer['started']['cgroup']).name, outer['started']['cpu_affinity']
    first, second = (_records(report['stderr']) for report in reports)
    for record in (first, second):
        assert record['started'] is not None, record['last']
        assert record['started']['cpu_affinity'] == cores
        assert record['started']['before']['enclosing_scope'] == scope
        assert record['started']['before']['enclosing_claim']['pid'] == int(scope.split('-')[2])
    # The first exceeds its own 96 MiB and is ended alone; the second, below its 512 MiB, is untouched.
    assert reports[0]['returncode'] == 125 and 'OOM' in first['last']
    assert first['finished']['after']['memory_events']['oom_kill'] > 0
    assert reports[1]['returncode'] == 0 and f'SEEN {json.dumps(cores)}' in reports[1]['stdout']
    assert second['finished']['after']['memory_events'] == {'oom': 0, 'oom_kill': 0}
    assert result.returncode == 0 and outer['finished']['after']['memory_events'] == {'oom': 0, 'oom_kill': 0}
    # Neither nested command left a claim of its own: only the parent's files exist, and they are empty again.
    claims = sorted((tmp_path / 'ro' / 'claims').iterdir())
    assert [path.name for path in claims] == sorted(f'cpu{cpu}' for cpu in cores)
    assert all(path.read_bytes() == b'' for path in claims)


def test_readonly_parent_counts_the_memory_of_its_nested_command_once(tmp_path):
    """The parent's supervised sum holds the nested tree exactly once; the nested wrapper measures only its own."""
    _cores(2)
    program = "payload=bytearray(200*1024*1024); import time; time.sleep(2.5)"
    result, outer, report, inner = _nested(tmp_path, (2, caps.GIB), (2, 512 * MIB), [sys.executable, '-c', program])
    assert result.returncode == 0 and report['returncode'] == 0, (outer['last'], inner['last'] if inner else None)
    outer_peak = outer['finished']['after']['memory_peak_bytes']
    inner_peak = inner['finished']['after']['memory_peak_bytes']
    assert 200 * MIB <= inner_peak < 260 * MIB
    assert inner_peak <= outer_peak < 380 * MIB


# A plain process, the candidate's parent, holds the claim lock of the candidate's first core or of both. The
# candidate is pinned to the two cores; each case changes one fact about the claim files or the affinity.
CLAIM_FILES = r"""
import fcntl, os, subprocess
case = CASE
def started(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
usable = sorted(os.sched_getaffinity(0))
cores, spare = usable[:2], usable[2:4]
me = os.getpid()
own = (json.dumps({'pid': me, 'scope': f'deepmzyme-g2-{me}-0123456789abcdef', 'start': started(me)},
                  sort_keys=True) + '\n').encode()
caps.CPU_CLAIM_ROOT.mkdir(mode=0o700, exist_ok=True)
paths = [caps.CPU_CLAIM_ROOT / f'cpu{cpu}' for cpu in cores]
held, locker = [], None
def hold(path):
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.ftruncate(descriptor, 0); os.pwrite(descriptor, own, 0)
    held.append(descriptor)
if case == 'symbolic_link':
    hold(paths[0]); os.symlink(paths[0], paths[1])
elif case == 'second_hard_link':
    hold(paths[0]); os.link(paths[0], paths[1])
elif case == 'locks_of_another_process':
    # The files name this process, but the locks belong to another process that is no ancestor of the candidate.
    for path in paths:
        path.write_bytes(own); path.chmod(0o600)
    program = ("import fcntl,os,sys,time\n"
               "fds=[os.open(p,os.O_RDWR) for p in sys.argv[1:]]\n"
               "[fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB) for fd in fds]\n"
               "print('locked',flush=True); time.sleep(120)")
    locker = subprocess.Popen([sys.executable, '-c', program, *map(str, paths)], stdout=subprocess.PIPE, text=True)
    assert locker.stdout.readline().strip() == 'locked'
else:
    for path in paths:
        hold(path)
candidate = '''
import json, os, sys
from types import SimpleNamespace
exec(open(sys.argv[2]).read())
case, cores, spare = sys.argv[3], json.loads(sys.argv[4]), json.loads(sys.argv[5])
os.sched_setaffinity(0, cores)
if case == 'other_owner':
    real = os.fstat
    def fstat(descriptor):
        info = real(descriptor)
        return SimpleNamespace(st_mode=info.st_mode, st_uid=info.st_uid + 1, st_size=info.st_size,
                               st_nlink=info.st_nlink)
    caps.os.fstat = fstat
calls = []
if case == 'affinity_changes':
    def affinity(pid):
        calls.append(pid)
        return set(cores) if len(calls) == 1 else set(spare)
    caps.os.sched_getaffinity = affinity
scope = caps._ReadonlyScope(root, 2 * caps.PERIOD_US, 1024**3)
try:
    scope.establish()
    out = {'cpus': list(scope.cpus), 'enclosing': scope.snapshot()['enclosing_scope'], 'affinity_reads': len(calls)}
except (caps.CapError, OSError) as exc:
    out = {'refused': str(exc)}
finally:
    scope.cleanup()
print(json.dumps(out))
'''
driver = root / 'driver.py'; driver.write_text(DRIVER)
env = {key: value for key, value in os.environ.items() if key != caps.SCOPE_ENV}
done = subprocess.run([sys.executable, '-c', candidate, str(root), str(driver), case, json.dumps(cores), json.dumps(spare)],
                      capture_output=True, text=True, env=env)
if locker is not None:
    locker.kill(); locker.wait(); locker.stdout.close()
out = json.loads(done.stdout.strip().splitlines()[-1]) if done.returncode == 0 else {'failed': done.stderr[-800:]}
out.update(holder=f'deepmzyme-g2-{me}-0123456789abcdef', cores=cores, spare=spare)
print(json.dumps(out))
"""


@pytest.mark.parametrize('case', ['symbolic_link', 'second_hard_link', 'other_owner', 'affinity_changes'])
def test_readonly_claim_files_name_a_parent_only_as_plain_private_files_for_the_checked_cores(tmp_path, case):
    _cores(4)
    result = _supervised(tmp_path, CLAIM_FILES.replace('CASE', repr(case)).replace('DRIVER', repr(READONLY_DRIVER)))
    assert 'failed' not in result, result
    if case == 'symbolic_link':
        # A link is never followed: the command ends as refused before it reads anything through it.
        assert 'refused' in result and 'cpus' not in result, result
    elif case == 'affinity_changes':
        # The affinity changes after the holder check (a cpuset change, taskset -p). The command stays on the
        # cores whose holder it checked and never takes cores from a later reading.
        assert result['cpus'] == result['cores'] and result['enclosing'] == result['holder'], result
        assert result['affinity_reads'] == 1
    else:
        # The ancestor holds one core and a second name for the same file stands in for the other core, or the
        # files belong to another user: no parent is found.
        assert result.get('refused') == TOO_FEW, result


def test_readonly_claim_records_are_trusted_as_written_by_processes_of_the_same_account(tmp_path):
    """A stated limit, not a protection: a record is believed without proof that the process it names holds the lock.

    Another process of the same account holds the locks, and the files name the candidate's parent. The candidate
    is accepted as a nested command of that parent. Only a process of the same account can arrange this, and such
    a process can as well set the scope variable, which is trusted in the same way.
    """
    _cores(4)
    result = _supervised(tmp_path, CLAIM_FILES.replace('CASE', repr('locks_of_another_process'))
                         .replace('DRIVER', repr(READONLY_DRIVER)))
    assert 'failed' not in result, result
    assert result['cpus'] == result['cores'] and result['enclosing'] == result['holder'], result


def test_readonly_nested_command_is_refused_once_the_holder_of_its_cores_has_ended(tmp_path):
    """The outer wrapper is killed while its nested step runs, so its cores are free for any other command."""
    _cores(2)
    _helper(tmp_path)
    env = _unscoped_environment()
    begun, ended = tmp_path / 'inner.begun', tmp_path / 'inner.ended'
    work = (f"import pathlib,time; pathlib.Path({str(begun)!r}).write_text('1'); time.sleep(12); "
            f"pathlib.Path({str(ended)!r}).write_text('1')")
    nested = [*_prefix(tmp_path, 2, caps.GIB), sys.executable, '-c', work]
    holder = subprocess.Popen([*_prefix(tmp_path, 2, caps.GIB), sys.executable, '-c', NESTED_REPORT, *nested],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, start_new_session=True)
    group = holder.pid
    try:
        first = json.loads(holder.stderr.readline())['container_caps_started']
        cores, scope = first['cpu_affinity'], Path(first['cgroup']).name
        until = time.monotonic() + 30
        while not begun.exists() and time.monotonic() < until:
            time.sleep(.05)
        assert begun.exists()
        os.kill(holder.pid, signal.SIGKILL)  # The wrapper alone: its tree lives on, its locks end.
        holder.wait(timeout=10)
        killed = time.monotonic()
        output = holder.stdout.read()  # Ends when the step's own wrapper has ended and the report is written.
        waited = time.monotonic() - killed
        stranger = subprocess.run([sys.executable, '-c', PINNED, json.dumps(cores), *_prefix(tmp_path, 2, caps.GIB),
                                   sys.executable, '-c', SEEN], capture_output=True, text=True, timeout=60, env=env)
    finally:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        holder.stdout.close()
        holder.stderr.close()
    report = _child_report(output)
    inner = _records(report['stderr'])
    assert inner['started']['before']['enclosing_scope'] == scope
    assert inner['started']['before']['enclosing_claim']['pid'] == int(scope.split('-')[2])
    # The nested wrapper noticed within its next polls, ended its step and reported no clean run.
    assert inner['last'] == ("container caps refused: The capped command that holds this command's cores has ended; "
                             "its outputs cannot qualify.")
    assert report['returncode'] == 125 and inner['finished'] is None
    assert waited < 8 and not ended.exists()
    # The cores are free again, and the next command reserves them as a command without a parent.
    assert stranger.returncode == 0, stranger.stderr[-300:]
    assert _records(stranger.stderr)['started']['cpu_affinity'] == cores
    assert _records(stranger.stderr)['started']['before']['enclosing_scope'] is None


def test_readonly_holder_is_checked_in_every_poll_and_before_every_record(monkeypatch):
    """The holder is looked up by process number and start time; a nested command without one ends its tree."""
    me = os.getpid()
    start = Path(f'/proc/{me}/stat').read_text().rsplit(')', 1)[1].split()[19]
    ended = subprocess.Popen([sys.executable, '-c', 'pass'])
    until = time.monotonic() + 20
    while (Path(f'/proc/{ended.pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
           and time.monotonic() < until):
        time.sleep(.02)
    exited = Path(f'/proc/{ended.pid}/stat').read_text().rsplit(')', 1)[1].split()
    scope = object.__new__(caps._ReadonlyScope)
    scope.polls, kills = 0, []
    monkeypatch.setattr(scope, 'kill', lambda: kills.append(True) or 0)
    try:
        # Not yet collected by its parent, restarted number, no such process: none of them is the holder any more.
        for claim in ({'pid': ended.pid, 'start': exited[19]}, {'pid': me, 'start': str(int(start) + 1)},
                      {'pid': 2**22 + 1, 'start': start}):
            assert exited[0] == 'Z'
            scope.parent_claim = claim
            for check in (scope.snapshot, scope.enforce, scope._require_holder):
                with pytest.raises(caps.CapError, match="holds this command's cores has ended"):
                    check()
        assert len(kills) == 9 and scope.polls == 3
    finally:
        ended.wait()
    # A live holder passes, and a command that found its parent in the environment has no holder to look up.
    scope.parent_claim = {'pid': me, 'start': start}
    scope._require_holder()
    scope.parent_claim = None
    monkeypatch.setattr(caps, '_process_identity', lambda pid: pytest.fail('no holder is recorded'))
    scope._require_holder()
    assert len(kills) == 9


# Every probe step returns only when its whole time limit has passed: the slowest host that breaks no single limit.
SLOWEST_PROBE = """
import time as _time
from types import SimpleNamespace
skew, limits, run = [0.0], [], caps._run
caps.time = SimpleNamespace(monotonic=lambda: _time.monotonic() + skew[0], sleep=_time.sleep, time=_time.time)
def slowest(*args, **kwargs):
    limits.append(kwargs['timeout'])
    result = run(*args, **kwargs)
    skew[0] += kwargs['timeout']
    return result
caps._run = slowest
try:
    caps.probe(root); refusal = None
except caps.CapError as exc:
    refusal = str(exc)
print(json.dumps({'limits': limits, 'refusal': refusal}))
"""
# Host admission waits this long for the whole probe (workstation.py, both probe calls).
ADMISSION_SECONDS = 90
PROBE_OUT_OF_TIME = 'The caps probe exceeded its time limit.'


def test_readonly_probe_steps_together_end_inside_the_time_host_admission_waits(tmp_path):
    """The step limits add up to more than admission waits, so the probe keeps one limit for all of them."""
    result = _supervised(tmp_path, SLOWEST_PROBE)
    limits = result['limits']
    # The enforcement and profile steps used up the probe's time; the nested step was never started. Each step
    # had at most its own limit, and a late one only what was left.
    assert limits[0] == 30 and len(limits) <= 4, limits
    assert all(0 < got <= own for got, own in zip(limits, (30, 15, 15, 15))) and limits[-1] < 15, limits
    assert sum(limits) <= caps.PROBE_SECONDS and result['refusal'] == PROBE_OUT_OF_TIME
    # What is left covers the end of a step that ran out of time: kill, wait and cleanup, five seconds each.
    assert caps.PROBE_SECONDS + 3 * 5 < ADMISSION_SECONDS


PROBE_WITH_A_STUCK_NESTED_COMMAND = """
import contextlib, fcntl, io, os, time
caps.NESTED_PROBE_SECONDS = 6
seen, run = [], caps._run
def recorded(*args, **kwargs):
    result = run(*args, **kwargs)
    seen.append(json.loads(json.dumps(result)))
    return result
caps._run = recorded
errors = io.StringIO()
begin = time.monotonic()
with contextlib.redirect_stderr(errors):
    code = caps.main(['probe', '--cgroup-root', str(root)])
elapsed = time.monotonic() - begin
left = [pid for pid, state in caps._descendants(os.getpid()).items() if state != 'Z']
claims = sorted(caps.CPU_CLAIM_ROOT.iterdir())
free = []
for path in claims:
    descriptor = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB); free.append(True)
    except OSError:
        free.append(False)
    os.close(descriptor)
print(json.dumps({'code': code, 'stderr': errors.getvalue(), 'elapsed': elapsed, 'left': left, 'outer': seen[-1],
                  'commands': len(seen), 'content': [path.read_text() for path in claims], 'free': free}))
"""


@pytest.mark.parametrize('fault,returncode,said', [
    ("import time; time.sleep(600)", -signal.SIGTERM, ''),
    ("arguments[-1] = 'import time; time.sleep(600)'", 125,
     'container caps refused: Resource-cap wrapper interrupted by signal 15.'),
], ids=['helper_never_returns', 'inner_command_never_returns'])
def test_readonly_probe_ends_a_stuck_nested_command_itself_and_leaves_nothing_behind(tmp_path, fault, returncode, said):
    """The outer command stops its inner helper inside its own limit; the helper then removes what it started."""
    result = _supervised(tmp_path, PROBE_WITH_A_STUCK_NESTED_COMMAND, helper=fault)
    assert result['code'] == 125
    assert result['stderr'] == 'container caps refused: The nested cap probe exceeded its time limit.\n'
    # The outer command itself ended in time and reported the helper it had stopped. Its wrapper found no
    # process left to end: the stopped helper had already ended its own child.
    outer = result['outer']
    report = json.loads(outer['stdout'].strip().splitlines()[-1])
    assert result['commands'] == 5 and outer['returncode'] == 0 and outer['descendant_cleanup_count'] == 0
    assert report['timed_out'] is True and report['returncode'] == returncode
    assert _records(report['stderr'])['last'] == said and _records(report['stderr'])['finished'] is None
    assert result['left'] == [] and all(text == '' for text in result['content']) and all(result['free'])
    assert result['elapsed'] < 30
    # Nothing that was started for this case is still alive anywhere on the machine.
    stray = []
    for entry in os.listdir('/proc'):
        if entry.isdigit() and int(entry) != os.getpid():
            try:
                command = Path(f'/proc/{entry}/cmdline').read_bytes().decode(errors='replace')
            except OSError:
                continue
            if str(tmp_path) in command:
                stray.append((entry, command[:120]))
    assert not stray, stray


# How the kernel fakes know the outer command of the nested probe: it starts the inner helper under this name.
NESTED_OUTER = "job=subprocess.Popen("


def _nested_report(argv, membership):
    """What the nested probe's outer command prints, with its inner command run through the same kernel fake."""
    inner = ast.literal_eval(re.search(r"job=subprocess\.Popen\((\[.*\]),env=env", argv[-1])[1])
    assert inner[:4] == [*caps._helper_command(), 'run', '--cgroup-root'] and inner[9] == '--'
    errors = io.StringIO()
    with contextlib.redirect_stderr(errors):
        try:
            code = caps._run(Path(inner[4]), int(inner[6]) * caps.PERIOD_US, int(inner[8]), inner[10:],
                             emit=True)['returncode']
        except caps.CapError as exc:
            print(f'container caps refused: {exc}', file=sys.stderr)
            code = 125
    finished = _records(errors.getvalue())['finished']
    return json.dumps({'environment': ['HOME', 'PATH'], 'returncode': code, 'timed_out': False,
                       'stderr': errors.getvalue(),
                       'stdout': membership(Path(finished['cgroup']).name) if finished else ''}) + '\n'


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
            elif NESTED_OUTER in self.argv[-1]:
                return _nested_report(self.argv, lambda name: f"0::/{name}\n"), ""
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
    # Four separate commands as before, then the nested pair: an outer scope and the inner scope started inside it.
    assert len(created) == len(children) == 6 and not any(path.exists() for path in created)
    assert evidence["profiles"]["esm1b"]["limits"]["memory.max"] == str(24*caps.GIB)
    assert evidence["memory_oom_test"]["returncode"] == -signal.SIGKILL
    assert evidence["cpu_throttle_test"]["after"]["cpu_stat"]["nr_throttled"] > 0
    outer = evidence["nested_test"]
    assert outer["limits"]["memory.max"] == str(24*caps.GIB) and outer["limits"]["cpu.max"] == "400000 100000"
    assert outer["nested"]["limits"]["memory.max"] == str(8*caps.GIB)
    assert outer["nested"]["limits"]["cpu.max"] == "400000 100000" and outer["nested"]["cgroup"] != outer["cgroup"]
    assert [str(path) for path in created[4:]] == [outer["cgroup"], outer["nested"]["cgroup"]]


@pytest.mark.parametrize("backend", ["kernel", "kernel_v1"])
def test_probe_fails_on_a_kernel_backend_whose_nested_command_is_refused(request, monkeypatch, backend):
    """The probe now also rejects a host on which a capped command cannot start inside another one."""
    root = request.getfixturevalue(backend)[0]
    run = caps._run

    def refuse_nested(*args, **kwargs):
        if kwargs.get("emit"):
            raise caps.CapError("The v1 mount must expose only this process's own container cgroup.")
        return run(*args, **kwargs)

    monkeypatch.setattr(caps, "_run", refuse_nested)
    with pytest.raises(caps.CapError, match="inside another one with a rebuilt environment did not run: The v1 mount"):
        caps.probe(root)


def test_nested_probe_time_limit_is_a_short_refusal(kernel, monkeypatch):
    root, created, _, _ = kernel
    communicate = caps.subprocess.Popen.communicate

    def slow(child, timeout=None):
        if NESTED_OUTER in child.argv[-1]:
            raise subprocess.TimeoutExpired(child.argv, timeout)
        return communicate(child, timeout)

    monkeypatch.setattr(caps.subprocess.Popen, "communicate", slow)
    with pytest.raises(caps.CapError, match=r"^The nested cap probe exceeded its time limit\.$"):
        caps.probe(root)
    assert len(created) == 5 and not any(path.exists() for path in created)


@pytest.mark.parametrize("backend", ["kernel", "kernel_v1"])
def test_probe_steps_together_end_inside_the_time_host_admission_waits(request, monkeypatch, backend):
    """Each step returns only when its whole time limit has passed. The nested step then gets what is left."""
    root, created = request.getfixturevalue(backend)[:2]
    clock, limits, pause, run = [1000.0], [], [0.0], caps._run
    monkeypatch.setattr(caps, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=lambda seconds: None,
                                                      time=time.time))

    def slowest(*args, **kwargs):
        result = run(*args, **kwargs)
        if "timeout" in kwargs:  # The inner command of the nested pair has no limit of its own.
            limits.append(kwargs["timeout"])
            clock[0] += kwargs["timeout"] + pause[0]
        return result

    monkeypatch.setattr(caps, "_run", slowest)
    evidence = caps.probe(root)
    assert caps.validate_probe_evidence(evidence) == evidence
    assert limits == [15, 15, 15, 15, caps.PROBE_SECONDS - 60] and limits[-1] < caps.NESTED_PROBE_SECONDS
    assert sum(limits) == caps.PROBE_SECONDS and caps.PROBE_SECONDS + 3 * 5 < ADMISSION_SECONDS
    # A host that also loses time between its steps has nothing left for the nested step: a short refusal.
    del limits[:]
    pause[0] = 3.0
    with pytest.raises(caps.CapError, match=r"^The caps probe exceeded its time limit\.$"):
        caps.probe(root)
    assert len(limits) == 4 and sum(limits) <= caps.PROBE_SECONDS
    assert not any(path.exists() for path in created)


@pytest.mark.parametrize("backend", ["kernel", "kernel_v1"])
def test_kernel_backends_run_a_command_without_claims_or_ancestry(request, monkeypatch, tmp_path, backend):
    """Core claims and parent recognition belong to the read-only backend alone."""
    root = request.getfixturevalue(backend)[0]

    def unused(*args, **kwargs):
        raise AssertionError("a kernel cgroup backend consulted the read-only parent recognition")

    for name in ("_read_claim", "_write_claim", "_claim_record", "_live_ancestor", "_process_identity"):
        monkeypatch.setattr(caps, name, unused, raising=False)
    monkeypatch.setattr(caps, "CPU_CLAIM_ROOT", tmp_path / "claims-must-stay-absent")
    monkeypatch.setenv(caps.SCOPE_ENV, "deepmzyme-g2-1-0123456789abcdef")
    result = caps._run(root, 2*caps.PERIOD_US, 8*caps.GIB, [sys.executable, "-c", "pass"])
    expected = {"cgroup", "limits", "before", "backend", "memory_limit_bytes", "after", "returncode",
                "membership_verified", "cleanup_verified"}
    if backend == "kernel_v1":
        expected |= {"controller_cgroups", "memsw_accounting_available", "descendant_cleanup_count"}
    assert set(result) == expected and "enclosing_scope" not in result["before"]
    assert not (tmp_path / "claims-must-stay-absent").exists()


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
    lambda item: item.pop("nested_test"),
    lambda item: item["nested_test"].pop("nested"),
    lambda item: item["nested_test"]["nested"].update({"helper_returncode": 125}),
    lambda item: item["nested_test"]["nested"].update({"returncode": 1}),
    lambda item: item["nested_test"]["nested"]["limits"].update({"memory.max": str(24*caps.GIB)}),
    lambda item: item["nested_test"]["nested"].update({"membership_verified": False}),
    lambda item: item["nested_test"]["nested"].update({"cgroup": item["nested_test"]["cgroup"]}),
    lambda item: item["nested_test"]["nested"].update({"stdout": "0::/\n"}),
    lambda item: item["nested_test"]["nested"]["after"]["memory_events"].update({"oom_kill": 1}),
    lambda item: item["nested_test"]["environment"].append("DEEPMZYME_CAPS_SCOPE"),
    lambda item: item["nested_test"].update({"cleanup_verified": False}),
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
    monkeypatch.setattr(caps, '_v1_roots', lambda path, **kw: {'cpu': cpu, 'memory': memory})
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
            nested = None
            if NESTED_OUTER in self.argv[-1]:
                # This fake lets the inner command create its own leaves. A real writable v1 host is expected to
                # refuse it (the caller no longer sits in the container's own cgroup), which fails the probe.
                nested = _nested_report(self.argv, lambda name: f'3:cpu,cpuacct:/docker/owned/{name}\n5:memory:/docker/owned/{name}\n')
            for path in self.scope.paths.values():
                (path / 'cgroup.procs').write_text('')
            if nested is not None:
                return nested, ''
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
    # Four separate commands as before, then the nested pair; each command has a cpu leaf and a memory leaf.
    assert len(created) == 12 and len(children) == 6 and not any(p.exists() for p in created)
    inner = evidence['nested_test']['nested']
    assert inner['limits']['memory']['memory.limit_in_bytes'] == str(8*caps.GIB)
    assert inner['limits']['cpu']['cpu.cfs_quota_us'] == '400000' and inner['cgroup'] != evidence['nested_test']['cgroup']
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
    lambda e: e.pop('nested_test'),
    lambda e: e['nested_test']['nested']['limits']['memory'].update({'memory.limit_in_bytes': str(24*caps.GIB)}),
    lambda e: e['nested_test']['nested']['controller_cgroups'].update({'memory': '/host/other'}),
    lambda e: e['nested_test']['nested'].update({'helper_returncode': 125}),
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
