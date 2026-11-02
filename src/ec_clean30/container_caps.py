"""Direct kernel cgroup caps for the G2 container route; no advisory fallback.

The provider must delegate v2 cpu/memory controllers or container-scoped v1
controller mounts on a genuinely swap-free host. This module changes only new
owned leaves, never ancestor limits or mounts. v1 rejects any host swap backing
or swap I/O during a child; swappiness=0 alone is not a zero-swap guarantee.
"""
from __future__ import annotations

import argparse
import ctypes
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


def _pidfd_open(pid: int) -> int:
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    # Conda builds may omit the Python wrapper despite a modern runtime libc.
    function = ctypes.CDLL(None, use_errno=True).pidfd_open
    function.argtypes, function.restype = [ctypes.c_int, ctypes.c_uint], ctypes.c_int
    descriptor = function(pid, 0)
    if descriptor < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return descriptor


def _pidfd_available() -> bool:
    return (hasattr(os, 'pidfd_open') or hasattr(ctypes.CDLL(None), 'pidfd_open')) and hasattr(signal, 'pidfd_send_signal')


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


def _no_host_swap() -> dict[str, Any]:
    """v1 has no separate swap.max; require no swap backing anywhere on the host."""
    for target in ('/proc/meminfo', '/proc/swaps', '/proc/vmstat'):
        candidates = []
        for line in Path('/proc/self/mountinfo').read_text().splitlines():
            left, right = line.split(' - ', 1)
            mount = Path(_mount_unescape(left.split()[4]))
            if Path(target) == mount or mount in Path(target).parents:
                candidates.append((len(mount.parts), right.split()[0], _mount_unescape(left.split()[3]), str(mount)))
        deepest = max((entry[0] for entry in candidates), default=-1)
        covering = [entry for entry in candidates if entry[0] == deepest]
        _require(len(covering) == 1 and covering[0][1:] == ('proc', '/', '/proc') and Path(target).resolve() == Path(target),
                 'The no-swap host evidence must come from genuine, unobscured procfs.')
    memory = {line.split(':', 1)[0]: line.split(':', 1)[1].strip()
              for line in Path('/proc/meminfo').read_text().splitlines()}
    swaps = Path('/proc/swaps').read_text().splitlines()
    _require(memory.get('SwapTotal') == memory.get('SwapFree') == '0 kB' and len(swaps) == 1
             and swaps[0].split() == ['Filename', 'Type', 'Size', 'Used', 'Priority'],
             'cgroup v1 requires a host with no configured swap backing; swappiness=0 alone is insufficient.')
    vmstat = _fields(Path('/proc/vmstat'))
    return {'swap_total_kib': 0, 'swap_devices': [], 'basis': 'host_has_no_swap_backing',
            'pswpin': vmstat['pswpin'], 'pswpout': vmstat['pswpout'],
            'boot_id': _read(Path('/proc/sys/kernel/random/boot_id'))}


def _v1_roots(root: Path) -> dict[str, Path]:
    _require(_pidfd_available()
             and hasattr(os, 'eventfd') and hasattr(os, 'eventfd_read'),
             'v1 requires race-safe pidfd cleanup and eventfd OOM notifications.')
    memberships = {}
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        _, controllers, member = line.split(':', 2)
        for controller in controllers.split(','):
            memberships[controller] = member
    found = {}
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left, right = line.split(' - ', 1)
        fields, details = left.split(), right.split()
        mount = Path(_mount_unescape(fields[4]))
        if details[0] != 'cgroup' or mount.parent != root:
            continue
        for controller in ('cpu', 'memory'):
            if controller not in details[2].split(','):
                continue
            member, mounted_root = memberships.get(controller), _mount_unescape(fields[3])
            _require(mount.resolve() == mount and 'rw' in fields[5].split(',')
                     and not os.statvfs(mount).f_flag & os.ST_RDONLY,
                     'Both v1 controller mounts must be canonical and writable.')
            # A host-wide cgroup mount is never an acceptable delegation.
            _require(mounted_root != '/' and member in (mounted_root, '/')
                     and str(os.getpid()) in _read(mount / 'cgroup.procs').split(),
                     'The v1 mount must expose only this process\'s own container cgroup.')
            _require(controller not in found, 'Ambiguous v1 controller mount.')
            found[controller] = mount
    _require(set(found) == {'cpu', 'memory'}, 'Writable container-scoped v1 cpu and memory mounts are required.')
    _no_host_swap()
    return found


def validate_root(root: Path) -> Path:
    root = Path(root)
    _require(root.is_absolute() and root.is_dir(), "The delegated cgroup root must be an existing absolute directory.")
    _require(root == root.resolve(), "The delegated cgroup root must be canonical and contain no symlinks.")
    if not (root / 'cgroup.type').exists():
        _v1_roots(root)
        return root
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

    def kill(self) -> None:
        _write(self.path / 'cgroup.kill', '1')


class _V1Scope:
    """Two container-scoped controller leaves; no ancestor limit is modified."""
    backend = 'cgroup_v1'

    def __init__(self, root: Path, cpu_quota_us: int, memory_bytes: int):
        name = f'deepmzyme-g2-{os.getpid()}-{secrets.token_hex(8)}'
        self.path = root / name  # Common logical identity across the split controllers.
        self.paths = {key: path / name for key, path in _v1_roots(root).items()}
        self.created = []
        self.memory_bytes = memory_bytes
        self.raw_limits = {'cpu': {'cpu.cfs_quota_us': str(cpu_quota_us), 'cpu.cfs_period_us': str(PERIOD_US)},
                           'memory': {'memory.limit_in_bytes': str(memory_bytes),
                                      'memory.swappiness': '0', 'memory.use_hierarchy': '1'}}
        self.limits = self.raw_limits
        self.host_swap = _no_host_swap()
        self.oom_events = 0
        self.kill_count = 0
        self.oom_descriptor = self.event_descriptor = None
        self.memsw_available = False

    def establish(self) -> None:
        for controller, path in self.paths.items():
            path.mkdir(mode=0o700)
            self.created.append(path)
            _require(not _read(path / 'cgroup.procs'), 'The new v1 leaf unexpectedly contains processes.')
            if controller == 'cpu' and (path / 'cpu.cfs_burst_us').exists():
                self.raw_limits['cpu']['cpu.cfs_burst_us'] = '0'
            if controller == 'memory':
                self.memsw_available = (path / 'memory.memsw.limit_in_bytes').is_file()
                if self.memsw_available:
                    self.raw_limits['memory']['memory.memsw.limit_in_bytes'] = str(self.memory_bytes)
            for name, value in self.raw_limits[controller].items():
                if _read(path / name) != value:
                    _write(path / name, value)
        _write(self.paths['memory'] / 'memory.oom_control', '0')
        self.event_descriptor = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self.oom_descriptor = os.open(self.paths['memory'] / 'memory.oom_control', os.O_RDONLY | os.O_CLOEXEC)
        _write(self.paths['memory'] / 'cgroup.event_control', f'{self.event_descriptor} {self.oom_descriptor}')
        self.snapshot()

    def snapshot(self, *, collect_events: bool = True) -> dict[str, Any]:
        host_swap = _no_host_swap()
        _require(host_swap == self.host_swap, 'Host swap configuration or swap I/O counters changed during the capped command.')
        raw = {key: {name: _read(self.paths[key] / name) for name in controls}
               for key, controls in self.raw_limits.items()}
        _require(raw == self.raw_limits, 'A v1 CPU, RAM, hierarchy or no-swap control differs from its request.')
        cpu = _fields(self.paths['cpu'] / 'cpu.stat')
        memory = self.paths['memory']
        oom = _fields(memory / 'memory.oom_control')
        _require(oom['oom_kill_disable'] == 0 and 'oom_kill' in oom,
                 'v1 requires an enabled kernel OOM killer and observable OOM-kill counter.')
        stat = _fields(memory / 'memory.stat')
        while collect_events and self.event_descriptor is not None:
            try:
                self.oom_events += os.eventfd_read(self.event_descriptor)
            except BlockingIOError:
                break
        swap = stat.get('total_swap', stat.get('swap'))
        _require(swap in (0, None) and (not self.memsw_available or swap is not None),
                 'The v1 group has nonzero swap or inconsistent swap-accounting evidence.')
        return {'limits': self.limits, 'backend': self.backend, 'raw_limits': raw,
                'zero_swap_basis': host_swap, 'oom_control': oom,
                'memsw_accounting_available': self.memsw_available, 'swap_stat_bytes': swap,
                'swap_current_source': 'kernel_counter' if swap is not None else 'host_no_swap_invariant',
                'cpu_stat': {**cpu, 'throttled_usec': cpu['throttled_time'] // 1000},
                'memory_failcnt': int(_read(memory / 'memory.failcnt')),
                'memory_events': {'oom': self.oom_events, 'oom_kill': oom['oom_kill']},
                'swap_current_bytes': 0, 'memory_current_bytes': int(_read(memory / 'memory.usage_in_bytes'))}

    def attach(self) -> None:
        for path in self.paths.values():
            _write(path / 'cgroup.procs', str(os.getpid()))
            _require(str(os.getpid()) in _read(path / 'cgroup.procs').split(),
                     'The child did not join both v1 capped leaves before exec.')
        _require(os.sched_getscheduler(0) == os.SCHED_OTHER,
                 'CPU bandwidth containment requires the ordinary fair scheduler.')
        self.snapshot(collect_events=False)

    def _members(self) -> set[int]:
        members = set()
        for path in self.created:
            for group in [path, *(p for p in path.rglob('*') if p.is_dir())]:
                members.update(int(pid) for pid in _read(group / 'cgroup.procs').split())
        return members

    def kill(self) -> None:
        # v1 lacks cgroup.kill. Enumerate both owned subtrees until they are empty;
        # recheck membership immediately before each kill to exclude recycled PIDs.
        until = time.monotonic() + 5
        while True:
            members = self._members()
            if not members:
                return
            _require(time.monotonic() < until, 'v1 capped descendants remain after recursive termination.')
            for pid in members:
                _require(pid > 0 and pid != os.getpid(), 'Refusing an invalid or wrapper PID in v1 cleanup.')
                try:
                    descriptor = _pidfd_open(pid)
                    try:
                        lines = Path(f'/proc/{pid}/cgroup').read_text().splitlines()
                        if any(self.path.name in Path(line.split(':', 2)[2]).parts for line in lines):
                            signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                            self.kill_count += 1
                    finally:
                        os.close(descriptor)
                except ProcessLookupError:
                    pass
                except FileNotFoundError:
                    pass
            time.sleep(.02)

    def cleanup(self) -> None:
        if not self.created:
            return
        self.kill()
        for descriptor in (self.event_descriptor, self.oom_descriptor):
            if descriptor is not None:
                os.close(descriptor)
        self.event_descriptor = self.oom_descriptor = None
        for path in reversed(self.created):
            for child in sorted((p for p in path.rglob('*') if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
                child.rmdir()
            path.rmdir()


@contextmanager
def _scope(root: Path, cpu_quota_us: int, memory_bytes: int) -> Iterator[_Scope]:
    _require(type(cpu_quota_us) is int and cpu_quota_us > 0, "A positive CPU quota is required.")
    _require(type(memory_bytes) is int and memory_bytes > 0, "A positive memory limit is required.")
    root = validate_root(root)
    scope = (_Scope if (root / 'cgroup.type').exists() else _V1Scope)(root, cpu_quota_us, memory_bytes)
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
                    "backend": getattr(scope, 'backend', 'cgroup_v2'),
                    "memory_limit_bytes": memory_bytes}
        if isinstance(scope, _V1Scope):
            evidence['controller_cgroups'] = {key: str(path) for key, path in scope.paths.items()}
            evidence['memsw_accounting_available'] = scope.memsw_available
        if emit:
            print(json.dumps({"container_caps_started": evidence}, sort_keys=True), file=sys.stderr, flush=True)
        child = subprocess.Popen(list(argv), preexec_fn=scope.attach,
                                 stdout=subprocess.PIPE if capture else None,
                                 stderr=subprocess.PIPE if capture else None, text=capture)
        try:
            if isinstance(scope, _V1Scope):
                until = time.monotonic() + timeout if timeout is not None else None
                while True:
                    try:
                        stdout, stderr = child.communicate(timeout=min(.25, max(.001, until-time.monotonic())) if until else .25)
                        break
                    except subprocess.TimeoutExpired:
                        _require(_no_host_swap() == scope.host_swap,
                                 'Host swap configuration or swap I/O counters changed during the capped command.')
                        _require(until is None or time.monotonic() < until, 'The capped v1 command exceeded its timeout.')
            else:
                stdout, stderr = child.communicate(timeout=timeout)
        except BaseException:
            scope.kill()
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
    if isinstance(scope, _V1Scope):
        evidence['descendant_cleanup_count'] = scope.kill_count
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
    if memory['backend'] == 'cgroup_v1':
        _require(memory['after']['memory_failcnt'] > memory['before']['memory_failcnt']
                 and memory['after']['memory_events']['oom'] > memory['before']['memory_events']['oom'],
                 'The v1 OOM proof must hit its own limit and emit a memcg OOM notification.')
    memory["status"] = "passed"
    profiles = {}
    for name, cores, gib in (("fit", 2, 8), ("esm1b", 4, 24)):
        program = "from pathlib import Path; print(Path('/proc/self/cgroup').read_text())"
        if not (root / 'cgroup.type').exists():
            program = ("from pathlib import Path; import subprocess,sys; "
                "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],"
                "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True); "
                "own=Path('/proc/self/cgroup').read_text(); "
                "assert Path('/proc/'+str(child.pid)+'/cgroup').read_text()==own; "
                "print(own); print('descendant_membership_verified='+str(child.pid))")
        evidence = _run(root, cores * PERIOD_US, gib * GIB,
                        [sys.executable, "-c", program],
                        timeout=15, capture=True)
        _require(evidence["returncode"] == 0 and Path(evidence["cgroup"]).name in evidence["stdout"],
                 "The profile probe did not confirm its actual cgroup membership.")
        evidence.update({"cpu_cores": cores, "status": "passed"})
        if evidence['backend'] == 'cgroup_v1':
            _require('descendant_membership_verified=' in evidence['stdout']
                     and evidence['descendant_cleanup_count'] > 0,
                     'v1 profile qualification must exercise actual inherited membership and pidfd descendant cleanup.')
        profiles[name] = evidence
    return {"format": PROBE_FORMAT, "status": "passed", "cgroup_root": str(root),
            'backend': 'cgroup_v2' if (root / 'cgroup.type').exists() else 'cgroup_v1',
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
            backend = evidence.get('backend', 'cgroup_v2')
            _require(backend in ('cgroup_v1', 'cgroup_v2')
                     and record.get('backend', 'cgroup_v2') == backend, 'Cgroup backend evidence differs.')
            if backend == 'cgroup_v1':
                expected = {'cpu': {'cpu.cfs_quota_us': str(quota), 'cpu.cfs_period_us': str(PERIOD_US)},
                            'memory': {'memory.limit_in_bytes': str(memory),
                                       'memory.swappiness': '0', 'memory.use_hierarchy': '1'}}
                _require(type(record['memsw_accounting_available']) is bool, 'The v1 accounting availability is not explicit.')
                if record['memsw_accounting_available']:
                    expected['memory']['memory.memsw.limit_in_bytes'] = str(memory)
                if 'cpu.cfs_burst_us' in record['limits']['cpu']:
                    expected['cpu']['cpu.cfs_burst_us'] = '0'
                paths = record['controller_cgroups']
                _require(set(paths) == {'cpu', 'memory'}
                         and all(Path(path).parent.parent == root and Path(path).name == Path(record['cgroup']).name
                                 for path in paths.values()), 'Split v1 cgroup identity is inconsistent.')
            else:
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
                if backend == 'cgroup_v1':
                    basis = snapshot['zero_swap_basis']
                    _require(snapshot['raw_limits'] == expected and snapshot['oom_control']['oom_kill_disable'] == 0
                             and snapshot['memsw_accounting_available'] == record['memsw_accounting_available']
                             and snapshot['swap_stat_bytes'] in (None, 0)
                             and (not record['memsw_accounting_available'] or snapshot['swap_stat_bytes'] is not None)
                             and snapshot['swap_current_source'] == ('kernel_counter' if snapshot['swap_stat_bytes'] is not None else 'host_no_swap_invariant')
                             and basis['basis'] == 'host_has_no_swap_backing' and basis['swap_total_kib'] == 0
                             and basis['swap_devices'] == []
                             and basis['boot_id'] == evidence['boot_id']
                             and all(type(basis[key]) is int and basis[key] >= 0 for key in ('pswpin', 'pswpout'))
                             and basis == record['before']['zero_swap_basis'],
                             'v1 requires exact raw controls and unchanged genuine no-swap host evidence.')
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
            if evidence.get('backend') == 'cgroup_v1':
                _require('descendant_membership_verified=' in record['stdout']
                         and type(record['descendant_cleanup_count']) is int and record['descendant_cleanup_count'] > 0,
                         'The v1 profile lacks exercised descendant cleanup.')
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
        if evidence.get('backend') == 'cgroup_v1':
            _require(memory['after']['memory_failcnt'] > memory['before']['memory_failcnt']
                     and memory['after']['memory_events']['oom'] > memory['before']['memory_events']['oom'],
                     'The v1 OOM proof lacks a local limit hit and genuine OOM notification.')
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
