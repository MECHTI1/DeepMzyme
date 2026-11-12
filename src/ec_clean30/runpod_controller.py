"""Bounded allocation or named adoption with a workstation stop watchdog.

The watchdog survives the submitting process, but is not a provider-side stop
guarantee: workstation power/network/authentication loss can delay shutdown.
Stopped allocations are deleted only after independent preservation readback.
No method replaces, restarts, or resumes an allocation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import threading
import time
from typing import Any, Mapping
import uuid
import urllib.parse
import urllib.request

class RunpodError(RuntimeError):
    """A configuration, ownership, budget, or shutdown check failed."""


class StockUnavailable(RunpodError):
    """A fresh read lost compatible capacity before a provider POST."""


class _HeartbeatGuard:
    """A same-process ticker; the independent watchdog remains the stop owner."""

    def __init__(self, controller: 'RunpodController', interval: float):
        self.controller, self.interval = controller, interval
        self.pid = os.getpid()
        self.error: str | None = None
        self.finished = threading.Event()
        self.thread: threading.Thread | None = None

    def check(self) -> None:
        _require(os.getpid() == self.pid, "Continuous heartbeat belongs to another coordinator process.")
        _require(self.error is None, f"Continuous owner heartbeat failed; admission is closed: {self.error}")
        if self.thread is not None and not self.finished.is_set():
            _require(self.thread.is_alive(), "Continuous owner heartbeat ended unexpectedly; admission is closed.")

    def _tick(self) -> None:
        try:
            self.controller.heartbeat()
        except BaseException as exc:
            self.error = self.controller._redact(f'{type(exc).__name__}: {exc}')
            # A writable intent asks the independent guard to stop immediately.
            # If bookkeeping itself fails, its stale-owner rule still applies.
            try:
                self.controller._cleanup_update(stop_requested=True, heartbeat_error=self.error)
            except BaseException:
                pass
            self.finished.set()

    def _run(self) -> None:
        while not self.finished.wait(self.interval):
            self._tick()

    def start(self) -> None:
        self._tick()
        self.check()
        thread = threading.Thread(target=self._run, name='runpod-owner-heartbeat', daemon=True)
        # Publish only a started thread: heartbeat() itself checks this guard.
        thread.start()
        self.thread = thread

    def close(self) -> None:
        _require(self.controller._cleanup_state().get('phase') in {'stopped', 'deleted'},
                 "Continuous heartbeat cannot end before verified shutdown.")
        self.finished.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
            _require(not self.thread.is_alive(), "Continuous heartbeat did not terminate after verified shutdown.")


def _require(value: Any, message: str) -> None:
    if not value:
        raise RunpodError(message)


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial")
    with temporary.open("x") as handle:
        os.chmod(temporary, 0o600)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _utc(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace("+00:00", "Z")


def _epoch(value: Any) -> float:
    _require(isinstance(value, str), "A timezone-aware timestamp is required.")
    # Older runpodctl builds render Go's timestamp string.
    value = re.sub(r" \+0000 UTC$", "+00:00", value)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require(parsed.tzinfo is not None, "A timezone-aware timestamp is required.")
    return parsed.timestamp()


def _number(value: Any, name: str, *, zero: bool = False) -> float:
    _require(type(value) in (int, float) and math.isfinite(value) and (value >= 0 if zero else value > 0),
             f"{name} must be finite and {'nonnegative' if zero else 'positive'}.")
    return float(value)


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    _require(config.get("provider") == "runpod", "An explicit Runpod provider is required.")
    settings = dict(config.get("runpod", {}))
    settings.setdefault("state_root", config.get("state_root"))
    return settings


def _reject_inline_secrets(value: Any) -> None:
    if isinstance(value, Mapping):
        _require(not any(str(key).lower() in {"api_key", "runpod_api_key", "authorization", "access_token", "password"}
                         for key in value), "Credentials belong in the protected credential file or environment, not launch configuration.")
        for item in value.values():
            _reject_inline_secrets(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_inline_secrets(item)
    elif isinstance(value, str):
        _require(not re.search(r"rpa_[A-Za-z0-9_-]{10,}|(?i:Bearer\s+)\S+", value),
                 "Launch configuration contains a credential-like value.")


def _validate(config: Mapping[str, Any], *, fresh: bool) -> dict[str, Any]:
    _reject_inline_secrets(config)
    cfg = _settings(config)
    _require(type(cfg.get('keep_alive_for_operator', False)) is bool,
             'keep_alive_for_operator must be a boolean.')
    search = _number(cfg.get('stock_search_seconds', 0), 'stock_search_seconds', zero=True)
    interval = _number(cfg.get('stock_poll_interval_seconds', 45), 'stock_poll_interval_seconds')
    _require(search <= 1800 and 30 <= interval <= 45, 'Stock search is limited to 1800 seconds with 30–45 second polls.')
    _require(not search or (cfg.get('data_center_id') == 'EU-RO-1' and not cfg.get('adopt_pod_id')),
             'Stock search requires a new EU-RO-1 allocation.')
    hours = _number(cfg.get("requested_hours"), "requested_hours")
    duration = _number(cfg.get("max_runtime_seconds"), "max_runtime_seconds")
    ceiling = _number(cfg.get("gross_ceiling_usd"), "gross_ceiling_usd")
    reserve = _number(cfg.get("stop_reserve_seconds", 60), "stop_reserve_seconds")
    _require(duration <= hours * 3600, "Runpod runtime exceeds the configured requested hours.")
    _require(min(120, duration / 2) <= reserve < duration, "Insufficient shutdown-request reserve.")
    heartbeat = _number(cfg.get("heartbeat_timeout_seconds", 900), "heartbeat_timeout_seconds")
    _require(heartbeat <= 900, "Owner heartbeat timeout exceeds 900 seconds.")
    if not cfg.get("adopt_pod_id") and "absolute_hard_stop_unix" in cfg:
        hard = _number(cfg["absolute_hard_stop_unix"], "absolute_hard_stop_unix")
        if fresh:
            _require(min(duration, hard - time.time()) - reserve >= 900,
                     "Absolute launch deadline leaves no 900-second closeout reserve.")
    if cfg.get("adopt_pod_id") is not None:
        _require(re.fullmatch(r"[A-Za-z0-9_-]+", str(cfg["adopt_pod_id"])), "Invalid explicitly adopted pod ID.")
        started = _epoch(cfg.get("adopt_started_at"))
        hard = _number(cfg.get("absolute_hard_stop_unix"), "absolute_hard_stop_unix")
        _require(started <= hard <= started + duration, "Adoption cannot reset or extend the original runtime window.")
    for key in ("state_root", "cli_path", "ssh_private_key"):
        _require(isinstance(cfg.get(key), str) and Path(cfg[key]).is_absolute()
                 and not any(x in cfg[key] for x in "\n\r\0"), f"{key} needs an explicit safe absolute path.")
    for key in ("pod_name", "ssh_alias"):
        _require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{1,100}", str(cfg.get(key, ""))), f"Invalid {key}.")
    for key in ("gpu_id", "data_center_id", "image", "min_cuda_version"):
        _require(isinstance(cfg.get(key), str) and cfg[key] and not any(x in cfg[key] for x in "\n\r\0"),
                 f"Explicit {key} is required.")
    profiles = {"runpod_l4": "NVIDIA L4", "runpod_rtx4090": "NVIDIA GeForce RTX 4090"}
    _require(config.get("hardware_profile") in profiles
             and cfg["gpu_id"] == profiles[config["hardware_profile"]], "Runpod GPU and approved hardware profile differ.")
    _require(re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", cfg["image"]), "Runpod image must be pinned by SHA-256 digest.")
    _require(cfg.get("runtime_environment") == {"NVIDIA_TF32_OVERRIDE": "0"},
             "Runpod requires the fixed full-FP32 runtime environment.")
    precision_origin = cfg.get("precision_environment_origin", "pod")
    _require(precision_origin == "pod" or (cfg.get("adopt_pod_id") and precision_origin == "transport"),
             "Only an explicitly adopted pod may use transport-enforced precision settings.")
    _require(cfg["data_center_id"] != "ANY_SECURE" or cfg.get("adopt_pod_id"),
             "New allocations require an exact data center.")
    ephemeral = cfg.get("storage_mode") == "ephemeral_host_pull"
    _require(cfg.get("storage_mode", "persistent") in {"persistent", "ephemeral_host_pull"}, "Unknown storage mode.")
    _require(not ephemeral or bool(cfg.get("adopt_pod_id")), "Ephemeral HostPull is restricted to an explicitly adopted pod.")
    _require(type(cfg.get("container_disk_gb")) is int and cfg["container_disk_gb"] > 0,
             "A positive integer container_disk_gb is required.")
    _require(type(cfg.get("volume_gb")) is int and (cfg["volume_gb"] == 0 if ephemeral else cfg["volume_gb"] > 0),
             "A positive volume_gb is required unless explicit ephemeral HostPull adoption is selected.")
    if ephemeral:
        roots = []
        devices = []
        for key in ("durable_root", "mirror_root"):
            value = config.get(key)
            _require(isinstance(value, str) and Path(value).is_absolute() and ".." not in Path(value).parts,
                     "Ephemeral HostPull requires explicit independent local destinations.")
            root = Path(value).resolve()
            roots.append(root)
            ancestor = root
            while not ancestor.exists():
                ancestor = ancestor.parent
            devices.append(ancestor.stat().st_dev)
        _require(devices[0] != devices[1] and roots[0] not in roots[1].parents and roots[1] not in roots[0].parents,
                 "Ephemeral HostPull requires primary and mirror destinations on independent devices.")
    mount = cfg.get("volume_mount_path", "/workspace")
    _require(isinstance(mount, str) and Path(mount).is_absolute() and ".." not in Path(mount).parts
             and not any(x in mount for x in "\n\r\0"), "Unsafe persistent volume path.")
    quote = cfg.get("quote", {})
    observed = _epoch(quote.get("observed_at"))
    age_limit = _number(quote.get("max_age_seconds", 900), "quote max_age_seconds")
    _require(age_limit <= 900, "Price quote freshness cannot exceed 900 seconds.")
    if fresh:
        _require(-5 <= time.time() - observed <= age_limit, "Runpod price quote is stale or in the future.")
    _require(isinstance(quote.get("source"), str) and quote["source"], "Price quote evidence source is required.")
    compute = _number(quote.get("hourly_compute_usd"), "hourly_compute_usd")
    storage = _number(quote.get("running_storage_hourly_usd"), "running_storage_hourly_usd", zero=True)
    stopped = _number(quote.get("stopped_storage_reserve_usd"), "stopped_storage_reserve_usd", zero=True)
    retention = cfg.get("retention_policy", {})
    retention_seconds = _number(retention.get("retention_seconds"), "retention_seconds")
    stopped_rate = _number(quote.get("stopped_storage_hourly_usd"), "stopped_storage_hourly_usd", zero=ephemeral)
    _require((stopped == stopped_rate == 0) if ephemeral else
             (stopped > 0 and stopped >= stopped_rate * retention_seconds / 3600),
             "Stopped storage reserve does not cover the defined retention period.")
    _require(retention.get("cleanup_reviewed") is True
             and retention.get("action") == ("host_pull_before_stop" if ephemeral else "verified_backup_then_delete")
             and all(isinstance(retention.get(key), str) and retention[key].strip() for key in ("owner", "evidence")),
             "A reviewed finite storage retention and backup-dependent cleanup policy is required.")
    egress = _number(quote.get("egress_reserve_usd"), "egress_reserve_usd", zero=True)
    prior = _number(quote.get("prior_charge_usd"), "prior_charge_usd", zero=True)
    margin = _number(quote.get("margin_fraction"), "margin_fraction")
    _require(margin >= .15, "Runpod cost margin must be at least 15%.")
    total = prior + ((compute + storage) * hours + stopped + egress) * (1 + margin)
    _require(total <= ceiling, "Conservative Runpod gross envelope exceeds the authorized ceiling.")
    return {"status": "ready", "gross_ceiling_usd": ceiling, "estimated_gross_usd": total,
            "requested_hours": hours, "max_total_seconds": duration, "stop_reserve_seconds": reserve,
            "quote_observed_at": quote["observed_at"], "quote_source": quote["source"],
            "prior_charge_usd": prior, "running_compute_usd": compute * hours,
            "running_storage_usd": storage * hours, "stopped_storage_reserve_usd": stopped,
            "egress_reserve_usd": egress, "margin_fraction": margin,
            "retention_seconds": retention_seconds, "stopped_storage_hourly_usd": stopped_rate,
            "retention_policy": retention, "automatic_storage_cleanup": False,
            "storage_cleanup_method": "host_pull_before_stop" if ephemeral else "closeout_after_independent_preservation",
            "ceiling_scope": "execution_only_no_stopped_storage" if ephemeral else "execution_plus_bounded_retention_estimate",
            "provider_enforced_stop": False, "stop_mechanism": "independent workstation process"}


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate local configuration and a fresh fully priced envelope; no API call."""
    return _validate(config, fresh=True)


def _status(pod: Mapping[str, Any]) -> tuple[str, str]:
    return str(pod.get("desiredStatus", pod.get("status", ""))).upper(), str(pod.get("runtimeStatus", "")).lower()


def _stopped(pod: Mapping[str, Any]) -> bool:
    desired, runtime = _status(pod)
    return desired == "EXITED" and runtime == "stopped"


def _gpu_count(pod: Mapping[str, Any]) -> int | None:
    count = pod.get("gpuCount", pod.get("gpu", {}).get("count"))
    return count if type(count) is int else None


def _public_pod(pod: Mapping[str, Any]) -> dict[str, Any]:
    """Provider environment and authentication fields must never enter receipts."""
    keys = ("id", "name", "desiredStatus", "runtimeStatus", "runtimeStatusReason", "createdAt",
            "startedAt", "costPerHr", "cost", "gpuCount", "gpuId", "imageName", "image",
            "containerDiskInGb", "volumeInGb", "volumeMountPath", "dataCenterId")
    result = {key: pod[key] for key in keys if key in pod}
    machine = pod.get("machine", {})
    result["placement"] = {"data_center_id": machine.get("dataCenterId") or pod.get("dataCenterId") or None,
                           "secure_cloud": machine.get("secureCloud")}
    return result


class RunpodController:
    """Bound one immutable launch intent to one pod; cleanup never widens ownership."""

    def __init__(self, config: Mapping[str, Any], config_path: Path, *, _snapshot: Mapping[str, Any] | None = None):
        self.config = json.loads(json.dumps(config))
        self.cfg = _settings(self.config)
        self.cost = _validate(self.config, fresh=False)
        self.config_path = Path(config_path).resolve()
        self.root = Path(self.cfg["state_root"]).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "session.json"
        self.snapshot_path = self.root / "runpod_config.snapshot.json"
        self.owner_path = self.root / "ownership.json"
        self.heartbeat_path = self.root / "owner_heartbeat.json"
        self.watch_path = self.root / "watchdog_heartbeat.json"
        self._cached_state: dict[str, Any] | None = None
        self._cached_owner: dict[str, Any] | None = None
        self._heartbeat_guard: _HeartbeatGuard | None = None
        self.source_sha = _snapshot["source_sha256"] if _snapshot else _sha(self.config_path)
        self.config_sha = _digest(self.config)
        self.cli_sha = _snapshot["cli_sha256"] if _snapshot else _sha(Path(self.cfg["cli_path"]))
        _require(_sha(Path(self.cfg["cli_path"])) == self.cli_sha, "Runpod CLI changed during this pass.")

    @contextmanager
    def _lock(self):
        with (self.root / "controller.lock").open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield

    def _state(self) -> dict[str, Any]:
        state = _read(self.state_path)
        _require(state.get("config_sha256") == self.config_sha, "Controller state belongs to another configuration.")
        self._cached_state = dict(state)
        return state

    def _cleanup_state(self) -> dict[str, Any]:
        try:
            return self._state()
        except (OSError, ValueError):
            _require(self._cached_state is not None, "No authenticated launch state is available for cleanup.")
            return dict(self._cached_state)

    def _update(self, **values: Any) -> dict[str, Any]:
        with self._lock():
            state = self._state()
            state.update(values)
            _write(self.state_path, state)
            self._cached_state = dict(state)
            return state

    def _cleanup_update(self, **values: Any) -> dict[str, Any]:
        """Disk failure must close admission without preventing provider shutdown."""
        try:
            return self._update(**values)
        except (OSError, ValueError):
            state = self._cleanup_state()
            state.update(values)
            self._cached_state = dict(state)
            return state

    def _unchanged(self) -> None:
        _require(_sha(self.config_path) == self.source_sha and _read(self.config_path) == self.config,
                 "Launch configuration changed; admission is closed.")

    def _environment(self) -> tuple[dict[str, str], str]:
        env = os.environ.copy()
        key = env.get("RUNPOD_API_KEY", "")
        if not key:
            path = Path(self.cfg.get("credential_file", ""))
            _require(path.is_absolute() and path.is_file() and not path.is_symlink(), "Missing Runpod credential file.")
            mode = stat.S_IMODE(path.stat().st_mode)
            _require(mode == 0o600 and path.stat().st_uid == os.getuid(), "Credential file must be owned by this user and mode 0600.")
            key = _read(path).get("api_key", "")
        _require(isinstance(key, str) and key and not any(x in key for x in "\n\r\0"), "Missing or invalid Runpod credential.")
        env["RUNPOD_API_KEY"] = key
        return env, key

    @staticmethod
    def _redact(text: str, key: str = "") -> str:
        if key:
            text = text.replace(key, "[REDACTED]")
        text = re.sub(r"rpa_[A-Za-z0-9_-]+", "[REDACTED]", text)
        return re.sub(r"(?i)(Bearer\s+)[A-Za-z0-9._-]+", r"\1[REDACTED]", text)[:1200]

    def _call(self, *args: str, timeout: float = 30) -> Any:
        env, key = self._environment()
        if args[0] == "create":
            _require(not self.cfg.get("adopt_pod_id"), "An adoption configuration can never create a pod.")
            try:
                self._unchanged()
                self._create_cost()
                with self._lock():
                    state = self._state()
                    _require(state.get("phase") == "creating" and state.get("create_attempted") is True
                             and not state.get("stop_requested") and time.time() < state["deadline_unix"]
                             and ("absolute_hard_stop_unix" not in self.cfg
                                  or state["deadline_unix"] - time.time() >= 900),
                             "Create admission closed before the provider request.")
                    self._watchdog_healthy()
            except BaseException:
                # No provider request has been made on this path. Preserve the
                # intent without leaving its watchdog reconciling a nonexistent pod.
                self._cleanup_update(create_cancelled_before_post=True)
                raise
        try:
            result = subprocess.run([self.cfg["cli_path"], "pod", *args], stdin=subprocess.DEVNULL,
                                    capture_output=True, text=True, timeout=timeout, env=env)
        except (subprocess.TimeoutExpired, OSError) as exc:
            raise RunpodError(f"Runpod {args[0]} result is uncertain ({type(exc).__name__}).") from None
        if result.returncode:
            raw = result.stderr or result.stdout
            try:
                error = json.loads(raw)
                message = f"{error.get('code', 'unclassified')}: {error.get('error', 'operation failed')}"
            except (ValueError, AttributeError):
                message = "unclassified command failure"
            raise RunpodError(f"Runpod {args[0]} failed: {self._redact(message, key)}")
        try:
            parsed = json.loads(result.stdout)
        except ValueError:
            if args[0] == "delete" and not result.stdout.strip():
                return {"status": "delete_requested"}
            raise RunpodError(f"Runpod {args[0]} returned invalid JSON.") from None
        _require(not isinstance(parsed, dict) or not parsed.get("error"), f"Runpod {args[0]} returned an error payload.")
        return parsed

    def _catalog_call(self, path: str, query: Mapping[str, Any]) -> dict[str, Any]:
        """Read only public catalog/account billing routes; never log credentials."""
        allowed = '/v2/catalog/gpus/' + urllib.parse.quote(self.cfg['gpu_id'], safe='')
        _require(path in {allowed, '/v2/billing'}, 'Unsupported stock-search read route.')
        _, key = self._environment()
        request = urllib.request.Request('https://api.runpod.io' + path + '?' + urllib.parse.urlencode(query),
                                         headers={'Authorization': 'Bearer ' + key})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                result = json.load(response)
        except Exception as exc:
            raise RunpodError(f'Stock-search read failed ({type(exc).__name__}).') from None
        _require(isinstance(result, dict), 'Stock-search response has an invalid shape.')
        return result

    def _stock_available(self) -> bool:
        row = self._catalog_call('/v2/catalog/gpus/' + urllib.parse.quote(self.cfg['gpu_id'], safe=''),
            {'include': 'AVAILABILITY', 'product': 'POD', 'cloud': 'SECURE', 'count': 1,
             'countryCodes': 'RO', 'minCudaVersion': self.cfg['min_cuda_version']})
        _require(row.get('id') == self.cfg['gpu_id'] and row.get('secure') is True,
                 'Stock read returned a different GPU or cloud.')
        _require(row.get('price', {}).get('secure') == self.cfg['quote']['hourly_compute_usd'],
                 'Live secure compute price differs from the bounded quote.')
        floor = tuple(map(int, self.cfg['min_cuda_version'].split('.')))
        cuda = any(v.get('available') is True and re.fullmatch(r'\d+\.\d+', str(v.get('version', '')))
                   and tuple(map(int, v['version'].split('.'))) >= floor for v in row.get('cudaVersions', []))
        available = cuda and any(v.get('id') == self.cfg['data_center_id']
            and v.get('availability') in {'LOW', 'MEDIUM', 'HIGH'} for v in row.get('dataCenters', []))
        observation = {'config_sha256': self.config_sha, 'observed_unix': time.time(),
                       'available': bool(available), 'catalog': row}
        _write(self.root / 'stock_observation.json', observation)
        with (self.root / 'stock_polls.jsonl').open('a') as handle:
            handle.write(json.dumps(observation, sort_keys=True) + '\n')
        return bool(available)

    def _search_admission(self) -> None:
        # Consume the positive poll rather than racing another catalog read.
        observation = _read(self.root / 'stock_observation.json')
        _require(observation.get('config_sha256') == self.config_sha
                 and observation.get('available') is True
                 and 0 <= time.time() - observation.get('observed_unix', 0) <= 45,
                 'A fresh positive stock poll is required before create.')
        billing = self._catalog_call('/v2/billing', {'bucketSize': 'hour', 'lastN': 25})
        total = _number(billing.get('metadata', {}).get('totals', {}).get('totalAmount'), 'fresh rolling billing', zero=True)
        _require(total <= self.cfg['quote']['prior_charge_usd'], 'Fresh billing exceeds the recorded prior-charge reserve.')
        _write(self.root / 'stock_billing.json', {'observed_unix': time.time(), 'total_usd': total,
                                                'config_sha256': self.config_sha, 'billing': billing})

    def _create_cost(self) -> dict[str, Any]:
        if not self.cfg.get('stock_search_seconds'):
            return validate_config(self.config)
        # Live price and billing replace only quote-age validation, never the caps.
        cost = _validate(self.config, fresh=False)
        self._search_admission()
        return cost

    def _await_stock(self) -> None:
        path = self.root / 'stock_search.json'
        if path.exists():
            search = _read(path)
            _require(search.get('config_sha256') == self.config_sha, 'Stock search belongs to another configuration.')
            _require(0 <= search['deadline_unix'] - search['started_unix'] <= self.cfg['stock_search_seconds']
                     and search['started_unix'] <= time.time(), 'Invalid persisted stock-search window.')
        else:
            now = time.time()
            search = {'config_sha256': self.config_sha, 'started_unix': now,
                      'deadline_unix': now + self.cfg['stock_search_seconds']}
            _write(path, search)
        while time.time() < search['deadline_unix']:
            self._unchanged()
            if self.state_path.exists():
                state = self._state()
                _require(not state.get('stop_requested') and time.time() < state['deadline_unix'] - 900,
                         'Stock retry reached the original runtime closeout boundary.')
                self.heartbeat()
                self._watchdog_healthy()
            rows = self._pods()
            self._no_other_gpu(rows)
            _require(not any(p.get('name') == self.cfg['pod_name'] for p in rows),
                     'A matching pod appeared during stock polling; reconciliation is required.')
            if self._stock_available():
                return
            time.sleep(min(self.cfg.get('stock_poll_interval_seconds', 45), max(0, search['deadline_unix'] - time.time())))
        raise RunpodError('The bounded 30-minute stock search is exhausted.')

    def _pods(self) -> list[dict[str, Any]]:
        rows = self._call("list", "--all")
        if isinstance(rows, dict):
            rows = rows.get("pods")
        _require(isinstance(rows, list) and all(isinstance(p, dict) and p.get("id") for p in rows),
                 "Runpod list has an unrecognized or incomplete shape.")
        return rows

    def _no_other_gpu(self, rows: list[dict[str, Any]], own: str | None = None) -> None:
        for pod in rows:
            if pod["id"] == own or _stopped(pod):
                continue
            _require(_gpu_count(pod) == 0, "Another Runpod GPU is active or its state is uncertain.")

    def _bind(self, pod: Mapping[str, Any], *, source: str) -> str:
        pod_id = str(pod.get("id", ""))
        _require(re.fullmatch(r"[A-Za-z0-9_-]+", pod_id), "Provider returned an invalid pod ID.")
        with self._lock():
            state = self._state()
            adoption = (source == "explicit_named_adoption" and state.get("adopted") is True
                        and pod_id == self.cfg.get("adopt_pod_id") == state.get("adopt_pod_id"))
            _require(pod.get("name") == self.cfg["pod_name"] and (adoption or
                     (pod_id not in state["preexisting_pod_ids"]
                      and _epoch(pod.get("createdAt")) >= state["allocation_started_unix"] - 5)),
                     "Provider response does not identify this new launch intent.")
            owned = {"pod_id": pod_id, "session_id": state["session_id"],
                     "config_sha256": self.config_sha, "source": source, "bound_at": _utc(time.time())}
            if self.owner_path.exists():
                owned = _read(self.owner_path)
                _require(owned.get("pod_id") == pod_id and owned.get("session_id") == state["session_id"]
                         and owned.get("config_sha256") == self.config_sha,
                         "A second pod cannot replace this pass's owned pod.")
            self._cached_owner = dict(owned)
            if not self.owner_path.exists():
                _write(self.owner_path, owned)
            if not adoption:
                started = min(state["allocation_started_unix"], _epoch(pod.get("createdAt")))
                hard = min(state["hard_stop_unix"], started + self.cost["max_total_seconds"])
                state.update(allocation_started_unix=started, hard_stop_unix=hard,
                             deadline_unix=min(state["deadline_unix"], hard - self.cost["stop_reserve_seconds"]),
                             max_total_seconds=hard - started)
            state.update(pod_id=pod_id, phase="allocated")
            _write(self.state_path, state)
            self._cached_state = dict(state)
        return pod_id

    def _owned_id(self) -> str | None:
        try:
            owned = _read(self.owner_path) if self.owner_path.exists() else self._cached_owner
        except (OSError, ValueError):
            owned = self._cached_owner
        if owned is None:
            return None
        state = self._cleanup_state()
        _require(owned.get("config_sha256") == self.config_sha and owned.get("session_id") == state["session_id"]
                 and state.get("pod_id") in (None, owned.get("pod_id")), "Runpod ownership record disagrees with state.")
        self._cached_owner = dict(owned)
        return owned["pod_id"]

    def _resolve(self) -> str | None:
        owned = self._owned_id()
        if owned:
            return owned
        state = self._cleanup_state()
        if state.get("adopted") is True:
            pod = self._call("get", self.cfg["adopt_pod_id"], "--include-machine")
            self._verify_adopt_identity(pod)
            return self._bind(pod, source="explicit_named_adoption")
        if not state.get("create_attempted") or state.get("create_cancelled_before_post"):
            return None
        matches = [p for p in self._pods() if p.get("name") == self.cfg["pod_name"]]
        _require(len(matches) <= 1, "Ambiguous pod name; no resource can be selected safely.")
        if not matches:
            return None
        pod = matches[0]
        _require(pod["id"] not in state["preexisting_pod_ids"]
                 and _epoch(pod.get("createdAt")) >= state["allocation_started_unix"] - 5,
                 "Matching pod predates this launch intent.")
        try:
            return self._bind(pod, source="unique_name_reconciliation")
        except OSError:
            # A successful authenticated binding remains usable in this process
            # even when the private receipt filesystem is full.
            owned = self._owned_id()
            if owned:
                return owned
            raise

    def heartbeat(self) -> None:
        if self._heartbeat_guard is not None:
            self._heartbeat_guard.check()
        state = self._state()
        _write(self.heartbeat_path, {"session_id": state["session_id"], "updated_unix": time.time(), "pid": os.getpid()})

    @contextmanager
    def maintain_heartbeat(self, *, interval: float = 30):
        """Cover blocking work; end the ticker only after verified terminal state.

        Enter after start/adopt. The caller may perform its normal preservation
        and closeout inside the context. Leaving early still requests STOP.
        """
        interval = _number(interval, 'heartbeat interval')
        _require(interval <= min(30, self.cfg.get('heartbeat_timeout_seconds', 900) / 3),
                 "Continuous heartbeat interval exceeds its safe owner-timeout margin.")
        _require(self._heartbeat_guard is None, "A continuous heartbeat is already active for this coordinator.")
        state = self._state()
        _require(state.get('phase') not in {'stopped', 'deleted'} and not state.get('stop_requested'),
                 "A closed allocation cannot acquire a continuous heartbeat.")
        guard = _HeartbeatGuard(self, interval)
        self._heartbeat_guard = guard
        try:
            guard.start()
            yield guard
            guard.check()
        finally:
            # Keep the ticker and independent watchdog alive if STOP cannot be
            # confirmed. No new work is admitted after stop_requested is set.
            state = self._cleanup_state()
            if state.get('phase') not in {'stopped', 'deleted'}:
                self.stop()
            guard.close()
            self._heartbeat_guard = None

    def _watchdog_healthy(self) -> None:
        watch = _read(self.watch_path) if self.watch_path.exists() else {}
        _require(watch.get("session_id") == self._state()["session_id"]
                 and watch.get("config_sha256") == self.config_sha
                 and time.time() - watch.get("updated_unix", 0) < 45
                 and watch.get("status") == "armed", "Independent stop watchdog is not healthy.")
        try:
            os.kill(int(watch["pid"]), 0)
        except (KeyError, ValueError, ProcessLookupError):
            raise RunpodError("Independent stop watchdog is absent.") from None

    def _arm_watchdog(self) -> None:
        snapshot = {"config": self.config, "config_path": str(self.config_path),
                    "source_sha256": self.source_sha, "cli_sha256": self.cli_sha}
        _write(self.snapshot_path, snapshot)
        log = self.root / "watchdog.log"
        with log.open("ab") as handle:
            os.chmod(log, 0o600)
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--watchdog", str(self.snapshot_path)],
                                       stdin=subprocess.DEVNULL, stdout=handle, stderr=handle, start_new_session=True)
        self._update(watchdog_pid=process.pid)
        until = time.monotonic() + 10
        while time.monotonic() < until:
            if process.poll() is not None:
                raise RunpodError("Stop watchdog failed before allocation; create is blocked.")
            if self.watch_path.exists():
                self._watchdog_healthy()
                return
            time.sleep(.05)
        raise RunpodError("Stop watchdog did not acknowledge readiness; create is blocked.")

    def _create_args(self) -> list[str]:
        cfg = self.cfg
        return ["create", "--name", cfg["pod_name"], "--gpu-id", cfg["gpu_id"], "--gpu-count", "1",
                "--data-center-ids", cfg["data_center_id"], "--cloud-type", "SECURE", "--image", cfg["image"],
                "--min-cuda-version", cfg["min_cuda_version"], "--container-disk-in-gb", str(cfg["container_disk_gb"]),
                "--volume-in-gb", str(cfg["volume_gb"]), "--volume-mount-path", cfg.get("volume_mount_path", "/workspace"),
                "--env", json.dumps(cfg["runtime_environment"], sort_keys=True), "--ports", "22/tcp"]

    def start(self) -> dict[str, Any]:
        _require(not self.cfg.get("adopt_pod_id"), "An adoption configuration can never create a pod; use adopt().")
        self._unchanged()
        search = bool(self.cfg.get('stock_search_seconds'))
        if search:
            _require(not self.state_path.exists(), 'This pass already has a launch intent; it cannot create again.')
            while True:
                self._await_stock()
                try:
                    self.cost = self._create_cost()
                    break
                except StockUnavailable:
                    time.sleep(min(self.cfg.get('stock_poll_interval_seconds', 45),
                        max(0, _read(self.root / 'stock_search.json')['deadline_unix'] - time.time())))
        else:
            self.cost = self._create_cost()
        with self._lock():
            _require(not self.state_path.exists(), "This pass already has a launch intent; it cannot create again.")
            rows = self._pods()
            self._no_other_gpu(rows)
            _require(not any(p.get("name") == self.cfg["pod_name"] for p in rows), "Pod name already exists.")
            begun = time.time()
            hard = min(begun + self.cost["max_total_seconds"],
                       self.cfg.get("absolute_hard_stop_unix", begun + self.cost["max_total_seconds"]))
            _require("absolute_hard_stop_unix" not in self.cfg or hard - begun - self.cost["stop_reserve_seconds"] >= 900,
                     "Absolute launch deadline leaves no 900-second closeout reserve.")
            _write(self.state_path, {"schema_version": 1, "session_id": uuid.uuid4().hex, "phase": "arming",
                                    "config_sha256": self.config_sha, "config_source_sha256": self.source_sha,
                                    "preexisting_pod_ids": [p["id"] for p in rows], "create_attempted": False,
                                    "allocation_started_unix": begun, "hard_stop_unix": hard,
                                    "deadline_unix": hard - self.cost["stop_reserve_seconds"],
                                    "max_total_seconds": hard - begun,
                                    "cost_admission": self.cost, "stop_requested": False})
        self.heartbeat()
        self._arm_watchdog()
        self._unchanged()
        if not search:
            validate_config(self.config)
        with self._lock():
            state = self._state()
            _require(state["phase"] == "arming" and not state.get("stop_requested")
                     and not state.get("create_attempted"), "Launch intent is closed; create is blocked.")
            self._watchdog_healthy()
            self.heartbeat()
            state.update(create_attempted=True, phase="creating")
            _write(self.state_path, state)
            self._cached_state = dict(state)
        try:
            while True:
                try:
                    if search:
                        until = _read(self.root / 'stock_search.json')['deadline_unix']
                        if time.time() >= until:
                            self._update(create_cancelled_before_post=True)
                            raise RunpodError('The bounded stock search expired before create.')
                    with self._lock():
                        current = self._state()
                        _require(not current.get('stop_requested'), 'Create intent is closed.')
                        current.pop('create_error', None)
                        current.pop('create_cancelled_before_post', None)
                        current['create_attempt_count'] = current.get('create_attempt_count', 0) + 1
                        _write(self.state_path, current)
                    created = self._call(*self._create_args(), timeout=min(120, self.cost["max_total_seconds"],
                        max(1, until - time.time()) if search else 120))
                    self._bind(created, source="successful_create_response")
                    break
                except RunpodError as exc:
                    self._update(create_error=str(exc))
                    if search and isinstance(exc, StockUnavailable):
                        time.sleep(min(self.cfg.get('stock_poll_interval_seconds', 45), max(0, until - time.time())))
                        self._await_stock()
                        continue
                    # An uncertain response can bind a uniquely reconciled pod, never retry POST.
                    if self._resolve():
                        break
                    rejection = str(exc)
                    definite = (rejection.startswith('Runpod create failed: bad_request: failed to create pod:')
                        and 'There are no longer any instances available with the requested specifications.' in rejection)
                    if not search or not definite:
                        raise
                    rows = self._pods()
                    self._no_other_gpu(rows)
                    _require(not any(p.get('name') == self.cfg['pod_name'] for p in rows),
                             'A pod appeared after rejection; retry is blocked.')
                    with (self.root / 'create_rejections.jsonl').open('a') as handle:
                        handle.write(json.dumps({'at': _utc(time.time()), 'error': rejection,
                            'absence_verified': True, 'observed_ids': sorted(p['id'] for p in rows)}) + '\n')
                    remaining = until - time.time()
                    _require(remaining > 0, 'The bounded stock search is exhausted after capacity rejection.')
                    time.sleep(min(self.cfg.get('stock_poll_interval_seconds', 45), remaining))
                    self._await_stock()
            until = min(time.time() + 600, self._state()["deadline_unix"])
            while time.time() < until:
                self.heartbeat()
                pod = self._call("get", self._owned_id())
                if _status(pod) == ("RUNNING", "running"):
                    return self.session()
                _require(not _stopped(pod) and _status(pod)[1] not in {"error", "terminated"}, "Pod did not become usable.")
                time.sleep(min(5, max(.05, until - time.time())))
            raise RunpodError("Pod readiness exceeded the bounded allocation window.")
        except BaseException:
            self._cleanup_update(stop_requested=True, phase="closing_after_failure")
            try:
                self.stop()
            except Exception as exc:
                self._cleanup_update(stop_error=self._redact(str(exc)))
            raise

    def _verify_adopt_identity(self, pod: Mapping[str, Any]) -> None:
        """Establish the exact user-named identity before taking shutdown ownership."""
        cfg = self.cfg
        _require(pod.get("id") == cfg.get("adopt_pod_id") and pod.get("name") == cfg["pod_name"],
                 "Provider identity differs from the explicitly adopted pod.")
        _require(pod.get("gpuId", pod.get("gpu", {}).get("id")) == cfg["gpu_id"], "Adopted GPU type differs.")
        _require(self._placement_matches(pod), "Adopted secure-cloud placement differs.")
        actual_start = _epoch(pod.get("startedAt") or pod.get("createdAt"))
        expected_start = _epoch(cfg["adopt_started_at"])
        created = _epoch(pod.get("createdAt"))
        original_creation_matches = -1e-6 <= created - expected_start <= 5
        _require(original_creation_matches and actual_start >= expected_start - 1e-6,
                 "Adopted pod start time differs; its allocation clock cannot reset.")

    def _placement_matches(self, pod: Mapping[str, Any]) -> bool:
        machine = pod.get("machine", {})
        actual = machine.get("dataCenterId") or pod.get("dataCenterId") or None
        any_secure = self.cfg.get("adopt_pod_id") and self.cfg["data_center_id"] == "ANY_SECURE"
        return machine.get("secureCloud") is True and (bool(any_secure) or actual == self.cfg["data_center_id"])

    @staticmethod
    def _session_metadata(state: Mapping[str, Any]) -> dict[str, Any]:
        return {key: state[key] for key in ("session_id", "pod_id", "allocation_started_unix", "deadline_unix",
                                           "hard_stop_unix", "max_total_seconds", "cost_admission", "provider_verified_at")
                if key in state}

    def adopt(self, pod_id: str, *, qualify: bool = False) -> dict[str, Any]:
        """Arm shutdown for one explicitly handed-over pod without creating or restarting it.

        Initial adoption intentionally permits the existing image, environment,
        and volume to await configuration. session() remains mandatory before
        any workload and checks the complete approved execution configuration.
        """
        _require(pod_id == self.cfg.get("adopt_pod_id"), "Only the explicitly named pod can be adopted.")
        self._unchanged()
        self.cost = validate_config(self.config)
        if self.state_path.exists():
            state = self._state()
            _require(state.get("adopted") is True and self._owned_id() == pod_id
                     and not state.get("stop_requested") and state["phase"] not in {"stopped", "deleted"},
                     "This adoption is closed or belongs to another allocation.")
            self._watchdog_healthy()
            self.heartbeat()
            return self.session(state["session_id"]) if qualify else {**self._session_metadata(state), "pending_qualification": True}
        pod = self._call("get", pod_id, "--include-machine")
        self._verify_adopt_identity(pod)
        _require(_status(pod)[0] == "RUNNING" and _gpu_count(pod) == 1,
                 "Only a running single-GPU pod can be adopted.")
        rows = self._pods()
        started = _epoch(self.cfg["adopt_started_at"])
        hard = min(started + self.cost["max_total_seconds"], self.cfg["absolute_hard_stop_unix"])
        with self._lock():
            _require(not self.state_path.exists(), "This pass already has an allocation intent.")
            state = {"schema_version": 1, "session_id": uuid.uuid4().hex, "phase": "arming",
                     "config_sha256": self.config_sha, "config_source_sha256": self.source_sha,
                     "preexisting_pod_ids": [p["id"] for p in rows], "create_attempted": False,
                     "adopted": True, "adopt_pod_id": pod_id, "allocation_started_unix": started,
                     "hard_stop_unix": hard, "deadline_unix": hard - self.cost["stop_reserve_seconds"],
                     "max_total_seconds": hard - started, "cost_admission": self.cost, "stop_requested": False}
            _write(self.state_path, state)
            self._cached_state = dict(state)
        try:
            self._bind(pod, source="explicit_named_adoption")
            self.heartbeat()
            self._arm_watchdog()
            self._require_adoption_open()
            rate = _number(pod.get("costPerHr", pod.get("cost")), "actual adopted hourly price")
            _require(rate <= self.cfg["quote"]["hourly_compute_usd"] + self.cfg["quote"]["running_storage_hourly_usd"],
                     "Adopted pod price exceeds the admitted quote.")
            self._no_other_gpu(rows, own=pod_id)
            return self.session() if qualify else {**self._session_metadata(self._state()), "pending_qualification": True}
        except BaseException:
            self._cleanup_update(stop_requested=True, phase="closing_after_failure")
            try:
                self.stop()
            except Exception as exc:
                self._cleanup_update(stop_error=self._redact(str(exc)))
            raise

    def _require_adoption_open(self) -> None:
        state = self._state()
        _require(not state.get("stop_requested") and state["phase"] not in {"stopped", "deleted"}
                 and time.time() < state["deadline_unix"], "The adopted pod's original stop deadline has elapsed.")
        self._watchdog_healthy()

    def session(self, previous: Mapping[str, Any] | str | None = None) -> dict[str, Any]:
        if self._heartbeat_guard is not None:
            self._heartbeat_guard.check()
        self._unchanged()
        state = self._state()
        _require(not state.get("stop_requested") and state["phase"] != "stopped", "This allocation is closing or stopped.")
        _require(time.time() < state["deadline_unix"], "Runpod stop-request deadline reached.")
        self._watchdog_healthy()
        owned = self._owned_id()
        _require(owned, "No owned Runpod allocation is bound.")
        pod = self._call("get", owned, "--include-machine")
        _require(pod.get("id") == owned and pod.get("name") == self.cfg["pod_name"], "Provider identity differs from owned pod.")
        _require(_status(pod) == ("RUNNING", "running") and _gpu_count(pod) == 1, "Owned pod is not a running single GPU.")
        _require(pod.get("gpuId", pod.get("gpu", {}).get("id")) == self.cfg["gpu_id"], "Allocated GPU type differs.")
        _require(pod.get("imageName", pod.get("image")) == self.cfg["image"], "Allocated image differs.")
        if self.cfg.get("precision_environment_origin", "pod") == "pod":
            _require(isinstance(pod.get("env"), dict) and pod["env"].get("NVIDIA_TF32_OVERRIDE") == "0",
                     "Allocated runtime environment differs from the fixed full-FP32 contract.")
        _require(self._placement_matches(pod), "Allocated data center or secure-cloud placement differs.")
        ephemeral = self.cfg.get("storage_mode") == "ephemeral_host_pull"
        _require(pod.get("volumeInGb") == self.cfg["volume_gb"]
                 and (ephemeral or pod.get("volumeMountPath") == self.cfg.get("volume_mount_path", "/workspace")),
                 "Allocated persistent volume differs from the approved request.")
        if ephemeral:
            _require(pod.get("containerDiskInGb") == self.cfg["container_disk_gb"], "Adopted ephemeral disk size differs.")
        rate = _number(pod.get("costPerHr", pod.get("cost")), "actual pod hourly price")
        _require(rate <= self.cfg["quote"]["hourly_compute_usd"] + self.cfg["quote"]["running_storage_hourly_usd"],
                 "Actual pod price exceeds the admitted quote.")
        self._no_other_gpu(self._pods(), own=owned)
        if isinstance(previous, str):
            _require(previous == state["session_id"], "Runpod session identity changed.")
        elif previous:
            for key in ("session_id", "pod_id", "allocation_started_unix", "deadline_unix", "hard_stop_unix"):
                if key in previous:
                    _require(previous[key] == state.get(key), "Runpod session identity or deadline changed.")
        self.heartbeat()
        # A watchdog STOP may win while the provider reads are in flight.
        # Recheck and publish admission under its lock so terminal state cannot
        # be overwritten by a stale successful provider response.
        with self._lock():
            current = self._state()
            _require(current.get("session_id") == state["session_id"]
                     and not current.get("stop_requested")
                     and current["phase"] not in {"stopped", "deleted"}
                     and time.time() < current["deadline_unix"],
                     "Session admission closed during provider verification.")
            current.update(phase="running", provider_verified_at=_utc(time.time()), pod=_public_pod(pod),
                           precision_environment_origin=self.cfg.get("precision_environment_origin", "pod"))
            _write(self.state_path, current)
            self._cached_state = dict(current)
            state = current
        return self._session_metadata(state)

    def stop(self) -> dict[str, Any]:
        """Request stop and verify a fresh stopped provider state; never delete data."""
        state = self._cleanup_update(stop_requested=True)
        if (not state.get("create_attempted") and not state.get("adopted")) or state.get("create_cancelled_before_post"):
            receipt = {"status": "CREATE_CANCELLED_BEFORE_POST" if state.get("create_cancelled_before_post")
                       else "NO_CREATE_ATTEMPT", "verified_at": _utc(time.time())}
            # Before create, the closed intent must be durable before the guard
            # exits; otherwise a recovering writer could revive admission.
            self._update(phase="stopped", stop_receipt=receipt)
            return receipt
        owned = self._resolve()
        rejection = state.get("create_error", "")
        if (not owned and isinstance(rejection, str)
                and rejection.startswith("Runpod create failed: bad_request: failed to create pod:")
                and "There are no longer any instances available with the requested specifications." in rejection):
            # An explicit capacity rejection is different from a lost POST reply.
            # Reconcile a fresh complete inventory before ending its stop guard.
            rows = self._pods()
            _require(not any(pod.get("name") == self.cfg["pod_name"] for pod in rows),
                     "A pod appeared after the capacity rejection; shutdown remains unresolved.")
            receipt = {"status": "CREATE_REJECTED_NO_POD", "provider_absence_verified": True,
                       "create_error_sha256": hashlib.sha256(rejection.encode()).hexdigest(),
                       "observed_pod_ids": sorted(pod["id"] for pod in rows), "verified_at": _utc(time.time())}
            self._cleanup_update(phase="stopped", stop_receipt=receipt)
            return receipt
        _require(owned, "Allocation result remains unresolved; watchdog continues name reconciliation.")
        until = time.monotonic() + 90
        last_error = ""
        while time.monotonic() < until:
            try:
                pod = self._call("get", owned)
                _require(pod.get("id") == owned, "Stop read returned another pod.")
                if _stopped(pod):
                    receipt = {"pod_id": owned, "status": "EXITED", "runtime_status": "stopped",
                               "verified_at": _utc(time.time()), "pod": _public_pod(pod)}
                    self._cleanup_update(phase="stopped", stop_receipt=receipt)
                    return receipt
                self._call("stop", owned)
            except RunpodError as exc:
                last_error = str(exc)
            time.sleep(1)
        raise RunpodError(f"Stop is not verified; independent watchdog keeps retrying. {self._redact(last_error)}")

    def admit_workload(self) -> None:
        """Permanently distinguish an execution from a pre-workload qualification failure."""
        self.session()
        self._update(workload_admitted=True)

    def register_output_roots(self, roots: list[dict[str, str]]) -> None:
        """Bind every execution tree before additional work can require its preservation."""
        self._unchanged()
        _require(isinstance(roots, list) and roots, "Execution output roots are required.")
        local_roots, remote_roots = [], []
        for row in roots:
            _require(set(row) == {"execution_root", "primary_root", "mirror_root"}, "Invalid execution output root binding.")
            remote = Path(row["execution_root"])
            _require(remote.is_absolute() and ".." not in remote.parts and str(remote) == row["execution_root"],
                     "Execution output root must be canonical and absolute.")
            remote_roots.append(remote)
            for key in ("primary_root", "mirror_root"):
                root = Path(row[key])
                _require(root.is_absolute() and root.resolve() == root, "Preservation destination must be canonical.")
                local_roots.append(root)
        for paths in (local_roots, remote_roots):
            _require(all(a != b and a not in b.parents and b not in a.parents
                         for index, a in enumerate(paths) for b in paths[index + 1:]),
                     "Execution preservation roots cannot alias or overlap.")
        current = self._state().get("required_preservation_roots")
        _require(current is None or current == roots, "The registered output roots cannot change during a session.")
        admitted = self._state().get("admitted_output_roots", [])
        if current is None and self._state().get("workload_admitted"):
            admitted = [roots[0]["execution_root"]]
        self._update(required_preservation_roots=roots, admitted_output_roots=admitted)

    def admit_output_root(self, execution_root: str) -> None:
        """Mark a registered tree before its first remote execution request."""
        self._unchanged()
        state = self._state()
        _require(not state.get("stop_requested") and state.get("phase") not in {"stopped", "deleted"},
                 "Output admission is closed with the allocation.")
        _require(execution_root in [row["execution_root"] for row in state.get("required_preservation_roots", [])],
                 "Output admission requires a pre-registered execution root.")
        admitted = list(state.get("admitted_output_roots", []))
        if execution_root not in admitted:
            admitted.append(execution_root)
            self._update(admitted_output_roots=admitted)

    def _verify_preservation(self, path: Path, *, _allow_set: bool = True) -> str:
        """Verify the complete collected inventory in two independent local copies."""
        _require(path.is_absolute() and path.is_file() and not path.is_symlink(), "Preservation receipt must be a regular local file.")
        proof, state = _read(path), self._state()
        identity = {"pod_id": self._owned_id(), "session_id": state["session_id"], "config_sha256": self.config_sha}
        if proof.get("format") == "deepmzyme_ec_clean30_runpod_preservation_set_v1":
            _require(_allow_set and all(proof.get(key) == value for key, value in identity.items()),
                     "Preservation set is nested or belongs to another allocation.")
            children = proof.get("receipts")
            _require(isinstance(children, list) and children, "Preservation set has no execution receipts.")
            roots, seen = [], set()
            for child in children:
                _require(isinstance(child, dict) and set(child) == {"path", "sha256"}, "Invalid preservation-set member.")
                child_path = Path(child["path"])
                _require(child_path.resolve() not in seen and child_path.resolve() != path.resolve(),
                         "Duplicate or recursive preservation-set receipt.")
                seen.add(child_path.resolve())
                _require(self._verify_preservation(child_path, _allow_set=False) == child["sha256"],
                         "Preservation-set member checksum differs.")
                record = _read(child_path)
                roots.append({key: record.get(key) for key in ("execution_root", "primary_root", "mirror_root")})
            expected = state.get("required_preservation_roots")
            _require(expected and roots == expected, "Preservation set omits or changes a registered execution root.")
            return _sha(path)
        _require(not _allow_set or not state.get("required_preservation_roots"),
                 "This allocation needs a complete preservation set before deletion.")
        _require(proof.get("format") == "deepmzyme_ec_clean30_runpod_preservation_v1"
                 and all(proof.get(key) == value for key, value in identity.items()),
                 "Preservation receipt belongs to another allocation.")
        roots = [Path(proof.get(key, "")) for key in ("primary_root", "mirror_root")]
        _require(all(root.is_absolute() and root.is_dir() and root.resolve() == root for root in roots)
                 and roots[0].stat().st_dev != roots[1].stat().st_dev,
                 "Preservation needs existing roots on two independent devices.")
        files = proof.get("files")
        _require(isinstance(files, list), "Preservation inventory is missing.")
        members: dict[str, dict[str, Any]] = {}
        for entry in files:
            _require(isinstance(entry, dict), "Invalid preservation inventory member.")
            name = entry.get("path", "")
            relative = Path(name)
            _require(isinstance(name, str) and name and not relative.is_absolute()
                     and ".." not in relative.parts and str(relative) == name and "\\" not in name
                     and name not in members and type(entry.get("size_bytes")) is int and entry["size_bytes"] >= 0
                     and re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", ""))),
                     "Unsafe or invalid preservation inventory member.")
            for root in roots:
                item = root / relative
                _require(root in item.resolve().parents and not item.is_symlink() and item.is_file()
                         and item.stat().st_size == entry["size_bytes"] and _sha(item) == entry["sha256"],
                         "Preservation bytes are missing or differ between local copies.")
            members[name] = entry
        from ec_clean30.execution_v5 import inherited_artifacts
        inherited = inherited_artifacts(self.config)
        # A recovered execution state alone cannot prove that local hot data was copied.
        for name in members:
            if Path(name).name in {"pretrain_attachment.json", "scientific_outputs_pending.json"}:
                remote_root = proof.get("execution_root")
                _require(isinstance(remote_root, str) and Path(remote_root).is_absolute(),
                         "External-output preservation needs its original execution root.")
                for root in roots:
                    try:
                        if Path(name).name == "pretrain_attachment.json":
                            from ec_clean30 import phase3

                            record = phase3.verify_preserved_pretrain_attachment(root / name,
                                original_attempt=Path(remote_root) / Path(name).parent)
                            owner = record["identity"]
                        else:
                            from ec_clean30 import phase4

                            phase4.verify_preserved_scientific_outputs(root / name,
                                original_attempt=Path(remote_root) / Path(name).parent)
                            owner = _read((root / name).parent / "request.json")
                        _require(all(owner.get(key) == identity[key] for key in ("session_id", "pod_id"))
                                 or inherited.get(str(Path(remote_root) / name)) == members[name]["sha256"],
                                 "External-output snapshot belongs to another allocation.")
                    except (OSError, ValueError, KeyError) as exc:
                        raise RunpodError(f"External scientific outputs are not completely preserved: {exc}") from exc
            if re.fullmatch(r"runs/[^/]+/attempt[1-9][0-9]*/hot_storage\.json", name):
                from ec_clean30 import hot_storage

                remote_root = proof.get("execution_root")
                _require(isinstance(remote_root, str) and Path(remote_root).is_absolute(),
                         "Hot preservation needs its original execution root.")
                for root in roots:
                    try:
                        hot_storage.verify_preserved_attempt((root / name).parent,
                            original_attempt=Path(remote_root) / Path(name).parent)
                    except (OSError, ValueError, hot_storage.HotStorageError) as exc:
                        raise RunpodError(f"Hot attempt is not completely preserved: {exc}") from exc
        inventory_name = proof.get("remote_inventory_path")
        _require(inventory_name in members, "Complete remote inventory must be preserved in both copies.")
        inventory = _read(roots[0] / inventory_name)
        pending = inventory.get("pending_unit")
        _require(inventory.get("format") == "deepmzyme_ec_clean30_runpod_remote_inventory_v1"
                 and all(inventory.get(key) == value for key, value in identity.items())
                 and inventory.get("pending_transfer") is None
                 and (pending is None or (isinstance(pending, dict) and pending.get("persisted") is True
                                          and pending.get("status") != "running")),
                 "Remote inventory is foreign or contains unacknowledged work.")
        remote_files = inventory.get("files")
        _require(isinstance(remote_files, list) and all(isinstance(entry, dict) for entry in remote_files)
                 and len({entry.get("path") for entry in remote_files}) == len(remote_files)
                 and all(entry == members.get(entry.get("path")) for entry in remote_files),
                 "Collected bytes do not cover the complete remote inventory.")
        manifests = proof.get("host_pull_manifests", [])
        _require(isinstance(manifests, list) and len(set(manifests)) == len(manifests), "Invalid HostPull manifest list.")
        inherited_only = bool(inherited) and bool(remote_files) and bool(proof.get("execution_root")) and all(
            inherited.get(str(Path(proof["execution_root"]) / entry["path"])) == entry["sha256"] for entry in remote_files)
        admitted = state.get("workload_admitted")
        if not _allow_set and state.get("required_preservation_roots"):
            _require(proof.get("execution_root") in [row["execution_root"] for row in state["required_preservation_roots"]],
                     "Preservation child names an unregistered execution root.")
            admitted = proof["execution_root"] in state.get("admitted_output_roots", [])
        if admitted:
            _require(remote_files and (manifests or inherited_only), "An admitted workload needs its complete artifacts and HostPull acknowledgments.")
        else:
            _require(not remote_files or manifests or inherited_only, "Nonempty execution artifacts require HostPull acknowledgments.")
        for name in manifests:
            _require(name in members, "A HostPull manifest is absent from the preserved inventory.")
            manifest = _read(roots[0] / name)
            ack_name = manifest.get("ack_path")
            _require(manifest.get("persistence_mode") == "host_pull" and manifest.get("schema_version") == 1
                     and manifest.get("destination_root") == str(roots[0]) and ack_name in members,
                     "HostPull manifest or acknowledgment is missing or foreign.")
            ack = _read(roots[0] / ack_name)
            payload = manifest.get("files", [])
            _require(isinstance(payload, list) and ack.get("manifest_sha256") == members[name]["sha256"]
                     and ack.get("verified_root") == str(roots[0]) and ack.get("file_count") == len(payload)
                     and all(entry == members.get(entry.get("path")) for entry in payload),
                     "HostPull acknowledgment does not cover its preserved payload.")
        return _sha(path)

    def closeout(self, preservation_path: Path) -> dict[str, Any]:
        """Verify preservation and stop; delete only allocations this controller created."""
        owned = self._owned_id()
        _require(owned, "No owned Runpod allocation is bound for closeout.")
        state = self._state()
        if state.get("adopted") is True:
            try:
                proof_sha = self._verify_preservation(Path(preservation_path))
            except BaseException:
                self.stop()
                raise
            stopped = self.stop()
            receipt = {"pod_id": owned, "session_id": state["session_id"], "status": "STOPPED",
                       "provider_stop_verified": stopped.get("runtime_status") == "stopped",
                       "preservation_sha256": proof_sha, "verified_at": _utc(time.time()),
                       "ephemeral_data_erased_on_stop": self.cfg.get("storage_mode") == "ephemeral_host_pull",
                       "deletion_authorized": False}
            self._update(preservation_sha256=proof_sha, preservation_path=str(preservation_path), closeout_receipt=receipt)
            return receipt
        proof_sha = self._verify_preservation(Path(preservation_path))
        if state.get("phase") != "deleted":
            self.stop()
            self._update(preservation_sha256=proof_sha, preservation_path=str(preservation_path))
            try:
                self._call("delete", owned)
            except RunpodError:
                # A lost DELETE reply is reconciled from a fresh list, never a new pod.
                if any(pod["id"] == owned for pod in self._pods()):
                    raise
        _require(not any(pod["id"] == owned for pod in self._pods()), "Provider deletion is not verified.")
        receipt = {"pod_id": owned, "session_id": state["session_id"], "status": "DELETED",
                   "provider_absence_verified": True, "preservation_sha256": proof_sha, "verified_at": _utc(time.time())}
        self._update(phase="deleted", delete_receipt=receipt)
        return receipt

    def write_ssh_config(self) -> Path:
        self.session()
        pod = self._call("get", self._owned_id())
        ssh = pod.get("ssh", {})
        direct = ssh.get("direct", ssh)
        host, port = direct.get("host", direct.get("ip")), direct.get("port")
        try:
            ipaddress.ip_address(host)
        except ValueError:
            raise RunpodError("Provider did not return a direct SSH IP address.") from None
        _require(type(port) is int and 1 <= port <= 65535, "Provider did not return a valid SSH port.")
        username = direct.get("username", "root")
        _require(re.fullmatch(r"[A-Za-z0-9_-]+", username), "Unsafe SSH username.")
        key = Path(self.cfg["ssh_private_key"])
        _require(key.is_file() and not stat.S_IMODE(key.stat().st_mode) & 0o077, "SSH private key is absent or permissions are unsafe.")
        path = self.root / "ssh_config"
        contents = (f"Host {self.cfg['ssh_alias']}\n  HostName {host}\n  Port {port}\n  User {username}\n"
                    f"  IdentityFile {json.dumps(str(key))}\n  IdentitiesOnly yes\n  BatchMode yes\n"
                    f"  UserKnownHostsFile {json.dumps(str(self.root / 'known_hosts'))}\n  StrictHostKeyChecking accept-new\n")
        path.write_text(contents)
        os.chmod(path, 0o600)
        return path


def _watchdog(snapshot_path: Path) -> None:
    snapshot = _read(snapshot_path)
    controller = RunpodController(snapshot["config"], Path(snapshot["config_path"]), _snapshot=snapshot)
    cfg = controller.cfg
    interval = min(10, controller.cost["stop_reserve_seconds"] / 4)
    while True:
        state = controller._cleanup_state()
        if state["phase"] in {"stopped", "deleted"}:
            return
        bookkeeping_failed = False
        try:
            _write(controller.watch_path, {"session_id": state["session_id"], "config_sha256": controller.config_sha,
                                          "pid": os.getpid(), "updated_unix": time.time(), "status": "armed"})
            beat = _read(controller.heartbeat_path) if controller.heartbeat_path.exists() else {}
        except (OSError, ValueError):
            bookkeeping_failed, beat = True, {}
        owner_stale = beat.get("session_id") != state["session_id"] or time.time() - beat.get("updated_unix", 0) > cfg.get("heartbeat_timeout_seconds", 900)
        expired = state.get("deadline_unix") is not None and time.time() >= state["deadline_unix"]
        if state.get("stop_requested") or expired or owner_stale or bookkeeping_failed:
            try:
                # stop() re-reads and closes the intent under the same lock used
                # for create admission; a stale pre-create read cannot win later.
                controller.stop()
                return
            except Exception as exc:
                # Transient network/auth failures must not disable the stop loop.
                controller._cleanup_update(watchdog_last_error=controller._redact(str(exc)), watchdog_last_retry_at=_utc(time.time()))
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watchdog", type=Path, required=True)
    args = parser.parse_args()
    _watchdog(args.watchdog)


if __name__ == "__main__":
    main()
