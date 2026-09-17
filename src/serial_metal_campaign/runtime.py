"""Durable, serial execution accounting for the bounded metal campaign.

This module never provisions or stops a VM. A session starts when its GPU is
allocated, before setup, and closes only after the allocation is stopped.
Interrupted fits restart in a new directory; checkpoints are not resumed.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
import uuid


TOTAL_SECONDS = 20 * 3600
DISCOVERY_SECONDS = 6 * 3600
CONFIRMATION_SECONDS = 10 * 3600
OPERATIONS_SECONDS = 4 * 3600
SESSION_SECONDS = 4 * 3600
CLOSEOUT_SECONDS = 15 * 60
ADMISSION_FACTOR = 1.25
POLL_SECONDS = 0.2
TERM_GRACE_SECONDS = 10.0
KILL_GRACE_SECONDS = 5.0
STAGES = {"discovery", "confirmation", "operations"}
TERMINAL_STATES = {"completed", "failed", "interrupted", "deadline_stopped"}
BUDGET_LIMIT_KEYS = ("total_seconds", "discovery_seconds", "confirmation_seconds",
                     "operations_seconds", "session_seconds", "closeout_seconds")
DEFAULT_BUDGET_LIMITS = {
    "total_seconds": TOTAL_SECONDS,
    "discovery_seconds": DISCOVERY_SECONDS,
    "confirmation_seconds": CONFIRMATION_SECONDS,
    "operations_seconds": OPERATIONS_SECONDS,
    "session_seconds": SESSION_SECONDS,
    "closeout_seconds": CLOSEOUT_SECONDS,
}


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _number(value, name):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite nonnegative number.")
    return number


def _now(value=None):
    return _number(time.time() if value is None else value, "now")


def _limit_values(value, name):
    if not isinstance(value, dict) or set(value) != set(BUDGET_LIMIT_KEYS):
        raise ValueError(f"{name} must contain exactly {', '.join(BUDGET_LIMIT_KEYS)}.")
    return {key: _number(value[key], f"{name}.{key}") for key in BUDGET_LIMIT_KEYS}


def budget_limits(output):
    """Return default or explicitly user-authorized cumulative GPU ceilings.

    Authorization records may only raise cumulative phase/total ceilings. The
    four-hour per-session limit and fifteen-minute closeout reserve stay fixed.
    """
    records = read_json(Path(output) / "budget_authorizations.json", [])
    if not isinstance(records, list):
        raise ValueError("Budget authorizations must be a JSON list.")
    limits = {key: float(value) for key, value in DEFAULT_BUDGET_LIMITS.items()}
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError("Each budget authorization must be an object.")
        if (record.get("sequence") != index or record.get("status") != "authorized"
                or record.get("authorized_by") != "user"
                or record.get("decision") != "continue_beyond_planned_ceiling"
                or record.get("held_out_evaluation") is not False
                or not str(record.get("authorized_at", "")).strip()
                or not str(record.get("reason", "")).strip()):
            raise ValueError("Budget authorization lacks the required user decision and audit fields.")
        previous = _limit_values(record.get("previous_limits_seconds"), "previous_limits_seconds")
        if previous != limits:
            raise ValueError("Budget authorization chain does not match the preceding ceilings.")
        extended = _limit_values(record.get("authorized_limits_seconds"), "authorized_limits_seconds")
        for key in ("total_seconds", "discovery_seconds", "confirmation_seconds", "operations_seconds"):
            if extended[key] < limits[key]:
                raise ValueError("An authorization cannot reduce a prior cumulative ceiling.")
        if (extended["session_seconds"] != DEFAULT_BUDGET_LIMITS["session_seconds"]
                or extended["closeout_seconds"] != DEFAULT_BUDGET_LIMITS["closeout_seconds"]):
            raise ValueError("Budget authorization cannot weaken session or closeout safety limits.")
        if extended["total_seconds"] < sum(extended[f"{stage}_seconds"] for stage in STAGES):
            raise ValueError("The total ceiling must cover all authorized phase ceilings.")
        if extended == limits:
            raise ValueError("A budget authorization must increase at least one cumulative ceiling.")
        limits = extended
    digest = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**limits, "authorization_count": len(records), "authorization_sha256": digest}


def authorize_budget(output, authorization):
    """Append one audited user decision while no allocation or fit is active."""
    with _lock(output) as output:
        if any(row.get("stopped_epoch") is None for row in _sessions(output)):
            raise RuntimeError("Close the active allocation before changing cumulative ceilings.")
        if any(row.get("status") == "running" for row in _attempts(output)):
            raise RuntimeError("Reconcile the active fit before changing cumulative ceilings.")
        records = read_json(output / "budget_authorizations.json", [])
        if not isinstance(records, list):
            raise ValueError("Budget authorizations must be a JSON list.")
        candidate = [*records, authorization]
        with tempfile.TemporaryDirectory() as directory:
            atomic_json(Path(directory) / "budget_authorizations.json", candidate)
            checked = budget_limits(directory)
        atomic_json(output / "budget_authorizations.json", candidate)
        return checked


def _identity():
    boot = Path("/proc/sys/kernel/random/boot_id")
    if not boot.is_file():
        raise RuntimeError("Linux boot identity is required for safe process ownership.")
    return {"hostname": socket.gethostname(), "boot_id": boot.read_text().strip()}


def _process_info(pid):
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        return {"state": fields[0], "pgrp": int(fields[2]), "start_token": fields[19]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def _group_alive(pgid):
    # Ignore unreapable orphan zombies; their training work is no longer live.
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            info = _process_info(entry.name)
            if info and info["pgrp"] == int(pgid) and info["state"] != "Z":
                return True
    return False


@contextmanager
def _lock(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".runtime.lock").open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another campaign controller holds the serial execution lock.") from exc
        try:
            yield output
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def _worker_lock():
    """One training worker across campaign directories on the current host."""
    path = Path(tempfile.gettempdir()) / f"deepmzyme_serial_metal_worker_{os.getuid()}.lock"
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another serial metal campaign already owns the host training worker.") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _sessions(output):
    return read_json(Path(output) / "sessions.json", [])


def _attempts(output):
    return read_json(Path(output) / "attempts.json", [])


def _current_session(output):
    opened = [row for row in _sessions(output) if row.get("stopped_epoch") is None]
    if len(opened) != 1:
        raise RuntimeError("Exactly one open GPU allocation session is required.")
    return opened[0]


def _active_dead(output, active, session_override=None):
    session = session_override or next((row for row in _sessions(output) if row["session_id"] == active["session_id"]), None)
    evidence = (session or {}).get("stop_evidence", {})
    stopped = (session or {}).get("stopped_epoch")
    verified_stopped = (stopped is not None and evidence.get("provider_verified_stopped") is True
                        and evidence.get("session_id") == active["session_id"]
                        and evidence.get("stopped_epoch") == stopped)
    same_host = all(active.get(key) == value for key, value in _identity().items())
    if same_host:
        pid = active.get("pid")
        if pid is None:
            controller = _process_info(active["controller_pid"])
            if controller and controller["start_token"] == active["controller_start_token"]:
                raise RuntimeError("The previous launch controller is still live.")
            if verified_stopped:
                return "provider_verified_session_stopped"
            raise RuntimeError("Cannot prove launch-window child death without provider-verified session stop evidence.")
        elif _group_alive(active["pgid"]):
            raise RuntimeError("The previous owned training process group is still live.")
        return "local_process_group_absent"
    if verified_stopped:
        return "provider_verified_session_stopped"
    raise RuntimeError("Cannot prove the previous remote session stopped; record provider stop evidence first.")


def open_session(output, session_id, started_epoch, hardware, now=None):
    """Register an actual allocation, including setup time, without allocating it."""
    now = _now(now)
    started = _number(started_epoch, "started_epoch")
    if started > now or not str(session_id).strip() or not isinstance(hardware, dict) or not hardware:
        raise ValueError("Provide a session ID, hardware identity, and an allocation start no later than now.")
    with _lock(output) as output:
        sessions = _sessions(output)
        existing = next((row for row in sessions if row["session_id"] == session_id), None)
        if existing:
            if existing["started_epoch"] != started or existing["hardware"] != hardware:
                raise ValueError("Session allocation inputs are immutable.")
            if existing.get("stopped_epoch") is not None:
                raise ValueError("A closed session cannot be reopened.")
            return existing
        if any(row.get("stopped_epoch") is None for row in sessions):
            raise RuntimeError("Close the previous allocation before registering another session.")
        if sessions and started < max(row["stopped_epoch"] for row in sessions):
            raise ValueError("GPU allocation intervals must not overlap.")
        limits = budget_limits(output)
        if sum(row["stopped_epoch"] - row["started_epoch"] for row in sessions) >= limits["total_seconds"]:
            raise RuntimeError("The cumulative campaign allocation budget is exhausted.")
        active = read_json(output / "active_process.json")
        if active:
            _active_dead(output, active)
        record = {"session_id": str(session_id), "started_epoch": started,
                  "hardware": hardware, "stopped_epoch": None, **_identity()}
        sessions.append(record)
        atomic_json(output / "sessions.json", sessions)
        return record


def close_session(output, session_id, stopped_epoch, *, stop_evidence=None):
    """Record actual teardown only after matching provider stop verification."""
    stopped = _number(stopped_epoch, "stopped_epoch")
    with _lock(output) as output:
        sessions = _sessions(output)
        session = next((row for row in sessions if row["session_id"] == session_id), None)
        if session is None:
            raise ValueError("Unknown session ID.")
        if stopped < session["started_epoch"]:
            raise ValueError("A session cannot stop before it started.")
        if session.get("stopped_epoch") is not None:
            if session["stopped_epoch"] != stopped:
                raise ValueError("A closed allocation interval is immutable.")
            return session
        same_host = all(session.get(key) == value for key, value in _identity().items())
        evidence = stop_evidence or {}
        if (evidence.get("provider_verified_stopped") is not True
                or evidence.get("session_id") != session_id
                or evidence.get("stopped_epoch") != stopped):
            raise RuntimeError("Every allocation needs matching provider-verified stop evidence.")
        active = read_json(output / "active_process.json")
        if active and active["session_id"] == session_id and same_host:
            _active_dead(output, active, {**session, "stopped_epoch": stopped, "stop_evidence": evidence})
        attempt_ends = [row.get("ended_epoch", row["started_epoch"]) for row in _attempts(output)
                        if row["session_id"] == session_id]
        if attempt_ends and stopped < max(attempt_ends):
            raise ValueError("Allocation stop precedes recorded attempt work.")
        session["stopped_epoch"] = stopped
        session["stop_evidence"] = evidence
        atomic_json(output / "sessions.json", sessions)
        return session


def budget_status(output, now=None):
    """Account for all allocated seconds, including setup, failures and idle time."""
    now = _now(now)
    sessions = _sessions(output)
    allocated = 0.0
    by_session = {}
    previous_end = None
    for session in sessions:
        started = _number(session["started_epoch"], "allocation start")
        end = session.get("stopped_epoch")
        end = now if end is None else _number(end, "allocation stop")
        if end < started:
            raise ValueError("Clock precedes a recorded allocation start.")
        if session["session_id"] in by_session or (previous_end is not None and started < previous_end):
            raise ValueError("Duplicate or overlapping allocation sessions.")
        if previous_end is not None and any(row.get("stopped_epoch") is None for row in sessions[:len(by_session)]):
            raise ValueError("An open allocation cannot precede another allocation.")
        previous_end = end
        allocated += end - started
        by_session[session["session_id"]] = (started, end)
    used = {stage: 0.0 for stage in STAGES}
    active_elapsed = 0.0
    active_remaining = 0.0
    active_count = 0
    seen = set()
    intervals = {}
    attempts = _attempts(output)
    for attempt in attempts:
        if attempt["attempt_id"] in seen or attempt["stage"] not in STAGES:
            raise ValueError("Duplicate attempt identity or invalid requested stage.")
        seen.add(attempt["attempt_id"])
        if attempt["status"] not in TERMINAL_STATES | {"running"}:
            raise ValueError("Unknown attempt accounting state.")
        if attempt["status"] in TERMINAL_STATES and "ended_epoch" not in attempt:
            raise ValueError("Terminal attempts require a frozen stop timestamp.")
        if attempt["session_id"] not in by_session:
            raise ValueError("Attempt refers to an unknown allocation.")
        start, session_end = by_session[attempt["session_id"]]
        began = _number(attempt["started_epoch"], "attempt start")
        end = _number(attempt.get("ended_epoch", min(now, session_end)), "attempt stop")
        if began < start or end > session_end + 0.001 or end < began:
            raise ValueError("Attempt timing falls outside its allocated session.")
        intervals.setdefault(attempt["session_id"], []).append((began, end))
        elapsed = end - began
        if attempt["status"] == "running":
            active_count += 1
            active_elapsed += elapsed
            active_remaining += max(0.0, attempt.get("admission", {}).get("deadline", now) - now)
        elif attempt["status"] == "completed" and attempt["stage"] in {"discovery", "confirmation"}:
            used[attempt["stage"]] += elapsed
    if active_count > 1:
        raise ValueError("Multiple active attempts violate serial allocation accounting.")
    for session_intervals in intervals.values():
        ordered = sorted(session_intervals)
        if any(right[0] < left[1] for left, right in zip(ordered, ordered[1:])):
            raise ValueError("Overlapping attempt intervals would double-count allocation time.")
    # Every failed/interrupted fit, setup and idle second belongs to operations.
    # Active time is unsettled; it becomes scientific spend only after success.
    operations = allocated - used["discovery"] - used["confirmation"] - active_elapsed
    if operations < -0.001:
        raise ValueError("Attempt accounting exceeds the actual allocation intervals.")
    operations = max(0.0, operations)
    limits = budget_limits(output)
    closed = read_json(Path(output) / "discovery_closed.json")
    if closed and (closed.get("status") != "closed" or closed.get("accounting_version") != 2
                   or closed.get("discovery_spent_seconds") != used["discovery"]
                   or closed.get("transferred_seconds") != max(0.0, limits["discovery_seconds"] - used["discovery"])):
        raise ValueError("Invalid or changed frozen discovery closure.")
    transferred = closed["transferred_seconds"] if closed else 0.0
    caps = {"discovery": limits["discovery_seconds"],
            "confirmation": limits["confirmation_seconds"] + transferred,
            "operations": limits["operations_seconds"]}
    used["operations"] = operations
    reservations = _reservation_totals(output, attempts, now)
    return {"allocated_seconds": allocated,
            "total_remaining_seconds": max(0.0, limits["total_seconds"] - allocated),
            "used_seconds": used, "caps_seconds": caps,
            "remaining_seconds": {key: max(0.0, caps[key] - used[key]) for key in STAGES},
            "reserved_seconds": reservations, "active_elapsed_seconds": active_elapsed,
            "active_remaining_seconds": active_remaining,
            "failure_liability_seconds": active_elapsed + active_remaining,
            "accounting_version": 2,
            "discovery_closed": bool(closed), "discovery_transferred_seconds": transferred,
            "limit_seconds": limits}


def _reservation_remaining(reservation, attempts, now):
    if reservation.get("released_reason"):
        return 0.0
    remaining = 0.0
    for run_id, forecast in reservation["forecasts"].items():
        matching = [row for row in attempts if row["run_id"] == run_id]
        if any(row["status"] == "completed" for row in matching):
            continue
        active = next((row for row in matching if row["status"] == "running"), None)
        elapsed = max(0.0, now - active["started_epoch"]) if active else 0.0
        remaining += max(float(forecast) * ADMISSION_FACTOR, elapsed)
    return remaining


def _reservation_totals(output, attempts, now, *, exclude=None):
    totals = {stage: 0.0 for stage in STAGES}
    for row in read_json(Path(output) / "comparisons.json", []):
        if row["block_id"] != exclude:
            totals[row["stage"]] += _reservation_remaining(row, attempts, now)
    return totals


def reserve_comparison(output, block_id, stage, runs, forecasts, confirmation_reserve_seconds=0, now=None):
    """Reserve the full logical comparison cumulatively, across multiple sessions."""
    now = _now(now)
    if stage not in STAGES or not block_id:
        raise ValueError("A comparison needs an identity and valid stage.")
    identities = [row["id"] if isinstance(row, dict) else str(row) for row in runs]
    if len(identities) != len(set(identities)) or set(identities) != set(forecasts) or not identities:
        raise ValueError("Reserve every distinct comparison unit with its measured forecast.")
    checked = {key: _number(value, "unit forecast") for key, value in forecasts.items()}
    if any(value <= 0 for value in checked.values()):
        raise ValueError("Comparison forecasts must be positive.")
    identity = {"block_id": str(block_id), "stage": stage, "run_ids": identities, "forecasts": checked,
                "run_fingerprints": {row["id"]: _fingerprint(row) for row in runs if isinstance(row, dict)}}
    with _lock(output) as output:
        reservations = read_json(output / "comparisons.json", [])
        existing = next((row for row in reservations if row["block_id"] == block_id), None)
        if existing:
            if any(existing.get(key) != value for key, value in identity.items()):
                raise ValueError("Logical comparison membership and initial forecast are immutable.")
            return existing
        if any(set(row["run_ids"]) & set(identities) for row in reservations):
            raise ValueError("A run cannot be reserved in multiple logical comparisons.")
        status = budget_status(output, now)
        if any(row["status"] == "running" for row in _attempts(output)):
            raise RuntimeError("Reserve comparisons between attempts, after active costs settle.")
        if stage == "discovery" and status["discovery_closed"]:
            raise ValueError("Discovery is already closed.")
        required = sum(checked.values()) * ADMISSION_FACTOR
        closeout = status["limit_seconds"]["closeout_seconds"]
        total_free = status["total_remaining_seconds"] - sum(status["reserved_seconds"].values()) - closeout
        if stage != "confirmation":
            total_free -= _number(confirmation_reserve_seconds, "confirmation reserve")
        free_stage = status["remaining_seconds"][stage] - status["reserved_seconds"][stage]
        if required > min(total_free, free_stage):
            raise RuntimeError("The full logical comparison does not fit cumulative unreserved budgets.")
        record = {**identity, "created_epoch": now, "required_seconds": required, "accounting_version": 2}
        reservations.append(record)
        atomic_json(output / "comparisons.json", reservations)
        return record


def comparison_status(output, block_id, now=None):
    record = next((row for row in read_json(Path(output) / "comparisons.json", []) if row["block_id"] == block_id), None)
    if record is None:
        raise ValueError("Unknown logical comparison.")
    attempts = _attempts(output)
    completed = {row["run_id"] for row in attempts if row["status"] == "completed"}
    pending = [run_id for run_id in record["run_ids"] if run_id not in completed]
    return {**record, "pending_run_ids": pending, "complete": not pending,
            "remaining_reserved_seconds": _reservation_remaining(record, attempts, _now(now))}


def release_comparison(output, block_id, reason):
    if not str(reason).strip():
        raise ValueError("A deferred comparison needs an explicit reason.")
    with _lock(output) as output:
        records = read_json(output / "comparisons.json", [])
        row = next((row for row in records if row["block_id"] == block_id), None)
        if row is None:
            raise ValueError("Unknown logical comparison.")
        if any(attempt["status"] == "running" and attempt["run_id"] in row["run_ids"] for attempt in _attempts(output)):
            raise RuntimeError("Cannot release a comparison while its fit is active.")
        if row.get("released_reason") not in (None, reason):
            raise ValueError("Comparison deferral reason is immutable.")
        row["released_reason"] = str(reason)
        atomic_json(output / "comparisons.json", records)
        return row


def freeze_discovery(output, now=None):
    with _lock(output) as output:
        now = _now(now)
        status = budget_status(output, now)
        existing = read_json(output / "discovery_closed.json")
        if existing:
            return existing
        if any(row["status"] == "running" and row["stage"] == "discovery" for row in _attempts(output)):
            raise RuntimeError("Cannot close discovery while a fit is active.")
        if status["reserved_seconds"]["discovery"] > 0:
            raise RuntimeError("Complete or explicitly defer outstanding discovery comparisons first.")
        record = {"status": "closed", "accounting_version": 2, "closed_epoch": now,
                  "discovery_spent_seconds": status["used_seconds"]["discovery"],
                  "transferred_seconds": max(0.0, status["caps_seconds"]["discovery"]
                                             - status["used_seconds"]["discovery"])}
        atomic_json(output / "discovery_closed.json", record)
        return record


def admission(output, stage, forecast_seconds, confirmation_reserve_seconds=0, now=None, *, block_id=None):
    """Admit one session-sized execution unit with a 25% forecast margin."""
    if stage not in STAGES:
        raise ValueError(f"Unknown campaign stage: {stage}")
    now = _now(now)
    forecast = _number(forecast_seconds, "forecast_seconds")
    reserve = _number(confirmation_reserve_seconds, "confirmation_reserve_seconds")
    if forecast <= 0:
        raise ValueError("A positive measured block forecast is required.")
    status = budget_status(output, now)
    limits = status["limit_seconds"]
    session = _current_session(output)
    reasons = []
    if status["active_elapsed_seconds"] or any(row["status"] == "running" for row in _attempts(output)):
        reasons.append("A fit is already active.")
    if stage == "discovery" and status["discovery_closed"]:
        reasons.append("Discovery was closed; its unused allocation cannot be reopened.")
    if block_id is not None:
        comparison = comparison_status(output, block_id, now)
        if comparison["stage"] != stage or comparison.get("released_reason"):
            reasons.append("Logical comparison stage differs or its reservation was released.")
    other = _reservation_totals(output, _attempts(output), now, exclude=block_id)
    available = status["total_remaining_seconds"] - limits["closeout_seconds"] - sum(other.values())
    if stage != "confirmation":
        available -= reserve
    available = min(available, status["remaining_seconds"][stage] - other[stage],
                    session["started_epoch"] + limits["session_seconds"] - limits["closeout_seconds"] - now)
    if status["remaining_seconds"]["operations"] < limits["closeout_seconds"]:
        reasons.append("Operations budget cannot cover the closeout reserve.")
    # A fit that fails must fit entirely in operations, without using closeout.
    available = min(available, status["remaining_seconds"]["operations"] - other["operations"]
                    - limits["closeout_seconds"])
    needed = forecast * ADMISSION_FACTOR
    if needed > max(0.0, available):
        reasons.append("The complete block forecast plus 25% margin does not fit the remaining budget.")
    return {"allowed": not reasons, "reason": " ".join(reasons) if reasons else "Complete block fits all budgets.",
            "deadline": now + min(needed, max(0.0, available)), "remaining": max(0.0, available),
            "forecast_seconds": forecast, "required_seconds": needed, "session_id": session["session_id"],
            "budget": status}


def artifact_inventory(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Artifact directory is missing: {directory}")
    inventory = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Artifact inventories do not accept symlinks: {path}")
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            inventory[str(path.relative_to(directory))] = {"sha256": digest.hexdigest(), "size": path.stat().st_size}
    if not inventory:
        raise ValueError("An empty artifact directory is not a persistence receipt.")
    return inventory


def _is_drive_mount(directory):
    directory = Path(directory).resolve()
    for line in Path("/proc/mounts").read_text().splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        source, mount, filesystem = fields[:3]
        if "drive" in (source + filesystem).lower() and filesystem.startswith("fuse"):
            mount = Path(mount.replace("\\040", " ")).resolve()
            if directory == mount or mount in directory.parents:
                return True
    return False


def _verify_receipt(attempt, receipt, *, verify_bytes=True):
    source = Path(attempt["run_dir"])
    observed = attempt["artifacts"]
    if receipt.get("artifacts") != observed:
        raise ValueError("Persistence inventory does not match immutable completed artifacts.")
    if receipt.get("attempt_id") != attempt["attempt_id"]:
        raise ValueError("Persistence receipt identifies a different attempt.")
    if not str(receipt.get("readback_root", "")).strip():
        raise ValueError("Persistence receipt requires an explicit readback directory.")
    readback = Path(receipt["readback_root"]).expanduser().resolve()
    method = receipt.get("method")
    if method == "mounted_drive":
        if verify_bytes and not _is_drive_mount(readback):
            raise ValueError("Mounted-Drive persistence requires an actual Drive mount.")
    elif method == "verified_transfer":
        if not receipt.get("destination_uri") or readback == source.resolve():
            raise ValueError("Transfer persistence needs a destination URI and independent readback directory.")
    else:
        raise ValueError("Persistence method must be mounted_drive or verified_transfer.")
    if verify_bytes:
        if artifact_inventory(source) != observed:
            raise ValueError("Persistence inventory does not match immutable completed artifacts.")
        if artifact_inventory(readback) != observed:
            raise ValueError("Persisted artifact readback hashes do not match the completed attempt.")
    return observed


def record_persistence(output, attempt_id, receipt):
    """Verify actual bytes after mounting or transferring, then freeze the receipt."""
    with _lock(output) as output:
        attempts = _attempts(output)
        terminal = [row for row in attempts if row["status"] in TERMINAL_STATES]
        attempt = next((row for row in terminal if row["attempt_id"] == attempt_id), None)
        if attempt is None or attempt is not terminal[-1]:
            raise ValueError("Persistence must identify the last terminal attempt, including failures.")
        payload = {**receipt, "attempt_id": attempt_id}
        _verify_receipt(attempt, payload)
        path = output / "persistence" / f"{attempt_id}.json"
        existing = read_json(path)
        if existing and existing != payload:
            raise ValueError("Verified persistence receipts are immutable.")
        atomic_json(path, payload)
        return payload


def _check_persistence(output, attempts, *, full):
    completed = [attempt for attempt in attempts if attempt["status"] in TERMINAL_STATES]
    for index, attempt in enumerate(completed):
        receipt = read_json(output / "persistence" / f"{attempt['attempt_id']}.json")
        if not receipt:
            raise RuntimeError(f"Verify persistent artifacts for {attempt['attempt_id']} before the next fit.")
        _verify_receipt(attempt, receipt, verify_bytes=full or index == len(completed) - 1)
    return [attempt["attempt_id"] for attempt in completed]


def verify_persistence(output, *, full=True):
    """Rehash all persisted artifacts for explicit verification/closeout by default."""
    return _check_persistence(Path(output), _attempts(output), full=full)


def _require_persistence(output, attempts):
    return _check_persistence(output, attempts, full=False)


def _pending_intents(output):
    attempts = {row["attempt_id"]: row for row in _attempts(output)}
    pending = []
    for intent in read_json(Path(output) / "launch_intents.json", []):
        attempt = attempts.get(intent["attempt_id"])
        if attempt:
            if any(attempt.get(key) != intent.get(key) for key in ("run_id", "session_id", "run_fingerprint")):
                raise ValueError("Attempt identity differs from its immutable launch intent.")
            if attempt["status"] in TERMINAL_STATES:
                continue
        pending.append(intent)
    if len(pending) > 1:
        raise RuntimeError("Multiple unresolved launch intents violate serial execution.")
    return pending


def _next_attempt_id(output):
    identifiers = {row["attempt_id"] for row in _attempts(output)}
    identifiers.update(row["attempt_id"] for row in read_json(Path(output) / "launch_intents.json", []))
    number = max((int(value.removeprefix("attempt_")) for value in identifiers), default=0) + 1
    return f"attempt_{number:05d}"


def _attempt_layout(output, run, previous):
    directory = Path(run["run_dir"]).resolve()
    if Path(output).resolve() not in directory.parents:
        raise ValueError("Run outputs must be inside the campaign output directory.")
    command = [str(value) for value in run["command"]]
    if previous:
        directory = directory.with_name(directory.name + "__retry1")
        if "--run-name" not in command or command.index("--run-name") + 1 >= len(command):
            raise ValueError("Retry commands require an explicit --run-name to preserve the original output.")
        command[command.index("--run-name") + 1] = directory.name
    return directory, command


def prepare_launch_intent(output, run):
    """Freeze prelaunch ownership before exporting non-Drive controller state.

    An intent alone never launches a process. A lost VM leaves this durable
    identity behind, so recovery cannot silently reset the one-retry limit.
    """
    from serial_metal_campaign.control import require_running

    require_running(output)
    with _lock(output) as output:
        session = _current_session(output)
        if any(session.get(key) != value for key, value in _identity().items()):
            raise RuntimeError("Launch intent must belong to the current worker host and boot.")
        fingerprint = _fingerprint(run)
        pending = _pending_intents(output)
        if pending:
            intent = pending[0]
            if (intent["run_id"] != run["id"] or intent["run_fingerprint"] != fingerprint
                    or intent["session_id"] != session["session_id"]):
                raise RuntimeError("Reconcile the unresolved prior launch intent before preparing another fit.")
            return intent
        if read_json(output / "active_process.json") or any(row["status"] == "running" for row in _attempts(output)):
            raise RuntimeError("Reconcile the interrupted process before preparing another launch intent.")
        attempts = _attempts(output)
        previous = [row for row in attempts if row["run_id"] == run["id"]]
        if any(row["run_fingerprint"] != fingerprint for row in previous):
            raise ValueError("A run identity cannot be reused with different inputs.")
        if any(row["status"] == "completed" for row in previous):
            return None
        if len(previous) >= 2:
            raise RuntimeError("Only one linked retry is permitted for a configuration.")
        _require_persistence(output, attempts)
        directory, command = _attempt_layout(output, run, previous)
        if directory.exists():
            raise FileExistsError(f"Refusing to overwrite untracked attempt output: {directory}")
        intent = {"intent_id": uuid.uuid4().hex, "attempt_id": _next_attempt_id(output),
                  "run_id": run["id"], "run_fingerprint": fingerprint, "session_id": session["session_id"],
                  "stage": run["stage"], "block": run["block"], "family": run.get("family"),
                  "arm": run.get("arm"), "epochs": run["epochs"], "run_dir": str(directory), "command": command,
                  "created_epoch": _now(), "retry_of": previous[-1]["attempt_id"] if previous else None,
                  **_identity()}
        intents = read_json(output / "launch_intents.json", [])
        intents.append(intent)
        atomic_json(output / "launch_intents.json", intents)
        return intent


def _reconcile_intents(output, allow_intent_id=None):
    for intent in _pending_intents(output):
        attempts = _attempts(output)
        existing = next((row for row in attempts if row["attempt_id"] == intent["attempt_id"]), None)
        if intent["intent_id"] == allow_intent_id and existing is None:
            return None  # The acknowledged prelaunch intent is about to execute.
        session = next((row for row in _sessions(output) if row["session_id"] == intent["session_id"]), None)
        proof = (session or {}).get("stop_evidence", {})
        stopped = (session or {}).get("stopped_epoch")
        if (stopped is None or proof.get("provider_verified_stopped") is not True
                or proof.get("session_id") != intent["session_id"] or proof.get("stopped_epoch") != stopped):
            raise RuntimeError("Unresolved launch intent requires provider-verified allocation death before recovery.")
        if stopped < intent["created_epoch"]:
            raise ValueError("Verified allocation stop precedes its launch intent.")
        directory = Path(intent["run_dir"])
        if output.resolve() not in directory.resolve().parents:
            raise ValueError("Recovered intent output escapes the campaign directory.")
        missing = not directory.is_dir()
        directory.mkdir(parents=True, exist_ok=True)
        attempt = existing or {key: intent[key] for key in (
            "attempt_id", "run_id", "run_fingerprint", "session_id", "stage", "block", "family",
            "arm", "epochs", "run_dir", "command", "retry_of")}
        started = attempt.get("started_epoch", intent["created_epoch"])
        attempt.update(status="interrupted", started_epoch=started, ended_epoch=stopped,
                       elapsed_seconds=stopped-started, charged_stage="operations",
                       recovered_from_launch_intent=intent["intent_id"], launch_status_unknown=existing is None,
                       artifact_loss_detected=missing, death_proof="provider_verified_session_stopped")
        atomic_json(directory / "interruption.json", {key: value for key, value in attempt.items() if key != "artifacts"})
        attempt["artifacts"] = artifact_inventory(directory)
        if existing is None:
            attempts.append(attempt)
        atomic_json(output / "attempts.json", attempts)
        return attempt
    return None


def _reconcile_interrupted(output, allow_intent_id=None):
    active = read_json(output / "active_process.json")
    if not active:
        recovered = _reconcile_intents(output, allow_intent_id)
        attempts = _attempts(output)
        running = [row for row in attempts if row["status"] == "running"]
        if len(running) > 1:
            raise RuntimeError("Multiple ownerless running attempts require a ledger audit.")
        if running:
            # A durable ledger can survive a VM loss between the attempt write
            # and the process-token write. Only provider-confirmed VM absence
            # proves that no launch-window child could remain alive.
            attempt = running[0]
            session = next((row for row in _sessions(output) if row["session_id"] == attempt["session_id"]), None)
            evidence = (session or {}).get("stop_evidence", {})
            stopped = (session or {}).get("stopped_epoch")
            if (stopped is None or evidence.get("provider_verified_stopped") is not True
                    or evidence.get("session_id") != attempt["session_id"] or evidence.get("stopped_epoch") != stopped):
                raise RuntimeError("Ownerless running attempt requires provider-verified allocation death before recovery.")
            if stopped < attempt["started_epoch"]:
                raise ValueError("Verified allocation stop precedes its recorded attempt.")
            directory = Path(attempt["run_dir"])
            if output.resolve() not in directory.resolve().parents:
                raise ValueError("Recovered attempt output escapes the campaign directory.")
            missing = not directory.is_dir()
            directory.mkdir(parents=True, exist_ok=True)
            attempt.update(status="interrupted", ended_epoch=stopped,
                           elapsed_seconds=stopped-attempt["started_epoch"], charged_stage="operations",
                           recovered_without_process_token=True, launch_status_unknown=True,
                           artifact_loss_detected=missing, death_proof="provider_verified_session_stopped")
            atomic_json(directory / "interruption.json", {key: value for key, value in attempt.items() if key != "artifacts"})
            attempt["artifacts"] = artifact_inventory(directory)
            atomic_json(output / "attempts.json", attempts)
            return attempt
        return recovered
    proof = _active_dead(output, active)
    attempts = _attempts(output)
    attempt = next((row for row in attempts if row["attempt_id"] == active["attempt_id"]), None)
    if attempt is None:
        raise RuntimeError("Active process is missing its immutable attempt identity.")
    if attempt["status"] == "running":
        session = next(row for row in _sessions(output) if row["session_id"] == attempt["session_id"])
        ended = session.get("stopped_epoch")
        ended = _now() if ended is None else ended
        attempt.update(status="interrupted", ended_epoch=ended,
                       elapsed_seconds=max(0.0, ended - attempt["started_epoch"]), death_proof=proof,
                       charged_stage="operations")
        directory = Path(attempt["run_dir"])
        if not directory.is_dir():
            raise RuntimeError("Recover interrupted attempt artifacts before reconciling its ledger.")
        atomic_json(directory / "interruption.json", {"attempt_id": attempt["attempt_id"],
                    "death_proof": proof, "ended_epoch": ended})
        attempt["artifacts"] = artifact_inventory(directory)
        atomic_json(output / "attempts.json", attempts)
    (output / "active_process.json").unlink()
    return attempt


def reconcile_interrupted(output):
    """Reconcile only after proof the former process group or allocation is dead."""
    with _lock(output) as output:
        return _reconcile_interrupted(output)


def _fingerprint(run):
    ephemeral = {"forecast_seconds", "block_forecast_seconds", "estimated_seconds", "profiled_seconds", "measured_seconds"}
    identity = {key: value for key, value in run.items() if key not in ephemeral}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _stop_group(process, deadline=None):
    current = _process_info(process.pid)
    expected = getattr(process, "_campaign_start_token", None)
    if current and expected is not None and current["start_token"] != expected:
        raise RuntimeError("Process identity changed; refusing to signal an unowned process group.")
    if process.poll() is None or _group_alive(process.pid):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        term_grace = TERM_GRACE_SECONDS
        if deadline is not None:
            term_grace = min(term_grace, max(0.0, deadline - _now()) / 2)
        end = time.monotonic() + term_grace
        while _group_alive(process.pid) and time.monotonic() < end:
            process.poll()
            time.sleep(min(POLL_SECONDS, max(0.0, end - time.monotonic())))
        if _group_alive(process.pid):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        kill_grace = KILL_GRACE_SECONDS if deadline is None else min(KILL_GRACE_SECONDS, max(0.01, deadline - _now()))
        process.wait(timeout=kill_grace)
        if _group_alive(process.pid):
            raise RuntimeError("Owned training process group is still live after termination.")


def execute_attempt(output, run, root, verify_result, confirmation_reserve_seconds=0,
                    block_forecast_seconds=None, *, block_id=None, launch_guard=None,
                    training_deadline_epoch=None, launch_intent_id=None):
    """Execute one fit/retry under an exclusive lock; never advance automatically."""
    from serial_metal_campaign.control import require_running

    require_running(output)
    with _lock(output) as output, _worker_lock():
        _reconcile_interrupted(output, allow_intent_id=launch_intent_id)
        session = _current_session(output)
        if any(session.get(key) != value for key, value in _identity().items()):
            raise RuntimeError("The open session belongs to another host or boot; register the actual new allocation first.")
        attempts = _attempts(output)
        fingerprint = _fingerprint(run)
        intent = next((row for row in read_json(output / "launch_intents.json", [])
                       if row["intent_id"] == launch_intent_id), None)
        if launch_intent_id is not None and (intent is None or intent["run_id"] != run["id"]
                or intent["run_fingerprint"] != fingerprint or intent["session_id"] != session["session_id"]):
            raise ValueError("Persisted launch intent does not match this run and allocation.")
        if block_id is not None:
            comparison = comparison_status(output, block_id)
            if (run["id"] not in comparison["run_ids"] or comparison["stage"] != run["stage"]
                    or comparison.get("released_reason")
                    or comparison["run_fingerprints"].get(run["id"], fingerprint) != fingerprint):
                raise ValueError("Run identity does not match its frozen logical comparison.")
        previous = [row for row in attempts if row["run_id"] == run["id"]]
        if any(row["run_fingerprint"] != fingerprint for row in previous):
            raise ValueError("A run identity cannot be reused with different inputs.")
        completed = next((row for row in previous if row["status"] == "completed"), None)
        if completed:
            if artifact_inventory(completed["run_dir"]) != completed["artifacts"]:
                raise ValueError("Completed run artifacts changed.")
            verify_result(run, Path(completed["run_dir"]))
            return completed
        if len(previous) >= 2:
            raise RuntimeError("Only one linked retry is permitted for a configuration.")
        _require_persistence(output, attempts)
        forecast = block_forecast_seconds if block_forecast_seconds is not None else run.get("forecast_seconds")
        if forecast is None:
            raise ValueError("A measured full-block forecast is required before launch.")
        decision = admission(output, run["stage"], forecast, confirmation_reserve_seconds, block_id=block_id)
        if training_deadline_epoch is not None:
            guard_remaining = _number(training_deadline_epoch, "training_deadline_epoch") - _now()
            decision["remaining"] = min(decision["remaining"], max(0.0, guard_remaining))
            decision["deadline"] = min(decision["deadline"], training_deadline_epoch)
            if decision["required_seconds"] > guard_remaining:
                decision.update(allowed=False, reason="The complete unit cannot fit the verified host training deadline.")
        if not decision["allowed"]:
            raise RuntimeError(decision["reason"])
        directory, command = _attempt_layout(output, run, previous)
        if intent and (intent["run_dir"] != str(directory) or intent["command"] != command
                       or intent["retry_of"] != (previous[-1]["attempt_id"] if previous else None)):
            raise ValueError("Attempt layout differs from its persisted launch intent.")
        if directory.exists():
            raise FileExistsError(f"Refusing to overwrite untracked attempt output: {directory}")
        directory.mkdir(parents=True)
        attempt_id = intent["attempt_id"] if intent else _next_attempt_id(output)
        attempt = {"attempt_id": attempt_id, "run_id": run["id"], "run_fingerprint": fingerprint,
                   "stage": run["stage"], "block": run["block"], "family": run.get("family"),
                   "arm": run.get("arm"), "epochs": run["epochs"], "session_id": session["session_id"],
                   "run_dir": str(directory), "command": command, "status": "running",
                   "started_epoch": _now(), "admission": decision,
                   "logical_comparison_id": block_id,
                   "launch_intent_id": launch_intent_id,
                   "retry_of": previous[-1]["attempt_id"] if previous else None}
        attempts.append(attempt)
        atomic_json(output / "attempts.json", attempts)
        controller = _process_info(os.getpid())
        active = {"attempt_id": attempt_id, "session_id": session["session_id"],
                  "token": uuid.uuid4().hex, "controller_pid": os.getpid(),
                  "controller_start_token": controller["start_token"], **_identity()}
        atomic_json(output / "active_process.json", active)
        process = None
        caught = None
        previous_handler = signal.getsignal(signal.SIGTERM)

        def interrupted(_signum, _frame):
            raise InterruptedError("Campaign controller was terminated.")

        try:
            signal.signal(signal.SIGTERM, interrupted)
            with (directory / "execution.log").open("x") as log:
                if launch_guard is not None:
                    launch_guard()
                require_running(output)
                process = subprocess.Popen(command, cwd=Path(root), env={**os.environ, **run.get("env", {}),
                                            "PYTHONUNBUFFERED": "1"}, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                info = _process_info(process.pid)
                process._campaign_start_token = info["start_token"] if info else None
                active.update(pid=process.pid, pgid=process.pid, process_start_token=info["start_token"] if info else None)
                atomic_json(output / "active_process.json", active)
                while process.poll() is None:
                    shutdown_reserve = min(TERM_GRACE_SECONDS + KILL_GRACE_SECONDS, decision["required_seconds"] * 0.2)
                    if _now() >= decision["deadline"] - shutdown_reserve:
                        attempt["status"] = "deadline_stopped"
                        _stop_group(process, decision["deadline"])
                        break
                    time.sleep(POLL_SECONDS)
                if _group_alive(process.pid):
                    _stop_group(process)
                attempt["returncode"] = process.wait()
            if attempt["status"] == "running":
                if attempt["returncode"] != 0:
                    attempt["status"] = "failed"
                    attempt["error"] = f"Training command exited with {attempt['returncode']}."
                else:
                    attempt["result"] = verify_result(run, directory)
                    attempt["artifacts"] = artifact_inventory(directory)
                    attempt["status"] = "completed"
        except BaseException as exc:
            caught = exc
            attempt["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else "failed"
            attempt["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                if process is not None:
                    _stop_group(process)
            finally:
                signal.signal(signal.SIGTERM, previous_handler)
                attempt["ended_epoch"] = _now()
                attempt["elapsed_seconds"] = attempt["ended_epoch"] - attempt["started_epoch"]
                attempt["charged_stage"] = attempt["stage"] if attempt["status"] == "completed" else "operations"
                # Preserve logs and partial outputs on failures as well as successes.
                # The summary deliberately excludes its own inventory to avoid recursion.
                atomic_json(directory / "attempt_outcome.json", {key: value for key, value in attempt.items()
                            if key not in {"artifacts", "admission"}})
                attempt["artifacts"] = artifact_inventory(directory)
                atomic_json(output / "attempts.json", attempts)
                if process is None or not _group_alive(process.pid):
                    (output / "active_process.json").unlink(missing_ok=True)
        if caught is not None:
            raise caught
        return attempt
