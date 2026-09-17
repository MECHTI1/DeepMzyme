"""Durable user pause and exclusive host ownership; no provider operations."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

from serial_metal_campaign import runtime


def _legacy_pauses(output):
    records = {}
    for path in sorted(Path(output).glob("USER_REQUESTED_PAUSE*.json")):
        value = runtime.read_json(path)
        if not isinstance(value, dict) or not value:
            raise ValueError("Malformed existing user pause: " + str(path))
        records[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return records


def _events(output):
    events = runtime.read_json(Path(output) / "control_events.json", [])
    if not isinstance(events, list):
        raise ValueError("Campaign control events must be a list.")
    seen = set()
    for sequence, event in enumerate(events, 1):
        if (not isinstance(event, dict) or event.get("sequence") != sequence
                or event.get("action") not in {"pause", "resume"}
                or event.get("authorized_by") != "user"
                or not event.get("request_id") or event["request_id"] in seen
                or not str(event.get("reason", "")).strip()
                or not isinstance(event.get("acknowledged_legacy_pauses"), dict)):
            raise ValueError("Invalid campaign control event or request identity.")
        if event["action"] == "resume" and not str(event.get("authorization", "")).strip():
            raise ValueError("Resume requires an explicit user authorization record.")
        runtime._number(event.get("epoch"), "control event epoch")
        seen.add(event["request_id"])
    return events


def status(output):
    events, legacy = _events(output), _legacy_pauses(output)
    latest = events[-1] if events else None
    resumed = latest is not None and latest["action"] == "resume"
    unseen = legacy != (latest or {}).get("acknowledged_legacy_pauses", {})
    paused = bool(legacy and (not resumed or unseen)) or bool(latest and not resumed)
    payload = {"events": events, "legacy_pauses": legacy}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"paused": paused, "sequence": len(events), "latest": latest,
            "legacy_pauses": legacy, "control_sha256": digest,
            "has_explicit_control": bool(events or legacy)}


def require_running(output):
    value = status(output)
    if value["paused"]:
        raise RuntimeError("Campaign is user-paused; record an explicit user resume authorization before new work.")
    return value


def record(output, action, *, request_id, reason, authorization=""):
    """Append an idempotent user decision without modifying old pause receipts."""
    if action not in {"pause", "resume"} or not str(request_id).strip() or not str(reason).strip():
        raise ValueError("Provide pause/resume, a unique request ID and a reason.")
    if action == "resume" and not str(authorization).strip():
        raise ValueError("Resume requires an explicit user authorization record.")
    output = Path(output)
    # A training attempt holds the runtime lock for its lifetime. User STOP
    # must remain writable immediately while that attempt is still active.
    with runtime._lock(output / "control_lock"):
        events = _events(output)
        existing = next((row for row in events if row["request_id"] == request_id), None)
        if existing:
            if any(existing[key] != value for key, value in
                   (("action", action), ("reason", reason), ("authorization", authorization))):
                raise ValueError("Control request ID was already used with different contents.")
            return status(output)
        events.append({"sequence": len(events) + 1, "action": action, "request_id": request_id,
                       "reason": reason, "authorization": authorization, "authorized_by": "user",
                       "epoch": time.time(), "acknowledged_legacy_pauses": _legacy_pauses(output)})
        runtime.atomic_json(output / "control_events.json", events)
        return status(output)


def live_controller(output):
    """Prove the active supervisor's host, boot, PID and process-start identity."""
    lease = runtime.read_json(Path(output) / "host_control/controller_lease.json", {})
    if (lease.get("status") != "active"
            or any(lease.get(key) != value for key, value in runtime._identity().items())):
        return False
    process = runtime._process_info(lease.get("pid")) if lease.get("pid") else None
    return bool(process and process["state"] != "Z"
                and process["start_token"] == lease.get("process_start_token"))


@contextmanager
def controller_lease(output):
    """Hold a campaign-wide host lease, recovering only a provably dead owner."""
    root = Path(output) / "host_control"
    path = root / "controller_lease.json"
    with runtime._lock(root / "controller_lease_lock"):
        previous = runtime.read_json(path)
        identity = runtime._identity()
        if previous and previous.get("status") == "active":
            if previous.get("hostname") != identity["hostname"]:
                raise RuntimeError("Previous host-controller owner is on another host; its death is unresolved.")
            if previous.get("boot_id") == identity["boot_id"]:
                process = runtime._process_info(previous["pid"])
                if (process and process["state"] != "Z"
                        and process["start_token"] == previous["process_start_token"]):
                    raise RuntimeError("Previous host-controller process is still live.")
            elif not previous.get("boot_id"):
                raise RuntimeError("Previous host-controller boot identity is missing.")
        elif previous and previous.get("status") != "released":
            raise ValueError("Malformed host-controller lease.")
        process = runtime._process_info(os.getpid())
        if process is None:
            raise RuntimeError("Cannot identify the current host-controller process.")
        lease = {"token": uuid.uuid4().hex, "status": "active", "pid": os.getpid(),
                 "process_start_token": process["start_token"], "acquired_epoch": time.time(),
                 "previous_token": previous.get("token") if previous else None, **identity}
        runtime.atomic_json(path, lease)
        try:
            yield lease
        finally:
            current = runtime.read_json(path)
            if current != lease:
                raise RuntimeError("Host-controller lease changed while held.")
            runtime.atomic_json(path, {**lease, "status": "released", "released_epoch": time.time()})
