"""Explicit owned Colab allocation control for the serial metal campaign.

Importing or preparing this module never allocates a GPU. Only ``allocate`` does.
The detached host watchdog requests teardown five minutes before the earlier of
the four-hour allocation cap and remaining authorized cumulative cap. Provider
absence, rather than a successful CLI exit, closes an allocation interval.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serial_metal_campaign import runtime

GPUS = {"T4", "L4", "G4", "H100", "A100"}
STOP_MARGIN_SECONDS = 300
HEARTBEAT_SECONDS = 5
HEARTBEAT_MAX_AGE = 20
STARTUP_SECONDS = 20
BACKEND_TIMEOUT = 60


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _root(output):
    return Path(output).resolve() / "host_control"


def _session(output, session_id):
    if not re.fullmatch(r"deepmzyme-[a-z0-9][a-z0-9-]{1,100}", session_id):
        raise ValueError("Use a fresh named session matching deepmzyme-[a-z0-9-]+.")
    return _root(output) / "sessions" / session_id


def _ledger(output):
    rows = runtime.read_json(_root(output) / "allocations.json", [])
    seen = set()
    prior_end = None
    for row in rows:
        runtime._number(row["started_epoch"], "host allocation start")
        if row["session_id"] in seen or (prior_end is not None and row["started_epoch"] < prior_end):
            raise ValueError("Host allocation history contains duplicate or overlapping intervals.")
        if seen and prior_end is None:
            raise ValueError("An unresolved allocation cannot precede another allocation.")
        seen.add(row["session_id"])
        prior_end = row.get("stopped_epoch")
        if prior_end is not None:
            runtime._number(prior_end, "host allocation stop")
        if prior_end is not None and prior_end < row["started_epoch"]:
            raise ValueError("Host allocation stop precedes its request.")
    return rows


def host_budget(output, now=None):
    now = runtime._number(time.time() if now is None else now, "now")
    rows = _ledger(output)
    spent = sum((row.get("stopped_epoch") if row.get("stopped_epoch") is not None else now)
                - row["started_epoch"] for row in rows)
    if spent < 0:
        raise ValueError("Clock precedes allocation history.")
    limits = runtime.budget_limits(output)
    return {"allocated_seconds": spent, "total_remaining_seconds": max(0, limits["total_seconds"] - spent),
            "limit_seconds": limits,
            "open_session_ids": [row["session_id"] for row in rows if row.get("stopped_epoch") is None]}


class CLIBackend:
    """Bounded CLI calls and token-free structured inspection; no interactive auth."""

    def __init__(self, cli_python=None, history_dir=None):
        if cli_python is None:
            executable = shutil.which("colab")
            if not executable:
                raise RuntimeError("Install the Colab CLI before an explicit allocation action.")
            first = Path(executable).read_text().splitlines()[0]
            if not first.startswith("#!/") or " " in first[2:]:
                raise RuntimeError("Pass --cli-python for the installed Colab CLI environment.")
            cli_python = first[2:]
        # Keep a virtual-environment interpreter's path. Resolving its symlink
        # to the base Python drops the venv's site-packages, including colab_cli.
        self.cli_python = str(Path(cli_python).expanduser().absolute())
        self.history_dir = Path(history_dir or "~/.config/colab-cli/history").expanduser()

    def _invoke(self, mode, arguments=(), timeout=BACKEND_TIMEOUT):
        # Run in the CLI's own environment without copying OAuth tokens into this
        # process. Deny its only interactive flow before importing its commands.
        source = """
import json, sys
import colab_cli.auth as auth
def deny_auth(*args, **kwargs):
    raise RuntimeError('Interactive authentication is disabled for campaign control')
auth._run_remote_flow = deny_auth
mode = sys.argv[1]
if mode == 'cli':
    from colab_cli.cli import app
    sys.argv = ['colab'] + sys.argv[2:]
    app()
else:
    from colab_cli.common import state
    if mode == 'assignments':
        value = [{'endpoint': a.endpoint, 'accelerator': a.accelerator.value, 'variant': a.variant.name}
                 for a in state.client.list_assignments()]
    elif mode == 'mapping':
        entry = state.store.get(sys.argv[2])
        value = None if entry is None else {'endpoint': entry.endpoint, 'accelerator': entry.accelerator, 'variant': entry.variant}
    elif mode == 'stop_owned':
        from colab_cli.state import SessionState
        from colab_cli.commands.session import stop
        import contextlib, io
        name, endpoint, accelerator, variant = sys.argv[2:]
        store = state.store
        # Hold the CLI state lock through its named stop so another controller
        # cannot retarget the same name between verification and unassignment.
        with store._lock_exclusive() as stream:
            entries = store._load_raw(stream)
            entry = entries.get(name)
            if entry is not None and entry.endpoint != endpoint:
                raise RuntimeError('Refusing to stop an unrelated endpoint mapping')
            if entry is None:
                entry = SessionState(name=name, endpoint=endpoint, token='', url='', accelerator=accelerator, variant=variant)
                entries[name] = entry
                store._save_raw(stream, entries)
            class OwnedLockedStore:
                def get(self, requested):
                    if requested != name:
                        raise RuntimeError('Named stop escaped its owned session')
                    return entry
                def remove(self, requested):
                    if requested != name:
                        raise RuntimeError('Named removal escaped its owned session')
                    entries.pop(name, None)
                    store._save_raw(stream, entries)
            state._store = OwnedLockedStore()
            with contextlib.redirect_stdout(io.StringIO()):
                stop(session=name)
        value = {'named_stop_requested': True}
    else:
        raise ValueError('Unknown backend action')
    print(json.dumps(value))
"""
        result = subprocess.run([self.cli_python, "-c", source, mode, *map(str, arguments)],
                                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            # CLI errors can contain signed URLs or credentials; store only type/code.
            raise RuntimeError(f"Colab {mode} failed with exit {result.returncode}; owned teardown remains required.")
        return result.stdout if mode == "cli" else json.loads(result.stdout)

    def history(self, name):
        path = self.history_dir / f"{name}.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []

    def mapping(self, name):
        return self._invoke("mapping", [name])

    def assignments(self):
        return self._invoke("assignments")

    def create(self, name, gpu):
        return self._invoke("cli", ["new", "--gpu", gpu, "-s", name], timeout=180)

    def status(self, name):
        self._invoke("cli", ["status", "-s", name])

    def stop(self, name, owned):
        self._invoke("stop_owned", [name, owned["endpoint"], owned["accelerator"], owned["variant"]])


def prepare(output, session_id, gpu, *, manifest_path=None, cli_python=None):
    """Freeze one fresh session identity and bind it to the campaign manifest."""
    if gpu not in GPUS:
        raise ValueError(f"Unsupported accelerator; use an exact name from {sorted(GPUS)}.")
    output = Path(output).resolve()
    directory = _session(output, session_id)
    manifest = Path(manifest_path or output / "campaign_manifest.json").resolve()
    if not manifest.is_file():
        raise ValueError("A frozen campaign manifest is required before host preparation.")
    with runtime._lock(_root(output)):
        identity_path = _root(output) / "campaign_identity.json"
        identity = {"manifest": str(manifest), "manifest_sha256": _sha(manifest)}
        existing_identity = runtime.read_json(identity_path)
        if existing_identity and existing_identity != identity:
            raise ValueError("Host allocation history cannot be rebound to a different campaign manifest.")
        if not (_root(output) / "allocations.json").exists():
            prior = runtime.read_json(output / "sessions.json", [])
            imported = []
            for row in prior:
                proof = row.get("stop_evidence", {})
                if (row.get("stopped_epoch") is None or proof.get("provider_verified_stopped") is not True
                        or proof.get("session_id") != row["session_id"] or proof.get("stopped_epoch") != row["stopped_epoch"]):
                    raise RuntimeError("Prior worker allocations need verified stop receipts before host history import.")
                imported.append({**row, "status": "stopped_verified", "imported_worker_history": True})
            runtime.atomic_json(_root(output) / "allocations.json", imported)
        known = {row["session_id"]: row for row in _ledger(output)}
        for row in runtime.read_json(output / "sessions.json", []):
            recorded = known.get(row["session_id"])
            if recorded is None or recorded["started_epoch"] != row["started_epoch"]:
                raise RuntimeError("Worker allocation history differs from the cumulative host ledger.")
            if row.get("stopped_epoch") is not None and recorded.get("stopped_epoch") != row["stopped_epoch"]:
                raise RuntimeError("Worker and host allocation stop timestamps differ.")
        if host_budget(output)["open_session_ids"]:
            raise RuntimeError("Verify the preceding owned allocation stopped before preparing another.")
        if runtime.read_json(directory / "config.json") or any(row["session_id"] == session_id for row in _ledger(output)):
            raise FileExistsError("Prepared session identities are immutable; choose a fresh name.")
        limits = runtime.budget_limits(output)
        config = {"session_id": session_id, "gpu": gpu, "output": str(output), "manifest": str(manifest),
                  "manifest_sha256": _sha(manifest), "cli_python": cli_python,
                  "budget_limits": limits, "total_cap_seconds": limits["total_seconds"],
                  "session_cap_seconds": limits["session_seconds"],
                  "closeout_seconds": limits["closeout_seconds"], "stop_margin_seconds": STOP_MARGIN_SECONDS,
                  "operator_sha256": _sha(__file__), "runtime_sha256": _sha(runtime.__file__)}
        runtime.atomic_json(directory / "config.json", config)
        runtime.atomic_json(identity_path, identity)
        return {"status": "prepared_not_allocated", **config}


def _config(output, session_id):
    value = runtime.read_json(_session(output, session_id) / "config.json")
    if not value or value["session_id"] != session_id or value["output"] != str(Path(output).resolve()):
        raise ValueError("Prepare this exact campaign/session identity first.")
    return value


def _ownership(config, request, backend):
    rows = [row for row in backend.history(config["session_id"]) if row.get("event_type") == "session_created"]
    if len(rows) != 1:
        raise RuntimeError("Exactly one fresh session_created record is required to prove endpoint ownership.")
    row = rows[0]
    created = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")).timestamp()
    if not request["started_epoch"] <= created <= time.time() + 1:
        raise RuntimeError("Creation history does not belong to this allocation request.")
    if not row.get("endpoint"):
        raise RuntimeError("Creation history lacks an owned endpoint.")
    return {"session_id": config["session_id"], "endpoint": row["endpoint"], "accelerator": row.get("accelerator"),
            "variant": row.get("variant"), "creation_epoch": created,
            "creation_record_sha256": hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()}


def _request(output, session_id):
    row = next((row for row in _ledger(output) if row["session_id"] == session_id), None)
    if row is None:
        raise ValueError("This session has no allocation request.")
    return row


def watchdog_alive(output, session_id, *, now=None):
    now = time.time() if now is None else now
    directory = _session(output, session_id)
    ack = runtime.read_json(directory / "watchdog_ack.json", {})
    process = runtime.read_json(directory / "watchdog_process.json", {})
    request = _request(output, session_id)
    if (not ack or not process or ack.get("token") != request["watchdog_token"]
            or process.get("token") != ack.get("token") or process.get("pid") != ack.get("pid")
            or process.get("process_start_token") != ack.get("process_start_token")
            or any(ack.get(key) != value for key, value in runtime._identity().items())):
        return False
    info = runtime._process_info(ack["pid"])
    return bool(info and info["state"] != "Z" and info["start_token"] == ack.get("process_start_token")
                and 0 <= now - ack.get("heartbeat_epoch", 0) <= HEARTBEAT_MAX_AGE
                and ack.get("hard_deadline_epoch") == request["hard_deadline_epoch"])


def start_watchdog(output, session_id):
    directory = _session(output, session_id)
    request = _request(output, session_id)
    with (directory / "watchdog.log").open("x") as stream:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "watchdog", "--output", str(output),
                                    "--session-id", session_id, "--token", request["watchdog_token"]],
                                   stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    info = runtime._process_info(process.pid)
    runtime.atomic_json(directory / "watchdog_process.json", {"pid": process.pid, "token": request["watchdog_token"],
                        "process_start_token": info["start_token"] if info else None, **runtime._identity()})
    deadline = time.monotonic() + STARTUP_SECONDS
    while time.monotonic() < deadline:
        if watchdog_alive(output, session_id):
            return process.pid
        if process.poll() is not None:
            break
        time.sleep(0.1)
    raise RuntimeError("Watchdog startup acknowledgement failed; allocation was not requested.")


def allocate(output, session_id, *, backend=None, starter=None):
    config = _config(output, session_id)
    backend = backend or CLIBackend(config.get("cli_python"))
    directory = _session(output, session_id)
    with runtime._lock(_root(output)):
        rows = _ledger(output)
        budget = host_budget(output)
        if budget["open_session_ids"] or any(row["session_id"] == session_id for row in rows):
            raise RuntimeError("An unresolved or previously used allocation prevents another request.")
        known = {row["session_id"]: row for row in rows}
        for worker in runtime.read_json(Path(output) / "sessions.json", []):
            recorded = known.get(worker["session_id"])
            if (recorded is None or recorded["started_epoch"] != worker["started_epoch"]
                    or recorded.get("stopped_epoch") != worker.get("stopped_epoch")):
                raise RuntimeError("Reconcile worker allocation history before creating another owned session.")
        if backend.history(session_id) or backend.mapping(session_id):
            raise RuntimeError("Session name has prior CLI history or mapping; never reuse it.")
        if (_sha(config["manifest"]) != config["manifest_sha256"]
                or _sha(__file__) != config["operator_sha256"] or _sha(runtime.__file__) != config["runtime_sha256"]):
            raise ValueError("Prepared source or campaign manifest changed.")
        if runtime.budget_limits(output) != config["budget_limits"]:
            raise ValueError("Budget authorization changed after session preparation.")
        duration = min(config["session_cap_seconds"], budget["total_remaining_seconds"])
        if duration <= config["closeout_seconds"]:
            raise RuntimeError("Cumulative allocation budget cannot cover a new session and closeout.")
        started = time.time()
        request = {"session_id": session_id, "started_epoch": started, "started_monotonic": time.monotonic(),
                   "stopped_epoch": None, "hard_deadline_epoch": started + duration,
                   "training_deadline_epoch": started + duration - config["closeout_seconds"],
                   "stop_request_epoch": started + duration - STOP_MARGIN_SECONDS,
                   "prior_allocated_seconds": budget["allocated_seconds"], "status": "arming_watchdog",
                   "watchdog_token": uuid.uuid4().hex, **runtime._identity()}
        rows.append(request)
        runtime.atomic_json(_root(output) / "allocations.json", rows)
    try:
        (starter or start_watchdog)(output, session_id)
        if not watchdog_alive(output, session_id):
            raise RuntimeError("Watchdog startup did not prove liveness.")
    except BaseException:
        with runtime._lock(_root(output)):
            rows = _ledger(output)
            row = next(row for row in rows if row["session_id"] == session_id)
            row.update(status="cancelled_before_request", stopped_epoch=row["started_epoch"])
            runtime.atomic_json(_root(output) / "allocations.json", rows)
        raise
    with runtime._lock(_root(output)):
        rows = _ledger(output)
        next(row for row in rows if row["session_id"] == session_id)["status"] = "allocation_requested"
        runtime.atomic_json(_root(output) / "allocations.json", rows)
    try:
        backend.create(session_id, config["gpu"])
        owned = _ownership(config, request, backend)
        runtime.atomic_json(directory / "owned_session.json", owned)
        backend.status(session_id)
        observed = next((row for row in backend.assignments() if row["endpoint"] == owned["endpoint"]), None)
        if observed is None or observed.get("accelerator") != config["gpu"] or observed.get("variant") != "GPU":
            raise RuntimeError("Assigned hardware differs from the requested named GPU.")
        if not watchdog_alive(output, session_id) or time.time() >= request["training_deadline_epoch"]:
            raise RuntimeError("Watchdog liveness or allocation deadline prevents any work.")
        receipt = {"status": "owned_ready", **request, "hardware": observed, "ownership": owned}
        receipt["status"] = "owned_ready"
        runtime.atomic_json(directory / "allocation_receipt.json", receipt)
        return receipt
    except BaseException as exc:
        runtime.atomic_json(directory / "stop_required.json", {"reason": type(exc).__name__, "epoch": time.time()})
        try:
            stop(output, session_id, backend=backend)
        except Exception:
            pass  # Watchdog keeps reconciling lost responses; interval remains open.
        raise


def stop(output, session_id, *, backend=None):
    """Stop only a creation-history-owned endpoint and verify provider absence."""
    config = _config(output, session_id)
    directory = _session(output, session_id)
    backend = backend or CLIBackend(config.get("cli_python"))
    with runtime._lock(directory / "stop_lock"):
        existing = runtime.read_json(directory / "session_stopped.json")
        if existing:
            return existing
        request = _request(output, session_id)
        if request["status"] == "cancelled_before_request":
            return {"status": "cancelled_before_request", "session_id": session_id}
        if request.get("stopped_epoch") is not None:
            receipt = request["stop_evidence"]
            runtime.atomic_json(directory / "session_stopped.json", receipt)
            return receipt
        try:
            owned = _ownership(config, request, backend)
            previous = runtime.read_json(directory / "owned_session.json")
            if previous and previous != owned:
                raise RuntimeError("Owned endpoint differs from immutable creation history.")
            runtime.atomic_json(directory / "owned_session.json", owned)
            mapping = backend.mapping(session_id)
            if mapping and mapping["endpoint"] != owned["endpoint"]:
                raise RuntimeError("Local named mapping belongs to an unrelated endpoint.")
            stop_error = None
            try:
                backend.stop(session_id, owned)
            except Exception as exc:
                stop_error = type(exc).__name__
            assignments = backend.assignments()
            if any(row["endpoint"] == owned["endpoint"] for row in assignments):
                raise RuntimeError("Owned endpoint remains assigned after the stop request.")
            ended = time.time()
            receipt = {"status": "stopped_verified", "provider_verified_stopped": True,
                       "session_id": session_id, "endpoint": owned["endpoint"], "stopped_epoch": ended,
                       "started_epoch": request["started_epoch"], "stop_response_error": stop_error,
                       "hard_deadline_met": ended <= request["hard_deadline_epoch"],
                       "observed_endpoints": sorted(row["endpoint"] for row in assignments)}
            with runtime._lock(_root(output)):
                rows = _ledger(output)
                row = next(row for row in rows if row["session_id"] == session_id)
                receipt["cumulative_allocated_seconds"] = sum(
                    (ended if item["session_id"] == session_id else item["stopped_epoch"]) - item["started_epoch"] for item in rows)
                row.update(stopped_epoch=ended, status="stopped_verified", stop_evidence=receipt)
                runtime.atomic_json(_root(output) / "allocations.json", rows)
            runtime.atomic_json(directory / "session_stopped.json", receipt)
            return receipt
        except Exception as exc:
            runtime.atomic_json(directory / "stop_required.json", {"reason": type(exc).__name__, "epoch": time.time(),
                                "status": "stop_unverified", "session_id": session_id})
            raise


def status(output, session_id, *, backend=None):
    request = _request(output, session_id)
    if request.get("stopped_epoch") is not None:
        return {"work_allowed": False, "allocation": request, "budget": host_budget(output)}
    alive = watchdog_alive(output, session_id)
    directory = _session(output, session_id)
    allowed = alive and time.time() < request["training_deadline_epoch"] and not (directory / "stop_required.json").exists()
    if not alive:
        runtime.atomic_json(directory / "stop_required.json", {"reason": "watchdog_not_live", "epoch": time.time()})
        stop(output, session_id, backend=backend)
    return {"work_allowed": allowed, "watchdog_alive": alive, "allocation": _request(output, session_id),
            "budget": host_budget(output)}


def worker_receipt(output, session_id, *, backend=None):
    """Issue a short-lived launch gate after live watcher and provider inspection."""
    config = _config(output, session_id)
    backend = backend or CLIBackend(config.get("cli_python"))
    checked = status(output, session_id, backend=backend)
    if not checked["work_allowed"]:
        raise RuntimeError("The host watchdog does not authorize another worker fit.")
    request = checked["allocation"]
    owned = _ownership(config, request, backend)
    frozen = runtime.read_json(_session(output, session_id) / "owned_session.json")
    if frozen != owned:
        raise RuntimeError("Worker authorization requires immutable verified endpoint ownership.")
    observed = next((row for row in backend.assignments() if row["endpoint"] == owned["endpoint"]), None)
    mapping = backend.mapping(session_id)
    if (observed is None or observed.get("accelerator") != config["gpu"] or observed.get("variant") != "GPU"
            or mapping is None or mapping["endpoint"] != owned["endpoint"]):
        raise RuntimeError("Provider assignment or named mapping changed; no worker authorization issued.")
    now = time.time()
    if (not watchdog_alive(output, session_id, now=now) or now >= request["training_deadline_epoch"]
            or (_session(output, session_id) / "stop_required.json").exists()):
        raise RuntimeError("Watchdog receipt expired during provider verification.")
    if _sha(config["manifest"]) != config["manifest_sha256"]:
        raise ValueError("Frozen campaign manifest changed.")
    manifest = runtime.read_json(config["manifest"])
    return {"session_id": session_id, "allocation_started_epoch": request["started_epoch"],
            "hard_stop_epoch": request["hard_deadline_epoch"], "training_stop_epoch": request["training_deadline_epoch"],
            "endpoint": owned["endpoint"], "hardware": observed, "watchdog_verified": True, "checked_epoch": now,
            "expires_epoch": min(now + 120, request["training_deadline_epoch"]),
            "campaign_profile": manifest.get("profile", manifest.get("campaign_profile")),
            "campaign_manifest_sha256": config["manifest_sha256"], "prior_allocated_seconds": request["prior_allocated_seconds"],
            "budget_authorization_sha256": config["budget_limits"]["authorization_sha256"],
            "total_cap_seconds": config["total_cap_seconds"],
            "watchdog_host": runtime._identity()}


def watchdog(output, session_id, token, *, backend=None, once=False):
    """Independent deadline owner. Its acknowledgement is required before create."""
    directory = _session(output, session_id)
    request = _request(output, session_id)
    if token != request["watchdog_token"] or any(request[key] != value for key, value in runtime._identity().items()):
        raise RuntimeError("Watchdog token or host boot differs from the allocation request.")
    process = runtime._process_info(os.getpid())
    stop_signal = False
    def request_stop(_signum, _frame):
        nonlocal stop_signal
        stop_signal = True
    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        while True:
            request = _request(output, session_id)
            if request.get("stopped_epoch") is not None:
                return request
            now = time.time()
            runtime.atomic_json(directory / "watchdog_ack.json", {"pid": os.getpid(), "token": token,
                                "process_start_token": process["start_token"], "heartbeat_epoch": now,
                                "hard_deadline_epoch": request["hard_deadline_epoch"], **runtime._identity()})
            if stop_signal:
                runtime.atomic_json(directory / "stop_required.json", {"reason": "watchdog_signal", "epoch": now})
            monotonic_due = time.monotonic() - request["started_monotonic"] >= (
                request["stop_request_epoch"] - request["started_epoch"])
            if now >= request["stop_request_epoch"] or monotonic_due or (directory / "stop_required.json").exists():
                try:
                    return stop(output, session_id, backend=backend)
                except Exception as exc:
                    runtime.atomic_json(directory / "watchdog_stop_error.json", {"error_type": type(exc).__name__,
                                        "epoch": time.time(), "hard_deadline_exceeded": time.time() >= request["hard_deadline_epoch"]})
            if once:
                return {"status": "watchdog_armed"}
            time.sleep(HEARTBEAT_SECONDS)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "allocate", "status", "watchdog", "stop"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--gpu", choices=sorted(GPUS), default="G4")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cli-python")
    parser.add_argument("--worker-receipt", type=Path, help="Write a fresh verified worker launch receipt with status.")
    parser.add_argument("--token", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.output, args.session_id, args.gpu, manifest_path=args.manifest, cli_python=args.cli_python)
    elif args.action == "watchdog":
        result = watchdog(args.output, args.session_id, args.token)
    elif args.action == "status" and args.worker_receipt:
        result = worker_receipt(args.output, args.session_id)
        runtime.atomic_json(args.worker_receipt, result)
    else:
        result = globals()[args.action](args.output, args.session_id)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
