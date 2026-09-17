"""Idempotent worker operations for the maintained host queue.

No operation allocates a VM. Launches delegate to the existing scientific
workflow with its original admission, retry, persistence and receipt gates.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from serial_metal_campaign import control, evidence, profile, runtime, workflow

base = profile.base


def _sha(path):
    return base.digest(Path(path))


def worker_python(manifest):
    """Use the single exported interpreter, independent of the Colab kernel."""
    runs = manifest.get("runs", [])
    commands = [row.get("command") for row in runs]
    commands += [row.get("command") for row in manifest.get("templates", {}).values()]
    base.require(runs and all(isinstance(cmd, list) and cmd for cmd in commands),
                 "Frozen campaign commands must declare their worker interpreter")
    interpreters = {cmd[0] for cmd in commands}
    base.require(len(interpreters) == 1, "Frozen campaign commands use conflicting worker interpreters")
    interpreter = next(iter(interpreters))
    base.require(isinstance(interpreter, str) and Path(interpreter).is_absolute()
                 and ".." not in Path(interpreter).parts, "Frozen worker interpreter must be an absolute path")
    return interpreter


def _archive(root, files, target):
    root, target = Path(root), Path(target)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        with tarfile.open(temporary, "w:gz") as stream:
            for name in sorted(files):
                stream.add(root / name, arcname=name, recursive=False)
        os.replace(temporary, target)
    return {"archive": str(target), "archive_sha256": _sha(target)}


def verify_archive(path, inventory, destination):
    """Reject extra members, traversal, links, duplicate members and corrupt bytes."""
    destination = Path(destination)
    with tarfile.open(path, "r:gz") as stream:
        members = stream.getmembers()
        if len(members) != len(inventory) or {m.name for m in members} != set(inventory):
            raise ValueError("Archive inventory differs from the frozen receipt.")
        for member in members:
            relative = Path(member.name)
            if not member.isfile() or relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Archive contains an unsafe member.")
            content = stream.extractfile(member).read()
            expected = inventory[member.name]
            digest = expected["sha256"] if isinstance(expected, dict) else expected
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError("Archive member checksum differs: " + member.name)
            if isinstance(expected, dict) and len(content) != expected["size"]:
                raise ValueError("Archive member length differs: " + member.name)
        for member in members:
            target = destination / member.name
            if target.is_symlink() or not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError("Independent readback path escapes its destination.")
            if target.exists():
                expected = inventory[member.name]
                digest = expected["sha256"] if isinstance(expected, dict) else expected
                if not target.is_file() or _sha(target) != digest:
                    raise ValueError("Existing independent readback differs: " + member.name)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as handle:
                    handle.write(stream.extractfile(member).read())
                    handle.flush()
                    os.fsync(handle.fileno())


def verify_setup(output, manifest_sha256, host_receipt):
    output = Path(output)
    control.require_running(output)
    base.require(_sha(output / "campaign_manifest.json") == manifest_sha256, "Host/worker manifest differs")
    manifest = profile.verify_manifest(output)
    base.require(Path(sys.executable).absolute() == Path(worker_python(manifest)).absolute(),
                 "Worker operation is not running under the frozen campaign interpreter")
    evidence.verify_preparation(output)
    workflow.verify_host_guard(output, host_receipt)
    runtime.verify_persistence(output, full=True)
    return {"status": "verified_existing_setup", "manifest_sha256": manifest_sha256,
            "source_rebuilt": False, "features_regenerated": False}


def observe(output):
    output = Path(output)
    attempts = runtime._attempts(output)
    pending = [row for row in attempts if row["status"] != "running"
               and not (output / "persistence" / (row["attempt_id"] + ".json")).is_file()]
    return {"phase": runtime.read_json(output / "queue.json", {})["phase"],
            "active": runtime.read_json(output / "active_process.json"),
            "running": [row["attempt_id"] for row in attempts if row["status"] == "running"],
            "pending_persistence": [row["attempt_id"] for row in pending],
            "last": {key: attempts[-1].get(key) for key in
                     ("attempt_id", "run_id", "status", "elapsed_seconds")} if attempts else None}


def package_attempt(output, attempt_id):
    output = Path(output)
    with workflow.controller_lock(output):
        attempt = next(row for row in runtime._attempts(output) if row["attempt_id"] == attempt_id)
        base.require(attempt["status"] in runtime.TERMINAL_STATES, "Attempt has not terminated")
        base.require(runtime.artifact_inventory(attempt["run_dir"]) == attempt["artifacts"], "Attempt artifacts changed")
        archive = output / "queue_transport" / (attempt_id + ".tar.gz")
        return {"attempt": attempt, **_archive(attempt["run_dir"], attempt["artifacts"], archive)}


def acknowledge_attempt(output, attempt_id, archive, archive_sha256, receipt):
    base.require(_sha(archive) == archive_sha256, "Independent attempt archive differs")
    with workflow.controller_lock(output):
        verify_archive(archive, receipt["artifacts"], receipt["readback_root"])
        return runtime.record_persistence(output, attempt_id, receipt)


def package_state(output, *, prepare_next=False, host_receipt=None, persistence_seconds=None):
    output = Path(output)
    next_info = None
    rollover = False
    with workflow.controller_lock(output):
        runtime.verify_persistence(output, full=False)
        if prepare_next:
            persistence_seconds = runtime._number(persistence_seconds, "measured per-fit persistence allowance")
            base.require(persistence_seconds > 0, "A positive measured persistence allowance is required")
            control.require_running(output)
            guard = workflow.verify_host_guard(output, host_receipt)
            run = workflow.next_run(output)
            transitions = 0
            while run is None and runtime.read_json(output / "queue.json")["phase"] != "complete":
                workflow.advance(output)
                transitions += 1
                base.require(transitions < 32, "Queue transition limit reached")
                run = workflow.next_run(output)
            if run is not None:
                if run["stage"] != "operations":
                    workflow.admit(output)
                if run["stage"] == "confirmation" and run["block"] == "ring" and not workflow.ring_audit_directory(output, run["fold_index"]):
                    run, _ = workflow.prepare_ring_audit(output, run["fold_index"])
                forecast = 600.0 if run["stage"] == "operations" else profile.forecast(run, workflow._timings(output))
                rollover = time.time() + runtime.ADMISSION_FACTOR * forecast + persistence_seconds > guard["training_stop_epoch"]
                if not rollover:
                    intent = runtime.prepare_launch_intent(output, run)
                    base.require(intent is not None, "Selected next unit is already completed")
                    next_info = {"intent_id": intent["intent_id"], "attempt_id": intent["attempt_id"],
                                 "run_id": run["id"], "kind": run.get("kind"), "fold_index": run.get("fold_index")}
        snapshot = workflow.export_state(output)
        archive = output / "queue_transport" / ("state_" + snapshot["state_sha256"] + ".tar.gz")
        return {**snapshot, **_archive(snapshot["source_root"], snapshot["files"], archive), "next": next_info,
                "phase": runtime.read_json(output / "queue.json")["phase"], "rollover_required": rollover}


def acknowledge_state(output, archive, archive_sha256, receipt):
    base.require(_sha(archive) == archive_sha256, "Independent state archive differs")
    with workflow.controller_lock(output):
        snapshot = workflow.export_state(output)
        base.require(snapshot["state_sha256"] == receipt["state_sha256"], "State changed during persistence")
        verify_archive(archive, snapshot["files"], receipt["readback_root"])
        return workflow.verify_state_transfer(output, receipt)


def _existing_launch(output, next_info):
    """Read atomic receipts without contending with a live fit's lifetime lock."""
    matches = [row for row in runtime._attempts(output) if row.get("launch_intent_id") == next_info["intent_id"]]
    base.require(len(matches) <= 1, "Multiple attempts claim the same persisted launch intent")
    if matches:
        existing = matches[0]
        base.require(all(existing.get(key) == next_info.get(key) for key in ("attempt_id", "run_id")),
                     "Replayed request differs from the registered attempt")
        return {"status": "reconciled_existing_attempt", "attempt_id": existing["attempt_id"]}
    previous = runtime.read_json(output / "queue_transport" / (next_info["intent_id"] + "_dispatch.json"))
    if previous:
        base.require(all(previous.get(key) == next_info.get(key) for key in ("intent_id", "attempt_id", "run_id")),
                     "Ambiguous prior launch or mismatched dispatch identity; reconcile before any retry")
        if all(previous.get(key) == value for key, value in runtime._identity().items()):
            for pid_key, token_key in (("pid", "process_start_token"), ("controller_pid", "controller_start_token")):
                if pid_key == "controller_pid" and previous.get("status") != "dispatching":
                    continue
                process = runtime._process_info(previous[pid_key]) if previous.get(pid_key) else None
                if process and process["state"] != "Z" and process["start_token"] == previous.get(token_key):
                    return {"status": "reconciled_existing_dispatch", "attempt_id": next_info["attempt_id"]}
        raise RuntimeError("Ambiguous prior launch; reconcile dispatch and provider state before any retry.")
    return None


def launch(output, next_info, host_receipt):
    """Submit each persisted intent at most once, including after a lost reply."""
    output = Path(output)
    ident = next_info.get("intent_id")
    base.require(isinstance(ident, str) and ident not in {"", ".", ".."} and Path(ident).name == ident,
                 "A valid persisted intent identity is required")
    workflow.verify_host_guard(output, host_receipt)
    existing = _existing_launch(output, next_info)
    if existing:
        return existing
    with workflow.controller_lock(output):
        workflow.verify_host_guard(output, host_receipt)
        existing = _existing_launch(output, next_info)
        if existing:
            return existing
        base.require(not runtime.read_json(output / "active_process.json"), "Another attempt is active")
        intent = next((row for row in runtime._pending_intents(output) if row["intent_id"] == next_info["intent_id"]), None)
        base.require(intent and intent["attempt_id"] == next_info["attempt_id"] and intent["run_id"] == next_info["run_id"],
                     "Launch request differs from the persisted intent")
        dispatch = output / "queue_transport" / (next_info["intent_id"] + "_dispatch.json")
        workflow.require_persistent_state(output)
        manifest = runtime.read_json(output / "campaign_manifest.json")
        command = [worker_python(manifest), str(Path(manifest["root"]) / "src/run_metal_single_gpu_campaign.py")]
        command += ["ring-audit", "--fold-index", str(next_info["fold_index"])] if next_info.get("kind") == "ring_audit" else ["execute"]
        command += ["--output-dir", str(output), "--host-receipt-json", str(host_receipt)]
        owner = runtime._process_info(os.getpid())
        record = {"status": "dispatching", "intent_id": intent["intent_id"], "attempt_id": intent["attempt_id"],
                  "run_id": intent["run_id"], "command": command,
                  "controller_pid": os.getpid(), "controller_start_token": owner["start_token"], **runtime._identity()}
        runtime.atomic_json(dispatch, record)
    # The CLI child takes this same nonblocking controller lock. Reserve the
    # immutable dispatch under the lock, then release it before spawning.
    workflow.verify_host_guard(output, host_receipt)
    control.require_running(output)
    with dispatch.with_suffix(".log").open("x") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    info = runtime._process_info(process.pid)
    runtime.atomic_json(dispatch, {**record, "status": "submitted", "pid": process.pid,
                                  "process_start_token": info["start_token"] if info else None})
    return {"status": "submitted", "attempt_id": intent["attempt_id"]}
