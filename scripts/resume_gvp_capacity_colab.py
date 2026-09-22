"""Recover the frozen, reduced GVP capacity study without changing its science.

Status is read-only. Resume preserves completed fits and prior charges, then
uses one replacement G4 allocation. The original controller and all training
modules are loaded from the checksum-verified input archive, never this checkout.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
RELATIVE_OUTPUT = Path("DeepMzyme_Data/notebook_outputs/plans/gvp_capacity_diagnostic_v1")
CONTROLLER = "scripts/run_gvp_capacity_colab.py"
TOTAL_SECONDS = 6 * 3600
MAXIMUM_FITS = 34


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_once(path, value):
    """Immutable recovery receipts may be replayed, but never replaced."""
    if path.exists():
        require(read_json(path) == value, f"Conflicting receipt: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")


@contextmanager
def ownership_lock(execution):
    with (execution / "host.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def verify_archive(output):
    execution = output / "execution"
    required = (output / "manifest.json", output / "splits.json", output / "discovery_matrix.json",
                execution / "transfer.json", execution / "input_inventory.json", execution / "inputs.tar.gz")
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, "Frozen study artifacts are missing; restore the original study before recovery: " + ", ".join(missing))
    transfer = read_json(execution / "transfer.json")
    inventory = read_json(execution / "input_inventory.json")
    manifest = read_json(output / "manifest.json")
    require(manifest["study"] == "gvp_capacity_diagnostic_v1", "Wrong study")
    require(manifest["gpu_ceiling_seconds"] == TOTAL_SECONDS, "Budget changed")
    require(not manifest["held_out_evaluation"] and not manifest["promotion"], "Unapproved scientific scope")
    require(manifest["reduced_maximum_fits"] == MAXIMUM_FITS, "Reduced matrix changed")
    require(digest(output / "manifest.json") == transfer["study_manifest_sha256"], "Study manifest changed")
    require(digest(execution / "inputs.tar.gz") == transfer["archive_sha256"], "Input archive changed")
    for part in transfer["parts"]:
        path = execution / part["name"]
        require(path.parent == execution and digest(path) == part["sha256"], "Input transfer part changed")
    with tarfile.open(execution / "inputs.tar.gz") as archive:
        members = archive.getmembers()
        require(len({m.name for m in members}) == len(members), "Duplicate archive member")
        require(all(m.isfile() and not Path(m.name).is_absolute() and ".." not in Path(m.name).parts
                    for m in members), "Unsafe input archive member")
        for name, expected in inventory.items():
            source = archive.extractfile(name)
            require(source is not None and hashlib.file_digest(source, "sha256").hexdigest() == expected,
                    f"Frozen archive mismatch: {name}")
        for name in ("manifest.json", "splits.json", "discovery_matrix.json"):
            require(digest(output / name) == inventory[str(RELATIVE_OUTPUT / name)], f"Frozen {name} changed")
        stored = archive.extractfile(str(RELATIVE_OUTPUT / "execution/input_inventory.json"))
        require(stored is not None and json.load(stored) == inventory, "Input inventory changed")
    require(all(inventory[name] == value for name, value in manifest["source_files"].items()), "Source seal mismatch")
    return manifest, inventory


def require_admitted(controller, ledger, results):
    gate = controller.cost_gate(controller.charge(ledger["allocations"], time.time()),
                                [r["total_seconds"] for r in results], len(results), MAXIMUM_FITS)
    require(gate["admitted"], "Protected comparison no longer fits the six-hour ceiling")
    return gate


def stage_source(output, inventory):
    recovery = output / "execution/recovery"
    destination = recovery / "frozen_source"
    if not destination.exists():
        temporary = recovery / "frozen_source.extracting"
        require(not temporary.exists(), "Partial source extraction requires inspection")
        temporary.mkdir(parents=True)
        with tarfile.open(output / "execution/inputs.tar.gz") as archive:
            archive.extractall(temporary, filter="data")
        for name, expected in inventory.items():
            require(digest(temporary / name) == expected, f"Extracted source mismatch: {name}")
        temporary.rename(destination)
    for name, expected in inventory.items():
        require(digest(destination / name) == expected, f"Staged input changed: {name}")
    return destination


def load_controller(source, output):
    require("gvp_capacity_study" not in sys.modules, "Training module was imported before source isolation")
    sys.path.insert(0, str(source / "src"))
    spec = importlib.util.spec_from_file_location("capacity_frozen_controller", source / CONTROLLER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.checked_manifest(source, output)
    return module


def confirm_absence(listing, session, endpoint=None):
    require("No active sessions found on server." in listing or
            re.search(r"^\[[^\]]+\].*Hardware:", listing, re.MULTILINE), "Unrecognized provider listing")
    require(f"[{session}]" not in listing and (not endpoint or endpoint not in listing),
            "Owned session is still present or ambiguous")


def reconciled_ledger(ledger, receipt):
    """Return an idempotent accounting preview; absence time is an upper bound."""
    result = copy.deepcopy(ledger)
    require(result["ceiling_seconds"] == TOTAL_SECONDS, "Allocation budget changed")
    allocations = result["allocations"]
    require(len({a["session"] for a in allocations}) == len(allocations), "Duplicate allocation identity")
    require(receipt["returncode"] == 0 and receipt["command"] == ["colab", "sessions"], "Invalid absence receipt")
    confirm_absence(receipt["stdout"] + receipt.get("stderr", ""), receipt["session"], receipt.get("endpoint"))
    matching = [a for a in allocations if a["session"] == receipt["session"]]
    require(len(matching) == 1, "Absence receipt has no unique owned allocation")
    allocation = matching[0]
    require(receipt["observed_at_unix"] >= allocation["started_at_unix"], "Absence predates allocation")
    if "stopped_at_unix" not in allocation:
        allocation.update(stopped_at_unix=receipt["observed_at_unix"],
                          provider_listing_verified_absent=True,
                          termination_kind="provider_loss",
                          accounting_end_basis="conservative upper bound at observed absence")
        if receipt.get("endpoint"):
            allocation["endpoint"] = receipt["endpoint"]
    else:
        require(allocation.get("provider_listing_verified_absent"), "Closed allocation lacks provider verification")
    return result


def absence_receipt(output, ledger, controller):
    """Capture live absence only when no earlier durable observation exists."""
    recovery = output / "execution/recovery"
    path = recovery / "initial_provider_absence.json"
    if path.exists():
        return read_json(path)
    first = ledger["allocations"][0]
    require(len(ledger["allocations"]) == 1, "Missing original absence evidence after replacement allocation")
    status_path = output / "execution/session_1/provider_status.txt"
    status = status_path.read_text()
    match = re.search(r"\[" + re.escape(first["session"]) + r"\] (\S+) \| Hardware: G4 \|", status)
    require(match is not None, "Original allocation hardware/endpoint evidence missing")
    recovery.mkdir(parents=True, exist_ok=True)
    provider = controller.Provider(first["session"], recovery / "provider_preflight.log")
    listing = provider.call(["colab", "sessions"], timeout=60)
    observed = time.time()
    confirm_absence(listing, first["session"], match.group(1))
    receipt = {"session": first["session"], "endpoint": match.group(1), "observed_at_unix": observed,
               "command": ["colab", "sessions"], "returncode": 0, "stdout": listing, "stderr": "",
               "observation": "provider confirms absence; exact termination time unknown",
               "accounting_end_basis": "conservative upper bound at observed absence"}
    write_once(path, receipt)
    return receipt


def phase_matrices(output):
    matrices = {}
    for phase in ("discovery", "confirmation"):
        path = output / f"{phase}_matrix.json"
        if path.exists():
            rows = read_json(path)
            require(len({r["id"] for r in rows}) == len(rows), "Duplicate matrix cell")
            matrices.update({r["id"]: r for r in rows})
    return matrices


def verify_run(output, path, planned):
    row = read_json(path / "capacity_result.json")
    require(row["spec"] == planned and planned["id"] == path.name, "Result scientific identity changed")
    require(not row["held_out_evaluation"] and row["manifest_sha256"] == digest(output / "manifest.json"),
            "Result study identity changed")
    require(bool(row["files"]), "Result has no bound artifacts")
    for name, expected in row["files"].items():
        require(Path(name).name == name and digest(path / name) == expected, f"Corrupt result: {path.name}/{name}")
    retrieval = output / "execution/retrieval" / path.name
    receipt = read_json(retrieval / "verified.json")
    require(receipt["id"] == path.name, "Retrieval identity changed")
    archive_path = retrieval / "run.tar.gz"
    require(digest(archive_path) == receipt["archive_sha256"], "Result archive changed")
    with tarfile.open(archive_path) as archive:
        member = archive.extractfile(path.name + "/capacity_result.json")
        require(member is not None and member.read() == (path / "capacity_result.json").read_bytes(),
                "Result record differs from retrieved archive")
    return row


def verified_results(output):
    planned = phase_matrices(output)
    results = []
    for path in sorted((output / "runs").iterdir()):
        if not path.is_dir():
            continue
        require(path.name in planned, f"Unplanned local run: {path.name}")
        require((path / "capacity_result.json").is_file(), f"Partial local run requires recovery: {path.name}")
        results.append(verify_run(output, path, planned[path.name]))
    return results


def prior_attempts(execution, spec):
    records = []
    for path in sorted(execution.glob(f"session_*/launch_intent_{spec['id']}.json")):
        require(read_json(path) == spec, "Prior attempt scientific identity differs")
        records.append({"path": str(path.relative_to(execution)), "sha256": digest(path)})
    require(len(records) < 2, "Interrupted fit retry limit exhausted")
    return records


def retrieve_with_recovery(controller, provider, output, status):
    """Retry transfer, never training; retain partial downloads for diagnosis."""
    run = output / "runs" / status["id"]
    if run.exists():
        return verify_run(output, run, phase_matrices(output)[status["id"]])
    destination = output / "execution/retrieval" / status["id"]
    for attempt in range(3):
        if destination.exists():
            quarantine = output / "execution/recovery/partial_retrievals" / status["id"] / str(time.time_ns())
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            destination.rename(quarantine)
        try:
            controller.retrieve(provider, output, status)
            return verify_run(output, run, phase_matrices(output)[status["id"]])
        except Exception:
            if run.exists() or attempt == 2:
                raise
            atomic_json(destination / "retrieval_failure.json", {"error": traceback.format_exc()})
    raise AssertionError("Unreachable transfer state")


def stop_confirmed(controller, provider, directory, allocation, ledger, ledger_path):
    """A failed stop request is not proof of either presence or absence."""
    for attempt in range(3):
        requested = False
        try:
            text = provider.command("stop", timeout=90)
            (directory / f"provider_stop_{attempt}.txt").write_text(text)
            requested = True
        except Exception:
            (directory / f"stop_request_error_{attempt}.txt").write_text(traceback.format_exc())
        try:
            listing = provider.call(["colab", "sessions"], timeout=60)
            (directory / f"provider_sessions_after_{attempt}.txt").write_text(listing)
            confirm_absence(listing, provider.session, allocation.get("endpoint"))
            allocation.update(stopped_at_unix=time.time(), provider_listing_verified_absent=True,
                              termination_kind="requested_stop" if requested else "provider_loss",
                              accounting_end_basis="conservative upper bound at observed absence")
            controller.atomic_json(ledger_path, ledger)
            atomic_json(directory / "provider_stopped.json", allocation)
            return
        except Exception:
            (directory / f"stop_verification_error_{attempt}.txt").write_text(traceback.format_exc())
    raise RuntimeError("Provider absence unverified; watchdog must remain active")


def describe(root, output):
    manifest, _ = verify_archive(output)
    results = verified_results(output)
    execution = output / "execution"
    ledger = read_json(execution / "allocation_ledger.json")
    receipt = execution / "recovery/initial_provider_absence.json"
    if receipt.exists():
        ledger = reconciled_ledger(ledger, read_json(receipt))
    now = time.time()
    seconds = sum(a.get("stopped_at_unix", now) - a["started_at_unix"] for a in ledger["allocations"])
    return {"study": manifest["study"], "completed_fits": len(results),
            "completed_ids": [r["spec"]["id"] for r in results],
            "missing_fits": MAXIMUM_FITS - len(results), "maximum_fits": MAXIMUM_FITS,
            "allocated_seconds": seconds, "remaining_seconds": TOTAL_SECONDS - seconds,
            "accounting_is_preview": True,
            "working_source_differences": [name for name, expected in manifest["source_files"].items()
                                           if not (root / name).is_file() or digest(root / name) != expected],
            "execution_source": "verified original input archive", "held_out_evaluation": False}


def interrupted(signum, _frame):
    raise SystemExit(f"Controller interrupted by signal {signum}")


def resume(root, output):
    execution = output / "execution"
    recovery = execution / "recovery"
    require(execution.is_dir(), "Study execution directory is missing; restore the original archive, results and ledger before recovery")
    with ownership_lock(execution):
        manifest, inventory = verify_archive(output)
        require(read_json(execution / "progress.json")["reduced"] is True, "Recovery requires the recorded reduced schedule")
        require(not (execution / "STOP_REQUESTED").exists(), "Stop requested")
        source = stage_source(output, inventory)
        controller = load_controller(source, output)
        results = verified_results(output)
        completed_ids = {r["spec"]["id"] for r in results}
        ledger_path = execution / "allocation_ledger.json"
        original_ledger = read_json(ledger_path)
        for name in ("allocation_ledger.json", "progress.json", "latest_cost_gate.json", "worker_status.json"):
            if (execution / name).exists() and not (recovery / "before" / name).exists():
                write_once(recovery / "before" / name, read_json(execution / name))
        receipt = absence_receipt(output, original_ledger, controller)
        ledger = reconciled_ledger(original_ledger, receipt)
        require(all(a.get("provider_listing_verified_absent") for a in ledger["allocations"]),
                "An owned allocation still needs reconciliation")
        # Confirm no replacement can overlap any prior owned endpoint.
        probe = subprocess.run(["colab", "sessions"], capture_output=True, text=True, timeout=60, check=True)
        listing = probe.stdout + probe.stderr
        for allocation in ledger["allocations"]:
            confirm_absence(listing, allocation["session"], allocation.get("endpoint"))
        require(not re.search(r"^\[deepmzyme-gvp-capacity[^\]]*\]", listing, re.MULTILINE), "Another capacity session is active")
        write_once(recovery / "reconciled_first_allocation.json", ledger["allocations"][0])
        if ledger != original_ledger:
            atomic_json(ledger_path, ledger)
        if len(results) == MAXIMUM_FITS:
            from gvp_capacity_study import report
            return read_json(output / "confirmation_report.json") if (output / "confirmation_report.json").exists() else report(output)
        require(len(ledger["allocations"]) == 1, "The one approved replacement allocation was already used")
        controller.charge(ledger["allocations"], time.time())
        gate = require_admitted(controller, ledger, results)
        write_once(recovery / "preallocation_cost_gate.json", gate)
        write_once(recovery / "operational_manifest.json", {
            "recovery_script_sha256": digest(Path(__file__).resolve()),
            "original_controller_sha256": inventory[CONTROLLER],
            "study_manifest_sha256": digest(output / "manifest.json"),
            "absence_receipt_sha256": digest(recovery / "initial_provider_absence.json"),
            "preserved_results": {r["spec"]["id"]: digest(output / "runs" / r["spec"]["id"] / "capacity_result.json")
                                  for r in results},
            "maximum_fits": MAXIMUM_FITS, "reduced": True,
            "authorization": "User approved recovery plan and requested implementation; existing six-hour ceiling retained"})
        directory = execution / "session_2"
        directory.mkdir()
        session = "deepmzyme-gvp-capacity-v1-s2-20260917"
        start = time.time()
        duration = min(controller.SESSION_CEILING, TOTAL_SECONDS - controller.charge(ledger["allocations"], start))
        allocation = {"session": session, "started_at_unix": start, "requested_gpu": "G4",
                      "ceiling_seconds": duration, "stop_deadline_unix": start + duration - 240}
        ledger["allocations"].append(allocation)
        atomic_json(ledger_path, ledger)
        write_once(directory / "allocation.json", allocation)
        guardian = None
        provider = controller.Provider(session, directory / "provider.log")
        error = None
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, interrupted)
        try:
            guardian = subprocess.Popen([sys.executable, str(source / CONTROLLER), "watchdog", "--output", str(directory)],
                                        stdout=(directory / "watchdog.log").open("x"), stderr=subprocess.STDOUT,
                                        start_new_session=True)
            write_once(directory / "watchdog_started.json", {"pid": guardian.pid, "deadline": allocation["stop_deadline_unix"]})
            provider.command("new", "--gpu", "G4", timeout=180)
            status = provider.command("status")
            (directory / "provider_status.txt").write_text(status)
            require("Hardware: G4 |" in status, "Assigned accelerator differs from G4")
            match = re.search(r"\] (\S+) \|", status)
            require(match is not None, "Provider endpoint missing")
            allocation["endpoint"] = match.group(1)
            atomic_json(ledger_path, ledger)
            hardware = controller.setup_remote(provider, output)
            write_once(directory / "hardware.json", hardware)
            print(json.dumps({"event": "session_ready", "session": session, "hardware": hardware}), flush=True)
            specs = read_json(output / "discovery_matrix.json")
            phase = "discovery"
            while True:
                require(not (execution / "STOP_REQUESTED").exists(), "Stop requested")
                require(guardian.poll() is None, "Independent watchdog exited unexpectedly")
                remaining = [s for s in specs if s["id"] not in completed_ids]
                if not remaining:
                    if phase == "confirmation":
                        break
                    coverage = {"reduced": True, "maximum_fits": MAXIMUM_FITS, "confirmation_outcomes_seen": False,
                                "reason": "Preserved pre-interruption reduced schedule; no optional refinement"}
                    write_once(execution / "coverage_decision.json", coverage)
                    confirmation = output / "confirmation_matrix.json"
                    if confirmation.exists():
                        candidates = read_json(output / "confirmation_candidates.json")
                        require(not candidates["include_performance_authorized_by_cost_gate"], "Confirmation scope changed")
                        specs = read_json(confirmation)
                    else:
                        specs = controller.freeze_confirmation(output, include_performance=False)
                    require(len(specs) == 20, "Protected five-fold/two-seed matrix changed")
                    phase = "confirmation"
                    continue
                gate = controller.cost_gate(controller.charge(ledger["allocations"], time.time()),
                                            [r["total_seconds"] for r in results], len(results), MAXIMUM_FITS)
                atomic_json(execution / "latest_cost_gate.json", gate)
                require(gate["admitted"], "Protected comparison no longer fits the six-hour ceiling")
                require(time.time() + gate["worst_measured_fit_seconds"] * 1.25 + controller.SHUTDOWN_RESERVE < start + duration,
                        "Replacement session reached its protected shutdown reserve")
                spec = remaining[0]
                prior = prior_attempts(execution, spec)
                write_once(directory / ("attempt_" + spec["id"] + ".json"), {
                    "spec": spec, "attempt_number": len(prior) + 1, "prior_launch_intents": prior,
                    "recovery": "restart once from original seed after verified provider loss" if prior else None})
                write_once(directory / ("launch_intent_" + spec["id"] + ".json"), spec)
                launch = controller.launch_fit(provider, output, spec)
                write_once(directory / ("launched_" + spec["id"] + ".json"), launch)
                print(json.dumps({"event": "fit_started", "id": spec["id"], "completed": len(results)}), flush=True)
                while True:
                    time.sleep(30)
                    require(guardian.poll() is None, "Independent watchdog exited unexpectedly")
                    status = controller.poll_fit(provider, spec)
                    atomic_json(execution / "worker_status.json", status)
                    if status["status"] != "running":
                        break
                    epochs = re.findall(r"(?:^|\n)epoch=(\d+)", status.get("tail", ""))
                    print(json.dumps({"event": "fit_progress", "id": spec["id"], "epoch": int(epochs[-1]) if epochs else None}), flush=True)
                    require(time.time() < start + duration - controller.SHUTDOWN_RESERVE, "Fit entered shutdown reserve")
                write_once(directory / ("terminal_" + spec["id"] + ".json"), status)
                require(status["status"] == "completed", f"Fit failed: {status.get('error', status)}")
                result = retrieve_with_recovery(controller, provider, output, status)
                results.append(result)
                completed_ids.add(spec["id"])
                atomic_json(execution / "progress.json", {"phase": phase, "completed_ids": sorted(completed_ids), "reduced": True,
                                                          "allocated_seconds": controller.charge(ledger["allocations"], time.time())})
                print(json.dumps({"event": "fit_verified_locally", "id": spec["id"], "completed": len(results),
                                  "ba": result["selected"]["val_metal_balanced_acc"]}), flush=True)
        except BaseException:
            error = traceback.format_exc()
            atomic_json(recovery / "failure.json", {"error": error, "time": time.time()})
            raise
        finally:
            stop_confirmed(controller, provider, directory, allocation, ledger, ledger_path)
            if guardian is not None:
                guardian.terminate()
                guardian.wait(timeout=10)
            closeout = {"status": "complete" if len(results) == MAXIMUM_FITS and error is None else "incomplete",
                        "reason": error, "completed_fits": len(results), "missing_fits": MAXIMUM_FITS - len(results),
                        "reduced": True, "allocated_seconds": controller.charge(ledger["allocations"], time.time()),
                        "ceiling_seconds": TOTAL_SECONDS,
                        "all_owned_sessions_provider_verified_stopped": all(a.get("provider_listing_verified_absent") for a in ledger["allocations"]),
                        "allocation_note": "Lost-session duration conservatively ends at confirmed provider absence"}
            atomic_json(execution / "closeout.json", closeout)
            print(json.dumps(closeout), flush=True)
        if len(results) == MAXIMUM_FITS:
            from gvp_capacity_study import report
            return report(output)
        return closeout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "resume"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / RELATIVE_OUTPUT).resolve()
    try:
        result = describe(root, output) if args.action == "status" else resume(root, output)
    except (ValueError, FileNotFoundError, BlockingIOError) as error:
        parser.exit(2, f"Capacity recovery blocked: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
