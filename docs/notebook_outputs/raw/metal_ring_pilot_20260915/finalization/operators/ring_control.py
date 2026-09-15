"""Serial one-step runner and immutable incremental archives on the owned VM."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time

from ring_common import allocation_deadline, alive, read, require, require_idle, save, sha

TOP_FILES = ("campaign_manifest.json", "campaign_state.json", "campaign_attempt_ledger.json", "campaign_attempt_ledger.csv",
    "commands.txt", "run_matrix.csv", "expected_split.json", "readiness.json", "training_cache_audit.json",
    "persistent_storage.json", "bootstrap_provenance.json", "source_snapshot_receipt.json", "operator_control_history.json",
    "ring_validation_results.json", "ring_coverage.json", "ring_screen.csv", "ring_decision_record.md",
    "ring_diagnostics.json", "ring_input_audit.json", "ring_normalization_controls.json", "gvp_admission.json", "late_admission.json",
    "host_last_worker_result.json", "host_step_process.json", "active_process.json")


def context():
    cfg = read("/content/ring_session_config.json")
    for name in ("ring_common.py", "ring_control.py"):
        require(sha(Path("/content")/name) == cfg["operator_files"][name], "Operator source changed: "+name)
    root, output, budget = Path(cfg["repo_root"]), Path(cfg["output_dir"]), Path(cfg["budget_root"])
    return cfg, root, output, budget


def worker(phase):
    cfg, root, output, budget = context()
    require(read(output/"bootstrap_provenance.json", {}).get("status") == "ready", "Pinned-data bootstrap is not ready")
    require(time.time() < allocation_deadline(cfg, training=True), "Cumulative training deadline reached")
    sys.path.insert(0, str(root/"src"))
    import run_metal_ring_pilot as pilot
    if phase == "plan":
        result = pilot.plan(root, Path(cfg["data_root"]), output, cfg["source_commit"], Path(cfg["parent_reference_dir"]),
            external_features_root_dir=cfg["external_features_root_dir"], feature_overlay_manifest=cfg["feature_overlay_manifest"], budget_root=budget)
        save(output/"source_snapshot_receipt.json", dict(source_archive=cfg["source_archive"], source_sha256=cfg["source_sha256"],
             operator_files=cfg["operator_files"], budget_handoff_sha256=cfg["budget_handoff_sha256"]))
        result = dict(status="planned", manifest_sha256=sha(output/"campaign_manifest.json"))
    else:
        operation = pilot.preflight if phase == "preflight" else pilot.execute
        kwargs = dict(persistence_receipt="/content/metal_ring_persistence_receipt.json", budget_root=budget)
        if phase == "execute":
            kwargs["max_runs"] = 1
        result = operation(root, output, cfg["allocation_started_epoch"], **kwargs)
    save(output/"host_last_worker_result.json", result)
    return result


def package_archive(output, budget, latest, *, exports, extra_entries=()):
    ident = latest["attempt_id"]
    require(latest["status"] != "running", "Active attempt cannot be archived")
    exports.mkdir(parents=True, exist_ok=True)
    descriptor_path = exports/(ident+"_archive.json")
    old = read(descriptor_path)
    if old:
        require(sha(Path(old["archive"])) == old["archive_sha256"], "Existing archive changed")
        return old
    archive = exports/(ident+".tar.gz")
    require(not archive.exists(), "Unverified existing archive preserved; reconcile before retry")
    with tarfile.open(archive, "w:gz", compresslevel=3) as stream:
        for name in TOP_FILES:
            path = output/name
            if path.is_file():
                stream.add(path, arcname=name, recursive=False)
        for key in ("run_dir", "log"):
            if latest.get(key):
                path = Path(latest[key])
                require(path.is_relative_to(output), "Attempt output escapes campaign directory")
                if path.exists():
                    stream.add(path, arcname=str(path.relative_to(output)))
        if latest.get("block") == "P":
            for name in ("input_audit_runs", "ring_preflight_late_ring0", "ring_preflight_late_ring1"):
                if (output/name).is_dir():
                    stream.add(output/name, arcname=name)
        if (output/"transfer_receipts").is_dir():
            stream.add(output/"transfer_receipts", arcname="transfer_receipts")
        for name in extra_entries:
            stream.add(output/name, arcname=name)
        for name in ("budget_handoff.json", "allocation_ledger.json", "bootstrap_budget_usage.json"):
            path = budget/name
            require(path.is_file(), "Missing shared budget evidence: "+name)
            stream.add(path, arcname="budget/"+name, recursive=False)
    parts = []
    with archive.open("rb") as source:
        for index, data in enumerate(iter(lambda: source.read(16*1024*1024), b"")):
            part = archive.with_name(archive.name+f".part{index:03d}")
            part.write_bytes(data)
            parts.append(dict(path=str(part), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    result = dict(attempt_id=ident, archive=str(archive), archive_sha256=sha(archive), bytes=archive.stat().st_size,
                  manifest_sha256=sha(output/"campaign_manifest.json"), parts=parts)
    save(descriptor_path, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("launch", "worker", "status", "archive", "finalize"))
    parser.add_argument("--phase", choices=("plan", "preflight", "execute"), default="execute")
    args = parser.parse_args()
    cfg, root, output, budget = context()
    attempts = read(output/"campaign_attempt_ledger.json", [])
    if args.action == "worker":
        result = worker(args.phase)
    elif args.action == "launch":
        require_idle(output)
        require(time.time() < allocation_deadline(cfg, training=True), "Cumulative training deadline reached")
        require(read(output/"bootstrap_provenance.json", {}).get("status") == "ready", "Bootstrap is not ready")
        command = [sys.executable, "/content/ring_control.py", "worker", "--phase", args.phase]
        path = output/(f"host_step_{len(attempts)+1:03d}_{args.phase}.log")
        with path.open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        result = dict(pid=process.pid, phase=args.phase, started_epoch=time.time(), log=str(path), command=command)
        save(output/"host_step_process.json", result)
    elif args.action == "status":
        record = read(output/"host_step_process.json", {})
        bootstrap = read(output/"bootstrap_provenance.json", {})
        worker_result = read(output/"host_last_worker_result.json", {})
        result = dict(alive=alive(record) or alive(read(output/"active_process.json", {})),
            bootstrap_alive=alive(read(output/"bootstrap_process.json", {})), bootstrap={key: bootstrap.get(key) for key in ("status", "elapsed_seconds", "error_type", "error")},
            state=read(output/"campaign_state.json"), readiness=read(output/"readiness.json"), attempts=len(attempts),
            worker_result={key: worker_result[key] for key in ("status", "completed_full_runs", "completed_smokes", "state") if key in worker_result})
        complete = [a for a in attempts if a["status"] == "completed"]
        result["completed_full_runs"] = len({a["run_id"] for a in complete if a["block"] not in ("P", "S")})
        result["completed_smokes"] = len({a["run_id"] for a in complete if a["block"] == "S"})
        if attempts:
            last = attempts[-1]
            result["latest"] = {key: last.get(key) for key in ("attempt_id", "run_id", "block", "status", "elapsed_seconds", "error")}
            result["latest_transfer_verified"] = read(output/"transfer_receipts"/(last["attempt_id"]+".json"), {}).get("drive_verified", False)
            metrics = Path(last["run_dir"])/"epoch_metrics.csv"
            if metrics.is_file():
                with metrics.open() as stream:
                    history = list(csv.DictReader(stream))
                result["epochs"] = len(history)
                if history:
                    result["last_val_ba"] = history[-1].get("val_metal_balanced_acc")
        if record.get("log") and Path(record["log"]).is_file():
            result["worker_log_tail"] = Path(record["log"]).read_text()[-1800:]
    elif args.action == "finalize":
        require(sha(Path('/content/ring_finalization.py')) == cfg['operator_files']['ring_finalization.py'], "Finalization operator changed")
        from ring_finalization import capture
        result = capture(cfg, root, output, budget, exports=Path("/content/verified_exports/metal_ring_pilot_v1/finalization"))
    else:
        require_idle(output)
        require(attempts, "There is no completed attempt to archive")
        result = package_archive(output, budget, attempts[-1], exports=Path("/content/verified_exports/metal_ring_pilot_v1"))
    print("RING_CONTROL="+json.dumps(result))


if __name__ == "__main__":
    main()
