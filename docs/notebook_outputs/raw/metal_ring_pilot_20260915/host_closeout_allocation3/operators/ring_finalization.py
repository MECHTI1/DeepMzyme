"""Terminal metadata capture with selected checkpoint verification; no inference."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

from ring_common import read, require, require_idle, save, sha


RUN_METADATA = ("run_config.json", "run_metadata.json", "dataset_summary.json", "epoch_metrics.csv")


def checkpoint_binding(row, *, load_checkpoint=None):
    if load_checkpoint is None:
        import torch
        load_checkpoint = lambda path: torch.load(path, map_location="cpu", weights_only=True)
    directory = Path(row["run_dir"])
    before = {name: sha(directory/name) for name in (*RUN_METADATA, "best_model_checkpoint.pt", "last_model_checkpoint.pt")}
    metadata, saved = read(directory/"run_metadata.json"), read(directory/"run_config.json")
    checkpoint = load_checkpoint(directory/"best_model_checkpoint.pt")
    metric = "val_metal_balanced_acc"
    selected = next((record for record in saved["history"] if record["epoch"] == row["selected_epoch"]), None)
    require(selected is not None, "Selected epoch is missing from saved history")
    require(checkpoint.get("epoch") == metadata["selected_checkpoint_epoch"] == row["selected_epoch"], "Selected checkpoint epoch differs")
    require(checkpoint.get("selection_metric") == metadata.get("selection_metric") == metric, "Selected checkpoint metric differs")
    import torch
    def jsonable(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {str(key): jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [jsonable(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        return value
    require(jsonable(checkpoint.get("config")) == saved["config"], "Selected checkpoint configuration differs")
    normalization = jsonable(checkpoint.get("normalization_stats"))
    require(normalization is not None and normalization == metadata.get("normalization_stats"), "Selected checkpoint normalization differs")
    normalization_sha = hashlib.sha256(json.dumps(normalization, sort_keys=True, default=str).encode()).hexdigest()
    require(normalization_sha == row["normalization_stats_sha256"], "Selected checkpoint normalization fingerprint differs")
    expected = max(float(record[metric]) for record in saved["history"])
    require(math.isfinite(expected) and all(math.isclose(float(value), expected, abs_tol=1e-10, rel_tol=0)
            for value in (checkpoint["selection_metric_value"], selected[metric], row["balanced_accuracy"])),
            "Selected checkpoint value differs from the native history maximum")
    require(before == {name: sha(directory/name) for name in before}, "Run evidence changed during checkpoint verification")
    return dict(run_id=row["id"], block=row["block"], selected_epoch=row["selected_epoch"],
                balanced_accuracy=row["balanced_accuracy"], normalization_stats_sha256=row["normalization_stats_sha256"],
                files=before, run_dir=str(directory))


def capture(cfg, root, output, budget, *, exports, operator_root=Path("/content"), pilot=None, load_checkpoint=None):
    from ring_control import TOP_FILES, package_archive
    require_idle(output)
    state = read(output/"campaign_state.json", {})
    require(state.get("status") in ("completed", "budget_stopped"), "Finalization requires a terminal completed or budget_stopped campaign")
    require(not any(a.get("status") == "running" for a in read(output/"campaign_attempt_ledger.json", [])), "A ledger attempt is still active")
    if pilot is None:
        sys.path.insert(0, str(root/"src"))
        import run_metal_ring_pilot as pilot
    manifest = pilot.verify_manifest(root, output)
    attempts = read(output/"campaign_attempt_ledger.json", [])
    pilot.base._require_transfer(output, attempts, {"method": "verified_archive_transfer"})
    elapsed = pilot.base.allocation_elapsed(budget, cfg["allocation_started_epoch"])
    summary = pilot.summarize(output)
    require(summary["state"] == state, "Summary terminal state differs")
    if state["status"] == "completed":
        require(summary["completed_smokes"] == 4 and summary["completed_full_runs"] == 16, "Completed state lacks all planned runs")
    rows = pilot.rows(output, manifest)
    bindings = [checkpoint_binding(row, load_checkpoint=load_checkpoint) for row in rows]
    source_files = {}
    for name, expected in manifest["source_files"].items():
        require(sha(root/name) == expected, "Frozen source changed during capture: "+name)
        source_files[name] = expected
    for name, expected in cfg["operator_files"].items():
        require(sha(operator_root/name) == expected, "Operator source changed during capture: "+name)
    old = read(exports/"finalization_archive.json")
    if old:
        prior = read(output/"finalization_stage/final_capture_receipt.json")
        require(prior and prior["runs"] == bindings and prior["terminal_state"] == state,
                "Existing finalization belongs to different completed evidence")
        require(sha(Path(old["archive"])) == old["archive_sha256"], "Existing final archive changed")
        return old
    staging = output/"finalization_stage"
    require(not staging.exists(), "Unverified finalization staging preserved; reconcile before retry")
    staging.mkdir()
    for name in ("completed_runs", "source", "operators"):
        (staging/name).mkdir()
    for name in TOP_FILES:
        if (output/name).is_file():
            shutil.copyfile(output/name, staging/name)
    if (output/"transfer_receipts").is_dir():
        shutil.copytree(output/"transfer_receipts", staging/"transfer_receipts")
    for binding in bindings:
        directory = Path(binding["run_dir"])
        for name in RUN_METADATA:
            target = staging/"completed_runs"/binding["run_id"]/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(directory/name, target)
            require(sha(target) == binding["files"][name], "Copied run metadata changed")
    for prefix, source_root, entries in (("source", root, source_files), ("operators", operator_root, cfg["operator_files"])):
        for name, expected in entries.items():
            target = staging/prefix/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_root/name, target)
            require(sha(target) == expected, "Copied frozen source changed")
    save(staging/"final_capture_receipt.json", dict(status="terminal_metadata_verified_gpu_not_stopped", captured_epoch=time.time(),
         terminal_state=state, manifest_sha256=sha(output/"campaign_manifest.json"), source_archive_sha256=cfg["source_sha256"],
         operator_files=cfg["operator_files"], completed_smokes=summary["completed_smokes"], completed_full_runs=summary["completed_full_runs"],
         required_gvp_complete=summary["coverage"]["GVP"]["complete"], allocated_seconds_at_capture=elapsed,
         budget_handoff_sha256=sha(budget/"budget_handoff.json"), runs=bindings, held_out_evaluation=False, checkpoint_inference_performed=False))
    return package_archive(staging, budget, {"attempt_id": "finalization", "status": state["status"]}, exports=exports,
                           extra_entries=("final_capture_receipt.json", "completed_runs", "source", "operators"))
