"""Run the separate five-arm coordination-geometry comparison on a shared budget.

The original architecture campaign stays in its frozen checkout. This campaign
has a new source/manifest identity and charges the original allocation ledger.
It runs five one-epoch smokes, ten seed-42 fits, then five matched seed-43 fits.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import shlex
import statistics
import sys
import time

import run_metal_architecture_pilot as base

PROFILE = "metal_coordination_geometry_pilot_v1"
ARMS = {
    "A": ("none", "none"),
    "B": ("counts", "none"),
    "C": ("counts_angles", "none"),
    "D": ("counts", "per_metal"),
    "E": ("counts_angles", "per_metal"),
}
BLOCKS = ("S", "G1", "G2", "GR")
DATASET, BUNDLE, BUNDLE_SHA, METRIC = base.DATASET, base.BUNDLE, base.BUNDLE_SHA, base.METRIC
require, read, save = base.require, base.read, base.save


def validate(config, arm, epochs=50):
    geometry, nodes = ARMS[arm]
    require(config.site_geometry_features == geometry and config.metal_node_mode == nodes,
            f"Wrong geometry controls for arm {arm}")
    require(config.model_architecture == "only_gvp" and not config.use_early_esm and not config.use_esm_branch,
            "Geometry campaign requires Only-GVP")
    require(config.metal_label_scheme == "merge_fe_class_viii" and config.device == "cuda",
            "Geometry campaign requires direct four-class CUDA training")
    require(config.structural_readout_scope == "residue_only", "All five arms pool residues only")
    # Apply every frozen baseline guard, allowing just the declared graph change.
    base.validate(replace(config, metal_node_mode="none"), epochs)


def plan(root, data, output, source_commit, parent_campaign_dir,
         external_features_root_dir=None, feature_overlay_manifest=None):
    root, data, output, parent = [Path(p).resolve() for p in (root, data, output, parent_campaign_dir)]
    parent_manifest = read(parent / "campaign_manifest.json")
    require(parent_manifest and parent_manifest["profile"] == base.PROFILE, "Expected the original architecture campaign")
    require(parent_manifest["dataset"] == DATASET and parent_manifest["bundle_sha256"] == BUNDLE_SHA,
            "Parent dataset identity does not match the geometry campaign")
    reference = read(parent / "expected_split.json")
    require(reference is not None, "The parent campaign must have a certified retained split")
    external = Path(external_features_root_dir or parent_manifest["external_features_root_dir"]).resolve()
    overlay = feature_overlay_manifest or (parent_manifest.get("feature_overlay_manifest") or {}).get("path")
    require(overlay is not None, "A complete, checksum-verified external-feature overlay is required")
    overlay = Path(overlay).resolve()
    base.validate_feature_overlay(data, external, overlay)
    template = base._templates(root, data, output, external)["Only-GVP"]
    runs = []

    def add(block, arm, lr, seed, epochs):
        geometry, nodes = ARMS[arm]
        ident = f"{block}_{arm}_{geometry}_{nodes}_lr{lr:g}_s{seed}"
        command = list(map(str, template["command"]))
        command[0] = sys.executable
        for flag, value in (("--site-geometry-features", geometry), ("--metal-node-mode", nodes),
                            ("--structural-readout-scope", "residue_only"),
                            ("--metal-label-scheme", "four_class"), ("--selection-metric", METRIC),
                            ("--edge-radius", 6), ("--classifier-pool-distance-cutoff", 0),
                            ("--epochs", epochs), ("--learning-rate", lr), ("--seed", seed),
                            ("--run-name", ident), ("--runs-dir", output / "runs")):
            command = base.replace_option(command, flag, value)
        config = base.parse_config(command)
        validate(config, arm, epochs)
        payload = json.loads(json.dumps(asdict(config), default=str))
        runs.append(dict(id=ident, block=block, arm=arm, family="Only-GVP", scheme="four_class", lr=lr, seed=seed,
                         epochs=epochs, geometry=geometry, metal_node_mode=nodes, command=command,
                         env={**template.get("env", {}), "DEEPGM_METAL_LABEL_SCHEME": "four_class"},
                         run_dir=str(output / "runs" / ident), config=payload, config_sha256=base.fingerprint(payload)))

    for arm in ARMS:
        add("S", arm, base.LRS[0], 42, 1)
    for block, lr in zip(("G1", "G2"), base.LRS):
        for arm in ARMS:
            add(block, arm, lr, 42, 50)
    for arm in ARMS:
        for lr in base.LRS:
            add("GR", arm, lr, 43, 50)
    source_paths = sorted((root / "src").rglob("*.py")) + [
        root / "notebooks/DeepMzyme_training_colab.ipynb", root / "docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md"]
    train = data / DATASET / "train"
    dataset_paths = [train / "structure_manifest.csv", train / "final_data_summarazing_table_transition_metals_only_catalytic.csv"]
    manifest = dict(profile=PROFILE, source_commit=source_commit, dataset=DATASET, bundle=BUNDLE, bundle_sha256=BUNDLE_SHA,
                    data_root=str(data), output_dir=str(output), parent_campaign_dir=str(parent),
                    parent_manifest_sha256=base.digest(parent / "campaign_manifest.json"),
                    expected_split_sha256=base.digest(parent / "expected_split.json"),
                    cohort_sha256=base.fingerprint(base.cohort_identity(reference)),
                    external_features_root_dir=str(external),
                    feature_overlay_manifest={"path": str(overlay), "sha256": base.digest(overlay)},
                    source_files={str(p.relative_to(root)): base.digest(p) for p in source_paths},
                    dataset_files={str(p.relative_to(data)): base.digest(p) for p in dataset_paths},
                    runs=runs, block_order=list(BLOCKS), arms={k: list(v) for k, v in ARMS.items()}, selection_metric=METRIC,
                    maximum_smokes=5, maximum_full_runs=15, held_out_evaluation=False,
                    geometry_slots=8, site_encoder_input_dim=12, structural_readout_scope="residue_only",
                    node_type_embeddings="generic_residue_metal_module_identical_all_explicit_arms",
                    budget_policy=dict(shared_with_parent=True, training_deadline_minutes=570,
                                       admission_factor=1.25, main_minutes=450, retry_minutes=60))
    existing = read(output / "campaign_manifest.json")
    require(existing is None or existing == manifest, "Geometry manifest changed; use a new campaign identity")
    save(output / "campaign_manifest.json", manifest)
    (output / "expected_split.json").write_bytes((parent / "expected_split.json").read_bytes())
    (output / "commands.txt").write_text("\n\n".join(shlex.join(r["command"]) for r in runs) + "\n")
    base.csv_save(output / "run_matrix.csv", runs,
                  ["id", "block", "arm", "geometry", "metal_node_mode", "lr", "seed", "epochs", "run_dir"])
    return manifest


def verify_manifest(root, output):
    manifest = read(output / "campaign_manifest.json")
    require(manifest and manifest["profile"] == PROFILE, "Missing geometry campaign manifest")
    for name, checksum in manifest["source_files"].items():
        require(base.digest(root / name) == checksum, f"Geometry source changed after planning: {name}")
    for name, checksum in manifest["dataset_files"].items():
        require(base.digest(Path(manifest["data_root"]) / name) == checksum, f"Training data changed: {name}")
    parent = Path(manifest["parent_campaign_dir"])
    require(base.digest(parent / "campaign_manifest.json") == manifest["parent_manifest_sha256"], "Parent manifest changed")
    require(base.digest(output / "expected_split.json") == manifest["expected_split_sha256"], "Expected retained split changed")
    overlay = manifest["feature_overlay_manifest"]
    require(base.digest(overlay["path"]) == overlay["sha256"], "External overlay manifest changed")
    base.validate_feature_overlay(manifest["data_root"], manifest["external_features_root_dir"], overlay["path"])
    return manifest


def rows(output, manifest):
    expected = read(output / "expected_split.json")
    planned = {r["id"]: r for r in manifest["runs"]}
    result = []
    for attempt in base._attempts(output):
        if attempt["status"] == "completed" and attempt["run_id"] in planned:
            run = planned[attempt["run_id"]]
            row = base.completed_result(run, attempt["run_dir"], expected)
            normalization = read(Path(attempt["run_dir"]) / "run_metadata.json").get("normalization_stats")
            require(normalization is not None, "Missing training-fitted normalization metadata")
            row.update(arm=run["arm"], geometry=run["geometry"], metal_node_mode=run["metal_node_mode"],
                       elapsed_seconds=attempt["elapsed_seconds"], normalization_stats_sha256=base.fingerprint(normalization))
            result.append(row)
    return result


def selected_lrs(results):
    selected = {}
    for arm in ARMS:
        candidates = [r for r in results if r["arm"] == arm and r["seed"] == 42 and r["block"] in ("G1", "G2")]
        if len(candidates) == 2 and {r["lr"] for r in candidates} == set(base.LRS):
            selected[arm] = max(candidates, key=lambda r: (r["balanced_accuracy"], r["minimum_recall"], -r["lr"]))["lr"]
    return selected


def active_runs(manifest, results, *, allow_repeat_placeholders=False):
    selected = selected_lrs(results)
    return [r for r in manifest["runs"] if r["block"] != "GR"
            or r["lr"] == selected.get(r["arm"], base.LRS[0] if allow_repeat_placeholders else None)]


def forecast(run, attempts, manifest):
    by_id = {r["id"]: r for r in manifest["runs"]}
    measured = [a["setup_seconds"] + a["epoch_seconds"] * run["epochs"] for a in attempts
                if a["status"] == "completed" and a["run_id"] in by_id
                and by_id[a["run_id"]]["arm"] == run["arm"] and "epoch_seconds" in a]
    require(measured, f"Arm {run['arm']} has no successful timing smoke")
    return max(measured)


def shared_budget_available(output, budget_root, elapsed, *, retry=False):
    """Include geometry preparation/smokes in the shared main/retry allowances."""
    parent_attempts = base._attempts(budget_root)
    own_attempts = base._attempts(output)
    parent_main = sum(float(a.get("elapsed_seconds") or 0) for a in parent_attempts
                      if a["block"] not in ("P", "S") and not a.get("retry_of"))
    own_main = sum(float(a.get("elapsed_seconds") or 0) for a in own_attempts if not a.get("retry_of"))
    retry_used = sum(float(a.get("elapsed_seconds") or 0) for a in parent_attempts + own_attempts if a.get("retry_of"))
    category_left = 60 * (60 if retry else 450) - (retry_used if retry else parent_main + own_main)
    return max(0, min(570 * 60 - elapsed, category_left))


def publish_budget_usage(output, budget_root):
    attempts = base._attempts(output)
    save(budget_root / "coordination_geometry_budget_usage.json", dict(
        campaign_dir=str(output), manifest_sha256=base.digest(output / "campaign_manifest.json"),
        normal_elapsed_seconds=sum(float(a.get("elapsed_seconds") or 0) for a in attempts if not a.get("retry_of")),
        retry_elapsed_seconds=sum(float(a.get("elapsed_seconds") or 0) for a in attempts if a.get("retry_of")),
        original_runner_integration="The host must subtract these costs when admitting later original-campaign blocks; the shared allocation ledger enforces the cumulative deadline."))


def ensure_parent_idle(parent):
    active = read(parent / "active_process.json", {})
    if active.get("pid") and active.get("status") == "running":
        try:
            os.kill(int(active["pid"]), 0)
        except ProcessLookupError:
            return
        raise ValueError("The original campaign still has an active training process")


def require_parent_architecture_screen(parent):
    manifest = read(parent / "campaign_manifest.json")
    required = {r["id"] for r in manifest["runs"] if r["block"] in ("A1", "A2")}
    require(len(required) == 8, "Original architecture screen must contain eight A1/A2 runs")
    complete = {r["id"] for r in base.completed_rows(parent, manifest)}
    require(required <= complete, "Complete and verify original A1/A2 before executing geometry runs")


def check_normalization_controls(results):
    smokes = {r["arm"]: r for r in results if r["block"] == "S"}
    require(set(smokes) == set(ARMS), "All five geometry smokes must pass")
    for group in (("A", "B", "C"), ("D", "E")):
        require(len({smokes[a]["normalization_stats_sha256"] for a in group}) == 1,
                f"Geometry-only arms have different fitted normalization: {group}")
    for row in results:
        require(row["normalization_stats_sha256"] == smokes[row["arm"]]["normalization_stats_sha256"],
                f"Fitted normalization changed within arm {row['arm']}")
    return {arm: row["normalization_stats_sha256"] for arm, row in smokes.items()}


def _context(root, output, budget_root, started, receipt):
    root, output, budget_root = [Path(p).resolve() for p in (root, output, budget_root)]
    manifest = verify_manifest(root, output)
    require(budget_root == Path(manifest["parent_campaign_dir"]), "Geometry must charge the original campaign budget root")
    require((budget_root / "allocation_ledger.json").is_file(), "Original cumulative allocation ledger must be restored first")
    ensure_parent_idle(budget_root)
    storage = base.storage_check(output, receipt)
    elapsed = base.allocation_elapsed(budget_root, started)
    attempts = base._attempts(output)
    base._recover_interrupted(output, attempts)
    base._require_transfer(output, attempts, storage)
    return root, output, budget_root, manifest, storage, elapsed, attempts


def preflight(root, output, allocation_started_epoch, persistence_receipt=None, budget_root=None):
    require(budget_root is not None, "Supply the original campaign --budget-root")
    root, output, budget_root, manifest, _, elapsed, attempts = _context(
        root, output, budget_root, allocation_started_epoch, persistence_receipt)
    ready = read(output / "readiness.json", {})
    if ready.get("status") == "passed":
        require(ready["manifest_sha256"] == base.digest(output / "campaign_manifest.json"), "Readiness source identity changed")
        base.verify_training_cache(output)
        return ready
    base.verify_training_cache(budget_root)
    available = shared_budget_available(output, budget_root, elapsed,
                                       retry=any(a["run_id"] == "geometry_readiness" for a in attempts))
    run = dict(id="geometry_readiness", block="P", run_dir=str(output / "geometry_preflight"))
    command = [sys.executable, str(root / "src/run_metal_coordination_geometry_pilot.py"), "_preflight", "--output-dir", str(output)]
    try:
        base._record_attempt(root, output, run, command, {"DEEPGM_METAL_LABEL_SCHEME": "four_class"}, available, attempts)
    finally:
        publish_budget_usage(output, budget_root)
        base.allocation_elapsed(budget_root, allocation_started_epoch)
    return read(output / "readiness.json")


def _preflight_worker(root, output, run_name="geometry_preflight"):
    import torch
    from torch_geometric.data import Batch
    from model import residue_node_mask, metal_distance_pool_mask
    from training.run import prepare_run, set_seed
    manifest = verify_manifest(root, output)
    parent = Path(manifest["parent_campaign_dir"])
    base.verify_training_cache(parent)
    require(torch.cuda.is_available(), "Geometry preflight requires CUDA")
    template = next(r for r in manifest["runs"] if r["block"] == "S" and r["arm"] == "A")
    command = base.replace_option(template["command"], "--runs-dir", output)
    command = base.replace_option(command, "--run-name", run_name)
    config = base.parse_config(command)
    validate(config, "A", 1)
    set_seed(config.seed, deterministic=config.deterministic)
    prepared = prepare_run(config)
    identity = base.cohort_identity(prepared.dataset_summary)
    require(base.fingerprint(identity) == manifest["cohort_sha256"], "Geometry retained cohort differs from the original campaign")
    model = prepared.model
    require(model.site_feature_encoder[0].in_features == 12 and model.node_type_embedding is not None,
            "Explicit geometry must keep twelve site-input dimensions and generic node-type embeddings")
    diagnostics = []
    for part, loader in (("train", prepared.train_loader), ("validation", prepared.val_loader)):
        require(loader is not None, "No validation loader")
        for index in range(len(loader.dataset)):
            graph = loader.dataset[index]
            batch = Batch.from_data_list([graph])
            mask = residue_node_mask(batch)
            pooled = metal_distance_pool_mask(batch, 0, mask)
            raw = graph.site_ligand_angle_stats_raw
            require(raw.shape == (1, 8) and torch.isfinite(raw).all(), "Invalid fixed eight-slot geometry tensor")
            require(int(mask.sum()) == int(pooled.sum()) > 0, "Residue readout changed")
            require(set(graph.node_type_id.tolist()).issubset({0, 1}), "Non-generic node type ID")
            diagnostics.append(dict(part=part, pocket_id=identity[part][index]["pocket_id"],
                                    residues=int(mask.sum()), site_metal_count=int(graph.metal_count.item()),
                                    raw_geometry_slots=raw.flatten().tolist()))
    batch = next(iter(prepared.train_loader)).to(config.device)
    result = model(batch)
    require(torch.isfinite(result["loss"]), "Nonfinite actual-data forward loss")
    result["loss"].backward()
    torch.cuda.synchronize()
    save(output / "training_cache_audit.json", read(parent / "training_cache_audit.json"))
    save(output / "geometry_diagnostics.json", dict(site_encoder_input_dim=12, geometry_slots=8,
         structural_readout_scope="residue_only", pockets=diagnostics,
         model_parameter_count=sum(p.numel() for p in model.parameters()),
         mode_coverage="A actual-data preflight; all A-E require a completed one-epoch smoke before full-run admission"))
    save(output / "readiness.json", dict(status="passed", manifest_sha256=base.digest(output / "campaign_manifest.json"),
         parent_manifest_sha256=manifest["parent_manifest_sha256"], cohort_sha256=manifest["cohort_sha256"],
         gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
         source_specific_actual_data_forward_backward=True, held_out_evaluation=False))


def execute(root, output, allocation_started_epoch, persistence_receipt=None, budget_root=None, max_runs=1):
    require(max_runs == 1 and budget_root is not None, "Execute one attempt using the original campaign --budget-root")
    root, output, budget_root, manifest, storage, elapsed, attempts = _context(
        root, output, budget_root, allocation_started_epoch, persistence_receipt)
    require_parent_architecture_screen(budget_root)
    ready = read(output / "readiness.json", {})
    require(ready.get("status") == "passed" and ready["manifest_sha256"] == base.digest(output / "campaign_manifest.json"),
            "Complete source-specific geometry preflight before execution")
    base.verify_training_cache(output)
    results = rows(output, manifest)
    complete = {r["id"] for r in results}
    pending = [r for r in active_runs(manifest, results) if r["id"] not in complete]
    if not pending:
        require(len([r for r in results if r["block"] != "S"]) == 15, "Geometry completion requires all fifteen full runs")
        save(output / "campaign_state.json", dict(status="completed", allocated_seconds=elapsed))
        return summarize(output)
    run = pending[0]
    retry = any(a["run_id"] == run["id"] for a in attempts)
    available = shared_budget_available(output, budget_root, elapsed, retry=retry)
    if run["block"] != "S":
        normalization_checks = check_normalization_controls(results)
        save(output / "geometry_normalization_controls.json", dict(arm_sha256=normalization_checks,
             same_graph_pairs_verified=True,
             metal_node_comparison="Adding metal nodes also changes the fitted edge normalization distribution"))
        remaining = [r for r in active_runs(manifest, results, allow_repeat_placeholders=True)
                     if r["block"] != "S" and r["id"] not in complete]
        seconds = sum(forecast(r, attempts, manifest) for r in remaining) * 1.25
        normal_available = shared_budget_available(output, budget_root, elapsed)
        if seconds > normal_available or available <= 20:
            save(output / "campaign_state.json", dict(status="budget_stopped", next_run_id=run["id"],
                 reason="The remaining geometry comparison does not fit the measured forecast plus 25% margin",
                 forecast_seconds=seconds, available_seconds=normal_available, allocated_seconds=elapsed))
            return summarize(output)
        admissions = read(output / "geometry_campaign_admission.json")
        if admissions is None:
            save(output / "geometry_campaign_admission.json", dict(full_run_count=15, forecast_seconds=seconds,
                 allocated_seconds=elapsed, admitted_epoch=time.time(), margin=1.25))
        if run["block"] == "GR":
            selected = selected_lrs(results)
            require(len(selected) == 5, "Freeze all five LR selections before seed repeats")
            old = read(output / "geometry_selected_learning_rates.json")
            require(old is None or old == selected, "Selected geometry learning rates changed")
            save(output / "geometry_selected_learning_rates.json", selected)
    elif available <= 20:
        save(output / "campaign_state.json", dict(status="budget_stopped", next_run_id=run["id"], allocated_seconds=elapsed))
        return summarize(output)
    validate(base.parse_config(run["command"]), run["arm"], run["epochs"])
    save(output / "campaign_state.json", dict(status="running", run_id=run["id"], block=run["block"], arm=run["arm"], allocated_seconds=elapsed))
    try:
        attempt = base._record_attempt(root, output, run, run["command"], run["env"], available, attempts)
    except BaseException:
        save(output / "campaign_state.json", dict(status="failed", run_id=run["id"], block=run["block"]))
        raise
    finally:
        publish_budget_usage(output, budget_root)
        base.allocation_elapsed(budget_root, allocation_started_epoch)
    save(output / "campaign_state.json", dict(status="awaiting_archive_transfer" if storage["method"] == "verified_archive_transfer"
         else "step_completed", run_id=run["id"], attempt_id=attempt["attempt_id"], attempt_status=attempt["status"]))
    return summarize(output)


def summarize(output):
    output = Path(output)
    manifest = read(output / "campaign_manifest.json")
    results = rows(output, manifest)
    complete = {r["id"] for r in results}
    active = active_runs(manifest, results)
    coverage = {block: dict(expected_runs=5, completed_runs=sum(r["block"] == block for r in results),
                           complete=sum(r["block"] == block for r in results) == 5)
                for block in BLOCKS}
    full = [r for r in results if r["block"] != "S"]
    for row in full:
        row["block_complete"] = coverage[row["block"]]["complete"]
    selected = selected_lrs(results)
    repeats = []
    for arm, lr in selected.items():
        pair = [r for r in full if r["arm"] == arm and r["lr"] == lr]
        if len(pair) == 2 and {r["seed"] for r in pair} == {42, 43}:
            repeats.append(dict(arm=arm, lr=lr, seeds=[42, 43],
                 mean_native_balanced_accuracy=sum(r["balanced_accuracy"] for r in pair) / 2,
                 sample_sd_native_balanced_accuracy=statistics.stdev(r["balanced_accuracy"] for r in pair),
                 worst_native_minimum_recall=min(r["minimum_recall"] for r in pair),
                 run_ids=[r["id"] for r in pair], descriptive_only=True))
    summary = dict(profile=PROFILE, completed_smokes=len(results) - len(full), completed_full_runs=len(full),
                   coverage=coverage, selected_learning_rates=selected, matched_two_seed_repeats=repeats,
                   promoted=False, held_out_evaluation=False, confidence_intervals_computed=False,
                   state=read(output / "campaign_state.json", {"status": "planned"}), runs=full)
    save(output / "geometry_validation_results.json", summary)
    save(output / "geometry_coverage.json", [dict(id=r["id"], arm=r["arm"], block=r["block"], completed=r["id"] in complete)
                                             for r in active])
    base.csv_save(output / "geometry_screen.csv", full,
                  ["id", "arm", "geometry", "metal_node_mode", "block", "lr", "seed", "selected_epoch", "balanced_accuracy",
                   "collapsed4_balanced_accuracy", "minimum_recall", "per_class_recall", "elapsed_seconds", "normalization_stats_sha256", "block_complete", "run_dir"])
    (output / "geometry_decision_record.md").write_text(
        f"# {PROFILE}\n\nCompleted smokes: {summary['completed_smokes']}/5; full runs: {len(full)}/15.\n\n"
        "All arms retain the same four base site features, eight explicit geometry slots, generic node-type embeddings, "
        "and residue-only classifier pooling. The five arms vary active geometry slots and generic metal-node insertion. "
        "Metal-node insertion also changes the fitted edge normalization distribution; its contrast estimates the full added-node representation package. "
        "The legacy original GVP run is a separate baseline; the explicit none arm is not an identical old-model rerun.\n\n"
        "Native validation balanced accuracy selects checkpoints and each arm's LR, with minimum recall then lower LR as ties. "
        "Only complete blocks support comparisons. Two-seed summaries are descriptive; no promotion or confidence claim is made.\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "preflight", "execute", "summarize", "_preflight", "budget-status"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--parent-campaign-dir", type=Path)
    parser.add_argument("--external-features-root-dir", type=Path)
    parser.add_argument("--feature-overlay-manifest", type=Path)
    parser.add_argument("--allocation-started-epoch", type=float)
    parser.add_argument("--budget-root", type=Path)
    parser.add_argument("--persistence-receipt", type=Path)
    parser.add_argument("--max-runs", type=int, default=1)
    parser.add_argument("--preflight-run-name", default="geometry_preflight", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.phase == "plan":
        require(args.data_root and args.source_commit and args.parent_campaign_dir, "Plan needs data, source identity, and parent campaign")
        result = plan(root, args.data_root, args.output_dir, args.source_commit, args.parent_campaign_dir,
                      args.external_features_root_dir, args.feature_overlay_manifest)
        print(f"Saved {len(result['runs'])} templates: five smokes and at most fifteen full fits")
    elif args.phase in ("preflight", "execute"):
        require(args.allocation_started_epoch is not None and args.budget_root, "Supply original budget root and actual allocation start")
        operation = preflight if args.phase == "preflight" else execute
        keywords = {"max_runs": args.max_runs} if args.phase == "execute" else {}
        print(json.dumps(operation(root, args.output_dir, args.allocation_started_epoch,
                                   args.persistence_receipt, args.budget_root, **keywords), indent=2))
    elif args.phase == "_preflight":
        _preflight_worker(root, args.output_dir, args.preflight_run_name)
    elif args.phase == "budget-status":
        require(args.allocation_started_epoch is not None and args.budget_root, "Supply budget root and allocation start")
        elapsed = base.allocation_elapsed(args.budget_root, args.allocation_started_epoch)
        print(json.dumps(dict(allocated_seconds=elapsed, normal_available_seconds=shared_budget_available(args.output_dir, args.budget_root, elapsed),
                              retry_available_seconds=shared_budget_available(args.output_dir, args.budget_root, elapsed, retry=True))))
    else:
        print(json.dumps(summarize(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
