"""Bounded, matched RING comparison charged to the original pilot budget.

Four timing smokes precede eight Only-GVP fits. A separate eight-fit late-fusion
block is admitted only when its full measured forecast fits. No test is opened.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import shlex
import sys
import time

import run_metal_architecture_pilot as base

PROFILE = "metal_ring_pilot_v1"
BUDGET_PROFILE = "metal_ring_continuation_budget_v1"
FAMILIES = {"GVP": "Only-GVP", "LATE": "GVP + late fusion"}
BLOCKS = ("S", "GVP", "LATE")
require, read, save = base.require, base.read, base.save


def validate(config, ring, family, epochs=50):
    require(config.use_ring_edges == ring and config.require_ring_edges == ring
            and not config.prepare_missing_ring_edges, "Wrong RING controls")
    require(config.shell_role_source == "geometry", "Freeze shell roles geometrically in both arms")
    require(config.site_geometry_features == "legacy" and config.metal_node_mode == "none",
            "RING comparison retains legacy site inputs without metal nodes")
    require(config.metal_label_scheme == "merge_fe_class_viii" and config.device == "cuda",
            "Direct four-class CUDA training is required")
    if family == "Only-GVP":
        require(config.model_architecture == "only_gvp" and not config.use_esm_branch
                and not config.use_early_esm, "Unexpected model family")
    else:
        require(family == FAMILIES["LATE"] and config.model_architecture == "gvp"
                and not config.use_early_esm and config.use_esm_branch
                and config.fusion_mode == "late_fusion", "Unexpected model family")
    base.validate(replace(config, use_ring_edges=False, require_ring_edges=False), epochs)


def budget_handoff(budget_root):
    handoff = read(Path(budget_root) / "budget_handoff.json")
    require(handoff and handoff["profile"] == BUDGET_PROFILE, "Missing original-budget handoff")
    for name, expected in (("total_cap_seconds", 36000), ("training_cap_seconds", 34200),
                           ("main_cap_seconds", 27000)):
        require(handoff[name] == expected, f"Original budget changed: {name}")
    intervals = handoff["prior_intervals"]
    require(len(intervals) == 2 and all(x["ended_epoch"] >= x["started_epoch"] for x in intervals),
            "Require the two closed previous allocation intervals")
    require(math.isclose(sum(x["ended_epoch"] - x["started_epoch"] for x in intervals),
                         handoff["prior_allocated_seconds"], abs_tol=1e-6), "Incorrect cumulative allocation")
    require(all(math.isfinite(handoff[k]) and handoff[k] >= 0 for k in
                ("prior_main_seconds", "prior_retry_seconds")), "Invalid previous attempt costs")
    return handoff


def plan(root, data, output, source_commit, parent_reference_dir,
         external_features_root_dir=None, feature_overlay_manifest=None, budget_root=None):
    root, data, output, parent = map(lambda p: Path(p).resolve(), (root, data, output, parent_reference_dir))
    require(budget_root is not None, "Supply the original-budget continuation directory")
    budget_root = Path(budget_root).resolve()
    handoff = budget_handoff(budget_root)
    original = read(parent / "campaign_manifest.json")
    require(original and original["profile"] == base.PROFILE and original["dataset"] == base.DATASET
            and original["bundle_sha256"] == base.BUNDLE_SHA, "Wrong parent scientific dataset")
    reference = read(parent / "expected_split.json")
    require(reference is not None, "Require the certified parent retained split")
    cache_proofs = ("expected_ring_files.json", "expected_feature_files.json", "expected_cache_inventory_provenance.json")
    require(all((parent / name).is_file() for name in cache_proofs), "Require pinned RING/ESM/external feature inventories")
    provenance = read(parent / cache_proofs[-1])
    require(provenance["v12_bundle_sha256"] == base.BUNDLE_SHA
            and provenance["expected_ring_files_sha256"] == base.digest(parent / cache_proofs[0])
            and provenance["expected_feature_files_sha256"] == base.digest(parent / cache_proofs[1]),
            "Feature inventories differ from their pinned provenance")
    external = Path(external_features_root_dir or original["external_features_root_dir"]).resolve()
    overlay = Path(feature_overlay_manifest or original["feature_overlay_manifest"]["path"]).resolve()
    base.validate_feature_overlay(data, external, overlay)
    templates = base._templates(root, data, output, external)
    runs = []
    for block in BLOCKS:
        for key, family in FAMILIES.items():
            if block != "S" and block != key:
                continue
            for lr in (base.LRS[:1] if block == "S" else base.LRS):
                for seed in ((42,) if block == "S" else (42, 43)):
                    for ring in (False, True):
                        ident = f"{block}_{key}_ring{int(ring)}_lr{lr:g}_s{seed}"
                        template = templates[family]
                        command = [str(x) for x in template["command"] if str(x) not in
                                   ("--use-ring-edges", "--require-ring-edges", "--prepare-missing-ring-edges")]
                        command[0] = sys.executable
                        if "--no-prepare-missing-ring-edges" not in command:
                            command.append("--no-prepare-missing-ring-edges")
                        if ring:
                            command.extend(("--use-ring-edges", "--require-ring-edges"))
                        epochs = 1 if block == "S" else 50
                        for flag, value in (("--shell-role-source", "geometry"), ("--site-geometry-features", "legacy"),
                                            ("--metal-node-mode", "none"), ("--structural-readout-scope", "residue_only"),
                                            ("--metal-label-scheme", "four_class"), ("--selection-metric", base.METRIC),
                                            ("--ring-features-dir", data / "RING_features"), ("--edge-radius", 6),
                                            ("--classifier-pool-distance-cutoff", 0), ("--epochs", epochs),
                                            ("--learning-rate", lr), ("--seed", seed), ("--run-name", ident),
                                            ("--runs-dir", output / "runs")):
                            command = base.replace_option(command, flag, value)
                        config = base.parse_config(command)
                        validate(config, ring, family, epochs)
                        payload = json.loads(json.dumps(asdict(config), default=str))
                        runs.append(dict(id=ident, block=block, family=family, ring=ring, scheme="four_class",
                                         lr=lr, seed=seed, epochs=epochs, command=command,
                                         env={**template.get("env", {}), "DEEPGM_METAL_LABEL_SCHEME": "four_class"},
                                         run_dir=str(output / "runs" / ident), config=payload,
                                         config_sha256=base.fingerprint(payload)))
    source = sorted((root / "src").rglob("*.py")) + [root / "notebooks/DeepMzyme_training_colab.ipynb",
                                                                  root / "docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md"]
    train = data / base.DATASET / "train"
    dataset_files = [train / "structure_manifest.csv", train / "final_data_summarazing_table_transition_metals_only_catalytic.csv"]
    manifest = dict(profile=PROFILE, source_commit=source_commit, dataset=base.DATASET,
                    bundle=base.BUNDLE, bundle_sha256=base.BUNDLE_SHA, data_root=str(data), output_dir=str(output),
                    parent_reference_dir=str(parent), parent_manifest_sha256=base.digest(parent / "campaign_manifest.json"),
                    expected_split_sha256=base.digest(parent / "expected_split.json"),
                    feature_inventory_files={name: base.digest(parent / name) for name in cache_proofs},
                    cohort_sha256=base.fingerprint(base.cohort_identity(reference)), budget_root=str(budget_root),
                    budget_handoff_sha256=base.digest(budget_root / "budget_handoff.json"),
                    external_features_root_dir=str(external),
                    feature_overlay_manifest=dict(path=str(overlay), sha256=base.digest(overlay)),
                    source_files={str(p.relative_to(root)): base.digest(p) for p in source},
                    dataset_files={str(p.relative_to(data)): base.digest(p) for p in dataset_files},
                    runs=runs, block_order=list(BLOCKS), maximum_smokes=4, maximum_full_runs=16,
                    required_full_runs=8, held_out_evaluation=False, selection_metric=base.METRIC,
                    shell_role_source="geometry", normalization="training_fitted_per_arm",
                    estimand="RING edges and interaction features with their training-fitted edge normalization",
                    budget_policy=dict(prior_allocated_seconds=handoff["prior_allocated_seconds"],
                                       admission_factor=1.25, setup_cap_seconds=1800, readiness_cap_seconds=1200,
                                       optional_block="LATE", reset_original_budget=False))
    old = read(output / "campaign_manifest.json")
    require(old is None or old == manifest, "Frozen RING manifest changed; use a new identity")
    save(output / "campaign_manifest.json", manifest)
    (output / "expected_split.json").write_bytes((parent / "expected_split.json").read_bytes())
    (output / "commands.txt").write_text("\n\n".join(shlex.join(r["command"]) for r in runs) + "\n")
    base.csv_save(output / "run_matrix.csv", runs, ["id", "block", "family", "ring", "lr", "seed", "epochs", "run_dir"])
    return manifest


def verify_manifest(root, output):
    root, output = Path(root), Path(output)
    manifest = read(output / "campaign_manifest.json")
    require(manifest and manifest["profile"] == PROFILE, "Missing RING manifest")
    for name, checksum in manifest["source_files"].items():
        require(base.digest(root / name) == checksum, f"Source changed after freeze: {name}")
    for name, checksum in manifest["dataset_files"].items():
        require(base.digest(Path(manifest["data_root"]) / name) == checksum, f"Data changed: {name}")
    require(base.digest(Path(manifest["parent_reference_dir"]) / "campaign_manifest.json") ==
            manifest["parent_manifest_sha256"], "Parent manifest changed")
    require(base.digest(output / "expected_split.json") == manifest["expected_split_sha256"], "Cohort changed")
    for name, checksum in manifest["feature_inventory_files"].items():
        require(base.digest(Path(manifest["parent_reference_dir"]) / name) == checksum, "Pinned feature inventory changed")
    require(base.digest(Path(manifest["budget_root"]) / "budget_handoff.json") ==
            manifest["budget_handoff_sha256"], "Original-budget handoff changed")
    overlay = manifest["feature_overlay_manifest"]
    require(base.digest(overlay["path"]) == overlay["sha256"], "Feature-overlay manifest changed")
    base.validate_feature_overlay(manifest["data_root"], manifest["external_features_root_dir"], overlay["path"])
    return manifest


def available_seconds(output, budget_root, elapsed, *, retry=False):
    handoff = budget_handoff(budget_root)
    attempts = base._attempts(output)
    used = sum(float(a.get("elapsed_seconds") or 0) for a in attempts if bool(a.get("retry_of")) == retry)
    if retry:
        remaining = 3600 - handoff["prior_retry_seconds"] - used
    else:
        bootstrap = read(Path(budget_root) / "bootstrap_budget_usage.json", {})
        setup = float(bootstrap.get("normal_elapsed_seconds", 0))
        require(math.isfinite(setup) and setup >= 0, "Invalid bootstrap cost")
        remaining = handoff["main_cap_seconds"] - handoff["prior_main_seconds"] - setup - used
    return max(0., min(handoff["training_cap_seconds"] - elapsed, remaining))


def _context(root, output, budget_root, started, receipt):
    root, output, budget_root = map(lambda p: Path(p).resolve(), (root, output, budget_root))
    manifest = verify_manifest(root, output)
    require(budget_root == Path(manifest["budget_root"]), "Wrong cumulative budget directory")
    handoff = budget_handoff(budget_root)
    ledger = read(budget_root / "allocation_ledger.json")
    require(ledger and ledger["intervals"][:2] == handoff["prior_intervals"], "Original closed intervals changed")
    storage = base.storage_check(output, receipt)
    elapsed = base.allocation_elapsed(budget_root, started)
    attempts = base._attempts(output)
    base._recover_interrupted(output, attempts)
    base._require_transfer(output, attempts, storage)
    return root, output, budget_root, manifest, storage, elapsed, attempts


def rows(output, manifest):
    results = base.completed_rows(output, manifest)
    planned = {r["id"]: r for r in manifest["runs"]}
    for row in results:
        row["ring"] = planned[row["id"]]["ring"]
        stats = read(Path(row["run_dir"]) / "run_metadata.json").get("normalization_stats")
        require(stats is not None, "Missing fitted normalization metadata")
        row["normalization_stats_sha256"] = base.fingerprint(stats)
    return results


def forecast(run, attempts, manifest):
    planned = {r["id"]: r for r in manifest["runs"]}
    measured = [a["setup_seconds"] + a["epoch_seconds"] * run["epochs"] for a in attempts
                if a["status"] == "completed" and a["run_id"] in planned and "epoch_seconds" in a
                and all(planned[a["run_id"]][k] == run[k] for k in ("family", "ring"))]
    require(measured, "Both RING modes in each family require successful timing smokes")
    return max(measured)


def preflight(root, output, allocation_started_epoch, persistence_receipt=None, budget_root=None):
    require(budget_root is not None, "Supply cumulative budget directory")
    root, output, budget_root, manifest, _, elapsed, attempts = _context(
        root, output, budget_root, allocation_started_epoch, persistence_receipt)
    ready = read(output / "readiness.json", {})
    if ready.get("status") == "passed":
        require(ready["manifest_sha256"] == base.digest(output / "campaign_manifest.json"), "Readiness identity changed")
        base.verify_training_cache(output)
        return ready
    retry = any(a["run_id"] == "ring_readiness" for a in attempts)
    available = min(manifest["budget_policy"]["readiness_cap_seconds"],
                    available_seconds(output, budget_root, elapsed, retry=retry))
    run = dict(id="ring_readiness", block="P", run_dir=str(output / "ring_preflight"))
    command = [sys.executable, str(root / "src/run_metal_ring_pilot.py"), "_preflight", "--output-dir", str(output)]
    try:
        audit_env = {"DEEPGM_METAL_LABEL_SCHEME": "four_class", "OMP_NUM_THREADS": "1",
                     "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        base._record_attempt(root, output, run, command, audit_env, available, attempts)
    finally:
        base.allocation_elapsed(budget_root, allocation_started_epoch)
    return read(output / "readiness.json")


def _preflight_worker(root, output, run_name="ring_preflight"):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from audit_metal_ring_inputs import audit
    from training.run import prepare_run, set_seed
    manifest = verify_manifest(root, output)
    receipt = read(Path("/content/v12_ready.json"), {})
    require(receipt.get("sha256") == base.BUNDLE_SHA and torch.cuda.is_available(), "Verified v12 and CUDA required")
    configs = []
    for ring in (False, True):
        run = next(r for r in manifest["runs"] if r["block"] == "S" and r["family"] == "Only-GVP" and r["ring"] == ring)
        config = base.parse_config(run["command"])
        validate(config, ring, run["family"], 1)
        configs.append(replace(config, device="cpu", esm_embeddings_dir=None, require_esm_embeddings=False,
                               runs_dir=output, run_name=f"{run_name}_ring{int(ring)}"))
    parent = Path(manifest["parent_reference_dir"])
    audit(*configs, read(output / "expected_split.json"), output,
          expected_ring_files=read(parent / "expected_ring_files.json"),
          expected_feature_files=read(parent / "expected_feature_files.json"))
    for ring in (False, True):
        run = next(r for r in manifest["runs"] if r["block"] == "S" and r["family"] == FAMILIES["LATE"] and r["ring"] == ring)
        config = replace(base.parse_config(run["command"]), runs_dir=output, run_name=f"{run_name}_late_ring{int(ring)}")
        set_seed(config.seed, deterministic=config.deterministic)
        prepared = prepare_run(config)
        require(base.fingerprint(base.cohort_identity(prepared.dataset_summary)) == manifest["cohort_sha256"], "Late cohort changed")
        result = prepared.model(next(iter(prepared.train_loader)).to(config.device))
        require(torch.isfinite(result["loss"]), "Nonfinite actual-data loss")
        result["loss"].backward()
        torch.cuda.synchronize()
        del prepared, result
        torch.cuda.empty_cache()
    base.verify_training_cache(output)
    save(output / "readiness.json", dict(status="passed", manifest_sha256=base.digest(output / "campaign_manifest.json"),
         cohort_sha256=manifest["cohort_sha256"], gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
         cuda=torch.version.cuda, audit_cpu_threads=torch.get_num_threads(),
         audit_interop_threads=torch.get_num_interop_threads(),
         actual_data_forward_backward_both_ring_modes=True, held_out_evaluation=False))


def execute(root, output, allocation_started_epoch, persistence_receipt=None, budget_root=None, max_runs=1):
    require(max_runs == 1 and budget_root is not None, "One attempt at a time, charged to original budget")
    root, output, budget_root, manifest, storage, elapsed, attempts = _context(
        root, output, budget_root, allocation_started_epoch, persistence_receipt)
    ready = read(output / "readiness.json", {})
    require(ready.get("status") == "passed" and ready["manifest_sha256"] == base.digest(output / "campaign_manifest.json"),
            "Complete source-specific preflight first")
    base.verify_training_cache(output)
    results = rows(output, manifest)
    completed = {r["id"] for r in results}
    pending = [r for r in manifest["runs"] if r["id"] not in completed]
    if not pending:
        save(output / "campaign_state.json", dict(status="completed", allocated_seconds=elapsed))
        return summarize(output)
    run = pending[0]
    retry = any(a["run_id"] == run["id"] for a in attempts)
    available = available_seconds(output, budget_root, elapsed, retry=retry)
    if run["block"] != "S":
        require(sum(r["block"] == "S" for r in results) == 4, "All four timing smokes must pass")
        remaining = [r for r in pending if r["block"] == run["block"]]
        projected = sum(forecast(r, attempts, manifest) for r in remaining) * 1.25
        normal_available = available_seconds(output, budget_root, elapsed)
        # A retry is charged separately, but admission remains conservative.
        if projected > normal_available or available <= 20:
            save(output / "campaign_state.json", dict(status="budget_stopped", next_run_id=run["id"],
                 block=run["block"], forecast_seconds=projected, available_seconds=normal_available,
                 reason="Remaining full matched family block does not fit forecast plus 25% margin", allocated_seconds=elapsed))
            return summarize(output)
        admission = output / f"{run['block'].lower()}_admission.json"
        if not admission.exists():
            save(admission, dict(block=run["block"], full_run_count=8, forecast_seconds=projected,
                                 available_seconds=normal_available, admitted_epoch=time.time(), margin=1.25))
    elif available <= 20:
        save(output / "campaign_state.json", dict(status="budget_stopped", allocated_seconds=elapsed))
        return summarize(output)
    validate(base.parse_config(run["command"]), run["ring"], run["family"], run["epochs"])
    save(output / "campaign_state.json", dict(status="running", run_id=run["id"], block=run["block"], allocated_seconds=elapsed))
    try:
        attempt = base._record_attempt(root, output, run, run["command"], run["env"], available, attempts)
    except BaseException:
        save(output / "campaign_state.json", dict(status="failed", run_id=run["id"], block=run["block"]))
        raise
    finally:
        base.allocation_elapsed(budget_root, allocation_started_epoch)
    save(output / "campaign_state.json", dict(status="awaiting_archive_transfer" if storage["method"] == "verified_archive_transfer"
         else "step_completed", run_id=run["id"], attempt_id=attempt["attempt_id"], attempt_status=attempt["status"]))
    return summarize(output)


def summarize(output):
    output = Path(output)
    manifest = read(output / "campaign_manifest.json")
    results = rows(output, manifest)
    coverage = {b: dict(expected_runs=4 if b == "S" else 8, completed_runs=sum(r["block"] == b for r in results),
                       complete=sum(r["block"] == b for r in results) == (4 if b == "S" else 8)) for b in BLOCKS}
    full = [r for r in results if r["block"] != "S"]
    pairs = []
    for key, family in FAMILIES.items():
        if not coverage[key]["complete"]:
            continue
        for lr in base.LRS:
            for seed in (42, 43):
                matched = {r["ring"]: r for r in full if r["family"] == family and r["lr"] == lr and r["seed"] == seed}
                pairs.append(dict(family=family, lr=lr, seed=seed,
                                  delta_balanced_accuracy=matched[True]["balanced_accuracy"] - matched[False]["balanced_accuracy"],
                                  ring_off_minimum_recall=matched[False]["minimum_recall"],
                                  ring_on_minimum_recall=matched[True]["minimum_recall"],
                                  run_ids=[matched[x]["id"] for x in (False, True)]))
    summary = dict(profile=PROFILE, completed_smokes=len(results)-len(full), completed_full_runs=len(full),
                   coverage=coverage, paired_fixed_recipe_deltas=pairs, state=read(output / "campaign_state.json", {"status": "planned"}),
                   estimand=manifest["estimand"], promoted=False, held_out_evaluation=False,
                   confidence_intervals_computed=False, independent_validation_folds=1, runs=full)
    save(output / "ring_validation_results.json", summary)
    save(output / "ring_coverage.json", coverage)
    base.csv_save(output / "ring_screen.csv", full, ["id", "block", "family", "ring", "lr", "seed", "selected_epoch",
                  "balanced_accuracy", "minimum_recall", "per_class_recall", "elapsed_seconds", "normalization_stats_sha256", "run_dir"])
    (output / "ring_decision_record.md").write_text(
        f"# {PROFILE}\n\nSmokes: {summary['completed_smokes']}/4. Full fits: {len(full)}/16 maximum.\n\n"
        "Fresh direct-four-class controls share a single PDB-grouped development split, two learning rates and two seeds. "
        "Only-GVP precedes an optional complete graph-level late-fusion block. Shell roles are geometry-derived in both arms. "
        "RING changes edges/interaction features and their training-fitted edge normalization. "
        "It does not introduce explicit metal nodes or consume RING's raw Angle column.\n\n"
        "Only complete family blocks support paired summaries. These are descriptive fixed-split results; "
        "they do not establish general superiority, grouped-fold promotion, or performance on held-out test data.\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "preflight", "execute", "summarize", "_preflight", "budget-status"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--parent-reference-dir", type=Path)
    parser.add_argument("--external-features-root-dir", type=Path)
    parser.add_argument("--feature-overlay-manifest", type=Path)
    parser.add_argument("--budget-root", type=Path)
    parser.add_argument("--allocation-started-epoch", type=float)
    parser.add_argument("--persistence-receipt", type=Path)
    parser.add_argument("--max-runs", type=int, default=1)
    parser.add_argument("--preflight-run-name", default="ring_preflight", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.phase == "plan":
        require(args.data_root and args.source_commit and args.parent_reference_dir and args.budget_root,
                "Plan requires data, source identity, parent reference, and budget handoff")
        result = plan(root, args.data_root, args.output_dir, args.source_commit, args.parent_reference_dir,
                      args.external_features_root_dir, args.feature_overlay_manifest, args.budget_root)
        print(f"Saved {len(result['runs'])} runs: four smokes, eight GVP fits, optional eight late-fusion fits")
    elif args.phase in ("preflight", "execute"):
        require(args.allocation_started_epoch is not None and args.budget_root, "Supply actual start and cumulative budget")
        operation = preflight if args.phase == "preflight" else execute
        kwargs = {"max_runs": args.max_runs} if args.phase == "execute" else {}
        result = operation(root, args.output_dir, args.allocation_started_epoch, args.persistence_receipt, args.budget_root, **kwargs)
        print(json.dumps(result, indent=2))
    elif args.phase == "_preflight":
        _preflight_worker(root, args.output_dir, args.preflight_run_name)
    elif args.phase == "budget-status":
        require(args.allocation_started_epoch is not None and args.budget_root, "Supply actual start and cumulative budget")
        elapsed = base.allocation_elapsed(args.budget_root, args.allocation_started_epoch)
        print(json.dumps(dict(allocated_seconds=elapsed, normal_available_seconds=available_seconds(args.output_dir, args.budget_root, elapsed),
                              retry_available_seconds=available_seconds(args.output_dir, args.budget_root, elapsed, retry=True))))
    else:
        print(json.dumps(summarize(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
