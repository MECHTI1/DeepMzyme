"""Campaign-specific validation-to-refit-to-reference bridge.

Development never enters the reference-data branch. The reference route is a
secondary, possibly overlapping known-ion comparison, not the primary final test.
"""
from __future__ import annotations

import csv
import json
import shlex
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from benchmarking import pmm_ion_campaign as campaign
from training.source_cohort import read_cohort_csv, sha256_file

ROUTE = "zenodo_pmm_secondary_reference"
CONTROL = "only_gvp__four_class__none"
FINAL_EPOCHS = 50
FINAL_SEED = 42
TIE_EPSILON = 0.002
REFERENCE_SOURCE_SHA256 = "ec6427ad0b8a18261dbc1d6822bdcfe7194730c7da4a36c208957bcb00f6a0f2"


class FinalReportError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalReportError(message)


def select_configuration(collected: dict[str, Any], contrasts: list[dict]) -> dict:
    """Select only tested configurations passing the predeclared gates.

The six scientific contrasts retain their own multiplicity rule. Comparisons
against the fixed control here are selection diagnostics, not extra superiority
claims. A tie-break is applied only among candidates that are already eligible.
"""
    from benchmarking.pmm_ion_analysis import COMMON_FOUR_LABELS, fold_bootstrap, mean_class_recalls

    expected = {config.config_id for config in campaign.grid_configs()}
    require(set(collected) == expected, "Final selection requires the exact nine-configuration grid")
    require(all(entry["complete"] and set(entry["folds"]) == set(range(5)) for entry in collected.values()),
            "Final selection requires every planned fold")
    matched = {row["challenger"]: row for row in contrasts}
    control = collected[CONTROL]
    baseline_recalls = mean_class_recalls(control["folds"])
    rows = []
    for config_id, entry in collected.items():
        config = entry["config"]
        bas = [float(entry["folds"][k]["metrics"]["common4"]["balanced_accuracy"]) for k in range(5)]
        recalls = mean_class_recalls(entry["folds"])
        minimum = [min(entry["folds"][k]["metrics"]["common4"]["recall"].values()) for k in range(5)]
        differences = np.asarray([bas[k] - control["folds"][k]["metrics"]["common4"]["balanced_accuracy"]
                                  for k in range(5)])
        interval = fold_bootstrap(differences)
        matched_gate = (config.target == "four_class" and config.readout == "none") or bool(
            matched.get(config_id, {}).get("promoted"))
        recall_gate = all(recalls[label] is not None and recalls[label] > 0
                          and baseline_recalls[label] is not None
                          and baseline_recalls[label] - recalls[label] <= 0.03 for label in COMMON_FOUR_LABELS)
        control_gate = interval["mean_difference"] > 0 and interval["ci95_low"] > 0 and recall_gate
        rows.append({"config_id": config_id, "family": config.family, "target": config.target,
                     "readout": config.readout, "mean_common4_ba": float(np.mean(bas)),
                     "mean_min_recall": float(np.mean(minimum)), "worst_fold_common4_ba": min(bas),
                     "sd_common4_ba": float(np.std(bas, ddof=1)),
                     "complexity_proxy": {"only_esm": 0, "only_gvp": 1, "gvp_late_fusion": 2}[config.family],
                     "matched_promotion_passed": matched_gate,
                     "control_ci_low": interval["ci95_low"], "control_mean_difference": interval["mean_difference"],
                     "control_recall_gate": recall_gate,
                     "eligible": config_id == CONTROL or (matched_gate and control_gate)})
    eligible = [row for row in rows if row["eligible"]]
    challengers = [row for row in eligible if row["config_id"] != CONTROL]
    if challengers:
        largest = max(row["mean_common4_ba"] for row in challengers)
        tied = [row for row in challengers if largest - row["mean_common4_ba"] <= TIE_EPSILON]
        chosen = min(tied, key=lambda row: (-row["mean_min_recall"], -row["worst_fold_common4_ba"],
                                           row["sd_common4_ba"], row["complexity_proxy"], row["config_id"]))
        reason = "passed_matched_and_control_gates" if len(tied) == 1 else "eligible_candidate_tie_break"
    else:
        chosen = next(row for row in rows if row["config_id"] == CONTROL)
        reason = "retained_predeclared_control_no_supported_replacement"
    return {"selected_config_id": chosen["config_id"], "selection_reason": reason, "control": CONTROL,
            "ranked_candidates": sorted(rows, key=lambda row: (-row["mean_common4_ba"], row["config_id"])),
            "rank_metric": "mean_common4_balanced_accuracy", "tie_epsilon": TIE_EPSILON,
            "tie_breakers": ["mean_min_recall_desc", "worst_fold_common4_ba_desc", "sd_common4_ba_asc",
                             "complexity_proxy_asc", "config_id_asc"],
            "evidence_design": "one-seed fivefold grouped comparison; secondary reference selection"}


def _set_argument(argv: list[str], flag: str, value: str | None = None, *, remove: bool = False) -> None:
    while flag in argv:
        index = argv.index(flag)
        del argv[index]
        if index < len(argv) and not argv[index].startswith("--"):
            del argv[index]
    if not remove:
        argv.append(flag)
        if value is not None:
            argv.append(value)


def build_refit_command(paths: campaign.CampaignPaths, train_dir: Path, config_id: str, *,
                        python_bin: str, device: str, load_workers: int | None = None) -> tuple[list[str], dict, dict]:
    from training.config import config_to_payload, parse_args

    config = next((config for config in campaign.grid_configs() if config.config_id == config_id), None)
    require(config is not None, "Refit configuration was not in the evaluated grid")
    command, env, _ = campaign.build_train_command(paths, python_bin=python_bin, train_dir=train_dir,
                                                  config=config, fold=0, seed=FINAL_SEED, device=device,
                                                  runs_dir=paths.root / "final_refit", load_workers=load_workers)
    for flag in ("--n-folds", "--fold-index", "--fold-membership-csv", "--fold-membership-sha256",
                 "--export-validation-predictions", "--campaign-run-identity"):
        _set_argument(command, flag, remove=True)
    _set_argument(command, "--val-fraction", "0.0")
    _set_argument(command, "--selection-metric", "train_loss")
    _set_argument(command, "--run-name", f"stage6b__{config_id}__seed42")
    bindings = read_cohort_csv(paths.cohort, campaign.load_campaign(paths)["cohort"]["sha256"])
    counts = Counter(campaign.COMMON_FOUR[b.native_element] for b in bindings)
    require(len(counts) == 4, "Full-training refit lacks an endpoint class")
    weights = {label: len(bindings) / (4 * count) for label, count in counts.items()}
    values = {"mn": weights["Mn"], "cu": weights["Cu"], "zn": weights["Zn"]}
    values.update({key: weights["Class VIII"] for key in
                   (("class-viii",) if config.target == "four_class" else ("fe", "co", "ni"))})
    for key in ("mn", "cu", "zn", "fe", "co", "ni", "class-viii"):
        _set_argument(command, f"--{key}-loss-multiplier", remove=True)
    for key, weight in values.items():
        _set_argument(command, f"--{key}-loss-multiplier", repr(weight))
    resolved = config_to_payload(parse_args(command[3:]))
    identity = {"campaign_id": campaign.load_campaign(paths)["campaign_id"], "phase": "stage6b_full_train",
                "selected_config_id": config_id, "cohort_sha256": sha256_file(paths.cohort),
                "feature_inventory_sha256": campaign.inventory_identity_sha256(paths),
                "source_tree_sha256": campaign.source_tree_sha256(), "model_seed": FINAL_SEED,
                "epochs": FINAL_EPOCHS, "checkpoint_rule": "terminal_epoch_50",
                "common_four_weights": weights,
                "resolved_config_sha256": campaign.stable_hash({key: value for key, value in resolved.items()
                                                                 if key not in campaign.NON_IDENTITY_CONFIG_KEYS})}
    command += ["--campaign-run-identity", json.dumps(identity, sort_keys=True)]
    return command, env, identity


def preview_refit(paths: campaign.CampaignPaths, train_dir: Path, *, python_bin: str, device: str,
                  load_workers: int | None = None) -> dict:
    from benchmarking.pmm_ion_analysis import collect_units
    from benchmarking.pmm_comparator import verify_comparator_outputs

    decision_path = paths.root / "validation_decision.json"
    campaign.campaign_manifest_guard(paths, require_context=True)
    decision = campaign.read_json(decision_path)
    require(decision.get("status") == "complete" and not decision.get("replay_reconciliation_mismatches"),
            "Stage 6B needs complete reconciled development results")
    verify_comparator_outputs(paths.root)
    require(decision["code_tree_sha256"] == campaign.source_tree_sha256(),
            "Source changed since the completed validation campaign")
    selected = select_configuration(collect_units(paths, train_dir), decision["predeclared_contrasts"])
    command, env, identity = build_refit_command(paths, train_dir, selected["selected_config_id"],
                                                 python_bin=python_bin, device=device, load_workers=load_workers)
    preview = {**selected, "status": "selected_for_final_refit", "launch_final_refit": False,
               "route": ROUTE, "validation_decision_sha256": sha256_file(decision_path),
               "identity": identity, "command": command, "environment": env,
               "reference_policy": {"secondary_only": True, "possibly_overlapping": True,
                                    "no_calibration": True, "no_ensemble": True, "one_model_per_system": True,
                                    "bootstrap_resamples": 10000, "bootstrap_seed": 42,
                                    "bootstrap_unit": "PDB/alias group"}}
    existing = paths.root / "stage6b_decision.json"
    if existing.exists():
        old = campaign.read_json(existing)
        require(old["identity"] == identity and old["validation_decision_sha256"] == preview["validation_decision_sha256"],
                "Existing final selection differs; refusing to replace its frozen protocol")
        return old
    campaign.write_json(existing, preview)
    ranked_path = paths.root / "stage6b_ranked_candidates.csv"
    with ranked_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected["ranked_candidates"][0]))
        writer.writeheader()
        writer.writerows(selected["ranked_candidates"])
    (paths.root / "stage6b_final_refit_command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    return preview


def verify_final_refit(paths: campaign.CampaignPaths) -> dict:
    receipt = campaign.read_json(paths.root / "stage6b_selected_final_refit_candidate.json")
    decision = campaign.read_json(paths.root / "stage6b_decision.json")
    require(receipt["identity"] == decision["identity"], "Final refit identity differs from selected protocol")
    checkpoint_path = paths.root / receipt["checkpoint_path"]
    require(checkpoint_path.resolve().is_relative_to((paths.root / "final_refit").resolve()),
            "Final checkpoint path escapes the final-refit directory")
    require(receipt.get("status") == "completed" and receipt.get("held_out_test_accessed") is False,
            "Final refit receipt is incomplete or includes held-out access")
    require(sha256_file(checkpoint_path) == receipt["checkpoint_sha256"], "Final checkpoint hash differs")
    for name in ("run_metadata", "run_config"):
        require(sha256_file(checkpoint_path.parent / f"{name}.json") == receipt[f"{name}_sha256"],
                f"Final refit {name} changed after certification")
    require(receipt["completed_epochs"] == FINAL_EPOCHS and receipt["terminal_checkpoint"] is True,
            "Final source is not the completed terminal epoch-50 checkpoint")
    return receipt


def verify_pmm_refit(paths: campaign.CampaignPaths) -> dict:
    from benchmarking.pmm_comparator import verify_refit

    directory = paths.root / "final_refit" / "pmm"
    receipt = verify_refit(paths.root, directory)
    require(receipt.get("status") == "completed", "PMM full-training refit is not complete")
    require(receipt["cohort_sha256"] == sha256_file(paths.cohort), "PMM refit used a different training cohort")
    model_path = directory / receipt["model_path"]
    require(model_path.resolve().is_relative_to(directory.resolve()), "PMM model path escapes its frozen directory")
    require(sha256_file(model_path) == receipt["model_sha256"], "PMM refit model hash differs")
    return receipt


def require_reference_authorization(paths: campaign.CampaignPaths, source_dir: Path, route_path: Path) -> dict:
    """Validate train-side gates before any caller inspects reference inputs."""
    route = campaign.read_json(route_path)
    require(route.get("route") == ROUTE and route.get("selection_frozen") is True,
            "A frozen secondary reference route is required")
    parent = campaign.CampaignPaths(Path(route["campaign_dir"]))
    decision = campaign.read_json(parent.root / "stage6b_decision.json")
    require(sha256_file(parent.root / "stage6b_decision.json") == route["stage6b_decision_sha256"],
            "Final selection changed after reference route freeze")
    require(decision["validation_decision_sha256"] == sha256_file(parent.root / "validation_decision.json"),
            "Validation decision changed after final selection")
    require(decision["identity"]["source_tree_sha256"] == campaign.source_tree_sha256(),
            "Scientific source changed after validation/refit")
    neural, pmm = verify_final_refit(parent), verify_pmm_refit(parent)
    require(neural["checkpoint_sha256"] == route["deepmzyme_refit_checkpoint_sha256"]
            and pmm["model_sha256"] == route["pmm_refit_model_sha256"], "Final refit differs from the frozen route")
    require(route["campaign_cohort_sha256"] == sha256_file(parent.cohort), "Training cohort changed")
    require(route["reference_source_sha256"] == REFERENCE_SOURCE_SHA256, "Reference source is not the pinned release")
    require(Path(source_dir).resolve() == Path(route["reference_dir"]).resolve()
            and Path(source_dir).name == "test", "Only the declared original reference side is permitted")
    require(paths.root.resolve() == (parent.root / "reference" / "inputs_profile").resolve(),
            "Reference inputs must stay in their separately named profile")
    return route


def finish_refit_receipt(paths: campaign.CampaignPaths, run_dir: Path, identity: dict) -> dict:
    import torch

    metadata = campaign.read_json(run_dir / "run_metadata.json")
    config = campaign.read_json(run_dir / "run_config.json")
    require(metadata.get("fit_status") == "completed" and metadata.get("test_report") is None,
            "Final refit incomplete or evaluated test during training")
    require(metadata.get("campaign_run_identity") == identity, "Final refit configuration differs")
    require([row["epoch"] for row in config["history"]] == list(range(1, FINAL_EPOCHS + 1)),
            "Final refit must complete all 50 epochs")
    checkpoint_path = run_dir / "last_model_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(len(checkpoint["history"]) == FINAL_EPOCHS, "Terminal checkpoint history is incomplete")
    require(checkpoint["config"]["val_fraction"] == 0.0 and checkpoint["config"]["n_folds"] is None,
            "Refit checkpoint was trained with a validation partition")
    require(json.loads(checkpoint["config"]["campaign_run_identity"]) == identity, "Checkpoint identity differs")
    receipt = {"status": "completed", "identity": identity,
               "checkpoint_path": str(checkpoint_path.relative_to(paths.root)),
               "checkpoint_sha256": sha256_file(checkpoint_path), "completed_epochs": FINAL_EPOCHS,
               "terminal_checkpoint": True, "run_metadata_sha256": sha256_file(run_dir / "run_metadata.json"),
               "run_config_sha256": sha256_file(run_dir / "run_config.json"), "held_out_test_accessed": False}
    campaign.write_json(paths.root / "stage6b_selected_final_refit_candidate.json", receipt)
    return receipt


def execute_refit(paths: campaign.CampaignPaths, train_dir: Path, *, python_bin: str, pmm_python: str,
                  device: str, execution, estimated_seconds: float, load_workers: int | None = None) -> dict:
    import os

    preview = preview_refit(paths, train_dir, python_bin=python_bin, device=device, load_workers=load_workers)
    receipt_path = paths.root / "stage6b_selected_final_refit_candidate.json"
    if receipt_path.exists():
        receipt = verify_final_refit(paths)
    else:
        command = preview["command"]
        name = command[command.index("--run-name") + 1]
        run_dir = paths.root / "final_refit" / name
        require(not run_dir.exists(), "Incomplete final refit exists; preserve and diagnose it before a linked restart")
        execution.admit(name, estimated_seconds)
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        log_path = run_dir.parent / f"{name}.log"
        started = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            result = execution.run_subprocess(command, cwd=campaign.REPO_ROOT,
                                    env={**os.environ, **preview["environment"]}, stdout=log, stderr=subprocess.STDOUT)
        try:
            require(result.returncode == 0, f"Final refit failed; see {log_path}")
            receipt = finish_refit_receipt(paths, run_dir, preview["identity"])
        except Exception:
            execution.record_result(name, "failed", time.time() - started)
            execution.persist([path for path in (run_dir, log_path) if path.exists()])
            raise
        execution.record_result(name, "completed", time.time() - started)
        execution.persist([run_dir, log_path, receipt_path, paths.root / "stage6b_decision.json"])
        if execution.should_stop:
            return {**receipt, "next_action": "verify_host_transfer_before_pmm_refit"}
    pmm_dir = paths.root / "final_refit" / "pmm"
    if (pmm_dir / "pmm_refit_receipt.json").exists():
        verify_pmm_refit(paths)
        return receipt
    execution.admit("pmm_final_refit", estimated_seconds)
    started = time.time()
    try:
        execution.run_subprocess([pmm_python, str(campaign.SRC_ROOT / "benchmarking" / "pmm_comparator.py"),
                                  "--campaign-dir", str(paths.root), "--action", "refit", "--refit-dir", str(pmm_dir)],
                                 check=True)
        verify_pmm_refit(paths)
    except BaseException:
        execution.record_result("pmm_final_refit", "failed", time.time() - started)
        execution.persist([pmm_dir] if pmm_dir.exists() else [])
        raise
    execution.record_result("pmm_final_refit", "completed", time.time() - started)
    execution.persist([pmm_dir])
    return receipt


def predict_reference(paths: campaign.CampaignPaths, reference: campaign.CampaignPaths,
                      reference_dir: Path, route_path: Path, *, device: str) -> dict:
    """Inference with the frozen terminal model and saved training normalization."""
    import torch
    from torch_geometric.loader import DataLoader
    from benchmarking.pmm_ion_analysis import native_labels_for, prediction_metrics
    from benchmarking.pmm_ion_features import verify_frozen_feature_inventory
    from training.campaign_runtime import load_campaign_prediction_components
    from training.data import load_training_pockets_with_report_from_dir
    from training.final_test_reporting import collapse_metal_probabilities, collapse_metal_targets
    from training.graph_dataset import PocketGraphDataset, build_graph_data_list
    from training.loop import evaluate_epoch_with_predictions

    route = require_reference_authorization(reference, reference_dir, route_path)
    input_receipt = verify_frozen_feature_inventory(reference, reference_dir, reference_route_json=route_path)
    receipt = verify_final_refit(paths)
    checkpoint = torch.load(paths.root / receipt["checkpoint_path"], map_location="cpu", weights_only=False)
    model, normalization, graph_options = load_campaign_prediction_components(checkpoint, device=device)
    config = checkpoint["config"]
    inventory = campaign.read_json(reference.feature_inventory)
    uses_esm = bool(config["require_esm_embeddings"])
    loaded = load_training_pockets_with_report_from_dir(
        reference_dir, required_targets=("metal",), metal_example_unit="ion", metal_eligibility_scheme="six_class",
        esm_dim=config["esm_dim"], esm_embeddings_dir=(Path(inventory["esm"]["embeddings_dir"])
                                                     if uses_esm else reference.empty_esm),
        require_esm_embeddings=uses_esm, require_external_features=False,
        external_features_root_dir=reference.empty_external_features, external_feature_source="updated",
        unsupported_metal_policy="error", invalid_structure_policy="error",
        source_cohort_csv=reference.cohort, source_cohort_sha256=route["reference_cohort_sha256"],
        load_workers=1)
    pockets = loaded.pockets
    graphs = build_graph_data_list(pockets, **graph_options)
    loader = DataLoader(PocketGraphDataset(pockets, precomputed_data=graphs, normalization_stats=normalization,
                                         **graph_options), batch_size=config["batch_size"], shuffle=False,
                        num_workers=0, generator=torch.Generator().manual_seed(FINAL_SEED + 2))
    predictions = evaluate_epoch_with_predictions(model, loader, device=device)
    require(bool(torch.isfinite(predictions["metal_logits"]).all()), "Reference inference produced nonfinite logits")
    probabilities = predictions["metal_logits"].float().softmax(-1)
    targets = predictions["metal_y"].long()
    require(targets.tolist() == [int(pocket.y_metal) for pocket in pockets], "Reference inference ordering differs")
    common = collapse_metal_probabilities(probabilities)
    common_targets = collapse_metal_targets(targets)
    selected_id = receipt["identity"]["selected_config_id"]
    grid_config = next(c for c in campaign.grid_configs() if c.config_id == selected_id)
    labels = native_labels_for(grid_config)
    rows = []
    for index, pocket in enumerate(pockets):
        rows.append({"source_uid": pocket.metadata["source_uid"],
                     "physical_ion_id": pocket.metadata["physical_ion_id"],
                     "group_id": pocket.metadata["cohort_group_id"], "native_element": pocket.metal_element,
                     "parent_pocket_id": pocket.metadata.get("parent_pocket_id"),
                     "checkpoint_sha256": receipt["checkpoint_sha256"],
                     "y_native": int(targets[index]), "pred_native": int(probabilities[index].argmax()),
                     "y_common4": int(common_targets[index]), "pred_common4": int(common[index].argmax()),
                     **{f"p_native_{label.replace(' ', '_')}": float(probabilities[index, j]) for j, label in enumerate(labels)},
                     **{f"p_common4_{label.replace(' ', '_')}": float(common[index, j])
                        for j, label in enumerate(("Mn", "Cu", "Zn", "Class VIII"))}})
    out = paths.root / "reference" / "neural_predictions.csv"
    with out.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"predictions": {"path": str(out.relative_to(paths.root)), "sha256": sha256_file(out)},
            "metrics": prediction_metrics(rows, native_labels=labels), "n_rows": len(rows),
            "normalization_refitted": False, "configuration": selected_id, "verified_inputs": input_receipt}


def execute_reference_report(paths: campaign.CampaignPaths, args, *, execution=None) -> dict:
    """One-shot reporting; repeated successful invocations return verified artifacts."""
    from benchmarking.pmm_ion_analysis import read_prediction_rows, prediction_metrics, group_bootstrap_pooled
    from benchmarking.pmm_ion_cohort import run_reference_audit
    from benchmarking.pmm_ion_features import plan_embeddings, generate_embeddings, certify_inventory
    neural, pmm = verify_final_refit(paths), verify_pmm_refit(paths)
    decision = campaign.read_json(paths.root / "stage6b_decision.json")
    require(decision["identity"]["source_tree_sha256"] == campaign.source_tree_sha256(),
            "Frozen validation/refit source differs from running code")
    require(args.reference_dir is not None and args.reference_source_csv is not None,
            "Reference reporting requires explicit reference directory and pinned source CSV")
    protocol = {"route": ROUTE, "selection_frozen": True, "campaign_dir": str(paths.root.resolve()),
                "reference_dir": str(args.reference_dir.resolve()),
                "reference_source_sha256": REFERENCE_SOURCE_SHA256,
                "campaign_cohort_sha256": sha256_file(paths.cohort),
                "stage6b_decision_sha256": sha256_file(paths.root / "stage6b_decision.json"),
                "deepmzyme_refit_checkpoint_sha256": neural["checkpoint_sha256"],
                "pmm_refit_model_sha256": pmm["model_sha256"], "reporting_policy": decision["reference_policy"]}
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return protocol
    require(execution is not None, "Reference work requires owned runtime admission")
    root = paths.root / "reference"
    report_path = root / "reference_report.json"
    if report_path.exists():
        report = campaign.read_json(report_path)
        require(report["protocol"] == protocol, "Completed reference report used another frozen protocol")
        for item in report["prediction_artifacts"]:
            require(sha256_file(paths.root / item["path"]) == item["sha256"], "Reference prediction hash changed")
        return report
    root.mkdir(exist_ok=True)
    access_path = root / "reference_access.json"
    require(not access_path.exists(), "A reference attempt already exists; reconcile its saved state before recovery")
    execution.admit("reference_report", args.estimated_fit_seconds)
    started = time.time()
    with access_path.open("x", encoding="utf-8") as handle:
        json.dump({"status": "started", "opened_at_unix": started, "protocol": protocol,
                   "selection_changed_after_access": False}, handle, indent=2)
    route_path = root / "reference_route.json"
    campaign.write_json(route_path, protocol)
    reference = campaign.CampaignPaths(root / "inputs_profile")
    try:
        require_reference_authorization(reference, args.reference_dir, route_path)
        # The source and structures are opened for the first time only below.
        run_reference_audit(args.reference_dir, args.reference_source_csv, reference.root,
                            route_path=route_path, workers=args.load_workers or 2)
        protocol_with_cohort = {**protocol, "reference_cohort_sha256": sha256_file(reference.cohort)}
        campaign.write_json(route_path, protocol_with_cohort)
        plan_embeddings(reference, args.reference_dir, reference_route_json=route_path)
        selected = next(c for c in campaign.grid_configs() if c.config_id == neural["identity"]["selected_config_id"])
        esm_dir = None
        if campaign.FAMILIES[selected.family]["uses_esm"]:
            esm_dir = reference.root / "inputs" / "esm_embeddings_esmc600m_v1"
            generate_embeddings(reference, args.reference_dir, esm_dir, device=args.device,
                                reference_route_json=route_path)
        certify_inventory(reference, args.reference_dir, esm_dir, load_workers=args.load_workers,
                          reference_route_json=route_path)
        # Common evaluation membership is fixed by the same reconstruction rules.
        train_bindings, reference_bindings = read_cohort_csv(paths.cohort), read_cohort_csv(reference.cohort)
        train_groups = {b.group_id for b in train_bindings}
        train_ions = {b.physical_ion_id for b in train_bindings}
        overlap = {"pdb_groups": sorted(train_groups & {b.group_id for b in reference_bindings}),
                   "physical_ions": sorted(train_ions & {b.physical_ion_id for b in reference_bindings}),
                   "reference_ions": len(reference_bindings), "training_ions": len(train_bindings),
                   "interpretation": "secondary possibly overlapping source-split reference, not an independent primary test"}
        campaign.write_json(root / "coverage_and_overlap.json", overlap)
        neural_result = predict_reference(paths, reference, args.reference_dir, route_path, device=args.device)
        pmm_output = root / "pmm"
        execution.run_subprocess([args.pmm_python, str(campaign.SRC_ROOT / "benchmarking" / "pmm_comparator.py"),
                        "--campaign-dir", str(paths.root), "--action", "reference-predict",
                        "--refit-dir", str(paths.root / "final_refit" / "pmm"),
                        "--reference-route-json", str(route_path),
                        "--reference-cohort-csv", str(reference.cohort),
                        "--reference-source-csv", str(args.reference_source_csv),
                        "--output-dir", str(pmm_output)], check=True)
        neural_rows = read_prediction_rows(root / "neural_predictions.csv")
        pmm_rows = read_prediction_rows(pmm_output / "predictions.csv")
        keyed = {row["source_uid"]: row for row in pmm_rows}
        require(len(keyed) == len(pmm_rows) == len(neural_rows), "Reference systems have different row counts")
        for row in neural_rows:
            other = keyed.get(row["source_uid"])
            require(other is not None and all(row[field] == other[field] for field in
                    ("physical_ion_id", "group_id", "native_element", "y_common4")),
                    "Reference systems do not have identical evaluated ions and labels")
        artifacts = [neural_result["predictions"], {"path": str((pmm_output / "predictions.csv").relative_to(paths.root)),
                                                   "sha256": sha256_file(pmm_output / "predictions.csv")}]
        report = {"status": "completed", "protocol": protocol,
                  "label": "secondary Zenodo PMM known-ion reference comparison",
                  "prediction_artifacts": artifacts, "deepmzyme": neural_result,
                  "pmm": prediction_metrics(pmm_rows, native_labels=None), "coverage_and_overlap": overlap,
                  "paired_pooled_difference": group_bootstrap_pooled(pmm_rows, neural_rows),
                  "held_out_test_accessed": True, "selection_changed_after_access": False,
                  "published_paper_parity": "not established; matched system rerun only",
                  "completed_at_unix": time.time()}
        campaign.write_json(report_path, report)
        campaign.write_json(access_path, {"status": "completed", "opened_at_unix": started,
                                          "completed_at_unix": time.time(), "protocol": protocol,
                                          "selection_changed_after_access": False,
                                          "report_sha256": sha256_file(report_path)})
    except BaseException as exc:
        campaign.write_json(access_path, {"status": "failed", "opened_at_unix": started,
                                          "protocol": protocol, "error": str(exc),
                                          "selection_changed_after_access": False})
        execution.record_result("reference_report", "failed", time.time() - started)
        execution.persist([root])
        raise
    execution.record_result("reference_report", "completed", time.time() - started)
    execution.persist([root])
    return report
