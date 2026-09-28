"""Validation-only assessment and non-executing refit bridge for PMM core v2.

The frozen scientific tree owns prediction metrics and training. This additive
consumer owns the amended nine-configuration grid; it never invokes a fit or
opens reference/test inputs, and never rewrites legacy assessment/fit receipts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import json
from pathlib import Path
import shlex
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE_SHA256 = "adc95c42261448dc9d35572a138a3b8349de79124d9708618f511c27a848dd23"
SCOPE_PATH = ROOT / "docs/plans/pmm_core_scope_v2.json"
FAMILIES = ("only_esm", "only_gvp", "gvp_late_fusion")
TARGETS = ("four_class", "five_class", "six_class")
COMMON4 = ("Mn", "Cu", "Zn", "Class VIII")
NATIVE = {"four_class": COMMON4, "five_class": ("Mn", "Cu", "Zn", "Fe", "Class VIII"),
          "six_class": ("Mn", "Cu", "Zn", "Fe", "Co", "Ni")}
CONTROL = "only_gvp__four_class__none"
ROUTE = "zenodo_pmm_secondary_reference"


class CoreAssessmentError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise CoreAssessmentError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def bind_source(source_root):
    """Use the shared replay consumer's source guard before scientific imports."""
    from pmm_core_replay import ReplayValidationError, bind_frozen_source
    try:
        bind_frozen_source(source_root)
    except ReplayValidationError as exc:
        raise CoreAssessmentError(str(exc)) from exc
    return importlib.import_module("benchmarking.pmm_ion_analysis"), importlib.import_module("benchmarking.pmm_ion_campaign")


def load_scope(scope_path=SCOPE_PATH):
    scope = read_json(scope_path)
    expected = {"scope_id": "pmm-core-v2", "campaign_id": "pmm_ion_metal_v2_context",
                "source_tree_sha256": SOURCE_SHA256, "families": list(FAMILIES), "targets": list(TARGETS),
                "readouts": ["none"], "folds": list(range(5)), "model_seeds": [42], "epochs": 50,
                "required_core_fits": 45, "paused_readouts": ["first_shell_bias"]}
    for key, value in expected.items():
        require(scope.get(key) == value, f"Unsupported core scope field {key}")
    for key, value in {"required_configurations": 9, "required_folds_per_configuration": 5,
                       "required_pmm_folds": 5, "bootstrap_resamples": 10000, "bootstrap_seed": 42,
                       "confidence": .95, "simultaneous_correction_denominator": 9,
                       "rare_class_recall_drop_tolerance": .03}.items():
        require(scope.get("assessment", {}).get(key) == value, f"Unsupported assessment policy {key}")
    return scope


def config_ids():
    return [f"{family}__{target}__none" for family in FAMILIES for target in TARGETS]


def validate_grid_shape(collected):
    require(set(collected) == set(config_ids()), "Assessment requires the exact nine core configurations")
    require(all(set(entry["folds"]) == set(range(5)) for entry in collected.values()),
            "Assessment requires every planned fold: 45 qualified fits")


def compare_pair(control_id, challenger_id, collected, analysis):
    control, challenger = collected[control_id]["folds"], collected[challenger_id]["folds"]
    differences = np.asarray([challenger[k]["metrics"]["common4"]["balanced_accuracy"] -
                              control[k]["metrics"]["common4"]["balanced_accuracy"] for k in range(5)])
    interval = analysis.fold_bootstrap(differences, resamples=10000, seed=42, n_contrasts=9)
    a, b = analysis.mean_class_recalls(control), analysis.mean_class_recalls(challenger)
    drops = {label: None if a[label] is None or b[label] is None else a[label] - b[label] for label in COMMON4}
    gates = {"positive_mean_difference": interval["mean_difference"] > 0,
             "positive_ci95_lower_bound": interval["ci95_low"] > 0,
             "no_missing_class_recall": all(value is not None for value in b.values()),
             "no_zero_mean_class_recall": all(value is not None and value > 0 for value in b.values()),
             "no_class_recall_drop_over_0.03": all(value is not None and value <= .03 for value in drops.values())}
    return {"control": control_id, "challenger": challenger_id, "fold_differences": differences.tolist(),
            **interval, "multiplicity_denominator": 9, "control_mean_class_recall": a,
            "challenger_mean_class_recall": b, "class_recall_drop": drops, "gates": gates,
            "promoted": all(gates.values()),
            "simultaneous_claim_supported": all(gates.values()) and interval["bonferroni_low"] > 0}


def select_core_configuration(collected, contrasts, analysis, *, tie_epsilon):
    """Preserve frozen final-report selection gates, extended to the core grid."""
    validate_grid_shape(collected)
    matched = {row["challenger"]: row for row in contrasts}
    rows = []
    for config_id, entry in collected.items():
        family, target, _ = config_id.split("__")
        scores = [entry["folds"][k]["metrics"]["common4"]["balanced_accuracy"] for k in range(5)]
        recalls = analysis.mean_class_recalls(entry["folds"])
        minimum = [entry["folds"][k]["metrics"]["common4"]["min_recall"] for k in range(5)]
        vs_control = compare_pair(CONTROL, config_id, collected, analysis)
        matched_pass = target == "four_class" or matched.get(config_id, {}).get("promoted", False)
        class_coverage = all(value is not None and value > 0 for value in recalls.values())
        rows.append({"config_id": config_id, "family": family, "target": target,
                     "mean_common4_ba": float(np.mean(scores)), "mean_min_recall": float(np.mean(minimum)),
                     "worst_fold_common4_ba": min(scores), "sd_common4_ba": float(np.std(scores, ddof=1)),
                     "complexity_proxy": FAMILIES.index(family), "matched_promotion_passed": bool(matched_pass),
                     "control_ci_low": vs_control["ci95_low"],
                     "control_mean_difference": vs_control["mean_difference"],
                     "control_recall_gate": all(v for k, v in vs_control["gates"].items() if "difference" not in k and "bound" not in k),
                     "eligible": class_coverage and (config_id == CONTROL or (matched_pass and vs_control["promoted"]))})
    eligible = [row for row in rows if row["eligible"]]
    require(any(row["config_id"] == CONTROL for row in eligible), "Predeclared control has missing/zero class recall; refit selection blocked")
    challengers = [row for row in eligible if row["config_id"] != CONTROL]
    if challengers:
        largest = max(row["mean_common4_ba"] for row in challengers)
        tied = [row for row in challengers if largest - row["mean_common4_ba"] <= tie_epsilon]
        chosen = min(tied, key=lambda row: (-row["mean_min_recall"], -row["worst_fold_common4_ba"],
                                           row["sd_common4_ba"], row["complexity_proxy"], row["config_id"]))
        reason = "passed_matched_and_control_gates" if len(tied) == 1 else "eligible_candidate_tie_break"
    else:
        chosen = next(row for row in rows if row["config_id"] == CONTROL)
        reason = "retained_predeclared_control_no_supported_replacement"
    return {"selected_config_id": chosen["config_id"], "selection_reason": reason, "control": CONTROL,
            "rank_metric": "mean_common4_balanced_accuracy", "tie_epsilon": tie_epsilon,
            "tie_policy_source": "frozen src/benchmarking/pmm_final_report.py:TIE_EPSILON and select_configuration",
            "tie_breakers": ["mean_min_recall_desc", "worst_fold_common4_ba_desc", "sd_common4_ba_asc", "complexity_proxy_asc", "config_id_asc"],
            "ranked_candidates": sorted(rows, key=lambda row: (-row["mean_common4_ba"], row["config_id"])),
            "control_comparisons_are_selection_diagnostics": True}


def _add_evidence(evidence, additions):
    for path, value in additions.items():
        key = str(Path(path).resolve())
        require(key not in evidence or evidence[key] == value, f"Evidence changed during assessment: {key}")
        evidence[key] = value


def collect_core(campaign_dir, train_dir, *, source_root, scope_path=SCOPE_PATH):
    """All-or-nothing collection; replay qualification belongs to its own consumer."""
    analysis, campaign = bind_source(source_root)
    from benchmarking import pmm_comparator as comparator
    from pmm_core_replay import validate_core_unit

    campaign_dir, train_dir = Path(campaign_dir).resolve(), Path(train_dir).resolve()
    require(train_dir.name == "train" and train_dir.is_dir(), "Only an existing training-side directory named train is allowed")
    scope = load_scope(scope_path)
    manifest, membership, cohort = comparator.read_campaign_contract(campaign_dir)
    require(manifest.get("campaign_id") == scope["campaign_id"], "Campaign identity differs from core scope")
    comparator.verify_comparator_outputs(campaign_dir)
    evidence = {str(Path(scope_path).resolve()): sha256(scope_path), str(Path(__file__).resolve()): sha256(__file__)}
    for name in ("campaign_manifest.json", "train_cohort.csv", "fold_membership.csv", "fold_class_weights.json", "feature_inventory.json",
                 "pmm_comparator/pmm_comparator_manifest.json"):
        path = campaign_dir / name
        evidence[str(path)] = sha256(path)
    evidence[str(comparator.PMM_SOURCE_DEFAULT)] = sha256(comparator.PMM_SOURCE_DEFAULT)
    collected = {}
    for config_id in config_ids():
        family, target, _ = config_id.split("__")
        folds = {}
        for fold in range(5):
            unit = validate_core_unit(campaign_dir, train_dir, family, target, fold, source_root=Path(source_root))
            expected = {"campaign_id": scope["campaign_id"], "family": family, "target_scheme": target,
                        "readout": "none", "fold": fold, "model_seed": 42, "epochs": 50,
                        "source_tree_sha256": SOURCE_SHA256, "cohort_sha256": sha256(campaign_dir / "train_cohort.csv"),
                        "fold_membership_sha256": sha256(campaign_dir / "fold_membership.csv")}
            require(all(unit["identity"].get(k) == v for k, v in expected.items()), f"{config_id} fold {fold}: normalized unit identity differs")
            comparator.validate_prediction_rows(unit["rows"], membership, fold, seed=42,
                                               checkpoint_sha256=unit["receipt"]["selected_checkpoint_sha256"])
            metrics = analysis.prediction_metrics(unit["rows"], native_labels=NATIVE[target])
            require(metrics == unit["metrics"], f"{config_id} fold {fold}: normalized unit metrics differ")
            require(bool(unit.get("replay_policy_id")) and bool(unit.get("evidence_files")), "Replay qualification lacks policy/evidence bindings")
            folds[fold] = unit
            _add_evidence(evidence, unit["evidence_files"])
        analysis.check_oof_coverage(list(cohort), folds)
        collected[config_id] = {"folds": folds}
    pmm = {}
    for fold in range(5):
        path = campaign_dir / "pmm_comparator" / f"fold{fold}_predictions.csv"
        rows = analysis.read_prediction_rows(path)
        comparator.validate_prediction_rows(rows, membership, fold)
        pmm[fold] = {"run_name": f"pmm_fold{fold}", "rows": rows,
                     "metrics": analysis.prediction_metrics(rows, native_labels=None)}
        evidence[str(path)] = sha256(path)
    analysis.check_oof_coverage(list(cohort), pmm)
    validate_grid_shape(collected)
    return collected, pmm, cohort, evidence, analysis, campaign


def evaluate_core(collected, pmm, cohort, analysis):
    validate_grid_shape(collected)
    require(set(pmm) == set(range(5)), "All five aligned PMM folds are required")
    analysis.check_oof_coverage(list(cohort), pmm)
    for entry in collected.values():
        analysis.check_oof_coverage(list(cohort), entry["folds"])
    contrasts = [compare_pair(f"{family}__four_class__none", f"{family}__{target}__none", collected, analysis)
                 for family in FAMILIES for target in ("five_class", "six_class")]
    from benchmarking.pmm_final_report import TIE_EPSILON
    selection = select_core_configuration(collected, contrasts, analysis, tie_epsilon=TIE_EPSILON)
    summaries = {}
    for config_id, folds in [(k, v["folds"]) for k, v in collected.items()] + [("pmm", pmm)]:
        rows = [row for fold in range(5) for row in folds[fold]["rows"]]
        target = config_id.split("__")[1] if config_id != "pmm" else None
        summaries[config_id] = {"mean_common4_balanced_accuracy": float(np.mean([folds[k]["metrics"]["common4"]["balanced_accuracy"] for k in range(5)])),
                                "mean_common4_macro_f1": float(np.mean([folds[k]["metrics"]["common4"]["macro_f1"] for k in range(5)])),
                                "mean_common4_class_recall": analysis.mean_class_recalls(folds),
                                "pooled_oof": analysis.prediction_metrics(rows, native_labels=NATIVE.get(target)),
                                "native_label_meaning": "Class VIII = Co+Ni" if target == "five_class" else "Class VIII = Fe+Co+Ni" if target == "four_class" else None}
    return {"status": "complete", "required_core_fits": 45, "qualified_core_fits": 45,
            "configurations": 9, "folds_per_configuration": 5, "pmm_folds": 5, "seed": 42,
            "selection": selection, "active_contrasts": contrasts, "active_contrast_count": 6,
            "multiplicity_denominator": 9, "paused_awareness_contrasts": 3,
            "five_vs_six_is_descriptive_only": True, "summaries": summaries,
            "held_out_test_accessed": False, "refit_executed": False,
            "limitations": ["One-seed, fivefold grouped validation with native checkpoint selection; not nested CV or seed-stability evidence.",
                            "Pooled OOF metrics and mean-fold metrics are distinct estimands.",
                            "PMM is refitted on matched campaign folds; paper-score parity and the primary final-test route remain unresolved.",
                            "Replay qualification labels and any retrospective limitations remain attached to every unit."]}


def _new_output_dir(out_dir, campaign_dir):
    out_dir = Path(out_dir).resolve()
    require(out_dir.is_relative_to(Path(campaign_dir).resolve()), "Core outputs must stay within this campaign")
    require(not out_dir.exists(), "Output directory already exists; preserve earlier reports and choose a new directory")
    return out_dir


def assess_core(campaign_dir, train_dir, *, source_root, out_dir, scope_path=SCOPE_PATH):
    out_dir = _new_output_dir(out_dir, campaign_dir)
    collected, pmm, cohort, evidence, analysis, _ = collect_core(campaign_dir, train_dir, source_root=source_root, scope_path=scope_path)
    decision = evaluate_core(collected, pmm, cohort, analysis)
    fold_rows, oof_rows, qualifications = [], [], []
    for config_id, folds in [(k, v["folds"]) for k, v in collected.items()] + [("pmm", pmm)]:
        for fold in range(5):
            unit = folds[fold]
            qualification = {"config_id": config_id, "fold": fold, "run_name": unit["run_name"],
                             "replay_policy_id": unit.get("replay_policy_id", "pmm_comparator_frozen_manifest"),
                             "replay_qualification": unit.get("replay_qualification")}
            qualifications.append(qualification)
            for view in ("common4", "native"):
                if view in unit["metrics"]:
                    fold_rows.append({"config_id": config_id, "fold": fold, "view": view,
                                      "n": unit["metrics"]["n"], **unit["metrics"][view],
                                      "checkpoint_sha256": unit.get("receipt", {}).get("selected_checkpoint_sha256"),
                                      "replay_policy_id": qualification["replay_policy_id"]})
            oof_rows.extend({**row, "config_id": config_id, "replay_policy_id": qualification["replay_policy_id"]} for row in unit["rows"])
    decision.update(scope_id="pmm-core-v2", campaign_dir=str(Path(campaign_dir).resolve()),
                    train_dir=str(Path(train_dir).resolve()), source_tree_sha256=SOURCE_SHA256,
                    evidence_files=evidence, replay_qualifications=qualifications)
    out_dir.mkdir(parents=True)
    outputs = {"core_cv_fold_metrics.csv": fold_rows, "core_cv_oof_predictions.csv": oof_rows,
               "core_cv_paired_deltas.csv": decision["active_contrasts"]}
    for name, rows in outputs.items():
        analysis.write_rows(out_dir / name, [{k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows])
    decision["output_files"] = {name: sha256(out_dir / name) for name in outputs}
    write_json(out_dir / "core_validation_decision.json", decision)
    return decision


def verify_decision(decision_path, campaign_dir, train_dir, *, source_root, scope_path=SCOPE_PATH):
    decision_path = Path(decision_path).resolve()
    require(decision_path.name == "core_validation_decision.json" and decision_path.is_relative_to(Path(campaign_dir).resolve()),
            "Only a core validation decision within this campaign is accepted")
    decision = read_json(decision_path)
    require(decision.get("status") == "complete" and decision.get("qualified_core_fits") == 45 and
            decision.get("source_tree_sha256") == SOURCE_SHA256 and decision.get("held_out_test_accessed") is False,
            "Refit preview needs complete, frozen, validation-only core results")
    # Recollect known training-side paths first. Never open arbitrary paths from
    # an edited decision's evidence map (which could point at held-out inputs).
    collected, pmm, cohort, evidence, analysis, campaign = collect_core(campaign_dir, train_dir, source_root=source_root, scope_path=scope_path)
    require(decision.get("evidence_files") == evidence, "Assessment evidence changed")
    require(set(decision.get("output_files", {})) == {"core_cv_fold_metrics.csv", "core_cv_oof_predictions.csv", "core_cv_paired_deltas.csv"},
            "Core decision lacks the complete assessment output set")
    for name, expected in decision.get("output_files", {}).items():
        path = (decision_path.parent / name).resolve()
        require(path.parent == decision_path.parent and sha256(path) == expected, "Assessment output changed or escaped its directory")
    current = evaluate_core(collected, pmm, cohort, analysis)
    require(decision["evidence_files"] == evidence and all(decision.get(k) == v for k, v in current.items()),
            "Core decision differs from the revalidated evidence or selection")
    return decision, campaign


def _build_refit(campaign, campaign_dir, train_dir, selected, decision_hash, *, python_bin, device, load_workers=None):
    from benchmarking.pmm_final_report import _set_argument
    from training.config import config_to_payload, parse_args
    paths = campaign.CampaignPaths(Path(campaign_dir))
    family, target, readout = selected.split("__")
    require(selected in config_ids(), "Refit configuration was not in the core grid")
    base_target = "four_class" if target == "five_class" else target
    command, env, _ = campaign.build_train_command(paths, python_bin=python_bin, train_dir=Path(train_dir),
        config=campaign.GridConfig(family, base_target, readout), fold=0, seed=42, epochs=50,
        device=device, runs_dir=paths.root / "core_final_refit", load_workers=load_workers)
    for flag in ("--n-folds", "--fold-index", "--fold-membership-csv", "--fold-membership-sha256",
                 "--export-validation-predictions", "--campaign-run-identity"):
        _set_argument(command, flag, remove=True)
    for flag, value in (("--val-fraction", "0.0"), ("--selection-metric", "train_loss"),
                        ("--run-name", f"core_stage6b__{selected}__seed42"), ("--metal-label-scheme", target)):
        _set_argument(command, flag, value)
    bindings = campaign.read_cohort_csv(paths.cohort, campaign.load_campaign(paths)["cohort"]["sha256"])
    counts = Counter(campaign.COMMON_FOUR[b.native_element] for b in bindings)
    require(set(counts) == set(COMMON4), "Full-training cohort lacks an endpoint class")
    weights = {label: len(bindings) / (4 * count) for label, count in counts.items()}
    multipliers = {"mn": weights["Mn"], "cu": weights["Cu"], "zn": weights["Zn"]}
    metal_keys = {"four_class": ("class-viii",), "five_class": ("fe", "class-viii"), "six_class": ("fe", "co", "ni")}
    multipliers.update({key: weights["Class VIII"] for key in metal_keys[target]})
    for key in ("mn", "cu", "zn", "fe", "co", "ni", "class-viii"):
        _set_argument(command, f"--{key}-loss-multiplier", remove=True)
    for key, value in multipliers.items():
        _set_argument(command, f"--{key}-loss-multiplier", repr(value))
    resolved = config_to_payload(parse_args(command[3:]))
    identity = {"campaign_id": campaign.load_campaign(paths)["campaign_id"], "scope_id": "pmm-core-v2",
                "phase": "core_stage6b_full_train", "selected_config_id": selected,
                "core_validation_decision_sha256": decision_hash, "cohort_sha256": sha256(paths.cohort),
                "feature_inventory_sha256": campaign.inventory_identity_sha256(paths),
                "source_tree_sha256": SOURCE_SHA256, "model_seed": 42, "epochs": 50,
                "checkpoint_rule": "terminal_epoch_50", "common_four_weights": weights,
                "resolved_config_sha256": campaign.stable_hash({k: v for k, v in resolved.items() if k not in campaign.NON_IDENTITY_CONFIG_KEYS})}
    command += ["--campaign-run-identity", json.dumps(identity, sort_keys=True)]
    return command, env, identity


def preview_core_refit(campaign_dir, train_dir, *, source_root, decision_path, out_dir, route,
                       python_bin=sys.executable, device="cuda", load_workers=None, scope_path=SCOPE_PATH):
    require(route == ROUTE, "Declare the supported secondary reference route; primary final-test route remains unresolved")
    out_dir = _new_output_dir(out_dir, campaign_dir)
    decision, campaign = verify_decision(decision_path, campaign_dir, train_dir, source_root=source_root, scope_path=scope_path)
    command, env, identity = _build_refit(campaign, campaign_dir, train_dir, decision["selection"]["selected_config_id"],
                                        sha256(decision_path), python_bin=python_bin, device=device, load_workers=load_workers)
    preview = {"status": "preview_only", "launch_final_refit": False, "held_out_access_authorized": False,
               "route": route, "secondary_only": True, "possibly_overlapping": True,
               "train_dir": str(Path(train_dir).resolve()), "decision_path": str(Path(decision_path).resolve()),
               "validation_decision_sha256": sha256(decision_path), "identity": identity,
               "command": command, "environment": env, "selection": decision["selection"],
               "normalization": "recomputed on the complete frozen training cohort by the frozen trainer",
               "pmm_refit_required": True, "execution_requires_separate_authorization": True}
    out_dir.mkdir(parents=True)
    write_json(out_dir / "core_stage6b_decision.json", preview)
    (out_dir / "core_stage6b_final_refit_command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    from benchmarking.pmm_ion_analysis import write_rows
    write_rows(out_dir / "core_stage6b_ranked_candidates.csv", decision["selection"]["ranked_candidates"])
    return preview


def verify_core_refit(campaign_dir, train_dir, *, source_root, preview_path, run_dir, pmm_refit_dir=None, scope_path=SCOPE_PATH):
    """Verify an existing neural terminal refit; does not authorize final reporting."""
    preview_path, run_dir = Path(preview_path).resolve(), Path(run_dir).resolve()
    require(preview_path.name == "core_stage6b_decision.json" and preview_path.is_relative_to(Path(campaign_dir).resolve()), "Core refit preview must belong to this campaign")
    preview = read_json(preview_path)
    require(preview.get("status") == "preview_only" and preview.get("launch_final_refit") is False and
            preview.get("held_out_access_authorized") is False and preview.get("route") == ROUTE,
            "Unsupported refit preview/route")
    require(sha256(preview["decision_path"]) == preview["validation_decision_sha256"], "Validation decision changed after refit preview")
    decision, campaign = verify_decision(preview["decision_path"], campaign_dir, train_dir, source_root=source_root, scope_path=scope_path)
    require(preview["identity"]["selected_config_id"] == decision["selection"]["selected_config_id"], "Refit selection differs")
    require(run_dir.parent == (Path(campaign_dir).resolve() / "core_final_refit") and
            run_dir.name == f"core_stage6b__{decision['selection']['selected_config_id']}__seed42", "Refit directory differs from selected core unit")
    command, _env, identity = _build_refit(campaign, campaign_dir, train_dir, decision["selection"]["selected_config_id"],
                                          sha256(preview["decision_path"]), python_bin=sys.executable, device="cpu")
    require(preview["identity"] == identity, "Refit identity differs from the full-cohort recipe")
    metadata, run_config = read_json(run_dir / "run_metadata.json"), read_json(run_dir / "run_config.json")
    require(metadata.get("fit_status") == "completed" and metadata.get("campaign_run_identity") == identity,
            "Refit is incomplete or has the wrong identity")
    require(not (run_dir / "test_report.json").exists(), "Refit contains a test report")
    from dataclasses import fields
    from training.config import TrainConfig
    from training.run import to_jsonable
    import torch
    checkpoint_path = run_dir / "last_model_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    histories = [run_config.get("history", []), checkpoint.get("history", [])]
    require(all([row["epoch"] for row in history] == list(range(1, 51)) for history in histories), "Refit must complete terminal epoch 50")
    for payload in (metadata, run_config, checkpoint):
        config = payload["config"]
        recorded = config.get("campaign_run_identity")
        require((json.loads(recorded) if isinstance(recorded, str) else recorded) == identity, "Refit config/checkpoint identity differs")
        scientific = {field.name: config[field.name] for field in fields(TrainConfig)
                      if field.name not in campaign.NON_IDENTITY_CONFIG_KEYS}
        require(campaign.stable_hash(scientific) == identity["resolved_config_sha256"], "Refit resolved scientific configuration differs")
        require(config.get("val_fraction") == 0 and config.get("n_folds") is None and
                config.get("fold_membership_csv") is None and config.get("source_cohort_sha256") == identity["cohort_sha256"], "Refit is not full-cohort training")
        require(not config.get("run_test_eval") and config.get("test_structure_dir") is None and
                config.get("test_summary_csv") is None and payload.get("test_report") is None,
                "Refit requested or recorded held-out evaluation")
    require(run_config.get("normalization_stats") and run_config["normalization_stats"] == metadata.get("normalization_stats") == checkpoint.get("normalization_stats"), "Refit normalization is absent or inconsistent")
    dataset = run_config.get("dataset_summary")
    require(dataset and dataset == metadata.get("dataset_summary") == to_jsonable(checkpoint.get("dataset_summary")),
            "Refit retained training identities are absent or inconsistent")
    bindings = campaign.read_cohort_csv(Path(campaign_dir) / "train_cohort.csv", identity["cohort_sha256"])
    expected = {binding.example_id(): binding for binding in bindings}
    actual_rows = dataset["retained_split_identity"]["train"]["examples"]
    actual = {row["pocket_id"]: row for row in actual_rows}
    require(len(actual) == len(actual_rows) and set(actual) == set(expected) and
            dataset["n_train_pockets"] == len(bindings) and dataset["n_val_pockets"] == 0 and
            dataset["retained_split_identity"]["validation"]["examples"] == [],
            "Refit retained examples do not cover the complete frozen cohort exactly once")
    target = identity["selected_config_id"].split("__")[1]
    labels = NATIVE[target]
    for name, binding in expected.items():
        element_index = ("MN", "CU", "ZN", "FE", "CO", "NI").index(binding.native_element)
        require(actual[name]["structure_id"] == binding.structure_stem and actual[name]["group"] == binding.pdbid
                and actual[name]["y_metal"] == min(element_index, len(labels) - 1), "Refit retained structure/group/target differs")
    pmm_receipt = None
    if pmm_refit_dir is not None:
        from benchmarking.pmm_comparator import verify_refit
        pmm_refit_dir = Path(pmm_refit_dir).resolve()
        require(pmm_refit_dir == Path(campaign_dir).resolve() / "core_final_refit" / "pmm", "PMM refit must stay in the core final-refit directory")
        pmm_receipt = verify_refit(Path(campaign_dir), pmm_refit_dir)
    return {"status": "completed", "identity": identity, "completed_epochs": 50, "terminal_checkpoint": True,
            "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
            "preview_sha256": sha256(preview_path), "run_config_sha256": sha256(run_dir / "run_config.json"),
            "run_metadata_sha256": sha256(run_dir / "run_metadata.json"), "held_out_test_accessed": False,
            "pmm_refit_verified": pmm_receipt is not None, "pmm_refit_receipt": pmm_receipt,
            "final_reporting_authorized": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = assess_core(**vars(args))
    print(json.dumps({"status": result["status"], "qualified_core_fits": result["qualified_core_fits"], "selection": result["selection"]["selected_config_id"]}))


if __name__ == "__main__":
    main()
