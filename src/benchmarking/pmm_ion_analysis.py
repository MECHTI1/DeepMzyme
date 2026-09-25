"""Validation-only assessment of the PMM ion campaign (plan Step 6).

Primary estimand: mean common-four balanced accuracy over the five selected fold
checkpoints. Every metric is recomputed from the UID-keyed ``val_predictions.csv``
of the selected checkpoint and reconciled with the run's own receipt; native and
common-four values therefore always describe the same checkpoint.

Paired inference follows Stage 6: per-fold differences, 10,000 fold-bootstrap
resamples, seed 42, 95 % percentile intervals, plus a Bonferroni bound at
``1 - 0.05/6`` for a simultaneous claim. A pooled-OOF interval, when reported,
resamples whole PDB groups and is labeled conditional on the fitted models.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from benchmarking.pmm_ion_campaign import (
    ASSESSMENT_IGNORED_IDENTITY_KEYS,
    MODEL_SEEDS,
    N_FOLDS,
    PREDECLARED_CONTRASTS,
    CampaignPaths,
    GridConfig,
    build_train_command,
    completed_run_receipt,
    grid_configs,
    run_name,
    write_json,
)
from training.source_cohort import read_cohort_csv, sha256_file
from benchmarking.pmm_comparator import read_campaign_contract, validate_prediction_rows, verify_comparator_outputs

COMMON_FOUR_LABELS = ("Mn", "Cu", "Zn", "Class VIII")
SIX_LABELS = ("Mn", "Cu", "Zn", "Fe", "Co", "Ni")
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42
CONFIDENCE = 0.95
N_PREDECLARED = len(PREDECLARED_CONTRASTS)
RARE_RECALL_TOLERANCE = 0.03


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def classification_summary(y_true: np.ndarray, y_pred: np.ndarray, labels: tuple[str, ...]) -> dict[str, Any]:
    matrix = confusion(y_true, y_pred, len(labels))
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    tp = np.diag(matrix)
    recall = [float(tp[i] / support[i]) if support[i] else None for i in range(len(labels))]
    precision = [float(tp[i] / predicted[i]) if predicted[i] else 0.0 for i in range(len(labels))]
    f1 = []
    for i in range(len(labels)):
        if support[i] == 0:
            continue
        p, r = precision[i], recall[i] or 0.0
        f1.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    present = [value for value in recall if value is not None]
    return {
        "balanced_accuracy": float(np.mean(present)) if present else None,
        "accuracy": float(tp.sum() / max(1, matrix.sum())),
        "macro_f1": float(np.mean(f1)) if f1 else None,
        "min_recall": float(min(present)) if present else None,
        "recall": dict(zip(labels, recall)),
        "support": dict(zip(labels, [int(value) for value in support])),
        "confusion_matrix": matrix.tolist(),
    }


def read_prediction_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def prediction_metrics(rows: list[dict[str, str]], *, native_labels: tuple[str, ...] | None) -> dict[str, Any]:
    y4 = np.array([int(row["y_common4"]) for row in rows])
    p4 = np.array([int(row["pred_common4"]) for row in rows])
    metrics = {"common4": classification_summary(y4, p4, COMMON_FOUR_LABELS), "n": len(rows)}
    if native_labels is not None:
        yn = np.array([int(row["y_native"]) for row in rows])
        pn = np.array([int(row["pred_native"]) for row in rows])
        metrics["native"] = classification_summary(yn, pn, native_labels)
    return metrics


def native_labels_for(config: GridConfig) -> tuple[str, ...]:
    return SIX_LABELS if config.target == "six_class" else COMMON_FOUR_LABELS


def expected_identity(paths: CampaignPaths, train_dir: Path, config: GridConfig, fold: int, seed: int) -> dict[str, Any]:
    """The identity the campaign command builder assigns to this unit (placement-independent)."""
    return build_train_command(paths, python_bin="python", train_dir=train_dir, config=config, fold=fold,
                               seed=seed, device="cuda", runs_dir=paths.runs)[2]


def collect_units(paths: CampaignPaths, train_dir: Path) -> dict[str, Any]:
    """Verified selected-checkpoint predictions for every planned unit; missing ones are listed.

    A unit counts only when its recorded identity equals the expected one, except for
    the code-tree hash; all counted units must then share one code tree.
    """
    collected: dict[str, Any] = {}
    _manifest, membership, _cohort = read_campaign_contract(paths.root)
    for config in grid_configs():
        folds: dict[int, dict[str, Any]] = {}
        missing = []
        for fold in range(N_FOLDS):
            for seed in MODEL_SEEDS:
                name = run_name(config, fold, seed)
                run_dir = paths.runs / name
                receipt = completed_run_receipt(run_dir, expected_identity(paths, train_dir, config, fold, seed),
                                                ignore=ASSESSMENT_IGNORED_IDENTITY_KEYS)
                if receipt is None:
                    missing.append(name)
                    continue
                rows = read_prediction_rows(run_dir / receipt["validation_predictions"]["path"])
                if receipt.get("reconciliation_status") != "match":
                    raise ValueError(f"{name}: selected-checkpoint validation replay is not reconciled")
                validate_prediction_rows(rows, membership, fold, native_labels=native_labels_for(config),
                                         seed=seed, checkpoint_sha256=receipt["selected_checkpoint_sha256"])
                metrics = prediction_metrics(rows, native_labels=native_labels_for(config))
                recorded = receipt["metrics"]
                # The receipt's metrics are the exported replay of the selected checkpoint.
                for key, value in (("val_metal_collapsed4_balanced_acc", metrics["common4"]["balanced_accuracy"]),
                                   ("val_metal_balanced_acc", metrics["native"]["balanced_accuracy"])):
                    if abs(float(recorded[key]) - float(value)) > 1e-9:
                        raise ValueError(f"{name}: {key} from predictions {value} != receipt {recorded[key]}")
                folds[fold] = {"run_name": name, "receipt": receipt, "rows": rows, "metrics": metrics}
        collected[config.config_id] = {"config": config, "folds": folds, "missing": missing,
                                       "complete": not missing}
    return collected


def check_oof_coverage(cohort_uids: list[str], folds: dict[int, dict[str, Any]]) -> None:
    seen: dict[str, int] = {}
    for fold, unit in folds.items():
        for row in unit["rows"]:
            if int(row["fold"]) != fold:
                raise ValueError(f"{unit['run_name']}: row fold {row['fold']} != {fold}")
            if row["source_uid"] in seen:
                raise ValueError(f"{row['source_uid']} predicted in folds {seen[row['source_uid']]} and {fold}")
            seen[row["source_uid"]] = fold
    if set(seen) != set(cohort_uids):
        raise ValueError(f"OOF predictions cover {len(seen)} of {len(cohort_uids)} cohort ions")


def fold_bootstrap(differences: np.ndarray, *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
                   n_contrasts: int = N_PREDECLARED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(differences), size=(resamples, len(differences)))
    means = differences[index].mean(axis=1)
    alpha = 1.0 - CONFIDENCE
    bonferroni_alpha = alpha / n_contrasts
    return {
        "mean_difference": float(differences.mean()),
        "ci95_low": float(np.quantile(means, alpha / 2)),
        "ci95_high": float(np.quantile(means, 1 - alpha / 2)),
        "bonferroni_confidence": 1 - bonferroni_alpha,
        "bonferroni_low": float(np.quantile(means, bonferroni_alpha / 2)),
        "bonferroni_high": float(np.quantile(means, 1 - bonferroni_alpha / 2)),
        "resamples": resamples,
        "seed": seed,
    }


def group_bootstrap_pooled(control: list[dict[str, str]], challenger: list[dict[str, str]], *,
                           resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict[str, float]:
    """Pooled-OOF BA difference resampling whole PDB groups (conditional on fitted models)."""
    by_uid = {row["source_uid"]: row for row in challenger}
    uids = [row["source_uid"] for row in control]
    groups = sorted({row["group_id"] for row in control})
    group_index = {group: i for i, group in enumerate(groups)}
    g = np.array([group_index[row["group_id"]] for row in control])
    y = np.array([int(row["y_common4"]) for row in control])
    correct_a = np.array([int(row["pred_common4"]) == int(row["y_common4"]) for row in control], dtype=float)
    correct_b = np.array([int(by_uid[uid]["pred_common4"]) == int(by_uid[uid]["y_common4"]) for uid in uids], dtype=float)
    n_groups = len(groups)
    rng = np.random.default_rng(seed)

    def ba(weights: np.ndarray, correct: np.ndarray) -> float:
        recalls = []
        for label in range(len(COMMON_FOUR_LABELS)):
            mask = y == label
            total = weights[mask].sum()
            if total > 0:
                recalls.append((weights[mask] * correct[mask]).sum() / total)
        return float(np.mean(recalls))

    ones = np.ones(len(control))
    observed = ba(ones, correct_b) - ba(ones, correct_a)
    diffs = np.empty(resamples)
    for i in range(resamples):
        counts = np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)
        weights = counts[g].astype(float)
        diffs[i] = ba(weights, correct_b) - ba(weights, correct_a)
    return {"pooled_oof_difference": observed, "ci95_low": float(np.quantile(diffs, 0.025)),
            "ci95_high": float(np.quantile(diffs, 0.975)), "resampling_unit": "PDB group",
            "interpretation": "conditional on the fitted fold models; not an estimate of mean-fold BA"}


def mean_class_recalls(folds: dict[int, dict[str, Any]]) -> dict[str, float | None]:
    out = {}
    for label in COMMON_FOUR_LABELS:
        values = [unit["metrics"]["common4"]["recall"][label] for unit in folds.values()]
        out[label] = None if any(value is None for value in values) else float(np.mean(values))
    return out


def evaluate_contrast(name: str, control: dict[str, Any], challenger: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"contrast": name, "control": control["config"].config_id,
                              "challenger": challenger["config"].config_id}
    if not (control["complete"] and challenger["complete"]):
        result.update(status="incomplete", missing=control["missing"] + challenger["missing"], promoted=False)
        return result
    folds = sorted(control["folds"])
    differences = np.array([challenger["folds"][k]["metrics"]["common4"]["balanced_accuracy"]
                            - control["folds"][k]["metrics"]["common4"]["balanced_accuracy"] for k in folds])
    result["fold_differences"] = differences.tolist()
    result.update(fold_bootstrap(differences))
    control_recall, challenger_recall = mean_class_recalls(control["folds"]), mean_class_recalls(challenger["folds"])
    recall_drops = {label: (None if control_recall[label] is None or challenger_recall[label] is None
                            else control_recall[label] - challenger_recall[label]) for label in COMMON_FOUR_LABELS}
    gates = {
        "positive_mean_difference": result["mean_difference"] > 0,
        "positive_ci95_lower_bound": result["ci95_low"] > 0,
        "no_missing_class_recall": all(value is not None for value in challenger_recall.values()),
        "no_zero_mean_class_recall": all(value not in (None, 0.0) for value in challenger_recall.values()),
        "no_class_recall_drop_over_0.03": all(drop is not None and drop <= RARE_RECALL_TOLERANCE
                                              for drop in recall_drops.values()),
    }
    control_rows = [row for unit in control["folds"].values() for row in unit["rows"]]
    challenger_rows = [row for unit in challenger["folds"].values() for row in unit["rows"]]
    result.update(
        status="complete",
        control_mean_class_recall=control_recall,
        challenger_mean_class_recall=challenger_recall,
        class_recall_drop=recall_drops,
        gates=gates,
        promoted=all(gates.values()),
        simultaneous_claim_supported=all(gates.values()) and result["bonferroni_low"] > 0,
        pooled_oof=group_bootstrap_pooled(control_rows, challenger_rows),
        label="exploratory predeclared contrast",
    )
    return result


def config_summary(entry: dict[str, Any]) -> dict[str, Any]:
    config: GridConfig = entry["config"]
    summary: dict[str, Any] = {"config_id": config.config_id, "family": config.family, "target": config.target,
                               "readout": config.readout, "complete": entry["complete"],
                               "n_completed_folds": len(entry["folds"]), "missing_units": entry["missing"]}
    if not entry["folds"]:
        return summary
    bas = [unit["metrics"]["common4"]["balanced_accuracy"] for unit in entry["folds"].values()]
    summary["mean_fold_common4_ba"] = float(np.mean(bas)) if entry["complete"] else None
    summary["sd_fold_common4_ba"] = float(np.std(bas, ddof=1)) if entry["complete"] and len(bas) > 1 else None
    summary["mean_fold_common4_macro_f1"] = (
        float(np.mean([unit["metrics"]["common4"]["macro_f1"] for unit in entry["folds"].values()]))
        if entry["complete"] else None)
    summary["mean_class_recall"] = mean_class_recalls(entry["folds"]) if entry["complete"] else None
    if entry["complete"]:
        rows = [row for unit in entry["folds"].values() for row in unit["rows"]]
        pooled = prediction_metrics(rows, native_labels=native_labels_for(config))
        summary["pooled_oof_common4"] = pooled["common4"]
        summary["pooled_oof_native"] = pooled["native"]
        if config.target == "six_class":
            summary["mean_fold_native_recall_fe_co_ni"] = {
                label: float(np.mean([unit["metrics"]["native"]["recall"][label] for unit in entry["folds"].values()]))
                for label in ("Fe", "Co", "Ni")}
    return summary


def pmm_summary(paths: CampaignPaths) -> dict[str, Any] | None:
    pmm_dir = paths.root / "pmm_comparator"
    manifest = pmm_dir / "pmm_comparator_manifest.json"
    if not manifest.is_file():
        return None
    verified = verify_comparator_outputs(paths.root)
    folds = {}
    for fold in range(N_FOLDS):
        path = pmm_dir / f"fold{fold}_predictions.csv"
        if path.is_file():
            rows = read_prediction_rows(path)
            folds[fold] = {"rows": rows, "metrics": prediction_metrics(rows, native_labels=None)}
    complete = len(folds) == N_FOLDS
    out = {"complete": complete, "manifest": verified, "folds": folds}
    if complete:
        out["mean_fold_common4_ba"] = float(np.mean([unit["metrics"]["common4"]["balanced_accuracy"]
                                                     for unit in folds.values()]))
        out["mean_fold_common4_macro_f1"] = float(np.mean([unit["metrics"]["common4"]["macro_f1"]
                                                           for unit in folds.values()]))
        out["mean_class_recall"] = mean_class_recalls(folds)
        out["pooled_oof_common4"] = prediction_metrics(
            [row for unit in folds.values() for row in unit["rows"]], native_labels=None)["common4"]
    return out


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    for row in rows[1:]:
        fields += [key for key in row if key not in fields]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_report(paths: CampaignPaths, decision: dict[str, Any], pmm: dict[str, Any] | None) -> None:
    """Render measured validation evidence and the fixed system-level comparison contract."""
    cohort_hash = decision["cohort_sha256"]
    fold_hash = sha256_file(paths.fold_membership)
    records = []
    feature_sets = {
        "only_esm": "ESMC-600M residue embeddings (1152 dimensions); target-centered structural pocket membership",
        "only_gvp": "Conservative protein residue geometry; radius-only edges; masked external feature channels",
        "gvp_late_fusion": "Conservative GVP protein geometry plus graph-level late ESMC-600M fusion",
    }
    for config in grid_configs():
        records.append({"config_id": config.config_id, "system": "DeepMzyme", "training_target": config.target,
                        "reporting_endpoint": "Mn,Cu,Zn,Class VIII=Fe+Co+Ni", "features": feature_sets[config.family],
                        "readout": config.readout, "known_site_information": "target ion coordinate; protein structure",
                        "imbalance": "training-fold common-four equalized per-ion loss weights; ordinary shuffle",
                        "preprocessing": "normalization fitted on training fold only",
                        "checkpoint_rule": "earliest highest native validation BA",
                        "cohort_sha256": cohort_hash, "fold_membership_sha256": fold_hash,
                        "paper_protocol_parity_verified": False})
    pmm_features = pmm["manifest"]["feature_columns"] if pmm is not None else None
    records.append({"config_id": "pmm_released_recipe", "system": "PinMyMetal released recipe",
                    "training_target": "four_class", "reporting_endpoint": "Mn,Cu,Zn,Class VIII=Fe+Co+Ni",
                    "features": (json.dumps(pmm_features) if pmm_features is not None else "not yet verified"),
                    "readout": "soft-voting ensemble", "known_site_information": "released known-site feature table",
                    "imbalance": "published internal estimator resampling confined to training fold",
                    "preprocessing": "dropna then published feature selection; no scaling",
                    "checkpoint_rule": "fixed released recipe; one fit per frozen training fold",
                    "cohort_sha256": cohort_hash, "fold_membership_sha256": fold_hash,
                    "paper_protocol_parity_verified": False})
    write_rows(paths.root / "comparison_protocol.csv", records)

    def percent(value):
        return "—" if value is None else f"{100 * float(value):.3f}%"

    def points(value):
        return "—" if value is None else f"{100 * float(value):+.3f} pp"

    lines = ["# PMM ion-level validation comparison", "", f"Status: **{decision['status']}**.", "",
             "This is a matched classification comparison at known ion sites. DeepMzyme and PMM use different "
             "feature recipes and model capacities; published paper scores are contextual, not matched controls.", "",
             "All candidates use the same eligible source ions and five PDB-grouped folds. Neural checkpoints are "
             "selected by native validation balanced accuracy; comparisons use the common four-class endpoint. "
             "The design has one model seed (42), and these validation-selected estimates are not nested estimates.", "",
             "| Configuration | Complete | Mean fold common-four BA | Mean fold macro-F1 |",
             "|---|---|---:|---:|"]
    for config in decision["configs"]:
        lines.append(f"| {config['config_id']} | {config['complete']} | {percent(config.get('mean_fold_common4_ba'))} | "
                     f"{percent(config.get('mean_fold_common4_macro_f1'))} |")
    if pmm is not None:
        lines.append(f"| PMM released recipe | {pmm['complete']} | {percent(pmm.get('mean_fold_common4_ba'))} | "
                     f"{percent(pmm.get('mean_fold_common4_macro_f1'))} |")
    lines += ["", "Mean-fold BA and pooled OOF BA are distinct estimands. Class support, recalls and confusion matrices "
              "are in `cv_fold_metrics.csv`; prediction identities and probabilities are in `cv_oof_predictions.csv`.", "",
              "| Predeclared contrast | Mean BA difference | Paired 95% interval | Promoted |",
              "|---|---:|---|---|"]
    for contrast in decision["predeclared_contrasts"]:
        interval = (f"[{points(contrast.get('ci95_low'))}, {points(contrast.get('ci95_high'))}]"
                    if contrast["status"] == "complete" else "incomplete")
        lines.append(f"| {contrast['contrast']} | {points(contrast.get('mean_difference'))} | {interval} | {contrast['promoted']} |")
    lines += ["", "Promotion requires a positive paired fold-bootstrap lower bound and the predeclared per-class recall "
              "protection. Six contrasts remain exploratory; simultaneous claims additionally require the saved "
              "Bonferroni bound. See `validation_decision.json` for every gate and pooled whole-group uncertainty.", "",
              "Protein symmetry-context exclusions define a matched subset. Remaining inputs use all local protein chains "
              "of the frozen asymmetric unit. Complete historical source assembly/paper-protocol parity is unverified.", "",
              "Reference-report gate: validation assessment alone does not authorize reference data access. "
              "One selected evaluated configuration and PMM must complete their frozen full-training refits before "
              "the separately recorded one-shot secondary reference report. The primary final-test route remains separate.", ""]
    if decision["completion_blockers"] or decision["incomplete_units"]:
        lines += ["Outstanding work: " + "; ".join(decision["completion_blockers"] +
                                                   [f"{len(decision['incomplete_units'])} neural units incomplete"]), ""]
    (paths.root / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def assess_campaign(paths: CampaignPaths, train_dir: Path) -> dict[str, Any]:
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    cohort_uids = [b.source_uid for b in read_cohort_csv(paths.cohort, expected_sha256=manifest["cohort"]["sha256"])]
    collected = collect_units(paths, train_dir)
    trees = {unit["receipt"]["campaign_run_identity"]["source_tree_sha256"]
             for entry in collected.values() for unit in entry["folds"].values()}
    if len(trees) > 1:
        raise ValueError(f"Completed units were produced by {len(trees)} different code trees; freeze the code")
    reconciliation_mismatches = [unit["run_name"] for entry in collected.values() for unit in entry["folds"].values()
                                 if unit["receipt"].get("reconciliation_status") == "mismatch"]
    if reconciliation_mismatches:
        raise ValueError("Cannot assess unreconciled selected checkpoints: " + ", ".join(reconciliation_mismatches))
    for entry in collected.values():
        if entry["complete"]:
            check_oof_coverage(cohort_uids, entry["folds"])

    fold_rows, oof_rows = [], []
    for entry in collected.values():
        config = entry["config"]
        for fold, unit in sorted(entry["folds"].items()):
            c4, native = unit["metrics"]["common4"], unit["metrics"]["native"]
            row = {"config_id": config.config_id, "family": config.family, "target": config.target,
                   "readout": config.readout, "fold": fold, "seed": MODEL_SEEDS[0],
                   "selected_epoch": unit["receipt"]["selected_epoch"],
                   "checkpoint_sha256": unit["receipt"]["selected_checkpoint_sha256"], "n_val": unit["metrics"]["n"],
                   "common4_balanced_acc": c4["balanced_accuracy"], "common4_macro_f1": c4["macro_f1"],
                   "common4_min_recall": c4["min_recall"], "native_balanced_acc": native["balanced_accuracy"],
                   "native_macro_f1": native["macro_f1"]}
            for label in COMMON_FOUR_LABELS:
                row[f"common4_recall_{label.replace(' ', '_')}"] = c4["recall"][label]
                row[f"common4_support_{label.replace(' ', '_')}"] = c4["support"][label]
            for label in native_labels_for(config):
                row[f"native_recall_{label.replace(' ', '_')}"] = native["recall"][label]
            row["common4_confusion_matrix"] = json.dumps(c4["confusion_matrix"])
            fold_rows.append(row)
            for prediction in unit["rows"]:
                oof_rows.append({"config_id": config.config_id, **prediction})
    pmm = pmm_summary(paths)
    if pmm is not None:
        for fold, unit in sorted(pmm["folds"].items()):
            c4 = unit["metrics"]["common4"]
            row = {"config_id": "pmm_released_recipe", "family": "pmm", "target": "four_class", "readout": "n/a",
                   "fold": fold, "seed": "n/a", "n_val": unit["metrics"]["n"],
                   "common4_balanced_acc": c4["balanced_accuracy"], "common4_macro_f1": c4["macro_f1"],
                   "common4_min_recall": c4["min_recall"]}
            for label in COMMON_FOUR_LABELS:
                row[f"common4_recall_{label.replace(' ', '_')}"] = c4["recall"][label]
                row[f"common4_support_{label.replace(' ', '_')}"] = c4["support"][label]
            row["common4_confusion_matrix"] = json.dumps(c4["confusion_matrix"])
            fold_rows.append(row)
            for prediction in unit["rows"]:
                oof_rows.append({"config_id": "pmm_released_recipe", **prediction})

    contrasts = [evaluate_contrast(name, collected[control.config_id], collected[challenger.config_id])
                 for name, control, challenger in PREDECLARED_CONTRASTS]
    delta_rows = []
    for contrast in contrasts:
        row = {key: contrast.get(key) for key in ("contrast", "control", "challenger", "status", "mean_difference",
                                                  "ci95_low", "ci95_high", "bonferroni_low", "bonferroni_high",
                                                  "promoted", "simultaneous_claim_supported")}
        row["fold_differences"] = json.dumps(contrast.get("fold_differences"))
        row["class_recall_drop"] = json.dumps(contrast.get("class_recall_drop"))
        row["gates"] = json.dumps(contrast.get("gates"))
        delta_rows.append(row)

    pmm_descriptive = []
    if pmm is not None and pmm["complete"]:
        for entry in collected.values():
            if not entry["complete"]:
                continue
            differences = np.array([entry["folds"][k]["metrics"]["common4"]["balanced_accuracy"]
                                    - pmm["folds"][k]["metrics"]["common4"]["balanced_accuracy"] for k in range(N_FOLDS)])
            pmm_descriptive.append({"config_id": entry["config"].config_id, "vs": "pmm_released_recipe",
                                    **fold_bootstrap(differences),
                                    "label": "descriptive; different inputs; not a promotion contrast"})

    decisions = {}
    by_id = {contrast["challenger"]: contrast for contrast in contrasts}
    for family in sorted({config.family for config in grid_configs()}):
        six = by_id[f"{family}__six_class__none"]
        aware = by_id[f"{family}__four_class__first_shell_bias"]
        decisions[family] = {
            "target_formulation": ("six_class (promoted on common-four validation)" if six["promoted"]
                                   else "four_class (retained: six_class not promoted)" if six["status"] == "complete"
                                   else "undecided (incomplete units)"),
            "readout": ("first_shell_bias (promoted with direct four_class only)" if aware["promoted"]
                        else "none (retained: first_shell_bias not promoted)" if aware["status"] == "complete"
                        else "undecided (incomplete units)"),
        }
    summaries = [config_summary(entry) for entry in collected.values()]
    decision = {
        "campaign_id": manifest["campaign_id"],
        "cohort_sha256": manifest["cohort"]["sha256"],
        "estimand": "mean common-four balanced accuracy over five selected fold checkpoints (one seed)",
        "status": "complete" if (all(entry["complete"] for entry in collected.values())
                                    and pmm is not None and pmm["complete"] and not reconciliation_mismatches) else "incomplete",
        "neural_grid_complete": all(entry["complete"] for entry in collected.values()),
        "completion_blockers": ([] if pmm is not None and pmm["complete"] else ["PMM comparator incomplete"])
                               + (["Checkpoint replay mismatch"] if reconciliation_mismatches else []),
        "incomplete_units": [name for entry in collected.values() for name in entry["missing"]],
        "code_tree_sha256": next(iter(trees), None),
        "replay_reconciliation_mismatches": reconciliation_mismatches,
        "configs": summaries,
        "predeclared_contrasts": contrasts,
        "decisions": decisions,
        "pmm_comparator": ({key: pmm[key] for key in ("complete", "mean_fold_common4_ba", "mean_fold_common4_macro_f1",
                                                     "mean_class_recall", "pooled_oof_common4") if key in pmm}
                           if pmm is not None else {"status": "not run"}),
        "pmm_descriptive_differences": pmm_descriptive,
        "caveats": [
            "validation-selected, one-seed grouped-fold CV; not a nested estimate of the selection process",
            "not PMM's original folds; not seed-stability evidence; not remote-protein generalization",
            "PDB-grouped folds are not homology-disjoint",
            "no held-out test data were read in this validation chain",
        ],
        "held_out_test_accessed": False,
    }
    paths.root.joinpath("analysis").mkdir(exist_ok=True)
    write_rows(paths.root / "cv_fold_metrics.csv", fold_rows)
    write_rows(paths.root / "cv_oof_predictions.csv", oof_rows)
    write_rows(paths.root / "cv_paired_deltas.csv", delta_rows)
    write_json(paths.root / "validation_decision.json", decision)
    write_comparison_report(paths, decision, pmm)
    return decision
