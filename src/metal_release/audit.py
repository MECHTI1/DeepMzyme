"""Reconstruct fixed metal ensembles from hash-bound development predictions only.

This is an evidence audit, not a selection or promotion procedure. Its confidence
intervals condition on the already chosen candidates and do not correct selection
optimism. No model, dataset loader, held-out evaluator, or network client is used.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import (
    COMMON4, NATIVE, ReleaseError, aggregate_probabilities, candidate_member_ids, development_artifact, require,
    sha256_file, validate_manifest,
)

SCHEMES = {
    "four_class": {"four_class", "merge_fe_class_viii"},
    "five_class": {"five_class"},
    "six_class": {"six_class", "split_all_metals"},
}
ELEMENTS = ("MN", "CU", "ZN", "FE", "CO", "NI")
PROBABILITY_TOLERANCE = 2e-6  # Existing validation exports use eight decimals.
CONFIDENCE_THRESHOLDS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
SUMMARY_CONFIDENCE = 0.9
ECE_BINNING = "10 equal-width bins of the maximum common-four probability on [0, 1]; last bin closed"
# An absolute path starts after whitespace, punctuation or a quote, never inside a word such as "Fe/Co".
ABSOLUTE_PATH = re.compile(r"(?<![\w.~-])/[^\s'\",\]]+")
ACCURACY_COLUMN = f"accuracy_percent_at_max_probability_{SUMMARY_CONFIDENCE:g}"
COVERAGE_COLUMN = f"coverage_percent_at_max_probability_{SUMMARY_CONFIDENCE:g}"
SIMPLICITY_RULE = ("Plan section 7 sequential rule: start from each simplest candidate and move to the least complex "
                   "more complex candidate whose paired 95% CI of OOF balanced-accuracy advantage over the current "
                   "choice excludes zero; stop when none does. Reported for the fold bootstrap, the related-group "
                   "bootstrap and both together. Complexity order: checkpoints, then members per OOF example.")


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(required <= set(reader.fieldnames or ()),
                f"Missing columns in {path.name}: {sorted(required - set(reader.fieldnames or ()))}")
        rows = list(reader)
    require(bool(rows), f"Empty CSV: {path}")
    return rows


def _membership(manifest: dict) -> list[dict[str, str]]:
    path = development_artifact(manifest.get("fold_membership"), "fold_membership")
    rows = _csv(path, {"source_uid", "physical_ion_id", "group_id", "native_element", "fold"})
    uids = [r["source_uid"] for r in rows]
    physical = [r["physical_ion_id"] for r in rows]
    require(all(uids) and len(set(uids)) == len(uids), "Duplicate or empty membership source_uid")
    require(all(physical) and len(set(physical)) == len(physical), "Duplicate or empty physical ion")
    require({int(r["fold"]) for r in rows} == set(range(5)),
            "Audit requires the complete frozen five-fold membership (folds 0..4)")
    grouped: dict[tuple[str, str], int] = {}
    for row in rows:
        require(row["native_element"].upper() in ELEMENTS, "Unsupported membership native element")
        for field in ("group_id", "pdbid", "near_copy_group", "cluster_id"):
            if field not in row:
                continue
            value = row[field]
            require(bool(value), f"Missing membership {field}")
            key = (field, value)
            require(key not in grouped or grouped[key] == int(row["fold"]),
                    f"{field} {value} crosses training/validation folds")
            grouped[key] = int(row["fold"])
    return sorted(rows, key=lambda r: r["source_uid"])


def _target_index(element: str, target: str) -> int:
    index = ELEMENTS.index(element.upper())
    return min(index, {"four_class": 3, "five_class": 4, "six_class": 5}[target])


def _probabilities(rows: list[dict[str, str]], prefix: str, names: tuple[str, ...]) -> np.ndarray:
    value = np.array([[float(row[f"{prefix}_{name.replace(' ', '_')}"]) for name in names]
                      for row in rows], dtype=np.float64)
    require(np.isfinite(value).all() and (value >= 0).all() and (value <= 1).all()
            and np.all(np.abs(value.sum(axis=1) - 1) <= PROBABILITY_TOLERANCE),
            f"Invalid {prefix} probabilities")
    return value


def _metrics(y: np.ndarray, probs: np.ndarray, names: tuple[str, ...] = COMMON4) -> dict:
    pred = probs.argmax(axis=1)
    matrix = np.zeros((len(names), len(names)), dtype=np.int64)
    np.add.at(matrix, (y, pred), 1)
    support = matrix.sum(axis=1)
    recalls = np.divide(matrix.diagonal(), support, out=np.full(len(names), np.nan), where=support > 0)
    return {
        "n": int(len(y)), "accuracy": float(np.mean(pred == y)) if len(y) else None,
        "balanced_accuracy": float(np.mean(recalls[np.isfinite(recalls)])) if np.isfinite(recalls).any() else None,
        "all_classes_present": bool((support > 0).all()),
        "recall": {name: float(r) if np.isfinite(r) else None for name, r in zip(names, recalls)},
        "support": dict(zip(names, map(int, support))), "confusion_matrix": matrix.tolist(),
        "native_label_note": "Five-class Class VIII means Co+Ni; common-four Class VIII means Fe+Co+Ni.",
    }


def _validate_retained_split(member: dict, membership: list[dict], predictions: list[dict], fold: int) -> None:
    metadata = member.get("metadata", {})
    ref = metadata.get("dataset_summary.json") or metadata.get("run_metadata.json")
    payload = _json(development_artifact(ref, "retained development split metadata"))
    summary = payload.get("dataset_summary", payload)
    retained = summary.get("retained_split_identity", {})
    train, validation = retained.get("train", {}), retained.get("validation", {})
    expected_val = [row for row in membership if int(row["fold"]) == fold]
    expected_train = [row for row in membership if int(row["fold"]) != fold]
    for partition, expected in ((train, expected_train), (validation, expected_val)):
        examples = partition.get("examples")
        require(isinstance(examples, list) and partition.get("n_examples") == len(expected)
                and len(examples) == len(expected), "Retained split example count differs")
        raw = json.dumps(examples, sort_keys=True, separators=(",", ":")).encode()
        require(hashlib.sha256(raw).hexdigest() == partition.get("ordered_examples_sha256"),
                "Retained split ordered-example hash differs")
        ids = [(r["structure_id"], r["pocket_id"]) for r in examples]
        require(len(ids) == len(set(ids)), "Duplicate retained split example")
        expected_groups = {row.get("pdbid", row["group_id"]) for row in expected}
        require({row["group"] for row in examples} == expected_groups,
                "Retained split PDB groups differ from frozen fold eligibility")
    train_ids = {(r["structure_id"], r["pocket_id"]) for r in train["examples"]}
    val_ids = {(r["structure_id"], r["pocket_id"]) for r in validation["examples"]}
    require(not train_ids & val_ids, "A validation example occurs in retained training examples")
    require(val_ids == {(r["structure_id"], r["example_id"]) for r in predictions},
            "Validation exports differ from retained validation examples")
    verification = summary.get("fold_membership", {})
    identity_text = "\n".join(
        f"{r['source_uid']}\t{r['physical_ion_id']}\t{r['group_id']}\t{int(r['fold'])}" for r in membership)
    require(verification.get("verified") is True and verification.get("fold_index") == fold
            and verification.get("n_train") == len(expected_train) and verification.get("n_val") == len(expected_val)
            and verification.get("membership_identity_sha256") == hashlib.sha256(identity_text.encode()).hexdigest(),
            "Retained split lacks matching frozen-membership verification")


def _verify_refit(manifest: dict, member_id: str, membership: list[dict]) -> dict:
    member = manifest["members"][member_id]
    paths = {role: development_artifact(member.get(role), f"{member_id}/{role}")
             for role in ("checkpoint", "run_config", "selected_checkpoint")}
    payload, receipt = _json(paths["run_config"]), _json(paths["selected_checkpoint"])
    config = payload.get("config", payload)
    identity = config.get("campaign_run_identity")
    if isinstance(identity, str):
        identity = json.loads(identity)
    require(isinstance(identity, dict) and identity.get("phase") == "stage6b_full_non_test_refit"
            and identity.get("validation") == "none" and identity.get("model_seed") == member["seed"]
            and identity.get("target_scheme") == member["target"], "Refit configuration identity differs")
    require(config.get("run_test_eval") is False and not config.get("test_structure_dir")
            and not config.get("test_summary_csv") and config.get("fold_index") is None
            and config.get("metal_example_unit") == "ion" and config.get("task") == "metal"
            and config.get("checkpoint_rule") == "terminal", "Refit is not the frozen development-only source")
    require(receipt.get("passed") is True and receipt.get("parameters_finite") is True
            and receipt.get("n_ions") == len(membership)
            and receipt.get("checkpoint_sha256") == member["checkpoint"]["sha256"],
            "Refit reload receipt does not bind the full-cohort checkpoint")
    return {"status": "artifact_verified", "interpretation": "Full non-test refit; no OOF predictions",
            "checkpoint_sha256": member["checkpoint"]["sha256"]}


def _load_member(manifest: dict, member_id: str, membership: list[dict[str, str]]) -> dict:
    member = manifest["members"][member_id]
    require(member.get("fold") in range(5), f"{member_id}: full refit has no OOF predictions")
    paths = {role: development_artifact(member.get(role), f"{member_id}/{role}")
             for role in ("checkpoint", "run_config", "selected_checkpoint", "validation_predictions")}
    payload, receipt = _json(paths["run_config"]), _json(paths["selected_checkpoint"])
    config = payload.get("config", payload)
    target, fold, seed = member["target"], member["fold"], member["seed"]
    require(target in NATIVE, f"{member_id}: unsupported target")
    require(config.get("run_test_eval") is False and not config.get("test_structure_dir")
            and not config.get("test_summary_csv"), f"{member_id}: config is not development-only")
    require(config.get("fold_index") == fold and config.get("seed") == seed
            and config.get("metal_label_scheme") in SCHEMES[target],
            f"{member_id}: config fold/seed/target differs")
    require(config.get("metal_example_unit") == "ion" and config.get("task") == "metal",
            f"{member_id}: config does not describe metal ion classification")
    require(config.get("fold_split_source") == "membership"
            and config.get("fold_membership_sha256") == manifest["fold_membership"]["sha256"],
            f"{member_id}: config does not bind the frozen development membership")
    require(config.get("checkpoint_rule") == "terminal" and receipt.get("checkpoint_rule") == "terminal"
            and receipt.get("selected_epoch") == config.get("epochs")
            and receipt.get("completed_epochs") == config.get("epochs")
            and receipt.get("fit_status") == "completed" and receipt.get("reconciliation_status") == "match",
            f"{member_id}: terminal checkpoint receipt is incomplete or inconsistent")
    require(receipt.get("selected_checkpoint_sha256") == member["checkpoint"]["sha256"]
            and receipt.get("selected_checkpoint") == paths["checkpoint"].name,
            f"{member_id}: receipt checkpoint binding differs")
    prediction_receipt = receipt.get("validation_predictions", {})
    require(prediction_receipt.get("sha256") == member["validation_predictions"]["sha256"]
            and prediction_receipt.get("path") == paths["validation_predictions"].name,
            f"{member_id}: receipt validation-prediction binding differs")
    identity = config.get("campaign_run_identity")
    if isinstance(identity, str):
        identity = json.loads(identity)
    require(isinstance(identity, dict) and identity == receipt.get("campaign_run_identity"),
            f"{member_id}: config and receipt campaign identity differ")
    require(identity.get("fold") == fold and identity.get("model_seed") == seed
            and identity.get("target_scheme") == target
            and identity.get("fold_membership_sha256") == manifest["fold_membership"]["sha256"],
            f"{member_id}: campaign identity does not bind fold/seed/target/membership")

    required = {"source_uid", "physical_ion_id", "group_id", "native_element", "fold", "model_seed", "structure_id", "example_id",
                "metal_label_scheme", "selected_epoch", "checkpoint_sha256", "y_native", "pred_native",
                "y_common4", "pred_common4"}
    required.update(f"p_native_{name.replace(' ', '_')}" for name in NATIVE[target])
    required.update(f"p_common4_{name.replace(' ', '_')}" for name in COMMON4)
    rows = _csv(paths["validation_predictions"], required)
    expected = {r["source_uid"]: r for r in membership if int(r["fold"]) == fold}
    uids = [r["source_uid"] for r in rows]
    require(len(set(uids)) == len(uids) and set(uids) == set(expected),
            f"{member_id}: predictions must cover exactly its excluded fold once")
    require(prediction_receipt.get("n_rows") == len(rows), f"{member_id}: receipt row count differs")
    rows.sort(key=lambda row: row["source_uid"])
    _validate_retained_split(member, membership, rows, fold)
    for row in rows:
        item = expected[row["source_uid"]]
        require(row["physical_ion_id"] == item["physical_ion_id"]
                and row["group_id"] == item.get("pdbid", item["group_id"])
                and row["native_element"].upper() == item["native_element"].upper(),
                f"{member_id}: prediction example identity differs from membership")
        require(int(row["fold"]) == fold and int(row["model_seed"]) == seed
                and row["metal_label_scheme"] in SCHEMES[target]
                and int(row["selected_epoch"]) == receipt["selected_epoch"]
                and row["checkpoint_sha256"] == member["checkpoint"]["sha256"],
                f"{member_id}: prediction checkpoint/fold/seed/target binding differs")
        require(int(row["y_native"]) == _target_index(item["native_element"], target)
                and int(row["y_common4"]) == _target_index(item["native_element"], "four_class"),
                f"{member_id}: prediction labels differ from frozen membership")
    native = _probabilities(rows, "p_native", NATIVE[target])
    common = _probabilities(rows, "p_common4", COMMON4)
    collapsed = np.column_stack([native[:, :3], native[:, 3:].sum(axis=1)])
    require(np.all(np.abs(common - collapsed) <= PROBABILITY_TOLERANCE),
            f"{member_id}: invalid native-to-common-four probability collapse")
    for probs, key in ((native, "pred_native"), (common, "pred_common4")):
        predictions = np.array([int(r[key]) for r in rows])
        require(((predictions >= 0) & (predictions < probs.shape[1])).all()
                and np.all(probs.max(axis=1) - probs[np.arange(len(rows)), predictions] <= PROBABILITY_TOLERANCE),
                f"{member_id}: {key} disagrees with probabilities")
    native_y = np.array([int(r["y_native"]) for r in rows])
    native_metrics = _metrics(native_y, native, NATIVE[target])
    common_metrics = _metrics(np.array([int(r["y_common4"]) for r in rows]), common)
    for key, metric in (("val_metal_balanced_acc", native_metrics["balanced_accuracy"]),
                        ("val_metal_collapsed4_balanced_acc", common_metrics["balanced_accuracy"])):
        recorded = receipt.get("metrics", {}).get(key)
        require(metric is not None and isinstance(recorded, (float, int))
                and abs(recorded - metric) <= 1e-9, f"{member_id}: {key} receipt metric does not reconcile")
    return {"common": common, "native_metrics": native_metrics,
            "target": target, "fold": fold, "seed": seed, "n": len(rows),
            "checkpoint_sha256": member["checkpoint"]["sha256"]}


def _reliability_bins(confidence: np.ndarray, correct: np.ndarray) -> tuple[list, float]:
    bins, ece = [], 0.0
    for i in range(10):
        lower, upper = i / 10, (i + 1) / 10
        mask = (confidence >= lower) & ((confidence < upper) if i < 9 else (confidence <= upper))
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else None
        mean_confidence = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / len(confidence) * abs(accuracy - mean_confidence)
        bins.append({"lower": lower, "upper": upper, "n": count,
                     "accuracy": accuracy, "mean_confidence": mean_confidence})
    return bins, ece


def _reliability(y: np.ndarray, probs: np.ndarray) -> dict:
    confidence, pred = probs.max(axis=1), probs.argmax(axis=1)
    correct = pred == y
    bins, ece = _reliability_bins(confidence, correct)
    by_class = {}
    for i, name in enumerate(COMMON4):
        class_bins, class_ece = _reliability_bins(probs[:, i], y == i)
        by_class[name] = {"interpretation": "one-vs-rest: accuracy field is observed class frequency",
                          "n_positive": int((y == i).sum()), "reliability_bins": class_bins,
                          "ece_10_equal_width_bins": class_ece,
                          "brier": float(np.square(probs[:, i] - (y == i)).mean())}
    coverage = []
    for threshold in CONFIDENCE_THRESHOLDS:
        keep = confidence >= threshold
        row: dict[str, Any] = {"threshold": threshold, "retained": int(keep.sum()),
                               "coverage": float(keep.mean()),
                               "accuracy": float(correct[keep].mean()) if keep.any() else None}
        row["by_true_class"] = {}
        row["by_predicted_class"] = {}
        for i, name in enumerate(COMMON4):
            for key, group in (("by_true_class", y == i), ("by_predicted_class", pred == i)):
                retained = keep & group
                row[key][name] = {"n": int(group.sum()), "retained": int(retained.sum()),
                                  "coverage": float(retained.sum() / group.sum()) if group.any() else None,
                                  "accuracy": float(correct[retained].mean()) if retained.any() else None}
        coverage.append(row)
    return {"status": "descriptive OOF diagnostics; no calibrator or threshold fitted",
            "nll": float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-15, 1)).mean()),
            "multiclass_brier": float(np.square(probs - np.eye(4)[y]).sum(axis=1).mean()),
            "ece_10_equal_width_bins": float(ece), "reliability_bins": bins, "by_class": by_class,
            "coverage": coverage}


def _candidate_oof(manifest: dict, candidate_id: str, members: dict, membership: list[dict]) -> np.ndarray:
    probabilities = np.empty((len(membership), 4), dtype=np.float64)
    folds = np.array([int(r["fold"]) for r in membership])
    candidate = manifest["candidates"][candidate_id]
    for group in candidate["groups"]:
        signatures = {(manifest["members"][mid]["target"], manifest["members"][mid]["seed"])
                      for mid in group["members"]}
        for target, seed in signatures:
            found = [manifest["members"][mid]["fold"] for mid in group["members"]
                     if (manifest["members"][mid]["target"], manifest["members"][mid]["seed"]) == (target, seed)]
            require(sorted(found) == list(range(5)),
                    f"{candidate_id}/{group['id']}: incomplete or duplicate target/seed fold set")
    for fold in range(5):
        available = {mid: members[mid]["common"] for mid in candidate_member_ids(manifest, candidate_id)
                     if manifest["members"][mid]["fold"] == fold}
        value = aggregate_probabilities(manifest, candidate_id, available, fold=fold)
        require(value.shape == (int((folds == fold).sum()), 4), f"{candidate_id}: OOF shape differs")
        probabilities[folds == fold] = value
    require(np.isfinite(probabilities).all() and (probabilities >= 0).all()
            and np.all(np.abs(probabilities.sum(axis=1) - 1) <= PROBABILITY_TOLERANCE),
            f"{candidate_id}: invalid normalized ensemble probabilities")
    return probabilities


def related_groups(rows: list[dict]) -> tuple[str, np.ndarray]:
    """Return the coarsest available related-group field and a dense group index per row."""
    group_field = next((field for field in ("near_copy_group", "cluster_id", "group_id")
                        if all(r.get(field) for r in rows)), None)
    require(group_field is not None, "Related-group membership is missing")
    _, ids = np.unique([r[group_field] for r in rows], return_inverse=True)
    return group_field, ids


def related_group_deltas(y: np.ndarray, candidate_correct: np.ndarray, reference_correct: np.ndarray,
                         group_ids: np.ndarray, samples: int, rng: np.random.Generator) -> tuple[list[float], int]:
    """Paired balanced-accuracy deltas over related groups resampled with replacement.

    Resamples missing any class are skipped and counted, because balanced accuracy is undefined there.
    """
    n_groups = int(group_ids.max()) + 1
    support = np.zeros((n_groups, 4), dtype=np.int64)
    correct_delta = np.zeros_like(support)
    np.add.at(support, (group_ids, y), 1)
    np.add.at(correct_delta, (group_ids, y),
              np.asarray(candidate_correct, dtype=int) - np.asarray(reference_correct, dtype=int))
    deltas, invalid = [], 0
    for start in range(0, samples, 128):
        indices = rng.integers(0, n_groups, size=(min(128, samples - start), n_groups))
        denominator = support[indices].sum(axis=1)
        numerator = correct_delta[indices].sum(axis=1)
        valid = (denominator > 0).all(axis=1)
        invalid += int((~valid).sum())
        deltas.extend((numerator[valid] / denominator[valid]).mean(axis=1).tolist())
    require(bool(deltas), "No related-group bootstrap resample supports all four classes")
    return deltas, invalid


def _calibration(reliability: dict) -> dict:
    """Compact public view of the descriptive reliability block."""
    row = next(item for item in reliability["coverage"] if item["threshold"] == SUMMARY_CONFIDENCE)
    return {"ece": reliability["ece_10_equal_width_bins"], "ece_binning": ECE_BINNING,
            "nll": reliability["nll"], "multiclass_brier": reliability["multiclass_brier"],
            "max_probability_threshold": SUMMARY_CONFIDENCE,
            "coverage_at_threshold": row["coverage"], "accuracy_at_threshold": row["accuracy"],
            "by_true_class_at_threshold": {name: {"coverage": value["coverage"], "accuracy": value["accuracy"]}
                                           for name, value in row["by_true_class"].items()}}


def sequential_walk(order: list[str], supported: set[tuple[str, str]], start: str) -> list[str]:
    """Path of Plan's sequential rule from `start`; `supported` holds (simpler, more complex) pairs whose
    interval excludes zero, and each step takes the least complex supported candidate in `order`."""
    path = [start]
    while True:
        later = order[order.index(path[-1]) + 1:]
        step = next((key for key in later if (path[-1], key) in supported), None)
        if step is None:
            return path
        path.append(step)


def _simplicity_rule(y: np.ndarray, outputs: dict[str, np.ndarray], candidates: dict, rows: list[dict],
                     samples: int, seed: int) -> dict:
    """Walk Plan's sequential rule over the fold ensembles from each simplest candidate.

    The answer depends on which paired interval is required, so all three readings are reported.
    """
    eligible = sorted((key for key in outputs if candidates[key]["oof_source"] == key),
                      key=lambda key: (candidates[key]["checkpoints"], candidates[key]["members_per_oof_example"], key))
    pairs = []
    for i, simpler in enumerate(eligible):
        for complex_ in eligible[i + 1:]:
            if candidates[complex_]["checkpoints"] == candidates[simpler]["checkpoints"]:
                continue
            contrast = _paired_comparison(y, outputs[complex_], outputs[simpler], rows, samples, seed)
            pairs.append({"simpler": simpler, "more_complex": complex_,
                          "pooled_delta_pp": contrast["pooled_delta_pp"],
                          "fold_bootstrap_ci95_pp": contrast["fold_bootstrap_ci95_pp"],
                          "related_group_ci95_pp": contrast["related_group_bootstrap"]["ci95_pp"],
                          "fold_interval_excludes_zero": contrast["fold_bootstrap_ci95"][0] > 0,
                          "related_group_interval_excludes_zero": contrast["related_group_bootstrap"]["ci95"][0] > 0})
    readings = {"both_intervals": ("fold_interval_excludes_zero", "related_group_interval_excludes_zero"),
                "related_group_interval": ("related_group_interval_excludes_zero",),
                "fold_interval": ("fold_interval_excludes_zero",)}
    fewest = min(candidates[key]["checkpoints"] for key in eligible) if eligible else None
    starts = [key for key in eligible if candidates[key]["checkpoints"] == fewest]
    walks = {}
    for reading, flags in readings.items():
        supported = {(pair["simpler"], pair["more_complex"]) for pair in pairs if all(pair[flag] for flag in flags)}
        walks[reading] = {start: sequential_walk(eligible, supported, start) for start in starts}
    return {"rule": SIMPLICITY_RULE, "complexity_order": eligible, "pairs": pairs, "walks": walks,
            "stopping_points": {reading: sorted({path[-1] for path in paths.values()}, key=eligible.index)
                                for reading, paths in walks.items()},
            "interpretation": "Descriptive OOF evidence on historically chosen candidates; not a promotion."}


def _paired_comparison(y: np.ndarray, candidate: np.ndarray, reference: np.ndarray,
                       rows: list[dict], samples: int, seed: int) -> dict:
    folds = np.array([int(r["fold"]) for r in rows])
    deltas, recall_deltas = [], []
    for fold in range(5):
        mask = folds == fold
        left, right = _metrics(y[mask], candidate[mask]), _metrics(y[mask], reference[mask])
        require(left["all_classes_present"] and right["all_classes_present"],
                "Every fold must support all four classes for paired fold comparison")
        recall_deltas.append([left["recall"][name] - right["recall"][name] for name in COMMON4])
        deltas.append(left["balanced_accuracy"] - right["balanced_accuracy"])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, 5, size=(samples, 5))
    fold_boot = np.array(deltas)[draws].mean(axis=1)
    recall_boot = np.array(recall_deltas)[draws].mean(axis=1)
    group_field, group_ids = related_groups(rows)
    group_boot, invalid = related_group_deltas(
        y, candidate.argmax(axis=1) == y, reference.argmax(axis=1) == y, group_ids, samples, rng)
    pooled_candidate, pooled_reference = _metrics(y, candidate), _metrics(y, reference)
    pooled_recall_delta = {name: pooled_candidate["recall"][name] - pooled_reference["recall"][name]
                           for name in COMMON4}
    return {
        "interpretation": "descriptive paired intervals, conditional on fixed historical candidates; not promotion evidence",
        "bootstrap_samples": samples, "bootstrap_seed": seed, "units": "balanced-accuracy fraction",
        "fold_mean_delta": float(np.mean(deltas)), "fold_deltas": deltas,
        "fold_mean_delta_pp": 100 * float(np.mean(deltas)),
        "fold_bootstrap_ci95": np.quantile(fold_boot, [0.025, 0.975]).tolist(),
        "fold_bootstrap_ci95_pp": (100 * np.quantile(fold_boot, [0.025, 0.975])).tolist(),
        "fold_mean_recall_delta": dict(zip(COMMON4, np.array(recall_deltas).mean(axis=0).tolist())),
        "fold_mean_recall_delta_pp": dict(zip(COMMON4, (100 * np.array(recall_deltas).mean(axis=0)).tolist())),
        "fold_recall_delta_ci95": {name: np.quantile(recall_boot[:, i], [0.025, 0.975]).tolist()
                                   for i, name in enumerate(COMMON4)},
        "pooled_delta": pooled_candidate["balanced_accuracy"] - pooled_reference["balanced_accuracy"],
        "pooled_delta_pp": 100 * (pooled_candidate["balanced_accuracy"] - pooled_reference["balanced_accuracy"]),
        "pooled_recall_delta_pp": {name: 100 * value for name, value in pooled_recall_delta.items()},
        "related_group_bootstrap": {"group_field": group_field, "n_groups": int(group_ids.max()) + 1,
                                    "valid_resamples": len(group_boot), "missing_class_resamples": invalid,
                                    "ci95": np.quantile(group_boot, [0.025, 0.975]).tolist(),
                                    "ci95_pp": (100 * np.quantile(group_boot, [0.025, 0.975])).tolist(),
                                    "one_sided95_lower_pp": 100 * float(np.quantile(group_boot, 0.05))},
        "prediction_disagreement_n": int((candidate.argmax(axis=1) != reference.argmax(axis=1)).sum()),
        "prediction_disagreement_rate": float((candidate.argmax(axis=1) != reference.argmax(axis=1)).mean()),
        "selection_optimism_corrected": False, "promotion_authorized": False,
    }


def audit(manifest: dict, output_dir: str | Path, bootstrap_samples: int = 10000, seed: int = 42,
          similarity_margin_pp: float = 1.5, recall_floor_pp: float = 3.0) -> dict:
    """Write an additive audit; missing reference evidence blocks every numeric score."""
    output = Path(output_dir)
    require(not output.exists(), f"Output directory already exists: {output}")
    require(isinstance(bootstrap_samples, int) and bootstrap_samples > 0, "bootstrap_samples must be positive")
    require(isinstance(seed, int) and seed >= 0, "seed must be a nonnegative integer")
    require(similarity_margin_pp >= 0 and recall_floor_pp >= 0, "Similarity margins must be nonnegative")
    report: dict[str, Any] = {
        "schema": "deepmzyme-metal-release-audit-v1", "status": "blocked",
        "reference": manifest["reference"], "held_out_accessed": False,
        "catalog_id": manifest.get("catalog_id"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "manifest_path": manifest.get("manifest_path"),
        "selection_provenance": manifest.get("selection_provenance"),
        "inventory_report": manifest.get("inventory_report"),
        "selection_or_promotion": False, "selection_optimism_corrected": False,
        "disclosure": "Historical fixed-shortlist OOF reconstruction; no unbiased estimate of the selected deployment ensemble is claimed.",
        "similarity_bar": {
            "definition": (f"One-sided 95% related-group bootstrap lower bound of (candidate - reference) pooled "
                           f"common-four BA >= -{similarity_margin_pp:g} pp, and no class's pooled recall more than "
                           f"{recall_floor_pp:g} pp below the reference."),
            "margin_pp": similarity_margin_pp, "class_recall_floor_pp": recall_floor_pp,
            "declared": "after the OOF results were seen; descriptive only, never a selection or promotion rule",
        },
        "blockers": [], "candidates": {}, "comparisons": {}, "member_diagnostics": {},
    }
    outputs: dict[str, np.ndarray] = {}
    members: dict[str, dict] = {}
    failures: dict[str, str] = {}
    rows: list[dict] = []
    try:
        validate_manifest(manifest)
        rows = _membership(manifest)
        require(manifest["reference"] in manifest["candidates"], "Reference candidate is absent")
        for candidate_id, candidate in manifest["candidates"].items():
            source = candidate.get("oof_proxy") or candidate_id
            require(source in manifest["candidates"], f"Unknown OOF proxy {source}")
            require(not manifest["candidates"][source].get("oof_proxy"), "Nested OOF proxies are not supported")
            if source != candidate_id:
                try:
                    for member_id in candidate_member_ids(manifest, candidate_id):
                        require(manifest["members"][member_id]["fold"] is None,
                                "Only full-data refits may use configuration OOF proxies")
                        _verify_refit(manifest, member_id, rows)
                except (ReleaseError, OSError, ValueError, KeyError, TypeError) as error:
                    report["candidates"][candidate_id] = {"status": "blocked", "error": str(error)}
                    continue
            for member_id in candidate_member_ids(manifest, source):
                if member_id in members or member_id in failures:
                    continue
                try:
                    members[member_id] = _load_member(manifest, member_id, rows)
                except (ReleaseError, OSError, ValueError, KeyError, TypeError) as error:
                    failures[member_id] = str(error)
            missing = {mid: failures[mid] for mid in candidate_member_ids(manifest, source) if mid in failures}
            if missing:
                report["candidates"][candidate_id] = {"status": "blocked", "member_errors": missing}
                continue
            try:
                outputs[candidate_id] = _candidate_oof(manifest, source, members, rows)
                report["candidates"][candidate_id] = {
                    "status": "verified", "oof_source": source,
                    "checkpoints": len(candidate_member_ids(manifest, candidate_id)),
                    # Complete fold sets per seed make this count identical for every fold.
                    "members_per_oof_example": sum(manifest["members"][mid]["fold"] == 0
                                                   for mid in candidate_member_ids(manifest, source)),
                    "interpretation": ("configuration OOF proxy only; no deployed full-refit OOF score"
                                       if source != candidate_id else
                                       "fold-eligible OOF configuration mixture; deployment averages additional fold models"),
                }
            except (ReleaseError, ValueError, KeyError, TypeError) as error:
                report["candidates"][candidate_id] = {"status": "blocked", "error": str(error)}
        if manifest["reference"] not in outputs:
            report["blockers"].append("Reference OOF evidence is incomplete or invalid; all numeric scores suppressed")
            outputs.clear()
        else:
            y = np.array([_target_index(row["native_element"], "four_class") for row in rows])
            folds = np.array([int(row["fold"]) for row in rows])
            for member_id, member in members.items():
                report["member_diagnostics"][member_id] = {k: v for k, v in member.items() if k != "common"}
            for candidate_id, probabilities in outputs.items():
                candidate_report = report["candidates"][candidate_id]
                candidate_report["common_four"] = _metrics(y, probabilities)
                candidate_report["folds"] = {str(f): _metrics(y[folds == f], probabilities[folds == f]) for f in range(5)}
                require(all(item["all_classes_present"] for item in candidate_report["folds"].values()),
                        "Every development fold must contain all four reporting classes")
                candidate_report["fold_mean_balanced_accuracy"] = float(np.mean([
                    value["balanced_accuracy"] for value in candidate_report["folds"].values()]))
                candidate_report["reliability"] = _reliability(y, probabilities)
                ordered = np.sort(probabilities, axis=1)
                candidate_report["near_ties_within_export_tolerance"] = int(
                    ((ordered[:, -1] - ordered[:, -2]) <= PROBABILITY_TOLERANCE).sum())
                reported = manifest["candidates"][candidate_id].get("reported_oof_ba_percent")
                if reported is not None:
                    actual = 100 * candidate_report["common_four"]["balanced_accuracy"]
                    candidate_report["historical_score_reconciliation"] = {
                        "reported_percent": reported, "reconstructed_percent": actual,
                        "matches_reported_rounding": abs(actual - reported) <= 0.00500001,
                    }
                contrast = _paired_comparison(
                    y, probabilities, outputs[manifest["reference"]], rows, bootstrap_samples, seed)
                if candidate_id != manifest["reference"]:
                    contrast["meets_similarity_bar"] = bool(
                        contrast["related_group_bootstrap"]["one_sided95_lower_pp"] >= -similarity_margin_pp
                        and min(contrast["pooled_recall_delta_pp"].values()) >= -recall_floor_pp)
                    # The reference is the more complex model; its advantage needs both intervals below zero.
                    contrast["reference_advantage_supported"] = bool(
                        contrast["fold_bootstrap_ci95"][1] < 0 and contrast["related_group_bootstrap"]["ci95"][1] < 0)
                report["comparisons"][candidate_id] = contrast
            report["simplicity_rule"] = _simplicity_rule(y, outputs, report["candidates"], rows,
                                                         bootstrap_samples, seed)
            blocked = [key for key, value in report["candidates"].items() if value["status"] == "blocked"]
            report["blockers"].extend(f"Candidate evidence unavailable: {key}" for key in blocked)
            report["status"] = "blocked" if blocked else "complete"
            report["n_examples"] = len(rows)
            report["related_group_field"], group_ids = related_groups(rows)
            report["n_related_groups"] = int(group_ids.max()) + 1
    except (ReleaseError, OSError, ValueError, KeyError, TypeError) as error:
        report["blockers"].append(str(error))
        report["status"] = "blocked"
        # A top-level eligibility failure must never leave a partial numeric report.
        report["comparisons"], report["member_diagnostics"] = {}, {}
        report.pop("simplicity_rule", None)
        report["candidates"] = {key: {"status": "blocked", "error": "Audit prerequisite failed"}
                                for key in manifest.get("candidates", {})}
        outputs.clear()
    output.mkdir(parents=True, exist_ok=False)
    if outputs:
        _write_comparison(output / "comparison.csv", report, manifest["reference"])
        with (output / "development_oof_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            fieldnames = ["candidate", "source_uid", "physical_ion_id", "group_id", "fold", "native_element",
                          "y_common4", "pred_common4", *(f"p_common4_{name.replace(' ', '_')}" for name in COMMON4)]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for candidate_id, probabilities in outputs.items():
                for row, probability in zip(rows, probabilities):
                    record = {key: row[key] for key in ("source_uid", "physical_ion_id", "group_id", "fold", "native_element")}
                    record.update(candidate=candidate_id, y_common4=_target_index(row["native_element"], "four_class"),
                                  pred_common4=int(probability.argmax()))
                    record.update({f"p_common4_{name.replace(' ', '_')}": float(p) for name, p in zip(COMMON4, probability)})
                    writer.writerow(record)
    (output / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = public_summary(report, {name: output / name for name in
                                      ("audit.json", "comparison.csv", "development_oof_predictions.csv")})
    (output / "audit_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                                               encoding="utf-8")
    return report


def _percent(value: float | None) -> float | None:
    return None if value is None else 100 * value


def _write_comparison(path: Path, report: dict, reference: str) -> None:
    """One row per candidate, simplest first; deltas are candidate minus reference."""
    fields = ["candidate", "checkpoints", "members_per_oof_example", "pooled_ba_percent", "fold_mean_ba_percent",
              "pooled_delta_pp", "fold_mean_delta_pp", "fold_delta_ci95_lower_pp", "fold_delta_ci95_upper_pp",
              "related_group_delta_ci95_lower_pp", "related_group_delta_ci95_upper_pp",
              "related_group_delta_one_sided95_lower_pp",
              *(f"pooled_recall_delta_pp_{name.replace(' ', '_')}" for name in COMMON4),
              "ece", ACCURACY_COLUMN, COVERAGE_COLUMN,
              "meets_similarity_bar", "reference_advantage_supported", "disagreement_n", "interpretation"]
    candidates = report["candidates"]
    order = sorted(report["comparisons"], key=lambda key: (
        candidates[key]["checkpoints"], candidates[key]["members_per_oof_example"], key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for candidate_id in order:
            item, contrast = candidates[candidate_id], report["comparisons"][candidate_id]
            calibration = _calibration(item["reliability"])
            group = contrast["related_group_bootstrap"]
            row = {"candidate": candidate_id, "checkpoints": item["checkpoints"],
                   "members_per_oof_example": item["members_per_oof_example"],
                   "pooled_ba_percent": 100 * item["common_four"]["balanced_accuracy"],
                   "fold_mean_ba_percent": 100 * item["fold_mean_balanced_accuracy"],
                   "pooled_delta_pp": contrast["pooled_delta_pp"],
                   "fold_mean_delta_pp": contrast["fold_mean_delta_pp"],
                   "fold_delta_ci95_lower_pp": contrast["fold_bootstrap_ci95_pp"][0],
                   "fold_delta_ci95_upper_pp": contrast["fold_bootstrap_ci95_pp"][1],
                   "related_group_delta_ci95_lower_pp": group["ci95_pp"][0],
                   "related_group_delta_ci95_upper_pp": group["ci95_pp"][1],
                   "related_group_delta_one_sided95_lower_pp": group["one_sided95_lower_pp"],
                   "ece": calibration["ece"],
                   ACCURACY_COLUMN: _percent(calibration["accuracy_at_threshold"]),
                   COVERAGE_COLUMN: _percent(calibration["coverage_at_threshold"]),
                   "meets_similarity_bar": contrast.get("meets_similarity_bar", ""),
                   "reference_advantage_supported": contrast.get("reference_advantage_supported", ""),
                   "disagreement_n": contrast["prediction_disagreement_n"],
                   "interpretation": "reference" if candidate_id == reference else item["interpretation"]}
            row.update({f"pooled_recall_delta_pp_{name.replace(' ', '_')}": value
                        for name, value in contrast["pooled_recall_delta_pp"].items()})
            writer.writerow({key: round(value, 4) if isinstance(value, float) else value
                             for key, value in row.items()})


def _without_paths(value: Any) -> Any:
    """Replace absolute paths in blocker and error text; the private audit.json keeps them."""
    if isinstance(value, str):
        return ABSOLUTE_PATH.sub("<path>", value)
    if isinstance(value, dict):
        return {key: _without_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_without_paths(item) for item in value]
    return value


def public_summary(report: dict, evidence: dict[str, Path]) -> dict:
    """Shareable audit view: no file paths, member diagnostics or full reliability bins."""
    private = {"manifest_path", "inventory_report", "member_diagnostics", "candidates"}
    summary = {key: value for key, value in report.items() if key not in private}
    summary["candidates"] = {}
    for candidate_id, item in report["candidates"].items():
        view = {key: value for key, value in item.items() if key != "reliability"}
        if "reliability" in item:
            view["calibration"] = _calibration(item["reliability"])
        summary["candidates"][candidate_id] = view
    summary["calibration_note"] = ("OOF calibration depends on members per OOF example; a deployed fold ensemble "
                                   "averages more members per input than its OOF reconstruction. No calibrator fitted.")
    summary["raw_evidence"] = {name: {"sha256": sha256_file(path)} for name, path in evidence.items() if path.is_file()}
    summary["scope"] = "Saved development predictions and metadata only; no held-out reads, inference or training."
    return _without_paths(summary)
