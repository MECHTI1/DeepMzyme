"""Fixed-denominator EC-class metrics (numpy only).

One implementation serves the trainer, the campaign selection and the scorer.
Classes are the seven EC classes in index order 0..6 (EC classes 1..7). Two
summaries are computed over EC classes 1-6 from seven-way predictions: the
accuracy over the proteins of those classes (``ec_classes_1_6_accuracy``: the
correctly predicted weight of the rows whose true class is one of EC classes 1-6,
over the weight of those rows) and the balanced accuracy over the EC classes 1-6
present. A prediction of class 7 for such a protein counts as an error; class-7
rows never enter the accuracy over EC classes 1-6, and class 7 is not averaged in
the balanced accuracy. The campaign decides which summary is primary. Classes
absent from the true labels are left out of an average and reported through the
``*_n_present`` counts.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np


N_EC_CLASSES = 7
PRIMARY_CLASS_INDICES = tuple(range(6))
ALL_CLASS_INDICES = tuple(range(N_EC_CLASSES))


def _as_index_array(values: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}.")
    if array.size and not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.equal(np.mod(array, 1), 0)):
            raise ValueError(f"{name} must contain integer class indices.")
        array = array.astype(np.int64)
    array = array.astype(np.int64, copy=False)
    if array.size and (array.min() < 0 or array.max() >= N_EC_CLASSES):
        raise ValueError(f"{name} holds class indices outside 0..{N_EC_CLASSES - 1}.")
    return array


def _class_average(values: np.ndarray, support: np.ndarray, classes: Sequence[int]) -> tuple[float, int]:
    present = [index for index in classes if support[index] > 0]
    if not present:
        return float("nan"), 0
    return float(np.mean([values[index] for index in present])), len(present)


def ec_class_metrics(
    true_idx: Sequence[int] | np.ndarray,
    pred_idx: Sequence[int] | np.ndarray,
    *,
    weights: Optional[Sequence[float] | np.ndarray] = None,
) -> dict[str, Any]:
    true = _as_index_array(true_idx, "true_idx")
    pred = _as_index_array(pred_idx, "pred_idx")
    if true.size == 0:
        raise ValueError("EC-class metrics need at least one example.")
    if true.shape != pred.shape:
        raise ValueError(f"true_idx and pred_idx differ in length: {true.shape} vs {pred.shape}.")
    if weights is None:
        sample_weight = np.ones(true.shape[0], dtype=np.float64)
    else:
        sample_weight = np.asarray(weights, dtype=np.float64)
        if sample_weight.shape != true.shape:
            raise ValueError(f"weights must match the labels: {sample_weight.shape} vs {true.shape}.")
        if np.any(sample_weight < 0) or not np.all(np.isfinite(sample_weight)) or sample_weight.sum() <= 0:
            raise ValueError("weights must be finite, non-negative and not all zero.")

    confusion = np.zeros((N_EC_CLASSES, N_EC_CLASSES), dtype=np.float64)
    np.add.at(confusion, (true, pred), sample_weight)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    correct = np.diag(confusion)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, correct / np.where(support > 0, support, 1.0), 0.0)
        precision = np.where(predicted > 0, correct / np.where(predicted > 0, predicted, 1.0), 0.0)
        f1_denominator = precision + recall
        f1 = np.where(f1_denominator > 0, 2.0 * precision * recall / np.where(f1_denominator > 0, f1_denominator, 1.0), 0.0)

    primary_ba, primary_present = _class_average(recall, support, PRIMARY_CLASS_INDICES)
    seven_ba, seven_present = _class_average(recall, support, ALL_CLASS_INDICES)
    primary_f1, _ = _class_average(f1, support, PRIMARY_CLASS_INDICES)
    seven_f1, _ = _class_average(f1, support, ALL_CLASS_INDICES)
    primary_rows = list(PRIMARY_CLASS_INDICES)
    primary_weight = float(support[primary_rows].sum())
    primary_accuracy = float(correct[primary_rows].sum() / primary_weight) if primary_weight > 0 else float("nan")
    return {
        "n_examples": int(true.shape[0]),
        "total_weight": float(sample_weight.sum()),
        "accuracy": float(correct.sum() / sample_weight.sum()),
        "ec_classes_1_6_accuracy": primary_accuracy,
        "ec_classes_1_6_total_weight": primary_weight,
        "ec_classes_1_6_balanced_accuracy": primary_ba,
        "ec_classes_1_6_n_present": primary_present,
        "ec_classes_1_6_label": f"{primary_present}-class BA",
        "ec_classes_1_6_macro_f1": primary_f1,
        "seven_class_balanced_accuracy": seven_ba,
        "seven_class_n_present": seven_present,
        "seven_class_full_support": bool(seven_present == N_EC_CLASSES),
        "seven_class_macro_f1": seven_f1,
        "per_class_support": support.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_precision": precision.tolist(),
        "per_class_f1": f1.tolist(),
        "confusion_matrix": confusion.tolist(),
    }


def ec_class_metrics_from_logits(
    logits: Sequence[Sequence[float]] | np.ndarray,
    true_idx: Sequence[int] | np.ndarray,
    *,
    weights: Optional[Sequence[float] | np.ndarray] = None,
) -> dict[str, Any]:
    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != N_EC_CLASSES:
        raise ValueError(f"logits must have shape (n, {N_EC_CLASSES}), got {scores.shape}.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("logits must be finite.")
    return ec_class_metrics(true_idx, scores.argmax(axis=1), weights=weights)
