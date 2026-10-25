"""Post-hoc logit adjustment by training class priors (lever L9; numpy only).

Adjusted logits are ``logits - tau * log(prior)`` with ``tau`` from the frozen grid ``TAUS``.
Priors are protein-level training class frequencies over the fixed seven EC classes with
add-one smoothing: ``(n_c + 1) / (N + 7)``, each training protein counted once.

Selection adjusts every inner fit with its own training priors before the out-of-fold rows are
pooled; the final ensemble averages logits over seeds first and then adjusts with the refit
priors. Predictions take the arg-max; exact ties go to the lowest class index (``numpy.argmax``
returns the first maximum), so the rule is deterministic.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from training.ec_class_metrics import N_EC_CLASSES


TAUS: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
PRIOR_SMOOTHING = 1.0
_PRIOR_SUM_TOLERANCE = 1e-9


def validate_tau(tau: float) -> float:
    value = float(tau)
    if value not in TAUS:
        raise ValueError(f"tau must be one of the frozen values {TAUS}, got {tau!r}.")
    return value


def protein_class_priors(
    train_class_indices: Sequence[int] | np.ndarray,
    n_classes: int = N_EC_CLASSES,
    *,
    protein_ids: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Add-one smoothed class priors over training proteins.

    ``train_class_indices`` holds one class index per protein. When rows are pockets, pass
    ``protein_ids`` so each protein is counted once; a protein with two classes is refused.
    """
    if int(n_classes) != N_EC_CLASSES:
        raise ValueError(f"Priors use the fixed {N_EC_CLASSES}-class EC vocabulary, got n_classes={n_classes}.")
    classes = np.asarray(train_class_indices)
    if classes.ndim != 1 or classes.size == 0:
        raise ValueError("Priors need a non-empty one-dimensional array of training class indices.")
    if not np.issubdtype(classes.dtype, np.integer):
        raise ValueError("Training class indices must be integers.")
    if classes.min() < 0 or classes.max() >= N_EC_CLASSES:
        raise ValueError(f"Training class indices must lie in 0..{N_EC_CLASSES - 1}.")
    if protein_ids is not None:
        ids = [str(value) for value in protein_ids]
        if len(ids) != classes.size:
            raise ValueError("protein_ids must match the training class indices in length.")
        per_protein: dict[str, int] = {}
        for pid, value in zip(ids, classes.tolist()):
            if per_protein.setdefault(pid, value) != value:
                raise ValueError(f"Protein {pid!r} carries more than one EC class.")
        classes = np.fromiter(per_protein.values(), dtype=np.int64, count=len(per_protein))
    counts = np.bincount(classes.astype(np.int64), minlength=N_EC_CLASSES).astype(np.float64)
    return (counts + PRIOR_SMOOTHING) / (counts.sum() + PRIOR_SMOOTHING * N_EC_CLASSES)


def _check_priors(priors: np.ndarray, n_rows: Optional[int]) -> np.ndarray:
    values = np.asarray(priors, dtype=np.float64)
    if values.shape != (N_EC_CLASSES,) and (n_rows is None or values.shape != (n_rows, N_EC_CLASSES)):
        raise ValueError(f"priors must have shape ({N_EC_CLASSES},) or (n_rows, {N_EC_CLASSES}), got {values.shape}.")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("priors must be finite and positive.")
    if np.any(np.abs(values.sum(axis=-1) - 1.0) > _PRIOR_SUM_TOLERANCE):
        raise ValueError("priors must sum to 1.")
    return values


def adjust(logits: Sequence[Sequence[float]] | np.ndarray, priors: Sequence[float] | np.ndarray, tau: float) -> np.ndarray:
    """``logits - tau * log(priors)``; priors are one vector or one vector per row."""
    tau = validate_tau(tau)
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim not in (1, 2) or values.shape[-1] != N_EC_CLASSES:
        raise ValueError(f"logits must have shape ({N_EC_CLASSES},) or (n, {N_EC_CLASSES}), got {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("logits must be finite.")
    prior_values = _check_priors(priors, values.shape[0] if values.ndim == 2 else None)
    if tau == 0.0:
        return values.copy()
    return values - tau * np.log(prior_values)


def predict_ec_class(logits: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Arg-max class index per row; exact ties go to the lowest class index."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim not in (1, 2) or values.shape[-1] != N_EC_CLASSES:
        raise ValueError(f"logits must have shape ({N_EC_CLASSES},) or (n, {N_EC_CLASSES}), got {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("logits must be finite.")
    return values.argmax(axis=-1)
