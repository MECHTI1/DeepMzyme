"""Pre-registered inner-fold selection for the EC class versus CLEAN30 campaign (numpy only).

Every outer fold runs the same steps on its own training proteins only:

- ``promote``: a challenger replaces the incumbent only if (1) the seed-averaged pooled inner
  out-of-fold gain in balanced accuracy over EC classes 1-6 is at least 1.0 point; (2) the
  one-sided 96% paired bootstrap lower bound of that gain (alpha 0.04; L9 alone 0.01, its own
  ``PromotionConfig``), resampling inner identity components (10,000 draws, frozen seed), is
  above 0; (3) no class with at least 30 training proteins loses
  at least 5 recall points that are also at least 3 net proteins; (4) no EC class 1-6 recall
  falls to zero, read as: positive for the incumbent and zero for the challenger.
- S1: the learning rate per family (higher pooled BA; within 0.5 points the first-listed
  setting of ``S1_LEARNING_RATE_GRID``, whose labels key the inputs) and the family escalation
  (anchor F2; F5 vs F2; then F4 vs F5 if the whole-protein branch was adopted, else F3 vs F2;
  F1 and C-G are reported, never selected). This historical rule remains available for replay.
  The versioned amended ``s1_family_screen`` compares F3, F5 and F4 directly with F2 in one
  Holm family at alpha 0.04, retaining all three hypotheses even when a promotion guard fails.
- S2: each lever alone against the S1 winner; every decision must name that winner as its
  incumbent. L7 compares the better of 12 and 20 epochs with 30; L9 picks tau on the pooled
  out-of-fold predictions and compares it with tau = 0; L5 counts as tested only with SupCon
  positives (batch or memory) on at least 90% of training minibatches of every required inner fit. Inapplicable levers are logged "not applicable".
- S3: two or more passing levers are combined; the combination is adopted only if it passes
  ``promote`` against the best single lever, otherwise the best single lever (or S1) stays.
- S4: the refit specification (a grid learning-rate setting, seeds 42-46).
- Hash-chained step records and a null operating-characteristics simulator.

Values named ``*_points`` are percentage points. Definitions that Freeze 1 must confirm are
explicit fields of ``PromotionConfig``; the other tie rules are stated where they apply.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from ec_clean30.logit_adjustment import TAUS, adjust, validate_tau
from ec_clean30.stats import (
    DEFAULT_BOOTSTRAP_SEED, DEFAULT_MAX_SKIP_FRACTION, DEFAULT_N_DRAWS, BootstrapAbort, BootstrapResult, Contrast,
    build_cluster_count_table, paired_cluster_bootstrap,
)
from training.ec_class_metrics import N_EC_CLASSES, PRIMARY_CLASS_INDICES, ec_class_metrics
from training.ec_classes import EC_CLASS_TOKEN_TO_INDEX


POINTS = 100.0
FLOAT_TOLERANCE = 1e-9

SCREENING_SEEDS: tuple[int, ...] = (42, 43)
REFIT_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)
N_INNER_FOLDS = 4

MIN_GAIN_POINTS = 1.0
# D1 (P2-9a, confirmed on fresh simulation draws 2026-10-30): alpha 0.04 for every promotion,
# including L7's two looks; L9's four looks take alpha / 4 = 0.01 through its own PromotionConfig.
PROMOTION_ALPHA = 0.04
L9_PROMOTION_ALPHA = 0.01
LR_TIE_BAND_POINTS = 0.5
GUARD_MIN_TRAINING_PROTEINS = 30
GUARD_RECALL_DROP_POINTS = 5.0
GUARD_NET_PROTEINS = 3.0
ZERO_RECALL_UNITS = ("seed_mean", "per_seed")
GAIN_CONTRAST = "challenger_minus_incumbent"

FAMILIES: tuple[str, ...] = ("F1", "F2", "F3", "F4", "F5", "C-G")
ANCHOR_FAMILY = "F2"
CONTROL_FAMILIES: tuple[str, ...] = ("F1", "C-G")
GVP_FAMILIES = frozenset({"F1", "F3", "F4"})
GLOBAL_BRANCH_FAMILIES = frozenset({"F4", "F5", "C-G"})

HISTORICAL_S1_RULE = "family_escalation_v1"
AMENDED_S1_RULE = "fixed_anchor_holm_v1"
S1_FAMILY_ALPHA = 0.04
S1_FAMILY_TIE_BAND_POINTS = 0.5
S1_CANDIDATES: tuple[str, ...] = ("F3", "F5", "F4")
S1_TIE_PREFERENCE: tuple[str, ...] = ("F5", "F3", "F4")

# Listed order matters: within the tie band the first-listed setting wins.
_SINGLE_LR = ((("learning_rate", 1e-4),), (("learning_rate", 3e-4),))
_SPLIT_LR = (
    (("gvp_learning_rate", 1e-4), ("learning_rate", 1e-4)),
    (("gvp_learning_rate", 3e-4), ("learning_rate", 1e-4)),
)
S1_LEARNING_RATE_GRID = MappingProxyType(
    {"F1": _SINGLE_LR, "F2": _SINGLE_LR, "F3": _SPLIT_LR, "F4": _SPLIT_LR, "F5": _SINGLE_LR, "C-G": _SINGLE_LR}
)


def learning_rate_label(setting: Iterable[tuple[str, float]] | Mapping[str, float]) -> str:
    """Stable label of one learning-rate setting, e.g. ``gvp_learning_rate=0.0003,learning_rate=0.0001``."""
    items = setting.items() if isinstance(setting, Mapping) else setting
    return ",".join(f"{key}={float(value):g}" for key, value in sorted((str(k), v) for k, v in items))


def s1_learning_rate_labels(family: str) -> tuple[str, ...]:
    """The family's grid labels in listed order."""
    return tuple(learning_rate_label(setting) for setting in S1_LEARNING_RATE_GRID[_check_family(family)])


def _grid_setting(family: str, setting: Mapping[str, float]) -> dict[str, float]:
    """The grid entry equal to ``setting``; anything outside the family's grid is refused."""
    given = {str(key): float(value) for key, value in setting.items()}
    for entry in S1_LEARNING_RATE_GRID[_check_family(family)]:
        if dict(entry) == given:
            return dict(entry)
    raise ValueError(f"{family}: learning-rate setting {given} is not in the S1 grid {s1_learning_rate_labels(family)}.")

LEVERS: tuple[str, ...] = ("L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9")
EPOCH_LEVER = "L7"
LOGIT_ADJUSTMENT_LEVER = "L9"
DEFAULT_EPOCHS = 30
EPOCH_LOOKS: tuple[int, ...] = (12, 20)
SUPCON_MIN_POSITIVE_STEP_FRACTION = 0.90
_LEVER_SCOPE = {"L1": GVP_FAMILIES, "L2": GVP_FAMILIES, "L8": GLOBAL_BRANCH_FAMILIES}

LEVER_PASSED = "passed"
LEVER_FAILED = "failed"
LEVER_NOT_APPLICABLE = "not_applicable"
LEVER_NOT_TESTED = "not_tested"

GENESIS_SHA256 = "0" * 64
NULL_ADOPTION_TARGET = 0.10
DEFAULT_SIMULATION_SEED = 20261024


# ---------------------------------------------------------------------------
# Canonical JSON and hashes


def to_jsonable(value: Any) -> Any:
    """Plain JSON types; tuples become lists and non-finite floats become null."""
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"Cannot convert {type(value).__name__} to JSON.")


def canonical_json(value: Any) -> str:
    return json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha256_canonical(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


# ---------------------------------------------------------------------------
# Pooled inner out-of-fold predictions


@dataclass(frozen=True, eq=False)
class PooledOof:
    """Pooled inner out-of-fold protein predictions of one configuration over its screening seeds.

    ``logits[seed, protein, class]``. ``row_priors`` (same shape) holds the training priors of
    the inner fit that predicted each row; only logit adjustment needs it.
    """

    name: str
    protein_ids: tuple[str, ...]
    true_idx: np.ndarray
    seeds: tuple[int, ...]
    logits: np.ndarray
    row_priors: Optional[np.ndarray] = None
    inner_fold: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        ids = tuple(str(value) for value in self.protein_ids)
        if not ids:
            raise ValueError(f"{self.name}: pooled predictions need at least one protein.")
        if len(set(ids)) != len(ids):
            raise ValueError(f"{self.name}: protein IDs must be unique.")
        true = np.asarray(self.true_idx)
        if true.shape != (len(ids),) or not np.issubdtype(true.dtype, np.integer):
            raise ValueError(f"{self.name}: true_idx must hold one integer class per protein.")
        if true.min() < 0 or true.max() >= N_EC_CLASSES:
            raise ValueError(f"{self.name}: true_idx holds classes outside 0..{N_EC_CLASSES - 1}.")
        seeds = tuple(int(seed) for seed in self.seeds)
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError(f"{self.name}: seeds must be non-empty and distinct.")
        logits = np.asarray(self.logits, dtype=np.float64)
        if logits.shape != (len(seeds), len(ids), N_EC_CLASSES) or not np.all(np.isfinite(logits)):
            raise ValueError(
                f"{self.name}: logits must be finite with shape ({len(seeds)}, {len(ids)}, {N_EC_CLASSES}), got {logits.shape}."
            )
        priors = self.row_priors
        if priors is not None:
            priors = np.asarray(priors, dtype=np.float64)
            if priors.shape != logits.shape or not np.all(np.isfinite(priors)) or np.any(priors <= 0):
                raise ValueError(f"{self.name}: row_priors must be positive and shaped like the logits.")
        folds = self.inner_fold
        if folds is not None:
            folds = np.asarray(folds)
            if folds.shape != (len(ids),) or not np.issubdtype(folds.dtype, np.integer):
                raise ValueError(f"{self.name}: inner_fold must hold one integer per protein.")
        object.__setattr__(self, "protein_ids", ids)
        object.__setattr__(self, "true_idx", true.astype(np.int64))
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "logits", logits)
        object.__setattr__(self, "row_priors", priors)
        object.__setattr__(self, "inner_fold", None if folds is None else folds.astype(np.int64))

    @property
    def predictions(self) -> np.ndarray:
        """``[seed, protein]`` arg-max classes; exact ties go to the lowest class index."""
        return self.logits.argmax(axis=-1)


@dataclass(frozen=True, eq=False)
class InnerFoldPredictions:
    """Protein-level logits of one inner fit (one seed, one held-out inner fold)."""

    seed: int
    inner_fold: int
    protein_ids: tuple[str, ...]
    logits: np.ndarray
    training_priors: Optional[np.ndarray] = None


def _examples(values: Iterable[str], limit: int = 5) -> list[str]:
    return sorted(values)[:limit]


def pool_inner_oof(
    name: str,
    parts: Sequence[InnerFoldPredictions],
    true_class_of: Mapping[str, int],
    *,
    expected_seeds: Optional[Sequence[int]] = SCREENING_SEEDS,
    expected_inner_folds: Optional[int] = N_INNER_FOLDS,
) -> PooledOof:
    """Concatenate inner-fold predictions; every eligible protein must appear exactly once per seed."""
    if not parts:
        raise ValueError(f"{name}: no inner-fold predictions given.")
    eligible = sorted(str(pid) for pid in true_class_of)
    index = {pid: position for position, pid in enumerate(eligible)}
    keys = [(int(part.seed), int(part.inner_fold)) for part in parts]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{name}: a (seed, inner fold) pair is given more than once.")
    seeds = tuple(sorted({seed for seed, _ in keys}))
    if expected_seeds is not None and seeds != tuple(sorted(int(seed) for seed in expected_seeds)):
        raise ValueError(f"{name}: seeds {seeds} differ from the expected {tuple(expected_seeds)}.")
    folds_by_seed = {seed: sorted(fold for s, fold in keys if s == seed) for seed in seeds}
    reference_folds = folds_by_seed[seeds[0]]
    if any(folds != reference_folds for folds in folds_by_seed.values()):
        raise ValueError(f"{name}: seeds cover different inner folds: {folds_by_seed}.")
    if expected_inner_folds is not None and len(reference_folds) != expected_inner_folds:
        raise ValueError(f"{name}: expected {expected_inner_folds} inner folds, got {reference_folds}.")
    with_priors = {part.training_priors is not None for part in parts}
    if len(with_priors) != 1:
        raise ValueError(f"{name}: training priors must be given for every inner fit or for none.")

    n = len(eligible)
    logits = np.full((len(seeds), n, N_EC_CLASSES), np.nan)
    priors = np.full_like(logits, np.nan) if with_priors == {True} else None
    fold_of: dict[str, int] = {}
    for part in sorted(parts, key=lambda item: (int(item.seed), int(item.inner_fold))):
        seed_index = seeds.index(int(part.seed))
        ids = [str(pid) for pid in part.protein_ids]
        values = np.asarray(part.logits, dtype=np.float64)
        if values.shape != (len(ids), N_EC_CLASSES):
            raise ValueError(f"{name}: inner fold {part.inner_fold} seed {part.seed} logits have shape {values.shape}.")
        extra = [pid for pid in ids if pid not in index]
        if extra:
            raise ValueError(f"{name}: {len(extra)} predicted proteins are not eligible, e.g. {_examples(extra)}.")
        rows = np.array([index[pid] for pid in ids], dtype=np.int64)
        repeated = {pid for k, pid in enumerate(ids) if not np.isnan(logits[seed_index, rows[k], 0])}
        repeated |= {pid for pid in ids if ids.count(pid) > 1} if len(set(ids)) != len(ids) else set()
        if repeated:
            raise ValueError(f"{name}: proteins predicted more than once for seed {part.seed}, e.g. {_examples(repeated)}.")
        for pid in ids:
            if fold_of.setdefault(pid, int(part.inner_fold)) != int(part.inner_fold):
                raise ValueError(f"{name}: protein {pid!r} sits in different inner folds across seeds.")
        logits[seed_index, rows] = values
        if priors is not None:
            fit_priors = np.asarray(part.training_priors, dtype=np.float64)
            if fit_priors.shape != (N_EC_CLASSES,):
                raise ValueError(f"{name}: training priors must have shape ({N_EC_CLASSES},).")
            priors[seed_index, rows] = fit_priors
    for seed_index, seed in enumerate(seeds):
        missing = [eligible[row] for row in np.flatnonzero(np.isnan(logits[seed_index, :, 0]))]
        if missing:
            raise ValueError(f"{name}: {len(missing)} eligible proteins lack a prediction for seed {seed}, e.g. {_examples(missing)}.")
    true = np.array([int(true_class_of[pid]) for pid in eligible], dtype=np.int64)
    folds = np.array([fold_of[pid] for pid in eligible], dtype=np.int64)
    return PooledOof(name, tuple(eligible), true, seeds, logits, priors, folds)


def predictions_sha256(pooled: PooledOof) -> str:
    header = {
        "protein_ids": list(pooled.protein_ids),
        "true_idx": pooled.true_idx,
        "seeds": list(pooled.seeds),
        "inner_fold": pooled.inner_fold,
        "shape": list(pooled.logits.shape),
        "has_row_priors": pooled.row_priors is not None,
    }
    digest = hashlib.sha256(canonical_json(header).encode("ascii"))
    digest.update(np.ascontiguousarray(pooled.logits, dtype="<f8").tobytes())
    if pooled.row_priors is not None:
        digest.update(np.ascontiguousarray(pooled.row_priors, dtype="<f8").tobytes())
    return digest.hexdigest()


def component_table_sha256(component_of: Mapping[str, str], protein_ids: Sequence[str]) -> str:
    """Hash of the component assignment restricted to the given proteins."""
    return sha256_canonical(sorted([str(pid), str(component_of[pid])] for pid in protein_ids))


def apply_tau(pooled: PooledOof, tau: float) -> PooledOof:
    """Adjust every row with the training priors of the inner fit that predicted it."""
    tau = validate_tau(tau)
    if pooled.row_priors is None:
        raise ValueError(f"{pooled.name}: logit adjustment needs the training priors of every inner fit.")
    adjusted = np.stack([adjust(pooled.logits[k], pooled.row_priors[k], tau) for k in range(len(pooled.seeds))])
    return replace(pooled, name=f"{pooled.name}/tau={tau:+.1f}", logits=adjusted)


# ---------------------------------------------------------------------------
# Seed-averaged metrics


@dataclass(frozen=True, eq=False)
class _SeedSummary:
    ba_points_by_seed: tuple[float, ...]
    recall: np.ndarray
    correct: np.ndarray
    support: np.ndarray

    @property
    def ba_points(self) -> float:
        return float(np.mean(self.ba_points_by_seed))


def _summarise(pooled: PooledOof) -> _SeedSummary:
    rows = [ec_class_metrics(pooled.true_idx, predicted) for predicted in pooled.predictions]
    ba = tuple(POINTS * row["ec_classes_1_6_balanced_accuracy"] for row in rows)
    if not all(math.isfinite(value) for value in ba):
        raise ValueError(f"{pooled.name}: no protein of EC classes 1-6 in the pooled predictions.")
    return _SeedSummary(
        ba_points_by_seed=ba,
        recall=np.array([row["per_class_recall"] for row in rows]),
        correct=np.array([np.diag(np.asarray(row["confusion_matrix"])) for row in rows]),
        support=np.asarray(rows[0]["per_class_support"]),
    )


def seed_mean_ba_points(pooled: PooledOof) -> float:
    """Seed-averaged pooled balanced accuracy over EC classes 1-6, in points."""
    return _summarise(pooled).ba_points


# ---------------------------------------------------------------------------
# Promotion rule


@dataclass(frozen=True)
class PromotionConfig:
    """Promotion-rule parameters. ``stratified``, ``max_skip_fraction`` and ``zero_recall_unit``
    are open Freeze-1 choices; the defaults are an unstratified component bootstrap, a 1% skip
    limit and seed-averaged recall."""

    min_gain_points: float = MIN_GAIN_POINTS
    alpha: float = PROMOTION_ALPHA
    n_draws: int = DEFAULT_N_DRAWS
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED
    stratified: bool = False
    max_skip_fraction: float = DEFAULT_MAX_SKIP_FRACTION
    guard_min_training_proteins: int = GUARD_MIN_TRAINING_PROTEINS
    guard_recall_drop_points: float = GUARD_RECALL_DROP_POINTS
    guard_net_proteins: float = GUARD_NET_PROTEINS
    zero_recall_unit: str = "seed_mean"

    def __post_init__(self) -> None:
        if self.zero_recall_unit not in ZERO_RECALL_UNITS:
            raise ValueError(f"zero_recall_unit must be one of {ZERO_RECALL_UNITS}, got {self.zero_recall_unit!r}.")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1.")
        if int(self.n_draws) < 1:
            raise ValueError("n_draws must be positive.")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


# D1: the promotion level per lever; every lever not listed uses PROMOTION_ALPHA (L7 included).
LEVER_PROMOTION_ALPHA: Mapping[str, float] = {LOGIT_ADJUSTMENT_LEVER: L9_PROMOTION_ALPHA}


def promotion_config_for(lever: Optional[str], base: Optional[PromotionConfig] = None) -> PromotionConfig:
    """``base`` (default: the frozen config) with the promotion level D1 sets for ``lever``.

    ``lever`` None (S1 and S3) and every lever except L9 take ``PROMOTION_ALPHA``; L9 takes
    ``L9_PROMOTION_ALPHA``. Only ``alpha`` changes, so the other Freeze-1 choices carry over.
    """
    base = PromotionConfig() if base is None else base
    if lever is not None and lever not in LEVERS:
        raise ValueError(f"Unknown lever {lever!r}; expected one of {LEVERS}.")
    return replace(base, alpha=LEVER_PROMOTION_ALPHA.get(lever, PROMOTION_ALPHA) if lever else PROMOTION_ALPHA)


@dataclass(frozen=True)
class PromotionDecision:
    incumbent: str
    challenger: str
    promoted: bool
    gain_points: float
    incumbent_ba_points: float
    challenger_ba_points: float
    incumbent_ba_points_by_seed: tuple[float, ...]
    challenger_ba_points_by_seed: tuple[float, ...]
    seeds: tuple[int, ...]
    lower_bound_points: Optional[float]
    p_value: Optional[float]
    checks: Mapping[str, Optional[bool]]
    vetoes: tuple[str, ...]
    recall_guard: tuple[Mapping[str, Any], ...]
    zero_recall_ec_classes: tuple[int, ...]
    bootstrap: Optional[Mapping[str, Any]]
    config: Mapping[str, Any]
    inputs: Mapping[str, str]

    def to_record(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


def _training_counts(values: Sequence[int] | np.ndarray) -> np.ndarray:
    counts = np.asarray(values)
    if counts.shape != (N_EC_CLASSES,) or np.any(counts < 0):
        raise ValueError(f"training_class_counts must hold {N_EC_CLASSES} non-negative protein counts.")
    return counts.astype(np.float64)


def _require_paired(incumbent: PooledOof, challenger: PooledOof) -> None:
    if incumbent.protein_ids != challenger.protein_ids:
        raise ValueError(f"{incumbent.name} and {challenger.name} cover different proteins.")
    if not np.array_equal(incumbent.true_idx, challenger.true_idx):
        raise ValueError(f"{incumbent.name} and {challenger.name} disagree on true classes.")
    if incumbent.seeds != challenger.seeds:
        raise ValueError(f"{incumbent.name} and {challenger.name} use different seeds.")


def _require_components(component_of: Mapping[str, str], pooled: PooledOof) -> None:
    missing = [pid for pid in pooled.protein_ids if pid not in component_of]
    if missing:
        raise ValueError(
            f"{pooled.name}: {len(missing)} pooled proteins have no inner identity component, e.g. {_examples(missing)}."
        )


def _recall_guard(
    incumbent: _SeedSummary, challenger: _SeedSummary, training_counts: np.ndarray, config: PromotionConfig
) -> tuple[tuple[dict[str, Any], ...], bool]:
    rows = []
    for index in range(N_EC_CLASSES):
        if training_counts[index] < config.guard_min_training_proteins or incumbent.support[index] <= 0:
            continue
        drop = POINTS * float(incumbent.recall[:, index].mean() - challenger.recall[:, index].mean())
        net = float(incumbent.correct[:, index].mean() - challenger.correct[:, index].mean())
        vetoed = (drop >= config.guard_recall_drop_points - FLOAT_TOLERANCE
                  and net >= config.guard_net_proteins - FLOAT_TOLERANCE)
        rows.append({
            "ec_class": index + 1,
            "training_proteins": int(training_counts[index]),
            "recall_drop_points": drop,
            "net_proteins_lost": net,
            "vetoed": vetoed,
        })
    return tuple(rows), not any(row["vetoed"] for row in rows)


def _zero_recall_classes(incumbent: _SeedSummary, challenger: _SeedSummary, unit: str) -> tuple[int, ...]:
    classes = []
    for index in PRIMARY_CLASS_INDICES:
        if incumbent.support[index] <= 0:
            continue
        if unit == "seed_mean":
            falls = incumbent.recall[:, index].mean() > 0 and challenger.recall[:, index].mean() <= 0
        else:
            falls = bool(np.any((incumbent.recall[:, index] > 0) & (challenger.recall[:, index] <= 0)))
        if falls:
            classes.append(index + 1)
    return tuple(classes)


def _promotion_bootstrap(
    incumbent: PooledOof, challenger: PooledOof, component_of: Mapping[str, str], config: PromotionConfig
) -> BootstrapResult:
    arms: dict[str, np.ndarray] = {}
    systems: dict[str, list[str]] = {"incumbent": [], "challenger": []}
    for role, pooled in (("incumbent", incumbent), ("challenger", challenger)):
        for position, seed in enumerate(pooled.seeds):
            arm = f"{role}/seed_{seed}"
            arms[arm] = pooled.predictions[position]
            systems[role].append(arm)
    table = build_cluster_count_table(incumbent.protein_ids, incumbent.true_idx, arms, component_of)
    return paired_cluster_bootstrap(
        table,
        [Contrast.difference(GAIN_CONTRAST, "challenger", "incumbent")],
        systems=systems,
        n_draws=config.n_draws,
        seed=config.bootstrap_seed,
        stratified=config.stratified,
        max_skip_fraction=config.max_skip_fraction,
    )


def promote(
    incumbent: PooledOof,
    challenger: PooledOof,
    *,
    component_of: Mapping[str, str],
    training_class_counts: Sequence[int] | np.ndarray,
    config: PromotionConfig = PromotionConfig(),
    evaluate_all_checks: bool = True,
) -> PromotionDecision:
    """Apply the promotion rule to seed-averaged pooled inner out-of-fold predictions.

    ``component_of`` maps every pooled protein to its inner identity component, the bootstrap
    unit. ``training_class_counts`` holds the fold's training proteins per EC class 1-7 and
    decides which classes the recall guard covers. Both arms share every bootstrap draw. With
    ``evaluate_all_checks=False`` the bootstrap runs only when the cheaper checks pass (same
    decision, used by the simulator).
    """
    _require_paired(incumbent, challenger)
    _require_components(component_of, incumbent)
    counts = _training_counts(training_class_counts)
    incumbent_summary, challenger_summary = _summarise(incumbent), _summarise(challenger)
    gain = challenger_summary.ba_points - incumbent_summary.ba_points
    gain_ok = gain >= config.min_gain_points - FLOAT_TOLERANCE
    guard_rows, guard_ok = _recall_guard(incumbent_summary, challenger_summary, counts, config)
    zero_classes = _zero_recall_classes(incumbent_summary, challenger_summary, config.zero_recall_unit)
    zero_ok = not zero_classes

    bootstrap_record, lower_bound, p_value, bound_ok = None, None, None, None
    if evaluate_all_checks or (gain_ok and guard_ok and zero_ok):
        result = _promotion_bootstrap(incumbent, challenger, component_of, config)
        test = result.test(GAIN_CONTRAST, config.alpha)
        lower_bound, p_value, bound_ok = POINTS * test.lower_bound, test.p_value, test.passed
        bootstrap_record = result.to_record()

    checks = {"gain": gain_ok, "lower_bound": bound_ok, "recall_guard": guard_ok, "zero_recall": zero_ok}
    vetoes = tuple(name for name, passed in checks.items() if passed is False)
    return PromotionDecision(
        incumbent=incumbent.name,
        challenger=challenger.name,
        promoted=bool(gain_ok and guard_ok and zero_ok and bound_ok),
        gain_points=gain,
        incumbent_ba_points=incumbent_summary.ba_points,
        challenger_ba_points=challenger_summary.ba_points,
        incumbent_ba_points_by_seed=incumbent_summary.ba_points_by_seed,
        challenger_ba_points_by_seed=challenger_summary.ba_points_by_seed,
        seeds=incumbent.seeds,
        lower_bound_points=lower_bound,
        p_value=p_value,
        checks=checks,
        vetoes=vetoes,
        recall_guard=guard_rows,
        zero_recall_ec_classes=zero_classes,
        bootstrap=bootstrap_record,
        config=config.to_record(),
        inputs={
            "incumbent_predictions_sha256": predictions_sha256(incumbent),
            "challenger_predictions_sha256": predictions_sha256(challenger),
            "component_table_sha256": component_table_sha256(component_of, incumbent.protein_ids),
        },
    )


# ---------------------------------------------------------------------------
# S1: learning rate per family and family escalation


def _check_family(family: str) -> str:
    if family not in FAMILIES:
        raise ValueError(f"Unknown family {family!r}; expected one of {FAMILIES}.")
    return family


@dataclass(frozen=True, eq=False)
class LearningRateChoice:
    family: str
    chosen: str
    best: str
    chosen_setting: Mapping[str, float]
    scores_points: tuple[tuple[str, float], ...]
    band_points: float
    setting_predictions_sha256: tuple[tuple[str, str], ...]
    winner: PooledOof = field(repr=False)

    def to_record(self) -> dict[str, Any]:
        return to_jsonable({
            "family": self.family,
            "chosen": self.chosen,
            "best": self.best,
            "chosen_setting": dict(self.chosen_setting),
            "scores_points": self.scores_points,
            "band_points": self.band_points,
            "setting_predictions_sha256": self.setting_predictions_sha256,
            "winner_predictions_sha256": predictions_sha256(self.winner),
        })


def s1_choose_learning_rate(
    family: str, settings: Mapping[str, PooledOof], *, band_points: float = LR_TIE_BAND_POINTS
) -> LearningRateChoice:
    """``settings`` maps each grid label (``s1_learning_rate_labels``) to its pooled predictions.

    Scores are read in the grid's listed order, whatever the mapping order; the first-listed
    setting within ``band_points`` of the best wins.
    """
    labels = s1_learning_rate_labels(family)
    given = [str(label) for label in settings]
    if sorted(given) != sorted(labels):
        raise ValueError(f"{family}: learning-rate settings must be keyed by exactly the grid labels {labels}, got {given}.")
    ordered = [settings[label] for label in labels]
    for pooled in ordered[1:]:
        _require_paired(ordered[0], pooled)
    scores = [seed_mean_ba_points(pooled) for pooled in ordered]
    top = max(scores)
    best = next(k for k, score in enumerate(scores) if score >= top - FLOAT_TOLERANCE)
    chosen = next(k for k, score in enumerate(scores) if top - score <= band_points + FLOAT_TOLERANCE)
    return LearningRateChoice(
        family=family,
        chosen=labels[chosen],
        best=labels[best],
        chosen_setting=dict(S1_LEARNING_RATE_GRID[family][chosen]),
        scores_points=tuple(zip(labels, scores)),
        band_points=float(band_points),
        setting_predictions_sha256=tuple((label, predictions_sha256(pooled)) for label, pooled in zip(labels, ordered)),
        winner=ordered[chosen],
    )


@dataclass(frozen=True, eq=False)
class FamilyEscalation:
    winner_family: str
    global_branch_adopted: bool
    pocket_graph_adopted: bool
    global_decision: PromotionDecision
    pocket_graph_comparison: tuple[str, str]
    pocket_graph_decision: PromotionDecision
    control_scores_points: Mapping[str, float]
    winner: PooledOof = field(repr=False)

    def to_record(self) -> dict[str, Any]:
        return to_jsonable({
            "winner_family": self.winner_family,
            "global_branch_adopted": self.global_branch_adopted,
            "pocket_graph_adopted": self.pocket_graph_adopted,
            "global_comparison": ["F2", "F5"],
            "global_decision": self.global_decision.to_record(),
            "pocket_graph_comparison": self.pocket_graph_comparison,
            "pocket_graph_decision": self.pocket_graph_decision.to_record(),
            "control_scores_points": self.control_scores_points,
        })


def s1_family_escalation(
    winners: Mapping[str, PooledOof],
    *,
    component_of: Mapping[str, str],
    training_class_counts: Sequence[int] | np.ndarray,
    config: PromotionConfig = PromotionConfig(),
) -> FamilyEscalation:
    """``winners`` maps each family to its S1 learning-rate winner; F1 and C-G are reported only."""
    for family in winners:
        _check_family(family)
    missing = [family for family in ("F2", "F3", "F4", "F5") if family not in winners]
    if missing:
        raise ValueError(f"Family escalation needs F2, F3, F4 and F5; missing {missing}.")
    kwargs = {"component_of": component_of, "training_class_counts": training_class_counts, "config": config}
    global_decision = promote(winners[ANCHOR_FAMILY], winners["F5"], **kwargs)
    incumbent_family, challenger_family = ("F5", "F4") if global_decision.promoted else (ANCHOR_FAMILY, "F3")
    pocket_decision = promote(winners[incumbent_family], winners[challenger_family], **kwargs)
    winner_family = challenger_family if pocket_decision.promoted else incumbent_family
    controls = {family: seed_mean_ba_points(winners[family]) for family in CONTROL_FAMILIES if family in winners}
    return FamilyEscalation(
        winner_family=winner_family,
        global_branch_adopted=global_decision.promoted,
        pocket_graph_adopted=pocket_decision.promoted,
        global_decision=global_decision,
        pocket_graph_comparison=(incumbent_family, challenger_family),
        pocket_graph_decision=pocket_decision,
        control_scores_points=controls,
        winner=winners[winner_family],
    )


def s1_policy_record() -> dict[str, Any]:
    """Fresh serializable descriptor of the amended rule, for its authenticated amendment."""
    return {
        "rule_id": AMENDED_S1_RULE,
        "historical_rule_id": HISTORICAL_S1_RULE,
        "anchor_family": ANCHOR_FAMILY,
        "candidate_families": list(S1_CANDIDATES),
        "control_families": list(CONTROL_FAMILIES),
        "multiplicity": "Holm step-down over all three fixed contrasts",
        "family_alpha": S1_FAMILY_ALPHA,
        "gain_and_recall_reference": ANCHOR_FAMILY,
        "promotion_config": PromotionConfig().to_record(),
        "learning_rate_tie_band_points": LR_TIE_BAND_POINTS,
        "learning_rate_tie_rule": "first listed grid setting within the band of the maximum",
        "family_tie_band_points": S1_FAMILY_TIE_BAND_POINTS,
        "family_tie_preference": list(S1_TIE_PREFERENCE),
        "family_tie_rule": "preferred passing family within the band of the maximum passing score",
        "fallback_family": ANCHOR_FAMILY,
        "incremental_superiority_claim": False,
    }


@dataclass(frozen=True, eq=False)
class FamilyScreen:
    winner_family: str
    candidate_scores_points: Mapping[str, float]
    control_scores_points: Mapping[str, float]
    comparisons: tuple[Mapping[str, Any], ...]
    passing_families: tuple[str, ...]
    tied_families: tuple[str, ...]
    anchor_score_points: float
    bootstrap: Mapping[str, Any]
    inputs: Mapping[str, Any]
    winner: PooledOof = field(repr=False)

    def to_record(self) -> dict[str, Any]:
        return to_jsonable({
            "rule_id": AMENDED_S1_RULE,
            "policy": s1_policy_record(),
            "winner_family": self.winner_family,
            "anchor_score_points": self.anchor_score_points,
            "candidate_scores_points": self.candidate_scores_points,
            "control_scores_points": self.control_scores_points,
            "comparisons": self.comparisons,
            "passing_families": self.passing_families,
            "tied_families": self.tied_families,
            "bootstrap": self.bootstrap,
            "inputs": self.inputs,
        })


def s1_family_screen(
    winners: Mapping[str, PooledOof],
    *,
    component_of: Mapping[str, str],
    training_class_counts: Sequence[int] | np.ndarray,
    config: PromotionConfig = PromotionConfig(),
) -> FamilyScreen:
    """Compare every amended S1 candidate with F2 using one fixed three-test Holm family.

    Every candidate gets a bootstrap test, even if its gain or recall guard fails. Only
    statistical rejection controls Holm continuation; the promotion guards separately decide
    eligibility. All comparisons share the same paired component draws. ``config.alpha`` is
    ignored: the family level is always 0.04. Selection does not claim that a passing winner
    outperforms the other passing candidates, nor protect their recall gains relative to F2.
    """
    for family in winners:
        _check_family(family)
    missing = [family for family in (ANCHOR_FAMILY, *S1_CANDIDATES) if family not in winners]
    if missing:
        raise ValueError(f"Family screen needs F2, F3, F5 and F4; missing {missing}.")
    config = replace(config, alpha=S1_FAMILY_ALPHA)
    anchor = winners[ANCHOR_FAMILY]
    _require_components(component_of, anchor)
    counts = _training_counts(training_class_counts)
    summaries = {}
    arms: dict[str, np.ndarray] = {}
    systems: dict[str, list[str]] = {}
    for family in (ANCHOR_FAMILY, *S1_CANDIDATES):
        pooled = winners[family]
        _require_paired(anchor, pooled)
        summaries[family] = _summarise(pooled)
        systems[family] = []
        for position, seed in enumerate(pooled.seeds):
            arm = f"{family}/seed_{seed}"
            arms[arm] = pooled.predictions[position]
            systems[family].append(arm)
    contrasts = [Contrast.difference(f"{family}_minus_{ANCHOR_FAMILY}", family, ANCHOR_FAMILY)
                 for family in S1_CANDIDATES]
    table = build_cluster_count_table(anchor.protein_ids, anchor.true_idx, arms, component_of)
    result = paired_cluster_bootstrap(
        table, contrasts, systems=systems, n_draws=config.n_draws, seed=config.bootstrap_seed,
        stratified=config.stratified, max_skip_fraction=config.max_skip_fraction,
    )
    p_values = {family: result.p_value(contrast.name) for family, contrast in zip(S1_CANDIDATES, contrasts)}
    order = sorted(S1_CANDIDATES, key=p_values.__getitem__)
    records = {}
    still_testing, adjusted_p = True, 0.0
    for index, family in enumerate(order):
        alpha = S1_FAMILY_ALPHA / (len(S1_CANDIDATES) - index)
        contrast = f"{family}_minus_{ANCHOR_FAMILY}"
        test = result.test(contrast, alpha)
        adjusted_p = max(adjusted_p, min(1.0, (len(S1_CANDIDATES) - index) * p_values[family]))
        tested = still_testing
        holm_passed = tested and test.passed
        still_testing = holm_passed
        summary = summaries[family]
        gain = summary.ba_points - summaries[ANCHOR_FAMILY].ba_points
        guard_rows, guard_ok = _recall_guard(summaries[ANCHOR_FAMILY], summary, counts, config)
        zero_classes = _zero_recall_classes(summaries[ANCHOR_FAMILY], summary, config.zero_recall_unit)
        checks = {"gain": gain >= config.min_gain_points - FLOAT_TOLERANCE,
                  "holm": holm_passed, "recall_guard": guard_ok, "zero_recall": not zero_classes}
        vetoes = tuple(name for name, passed in checks.items() if not passed)
        records[family] = {
            "family": family, "incumbent_family": ANCHOR_FAMILY, "contrast": contrast,
            "holm_rank": index + 1, "alpha": alpha, "p_value": p_values[family],
            "adjusted_p_value": adjusted_p, "tested": tested, "holm_passed": holm_passed,
            "lower_bound_points": POINTS * test.lower_bound,
            "gain_points": gain, "incumbent_ba_points": summaries[ANCHOR_FAMILY].ba_points,
            "challenger_ba_points": summary.ba_points,
            "incumbent_ba_points_by_seed": summaries[ANCHOR_FAMILY].ba_points_by_seed,
            "challenger_ba_points_by_seed": summary.ba_points_by_seed,
            "passed": not vetoes, "checks": checks, "vetoes": vetoes,
            "reason": None if not vetoes else ", ".join(vetoes),
            "recall_guard": guard_rows, "zero_recall_ec_classes": zero_classes,
            "config": config.to_record(),
        }
    passing = tuple(family for family in S1_CANDIDATES if records[family]["passed"])
    scores = {family: summaries[family].ba_points for family in S1_CANDIDATES}
    tied = ()
    winner = ANCHOR_FAMILY
    if passing:
        maximum = max(scores[family] for family in passing)
        tied = tuple(family for family in S1_TIE_PREFERENCE if family in passing
                     and maximum - scores[family] <= S1_FAMILY_TIE_BAND_POINTS + FLOAT_TOLERANCE)
        winner = tied[0]
    return FamilyScreen(
        winner_family=winner, candidate_scores_points=scores,
        control_scores_points={family: seed_mean_ba_points(winners[family])
                              for family in CONTROL_FAMILIES if family in winners},
        comparisons=tuple(records[family] for family in S1_CANDIDATES),
        passing_families=passing, tied_families=tied, anchor_score_points=summaries[ANCHOR_FAMILY].ba_points,
        bootstrap=result.to_record(),
        inputs={"predictions_sha256": {family: predictions_sha256(winners[family])
                                       for family in (ANCHOR_FAMILY, *S1_CANDIDATES)},
                "component_table_sha256": component_table_sha256(component_of, anchor.protein_ids)},
        winner=winners[winner],
    )


# ---------------------------------------------------------------------------
# S2: lever screen


def lever_applies(lever: str, family: str) -> bool:
    if lever not in LEVERS:
        raise ValueError(f"Unknown lever {lever!r}; expected one of {LEVERS}.")
    _check_family(family)
    return family in _LEVER_SCOPE.get(lever, frozenset(FAMILIES))


SUPCON_LEVER = "L5"


def supcon_lever_testable(positive_step_fraction: float) -> bool:
    """L5 counts as tested only if SupCon positives appeared on at least 90% of training minibatches."""
    value = float(positive_step_fraction)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"positive_step_fraction must lie in [0, 1], got {positive_step_fraction!r}.")
    return value >= SUPCON_MIN_POSITIVE_STEP_FRACTION - FLOAT_TOLERANCE


def _is_count(value: Any) -> bool:
    """A non-negative integer count; booleans are refused (``True`` is an ``int`` in Python)."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def supcon_positive_step_fraction(
    coverage_by_fit: Mapping[str, Optional[Mapping[str, Any]]],
) -> tuple[Optional[float], dict[str, Any]]:
    """L5's coverage over its required inner fits (aggregation rule of 2026-10-30).

    ``coverage_by_fit`` maps every required L5 inner fit to the ``ec_supcon_coverage`` record of
    its prediction receipt. A fit's share is ``batches_with_positive / train_batches`` (training
    minibatches; a positive is a same-label candidate of another protein among the batch and the
    memory). The lever's fraction is the smallest share, so the 90% condition must hold for each
    required fit. A fit without counters leaves the lever uncertified: the fraction is None and the
    lever is logged "not tested", never assumed to pass.

    This function only aggregates the records it is given. Certification also needs the caller
    (``select``) to derive the complete grid of required fits (inner folds × seeds) from the frozen
    plan, reject missing, duplicate or foreign fits, and check each record against the per-epoch
    counters of the fit's hash-bound checkpoint and run records; this function does neither.
    """
    if not coverage_by_fit:
        raise ValueError("L5 coverage needs at least one required fit.")
    shares: dict[str, Optional[float]] = {}
    for fit, record in coverage_by_fit.items():
        batches = None if record is None else record.get("train_batches")
        positive = None if record is None else record.get("batches_with_positive")
        valid = (_is_count(batches) and _is_count(positive) and batches > 0 and 0 <= positive <= batches)
        shares[str(fit)] = positive / batches if valid else None
    uncertified = sorted(fit for fit, share in shares.items() if share is None)
    fraction = None if uncertified else min(share for share in shares.values() if share is not None)
    details = {"aggregation": "minimum over the required inner fits", "per_fit": shares,
               "uncertified_fits": uncertified, "positive_step_fraction": fraction}
    return fraction, details


@dataclass(frozen=True, eq=False)
class LeverOutcome:
    lever: str
    status: str
    reason: str = ""
    decision: Optional[PromotionDecision] = None
    details: Mapping[str, Any] = field(default_factory=dict)
    challenger: Optional[PooledOof] = field(default=None, repr=False)

    def to_record(self) -> dict[str, Any]:
        return to_jsonable({
            "lever": self.lever,
            "status": self.status,
            "reason": self.reason,
            "details": self.details,
            "decision": None if self.decision is None else self.decision.to_record(),
        })


def _promotion_outcome(lever: str, decision: PromotionDecision, challenger: PooledOof,
                       details: Optional[Mapping[str, Any]] = None) -> LeverOutcome:
    status = LEVER_PASSED if decision.promoted else LEVER_FAILED
    return LeverOutcome(lever, status, decision=decision, details=dict(details or {}), challenger=challenger)


def _require_applicable(lever: str, family: str) -> None:
    if not lever_applies(lever, family):
        raise ValueError(f"Lever {lever} does not apply to {family}; log it as not applicable.")


def screen_lever(
    lever: str,
    family: str,
    s1_winner: PooledOof,
    challenger: PooledOof,
    *,
    component_of: Mapping[str, str],
    training_class_counts: Sequence[int] | np.ndarray,
    config: PromotionConfig = PromotionConfig(),
    positive_step_fraction: Optional[float] = None,
) -> LeverOutcome:
    """One lever applied alone to the S1 winner (L7 and L9 have their own rules).

    L5 needs ``positive_step_fraction``, the share of its training minibatches with a SupCon
    positive (``supcon_positive_step_fraction``); below 90% it is logged "not tested" and not compared.
    """
    if lever in (EPOCH_LEVER, LOGIT_ADJUSTMENT_LEVER):
        raise ValueError(f"Lever {lever} has its own screening rule.")
    _require_applicable(lever, family)
    details: dict[str, Any] = {}
    if lever == SUPCON_LEVER:
        if positive_step_fraction is None:
            raise ValueError("L5 needs positive_step_fraction, the share of minibatches with a SupCon positive.")
        fraction = float(positive_step_fraction)
        details["positive_step_fraction"] = fraction
        if not supcon_lever_testable(fraction):
            reason = (f"SupCon positives on {fraction:.1%} of training minibatches, below "
                      f"{SUPCON_MIN_POSITIVE_STEP_FRACTION:.0%}")
            return LeverOutcome(lever, LEVER_NOT_TESTED, reason=reason, details=details)
    elif positive_step_fraction is not None:
        raise ValueError(f"positive_step_fraction applies to L5 only, not {lever}.")
    decision = promote(s1_winner, challenger, component_of=component_of,
                       training_class_counts=training_class_counts, config=config)
    return _promotion_outcome(lever, decision, challenger, details)


def screen_epoch_lever(
    family: str,
    s1_winner: PooledOof,
    looks: Mapping[int, PooledOof],
    *,
    component_of: Mapping[str, str],
    training_class_counts: Sequence[int] | np.ndarray,
    config: PromotionConfig = PromotionConfig(),
) -> LeverOutcome:
    """L7: the better of 12 and 20 epochs (equal scores: 12) must pass ``promote`` against 30."""
    _require_applicable(EPOCH_LEVER, family)
    if set(looks) != set(EPOCH_LOOKS):
        raise ValueError(f"L7 needs exactly the looks {EPOCH_LOOKS}, got {sorted(looks)}.")
    for epochs in EPOCH_LOOKS:
        _require_paired(s1_winner, looks[epochs])
    scores = {epochs: seed_mean_ba_points(looks[epochs]) for epochs in EPOCH_LOOKS}
    top = max(scores.values())
    chosen = next(epochs for epochs in EPOCH_LOOKS if scores[epochs] >= top - FLOAT_TOLERANCE)
    decision = promote(s1_winner, looks[chosen], component_of=component_of,
                       training_class_counts=training_class_counts, config=config)
    details = {"incumbent_epochs": DEFAULT_EPOCHS, "look_scores_points": scores, "challenger_epochs": chosen}
    return _promotion_outcome(EPOCH_LEVER, decision, looks[chosen], details)


def screen_logit_adjustment(
    family: str,
    s1_winner: PooledOof,
    *,
    component_of: Mapping[str, str],
    training_class_counts: Sequence[int] | np.ndarray,
    config: PromotionConfig = promotion_config_for(LOGIT_ADJUSTMENT_LEVER),
) -> LeverOutcome:
    """L9: the best tau other than 0 by pooled BA must pass ``promote`` against tau = 0.

    Its four looks take their own promotion level (D1: ``L9_PROMOTION_ALPHA``), the default here.

    No extra fits: each row is adjusted with its own inner fit's training priors. Equal scores
    go to the smaller |tau|, then to the negative value (the order of ``TAUS``).
    """
    _require_applicable(LOGIT_ADJUSTMENT_LEVER, family)
    scores = {tau: seed_mean_ba_points(apply_tau(s1_winner, tau)) for tau in TAUS}
    candidates = sorted((tau for tau in TAUS if tau != 0.0), key=lambda tau: (abs(tau), tau))
    top = max(scores[tau] for tau in candidates)
    chosen = next(tau for tau in candidates if scores[tau] >= top - FLOAT_TOLERANCE)
    challenger = apply_tau(s1_winner, chosen)
    decision = promote(s1_winner, challenger, component_of=component_of,
                       training_class_counts=training_class_counts, config=config)
    details = {"tau_scores_points": {f"{tau:+.1f}": score for tau, score in scores.items()}, "chosen_tau": chosen}
    return _promotion_outcome(LOGIT_ADJUSTMENT_LEVER, decision, challenger, details)


def lever_not_applicable(lever: str, family: str) -> LeverOutcome:
    if lever_applies(lever, family):
        raise ValueError(f"Lever {lever} applies to {family}; it must be screened or logged as not tested.")
    return LeverOutcome(lever, LEVER_NOT_APPLICABLE, reason=f"not applicable to {family}")


def lever_not_tested(lever: str, family: str, reason: str) -> LeverOutcome:
    """An applicable lever left untested, e.g. L3 over its cost ceiling or L5 without positives."""
    _require_applicable(lever, family)
    if not reason.strip():
        raise ValueError("A lever left untested needs a reason.")
    return LeverOutcome(lever, LEVER_NOT_TESTED, reason=reason)


@dataclass(frozen=True, eq=False)
class LeverScreen:
    family: str
    s1_winner_sha256: str
    outcomes: tuple[LeverOutcome, ...]

    @property
    def passing(self) -> tuple[LeverOutcome, ...]:
        return tuple(outcome for outcome in self.outcomes if outcome.status == LEVER_PASSED)

    def to_record(self) -> dict[str, Any]:
        return {"family": self.family, "s1_winner_predictions_sha256": self.s1_winner_sha256,
                "outcomes": [outcome.to_record() for outcome in self.outcomes],
                "passing": [outcome.lever for outcome in self.passing]}


def _check_screened_outcome(lever: str, outcome: LeverOutcome, s1_sha256: str) -> None:
    """A passed or failed lever must carry a decision that matches its status, against the S1 winner."""
    if outcome.decision is None or outcome.challenger is None:
        raise ValueError(f"Lever {lever} was logged {outcome.status!r} without a promotion decision.")
    if outcome.decision.promoted != (outcome.status == LEVER_PASSED):
        raise ValueError(f"Lever {lever} was logged {outcome.status!r} but its decision has promoted="
                         f"{outcome.decision.promoted}.")
    if outcome.decision.inputs["incumbent_predictions_sha256"] != s1_sha256:
        raise ValueError(f"Lever {lever} was compared with {outcome.decision.incumbent!r}, not the S1 winner.")
    if outcome.decision.inputs["challenger_predictions_sha256"] != predictions_sha256(outcome.challenger):
        raise ValueError(f"Lever {lever}'s challenger predictions differ from the ones its decision scored.")
    if lever == SUPCON_LEVER:
        fraction = outcome.details.get("positive_step_fraction")
        if fraction is None or not supcon_lever_testable(fraction):
            raise ValueError("L5 was screened without SupCon positives on at least 90% of minibatches.")


def s2_lever_screen(family: str, s1_winner: PooledOof, outcomes: Iterable[LeverOutcome]) -> LeverScreen:
    """Check that every lever has exactly one outcome consistent with the family and with the S1
    winner (screened levers must be decided against it); order by lever."""
    _check_family(family)
    s1_sha256 = predictions_sha256(s1_winner)
    by_lever: dict[str, LeverOutcome] = {}
    for outcome in outcomes:
        if outcome.lever in by_lever:
            raise ValueError(f"Lever {outcome.lever} has more than one outcome.")
        if outcome.lever not in LEVERS:
            raise ValueError(f"Unknown lever {outcome.lever!r}.")
        by_lever[outcome.lever] = outcome
    missing = [lever for lever in LEVERS if lever not in by_lever]
    if missing:
        raise ValueError(f"Levers without an outcome: {missing}.")
    for lever in LEVERS:
        outcome = by_lever[lever]
        applicable = lever_applies(lever, family)
        if applicable == (outcome.status == LEVER_NOT_APPLICABLE):
            raise ValueError(f"Lever {lever} is {'applicable' if applicable else 'not applicable'} to {family} "
                             f"but was logged {outcome.status!r}.")
        if outcome.status in (LEVER_PASSED, LEVER_FAILED):
            _check_screened_outcome(lever, outcome, s1_sha256)
        elif outcome.status in (LEVER_NOT_APPLICABLE, LEVER_NOT_TESTED):
            if outcome.decision is not None or outcome.challenger is not None:
                raise ValueError(f"Lever {lever} was logged {outcome.status!r} but carries a comparison.")
        else:
            raise ValueError(f"Lever {lever} has an unknown status {outcome.status!r}.")
    return LeverScreen(family, s1_sha256, tuple(by_lever[lever] for lever in LEVERS))


# ---------------------------------------------------------------------------
# S3: combination; S4: refit specification


def best_single_lever(screen: LeverScreen) -> Optional[LeverOutcome]:
    """The passing lever with the highest pooled BA (equal scores: the earlier lever).

    ``s2_lever_screen`` checks that every lever was decided against the same S1 winner, so this
    is also the largest gain.
    """
    best, best_score = None, -math.inf
    for outcome in screen.passing:
        score = seed_mean_ba_points(outcome.challenger)
        if score > best_score + FLOAT_TOLERANCE:
            best, best_score = outcome, score
    return best


@dataclass(frozen=True, eq=False)
class CombinationOutcome:
    selected: str
    adopted_levers: tuple[str, ...]
    best_single_lever: Optional[str]
    decision: Optional[PromotionDecision]
    reason: str
    selected_oof: PooledOof = field(repr=False)

    def to_record(self) -> dict[str, Any]:
        return to_jsonable({
            "selected": self.selected,
            "adopted_levers": self.adopted_levers,
            "best_single_lever": self.best_single_lever,
            "decision": None if self.decision is None else self.decision.to_record(),
            "reason": self.reason,
            "selected_predictions_sha256": predictions_sha256(self.selected_oof),
        })


def s3_combination(
    s1_winner: PooledOof,
    screen: LeverScreen,
    combination: Optional[PooledOof] = None,
    *,
    component_of: Mapping[str, str],
    training_class_counts: Sequence[int] | np.ndarray,
    config: PromotionConfig = PromotionConfig(),
) -> CombinationOutcome:
    """Combine two or more passing levers; adopt the combination only if it beats the best single lever.

    ``combination`` is the pooled prediction of one fit with every passing lever (L9 applied
    with the tau chosen in S2); it is required exactly when two or more levers passed.
    """
    if predictions_sha256(s1_winner) != screen.s1_winner_sha256:
        raise ValueError(f"The lever screen was built against another S1 winner than {s1_winner.name!r}.")
    passing = screen.passing
    if len(passing) < 2 and combination is not None:
        raise ValueError("A combination is evaluated only when at least two levers passed.")
    if not passing:
        return CombinationOutcome("s1_winner", (), None, None, "no lever passed; the S1 winner stays", s1_winner)
    best = best_single_lever(screen)
    if len(passing) == 1:
        return CombinationOutcome(best.lever, (best.lever,), best.lever, None, "one lever passed", best.challenger)
    if combination is None:
        raise ValueError(f"Levers {[o.lever for o in passing]} passed; S3 needs the combined fit's predictions.")
    decision = promote(best.challenger, combination, component_of=component_of,
                       training_class_counts=training_class_counts, config=config)
    if decision.promoted:
        levers = tuple(outcome.lever for outcome in passing)
        return CombinationOutcome("combination", levers, best.lever, decision,
                                  "the combination passed against the best single lever", combination)
    return CombinationOutcome(best.lever, (best.lever,), best.lever, decision,
                              "the combination did not pass; the best single lever stays", best.challenger)


def s4_refit_specification(
    *,
    family: str,
    learning_rate_setting: Mapping[str, float],
    adopted_levers: Sequence[str],
    epochs: int = DEFAULT_EPOCHS,
    tau: float = 0.0,
    seeds: Sequence[int] = REFIT_SEEDS,
) -> dict[str, Any]:
    """The selected recipe for the fold refit, checked for internal consistency.

    ``learning_rate_setting`` must be an entry of the family's S1 grid (the S1 choice's
    ``chosen_setting``); the seeds must be exactly ``REFIT_SEEDS``.
    """
    _check_family(family)
    if family in CONTROL_FAMILIES:
        raise ValueError(f"{family} is a control and is never refit as the selected model.")
    setting = _grid_setting(family, learning_rate_setting)
    levers = [lever for lever in LEVERS if lever in set(adopted_levers)]
    if len(levers) != len(adopted_levers) or len(set(adopted_levers)) != len(adopted_levers):
        raise ValueError(f"Adopted levers must be distinct known levers, got {list(adopted_levers)}.")
    for lever in levers:
        _require_applicable(lever, family)
    if (EPOCH_LEVER in levers) != (int(epochs) in EPOCH_LOOKS) or int(epochs) not in (*EPOCH_LOOKS, DEFAULT_EPOCHS):
        raise ValueError(f"Epochs {epochs} do not match the adopted levers {levers}.")
    tau = validate_tau(tau)
    if (LOGIT_ADJUSTMENT_LEVER in levers) != (tau != 0.0):
        raise ValueError(f"tau {tau} does not match the adopted levers {levers}.")
    seeds = [int(seed) for seed in seeds]
    if tuple(seeds) != REFIT_SEEDS:
        raise ValueError(f"The refit uses seeds {list(REFIT_SEEDS)}, got {seeds}.")
    recompute = ["normalisation", "class_weights", "class_priors"]
    if "L1" in levers:
        recompute.append("gvp_weight_decay_for_target_shrink")
    return {
        "family": family,
        "learning_rate_setting": setting,
        "learning_rate_label": learning_rate_label(setting),
        "adopted_levers": levers,
        "epochs": int(epochs),
        "logit_adjustment_tau": tau,
        "seeds": seeds,
        "recompute_on_refit_training_set": recompute,
    }


# ---------------------------------------------------------------------------
# Hash-chained step records


def record_sha256(record: Mapping[str, Any]) -> str:
    return sha256_canonical(record)


def _check_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be a lower-case SHA-256 hex digest.")
    return value


class StepChain:
    """Append-only step records; each holds the SHA-256 of the previous record's canonical JSON.

    The first record points at ``anchor_sha256`` (for example the Freeze-1 manifest hash).
    """

    def __init__(self, anchor_sha256: str = GENESIS_SHA256, records: Sequence[Mapping[str, Any]] = ()):
        self.anchor_sha256 = _check_sha256(anchor_sha256, "anchor_sha256")
        self._records = [to_jsonable(record) for record in records]
        verify_step_chain(self._records, anchor_sha256=self.anchor_sha256)

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._records)

    @property
    def head_sha256(self) -> str:
        return record_sha256(self._records[-1]) if self._records else self.anchor_sha256

    def append(self, step: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not step:
            raise ValueError("A step record needs a step name.")
        record = to_jsonable({
            "index": len(self._records),
            "step": step,
            "previous_sha256": self.head_sha256,
            "payload": payload,
        })
        self._records.append(record)
        return record


def verify_step_chain(records: Sequence[Mapping[str, Any]], *, anchor_sha256: str = GENESIS_SHA256) -> str:
    """Raise on a broken link or index; return the head hash."""
    previous = _check_sha256(anchor_sha256, "anchor_sha256")
    for position, record in enumerate(records):
        if set(record) != {"index", "step", "previous_sha256", "payload"}:
            raise ValueError(f"Step record {position} has unexpected fields {sorted(record)}.")
        if record["index"] != position:
            raise ValueError(f"Step record {position} carries index {record['index']}.")
        if record["previous_sha256"] != previous:
            raise ValueError(f"Step record {position} does not link to the previous record.")
        previous = record_sha256(record)
    return previous


def append_step_record(path: Path, step: str, payload: Mapping[str, Any], *,
                       anchor_sha256: str = GENESIS_SHA256) -> dict[str, Any]:
    """Verify the chain in a JSON-lines file and append one record."""
    path = Path(path)
    chain = StepChain(anchor_sha256, read_step_chain(path, anchor_sha256=anchor_sha256) if path.exists() else ())
    record = chain.append(step, payload)
    with path.open("a", encoding="ascii") as handle:
        handle.write(canonical_json(record) + "\n")
    return record


def read_step_chain(path: Path, *, anchor_sha256: str = GENESIS_SHA256) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in Path(path).read_text(encoding="ascii").splitlines() if line.strip()]
    verify_step_chain(records, anchor_sha256=anchor_sha256)
    return records


# ---------------------------------------------------------------------------
# Null operating characteristics


@dataclass(frozen=True)
class NullNoiseModel:
    """Declared noise model for the share of null levers the promotion rule adopts.

    ``n_proteins`` training proteins fall into identity components with geometric sizes (mean
    ``mean_component_size``). Each component draws one EC class from ``class_proportions``;
    each member keeps it with probability ``component_class_purity`` and otherwise redraws.
    Fit f predicts protein i correctly with probability

        sigmoid(logit(class_recall[c_i]) + u[component(i)] + v[i] + s[f] + e[f, i] + shift)

    with u ~ N(0, component_sd) and v ~ N(0, protein_sd) shared by all fits (difficulty),
    s ~ N(0, fit_shift_sd) per fit, e ~ N(0, fit_noise_sd) per fit and protein, and ``shift``
    0 for the incumbent and ``challenger_shift`` for challengers (0 under the null). A wrong
    prediction draws another class in proportion to ``class_proportions``. Looks (L7: 2,
    L9: 4) are simulated as independent challenger fits; L9's looks reuse one fit, so this
    overstates its multiplicity. The defaults are illustrative, not fitted to campaign data.

    The fit-level shift ``s`` is invisible to a bootstrap over proteins, so it raises null
    adoption above the bound's nominal level; ``fit_shift_sd`` should be set from measured
    seed-to-seed variation before the simulation is recorded.
    """

    n_proteins: int = 600
    class_proportions: tuple[float, ...] = (0.24, 0.26, 0.30, 0.10, 0.04, 0.055, 0.005)
    mean_component_size: float = 1.6
    component_class_purity: float = 0.9
    class_recall: tuple[float, ...] = (0.75, 0.75, 0.80, 0.65, 0.45, 0.55, 0.25)
    component_sd: float = 1.0
    protein_sd: float = 1.0
    fit_shift_sd: float = 0.15
    fit_noise_sd: float = 0.8
    challenger_shift: float = 0.0

    def __post_init__(self) -> None:
        proportions = np.asarray(self.class_proportions, dtype=np.float64)
        recall = np.asarray(self.class_recall, dtype=np.float64)
        if proportions.shape != (N_EC_CLASSES,) or np.any(proportions < 0) or abs(proportions.sum() - 1.0) > 1e-6:
            raise ValueError(f"class_proportions must be {N_EC_CLASSES} non-negative values summing to 1.")
        if recall.shape != (N_EC_CLASSES,) or np.any(recall <= 0) or np.any(recall >= 1):
            raise ValueError(f"class_recall must be {N_EC_CLASSES} values strictly between 0 and 1.")
        if self.n_proteins < 2 or self.mean_component_size < 1.0 or not 0.0 <= self.component_class_purity <= 1.0:
            raise ValueError("Invalid population size, component size or class purity.")
        if min(self.component_sd, self.protein_sd, self.fit_shift_sd, self.fit_noise_sd) < 0:
            raise ValueError("Standard deviations must be non-negative.")

    def to_record(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass(frozen=True)
class OperatingCharacteristics:
    n_replicates: int
    looks: int
    n_adopted: int
    adoption_rate: float
    adoption_rate_standard_error: float
    target_rate: float
    gain_pass_rate: float
    recall_guard_veto_rate: float
    zero_recall_veto_rate: float
    n_bootstrap_evaluated: int
    n_lower_bound_passed: int
    n_bootstrap_aborted: int
    gain_points_mean: float
    gain_points_sd: float
    seed: int
    noise_model: Mapping[str, Any]
    promotion_config: Mapping[str, Any]

    def to_record(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


def _simulated_population(rng: np.random.Generator, model: NullNoiseModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    proportions = np.asarray(model.class_proportions, dtype=np.float64)
    proportions = proportions / proportions.sum()
    sizes: list[int] = []
    while sum(sizes) < model.n_proteins:
        sizes.append(int(rng.geometric(1.0 / model.mean_component_size)))
    sizes[-1] -= sum(sizes) - model.n_proteins
    component = np.repeat(np.arange(len(sizes)), sizes)
    true = rng.choice(N_EC_CLASSES, size=len(sizes), p=proportions)[component]
    redraw = rng.random(model.n_proteins) >= model.component_class_purity
    true[redraw] = rng.choice(N_EC_CLASSES, size=int(redraw.sum()), p=proportions)
    recall = np.asarray(model.class_recall, dtype=np.float64)
    difficulty = (np.log(recall / (1.0 - recall))[true]
                  + rng.normal(0.0, model.component_sd, len(sizes))[component]
                  + rng.normal(0.0, model.protein_sd, model.n_proteins))
    return true.astype(np.int64), component, difficulty


def _simulated_fit(rng: np.random.Generator, model: NullNoiseModel, true: np.ndarray,
                   difficulty: np.ndarray, shift: float) -> np.ndarray:
    eta = difficulty + rng.normal(0.0, model.fit_shift_sd) + rng.normal(0.0, model.fit_noise_sd, true.size) + shift
    correct = rng.random(true.size) < 1.0 / (1.0 + np.exp(-eta))
    predicted = true.copy()
    wrong = np.flatnonzero(~correct)
    if wrong.size:
        weights = np.tile(np.asarray(model.class_proportions, dtype=np.float64), (wrong.size, 1))
        weights[np.arange(wrong.size), true[wrong]] = 0.0
        cumulative = np.cumsum(weights / weights.sum(axis=1, keepdims=True), axis=1)
        draws = rng.random(wrong.size)[:, None]
        predicted[wrong] = np.minimum((cumulative < draws).sum(axis=1), N_EC_CLASSES - 1)
    return predicted


def _one_hot(predictions: np.ndarray) -> np.ndarray:
    return np.eye(N_EC_CLASSES)[predictions]


def simulate_null_adoption(
    noise_model: NullNoiseModel = NullNoiseModel(),
    *,
    n_replicates: int = 200,
    looks: int = 1,
    seeds: Sequence[int] = SCREENING_SEEDS,
    config: PromotionConfig = PromotionConfig(),
    seed: int = DEFAULT_SIMULATION_SEED,
) -> OperatingCharacteristics:
    """Share of simulated levers adopted by ``promote``; under the null the target is about 10%.

    Each replicate draws a population, an incumbent with ``len(seeds)`` fits and ``looks``
    challengers; the challenger with the best seed-mean BA (equal scores: the first) is tested.
    A bootstrap abort (too many draws lacking an EC class 1-6) counts as not adopted and is
    reported in ``n_bootstrap_aborted``.
    """
    if n_replicates < 1 or looks < 1:
        raise ValueError("n_replicates and looks must be positive.")
    rng = np.random.default_rng(int(seed))
    seeds = tuple(int(value) for value in seeds)
    ids = tuple(f"p{index:05d}" for index in range(noise_model.n_proteins))
    adopted = gain_pass = guard_veto = zero_veto = evaluated = bound_pass = aborted = 0
    gains = []
    for _ in range(int(n_replicates)):
        true, component, difficulty = _simulated_population(rng, noise_model)
        component_of = {pid: f"c{value}" for pid, value in zip(ids, component.tolist())}
        counts = np.bincount(true, minlength=N_EC_CLASSES)

        def fits(name: str, shift: float) -> PooledOof:
            predictions = np.stack([_simulated_fit(rng, noise_model, true, difficulty, shift) for _ in seeds])
            return PooledOof(name, ids, true, seeds, _one_hot(predictions))

        incumbent = fits("incumbent", 0.0)
        candidates = [fits(f"look_{look}", noise_model.challenger_shift) for look in range(int(looks))]
        scores = [seed_mean_ba_points(candidate) for candidate in candidates]
        best = next(k for k, score in enumerate(scores) if score >= max(scores) - FLOAT_TOLERANCE)
        try:
            decision = promote(incumbent, candidates[best], component_of=component_of, training_class_counts=counts,
                               config=config, evaluate_all_checks=False)
        except BootstrapAbort:
            # Reached only after the gain and recall checks passed; the abort counts as not adopted.
            gains.append(scores[best] - seed_mean_ba_points(incumbent))
            gain_pass += 1
            evaluated += 1
            aborted += 1
            continue
        gains.append(decision.gain_points)
        adopted += decision.promoted
        gain_pass += bool(decision.checks["gain"])
        guard_veto += not decision.checks["recall_guard"]
        zero_veto += not decision.checks["zero_recall"]
        evaluated += decision.checks["lower_bound"] is not None
        bound_pass += decision.checks["lower_bound"] is True
    rate = adopted / n_replicates
    return OperatingCharacteristics(
        n_replicates=int(n_replicates),
        looks=int(looks),
        n_adopted=int(adopted),
        adoption_rate=rate,
        adoption_rate_standard_error=math.sqrt(rate * (1.0 - rate) / n_replicates),
        target_rate=NULL_ADOPTION_TARGET,
        gain_pass_rate=gain_pass / n_replicates,
        recall_guard_veto_rate=guard_veto / n_replicates,
        zero_recall_veto_rate=zero_veto / n_replicates,
        n_bootstrap_evaluated=int(evaluated),
        n_lower_bound_passed=int(bound_pass),
        n_bootstrap_aborted=int(aborted),
        gain_points_mean=float(np.mean(gains)),
        gain_points_sd=float(np.std(gains, ddof=1)) if len(gains) > 1 else 0.0,
        seed=int(seed),
        noise_model=noise_model.to_record(),
        promotion_config=config.to_record(),
    )


# ---------------------------------------------------------------------------
# Noise-model calibration from a certified inner membership


def membership_noise_statistics(rows: Iterable[Mapping[str, Any]], components: Mapping[str, str]) -> dict[str, Any]:
    """Population statistics of one fold's certified inner membership, as ``NullNoiseModel`` reads them.

    ``rows`` are membership rows with ``accession``, ``group_id`` and ``ec_class`` (token "1".."7");
    ``components`` maps every node of the inner identity graph to its component, and each row's
    ``group_id`` must equal its accession's component. Component sizes count the membership's
    proteins only, the unit the promotion bootstrap resamples.

    Purity is a moment estimate under the model's own assumption: two members of one component
    share their class with probability ``q**2 + (1 - q**2) * S``, where ``S`` is the sum of the
    squared class proportions, so ``q = sqrt((A - S) / (1 - S))`` from the agreement ``A`` pooled
    over all within-component pairs (clipped to [0, 1]). Without a component of two or more
    proteins ``q`` is not identified and is set to 1; it then has no effect, because the mean
    component size is 1 and every simulated component is a singleton. The geometric size model
    is compared with the observed sizes through their variances.

    The result holds per-fold class proportions: it belongs in private records only. Refusals
    name row numbers and counts, never an accession, a group or a class: a step that reads every
    fold's membership must not print another fold's proteins.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("The inner membership has no rows.")
    class_counts = np.zeros(N_EC_CLASSES, dtype=np.int64)
    members: dict[str, list[int]] = {}
    seen: set[str] = set()
    failed: dict[str, list[int]] = {}
    for number, row in enumerate(rows, start=1):
        accession, group = str(row.get("accession") or ""), str(row.get("group_id") or "")
        token = str(row.get("ec_class", ""))
        if not accession or not group:
            reason = "lack an accession or a group"
        elif accession in seen:
            reason = "repeat the accession of an earlier row"
        elif token not in EC_CLASS_TOKEN_TO_INDEX:
            reason = "have no EC class 1-7"
        elif accession not in components:
            reason = "have no inner identity component"
        elif str(components[accession]) != group:
            reason = "name a group other than their inner component"
        else:
            seen.add(accession)
            index = EC_CLASS_TOKEN_TO_INDEX[token]
            class_counts[index] += 1
            members.setdefault(group, []).append(index)
            continue
        seen.add(accession)
        failed.setdefault(reason, []).append(number)
    if failed:
        raise ValueError("The inner membership cannot calibrate the noise model: " + "; ".join(
            f"{len(numbers)} row(s) {reason} (first at data row {numbers[0]})" for reason, numbers in failed.items()))
    n = len(rows)
    if int(np.count_nonzero(class_counts)) < 2:
        raise ValueError("The membership holds fewer than two EC classes; the noise model needs at least two.")
    proportions = class_counts / n
    concentration = float(np.sum(proportions ** 2))
    sizes = np.array([len(classes) for classes in members.values()], dtype=np.float64)
    pairs = agreeing = 0
    for classes in members.values():
        counts = np.bincount(classes, minlength=N_EC_CLASSES)
        pairs += len(classes) * (len(classes) - 1) // 2
        agreeing += int(np.sum(counts * (counts - 1) // 2))
    agreement = agreeing / pairs if pairs else None
    if agreement is None:
        purity = 1.0
    else:
        purity = math.sqrt(min(1.0, max(0.0, (agreement - concentration) / (1.0 - concentration))))
    mean_size = n / len(members)
    return {
        "n_proteins": n,
        "class_proportions": tuple(float(value) for value in proportions),
        "n_ec_classes_present": int(np.count_nonzero(class_counts)),
        "n_components": len(members),
        "mean_component_size": mean_size,
        "largest_component_size": int(sizes.max()),
        "singleton_components": int(np.sum(sizes == 1)),
        "component_size_variance": float(sizes.var()),
        "geometric_size_variance": mean_size * (mean_size - 1.0),
        "within_component_pairs": int(pairs),
        "within_component_pair_agreement": agreement,
        "class_concentration": concentration,
        "component_class_purity": purity,
        "purity_identified": agreement is not None,
    }


def noise_model_from_membership(
    rows: Iterable[Mapping[str, Any]],
    components: Mapping[str, str],
    *,
    base: NullNoiseModel = NullNoiseModel(),
) -> NullNoiseModel:
    """``base`` with the population taken from a certified inner membership.

    Sets ``n_proteins``, ``class_proportions``, ``mean_component_size`` and
    ``component_class_purity`` (``membership_noise_statistics``); the recall, difficulty and
    fit-noise parameters stay as in ``base``.
    """
    stats = membership_noise_statistics(rows, components)
    return replace(
        base,
        n_proteins=stats["n_proteins"],
        class_proportions=stats["class_proportions"],
        mean_component_size=stats["mean_component_size"],
        component_class_purity=stats["component_class_purity"],
    )
