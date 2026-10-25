"""Paired cluster bootstrap, one-sided bounds, intersection-union and Holm tests (numpy only).

One engine serves the inner-fold selection and the frozen scorer, so neither imports the trainer.
Every system is scored on the same proteins. Per-cluster confusion counts are built once; a
bootstrap draw is a vector of cluster multiplicities, and the resampled confusion matrices are a
matrix product of multiplicities and counts, so no protein is re-indexed per draw.

Conventions:
- Metric values are fractions (0..1); callers convert to points.
- One table per population. Strata and resampling use every row of a table, so a population
  (for example B1-NH30) gets its own table; subsets serve contrasts within one population, such
  as a difference in differences. A table whose subsets leave rows uncovered is refused unless
  the caller opts in with ``allow_rows_outside_subsets``; the count is recorded.
- Strata: each cluster joins the stratum of the rarest EC class among classes 1-6 it contains.
  Rarity is the class's total protein weight in the population; equal weights go to the lower
  class number. Clusters holding only class-7 proteins form their own stratum. Each stratum is
  resampled with replacement to its own cluster count. Unstratified runs use one stratum.
- A draw is skipped, and counted, when it lacks an EC class 1-6 that is present in the
  population (per subset, over that subset's own population). Class 7 never causes a skip.
  More than ``max_skip_fraction`` skipped draws aborts the run.
- The one-sided lower bound at ``alpha`` is the order statistic d[k] (0-based) of the sorted
  valid draws, k = floor(alpha * n). The p-value for H0: difference <= 0 is the share of valid
  draws <= 0. Draws exactly at 0 count toward H0, so identical systems give p = 1 and a lower
  bound of 0. With these definitions the lower bound is > 0 exactly when p <= alpha.
- Draws depend on the seed, the draw count, the stratum order and ``DRAW_BATCH_SIZE``; all are
  fixed here so a frozen seed reproduces the same draws. They also depend on the NumPy version
  and bit generator, which the record carries. With dyadic counts (integer or half weights) the
  matrix-product sums are exact; with other weights (for example 1/3 for an accession listed
  three times) the summation order of the linear-algebra library can change the last bits, and
  with them ``contrast_draws_sha256``. The record says which case applies.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from training.ec_class_metrics import N_EC_CLASSES, PRIMARY_CLASS_INDICES


METRIC_EC_CLASSES_1_6_BALANCED_ACCURACY = "ec_classes_1_6_balanced_accuracy"
METRIC_EC_CLASSES_1_6_MACRO_F1 = "ec_classes_1_6_macro_f1"
METRIC_ACCURACY = "accuracy"
BOOTSTRAP_METRICS = (
    METRIC_EC_CLASSES_1_6_BALANCED_ACCURACY,
    METRIC_EC_CLASSES_1_6_MACRO_F1,
    METRIC_ACCURACY,
)

DEFAULT_N_DRAWS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20261023
DEFAULT_MAX_SKIP_FRACTION = 0.01
DRAW_BATCH_SIZE = 1_000
DEFAULT_SUBSET = "all"
CLASS_7_ONLY_STRATUM = "class_7_only"
UNSTRATIFIED_STRATUM = "all_clusters"

# Absorbs float rounding of alpha * n (e.g. 0.05 / 2 * 10_000) and of p-value comparisons.
_ORDER_TOLERANCE = 1e-9
P_VALUE_TOLERANCE = 1e-12
# Counts that are multiples of 2**-20 sum exactly in float64 at campaign sizes.
_DYADIC_SCALE = 2.0 ** 20


class BootstrapAbort(RuntimeError):
    """Raised when too many bootstrap draws lack a required EC class."""


def rarest_class_stratum(class_index: int) -> str:
    return f"rarest_ec_class_{class_index + 1}"


def _stratum_order_key(name: str) -> tuple[int, str]:
    for index in PRIMARY_CLASS_INDICES:
        if name == rarest_class_stratum(index):
            return index, name
    return len(PRIMARY_CLASS_INDICES), name


def _check_alpha(alpha: float) -> float:
    value = float(alpha)
    if not 0.0 < value < 1.0:
        raise ValueError(f"alpha must lie strictly between 0 and 1, got {alpha!r}.")
    return value


def _class_indices(values: Sequence[int] | np.ndarray, name: str, length: int) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {array.shape}.")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must hold integer class indices.")
    array = array.astype(np.int64, copy=False)
    if array.size and (array.min() < 0 or array.max() >= N_EC_CLASSES):
        raise ValueError(f"{name} holds class indices outside 0..{N_EC_CLASSES - 1}.")
    return array


@dataclass(frozen=True, eq=False)
class ClusterCountTable:
    """Weighted confusion counts per cluster: ``counts[cluster, subset, arm, true, predicted]``.

    ``class_support[cluster, class]`` holds the true-class weights of every row, whatever its
    subsets; strata are assigned from it.
    """

    cluster_ids: tuple[str, ...]
    arm_names: tuple[str, ...]
    subset_names: tuple[str, ...]
    counts: np.ndarray
    class_support: np.ndarray
    n_rows: int
    n_rows_outside_subsets: int = 0

    def subset_support(self, subset: str = DEFAULT_SUBSET) -> np.ndarray:
        index = self.subset_names.index(subset)
        return self.counts[:, index, 0].sum(axis=(0, 2))

    @property
    def dyadic_counts(self) -> bool:
        scaled = self.counts * _DYADIC_SCALE
        return bool(np.all(scaled == np.round(scaled)))


def build_cluster_count_table(
    protein_ids: Sequence[str],
    true_idx: Sequence[int] | np.ndarray,
    arm_predictions: Mapping[str, Sequence[int] | np.ndarray],
    cluster_of: Mapping[str, str],
    *,
    weights: Optional[Sequence[float] | np.ndarray] = None,
    subsets: Optional[Mapping[str, Sequence[bool] | np.ndarray]] = None,
    allow_rows_outside_subsets: bool = False,
) -> ClusterCountTable:
    """Per-cluster confusion counts for paired arms scored on the same rows.

    An arm is one prediction vector (one system and seed). A protein listed on several rows
    (for example a repeated accession weighted to a total of one) maps to one cluster, so it is
    resampled as a unit. The rows are the population: every row must lie in some subset unless
    ``allow_rows_outside_subsets`` is set (see the module conventions).
    """
    ids = [str(value) for value in protein_ids]
    n_rows = len(ids)
    if n_rows == 0:
        raise ValueError("The bootstrap needs at least one protein.")
    true = _class_indices(true_idx, "true_idx", n_rows)
    if not arm_predictions:
        raise ValueError("At least one prediction arm is required.")
    arm_names = tuple(str(name) for name in arm_predictions)
    if len(set(arm_names)) != len(arm_names):
        raise ValueError("Arm names must be unique.")
    predictions = [_class_indices(arm_predictions[name], f"predictions of {name!r}", n_rows) for name in arm_predictions]
    if weights is None:
        row_weight = np.ones(n_rows, dtype=np.float64)
    else:
        row_weight = np.asarray(weights, dtype=np.float64)
        if row_weight.shape != (n_rows,):
            raise ValueError(f"weights must have shape ({n_rows},), got {row_weight.shape}.")
        if not np.all(np.isfinite(row_weight)) or np.any(row_weight <= 0):
            raise ValueError("weights must be finite and positive.")
    missing = sorted({pid for pid in ids if pid not in cluster_of})
    if missing:
        raise ValueError(f"{len(missing)} proteins have no cluster, e.g. {missing[:5]}.")
    row_clusters = [str(cluster_of[pid]) for pid in ids]
    cluster_ids = tuple(sorted(set(row_clusters)))
    cluster_index = {value: index for index, value in enumerate(cluster_ids)}
    row_cluster = np.array([cluster_index[value] for value in row_clusters], dtype=np.int64)

    if subsets is None:
        subsets = {DEFAULT_SUBSET: np.ones(n_rows, dtype=bool)}
    subset_names = tuple(str(name) for name in subsets)
    masks = []
    for name in subsets:
        mask = np.asarray(subsets[name])
        if mask.shape != (n_rows,) or mask.dtype != np.bool_:
            raise ValueError(f"Subset {name!r} must be a boolean mask of shape ({n_rows},).")
        if not mask.any():
            raise ValueError(f"Subset {name!r} is empty.")
        masks.append(mask)
    outside = int(np.count_nonzero(~np.logical_or.reduce(masks)))
    if outside and not allow_rows_outside_subsets:
        raise ValueError(
            f"{outside} of {n_rows} rows lie in no subset. Strata and resampling use every row of the "
            "table, so build one table per population (subsets are for contrasts within it), or pass "
            "allow_rows_outside_subsets=True."
        )

    counts = np.zeros((len(cluster_ids), len(subset_names), len(arm_names), N_EC_CLASSES, N_EC_CLASSES))
    for subset_index, mask in enumerate(masks):
        for arm_index, predicted in enumerate(predictions):
            np.add.at(counts, (row_cluster[mask], subset_index, arm_index, true[mask], predicted[mask]), row_weight[mask])
    class_support = np.zeros((len(cluster_ids), N_EC_CLASSES))
    np.add.at(class_support, (row_cluster, true), row_weight)
    return ClusterCountTable(
        cluster_ids=cluster_ids,
        arm_names=arm_names,
        subset_names=subset_names,
        counts=counts,
        class_support=class_support,
        n_rows=n_rows,
        n_rows_outside_subsets=outside,
    )


def assign_cluster_strata(class_support: np.ndarray) -> tuple[str, ...]:
    """Stratum per cluster: the rarest EC class 1-6 it contains, else the class-7-only stratum."""
    support = np.asarray(class_support, dtype=np.float64)
    if support.ndim != 2 or support.shape[1] != N_EC_CLASSES:
        raise ValueError(f"class_support must have shape (n_clusters, {N_EC_CLASSES}), got {support.shape}.")
    population = support.sum(axis=0)
    present = [index for index in PRIMARY_CLASS_INDICES if population[index] > 0]
    rank = {index: position for position, index in enumerate(sorted(present, key=lambda c: (population[c], c)))}
    strata = []
    for row in support:
        contained = [index for index in present if row[index] > 0]
        strata.append(rarest_class_stratum(min(contained, key=rank.__getitem__)) if contained else CLASS_7_ONLY_STRATUM)
    return tuple(strata)


def stratum_members(strata: Sequence[str]) -> dict[str, np.ndarray]:
    """Cluster indices per stratum, in the fixed stratum order used for drawing."""
    names = sorted(set(strata), key=_stratum_order_key)
    labels = np.asarray(strata, dtype=object)
    return {name: np.flatnonzero(labels == name) for name in names}


def stratified_multiplicities(
    rng: np.random.Generator,
    members: Sequence[np.ndarray],
    n_clusters: int,
    n_draws: int,
) -> np.ndarray:
    """Cluster multiplicities ``[n_draws, n_clusters]``; each stratum keeps its cluster count."""
    multiplicity = np.zeros((n_draws, n_clusters), dtype=np.float64)
    offsets = np.arange(n_draws, dtype=np.int64)[:, None]
    for indices in members:
        size = int(indices.size)
        picks = rng.integers(0, size, size=(n_draws, size))
        counts = np.bincount((picks + offsets * size).ravel(), minlength=n_draws * size)
        multiplicity[:, indices] = counts.reshape(n_draws, size)
    return multiplicity


def confusion_metric(confusion: np.ndarray, metric: str, classes: Sequence[int]) -> np.ndarray:
    """Metric over the trailing (true, predicted) axes; ``classes`` are the averaged class indices.

    Balanced accuracy is NaN when an averaged class has no support; macro-F1 follows
    ``training.ec_class_metrics`` (zero precision or F1 where undefined).
    """
    matrix = np.asarray(confusion, dtype=np.float64)
    if matrix.shape[-2:] != (N_EC_CLASSES, N_EC_CLASSES):
        raise ValueError(f"confusion must end in ({N_EC_CLASSES}, {N_EC_CLASSES}), got {matrix.shape}.")
    diagonal = np.diagonal(matrix, axis1=-2, axis2=-1)
    if metric == METRIC_ACCURACY:
        return diagonal.sum(axis=-1) / matrix.sum(axis=(-2, -1))
    selected = np.asarray(classes, dtype=np.int64)
    if selected.size == 0:
        raise ValueError("At least one class must be averaged.")
    support = matrix.sum(axis=-1)
    recall = np.divide(diagonal, support, out=np.full_like(diagonal, np.nan), where=support > 0)
    if metric == METRIC_EC_CLASSES_1_6_BALANCED_ACCURACY:
        return recall[..., selected].mean(axis=-1)
    if metric == METRIC_EC_CLASSES_1_6_MACRO_F1:
        predicted = matrix.sum(axis=-2)
        precision = np.divide(diagonal, predicted, out=np.zeros_like(diagonal), where=predicted > 0)
        recall = np.nan_to_num(recall, nan=0.0)
        denominator = precision + recall
        f1 = np.divide(2.0 * precision * recall, denominator, out=np.zeros_like(diagonal), where=denominator > 0)
        return f1[..., selected].mean(axis=-1)
    raise ValueError(f"Unknown bootstrap metric {metric!r}; expected one of {BOOTSTRAP_METRICS}.")


@dataclass(frozen=True)
class Contrast:
    """A linear combination of system values: ``sum(coefficient * value[system, subset])``."""

    name: str
    terms: tuple[tuple[float, str, str], ...]

    @classmethod
    def difference(cls, name: str, minuend: str, subtrahend: str, subset: str = DEFAULT_SUBSET) -> "Contrast":
        return cls(name, ((1.0, minuend, subset), (-1.0, subtrahend, subset)))


def one_sided_lower_bound(draws: Sequence[float] | np.ndarray, alpha: float) -> float:
    """Order statistic d[floor(alpha * n)] of the sorted draws (see the module conventions)."""
    values = np.sort(np.asarray(draws, dtype=np.float64))
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Lower bounds need a non-empty one-dimensional array of finite draws.")
    index = int(math.floor(_check_alpha(alpha) * values.size + _ORDER_TOLERANCE))
    return float(values[index])


def one_sided_p_value(draws: Sequence[float] | np.ndarray) -> float:
    """Bootstrap p-value for H0: difference <= 0, the share of draws <= 0."""
    values = np.asarray(draws, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("p-values need a non-empty one-dimensional array of finite draws.")
    return float(np.count_nonzero(values <= 0.0) / values.size)


@dataclass(frozen=True)
class OneSidedTest:
    name: str
    estimate: float
    alpha: float
    lower_bound: float
    p_value: float
    n_valid_draws: int

    @property
    def passed(self) -> bool:
        return self.lower_bound > 0.0

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "estimate": self.estimate,
            "alpha": self.alpha,
            "lower_bound": self.lower_bound,
            "p_value": self.p_value,
            "n_valid_draws": self.n_valid_draws,
            "passed": self.passed,
        }


def one_sided_test(name: str, estimate: float, draws: Sequence[float] | np.ndarray, alpha: float) -> OneSidedTest:
    values = np.asarray(draws, dtype=np.float64)
    return OneSidedTest(
        name=name,
        estimate=float(estimate),
        alpha=_check_alpha(alpha),
        lower_bound=one_sided_lower_bound(values, alpha),
        p_value=one_sided_p_value(values),
        n_valid_draws=int(values.size),
    )


@dataclass(frozen=True, eq=False)
class BootstrapResult:
    metric: str
    seed: int
    n_draws: int
    n_skipped: int
    max_skip_fraction: float
    stratified: bool
    strata_sizes: dict[str, int]
    averaged_classes: dict[str, tuple[int, ...]]
    subset_names: tuple[str, ...]
    system_names: tuple[str, ...]
    system_point: np.ndarray
    system_draws: np.ndarray
    contrast_names: tuple[str, ...]
    contrast_point: np.ndarray
    contrast_draws: np.ndarray
    n_rows: int
    n_rows_outside_subsets: int
    dyadic_counts: bool
    numpy_version: str
    bit_generator: str

    @property
    def n_valid(self) -> int:
        return int(self.contrast_draws.shape[0])

    def estimate(self, contrast: str) -> float:
        return float(self.contrast_point[self.contrast_names.index(contrast)])

    def draws(self, contrast: str) -> np.ndarray:
        return self.contrast_draws[:, self.contrast_names.index(contrast)]

    def system_value(self, system: str, subset: str = DEFAULT_SUBSET) -> float:
        return float(self.system_point[self.subset_names.index(subset), self.system_names.index(system)])

    def lower_bound(self, contrast: str, alpha: float) -> float:
        return one_sided_lower_bound(self.draws(contrast), alpha)

    def p_value(self, contrast: str) -> float:
        return one_sided_p_value(self.draws(contrast))

    def test(self, contrast: str, alpha: float) -> OneSidedTest:
        return one_sided_test(contrast, self.estimate(contrast), self.draws(contrast), alpha)

    def draws_sha256(self) -> str:
        return hashlib.sha256(np.ascontiguousarray(self.contrast_draws, dtype="<f8").tobytes()).hexdigest()

    def to_record(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "seed": self.seed,
            "n_draws": self.n_draws,
            "n_skipped": self.n_skipped,
            "n_valid": self.n_valid,
            "max_skip_fraction": self.max_skip_fraction,
            "stratified": self.stratified,
            "strata_sizes": dict(self.strata_sizes),
            "averaged_ec_classes": {name: [index + 1 for index in classes] for name, classes in self.averaged_classes.items()},
            "system_point": {
                subset: {system: _finite_or_none(self.system_point[s, k]) for k, system in enumerate(self.system_names)}
                for s, subset in enumerate(self.subset_names)
            },
            "contrast_point": {name: _finite_or_none(self.contrast_point[k]) for k, name in enumerate(self.contrast_names)},
            "contrast_draws_sha256": self.draws_sha256(),
            "draw_batch_size": DRAW_BATCH_SIZE,
            "n_rows": self.n_rows,
            "n_rows_outside_subsets": self.n_rows_outside_subsets,
            "dyadic_counts": self.dyadic_counts,
            "numpy_version": self.numpy_version,
            "bit_generator": self.bit_generator,
        }


def _finite_or_none(value: float) -> Optional[float]:
    value = float(value)
    return value if math.isfinite(value) else None


def _resolve_systems(table: ClusterCountTable, systems: Optional[Mapping[str, Sequence[str]]]) -> dict[str, np.ndarray]:
    if systems is None:
        return {name: np.array([index]) for index, name in enumerate(table.arm_names)}
    resolved = {}
    for name, arms in systems.items():
        arms = list(arms)
        unknown = [arm for arm in arms if arm not in table.arm_names]
        if not arms or unknown or len(set(arms)) != len(arms):
            raise ValueError(f"System {name!r} needs distinct known arms; unknown: {unknown}.")
        resolved[str(name)] = np.array([table.arm_names.index(arm) for arm in arms])
    return resolved


def _contrast_matrix(
    contrasts: Sequence[Contrast], subset_names: Sequence[str], system_names: Sequence[str]
) -> np.ndarray:
    names = [contrast.name for contrast in contrasts]
    if not contrasts or len(set(names)) != len(names):
        raise ValueError("Contrasts must be non-empty with unique names.")
    matrix = np.zeros((len(contrasts), len(subset_names), len(system_names)))
    for row, contrast in enumerate(contrasts):
        if not contrast.terms:
            raise ValueError(f"Contrast {contrast.name!r} has no terms.")
        for coefficient, system, subset in contrast.terms:
            if system not in system_names or subset not in subset_names:
                raise ValueError(f"Contrast {contrast.name!r} names an unknown system or subset: {system!r}, {subset!r}.")
            matrix[row, subset_names.index(subset), system_names.index(system)] += float(coefficient)
    return matrix


def _system_values(
    confusion: np.ndarray,
    metric: str,
    averaged: Sequence[tuple[int, ...]],
    systems: Mapping[str, np.ndarray],
) -> np.ndarray:
    """``confusion[..., subset, arm, 7, 7]`` to system values ``[..., subset, system]`` (mean over arms)."""
    per_subset = [confusion_metric(confusion[..., s, :, :, :], metric, averaged[s]) for s in range(len(averaged))]
    arm_values = np.stack(per_subset, axis=-2)
    return np.stack([arm_values[..., arms].mean(axis=-1) for arms in systems.values()], axis=-1)


def paired_cluster_bootstrap(
    table: ClusterCountTable,
    contrasts: Sequence[Contrast],
    *,
    systems: Optional[Mapping[str, Sequence[str]]] = None,
    metric: str = METRIC_EC_CLASSES_1_6_BALANCED_ACCURACY,
    n_draws: int = DEFAULT_N_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    stratified: bool = True,
    max_skip_fraction: float = DEFAULT_MAX_SKIP_FRACTION,
) -> BootstrapResult:
    """Paired (stratified) cluster bootstrap of system metrics and their contrasts.

    A system's value is the mean of its arms' metrics (a seed mean); an ensemble is its own arm.
    All systems and contrasts share the same draws and the same skipped draws.
    """
    if metric not in BOOTSTRAP_METRICS:
        raise ValueError(f"Unknown bootstrap metric {metric!r}; expected one of {BOOTSTRAP_METRICS}.")
    if int(n_draws) < 1:
        raise ValueError("n_draws must be positive.")
    if not 0.0 <= float(max_skip_fraction) <= 1.0:
        raise ValueError("max_skip_fraction must lie in [0, 1].")
    n_draws = int(n_draws)
    resolved = _resolve_systems(table, systems)
    system_names = tuple(resolved)
    weights = _contrast_matrix(contrasts, table.subset_names, system_names)

    averaged = []
    for subset in table.subset_names:
        support = table.subset_support(subset)
        classes = tuple(index for index in PRIMARY_CLASS_INDICES if support[index] > 0)
        if not classes:
            raise ValueError(f"Subset {subset!r} holds no protein of EC classes 1-6.")
        averaged.append(classes)

    strata = assign_cluster_strata(table.class_support) if stratified else (UNSTRATIFIED_STRATUM,) * len(table.cluster_ids)
    members = stratum_members(strata)
    n_clusters = len(table.cluster_ids)
    flat_counts = table.counts.reshape(n_clusters, -1)
    tail_shape = table.counts.shape[1:]

    system_point = _system_values(table.counts.sum(axis=0), metric, averaged, resolved)
    contrast_point = np.einsum("csk,sk->c", weights, system_point)

    rng = np.random.default_rng(int(seed))
    system_chunks, skipped = [], 0
    for start in range(0, n_draws, DRAW_BATCH_SIZE):
        batch = min(DRAW_BATCH_SIZE, n_draws - start)
        multiplicity = stratified_multiplicities(rng, list(members.values()), n_clusters, batch)
        confusion = (multiplicity @ flat_counts).reshape((batch, *tail_shape))
        support = confusion[:, :, 0].sum(axis=-1)
        valid = np.ones(batch, dtype=bool)
        for s, classes in enumerate(averaged):
            valid &= (support[:, s, list(classes)] > 0).all(axis=1)
        skipped += int(batch - valid.sum())
        if valid.any():
            system_chunks.append(_system_values(confusion[valid], metric, averaged, resolved))

    if skipped > max_skip_fraction * n_draws + _ORDER_TOLERANCE or not system_chunks:
        raise BootstrapAbort(
            f"{skipped} of {n_draws} bootstrap draws lacked an EC class 1-6 present in the population "
            f"(limit {max_skip_fraction:.2%}); strata {({name: len(v) for name, v in members.items()})}."
        )
    system_draws = np.concatenate(system_chunks, axis=0)
    contrast_draws = np.einsum("dsk,csk->dc", system_draws, weights)
    return BootstrapResult(
        metric=metric,
        seed=int(seed),
        n_draws=n_draws,
        n_skipped=skipped,
        max_skip_fraction=float(max_skip_fraction),
        stratified=bool(stratified),
        strata_sizes={name: int(indices.size) for name, indices in members.items()},
        averaged_classes=dict(zip(table.subset_names, averaged)),
        subset_names=table.subset_names,
        system_names=system_names,
        system_point=system_point,
        system_draws=system_draws,
        contrast_names=tuple(contrast.name for contrast in contrasts),
        contrast_point=contrast_point,
        contrast_draws=contrast_draws,
        n_rows=table.n_rows,
        n_rows_outside_subsets=table.n_rows_outside_subsets,
        dyadic_counts=table.dyadic_counts,
        numpy_version=np.__version__,
        bit_generator=type(rng.bit_generator).__name__,
    )


@dataclass(frozen=True)
class IntersectionUnionResult:
    """Passes only if every component's one-sided lower bound is above 0; p is the largest component p."""

    name: str
    alpha: float
    components: tuple[OneSidedTest, ...]

    @property
    def p_value(self) -> float:
        return max(component.p_value for component in self.components)

    @property
    def passed(self) -> bool:
        return all(component.passed for component in self.components)

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "alpha": self.alpha,
            "p_value": self.p_value,
            "passed": self.passed,
            "components": [component.to_record() for component in self.components],
        }


def intersection_union_test(
    result: BootstrapResult, components: Sequence[str], alpha: float, *, name: str
) -> IntersectionUnionResult:
    """Intersection-union test over contrasts of one bootstrap run, so all components share draws."""
    if not components or len(set(components)) != len(components):
        raise ValueError("An intersection-union test needs distinct component contrasts.")
    return IntersectionUnionResult(name, _check_alpha(alpha), tuple(result.test(item, alpha) for item in components))


def intersection_union_p_value(result: BootstrapResult, components: Sequence[str]) -> float:
    if not components:
        raise ValueError("An intersection-union test needs at least one component.")
    return max(result.p_value(item) for item in components)


@dataclass(frozen=True)
class HolmStep:
    name: str
    rank: int
    p_value: float
    alpha: float
    adjusted_p_value: float
    tested: bool
    rejected: bool

    def to_record(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _holm_order(p_values: Mapping[str, float]) -> list[tuple[str, float]]:
    # Stable sort: equal p-values keep the caller's order.
    items = [(str(name), float(value)) for name, value in p_values.items()]
    if not items:
        raise ValueError("Holm needs at least one hypothesis.")
    if any(not 0.0 <= value <= 1.0 for _, value in items):
        raise ValueError("p-values must lie in [0, 1].")
    return sorted(items, key=lambda item: item[1])


def holm_step_down(p_values: Mapping[str, float], family_alpha: float = 0.05) -> tuple[HolmStep, ...]:
    """Holm step-down: the j-th smallest p (0-based) is tested at alpha / (m - j) while all earlier ones reject."""
    family_alpha = _check_alpha(family_alpha)
    ordered = _holm_order(p_values)
    m = len(ordered)
    steps, still_testing, running = [], True, 0.0
    for rank, (name, value) in enumerate(ordered):
        alpha = family_alpha / (m - rank)
        running = max(running, min(1.0, (m - rank) * value))
        rejected = still_testing and value <= alpha + P_VALUE_TOLERANCE
        steps.append(HolmStep(name, rank + 1, value, alpha, running, still_testing, rejected))
        still_testing = rejected
    return tuple(steps)


@dataclass(frozen=True)
class HolmContrastStep:
    name: str
    rank: int
    alpha: float
    p_value: float
    tested: bool
    passed: bool
    test: Optional[IntersectionUnionResult]

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rank": self.rank,
            "alpha": self.alpha,
            "p_value": self.p_value,
            "tested": self.tested,
            "passed": self.passed,
            "test": None if self.test is None else self.test.to_record(),
        }


def holm_intersection_union(
    contrasts: Mapping[str, tuple[BootstrapResult, Sequence[str]]],
    family_alpha: float = 0.05,
) -> tuple[HolmContrastStep, ...]:
    """Holm across intersection-union contrasts, each from its own bootstrap run.

    Contrasts are ordered by their intersection-union p-value (equal values keep the given
    order). The j-th contrast is tested at ``family_alpha / (m - j)`` with lower bounds at that
    level, and only while every earlier contrast passed.
    """
    family_alpha = _check_alpha(family_alpha)
    p_values = {name: intersection_union_p_value(result, components) for name, (result, components) in contrasts.items()}
    ordered = _holm_order(p_values)
    m = len(ordered)
    steps, still_testing = [], True
    for rank, (name, value) in enumerate(ordered):
        alpha = family_alpha / (m - rank)
        test = None
        if still_testing:
            result, components = contrasts[name]
            test = intersection_union_test(result, components, alpha, name=name)
        passed = bool(test is not None and test.passed)
        steps.append(HolmContrastStep(name, rank + 1, alpha, value, still_testing, passed, test))
        still_testing = passed
    return tuple(steps)


CO_PRIMARY_FAMILY_ALPHA = 0.05


@dataclass(frozen=True)
class CoPrimaryResult:
    """Both co-primary steps. ``steps`` is in test order (smaller p first), not argument order,
    so claims are read by name with ``step``; the result cannot be unpacked by position."""

    steps: tuple[HolmContrastStep, HolmContrastStep]

    def step(self, name: str) -> HolmContrastStep:
        for item in self.steps:
            if item.name == name:
                return item
        raise KeyError(f"No co-primary contrast named {name!r}; have {[item.name for item in self.steps]}.")

    def passed(self, name: str) -> bool:
        return self.step(name).passed

    def to_record(self) -> dict[str, Any]:
        return {"family_alpha": CO_PRIMARY_FAMILY_ALPHA, "steps": [item.to_record() for item in self.steps]}


def co_primary_holm(
    contrast_a: tuple[str, BootstrapResult, Sequence[str]],
    contrast_b: tuple[str, BootstrapResult, Sequence[str]],
) -> CoPrimaryResult:
    """The two co-primary contrasts: the smaller p at 0.025, the other at 0.05 only if the first passes.

    This is Holm with two hypotheses at a family level of 0.05, each contrast an
    intersection-union test over its components. Argument order carries no meaning.
    """
    if contrast_a[0] == contrast_b[0]:
        raise ValueError("The co-primary contrasts need distinct names.")
    steps = holm_intersection_union(
        {contrast_a[0]: (contrast_a[1], contrast_a[2]), contrast_b[0]: (contrast_b[1], contrast_b[2])},
        family_alpha=CO_PRIMARY_FAMILY_ALPHA,
    )
    return CoPrimaryResult((steps[0], steps[1]))
