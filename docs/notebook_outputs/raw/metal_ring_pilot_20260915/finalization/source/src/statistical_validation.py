from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Iterable


@dataclass(frozen=True)
class PairedBootstrapCI:
    n_pairs: int
    mean_difference: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_bootstrap: int
    seed: int
    raw_improvement_threshold: float
    passes: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


def _as_float_list(values: Iterable[float | int]) -> list[float]:
    parsed = [float(value) for value in values]
    if not parsed:
        raise ValueError("paired bootstrap requires at least one paired value.")
    return parsed


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a quantile of an empty list.")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    return sorted_values[lower_index] * (1.0 - fraction) + sorted_values[upper_index] * fraction


def paired_bootstrap_ci(
    candidate_a_values: Iterable[float | int],
    candidate_b_values: Iterable[float | int],
    *,
    n_bootstrap: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 0,
    raw_improvement_threshold: float = 0.0,
) -> PairedBootstrapCI:
    """Bootstrap paired differences A - B and return a confidence interval."""
    a_values = _as_float_list(candidate_a_values)
    b_values = _as_float_list(candidate_b_values)
    if len(a_values) != len(b_values):
        raise ValueError(
            "paired bootstrap requires the same number of A and B values; "
            f"got {len(a_values)} and {len(b_values)}."
        )
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be positive, got {n_bootstrap}.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}.")

    differences = [a - b for a, b in zip(a_values, b_values)]
    n_pairs = len(differences)
    mean_difference = sum(differences) / float(n_pairs)
    rng = random.Random(seed)
    boot_means = []
    for _ in range(int(n_bootstrap)):
        total = sum(differences[rng.randrange(n_pairs)] for _ in range(n_pairs))
        boot_means.append(total / float(n_pairs))
    boot_means.sort()

    alpha = 1.0 - float(confidence_level)
    ci_lower = _quantile(boot_means, alpha / 2.0)
    ci_upper = _quantile(boot_means, 1.0 - alpha / 2.0)
    passes = (
        mean_difference > 0.0
        and ci_lower > 0.0
        and mean_difference >= float(raw_improvement_threshold)
    )
    return PairedBootstrapCI(
        n_pairs=n_pairs,
        mean_difference=mean_difference,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=float(confidence_level),
        n_bootstrap=int(n_bootstrap),
        seed=int(seed),
        raw_improvement_threshold=float(raw_improvement_threshold),
        passes=passes,
    )
