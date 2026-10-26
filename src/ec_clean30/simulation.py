"""Null-adoption sweeps of the promotion rule for the EC class versus CLEAN30 campaign (numpy only).

The runner's ``simulate`` action (P2-9) calls ``run_simulate_action``, which calls ``run_sweep``. A
sweep file names the promotion rules, the looks (1 for a single lever, 2 for the two epoch looks of
L7, 4 for the non-zero taus of L9), the noise points, the power points (a true gain in BA points)
and the outer fold whose population each point uses; ``simulate_null_adoption`` runs every point.

Populations come from each fold's certified inner membership, read with the identity loaders from
the data overlay's copy (``--calibrate-from K=<overlay>/identity/fold<K>``). A folder under the
campaign's own ``identity/`` folder is refused, because that folder holds test-side content once the
identity audit has run; a real run also needs ``overlay_manifest.json`` two levels above the folder.
The calibrated parameters are the eligible protein count, the class proportions, the mean
component size and the class purity (``selection.noise_model_from_membership``); recall, difficulty
and fit noise come from the sweep file. A noise point may give the seed-to-seed spread of pooled BA
in points (``seed_spread_points``) instead of ``fit_shift_sd``; it is converted on the fold's
population. A power point's true gain is converted into ``challenger_shift`` the same way.

Outputs, written once into ``--out``: ``sweep.json`` (a byte copy of the sweep file),
``records/<point>.json`` (calibration hashes and statistics, noise model, ``PromotionConfig``,
operating characteristics, timing and provenance; the deterministic part is hashed as
``result_sha256``) and, once every point has a record, ``summary.tsv`` and ``sweep_receipt.json``.
Every call names one calibration folder for each fold the sweep uses (and no other) and resolves
every point before it writes anything, so a point that cannot be resolved is refused at once. A
later call with the same sweep file, code and calibrations continues where an earlier one stopped
(``--points`` times one point first); anything else is refused. A run stops before its next point
when the forecast for the whole sweep exceeds the CPU-hour limit. Records hold per-fold class
proportions and stay private, so ``--out`` may not lie under the campaign's identity folder, inside
a calibration folder or inside a data overlay; the printed payload, the summary and the receipt
carry no class proportions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ec_clean30 import identity
from ec_clean30.campaign import CAMPAIGN_ID, N_OUTER_FOLDS
from ec_clean30.logit_adjustment import TAUS
from ec_clean30.provenance import (
    names_test_split, provenance_record, real_action_git_state, refuse_test_paths, sha256_bytes, sha256_file,
    utc_now, write_new_json,
)
from ec_clean30.selection import (
    DEFAULT_SIMULATION_SEED, EPOCH_LOOKS, FLOAT_TOLERANCE, POINTS, SCREENING_SEEDS,
    NullNoiseModel, PromotionConfig, _simulated_population, membership_noise_statistics, noise_model_from_membership,
    sha256_canonical, simulate_null_adoption, to_jsonable,
)
from training.ec_class_metrics import PRIMARY_CLASS_INDICES


SWEEP_FORMAT = "deepmzyme_ec_clean30_simulation_sweep_v1"
RECORD_FORMAT = "deepmzyme_ec_clean30_simulation_record_v1"
RECEIPT_FORMAT = "deepmzyme_ec_clean30_simulation_sweep_receipt_v1"

# Looks per decision: a single lever, L7 (the better of 12 and 20 epochs) and L9 (the best non-zero tau).
ALLOWED_LOOKS: tuple[int, ...] = (1, len(EPOCH_LOOKS), sum(1 for tau in TAUS if tau != 0.0))
KIND_NULL = "null"
KIND_POWER = "power"
DEFAULT_N_REPLICATES = 200
DEFAULT_MAX_FORECAST_CPU_HOURS = 8.0

# Seed-spread conversion: common random numbers over this many populations of two fits each.
SEED_SPREAD_POPULATIONS = 1000
SEED_SPREAD_STREAM = 1
FIT_SHIFT_SD_UPPER = 3.0
FIT_SHIFT_SD_TOLERANCE = 1e-4
# True-gain conversion: Gauss-Hermite expectation of the seed-mean BA, solved by bisection.
QUADRATURE_NODES = 64
MAX_CHALLENGER_SHIFT = 20.0

STATUS_COMPLETE = "complete"
STATUS_PARTIAL = "partial"
STATUS_STOPPED = "stopped_forecast_over_limit"
EXIT_STOPPED = 4

SWEEP_FILE = "sweep.json"
RECORDS_FOLDER = "records"
SUMMARY_FILE = "summary.tsv"
RECEIPT_FILE = "sweep_receipt.json"
IDENTITY_FOLDER = "identity"
OVERLAY_MANIFEST = "overlay_manifest.json"

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,47}$")
MAX_POINT_ID_LENGTH = 200
_SWEEP_REQUIRED = ("format", "sweep_id", "rules", "noise_points", "groups")
_SWEEP_OPTIONAL = ("description", "n_replicates", "simulation_seed", "noise_base")
_GROUP_REQUIRED = ("group", "folds", "rules", "looks", "noise_points")
_GROUP_OPTIONAL = ("true_gain_points", "n_replicates")
NOISE_SD_KEYS: tuple[str, ...] = ("component_sd", "protein_sd", "fit_shift_sd", "fit_noise_sd")
NOISE_KEYS: tuple[str, ...] = (*NOISE_SD_KEYS, "class_recall", "class_recall_offset", "seed_spread_points")
EXCLUSIVE_NOISE_KEYS: tuple[tuple[str, str], ...] = (("fit_shift_sd", "seed_spread_points"),
                                                     ("class_recall", "class_recall_offset"))
CALIBRATED_NOISE_KEYS: tuple[str, ...] = ("n_proteins", "class_proportions", "mean_component_size",
                                          "component_class_purity")
_RULE_DEFAULTS = {item.name: item.default for item in fields(PromotionConfig)}
# Ranges of the float fields that PromotionConfig does not check itself (alpha it checks).
_RULE_RANGES: dict[str, tuple[Optional[float], Optional[float]]] = {
    "min_gain_points": (0.0, None),
    "max_skip_fraction": (0.0, 1.0),
    "guard_recall_drop_points": (0.0, None),
    "guard_net_proteins": (0.0, None),
}

SUMMARY_COLUMNS: tuple[str, ...] = (
    "point_id", "group", "kind", "fold", "n_proteins", "rule", "alpha", "min_gain_points", "stratified", "n_draws",
    "looks", "noise_point", "fit_shift_sd", "fit_noise_sd", "component_sd", "protein_sd", "class_recall_offset",
    "seed_spread_target_points", "implied_seed_spread_points", "true_gain_points", "challenger_shift",
    "expected_gain_points", "n_replicates", "n_adopted", "adoption_rate", "adoption_rate_standard_error",
    "target_rate", "meets_target", "gain_pass_rate", "recall_guard_veto_rate", "zero_recall_veto_rate",
    "n_bootstrap_evaluated", "n_lower_bound_passed", "n_bootstrap_aborted", "gain_points_mean", "gain_points_sd",
    "result_sha256",
)


class SimulationError(ValueError):
    """A sweep file, calibration folder or output folder cannot be used."""


# ---------------------------------------------------------------------------
# Sweep file


@dataclass(frozen=True)
class SweepPoint:
    point_id: str
    group: str
    kind: str
    fold: int
    rule: str
    looks: int
    noise_point: str
    true_gain_points: Optional[float]
    n_replicates: int

    def to_record(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass(frozen=True)
class SweepSpec:
    sweep_id: str
    simulation_seed: int
    rules: Mapping[str, Mapping[str, Any]]
    noise: Mapping[str, Mapping[str, Any]]
    points: tuple[SweepPoint, ...]
    spec_sha256: str

    def config(self, rule: str) -> PromotionConfig:
        return PromotionConfig(**self.rules[rule])

    def point(self, point_id: str) -> SweepPoint:
        for point in self.points:
            if point.point_id == point_id:
                return point
        raise SimulationError(f"The sweep has no point {point_id!r}.")


def _refuse(message: str) -> SimulationError:
    return SimulationError(f"Sweep file refused: {message}")


def _keys(mapping: Any, what: str, required: Sequence[str], optional: Sequence[str] = ()) -> Mapping[str, Any]:
    if not isinstance(mapping, Mapping):
        raise _refuse(f"{what} must be a JSON object.")
    missing = [key for key in required if key not in mapping]
    unknown = sorted(set(mapping) - set(required) - set(optional))
    if missing or unknown:
        raise _refuse(f"{what}: missing keys {missing}, unknown keys {unknown}.")
    return mapping


def _name(value: Any, what: str) -> str:
    if not isinstance(value, str) or not _NAME_RE.match(value) or "__" in value or value.endswith("_"):
        raise _refuse(f"{what} {value!r} must be 1-48 characters of letters, digits and . _ + - (no '__').")
    if names_test_split(value):
        raise _refuse(f"{what} {value!r} names a test split.")
    return value


def _number(value: Any, what: str, *, minimum: Optional[float] = None, maximum: Optional[float] = None,
            positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise _refuse(f"{what} must be a finite number, got {value!r}.")
    number = float(value)
    if positive and number <= 0.0:
        raise _refuse(f"{what} must be positive, got {value!r}.")
    if minimum is not None and number < minimum:
        raise _refuse(f"{what} must be at least {minimum:g}, got {value!r}.")
    if maximum is not None and number > maximum:
        raise _refuse(f"{what} must be at most {maximum:g}, got {value!r}.")
    return number


def _integer(value: Any, what: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _refuse(f"{what} must be an integer of at least {minimum}, got {value!r}.")
    return int(value)


def _distinct_list(value: Any, what: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise _refuse(f"{what} must be a non-empty list.")
    if len({json.dumps(item, sort_keys=True) for item in value}) != len(value):
        raise _refuse(f"{what} lists a value more than once.")
    return value


def _rule_overrides(name: str, overrides: Any) -> dict[str, Any]:
    if not isinstance(overrides, Mapping):
        raise _refuse(f"rule {name!r} must be a JSON object of PromotionConfig fields.")
    unknown = sorted(set(overrides) - set(_RULE_DEFAULTS))
    if unknown:
        raise _refuse(f"rule {name!r}: unknown PromotionConfig fields {unknown}.")
    values: dict[str, Any] = {}
    for key, value in overrides.items():
        default = _RULE_DEFAULTS[key]
        what = f"rule {name!r} {key}"
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise _refuse(f"{what} must be true or false.")
            values[key] = value
        elif isinstance(default, int):
            values[key] = _integer(value, what, minimum=1 if key == "n_draws" else 0)
        elif isinstance(default, float):
            minimum, maximum = _RULE_RANGES.get(key, (None, None))
            values[key] = _number(value, what, minimum=minimum, maximum=maximum)
        elif not isinstance(value, str):
            raise _refuse(f"{what} must be a string.")
        else:
            values[key] = value
    try:
        PromotionConfig(**values)
    except (TypeError, ValueError) as exc:
        raise _refuse(f"rule {name!r}: {exc}") from None
    return values


def _noise_overrides(name: str, overrides: Any) -> dict[str, Any]:
    if not isinstance(overrides, Mapping):
        raise _refuse(f"noise point {name!r} must be a JSON object.")
    calibrated = sorted(set(overrides) & set(CALIBRATED_NOISE_KEYS))
    if calibrated:
        raise _refuse(f"noise point {name!r} sets {calibrated}; these come from the fold's certified membership.")
    unknown = sorted(set(overrides) - set(NOISE_KEYS))
    if unknown:
        raise _refuse(f"noise point {name!r}: unknown keys {unknown} (allowed: {list(NOISE_KEYS)}).")
    values: dict[str, Any] = {}
    for key, value in overrides.items():
        what = f"noise point {name!r} {key}"
        if key in NOISE_SD_KEYS:
            values[key] = _number(value, what, minimum=0.0)
        elif key == "class_recall":
            if not isinstance(value, list) or len(value) != len(NullNoiseModel().class_recall):
                raise _refuse(f"{what} must list one recall per EC class 1-7.")
            values[key] = [_number(item, what) for item in value]
        elif key == "class_recall_offset":
            values[key] = _number(value, what)
        else:
            values[key] = _number(value, what, positive=True)
    for first, second in EXCLUSIVE_NOISE_KEYS:
        if first in values and second in values:
            raise _refuse(f"noise point {name!r} sets both {first} and {second}.")
    return values


def _merged_noise(base: Mapping[str, Any], point: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for first, second in EXCLUSIVE_NOISE_KEYS:
        if first in point or second in point:
            merged.pop(first, None)
            merged.pop(second, None)
    merged.update(point)
    return merged


def _model_values(base: NullNoiseModel, overrides: Mapping[str, Any]) -> dict[str, Any]:
    """``NullNoiseModel`` fields set by the overrides, except the seed-spread target."""
    values: dict[str, Any] = {key: float(overrides[key]) for key in NOISE_SD_KEYS if key in overrides}
    if "class_recall" in overrides:
        values["class_recall"] = tuple(float(value) for value in overrides["class_recall"])
    elif "class_recall_offset" in overrides:
        values["class_recall"] = tuple(value + float(overrides["class_recall_offset"]) for value in base.class_recall)
    return values


def _point_id(group: str, fold: int, rule: str, looks: int, noise: str, gain: Optional[float]) -> str:
    parts = [group, f"fold{fold}", rule, f"looks{looks}", noise] + ([f"gain{gain:g}"] if gain is not None else [])
    return "__".join(parts)


def parse_sweep(document: Any) -> SweepSpec:
    """Validate a sweep document and expand its groups into points (groups in order; within a group
    folds, rules, looks, noise points, then true gains)."""
    document = _keys(document, "the sweep file", _SWEEP_REQUIRED, _SWEEP_OPTIONAL)
    if document["format"] != SWEEP_FORMAT:
        raise _refuse(f"format must be {SWEEP_FORMAT!r}, got {document['format']!r}.")
    sweep_id = _name(document["sweep_id"], "sweep_id")
    if "description" in document and not isinstance(document["description"], str):
        raise _refuse("description must be a string.")
    n_replicates = _integer(document.get("n_replicates", DEFAULT_N_REPLICATES), "n_replicates", minimum=1)
    seed = _integer(document.get("simulation_seed", DEFAULT_SIMULATION_SEED), "simulation_seed", minimum=0)

    rules_in = document["rules"]
    if not isinstance(rules_in, Mapping) or not rules_in:
        raise _refuse("rules must be a non-empty JSON object of named rules.")
    rules = {_name(name, "rule name"): _rule_overrides(name, overrides) for name, overrides in rules_in.items()}
    base = _noise_overrides("noise_base", document.get("noise_base", {}))
    points_in = document["noise_points"]
    if not isinstance(points_in, Mapping) or not points_in:
        raise _refuse("noise_points must be a non-empty JSON object of named noise points.")
    noise: dict[str, dict[str, Any]] = {}
    for name, overrides in points_in.items():
        merged = _merged_noise(base, _noise_overrides(_name(name, "noise point name"), overrides))
        try:
            replace(NullNoiseModel(), **_model_values(NullNoiseModel(), merged))
        except ValueError as exc:
            raise _refuse(f"noise point {name!r}: {exc}") from None
        noise[name] = merged

    groups = document["groups"]
    if not isinstance(groups, list) or not groups:
        raise _refuse("groups must be a non-empty list.")
    points: list[SweepPoint] = []
    seen: set[str] = set()
    for position, group_in in enumerate(groups):
        group_in = _keys(group_in, f"group {position}", _GROUP_REQUIRED, _GROUP_OPTIONAL)
        group = _name(group_in["group"], "group name")
        folds = [_integer(value, f"group {group!r} fold", minimum=0) for value in
                 _distinct_list(group_in["folds"], f"group {group!r} folds")]
        if any(fold >= N_OUTER_FOLDS for fold in folds):
            raise _refuse(f"group {group!r}: folds must lie in 0..{N_OUTER_FOLDS - 1}, got {folds}.")
        group_rules = _distinct_list(group_in["rules"], f"group {group!r} rules")
        group_noise = _distinct_list(group_in["noise_points"], f"group {group!r} noise_points")
        for names, known, what in ((group_rules, rules, "rule"), (group_noise, noise, "noise point")):
            unnamed = [value for value in names if not isinstance(value, str)]
            if unnamed:
                raise _refuse(f"group {group!r} {what}s must be names (strings), got {unnamed}.")
            missing = [value for value in names if value not in known]
            if missing:
                raise _refuse(f"group {group!r} names unknown {what}s {missing}.")
        looks = [_integer(value, f"group {group!r} looks", minimum=1) for value in
                 _distinct_list(group_in["looks"], f"group {group!r} looks")]
        if any(value not in ALLOWED_LOOKS for value in looks):
            raise _refuse(f"group {group!r}: looks must be among {list(ALLOWED_LOOKS)}, got {looks}.")
        gains: list[Optional[float]] = [None]
        if "true_gain_points" in group_in:
            gains = [_number(value, f"group {group!r} true_gain_points", positive=True)
                     for value in _distinct_list(group_in["true_gain_points"], f"group {group!r} true_gain_points")]
        replicates = _integer(group_in.get("n_replicates", n_replicates), f"group {group!r} n_replicates", minimum=1)
        for fold in folds:
            for rule in group_rules:
                for look in looks:
                    for noise_point in group_noise:
                        for gain in gains:
                            point_id = _point_id(group, fold, rule, look, noise_point, gain)
                            if len(point_id) > MAX_POINT_ID_LENGTH:
                                raise _refuse(f"point {point_id!r} is longer than {MAX_POINT_ID_LENGTH} characters.")
                            if point_id in seen:
                                raise _refuse(f"point {point_id!r} is defined more than once.")
                            seen.add(point_id)
                            points.append(SweepPoint(point_id, group, KIND_NULL if gain is None else KIND_POWER, fold,
                                                     rule, look, noise_point, gain, replicates))
    spec_sha256 = sha256_canonical({"sweep": document})
    return SweepSpec(sweep_id, seed, rules, noise, tuple(points), spec_sha256)


def read_sweep(path: Path) -> tuple[SweepSpec, bytes]:
    data = Path(path).read_bytes()
    try:
        document = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _refuse(f"{path} is not UTF-8 JSON ({exc}).") from None
    return parse_sweep(document), data


# ---------------------------------------------------------------------------
# Conversions on a fold's population


def _expit(values: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * values))


def expected_ba_points(model: NullNoiseModel, shift: float = 0.0) -> float:
    """Expected seed-mean pooled BA over EC classes 1-6 (points) of fits with logit ``shift``.

    Under the model the logit of a correct prediction is ``logit(class_recall[c]) + Z + shift`` with
    ``Z`` normal of variance ``component_sd**2 + protein_sd**2 + fit_shift_sd**2 + fit_noise_sd**2``;
    recall only counts correct predictions, so BA is the mean over present classes of
    ``E[sigmoid(...)]`` (Gauss-Hermite quadrature).
    """
    proportions = np.asarray(model.class_proportions, dtype=np.float64)
    classes = [index for index in PRIMARY_CLASS_INDICES if proportions[index] > 0]
    if not classes:
        raise SimulationError("The noise model holds no protein of EC classes 1-6.")
    nodes, weights = np.polynomial.hermite_e.hermegauss(QUADRATURE_NODES)
    weights = weights / math.sqrt(2.0 * math.pi)
    scale = math.sqrt(model.component_sd ** 2 + model.protein_sd ** 2 + model.fit_shift_sd ** 2
                      + model.fit_noise_sd ** 2)
    recall = np.asarray(model.class_recall, dtype=np.float64)[classes]
    logits = np.log(recall / (1.0 - recall))
    expected = _expit(logits[:, None] + scale * nodes[None, :] + float(shift)) @ weights
    return POINTS * float(np.mean(expected))


def challenger_shift_for_gain(model: NullNoiseModel, gain_points: float) -> float:
    """The ``challenger_shift`` whose expected seed-mean BA gain over the incumbent is ``gain_points``."""
    target = float(gain_points)
    if not math.isfinite(target) or target <= 0.0:
        raise SimulationError(f"A true gain must be positive, got {gain_points!r}.")
    base = expected_ba_points(model)
    reachable = expected_ba_points(model, MAX_CHALLENGER_SHIFT) - base
    if reachable < target:
        raise SimulationError(f"A true gain of {target:g} BA points exceeds the {reachable:.3f} points this "
                              f"population can gain.")
    low, high = 0.0, MAX_CHALLENGER_SHIFT
    for _ in range(100):
        middle = 0.5 * (low + high)
        if expected_ba_points(model, middle) - base < target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


@dataclass(frozen=True, eq=False)
class _SpreadDraws:
    true: np.ndarray
    difficulty: np.ndarray
    fit_shift: np.ndarray
    fit_noise: np.ndarray
    uniform: np.ndarray
    seed: tuple[int, int]


def _spread_draws(model: NullNoiseModel, n_populations: int, seed: int) -> _SpreadDraws:
    """Populations as ``simulate_null_adoption`` draws them, each with two incumbent fits' standard draws."""
    if int(n_populations) < 2:
        raise SimulationError("The seed-spread conversion needs at least two populations.")
    stream = (int(seed), SEED_SPREAD_STREAM)
    rng = np.random.default_rng(list(stream))
    populations = [_simulated_population(rng, model) for _ in range(int(n_populations))]
    shape = (int(n_populations), 2, model.n_proteins)
    return _SpreadDraws(
        true=np.stack([population[0] for population in populations]),
        difficulty=np.stack([population[2] for population in populations]),
        fit_shift=rng.standard_normal(shape[:2]),
        fit_noise=rng.standard_normal(shape),
        uniform=rng.random(shape),
        seed=stream,
    )


def _ba_points(true: np.ndarray, correct: np.ndarray) -> np.ndarray:
    """``[population, fit]`` BA over EC classes 1-6 in points from correctness alone."""
    recalls = np.full((*correct.shape[:2], len(PRIMARY_CLASS_INDICES)), np.nan)
    for position, index in enumerate(PRIMARY_CLASS_INDICES):
        mask = true == index
        support = mask.sum(axis=1)
        present = support > 0
        hits = (correct & mask[:, None, :]).sum(axis=2)
        recalls[present, :, position] = hits[present] / support[present, None]
    if np.any(np.all(np.isnan(recalls), axis=2)):
        raise SimulationError("A simulated population holds no protein of EC classes 1-6.")
    return POINTS * np.nanmean(recalls, axis=2)


def _spread(draws: _SpreadDraws, fit_shift_sd: float, fit_noise_sd: float) -> float:
    eta = (draws.difficulty[:, None, :] + fit_shift_sd * draws.fit_shift[:, :, None]
           + fit_noise_sd * draws.fit_noise)
    ba = _ba_points(draws.true, draws.uniform < _expit(eta))
    return float(np.sqrt(np.mean((ba[:, 0] - ba[:, 1]) ** 2) / 2.0))


def _population_key(model: NullNoiseModel) -> str:
    """The noise-model fields that shape a simulated population (not the fit noise)."""
    record = model.to_record()
    return sha256_canonical({key: value for key, value in record.items()
                             if key not in ("fit_shift_sd", "fit_noise_sd", "challenger_shift")})


def seed_spread_points(model: NullNoiseModel, *, n_populations: int = SEED_SPREAD_POPULATIONS,
                       seed: int = DEFAULT_SIMULATION_SEED, draws: Optional[_SpreadDraws] = None) -> float:
    """Model-implied standard deviation (points) of pooled BA between two fits on one population."""
    draws = draws if draws is not None else _spread_draws(model, n_populations, seed)
    return _spread(draws, model.fit_shift_sd, model.fit_noise_sd)


def fit_shift_sd_for_seed_spread(
    model: NullNoiseModel,
    target_points: float,
    *,
    n_populations: int = SEED_SPREAD_POPULATIONS,
    seed: int = DEFAULT_SIMULATION_SEED,
    upper: float = FIT_SHIFT_SD_UPPER,
    tolerance: float = FIT_SHIFT_SD_TOLERANCE,
    draws: Optional[_SpreadDraws] = None,
) -> dict[str, Any]:
    """The ``fit_shift_sd`` at which the model's seed spread equals ``target_points``.

    Bisection over common random numbers, so the spread is evaluated on the same draws at every
    step. Refused when the target lies below the spread that per-protein noise gives on its own
    (``fit_shift_sd`` 0) or above the spread at ``upper``.
    """
    target = float(target_points)
    if not math.isfinite(target) or target <= 0.0:
        raise SimulationError(f"A seed spread must be positive, got {target_points!r}.")
    draws = draws if draws is not None else _spread_draws(model, n_populations, seed)
    floor = _spread(draws, 0.0, model.fit_noise_sd)
    if target <= floor:
        raise SimulationError(f"A seed spread of {target:g} BA points lies at or below the {floor:.3f} points that "
                              f"per-protein noise gives alone (fit_noise_sd {model.fit_noise_sd:g}).")
    ceiling = _spread(draws, float(upper), model.fit_noise_sd)
    if target > ceiling:
        raise SimulationError(f"A seed spread of {target:g} BA points exceeds the {ceiling:.3f} points at "
                              f"fit_shift_sd {upper:g}.")
    low, high = 0.0, float(upper)
    while high - low > tolerance:
        middle = 0.5 * (low + high)
        if _spread(draws, middle, model.fit_noise_sd) < target:
            low = middle
        else:
            high = middle
    return {
        "target_points": target,
        "fit_shift_sd": high,
        "achieved_points": _spread(draws, high, model.fit_noise_sd),
        "floor_points": floor,
        "n_populations": int(draws.true.shape[0]),
        "seed": list(draws.seed),
        "tolerance": float(tolerance),
    }


# ---------------------------------------------------------------------------
# Calibration from certified inner memberships


@dataclass(frozen=True, eq=False)
class FoldCalibration:
    """One fold's calibrated population; ``statistics`` holds class proportions (private records only)."""

    fold: int
    source: str
    inputs: Mapping[str, Any]
    statistics: Mapping[str, Any]
    noise_model: NullNoiseModel

    @classmethod
    def from_membership(cls, fold: int, rows: Sequence[Mapping[str, Any]], components: Mapping[str, str], *,
                        source: str, inputs: Mapping[str, Any]) -> "FoldCalibration":
        try:
            statistics = membership_noise_statistics(rows, components)
            model = noise_model_from_membership(rows, components)
        except ValueError as exc:
            raise SimulationError(f"Fold {fold} calibration: {exc}") from exc
        return cls(int(fold), str(source), dict(inputs), statistics, model)

    def result_record(self) -> dict[str, Any]:
        return to_jsonable({"fold": self.fold, "inputs": self.inputs, "statistics": self.statistics})

    def public_record(self) -> dict[str, Any]:
        """Printable: no class proportions."""
        return {"fold": self.fold, "n_proteins": self.statistics["n_proteins"],
                "membership_sha256": self.inputs.get("membership_sha256"),
                "certificate_sha256": self.inputs.get("certificate_sha256")}


def parse_calibrate_from(values: Sequence[str] | Mapping[int, Any]) -> dict[int, Path]:
    """``K=PATH`` entries (one per outer fold) as {fold: folder}."""
    if isinstance(values, Mapping):
        items = [(fold, path) for fold, path in values.items()]
    else:
        items = []
        for value in values:
            fold_text, separator, path = str(value).partition("=")
            if not separator or not path or not re.fullmatch(r"[0-9]+", fold_text.strip()):
                raise SimulationError(f"--calibrate-from takes K=PATH with an outer fold K, got {value!r}.")
            items.append((int(fold_text), path))
    mapping: dict[int, Path] = {}
    for fold, path in items:
        if isinstance(fold, bool) or not isinstance(fold, int) or not 0 <= fold < N_OUTER_FOLDS:
            raise SimulationError(f"--calibrate-from fold must lie in 0..{N_OUTER_FOLDS - 1}, got {fold!r}.")
        if fold in mapping:
            raise SimulationError(f"--calibrate-from names fold {fold} more than once.")
        mapping[fold] = Path(path)
    return mapping


def is_campaign_identity_path(path: Path) -> bool:
    """Whether ``path`` (as given or resolved) lies under an ``identity`` folder of the campaign root."""
    for candidate in (Path(os.path.abspath(path)), Path(path).resolve()):
        parts = candidate.parts
        if any(part == IDENTITY_FOLDER and parts[position - 1].startswith(CAMPAIGN_ID)
               for position, part in enumerate(parts) if position > 0):
            return True
    return False


def _calibration_file_names(fold: int) -> tuple[str, ...]:
    return (identity.INNER_CERTIFICATE, identity.INNER_ASSIGNMENT_RECEIPT,
            identity.INNER_MEMBERSHIP_NAME.format(fold=fold), f"components_{identity.SCOPE_INNER}.tsv",
            f"components_receipt_{identity.SCOPE_INNER}.json")


def check_calibration_folder(fold: int, work: Path, *, require_overlay: bool) -> Optional[Path]:
    """Refuse a folder of the campaign's own identity tree, test-named or symlinked inputs and, for real
    runs, a folder outside ``<overlay>/identity/``; return the overlay manifest when there is one."""
    work = Path(work)
    refuse_test_paths([work], what="--calibrate-from")
    if is_campaign_identity_path(work):
        raise SimulationError(f"--calibrate-from {fold}={work} lies under the campaign's identity folder, which holds "
                              "test-side content after the identity audit; read the data overlay's copy "
                              "(<overlay>/identity/fold<K>) instead.")
    if not work.is_dir():
        raise SimulationError(f"--calibrate-from {fold}={work} is not a folder.")
    for name in _calibration_file_names(fold):
        path = work / name
        if path.is_symlink() or is_campaign_identity_path(path):
            raise SimulationError(f"--calibrate-from {fold}: {name} is a link or lies under the campaign's identity "
                                  "folder; the overlay holds byte copies only.")
    manifest = work.parent.parent / OVERLAY_MANIFEST if work.parent.name == IDENTITY_FOLDER else None
    if manifest is not None and (manifest.is_symlink() or not manifest.is_file()):
        manifest = None
    if require_overlay and manifest is None:
        raise SimulationError(f"--calibrate-from {fold}={work}: a real run reads the data overlay's copy, "
                              f"<overlay>/{IDENTITY_FOLDER}/<fold folder> next to <overlay>/{OVERLAY_MANIFEST}.")
    return manifest


def _as_given_and_resolved(path: Path) -> tuple[Path, ...]:
    return tuple(dict.fromkeys((Path(os.path.abspath(path)), Path(path).resolve())))


def check_out_folder(out: Path, calibration_folders: Mapping[int, Path]) -> None:
    """Refuse an output folder under the campaign's identity folder, inside a calibration folder or
    inside a data overlay (any folder above it, or itself, holding ``overlay_manifest.json``).

    Records hold per-fold class proportions: the identity folder becomes test-bearing after the
    identity audit, and the overlay is checked file by file against its manifest.
    """
    out = Path(out)
    if is_campaign_identity_path(out):
        raise SimulationError(f"--out {out} lies under the campaign's identity folder, which holds test-side content "
                              "after the identity audit; write simulation outputs elsewhere.")
    candidates = _as_given_and_resolved(out)
    for fold, work in sorted(calibration_folders.items()):
        if any(candidate.is_relative_to(folder) for candidate in candidates
               for folder in _as_given_and_resolved(Path(work))):
            raise SimulationError(f"--out {out} lies inside the calibration folder of fold {fold}.")
    for candidate in candidates:
        for folder in (candidate, *candidate.parents):
            if os.path.lexists(folder / OVERLAY_MANIFEST):
                raise SimulationError(f"--out {out} lies inside a data overlay ({folder} holds {OVERLAY_MANIFEST}); "
                                      "write simulation outputs outside it.")


def load_fold_calibration(fold: int, work: Path, *, smoke: bool = False) -> FoldCalibration:
    """Read one fold's certified inner membership and inner components with the identity loaders."""
    work = Path(work)
    manifest = check_calibration_folder(fold, work, require_overlay=not smoke)
    try:
        chain = identity.load_inner_chain(work, fold)
        components, components_receipt_sha256 = identity._load_inner_components(work, fold, chain)
        membership, certificate_sha256 = identity._load_certified_membership(work, fold, chain)
    except (ValueError, KeyError, OSError) as exc:
        raise SimulationError(f"--calibrate-from {fold}={work}: {exc}") from exc
    inputs = {
        "membership_sha256": membership.file_sha256,
        "membership_identity_sha256": membership.identity_sha256,
        "certificate_sha256": certificate_sha256,
        "assignment_receipt_sha256": sha256_file(work / identity.INNER_ASSIGNMENT_RECEIPT),
        "components_receipt_sha256": components_receipt_sha256,
        "chain": chain.record(),
        "overlay_manifest_sha256": sha256_file(manifest) if manifest is not None else None,
    }
    return FoldCalibration.from_membership(fold, membership.rows, components, source=str(work), inputs=inputs)


# ---------------------------------------------------------------------------
# Points


@dataclass(frozen=True, eq=False)
class _ResolvedPoint:
    point: SweepPoint
    model: NullNoiseModel
    config: PromotionConfig
    resolution: Mapping[str, Any]
    implied: Mapping[str, Any]


def _resolve_points(spec: SweepSpec, points: Sequence[SweepPoint],
                    calibrations: Mapping[int, FoldCalibration]) -> list[_ResolvedPoint]:
    """Noise models, rules and diagnostics of ``points``, before any simulation runs.

    The seed-spread draws are large (about 48 bytes x proteins x ``SEED_SPREAD_POPULATIONS``), so
    the (fold, noise point) pairs are resolved one simulated population at a time and each
    population's draws are freed before the next; the results do not depend on that order.
    """
    first_point: dict[tuple[int, str], str] = {}
    for point in points:
        first_point.setdefault((point.fold, point.noise_point), point.point_id)
    unsolved: dict[tuple[int, str], NullNoiseModel] = {}
    by_population: dict[str, list[tuple[int, str]]] = {}
    for key, point_id in first_point.items():
        calibrated = calibrations[key[0]].noise_model
        try:
            model = replace(calibrated, **_model_values(calibrated, spec.noise[key[1]]))
        except ValueError as exc:
            raise SimulationError(f"Point {point_id}: {exc}") from exc
        unsolved[key] = model
        by_population.setdefault(_population_key(model), []).append(key)
    noise_cache: dict[tuple[int, str], tuple[NullNoiseModel, dict[str, Any], float]] = {}
    for keys in by_population.values():
        draws = _spread_draws(unsolved[keys[0]], SEED_SPREAD_POPULATIONS, spec.simulation_seed)
        for key in keys:
            model, overrides = unsolved[key], spec.noise[key[1]]
            try:
                resolution: dict[str, Any] = {"fit_shift_sd_from_seed_spread": None}
                if "seed_spread_points" in overrides:
                    solved = fit_shift_sd_for_seed_spread(model, overrides["seed_spread_points"], draws=draws)
                    model = replace(model, fit_shift_sd=solved["fit_shift_sd"])
                    resolution["fit_shift_sd_from_seed_spread"] = solved
                spread = seed_spread_points(model, draws=draws)
            except ValueError as exc:
                raise SimulationError(f"Point {first_point[key]}: {exc}") from exc
            noise_cache[key] = (model, resolution, spread)
        del draws
    resolved = []
    for point in points:
        model, resolution, spread = noise_cache[(point.fold, point.noise_point)]
        resolution = dict(resolution, challenger_shift_from_true_gain=None)
        if point.true_gain_points is not None:
            try:
                shift = challenger_shift_for_gain(model, point.true_gain_points)
            except ValueError as exc:
                raise SimulationError(f"Point {point.point_id}: {exc}") from exc
            model = replace(model, challenger_shift=shift)
            resolution["challenger_shift_from_true_gain"] = {"true_gain_points": point.true_gain_points,
                                                             "challenger_shift": shift}
        implied = {
            "expected_gain_points": expected_ba_points(model, model.challenger_shift) - expected_ba_points(model),
            "seed_spread_points": spread,
            "seed_spread_populations": SEED_SPREAD_POPULATIONS,
            "seed_spread_seed": [spec.simulation_seed, SEED_SPREAD_STREAM],
        }
        resolved.append(_ResolvedPoint(point, model, spec.config(point.rule), resolution, implied))
    return resolved


def _point_result(spec: SweepSpec, item: _ResolvedPoint, calibration: FoldCalibration,
                  characteristics: Mapping[str, Any]) -> dict[str, Any]:
    return to_jsonable({
        "calibration": calibration.result_record(),
        "noise_overrides": spec.noise[item.point.noise_point],
        "rule_overrides": spec.rules[item.point.rule],
        "noise_model": item.model.to_record(),
        "promotion_config": item.config.to_record(),
        "looks": item.point.looks,
        "seeds": list(SCREENING_SEEDS),
        "n_replicates": item.point.n_replicates,
        "simulation_seed": spec.simulation_seed,
        "resolution": item.resolution,
        "implied": item.implied,
        "operating_characteristics": characteristics,
    })


# ---------------------------------------------------------------------------
# Output folder


def _prepare_out(out: Path, sweep_bytes: bytes) -> None:
    copy = out / SWEEP_FILE
    if out.exists():
        if not out.is_dir() or not copy.is_file():
            raise SimulationError(f"--out {out} exists and is not a simulation folder of this sweep.")
        if copy.read_bytes() != sweep_bytes:
            raise SimulationError(f"--out {out} holds another sweep file; use a new folder for a changed sweep.")
    else:
        out.mkdir(parents=True)
        with copy.open("xb") as handle:
            handle.write(sweep_bytes)
    (out / RECORDS_FOLDER).mkdir(exist_ok=True)


def _record_path(out: Path, point_id: str) -> Path:
    return out / RECORDS_FOLDER / f"{point_id}.json"


def _check_existing_record(path: Path, spec: SweepSpec, point: SweepPoint, *, smoke: bool,
                           code_sha256: Mapping[str, str], calibration: FoldCalibration) -> dict[str, Any]:
    try:
        record = json.loads(path.read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SimulationError(f"{path.name} is not a readable record ({type(exc).__name__}); "
                              "use a new --out folder.") from None
    timing = record.get("timing") if isinstance(record, dict) else None
    if (not isinstance(record, dict)
            or not all(isinstance(record.get(key), dict) for key in ("point", "result", "provenance", "timing"))
            or isinstance(timing.get("cpu_seconds"), bool) or not isinstance(timing.get("cpu_seconds"), (int, float))):
        raise SimulationError(f"{path.name} is not a simulation record; use a new --out folder.")
    problems = []
    if record.get("format") != RECORD_FORMAT or record.get("spec_sha256") != spec.spec_sha256:
        problems.append("another sweep")
    if record.get("point") != point.to_record():
        problems.append("another point definition")
    if bool(record.get("smoke")) != bool(smoke):
        problems.append("smoke and real records mixed")
    if record["provenance"].get("code_sha256") != dict(code_sha256):
        problems.append("other campaign code")
    if record.get("result_sha256") != sha256_canonical(record["result"]):
        problems.append("a result that no longer matches its hash")
    if record["result"].get("calibration") != calibration.result_record():
        problems.append("another calibration")
    if problems:
        raise SimulationError(f"{path.name} was written for {', '.join(problems)}; use a new --out folder.")
    return record


def _forecast(spec: SweepSpec, records: Mapping[str, Mapping[str, Any]],
              limit_cpu_hours: Optional[float]) -> dict[str, Any]:
    per_replicate = max((record["timing"]["cpu_seconds"] / record["point"]["n_replicates"]
                         for record in records.values()), default=None)
    remaining = [point for point in spec.points if point.point_id not in records]
    remaining_replicates = sum(point.n_replicates for point in remaining)
    spent = sum(record["timing"]["cpu_seconds"] for record in records.values())
    total = None if per_replicate is None else (spent + per_replicate * remaining_replicates) / 3600.0
    return {
        "basis": "largest CPU seconds per replicate over the recorded points x the remaining replicates (rough)",
        "recorded_points": len(records),
        "recorded_cpu_hours": spent / 3600.0,
        "cpu_seconds_per_replicate": per_replicate,
        "remaining_points": len(remaining),
        "remaining_replicates": remaining_replicates,
        "forecast_total_cpu_hours": total,
        "limit_cpu_hours": limit_cpu_hours,
        "over_limit": bool(limit_cpu_hours is not None and total is not None and total > limit_cpu_hours),
    }


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".10g")
    return str(value)


def _summary_row(record: Mapping[str, Any]) -> dict[str, Any]:
    point, result = record["point"], record["result"]
    characteristics, model, config = result["operating_characteristics"], result["noise_model"], result["promotion_config"]
    overrides, resolution = result["noise_overrides"], result["resolution"]
    seed_spread = resolution.get("fit_shift_sd_from_seed_spread")
    offset = "custom" if "class_recall" in overrides else overrides.get("class_recall_offset")
    meets = None
    if point["kind"] == KIND_NULL:
        meets = characteristics["adoption_rate"] <= characteristics["target_rate"] + FLOAT_TOLERANCE
    return {
        "point_id": point["point_id"], "group": point["group"], "kind": point["kind"], "fold": point["fold"],
        "n_proteins": model["n_proteins"], "rule": point["rule"], "alpha": config["alpha"],
        "min_gain_points": config["min_gain_points"], "stratified": config["stratified"], "n_draws": config["n_draws"],
        "looks": point["looks"], "noise_point": point["noise_point"], "fit_shift_sd": model["fit_shift_sd"],
        "fit_noise_sd": model["fit_noise_sd"], "component_sd": model["component_sd"],
        "protein_sd": model["protein_sd"], "class_recall_offset": offset,
        "seed_spread_target_points": None if seed_spread is None else seed_spread["target_points"],
        "implied_seed_spread_points": result["implied"]["seed_spread_points"],
        "true_gain_points": point["true_gain_points"], "challenger_shift": model["challenger_shift"],
        "expected_gain_points": result["implied"]["expected_gain_points"],
        **{key: characteristics[key] for key in (
            "n_replicates", "n_adopted", "adoption_rate", "adoption_rate_standard_error", "target_rate")},
        "meets_target": meets,
        **{key: characteristics[key] for key in (
            "gain_pass_rate", "recall_guard_veto_rate", "zero_recall_veto_rate", "n_bootstrap_evaluated",
            "n_lower_bound_passed", "n_bootstrap_aborted", "gain_points_mean", "gain_points_sd")},
        "result_sha256": record["result_sha256"],
    }


def summary_text(records: Sequence[Mapping[str, Any]]) -> str:
    """Tab-separated summary, one row per point in sweep order; no class proportions, no timing."""
    lines = ["\t".join(SUMMARY_COLUMNS)]
    for record in records:
        row = _summary_row(record)
        lines.append("\t".join(_cell(row[column]) for column in SUMMARY_COLUMNS))
    return "\n".join(lines) + "\n"


def _write_summary(out: Path, spec: SweepSpec, records: Mapping[str, Mapping[str, Any]],
                   calibrations: Mapping[int, FoldCalibration], sweep_bytes: bytes,
                   provenance: Mapping[str, Any]) -> dict[str, Any]:
    text = summary_text([records[point.point_id] for point in spec.points]).encode("utf-8")
    path = out / SUMMARY_FILE
    if path.exists():
        if path.read_bytes() != text:
            raise SimulationError(f"{path} differs from the summary of the recorded points.")
    else:
        with path.open("xb") as handle:
            handle.write(text)
    summary = {"file": SUMMARY_FILE, "sha256": sha256_bytes(text), "n_points": len(spec.points)}
    receipt = out / RECEIPT_FILE
    if not receipt.exists():
        write_new_json(receipt, to_jsonable({
            "format": RECEIPT_FORMAT,
            "sweep_id": spec.sweep_id,
            "spec_sha256": spec.spec_sha256,
            "sweep_file_sha256": sha256_bytes(sweep_bytes),
            "summary": summary,
            "records": {point.point_id: {"file": f"{RECORDS_FOLDER}/{point.point_id}.json",
                                         "sha256": sha256_file(_record_path(out, point.point_id)),
                                         "result_sha256": records[point.point_id]["result_sha256"]}
                        for point in spec.points},
            "calibrations": {str(fold): calibration.public_record() for fold, calibration in sorted(calibrations.items())},
            "provenance": provenance,
        }))
    return summary


# ---------------------------------------------------------------------------
# Sweep runner


def run_sweep(
    sweep: Path,
    out: Path,
    calibrate_from: Sequence[str] | Mapping[int, Any],
    *,
    points: Optional[Sequence[str]] = None,
    smoke: bool = False,
    max_forecast_cpu_hours: Optional[float] = DEFAULT_MAX_FORECAST_CPU_HOURS,
    git: Optional[Mapping[str, Any]] = None,
    argv: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Run the selected points of a sweep (default: every point without a record) and return a printable
    payload (status, per-point adoption rates, forecast, calibration hashes; no class proportions).

    Every call names one calibration folder for each fold the sweep uses, and no other, and resolves
    every point of the sweep (seed-spread and true-gain conversions) before anything is written, so a
    point that cannot be resolved stops the first call, not a later one. A real run
    (``smoke=False``) needs a clean tree and calibration folders inside the data overlay. ``git``
    (synthetic checks only) stands in for the clean-tree check and the recorded git state; ``argv``
    replaces the recorded command line.
    """
    sweep, out = Path(sweep), Path(out)
    mapping = parse_calibrate_from(calibrate_from)
    refuse_test_paths([sweep, out], what="simulate path")
    if max_forecast_cpu_hours is not None and not float(max_forecast_cpu_hours) > 0.0:
        raise SimulationError("The forecast limit must be positive (CPU hours).")
    git = dict(git) if git is not None else real_action_git_state(smoke=smoke)
    spec, sweep_bytes = read_sweep(sweep)
    if points is None:
        selected = list(spec.points)
    else:
        wanted = [str(point_id) for point_id in points]
        unknown = [point_id for point_id in wanted if point_id not in {point.point_id for point in spec.points}]
        if unknown or not wanted:
            raise SimulationError(f"--points names points the sweep does not define: {unknown or wanted}.")
        selected = [point for point in spec.points if point.point_id in set(wanted)]
    sweep_folds = sorted({point.fold for point in spec.points})
    missing = [fold for fold in sweep_folds if fold not in mapping]
    if missing:
        raise SimulationError(f"The sweep uses folds {missing} without a --calibrate-from K=PATH; every call names "
                              "every fold of the sweep.")
    unused = sorted(set(mapping) - set(sweep_folds))
    if unused:
        raise SimulationError(f"--calibrate-from names folds {unused} that no point of the sweep uses.")
    check_out_folder(out, mapping)
    calibrations = {fold: load_fold_calibration(fold, mapping[fold], smoke=smoke) for fold in sorted(mapping)}
    resolution_cpu = time.process_time()
    resolved = _resolve_points(spec, spec.points, calibrations)
    resolution_cpu = time.process_time() - resolution_cpu

    provenance = provenance_record(smoke=smoke, git=git, argv=argv)
    _prepare_out(out, sweep_bytes)
    records: dict[str, dict[str, Any]] = {}
    for point in spec.points:
        path = _record_path(out, point.point_id)
        if path.exists():
            records[point.point_id] = _check_existing_record(
                path, spec, point, smoke=smoke, code_sha256=provenance["code_sha256"],
                calibration=calibrations[point.fold])

    status, reports = STATUS_PARTIAL, []
    limit = None if max_forecast_cpu_hours is None else float(max_forecast_cpu_hours)
    selected_ids = {point.point_id for point in selected}
    for item in resolved:
        point = item.point
        if point.point_id not in selected_ids:
            continue
        if point.point_id in records:
            reports.append(_point_report(records[point.point_id], "existing"))
            continue
        if _forecast(spec, records, limit)["over_limit"]:
            status = STATUS_STOPPED
            break
        started, cpu, wall = utc_now(), time.process_time(), time.perf_counter()
        characteristics = simulate_null_adoption(item.model, n_replicates=point.n_replicates, looks=point.looks,
                                                 config=item.config, seed=spec.simulation_seed).to_record()
        timing = {"started_utc": started, "cpu_seconds": time.process_time() - cpu,
                  "wall_seconds": time.perf_counter() - wall}
        result = _point_result(spec, item, calibrations[point.fold], characteristics)
        record = {
            "format": RECORD_FORMAT,
            "sweep_id": spec.sweep_id,
            "spec_sha256": spec.spec_sha256,
            "sweep_file_sha256": sha256_bytes(sweep_bytes),
            "point_id": point.point_id,
            "point": point.to_record(),
            "smoke": bool(smoke),
            "calibration_source": calibrations[point.fold].source,
            "result": result,
            "result_sha256": sha256_canonical(result),
            "timing": timing,
            "provenance": provenance,
        }
        write_new_json(_record_path(out, point.point_id), to_jsonable(record))
        records[point.point_id] = to_jsonable(record)
        reports.append(_point_report(records[point.point_id], "written"))

    summary = None
    if all(point.point_id in records for point in spec.points):
        status = STATUS_COMPLETE
        summary = _write_summary(out, spec, records, calibrations, sweep_bytes, provenance)
    return {
        "action": "simulate",
        "status": status,
        "smoke": bool(smoke),
        "sweep_id": spec.sweep_id,
        "spec_sha256": spec.spec_sha256,
        "out": str(out),
        "points_total": len(spec.points),
        "points_selected": len(selected),
        "points_recorded": len(records),
        "points": reports,
        "resolution_cpu_seconds": resolution_cpu,
        "forecast": _forecast(spec, records, limit),
        "calibrations": {str(fold): calibration.public_record() for fold, calibration in sorted(calibrations.items())},
        "summary": summary,
    }


def _point_report(record: Mapping[str, Any], status: str) -> dict[str, Any]:
    point, characteristics = record["point"], record["result"]["operating_characteristics"]
    return {
        "point_id": point["point_id"],
        "status": status,
        "kind": point["kind"],
        "looks": point["looks"],
        "n_replicates": characteristics["n_replicates"],
        "adoption_rate": characteristics["adoption_rate"],
        "adoption_rate_standard_error": characteristics["adoption_rate_standard_error"],
        "cpu_seconds": record["timing"]["cpu_seconds"],
        "result_sha256": record["result_sha256"],
    }


def run_simulate_action(args: Any) -> tuple[int, dict[str, Any]]:
    """The runner's ``simulate`` action: (exit code, payload to print).

    Reads ``args.sweep``, ``args.out``, ``args.calibrate_from`` (list of ``K=PATH``), ``args.points``
    (comma list or None), ``args.smoke`` and ``args.max_forecast_cpu_hours``. Exit code 0 when the
    sweep is complete or partial, ``EXIT_STOPPED`` when the forecast limit stopped it.
    """
    points_text = getattr(args, "points", None)
    points = [part.strip() for part in points_text.split(",") if part.strip()] if points_text else None
    payload = run_sweep(Path(args.sweep), Path(args.out), list(getattr(args, "calibrate_from", None) or []),
                        points=points, smoke=bool(getattr(args, "smoke", False)),
                        max_forecast_cpu_hours=getattr(args, "max_forecast_cpu_hours",
                                                       DEFAULT_MAX_FORECAST_CPU_HOURS))
    return (EXIT_STOPPED if payload["status"] == STATUS_STOPPED else 0), payload
