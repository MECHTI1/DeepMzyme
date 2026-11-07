"""Fold-0 futility rule of the EC class versus CLEAN30 campaign (amendment 2, rule A2-F).

    futility --fold 0 --step-chain STEP_RECORDS --plan <campaign root>/plans_v2/fold0.json --runs R [--runs R2 ...]
             --components C --freeze1-sha256 F --freeze1-amendment A --clean-r1-inner NPZ (x4)
             --clean-r2-inner NPZ (x4) --clean-sanity-record S --out FUTILITY [--smoke]

Once, after the fold-0 S1 and S2 records of ``amendment2_selection_v1`` and after the training-only CLEAN
R1-inner sanity record has passed, and before any fit of folds 1-4, this action writes the futility record. Every
constant is part of ``selection.selection_policy_record_v2`` (``policy_record``), which amendment 2 binds.

- Population U: the fold-0 inner-validation proteins of EC classes 1-6 whose full EC numbers all occur among the
  R1-inner training centres of their inner fold (``selector.inner_closure_rows``, shared with the historical gap
  gate, including its CLEAN query-set, in-sample and fold refusals), minus the proteins whose sequence equals a
  protein kept in their inner fold's R1-inner training set (from the fold-0 F6 corpus and the four R1-inner lists;
  the count is reported, and a record without them is never binding).
- Arms: DeepMzyme is the mean of the two screening-seed logits of the selected S1 configuration (re-pooled and
  matched to the S1 record's hash), adjusted with the S2 tau and that inner fit's training priors, then arg-max.
  CLEAN is R1-inner of the protein's inner fold (one seed), nearest-centre EC class; a distance tie is wrong.
- Units: the inner identity components of the overlay copy restricted to U; a component's stratum is its rarest
  EC class 1-6 by total protein count in U (``stats.assign_cluster_strata``).
- Projection (``project_success``): for r = 0..1999 with a generator seeded by (20261104, r): (a) resample the
  components within each stratum to its component count; (b) synthetic B1-all (526 proteins of EC classes 1-6):
  draw n1_s = max(1, floor(k_s * N1 / n_U + 0.5)) components with replacement, in draw order, from each stratum of
  the step-(a) sample, each copy its own cluster; (b') synthetic B1-NH30 (226, nested): the first
  n2_s = max(1, floor(n1_s * N2 / N1 + 0.5)) copies of each stratum of step (b); (c) score both synthetic
  populations as the scoring pass scores a co-primary
  population (``claim_engine_scorer``: primary metric, stratified paired cluster bootstrap with 10,000 draws, seed
  20261023, at most 1% skipped draws, the amended stratum gate; a gated or aborted population is unavailable with
  p = 1; intersection-union over the two R1 components; ``stats.co_primary_holm``); (d) replicate r succeeds if at
  least one population reads "Outperforms CLEAN".
- Decision (``futility_decision``): continue to folds 1-4 if at least 400 of the 2,000 replicates succeed, else
  stop for futility (integers). A stop closes the campaign without test scoring.
- Descriptive only: projected probabilities per population, for both and for either (Wilson 95% intervals); the
  observed inner primary difference and its component-bootstrap bound; conditional power (step (a) omitted, 500
  replicates); the projection with target sizes divided by 1.25 and 1.5 (500 replicates each); the historical G4
  computation (``selector.score_gap_gate(after="S1")`` on the same inputs) as ``historical_rule_outcome``; the
  same projection with independent (non-nested) draws (500 replicates); C-GF beside CLEAN R1-inner and the
  selected configuration on U (``cgf_beside_clean``).
- ``operating_characteristics`` (``futility-oc``) writes ``futility_oc.json`` once: on a declared synthetic U at
  true margins of 0, 2, 4, 6 and 8 points it runs the decision projection with 200 replicates per margin and
  reports the projected success probability and the decision the rule would take (descriptive, label-free).

The record is written once and holds totals only, no protein identifier; the printed summary holds the decision
and population totals. A binding record is then bound twice (rule A2-GOV-7(d), contract X7): a write-once pointer
file ``<campaign root>/amendment2/futility/fut_seq<n>.json`` (n = the chain's tip sequence) and a FUT step record
appended to the fold-0 step chain. ``verify_futility_decision`` reads the pointer files by exact name; any
``stop_futility`` pointer wins, a chain with a second FUT record or an S1 or S2 record after FUT is refused, and a
stop is never followed by another futility run. FUT runs once per fit chain: a pointer of the same fit chain
refuses a second run.

Binding (review fix): outside ``--smoke`` the action refuses before reading or writing anything unless the record
can be binding: the identical-sequence exclusion inputs (``--f6-corpus`` and the four ``--r1-inner-list`` files)
are given, no in-process test override is set and the claim engine has its frozen settings. A ``--smoke`` run needs
its own ``--campaign-root`` (never the plan's, and one without pointer files) and writes only the record: no
pointer file and no FUT step record, so it can neither stop nor block the campaign. The CLEAN inputs are bound: the
R1-inner and R2-inner exports must carry exactly the checkpoint SHA-256 values of the held-out sanity record that
the fold-0 S1 record binds, and outside ``--smoke`` the S1 record must bind that record. When the held-out
record's consequence (``clean_sanity.held_out_consequence``, choice C6b) leaves no co-primary available, the
decision is ``stop_futility`` with 0 successes and the projection is not run. The claim engine's primary metric and amended stratum gate come from ``ec_clean30.stats``
(``METRIC_EC_CLASSES_1_6_ACCURACY``, ``ENGINE_AMENDMENT2``, ``inference_gate``); without them the action refuses.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from ec_clean30 import stats
from ec_clean30.campaign import CAMPAIGN_ID, PLAN_FORMAT_V2, STEP_S1, STEP_S2
from ec_clean30.logit_adjustment import adjust, validate_tau
from ec_clean30.provenance import provenance_record, real_action_git_state, refuse_test_paths, sha256_file, write_new_json
from ec_clean30 import selection
from ec_clean30.selection import (
    FUTILITY_CONTINUE, FUTILITY_DESCRIPTIVE_REPLICATES, FUTILITY_DESCRIPTIVE_STREAMS, FUTILITY_FOLD,
    FUTILITY_MIN_SUCCESSES, FUTILITY_POPULATIONS, FUTILITY_REPLICATES, FUTILITY_RULE_V2, FUTILITY_SEED,
    FUTILITY_SENSITIVITY_DIVISORS, FUTILITY_STOP, FUTILITY_STOP_WORDING, FUTILITY_TARGET_SIZES, POINTS,
    PRIMARY_METRIC_V2, SELECTION_PIPELINE_V2, predictions_sha256, selection_policy_sha256_v2,
)
from training.ec_class_metrics import N_EC_CLASSES, PRIMARY_CLASS_INDICES


FUTILITY_RECORD_FORMAT = "deepmzyme_ec_clean30_futility_record_v1"
FUTILITY_POINTER_FORMAT = "deepmzyme_ec_clean30_futility_pointer_v1"
SCHEMA_VERSION = 1
POINTER_FOLDER = ("amendment2", "futility")
POINTER_NAME = "fut_seq{sequence}.json"
FIRST_POINTER_SEQUENCE = 2
DM_ARM = "deepmzyme"
CLEAN_ARM = "clean_r1_inner"
# The claim engine's system and contrast names (scoring.DM_FINAL, R1_SEED_MEAN, R1_ENSEMBLE, PRIMARY_COMPONENTS):
# the one CLEAN R1-inner arm stands for both the R1 seed mean and the R1 ensemble.
SYSTEMS: Mapping[str, tuple[str, ...]] = {"deepmzyme_final": (DM_ARM,), "r1_seed_mean": (CLEAN_ARM,),
                                          "r1_ensemble": (CLEAN_ARM,)}
COMPONENTS: tuple[str, ...] = ("deepmzyme_final_minus_r1_seed_mean", "deepmzyme_final_minus_r1_ensemble")
CONTRASTS: tuple[stats.Contrast, ...] = (
    stats.Contrast.difference(COMPONENTS[0], "deepmzyme_final", "r1_seed_mean"),
    stats.Contrast.difference(COMPONENTS[1], "deepmzyme_final", "r1_ensemble"),
)
WILSON_Z = 1.959963984540054
OBSERVED_BOUND_ALPHA = 0.025
CLEAN_SANITY_ACCEPTED = ("passed",)
CLEAN_SANITY_DISPOSITION_ACCEPTED = ("proceed_disclosed",)
PRINTED_TOTALS: tuple[str, ...] = (
    "n_inner_validation_proteins", "n_in_r1_inner_closure", "n_population", "n_components",
)
# In-process switches for the synthetic tests only (fewer replicates and draws, or a stand-in metric); no
# command-line flag reaches them. A record written with any of them is marked and is never binding.
_TEST_OVERRIDES: dict[str, Any] = {}
_OVERRIDABLE = ("n_replicates", "n_descriptive_replicates", "n_draws", "metric")
# In-process switch for the synthetic tests only: let a non-smoke run that cannot be binding write its record
# (only the record: never a pointer file or a FUT step record). No command-line flag reaches it.
_TEST_ALLOW_NON_BINDING = False

# Operating characteristics (descriptive; rule A2-FUT-8): a declared synthetic U and its true margins.
OC_FORMAT = "deepmzyme_ec_clean30_futility_oc_v1"
OC_MARGINS_POINTS: tuple[int, ...] = (0, 2, 4, 6, 8)
OC_REPLICATES = 200
OC_SEED = 20261105
OC_DESIGN: Mapping[str, Any] = {
    "proteins_per_ec_class_1_6": (120, 110, 100, 60, 30, 40),
    "component_size": 2,
    "clean_accuracy": 0.70,
    "wrong_prediction": "(true class + 1) mod 6",
    "margin_rule": ("DeepMzyme is right wherever CLEAN is right, plus round(margin * n_U / 100) CLEAN errors in a "
                    "fixed random order (common random numbers across margins)"),
}

FUTILITY_HELP = ("Fold-0 futility rule of amendment 2: projected probability that the final claim succeeds, from the "
                 "selected S1 configuration and CLEAN R1-inner on the R1-inner EC closure; continue only at >= 0.20.")


class FutilityInputError(ValueError):
    """``futility`` refuses its inputs; no record is written."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise FutilityInputError(message)


def policy_record() -> dict[str, Any]:
    """Rule A2-F as bound by amendment 2 (the futility block of ``selection.selection_policy_record_v2``)."""
    return selection.futility_policy_block_v2()


futility_policy_record = policy_record


# ---------------------------------------------------------------------------
# Projection inputs and synthetic populations


@dataclass(frozen=True, eq=False)
class ProjectionInputs:
    """Population U: per protein the true class, both arms' predicted classes and its component (an index)."""

    true: np.ndarray
    dm_predicted: np.ndarray
    clean_predicted: np.ndarray
    component: np.ndarray
    strata: tuple[str, ...]  # per component, in component-index order
    rows_of_component: tuple[np.ndarray, ...] = field(repr=False, default=())

    def __post_init__(self) -> None:
        n = int(np.asarray(self.true).size)
        for name in ("true", "dm_predicted", "clean_predicted", "component"):
            values = np.asarray(getattr(self, name))
            if values.shape != (n,) or not np.issubdtype(values.dtype, np.integer):
                raise ValueError(f"{name} must hold one integer per protein of U.")
            object.__setattr__(self, name, values.astype(np.int64))
        if n == 0:
            raise ValueError("Population U is empty.")
        if not np.isin(self.true, PRIMARY_CLASS_INDICES).all():
            raise ValueError("Population U holds only proteins of EC classes 1-6.")
        n_components = int(self.component.max()) + 1
        if sorted(set(self.component.tolist())) != list(range(n_components)) or len(self.strata) != n_components:
            raise ValueError("Components must be numbered 0..K-1 with one stratum each.")
        object.__setattr__(self, "rows_of_component",
                           tuple(np.flatnonzero(self.component == index) for index in range(n_components)))

    @property
    def n_proteins(self) -> int:
        return int(self.true.size)

    @property
    def n_components(self) -> int:
        return len(self.strata)

    def stratum_members(self) -> dict[str, np.ndarray]:
        """Component indices per stratum, in the claim engine's stratum order."""
        return stats.stratum_members(self.strata)

    def totals(self) -> dict[str, Any]:
        return {"n_population": self.n_proteins, "n_components": self.n_components,
                "n_strata": len(self.stratum_members()),
                "largest_component_size": int(max(rows.size for rows in self.rows_of_component)),
                "components_per_stratum": {name: int(indices.size) for name, indices in self.stratum_members().items()}}


@dataclass(frozen=True, eq=False)
class SyntheticPopulation:
    """Rows of drawn component copies; each copy is its own cluster."""

    protein_ids: tuple[str, ...]
    true: np.ndarray
    dm_predicted: np.ndarray
    clean_predicted: np.ndarray
    cluster_of: Mapping[str, str] = field(repr=False)

    @property
    def n_proteins(self) -> int:
        return len(self.protein_ids)


def projection_inputs(true: Sequence[int], dm_predicted: Sequence[int], clean_predicted: Sequence[int],
                      component_labels: Sequence[str]) -> ProjectionInputs:
    """``ProjectionInputs`` from per-protein arrays and component labels (strata from the class support in U)."""
    labels = [str(value) for value in component_labels]
    order = sorted(set(labels))
    index = {label: position for position, label in enumerate(order)}
    component = np.array([index[label] for label in labels], dtype=np.int64)
    true = np.asarray(true, dtype=np.int64)
    support = np.zeros((len(order), N_EC_CLASSES))
    np.add.at(support, (component, true), 1.0)
    strata = stats.assign_cluster_strata(support)
    return ProjectionInputs(true, np.asarray(dm_predicted, dtype=np.int64), np.asarray(clean_predicted, dtype=np.int64),
                            component, tuple(strata))


def synthetic_population(inputs: ProjectionInputs, components: Sequence[int]) -> SyntheticPopulation:
    """The rows of the drawn component copies (components move as units; copy j is cluster ``copy<j>``)."""
    rows, ids, cluster_of = [], [], {}
    for copy, component in enumerate(int(value) for value in components):
        members = inputs.rows_of_component[component]
        rows.append(members)
        cluster = f"copy{copy:06d}"
        for row in members.tolist():
            pid = f"{cluster}:{row}"
            ids.append(pid)
            cluster_of[pid] = cluster
    selected = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    return SyntheticPopulation(tuple(ids), inputs.true[selected], inputs.dm_predicted[selected],
                               inputs.clean_predicted[selected], cluster_of)


def stratum_draw_counts(inputs: ProjectionInputs, target: float) -> dict[str, int]:
    """Step (b): max(1, floor(k_s * N / n_U + 0.5)) components per stratum for a target size N."""
    return {name: max(1, int(math.floor(indices.size * float(target) / inputs.n_proteins + 0.5)))
            for name, indices in inputs.stratum_members().items()}


def nested_draw_counts(p1_counts: Mapping[str, int], n1: float, n2: float) -> dict[str, int]:
    """Step (b'): the first max(1, floor(n1_s * N2 / N1 + 0.5)) copies of each stratum of synthetic B1-all form
    synthetic B1-NH30, so every B1-NH30 cluster is also a B1-all cluster."""
    return {name: min(count, max(1, int(math.floor(count * float(n2) / float(n1) + 0.5))))
            for name, count in p1_counts.items()}


# ---------------------------------------------------------------------------
# Claim engine for one population


PopulationScorer = Callable[[SyntheticPopulation], tuple[Optional[stats.BootstrapResult], Optional[str]]]


def _snap(result: stats.BootstrapResult) -> stats.BootstrapResult:
    """Contrast zero-snapping as in the scoring pass, from ``stats`` (no post-fit analysis file is loaded)."""
    return stats.snap_contrasts(result)


def claim_engine_scorer(*, metric: Optional[str] = None, n_draws: int = stats.DEFAULT_N_DRAWS,
                        seed: int = stats.DEFAULT_BOOTSTRAP_SEED,
                        max_skip_fraction: float = stats.DEFAULT_MAX_SKIP_FRACTION) -> PopulationScorer:
    """A co-primary population as the amended scoring pass scores it: the stratum gate, then the stratified paired
    cluster bootstrap of the two R1 contrasts on the primary metric. A gated or aborted population is unavailable
    (None and its reason).

    ``metric`` defaults to the claim engine's primary metric (``stats.METRIC_EC_CLASSES_1_6_ACCURACY``) with the
    amended engine and gate when ``stats`` provides them; without them the scorer refuses. Population U holds no
    class-7 protein, so on these populations the historical ``accuracy`` equals the primary metric; the synthetic
    tests pass it explicitly.
    """
    engine_kwargs: dict[str, Any] = {}
    if metric is None:
        metric = getattr(stats, "METRIC_EC_CLASSES_1_6_ACCURACY", None)
        _require(metric is not None and metric in stats.BOOTSTRAP_METRICS_V2,
                 "The futility projection needs the amended claim engine (stats.METRIC_EC_CLASSES_1_6_ACCURACY); "
                 "it is not available.")
    if hasattr(stats, "ENGINE_AMENDMENT2"):
        engine_kwargs["engine"] = stats.ENGINE_AMENDMENT2
    _require(metric in stats.BOOTSTRAP_METRICS_V2, f"Unknown claim-engine metric {metric!r}.")
    gate_function = getattr(stats, "inference_gate", None)

    def score(population: SyntheticPopulation) -> tuple[Optional[stats.BootstrapResult], Optional[str]]:
        table = stats.build_cluster_count_table(
            population.protein_ids, population.true,
            {DM_ARM: population.dm_predicted, CLEAN_ARM: population.clean_predicted}, population.cluster_of)
        if gate_function is not None and "engine" in engine_kwargs:
            reasons = list(gate_function(table, metric, engine_kwargs["engine"]) or [])
        else:
            reasons = [f"single-cluster stratum {name}" for name in stats.single_cluster_strata(table, metric)]
        if reasons:
            return None, "stratum gate: " + "; ".join(reasons)
        try:
            result = stats.paired_cluster_bootstrap(
                table, list(CONTRASTS), systems={name: list(arms) for name, arms in SYSTEMS.items()}, metric=metric,
                n_draws=int(n_draws), seed=int(seed), stratified=True, max_skip_fraction=float(max_skip_fraction),
                **engine_kwargs)
        except stats.BootstrapAbort as exc:
            return None, f"bootstrap aborted: {exc}"
        return _snap(result), None

    score.settings = {"metric": metric, "n_draws": int(n_draws), "seed": int(seed), "stratified": True,  # type: ignore[attr-defined]
                      "max_skip_fraction": float(max_skip_fraction), **engine_kwargs,
                      "gate": "stats.inference_gate" if gate_function is not None and engine_kwargs
                      else "stats.single_cluster_strata"}
    return score


# ---------------------------------------------------------------------------
# Projection and decision


def wilson_interval(successes: int, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    """Wilson score interval of a binomial proportion (descriptive only)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def _replicate_generator(seed: int, replicate: int, stream: Optional[int]) -> np.random.Generator:
    return np.random.default_rng([int(seed), int(replicate)] if stream is None
                                 else [int(seed), int(stream), int(replicate)])


def project_success(
    inputs: ProjectionInputs,
    *,
    scorer: PopulationScorer,
    n_replicates: int = FUTILITY_REPLICATES,
    seed: int = FUTILITY_SEED,
    targets: Mapping[str, int] = FUTILITY_TARGET_SIZES,
    divisor: float = 1.0,
    resample_components: bool = True,
    stream: Optional[int] = None,
    nested: bool = selection.FUTILITY_NESTED_POPULATIONS,
) -> dict[str, Any]:
    """Steps (a) to (e) of rule A2-FUT-5 over ``n_replicates`` replicates; returns success counts (totals only).

    ``resample_components=False`` omits step (a) (conditional power); ``divisor`` divides the target sizes;
    ``stream`` selects a descriptive seed stream (the decision projection has none); ``nested=False`` draws the
    synthetic B1-NH30 independently of B1-all (descriptive only).
    """
    _require(n_replicates >= 1, "The projection needs at least one replicate.")
    _require(set(targets) == {"P1", "P2"}, "The projection scores the co-primary populations P1 and P2.")
    members = inputs.stratum_members()
    counts = {name: stratum_draw_counts(inputs, targets[name] / float(divisor)) for name in ("P1", "P2")}
    if nested:
        counts["P2"] = nested_draw_counts(counts["P1"], targets["P1"], targets["P2"])
    passed = {"P1": 0, "P2": 0}
    unavailable = {"P1": 0, "P2": 0}
    both = either = 0
    sizes = {"P1": [], "P2": []}
    reasons: dict[str, int] = {}
    for replicate in range(int(n_replicates)):
        rng = _replicate_generator(seed, replicate, stream)
        base = {name: (rng.choice(indices, size=indices.size, replace=True) if resample_components else indices)
                for name, indices in members.items()}
        results = {}
        p1_drawn = None
        for population in ("P1", "P2"):
            if nested and population == "P2":
                drawn = [copies[:counts["P2"][name]] for name, copies in zip(members, p1_drawn)]
            else:
                drawn = [rng.choice(base[name], size=counts[population][name], replace=True) for name in members]
            if population == "P1":
                p1_drawn = drawn
            synthetic = synthetic_population(inputs, np.concatenate(drawn))
            sizes[population].append(synthetic.n_proteins)
            result, reason = scorer(synthetic)
            if result is None:
                unavailable[population] += 1
                key = (reason or "unavailable").split(":")[0]
                reasons[key] = reasons.get(key, 0) + 1
            results[population] = result
        co_primary = stats.co_primary_holm(("P1", results["P1"], COMPONENTS), ("P2", results["P2"], COMPONENTS))
        p1, p2 = co_primary.passed("P1"), co_primary.passed("P2")
        passed["P1"] += int(p1)
        passed["P2"] += int(p2)
        both += int(p1 and p2)
        either += int(p1 or p2)
    n = int(n_replicates)
    summary = {
        "n_replicates": n,
        "n_successes": either,
        "projected_probability": either / n,
        "projected_probability_wilson95": list(wilson_interval(either, n)),
        "per_population": {name: {"population": FUTILITY_POPULATIONS[name], "target_size": targets[name] / float(divisor),
                                  "components_drawn_per_stratum_total": int(sum(counts[name].values())),
                                  "n_outperforms": passed[name], "probability": passed[name] / n,
                                  "wilson95": list(wilson_interval(passed[name], n)),
                                  "n_unavailable": unavailable[name],
                                  "realized_size_mean": float(np.mean(sizes[name])),
                                  "realized_size_min": int(min(sizes[name])), "realized_size_max": int(max(sizes[name]))}
                           for name in ("P1", "P2")},
        "both": {"n": both, "probability": both / n, "wilson95": list(wilson_interval(both, n))},
        "either": {"n": either, "probability": either / n, "wilson95": list(wilson_interval(either, n))},
        "unavailable_reasons": dict(sorted(reasons.items())),
        "step_a_resampling": bool(resample_components),
        "nested_populations": bool(nested),
        "target_divisor": float(divisor),
        "seed": int(seed),
        "seed_stream": stream,
    }
    return summary


def futility_decision(n_successes: int, n_replicates: int = FUTILITY_REPLICATES) -> str:
    """Continue if the success share reaches 400 / 2000 (an integer comparison); otherwise stop for futility."""
    _require(isinstance(n_successes, int) and isinstance(n_replicates, int) and 0 <= n_successes <= n_replicates
             and n_replicates > 0, "Success counts must be integers with 0 <= successes <= replicates.")
    return (FUTILITY_CONTINUE if n_successes * FUTILITY_REPLICATES >= FUTILITY_MIN_SUCCESSES * n_replicates
            else FUTILITY_STOP)


def observed_difference(inputs: ProjectionInputs, scorer: PopulationScorer) -> dict[str, Any]:
    """The inner primary difference DeepMzyme minus CLEAN on U, with its component bootstrap (descriptive)."""
    population = synthetic_population(inputs, range(inputs.n_components))
    dm = float(np.mean(population.dm_predicted == population.true)) * POINTS
    clean = float(np.mean(population.clean_predicted == population.true)) * POINTS
    record: dict[str, Any] = {"deepmzyme_points": dm, "clean_r1_inner_points": clean, "difference_points": dm - clean,
                              "bound_alpha": OBSERVED_BOUND_ALPHA}
    result, reason = scorer(population)
    if result is None:
        record.update({"available": False, "reason": reason})
        return record
    test = result.test(COMPONENTS[0], OBSERVED_BOUND_ALPHA)
    record.update({"available": True, "bootstrap_estimate_points": POINTS * test.estimate,
                   "lower_bound_points": POINTS * test.lower_bound, "p_value": test.p_value,
                   "n_valid_draws": test.n_valid_draws})
    return record


# ---------------------------------------------------------------------------
# Inputs from the fold-0 records


def deepmzyme_predictions(pooled: selection.PooledOof, tau: float) -> np.ndarray:
    """Arg-max of the seed-mean logits adjusted with tau and the inner fit's training priors (equal across seeds)."""
    tau = validate_tau(tau)
    mean = pooled.logits.mean(axis=0)
    if tau == 0.0:
        return mean.argmax(axis=-1)
    _require(pooled.row_priors is not None, "The selected configuration carries no training priors.")
    _require(all(np.array_equal(pooled.row_priors[0], pooled.row_priors[k]) for k in range(len(pooled.seeds))),
             "The screening seeds of one inner fold must share their training priors.")
    return adjust(mean, pooled.row_priors[0], tau).argmax(axis=-1)


def build_projection_inputs(configuration: Any, tau: float, r1_exports: Mapping[int, Any],
                            component_of: Mapping[str, str], *,
                            exclude_ids: Optional[set[str]] = None,
                            population_ids: Optional[list] = None) -> tuple[ProjectionInputs, dict[str, Any]]:
    """Population U, both arms and the components from the selected S1 configuration and the R1-inner exports;
    ``exclude_ids`` are the proteins whose sequence equals one CLEAN R1-inner keeps (``None``: not applied)."""
    from ec_clean30 import selector

    pooled = configuration.pooled
    rows = selector.inner_closure_rows(configuration, r1_exports)
    keep = rows["keep"]
    dm = deepmzyme_predictions(pooled, tau)[keep]
    clean = rows["arms"][CLEAN_ARM]["predicted"]
    true = rows["true"]
    in_population = np.isin(true, PRIMARY_CLASS_INDICES)
    n_identical = 0
    if exclude_ids is not None:
        identical = np.array([pid in exclude_ids for pid in rows["ids"]], dtype=bool)
        n_identical = int(np.count_nonzero(identical & in_population))
        in_population &= ~identical
    _require(in_population.any(), "No closure-eligible inner-validation protein of EC classes 1-6.")
    ids = [pid for pid, kept in zip(rows["ids"], in_population) if kept]
    if population_ids is not None:
        population_ids.extend(ids)  # in-memory only (C-GF descriptive); never written to a record
    missing = sum(1 for pid in ids if pid not in component_of)
    _require(missing == 0, f"{missing} proteins of U have no inner identity component.")
    inputs = projection_inputs(true[in_population], dm[in_population], clean[in_population],
                               [component_of[pid] for pid in ids])
    totals = {"n_inner_validation_proteins": len(pooled.protein_ids), "n_in_r1_inner_closure": int(keep.sum()),
              "n_outside_closure": int((~keep).sum()), "n_class_7_in_closure": int((~in_population).sum()),
              "n_clean_ties_in_population": int(np.count_nonzero(rows["arms"][CLEAN_ARM]["tie"][in_population])),
              "identical_sequence_exclusion_applied": exclude_ids is not None,
              "n_excluded_identical_to_kept_r1_inner_training": n_identical,
              **inputs.totals()}
    return inputs, totals


def _settings() -> dict[str, Any]:
    unknown = sorted(set(_TEST_OVERRIDES) - set(_OVERRIDABLE))
    _require(not unknown, f"Unknown futility test overrides {unknown}.")
    return {"n_replicates": FUTILITY_REPLICATES, "n_descriptive_replicates": FUTILITY_DESCRIPTIVE_REPLICATES,
            "n_draws": stats.DEFAULT_N_DRAWS, "metric": None, **_TEST_OVERRIDES}


def _verify_clean_sanity(path: Path, dispositions: Sequence[Path] = ()) -> Mapping[str, Any]:
    """The fold-0 CLEAN held-out sanity record through ``ec_clean30.clean_sanity.held_out_binding``; refuses without
    the module."""
    try:
        from ec_clean30 import clean_sanity
    except ImportError as exc:
        raise FutilityInputError("The futility record needs the CLEAN sanity verifier (ec_clean30.clean_sanity), "
                                 "which is not available.") from exc
    try:
        return clean_sanity.held_out_binding(Path(path), [Path(item) for item in dispositions])
    except clean_sanity.SanityError as exc:
        message = str(exc)
        if "failed sanity record needs" in message or "status" in message:
            message += "; comparator diagnosis is required; fix the defects first, without retuning"
        raise FutilityInputError(f"--clean-sanity-record: {message}") from exc


def clean_sanity_binding(path: Path, dispositions: Sequence[Path] = ()) -> dict[str, Any]:
    """The fold-0 CLEAN held-out sanity record (R1-inner and R2-inner) must have passed, or carry exactly one
    disposition "proceed, disclosed" (rule A2-CLN-2)."""
    path = Path(path)
    refuse_test_paths([path, *dispositions], what="--clean-sanity-record/--clean-sanity-disposition")
    _require(path.is_file() and not path.is_symlink(), "--clean-sanity-record is missing or a link.")
    result = _verify_clean_sanity(path, dispositions)
    _require(isinstance(result, Mapping), "The CLEAN sanity verifier returned no result.")
    status, disposition = result.get("status"), result.get("disposition")
    _require(status in CLEAN_SANITY_ACCEPTED or disposition in CLEAN_SANITY_DISPOSITION_ACCEPTED,
             f"The CLEAN held-out sanity record has status {status!r}: comparator diagnosis is required; fix the "
             "defects first, without retuning.")
    return {"name": path.name, "sha256": sha256_file(path), "status": status, "disposition": disposition,
            "dispositions": [sha256_file(Path(item)) for item in dispositions],
            "arm_status": result.get("arm_status"), "arm_disposition": result.get("arm_disposition"),
            "consequence": result.get("consequence"),
            "checkpoint_sha256_by_fit": result.get("checkpoint_sha256_by_fit")}


def check_clean_inputs(sanity: Mapping[str, Any], exports: Mapping[str, Mapping[int, Any]]) -> None:
    """The R1-inner and R2-inner exports must be the fits the held-out sanity record checked: their
    ``(arm, inner fold) -> checkpoint SHA-256`` map equals the record's."""
    recorded = sanity.get("checkpoint_sha256_by_fit")
    _require(isinstance(recorded, Mapping), "The CLEAN held-out sanity record names no checkpoint per fit.")
    given = {arm: {str(inner): export.receipt.get("checkpoint_sha256") for inner, export in sorted(arm_exports.items())}
             for arm, arm_exports in exports.items()}
    expected = {arm: dict(recorded.get(arm) or {}) for arm in exports}
    _require(given == expected, "The CLEAN R1-inner/R2-inner exports are not the fits the held-out sanity record "
                                "checked (checkpoint SHA-256 per arm and inner fold differ).")


# ---------------------------------------------------------------------------
# Exact-sequence copies kept by CLEAN R1-inner (rule A2-FUT-2)


def read_corpus_sequences(path: Path) -> dict[str, str]:
    """``accession -> sequence_sha256`` of an F6 corpus table (``corpus.tsv``; torch-free reader)."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        _require(reader.fieldnames is not None and reader.fieldnames[:3] == ["accession", "sequence_sha256", "ec_class"],
                 f"{Path(path).name} is not an F6 corpus table.")
        return {row["accession"]: row["sequence_sha256"] for row in reader}


def read_r1_inner_list(path: Path) -> set[str]:
    """The accessions of one CLEAN R1-inner exclusion list (``r1_inner_exclusions_fold0_inner<j>.tsv``)."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        _require(reader.fieldnames == ["accession", "reason"], f"{Path(path).name} is not an R1-inner exclusion list.")
        return {row["accession"] for row in reader}


def kept_identical_sequence_ids(inner_fold_of: Mapping[str, int], sequences: Mapping[str, str],
                                r1_lists: Mapping[int, set[str]]) -> set[str]:
    """Inner-validation proteins whose sequence equals a protein that CLEAN R1-inner keeps in its training set for
    their inner fold (an accession outside that fold's R1-inner exclusion list); CLEAN answers them from memory."""
    _require(sorted(r1_lists) == sorted(set(inner_fold_of.values())), "One R1-inner list per inner fold is needed.")
    by_sequence: dict[str, set[str]] = {}
    for accession, key in sequences.items():
        by_sequence.setdefault(key, set()).add(accession)
    excluded = set()
    for pid, inner in inner_fold_of.items():
        _require(pid in sequences, "An inner-validation protein is missing from the F6 corpus.")
        _require(pid in r1_lists[inner], "An inner-validation protein is missing from its R1-inner exclusion list.")
        if any(other not in r1_lists[inner] for other in by_sequence[sequences[pid]] - {pid}):
            excluded.add(pid)
    return excluded


# ---------------------------------------------------------------------------
# Precedence: pointer files and the FUT step record (rule A2-GOV-7(d), contract X7)


def pointer_path(campaign_root: Path, sequence: int) -> Path:
    return Path(campaign_root).joinpath(*POINTER_FOLDER, POINTER_NAME.format(sequence=int(sequence)))


def _read_pointer(path: Path) -> dict[str, Any]:
    _require(not path.is_symlink(), f"{path.name} is a link.")
    try:
        pointer = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FutilityInputError(f"{path.name} does not parse ({exc.__class__.__name__}).") from exc
    _require(isinstance(pointer, dict) and pointer.get("format") == FUTILITY_POINTER_FORMAT
             and pointer.get("decision") in (FUTILITY_CONTINUE, FUTILITY_STOP)
             and isinstance(pointer.get("futility_record_sha256"), str),
             f"{path.name} is not a futility pointer file.")
    _require(pointer.get("binding") is True, f"{path.name} is not binding: only binding futility records write "
                                             "pointer files.")
    return pointer


def read_pointers(campaign_root: Path, tip_sequence: int) -> dict[int, dict[str, Any]]:
    """The pointer files ``fut_seq<n>.json`` for n = 2 .. tip sequence, read by exact name (no glob)."""
    pointers = {}
    for sequence in range(FIRST_POINTER_SEQUENCE, int(tip_sequence) + 1):
        path = pointer_path(campaign_root, sequence)
        if path.exists() or path.is_symlink():
            pointers[sequence] = _read_pointer(path)
    return pointers


def _policy_and_chain(step_chain: Path, freeze1_sha256: str, amendment_tip: Optional[Path], smoke: bool,
                      chain_receipt: Optional[Path] = None, expected_chain: Optional[Sequence[str]] = None):
    from ec_clean30 import selector

    freeze1_sha256 = selector._check_sha256(freeze1_sha256, "--freeze1-sha256")
    policy = selector._selection_policy(amendment_tip, freeze1_sha256, smoke=smoke,
                                        **selector._chain_options(chain_receipt, expected_chain))
    _require(selector.is_pipeline_v2(policy),
             f"The futility rule belongs to {SELECTION_PIPELINE_V2}; supply the amendment-2 chain.")
    records = selector._read_chain(Path(step_chain), freeze1_sha256, "--step-chain")
    selector._check_policy_chain(records, policy)
    return policy, records, selector._fold_records(records, FUTILITY_FOLD, selector.STEP_RECORD_FORMAT_V2)


def verify_futility_decision(step_chain_path: Path, *, freeze1_sha256: str, amendment_tip: Path,
                             campaign_root: Path, smoke: bool = False, chain_receipt: Optional[Path] = None,
                             expected_chain: Optional[Sequence[str]] = None) -> dict[str, Any]:
    """The binding fold-0 futility decision (contract X7): ``{"decision", "record_sha256", "fit_chain",
    "scientific_identity_sha256", "pointer_sequence"}``. Any ``stop_futility`` pointer file wins; otherwise the
    chain's single FUT record must match a ``continue`` pointer of the same fit chain and be binding. Only binding
    records write pointer files and FUT step records (a smoke run writes neither), so a smoke chain never passes.
    The chain is authenticated by the selector (or its ``chain_receipt`` checked); a caller that authenticated it
    already passes the chain it got as ``expected_chain``."""
    from ec_clean30 import selector

    try:
        policy, _records, chain = _policy_and_chain(step_chain_path, freeze1_sha256, amendment_tip, smoke,
                                                    chain_receipt, expected_chain)
    except selector.SelectionInputError as exc:
        raise FutilityInputError(str(exc)) from exc
    tip = len(policy.get("amendment_tip_chain") or policy["freeze1_amendment_chain"])
    pointers = read_pointers(Path(campaign_root), tip)
    base = {"fit_chain": list(policy["freeze1_amendment_chain"]),
            "scientific_identity_sha256": policy.get("scientific_identity_sha256")}
    for sequence, pointer in sorted(pointers.items()):
        if pointer["decision"] == FUTILITY_STOP:
            return {**base, "decision": FUTILITY_STOP, "record_sha256": pointer["futility_record_sha256"],
                    "pointer_sequence": sequence}
    _require(selector.STEP_FUT in chain, "The fold-0 step chain holds no futility record.")
    payload, _sha256 = chain[selector.STEP_FUT]
    matching = [sequence for sequence, pointer in sorted(pointers.items())
                if pointer["futility_record_sha256"] == payload.get("futility_record_sha256")
                and pointer.get("fit_chain") == policy["freeze1_amendment_chain"]]
    _require(len(matching) == 1, "The chain's futility record has no matching pointer file of this fit chain.")
    _require(payload.get("decision") == pointers[matching[0]]["decision"],
             "The chain's futility decision differs from its pointer file.")
    _require(smoke or payload.get("binding") is True, "The fold-0 futility record is not binding.")
    return {**base, "decision": payload["decision"], "record_sha256": payload["futility_record_sha256"],
            "pointer_sequence": matching[0]}


def run_futility(
    *,
    fold: int,
    step_chain: Path,
    freeze1_sha256: str,
    plan_path: Path,
    runs_dirs: Sequence[Path],
    clean_r1_inner: Sequence[Path],
    clean_r2_inner: Sequence[Path],
    clean_sanity_record: Path,
    tables: Any,
    out: Path,
    freeze1_amendment_path: Optional[Path],
    smoke: bool = False,
    provenance: Optional[Mapping[str, Any]] = None,
    scorer: Optional[PopulationScorer] = None,
    clean_sanity_dispositions: Sequence[Path] = (),
    campaign_root: Optional[Path] = None,
    f6_corpus: Optional[Path] = None,
    r1_inner_lists: Sequence[Path] = (),
    chain_receipt: Optional[Path] = None,
) -> dict[str, Any]:
    """Rule A2-FUT on fold 0; writes the futility record ``out`` once, its pointer file and the FUT step record,
    and returns the record. ``campaign_root`` defaults to the plan's campaign root (``plans_v2/..``)."""
    from ec_clean30 import selector

    _require(fold == FUTILITY_FOLD, f"The futility rule runs on fold {FUTILITY_FOLD} only, not fold {fold}.")
    problems = binding_problems(f6_corpus=f6_corpus, r1_inner_lists=r1_inner_lists, scorer=scorer)
    _require(smoke or not problems or _TEST_ALLOW_NON_BINDING,
             "A real futility run must be able to bind (nothing was read or written): " + "; ".join(problems))
    plan_root = Path(plan_path).resolve().parent.parent
    if smoke:
        _require(campaign_root is not None and Path(campaign_root).resolve() != plan_root,
                 "A --smoke futility run needs its own --campaign-root, never the plan's campaign root.")
    try:
        freeze1_sha256 = selector._check_sha256(freeze1_sha256, "--freeze1-sha256")
        policy, chain_records, chain = _policy_and_chain(Path(step_chain), freeze1_sha256, freeze1_amendment_path,
                                                         smoke, chain_receipt)
        campaign_root = Path(campaign_root) if campaign_root is not None else plan_root
        tip = len(policy.get("amendment_tip_chain") or policy["freeze1_amendment_chain"])
        pointers = read_pointers(campaign_root, tip)
        _require(not (smoke and pointers), "A --smoke futility run refuses a campaign root that holds futility "
                                           "pointer files (real records).")
        _require(not any(pointer["decision"] == FUTILITY_STOP for pointer in pointers.values()),
                 "A stop_futility record exists in this campaign: S1, S2 and FUT are never rerun (A2-GOV-7(d)).")
        _require(not any(pointer.get("fit_chain") == list(policy["freeze1_amendment_chain"])
                         for pointer in pointers.values()),
                 "A futility record of this fit chain exists already: FUT runs once per fit chain (A2-FUT-1).")
        _require(tip not in pointers, f"The futility pointer file of chain sequence {tip} already exists (written once).")
        _require(selector.STEP_FUT not in chain, "The fold-0 step chain already holds a futility record.")
        out = Path(out)
        folder = tables.record.get("folder")
        selector._refuse_record_output(out, {fold: Path(folder)} if folder else {})
        _require(not out.exists(), f"--out {out.name} already exists (written once).")
        plan = selector.load_frozen_plan(plan_path, fold, plan_format=PLAN_FORMAT_V2)
        _require(STEP_S1 in chain and STEP_S2 in chain, "The futility rule needs the fold-0 S1 and S2 records.")
        s1_payload, s1_sha256 = chain[STEP_S1]
        s2_payload, s2_sha256 = chain[STEP_S2]
        for payload in (s1_payload, s2_payload):
            selector._check_chain_inputs(payload, plan_sha256=plan["plan_sha256"], freeze1_sha256=freeze1_sha256)
            _require(selector._components_binding(payload.get("components"))
                     == selector._components_binding(tables.record),
                     "--components differs from the inner tables of the fold-0 step records.")
        _require(s2_payload.get("s1_record_sha256") == s1_sha256, "The S2 record was built on another S1 record.")
        sanity = clean_sanity_binding(clean_sanity_record, clean_sanity_dispositions)
        bound = (s1_payload.get("best_inner_screen") or {}).get("clean_sanity_record_sha256")
        _require((smoke and bound is None) or bound == sanity["sha256"],
                 "The CLEAN held-out sanity record differs from the one the fold-0 S1 record binds"
                 + ("" if bound is not None else " (the S1 record binds none)") + ".")
        records = selector.read_runner_records(runs_dirs, fold=fold)
        selector._check_fit_policies(records, policy)
        configuration = selector.pool_selected_v2(s1_payload["selected"], plan, records, tables=tables,
                                                  freeze1_sha256=None if smoke else freeze1_sha256, smoke=smoke)
        selector._component_check(tables, configuration.pooled)
        _require(predictions_sha256(configuration.pooled) == s2_payload.get("s1_winner_predictions_sha256"),
                 "The selected S1 predictions differ from the ones the S2 record screened.")
        tau = float(s2_payload.get("adopted_tau", 0.0))
        r1 = selector._read_clean_arm(clean_r1_inner, "r1_inner")
        r2 = selector._read_clean_arm(clean_r2_inner, "r2_inner")
        check_clean_inputs(sanity, {"r1_inner": r1, "r2_inner": r2})
        exclude_ids = None
        if f6_corpus is not None or r1_inner_lists:
            _require(f6_corpus is not None and len(r1_inner_lists) == selection.N_INNER_FOLDS,
                     "The identical-sequence exclusion needs the fold-0 F6 corpus and the four R1-inner lists.")
            refuse_test_paths([f6_corpus, *r1_inner_lists], what="--f6-corpus/--r1-inner-list")
            lists = {}
            for path in r1_inner_lists:
                match = [inner for inner in range(selection.N_INNER_FOLDS)
                         if Path(path).name == f"r1_inner_exclusions_fold{fold}_inner{inner}.tsv"]
                _require(len(match) == 1 and match[0] not in lists, f"{Path(path).name} is not a fold-{fold} R1-inner "
                                                                    "exclusion list or repeats an inner fold.")
                lists[match[0]] = read_r1_inner_list(Path(path))
            exclude_ids = kept_identical_sequence_ids(tables.inner_fold_of or {}, read_corpus_sequences(Path(f6_corpus)),
                                                      lists)
        population_ids: list = []
        inputs, totals = build_projection_inputs(configuration, tau, r1, tables.component_of, exclude_ids=exclude_ids,
                                                 population_ids=population_ids)
        g4 = selector.score_gap_gate(configuration, r1, r2, after=STEP_S1)
        cgf = cgf_beside_clean(s1_payload, records, tables, r1, configuration, tau, population_ids)
    except selector.SelectionInputError as exc:
        raise FutilityInputError(str(exc)) from exc
    settings = _settings()
    if scorer is None:
        scorer = claim_engine_scorer(metric=settings["metric"], n_draws=settings["n_draws"])
    consequence = sanity.get("consequence") or {}
    if consequence.get("co_primaries_available") is False:
        # Choice C6b: a disposed R1-inner held-out arm leaves no co-primary available, so no replicate can succeed.
        decision_projection = {"n_replicates": FUTILITY_REPLICATES, "n_successes": 0, "projected_probability": 0.0,
                               "projection_run": False,
                               "reason": "the held-out CLEAN sanity consequence leaves no co-primary available "
                                         f"({consequence.get('rule_id')})"}
    else:
        decision_projection = project_success(inputs, scorer=scorer, n_replicates=settings["n_replicates"])
    decision = futility_decision(decision_projection["n_successes"], decision_projection["n_replicates"])
    descriptive_n = settings["n_descriptive_replicates"]
    descriptive = {
        "observed_difference": observed_difference(inputs, scorer),
        "conditional_power": project_success(inputs, scorer=scorer, n_replicates=descriptive_n,
                                             resample_components=False,
                                             stream=FUTILITY_DESCRIPTIVE_STREAMS["conditional_power"]),
        "sensitivity": {f"targets_divided_by_{divisor:g}": project_success(
            inputs, scorer=scorer, n_replicates=descriptive_n, divisor=divisor,
            stream=FUTILITY_DESCRIPTIVE_STREAMS[f"targets_divided_by_{divisor:g}"])
            for divisor in FUTILITY_SENSITIVITY_DIVISORS},
        "independent_draws": project_success(inputs, scorer=scorer, n_replicates=descriptive_n, nested=False,
                                             stream=FUTILITY_DESCRIPTIVE_STREAMS["independent_draws"]),
        "historical_rule_outcome": {"after": STEP_S1, **g4},
        "cgf_beside_clean": cgf,
        "decides": False,
    }
    overrides = dict(sorted(_TEST_OVERRIDES.items()))
    binding = not smoke and not binding_problems(f6_corpus=f6_corpus, r1_inner_lists=r1_inner_lists, scorer=scorer)
    record = {
        "format": FUTILITY_RECORD_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "fold": fold,
        "rule_id": FUTILITY_RULE_V2,
        "freeze1_sha256": freeze1_sha256,
        **policy,
        "policy": policy_record(),
        "plan_sha256": plan["plan_sha256"],
        "components": dict(tables.record),
        "s1_record_sha256": s1_sha256,
        "s2_record_sha256": s2_sha256,
        "step_chain_head_sha256": selector._chain_head(chain_records, freeze1_sha256),
        "selected": dict(s1_payload["selected"]),
        "adopted_tau": tau,
        "metric": PRIMARY_METRIC_V2,
        "claim_engine": dict(getattr(scorer, "settings", {"scorer": "injected"})),
        "clean_exports": {arm: {str(inner): {"npz_sha256": export.receipt.get("npz_sha256"),
                                             "seed": export.receipt.get("seed"),
                                             "checkpoint_sha256": export.receipt.get("checkpoint_sha256")}
                                for inner, export in sorted(exports.items())}
                          for arm, exports in (("r1_inner", r1), ("r2_inner", r2))},
        "clean_sanity": sanity,
        "population": totals,
        "projection": decision_projection,
        "decision": {
            "decision": decision,
            "n_successes": decision_projection["n_successes"],
            "n_replicates": decision_projection["n_replicates"],
            "min_successes_of_2000": FUTILITY_MIN_SUCCESSES,
            "projected_probability": decision_projection["projected_probability"],
            "binding": bool(binding),
            "wording": FUTILITY_STOP_WORDING if decision == FUTILITY_STOP else None,
            "stop_closes_campaign_without_test_scoring": decision == FUTILITY_STOP,
        },
        "descriptive": descriptive,
        "test_overrides": overrides,
        "smoke": bool(smoke),
        "provenance": dict(provenance) if provenance is not None else provenance_record(smoke=smoke),
    }
    _require(record["selection_policy_sha256"] == selection_policy_sha256_v2(),
             "The authenticated policy hash differs from the implemented amendment-2 policy.")
    record["futility_record_sha256"] = write_new_json(out, selection.to_jsonable(record))
    if not binding:
        # A smoke or test record is never bound: no pointer file and no FUT step record, so it can neither stop
        # nor block the campaign.
        record["pointer"] = None
        return record
    pointer = {"format": FUTILITY_POINTER_FORMAT, "schema_version": SCHEMA_VERSION, "campaign_id": CAMPAIGN_ID,
               "sequence": tip, "decision": decision, "binding": bool(binding),
               "futility_record_name": out.name, "futility_record_sha256": record["futility_record_sha256"],
               "fit_chain": list(policy["freeze1_amendment_chain"]),
               "amendment_tip_chain": list(policy.get("amendment_tip_chain") or policy["freeze1_amendment_chain"]),
               "step_chain_head_before_sha256": record["step_chain_head_sha256"]}
    path = pointer_path(campaign_root, tip)
    path.parent.mkdir(parents=True, exist_ok=True)
    pointer_sha256 = write_new_json(path, pointer)
    from ec_clean30 import selector

    step_payload = {
        "format": selector.STEP_RECORD_FORMAT_V2, "schema_version": selector.SCHEMA_VERSION_V2,
        "campaign_id": CAMPAIGN_ID, "fold": fold, "step": selector.STEP_FUT, "plan_sha256": plan["plan_sha256"],
        "freeze1_sha256": freeze1_sha256,
        **{key: value for key, value in policy.items() if key != "requires_futility_continue"},
        "smoke": bool(smoke), "components": dict(tables.record), "s1_record_sha256": s1_sha256,
        "s2_record_sha256": s2_sha256, "futility_record_sha256": record["futility_record_sha256"],
        "decision": decision, "binding": bool(binding), "pointer_sequence": tip, "pointer_sha256": pointer_sha256,
    }
    selection.append_step_record(Path(step_chain), selector.STEP_FUT, step_payload, anchor_sha256=freeze1_sha256)
    record["pointer"] = {"path": str(path), "sha256": pointer_sha256}
    return record


def frozen_scorer_settings() -> dict[str, Any]:
    """The claim-engine settings of a binding record (``claim_engine_scorer()`` with its defaults)."""
    return dict(claim_engine_scorer().settings)  # type: ignore[attr-defined]


def binding_problems(*, f6_corpus: Optional[Path], r1_inner_lists: Sequence[Path],
                     scorer: Optional[PopulationScorer]) -> list[str]:
    """Why a futility record could not be binding (empty: it can be)."""
    problems = []
    if f6_corpus is None or len(list(r1_inner_lists)) != selection.N_INNER_FOLDS:
        problems.append("the identical-sequence exclusion of rule A2-FUT-2 needs --f6-corpus and the four "
                        "--r1-inner-list files")
    if _TEST_OVERRIDES:
        problems.append(f"in-process test overrides are set ({sorted(_TEST_OVERRIDES)})")
    if scorer is not None and getattr(scorer, "settings", None) != frozen_scorer_settings():
        problems.append("the claim engine does not have its frozen settings")
    return problems


def cgf_beside_clean(s1_payload: Mapping[str, Any], records: Mapping[str, Any], tables: Any,
                     r1_exports: Mapping[int, Any], configuration: Any, tau: float,
                     population_ids: Sequence[str]) -> dict[str, Any]:
    """Descriptive (A2-FUT-8): C-GF beside CLEAN R1-inner and the selected configuration on the proteins of U.

    C-GF is re-pooled from the first F6 setting's control files named by the S1 record and must match the
    predictions hash the S1 record holds; totals and points only.
    """
    from ec_clean30 import selector

    control = s1_payload.get("whole_protein_control") or {}
    unit_ids = list(control.get("unit_ids") or [])
    _require(unit_ids and all(uid in records for uid in unit_ids),
             "The S1 record names no C-GF control fits, or a control fit is missing from --runs.")
    cgf, _controls = selector.pool_control(control.get("family", "C-GF"), {uid: records[uid] for uid in unit_ids},
                                           tables=tables, verifier=selector.FitVerifier())
    recorded = ((s1_payload.get("configurations") or {}).get(control.get("family", "C-GF")) or {})
    rows = selector.inner_closure_rows(cgf, r1_exports)
    keep = rows["keep"]
    cgf_predicted = cgf.pooled.logits.mean(axis=0).argmax(axis=-1)[keep]
    clean_predicted = rows["arms"][CLEAN_ARM]["predicted"]
    dm_rows = selector.inner_closure_rows(configuration, r1_exports)
    dm_predicted = dict(zip(dm_rows["ids"], deepmzyme_predictions(configuration.pooled, tau)[dm_rows["keep"]]))
    true = rows["true"]
    position = {pid: index for index, pid in enumerate(rows["ids"])}
    in_u = [position[pid] for pid in population_ids if pid in position]
    _require(len(in_u) == len(population_ids), "C-GF does not cover every protein of U.")
    in_u = np.asarray(in_u, dtype=np.int64)

    def points(predicted: np.ndarray) -> float:
        return float(np.mean(predicted[in_u] == true[in_u])) * POINTS if in_u.size else float("nan")

    dm = np.array([dm_predicted[pid] for pid in rows["ids"]], dtype=np.int64)
    record = {"control": control.get("family", "C-GF"), "n_population": int(in_u.size),
              "predictions_sha256": predictions_sha256(cgf.pooled),
              "matches_s1_record": predictions_sha256(cgf.pooled) == recorded.get("predictions_sha256"),
              "cgf_points": points(cgf_predicted), "clean_r1_inner_points": points(clean_predicted),
              "deepmzyme_points": points(dm)}
    record["cgf_minus_clean_points"] = record["cgf_points"] - record["clean_r1_inner_points"]
    record["deepmzyme_minus_cgf_points"] = record["deepmzyme_points"] - record["cgf_points"]
    _require(record["matches_s1_record"], "The re-pooled C-GF predictions differ from the S1 record's hash.")
    return record


def synthetic_oc_inputs(margin_points: float, *, seed: int = OC_SEED) -> ProjectionInputs:
    """The declared synthetic U of the operating characteristics (``OC_DESIGN``) at a true margin."""
    sizes = OC_DESIGN["proteins_per_ec_class_1_6"]
    true = np.repeat(np.arange(len(sizes)), sizes).astype(np.int64)
    n = int(true.size)
    size = int(OC_DESIGN["component_size"])
    labels = [f"c{cls}_{index // size:04d}" for cls, count in enumerate(sizes) for index in range(count)]
    rng = np.random.default_rng(int(seed))
    clean_right = rng.random(n) < float(OC_DESIGN["clean_accuracy"])
    order = rng.permutation(n)
    extra = int(round(float(margin_points) * n / POINTS))
    wrong_rows = [row for row in order.tolist() if not clean_right[row]]
    _require(extra <= len(wrong_rows), "The synthetic margin exceeds the CLEAN errors of the synthetic U.")
    dm_right = clean_right.copy()
    dm_right[wrong_rows[:extra]] = True
    wrong = (true + 1) % len(sizes)
    return projection_inputs(true, np.where(dm_right, true, wrong), np.where(clean_right, true, wrong), labels)


def operating_characteristics(*, out: Optional[Path] = None, n_replicates: int = OC_REPLICATES,
                              margins: Sequence[float] = OC_MARGINS_POINTS,
                              scorer: Optional[PopulationScorer] = None,
                              provenance: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Synthetic operating characteristics of the futility rule (descriptive; label-free; deterministic).

    For each true margin the declared synthetic U (``synthetic_oc_inputs``) is projected with ``n_replicates``
    replicates of the decision projection; the record gives the projected success probability, its Wilson
    interval and the decision the rule would take at that probability. ``out`` writes ``futility_oc.json`` once.
    """
    scorer = scorer if scorer is not None else claim_engine_scorer()
    rows = []
    for margin in margins:
        inputs = synthetic_oc_inputs(margin)
        summary = project_success(inputs, scorer=scorer, n_replicates=int(n_replicates))
        rows.append({"margin_points": float(margin), "n_population": inputs.n_proteins,
                     "n_components": inputs.n_components, "n_replicates": summary["n_replicates"],
                     "n_successes": summary["n_successes"], "projected_probability": summary["projected_probability"],
                     "wilson95": summary["projected_probability_wilson95"],
                     "decision_at_this_probability": futility_decision(summary["n_successes"],
                                                                       summary["n_replicates"]),
                     "per_population": {name: {key: value[key] for key in ("probability", "n_unavailable")}
                                        for name, value in summary["per_population"].items()}})
    record = {"format": OC_FORMAT, "schema_version": SCHEMA_VERSION, "campaign_id": CAMPAIGN_ID,
              "rule_id": FUTILITY_RULE_V2, "design": {**dict(OC_DESIGN), "seed": OC_SEED,
                                                       "proteins_per_ec_class_1_6":
                                                           list(OC_DESIGN["proteins_per_ec_class_1_6"])},
              "margins_points": [float(value) for value in margins], "n_replicates_per_margin": int(n_replicates),
              "claim_engine": dict(getattr(scorer, "settings", {"scorer": "injected"})),
              "targets": dict(FUTILITY_TARGET_SIZES), "rows": rows, "descriptive": True, "decides": False,
              "synthetic": True}
    if out is not None:
        refuse_test_paths([Path(out)], what="--out")
        record["provenance"] = dict(provenance) if provenance is not None else provenance_record(smoke=False)
        record["sha256"] = write_new_json(Path(out), selection.to_jsonable(record))
    return record


def action_futility_oc(args: argparse.Namespace) -> int:
    """The ``futility-oc`` action: write ``futility_oc.json`` once (WP9 step 7); prints the probability per margin."""
    try:
        git = real_action_git_state(smoke=False)
        record = operating_characteristics(out=Path(args.out), provenance=provenance_record(smoke=False, git=git))
    except (FutilityInputError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"sha256": record["sha256"], "rows": [{key: row[key] for key in (
        "margin_points", "projected_probability", "decision_at_this_probability")} for row in record["rows"]]},
        indent=2, sort_keys=True))
    return 0


def futility_allows_continuation(record: Mapping[str, Any]) -> bool:
    """True only for a binding, non-smoke fold-0 futility record of this format whose decision is continue."""
    decision = record.get("decision") or {}
    return (record.get("format") == FUTILITY_RECORD_FORMAT and record.get("fold") == FUTILITY_FOLD
            and record.get("rule_id") == FUTILITY_RULE_V2 and record.get("smoke") is False
            and not record.get("test_overrides") and decision.get("binding") is True
            and decision.get("decision") == FUTILITY_CONTINUE)


# ---------------------------------------------------------------------------
# Runner action


def add_futility_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The ``futility`` action's arguments (see the module docstring); the handler is ``action_futility``."""
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--step-chain", required=True, help="The fold-0 step-record chain (S1 and S2 of amendment 2).")
    parser.add_argument("--plan", required=True, help="The frozen plan <campaign root>/plans_v2/fold0.json.")
    parser.add_argument("--runs", action="append", required=True, help="Runs folder of the fold's fits (repeatable).")
    parser.add_argument("--components", required=True, help="The data overlay's certified inner identity copy.")
    parser.add_argument("--freeze1-sha256", required=True, help="Freeze-1 SHA-256, the step chain's anchor.")
    parser.add_argument("--freeze1-amendment", required=True, help="The authenticated amendment-2 record.")
    parser.add_argument("--clean-r1-inner", action="append", required=True,
                        help="CLEAN R1-inner C4 validation export (.npz), one per inner fold (repeatable).")
    parser.add_argument("--clean-r2-inner", action="append", required=True,
                        help="CLEAN R2-inner C4 validation export (.npz), one per inner fold (repeatable).")
    parser.add_argument("--clean-sanity-record", required=True,
                        help="The fold-0 CLEAN held-out sanity record (R1-inner and R2-inner; passed or disposed).")
    parser.add_argument("--clean-sanity-disposition", action="append", default=[],
                        help="The maintainer's disposition of a failed held-out sanity record.")
    parser.add_argument("--campaign-root", help="Campaign root of the pointer files (default: the plan's).")
    parser.add_argument("--f6-corpus", help="The fold-0 F6 corpus (identical-sequence exclusion of rule A2-FUT-2).")
    parser.add_argument("--r1-inner-list", action="append", default=[],
                        help="The fold-0 R1-inner exclusion lists, one per inner fold (repeatable).")
    parser.add_argument("--out", required=True, help="Futility record (JSON, written once).")
    parser.add_argument("--amendment-chain-receipt",
                        help="The verify-amendment-chain receipt, checked instead of verifying the chain again.")
    parser.add_argument("--smoke", action="store_true", help="Engineering use: allow a dirty tree and smoke fits.")
    return parser


def action_futility(args: argparse.Namespace) -> int:
    """The ``futility`` action: refusals before reading, the clean-tree check (real runs), the record and a summary
    of the decision and population totals. Returns 0 after writing the record and 2 when the inputs are refused."""
    from ec_clean30 import selector

    for name in ("step_chain", "plan", "out", "components", "freeze1_amendment", "clean_sanity_record"):
        refuse_test_paths([getattr(args, name)], what=f"--{name.replace('_', '-')}")
    for name in ("campaign_root", "f6_corpus", "amendment_chain_receipt"):
        if getattr(args, name, None):
            refuse_test_paths([getattr(args, name)], what=f"--{name.replace('_', '-')}")
    refuse_test_paths([*getattr(args, "clean_sanity_disposition", []), *getattr(args, "r1_inner_list", [])],
                      what="--clean-sanity-disposition/--r1-inner-list")
    refuse_test_paths(args.runs, what="--runs")
    refuse_test_paths([*args.clean_r1_inner, *args.clean_r2_inner], what="--clean-r1-inner/--clean-r2-inner")
    try:
        git = real_action_git_state(smoke=args.smoke)
        tables = selector.load_inner_tables(args.fold, Path(args.components), smoke=args.smoke)
        record = run_futility(fold=args.fold, step_chain=Path(args.step_chain), freeze1_sha256=args.freeze1_sha256,
                              plan_path=Path(args.plan), runs_dirs=[Path(path) for path in args.runs],
                              clean_r1_inner=[Path(path) for path in args.clean_r1_inner],
                              clean_r2_inner=[Path(path) for path in args.clean_r2_inner],
                              clean_sanity_record=Path(args.clean_sanity_record), tables=tables, out=Path(args.out),
                              freeze1_amendment_path=Path(args.freeze1_amendment), smoke=args.smoke,
                              provenance=provenance_record(smoke=args.smoke, git=git),
                              clean_sanity_dispositions=[Path(path) for path in
                                                         getattr(args, "clean_sanity_disposition", [])],
                              campaign_root=Path(args.campaign_root) if getattr(args, "campaign_root", None) else None,
                              f6_corpus=Path(args.f6_corpus) if getattr(args, "f6_corpus", None) else None,
                              r1_inner_lists=[Path(path) for path in getattr(args, "r1_inner_list", [])],
                              chain_receipt=(Path(args.amendment_chain_receipt)
                                             if getattr(args, "amendment_chain_receipt", None) else None))
    except (FutilityInputError, selector.SelectionInputError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    totals = {key: record["population"][key] for key in PRINTED_TOTALS}
    print(json.dumps({"decision": record["decision"]["decision"], "binding": record["decision"]["binding"],
                      "n_successes": record["decision"]["n_successes"],
                      "n_replicates": record["decision"]["n_replicates"],
                      "projected_probability": record["decision"]["projected_probability"], "totals": totals,
                      "futility_record_sha256": record["futility_record_sha256"]}, indent=2, sort_keys=True))
    return 0
