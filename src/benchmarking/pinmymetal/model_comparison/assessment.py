"""Pure validation-only decisions for the bounded metal completion campaign.

The collector must verify run/checkpoint/prediction identities before calling these
functions. This module validates complete metric grids; it never reads predictions,
opens held-out data, starts a fit, or grants promotion/refit authority. Additional
optimizer seeds on the existing development folds are not independent validation.
"""

from __future__ import annotations


import sys as _sys
from pathlib import Path as _Path
for _p in (str(_Path(__file__).resolve().parents[4]), str(_Path(__file__).resolve().parents[4] / "src")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
from project_paths import RELOCATED_FILES, install_relocated_module_aliases, resolve_existing_path, resolve_repo_file
install_relocated_module_aliases()
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import metal_completion_campaign as campaign
from pmm_v3_assessment import COMMON4, NATIVE, paired_intervals

TARGETS = ("four_class", "five_class", "six_class")
RECIPES = ("B", "P", "S")
MIN_GAIN = 0.002
RECALL_DROP_LIMIT = 0.03
TIE_BAND = 0.002
FLOAT_TOLERANCE = 1e-12
METRIC_TOLERANCE = 1e-9
INTERVAL_SPEC = {"confidence": 0.95, "bootstrap_resamples": 10000, "bootstrap_seed": 42}
EVIDENCE_BOUNDARY = (
    "Validation-only selection on reused development folds; extra optimizer seeds "
    "are not independent validation or independent evaluation. No final-refit, "
    "promotion or held-out evaluation authority follows from this assessment."
)
# Cross-target ranking uses one metric only; native five/six metrics stay descriptive.
RANKING_METRIC = "probability_summed_common_four_balanced_accuracy"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _probability(value: Any, label: str) -> float:
    _require(isinstance(value, Real) and not isinstance(value, bool), f"{label}: expected a number")
    value = float(value)
    _require(math.isfinite(value) and 0 <= value <= 1, f"{label}: expected finite value in [0, 1]")
    return value


def _integer(value: Any, label: str, *, positive: bool = False) -> int:
    _require(isinstance(value, Integral) and not isinstance(value, bool), f"{label}: expected integer")
    _require(value > 0 if positive else value >= 0, f"{label}: invalid integer support/index")
    return int(value)


def _metric(block: Any, labels: tuple[str, ...], context: str) -> dict[str, Any]:
    _require(isinstance(block, Mapping), f"{context}: missing metric block")
    recall, support = block.get("recall"), block.get("support")
    for name, values in (("recall", recall), ("support", support)):
        _require(isinstance(values, Mapping) and set(values) == set(labels),
                 f"{context}: {name} labels differ from {labels}")
    recall = {label: _probability(recall[label], f"{context}.recall.{label}") for label in labels}
    support = {label: _integer(support[label], f"{context}.support.{label}", positive=True)
               for label in labels}
    ba = _probability(block.get("balanced_accuracy"), f"{context}.balanced_accuracy")
    _require(abs(ba - math.fsum(recall.values()) / len(labels)) <= METRIC_TOLERANCE,
             f"{context}: balanced_accuracy differs from mean recall")
    if "min_recall" in block:
        minimum = _probability(block["min_recall"], f"{context}.min_recall")
        _require(abs(minimum - min(recall.values())) <= METRIC_TOLERANCE,
                 f"{context}: min_recall differs from recalls")
    result = {"balanced_accuracy": ba, "recall": recall, "support": support,
              "min_recall": min(recall.values())}
    if "confusion_matrix" in block:
        matrix = block["confusion_matrix"]
        _require(isinstance(matrix, (list, tuple)) and len(matrix) == len(labels)
                 and all(isinstance(row, (list, tuple)) and len(row) == len(labels) for row in matrix),
                 f"{context}: invalid confusion_matrix shape")
        matrix = [[_integer(value, f"{context}.confusion_matrix") for value in row] for row in matrix]
        for index, label in enumerate(labels):
            _require(sum(matrix[index]) == support[label], f"{context}: confusion_matrix support differs")
            _require(abs(matrix[index][index] / support[label] - recall[label]) <= METRIC_TOLERANCE,
                     f"{context}: confusion_matrix recall differs")
        result["confusion_matrix"] = matrix
    return result


def _normalize(records: Mapping[str, Any] | Sequence[Mapping[str, Any]], units) -> dict[str, dict[str, Any]]:
    """Require the exact expected grid and compatible class supports on each fold."""
    expected = {unit.name: unit for unit in units}
    if isinstance(records, Mapping):
        items = list(records.items())
    else:
        _require(isinstance(records, Sequence) and not isinstance(records, (str, bytes)),
                 "records must be a mapping or a sequence")
        _require(all(isinstance(record, Mapping) and isinstance(record.get("name"), str) for record in records),
                 "list records require a name")
        items = [(record["name"], record) for record in records]
    names = [name for name, _ in items]
    _require(all(isinstance(name, str) for name in names), "Unit record keys must be names")
    _require(len(set(names)) == len(names), "Duplicate unit records")
    missing, extra = sorted(set(expected) - set(names)), sorted(set(names) - set(expected))
    _require(not missing and not extra, f"Incomplete or mismatched grid; missing={missing}; unexpected={extra}")
    result: dict[str, dict[str, Any]] = {}
    fold_common_support: dict[int, dict[str, int]] = {}
    fold_native_support: dict[tuple[int, str], dict[str, int]] = {}
    for name, record in items:
        _require(isinstance(record, Mapping), f"{name}: invalid record")
        _require(record.get("complete") is True, f"{name}: unit is not complete")
        _require(record.get("name", name) == name, f"{name}: record name differs from key")
        unit = expected[name]
        for field in ("phase", "recipe", "target", "fold", "seed"):
            value = record.get(field)
            if field in ("fold", "seed"):
                value = _integer(value, f"{name}.{field}")
            _require(value == getattr(unit, field), f"{name}: {field} differs from expected unit")
        metrics = record.get("metrics")
        _require(isinstance(metrics, Mapping), f"{name}: missing metrics")
        common = _metric(metrics.get("common4"), COMMON4, f"{name}.common4")
        native = _metric(metrics.get("native"), NATIVE[unit.target], f"{name}.native")
        native_support = native["support"]
        collapsed_support = {label: native_support[label] for label in COMMON4[:3]}
        collapsed_support["Class VIII"] = sum(native_support[label] for label in NATIVE[unit.target][3:])
        _require(common["support"] == collapsed_support, f"{name}: native/common4 support differs")
        n = sum(common["support"].values())
        if "n" in metrics:
            _require(_integer(metrics["n"], f"{name}.n", positive=True) == n, f"{name}: n differs from support")
        if unit.target == "four_class":
            _require(all(abs(common["recall"][label] - native["recall"][label]) <= METRIC_TOLERANCE
                         for label in COMMON4), f"{name}: direct-four native/common4 recalls differ")
        _require(fold_common_support.setdefault(unit.fold, common["support"]) == common["support"],
                 f"{name}: common4 support differs across matched fold units")
        key = (unit.fold, unit.target)
        _require(fold_native_support.setdefault(key, native_support) == native_support,
                 f"{name}: native support differs across matched fold units")
        result[name] = {"name": name, **{field: getattr(unit, field) for field in
                                       ("phase", "recipe", "target", "fold", "seed")},
                        "complete": True, "metrics": {"n": n, "common4": common, "native": native}}
    # Five-class and six-class supports must represent the same frozen native ions.
    for fold in range(5):
        five = fold_native_support[(fold, "five_class")]
        six = fold_native_support[(fold, "six_class")]
        _require(five["Fe"] == six["Fe"] and five["Class VIII"] == six["Co"] + six["Ni"],
                 f"Fold {fold}: five/six native support differs")
    return result


def _mean(values) -> float:
    values = list(values)
    return math.fsum(values) / len(values)


def _config_name(recipe: str, target: str) -> str:
    return f"{recipe}__{target}"


def _configuration(records, recipe: str, target: str) -> dict[str, Any]:
    fits = sorted((r for r in records.values() if r["recipe"] == recipe and r["target"] == target),
                  key=lambda r: (r["fold"], r["seed"]))
    folds = []
    for fold in range(5):
        matched = [r for r in fits if r["fold"] == fold]
        folds.append({"fold": fold, "seeds": [r["seed"] for r in matched], **{
            block: {"balanced_accuracy": _mean(r["metrics"][block]["balanced_accuracy"] for r in matched),
                    "recall": {label: _mean(r["metrics"][block]["recall"][label] for r in matched)
                               for label in (COMMON4 if block == "common4" else NATIVE[target])}}
            for block in ("common4", "native")}})
    return {"config": _config_name(recipe, target), "recipe": recipe, "target": target,
            "mean_common4_balanced_accuracy": _mean(f["common4"]["balanced_accuracy"] for f in folds),
            "mean_native_balanced_accuracy": _mean(f["native"]["balanced_accuracy"] for f in folds),
            "mean_common4_recall": {label: _mean(f["common4"]["recall"][label] for f in folds) for label in COMMON4},
            "mean_native_recall": {label: _mean(f["native"]["recall"][label] for f in folds) for label in NATIVE[target]},
            "by_seed": {str(seed): {"common4_balanced_accuracy": _mean(r["metrics"]["common4"]["balanced_accuracy"]
                                                                             for r in fits if r["seed"] == seed),
                                    "native_balanced_accuracy": _mean(r["metrics"]["native"]["balanced_accuracy"]
                                                                            for r in fits if r["seed"] == seed)}
                        for seed in sorted({r["seed"] for r in fits})},
            "folds": folds, "fits": fits}


def _guard(candidate, control) -> dict[str, Any]:
    zero = [{"name": fit["name"], "class": label} for fit in candidate["fits"] for label in COMMON4
            if fit["metrics"]["common4"]["recall"][label] <= 0]
    change = {label: candidate["mean_common4_recall"][label] - control["mean_common4_recall"][label]
              for label in COMMON4}
    native_zero = [label for label, value in candidate["mean_native_recall"].items() if value <= 0]
    checks = {"common4_recall_nonzero_every_fit": not zero,
              "mean_common4_recall_drop_at_most_3pp": all(value >= -RECALL_DROP_LIMIT - FLOAT_TOLERANCE
                                                          for value in change.values()),
              "native_mean_recall_nonzero": not native_zero}
    return {"passed": all(checks.values()), "checks": checks, "common4_zero_recalls": zero,
            "common4_mean_recall_changes": change, "native_zero_mean_recalls": native_zero}


def discover(records: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Screen the exact 45 seed-42 units and freeze one shared B/P/S recipe.

    P/S qualify with at least one guarded target gain >0.2 points over that
    target's B control and positive mean gain on folds 1–4. Rank their largest
    qualifying gains, tie within 0.2 points, then mean gain over all targets,
    then P before S. This screen makes no interval-supported improvement claim.
    """
    records = _normalize(records, campaign.discovery_units())
    configs = {_config_name(recipe, target): _configuration(records, recipe, target)
               for recipe in RECIPES for target in TARGETS}
    recipes = {}
    for recipe in ("P", "S"):
        targets = {}
        for target in TARGETS:
            candidate, control = configs[_config_name(recipe, target)], configs[_config_name("B", target)]
            differences = [a["common4"]["balanced_accuracy"] - b["common4"]["balanced_accuracy"]
                           for a, b in zip(candidate["folds"], control["folds"])]
            mean, other_folds = _mean(differences), _mean(differences[1:])
            guard = _guard(candidate, control)
            checks = {"all_fold_mean_gain_gt_0_2pp": mean > MIN_GAIN + FLOAT_TOLERANCE,
                      "folds_1_4_mean_gain_positive": other_folds > FLOAT_TOLERANCE,
                      "recall_guard": guard["passed"]}
            targets[target] = {"qualifies": all(checks.values()), "checks": checks,
                               "fold_differences": differences, "mean_gain": mean,
                               "fold0_gain": differences[0], "folds_1_4_mean_gain": other_folds,
                               "recall_guard": guard}
        qualifying = [row["mean_gain"] for row in targets.values() if row["qualifies"]]
        recipes[recipe] = {"eligible": bool(qualifying), "targets": targets,
                           "largest_qualifying_mean_gain": max(qualifying) if qualifying else None,
                           "mean_gain_all_targets": _mean(row["mean_gain"] for row in targets.values())}
    eligible = [recipe for recipe in ("P", "S") if recipes[recipe]["eligible"]]
    tied = []
    if not eligible:
        selected, reason = "B", "No P/S recipe has a qualifying target; retain B for all targets."
    else:
        best = max(recipes[recipe]["largest_qualifying_mean_gain"] for recipe in eligible)
        tied = [recipe for recipe in eligible if best - recipes[recipe]["largest_qualifying_mean_gain"]
                <= TIE_BAND + FLOAT_TOLERANCE]
        best_all = max(recipes[recipe]["mean_gain_all_targets"] for recipe in tied)
        selected = next(recipe for recipe in ("P", "S") if recipe in tied
                        and best_all - recipes[recipe]["mean_gain_all_targets"] <= FLOAT_TOLERANCE)
        reason = "Largest qualifying target gain; 0.2-point tie band, then all-target mean gain, then P before S."
    return {"phase": "discovery", "complete": True, "shared_recipe": selected, "reason": reason,
            "recipes": recipes, "eligible_recipes": eligible, "primary_tie_recipes": tied,
            "configurations": configs, "unit_count": len(records), "seeds": [42], "folds": list(range(5)),
            "ranking_metric": RANKING_METRIC,
            "tie_break_note": ("The all-target mean gain used inside the tie band averages every target's "
                               "paired gain, including targets that did not qualify, as the frozen rule states."),
            "evidence_boundary": EVIDENCE_BOUNDARY, "independent_validation": False,
            "improvement_claim_supported": False,
            "unsupported_claims": ["Independent validation", "Interval-supported recipe improvement",
                                   "Final-refit or test readiness"]}


def _contrast(candidate, control, family_size: int) -> dict[str, Any]:
    differences = [a["common4"]["balanced_accuracy"] - b["common4"]["balanced_accuracy"]
                   for a, b in zip(candidate["folds"], control["folds"])]
    intervals = paired_intervals(differences, INTERVAL_SPEC, n_contrasts=family_size)
    guard = _guard(candidate, control)
    checks = {"mean_gain_gt_0_2pp": intervals["mean_difference"] > MIN_GAIN + FLOAT_TOLERANCE,
              "adjusted_bootstrap_lower_positive": intervals["adjusted"]["bootstrap"][0] > 0,
              "adjusted_t_lower_positive": intervals["adjusted"]["t"][0] > 0,
              "recall_guard": guard["passed"]}
    return {"candidate": candidate["config"], "control": control["config"],
            "passed": all(checks.values()), "checks": checks, "recall_guard": guard,
            "intervals": intervals,
            "estimand": "Paired seed-43/44 differences averaged within each of five folds; n=5, df=4."}


def confirmation(records: Mapping[str, Any] | Sequence[Mapping[str, Any]], shared_recipe: str) -> dict[str, Any]:
    """Assess the exact 30/40 seed-43/44 units without any missing-grid fallback.

    For P/S, the five fixed contrasts are R4/R5/R6 versus B4 and R5/R6 versus
    R4. R5/R6 must pass both applicable contrasts; R4 need not itself qualify.
    For B, the two contrasts are B5/B6 versus B4. Keep B4 if none qualify;
    otherwise select by common-four BA, with the 0.2-point tie band preferring
    fewer classes. This is a selection convention, not a five-versus-six claim.
    """
    _require(shared_recipe in RECIPES, "shared_recipe must be B, P or S")
    records = _normalize(records, campaign.confirmation_units(shared_recipe))
    pairs = [(shared_recipe, target) for target in TARGETS]
    if shared_recipe != "B":
        pairs.append(("B", "four_class"))
    configs = {_config_name(recipe, target): _configuration(records, recipe, target) for recipe, target in pairs}
    anchor = _config_name("B", "four_class")
    direct = _config_name(shared_recipe, "four_class")
    contrast_pairs = [(key, anchor) for key in configs if key != anchor]
    if shared_recipe != "B":
        contrast_pairs.extend((_config_name(shared_recipe, target), direct) for target in TARGETS[1:])
    family_size = 2 if shared_recipe == "B" else 5
    _require(len(contrast_pairs) == family_size, "Internal contrast family differs from frozen design")
    contrasts = {f"{candidate}_vs_{control}": _contrast(configs[candidate], configs[control], family_size)
                 for candidate, control in contrast_pairs}
    candidates = {}
    for target in TARGETS:
        candidate = _config_name(shared_recipe, target)
        if candidate == anchor:
            continue
        required = [f"{candidate}_vs_{anchor}"]
        if shared_recipe != "B" and target != "four_class":
            required.append(f"{candidate}_vs_{direct}")
        candidates[candidate] = {"eligible": all(contrasts[key]["passed"] for key in required),
                                 "required_contrasts": required, "recipe": shared_recipe, "target": target,
                                 "mean_common4_balanced_accuracy": configs[candidate]["mean_common4_balanced_accuracy"]}
    eligible = [name for name, row in candidates.items() if row["eligible"]]
    tied = []
    if not eligible:
        selected, reason = anchor, "No challenger passes every required gate; retain unchanged B four-class."
    else:
        best = max(configs[name]["mean_common4_balanced_accuracy"] for name in eligible)
        tied = [name for name in eligible if best - configs[name]["mean_common4_balanced_accuracy"]
                <= TIE_BAND + FLOAT_TOLERANCE]
        selected = min(tied, key=lambda name: TARGETS.index(configs[name]["target"]))
        reason = "Best eligible common-four mean; within 0.2 points prefer fewer classes (selection convention)."
    return {"phase": "confirmation", "complete": True, "shared_recipe": shared_recipe,
            "selected": selected, "selection": {"config": selected, "recipe": configs[selected]["recipe"],
                                                  "target": configs[selected]["target"], "reason": reason},
            "eligible": eligible, "selection_ties": tied, "candidates": candidates,
            "contrasts": contrasts, "family_size": family_size, "configurations": configs,
            "unit_count": len(records), "seeds": [43, 44], "folds": list(range(5)),
            "ranking_metric": RANKING_METRIC,
            "evidence_boundary": EVIDENCE_BOUNDARY, "independent_validation": False,
            "target_contrast_scope": "Matched target contrasts conditional on the discovery-selected shared recipe.",
            "unsupported_claims": ["Independent validation", "Fresh recipe effect versus B five/six controls",
                                   "Five-class versus six-class superiority", "Final-refit or test readiness"]}
