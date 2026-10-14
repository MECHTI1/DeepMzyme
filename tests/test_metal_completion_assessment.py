"""Synthetic decision-boundary tests; no campaign outputs or held-out files opened."""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import metal_completion_assessment as assess
import metal_completion_campaign as campaign


def metric(labels, recalls, support):
    return {"balanced_accuracy": math.fsum(recalls) / len(labels),
            "recall": dict(zip(labels, recalls)), "support": dict(zip(labels, support))}


def record(unit, shift=0.0):
    recall = 0.65 + unit.fold * 0.005 + shift
    common = metric(assess.COMMON4, [recall] * 4, [100, 100, 100, 300])
    support = {"four_class": [100, 100, 100, 300],
               "five_class": [100, 100, 100, 100, 200], "six_class": [100] * 6}[unit.target]
    native = metric(assess.NATIVE[unit.target], [recall] * len(support), support)
    return {"name": unit.name, "complete": True, "phase": unit.phase, "recipe": unit.recipe,
            "target": unit.target, "fold": unit.fold, "seed": unit.seed,
            "metrics": {"n": 600, "common4": common, "native": native}}


def grid(phase="discovery", recipe="B", shifts=None):
    shifts = shifts or {}
    units = campaign.discovery_units() if phase == "discovery" else campaign.confirmation_units(recipe)
    output = {}
    for unit in units:
        shift = shifts.get((unit.recipe, unit.target), 0.0)
        if callable(shift):
            shift = shift(unit.fold, unit.seed)
        elif isinstance(shift, (tuple, list)):
            shift = shift[unit.fold]
        output[unit.name] = record(unit, shift)
    return output


def matching(records, recipe, target, fold=None, seed=None):
    return [r for r in records.values() if r["recipe"] == recipe and r["target"] == target
            and (fold is None or r["fold"] == fold) and (seed is None or r["seed"] == seed)]


def set_recall(row, block, label, value):
    metric_block = row["metrics"][block]
    metric_block["recall"][label] = value
    metric_block["balanced_accuracy"] = math.fsum(metric_block["recall"].values()) / len(metric_block["recall"])
    if row["target"] == "four_class":
        row["metrics"]["native" if block == "common4" else "common4"] = copy.deepcopy(metric_block)


def test_discovery_no_gain_retains_shared_baseline_and_is_descriptive():
    result = assess.discover(grid())
    assert result["shared_recipe"] == "B"
    assert result["unit_count"] == 45
    assert not result["independent_validation"] and not result["improvement_claim_supported"]
    assert "not independent validation" in result["evidence_boundary"]
    assert len(result["configurations"]) == 9


def test_discovery_one_target_qualifies_the_shared_recipe_without_target_specific_selection():
    result = assess.discover(grid(shifts={("S", "six_class"): 0.025}))
    assert result["shared_recipe"] == "S"
    assert result["recipes"]["S"]["targets"]["six_class"]["qualifies"]
    assert not result["recipes"]["S"]["targets"]["four_class"]["qualifies"]


def test_discovery_fold0_alone_cannot_qualify_and_boundary_gain_is_insufficient():
    result = assess.discover(grid(shifts={("P", "four_class"): [0.05, -0.001, -0.001, -0.001, -0.001],
                                           ("S", "four_class"): 0.002}))
    assert result["shared_recipe"] == "B"
    p = result["recipes"]["P"]["targets"]["four_class"]
    assert p["mean_gain"] > 0.002 and p["folds_1_4_mean_gain"] < 0


def test_discovery_largest_qualifying_gain_is_primary_and_nonqualifier_is_excluded():
    records = grid(shifts={("P", "four_class"): 0.02, ("S", "six_class"): 0.04,
                          ("P", "five_class"): [0.4, -0.001, -0.001, -0.001, -0.001]})
    # Keep the deliberately large fold-0 metric within [0,1].
    for row in matching(records, "P", "five_class", 0):
        for label in assess.COMMON4:
            set_recall(row, "common4", label, 0.99)
        for label in assess.NATIVE["five_class"]:
            set_recall(row, "native", label, 0.99)
    result = assess.discover(records)
    assert result["shared_recipe"] == "S"
    assert result["recipes"]["P"]["largest_qualifying_mean_gain"] == pytest.approx(0.02)


def test_discovery_primary_tie_uses_all_target_gain_then_exact_tie_prefers_p():
    shifts = {("P", "four_class"): 0.02, ("S", "four_class"): 0.021,
              ("P", "five_class"): 0.01, ("P", "six_class"): 0.01}
    result = assess.discover(grid(shifts=shifts))
    assert result["primary_tie_recipes"] == ["P", "S"]
    assert result["shared_recipe"] == "P"
    tied = {(recipe, target): 0.01 for recipe in ("P", "S") for target in assess.TARGETS}
    assert assess.discover(grid(shifts=tied))["shared_recipe"] == "P"
    # Secondary ranking has no additional 0.2-point band.
    tied[("S", "six_class")] += 0.0001
    assert assess.discover(grid(shifts=tied))["shared_recipe"] == "S"


def test_discovery_zero_common_recall_in_one_fit_rejects_otherwise_large_gain():
    records = grid(shifts={("P", "five_class"): 0.12})
    set_recall(matching(records, "P", "five_class", 0)[0], "common4", "Mn", 0.0)
    result = assess.discover(records)
    checks = result["recipes"]["P"]["targets"]["five_class"]["recall_guard"]["checks"]
    assert not checks["common4_recall_nonzero_every_fit"]
    assert result["shared_recipe"] == "B"


def test_discovery_mean_recall_guard_has_inclusive_three_point_boundary():
    records = grid(shifts={("P", "four_class"): 0.03})
    for row in matching(records, "P", "four_class"):
        set_recall(row, "common4", "Mn", 0.65 + row["fold"] * 0.005 - 0.03)
    assert assess.discover(records)["shared_recipe"] == "P"
    for row in matching(records, "P", "four_class"):
        set_recall(row, "common4", "Mn", row["metrics"]["common4"]["recall"]["Mn"] - 0.00001)
    assert assess.discover(records)["shared_recipe"] == "B"


def test_baseline_confirmation_has_two_contrasts_and_retains_four_without_gain():
    result = assess.confirmation(grid("confirmation"), "B")
    assert result["family_size"] == 2 and len(result["contrasts"]) == 2
    assert result["unit_count"] == 30
    assert result["selection"]["recipe"] == "B" and result["selection"]["target"] == "four_class"
    assert not result["eligible"]


def test_confirmation_baseline_targets_use_common_four_and_ties_prefer_five():
    records = grid("confirmation", shifts={("B", "five_class"): 0.03, ("B", "six_class"): 0.031})
    for row in matching(records, "B", "six_class"):
        for label in assess.NATIVE["six_class"]:
            set_recall(row, "native", label, 0.99)
    result = assess.confirmation(records, "B")
    assert result["selection"]["target"] == "five_class"
    assert len(result["eligible"]) == 2
    assert "selection convention" in result["selection"]["reason"]


def test_nonbaseline_confirmation_has_five_fixed_contrasts_and_can_select_six():
    shifts = {("P", "four_class"): 0.02, ("P", "five_class"): 0.04, ("P", "six_class"): 0.06}
    result = assess.confirmation(grid("confirmation", "P", shifts), "P")
    assert result["unit_count"] == 40
    assert result["family_size"] == 5 and len(result["contrasts"]) == 5
    assert result["selection"]["target"] == "six_class"
    assert len(result["eligible"]) == 3
    for contrast in result["contrasts"].values():
        intervals = contrast["intervals"]
        assert intervals["adjusted"]["n_contrasts"] == 5
        assert intervals["adjusted"]["confidence"] == pytest.approx(0.99)
        assert intervals["bootstrap_resamples"] == 10000 and intervals["bootstrap_seed"] == 42
        assert intervals["n_folds"] == 5 and intervals["t_degrees_of_freedom"] == 4


def test_finer_target_must_beat_matched_direct_four_even_when_anchor_gain_is_clear():
    shifts = {("S", "four_class"): 0.03, ("S", "five_class"): 0.031, ("S", "six_class"): 0.028}
    result = assess.confirmation(grid("confirmation", "S", shifts), "S")
    assert result["eligible"] == ["S__four_class"]
    assert result["selection"]["target"] == "four_class"
    assert result["contrasts"]["S__five_class_vs_B__four_class"]["passed"]
    assert not result["contrasts"]["S__five_class_vs_S__four_class"]["passed"]


def test_finer_target_can_pass_when_shared_recipe_direct_four_does_not_qualify():
    shifts = {("P", "four_class"): 0.001, ("P", "five_class"): 0.025}
    result = assess.confirmation(grid("confirmation", "P", shifts), "P")
    assert result["eligible"] == ["P__five_class"]
    assert result["selection"]["target"] == "five_class"


def test_seed_differences_are_paired_then_averaged_within_fold_not_ten_independent_samples():
    shifts = {("B", "five_class"): lambda fold, seed: (0.06 if seed == 43 else -0.02) + fold * 0.001}
    result = assess.confirmation(grid("confirmation", shifts=shifts), "B")
    contrast = result["contrasts"]["B__five_class_vs_B__four_class"]
    assert contrast["intervals"]["fold_differences"] == pytest.approx([0.02, 0.021, 0.022, 0.023, 0.024])
    assert contrast["intervals"]["n_folds"] == 5 and contrast["intervals"]["t_degrees_of_freedom"] == 4
    assert result["selection"]["target"] == "five_class"
    assert not result["independent_validation"]


def test_confirmation_requires_both_intervals_positive_and_minimum_gain():
    shifts = {("B", "five_class"): [0.001, 0.001, 0.001, 0.001, 0.05], ("B", "six_class"): 0.002}
    result = assess.confirmation(grid("confirmation", shifts=shifts), "B")
    five = result["contrasts"]["B__five_class_vs_B__four_class"]
    assert five["checks"]["adjusted_bootstrap_lower_positive"]
    assert not five["checks"]["adjusted_t_lower_positive"]
    assert not result["eligible"]


def test_native_zero_guard_is_aggregate_not_a_per_fit_requirement():
    records = grid("confirmation", shifts={("B", "five_class"): 0.03})
    set_recall(matching(records, "B", "five_class", 0, 43)[0], "native", "Fe", 0.0)
    result = assess.confirmation(records, "B")
    assert result["selection"]["target"] == "five_class"
    for row in matching(records, "B", "five_class"):
        set_recall(row, "native", "Fe", 0.0)
    result = assess.confirmation(records, "B")
    assert result["selection"]["target"] == "four_class"
    guard = result["contrasts"]["B__five_class_vs_B__four_class"]["recall_guard"]
    assert guard["native_zero_mean_recalls"] == ["Fe"]


def test_common_zero_guard_is_per_fit_even_when_seed_mean_is_positive():
    records = grid("confirmation", shifts={("B", "five_class"): 0.12})
    set_recall(matching(records, "B", "five_class", 0, 43)[0], "common4", "Mn", 0.0)
    result = assess.confirmation(records, "B")
    assert not result["contrasts"]["B__five_class_vs_B__four_class"]["recall_guard"]["passed"]
    assert result["selection"]["target"] == "four_class"


@pytest.mark.parametrize("change, message", [
    (lambda row: row.update(complete=False), "not complete"),
    (lambda row: row.update(complete=1), "not complete"),
    (lambda row: row.update(name="other"), "name differs"),
    (lambda row: row.update(fold=True), "expected integer"),
    (lambda row: row.update(seed=43), "seed differs"),
    (lambda row: row.update(target="six_class"), "target differs"),
    (lambda row: row["metrics"]["common4"].update(balanced_accuracy=float("nan")), "finite"),
    (lambda row: row["metrics"]["common4"]["recall"].update(Mn=float("inf")), "finite"),
    (lambda row: row["metrics"]["common4"]["recall"].update(Mn=None), "expected a number"),
    (lambda row: row["metrics"]["common4"]["recall"].update(Mn=-0.01), "finite"),
    (lambda row: row["metrics"]["common4"]["recall"].update(Mn=True), "expected a number"),
    (lambda row: row["metrics"]["common4"]["support"].update(Mn=0), "support"),
    (lambda row: row["metrics"]["common4"]["support"].update(Mn=1.5), "integer"),
    (lambda row: row["metrics"]["common4"]["support"].update(Mn=True), "integer"),
    (lambda row: row["metrics"]["common4"].update(balanced_accuracy=0.99), "mean recall"),
    (lambda row: row["metrics"]["common4"].update(min_recall=0.99), "min_recall differs"),
    (lambda row: row["metrics"].update(n=599), "n differs"),
    (lambda row: row["metrics"]["common4"]["support"].update(Mn=101), "native/common4 support"),
])
def test_malformed_metrics_or_identity_fail_closed(change, message):
    records = grid()
    change(next(iter(records.values())))
    with pytest.raises(ValueError, match=message):
        assess.discover(records)


def test_missing_extra_duplicate_and_seed42_confirmation_records_fail_closed():
    records = grid()
    records.pop(next(iter(records)))
    with pytest.raises(ValueError, match="missing="):
        assess.discover(records)
    rows = list(grid().values())
    with pytest.raises(ValueError, match="Duplicate"):
        assess.discover(rows + [copy.deepcopy(rows[0])])
    confirmation = grid("confirmation")
    extra = next(iter(grid().values()))
    confirmation[extra["name"]] = extra
    with pytest.raises(ValueError, match="unexpected="):
        assess.confirmation(confirmation, "B")
    with pytest.raises(ValueError, match="shared_recipe"):
        assess.confirmation({}, "other")


def test_matched_fold_support_mismatch_is_rejected_even_if_internal_totals_agree():
    records = grid()
    row = matching(records, "P", "four_class", 0)[0]
    for block in ("common4", "native"):
        row["metrics"][block]["support"]["Mn"] += 1
    row["metrics"]["n"] += 1
    with pytest.raises(ValueError, match="support differs across matched"):
        assess.discover(records)


def test_five_six_native_support_must_agree_beyond_collapsed_totals():
    records = grid()
    for row in matching(records, "B", "five_class") + matching(records, "P", "five_class") + matching(records, "S", "five_class"):
        row["metrics"]["native"]["support"]["Fe"] += 1
        row["metrics"]["native"]["support"]["Class VIII"] -= 1
    with pytest.raises(ValueError, match="five/six native support differs"):
        assess.discover(records)


def test_optional_confusion_matrix_must_match_recalls_and_support():
    records = grid()
    row = next(iter(records.values()))
    row["metrics"]["common4"]["confusion_matrix"] = [[100, 0, 0, 0], [0, 100, 0, 0],
                                                       [0, 0, 100, 0], [0, 0, 0, 300]]
    with pytest.raises(ValueError, match="confusion_matrix recall differs"):
        assess.discover(records)


def test_list_and_mapping_input_agree_and_inputs_are_unchanged():
    records = grid(shifts={("S", "five_class"): 0.02})
    before = copy.deepcopy(records)
    mapping_result = assess.discover(records)
    assert assess.discover(list(records.values())) == mapping_result
    assert records == before
    assert assess.discover(records) == mapping_result
