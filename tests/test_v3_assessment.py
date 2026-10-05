"""Tests for the v3 assessor (pmm_v3_assessment.py) on synthetic metrics and one real tiny unit."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]

import pmm_v3_assessment as assess  # noqa: E402
import pmm_v3_campaign as v3  # noqa: E402
from test_v3_campaign import campaign, policy  # noqa: E402,F401  (fixture reuse)

SPEC = assess.SPEC_DEFAULTS
NOISE = (0.0, 0.004, -0.003, 0.002, -0.001)


# ---------------------------------------------------------------------------
# Synthetic units
# ---------------------------------------------------------------------------

def metrics(recalls4, native=None):
    recalls4 = [float(v) for v in recalls4]
    common = {"balanced_accuracy": float(np.mean(recalls4)), "min_recall": min(recalls4),
              "recall": dict(zip(assess.COMMON4, recalls4))}
    return {"common4": common, "native": {"recall": native if native is not None else dict(common["recall"])}}


def put(collected, family, target, recipe, recalls_by_fold, *, folds=range(5), seed=42, native=None):
    for fold, recalls in zip(folds, recalls_by_fold):
        unit = v3.Unit(family, target, recipe, fold, seed)
        native_recall = native
        if native_recall is None and target != "four_class":
            native_recall = dict(zip(assess.NATIVE[target], [recalls[0], recalls[1], recalls[2]]
                                     + [recalls[3]] * (len(assess.NATIVE[target]) - 3)))
        collected[unit.name] = {"name": unit.name, "status": "completed", "metrics": metrics(recalls, native_recall)}


def flat(base, shift=0.0, folds=range(5)):
    """Per-fold balanced recalls: every class equals base + shift + a fixed fold noise."""
    return [[base + shift + NOISE[k]] * 4 for k in folds]


def neutral_grid(shifts=None):
    """Baseline grid: four_class at 0.80; five/six shifted per (family, target)."""
    shifts = shifts or {}
    collected = {}
    for family in v3.FAMILIES:
        put(collected, family, "four_class", "baseline", flat(0.80))
        for target in ("five_class", "six_class"):
            put(collected, family, target, "baseline", flat(0.80, shifts.get((family, target), 0.0)))
    return collected


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------

def test_intervals_match_the_t_formula_and_agree_on_a_clear_gain():
    d = [0.01, 0.02, 0.03, 0.04, 0.05]
    result = assess.paired_intervals(d, SPEC)
    half = 2.7764451051977934 * np.std(d, ddof=1) / math.sqrt(5)
    assert result["t_degrees_of_freedom"] == 4
    assert result["unadjusted"]["t"] == pytest.approx([0.03 - half, 0.03 + half])
    assert result["unadjusted"]["outcome"] == "better" and result["unadjusted"]["methods_agree"]
    assert assess.paired_intervals([-v for v in d], SPEC)["unadjusted"]["outcome"] == "worse"


def test_bootstrap_and_t_disagreement_is_no_clear_difference():
    result = assess.paired_intervals([0.001, 0.001, 0.001, 0.001, 0.05], SPEC)["unadjusted"]
    assert result["bootstrap"][0] > 0 and result["t"][0] < 0
    assert result["outcome"] == "no clear difference" and not result["methods_agree"]


def test_bonferroni_adjustment_widens_both_intervals():
    result = assess.paired_intervals([0.004, 0.006, 0.008, 0.005, 0.007], SPEC, n_contrasts=6)
    plain, adjusted = result["unadjusted"], result["adjusted"]
    assert adjusted["confidence"] == pytest.approx(1 - 0.05 / 6)
    assert adjusted["t"][0] < plain["t"][0] and adjusted["bootstrap"][0] <= plain["bootstrap"][0]


def test_load_spec_is_refused_until_frozen(tmp_path):
    path = tmp_path / "spec.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="not frozen"):
        assess.load_spec(path, expected_sha256=None)
    with pytest.raises(ValueError, match="differs"):
        assess.load_spec(path, expected_sha256="0" * 64)


# ---------------------------------------------------------------------------
# Neutral target test
# ---------------------------------------------------------------------------

def test_a_target_tie_keeps_four_class():
    result = assess.neutral_target_test(neutral_grid(), SPEC)
    assert all(d["target"] == "four_class" for d in result["family_target_decisions"].values())
    assert all(c["verdict"] == "no clear difference" for c in result["contrasts"].values())
    assert result["claim_all_families_five_and_six_better"] is False


def test_supported_gains_pick_the_larger_target_and_ties_keep_five_class():
    result = assess.neutral_target_test(neutral_grid({
        ("only_esm", "five_class"): 0.02, ("only_esm", "six_class"): 0.04,
        ("only_gvp", "five_class"): 0.03, ("only_gvp", "six_class"): 0.031}), SPEC)
    decisions = result["family_target_decisions"]
    assert decisions["only_esm"]["target"] == "six_class"
    assert decisions["only_gvp"]["target"] == "five_class"  # 0.1 point apart: within the tie band
    assert decisions["gvp_late_fusion"]["target"] == "four_class"


def test_zero_native_recall_blocks_a_five_or_six_class_gain():
    collected = neutral_grid({("only_gvp", "six_class"): 0.03})
    native = {"Mn": 0.9, "Cu": 0.9, "Zn": 0.9, "Fe": 0.8, "Co": 0.0, "Ni": 0.7}
    put(collected, "only_gvp", "six_class", "baseline", flat(0.80, 0.03), native=native)
    result = assess.neutral_target_test(collected, SPEC)
    contrast = result["contrasts"]["only_gvp: six_class vs four_class"]
    assert contrast["unadjusted"]["outcome"] == "better" and not contrast["gates"]["no_zero_native_recall"]
    assert contrast["verdict"] == "blocked by a recall gate"
    assert result["family_target_decisions"]["only_gvp"]["target"] == "four_class"


def test_a_common_four_recall_drop_over_three_points_blocks():
    collected = neutral_grid()
    shifted = [[0.86 + NOISE[k], 0.86 + NOISE[k], 0.86 + NOISE[k], 0.76 + NOISE[k]] for k in range(5)]
    put(collected, "gvp_late_fusion", "five_class", "baseline", shifted)  # +2.5 BA, Class VIII -4 points
    contrast = assess.neutral_target_test(collected, SPEC)["contrasts"]["gvp_late_fusion: five_class vs four_class"]
    assert contrast["unadjusted"]["outcome"] == "better"
    assert contrast["class_recall_drop"]["Class VIII"] == pytest.approx(0.04)
    assert contrast["verdict"] == "blocked by a recall gate"


def test_an_incomplete_grid_is_reported_and_blocks_selection():
    collected = neutral_grid()
    del collected[v3.Unit("only_esm", "six_class", "baseline", 3, 42).name]
    collected[v3.Unit("only_gvp", "five_class", "baseline", 2, 42).name] = {
        "name": "x", "status": "failed", "final_failure": False}
    result = assess.assess_step_e(collected, SPEC)
    neutral = result["neutral_target_test"]
    assert neutral["contrasts"]["only_esm: six_class vs four_class"]["status"] == "incomplete"
    assert neutral["family_target_decisions"]["only_esm"]["target"] is None
    assert neutral["family_target_decisions"]["only_gvp"]["target"] is None
    assert neutral["claim_all_families_five_and_six_better"] is None and result["stage6_selection"] is None


# ---------------------------------------------------------------------------
# Improvement check and Stage 6
# ---------------------------------------------------------------------------

def test_improvement_check_uses_folds_1_to_4_and_reports_fold_0_separately():
    collected = neutral_grid()
    put(collected, "only_gvp", "four_class", "meanagg", flat(0.80, 0.03))
    small = [[0.80 + d + NOISE[k]] * 4 for k, d in zip(range(5), (0.05, 0.004, -0.002, 0.003, 0.001))]
    put(collected, "gvp_late_fusion", "four_class", "structlr", small)
    result = assess.improvement_check(collected, {"only_gvp": "meanagg", "gvp_late_fusion": "structlr"}, SPEC)
    assert result["only_gvp"]["folds_1_4"]["folds"] == [1, 2, 3, 4]
    assert result["only_gvp"]["development_fold0_difference"] == pytest.approx(0.03)
    assert result["only_gvp"]["status"] == "improvement interval-supported on folds 1-4"
    fusion = result["gvp_late_fusion"]
    assert fusion["status"] == "positive mean gain on folds 1-4, not interval-supported"
    assert fusion["stage6_matched_pass"] and fusion["folds_1_4"]["adjusted"]["n_contrasts"] == 2


def test_stage6_keeps_the_control_without_an_eligible_replacement():
    selection = assess.assess_step_e(neutral_grid(), SPEC)["stage6_selection"]
    assert selection["selected"] == assess.STAGE6_CONTROL and "kept" in selection["reason"]
    # Only-GVP six-class beats the control but not its own four-class arm, which is too noisy to
    # beat the control: nothing is eligible and the control is kept.
    collected = neutral_grid()
    swing = (0.05, -0.05, 0.04, -0.04, 0.0)
    put(collected, "only_gvp", "four_class", "baseline", [[0.83 + NOISE[k] + swing[k]] * 4 for k in range(5)])
    put(collected, "only_gvp", "six_class", "baseline", flat(0.80, 0.03))
    selection = assess.assess_step_e(collected, SPEC)["stage6_selection"]
    rows = {row["cell"]: row for row in selection["candidates"]}
    assert rows["only_gvp__six_class__baseline"]["versus_control"] == "better"
    assert not rows["only_gvp__six_class__baseline"]["matched_pass"]
    assert rows["only_gvp__four_class__baseline"]["versus_control"] == "no clear difference"
    assert selection["selected"] == assess.STAGE6_CONTROL


def test_stage6_tie_breakers_prefer_higher_minimum_recall_then_simpler_family():
    collected = neutral_grid()
    put(collected, "only_gvp", "four_class", "baseline", flat(0.80, 0.05))
    unbalanced = [[0.87 + NOISE[k], 0.87 + NOISE[k], 0.85 + NOISE[k], 0.813 + NOISE[k]] for k in range(5)]
    put(collected, "gvp_late_fusion", "four_class", "baseline", unbalanced)  # mean +0.08 points, lower min
    for target in ("five_class", "six_class"):
        put(collected, "only_gvp", target, "baseline", flat(0.80, 0.05))
        put(collected, "gvp_late_fusion", target, "baseline", unbalanced)
    selection = assess.assess_step_e(collected, SPEC)["stage6_selection"]
    assert selection["selected"] == "only_gvp__four_class__baseline" and selection["reason"] == "tie-breakers"
    put(collected, "gvp_late_fusion", "four_class", "baseline", flat(0.80, 0.05))  # identical to only_gvp
    for target in ("five_class", "six_class"):
        put(collected, "gvp_late_fusion", target, "baseline", flat(0.80, 0.05))
    assert assess.assess_step_e(collected, SPEC)["stage6_selection"]["selected"] == "only_gvp__four_class__baseline"


def test_stage6_tie_key_order_ends_with_simpler_family_then_baseline_recipe():
    row = {"mean_min_recall": 0.7, "worst_fold_common4_ba": 0.8, "sd_common4_ba": 0.01,
           "family": "gvp_late_fusion", "recipe": "baseline", "cell": "gvp_late_fusion__four_class__baseline"}
    improved = {**row, "recipe": "meanagg", "cell": "gvp_late_fusion__four_class__meanagg"}
    simpler = {**improved, "family": "only_gvp", "cell": "only_gvp__four_class__meanagg"}
    assert min([improved, row], key=assess.stage6_tie_key) is row
    assert min([row, simpler], key=assess.stage6_tie_key) is simpler  # family before recipe
    assert min([row, {**improved, "sd_common4_ba": 0.009}], key=assess.stage6_tie_key)["recipe"] == "meanagg"


# ---------------------------------------------------------------------------
# Step D screen and combination rule
# ---------------------------------------------------------------------------

def screen_grid(deltas_by_recipe, family="gvp_late_fusion", recall_override=None):
    collected = {}
    for seed in v3.SEEDS:
        put(collected, family, "four_class", "baseline", [[0.80] * 4], folds=(0,), seed=seed)
        for recipe, deltas in deltas_by_recipe.items():
            recalls = (recall_override or {}).get((recipe, seed), [0.80 + deltas[seed]] * 4)
            put(collected, family, "four_class", recipe, [recalls], folds=(0,), seed=seed)
    return collected


def test_screen_gate_needs_mean_gain_both_seeds_and_no_recall_loss():
    collected = screen_grid({"meanagg": {42: 0.02, 43: 0.012}, "resdrop01": {42: 0.04, 43: -0.001},
                             "structlr": {42: 0.01, 43: 0.01}, "gvpaux03": {42: 0.03, 43: 0.03}},
                            recall_override={("gvpaux03", 42): [0.86, 0.86, 0.86, 0.74],
                                             ("gvpaux03", 43): [0.86, 0.86, 0.86, 0.74]})
    results = {r: assess.screen_candidate(collected, "gvp_late_fusion", r, SPEC)
               for r in ("meanagg", "resdrop01", "structlr", "gvpaux03")}
    assert results["meanagg"]["passed"]
    assert not results["resdrop01"]["checks"]["positive_delta_both_seeds"]
    assert not results["structlr"]["checks"]["mean_delta_at_least_min"]
    assert not results["gvpaux03"]["checks"]["no_recall_change_below_limit"]


def test_screen_matched_control_and_failed_reruns():
    collected = {}
    for seed in v3.SEEDS:
        put(collected, "only_gvp", "four_class", "sitenone", [[0.70] * 4], folds=(0,), seed=seed)
        put(collected, "only_gvp", "four_class", "sitecountsangles", [[0.72] * 4], folds=(0,), seed=seed)
    result = assess.screen_candidate(collected, "only_gvp", "sitecountsangles", SPEC)
    assert result["control"] == "sitenone" and result["passed"]
    name = v3.Unit("only_gvp", "four_class", "sitecountsangles", 0, 43).name
    collected[name] = {"name": name, "status": "failed", "final_failure": False}
    assert assess.screen_candidate(collected, "only_gvp", "sitecountsangles", SPEC)["status"] == "incomplete"
    collected[name]["final_failure"] = True
    final = assess.screen_candidate(collected, "only_gvp", "sitecountsangles", SPEC)
    assert not final["passed"] and final["status"].startswith("did not pass")


def test_combination_rule_per_family():
    def screen(recipe, delta, passed=True):
        return {"family": "gvp_late_fusion", "recipe": recipe, "mean_delta": delta, "passed": passed}

    decide = assess.family_recipe_decision
    assert decide([screen("meanagg", 0.01, False)], "gvp_late_fusion")["recipe"] == "baseline"
    assert decide([screen("meanagg", 0.02)], "gvp_late_fusion")["recipe"] == "meanagg"
    # gvpaux03 and esmdrop02 are alternatives: only the better one enters the combination.
    plan = decide([screen("gvpaux03", 0.02), screen("esmdrop02", 0.03), screen("structlr", 0.025)], "gvp_late_fusion")
    assert plan["decision"] == "run combination" and plan["recipe"] == "combo-esmdrop02+structlr"
    assert decide([screen("gvpaux03", 0.02), screen("esmdrop02", 0.03)], "gvp_late_fusion")["recipe"] == "esmdrop02"
    screens = [screen("esmdrop02", 0.03), screen("structlr", 0.025)]
    adopted = decide(screens, "gvp_late_fusion", {**screen("combo-esmdrop02+structlr", 0.04)})
    assert adopted["decision"] == "combination" and adopted["recipe"] == "combo-esmdrop02+structlr"
    smaller = decide(screens, "gvp_late_fusion", {**screen("combo-esmdrop02+structlr", 0.029)})
    assert smaller["decision"] == "single" and smaller["recipe"] == "esmdrop02"


# ---------------------------------------------------------------------------
# Rows of a real tiny unit
# ---------------------------------------------------------------------------

def test_a_real_unit_is_collected_reconciled_and_tampering_is_refused(campaign, tmp_path):  # noqa: F811
    train, esm_dir, paths, _, _ = campaign
    unit = v3.Unit("gvp_late_fusion", "five_class", "baseline", 1, 42)
    result = v3.run_unit(paths, unit, lane=0, train_dir=train, esm_dir=esm_dir, python_bin=sys.executable,
                         device="cpu", execution_policy=policy(tmp_path), load_workers=1, epochs=2)
    assert result["status"] == "completed", result
    collected = assess.collect(paths, [unit, v3.Unit("only_esm", "four_class", "baseline", 1, 42)], epochs=2)
    record = collected[unit.name]
    assert record["status"] == "completed" and record["selected_epoch"] == 2
    assert set(record["metrics"]["native"]["recall"]) == {"Mn", "Cu", "Zn", "Fe", "Class VIII"}
    assert collected["only_esm__four_class__baseline__fold1__seed42"]["status"] == "missing"
    with pytest.raises(ValueError, match="identity differs"):
        assess.collect(paths, [unit], epochs=3)

    _, membership = assess.read_contract(paths)
    rows = record["rows"]
    kwargs = dict(membership=membership, fold=1, target="five_class", seed=42,
                  checkpoint_sha256=record["evidence"]["checkpoint_sha256"], selected_epoch=2)
    assess.validate_rows(rows, **kwargs)
    with pytest.raises(ValueError, match="exactly once"):
        assess.validate_rows(rows[1:], **kwargs)
    bad = [dict(row) for row in rows]
    bad[0]["p_common4_Class_VIII"], bad[0]["p_common4_Mn"] = bad[0]["p_common4_Mn"], bad[0]["p_common4_Class_VIII"]
    if bad[0]["p_common4_Mn"] != bad[0]["p_common4_Class_VIII"]:
        with pytest.raises(ValueError, match="probabilit"):
            assess.validate_rows(bad, **kwargs)
    with pytest.raises(ValueError, match="label scheme"):
        assess.validate_rows(rows, **{**kwargs, "target": "six_class"})

    out = assess.write_assessment(paths, "test", {"rows": [{"unit": unit.name, "ba": np.float64(0.5)}],
                                                  "unit": unit}, collected, "spec-sha")
    written = json.loads((out / "assessment.json").read_text())
    assert written["result"] == {"rows": [{"unit": unit.name, "ba": 0.5}], "unit": unit.name}
    assert written["unit_evidence"][unit.name]["checkpoint_sha256"] == record["evidence"]["checkpoint_sha256"]

    # Tamper with the stored predictions: the receipt hash must catch it.
    run_dir = paths.lane(0) / "runs" / unit.name
    predictions = run_dir / "val_predictions.csv"
    with predictions.open(encoding="utf-8", newline="") as handle:
        stored = list(csv.DictReader(handle))
    stored[0]["pred_native"] = str((int(stored[0]["pred_native"]) + 1) % 5)
    with predictions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stored[0]))
        writer.writeheader()
        writer.writerows(stored)
    with pytest.raises(ValueError, match="Predictions changed"):
        assess.collect(paths, [unit], epochs=2)
