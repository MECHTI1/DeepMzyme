"""Bounded paired trend and isolation-control rules for adaptive discovery."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serial_metal_campaign import adaptive, profile


def run(arm, parameters, ident="recipe"):
    return dict(arm=arm, parameters=parameters, id=ident,
                scheme="six_class" if arm.endswith("six") else "four_class", stage="discovery")


def rows(item, score=.7, recalls=None):
    native = recalls or {"Mn": .7, "Cu": .7, "Zn": .7, "VIII": .7}
    if item["scheme"] == "six_class" and recalls is None:
        native = {"Mn": .7, "Cu": .7, "Zn": .7, "Fe": .7, "Co": .4, "Ni": .6}
    return [{**item, "seed": seed, "balanced_accuracy": score, "cohort_sha256": "same-cohort",
             "per_class_recall": dict(native),
             "collapsed4_per_class_recall": {"Mn": .7, "Cu": .7, "Zn": .7, "VIII": .7}}
            for seed in profile.SEEDS]


def comparison(family="late", before=None, after=None):
    arms = [arm for arm in profile.ARMS if arm.startswith(family + "_")]
    parent = {arm: run(arm, before or profile.reference_parameters(), arm + "_parent") for arm in arms}
    child = {arm: run(arm, after or profile.reference_parameters(learning_rate=1e-4), arm + "_child") for arm in arms}
    evidence = [row for arm in arms for row in rows(parent[arm]) + rows(child[arm], .704)]
    return dict(parent_runs=parent, candidate_runs=child), evidence


def test_paired_gain_requires_both_seeds_mean_gain_and_native_recall_protection():
    base_run = run("late_six", profile.reference_parameters())
    before, after = rows(base_run), rows(base_run, .704)
    assert adaptive.paired_gain(before, after)["passed"]
    after[1]["balanced_accuracy"] = .699
    assert not adaptive.paired_gain(before, after)["passed"]
    after = rows(base_run, .702)
    assert not adaptive.paired_gain(before, after)["passed"]
    after = rows(base_run, .704)
    after[0]["per_class_recall"]["Co"] = 0
    assert not adaptive.paired_gain(before, after)["passed"]


def test_paired_gain_protects_each_common_class_and_rejects_partial_or_changed_groups():
    item = run("gvp_four", profile.reference_parameters())
    before, after = rows(item), rows(item, .71)
    after[0]["collapsed4_per_class_recall"]["Zn"] = .63
    assert not adaptive.paired_gain(before, after)["passed"]
    after[0]["collapsed4_per_class_recall"]["Zn"] = .64
    assert adaptive.paired_gain(before, after)["passed"]
    with pytest.raises(ValueError, match="exactly seeds"):
        adaptive.paired_gain(before[:1], after)
    after[1]["cohort_sha256"] = "different"
    with pytest.raises(ValueError, match="shared certified cohort"):
        adaptive.paired_gain(before, after)


def test_chain_is_matched_when_only_one_core_target_improves():
    comparison_record, evidence = comparison()
    for row in evidence:
        if row["id"] == "late_six_child":
            row["balanced_accuracy"] = .699
    proposal = adaptive.propose_next("late", [comparison_record], evidence)
    assert proposal["status"] == "continue"
    assert proposal["axis"] == "learning_rate" and proposal["direction"] == 1
    assert set(proposal["parameters"]) == {"late_four", "late_six"}
    assert all(params["learning_rate"] == 2e-4 for params in proposal["parameters"].values())
    assert proposal["required_fits_per_arm"] == {"late_four": 2, "late_six": 2}
    with pytest.raises(ValueError, match="preserve both targets"):
        adaptive.propose_next("late", [{**comparison_record, "parent_runs": {"late_four": comparison_record["parent_runs"]["late_four"]}}], evidence)


def test_missing_repeat_requests_control_only_and_reserves_followup_fit_allowance():
    comparison_record, evidence = comparison()
    evidence = [row for row in evidence if not (row["id"].endswith("parent") and row["seed"] == 43)]
    proposal = adaptive.propose_next("late", [comparison_record], evidence)
    assert proposal["status"] == "controls_required" and proposal["parameters"] == {}
    assert {control["arm"] for control in proposal["controls"]} == {"late_four", "late_six"}
    assert all(control["seeds"] == [43] for control in proposal["controls"])
    assert all(value == 3 for value in proposal["required_fits_per_arm"].values())
    assert adaptive.propose_next("late", [comparison_record], evidence,
        fits_used_per_arm={"late_four": 4, "late_six": 4})["status"] == "stop"


def test_coupled_regularization_requires_isolation_control_before_numeric_extension():
    original = profile.reference_parameters()
    improved = profile.reference_parameters(weight_decay=1e-3, head_mlp_dropout=.3)
    comparison_record, evidence = comparison(before=original, after=improved)
    assert adaptive.compare_axes(original, improved, "late") is None
    proposal = adaptive.propose_next("late", [comparison_record], evidence)
    assert proposal["status"] == "controls_required"
    assert proposal["axis"] == "weight_decay"
    assert all(control["parameters"]["weight_decay"] == 1e-4 and
               control["parameters"]["head_mlp_dropout"] == .3 for control in proposal["controls"])
    for control in proposal["controls"]:
        evidence.extend(rows(run(control["arm"], control["parameters"], control["arm"] + "_isolation"), .700))
    ready = adaptive.propose_next("late", [comparison_record], evidence,
                                 fits_used_per_arm={"late_four": 2, "late_six": 2})
    assert ready["status"] == "continue"
    assert all(params["weight_decay"] == 1e-2 and params["head_mlp_dropout"] == .3
               for params in ready["parameters"].values())


def test_categorical_diagnostics_cannot_start_a_numeric_chain():
    comparison_record, evidence = comparison(after=profile.reference_parameters(
        metal_class_weight_mode="effective_number"))
    assert adaptive.propose_next("late", [comparison_record], evidence)["status"] == "stop"


@pytest.mark.parametrize("family,upper", [("gvp", 3e-4), ("esm", 2e-4), ("late", 2e-4),
                                          ("early", 2e-4), ("hybrid", 1.5e-4)])
def test_lr_steps_and_hard_bounds(family, upper):
    assert adaptive.next_parameters(profile.reference_parameters(learning_rate=1e-4),
                                    "learning_rate", 1, family)["learning_rate"] == upper
    assert adaptive.next_parameters(profile.reference_parameters(learning_rate=1e-5),
                                    "learning_rate", -1, family)["learning_rate"] == 5e-6
    assert adaptive.next_parameters(profile.reference_parameters(learning_rate=8e-4),
                                    "learning_rate", 1, family) is None
    assert adaptive.next_parameters(profile.reference_parameters(learning_rate=1e-6),
                                    "learning_rate", -1, family) is None


def test_late_capacity_bundle_is_one_axis_with_fixed_other_settings_and_upper_stop():
    reference = profile.reference_parameters()
    large = profile.reference_parameters(**profile.LARGE_CAPACITIES["large_256"], capacity="large_256")
    assert adaptive.compare_axes(reference, large, "late") == ("late_capacity", 1)
    larger = adaptive.next_parameters(large, "late_capacity", 1, "late")
    assert tuple(larger[field] for field in adaptive.WIDTH_FIELDS) == (384, 48, 192)
    assert larger["head_mlp_layers"] == 2 and larger["gvp_layers"] == 4 and larger["esm_fusion_dim"] == 128
    largest = adaptive.next_parameters(larger, "late_capacity", 1, "late")
    assert adaptive.next_parameters(largest, "late_capacity", 1, "late") is None
    assert adaptive.next_parameters(large, "late_capacity", 1, "gvp") is None


def test_no_repeating_observed_setting_no_direction_switch_and_six_fit_cap():
    record, evidence = comparison()
    extra = {arm: run(arm, profile.reference_parameters(learning_rate=2e-4), arm + "_next")
             for arm in ("late_four", "late_six")}
    evidence.extend(row for item in extra.values() for row in rows(item, .710))
    assert adaptive.propose_next("late", [record], evidence)["status"] == "stop"
    second = dict(parent_runs=record["candidate_runs"], candidate_runs=extra)
    proposal = adaptive.propose_next("late", [record, second], evidence,
                                    fits_used_per_arm={"late_four": 4, "late_six": 4},
                                    active_axis="learning_rate", active_direction=1)
    assert proposal["status"] == "continue"
    assert proposal["parameters"]["late_four"]["learning_rate"] == 4e-4
    assert adaptive.propose_next("late", [second], evidence,
        fits_used_per_arm={"late_four": 6, "late_six": 6})["status"] == "stop"
    assert adaptive.propose_next("late", [second], evidence, active_direction=-1)["status"] == "stop"


def test_early_bottleneck_continuation_keeps_dropout_fixed():
    parameters = profile.reference_parameters(early_esm_dim=64, early_esm_dropout=.1)
    next_value = adaptive.next_parameters(parameters, "early_esm_dim", 1, "hybrid")
    assert next_value["early_esm_dim"] == 128 and next_value["early_esm_dropout"] == .1
    assert adaptive.next_parameters(next_value, "early_esm_dim", 1, "hybrid") is None


def test_opposite_target_boundaries_allow_one_qualifying_direction_and_reuse_other_target():
    parent = {
        "late_four": run("late_four", profile.reference_parameters(learning_rate=1e-5), "four_parent"),
        "late_six": run("late_six", profile.reference_parameters(learning_rate=1e-4), "six_parent"),
    }
    child = {
        "late_four": run("late_four", profile.reference_parameters(learning_rate=5e-6), "four_child"),
        "late_six": run("late_six", profile.reference_parameters(learning_rate=2e-4), "six_child"),
    }
    evidence = [row for arm in parent for row in rows(parent[arm], .70) + rows(child[arm], .705)]
    proposal = adaptive.propose_next("late", [dict(parent_runs=parent, candidate_runs=child)], evidence)
    # Fixed ordering selects downward. The six-class arm has already received
    # its next downward value, so its existing two-seed control is reused.
    assert proposal["status"] == "continue" and proposal["direction"] == -1
    assert proposal["parameters"]["late_four"]["learning_rate"] == 2.5e-6
    assert proposal["parameters"]["late_six"]["learning_rate"] == 1e-4
    assert proposal["gates"]["late_four"]["passed"]
    assert not proposal["gates"]["late_six"]["passed"]
    assert proposal["required_fits_per_arm"] == {"late_four": 2, "late_six": 0}
    assert proposal["next_missing_seeds"] == {"late_four": [42, 43], "late_six": []}
