"""Amendment-2 plan v2: per-fold fit lists, in-fit F6 pretraining, raw-RBF units, caps and v1 replay (synthetic)."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ec_clean30 import campaign, selection  # noqa: E402
from ec_clean30.campaign import (  # noqa: E402
    CampaignPlanError, PlannedUnit, Recipe, build_fold_plan, build_fold_plan_v2, canonical_json, plan_units,
)
from ec_clean30_amendment2_fixtures import install_family_table  # noqa: E402


@pytest.fixture(autouse=True)
def family_table(monkeypatch):
    install_family_table(monkeypatch)


def _units(fold, step=None):
    return [unit for unit in plan_units(build_fold_plan_v2(fold)) if step is None or unit.step == step]


def test_fold0_and_later_fold_counts():
    fold0 = build_fold_plan_v2(0)
    assert fold0["counts"] == {"S1": 112, "S2": 0, "S3": 0, "S4": 5, "total": 117}
    assert fold0["families"] == ["F1", "F2", "F3", "F4", "F5", "F6", "C-G", "C-GF"]
    for fold in range(1, 5):
        plan = build_fold_plan_v2(fold)
        assert plan["counts"] == {"S1": 96, "S2": 0, "S3": 0, "S4": 5, "total": 101}
        assert "F1" not in plan["families"] and not any(unit.family == "F1" for unit in _units(fold))
    assert sum(1 for unit in _units(0, "S1") if unit.family == "F1") == 16
    for fold in range(5):
        s1 = _units(fold, "S1")
        assert all(unit.epochs == 30 and unit.lever_configs == () for unit in s1)
        assert not any(unit.family == "C-GF" for unit in s1)
        assert {unit.family for unit in s1} | {"C-GF"} == set(campaign.FAMILIES_BY_FOLD_V2[fold])
        assert all(sum(1 for unit in s1 if unit.family == family) == 16 for family in {unit.family for unit in s1})


def test_f6_pretrains_inside_its_fits_and_the_plan_has_no_pretraining_units():
    # Rule K2 / A2-F6-5: no pretraining unit kind, no depends_on; C-GF comes from the F6 fits' control files.
    units = _units(0)
    assert {unit.step for unit in units} == {"S1", "S4"} and {unit.kind for unit in units} == {"inner", "refit"}
    f6 = [unit for unit in units if unit.step == "S1" and unit.family == "F6"]
    assert len(f6) == 16 and all(campaign.recipe_for_unit(unit).pretrains_global_branch for unit in f6)
    assert all("depends_on" not in unit.to_dict() for unit in units)
    s4 = [unit for unit in units if unit.step == "S4"]
    assert [unit.seed for unit in s4] == [42, 43, 44, 45, 46]
    assert all(unit.applies_to == ("F2", "F3", "F4", "F5", "F6") and unit.epochs == 30 for unit in s4)
    assert "F1" not in s4[0].applies_to and "C-G" not in s4[0].applies_to
    constants = build_fold_plan_v2(0)["constants"]
    assert constants["f6_pretraining"]["epochs"] == 30 and constants["f6_pretraining"]["batch_size"] == 64
    assert constants["families"]["C-GF"]["fits"] == 0
    assert constants["controls"]["whole_protein_control"]["prediction_file"] == "wp_pretrain_val_protein_predictions.csv"


def test_gvp_families_carry_the_raw_distance_default():
    for unit in _units(0, "S1"):
        raw = unit.family in ("F1", "F3", "F4")
        assert unit.rbf_distances == ("raw" if raw else None)
        assert unit.unit_id.split("__")[3].endswith("_rbfraw") == raw
        assert campaign.recipe_for_unit(unit).rbf_distances == ("raw" if raw else "normalized")
    constants = build_fold_plan_v2(0)["constants"]
    assert constants["rbf_distances"]["F3"] == "raw" and constants["rbf_distances"]["F2"] == "normalized"
    assert constants["search"]["screened_levers"] == ["L9"] and constants["search"]["s2_fits"] == 0
    assert constants["controls"]["selectable"] is False and constants["controls"]["f1_folds"] == [0]
    assert constants["selection_policy_sha256"] == selection.selection_policy_sha256_v2()
    # Raw units never share a unit ID with a v1 unit (rule A2-TRN-2).
    v1_ids = {item["unit_id"] for item in build_fold_plan(0)["units"]}
    raw_ids = {unit.unit_id for unit in _units(0, "S1") if unit.rbf_distances == "raw"}
    assert len(raw_ids) == 48 and not raw_ids & v1_ids


def test_plan_api_for_the_execution_lane():
    plan = build_fold_plan_v2(0)
    assert len(campaign.s1_units_for_plan(plan)) == 112 and len(campaign.s1_units_for_plan(build_fold_plan(0))) == 96
    assert len(campaign.fold_units_v2(plan, "S4")) == 5
    with pytest.raises(CampaignPlanError, match="only in S1 and S4"):
        campaign.fold_units_v2(plan, "S2")
    with pytest.raises(CampaignPlanError, match="needs a"):
        campaign.fold_units_v2(build_fold_plan(0), "S1")
    assert [campaign.plan_version_for_chain(chain) for chain in ([], ["a" * 64], ["a" * 64, "b" * 64])] == [1, 1, 2]
    with pytest.raises(CampaignPlanError):
        campaign.plan_version_for_chain(["a" * 64] * 3)
    timing = campaign.timing_recipes_v2()
    assert [name for name, _recipe in timing] == ["anchor", "F5", "F6", "F3_raw", "F4_raw", "C-G", "F1_raw"]
    assert timing[0][1].to_dict() == {"family": "F2", "learning_rate": {"learning_rate": 1e-4}, "epochs": 30,
                                      "lever_configs": []}
    assert [recipe.rbf_distances for _name, recipe in timing] == [
        "normalized", "normalized", "normalized", "raw", "raw", "normalized", "raw"]
    assert campaign.check_family_v2("F6") == "F6" and not campaign.config_applies_v2("L4", "F6")
    with pytest.raises(CampaignPlanError):
        campaign.check_family("F6")


def test_plan_v2_is_deterministic_and_round_trips():
    first, second = build_fold_plan_v2(2), build_fold_plan_v2(2)
    assert canonical_json(first) == canonical_json(second)
    assert first["format"] == "deepmzyme_ec_clean30_fold_plan_v2" and first["v1_plan_sha256"] == \
        build_fold_plan(2)["plan_sha256"]
    loaded = json.loads(json.dumps(first))
    units = plan_units(loaded)
    assert [unit.to_dict() for unit in units] == loaded["units"]
    loaded["units"][0]["epochs"] = 20
    with pytest.raises(CampaignPlanError, match="plan_sha256"):
        plan_units(loaded)


def test_v1_units_and_recipes_are_byte_identical():
    unit = PlannedUnit("u", 0, "S1", "inner", "fixed", "F2", (("learning_rate", 1e-4),), 30, (), 0, 42, ("F2",))
    assert "rbf_distances" not in unit.to_dict()
    for item in build_fold_plan(0)["units"]:
        assert "rbf_distances" not in item
    for family, grid in selection.S1_LEARNING_RATE_GRID.items():
        recipe = Recipe(family, tuple(grid[0]), 30)
        assert set(recipe.to_dict()) == {"family", "learning_rate", "epochs", "lever_configs"}


def test_plan_v2_refuses_without_the_amendment2_family_table(monkeypatch):
    monkeypatch.delattr(campaign, "FAMILY_SPECS_V2", raising=False)
    with pytest.raises(CampaignPlanError, match="FAMILY_SPECS_V2"):
        build_fold_plan_v2(0)
    monkeypatch.setattr(campaign, "FAMILY_SPECS_V2", dict(campaign.FAMILY_SPECS), raising=False)
    with pytest.raises(CampaignPlanError, match="lacks"):
        build_fold_plan_v2(0)
    from ec_clean30_amendment2_fixtures import F6_SPEC

    changed = {**campaign.FAMILY_SPECS, "F2": campaign.FamilySpec("F2", "gvp", None, False, True, "anchor"),
               "F6": F6_SPEC}
    monkeypatch.setattr(campaign, "FAMILY_SPECS_V2", changed, raising=False)
    with pytest.raises(CampaignPlanError, match="changes the Freeze-1 family F2"):
        build_fold_plan_v2(0)


def test_plan_v2_refuses_other_folds_and_caps(monkeypatch):
    with pytest.raises(CampaignPlanError, match="Outer fold"):
        build_fold_plan_v2(5)
    monkeypatch.setitem(campaign.STEP_FIT_CAPS_V2, "S1", 111)
    with pytest.raises(CampaignPlanError, match="S1 plans 112 fits"):
        build_fold_plan_v2(0)
    monkeypatch.setitem(campaign.STEP_FIT_CAPS_V2, "S1", 112)
    monkeypatch.setattr(campaign, "MAX_FITS_PER_FOLD_V2", 116)
    with pytest.raises(CampaignPlanError, match="above the cap of 116"):
        build_fold_plan_v2(0)


def test_g2_replay_anchor_unit_is_unchanged():
    anchor = "ecc30__fold0__S1__F2_lr1e-4_e30__inner0__seed42"
    v1 = next(item for item in build_fold_plan(0)["units"] if item["unit_id"] == anchor)
    v2 = next(item for item in build_fold_plan_v2(0)["units"] if item["unit_id"] == anchor)
    assert v1 == v2


def test_search_and_control_policy_records():
    search = campaign.search_policy_record_v2()
    assert search["rule_id"] == "l9_only_no_s3_v1" and search["s3"] == "not run" and search["epochs"] == 30
    assert search["dropped_levers"] == ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"]
    controls = campaign.controls_policy_record_v2()
    assert controls["controls_by_fold"]["0"] == ["F1", "C-G", "C-GF"]
    assert all(controls["controls_by_fold"][str(fold)] == ["C-G", "C-GF"] for fold in range(1, 5))
    assert controls["whole_protein_control"]["extra_fits"] == 0
