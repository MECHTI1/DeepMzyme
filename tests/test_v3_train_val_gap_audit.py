"""Tests for the exploratory train/validation gap audit (audit_v3_train_val_gap.py)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import audit_v3_train_val_gap as audit  # noqa: E402
import pmm_v3_campaign  # noqa: E402

REUSED_CONTROL = "gvp_late_fusion__four_class__baseline__fold0__seed42"
PROBE = "probe-full-fp32__" + REUSED_CONTROL


def write_run(root: Path, name: str, *, train_ba: float, val_ba: float, train_loss: float = 0.1,
              val_loss: float = 1.0, epochs: int = 10, seed: int | None = None, fold: int = 0, schedule: str = "cosine",
              fit_status: str = "completed", lane: str = "lane0", last_epoch: int | None = None):
    run = root / lane / "runs" / name
    run.mkdir(parents=True)
    if seed is None:
        seed = int(name.rsplit("__seed", 1)[1]) if "__seed" in name else 42
    fields = ["epoch", "train_loss", "val_loss", audit.TRAIN_BA, audit.VAL_BA, audit.TRAIN_MIN_RECALL, audit.VAL_MIN_RECALL]
    with (run / "epoch_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            logged = epoch % 10 == 0  # training metrics are logged every tenth epoch
            number = last_epoch if (epoch == epochs and last_epoch is not None) else epoch
            writer.writerow({"epoch": number, "train_loss": train_loss, "val_loss": val_loss + (epochs - epoch) * 0.01,
                             audit.TRAIN_BA: train_ba if logged else "", audit.VAL_BA: val_ba,
                             audit.TRAIN_MIN_RECALL: 0.9 if logged else "", audit.VAL_MIN_RECALL: 0.5})
    (run / "run_config.json").write_text(json.dumps({"config": {
        "lr_schedule": schedule, "checkpoint_rule": "terminal", "seed": seed, "fold_index": fold,
        "metal_label_scheme": "merge_fe_class_viii", "epochs": epochs}}))
    (run / "selected_checkpoint.json").write_text(json.dumps({"fit_status": fit_status, "selected_epoch": epochs}))
    return run


def test_the_control_map_matches_the_frozen_runner():
    assert audit.SCREEN_CONTROLS == pmm_v3_campaign.SCREEN_CONTROLS
    assert set(audit.SCREEN_CONTROLS.values()) <= pmm_v3_campaign.NON_COMBINABLE


def test_candidates_pair_with_their_same_seed_control_and_recipes_average_over_seeds(tmp_path):
    write_run(tmp_path, "only_gvp__four_class__baseline__fold0__seed42", train_ba=0.99, val_ba=0.65, val_loss=3.0)
    write_run(tmp_path, "only_gvp__four_class__baseline__fold0__seed43", train_ba=0.98, val_ba=0.63, val_loss=2.9)
    write_run(tmp_path, "only_gvp__four_class__wd10__fold0__seed42", train_ba=0.88, val_ba=0.67, val_loss=1.0)
    write_run(tmp_path, "only_gvp__four_class__wd10__fold0__seed43", train_ba=0.87, val_ba=0.66, val_loss=0.9, lane="lane2")
    write_run(tmp_path, "only_gvp__four_class__headdrop01__fold0__seed42", train_ba=0.99, val_ba=0.62, val_loss=3.5)
    # Opposite-sign gap changes over the two seeds: the mean alone would hide the disagreement.
    write_run(tmp_path, "only_gvp__four_class__resdrop01__fold0__seed42", train_ba=0.99, val_ba=0.60)  # gap 39, d +5
    write_run(tmp_path, "only_gvp__four_class__resdrop01__fold0__seed43", train_ba=0.90, val_ba=0.66)  # gap 24, d -11
    assessment = {"result": {"rounds": {"D-R": {"screens": [
        {"family": "only_gvp", "recipe": "wd10", "passed": True, "status": "passed the one-fold screen"},
        {"family": "only_gvp", "recipe": "resdrop01", "passed": False, "status": "did not pass the one-fold screen"}]}}}}
    report = audit.run_audit(tmp_path, max_epochs=10, assessment=assessment)
    assert sorted(report["controls"]) == ["only_gvp__four_class__baseline__fold0__seed42",
                                          "only_gvp__four_class__baseline__fold0__seed43"]
    assert report["controls"]["only_gvp__four_class__baseline__fold0__seed42"]["gap"] == pytest.approx(34.0)
    assert report["control_seed_spread"]["only_gvp"] == {"n_seeds": 2, "val_ba_spread_points": pytest.approx(2.0)}
    assert report["control_seed_spread"]["gvp_late_fusion"] == {"n_seeds": 0, "val_ba_spread_points": None}
    by_unit = {u["unit"]: u for u in report["units"]}
    wd42 = by_unit["only_gvp__four_class__wd10__fold0__seed42"]
    assert wd42["paired"] and wd42["control"] == "only_gvp__four_class__baseline__fold0__seed42"
    assert wd42["control_recipe"] == "baseline"
    assert wd42["gap"] == pytest.approx(21.0) and wd42["delta_gap"] == pytest.approx(-13.0)
    assert wd42["delta_val_ba"] == pytest.approx(2.0) and wd42["delta_train_ba"] == pytest.approx(-11.0)
    assert wd42["delta_gap"] == pytest.approx(wd42["delta_train_ba"] - wd42["delta_val_ba"])
    assert wd42["control_val_loss"] == pytest.approx(3.0) and wd42["val_loss"] == pytest.approx(1.0)
    assert wd42["train_ba_saturated_both"] is False
    assert by_unit["only_gvp__four_class__headdrop01__fold0__seed42"]["train_ba_saturated_both"] is True
    recipes = {(r["family"], r["recipe"]): r for r in report["recipes"]}
    wd10 = recipes[("only_gvp", "wd10")]
    assert wd10["seeds"] == ["seed42", "seed43"] and wd10["mean_delta_gap"] == pytest.approx(-13.5)
    assert wd10["mean_gap"] == pytest.approx(21.0) and wd10["mean_control_gap"] == pytest.approx(34.5)
    assert wd10["mean_delta_val_ba"] == pytest.approx(2.5) and wd10["mean_delta_train_ba"] == pytest.approx(-11.0)
    assert wd10["delta_gap_negative_both_seeds"] is True and wd10["screen"] == "passed the one-fold screen"
    assert wd10["train_ba_saturated_both_seeds"] is False
    head = recipes[("only_gvp", "headdrop01")]
    assert head["seeds"] == ["seed42"] and head["mean_delta_gap"] == pytest.approx(3.0)
    assert head["delta_gap_negative_both_seeds"] is None and head["screen"] == "no assessment record"
    assert head["train_ba_saturated_both_seeds"] is True
    res = recipes[("only_gvp", "resdrop01")]
    assert res["mean_delta_gap"] == pytest.approx(-3.0) and res["delta_gap_negative_both_seeds"] is False
    assert report["held_out_access"] is False and report["skipped"] == []
    text = audit.table(report)
    assert "only_gvp | wd10 | baseline | 42/43 | 21.0 | 34.5 | -13.5 | +2.5 | -11.0 | no | 0.950 (2.950) | 0.100 (0.100) | yes | passed the one-fold screen" in text
    assert "only_gvp | headdrop01 | baseline | 42 | " in text and "| n/a (one seed) | no assessment record" in text
    assert "only_gvp | resdrop01 | baseline | 42/43 | " in text and "| no | did not pass the one-fold screen" in text
    assert "only_gvp 2.0 (2 seeds)" in text and "gvp_late_fusion n/a (0 seeds)" in text


def test_site_geometry_pairs_with_its_matched_sitenone_control(tmp_path):
    write_run(tmp_path, "only_gvp__four_class__baseline__fold0__seed42", train_ba=0.99, val_ba=0.65)
    write_run(tmp_path, "only_gvp__four_class__sitenone__fold0__seed42", train_ba=0.98, val_ba=0.64)
    write_run(tmp_path, "only_gvp__four_class__sitecountsangles__fold0__seed42", train_ba=0.97, val_ba=0.66)
    write_run(tmp_path, "only_gvp__four_class__sitecountsangles__fold0__seed43", train_ba=0.97, val_ba=0.66)
    report = audit.run_audit(tmp_path, max_epochs=10)
    assert sorted(report["controls"]) == ["only_gvp__four_class__baseline__fold0__seed42",
                                          "only_gvp__four_class__sitenone__fold0__seed42"]
    assert report["control_seed_spread"]["only_gvp"]["n_seeds"] == 1  # the spread counts baseline seeds only
    by_unit = {u["unit"]: u for u in report["units"]}
    site42 = by_unit["only_gvp__four_class__sitecountsangles__fold0__seed42"]
    assert site42["paired"] and site42["control"] == "only_gvp__four_class__sitenone__fold0__seed42"
    assert site42["control_recipe"] == "sitenone" and site42["delta_val_ba"] == pytest.approx(2.0)
    site43 = by_unit["only_gvp__four_class__sitecountsangles__fold0__seed43"]
    assert site43["paired"] is False and site43["reason"] == "no completed same-seed sitenone control"
    assert "sitenone" not in {u["unit"].split("__")[2] for u in report["units"]}  # a control, never a candidate


def test_unmatched_and_incomplete_runs_are_listed_with_their_reasons(tmp_path):
    write_run(tmp_path, "only_gvp__four_class__baseline__fold0__seed42", train_ba=0.99, val_ba=0.65)
    # A candidate whose control has not completed is listed as unpaired, not silently dropped.
    write_run(tmp_path, "only_gvp__four_class__meanagg__fold0__seed43", train_ba=0.95, val_ba=0.68)
    # Incomplete, wrong target, probe, fixed-LR, Only-ESMC, misnumbered and mislabelled runs are outside the audit.
    write_run(tmp_path, "only_gvp__four_class__wd10__fold0__seed42", train_ba=0.9, val_ba=0.6, fit_status="failed")
    write_run(tmp_path, "only_gvp__five_class__baseline__fold0__seed42", train_ba=0.9, val_ba=0.6)
    write_run(tmp_path, "probe-w1-fp32-r1__gvp_late_fusion__four_class__baseline__fold0__seed42", train_ba=0.9, val_ba=0.6)
    write_run(tmp_path, "only_gvp__four_class__v2recipe__fold0__seed42", train_ba=0.9, val_ba=0.6, schedule="fixed")
    write_run(tmp_path, "only_esm__four_class__baseline__fold0__seed42", train_ba=0.9, val_ba=0.6)
    write_run(tmp_path, "only_gvp__four_class__wd01__fold0__seed42", train_ba=0.9, val_ba=0.6, last_epoch=11)
    write_run(tmp_path, "only_gvp__four_class__wd001__fold0__seed43", train_ba=0.9, val_ba=0.6, seed=42)
    # The reused step-B run is addressed by its campaign unit name through the settings' reuse map.
    write_run(tmp_path, PROBE, train_ba=0.99, val_ba=0.72, lane="lane1")
    write_run(tmp_path, "gvp_late_fusion__four_class__wd10__fold0__seed42", train_ba=0.98, val_ba=0.74, lane="lane2")
    report = audit.run_audit(tmp_path, max_epochs=10, reuse={REUSED_CONTROL: PROBE})
    assert sorted(report["controls"]) == [REUSED_CONTROL, "only_gvp__four_class__baseline__fold0__seed42"]
    assert report["controls"][REUSED_CONTROL]["run"] == PROBE
    by_unit = {u["unit"]: u for u in report["units"]}
    assert set(by_unit) == {"only_gvp__four_class__meanagg__fold0__seed43", "gvp_late_fusion__four_class__wd10__fold0__seed42"}
    assert by_unit["only_gvp__four_class__meanagg__fold0__seed43"]["paired"] is False
    fusion = by_unit["gvp_late_fusion__four_class__wd10__fold0__seed42"]
    assert fusion["paired"] and fusion["control"] == PROBE and fusion["delta_val_ba"] == pytest.approx(2.0)
    reasons = {row["run"]: row["reason"] for row in report["skipped"]}
    assert reasons["only_gvp__four_class__wd10__fold0__seed42"].startswith("fit_status is 'failed'")
    assert reasons["only_gvp__five_class__baseline__fold0__seed42"] == "not a four_class fold0 unit"
    assert reasons["probe-w1-fp32-r1__gvp_late_fusion__four_class__baseline__fold0__seed42"].startswith("not a campaign unit")
    assert reasons["only_gvp__four_class__v2recipe__fold0__seed42"].startswith("lr_schedule is 'fixed'")
    assert reasons["only_esm__four_class__baseline__fold0__seed42"].startswith("not a campaign unit")
    assert reasons["only_gvp__four_class__wd01__fold0__seed42"].startswith("not a completed 10-epoch history")
    assert reasons["only_gvp__four_class__wd001__fold0__seed43"].startswith("run_config seed/fold (42, 0) disagree")
    assert len(reasons) == 7
    text = audit.table(report)
    assert "Not paired: only_gvp__four_class__meanagg__fold0__seed43 (no completed same-seed baseline control)" in text
    assert "Skipped run directories: 7" in text


def test_two_completed_copies_of_one_unit_are_refused(tmp_path):
    write_run(tmp_path, "only_gvp__four_class__baseline__fold0__seed42", train_ba=0.99, val_ba=0.65)
    write_run(tmp_path, "only_gvp__four_class__baseline__fold0__seed42", train_ba=0.98, val_ba=0.64, lane="lane1")
    with pytest.raises(ValueError, match="two completed run directories"):
        audit.run_audit(tmp_path, max_epochs=10)
    # A probe mapped onto a unit that also exists as a campaign run is the same collision.
    root = tmp_path / "twin"
    write_run(root, REUSED_CONTROL, train_ba=0.99, val_ba=0.72)
    write_run(root, PROBE, train_ba=0.99, val_ba=0.72, lane="lane1")
    with pytest.raises(ValueError, match=REUSED_CONTROL):
        audit.run_audit(root, max_epochs=10, reuse={REUSED_CONTROL: PROBE})


def test_cli_finds_the_latest_settings_and_assessment_copies_and_refuses_an_existing_output_directory(tmp_path):
    durable = tmp_path / "durable"
    write_run(durable, "only_gvp__four_class__baseline__fold0__seed42", train_ba=0.99, val_ba=0.65)
    write_run(durable, "only_gvp__four_class__wd10__fold0__seed42", train_ba=0.9, val_ba=0.67)
    write_run(durable, PROBE, train_ba=0.99, val_ba=0.72, lane="lane1")
    write_run(durable, "gvp_late_fusion__four_class__wd10__fold0__seed42", train_ba=0.98, val_ba=0.74, lane="lane2")
    # Two evidence copies: the later one carries the reuse map; two assessment copies: the later one has the pass.
    for stamp, reuse in (("20260101T000000Z", {}), ("20260102T000000Z", {REUSED_CONTROL: PROBE})):
        settings = tmp_path / "step_d_evidence" / f"evidence_{stamp}" / "campaign" / "execution_settings.json"
        settings.parent.mkdir(parents=True)
        settings.write_text(json.dumps({"amp": False, "reuse": reuse}))
    for name, status in (("D-A_20260101T000000Z", "incomplete"), ("D-R_20260102T000000Z", "passed the one-fold screen")):
        path = tmp_path / "step_d_evidence" / "assessments" / name / "assessment.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"result": {"rounds": {"D-R": {"screens": [
            {"family": "only_gvp", "recipe": "wd10", "passed": status.startswith("passed"), "status": status}]}}}}))
    out = tmp_path / "audit"
    argv = ["--campaign", str(tmp_path), "--max-epochs", "10", "--out-dir", str(out)]
    assert audit.main(argv) == 0
    report = json.loads((out / "train_val_gap.json").read_text())
    assert report["settings"].endswith("evidence_20260102T000000Z/campaign/execution_settings.json")
    assert report["reuse"] == {REUSED_CONTROL: PROBE}
    assert report["assessment"].endswith("D-R_20260102T000000Z/assessment.json")
    by_unit = {u["unit"]: u for u in report["units"]}
    fusion = by_unit["gvp_late_fusion__four_class__wd10__fold0__seed42"]
    assert fusion["paired"] is True and fusion["control"] == PROBE
    recipes = {(r["family"], r["recipe"]): r for r in report["recipes"]}
    assert recipes[("only_gvp", "wd10")]["screen"] == "passed the one-fold screen"
    assert recipes[("gvp_late_fusion", "wd10")]["screen"] == "no assessment record"
    assert (out / "train_val_gap.txt").read_text().startswith("Controls:")
    with pytest.raises(FileExistsError):
        audit.main(argv)
    # An explicit settings file overrides the discovered copy.
    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps({"reuse": {}}))
    assert audit.main(argv[:-1] + [str(tmp_path / "audit2"), "--settings", str(plain)]) == 0
    report2 = json.loads((tmp_path / "audit2" / "train_val_gap.json").read_text())
    assert report2["reuse"] == {} and {u["unit"]: u["paired"] for u in report2["units"]}[
        "gvp_late_fusion__four_class__wd10__fold0__seed42"] is False
