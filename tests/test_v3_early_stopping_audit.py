"""Tests for the exploratory early-stopping audit (audit_v3_early_stopping.py)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import audit_v3_early_stopping as audit  # noqa: E402


def test_the_rule_is_sequential_strict_and_keeps_the_earliest_tie():
    rising = [0.1 * k for k in range(1, 9)]
    assert audit.simulate(rising, patience=3) == {"stop_epoch": 8, "stopped_early": False, "selected_epoch": 8,
                                                  "selected_value": pytest.approx(0.8)}
    # Best at epoch 2; epochs 3-5 do not improve, so the rule stops at epoch 5 and never sees epoch 6.
    values = [0.5, 0.7, 0.7, 0.6, 0.65, 0.99, 0.2]
    out = audit.simulate(values, patience=3)
    assert out == {"stop_epoch": 5, "stopped_early": True, "selected_epoch": 2, "selected_value": 0.7}
    assert audit.simulate(values, patience=4)["selected_epoch"] == 6  # a longer patience reaches the late peak
    # An exact tie is not an improvement: the earliest epoch stays selected and patience keeps counting.
    assert audit.simulate([0.6, 0.6, 0.6], patience=2) == {"stop_epoch": 3, "stopped_early": False,
                                                           "selected_epoch": 1, "selected_value": 0.6}
    # min_delta: gains of 0.01 no longer count as improvements.
    assert audit.simulate([0.50, 0.51, 0.52, 0.53], patience=2, min_delta=0.02)["selected_epoch"] == 1
    # Stopping exactly at the last epoch is not an early stop.
    assert audit.simulate([0.9, 0.1, 0.1], patience=2) == {"stop_epoch": 3, "stopped_early": False,
                                                           "selected_epoch": 1, "selected_value": 0.9}
    with pytest.raises(ValueError):
        audit.simulate([], patience=3)


def write_run(root: Path, name: str, values, *, schedule="cosine", losses=None, seed=42, fit_status="completed",
              extra_files=("terminal_model_checkpoint.pt", "last_model_checkpoint.pt"), best_epoch=None):
    run = root / "lane0" / "runs" / name
    run.mkdir(parents=True)
    losses = losses or [1.0] * len(values)
    with (run / "epoch_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "val_loss", audit.METRIC, *audit.RECALLS.values()])
        writer.writeheader()
        for epoch, (value, loss) in enumerate(zip(values, losses), start=1):
            writer.writerow({"epoch": epoch, "val_loss": loss, audit.METRIC: value,
                             **{column: value + 0.01 * k for k, column in enumerate(audit.RECALLS.values())}})
    (run / "run_config.json").write_text(json.dumps({"config": {
        "lr_schedule": schedule, "checkpoint_rule": "terminal" if schedule == "cosine" else "best_validation",
        "seed": seed, "fold_index": 0, "metal_label_scheme": "merge_fe_class_viii", "epochs": len(values)}}))
    (run / "selected_checkpoint.json").write_text(json.dumps({
        "fit_status": fit_status, "selected_epoch": len(values), "descriptive_best_epoch": best_epoch,
        "selected_checkpoint": "terminal_model_checkpoint.pt", "metrics": {audit.METRIC: values[-1]}}))
    for extra in extra_files:
        (run / extra).write_bytes(b"x")
    return run


def test_a_run_reports_the_simulated_rule_next_to_labelled_retrospective_context(tmp_path):
    values = [0.50, 0.70, 0.66, 0.65, 0.64, 0.80, 0.60, 0.62]
    losses = [1.0, 0.8, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7]
    run = write_run(tmp_path, "only_gvp__four_class__baseline__fold0__seed42", values, losses=losses)
    row = audit.analyse_run(run, patience=3, min_delta=0.0, max_epochs=8)
    assert row["analysed"] and (row["simulated_stop_epoch"], row["simulated_selected_epoch"]) == (5, 2)
    assert row["terminal_common4_ba"] == 0.62 and row["simulated_selected_minus_terminal"] == pytest.approx(0.08)
    assert row["simulated_selected_minus_terminal_recall"]["Class VIII"] == pytest.approx(0.08)
    assert row["epochs_saved"] == 3 and row["simulated_stopped_early"]
    assert row["later_improvement_missed"] and (row["best_later_epoch"], row["missed_gain"]) == (6, pytest.approx(0.10))
    assert row["retrospective"]["best_common4_ba_epoch"] == 6 and row["retrospective"]["min_val_loss_epoch"] == 3
    assert row["retrospective"]["label"].startswith("uses the full history")
    # Only the terminal checkpoint exists, so the selected epoch's score is a logged hypothetical.
    assert row["simulated_selected_checkpoint_saved"] is False
    assert "not a recovered or replayed model" in row["simulated_selected_score_status"]
    saved = write_run(tmp_path, "only_gvp__four_class__baseline__fold0__seed43", values, seed=43,
                      extra_files=("terminal_model_checkpoint.pt", "epoch_0002_checkpoint.pt"))
    assert audit.analyse_run(saved, patience=3, min_delta=0.0, max_epochs=8)["simulated_selected_checkpoint"] == \
        "epoch_0002_checkpoint.pt"
    short = write_run(tmp_path, "probe-w1__x__y__z__fold0__seed42", values[:4])
    assert not audit.analyse_run(short, patience=3, min_delta=0.0, max_epochs=8)["analysed"]
    failed = write_run(tmp_path, "only_gvp__four_class__wd01__fold0__seed42", values, fit_status="failed")
    assert not audit.analyse_run(failed, patience=3, min_delta=0.0, max_epochs=8)["analysed"]


def test_cosine_and_fixed_lr_runs_are_reported_apart_and_seeds_side_by_side(tmp_path):
    flat = [0.6] * 8
    write_run(tmp_path, "only_gvp__four_class__wd01__fold0__seed42", [0.5, 0.7, 0.6, 0.6, 0.6, 0.6, 0.6, 0.65])
    write_run(tmp_path, "only_gvp__four_class__wd01__fold0__seed43", flat, seed=43)
    write_run(tmp_path, "probe-full-fp32__gvp_late_fusion__four_class__baseline__fold0__seed42", flat)
    write_run(tmp_path, "only_esm__four_class__v2recipe__fold0__seed42", [0.9, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4],
              schedule="fixed", extra_files=("best_model_checkpoint.pt", "last_model_checkpoint.pt"), best_epoch=1)
    write_run(tmp_path, "probe-w1-fp32-r1__gvp_late_fusion__four_class__baseline__fold0__seed42", flat[:3])
    reuse = {"gvp_late_fusion__four_class__baseline__fold0__seed42":
             "probe-full-fp32__gvp_late_fusion__four_class__baseline__fold0__seed42"}
    report = audit.run_audit(tmp_path, patience=3, min_delta=0.0, max_epochs=8, reuse=reuse)
    cosine, fixed = report["cosine_campaign_runs"], report["fixed_lr_diagnostic_runs"]
    assert {row["unit"] for row in cosine["runs"]} == {
        "only_gvp__four_class__wd01__fold0__seed42", "only_gvp__four_class__wd01__fold0__seed43",
        "gvp_late_fusion__four_class__baseline__fold0__seed42"}  # the reused step-B run under its unit name
    assert [row["unit"] for row in fixed["runs"]] == ["only_esm__four_class__v2recipe__fold0__seed42"]
    assert fixed["runs"][0]["simulated_selected_checkpoint"] == "best_model_checkpoint.pt"
    assert [row["run"] for row in report["not_analysed"]] == [
        "probe-w1-fp32-r1__gvp_late_fusion__four_class__baseline__fold0__seed42"]
    assert cosine["summary"]["n_runs"] == 3 and cosine["summary"]["n_selected_below_terminal"] == 0
    pair = cosine["by_cell_with_several_seeds"]["only_gvp__four_class__wd01__fold0"]
    assert pair["seeds"] == [42, 43] and pair["simulated_selected_minus_terminal"] == [pytest.approx(0.05), 0.0]
    assert report["held_out_access"] is False and report["rule"]["status"].startswith("exploratory")
    assert any("selection optimism" in limit for limit in report["limits"])
    assert "terminal" in audit.table(cosine["runs"]) or "sel-term" in audit.table(cosine["runs"])
