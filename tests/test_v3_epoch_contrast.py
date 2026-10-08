"""Tests for the logged-epoch contrast audit (audit_v3_epoch_contrast.py) on synthetic run directories."""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

import audit_v3_epoch_contrast as audit  # noqa: E402

EPOCHS = 50
COMMON4 = audit.COMMON4


def history_row(epoch: int, ba: float, recalls: dict[str, float], confusion) -> dict[str, str]:
    support = {k: sum(row) for k, row in zip(COMMON4, confusion)}
    return {"epoch": str(epoch), audit.BA: repr(ba), audit.RECALLS: repr(recalls),
            audit.CONFUSION: repr(confusion), audit.SUPPORT: repr(support)}


def make_run(durable: Path, lane: int, family: str, target: str, recipe: str, fold: int, seed: int, *,
             ba10: float, ba50: float, epochs: int = EPOCHS, duplicate_epoch: int | None = None,
             identity_override: dict | None = None) -> Path:
    name = audit.unit_name(family, target, recipe, fold, seed)
    run = durable / f"lane{lane}" / "runs" / name
    run.mkdir(parents=True)
    confusion10 = [[90, 0, 0, 10], [0, 10, 0, 0], [0, 0, 50, 0], [5, 0, 0, 95]]
    confusion50 = [[95, 0, 0, 5], [0, 10, 0, 0], [0, 0, 50, 0], [20, 0, 0, 80]]
    rows = []
    for epoch in range(1, epochs + 1):
        ba = ba10 if epoch == 10 else ba50 if epoch == EPOCHS else 0.5
        confusion = confusion10 if epoch == 10 else confusion50
        recalls = {k: row[i] / sum(row) for i, (k, row) in enumerate(zip(COMMON4, confusion))}
        rows.append(history_row(epoch, ba, recalls, confusion))
    if duplicate_epoch is not None:
        rows.append(dict(rows[duplicate_epoch - 1]))
    with (run / "val_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    identity = {"family": family, "target_scheme": target, "recipe": recipe, "fold": fold, "model_seed": seed,
                "epochs": EPOCHS, "resolved_config_sha256": "x" * 64, **(identity_override or {})}
    receipt = {"campaign_run_identity": identity, "fit_status": "completed", "checkpoint_rule": "terminal",
               "planned_epochs": EPOCHS, "completed_epochs": EPOCHS, "metrics": {audit.BA: ba50}}
    (run / "selected_checkpoint.json").write_text(json.dumps(receipt))
    return run


VALUES = {0: (0.60, 0.70), 1: (0.81, 0.73), 2: (0.74, 0.73), 3: (0.74, 0.75), 4: (0.77, 0.78)}
CELL = ("only_esm", "four_class", "baseline")


def populate(durable: Path, *, order: list[int], lanes: list[int]) -> None:
    for fold, lane in zip(order, lanes):
        ba10, ba50 = VALUES[fold]
        make_run(durable, lane, *CELL, fold, 42, ba10=ba10, ba50=ba50)


def assessment_for(values=VALUES) -> dict:
    folds = sorted(values)
    return {"result": {"summaries": {"__".join(CELL): {"folds": folds,
                                                         "fold_common4_ba": [values[k][1] for k in folds]}}}}


def test_lane_and_directory_order_cannot_change_the_result(tmp_path):
    reports = []
    for trial in range(4):
        durable = tmp_path / f"d{trial}"
        order = list(VALUES)
        random.Random(trial).shuffle(order)
        populate(durable, order=order, lanes=[random.Random(trial + 10).randrange(3) for _ in order])
        reports.append(audit.run_audit(durable, [CELL], folds=[1, 2, 3, 4], seed=42, early=10, late=50,
                                       assessment=assessment_for()))
    expected = sum(VALUES[k][0] - VALUES[k][1] for k in (1, 2, 3, 4)) / 4
    for report in reports:
        item = report["cells"]["__".join(CELL)]
        assert item["folds"] == [1, 2, 3, 4]          # development fold 0 excluded, fold 4 included
        assert item["mean_ba_difference"] == pytest.approx(expected)
        assert item["mean_ba_epoch50"] == pytest.approx(sum(VALUES[k][1] for k in (1, 2, 3, 4)) / 4)
        assert [r["fold"] for r in item["per_fold"]] == [1, 2, 3, 4]
        assert all(r["unit"].endswith(f"fold{r['fold']}__seed42") for r in item["per_fold"])
    assert len({json.dumps(audit.jsonable(r)["cells"], sort_keys=True, default=str)
                .replace("lane0", "L").replace("lane1", "L").replace("lane2", "L") for r in reports}) == 1


def test_confusion_and_recall_changes_follow_true_rows_and_predicted_columns(tmp_path):
    populate(tmp_path, order=[1, 2], lanes=[0, 1])
    item = audit.run_audit(tmp_path, [CELL], folds=[1, 2], seed=42, early=10, late=50)["cells"]["__".join(CELL)]
    row = item["per_fold"][0]
    assert row["class_viii_to_mn"] == {10: 5, 50: 20}
    assert row["mn_to_class_viii"] == {10: 10, 50: 5}
    assert item["mean_recall_difference"]["Mn"] == pytest.approx(0.90 - 0.95)
    assert item["mean_recall_difference"]["Class VIII"] == pytest.approx(0.95 - 0.80)


def test_missing_fold_is_refused(tmp_path):
    populate(tmp_path, order=[0, 1, 2, 3], lanes=[0, 1, 2, 0])
    with pytest.raises(audit.AuditError, match="Missing run: .*fold4__seed42"):
        audit.run_audit(tmp_path, [CELL], folds=[1, 2, 3, 4], seed=42, early=10, late=50)


def test_duplicate_run_directories_across_lanes_are_refused(tmp_path):
    populate(tmp_path, order=[1, 2, 3, 4], lanes=[0, 1, 2, 0])
    make_run(tmp_path, 1, *CELL, 3, 42, ba10=0.74, ba50=0.75)  # fold 3 already sits in lane 2
    with pytest.raises(audit.AuditError, match="Duplicate run directories .*fold3__seed42"):
        audit.run_audit(tmp_path, [CELL], folds=[1, 2, 3, 4], seed=42, early=10, late=50)


def test_duplicate_history_record_is_refused(tmp_path):
    make_run(tmp_path, 0, *CELL, 1, 42, ba10=0.8, ba50=0.7, duplicate_epoch=10)
    with pytest.raises(audit.AuditError, match="duplicate history record for epoch 10"):
        audit.run_audit(tmp_path, [CELL], folds=[1], seed=42, early=10, late=50)


def test_incomplete_history_is_refused(tmp_path):
    make_run(tmp_path, 0, *CELL, 1, 42, ba10=0.8, ba50=0.7, epochs=49)
    with pytest.raises(audit.AuditError, match="not exactly 1..50"):
        audit.run_audit(tmp_path, [CELL], folds=[1], seed=42, early=10, late=50)


def test_identity_mismatch_is_refused(tmp_path):
    make_run(tmp_path, 0, *CELL, 1, 42, ba10=0.8, ba50=0.7, identity_override={"model_seed": 43})
    with pytest.raises(audit.AuditError, match="recorded identity"):
        audit.run_audit(tmp_path, [CELL], folds=[1], seed=42, early=10, late=50)


def test_wrong_seed_request_finds_no_run(tmp_path):
    populate(tmp_path, order=[1], lanes=[0])
    with pytest.raises(audit.AuditError, match="Missing run: .*seed43"):
        audit.run_audit(tmp_path, [CELL], folds=[1], seed=43, early=10, late=50)


def test_terminal_value_must_reconcile_with_the_assessment(tmp_path):
    populate(tmp_path, order=[1, 2, 3, 4], lanes=[0, 1, 2, 0])
    altered = dict(VALUES)
    altered[3] = (0.74, 0.7501)
    with pytest.raises(audit.AuditError, match="differs from the assessment"):
        audit.run_audit(tmp_path, [CELL], folds=[1, 2, 3, 4], seed=42, early=10, late=50,
                        assessment=assessment_for(altered))


def test_late_epoch_must_be_terminal_and_folds_distinct(tmp_path):
    populate(tmp_path, order=[1, 2], lanes=[0, 1])
    with pytest.raises(audit.AuditError, match="not the terminal epoch"):
        audit.run_audit(tmp_path, [CELL], folds=[1, 2], seed=42, early=10, late=40)
    with pytest.raises(audit.AuditError, match="distinct"):
        audit.run_audit(tmp_path, [CELL], folds=[1, 1], seed=42, early=10, late=50)


def test_cli_refuses_with_exit_code_and_writes_nothing(tmp_path, capsys):
    populate(tmp_path / "durable", order=[1, 2, 3], lanes=[0, 1, 2])
    out = tmp_path / "out"
    code = audit.main(["--durable", str(tmp_path / "durable"), "--cell", "only_esm:four_class:baseline",
                       "--out-dir", str(out)])
    assert code == 2 and not out.exists()
    assert "Missing run" in capsys.readouterr().err
