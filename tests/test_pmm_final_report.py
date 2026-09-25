"""Final selection and access gates use synthetic development evidence only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmarking import pmm_ion_campaign as campaign
from benchmarking.pmm_final_report import (
    CONTROL, FinalReportError, _set_argument, select_configuration, verify_final_refit,
    require_reference_authorization,
)
from training.source_cohort import sha256_file


def _grid():
    labels = ("Mn", "Cu", "Zn", "Class VIII")
    return {config.config_id: {"config": config, "complete": True, "folds": {
        k: {"metrics": {"common4": {"balanced_accuracy": 0.6,
                                      "recall": dict.fromkeys(labels, 0.6)}}} for k in range(5)}}
            for config in campaign.grid_configs()}


def _score(entry, value, recalls=None):
    for unit in entry["folds"].values():
        unit["metrics"]["common4"]["balanced_accuracy"] = value
        unit["metrics"]["common4"]["recall"] = recalls or dict.fromkeys(("Mn", "Cu", "Zn", "Class VIII"), value)


def test_inconclusive_selection_retains_predeclared_control():
    result = select_configuration(_grid(), [])
    assert result["selected_config_id"] == CONTROL
    assert result["selection_reason"] == "retained_predeclared_control_no_supported_replacement"


def test_untested_six_plus_aware_can_never_be_selected():
    grid = _grid()
    six = "only_esm__six_class__none"
    aware = "only_esm__four_class__first_shell_bias"
    _score(grid[six], 0.8)
    _score(grid[aware], 0.7)
    matched = [{"challenger": six, "promoted": True}, {"challenger": aware, "promoted": True}]
    result = select_configuration(grid, matched)
    assert result["selected_config_id"] == six
    assert result["selected_config_id"] in grid


def test_matched_and_control_rare_recall_gates_both_required():
    grid = _grid()
    challenger = "gvp_late_fusion__six_class__none"
    _score(grid[challenger], 0.8)
    assert select_configuration(grid, [])['selected_config_id'] == CONTROL
    _score(grid[challenger], 0.8, {"Mn": 0.9, "Cu": 0.4, "Zn": 0.9, "Class VIII": 1.0})
    result = select_configuration(grid, [{"challenger": challenger, "promoted": True}])
    assert result["selected_config_id"] == CONTROL


def test_missing_fold_blocks_final_selection():
    grid = _grid()
    del grid[CONTROL]["folds"][4]
    with pytest.raises(FinalReportError, match="every planned fold"):
        select_configuration(grid, [])


def test_direct_four_baseline_can_replace_control_with_paired_evidence():
    grid = _grid()
    winner = "only_esm__four_class__none"
    _score(grid[winner], 0.7)
    assert select_configuration(grid, [])["selected_config_id"] == winner


def test_argument_replacement_removes_fold_and_old_weights_cleanly():
    command = ["python", "-u", "train.py", "--n-folds", "5", "--export-validation-predictions",
               "--selection-metric", "val_metal_balanced_acc", "--mn-loss-multiplier", "1.0"]
    _set_argument(command, "--n-folds", remove=True)
    _set_argument(command, "--export-validation-predictions", remove=True)
    _set_argument(command, "--selection-metric", "train_loss")
    _set_argument(command, "--mn-loss-multiplier", "2.0")
    assert command == ["python", "-u", "train.py", "--selection-metric", "train_loss",
                       "--mn-loss-multiplier", "2.0"]


def _final_receipt(tmp_path):
    paths = campaign.CampaignPaths(tmp_path)
    run = tmp_path / "final_refit" / "selected"
    run.mkdir(parents=True)
    checkpoint = run / "last_model_checkpoint.pt"
    checkpoint.write_bytes(b"frozen terminal weights")
    for name in ("run_config", "run_metadata"):
        (run / f"{name}.json").write_text("{}")
    identity = {"selected_config_id": CONTROL}
    campaign.write_json(tmp_path / "stage6b_decision.json", {"identity": identity})
    receipt = {"status": "completed", "identity": identity, "terminal_checkpoint": True,
               "completed_epochs": 50, "held_out_test_accessed": False,
               "checkpoint_path": str(checkpoint.relative_to(tmp_path)),
               "checkpoint_sha256": sha256_file(checkpoint),
               **{f"{name}_sha256": sha256_file(run / f"{name}.json")
                  for name in ("run_config", "run_metadata")}}
    campaign.write_json(tmp_path / "stage6b_selected_final_refit_candidate.json", receipt)
    return paths, run, receipt


@pytest.mark.parametrize("changed", ["last_model_checkpoint.pt", "run_config.json", "run_metadata.json"])
def test_final_refit_rejects_artifact_drift(tmp_path, changed):
    paths, run, receipt = _final_receipt(tmp_path)
    assert verify_final_refit(paths) == receipt
    (run / changed).write_bytes(b"changed after selection")
    with pytest.raises(FinalReportError, match="hash differs|changed after certification"):
        verify_final_refit(paths)


def test_partial_refit_is_not_a_reference_source(tmp_path):
    paths, _run, receipt = _final_receipt(tmp_path)
    receipt["completed_epochs"] = 49
    campaign.write_json(tmp_path / "stage6b_selected_final_refit_candidate.json", receipt)
    with pytest.raises(FinalReportError, match="epoch-50"):
        verify_final_refit(paths)


def test_reference_inputs_are_not_opened_without_frozen_protocol(tmp_path, monkeypatch):
    route = tmp_path / "route.json"
    route.write_text(json.dumps({"route": "zenodo_pmm_secondary_reference", "selection_frozen": False}))
    reference_dir = tmp_path / "never_open_this_test"
    original = Path.open

    def guarded_open(path, *args, **kwargs):
        if path == reference_dir or reference_dir in path.parents:
            pytest.fail("Reference inputs opened before final refits were frozen")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(FinalReportError, match="frozen secondary reference route"):
        require_reference_authorization(campaign.CampaignPaths(tmp_path / "reference"), reference_dir, route)


@pytest.mark.parametrize("config_id", [CONTROL, "gvp_late_fusion__six_class__none"])
def test_refit_command_uses_full_training_and_equalized_common_four_weights(tmp_path, monkeypatch, config_id):
    from test_pmm_ion_campaign import _campaign_with_folds, _full_inventory
    from benchmarking.pmm_final_report import build_refit_command
    from training.config import parse_args
    from collections import Counter

    train, paths, _weights = _campaign_with_folds(tmp_path, monkeypatch)
    _full_inventory(paths, tmp_path)
    command, _env, identity = build_refit_command(paths, train, config_id, python_bin=sys.executable, device="cpu")
    config = parse_args(command[3:])
    assert config.val_fraction == 0 and config.n_folds is None
    assert config.fold_membership_csv is None and not config.export_validation_predictions
    assert config.selection_metric == "train_loss" and config.epochs == 50 and config.seed == 42
    assert not config.run_test_eval and config.test_structure_dir is None
    assert identity["checkpoint_rule"] == "terminal_epoch_50"
    bindings = campaign.read_cohort_csv(paths.cohort)
    counts = Counter(campaign.COMMON_FOUR[b.native_element] for b in bindings)
    assert all(identity["common_four_weights"][label] * count == pytest.approx(len(bindings) / 4)
               for label, count in counts.items())
