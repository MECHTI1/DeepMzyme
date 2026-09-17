"""Stage 6 command replay must preserve the selected scientific controls."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from label_schemes import active_metal_label_scheme_name, configure_active_metal_label_scheme
from stage6_standalone import (
    _build_rerun_configs,
    _build_train_command,
    _candidate_from_csv_row,
    _candidate_from_json_payload,
)
from training.config import parse_args


ROOT = Path(__file__).resolve().parents[1]
METRIC = "val_metal_balanced_acc"


@pytest.fixture(autouse=True)
def restore_metal_scheme(monkeypatch):
    original = active_metal_label_scheme_name()
    monkeypatch.setenv("DEEPGM_METAL_LABEL_SCHEME", original)
    yield
    configure_active_metal_label_scheme(original)


@pytest.mark.parametrize("source", ["json", "csv", "optuna_csv"])
@pytest.mark.parametrize("target_scheme", ["four_class", "six_class"])
def test_stage6_import_and_fold_command_keep_paired_metal_controls(tmp_path, source, target_scheme):
    selected_config = {
        "task": "metal",
        "model_architecture": "only_gvp",
        "metal_label_scheme": target_scheme,
        "metal_eligibility_scheme": "six_class",
        "split_stratify_by": "metal_site",
        "shell_role_source": "geometry",
        "site_geometry_features": "counts_angles",
        "metal_node_mode": "per_metal",
        "epochs": 50,
        "learning_rate": 1e-4,
        "run_test_eval": True,  # Historical flags must never reach a Stage 6 command.
    }
    if source == "json":
        candidate = _candidate_from_json_payload(
            {"config": selected_config}, tmp_path / "candidate.json", METRIC,
        )
    else:
        prefix = "user_attrs_" if source == "optuna_csv" else ""
        candidate = _candidate_from_csv_row(
            {prefix + key: str(value) for key, value in selected_config.items()},
            tmp_path / "candidates.csv", METRIC,
        )
    assert candidate is not None
    unit = {"model_seed": 43, "split_seed": 42, "n_folds": 5, "fold_index": 2,
            "validation_unit": "fold2_seed43", "fold_unit": "fold2"}
    rerun, = _build_rerun_configs(
        [candidate], [unit], output_runs_dir=tmp_path / "reruns", repo_dir=ROOT,
        epochs=None, device="cpu", selection_metric=METRIC,
    )
    config = parse_args(rerun["command"][2:])
    for key in ("metal_eligibility_scheme", "split_stratify_by", "shell_role_source",
                "site_geometry_features", "metal_node_mode", "learning_rate", "epochs"):
        assert getattr(config, key) == selected_config[key]
    assert config.metal_label_scheme == (
        "merge_fe_class_viii" if target_scheme == "four_class" else "split_all_metals"
    )
    assert (config.seed, config.split_seed, config.n_folds, config.fold_index) == (43, 42, 5, 2)
    assert config.split_by == "pdbid"
    assert config.selection_metric == METRIC
    assert config.val_fraction == 0.0
    assert not config.run_test_eval
    assert "--run-test-eval" not in rerun["command"]


@pytest.mark.parametrize("prefix", ["", "user_attrs_"])
def test_stage6_ec_summary_replays_group_class_weights(tmp_path, prefix):
    candidate = _candidate_from_csv_row(
        {prefix + "task": "ec", prefix + "model_architecture": "only_gvp",
         prefix + "ec_class_weight_unit": "group", prefix + "ec_group_weighting": "structure_id",
         prefix + "ec_label_depth": "1"},
        tmp_path / "ec.csv", "val_ec_group_level_1_balanced_acc",
    )
    config = parse_args(_build_train_command(candidate.config, ROOT)[2:])
    assert config.ec_class_weight_unit == "group"
    assert config.ec_group_weighting == "structure_id"
    assert config.ec_label_depth == 1
    assert config.split_stratify_by == "active_targets"
    assert not config.run_test_eval


def test_stage6_historical_missing_controls_keep_cli_defaults():
    config = parse_args(_build_train_command({"task": "metal"}, ROOT)[2:])
    assert config.metal_eligibility_scheme == "active"
    assert config.split_stratify_by == "active_targets"
    assert config.shell_role_source == "edge_mode"
    assert config.site_geometry_features == "legacy"
    assert config.ec_class_weight_unit == "pocket"
    assert not config.run_test_eval
