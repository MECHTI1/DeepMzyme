"""Tests for generalized 5-fold cross-validation runner across datasets and units."""

from pathlib import Path
import sys
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import run_metal_5fold_cv as rmc
import run_exact_pinmymetal_5fold_cv as rpc


DATA_ROOT = REPO_ROOT / "DeepMzyme_Data"

DATASETS_TO_TEST = [
    "train_and_test_sets_structures_exact_pinmymetal",
    "train_and_test_sets_structures_zenodo_pmm_exact",
    "train_and_test_sets_structures_non_overlapped_pinmymetal",
    "train_and_test_sets_structures_common_pdbid_70_30_pinmymetal",
    "CLEAN_30_train_test_split_0",
    "CARE_task1_30_clusterRes30_train_test_metallo",
]


@pytest.mark.parametrize("dataset_name", DATASETS_TO_TEST)
def test_dataset_layout_resolution(dataset_name):
    if not (DATA_ROOT / dataset_name).exists():
        pytest.skip(f"Dataset {dataset_name} not available locally")

    d_id, train_dir, train_csv, test_dir, test_csv = rmc.resolve_dataset_layout(
        DATA_ROOT, dataset_name
    )
    assert d_id == dataset_name
    assert train_dir.exists() and train_dir.is_dir()
    assert train_csv.exists() and train_csv.is_file()
    if test_dir is not None:
        assert test_dir.is_dir()
        if test_csv is not None:
            assert test_csv.is_file()


@pytest.mark.parametrize("unit", ["pocket", "ion"])
def test_generalized_runner_command_generation(unit):
    train_dir = DATA_ROOT / "train_and_test_sets_structures_exact_pinmymetal" / "train"
    train_csv = train_dir / "final_data_summarazing_table_transition_metals_only_catalytic.csv"
    runs_dir = Path("/tmp/test_runs")
    run_name = f"benchmark_enhanced_only_gvp_{unit}_fold0"

    cmd = rmc.build_fold_command(
        python_bin="python",
        model_key="benchmark_enhanced_only_gvp",
        fold_idx=0,
        n_folds=5,
        train_dir=train_dir,
        train_csv=train_csv,
        runs_dir=runs_dir,
        run_name=run_name,
        feat_dir=DATA_ROOT / "updated_feature_extraction",
        esm_dir=DATA_ROOT / "esm_embeddings",
        metal_example_unit=unit,
        train_val_split_by="pocket_id",
        epochs=10,
    )

    assert "--metal-example-unit" in cmd
    unit_idx = cmd.index("--metal-example-unit")
    assert cmd[unit_idx + 1] == unit

    assert "--train-val-split-by" in cmd
    split_idx = cmd.index("--train-val-split-by")
    assert cmd[split_idx + 1] == "pocket_id"

    assert "--run-name" in cmd
    name_idx = cmd.index("--run-name")
    assert cmd[name_idx + 1] == run_name
    assert unit in cmd[name_idx + 1]


def test_pocket_and_ion_run_names_do_not_collide():
    for f in range(5):
        pocket_name = f"benchmark_enhanced_only_gvp_pocket_fold{f}"
        ion_name = f"benchmark_enhanced_only_gvp_ion_fold{f}"
        assert pocket_name != ion_name


def test_exact_pinmymetal_runner_supports_both_units():
    # Pocket mode preserves legacy run_name
    p_name, p_cmd = rpc.build_fold_command(
        python_bin="python",
        model_key="benchmark_only_esm",
        fold_idx=1,
        n_folds=5,
        data_root=DATA_ROOT,
        runs_dir=Path("/tmp/runs"),
        metal_example_unit="pocket",
    )
    assert p_name == "benchmark_only_esm_fold1"
    assert "--metal-example-unit" in p_cmd
    assert p_cmd[p_cmd.index("--metal-example-unit") + 1] == "pocket"

    # Ion mode produces differentiated run_name
    i_name, i_cmd = rpc.build_fold_command(
        python_bin="python",
        model_key="benchmark_only_esm",
        fold_idx=1,
        n_folds=5,
        data_root=DATA_ROOT,
        runs_dir=Path("/tmp/runs"),
        metal_example_unit="ion",
    )
    assert i_name == "benchmark_only_esm_ion_fold1"
    assert "--metal-example-unit" in i_cmd
    assert i_cmd[i_cmd.index("--metal-example-unit") + 1] == "ion"


def test_dry_run_flag(monkeypatch, capsys):
    test_args = [
        "run_metal_5fold_cv.py",
        "--dataset",
        "train_and_test_sets_structures_exact_pinmymetal",
        "--metal-example-unit",
        "ion",
        "--models",
        "benchmark_enhanced_only_gvp",
        "--folds",
        "0",
        "--dry-run",
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    rmc.main()
    captured = capsys.readouterr()
    assert "DRY RUN VALIDATION: 5-FOLD BENCHMARK PLAN" in captured.out
    assert "Dry run validation successful" in captured.out

