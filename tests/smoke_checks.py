from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from data_structures import PocketRecord
from training.config import default_selection_metric_for_task, parse_args, required_targets_for_task
from training.run import ec_group_metrics_from_logits, validate_training_configuration
from training.splits import assign_ec_group_metadata


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


class SkipCheck(RuntimeError):
    """Raised when an optional smoke check needs local data that is absent."""


def run_help(script_path: Path) -> str:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [PYTHON, str(script_path), "--help"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def check_training_cli_help() -> None:
    help_text = run_help(REPO_ROOT / "src" / "train.py")
    expected_options = (
        "--deterministic",
        "--metal-loss-weight",
        "--ec-loss-weight",
        "--ec-group-weighting",
        "--fusion-mode",
        "--cross-attention-layers",
        "--cross-attention-heads",
        "--cross-attention-dropout",
        "--cross-attention-neighborhood",
        "--cross-attention-bidirectional",
        "--allow-train-loss-test-eval-debug",
    )
    missing = [option for option in expected_options if option not in help_text]
    if missing:
        raise AssertionError(f"Training CLI help is missing expected options: {missing}")


def check_test_eval_safety() -> None:
    unsafe_config = parse_args(
        [
            "--task",
            "metal",
            "--test-structure-dir",
            "/tmp/deepmzyme_missing_test_structures",
            "--test-summary-csv",
            "/tmp/deepmzyme_missing_test_summary.csv",
            "--run-test-eval",
        ]
    )
    try:
        validate_training_configuration(unsafe_config)
    except ValueError as exc:
        message = str(exc)
        if "--run-test-eval is for held-out reporting" not in message:
            raise AssertionError(f"Unsafe test-eval config failed with an unexpected error: {message}") from exc
    else:
        raise AssertionError("Unsafe test-eval config without validation was not rejected.")

    debug_config = parse_args(
        [
            "--task",
            "metal",
            "--test-structure-dir",
            "/tmp/deepmzyme_missing_test_structures",
            "--test-summary-csv",
            "/tmp/deepmzyme_missing_test_summary.csv",
            "--run-test-eval",
            "--allow-train-loss-test-eval-debug",
        ]
    )
    validate_training_configuration(debug_config)


def check_loss_weight_validation() -> None:
    default_config = parse_args([])
    if default_config.metal_loss_weight != 1.0:
        raise AssertionError(f"Expected default metal_loss_weight=1.0, got {default_config.metal_loss_weight}")
    if default_config.ec_loss_weight != 1.0:
        raise AssertionError(f"Expected default ec_loss_weight=1.0, got {default_config.ec_loss_weight}")

    for option in ("--metal-loss-weight", "--ec-loss-weight"):
        config = parse_args([option, "-0.1"])
        try:
            validate_training_configuration(config)
        except ValueError as exc:
            if option not in str(exc):
                raise AssertionError(f"{option} failed with an unexpected error: {exc}") from exc
        else:
            raise AssertionError(f"{option} accepted a negative value.")


def check_ec_group_weighting_config() -> None:
    default_config = parse_args([])
    if default_config.ec_group_weighting != "structure_id":
        raise AssertionError(
            f"Expected default ec_group_weighting='structure_id', got {default_config.ec_group_weighting!r}"
        )
    ec_config = parse_args(["--task", "ec", "--val-fraction", "0.2"])
    if ec_config.selection_metric != "val_ec_group_balanced_acc":
        raise AssertionError(f"Expected EC default selection metric to use group metrics, got {ec_config.selection_metric!r}")
    if default_selection_metric_for_task("joint", has_validation=True) != "val_joint_balanced_acc":
        raise AssertionError("Joint default selection metric changed unexpectedly.")

    metal_config = parse_args(["--task", "metal", "--ec-group-weighting", "pdbid"])
    validate_training_configuration(metal_config)
    if required_targets_for_task("metal") != ("metal",):
        raise AssertionError("Metal-only task unexpectedly requires EC supervision.")


def check_cross_attention_config() -> None:
    config = parse_args(
        [
            "--model-architecture",
            "gvp",
            "--fusion-mode",
            "cross_modal_attention",
            "--cross-attention-layers",
            "2",
            "--cross-attention-heads",
            "8",
            "--cross-attention-dropout",
            "0.2",
            "--cross-attention-neighborhood",
            "first_second_shell",
            "--cross-attention-bidirectional",
        ]
    )
    validate_training_configuration(config)
    expected = {
        "model_architecture": "gvp",
        "fusion_mode": "cross_modal_attention",
        "cross_attention_layers": 2,
        "cross_attention_heads": 8,
        "cross_attention_dropout": 0.2,
        "cross_attention_neighborhood": "first_second_shell",
        "cross_attention_bidirectional": True,
    }
    for key, expected_value in expected.items():
        observed_value = getattr(config, key)
        if observed_value != expected_value:
            raise AssertionError(f"Expected {key}={expected_value!r}, got {observed_value!r}")


def synthetic_pocket(structure_id: str, pocket_id: str, y_ec: int | None) -> PocketRecord:
    return PocketRecord(
        structure_id=structure_id,
        pocket_id=pocket_id,
        metal_element="ZN",
        metal_coords=[torch.zeros(3)],
        residues=[],
        y_ec=y_ec,
    )


def check_ec_group_weights_sum_per_group() -> None:
    pockets = [
        synthetic_pocket("1abc__chain_A__EC_1.1.1.1", "p0", 0),
        synthetic_pocket("1abc__chain_A__EC_1.1.1.1", "p1", 0),
        synthetic_pocket("2def__chain_B__EC_2.2.2.2", "p2", 1),
        synthetic_pocket("3ghi__chain_C__EC_3.3.3.3", "p3", None),
    ]
    assign_ec_group_metadata(pockets, weighting_mode="structure_id")
    sums: dict[str, float] = {}
    for pocket in pockets:
        if pocket.y_ec is None:
            continue
        group_key = str(pocket.metadata["ec_group_key"])
        sums[group_key] = sums.get(group_key, 0.0) + float(pocket.metadata["ec_sample_weight"])
    for group_key, total in sums.items():
        if abs(total - 1.0) > 1e-6:
            raise AssertionError(f"EC weights for group {group_key} sum to {total}, expected 1.0")


def check_ec_group_metric_aggregation() -> None:
    logits = torch.tensor(
        [
            [4.0, 0.0],
            [2.0, 0.0],
            [0.0, 3.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 0, 1], dtype=torch.long)
    group_indices = torch.tensor([0, 0, 1], dtype=torch.long)
    metrics = ec_group_metrics_from_logits(
        logits,
        targets,
        group_indices,
        ec_label_map={0: "1", 1: "2"},
        ec_label_depth=1,
    )
    if metrics["n_groups"] != 2 or metrics["n_conflicting_groups"] != 0:
        raise AssertionError(f"Unexpected EC group counts: {metrics}")
    if metrics["accuracy"] != 1.0 or metrics["balanced_accuracy"] != 1.0 or metrics["macro_f1"] != 1.0:
        raise AssertionError(f"Expected perfect EC group metrics, got {metrics}")
    if metrics["level_1_accuracy"] != 1.0:
        raise AssertionError(f"Expected perfect EC level-1 group metrics, got {metrics}")


def check_ec_group_id_batches_without_increment() -> None:
    loader = DataLoader(
        [
            Data(x=torch.zeros(2, 1), y_ec=torch.tensor([0]), ec_group_id=torch.tensor([0])),
            Data(x=torch.zeros(3, 1), y_ec=torch.tensor([0]), ec_group_id=torch.tensor([0])),
        ],
        batch_size=2,
        shuffle=False,
    )
    batch = next(iter(loader))
    if batch.ec_group_id.tolist() != [0, 0]:
        raise AssertionError(f"EC group IDs were shifted during PyG batching: {batch.ec_group_id.tolist()}")


def check_conflicting_ec_group_metrics_are_skipped() -> None:
    metrics = ec_group_metrics_from_logits(
        torch.tensor([[4.0, 0.0], [0.0, 4.0]], dtype=torch.float32),
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([0, 0], dtype=torch.long),
        ec_label_map={0: "1", 1: "2"},
        ec_label_depth=1,
    )
    if metrics["n_groups"] != 0 or metrics["n_conflicting_groups"] != 1:
        raise AssertionError(f"Expected one skipped conflicting EC group, got {metrics}")


def check_bundle_cli_help() -> None:
    help_text = run_help(REPO_ROOT / "src" / "build_colab_bundle.py")
    expected_options = (
        "--allow-multi-metal-structures",
        "--strict-single-metal-structures",
    )
    missing = [option for option in expected_options if option not in help_text]
    if missing:
        raise AssertionError(f"Bundle CLI help is missing expected options: {missing}")


def check_docs_do_not_use_broken_training_command() -> None:
    broken_module = ".".join(("src", "training", "run"))
    broken_patterns = (f"python -m {broken_module}", broken_module)
    for relative_path in ("README.md", "list_train_commands.md"):
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        matches = [pattern for pattern in broken_patterns if pattern in text]
        if matches:
            raise AssertionError(f"{relative_path} still contains broken command patterns: {matches}")


def check_multi_metal_site_level_granularity() -> None:
    structure_id = "1cob__chain_A__EC_1.15.1.1"
    dataset_root = REPO_ROOT / "DeepMzyme_Data" / "train_and_test_sets_structures_non_overlapped_pinmymetal"
    train_dir = dataset_root / "train"
    structure_path = train_dir / f"{structure_id}.pdb"
    site_summary_csv = train_dir / "final_data_summarazing_table_transition_metals_only_catalytic.csv"
    inspection_csv = (
        REPO_ROOT
        / "DeepMzyme_Data"
        / "DeepMzyme_Colab_Bundles"
        / dataset_root.name
        / f"{dataset_root.name}_train.csv"
    )

    required_paths = (structure_path, site_summary_csv, inspection_csv)
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise SkipCheck(f"local multi-metal fixture files are absent: {missing}")

    from graph.construction import pocket_to_pyg_data
    from label_schemes import METAL_TARGET_LABELS
    from training.site_filter import load_allowed_site_metal_labels, resolve_allowed_site_metal_labels
    from training.structure_loading import load_structure_pockets

    pockets, _feature_fallbacks, skipped_pockets = load_structure_pockets(
        structure_path=structure_path,
        structure_root=train_dir,
        allowed_site_metal_labels=resolve_allowed_site_metal_labels(site_summary_csv),
        esm_dim=960,
        embeddings_dir=train_dir / "embeddings",
        require_esm_embeddings=False,
        feature_root_dir=train_dir,
        external_feature_source="auto",
        require_external_features=False,
        ec_label_depth=1,
    )
    if skipped_pockets:
        raise AssertionError(f"Expected no skipped pockets for {structure_id}, got: {skipped_pockets}")
    if len(pockets) != 2:
        raise AssertionError(f"Expected {structure_id} to load as 2 pocket samples, got {len(pockets)}")

    observed_labels: dict[str, str] = {}
    for pocket in pockets:
        if pocket.y_metal is None:
            raise AssertionError(f"Pocket {pocket.pocket_id} is missing y_metal.")
        metal_label = METAL_TARGET_LABELS[int(pocket.y_metal)]
        data = pocket_to_pyg_data(pocket, esm_dim=960)
        if tuple(data.y_metal.shape) != (1,):
            raise AssertionError(f"Pocket {pocket.pocket_id} has non-scalar y_metal shape {tuple(data.y_metal.shape)}")
        if str(data.y_metal.dtype) != "torch.int64":
            raise AssertionError(f"Pocket {pocket.pocket_id} has non-integer y_metal dtype {data.y_metal.dtype}")
        if ";" in metal_label:
            raise AssertionError(f"Pocket {pocket.pocket_id} received joined metal label {metal_label!r}")
        observed_labels[pocket.pocket_id] = metal_label

    if sorted(observed_labels.values()) != ["Co", "Cu"]:
        raise AssertionError(f"Expected separate Co and Cu pocket labels, got {observed_labels}")

    import csv

    with inspection_csv.open("r", encoding="utf-8", newline="") as handle:
        row = next(
            (
                csv_row
                for csv_row in csv.DictReader(handle)
                if csv_row.get("structure_name") == structure_id
            ),
            None,
        )
    if row is None:
        raise AssertionError(f"Inspection CSV {inspection_csv} is missing row for {structure_id}")
    if row.get("metal_type") != "Co;Cu":
        raise AssertionError(f"Expected inspection CSV to contain Co;Cu metadata, got {row.get('metal_type')!r}")

    try:
        load_allowed_site_metal_labels(inspection_csv)
    except ValueError as exc:
        if "Missing required columns" not in str(exc):
            raise AssertionError(f"Inspection CSV was rejected for an unexpected reason: {exc}") from exc
    else:
        raise AssertionError("Structure-level inspection CSV was accepted as a site-level training summary CSV.")


def main() -> int:
    checks = (
        check_training_cli_help,
        check_test_eval_safety,
        check_loss_weight_validation,
        check_ec_group_weighting_config,
        check_cross_attention_config,
        check_ec_group_weights_sum_per_group,
        check_ec_group_metric_aggregation,
        check_ec_group_id_batches_without_increment,
        check_conflicting_ec_group_metrics_are_skipped,
        check_bundle_cli_help,
        check_docs_do_not_use_broken_training_command,
        check_multi_metal_site_level_granularity,
    )
    for check in checks:
        try:
            check()
        except SkipCheck as exc:
            print(f"SKIP {check.__name__}: {exc}")
        else:
            print(f"PASS {check.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
