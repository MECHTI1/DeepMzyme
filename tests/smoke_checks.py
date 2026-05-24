from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from data_structures import PocketRecord, ResidueRecord
from statistical_validation import paired_bootstrap_ci
from training.config import TrainConfig, default_selection_metric_for_task, parse_args, required_targets_for_task
from training.final_test_reporting import (
    equal_mass_ece,
    fit_temperature_from_logits,
    metal_bootstrap_metric_cis,
)
from training.loop import balanced_class_weights_from_pockets, class_weights_from_labels, train_epoch
from training.run import (
    build_run_dir,
    ec_group_metrics_from_logits,
    resolve_selection_metric,
    validate_training_configuration,
)
from training.splits import assign_ec_group_metadata, split_pockets_k_fold
from metal_objectives import (
    collapse_metal_logits_to_4,
    collapsed4_cross_entropy_from_logits,
    metal_loss_with_optional_collapsed4,
    validate_required_six_class_metal_labels,
)


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
        "--joint-loss-weighting",
        "--metal-label-scheme",
        "--metal-loss-weight",
        "--metal-collapsed-loss-weight",
        "--ec-loss-weight",
        "--ec-group-weighting",
        "--fusion-mode",
        "--use-ring-edges",
        "--cross-attention-layers",
        "--cross-attention-heads",
        "--cross-attention-dropout",
        "--cross-attention-neighborhood",
        "--cross-attention-bidirectional",
        "--position-noise-std",
        "--second-shell-dropout",
        "--outer-residue-dropout",
        "--head-mlp-dropout",
        "--esm-graph-encoder-dropout",
        "--metal-node-mode",
        "--structural-readout-scope",
        "--grad-clip-norm",
        "--amp",
        "--grad-accum-steps",
        "--num-workers",
        "--pin-memory",
        "--allow-train-loss-test-eval-debug",
        "--train-val-split-by",
        "--split-by",
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


def check_prelaunch_run_dir_reuse() -> None:
    with tempfile.TemporaryDirectory(prefix="deepmzyme_run_dir_") as tmp:
        runs_dir = Path(tmp) / "runs"
        prelaunch_dir = runs_dir / "trial_0000"
        prelaunch_dir.mkdir(parents=True)
        (prelaunch_dir / "active_run_config.json").write_text("{}", encoding="utf-8")
        (prelaunch_dir / "active_run_config.md").write_text("# Active Run Configuration\n", encoding="utf-8")
        (prelaunch_dir / "optuna_trial_status.json").write_text('{"status": "failed"}', encoding="utf-8")

        config = parse_args(["--runs-dir", str(runs_dir), "--run-name", "trial_0000"])
        if build_run_dir(config) != prelaunch_dir:
            raise AssertionError("Prelaunch notebook run directory was not reused.")
        if (prelaunch_dir / "optuna_trial_status.json").exists():
            raise AssertionError("Stale Optuna trial status was not removed before retrying the run.")

        completed_dir = runs_dir / "completed_run"
        completed_dir.mkdir()
        (completed_dir / "run_config.json").write_text("{}", encoding="utf-8")
        completed_config = parse_args(["--runs-dir", str(runs_dir), "--run-name", "completed_run"])
        try:
            build_run_dir(completed_config)
        except FileExistsError as exc:
            if "not an empty/notebook prelaunch directory" not in str(exc):
                raise AssertionError(f"Existing run directory failed with an unclear error: {exc}") from exc
        else:
            raise AssertionError("Existing completed run directory was reused.")


def check_loss_weight_validation() -> None:
    default_config = parse_args([])
    if default_config.joint_loss_weighting != "uncertainty":
        raise AssertionError(
            "Expected joint --joint-loss-weighting auto to resolve to uncertainty, "
            f"got {default_config.joint_loss_weighting!r}"
        )
    if default_config.metal_loss_weight != 1.0:
        raise AssertionError(f"Expected default metal_loss_weight=1.0, got {default_config.metal_loss_weight}")
    if default_config.ec_loss_weight != 1.0:
        raise AssertionError(f"Expected default ec_loss_weight=1.0, got {default_config.ec_loss_weight}")

    metal_config = parse_args(["--task", "metal"])
    if metal_config.joint_loss_weighting != "fixed":
        raise AssertionError(
            "Expected single-task --joint-loss-weighting auto to resolve to fixed, "
            f"got {metal_config.joint_loss_weighting!r}"
        )

    for option in ("--metal-loss-weight", "--ec-loss-weight"):
        config = parse_args([option, "-0.1"])
        try:
            validate_training_configuration(config)
        except ValueError as exc:
            if option not in str(exc):
                raise AssertionError(f"{option} failed with an unexpected error: {exc}") from exc
        else:
            raise AssertionError(f"{option} accepted a negative value.")

    invalid_single_task_config = parse_args(["--task", "metal", "--joint-loss-weighting", "uncertainty"])
    try:
        validate_training_configuration(invalid_single_task_config)
    except ValueError as exc:
        if "--joint-loss-weighting uncertainty requires --task joint" not in str(exc):
            raise AssertionError(f"Unexpected joint-loss weighting validation error: {exc}") from exc
    else:
        raise AssertionError("Single-task uncertainty loss weighting was not rejected.")

    default_metal_config = parse_args(["--task", "metal"])
    if default_metal_config.metal_collapsed_loss_weight != 0.0:
        raise AssertionError(
            "Expected default metal_collapsed_loss_weight=0.0, "
            f"got {default_metal_config.metal_collapsed_loss_weight}"
        )
    invalid_collapsed_config = parse_args(["--task", "metal", "--metal-collapsed-loss-weight", "1.1"])
    try:
        validate_training_configuration(invalid_collapsed_config)
    except ValueError as exc:
        if "--metal-collapsed-loss-weight" not in str(exc):
            raise AssertionError(f"Unexpected collapsed-loss validation error: {exc}") from exc
    else:
        raise AssertionError("--metal-collapsed-loss-weight accepted a value outside [0, 1].")
    invalid_ec_collapsed_config = parse_args(["--task", "ec", "--metal-collapsed-loss-weight", "0.3"])
    try:
        validate_training_configuration(invalid_ec_collapsed_config)
    except ValueError as exc:
        if "metal prediction head" not in str(exc):
            raise AssertionError(f"Unexpected EC collapsed-loss validation error: {exc}") from exc
    else:
        raise AssertionError("EC-only training accepted a metal collapsed-4 loss.")

    invalid_noise_config = parse_args(["--position-noise-std", "-0.1"])
    try:
        validate_training_configuration(invalid_noise_config)
    except ValueError as exc:
        if "--position-noise-std" not in str(exc):
            raise AssertionError(f"Unexpected position-noise validation error: {exc}") from exc
    else:
        raise AssertionError("--position-noise-std accepted a negative value.")

    invalid_dropout_config = parse_args(["--second-shell-dropout", "1.1"])
    try:
        validate_training_configuration(invalid_dropout_config)
    except ValueError as exc:
        if "--second-shell-dropout" not in str(exc):
            raise AssertionError(f"Unexpected second-shell dropout validation error: {exc}") from exc
    else:
        raise AssertionError("--second-shell-dropout accepted a value outside [0, 1].")

    invalid_outer_dropout_config = parse_args(["--outer-residue-dropout", "1.1"])
    try:
        validate_training_configuration(invalid_outer_dropout_config)
    except ValueError as exc:
        if "--outer-residue-dropout" not in str(exc):
            raise AssertionError(f"Unexpected outer-residue dropout validation error: {exc}") from exc
    else:
        raise AssertionError("--outer-residue-dropout accepted a value outside [0, 1].")

    invalid_head_dropout_config = parse_args(["--head-mlp-dropout", "-0.1"])
    try:
        validate_training_configuration(invalid_head_dropout_config)
    except ValueError as exc:
        if "--head-mlp-dropout" not in str(exc):
            raise AssertionError(f"Unexpected head MLP dropout validation error: {exc}") from exc
    else:
        raise AssertionError("--head-mlp-dropout accepted a value outside [0, 1].")

    invalid_esm_dropout_config = parse_args(["--esm-graph-encoder-dropout", "1.1"])
    try:
        validate_training_configuration(invalid_esm_dropout_config)
    except ValueError as exc:
        if "--esm-graph-encoder-dropout" not in str(exc):
            raise AssertionError(f"Unexpected ESM graph encoder dropout validation error: {exc}") from exc
    else:
        raise AssertionError("--esm-graph-encoder-dropout accepted a value outside [0, 1].")


def check_metal_label_scheme_options() -> None:
    from label_schemes import (
        configure_active_metal_label_scheme,
        metal_labels_for_scheme,
        metal_symbol_to_target_for_scheme,
    )

    five_class_labels = metal_labels_for_scheme("five_class")
    five_class_targets = metal_symbol_to_target_for_scheme("five_class")
    if five_class_labels != {0: "Mn", 1: "Cu", 2: "Zn", 3: "Fe", 4: "Class VIII"}:
        raise AssertionError(f"Unexpected five_class labels: {five_class_labels}")
    if five_class_targets["CO"] != five_class_targets["NI"]:
        raise AssertionError("five_class should map Co and Ni to the same target id.")
    if five_class_targets["FE"] == five_class_targets["CO"]:
        raise AssertionError("five_class should keep Fe separate from the grouped Co/Ni class.")

    try:
        five_class_config = parse_args(["--metal-label-scheme", "five_class"])
        if five_class_config.metal_label_scheme != "five_class":
            raise AssertionError(
                "Expected --metal-label-scheme five_class to normalize to 'five_class', "
                f"got {five_class_config.metal_label_scheme!r}."
            )
    finally:
        configure_active_metal_label_scheme("split_all_metals")


def check_training_efficiency_defaults_and_validation() -> None:
    default_config = parse_args([])
    expected_defaults = {
        "grad_clip_norm": 1.0,
        "use_amp": False,
        "grad_accum_steps": 1,
        "num_workers": 0,
        "pin_memory": False,
        "normalize_message_aggregation": False,
        "train_val_split_by": "pdbid",
        "split_by": "pdbid",
    }
    for field_name, expected_value in expected_defaults.items():
        observed_value = getattr(default_config, field_name)
        if observed_value != expected_value:
            raise AssertionError(f"Expected default {field_name}={expected_value!r}, got {observed_value!r}")

    resolved_metal = resolve_selection_metric(TrainConfig(task="metal", val_fraction=0.15))
    if resolved_metal.selection_metric != "val_metal_balanced_acc":
        raise AssertionError(f"Metal validation selection metric resolved incorrectly: {resolved_metal.selection_metric}")
    resolved_no_val = resolve_selection_metric(TrainConfig(task="metal", val_fraction=0.0))
    if resolved_no_val.selection_metric != "train_loss":
        raise AssertionError(f"No-validation selection metric resolved incorrectly: {resolved_no_val.selection_metric}")

    invalid_accum = parse_args(["--grad-accum-steps", "0"])
    try:
        validate_training_configuration(invalid_accum)
    except ValueError as exc:
        if "--grad-accum-steps" not in str(exc):
            raise AssertionError(f"Unexpected grad-accum validation error: {exc}") from exc
    else:
        raise AssertionError("--grad-accum-steps accepted a value below 1.")

    invalid_workers = parse_args(["--num-workers", "-1"])
    try:
        validate_training_configuration(invalid_workers)
    except ValueError as exc:
        if "--num-workers" not in str(exc):
            raise AssertionError(f"Unexpected num-workers validation error: {exc}") from exc
    else:
        raise AssertionError("--num-workers accepted a negative value.")

    normalized_gvp = parse_args(["--gvp-normalize-message-aggregation"])
    if not normalized_gvp.normalize_message_aggregation:
        raise AssertionError("GVP message aggregation normalization flag was not parsed.")

    disabled_reporting = parse_args(
        [
            "--disable-final-test-calibration",
            "--disable-final-test-temperature-scaling",
            "--disable-final-test-bootstrap-ci",
        ]
    )
    if (
        disabled_reporting.final_test_enable_calibration
        or disabled_reporting.final_test_enable_temperature_scaling
        or disabled_reporting.final_test_enable_bootstrap_ci
    ):
        raise AssertionError("Final-test disable aliases did not map to affirmative config fields.")

    for role in ("primary_preselected", "exploratory_posthoc"):
        role_config = parse_args(["--final-test-result-role", role])
        validate_training_configuration(role_config)
        if role_config.final_test_result_role != role:
            raise AssertionError(f"Final-test role {role!r} was not preserved by config parsing.")


class LinearLossModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, batch):
        return {"loss": (self.weight * batch.loss_scale.float()).sum()}


def check_grad_accum_final_partial_window() -> None:
    loader = DataLoader(
        [
            Data(x=torch.zeros(1, 1), loss_scale=torch.tensor([2.0])),
            Data(x=torch.zeros(1, 1), loss_scale=torch.tensor([4.0])),
            Data(x=torch.zeros(1, 1), loss_scale=torch.tensor([8.0])),
        ],
        batch_size=1,
        shuffle=False,
    )
    model = LinearLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    loss = train_epoch(
        model,
        loader,
        optimizer,
        device="cpu",
        grad_clip_norm=0.0,
        grad_accum_steps=2,
    )
    expected_weight = torch.tensor(-11.0)
    if not torch.allclose(model.weight.detach(), expected_weight, atol=1e-6):
        raise AssertionError(
            "Gradient accumulation did not flush/scale the final partial window correctly: "
            f"weight={float(model.weight.detach())}, expected={float(expected_weight)}"
        )
    expected_loss = -8.0
    if abs(loss - expected_loss) > 1e-8:
        raise AssertionError(
            f"Expected reported mean loss to use undivided batch losses, got {loss}, expected {expected_loss}"
        )


def check_uncertainty_task_loss_weighter() -> None:
    from model import TaskLossWeighter

    weighter = TaskLossWeighter(mode="uncertainty", predict_metal=True, predict_ec=True)
    metal_loss = torch.tensor(2.0, requires_grad=True)
    ec_loss = torch.tensor(1.0, requires_grad=True)
    total_loss, diagnostics = weighter({"metal": metal_loss, "ec": ec_loss})
    total_loss.backward()

    if "metal_loss_scale" not in diagnostics or "ec_loss_scale" not in diagnostics:
        raise AssertionError("Uncertainty task loss diagnostics are missing task scales.")
    if weighter.metal_log_variance is None or weighter.ec_log_variance is None:
        raise AssertionError("Joint uncertainty weighting did not create learnable log-variance parameters.")
    if weighter.metal_log_variance.grad is None or weighter.ec_log_variance.grad is None:
        raise AssertionError("Uncertainty weighting parameters did not receive gradients.")


def check_collapsed4_metal_loss_helpers() -> None:
    label_map = {
        0: "Zn",
        1: "Fe",
        2: "Mn",
        3: "Ni",
        4: "Cu",
        5: "Co",
    }
    logits = torch.tensor(
        [
            [-5.0, 1.0, 0.5, 2.0, 3.0, 4.0],
            [0.25, -1.0, 2.0, -2.0, 0.0, -3.0],
        ],
        dtype=torch.float32,
    )
    collapsed = collapse_metal_logits_to_4(logits, label_map=label_map, require_six_class=True)
    expected = torch.stack(
        [
            logits[:, 2],
            logits[:, 4],
            logits[:, 0],
            torch.logsumexp(logits[:, [1, 5, 3]], dim=-1),
        ],
        dim=-1,
    )
    if not torch.allclose(collapsed, expected, atol=1e-6):
        raise AssertionError(f"Collapsed-4 logsumexp marginalization changed: {collapsed} vs {expected}")

    targets = torch.tensor([2, 1], dtype=torch.long)
    six_weights = torch.tensor([1.4, 0.7, 1.1, 0.9, 1.2, 0.8], dtype=torch.float32)
    collapsed_weights = torch.tensor([1.0, 1.3, 0.6, 0.9], dtype=torch.float32)
    six_ce = F.cross_entropy(logits, targets, weight=six_weights)
    collapsed_ce = collapsed4_cross_entropy_from_logits(
        logits,
        targets,
        weight=collapsed_weights,
        label_map=label_map,
        require_six_class=True,
    )
    alpha0_loss, alpha0_aux = metal_loss_with_optional_collapsed4(
        six_ce,
        logits,
        targets,
        alpha=0.0,
        collapsed4_weight=collapsed_weights,
        label_map=label_map,
    )
    if alpha0_aux is not None or not torch.allclose(alpha0_loss, six_ce, atol=0.0, rtol=0.0):
        raise AssertionError("alpha=0 did not preserve the original six-class CE loss exactly.")

    alpha1_loss, alpha1_aux = metal_loss_with_optional_collapsed4(
        six_ce,
        logits,
        targets,
        alpha=1.0,
        collapsed4_weight=collapsed_weights,
        label_map=label_map,
    )
    if alpha1_aux is None or not torch.allclose(alpha1_loss, collapsed_ce, atol=1e-6):
        raise AssertionError("alpha=1 did not produce pure collapsed-4 CE.")

    try:
        validate_required_six_class_metal_labels({0: "Mn", 1: "Cu", 2: "Zn", 3: "Fe"})
    except ValueError as exc:
        message = str(exc)
        if "Co" not in message or "Ni" not in message or "Observed labels" not in message:
            raise AssertionError(f"Missing-label guard raised an unclear error: {message}") from exc
    else:
        raise AssertionError("Missing six-class metal labels were not rejected for collapsed-4 loss.")


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


def check_ring_edge_cli_config() -> None:
    default_config = parse_args([])
    if default_config.use_ring_edges:
        raise AssertionError("Raw CLI default should use radius-only edges unless --use-ring-edges is passed.")
    if not default_config.prepare_missing_ring_edges:
        raise AssertionError("Default training config should prepare missing RING edges when RING is enabled.")

    optional_config = parse_args(["--use-ring-edges"])
    if (
        not optional_config.use_ring_edges
        or optional_config.require_ring_edges
        or not optional_config.prepare_missing_ring_edges
    ):
        raise AssertionError("Expected --use-ring-edges to enable optional RING edges without requiring them.")

    required_config = parse_args(["--require-ring-edges"])
    if (
        not required_config.use_ring_edges
        or not required_config.require_ring_edges
        or not required_config.prepare_missing_ring_edges
    ):
        raise AssertionError("Expected --require-ring-edges to imply use_ring_edges.")

    prepared_config = parse_args(["--prepare-missing-ring-edges"])
    if not prepared_config.use_ring_edges or not prepared_config.prepare_missing_ring_edges:
        raise AssertionError("Expected --prepare-missing-ring-edges to imply use_ring_edges.")

    disabled_prepare_config = parse_args(["--use-ring-edges", "--no-prepare-missing-ring-edges"])
    if not disabled_prepare_config.use_ring_edges or disabled_prepare_config.prepare_missing_ring_edges:
        raise AssertionError("Expected --no-prepare-missing-ring-edges to disable automatic RING generation.")

    metal_node_config = parse_args(["--model-architecture", "only_gvp", "--metal-node-mode", "per_metal"])
    if metal_node_config.metal_node_mode != "per_metal":
        raise AssertionError("Expected --metal-node-mode per_metal to reach TrainConfig.")
    if metal_node_config.structural_readout_scope != "residue_and_metal":
        raise AssertionError("Expected structural readout auto mode to include metal nodes.")


def augmentation_fixture_pocket() -> PocketRecord:
    return PocketRecord(
        structure_id="augment_fixture",
        pocket_id="augment_fixture_site0",
        metal_element="ZN",
        metal_coords=[torch.tensor([0.0, 0.0, 0.0])],
        residues=[
            ResidueRecord(
                chain_id="A",
                resseq=1,
                icode="",
                resname="CYS",
                atoms={
                    "CA": torch.tensor([1.0, 0.0, 0.0]),
                    "CB": torch.tensor([1.2, 0.0, 0.0]),
                    "SG": torch.tensor([1.5, 0.0, 0.0]),
                },
            ),
            ResidueRecord(
                chain_id="A",
                resseq=2,
                icode="",
                resname="ALA",
                atoms={
                    "CA": torch.tensor([4.0, 0.0, 0.0]),
                    "CB": torch.tensor([4.2, 0.0, 0.0]),
                },
            ),
            ResidueRecord(
                chain_id="A",
                resseq=3,
                icode="",
                resname="ALA",
                atoms={
                    "CA": torch.tensor([10.0, 0.0, 0.0]),
                    "CB": torch.tensor([10.2, 0.0, 0.0]),
                },
            ),
        ],
        y_metal=0,
    )


def check_training_graph_augmentation() -> None:
    from graph.construction import pocket_to_pyg_data
    from training.graph_dataset import PocketGraphDataset

    pocket = augmentation_fixture_pocket()
    reference = pocket_to_pyg_data(pocket, esm_dim=2)

    train_dataset = PocketGraphDataset(
        [pocket],
        esm_dim=2,
        position_noise_std=0.5,
        second_shell_dropout=0.0,
    )
    first_epoch_graph = train_dataset[0]
    second_epoch_graph = train_dataset[0]
    if torch.allclose(first_epoch_graph.pos, second_epoch_graph.pos):
        raise AssertionError("Training coordinate noise did not vary across repeated dataset reads.")

    validation_dataset = PocketGraphDataset([pocket], esm_dim=2)
    validation_graph = validation_dataset[0]
    if not torch.allclose(validation_graph.pos, reference.pos):
        raise AssertionError("Validation/default graph dataset unexpectedly changed coordinates.")
    if not torch.allclose(pocket.residues[0].atoms["CA"], torch.tensor([1.0, 0.0, 0.0])):
        raise AssertionError("Augmentation mutated the source PocketRecord coordinates.")

    dropout_dataset = PocketGraphDataset(
        [pocket],
        esm_dim=2,
        edge_radius=12.0,
        position_noise_std=0.0,
        second_shell_dropout=1.0,
    )
    dropout_graph = dropout_dataset[0]
    kept_positions = {tuple(float(value) for value in row) for row in dropout_graph.pos.tolist()}
    if (4.0, 0.0, 0.0) in kept_positions:
        raise AssertionError("Second-shell dropout failed to remove the second-shell residue.")
    if (1.0, 0.0, 0.0) not in kept_positions or (10.0, 0.0, 0.0) not in kept_positions:
        raise AssertionError("Second-shell dropout removed a first-shell or non-second-shell residue.")

    outer_default_dataset = PocketGraphDataset(
        [pocket],
        esm_dim=2,
        edge_radius=12.0,
        position_noise_std=0.0,
        second_shell_dropout=0.0,
        outer_residue_dropout=0.0,
    )
    outer_default_graph = outer_default_dataset[0]
    if not torch.allclose(outer_default_graph.pos, reference.pos):
        raise AssertionError("outer_residue_dropout=0.0 unexpectedly changed the graph.")

    outer_dropout_dataset = PocketGraphDataset(
        [pocket],
        esm_dim=2,
        edge_radius=12.0,
        position_noise_std=0.0,
        second_shell_dropout=0.0,
        outer_residue_dropout=1.0,
    )
    outer_dropout_graph = outer_dropout_dataset[0]
    outer_kept_positions = {tuple(float(value) for value in row) for row in outer_dropout_graph.pos.tolist()}
    if (10.0, 0.0, 0.0) in outer_kept_positions:
        raise AssertionError("Outer-residue dropout failed to remove the outer residue.")
    if (1.0, 0.0, 0.0) not in outer_kept_positions or (4.0, 0.0, 0.0) not in outer_kept_positions:
        raise AssertionError("Outer-residue dropout removed a first-shell or second-shell residue.")


def check_esm_embedding_metadata_sidecar() -> None:
    from training.esm_feature_loading import (
        build_embedding_payload,
        embedding_metadata_from_payload,
        load_embedding_metadata_sidecar,
        summarize_esm_embedding_metadata,
        write_embedding_metadata_sidecar,
    )

    with tempfile.TemporaryDirectory(prefix="deepmzyme_esm_metadata_") as tmp:
        tmp_root = Path(tmp)
        structure_path = tmp_root / "1abc__chain_A__EC_1.1.1.1.pdb"
        structure_path.write_text("HEADER metadata smoke\n", encoding="utf-8")
        embeddings_dir = tmp_root / "embeddings"
        embeddings_dir.mkdir()
        embedding_path = embeddings_dir / "1abc__chain_A__EC_1.1.1.1_chain_A_esmc.pt"
        payload = build_embedding_payload(
            torch.zeros(2, 960),
            [("A", 1, ""), ("A", 2, "")],
            structure_id="1abc__chain_A__EC_1.1.1.1",
            chain_id="A",
            source_path=str(structure_path),
            metadata={
                "esm_model_name": "esmc_300m",
                "embedding_dim": 960,
                "generated_at": "2026-05-18T00:00:00+00:00",
            },
        )
        metadata = embedding_metadata_from_payload(payload)
        torch.save(payload, embedding_path)
        write_embedding_metadata_sidecar(embedding_path, metadata)
        loaded = load_embedding_metadata_sidecar(embedding_path)
        if loaded is None or loaded.get("esm_model_name") != "esmc_300m":
            raise AssertionError(f"ESM sidecar metadata was not round-tripped: {loaded}")
        summary = summarize_esm_embedding_metadata([structure_path], embeddings_dir)
        if summary["esm_model_names"] != ["esmc_300m"] or summary["embedding_dims"] != [960]:
            raise AssertionError(f"ESM metadata summary did not capture model/dim: {summary}")


def check_only_gvp_does_not_require_esm() -> None:
    only_gvp_config = parse_args(["--model-architecture", "only_gvp"])
    if only_gvp_config.require_esm_embeddings or only_gvp_config.use_esm_branch:
        raise AssertionError("Only-GVP runs should not require or generate ESM embeddings.")

    esm_config = parse_args(["--model-architecture", "only_esm"])
    if not esm_config.require_esm_embeddings or not esm_config.use_esm_branch:
        raise AssertionError("Only-ESM runs should require ESM embeddings by default.")


def check_graph_ring_edges_are_opt_in() -> None:
    from data_structures import EDGE_SOURCE_TO_INDEX, ResidueRecord
    from graph.construction import pocket_to_pyg_data

    with tempfile.TemporaryDirectory(prefix="deepmzyme_ring_opt_in_") as tmp:
        ring_path = Path(tmp) / "example_ringEdges"
        ring_path.write_text(
            "NodeId1\tNodeId2\tInteraction\tAtom1\tAtom2\n"
            "A:1:_:ALA\tA:2:_:GLU\tHBOND:SC_SC\tCA\tCA\n",
            encoding="utf-8",
        )
        pocket = PocketRecord(
            structure_id="example",
            pocket_id="example_A_1",
            metal_element="ZN",
            metal_coords=[torch.tensor([0.0, 0.0, 0.0])],
            residues=[
                ResidueRecord(
                    chain_id="A",
                    resseq=1,
                    icode="",
                    resname="ALA",
                    atoms={"CA": torch.tensor([1.0, 0.0, 0.0]), "CB": torch.tensor([1.5, 0.0, 0.0])},
                ),
                ResidueRecord(
                    chain_id="A",
                    resseq=2,
                    icode="",
                    resname="GLU",
                    atoms={
                        "CA": torch.tensor([3.0, 0.0, 0.0]),
                        "CB": torch.tensor([3.5, 0.0, 0.0]),
                        "OE1": torch.tensor([3.5, 0.5, 0.0]),
                        "OE2": torch.tensor([3.5, -0.5, 0.0]),
                    },
                ),
            ],
            metadata={"ring_edges_path": str(ring_path)},
        )

        default_graph = pocket_to_pyg_data(pocket, esm_dim=2)
        ring_idx = EDGE_SOURCE_TO_INDEX["ring"]
        if int((default_graph.edge_source_type[:, ring_idx] > 0.5).sum().item()) != 0:
            raise AssertionError("Default graph construction used RING edges; expected radius-only.")

        ring_graph = pocket_to_pyg_data(pocket, esm_dim=2, use_ring_edges=True)
        if int((ring_graph.edge_source_type[:, ring_idx] > 0.5).sum().item()) == 0:
            raise AssertionError("--use-ring-edges path did not include available RING edges.")


def check_metal_node_graph_and_gvp_forward() -> None:
    from data_structures import NODE_TYPE_GENERIC_METAL, NODE_TYPE_RESIDUE
    from graph.construction import pocket_to_pyg_data
    from model_variants import build_pocket_classifier
    from training.graph_dataset import (
        PocketGraphDataset,
        compute_feature_normalization_stats,
    )

    pocket = PocketRecord(
        structure_id="metal_node_fixture",
        pocket_id="metal_node_fixture_site0",
        metal_element="ZN",
        metal_coords=[torch.tensor([0.0, 0.0, 0.0])],
        residues=[
            ResidueRecord(
                chain_id="A",
                resseq=1,
                icode="",
                resname="CYS",
                atoms={
                    "CA": torch.tensor([1.0, 0.0, 0.0]),
                    "CB": torch.tensor([1.1, 0.0, 0.0]),
                    "SG": torch.tensor([1.5, 0.0, 0.0]),
                },
            ),
            ResidueRecord(
                chain_id="A",
                resseq=2,
                icode="",
                resname="HIS",
                atoms={
                    "CA": torch.tensor([0.0, 1.0, 0.0]),
                    "CB": torch.tensor([0.0, 1.1, 0.0]),
                    "ND1": torch.tensor([0.0, 1.5, 0.0]),
                },
            ),
            ResidueRecord(
                chain_id="A",
                resseq=3,
                icode="",
                resname="ASP",
                atoms={
                    "CA": torch.tensor([0.0, 0.0, 1.0]),
                    "CB": torch.tensor([0.0, 0.0, 1.1]),
                    "OD1": torch.tensor([0.0, 0.0, 1.5]),
                },
            ),
        ],
        y_metal=0,
    )

    graph = pocket_to_pyg_data(pocket, esm_dim=2, metal_node_mode="per_metal")
    if graph.num_nodes != 4:
        raise AssertionError(f"Expected 3 residue nodes + 1 metal node, got {graph.num_nodes}.")
    if int(graph.residue_node_mask.sum().item()) != 3 or int(graph.metal_node_mask.sum().item()) != 1:
        raise AssertionError(
            f"Unexpected residue/metal masks: {graph.residue_node_mask.tolist()}, {graph.metal_node_mask.tolist()}"
        )
    if graph.node_type_id[:3].tolist() != [NODE_TYPE_RESIDUE] * 3:
        raise AssertionError(f"Residue node_type_id values are wrong: {graph.node_type_id.tolist()}")
    if int(graph.node_type_id[3].item()) != NODE_TYPE_GENERIC_METAL:
        raise AssertionError(f"Metal node_type_id should be generic metal, got {graph.node_type_id.tolist()}")
    if not torch.allclose(graph.x_reschem[3], torch.zeros_like(graph.x_reschem[3])):
        raise AssertionError("Generic metal node leaked residue/element chemistry into x_reschem.")
    edge_pairs = set(zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist()))
    for residue_idx in range(3):
        if (residue_idx, 3) not in edge_pairs or (3, residue_idx) not in edge_pairs:
            raise AssertionError(f"Missing bidirectional residue-metal edge for residue {residue_idx}: {edge_pairs}")
    angle_stats = graph.site_ligand_angle_stats.view(-1)
    if int(angle_stats[0].item()) != 3 or int(angle_stats[1].item()) != 3:
        raise AssertionError(f"Unexpected ligand/angle counts in site_ligand_angle_stats: {angle_stats.tolist()}")
    if not 89.0 <= float(angle_stats[3].item()) <= 91.0:
        raise AssertionError(f"Expected mean ligand-metal-ligand angle near 90 degrees, got {angle_stats.tolist()}")

    normalization_stats = compute_feature_normalization_stats([graph], clamp_value=5.0)
    normalized_graph = PocketGraphDataset(
        [pocket],
        esm_dim=2,
        normalization_stats=normalization_stats,
        precomputed_data=[graph],
        metal_node_mode="per_metal",
    )[0]
    metal_mask = normalized_graph.metal_node_mask.to(dtype=torch.bool)
    for field_name in ("hydrophobicity_kd", "x_dist_raw", "x_misc", "x_env_burial", "x_env_electrostatics"):
        values = getattr(normalized_graph, field_name)[metal_mask]
        if not torch.allclose(values, torch.zeros_like(values)):
            raise AssertionError(f"Metal dummy node feature {field_name} was not kept at zero after normalization.")

    batch = next(iter(DataLoader([normalized_graph], batch_size=1)))
    model = build_pocket_classifier(
        model_architecture="only_gvp",
        esm_dim=2,
        hidden_s=16,
        hidden_v=4,
        edge_hidden=8,
        n_layers=1,
        n_metal=6,
        n_ec=1,
        predict_metal=True,
        predict_ec=False,
        structural_readout_scope="residue_and_metal",
        use_node_type_embedding=True,
        use_site_angle_features=True,
    )
    outputs = model(batch)
    if tuple(outputs["logits_metal"].shape) != (1, 6):
        raise AssertionError(f"Unexpected metal logits shape for metal-node GVP: {outputs['logits_metal'].shape}")
    if "loss" not in outputs:
        raise AssertionError("Metal-node GVP forward pass did not compute supervised loss.")


def check_colab_notebook_sweep_source() -> None:
    notebook_path = REPO_ROOT / "notebooks" / "DeepMzyme_training_colab.ipynb"
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in nb.get("cells", []))
    required_tokens = (
        '"run_training": False',
        'LAUNCH_PLANNED_MAIN_TRAINING_RUNS = False',
        'LAUNCH_PLANNED_TRAINING_RUNS = bool(LAUNCH_PLANNED_MAIN_TRAINING_RUNS)',
        'INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False',
        'LAUNCH_FINAL_HELD_OUT_TEST_EVAL = False',
        'DATASET_NAME = "train_and_test_sets_structures_exact_pinmymetal"',
        "train_and_test_sets_structures_common_pdbid_70_30_pinmymetal",
        "MODEL_PRESET =",
        'RING_EDGE_MODE = "with_ring"',
        'METAL_NODE_MODE = "per_metal"',
        'STRUCTURAL_READOUT_SCOPE = "auto"',
        'CLASSIFIER_POOL_DISTANCE_CUTOFF_VALUES_CSV = "0.0"',
        'ALLOW_MISSING_EXTERNAL_FEATURES = False',
        'OMIT_NODE_FEATURE_SETS = ""',
        'MAX_CONFIGURATION_RUNS',
        "CONFIG = {",
        "COLAB_DATA_SOURCE",
        "huggingface_link",
        "DeepMzyme_Data_v2.tar.zst",
        "12181d6bd7cb8e853cc0ea1d69dc50482dffe60392ad97089ccb3a5466059ba3",
        "site-level MAHOMES summary CSV",
        "structure-level inspection CSV",
        "MODEL_PRESET_MAP",
        "Only-GVP",
        "GVP + cross-modal attention",
        "SimpleGNN + ESM",
        "def parse_omit_node_feature_sets",
        "validate_node_feature_omissions",
        "def build_train_command",
        "ring_mode",
        "metal_node_mode",
        "structural_readout_scope",
        'HEAD_MLP_DROPOUT_VALUES_CSV = "0.2,0.3"',
        'ESM_GRAPH_ENCODER_DROPOUT_VALUES_CSV = "0.1,0.2"',
        'OUTER_RESIDUE_DROPOUTS_CSV = "0.0,0.1"',
        "omit_node_features",
        "--omit-node-features",
        "--head-mlp-dropout",
        "--esm-graph-encoder-dropout",
        "--outer-residue-dropout",
        "--use-ring-edges",
        "--ring-features-dir",
        "--metal-node-mode",
        "--structural-readout-scope",
        "--prepare-missing-ring-edges",
        "--no-prepare-missing-ring-edges",
        "--no-prepare-missing-esm-embeddings",
        'METAL_COLLAPSED_LOSS_WEIGHTS_CSV = "0.0"',
        "OPTUNA_MULTIOBJECTIVE = False",
        'RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = True',
        'TOP_CONFIG_REEVALUATION_MODE = "group_kfold_seed_repeat"',
        'TOP_K_CONFIGS_FOR_SEED_AND_CROSS_FOLD_REPEAT = "auto"',
        'USE_EXISTING_OPTUNA_TRIALS_FOR_STAGE6 = False',
        'EXISTING_OPTUNA_TRIALS_RUNS_DIR = ""',
        "Stage 6 controls and existing Optuna/HPO reuse",
        "Existing HPO candidate source",
        "Run All-safe no-op",
        "Rerun Build planned configuration commands after changing these controls.",
        "SEED_REPEAT_N_FOLDS = 5",
        "SEED_REPEAT_SPLIT_SEED = 42",
        "completed_trial_count < 50",
        "completed_trial_count < 150",
        "min(20, completed_trial_count)",
        "stage6_existing_trials_import_report.csv",
        "stage6_existing_trials_import_report.json",
        "original_validation_metric",
        "original_val_metal_balanced_acc",
        "import_existing_optuna_trials_for_stage6",
        "selected_from_imported_existing_optuna_hpo_trials",
        "total extra runs",
        "validation-only Stage 6",
        "def build_summary_basename",
        "AUTO_SUMMARY_BASENAME",
        "SUMMARY_BASENAME_WARNINGS",
        "summary_metadata_json",
        "stage6_ranked_candidates.csv",
        "stage6_selected_final_candidate.json",
        '"DATASET_NAME":',
        '"RESOLVED_DATASET_ROOT":',
        '"TRAIN_DIR":',
        '"TEST_DIR":',
        '"TRAIN_SITE_SUMMARY_CSV":',
        '"TEST_SITE_SUMMARY_CSV":',
        '"SPLIT_BY":',
        '"VAL_FRACTION":',
        '"SPLIT_SEED":',
        '"N_FOLDS":',
        '"FOLD_INDEX":',
        '"RUN_BATCH_ID":',
        'FINAL_TEST_WORKFLOW = "evaluate_stage6_selected_candidate"',
        "evaluate_stage6_selected_candidate",
        "exploratory_evaluate_all_stage6_ranked_candidates",
        "exploratory_final_test_all_stage6_ranked_candidates.csv",
        "exploratory_final_test_all_stage6_ranked_candidates.json",
        "exploratory_final_test_warning.txt",
        "primary_preselected",
        "exploratory_posthoc",
        "held-out test performance",
        "Do not choose the best held-out test result as the primary model.",
        "RING_EXE_PATH",
        '"src" / "report_runs.py"',
        "Recommended First Runs",
    )
    missing = [token for token in required_tokens if token not in source]
    if missing:
        raise AssertionError(f"Colab notebook source is missing expected tokens: {missing}")
    forbidden_tokens = ("ipywidgets", "widgets.", "MAX_SWEEP_RUNS", "RUN_SWEEP_MODE")
    present = [token for token in forbidden_tokens if token in source]
    if present:
        raise AssertionError(f"Colab notebook still contains retired widget/old-runner tokens: {present}")
    forbidden_fallback_tokens = ("choose_dataset_root_fallback", "FALLBACK_DATASET_NAME")
    present_fallbacks = [token for token in forbidden_fallback_tokens if token in source]
    if present_fallbacks:
        raise AssertionError(f"Colab notebook still contains exact-split fallback tokens: {present_fallbacks}")
    forbidden_final_test_tokens = (
        "EXPLORATORY_FINAL_TEST_TOP_K",
        "EXPLORATORY_FINAL_TEST_CANDIDATE_SCOPE",
        "ALLOW_EXPLORATORY_FINAL_TEST_BATCH",
        "exploratory_batch_evaluate_stage6_ranked_candidates",
        "loop_over_trials_results",
    )
    present_final_test_tokens = [token for token in forbidden_final_test_tokens if token in source]
    if present_final_test_tokens:
        raise AssertionError(f"Colab notebook still exposes retired final-test controls: {present_final_test_tokens}")
    if "sweep" in source.lower():
        raise AssertionError("Colab notebook should not contain user-facing sweep terminology.")


def check_colab_final_test_workflow_controls() -> None:
    notebook_path = REPO_ROOT / "notebooks" / "DeepMzyme_training_colab.ipynb"
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    final_source = next(
        (
            "".join(cell.get("source", []))
            for cell in nb.get("cells", [])
            if cell.get("cell_type") == "code"
            and "#@title Optional final held-out test evaluation" in "".join(cell.get("source", []))
        ),
        None,
    )
    if final_source is None:
        raise AssertionError("Could not find the Colab final held-out test cell.")

    expected_workflow_line = (
        'FINAL_TEST_WORKFLOW = "evaluate_stage6_selected_candidate"  #@param '
        '["evaluate_stage6_selected_candidate", "exploratory_evaluate_all_stage6_ranked_candidates"]'
    )
    if expected_workflow_line not in final_source:
        raise AssertionError("FINAL_TEST_WORKFLOW default or dropdown choices changed unexpectedly.")
    if 'LAUNCH_FINAL_HELD_OUT_TEST_EVAL = False  #@param {type:"boolean"}' not in final_source:
        raise AssertionError("Final held-out test launch is not disabled by default.")

    forbidden = (
        "EXPLORATORY_FINAL_TEST_TOP_K",
        "EXPLORATORY_FINAL_TEST_CANDIDATE_SCOPE",
        "ALLOW_EXPLORATORY_FINAL_TEST_BATCH",
    )
    present = [token for token in forbidden if token in final_source]
    if present:
        raise AssertionError(f"Final-test cell still exposes retired exploratory controls: {present}")

    required_tokens = (
        "Exploratory all-candidates mode requires stage6_ranked_candidates.csv",
        "Exploratory all-candidates mode requires stage6_selected_final_candidate.json",
        'role = "primary_preselected" if rank == 1 else "exploratory_posthoc"',
        "stage6_ranked_candidates.csv: {ranked_path}",
        "stage6_selected_final_candidate.json: {selected_path}",
        "Stage-6 selection will not be changed.",
        "exploratory_final_test_all_stage6_ranked_candidates.csv",
        "exploratory_final_test_all_stage6_ranked_candidates.json",
        "exploratory_final_test_warning.txt",
        "Stage 6 rank order; not sorted by held-out test performance",
    )
    missing = [token for token in required_tokens if token not in final_source]
    if missing:
        raise AssertionError(f"Final-test cell is missing expected workflow safeguards: {missing}")

    protected_write_tokens = (
        "stage6_selected_final_candidate_json.write_text",
        "stage6_ranked_candidates_csv.write_text",
        "stage6_selected_final_candidate.json\", \"w\"",
        "stage6_ranked_candidates.csv\", \"w\"",
    )
    present_writes = [token for token in protected_write_tokens if token in final_source]
    if present_writes:
        raise AssertionError(f"Final-test cell appears to modify Stage 6 source files: {present_writes}")


def check_colab_notebook_provenance_helpers() -> None:
    notebook_path = REPO_ROOT / "notebooks" / "DeepMzyme_training_colab.ipynb"
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    config_source = next(
        (
            "".join(cell.get("source", []))
            for cell in nb.get("cells", [])
            if cell.get("cell_type") == "code" and "def build_summary_basename" in "".join(cell.get("source", []))
        ),
        None,
    )
    if config_source is None:
        raise AssertionError("Could not find the Colab provenance helper cell.")
    helper_source = config_source.split("TOP_K_CONFIGS_FOR_SEED_REPEAT_RESOLVED =", 1)[0]
    namespace: dict[str, object] = {}
    exec(helper_source, namespace)
    basename = namespace["build_summary_basename"](
        "metal",
        "five_class",
        "GVP + hybrid fusion",
        "train_and_test_sets_structures_exact_pinmymetal",
        "pleasev3",
        "pdbid",
        "stage6-groupkfold",
    )
    required_fragments = ("metal", "five_class", "gvp_hybrid", "exact_pinmymetal", "batch-pleasev3", "split-pdbid", "stage6-groupkfold")
    missing = [fragment for fragment in required_fragments if fragment not in basename]
    if missing:
        raise AssertionError(f"Generated SUMMARY_BASENAME is missing provenance fragments {missing}: {basename}")

    warnings = namespace["summary_basename_consistency_warnings"](
        manual_summary_basename="splitpocket_single",
        dataset_name="train_and_test_sets_structures_exact_pinmymetal",
        run_batch_id="pleasev3",
        split_by="pdbid",
    )
    if len(warnings) < 3:
        raise AssertionError(f"Manual stale SUMMARY_BASENAME did not trigger strong provenance warnings: {warnings}")

    top_k_auto = namespace["parse_top_k_configs_for_seed_repeat"]("auto")
    top_k_int = namespace["parse_top_k_configs_for_seed_repeat"]("20")
    if top_k_auto != "auto" or top_k_int != 20:
        raise AssertionError(f"TOP_K parser failed for auto/integer values: {top_k_auto!r}, {top_k_int!r}")


def check_colab_exact_split_no_fallback() -> None:
    notebook_path = REPO_ROOT / "notebooks" / "DeepMzyme_training_colab.ipynb"
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    dataset_source = next(
        (
            "".join(cell.get("source", []))
            for cell in nb.get("cells", [])
            if cell.get("cell_type") == "code" and "def find_dataset_root" in "".join(cell.get("source", []))
        ),
        None,
    )
    if dataset_source is None:
        raise AssertionError("Could not find the Colab dataset-resolution cell.")

    with tempfile.TemporaryDirectory(prefix="deepmzyme_exact_split_guard_") as tmp:
        tmp_root = Path(tmp)
        common_split = tmp_root / "train_and_test_sets_structures_common_pdbid_70_30_pinmymetal"
        (common_split / "train").mkdir(parents=True)
        (common_split / "test").mkdir()
        namespace = {
            "CONFIG": {
                "data": {
                    "dataset_name": "train_and_test_sets_structures_exact_pinmymetal",
                    "train_dir_override": "",
                    "test_dir_override": "",
                    "train_csv_override": "",
                    "test_csv_override": "",
                }
            },
            "DATA_ROOT_CANDIDATES": [tmp_root],
        }
        try:
            exec(dataset_source, namespace)
        except FileNotFoundError as exc:
            message = str(exc)
            required_phrases = (
                "Exact PinMyMetal split was requested",
                "train_and_test_sets_structures_exact_pinmymetal",
                "train_and_test_sets_structures_common_pdbid_70_30_pinmymetal",
                "explicitly rerun the Main configuration cell with DATASET_NAME='train_and_test_sets_structures_common_pdbid_70_30_pinmymetal'",
            )
            missing = [phrase for phrase in required_phrases if phrase not in message]
            if missing:
                raise AssertionError(f"Exact-split guard error is missing {missing}: {message}") from exc
        else:
            raise AssertionError("Exact split request silently resolved despite only the Common-PDBID split being present.")


def check_colab_generated_training_commands_parse() -> None:
    notebook_path = REPO_ROOT / "notebooks" / "DeepMzyme_training_colab.ipynb"
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    command_builder_source = next(
        (
            "".join(cell.get("source", []))
            for cell in nb.get("cells", [])
            if cell.get("cell_type") == "code" and "def build_train_command" in "".join(cell.get("source", []))
        ),
        None,
    )
    if command_builder_source is None:
        raise AssertionError("Could not find the Colab command-builder cell.")
    optuna_runner_source = next(
        (
            "".join(cell.get("source", []))
            for cell in nb.get("cells", [])
            if cell.get("cell_type") == "code" and "def sample_optuna_config" in "".join(cell.get("source", []))
        ),
        None,
    )
    if optuna_runner_source is None:
        raise AssertionError("Could not find the Colab Optuna runner cell.")

    import contextlib
    import copy
    import io

    def base_config(tmp_root: Path) -> dict[str, object]:
        return {
            "basic": {
                "task": "metal",
                "epochs": 1,
                "run_training": False,
                "device": "cpu",
                "run_held_out_test_eval": False,
            },
            "configuration_comparison": {
                "recommended_run_set": "custom",
                "model_preset": "Only-GVP",
                "ring_edge_mode": "with_ring",
                "batch_sizes_csv": "4",
                "learning_rates_csv": "1e-4",
                "weight_decays_csv": "1e-4",
                "seeds_csv": "42",
                "max_configuration_runs": 24,
                "stop_on_first_failure": True,
                "skip_existing_runs": True,
                "allow_model_preset_mismatch": False,
                "allow_single_mode_to_truncate_comparison": False,
            },
            "optuna": {
                "intensity": "debug",
                "search_preset": "first_useful_only_gvp_narrow",
                "n_trials": 4,
                "timeout_minutes": 0,
                "max_epochs_per_trial": 3,
                "multiobjective": False,
                "selection_metric": "task_default",
                "direction": "maximize",
                "study_name": "smoke_optuna",
                "split_seed": 42,
                "sampler_seed": None,
                "storage": "",
                "allow_incompatible_study_reuse": False,
                "use_pruning": False,
                "pruner_type": "none",
                "pruning_min_epoch": 8,
                "pruner_min_resource": 8,
                "pruner_reduction_factor": 3,
                "auto_configure_budget": False,
                "retrain_best_config_after_hpo": False,
                "run_seed_repeat_evaluation": True,
                "top_k_configs_for_seed_repeat": "auto",
                "top_k": "auto",
                "repeat_seeds": "42",
                "top_config_reevaluation_mode": "group_kfold",
                "seed_repeat_n_folds": 5,
                "seed_repeat_split_seed": 42,
                "stage6_raw_improvement_threshold": 0.0,
                "allow_seed_repeat_model_preset_mismatch": False,
                "use_existing_optuna_trials_for_stage6": False,
                "existing_optuna_trials_runs_dir": "",
                "learning_rate_range": "1e-5,3e-4",
                "lr_schedules_csv": "fixed",
                "weight_decays_csv": "0.0,1e-5",
                "batch_sizes_csv": "4",
                "hidden_s_values_csv": "64,128",
                "hidden_v_values_csv": "8,16",
                "gvp_layers_values_csv": "2,3",
                "head_mlp_layers_values_csv": "1,2",
                "head_mlp_dropout_values_csv": "0.2",
                "esm_graph_encoder_dropout_values_csv": "0.1",
                "edge_hidden_values_csv": "32,64",
                "edge_radius_values_csv": "6.0,8.0",
                "esm_fusion_dim_values_csv": "64,128",
                "early_esm_dim_values_csv": "16,32",
                "cross_attention_layers_csv": "1",
                "cross_attention_heads_csv": "2,4",
                "metal_focal_gamma_values_csv": "1.5,2.0",
                "metal_loss_weight_values_csv": "1.0",
                "ec_loss_weight_values_csv": "1.0",
                "metal_class_weight_modes_csv": "none,inverse_frequency",
                "metal_loss_functions_csv": "cross_entropy",
                "metal_label_smoothing_values_csv": "0.0",
                "metal_collapsed_loss_weights_csv": "0.0",
                "balance_metal_site_symbols_csv": "False",
                "position_noise_stds_csv": "0.0",
                "second_shell_dropouts_csv": "0.0",
                "outer_residue_dropouts_csv": "0.0",
                "classifier_pool_distance_cutoff_values_csv": "0.0",
                "early_esm_dropout_values_csv": "0.0",
                "cross_attention_dropout_values_csv": "0.0",
            },
            "data": {
                "colab_data_source": "huggingface_link",
                "dataset_name": "train_and_test_sets_structures_exact_pinmymetal",
            },
            "esm": {
                "esm_embeddings_dir": "",
                "allow_missing_esm_embeddings": False,
                "prepare_missing_esm_embeddings": False,
                "esm_dim": 960,
            },
            "ring": {
                "ring_features_dir": str(tmp_root / "ring_features"),
                "ring_exe_path": str(tmp_root / "ring"),
                "require_ring_edges": False,
                "prepare_missing_ring_edges": True,
            },
            "node_features": {
                "node_feature_set": "conservative",
                "omit_node_feature_sets": "",
            },
            "advanced": {
                "val_fraction": 0.15,
                "split_by": "pdbid",
                "selection_metric": "val_metal_balanced_acc",
                "hidden_s_values_csv": "128",
                "hidden_v_values_csv": "16",
                "edge_hidden_values_csv": "64",
                "gvp_layers_values_csv": "4",
                "head_mlp_layers_values_csv": "2",
                "head_mlp_dropout": 0.2,
                "esm_graph_encoder_dropout": 0.1,
                "edge_radius_values_csv": "8.0",
                "esm_fusion_dim_values_csv": "128",
                "lr_schedules_csv": "fixed",
                "lr_step_size": 10,
                "lr_decay_gamma": 0.5,
                "node_rbf_sigma": 0.75,
                "edge_rbf_sigma": 0.75,
                "node_rbf_use_raw_distances": False,
                "metal_node_mode": "none",
                "structural_readout_scope": "auto",
                "classifier_pool_distance_cutoff_values_csv": "0.0",
                "position_noise_std": 0.0,
                "second_shell_dropout": 0.0,
                "outer_residue_dropout": 0.0,
                "early_esm_dim": 32,
                "early_esm_dropout": 0.2,
                "early_esm_raw": False,
                "early_esm_scope": "all",
                "cross_attention_layers_csv": "1",
                "cross_attention_heads_csv": "4",
                "cross_attention_dropout": 0.1,
                "cross_attention_neighborhood": "all",
                "cross_attention_bidirectional": False,
                "ec_label_depths_csv": "1",
                "ec_group_weighting": "structure_id",
                "ec_contrastive_weights_csv": "0.0",
                "ec_contrastive_temperature": 0.1,
                "metal_loss_function": "cross_entropy",
                "metal_focal_gamma": 2.0,
                "metal_label_smoothing": 0.0,
                "metal_collapsed_loss_weight": 0.0,
                "metal_loss_weight": 1.0,
                "ec_loss_weight": 1.0,
                "mn_loss_multiplier": 1.0,
                "cu_loss_multiplier": 1.0,
                "zn_loss_multiplier": 1.0,
                "fe_loss_multiplier": 1.0,
                "co_loss_multiplier": 1.0,
                "ni_loss_multiplier": 1.0,
                "class_viii_loss_multiplier": 1.0,
                "balance_metal_site_symbols": False,
                "require_all_task_classes": False,
                "allow_missing_external_features": False,
                "external_features_root_dir": "",
                "external_feature_source": "updated",
                "n_folds": "",
                "fold_index": "",
                "deterministic": False,
                "save_epoch_checkpoints": False,
                "allow_train_loss_test_eval_debug": False,
                "unsupported_metal_policy": "error",
                "invalid_structure_policy": "skip",
            },
            "output": {
                "runs_dir": str(tmp_root / "runs"),
                "run_name_prefix": "",
                "copy_outputs_to_drive": False,
                "summary_basename": "summary",
            },
        }

    def run_builder(config_updates: dict[str, dict[str, object]], *, return_namespace: bool = False):
        with tempfile.TemporaryDirectory(prefix="deepmzyme_colab_command_smoke_") as tmp:
            tmp_root = Path(tmp)
            config = base_config(tmp_root)
            for section, updates in config_updates.items():
                nested = config[section]
                if not isinstance(nested, dict):
                    raise AssertionError(f"Unexpected non-dict config section: {section}")
                nested.update(updates)
            ring_dir = Path(str(config["ring"]["ring_features_dir"]))
            ring_dir.mkdir(parents=True)
            (ring_dir / "example_ringEdges").write_text("NodeId1\tNodeId2\tInteraction\n", encoding="utf-8")
            ring_exe = Path(str(config["ring"]["ring_exe_path"]))
            ring_exe.write_text("#!/bin/sh\n", encoding="utf-8")
            ring_exe.chmod(0o755)
            train_dir = tmp_root / "train"
            test_dir = tmp_root / "test"
            train_dir.mkdir()
            test_dir.mkdir()
            train_csv = train_dir / "summary.csv"
            test_csv = test_dir / "summary.csv"
            train_csv.write_text("pdbid,metal residue number,EC number,metal residue type\n", encoding="utf-8")
            test_csv.write_text("pdbid,metal residue number,EC number,metal residue type\n", encoding="utf-8")
            namespace = {
                "CONFIG": config,
                "REPO_DIR": REPO_ROOT,
                "SRC_DIR": REPO_ROOT / "src",
                "TRAIN_DIR": train_dir,
                "TEST_DIR": test_dir,
                "TRAIN_CSV": train_csv,
                "TEST_CSV": test_csv,
                "TRAIN_SITE_SUMMARY_CSV": train_csv,
                "TEST_SITE_SUMMARY_CSV": test_csv,
                "TRAIN_STRUCTURES": [],
                "TEST_STRUCTURES": [],
                "DATA_ROOT": tmp_root,
                "DATASET_ROOT": tmp_root / "dataset",
                "DRIVE_DATA_DIR": tmp_root / "drive" / "DeepMzyme_Data",
            }
            with contextlib.redirect_stdout(io.StringIO()):
                exec(command_builder_source, namespace)
            if return_namespace:
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(optuna_runner_source, namespace)
                return namespace
            return copy.deepcopy(namespace["planned_runs"])

    def assert_training_command_parses(cmd: list[object]) -> None:
        parts = [str(part) for part in cmd]
        if parts[1] != str(REPO_ROOT / "src" / "train.py"):
            raise AssertionError(f"Notebook command used an unexpected training entry point: {parts[:2]}")
        config = parse_args(parts[2:])
        validate_training_configuration(config)

    default_runs = run_builder({})
    if len(default_runs) != 1:
        raise AssertionError(f"Expected one default planned command, got {len(default_runs)}")
    default_cmd = [str(part) for part in default_runs[0]["command"]]
    assert_training_command_parses(default_cmd)
    for expected_flag in ("--use-ring-edges", "--ring-features-dir", "--prepare-missing-ring-edges"):
        if expected_flag not in default_cmd:
            raise AssertionError(f"Default graph command is missing {expected_flag}.")
    if "--allow-missing-external-features" in default_cmd:
        raise AssertionError("Default graph command should require updated external features.")
    if "--esm-embeddings-dir" in default_cmd:
        raise AssertionError("Only-GVP default command should not require an ESM embeddings directory.")
    if "--omit-node-features" in default_cmd:
        raise AssertionError("Full-feature default command unexpectedly omits node features.")
    if "--metal-node-mode" in default_cmd or "--structural-readout-scope" in default_cmd:
        raise AssertionError("Default graph command should leave metal-node mode disabled.")
    if "--head-mlp-dropout" not in default_cmd:
        raise AssertionError("Default notebook command did not record --head-mlp-dropout.")
    if default_cmd[default_cmd.index("--head-mlp-dropout") + 1] != "0.2":
        raise AssertionError("Default notebook command changed the head MLP dropout default.")
    if "--esm-graph-encoder-dropout" in default_cmd:
        raise AssertionError("Only-GVP default command should keep ESM graph encoder dropout inactive.")

    outer_dropout_runs = run_builder({"advanced": {"outer_residue_dropouts_csv": "0.4"}})
    outer_dropout_cmd = [str(part) for part in outer_dropout_runs[0]["command"]]
    assert_training_command_parses(outer_dropout_cmd)
    if "--outer-residue-dropout" not in outer_dropout_cmd:
        raise AssertionError("Outer-residue dropout notebook command did not pass --outer-residue-dropout.")
    if outer_dropout_cmd[outer_dropout_cmd.index("--outer-residue-dropout") + 1] != "0.4":
        raise AssertionError("Outer-residue dropout notebook command passed the wrong value.")

    late_fusion_runs = run_builder(
        {
            "configuration_comparison": {"model_preset": "GVP + late fusion"},
            "esm": {"allow_missing_esm_embeddings": True},
        }
    )
    late_fusion_cmd = [str(part) for part in late_fusion_runs[0]["command"]]
    assert_training_command_parses(late_fusion_cmd)
    if "--esm-graph-encoder-dropout" not in late_fusion_cmd:
        raise AssertionError("ESM graph encoder command did not record --esm-graph-encoder-dropout for late fusion.")
    if late_fusion_cmd[late_fusion_cmd.index("--esm-graph-encoder-dropout") + 1] != "0.1":
        raise AssertionError("Late-fusion notebook command changed the ESM graph encoder dropout default.")

    group_kfold_runs = run_builder(
        {"advanced": {"val_fraction": 0.0, "n_folds": 5, "fold_index": 2, "split_by": "pdbid", "split_seed": 42}}
    )
    if len(group_kfold_runs) != 1:
        raise AssertionError(f"Expected one group-kfold planned command, got {len(group_kfold_runs)}")
    group_kfold_cmd = [str(part) for part in group_kfold_runs[0]["command"]]
    assert_training_command_parses(group_kfold_cmd)
    expected_flag_values = {
        "--n-folds": "5",
        "--fold-index": "2",
        "--train-val-split-by": "pdbid",
        "--split-seed": "42",
        "--val-fraction": "0.0",
    }
    for expected_flag, expected_value in expected_flag_values.items():
        if expected_flag not in group_kfold_cmd:
            raise AssertionError(f"Stage 6 group-kfold command is missing {expected_flag}: {group_kfold_cmd}")
        actual_value = group_kfold_cmd[group_kfold_cmd.index(expected_flag) + 1]
        if actual_value != expected_value:
            raise AssertionError(
                f"Stage 6 group-kfold command passed {expected_flag} {actual_value}, expected {expected_value}."
            )
    if "--run-test-eval" in group_kfold_cmd or "--allow-train-loss-test-eval-debug" in group_kfold_cmd:
        raise AssertionError("Stage 6 group-kfold command unexpectedly includes held-out test evaluation.")

    metal_node_runs = run_builder({"advanced": {"metal_node_mode": "per_metal"}})
    if len(metal_node_runs) != 1:
        raise AssertionError(f"Expected one metal-node planned command, got {len(metal_node_runs)}")
    metal_node_cmd = [str(part) for part in metal_node_runs[0]["command"]]
    assert_training_command_parses(metal_node_cmd)
    if "--metal-node-mode" not in metal_node_cmd:
        raise AssertionError("Metal-node notebook command did not pass --metal-node-mode.")
    if metal_node_cmd[metal_node_cmd.index("--metal-node-mode") + 1] != "per_metal":
        raise AssertionError("Metal-node notebook command passed the wrong mode.")
    if "--structural-readout-scope" in metal_node_cmd:
        raise AssertionError("Default auto structural readout should not be passed explicitly.")

    metal_only_readout_runs = run_builder(
        {"advanced": {"metal_node_mode": "per_metal", "structural_readout_scope": "metal_only"}}
    )
    metal_only_readout_cmd = [str(part) for part in metal_only_readout_runs[0]["command"]]
    assert_training_command_parses(metal_only_readout_cmd)
    if "--structural-readout-scope" not in metal_only_readout_cmd:
        raise AssertionError("Metal-only readout notebook command did not pass --structural-readout-scope.")
    if metal_only_readout_cmd[metal_only_readout_cmd.index("--structural-readout-scope") + 1] != "metal_only":
        raise AssertionError("Metal-only readout notebook command passed the wrong scope.")

    pool_cutoff_runs = run_builder(
        {
            "basic": {"run_mode": "manual_configurations"},
            "advanced": {"classifier_pool_distance_cutoff_values_csv": "0.0,6.0"},
        }
    )
    if len(pool_cutoff_runs) != 2:
        raise AssertionError(f"Expected two classifier-pool cutoff planned commands, got {len(pool_cutoff_runs)}")
    pool_cutoff_values = [float(run["classifier_pool_distance_cutoff"]) for run in pool_cutoff_runs]
    if pool_cutoff_values != [0.0, 6.0]:
        raise AssertionError(f"Classifier-pool cutoff grid produced wrong values: {pool_cutoff_values}")
    pool_cutoff_cmds = [[str(part) for part in run["command"]] for run in pool_cutoff_runs]
    for expected_value, command in zip(("0.0", "6.0"), pool_cutoff_cmds):
        assert_training_command_parses(command)
        if "--classifier-pool-distance-cutoff" not in command:
            raise AssertionError("Classifier-pool cutoff command did not pass --classifier-pool-distance-cutoff.")
        actual_value = command[command.index("--classifier-pool-distance-cutoff") + 1]
        if actual_value != expected_value:
            raise AssertionError(f"Classifier-pool cutoff command passed {actual_value}, expected {expected_value}.")

    optuna_namespace = run_builder(
        {
            "basic": {"run_mode": "controlled_hpo_optuna"},
            "optuna": {
                "search_preset": "custom",
                "selection_metric": "val_metal_balanced_acc",
                "classifier_pool_distance_cutoff_values_csv": "0.0,6.0",
            },
        },
        return_namespace=True,
    )

    class FakeTrial:
        number = 7

        def suggest_float(self, name, low, high, log=False):
            return low

        def suggest_categorical(self, name, values):
            return values[-1]

    trial_config, sampled = optuna_namespace["sample_optuna_config"](
        FakeTrial(),
        optuna_namespace["all_planned_runs"][0],
    )
    if float(sampled.get("classifier_pool_distance_cutoff")) != 6.0:
        raise AssertionError(f"Optuna did not sample classifier_pool_distance_cutoff: {sampled}")
    optuna_cmd = [str(part) for part in trial_config["command"]]
    assert_training_command_parses(optuna_cmd)
    actual_optuna_cutoff = optuna_cmd[optuna_cmd.index("--classifier-pool-distance-cutoff") + 1]
    if actual_optuna_cutoff != "6.0":
        raise AssertionError(f"Optuna command passed classifier-pool cutoff {actual_optuna_cutoff}, expected 6.0.")

    with tempfile.TemporaryDirectory(prefix="deepmzyme_stage6_import_") as tmp:
        tmp_root = Path(tmp)
        existing_dir = tmp_root / "existing_hpo"
        existing_dir.mkdir()
        stage6_optuna_dir = tmp_root / "stage6_outputs"
        stage6_optuna_dir.mkdir()
        imported_base = copy.deepcopy(optuna_namespace["all_planned_runs"][0])
        imported_base.update(
            {
                "epochs": 50,
                "val_fraction": 0.15,
                "n_folds": None,
                "fold_index": None,
                "selection_metric": "val_metal_balanced_acc",
            }
        )

        def write_imported_trial(
            *,
            trial_number: int,
            validation_metric: float,
            min_recall: float,
            dataset_name: str | None = None,
            final_test: bool = False,
        ) -> Path:
            run_dir = existing_dir / f"optuna_existing_trial{trial_number:04d}"
            run_dir.mkdir()
            cfg = copy.deepcopy(imported_base)
            cfg.update(
                {
                    "run_name": run_dir.name,
                    "run_dir": str(run_dir),
                    "dataset_name": dataset_name or imported_base["dataset_name"],
                    "learning_rate": 1e-4 * trial_number,
                    "weight_decay": 1e-5,
                    "epochs": 50,
                    "n_folds": None,
                    "fold_index": None,
                    "val_fraction": 0.15,
                }
            )
            active_payload = {
                "source_mode": "optuna trial",
                "result_stage": "validation-only",
                "run_config": cfg,
                "extra": {
                    "trial_number": trial_number,
                    "sampled_params": {
                        "learning_rate": cfg["learning_rate"],
                        "weight_decay": cfg["weight_decay"],
                    },
                },
                "command": "python train.py --selection-metric val_metal_balanced_acc",
            }
            (run_dir / "active_run_config.json").write_text(json.dumps(active_payload, default=str), encoding="utf-8")
            (run_dir / "run_config.json").write_text(json.dumps({"config": cfg}, default=str), encoding="utf-8")
            (run_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "selection_metric": "val_metal_balanced_acc",
                        "selected_metric_value": validation_metric,
                        "selected_checkpoint": str(run_dir / "best_model_checkpoint.pt"),
                        "history": [
                            {
                                "epoch": 50,
                                "val_metal_balanced_acc": validation_metric,
                                "val_metal_min_recall": min_recall,
                                "val_loss": 1.0 - validation_metric,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "val_metrics.csv").write_text(
                "epoch,val_metal_balanced_acc,val_metal_min_recall,val_loss\n"
                f"50,{validation_metric},{min_recall},{1.0 - validation_metric}\n",
                encoding="utf-8",
            )
            if final_test:
                (run_dir / "test_report.json").write_text('{"test_metric": 0.99}', encoding="utf-8")
            return run_dir

        low_val_high_test = write_imported_trial(trial_number=1, validation_metric=0.61, min_recall=0.42)
        high_val_low_test = write_imported_trial(trial_number=2, validation_metric=0.70, min_recall=0.39)
        final_test_run = write_imported_trial(trial_number=3, validation_metric=0.99, min_recall=0.99, final_test=True)
        mismatch_run = write_imported_trial(
            trial_number=4,
            validation_metric=0.98,
            min_recall=0.98,
            dataset_name="train_and_test_sets_structures_non_overlapped_pinmymetal",
        )
        (existing_dir / "top_trials.csv").write_text(
            "rank,candidate_id,trial_number,state,run_dir,selection_metric,validation_metric,val_metal_balanced_acc,val_metal_min_recall,held_out_test_metric_value\n"
            f"1,trial0001,1,COMPLETE,{low_val_high_test},val_metal_balanced_acc,0.61,0.61,0.42,0.99\n"
            f"2,trial0002,2,COMPLETE,{high_val_low_test},val_metal_balanced_acc,0.70,0.70,0.39,0.10\n"
            f"3,trial0003,3,COMPLETE,{final_test_run},val_metal_balanced_acc,0.99,0.99,0.99,0.99\n"
            f"4,trial0004,4,COMPLETE,{mismatch_run},val_metal_balanced_acc,0.98,0.98,0.98,0.98\n",
            encoding="utf-8",
        )

        import_optuna_config = copy.deepcopy(optuna_namespace["CONFIG"]["optuna"])
        import_optuna_config.update(
            {
                "study_name": "stage6_import_smoke",
                "top_k_configs_for_seed_repeat": "auto",
                "top_k": "auto",
                "seed_repeat_n_folds": 5,
                "seed_repeat_split_seed": 42,
            }
        )
        try:
            optuna_namespace["import_existing_optuna_trials_for_stage6"](
                tmp_root / "missing",
                imported_base,
                import_optuna_config,
                stage6_optuna_dir,
            )
        except RuntimeError as exc:
            if "does not exist" not in str(exc):
                raise AssertionError(f"Missing existing-trials directory failed unclearly: {exc}") from exc
        else:
            raise AssertionError("Missing existing-trials directory was accepted.")

        with contextlib.redirect_stdout(io.StringIO()):
            import_result = optuna_namespace["import_existing_optuna_trials_for_stage6"](
                existing_dir,
                imported_base,
                import_optuna_config,
                stage6_optuna_dir,
            )
        selected = import_result["selected_candidates"]
        if len(selected) != 2:
            raise AssertionError(f"Expected two compatible imported candidates, got {len(selected)}")
        if selected[0]["trial_number"] != 2:
            raise AssertionError("Imported candidates were not ranked by validation metrics only.")
        report_text = (stage6_optuna_dir / "stage6_existing_trials_import_report.csv").read_text(encoding="utf-8")
        if "held-out test artifact" not in report_text or "metadata mismatch" not in report_text:
            raise AssertionError("Import report did not record skipped final-test and incompatible-dataset candidates.")

        reevaluation_units = [
            {"validation_unit": "fold_0", "model_seed": 42, "split_seed": 42, "n_folds": 5, "fold_index": 0}
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            top_rows, top_payload, rerun_configs, _commands = optuna_namespace["build_stage6_imported_rerun_configs"](
                import_result,
                imported_base,
                import_optuna_config,
                stage6_optuna_dir,
                reevaluation_units,
            )
        if len(top_rows) != 2 or len(top_payload) != 2 or len(rerun_configs) != 2:
            raise AssertionError("Imported Stage 6 top-K command generation produced the wrong number of rows.")
        for rerun_config in rerun_configs:
            cmd = [str(part) for part in rerun_config["command"]]
            for expected_flag, expected_value in {
                "--n-folds": "5",
                "--fold-index": "0",
                "--train-val-split-by": "pdbid",
                "--split-seed": "42",
                "--val-fraction": "0.0",
            }.items():
                if expected_flag not in cmd:
                    raise AssertionError(f"Imported Stage 6 command is missing {expected_flag}: {cmd}")
                if cmd[cmd.index(expected_flag) + 1] != expected_value:
                    raise AssertionError(f"Imported Stage 6 command passed wrong {expected_flag}: {cmd}")
            if "--run-test-eval" in cmd:
                raise AssertionError("Imported Stage 6 command unexpectedly includes held-out test evaluation.")
            if "original_source_run_dir" not in rerun_config:
                raise AssertionError("Imported Stage 6 rerun config did not preserve original source run directory.")

        records = []
        for rerun_config in rerun_configs:
            fold_metric = 0.72 if rerun_config["original_trial_number"] == 2 else 0.60
            records.append(
                {
                    "candidate_id": rerun_config["candidate_id"],
                    "imported_candidate_id": rerun_config["imported_candidate_id"],
                    "source_top_rank": rerun_config["seed_repeat_rank"],
                    "source_trial_number": rerun_config["seed_repeat_trial_number"],
                    "original_trial_number": rerun_config["original_trial_number"],
                    "original_source_run_dir": rerun_config["original_source_run_dir"],
                    "original_validation_metric": rerun_config["original_validation_metric"],
                    "original_val_metal_balanced_acc": rerun_config["original_val_metal_balanced_acc"],
                    "original_val_metal_min_recall": rerun_config["original_val_metal_min_recall"],
                    "original_val_loss": rerun_config["original_val_loss"],
                    "run_dir": rerun_config["run_dir"],
                    "selected_checkpoint": str(Path(rerun_config["run_dir"]) / "best_model_checkpoint.pt"),
                    "validation_unit": rerun_config["validation_unit"],
                    "selected_best_validation_metric_value": fold_metric,
                    "val_metal_balanced_acc": fold_metric,
                    "val_metal_min_recall": 0.40,
                    "model_preset": rerun_config["model_preset"],
                    "model_architecture": rerun_config["model_architecture"],
                    "fusion_mode": rerun_config["fusion_mode"],
                    "learning_rate": rerun_config["learning_rate"],
                    "weight_decay": rerun_config["weight_decay"],
                    "batch_size": rerun_config["batch_size"],
                    "hidden_s": rerun_config["hidden_s"],
                    "hidden_v": rerun_config["hidden_v"],
                    "edge_hidden": rerun_config["edge_hidden"],
                    "gvp_layers": rerun_config["gvp_layers"],
                    "head_mlp_layers": rerun_config["head_mlp_layers"],
                    "metal_class_weight_mode": rerun_config["metal_class_weight_mode"],
                    "metal_loss_function": rerun_config["metal_loss_function"],
                    "metal_label_smoothing": rerun_config["metal_label_smoothing"],
                }
            )
        with contextlib.redirect_stdout(io.StringIO()):
            optuna_namespace["write_stage6_imported_summary_outputs"](
                records,
                stage6_optuna_dir,
                import_optuna_config,
                reevaluation_units,
                import_result,
                stage6_optuna_dir / "top_trial_configs.json",
                stage6_optuna_dir / "top_trials.csv",
                stage6_optuna_dir / "seed_repeat_results.csv",
                stage6_optuna_dir / "seed_repeat_summary.csv",
                stage6_optuna_dir / "seed_repeat_summary.json",
                stage6_optuna_dir / "seed_repeat_pairwise_bootstrap.csv",
                stage6_optuna_dir / "seed_repeat_pairwise_bootstrap.json",
                stage6_optuna_dir / "stage6_ranked_candidates.csv",
                stage6_optuna_dir / "stage6_selected_final_candidate.json",
                0.0,
            )
        selected_payload = json.loads((stage6_optuna_dir / "stage6_selected_final_candidate.json").read_text(encoding="utf-8"))
        ranked_text = (stage6_optuna_dir / "stage6_ranked_candidates.csv").read_text(encoding="utf-8")
        if "original_validation_metric" not in ranked_text or "original_val_metal_balanced_acc" not in ranked_text:
            raise AssertionError("Stage 6 ranked candidates CSV did not include original imported validation metrics.")
        if not selected_payload.get("selected_from_imported_existing_optuna_hpo_trials"):
            raise AssertionError("Stage 6 selected-candidate JSON did not record imported-HPO source mode.")
        if selected_payload.get("held_out_test_metrics_used") is not False:
            raise AssertionError("Stage 6 selected-candidate JSON did not explicitly exclude held-out test metrics.")

    collapsed_loss_runs = run_builder({"advanced": {"metal_collapsed_loss_weights_csv": "0.3"}})
    collapsed_loss_cmd = [str(part) for part in collapsed_loss_runs[0]["command"]]
    assert_training_command_parses(collapsed_loss_cmd)
    if "--metal-collapsed-loss-weight" not in collapsed_loss_cmd:
        raise AssertionError("Collapsed-loss notebook command did not pass --metal-collapsed-loss-weight.")
    if collapsed_loss_cmd[collapsed_loss_cmd.index("--metal-collapsed-loss-weight") + 1] != "0.3":
        raise AssertionError("Collapsed-loss notebook command passed the wrong alpha value.")

    radius_only_runs = run_builder({"configuration_comparison": {"ring_edge_mode": "without_ring"}})
    if len(radius_only_runs) != 1:
        raise AssertionError(f"Expected one radius-only ablation command, got {len(radius_only_runs)}")
    radius_only_cmd = [str(part) for part in radius_only_runs[0]["command"]]
    if "--use-ring-edges" in radius_only_cmd or "--require-ring-edges" in radius_only_cmd:
        raise AssertionError("Explicit radius-only ablation command unexpectedly enables RING edges.")

    omit_runs = run_builder(
        {
            "basic": {"run_mode": "manual_configurations"},
            "node_features": {"omit_node_feature_sets": ";v_cb_to_fg;v_cb_to_fg,v_res_to_metal"},
        }
    )
    if len(omit_runs) != 3:
        raise AssertionError(f"Expected three omission planned commands, got {len(omit_runs)}")
    omit_cmds = [[str(part) for part in run["command"]] for run in omit_runs]
    if "--omit-node-features" in omit_cmds[0]:
        raise AssertionError("Full-feature omission entry unexpectedly passed --omit-node-features.")
    if omit_cmds[1][omit_cmds[1].index("--omit-node-features") + 1] != "v_cb_to_fg":
        raise AssertionError("Single-feature omission command passed the wrong value.")
    if omit_cmds[2][omit_cmds[2].index("--omit-node-features") + 1] != "v_cb_to_fg,v_res_to_metal":
        raise AssertionError("Combined-feature omission command passed the wrong value.")


def check_ring_environment_overrides() -> None:
    from graph.ring_edges import canonical_ring_edges_output_path
    from embed_helpers.Interaction_edge import DEFAULT_RING_EXE, create_ring_edges_batch
    from training.runtime_preparation import prepare_runtime_inputs

    personal_ring_path = str(Path("/home") / "mechti" / "ring-4.0" / "out" / "bin" / "ring")
    if DEFAULT_RING_EXE.is_absolute():
        raise AssertionError(f"RING default fallback should be repo-relative, got: {DEFAULT_RING_EXE}")
    expected_default = Path("DeepMzyme_Data") / "ring-4.0" / "out" / "bin" / "ring"
    if DEFAULT_RING_EXE != expected_default:
        raise AssertionError(f"Unexpected RING default fallback: {DEFAULT_RING_EXE}")
    if str(DEFAULT_RING_EXE) == personal_ring_path:
        raise AssertionError(f"RING default fallback still uses a personal path: {DEFAULT_RING_EXE}")
    if personal_ring_path in str(DEFAULT_RING_EXE):
        raise AssertionError(f"RING default fallback unexpectedly contains a personal path: {DEFAULT_RING_EXE}")

    old_ring_features_dir = os.environ.get("RING_FEATURES_DIR")
    old_ring_exe_path = os.environ.get("RING_EXE_PATH")
    try:
        with tempfile.TemporaryDirectory(prefix="deepmzyme_ring_edges_smoke_") as tmp:
            tmp_root = Path(tmp)
            structure_dir = tmp_root / "structures"
            structure_dir.mkdir()
            ring_dir = tmp_root / "ring_edges"
            os.environ["RING_FEATURES_DIR"] = str(ring_dir)

            expected_path = canonical_ring_edges_output_path("/tmp/example_structure.pdb")
            if not str(expected_path).startswith(f"{ring_dir}/"):
                raise AssertionError(f"RING edge lookup did not honor RING_FEATURES_DIR: {expected_path}")

            report = prepare_runtime_inputs(
                structure_dir=structure_dir,
                esm_embeddings_dir=None,
                require_esm_embeddings=False,
                prepare_missing_esm_embeddings=False,
                use_ring_edges=False,
                require_ring_edges=False,
                prepare_missing_ring_edges=False,
            )
            if report["ring_edges_output_dir"] != str(ring_dir):
                raise AssertionError(f"RING edge output did not honor RING_FEATURES_DIR: {report}")

        os.environ["RING_EXE_PATH"] = "/tmp/deepmzyme_missing_ring_executable"
        try:
            create_ring_edges_batch([], dir_results="/tmp/deepmzyme_ring_edges_smoke")
        except FileNotFoundError as exc:
            if "/tmp/deepmzyme_missing_ring_executable" not in str(exc):
                raise AssertionError(f"RING executable error did not mention RING_EXE_PATH: {exc}") from exc
        else:
            raise AssertionError("Missing RING_EXE_PATH executable was not rejected.")
    finally:
        if old_ring_features_dir is None:
            os.environ.pop("RING_FEATURES_DIR", None)
        else:
            os.environ["RING_FEATURES_DIR"] = old_ring_features_dir
        if old_ring_exe_path is None:
            os.environ.pop("RING_EXE_PATH", None)
        else:
            os.environ["RING_EXE_PATH"] = old_ring_exe_path


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


def check_fold_class_weights_use_training_fold_only() -> None:
    pockets = [
        PocketRecord(
            structure_id=f"{index}abc__chain_A__EC_1.1.1.1",
            pocket_id=f"p{index}",
            metal_element="ZN",
            metal_coords=[torch.zeros(3)],
            residues=[],
            y_metal=0 if index < 3 else 1,
        )
        for index in range(5)
    ]
    split = split_pockets_k_fold(
        pockets,
        n_folds=5,
        fold_index=0,
        split_by="pdbid",
        seed=42,
        task="metal",
    )
    train_labels = [int(pocket.y_metal) for pocket in split.train_pockets]
    all_labels = [int(pocket.y_metal) for pocket in pockets]
    train_only_weights, _ec_weights = balanced_class_weights_from_pockets(
        split.train_pockets,
        n_metal_classes=2,
        n_ec_classes=1,
        metal_class_weight_mode="inverse_frequency",
    )
    expected_train_only = class_weights_from_labels(
        train_labels,
        n_classes=2,
        mode="inverse_frequency",
    )
    all_data_weights = class_weights_from_labels(
        all_labels,
        n_classes=2,
        mode="inverse_frequency",
    )
    if not torch.allclose(train_only_weights, expected_train_only):
        raise AssertionError(
            "Fold class weights were not computed from the training fold labels."
        )
    if torch.allclose(train_only_weights, all_data_weights):
        raise AssertionError(
            "Fold class weights matched all-data weights; validation labels may be leaking into weights."
        )


def check_paired_bootstrap_ci_helper() -> None:
    result = paired_bootstrap_ci(
        [0.70, 0.72, 0.74, 0.71, 0.73],
        [0.64, 0.66, 0.68, 0.65, 0.67],
        n_bootstrap=1000,
        seed=123,
        raw_improvement_threshold=0.01,
    )
    if result.n_pairs != 5:
        raise AssertionError(f"Unexpected paired bootstrap n_pairs: {result}")
    if result.mean_difference <= 0.0 or result.ci_lower <= 0.0 or not result.passes:
        raise AssertionError(f"Expected a positive paired bootstrap pass, got {result}")


def check_equal_mass_ece_helper() -> None:
    confidences = torch.tensor([0.55, 0.60, 0.80, 0.90], dtype=torch.float32)
    outcomes = torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float32)
    ece, bins = equal_mass_ece(confidences, outcomes, n_bins=2)
    expected = 0.1125
    if len(bins) != 2:
        raise AssertionError(f"Expected two equal-mass ECE bins, got {bins}")
    if abs(ece - expected) > 1.0e-5:
        raise AssertionError(f"Unexpected equal-mass ECE: {ece} != {expected}")


def check_temperature_scaling_helper() -> None:
    logits = torch.tensor(
        [
            [4.0, 0.0],
            [4.0, 0.0],
            [0.0, 4.0],
            [0.0, 4.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 1, 1], dtype=torch.long)
    before = torch.nn.functional.cross_entropy(logits, targets).item()
    temperature = fit_temperature_from_logits(logits, targets, max_iter=25)
    after = torch.nn.functional.cross_entropy(logits / temperature, targets).item()
    if temperature <= 0.0:
        raise AssertionError(f"Temperature must be positive, got {temperature}")
    if after > before + 1.0e-5:
        raise AssertionError(f"Temperature scaling increased NLL: before={before}, after={after}")


def check_final_test_bootstrap_ci_helper() -> None:
    targets = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], dtype=torch.long)
    probabilities = torch.full((targets.numel(), 6), 0.02, dtype=torch.float32)
    for index, target in enumerate(targets.tolist()):
        probabilities[index, target] = 0.90
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    cis = metal_bootstrap_metric_cis(
        probabilities,
        targets,
        n_bootstrap=50,
        confidence_level=0.95,
        seed=123,
        n_bins=3,
    )
    for key in ("test_metal_balanced_acc_ci95", "test_metal_collapsed4_balanced_acc_ci95"):
        value = cis.get(key)
        if not isinstance(value, list) or len(value) != 2:
            raise AssertionError(f"Missing bootstrap CI field {key}: {cis}")
        if value[0] > value[1]:
            raise AssertionError(f"Bootstrap CI is reversed for {key}: {value}")


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


def check_bundle_artifact_can_be_labeled_subset() -> None:
    from build_dataset_csv import validate_rows_match_structure_dir

    with tempfile.TemporaryDirectory(prefix="deepmzyme_bundle_subset_") as tmp:
        structure_dir = Path(tmp)
        (structure_dir / "labeled_structure.pdb").write_text("HEADER labeled\n", encoding="utf-8")
        (structure_dir / "unlabeled_structure.pdb").write_text("HEADER unlabeled\n", encoding="utf-8")
        rows = [
            {
                "structure_name": "labeled_structure",
                "ec_numbers": "1",
                "metal_type": "Zn",
            }
        ]
        try:
            validate_rows_match_structure_dir(structure_dir=structure_dir, rows=rows)
        except ValueError as exc:
            if "missing rows" not in str(exc):
                raise AssertionError(f"Strict validation failed for an unexpected reason: {exc}") from exc
        else:
            raise AssertionError("Strict structure CSV validation accepted a missing structure row.")

        validate_rows_match_structure_dir(
            structure_dir=structure_dir,
            rows=rows,
            allow_missing_structure_rows=True,
        )


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
        embeddings_dir=train_dir / "esm_embeddings",
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
        check_prelaunch_run_dir_reuse,
        check_loss_weight_validation,
        check_metal_label_scheme_options,
        check_training_efficiency_defaults_and_validation,
        check_grad_accum_final_partial_window,
        check_uncertainty_task_loss_weighter,
        check_collapsed4_metal_loss_helpers,
        check_ec_group_weighting_config,
        check_cross_attention_config,
        check_ring_edge_cli_config,
        check_training_graph_augmentation,
        check_esm_embedding_metadata_sidecar,
        check_only_gvp_does_not_require_esm,
        check_graph_ring_edges_are_opt_in,
        check_metal_node_graph_and_gvp_forward,
        check_colab_notebook_sweep_source,
        check_colab_final_test_workflow_controls,
        check_colab_notebook_provenance_helpers,
        check_colab_exact_split_no_fallback,
        check_colab_generated_training_commands_parse,
        check_ring_environment_overrides,
        check_ec_group_weights_sum_per_group,
        check_fold_class_weights_use_training_fold_only,
        check_paired_bootstrap_ci_helper,
        check_equal_mass_ece_helper,
        check_temperature_scaling_helper,
        check_final_test_bootstrap_ci_helper,
        check_ec_group_metric_aggregation,
        check_ec_group_id_batches_without_increment,
        check_conflicting_ec_group_metrics_are_skipped,
        check_bundle_cli_help,
        check_bundle_artifact_can_be_labeled_subset,
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
