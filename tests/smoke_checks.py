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
        "--grad-clip-norm",
        "--amp",
        "--grad-accum-steps",
        "--num-workers",
        "--pin-memory",
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


def check_training_efficiency_defaults_and_validation() -> None:
    default_config = parse_args([])
    expected_defaults = {
        "grad_clip_norm": 1.0,
        "use_amp": False,
        "grad_accum_steps": 1,
        "num_workers": 0,
        "pin_memory": False,
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


def check_colab_notebook_sweep_source() -> None:
    notebook_path = REPO_ROOT / "notebooks" / "DeepMzyme_training_colab.ipynb"
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in nb.get("cells", []))
    required_tokens = (
        '"run_training": False',
        'LAUNCH_PLANNED_TRAINING_RUNS = False',
        'INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False',
        'LAUNCH_FINAL_HELD_OUT_TEST_EVAL = False',
        'MODEL_PRESET = "Only-GVP"',
        'RING_EDGE_MODE = "with_ring"',
        'ALLOW_MISSING_EXTERNAL_FEATURES = False',
        'OMIT_NODE_FEATURE_SETS = ""',
        'MAX_CONFIGURATION_RUNS',
        "CONFIG = {",
        "COLAB_DATA_SOURCE",
        "huggingface_link",
        "DeepMzyme_Data_runtime_local_2026-05-18_ring_external.tar.zst",
        "740598ca2e657a016de81d5286f0fe6ff43d3f2504d26c3db022627ac0f8c8fa",
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
        "omit_node_features",
        "--omit-node-features",
        "--use-ring-edges",
        "--ring-features-dir",
        "--prepare-missing-ring-edges",
        "--no-prepare-missing-ring-edges",
        "--no-prepare-missing-esm-embeddings",
        "METAL_COLLAPSED_LOSS_WEIGHT = 0.0",
        "OPTUNA_MULTIOBJECTIVE = False",
        'OPTUNA_METAL_COLLAPSED_LOSS_WEIGHTS_CSV = "0.0"',
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
    if "sweep" in source.lower():
        raise AssertionError("Colab notebook should not contain user-facing sweep terminology.")


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
                "run_seed_repeat_evaluation": False,
                "top_k_configs_for_seed_repeat": 3,
                "position_noise_stds_csv": "0.0",
                "second_shell_dropouts_csv": "0.0",
            },
            "data": {"colab_data_source": "huggingface_link"},
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
                "edge_radius_values_csv": "8.0",
                "esm_fusion_dim_values_csv": "128",
                "lr_schedules_csv": "fixed",
                "lr_step_size": 10,
                "lr_decay_gamma": 0.5,
                "node_rbf_sigma": 0.75,
                "edge_rbf_sigma": 0.75,
                "node_rbf_use_raw_distances": False,
                "position_noise_std": 0.0,
                "second_shell_dropout": 0.0,
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

    def run_builder(config_updates: dict[str, dict[str, object]]) -> list[dict[str, object]]:
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

    collapsed_loss_runs = run_builder({"advanced": {"metal_collapsed_loss_weight": 0.3}})
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
        check_colab_notebook_sweep_source,
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
