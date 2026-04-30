from __future__ import annotations

import copy
import json
import random
import subprocess
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch_geometric.loader import DataLoader

from label_schemes import (
    COLLAPSED_METAL_LABELS,
    METAL_TARGET_LABELS,
    N_METAL_CLASSES,
    collapsed_metal_target_for_label_name,
)
from model_variants import build_pocket_classifier
from project_paths import resolve_runs_dir
from training.labels import ec_prefix_from_label_token, parse_structure_identity
from training.config import TrainConfig, config_to_payload, required_targets_for_task
from training.data import load_training_pockets_with_report_from_dir
from training.graph_dataset import (
    FeatureNormalizationStats,
    PocketGraphDataset,
    build_graph_data_list,
    compute_feature_normalization_stats,
)
from training.loop import (
    balanced_class_weights_from_pockets,
    classification_metrics_from_logits,
    evaluate_epoch_with_predictions,
    train_epoch,
)
from training.preflight import run_preflight_checks
from training.runtime_preparation import prepare_runtime_inputs
from training.splits import (
    PocketSplit,
    build_balanced_metal_site_sampler,
    build_dataset_summary,
    split_pockets,
    split_pockets_k_fold,
)


@dataclass(frozen=True)
class PreparedRun:
    config_payload: dict[str, Any]
    run_dir: Path
    split: PocketSplit
    dataset_summary: dict[str, Any]
    ec_labels: dict[int, str]
    ec_label_to_index: dict[str, int]
    normalization_stats: FeatureNormalizationStats
    train_loader: DataLoader
    val_loader: DataLoader | None
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler | None


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            warnings.warn(
                "This PyTorch version does not support warn_only for deterministic algorithms; "
                "using CUDA/cuDNN deterministic seeding flags only.",
                RuntimeWarning,
            )
        except Exception as exc:
            warnings.warn(f"Could not enable deterministic PyTorch algorithms: {exc}", RuntimeWarning)


def build_run_dir(config: TrainConfig) -> Path:
    runs_dir = resolve_runs_dir(config.runs_dir, create=True)
    effective_name = config.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = runs_dir / effective_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def infer_split_identity(config: TrainConfig) -> dict[str, Any]:
    path_text = " ".join(
        str(path)
        for path in (
            config.structure_dir,
            config.summary_csv,
            config.test_structure_dir,
            config.test_summary_csv,
        )
        if path is not None
    )
    normalized = path_text.lower()
    if "train_and_test_sets_structures_non_overlapped_pinmymetal" in normalized:
        return {
            "split_name": "train_and_test_sets_structures_non_overlapped_pinmymetal",
            "split_type": "non_overlapped_pinmymetal",
            "overlap_warning": None,
        }
    if "train_and_test_sets_structures_exact_pinmymetal" in normalized:
        return {
            "split_name": "train_and_test_sets_structures_exact_pinmymetal",
            "split_type": "exact_pinmymetal_possibly_overlapped",
            "overlap_warning": (
                "Exact PinMyMetal split may contain train/test overlap and should "
                "be interpreted only as a secondary/reference result."
            ),
        }
    return {
        "split_name": Path(config.structure_dir).parent.name or Path(config.structure_dir).name,
        "split_type": "custom_or_unknown",
        "overlap_warning": None,
    }


def pocket_identity_sets(pockets) -> dict[str, set[str]]:
    structure_ids: set[str] = set()
    pdb_ids: set[str] = set()
    pdb_chain_ids: set[str] = set()
    pocket_ids: set[str] = set()
    for pocket in pockets:
        structure_ids.add(str(pocket.structure_id))
        pocket_ids.add(str(pocket.pocket_id))
        pdb_id, chain_id, _ec = parse_structure_identity(pocket.structure_id)
        pdb_ids.add(str(pdb_id))
        pdb_chain_ids.add(f"{pdb_id}__chain_{chain_id}")
    return {
        "structure_id": structure_ids,
        "pdb_id": pdb_ids,
        "pdb_chain": pdb_chain_ids,
        "pocket_id": pocket_ids,
    }


def train_test_overlap_report(train_like_pockets, test_pockets) -> dict[str, Any]:
    train_sets = pocket_identity_sets(train_like_pockets)
    test_sets = pocket_identity_sets(test_pockets)
    overlap_counts = {
        key: len(train_sets[key].intersection(test_sets[key]))
        for key in sorted(train_sets)
    }
    overlap_examples = {
        key: sorted(train_sets[key].intersection(test_sets[key]))[:10]
        for key in sorted(train_sets)
    }
    detected = any(count > 0 for count in overlap_counts.values())
    return {
        "train_test_overlap_detected": detected,
        "overlap_counts": overlap_counts,
        "overlap_examples": overlap_examples,
        "overlap_warning": "Train/test overlap detected." if detected else None,
    }


def prepare_status_payload(*, stage: str, status: str, config_payload: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "stage": stage,
        "status": status,
        "config": config_payload,
    }
    if extra:
        payload.update(extra)
    return payload


def validate_training_configuration(config: TrainConfig) -> None:
    if config.gvp_layers < 1:
        raise ValueError(f"--gvp-layers must be at least 1, got {config.gvp_layers}")
    if config.head_mlp_layers < 1:
        raise ValueError(f"--head-mlp-layers must be at least 1, got {config.head_mlp_layers}")
    if config.ec_label_depth < 1:
        raise ValueError(f"--ec-label-depth must be at least 1, got {config.ec_label_depth}")
    if config.ec_contrastive_weight < 0.0:
        raise ValueError(
            f"--ec-contrastive-weight must be non-negative, got {config.ec_contrastive_weight}"
        )
    if config.ec_contrastive_temperature <= 0.0:
        raise ValueError(
            "--ec-contrastive-temperature must be positive, "
            f"got {config.ec_contrastive_temperature}"
        )
    if config.metal_loss_weight < 0.0:
        raise ValueError(f"--metal-loss-weight must be non-negative, got {config.metal_loss_weight}")
    if config.ec_loss_weight < 0.0:
        raise ValueError(f"--ec-loss-weight must be non-negative, got {config.ec_loss_weight}")
    has_validation = config.val_fraction > 0.0 or config.n_folds is not None
    if not has_validation and config.selection_metric.startswith("val_"):
        raise ValueError(
            "Selection metric "
            f"{config.selection_metric!r} requires validation, but --val-fraction is 0. "
            "Either enable validation or choose a train-based metric such as 'train_loss'."
        )
    if config.run_test_eval and not config.allow_train_loss_test_eval_debug:
        if not has_validation:
            raise ValueError(
                "--run-test-eval is for held-out reporting and must use a validation-selected "
                "checkpoint. Set --val-fraction > 0, or use --n-folds/--fold-index, so model "
                "selection is based on validation metrics rather than train_loss. For a tiny "
                "debug/smoke run only, pass --allow-train-loss-test-eval-debug explicitly."
            )
        if config.selection_metric == "train_loss":
            raise ValueError(
                "--run-test-eval cannot select the checkpoint with train_loss because that tunes "
                "the final held-out report from the training objective. Use a validation metric "
                "such as val_metal_balanced_acc, val_ec_balanced_acc, or val_joint_balanced_acc. "
                "For a tiny debug/smoke run only, pass --allow-train-loss-test-eval-debug explicitly."
            )
    if (config.n_folds is None) != (config.fold_index is None):
        raise ValueError("--n-folds and --fold-index must be provided together.")
    if config.n_folds is not None:
        if config.n_folds < 2:
            raise ValueError(f"--n-folds must be at least 2, got {config.n_folds}")
        if not 0 <= int(config.fold_index) < config.n_folds:
            raise ValueError(
                f"--fold-index must be in [0, {config.n_folds - 1}], got {config.fold_index}"
            )
    if config.task == "metal" and ("ec_" in config.selection_metric or "joint_" in config.selection_metric):
        raise ValueError(
            f"Selection metric {config.selection_metric!r} is incompatible with --task metal. "
            "Use train_loss or a metal validation metric."
        )
    if config.task == "ec" and ("metal_" in config.selection_metric or "joint_" in config.selection_metric):
        raise ValueError(
            f"Selection metric {config.selection_metric!r} is incompatible with --task ec. "
            "Use train_loss or an EC validation metric."
        )
    if config.lr_schedule == "step" and config.lr_step_size <= 0:
        raise ValueError("--lr-step-size must be positive when --lr-schedule step is selected.")
    if config.run_test_eval and (config.test_structure_dir is None or config.test_summary_csv is None):
        raise ValueError(
            "--run-test-eval requires both --test-structure-dir and --test-summary-csv "
            "so held-out reporting remains explicit."
        )


def task_predicts_metal(task: str) -> bool:
    return task in ("joint", "metal")


def task_predicts_ec(task: str) -> bool:
    return task in ("joint", "ec")


def present_metric_values(values: list[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None]


def metal_label_index(label_name: str) -> int | None:
    for label_idx, current_label_name in METAL_TARGET_LABELS.items():
        if current_label_name == label_name:
            return label_idx
    return None


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if config.lr_schedule == "fixed":
        return None
    if config.lr_schedule == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))
    if config.lr_schedule == "step":
        return StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_decay_gamma)
    raise ValueError(f"Unsupported lr schedule {config.lr_schedule!r}.")


def collapse_metal_logits(logits: torch.Tensor) -> torch.Tensor:
    grouped_logits: list[torch.Tensor] = []
    for collapsed_idx in sorted(COLLAPSED_METAL_LABELS):
        source_indices = [
            label_idx
            for label_idx, label_name in METAL_TARGET_LABELS.items()
            if collapsed_metal_target_for_label_name(label_name) == collapsed_idx
        ]
        if not source_indices:
            grouped_logits.append(
                torch.full(
                    (logits.size(0), 1),
                    fill_value=torch.finfo(logits.dtype).min,
                    dtype=logits.dtype,
                    device=logits.device,
                )
            )
            continue
        grouped_logits.append(torch.logsumexp(logits[:, source_indices], dim=-1, keepdim=True))
    return torch.cat(grouped_logits, dim=-1)


def collapse_metal_targets(targets: torch.Tensor) -> torch.Tensor:
    collapsed = [
        collapsed_metal_target_for_label_name(METAL_TARGET_LABELS[int(target_idx)])
        for target_idx in targets.tolist()
    ]
    return torch.tensor(collapsed, dtype=torch.long)


def ec_level_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ec_label_map: dict[int, str],
    level: int,
) -> dict[str, Any]:
    predicted_ids = logits.argmax(dim=-1).tolist()
    true_ids = targets.tolist()
    prefix_to_index: dict[str, int] = {}
    level_pred: list[int] = []
    level_true: list[int] = []

    for predicted_id, true_id in zip(predicted_ids, true_ids):
        predicted_prefix = ec_prefix_from_label_token(ec_label_map[int(predicted_id)], level=level)
        true_prefix = ec_prefix_from_label_token(ec_label_map[int(true_id)], level=level)
        if predicted_prefix is None or true_prefix is None:
            continue
        for prefix in (predicted_prefix, true_prefix):
            if prefix not in prefix_to_index:
                prefix_to_index[prefix] = len(prefix_to_index)
        level_pred.append(prefix_to_index[predicted_prefix])
        level_true.append(prefix_to_index[true_prefix])

    if not level_true:
        return {}
    return classification_metrics_from_logits(
        torch.nn.functional.one_hot(torch.tensor(level_pred), num_classes=len(prefix_to_index)).float(),
        torch.tensor(level_true, dtype=torch.long),
    )


def evaluate_split_metrics(
    model,
    loader: DataLoader | None,
    device: str,
    prefix: str,
    *,
    task: str,
    ec_label_map: dict[int, str],
    ec_label_depth: int,
) -> dict[str, Any]:
    if loader is None:
        return {}

    predictions = evaluate_epoch_with_predictions(model, loader, device=device)
    payload = {
        f"{prefix}_loss": predictions["loss"],
    }
    if task_predicts_metal(task) and "metal_y" in predictions and "metal_logits" in predictions:
        metal_metrics = classification_metrics_from_logits(predictions["metal_logits"], predictions["metal_y"])
        metal_recalls = present_metric_values(metal_metrics["per_class_recall"])
        mn_idx = metal_label_index("Mn")
        fe_idx = metal_label_index("Fe")
        class_viii_idx = metal_label_index("Class VIII")
        collapsed_logits = collapse_metal_logits(predictions["metal_logits"])
        collapsed_targets = collapse_metal_targets(predictions["metal_y"])
        collapsed_metrics = classification_metrics_from_logits(collapsed_logits, collapsed_targets)
        payload.update(
            {
                f"{prefix}_metal_acc": metal_metrics["accuracy"],
                f"{prefix}_metal_balanced_acc": metal_metrics["balanced_accuracy"],
                f"{prefix}_metal_macro_f1": metal_metrics["macro_f1"],
                f"{prefix}_metal_min_recall": float(min(metal_recalls)),
                f"{prefix}_metal_mn_recall": (
                    metal_metrics["per_class_recall"][mn_idx] if mn_idx is not None else None
                ),
                f"{prefix}_metal_fe_recall": (
                    metal_metrics["per_class_recall"][fe_idx] if fe_idx is not None else None
                ),
                f"{prefix}_metal_class_viii_recall": (
                    metal_metrics["per_class_recall"][class_viii_idx] if class_viii_idx is not None else None
                ),
                f"{prefix}_metal_per_class_recall": {
                    label_name: metal_metrics["per_class_recall"][label_idx]
                    for label_idx, label_name in METAL_TARGET_LABELS.items()
                },
                f"{prefix}_metal_collapsed4_acc": collapsed_metrics["accuracy"],
                f"{prefix}_metal_collapsed4_balanced_acc": collapsed_metrics["balanced_accuracy"],
                f"{prefix}_metal_collapsed4_macro_f1": collapsed_metrics["macro_f1"],
                f"{prefix}_metal_collapsed4_mn_recall": collapsed_metrics["per_class_recall"][0],
            }
        )
    else:
        payload[f"{prefix}_metal_acc"] = None
    if task_predicts_ec(task) and "ec_y" in predictions and "ec_logits" in predictions:
        ec_metrics = classification_metrics_from_logits(predictions["ec_logits"], predictions["ec_y"])
        payload.update(
            {
                f"{prefix}_ec_acc": ec_metrics["accuracy"],
                f"{prefix}_ec_balanced_acc": ec_metrics["balanced_accuracy"],
                f"{prefix}_ec_macro_f1": ec_metrics["macro_f1"],
                f"{prefix}_ec_per_class_recall": {
                    label_name: ec_metrics["per_class_recall"][label_idx]
                    for label_idx, label_name in ec_label_map.items()
                },
            }
        )
        for level in range(1, ec_label_depth + 1):
            level_metrics = ec_level_metrics_from_logits(
                predictions["ec_logits"],
                predictions["ec_y"],
                ec_label_map=ec_label_map,
                level=level,
            )
            if not level_metrics:
                continue
            payload.update(
                {
                    f"{prefix}_ec_level_{level}_acc": level_metrics["accuracy"],
                    f"{prefix}_ec_level_{level}_balanced_acc": level_metrics["balanced_accuracy"],
                    f"{prefix}_ec_level_{level}_macro_f1": level_metrics["macro_f1"],
                }
            )
    else:
        payload[f"{prefix}_ec_acc"] = None

    balanced_values = [
        value
        for value in (
            payload.get(f"{prefix}_metal_balanced_acc"),
            payload.get(f"{prefix}_ec_balanced_acc"),
        )
        if value is not None
    ]
    macro_f1_values = [
        value
        for value in (
            payload.get(f"{prefix}_metal_macro_f1"),
            payload.get(f"{prefix}_ec_macro_f1"),
        )
        if value is not None
    ]
    if balanced_values:
        payload[f"{prefix}_joint_balanced_acc"] = float(sum(balanced_values) / len(balanced_values))
    if macro_f1_values:
        payload[f"{prefix}_joint_macro_f1"] = float(sum(macro_f1_values) / len(macro_f1_values))
    return payload


def normalization_stats_payload(normalization_stats: FeatureNormalizationStats) -> dict[str, Any]:
    return {
        "means": normalization_stats.means,
        "stds": normalization_stats.stds,
        "clamp_value": normalization_stats.clamp_value,
    }


def checkpoint_payload(
    *,
    model_state_dict: dict[str, Any],
    optimizer_state_dict: dict[str, Any],
    scheduler_state_dict: dict[str, Any] | None,
    history: list[dict[str, Any]],
    config_payload: dict[str, Any],
    normalization_stats: FeatureNormalizationStats,
    dataset_summary: dict[str, Any],
    ec_labels: dict[int, str],
) -> dict[str, Any]:
    return {
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler_state_dict,
        "history": history,
        "config": config_payload,
        "metal_labels": METAL_TARGET_LABELS,
        "ec_labels": ec_labels,
        "normalization_stats": normalization_stats_payload(normalization_stats),
        "dataset_summary": dataset_summary,
    }


def format_epoch_log(record: dict[str, Any]) -> str:
    parts = [
        f"epoch={record['epoch']}",
        f"train_loss={record['train_loss']:.4f}",
        f"lr={record['lr']:.6g}",
    ]
    if record["train_metal_acc"] is not None:
        parts.append(f"train_metal_acc={record['train_metal_acc']:.4f}")
    if record["train_ec_acc"] is not None:
        parts.append(f"train_ec_acc={record['train_ec_acc']:.4f}")
    if "val_loss" in record:
        parts.append(f"val_loss={record['val_loss']:.4f}")
    if record.get("val_metal_acc") is not None:
        parts.append(f"val_metal_acc={record['val_metal_acc']:.4f}")
    if record.get("val_ec_acc") is not None:
        parts.append(f"val_ec_acc={record['val_ec_acc']:.4f}")
    if record.get("val_metal_min_recall") is not None:
        parts.append(f"val_metal_min_recall={record['val_metal_min_recall']:.4f}")
    if record.get("val_metal_fe_recall") is not None:
        parts.append(f"val_fe_recall={record['val_metal_fe_recall']:.4f}")
    if record.get("val_metal_class_viii_recall") is not None:
        parts.append(f"val_class_viii_recall={record['val_metal_class_viii_recall']:.4f}")
    if record.get("val_joint_balanced_acc") is not None:
        parts.append(f"val_joint_bal_acc={record['val_joint_balanced_acc']:.4f}")
    if record.get("val_joint_macro_f1") is not None:
        parts.append(f"val_joint_macro_f1={record['val_joint_macro_f1']:.4f}")
    return " ".join(parts)


def metric_sort_value(record: dict[str, Any], selection_metric: str) -> tuple[float, bool]:
    if selection_metric not in record or record[selection_metric] is None:
        raise ValueError(f"Selection metric {selection_metric!r} is missing from the epoch record.")
    metric_value = float(record[selection_metric])
    if selection_metric.endswith("_loss"):
        return metric_value, False
    return metric_value, True


def prepare_run(config: TrainConfig) -> PreparedRun:
    validate_training_configuration(config)
    config_payload = config_to_payload(config)
    config_payload.update(infer_split_identity(config))
    config_payload["git_commit"] = git_commit_hash()
    run_dir = build_run_dir(config)
    save_json(
        run_dir / "prepare_status.json",
        prepare_status_payload(stage="prepare_run", status="started", config_payload=config_payload),
    )
    try:
        runtime_preparation_report = prepare_runtime_inputs(
            structure_dir=config.structure_dir,
            esm_embeddings_dir=config.esm_embeddings_dir,
            require_esm_embeddings=config.require_esm_embeddings,
            prepare_missing_esm_embeddings=config.prepare_missing_esm_embeddings,
            external_features_root_dir=config.external_features_root_dir,
            external_feature_source=config.external_feature_source,
            require_external_features=config.require_external_features,
            require_ring_edges=config.require_ring_edges,
            prepare_missing_ring_edges=config.prepare_missing_ring_edges,
        )
        save_json(
            run_dir / "prepare_status.json",
            prepare_status_payload(
                stage="runtime_preparation",
                status="completed",
                config_payload=config_payload,
                extra={"runtime_preparation": runtime_preparation_report},
            ),
        )
        load_result = load_training_pockets_with_report_from_dir(
            structure_dir=config.structure_dir,
            require_full_labels=True,
            required_targets=required_targets_for_task(config.task),
            summary_csv=config.summary_csv,
            esm_dim=config.esm_dim,
            esm_embeddings_dir=config.esm_embeddings_dir,
            require_esm_embeddings=config.require_esm_embeddings,
            external_features_root_dir=config.external_features_root_dir,
            external_feature_source=config.external_feature_source,
            require_external_features=config.require_external_features,
            unsupported_metal_policy=config.unsupported_metal_policy,
            invalid_structure_policy=config.invalid_structure_policy,
            ec_label_depth=config.ec_label_depth,
        )
        pockets = load_result.pockets
        if not pockets:
            raise ValueError("No training pockets were loaded.")
        if task_predicts_ec(config.task) and not load_result.ec_index_to_label:
            raise ValueError(
                "No EC labels were available after applying the configured "
                f"--ec-label-depth {config.ec_label_depth}."
            )
        save_json(
            run_dir / "prepare_status.json",
            prepare_status_payload(
                stage="load_training_pockets",
                status="completed",
                config_payload=config_payload,
                extra={"n_loaded_pockets": len(pockets), "feature_load_report": load_result.feature_report},
            ),
        )

        if config.n_folds is not None:
            split = split_pockets_k_fold(
                pockets,
                n_folds=config.n_folds,
                fold_index=int(config.fold_index),
                split_by=config.split_by,
                seed=config.seed,
                task=config.task,
            )
        else:
            split = split_pockets(
                pockets,
                val_fraction=config.val_fraction,
                split_by=config.split_by,
                seed=config.seed,
                task=config.task,
            )
        dataset_summary = build_dataset_summary(
            split,
            config,
            feature_load_report=load_result.feature_report,
            ec_label_map=load_result.ec_index_to_label,
        )
        dataset_summary.update(
            {
                "split_name": config_payload.get("split_name"),
                "split_type": config_payload.get("split_type"),
                "overlap_warning": config_payload.get("overlap_warning"),
                "test_structure_dir": config_payload.get("test_structure_dir"),
                "test_summary_csv": config_payload.get("test_summary_csv"),
            }
        )
        dataset_summary["runtime_preparation"] = runtime_preparation_report
        train_graphs = build_graph_data_list(
            split.train_pockets,
            esm_dim=config.esm_dim,
            edge_radius=config.edge_radius,
            require_ring_edges=config.require_ring_edges,
            node_feature_set=config.node_feature_set,
        )
        val_graphs = (
            build_graph_data_list(
                split.val_pockets,
                esm_dim=config.esm_dim,
                edge_radius=config.edge_radius,
                require_ring_edges=config.require_ring_edges,
                node_feature_set=config.node_feature_set,
            )
            if split.val_pockets
            else None
        )
        save_json(
            run_dir / "prepare_status.json",
            prepare_status_payload(
                stage="build_graphs",
                status="completed",
                config_payload=config_payload,
                extra={
                    "n_train_graphs": len(train_graphs),
                    "n_val_graphs": 0 if val_graphs is None else len(val_graphs),
                },
            ),
        )
        dataset_summary["preflight"] = run_preflight_checks(
            split,
            config,
            ec_label_map=load_result.ec_index_to_label,
            train_graphs=train_graphs,
            val_graphs=val_graphs,
        )
        save_json(
            run_dir / "prepare_status.json",
            prepare_status_payload(
                stage="preflight",
                status="completed",
                config_payload=config_payload,
                extra={"dataset_summary": dataset_summary},
            ),
        )
        normalization_stats = compute_feature_normalization_stats(train_graphs, clamp_value=5.0)
        train_sampler = (
            build_balanced_metal_site_sampler(split.train_pockets)
            if config.balance_metal_site_symbols and task_predicts_metal(config.task)
            else None
        )
        train_loader = DataLoader(
            PocketGraphDataset(
                split.train_pockets,
                esm_dim=config.esm_dim,
                edge_radius=config.edge_radius,
                normalization_stats=normalization_stats,
                require_ring_edges=config.require_ring_edges,
                precomputed_data=train_graphs,
                node_feature_set=config.node_feature_set,
            ),
            batch_size=config.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
        )
        val_loader = (
            DataLoader(
                PocketGraphDataset(
                    split.val_pockets,
                    esm_dim=config.esm_dim,
                    edge_radius=config.edge_radius,
                    normalization_stats=normalization_stats,
                    require_ring_edges=config.require_ring_edges,
                    precomputed_data=val_graphs,
                    node_feature_set=config.node_feature_set,
                ),
                batch_size=config.batch_size,
                shuffle=False,
            )
            if split.val_pockets
            else None
        )

        metal_class_weights = None
        ec_class_weights = None
        computed_metal_weights, computed_ec_weights = balanced_class_weights_from_pockets(
            split.train_pockets,
            n_metal_classes=N_METAL_CLASSES,
            n_ec_classes=max(1, len(load_result.ec_index_to_label)),
        )
        if task_predicts_metal(config.task):
            metal_class_weights = computed_metal_weights
            mn_idx = metal_label_index("Mn")
            cu_idx = metal_label_index("Cu")
            zn_idx = metal_label_index("Zn")
            fe_idx = metal_label_index("Fe")
            co_idx = metal_label_index("Co")
            ni_idx = metal_label_index("Ni")
            class_viii_idx = metal_label_index("Class VIII")
            if mn_idx is not None:
                metal_class_weights[mn_idx] = metal_class_weights[mn_idx] * float(config.mn_loss_multiplier)
            if cu_idx is not None:
                metal_class_weights[cu_idx] = metal_class_weights[cu_idx] * float(config.cu_loss_multiplier)
            if zn_idx is not None:
                metal_class_weights[zn_idx] = metal_class_weights[zn_idx] * float(config.zn_loss_multiplier)
            if fe_idx is not None:
                metal_class_weights[fe_idx] = metal_class_weights[fe_idx] * float(config.fe_loss_multiplier)
            if co_idx is not None:
                metal_class_weights[co_idx] = metal_class_weights[co_idx] * float(config.co_loss_multiplier)
            if ni_idx is not None:
                metal_class_weights[ni_idx] = metal_class_weights[ni_idx] * float(config.ni_loss_multiplier)
            if class_viii_idx is not None:
                metal_class_weights[class_viii_idx] = (
                    metal_class_weights[class_viii_idx] * float(config.class_viii_loss_multiplier)
                )
        if task_predicts_ec(config.task):
            ec_class_weights = computed_ec_weights

        model = build_pocket_classifier(
            model_architecture=config.model_architecture,
            esm_dim=config.esm_dim,
            hidden_s=config.hidden_s,
            hidden_v=config.hidden_v,
            edge_hidden=config.edge_hidden,
            n_layers=config.gvp_layers,
            n_metal=N_METAL_CLASSES,
            n_ec=max(1, len(load_result.ec_index_to_label)),
            esm_fusion_dim=config.esm_fusion_dim,
            head_mlp_layers=config.head_mlp_layers,
            node_rbf_sigma=config.node_rbf_sigma,
            edge_rbf_sigma=config.edge_rbf_sigma,
            node_rbf_use_raw_distances=config.node_rbf_use_raw_distances,
            use_esm_branch=config.use_esm_branch,
            fusion_mode=config.fusion_mode,
            cross_attention_layers=config.cross_attention_layers,
            cross_attention_heads=config.cross_attention_heads,
            cross_attention_dropout=config.cross_attention_dropout,
            cross_attention_neighborhood=config.cross_attention_neighborhood,
            cross_attention_bidirectional=config.cross_attention_bidirectional,
            use_early_esm=config.use_early_esm,
            early_esm_dim=config.early_esm_dim,
            early_esm_dropout=config.early_esm_dropout,
            early_esm_raw=config.early_esm_raw,
            early_esm_scope=config.early_esm_scope,
            metal_loss_weight=config.metal_loss_weight,
            ec_loss_weight=config.ec_loss_weight,
            metal_loss_function=config.metal_loss_function,
            metal_focal_gamma=config.metal_focal_gamma,
            metal_label_smoothing=config.metal_label_smoothing,
            ec_contrastive_weight=config.ec_contrastive_weight,
            ec_contrastive_temperature=config.ec_contrastive_temperature,
            metal_class_weights=metal_class_weights,
            ec_class_weights=ec_class_weights,
            predict_metal=task_predicts_metal(config.task),
            predict_ec=task_predicts_ec(config.task),
        ).to(config.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = build_scheduler(optimizer, config)
        save_json(
            run_dir / "prepare_status.json",
            prepare_status_payload(stage="prepare_run", status="ready", config_payload=config_payload),
        )
        return PreparedRun(
            config_payload=config_payload,
            run_dir=run_dir,
            split=split,
            dataset_summary=dataset_summary,
            ec_labels=load_result.ec_index_to_label,
            ec_label_to_index=load_result.ec_label_to_index,
            normalization_stats=normalization_stats,
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    except Exception as exc:
        save_json(
            run_dir / "prepare_status.json",
            prepare_status_payload(
                stage="prepare_run",
                status="failed",
                config_payload=config_payload,
                extra={"error": str(exc)},
            ),
        )
        raise


def train_and_select_checkpoint(
    prepared: PreparedRun,
    config: TrainConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    history: list[dict[str, Any]] = []
    best_metric: float | None = None
    best_checkpoint = None
    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(prepared.model, prepared.train_loader, prepared.optimizer, device=config.device)
        train_metrics = evaluate_split_metrics(
            prepared.model,
            prepared.train_loader,
            config.device,
            prefix="train",
            task=config.task,
            ec_label_map=prepared.ec_labels,
            ec_label_depth=config.ec_label_depth,
        )
        train_metrics.pop("train_loss", None)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": float(prepared.optimizer.param_groups[0]["lr"]),
            **train_metrics,
        }

        val_metrics = evaluate_split_metrics(
            prepared.model,
            prepared.val_loader,
            config.device,
            prefix="val",
            task=config.task,
            ec_label_map=prepared.ec_labels,
            ec_label_depth=config.ec_label_depth,
        )
        record.update(val_metrics)
        current_metric, maximize = metric_sort_value(record, config.selection_metric)
        is_better = (
            best_metric is None
            or (maximize and current_metric > best_metric)
            or (not maximize and current_metric < best_metric)
        )
        if is_better:
            best_metric = current_metric
            best_checkpoint = checkpoint_payload(
                model_state_dict=copy.deepcopy(prepared.model.state_dict()),
                optimizer_state_dict=copy.deepcopy(prepared.optimizer.state_dict()),
                scheduler_state_dict=(
                    copy.deepcopy(prepared.scheduler.state_dict()) if prepared.scheduler is not None else None
                ),
                history=copy.deepcopy(history + [record]),
                config_payload=prepared.config_payload,
                normalization_stats=prepared.normalization_stats,
                dataset_summary=prepared.dataset_summary,
                ec_labels=prepared.ec_labels,
            )
            best_checkpoint["epoch"] = epoch
            best_checkpoint["selection_metric"] = config.selection_metric
            best_checkpoint["selection_metric_value"] = current_metric

        history.append(record)
        if config.save_epoch_checkpoints:
            epoch_checkpoint_path = prepared.run_dir / f"epoch_{epoch:04d}_checkpoint.pt"
            torch.save(
                checkpoint_payload(
                    model_state_dict=prepared.model.state_dict(),
                    optimizer_state_dict=prepared.optimizer.state_dict(),
                    scheduler_state_dict=(
                        prepared.scheduler.state_dict() if prepared.scheduler is not None else None
                    ),
                    history=history,
                    config_payload=prepared.config_payload,
                    normalization_stats=prepared.normalization_stats,
                    dataset_summary=prepared.dataset_summary,
                    ec_labels=prepared.ec_labels,
                ),
                epoch_checkpoint_path,
            )
        if prepared.scheduler is not None:
            prepared.scheduler.step()
        print(format_epoch_log(record))
    return history, best_checkpoint


def evaluate_held_out_test_split(
    prepared: PreparedRun,
    config: TrainConfig,
    *,
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not config.run_test_eval or config.test_structure_dir is None or config.test_summary_csv is None:
        return None

    current_state_dict = copy.deepcopy(prepared.model.state_dict())
    model_state_dict = (
        checkpoint["model_state_dict"] if checkpoint is not None else prepared.model.state_dict()
    )
    prepared.model.load_state_dict(model_state_dict)
    try:
        test_load_result = load_training_pockets_with_report_from_dir(
            structure_dir=config.test_structure_dir,
            require_full_labels=True,
            required_targets=required_targets_for_task(config.task),
            summary_csv=config.test_summary_csv,
            esm_dim=config.esm_dim,
            esm_embeddings_dir=config.esm_embeddings_dir,
            require_esm_embeddings=config.require_esm_embeddings,
            external_features_root_dir=config.external_features_root_dir,
            external_feature_source=config.external_feature_source,
            require_external_features=config.require_external_features,
            unsupported_metal_policy=config.unsupported_metal_policy,
            invalid_structure_policy=config.invalid_structure_policy,
            ec_label_depth=config.ec_label_depth,
            ec_label_to_index=prepared.ec_label_to_index,
        )
        if not test_load_result.pockets:
            raise ValueError("No held-out test pockets were loaded.")

        test_graphs = build_graph_data_list(
            test_load_result.pockets,
            esm_dim=config.esm_dim,
            edge_radius=config.edge_radius,
            require_ring_edges=config.require_ring_edges,
            node_feature_set=config.node_feature_set,
        )
        test_loader = DataLoader(
            PocketGraphDataset(
                test_load_result.pockets,
                esm_dim=config.esm_dim,
                edge_radius=config.edge_radius,
                normalization_stats=prepared.normalization_stats,
                require_ring_edges=config.require_ring_edges,
                precomputed_data=test_graphs,
                node_feature_set=config.node_feature_set,
            ),
            batch_size=config.batch_size,
            shuffle=False,
        )
        metrics = evaluate_split_metrics(
            prepared.model,
            test_loader,
            config.device,
            prefix="test",
            task=config.task,
            ec_label_map=prepared.ec_labels,
            ec_label_depth=config.ec_label_depth,
        )
        train_like_pockets = prepared.split.train_pockets + prepared.split.val_pockets
        overlap_report = train_test_overlap_report(train_like_pockets, test_load_result.pockets)
        split_identity = infer_split_identity(config)
        overlap_warning = overlap_report.get("overlap_warning") or split_identity.get("overlap_warning")
        return {
            "metrics": metrics,
            "n_test_pockets": len(test_load_result.pockets),
            "feature_load_report": test_load_result.feature_report,
            "ec_labels": prepared.ec_labels,
            "split_name": split_identity.get("split_name"),
            "split_type": split_identity.get("split_type"),
            "train_test_overlap_detected": overlap_report["train_test_overlap_detected"],
            "overlap_counts": overlap_report["overlap_counts"],
            "overlap_examples": overlap_report["overlap_examples"],
            "overlap_warning": overlap_warning,
        }
    finally:
        prepared.model.load_state_dict(current_state_dict)


def persist_run_outputs(
    prepared: PreparedRun,
    *,
    history: list[dict[str, float | int]],
    best_checkpoint: dict[str, Any] | None,
    test_report: dict[str, Any] | None = None,
) -> None:
    save_json(prepared.run_dir / "dataset_summary.json", prepared.dataset_summary)

    checkpoint_path = prepared.run_dir / "last_model_checkpoint.pt"
    torch.save(
        checkpoint_payload(
            model_state_dict=prepared.model.state_dict(),
            optimizer_state_dict=prepared.optimizer.state_dict(),
            scheduler_state_dict=prepared.scheduler.state_dict() if prepared.scheduler is not None else None,
            history=history,
            config_payload=prepared.config_payload,
            normalization_stats=prepared.normalization_stats,
            dataset_summary=prepared.dataset_summary,
            ec_labels=prepared.ec_labels,
        ),
        checkpoint_path,
    )

    if best_checkpoint is not None:
        best_checkpoint_path = prepared.run_dir / "best_model_checkpoint.pt"
        torch.save(best_checkpoint, best_checkpoint_path)
        selected_checkpoint_path = str(best_checkpoint_path)
        selected_checkpoint_epoch = best_checkpoint.get("epoch")
        selected_metric_value = best_checkpoint.get("selection_metric_value")
    else:
        selected_checkpoint_path = str(checkpoint_path)
        selected_checkpoint_epoch = None
        selected_metric_value = None

    run_metadata = {
        "config": prepared.config_payload,
        "dataset_summary": prepared.dataset_summary,
        "metal_labels": METAL_TARGET_LABELS,
        "ec_labels": prepared.ec_labels,
        "normalization_stats": normalization_stats_payload(prepared.normalization_stats),
        "selection_metric": prepared.config_payload.get("selection_metric"),
        "selected_checkpoint": selected_checkpoint_path,
        "selected_checkpoint_epoch": selected_checkpoint_epoch,
        "selected_metric_value": selected_metric_value,
        "split_name": prepared.config_payload.get("split_name"),
        "split_type": prepared.config_payload.get("split_type"),
        "train_test_overlap_detected": (
            test_report.get("train_test_overlap_detected") if test_report is not None else None
        ),
        "overlap_warning": (
            test_report.get("overlap_warning")
            if test_report is not None
            else prepared.config_payload.get("overlap_warning")
        ),
        "test_report": test_report,
    }

    save_json(
        prepared.run_dir / "run_config.json",
        {
            "config": prepared.config_payload,
            "dataset_summary": prepared.dataset_summary,
            "metal_labels": METAL_TARGET_LABELS,
            "ec_labels": prepared.ec_labels,
            "normalization_stats": normalization_stats_payload(prepared.normalization_stats),
            "selection_metric": prepared.config_payload.get("selection_metric"),
            "selected_checkpoint": selected_checkpoint_path,
            "selected_checkpoint_epoch": selected_checkpoint_epoch,
            "selected_metric_value": selected_metric_value,
            "history": history,
            "test_report": test_report,
        },
    )
    save_json(prepared.run_dir / "run_metadata.json", run_metadata)
    if test_report is not None:
        save_json(prepared.run_dir / "test_report.json", test_report)

    print(f"Saved checkpoint to {checkpoint_path}")
    if best_checkpoint is not None:
        print(f"Saved best checkpoint to {prepared.run_dir / 'best_model_checkpoint.pt'}")
    print(f"Saved dataset summary to {prepared.run_dir / 'dataset_summary.json'}")
    print(f"Saved run config to {prepared.run_dir / 'run_config.json'}")
    print(f"Saved run metadata to {prepared.run_dir / 'run_metadata.json'}")
    if test_report is not None:
        print(f"Saved test report to {prepared.run_dir / 'test_report.json'}")


def run_training(config: TrainConfig) -> Path:
    set_seed(config.seed, deterministic=config.deterministic)
    prepared = prepare_run(config)
    history, best_checkpoint = train_and_select_checkpoint(prepared, config)
    test_report = evaluate_held_out_test_split(prepared, config, checkpoint=best_checkpoint)
    persist_run_outputs(prepared, history=history, best_checkpoint=best_checkpoint, test_report=test_report)
    return prepared.run_dir
