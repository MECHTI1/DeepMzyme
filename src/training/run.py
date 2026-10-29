from __future__ import annotations

import copy
import csv
import json
import math
import random
import resource
import sys
from time import perf_counter
import warnings
from dataclasses import dataclass, replace
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch_geometric.loader import DataLoader

from label_schemes import (
    ACTIVE_METAL_LABEL_SCHEME,
    COLLAPSED_METAL_LABELS,
    METAL_TARGET_LABELS,
    configure_active_metal_label_scheme,
)
from metal_objectives import (
    collapse_metal_label_ids_to_4,
    collapse_metal_logits_to_4,
    collapse_metal_targets_to_4,
    validate_required_six_class_metal_labels,
)
from model_variants import build_pocket_classifier
from project_paths import resolve_runs_dir
from training.labels import ec_prefix_from_label_token, parse_structure_identity
from training.config import (
    DEFAULT_SELECTED_SITE_POCKET_GROUPING,
    VALID_SELECTED_SITE_POCKET_GROUPING_CHOICES,
    TrainConfig,
    config_to_payload,
    default_selection_metric_for_task,
    parse_ec_aux_levels,
    required_targets_for_task,
)
from training.data import load_training_pockets_with_report_from_dir
from training.ec_class_metrics import ec_class_metrics_from_logits
from training.ec_membership import ec_membership_split, read_ec_fold_membership
from training.ec_classes import complete_ec_prefix
from training.ec_classes import (
    EC_CLASS_TOKEN_TO_INDEX, assert_consistent_ec_protein_labels, ec_class_split_support,
)
from training.evaluation_protocols import enforce_held_out_overlap_policy
from training.final_test_reporting import (
    build_metal_final_reporting_payload,
    build_softmax_mean_ensemble_payload,
)
from training.graph_dataset import (
    FeatureNormalizationStats,
    PocketGraphDataset,
    build_graph_data_list,
    compute_feature_normalization_stats,
)
from training.loop import (
    balanced_class_weights_from_pockets,
    class_weights_from_labels,
    classification_metrics_from_logits,
    evaluate_epoch_with_predictions,
    train_epoch,
)
from training.preflight import run_preflight_checks
from training.runtime_preparation import prepare_runtime_inputs
from training.reproducibility import (
    runtime_environment_payload,
    source_artifacts_payload,
    source_control_payload,
)
from training.splits import (
    PocketSplit,
    assign_ec_group_metadata,
    build_balanced_metal_site_sampler,
    build_dataset_summary,
    build_split_diagnostics,
    ec_grouping_mode_for_metrics,
    format_split_diagnostics,
    split_pockets,
    split_pockets_k_fold,
)
from training.structure_loading import find_structure_files
from training.campaign_runtime import (
    binding_bias_state,
    export_selected_validation_predictions,
    first_shell_support,
    membership_split,
    read_fold_membership,
    verify_fold_membership,
)


@dataclass(frozen=True)
class PreparedRun:
    config_payload: dict[str, Any]
    runtime_environment: dict[str, Any]
    source_artifacts: dict[str, Any]
    source_control: dict[str, Any]
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
    train_eval_loader: DataLoader | None = None


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


def split_seed_for_config(config: TrainConfig) -> int:
    return int(config.seed if config.split_seed is None else config.split_seed)


def resolve_selection_metric(config: TrainConfig) -> TrainConfig:
    if config.selection_metric is not None:
        return config
    metric = default_selection_metric_for_task(
        "ec" if config.controlled_ec_auxiliary else config.task,
        has_validation=config.val_fraction > 0.0 or config.n_folds is not None or bool(config.explicit_membership_manifest),
    )
    return replace(config, selection_metric=metric)


def dataloader_runtime_kwargs(config: TrainConfig) -> dict[str, Any]:
    effective_pin_memory = bool(config.pin_memory) and str(config.device).startswith("cuda")
    kwargs: dict[str, Any] = {
        "num_workers": int(config.num_workers),
        "pin_memory": effective_pin_memory,
    }
    if config.num_workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs


PRELAUNCH_RUN_DIR_FILENAMES = frozenset(
    {
        "active_run_config.json",
        "active_run_config.md",
        # The bounded campaign controller opens this before launching the
        # trainer so subprocess output survives setup failures.
        "execution.log",
        "optuna_trial_status.json",
    }
)


def reusable_prelaunch_run_dir(run_dir: Path) -> bool:
    if not run_dir.is_dir():
        return False
    existing_names = {path.name for path in run_dir.iterdir()}
    return existing_names.issubset(PRELAUNCH_RUN_DIR_FILENAMES)


def build_run_dir(config: TrainConfig) -> Path:
    runs_dir = resolve_runs_dir(config.runs_dir, create=True)
    effective_name = config.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = runs_dir / effective_name
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        if reusable_prelaunch_run_dir(run_dir):
            stale_status_path = run_dir / "optuna_trial_status.json"
            if stale_status_path.exists():
                stale_status_path.unlink()
            return run_dir
        existing_names = sorted(path.name for path in run_dir.iterdir()) if run_dir.is_dir() else []
        detail = f" Existing entries: {', '.join(existing_names[:10])}." if existing_names else ""
        raise FileExistsError(
            f"Run directory already exists and is not an empty/notebook prelaunch directory: {run_dir}.{detail} "
            "Choose a new --run-name or remove the existing run directory."
        ) from exc
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
    return source_control_payload(Path(__file__).resolve().parents[2]).get("commit")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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
    if "train_and_test_sets_structures_harsh_pinmymetal" in normalized:
        return {
            "split_name": "Harsh Split PinMyMetal",
            "split_type": "harsh_pinmymetal",
            "overlap_warning": None,
        }
    if "train_and_test_sets_structures_non_overlapped_pinmymetal" in normalized:
        return {
            "split_name": "Non-overlapped PinMyMetal",
            "split_type": "non_overlapped_pinmymetal",
            "overlap_warning": None,
        }
    if "train_and_test_sets_structures_exact_pinmymetal" in normalized:
        return {
            "split_name": "Metal Split PinMyMetal",
            "split_type": "metal_split_pinmymetal_possibly_overlapped",
            "overlap_warning": (
                "Metal Split PinMyMetal follows the exact PinMyMetal train/test PDB-ID membership "
                "for available supported structures and may contain train/test overlap. "
                "It should "
                "be interpreted only as a secondary/reference result."
            ),
        }
    if "train_and_test_sets_structures_common_pdbid_70_30_pinmymetal" in normalized:
        return {
            "split_name": "Common-PDBID 70/30 Split PinMyMetal",
            "split_type": "common_pdbid_70_30_pinmymetal",
            "overlap_warning": (
                "Common-PDBID 70/30 Split PinMyMetal is a custom comparison split, "
                "not the trusted final held-out split."
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
        pocket_ids.add(str(pocket.metadata.get("parent_pocket_id", pocket.pocket_id)))
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
    overlap_identities = {
        key: sorted(train_sets[key].intersection(test_sets[key]))
        for key in sorted(train_sets)
    }
    detected = any(count > 0 for count in overlap_counts.values())
    return {
        "train_test_overlap_detected": detected,
        "overlap_counts": overlap_counts,
        "overlap_examples": overlap_examples,
        "overlap_identities": overlap_identities,
        "overlap_warning": "Train/test overlap detected." if detected else None,
    }


def structure_file_identity_sets(structure_paths: list[Path]) -> dict[str, set[str]]:
    filenames: set[str] = set()
    structure_ids: set[str] = set()
    pdb_ids: set[str] = set()
    pdb_chain_ids: set[str] = set()
    for path in structure_paths:
        filenames.add(path.name.lower())
        structure_id = path.stem
        structure_ids.add(structure_id.lower())
        try:
            pdb_id, chain_id, _ec = parse_structure_identity(structure_id)
        except ValueError as exc:
            raise ValueError(
                "Cannot verify held-out train/test group disjointness because a structure "
                f"filename does not encode the expected PDB/chain/EC identity: {path}"
            ) from exc
        pdb_ids.add(pdb_id)
        pdb_chain_ids.add(f"{pdb_id}__chain_{chain_id}")
    return {
        "filename": filenames,
        "structure_id": structure_ids,
        "pdb_id": pdb_ids,
        "pdb_chain": pdb_chain_ids,
    }


def held_out_structure_overlap_report(
    train_structure_dir: Path,
    test_structure_dir: Path,
) -> dict[str, Any]:
    train_sets = structure_file_identity_sets(find_structure_files(Path(train_structure_dir)))
    test_sets = structure_file_identity_sets(find_structure_files(Path(test_structure_dir)))
    overlap_counts = {
        key: len(train_sets[key].intersection(test_sets[key]))
        for key in sorted(train_sets)
    }
    overlap_examples = {
        key: sorted(train_sets[key].intersection(test_sets[key]))[:10]
        for key in sorted(train_sets)
    }
    overlap_identities = {
        key: sorted(train_sets[key].intersection(test_sets[key]))
        for key in sorted(train_sets)
    }
    detected = any(count > 0 for count in overlap_counts.values())
    return {
        "train_test_overlap_detected": detected,
        "overlap_counts": overlap_counts,
        "overlap_examples": overlap_examples,
        "overlap_identities": overlap_identities,
        "overlap_warning": "Train/test structure-group overlap detected." if detected else None,
    }


def validate_held_out_structure_disjointness(config: TrainConfig) -> dict[str, Any] | None:
    if not config.run_test_eval or config.test_structure_dir is None:
        return None
    report = held_out_structure_overlap_report(config.structure_dir, config.test_structure_dir)
    return enforce_held_out_overlap_policy(config, report, phase="structure_files_before_inference")


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
    from training.explicit_membership import validate_mode
    validate_mode(config)
    if config.metal_example_unit not in {"pocket", "ion"}:
        raise ValueError(f"Unsupported metal example unit: {config.metal_example_unit!r}")
    if config.metal_example_unit == "ion" and config.task != "metal":
        raise ValueError("--metal-example-unit ion currently requires --task metal")
    configure_active_metal_label_scheme(config.metal_label_scheme)
    config = resolve_selection_metric(config)
    if config.gvp_weight_decay is not None and config.gvp_weight_decay < 0.0:
        raise ValueError("--gvp-weight-decay must be >= 0")
    if config.gvp_target_shrink is not None:
        if not 0.0 < config.gvp_target_shrink < 1.0:
            raise ValueError("--gvp-target-shrink must be in (0, 1)")
        if config.gvp_weight_decay is not None:
            raise ValueError("--gvp-target-shrink cannot be combined with --gvp-weight-decay")
        if config.gvp_target_shrink_scope not in {"trunk", "all"}:
            raise ValueError(f"Unsupported --gvp-target-shrink-scope {config.gvp_target_shrink_scope!r}")
    if (config.gvp_weight_decay is not None or config.gvp_target_shrink is not None) and \
            config.model_architecture not in {"gvp", "only_gvp"}:
        raise ValueError("--gvp-weight-decay and --gvp-target-shrink require --model-architecture gvp or only_gvp")
    if config.site_extraction not in {"component", "selected_sites"}:
        raise ValueError(f"Unsupported site extraction: {config.site_extraction!r}")
    if config.site_extraction == "selected_sites" and (
        config.metal_example_unit != "pocket" or config.source_cohort_csv is not None or config.summary_csv is None
    ):
        raise ValueError("--site-extraction selected_sites requires pocket examples, a summary CSV and no source cohort")
    if config.selected_site_pocket_grouping not in VALID_SELECTED_SITE_POCKET_GROUPING_CHOICES:
        raise ValueError(f"Unsupported selected-site pocket grouping: {config.selected_site_pocket_grouping!r}")
    if config.selected_site_pocket_grouping != DEFAULT_SELECTED_SITE_POCKET_GROUPING and \
            config.site_extraction != "selected_sites":
        raise ValueError("--selected-site-pocket-grouping structure requires --site-extraction selected_sites")
    if uses_ec_fold_membership(config):
        if config.metal_example_unit != "pocket":
            raise ValueError("EC fold membership requires --metal-example-unit pocket")
        if config.fold_membership_csv is None or not config.fold_membership_sha256 or config.n_folds is None \
                or config.fold_index is None:
            raise ValueError("EC fold membership requires a membership file, its SHA-256, --n-folds and --fold-index")
        if config.run_test_eval or config.test_structure_dir is not None:
            raise ValueError("EC fold membership cannot be combined with a test set")
    if config.use_global_esm_branch and config.run_test_eval:
        raise ValueError("Held-out evaluation with the whole-protein branch is not supported by the training run")
    if config.use_global_esm_branch and (config.global_esm_cache_dir is None or not config.global_esm_manifest_sha256):
        raise ValueError("--use-global-esm-branch requires --global-esm-cache-dir and --global-esm-manifest-sha256")
    if config.model_architecture == "only_global" and not config.use_global_esm_branch:
        raise ValueError("--model-architecture only_global requires --use-global-esm-branch")
    if config.global_esm_standardize not in {"none", "train_proteins"}:
        raise ValueError(f"Unsupported --global-esm-standardize {config.global_esm_standardize!r}")
    if config.augmentation_rng not in {"global", "per_fit"}:
        raise ValueError(f"Unsupported --augmentation-rng {config.augmentation_rng!r}")
    if config.augmentation_rng == "per_fit" and config.num_workers != 0:
        raise ValueError("--augmentation-rng per_fit requires --num-workers 0")
    if config.ec_class_vocabulary not in {"observed", "fixed_seven"}:
        raise ValueError(f"Unsupported EC class vocabulary: {config.ec_class_vocabulary!r}")
    if config.ec_class_vocabulary == "fixed_seven":
        if config.task != "ec" or config.ec_label_depth != 1:
            raise ValueError("--ec-class-vocabulary fixed_seven requires --task ec and --ec-label-depth 1")
        if config.require_all_task_classes:
            raise ValueError("--ec-class-vocabulary fixed_seven cannot be combined with --require-all-task-classes")
    if config.ec_conflicting_labels not in {"skip", "fail"}:
        raise ValueError(f"Unsupported EC conflicting-label policy: {config.ec_conflicting_labels!r}")
    if config.ec_conflicting_labels == "fail" and config.task == "metal":
        raise ValueError("--ec-conflicting-labels fail requires an EC task")
    if config.selection_metric == "val_ec_group_ec_classes_1_6_balanced_acc" and config.ec_class_vocabulary != "fixed_seven":
        raise ValueError("val_ec_group_ec_classes_1_6_balanced_acc requires --ec-class-vocabulary fixed_seven")
    if config.export_validation_predictions:
        if config.source_cohort_csv is None or config.task != "metal" or config.metal_example_unit != "ion":
            raise ValueError("--export-validation-predictions requires --source-cohort-csv --task metal --metal-example-unit ion")
        if not (config.val_fraction > 0.0 or (config.n_folds is not None and config.fold_index is not None)):
            raise ValueError("--export-validation-predictions requires a validation split")
    if config.controlled_ec_auxiliary:
        requirements = {
            "--task joint": config.task == "joint",
            "--metal-label-scheme four_class": config.metal_label_scheme in {"four_class", "merge_fe_class_viii"},
            "GVP + graph-level late fusion": (
                config.model_architecture == "gvp" and config.fusion_mode == "late_fusion"
                and not config.use_early_esm
            ),
            "--joint-loss-weighting fixed": config.joint_loss_weighting == "fixed",
            "--ec-label-depth 1": config.ec_label_depth == 1,
            "--ec-loss-weight 1": config.ec_loss_weight == 1.0,
            "--ec-contrastive-weight 0": config.ec_contrastive_weight == 0.0,
            "--selection-metric val_ec_group_balanced_acc": config.selection_metric == "val_ec_group_balanced_acc",
            "grouped validation by pdbid": (
                config.train_val_split_by == "pdbid"
                and (config.val_fraction > 0.0 or config.n_folds is not None)
            ),
            "--ec-group-weighting structure_id": config.ec_group_weighting == "structure_id",
            "validation-only execution without test overrides": not (
                config.run_test_eval or config.allow_train_loss_test_eval_debug
                or config.allow_final_refit_test_eval
            ),
        }
        failures = [name for name, valid in requirements.items() if not valid]
        if failures:
            raise ValueError("Controlled EC auxiliary comparison requires: " + "; ".join(failures))
    if config.gvp_layers < 1:
        raise ValueError(f"--gvp-layers must be at least 1, got {config.gvp_layers}")
    if config.head_mlp_layers < 1:
        raise ValueError(f"--head-mlp-layers must be at least 1, got {config.head_mlp_layers}")
    if not 0.0 <= config.head_mlp_dropout <= 1.0:
        raise ValueError(
            "--head-mlp-dropout must be in [0, 1], "
            f"got {config.head_mlp_dropout}"
        )
    if not 0.0 <= config.esm_graph_encoder_dropout <= 1.0:
        raise ValueError(
            "--esm-graph-encoder-dropout must be in [0, 1], "
            f"got {config.esm_graph_encoder_dropout}"
        )
    if config.ec_label_depth < 1:
        raise ValueError(f"--ec-label-depth must be at least 1, got {config.ec_label_depth}")
    if config.grad_accum_steps < 1:
        raise ValueError(f"--grad-accum-steps must be at least 1, got {config.grad_accum_steps}.")
    if config.num_workers < 0:
        raise ValueError(f"--num-workers must be >= 0, got {config.num_workers}.")
    ec_grouping_mode_for_metrics(config.ec_group_weighting)
    if config.ec_contrastive_weight < 0.0:
        raise ValueError(
            f"--ec-contrastive-weight must be non-negative, got {config.ec_contrastive_weight}"
        )
    if config.ec_contrastive_temperature <= 0.0:
        raise ValueError(
            "--ec-contrastive-temperature must be positive, "
            f"got {config.ec_contrastive_temperature}"
        )
    if config.position_noise_std < 0.0:
        raise ValueError(f"--position-noise-std must be non-negative, got {config.position_noise_std}")
    if config.classifier_pool_distance_cutoff < 0.0:
        raise ValueError(
            "--classifier-pool-distance-cutoff must be non-negative, "
            f"got {config.classifier_pool_distance_cutoff}"
        )
    if config.metal_node_mode != "none" and config.model_architecture not in {"gvp", "only_gvp"}:
        raise ValueError(
            "--metal-node-mode is currently implemented only for GVP graph models "
            "(--model-architecture gvp or only_gvp)."
        )
    if config.structural_readout_scope == "metal_only" and config.metal_node_mode == "none":
        raise ValueError("--structural-readout-scope metal_only requires --metal-node-mode per_metal.")
    if not 0.0 <= config.second_shell_dropout <= 1.0:
        raise ValueError(
            "--second-shell-dropout must be in [0, 1], "
            f"got {config.second_shell_dropout}"
        )
    if not 0.0 <= config.outer_residue_dropout <= 1.0:
        raise ValueError(
            "--outer-residue-dropout must be in [0, 1], "
            f"got {config.outer_residue_dropout}"
        )
    if config.metal_loss_weight < 0.0:
        raise ValueError(f"--metal-loss-weight must be non-negative, got {config.metal_loss_weight}")
    if config.ec_loss_weight < 0.0:
        raise ValueError(f"--ec-loss-weight must be non-negative, got {config.ec_loss_weight}")
    if not 0.0 <= config.metal_collapsed_loss_weight <= 1.0:
        raise ValueError(
            "--metal-collapsed-loss-weight must be in [0, 1], "
            f"got {config.metal_collapsed_loss_weight}"
        )
    if config.metal_collapsed_loss_weight > 0.0:
        if not task_predicts_metal(config.task):
            raise ValueError(
                "--metal-collapsed-loss-weight is only valid when a metal prediction head "
                "is active (task=metal or task=joint); EC-only training has no metal loss."
            )
        if config.metal_loss_function != "cross_entropy":
            raise ValueError(
                "--metal-collapsed-loss-weight currently requires --metal-loss-function cross_entropy "
                "so L_total=(1-alpha)*CE_6class + alpha*CE_4class is well-defined."
            )
        validate_required_six_class_metal_labels(METAL_TARGET_LABELS)
    if config.source_cohort_csv is not None:
        if config.run_test_eval or config.test_structure_dir is not None or config.test_summary_csv is not None:
            raise ValueError("Source-cohort development runs must not receive held-out test paths or evaluation")
        if config.prepare_missing_esm_embeddings and config.require_esm_embeddings:
            raise ValueError(
                "Source-cohort runs require --no-prepare-missing-esm-embeddings: feature preparation "
                "belongs to the frozen feature inventory, not to a fit"
            )
        if config.explicit_membership_manifest:
            raise ValueError("--source-cohort-csv cannot be combined with --explicit-membership-manifest")
    if config.binding_residue_pooling != "none":
        if config.model_architecture not in {"gvp", "only_gvp", "only_esm"}:
            raise ValueError("--binding-residue-pooling requires gvp, only_gvp or only_esm")
        if config.fusion_mode == "cross_modal_attention" and config.model_architecture == "gvp":
            raise ValueError("--binding-residue-pooling is not defined for cross_modal_attention")
    if config.metal_class_weight_mode not in {"none", "manual", "inverse_frequency", "inverse_sqrt_frequency", "effective_number"}:
        raise ValueError(f"Unsupported --metal-class-weight-mode {config.metal_class_weight_mode!r}")
    for field_name in (
        "mn_loss_multiplier",
        "cu_loss_multiplier",
        "zn_loss_multiplier",
        "fe_loss_multiplier",
        "co_loss_multiplier",
        "ni_loss_multiplier",
        "class_viii_loss_multiplier",
    ):
        value = float(getattr(config, field_name))
        if value < 0.0:
            flag_name = "--" + field_name.replace("_", "-")
            raise ValueError(f"{flag_name} must be non-negative, got {value}")
    if config.joint_loss_weighting == "uncertainty" and config.task != "joint":
        raise ValueError(
            "--joint-loss-weighting uncertainty requires --task joint. "
            "Single-task metal or EC runs use fixed weighting."
        )
    has_validation = config.val_fraction > 0.0 or config.n_folds is not None
    if not has_validation and config.selection_metric.startswith("val_"):
        raise ValueError(
            "Selection metric "
            f"{config.selection_metric!r} requires validation, but --val-fraction is 0. "
            "Either enable validation or choose a train-based metric such as 'train_loss'."
        )
    if has_validation and config.selection_metric == "train_loss":
        warnings.warn(
            "selection_metric is 'train_loss' but a validation split is configured; "
            "model selection should normally use a validation metric.",
            RuntimeWarning,
        )
    if config.run_test_eval and config.allow_final_refit_test_eval and not config.final_test_selected_config_id:
        raise ValueError(
            "--allow-final-refit-test-eval requires --final-test-selected-config-id so the "
            "validation-selected configuration is recorded with the held-out test report."
        )
    if (
        config.run_test_eval
        and not config.allow_train_loss_test_eval_debug
        and not config.allow_final_refit_test_eval
    ):
        raise ValueError(
            "Reportable --run-test-eval requires the frozen Stage 6B final full-train refit. "
            "Pass --allow-final-refit-test-eval together with --final-test-selected-config-id "
            "only after Stage 6B promotion/refit provenance has been verified. For a tiny, "
            "non-reportable synthetic debug run only, use --allow-train-loss-test-eval-debug."
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
    if config.final_test_primary_report not in {"single_checkpoint", "softmax_mean_5_seeds"}:
        raise ValueError(f"Unsupported final_test_primary_report={config.final_test_primary_report!r}.")
    if config.final_test_ensemble_mode not in {"single_checkpoint", "softmax_mean_5_seeds"}:
        raise ValueError(f"Unsupported final_test_ensemble_mode={config.final_test_ensemble_mode!r}.")
    valid_final_test_roles = {
        "primary_final_report",
        "secondary_diagnostic_report",
        "primary_preselected",
        "exploratory_posthoc",
    }
    if config.final_test_result_role not in valid_final_test_roles:
        raise ValueError(f"Unsupported final_test_result_role={config.final_test_result_role!r}.")
    if config.final_test_calibration_bins < 1:
        raise ValueError("--final-test-calibration-bins must be positive.")
    if config.final_test_bootstrap_resamples < 1:
        raise ValueError("--final-test-bootstrap-resamples must be positive.")
    if not 0.0 < config.final_test_bootstrap_confidence_level < 1.0:
        raise ValueError("--final-test-bootstrap-confidence-level must be in (0, 1).")


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
    return collapse_metal_logits_to_4(logits, label_map=METAL_TARGET_LABELS, require_six_class=False)


def collapse_metal_targets(targets: torch.Tensor) -> torch.Tensor:
    return collapse_metal_targets_to_4(targets, label_map=METAL_TARGET_LABELS, require_six_class=False)


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


def assign_ec_auxiliary_targets(split: PocketSplit, config: TrainConfig) -> dict[str, Any]:
    """Training-only vocabularies for deeper EC levels; sets ``ec_aux_l{level}`` on every pocket.

    Labels are the single complete EC prefix at that level (``complete_ec_prefix``);
    validation pockets and training pockets without one get -1. Returns {} when no
    auxiliary objective is enabled.
    """
    levels = set(parse_ec_aux_levels(config.ec_aux_levels))
    if config.ec_supcon_weight > 0.0:
        levels.add(3)
    vocabularies: dict[str, Any] = {}
    for level in sorted(levels):
        prefixes = [complete_ec_prefix(_pocket_ec_numbers(pocket), level) for pocket in split.train_pockets]
        tokens = sorted({prefix for prefix in prefixes if prefix is not None})
        index = {token: position for position, token in enumerate(tokens)}
        for pocket, prefix in zip(split.train_pockets, prefixes):
            pocket.metadata[f"ec_aux_l{level}"] = index.get(prefix, -1) if prefix is not None else -1
        for pocket in split.val_pockets:
            pocket.metadata[f"ec_aux_l{level}"] = -1
        vocabularies[str(level)] = {
            "tokens": tokens,
            "n_train_pockets_labelled": sum(prefix is not None for prefix in prefixes),
            "n_train_pockets_masked": sum(prefix is None for prefix in prefixes),
        }
    return vocabularies


def _pocket_ec_numbers(pocket) -> tuple[str, ...]:
    values = pocket.metadata.get("ec_numbers")
    if values is None:
        return ()
    return tuple(values.split(";")) if isinstance(values, str) else tuple(str(value) for value in values)


AUGMENTATION_SEED_OFFSET = 7919


def attach_global_esm_vectors(pockets, config: TrainConfig) -> dict[str, Any]:
    """Load one chain-mean vector per loaded structure (exact names only) and attach it to every pocket."""
    from training.global_esm_features import attach_global_esm_to_pockets, load_global_esm_vectors

    structure_ids = sorted({pocket.structure_id for pocket in pockets})
    vectors = load_global_esm_vectors(
        Path(config.global_esm_cache_dir), str(config.global_esm_manifest_sha256), structure_ids,
        expected_layer=config.global_esm_layer, expected_dim=config.global_esm_dim,
    )
    attach = attach_global_esm_to_pockets(pockets, vectors)
    return {"cache_dir": str(config.global_esm_cache_dir), "manifest_sha256": config.global_esm_manifest_sha256,
            "layer": config.global_esm_layer, "dim": config.global_esm_dim, "n_structures": len(structure_ids),
            "n_pockets": len(pockets), "attach": attach}


def standardize_global_esm_vectors(split: PocketSplit, config: TrainConfig) -> dict[str, Any] | None:
    """Optionally z-score every pocket's vector with statistics of one vector per training protein."""
    from training.global_esm_features import GLOBAL_ESM_METADATA_KEY

    if config.global_esm_standardize == "none":
        return None
    per_protein = {}
    for pocket in split.train_pockets:
        per_protein.setdefault(pocket.structure_id, pocket.metadata[GLOBAL_ESM_METADATA_KEY])
    stacked = torch.stack([per_protein[key] for key in sorted(per_protein)], dim=0)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0, unbiased=False).clamp_min(1e-6)
    for pocket in [*split.train_pockets, *split.val_pockets]:
        pocket.metadata[GLOBAL_ESM_METADATA_KEY] = (pocket.metadata[GLOBAL_ESM_METADATA_KEY] - mean) / std
    return {"mode": "train_proteins", "n_train_proteins": len(per_protein),
            "mean": mean.tolist(), "std": std.tolist()}


def uses_ec_fold_membership(config: TrainConfig) -> bool:
    """EC runs split by a frozen per-structure fold file (no source cohort)."""
    return config.fold_split_source == "membership" and config.task == "ec" and config.source_cohort_csv is None


def ec_metric_options(config: TrainConfig) -> dict[str, bool]:
    """Grouped EC metric options derived from the run configuration."""
    return {
        "fail_on_conflict": config.ec_conflicting_labels == "fail",
        "fixed_ec_classes": config.ec_class_vocabulary == "fixed_seven",
    }


def ec_group_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    group_indices: torch.Tensor,
    *,
    ec_label_map: dict[int, str],
    ec_label_depth: int,
    fail_on_conflict: bool = False,
    fixed_ec_classes: bool = False,
) -> dict[str, Any]:
    if fixed_ec_classes and dict(ec_label_map) != {index: token for token, index in EC_CLASS_TOKEN_TO_INDEX.items()}:
        raise ValueError("Fixed EC-class metrics need the fixed seven-class label map")
    grouped_logits: dict[int, list[torch.Tensor]] = {}
    grouped_targets: dict[int, set[int]] = {}
    for logit, target, group_idx in zip(logits, targets, group_indices):
        group_id = int(group_idx.item())
        if group_id < 0:
            continue
        grouped_logits.setdefault(group_id, []).append(logit)
        grouped_targets.setdefault(group_id, set()).add(int(target.item()))

    mean_logits: list[torch.Tensor] = []
    group_targets: list[int] = []
    conflicting_groups = 0
    for group_id in sorted(grouped_logits):
        target_values = grouped_targets[group_id]
        if len(target_values) != 1:
            if fail_on_conflict:
                raise ValueError(f"EC group {group_id} holds pockets with different EC labels {sorted(target_values)}")
            conflicting_groups += 1
            continue
        mean_logits.append(torch.stack(grouped_logits[group_id], dim=0).mean(dim=0))
        group_targets.append(next(iter(target_values)))

    payload: dict[str, Any] = {
        "n_groups": len(group_targets),
        "n_conflicting_groups": conflicting_groups,
    }
    if not mean_logits:
        return payload

    group_logits = torch.stack(mean_logits, dim=0)
    group_y = torch.tensor(group_targets, dtype=torch.long)
    metrics = classification_metrics_from_logits(group_logits, group_y)
    payload.update(
        {
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "min_recall": min(present_metric_values(metrics["per_class_recall"])),
            "per_class_recall": {
                label: metrics["per_class_recall"][index] for index, label in ec_label_map.items()
            },
            "per_class_support": {
                label: metrics["per_class_support"][index] for index, label in ec_label_map.items()
            },
            "confusion_matrix": metrics["confusion_matrix"],
        }
    )
    if fixed_ec_classes:
        fixed = ec_class_metrics_from_logits(group_logits.detach().cpu().numpy(), group_y.numpy())
        for key in (
            "ec_classes_1_6_balanced_accuracy", "ec_classes_1_6_n_present", "ec_classes_1_6_macro_f1",
            "seven_class_balanced_accuracy", "seven_class_n_present", "seven_class_macro_f1",
        ):
            payload[key] = fixed[key]
    for level in range(1, ec_label_depth + 1):
        level_metrics = ec_level_metrics_from_logits(
            group_logits,
            group_y,
            ec_label_map=ec_label_map,
            level=level,
        )
        if not level_metrics:
            continue
        payload[f"level_{level}_accuracy"] = level_metrics["accuracy"]
        payload[f"level_{level}_balanced_accuracy"] = level_metrics["balanced_accuracy"]
        payload[f"level_{level}_macro_f1"] = level_metrics["macro_f1"]
    return payload


def metrics_from_predictions(
    predictions: dict[str, Any],
    prefix: str,
    *,
    task: str,
    ec_label_map: dict[int, str],
    ec_label_depth: int,
    fail_on_conflict: bool = False,
    fixed_ec_classes: bool = False,
) -> dict[str, Any]:
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
        collapsed_recalls = present_metric_values(collapsed_metrics["per_class_recall"])
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
                f"{prefix}_metal_per_class_support": {
                    label_name: metal_metrics["per_class_support"][label_idx]
                    for label_idx, label_name in METAL_TARGET_LABELS.items()
                },
                f"{prefix}_metal_confusion_matrix": metal_metrics["confusion_matrix"],
                f"{prefix}_metal_collapsed4_acc": collapsed_metrics["accuracy"],
                f"{prefix}_metal_collapsed4_balanced_acc": collapsed_metrics["balanced_accuracy"],
                f"{prefix}_metal_collapsed4_macro_f1": collapsed_metrics["macro_f1"],
                f"{prefix}_metal_collapsed4_min_recall": float(min(collapsed_recalls)),
                f"{prefix}_metal_collapsed4_mn_recall": collapsed_metrics["per_class_recall"][0],
                f"{prefix}_metal_collapsed4_cu_recall": collapsed_metrics["per_class_recall"][1],
                f"{prefix}_metal_collapsed4_zn_recall": collapsed_metrics["per_class_recall"][2],
                f"{prefix}_metal_collapsed4_class_viii_recall": collapsed_metrics["per_class_recall"][3],
                f"{prefix}_metal_collapsed4_per_class_recall": {
                    label_name: collapsed_metrics["per_class_recall"][label_idx]
                    for label_idx, label_name in COLLAPSED_METAL_LABELS.items()
                },
                f"{prefix}_metal_collapsed4_per_class_support": {
                    label_name: collapsed_metrics["per_class_support"][label_idx]
                    for label_idx, label_name in COLLAPSED_METAL_LABELS.items()
                },
                f"{prefix}_metal_collapsed4_confusion_matrix": collapsed_metrics["confusion_matrix"],
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
                f"{prefix}_ec_per_class_support": {
                    label_name: ec_metrics["per_class_support"][label_idx]
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
        if prefix in {"val", "test"} and "ec_group_id" in predictions:
            group_metrics = ec_group_metrics_from_logits(
                predictions["ec_logits"],
                predictions["ec_y"],
                predictions["ec_group_id"],
                ec_label_map=ec_label_map,
                ec_label_depth=ec_label_depth,
                fail_on_conflict=fail_on_conflict,
                fixed_ec_classes=fixed_ec_classes,
            )
            payload.update(
                {
                    f"{prefix}_ec_group_acc": group_metrics.get("accuracy"),
                    f"{prefix}_ec_group_balanced_acc": group_metrics.get("balanced_accuracy"),
                    f"{prefix}_ec_group_macro_f1": group_metrics.get("macro_f1"),
                    f"{prefix}_ec_group_min_recall": group_metrics.get("min_recall"),
                    f"{prefix}_ec_group_per_class_recall": group_metrics.get("per_class_recall"),
                    f"{prefix}_ec_group_per_class_support": group_metrics.get("per_class_support"),
                    f"{prefix}_ec_group_confusion_matrix": group_metrics.get("confusion_matrix"),
                    f"{prefix}_ec_group_n_groups": group_metrics["n_groups"],
                    f"{prefix}_ec_group_n_conflicting_groups": group_metrics["n_conflicting_groups"],
                }
            )
            if fixed_ec_classes:
                payload.update(
                    {
                        f"{prefix}_ec_group_ec_classes_1_6_balanced_acc": group_metrics.get("ec_classes_1_6_balanced_accuracy"),
                        f"{prefix}_ec_group_ec_classes_1_6_n_present": group_metrics.get("ec_classes_1_6_n_present"),
                        f"{prefix}_ec_group_ec_classes_1_6_macro_f1": group_metrics.get("ec_classes_1_6_macro_f1"),
                        f"{prefix}_ec_group_seven_class_balanced_acc": group_metrics.get("seven_class_balanced_accuracy"),
                        f"{prefix}_ec_group_seven_class_n_present": group_metrics.get("seven_class_n_present"),
                        f"{prefix}_ec_group_seven_class_macro_f1": group_metrics.get("seven_class_macro_f1"),
                    }
                )
            for level in range(1, ec_label_depth + 1):
                if f"level_{level}_accuracy" not in group_metrics:
                    continue
                payload.update(
                    {
                        f"{prefix}_ec_group_level_{level}_acc": group_metrics[f"level_{level}_accuracy"],
                        f"{prefix}_ec_group_level_{level}_balanced_acc": group_metrics[
                            f"level_{level}_balanced_accuracy"
                        ],
                        f"{prefix}_ec_group_level_{level}_macro_f1": group_metrics[f"level_{level}_macro_f1"],
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


def evaluate_split_metrics(
    model,
    loader: DataLoader | None,
    device: str,
    prefix: str,
    *,
    task: str,
    ec_label_map: dict[int, str],
    ec_label_depth: int,
    fail_on_conflict: bool = False,
    fixed_ec_classes: bool = False,
) -> dict[str, Any]:
    if loader is None:
        return {}

    predictions = evaluate_epoch_with_predictions(model, loader, device=device)
    return metrics_from_predictions(
        predictions,
        prefix,
        task=task,
        ec_label_map=ec_label_map,
        ec_label_depth=ec_label_depth,
        fail_on_conflict=fail_on_conflict,
        fixed_ec_classes=fixed_ec_classes,
    )


def normalization_stats_payload(normalization_stats: FeatureNormalizationStats) -> dict[str, Any]:
    return {
        "means": normalization_stats.means,
        "stds": normalization_stats.stds,
        "clamp_value": normalization_stats.clamp_value,
    }


def normalization_stats_from_payload(payload: dict[str, Any]) -> FeatureNormalizationStats:
    return FeatureNormalizationStats(
        means={str(key): torch.as_tensor(value).float() for key, value in payload.get("means", {}).items()},
        stds={str(key): torch.as_tensor(value).float() for key, value in payload.get("stds", {}).items()},
        clamp_value=float(payload.get("clamp_value", 5.0)),
    )


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
        "metal_label_scheme": config_payload.get("metal_label_scheme", ACTIVE_METAL_LABEL_SCHEME),
        "ec_labels": ec_labels,
        "normalization_stats": normalization_stats_payload(normalization_stats),
        "dataset_summary": dataset_summary,
    }


def task_loss_weighting_state(model) -> dict[str, Any]:
    weighter = getattr(model, "task_loss_weighter", None)
    if weighter is None:
        return {}
    payload: dict[str, Any] = {"joint_loss_weighting": getattr(weighter, "mode", None)}
    for task_name in ("metal", "ec"):
        base_weight = getattr(weighter, f"{task_name}_loss_weight", None)
        log_variance = getattr(weighter, f"{task_name}_log_variance", None)
        if log_variance is None:
            if base_weight is not None:
                payload[f"{task_name}_loss_scale"] = float(base_weight)
            continue
        detached_log_variance = log_variance.detach().cpu()
        payload[f"{task_name}_loss_log_variance"] = float(detached_log_variance.item())
        payload[f"{task_name}_loss_scale"] = float(
            (float(base_weight) * torch.exp(-detached_log_variance)).item()
        )
    return payload


def _ordered_metric_fields(rows: list[dict[str, Any]], predicate) -> list[str]:
    fields: set[str] = set()
    for row in rows:
        fields.update(key for key in row if predicate(key))
    ordered = [field for field in ("epoch", "lr", "train_loss", "val_loss") if field in fields]
    ordered.extend(sorted(field for field in fields if field not in ordered))
    return ordered


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    if not fieldnames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    tmp_path.replace(path)


def write_epoch_metric_files(run_dir: Path, history: list[dict[str, Any]]) -> None:
    """Write per-epoch metric snapshots so external monitors can follow training."""
    if not history:
        return
    base_fields = {"epoch", "lr"}
    loss_state_fields = {
        "joint_loss_weighting",
        "metal_loss_scale",
        "ec_loss_scale",
        "metal_loss_log_variance",
        "ec_loss_log_variance",
    }
    full_fields = _ordered_metric_fields(history, lambda _key: True)
    train_fields = _ordered_metric_fields(
        history,
        lambda key: key in base_fields
        or key == "train_loss"
        or key.startswith("train_")
        or key in loss_state_fields,
    )
    val_fields = _ordered_metric_fields(history, lambda key: key in base_fields or key.startswith("val_"))
    _write_metrics_csv(run_dir / "epoch_metrics.csv", history, full_fields)
    _write_metrics_csv(run_dir / "train_metrics.csv", history, train_fields)
    _write_metrics_csv(run_dir / "val_metrics.csv", history, val_fields)


def format_per_class_metric_block(record: dict[str, Any], key: str, support_key: str) -> str | None:
    values = record.get(key)
    if not isinstance(values, dict) or not values:
        return None
    supports = record.get(support_key)
    if not isinstance(supports, dict):
        supports = {}
    formatted = []
    for label, value in values.items():
        if value is None:
            value_text = "NA"
        else:
            value_text = f"{float(value):.4f}"
        support = supports.get(label)
        if support is None:
            formatted.append(f"{label}:{value_text}")
        else:
            formatted.append(f"{label}:{value_text}(n={support})")
    return ",".join(formatted)


def format_epoch_log(record: dict[str, Any], *, include_per_class: bool = False) -> str:
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
    if record.get("val_metal_balanced_acc") is not None:
        parts.append(f"val_metal_bal_acc={record['val_metal_balanced_acc']:.4f}")
    if record.get("val_ec_balanced_acc") is not None:
        parts.append(f"val_ec_bal_acc={record['val_ec_balanced_acc']:.4f}")
    if record.get("val_ec_group_balanced_acc") is not None:
        parts.append(f"val_ec_group_bal_acc={record['val_ec_group_balanced_acc']:.4f}")
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
    if record.get("metal_loss_scale") is not None and record.get("ec_loss_scale") is not None:
        parts.append(f"metal_loss_scale={record['metal_loss_scale']:.4f}")
        parts.append(f"ec_loss_scale={record['ec_loss_scale']:.4f}")
    if include_per_class:
        for label, key, support_key in (
            ("train_metal_class_recall", "train_metal_per_class_recall", "train_metal_per_class_support"),
            ("val_metal_class_recall", "val_metal_per_class_recall", "val_metal_per_class_support"),
            (
                "train_metal_collapsed4_class_recall",
                "train_metal_collapsed4_per_class_recall",
                "train_metal_collapsed4_per_class_support",
            ),
            (
                "val_metal_collapsed4_class_recall",
                "val_metal_collapsed4_per_class_recall",
                "val_metal_collapsed4_per_class_support",
            ),
            ("train_ec_class_recall", "train_ec_per_class_recall", "train_ec_per_class_support"),
            ("val_ec_class_recall", "val_ec_per_class_recall", "val_ec_per_class_support"),
        ):
            block = format_per_class_metric_block(record, key, support_key)
            if block:
                parts.append(f"{label}=[{block}]")
    return " ".join(parts)


GVP_TRUNK_PREFIXES = ("layers.", "node_scalar_encoder.", "edge_scalar_encoder.")
GVP_STRUCTURAL_PREFIXES = GVP_TRUNK_PREFIXES + ("init_vec_proj.", "gvp_attn_pool.", "gvp_fusion_proj.")


def gvp_learning_rate_prefixes(scope: str) -> tuple[str, ...]:
    """Parameter-name prefixes trained at --gvp-learning-rate (plan rank 4 extends the trunk)."""
    if scope == "trunk":
        return GVP_TRUNK_PREFIXES
    if scope == "structural":
        return GVP_STRUCTURAL_PREFIXES
    raise ValueError(f"Unsupported --gvp-lr-scope {scope!r}")


OPTIMIZER_GROUPS_SCHEMA_VERSION = 2


def scheduler_epoch_lr_factors(config: TrainConfig) -> list[float]:
    """Per-epoch learning-rate multipliers of the configured schedule (stepped once per epoch)."""
    probe = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([probe], lr=1.0)
    scheduler = build_scheduler(optimizer, config)
    factors = []
    for _ in range(int(config.epochs)):
        factors.append(float(optimizer.param_groups[0]["lr"]))
        if scheduler is not None:
            optimizer.step()
            scheduler.step()
    return factors


def build_optimizer_param_groups(
    model: torch.nn.Module,
    config: TrainConfig,
    *,
    steps_per_epoch: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """AdamW parameter groups plus the record written to optimizer_groups.json.

    The GVP trunk gets its own group when a distinct GVP learning rate, a GVP weight
    decay or a trunk target shrink is set (TECH-032: a GVP decay alone used to be
    ignored). Without a split, one group holds every parameter, as before. A target
    shrink sets the decay so that exp(-wd * sum of per-step learning rates) equals
    the target over this fit's planned optimizer steps.
    """
    has_trunk = hasattr(model, "layers")
    gvp_lr = config.gvp_learning_rate
    reasons = []
    if has_trunk and gvp_lr is not None and gvp_lr != config.learning_rate:
        reasons.append("distinct_gvp_learning_rate")
    if has_trunk and config.gvp_weight_decay is not None:
        reasons.append("gvp_weight_decay")
    if has_trunk and config.gvp_target_shrink is not None and config.gvp_target_shrink_scope == "trunk":
        reasons.append("gvp_target_shrink")
    if not has_trunk and (config.gvp_weight_decay is not None or config.gvp_target_shrink is not None):
        raise ValueError("GVP weight decay options need a model with a GVP trunk")
    ignored = []
    if not has_trunk and gvp_lr is not None:
        ignored.append("gvp_learning_rate")
    if has_trunk and not reasons and config.gvp_lr_scope != "trunk":
        ignored.append("gvp_lr_scope")

    if reasons:
        prefixes = gvp_learning_rate_prefixes(config.gvp_lr_scope)
        named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
        trunk = [(name, param) for name, param in named if any(name.startswith(prefix) for prefix in prefixes)]
        base = [(name, param) for name, param in named if not any(name.startswith(prefix) for prefix in prefixes)]
        specs = [
            ["gvp_trunk", trunk, gvp_lr if gvp_lr is not None else config.learning_rate,
             config.gvp_weight_decay if config.gvp_weight_decay is not None else config.weight_decay],
            ["base", base, config.learning_rate, config.weight_decay],
        ]
    else:
        # All parameters, including frozen ones, exactly as the single-group optimizer always did.
        specs = [["all", list(model.named_parameters()), config.learning_rate, config.weight_decay]]

    factors = scheduler_epoch_lr_factors(config)
    target_shrink = None
    if config.gvp_target_shrink is not None:
        target = float(config.gvp_target_shrink)
        shrink_groups = {"gvp_trunk"} if config.gvp_target_shrink_scope == "trunk" else {spec[0] for spec in specs}
        target_shrink = {"target": target, "scope": config.gvp_target_shrink_scope, "groups": {}}
        for spec in specs:
            if spec[0] not in shrink_groups:
                continue
            sum_lr = steps_per_epoch * spec[2] * sum(factors)
            if sum_lr <= 0:
                raise ValueError("A target shrink needs a positive learning-rate sum")
            spec[3] = -math.log(target) / sum_lr
            per_step = [spec[2] * factor * spec[3] for factor in factors]
            target_shrink["groups"][spec[0]] = {
                "computed_weight_decay": spec[3],
                "sum_lr": sum_lr,
                "planned_product_shrink": float(math.prod((1.0 - value) ** steps_per_epoch for value in per_step)),
                "peak_lr_times_weight_decay": max(per_step),
                "float32_step_rounds_to_one": bool(any(float(torch.tensor(1.0 - value, dtype=torch.float32)) == 1.0
                                                       for value in per_step if value > 0)),
            }

    param_groups = [{"params": [param for _, param in params], "lr": lr, "weight_decay": wd}
                    for _name, params, lr, wd in specs]
    by_name = {spec[0]: spec for spec in specs}
    record = {
        "schema_version": OPTIMIZER_GROUPS_SCHEMA_VERSION,
        "gvp_lr_scope": config.gvp_lr_scope,
        "gvp_learning_rate": gvp_lr,
        "learning_rate": config.learning_rate,
        "gvp_rate_parameters": [name for name, _ in by_name["gvp_trunk"][1]] if "gvp_trunk" in by_name else [],
        "base_rate_parameters": [name for name, _ in (by_name.get("base") or by_name["all"])[1]],
        "split_reason": reasons,
        "ignored_flags": ignored,
        "groups": [
            {"name": name, "lr": lr, "weight_decay": wd, "n_tensors": len(params),
             "n_values": int(sum(param.numel() for _, param in params)),
             "parameter_names": [param_name for param_name, _ in params]}
            for name, params, lr, wd in specs
        ],
        "optimizer_steps_per_epoch": int(steps_per_epoch),
        "planned_total_steps": int(steps_per_epoch) * int(config.epochs),
        "epoch_lr_factors": factors,
        "target_shrink": target_shrink,
    }
    return param_groups, record


def realised_optimizer_steps(optimizer: torch.optim.Optimizer) -> list[int | None]:
    """Largest AdamW step count per group; None when no parameter of the group was ever updated."""
    realised = []
    for group in optimizer.param_groups:
        steps = [int(float(optimizer.state[param]["step"])) for param in group["params"]
                 if param in optimizer.state and "step" in optimizer.state[param]]
        realised.append(max(steps) if steps else None)
    return realised


def finalize_optimizer_groups(
    run_dir: Path,
    optimizer: torch.optim.Optimizer,
    n_epochs_run: int,
    *,
    amp_active: bool = False,
) -> dict[str, Any]:
    """Add realised step counts to optimizer_groups.json; fail when they differ from the plan.

    Mixed precision may skip steps on overflow, so with AMP a difference is recorded, not raised.
    """
    path = run_dir / "optimizer_groups.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    planned = int(record["optimizer_steps_per_epoch"]) * int(n_epochs_run)
    realised = realised_optimizer_steps(optimizer)
    record["planned_steps_for_epochs_run"] = planned
    record["realised_total_steps"] = realised
    record["realised_steps_match"] = all(value is None or value == planned for value in realised)
    record["amp_active"] = bool(amp_active)
    save_json(path, record)
    if not record["realised_steps_match"] and not amp_active:
        raise ValueError(f"Optimizer steps {realised} differ from the planned {planned}; see {path}")
    return record


def metric_sort_value(record: dict[str, Any], selection_metric: str) -> tuple[float, bool]:
    if selection_metric not in record or record[selection_metric] is None:
        raise ValueError(f"Selection metric {selection_metric!r} is missing from the epoch record.")
    metric_value = float(record[selection_metric])
    if selection_metric.endswith("_loss"):
        return metric_value, False
    return metric_value, True


def prepare_run(config: TrainConfig) -> PreparedRun:
    configure_active_metal_label_scheme(config.metal_label_scheme)
    config = resolve_selection_metric(config)
    validate_training_configuration(config)
    from training.explicit_membership import load_membership, fixed_split
    membership = load_membership(config)
    config_payload = config_to_payload(config)
    config_payload.update(infer_split_identity(config))
    config_payload["model_seed"] = int(config.seed)
    config_payload["effective_split_seed"] = split_seed_for_config(config)
    if membership is not None:
        config_payload["explicit_membership"] = membership["receipt"]
        config_payload["split_name"] = membership["receipt"]["endpoint"]
        config_payload["split_type"] = "diagnostic_explicit_membership"
    config_payload["training_graph_augmentation"] = {
        "position_noise_std": float(config.position_noise_std),
        "second_shell_dropout": float(config.second_shell_dropout),
        "outer_residue_dropout": float(config.outer_residue_dropout),
        "rng": config.augmentation_rng,
        **({"rng_seed": int(config.seed) + AUGMENTATION_SEED_OFFSET} if config.augmentation_rng == "per_fit" else {}),
        "training_only": True,
        "validation_and_test_unchanged": True,
    }
    config_payload["training_sampler_seed"] = (
        int(config.seed)
        if config.balance_metal_site_symbols and task_predicts_metal(config.task)
        else None
    )
    runtime_environment = runtime_environment_payload()
    source_artifacts = source_artifacts_payload(config)
    source_control = source_control_payload(Path(__file__).resolve().parents[2])
    config_payload["git_commit"] = source_control.get("commit")
    config_payload["git_dirty"] = source_control.get("dirty")
    run_dir = build_run_dir(config)
    save_json(
        run_dir / "prepare_status.json",
        prepare_status_payload(stage="prepare_run", status="started", config_payload=config_payload),
    )
    try:
        runtime_preparation_report = ({
            "explicit_membership": membership["receipt"],
            "feature_generation": False,
            "outer_metadata_only_preflight": True,
        } if membership is not None else {
            # The frozen feature inventory certified coverage for exactly the cohort's
            # canonical files; a directory-wide scan would count unused chain aliases.
            "source_cohort": True,
            "feature_generation": False,
            "certified_by_feature_inventory_sha256": config.feature_inventory_sha256,
            "esm_embeddings_dir": config.esm_embeddings_dir,
            "external_features_root_dir": config.external_features_root_dir,
        } if config.source_cohort_csv is not None else prepare_runtime_inputs(
            structure_dir=config.structure_dir,
            esm_embeddings_dir=config.esm_embeddings_dir,
            require_esm_embeddings=config.require_esm_embeddings,
            prepare_missing_esm_embeddings=config.prepare_missing_esm_embeddings,
            ring_features_dir=config.ring_features_dir,
            external_features_root_dir=config.external_features_root_dir,
            external_feature_source=config.external_feature_source,
            require_external_features=config.require_external_features,
            prepare_missing_external_features=config.prepare_missing_external_features,
            use_ring_edges=config.use_ring_edges,
            require_ring_edges=config.require_ring_edges,
            prepare_missing_ring_edges=config.prepare_missing_ring_edges,
        ))
        test_runtime_preparation_report = None
        if config.run_test_eval and config.test_structure_dir is not None:
            test_runtime_preparation_report = prepare_runtime_inputs(
                structure_dir=config.test_structure_dir,
                esm_embeddings_dir=config.esm_embeddings_dir,
                require_esm_embeddings=config.require_esm_embeddings,
                prepare_missing_esm_embeddings=config.prepare_missing_esm_embeddings,
                ring_features_dir=config.ring_features_dir,
                external_features_root_dir=config.external_features_root_dir,
                external_feature_source=config.external_feature_source,
                require_external_features=config.require_external_features,
                prepare_missing_external_features=config.prepare_missing_external_features,
                use_ring_edges=config.use_ring_edges,
                require_ring_edges=config.require_ring_edges,
                prepare_missing_ring_edges=config.prepare_missing_ring_edges,
            )
        save_json(
            run_dir / "prepare_status.json",
            prepare_status_payload(
                stage="runtime_preparation",
                status="completed",
                config_payload=config_payload,
                extra={
                    "runtime_preparation": runtime_preparation_report,
                    "train_runtime_preparation": runtime_preparation_report,
                    "test_runtime_preparation": test_runtime_preparation_report,
                },
            ),
        )
        ec_fold_membership = (
            read_ec_fold_membership(Path(config.fold_membership_csv), str(config.fold_membership_sha256))
            if uses_ec_fold_membership(config) else None
        )
        load_result = load_training_pockets_with_report_from_dir(
            load_workers=config.load_workers,
            structure_dir=config.structure_dir,
            require_full_labels=True,
            required_targets=required_targets_for_task(config.task),
            metal_eligibility_scheme=config.metal_eligibility_scheme,
            metal_example_unit=config.metal_example_unit,
            summary_csv=config.summary_csv,
            esm_dim=config.esm_dim,
            esm_embeddings_dir=config.esm_embeddings_dir,
            require_esm_embeddings=config.require_esm_embeddings,
            ring_features_dir=config.ring_features_dir,
            external_features_root_dir=config.external_features_root_dir,
            external_feature_source=config.external_feature_source,
            require_external_features=config.require_external_features,
            unsupported_metal_policy=config.unsupported_metal_policy,
            invalid_structure_policy=config.invalid_structure_policy,
            ec_label_depth=config.ec_label_depth,
            ec_class_vocabulary=config.ec_class_vocabulary,
            site_extraction=config.site_extraction,
            selected_site_pocket_grouping=config.selected_site_pocket_grouping,
            source_cohort_csv=config.source_cohort_csv,
            source_cohort_sha256=config.source_cohort_sha256,
            feature_inventory_sha256=config.feature_inventory_sha256,
            **({"allowed_structure_ids": {
                row["structure_id"] for part in ("train", "inner_validation")
                for row in membership["partitions"][part]
            }} if membership is not None else {}),
            **({"allowed_structure_ids": set(ec_fold_membership.fold_by_structure)}
               if ec_fold_membership is not None else {}),
        )
        pockets = load_result.pockets
        if not pockets:
            raise ValueError("No training pockets were loaded.")
        if task_predicts_ec(config.task) and not load_result.ec_index_to_label:
            raise ValueError(
                "No EC labels were available after applying the configured "
                f"--ec-label-depth {config.ec_label_depth}."
            )
        global_esm_report = attach_global_esm_vectors(pockets, config) if config.use_global_esm_branch else None
        ec_protein_label_check = None
        if config.ec_conflicting_labels == "fail":
            ec_protein_label_check = assert_consistent_ec_protein_labels(pockets)
        save_json(
            run_dir / "prepare_status.json",
            prepare_status_payload(
                stage="load_training_pockets",
                status="completed",
                config_payload=config_payload,
                extra={
                    "n_loaded_pockets": len(pockets),
                    "feature_load_report": load_result.feature_report,
                    **({"ec_label_report": load_result.ec_label_report} if load_result.ec_label_report else {}),
                    **({"ec_protein_label_check": ec_protein_label_check} if ec_protein_label_check else {}),
                },
            ),
        )

        ec_fold_membership_receipt = None
        if membership is not None:
            split = fixed_split(pockets, membership)
        elif ec_fold_membership is not None:
            split, ec_fold_membership_receipt = ec_membership_split(
                pockets, ec_fold_membership, fold_index=int(config.fold_index), n_folds=int(config.n_folds),
            )
        elif config.fold_split_source == "membership":
            split = membership_split(
                pockets,
                read_fold_membership(Path(config.fold_membership_csv), str(config.fold_membership_sha256)),
                fold_index=int(config.fold_index),
                n_folds=int(config.n_folds),
            )
        elif config.n_folds is not None:
            split = split_pockets_k_fold(
                pockets,
                n_folds=config.n_folds,
                fold_index=int(config.fold_index),
                split_by=config.train_val_split_by,
                seed=split_seed_for_config(config),
                task=config.task,
                stratify_by=config.split_stratify_by,
            )
        else:
            split = split_pockets(
                pockets,
                val_fraction=config.val_fraction,
                split_by=config.train_val_split_by,
                seed=split_seed_for_config(config),
                task=config.task,
                stratify_by=config.split_stratify_by,
            )
        if config.export_validation_predictions and not split.val_pockets:
            raise ValueError("--export-validation-predictions requires a nonempty validation split")
        fold_membership_receipt = ec_fold_membership_receipt
        if config.fold_membership_csv is not None and ec_fold_membership is None:
            fold_membership_receipt = verify_fold_membership(
                train_pockets=split.train_pockets,
                val_pockets=split.val_pockets,
                fold_index=int(config.fold_index),
                membership=read_fold_membership(Path(config.fold_membership_csv), str(config.fold_membership_sha256)),
            )
        if task_predicts_ec(config.task):
            assign_ec_group_metadata(split.train_pockets, weighting_mode=config.ec_group_weighting)
            assign_ec_group_metadata(split.val_pockets, weighting_mode=config.ec_group_weighting)
        ec_auxiliary_vocabularies = assign_ec_auxiliary_targets(split, config)
        if global_esm_report is not None:
            global_esm_report["standardization"] = standardize_global_esm_vectors(split, config)
        split_diagnostics = build_split_diagnostics(
            split,
            config,
            ec_label_map=load_result.ec_index_to_label,
        )
        if config.ec_class_vocabulary == "fixed_seven":
            split_diagnostics["ec_class_support"] = ec_class_split_support(split.train_pockets, split.val_pockets)
            split_diagnostics["ec_label_report"] = load_result.ec_label_report
        print(format_split_diagnostics(split_diagnostics))
        save_json(run_dir / "split_diagnostics.json", split_diagnostics)
        dataset_summary = build_dataset_summary(
            split,
            config,
            feature_load_report=load_result.feature_report,
            ec_label_map=load_result.ec_index_to_label,
        )
        dataset_summary["split_diagnostics"] = split_diagnostics
        if fold_membership_receipt is not None:
            dataset_summary["fold_membership"] = fold_membership_receipt
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
        if ec_auxiliary_vocabularies:
            dataset_summary["ec_auxiliary_objectives"] = {"vocabularies": ec_auxiliary_vocabularies}
        if global_esm_report is not None:
            dataset_summary["global_esm"] = global_esm_report
        train_graphs = build_graph_data_list(
            split.train_pockets,
            esm_dim=config.esm_dim,
            edge_radius=config.edge_radius,
            use_ring_edges=config.use_ring_edges,
            require_ring_edges=config.require_ring_edges,
            shell_role_source=config.shell_role_source,
            node_feature_set=config.node_feature_set,
            omit_node_features=config.omit_node_features,
            metal_node_mode=config.metal_node_mode,
        )
        val_graphs = (
            build_graph_data_list(
                split.val_pockets,
                esm_dim=config.esm_dim,
                edge_radius=config.edge_radius,
                use_ring_edges=config.use_ring_edges,
                require_ring_edges=config.require_ring_edges,
                shell_role_source=config.shell_role_source,
                node_feature_set=config.node_feature_set,
                omit_node_features=config.omit_node_features,
                metal_node_mode=config.metal_node_mode,
            )
            if split.val_pockets
            else None
        )
        dataset_summary["first_shell_support"] = {
            "train": first_shell_support(train_graphs),
            "val": first_shell_support(val_graphs),
        }
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
        train_sampler = None
        if config.balance_metal_site_symbols and task_predicts_metal(config.task):
            sampler_generator = torch.Generator()
            sampler_generator.manual_seed(int(config.seed))
            train_sampler = build_balanced_metal_site_sampler(
                split.train_pockets,
                generator=sampler_generator,
            )
        loader_worker_kwargs = dataloader_runtime_kwargs(config)
        train_loader = DataLoader(
            PocketGraphDataset(
                split.train_pockets,
                esm_dim=config.esm_dim,
                edge_radius=config.edge_radius,
                normalization_stats=normalization_stats,
                use_ring_edges=config.use_ring_edges,
                require_ring_edges=config.require_ring_edges,
                shell_role_source=config.shell_role_source,
                precomputed_data=train_graphs,
                node_feature_set=config.node_feature_set,
                omit_node_features=config.omit_node_features,
                metal_node_mode=config.metal_node_mode,
                position_noise_std=config.position_noise_std,
                second_shell_dropout=config.second_shell_dropout,
                outer_residue_dropout=config.outer_residue_dropout,
                augmentation_generator=(
                    torch.Generator().manual_seed(int(config.seed) + AUGMENTATION_SEED_OFFSET)
                    if config.augmentation_rng == "per_fit" else None
                ),
            ),
            batch_size=config.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            generator=torch.Generator().manual_seed(int(config.seed)),
            **loader_worker_kwargs,
        )
        # Metrics use the unaugmented training examples and an independent RNG:
        # changing log frequency cannot consume shuffle/augmentation randomness.
        train_eval_loader = DataLoader(
            PocketGraphDataset(
                split.train_pockets, esm_dim=config.esm_dim, edge_radius=config.edge_radius,
                normalization_stats=normalization_stats, use_ring_edges=config.use_ring_edges,
                require_ring_edges=config.require_ring_edges, shell_role_source=config.shell_role_source,
                precomputed_data=train_graphs, node_feature_set=config.node_feature_set,
                omit_node_features=config.omit_node_features, metal_node_mode=config.metal_node_mode,
            ),
            batch_size=config.batch_size, shuffle=False,
            generator=torch.Generator().manual_seed(int(config.seed) + 1),
            **loader_worker_kwargs,
        )
        val_loader = (
            DataLoader(
                PocketGraphDataset(
                    split.val_pockets,
                    esm_dim=config.esm_dim,
                    edge_radius=config.edge_radius,
                    normalization_stats=normalization_stats,
                    use_ring_edges=config.use_ring_edges,
                    require_ring_edges=config.require_ring_edges,
                    shell_role_source=config.shell_role_source,
                    precomputed_data=val_graphs,
                    node_feature_set=config.node_feature_set,
                    omit_node_features=config.omit_node_features,
                    metal_node_mode=config.metal_node_mode,
                ),
                batch_size=config.batch_size,
                shuffle=False,
                generator=torch.Generator().manual_seed(int(config.seed) + 2),
                **loader_worker_kwargs,
            )
            if split.val_pockets
            else None
        )

        metal_class_weights = None
        metal_collapsed4_class_weights = None
        ec_class_weights = None
        computed_metal_weights, computed_ec_weights = balanced_class_weights_from_pockets(
            split.train_pockets,
            n_metal_classes=len(METAL_TARGET_LABELS),
            n_ec_classes=max(1, len(load_result.ec_index_to_label)),
            metal_class_weight_mode=config.metal_class_weight_mode,
            ec_class_weight_unit=config.ec_class_weight_unit if task_predicts_ec(config.task) else "pocket",
        )
        if task_predicts_metal(config.task) and config.metal_class_weight_mode != "none":
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
        if task_predicts_metal(config.task) and config.metal_collapsed_loss_weight > 0.0:
            train_metal_labels = [
                int(pocket.y_metal)
                for pocket in split.train_pockets
                if pocket.y_metal is not None
            ]
            collapsed_train_labels = collapse_metal_label_ids_to_4(
                train_metal_labels,
                label_map=METAL_TARGET_LABELS,
                require_six_class=True,
            )
            if config.metal_class_weight_mode != "none":
                metal_collapsed4_class_weights = class_weights_from_labels(
                    collapsed_train_labels,
                    len(COLLAPSED_METAL_LABELS),
                    mode=config.metal_class_weight_mode,
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
            n_metal=len(METAL_TARGET_LABELS),
            n_ec=max(1, len(load_result.ec_index_to_label)),
            esm_fusion_dim=config.esm_fusion_dim,
            head_mlp_layers=config.head_mlp_layers,
            head_mlp_dropout=config.head_mlp_dropout,
            esm_graph_encoder_dropout=config.esm_graph_encoder_dropout,
            node_rbf_sigma=config.node_rbf_sigma,
            edge_rbf_sigma=config.edge_rbf_sigma,
            node_rbf_use_raw_distances=config.node_rbf_use_raw_distances,
            edge_rbf_use_raw_distances=config.edge_rbf_use_raw_distances,
            classifier_pool_distance_cutoff=config.classifier_pool_distance_cutoff,
            structural_readout_scope=config.structural_readout_scope,
            use_node_type_embedding=config.metal_node_mode != "none",
            use_site_angle_features=config.metal_node_mode != "none",
            site_geometry_features=config.site_geometry_features,
            normalize_message_aggregation=config.normalize_message_aggregation,
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
            joint_loss_weighting=config.joint_loss_weighting,
            metal_loss_weight=config.metal_loss_weight,
            ec_loss_weight=config.ec_loss_weight,
            metal_loss_function=config.metal_loss_function,
            metal_focal_gamma=config.metal_focal_gamma,
            metal_label_smoothing=config.metal_label_smoothing,
            metal_collapsed_loss_weight=config.metal_collapsed_loss_weight,
            ec_contrastive_weight=config.ec_contrastive_weight,
            ec_contrastive_temperature=config.ec_contrastive_temperature,
            metal_class_weights=metal_class_weights,
            metal_collapsed4_class_weights=metal_collapsed4_class_weights,
            ec_class_weights=ec_class_weights,
            predict_metal=task_predicts_metal(config.task),
            predict_ec=task_predicts_ec(config.task),
            binding_residue_pooling=config.binding_residue_pooling,
            gvp_residual_dropout=config.gvp_residual_dropout,
            gvp_vector_norm=config.gvp_vector_norm,
            gvp_auxiliary_loss_weight=config.gvp_auxiliary_loss_weight,
            esm_modality_dropout=config.esm_modality_dropout,
            use_global_esm_branch=config.use_global_esm_branch,
            global_esm_dim=config.global_esm_dim,
            global_esm_dropout=config.global_esm_dropout,
        ).to(config.device)
        if ec_auxiliary_vocabularies:
            from ec_auxiliary_objectives import attach_ec_auxiliary_objectives

            attachments = attach_ec_auxiliary_objectives(
                model,
                aux_level_sizes={level: len(ec_auxiliary_vocabularies[str(level)]["tokens"])
                                 for level in parse_ec_aux_levels(config.ec_aux_levels)},
                aux_weight=config.ec_aux_weight,
                supcon_weight=config.ec_supcon_weight,
                supcon_temperature=config.ec_supcon_temperature,
                supcon_memory=config.ec_supcon_memory,
                supcon_projection_dim=config.ec_supcon_projection_dim,
            )
            model = model.to(config.device)
            dataset_summary["ec_auxiliary_objectives"]["attachments"] = attachments
        steps_per_epoch = math.ceil(len(train_loader) / config.grad_accum_steps)
        param_groups, optimizer_groups = build_optimizer_param_groups(model, config, steps_per_epoch=steps_per_epoch)
        for flag in optimizer_groups["ignored_flags"]:
            print(f"[WARN] {flag} has no effect for this model; recorded in optimizer_groups.json", flush=True)
        save_json(run_dir / "optimizer_groups.json", optimizer_groups)
        optimizer = torch.optim.AdamW(param_groups)
        scheduler = build_scheduler(optimizer, config)
        save_json(
            run_dir / "prepare_status.json",
            prepare_status_payload(stage="prepare_run", status="ready", config_payload=config_payload),
        )
        return PreparedRun(
            config_payload=config_payload,
            runtime_environment=runtime_environment,
            source_artifacts=source_artifacts,
            source_control=source_control,
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
            train_eval_loader=train_eval_loader,
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
    config = resolve_selection_metric(config)
    history: list[dict[str, Any]] = []
    best_metric: float | None = None
    best_checkpoint = None
    amp_active = bool(config.use_amp) and torch.cuda.is_available() and str(config.device).startswith("cuda")
    grad_scaler = torch.amp.GradScaler("cuda") if amp_active else None
    for epoch in range(1, config.epochs + 1):
        phase_started = perf_counter()
        train_loss = train_epoch(
            prepared.model,
            prepared.train_loader,
            prepared.optimizer,
            device=config.device,
            grad_clip_norm=config.grad_clip_norm,
            grad_accum_steps=config.grad_accum_steps,
            use_amp=config.use_amp,
            scaler=grad_scaler,
        )
        _record_runtime_phase(prepared.run_dir, config, "train_epoch", phase_started)
        every = max(1, int(config.train_metrics_every_n_epochs))
        if epoch % every == 0 or epoch == config.epochs or every == 1:
            phase_started = perf_counter()
            train_metrics = evaluate_split_metrics(
                prepared.model,
                prepared.train_eval_loader if prepared.train_eval_loader is not None else prepared.train_loader,
                config.device,
                prefix="train",
                task=config.task,
                ec_label_map=prepared.ec_labels,
                ec_label_depth=config.ec_label_depth,
                **ec_metric_options(config),
            )
            train_metrics.pop("train_loss", None)
            _record_runtime_phase(prepared.run_dir, config, "train_metric_evaluation", phase_started)
        else:
            # Logging-only skip; the selected checkpoint depends on validation metrics alone.
            train_metrics = {"train_metal_acc": None, "train_ec_acc": None, "train_metrics_skipped": True}
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": float(prepared.optimizer.param_groups[0]["lr"]),
            "lr_groups": [float(group["lr"]) for group in prepared.optimizer.param_groups],
            **train_metrics,
            **task_loss_weighting_state(prepared.model),
            **binding_bias_state(prepared.model),
        }

        phase_started = perf_counter()
        val_metrics = evaluate_split_metrics(
            prepared.model,
            prepared.val_loader,
            config.device,
            prefix="val",
            task=config.task,
            ec_label_map=prepared.ec_labels,
            ec_label_depth=config.ec_label_depth,
            **ec_metric_options(config),
        )
        if prepared.val_loader is not None:
            _record_runtime_phase(prepared.run_dir, config, "validation", phase_started)
        phase_started = perf_counter()
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
        write_epoch_metric_files(prepared.run_dir, history)
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
        _record_runtime_phase(prepared.run_dir, config, "checkpoint_save", phase_started)
        if prepared.scheduler is not None:
            prepared.scheduler.step()
        print(format_epoch_log(record, include_per_class=config.log_per_class_metrics))
    if config.checkpoint_rule == "terminal":
        return history, terminal_checkpoint(prepared, config, history, best_checkpoint)
    return history, best_checkpoint


def terminal_checkpoint(
    prepared: PreparedRun,
    config: TrainConfig,
    history: list[dict[str, Any]],
    best_checkpoint: dict[str, Any] | None,
) -> dict[str, Any]:
    """The final-epoch weights as the selected checkpoint; the best epoch stays descriptive only."""
    selected = checkpoint_payload(
        model_state_dict=copy.deepcopy(prepared.model.state_dict()),
        optimizer_state_dict=copy.deepcopy(prepared.optimizer.state_dict()),
        scheduler_state_dict=(copy.deepcopy(prepared.scheduler.state_dict()) if prepared.scheduler is not None else None),
        history=copy.deepcopy(history),
        config_payload=prepared.config_payload,
        normalization_stats=prepared.normalization_stats,
        dataset_summary=prepared.dataset_summary,
        ec_labels=prepared.ec_labels,
    )
    selected["epoch"] = len(history)
    selected["checkpoint_rule"] = "terminal"
    selected["selection_metric"] = "terminal_epoch"
    selected["selection_metric_value"] = None
    selected["descriptive_best_metric"] = config.selection_metric
    selected["descriptive_best_epoch"] = None if best_checkpoint is None else best_checkpoint.get("epoch")
    selected["descriptive_best_metric_value"] = (
        None if best_checkpoint is None else best_checkpoint.get("selection_metric_value")
    )
    return selected


def selected_checkpoint_filename(checkpoint: dict[str, Any] | None) -> str:
    """File name of the selected checkpoint: terminal-rule runs never write a "best" file."""
    if checkpoint is not None and checkpoint.get("checkpoint_rule") == "terminal":
        return "terminal_model_checkpoint.pt"
    return "best_model_checkpoint.pt"


def _record_runtime_phase(run_dir: Path, config: TrainConfig, phase: str, started: float, *, reset: bool = False) -> None:
    """Campaign wall timings and process high-water memory; never synchronize CUDA."""
    if not config.campaign_run_identity:
        return
    elapsed = perf_counter() - started
    path = run_dir / "runtime_profile.json"
    profile = json.loads(path.read_text(encoding="utf-8")) if path.is_file() and not reset else {
        "schema_version": 1,
        "phase_seconds": {key: 0.0 for key in ("prepare", "train_epoch", "train_metric_evaluation",
                                               "validation", "checkpoint_save", "selected_export")},
        "notes": ["Wall times do not add CUDA synchronization; asynchronous work may cross phase boundaries.",
                  "CPU peak RSS covers this process, excluding loader workers; CUDA peaks are process allocator high-water marks.",
                  "Profile I/O and orchestration overhead are excluded; preparation is recorded after successful completion."],
    }
    profile["phase_seconds"][phase] += elapsed
    profile["process_peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (
        1 if sys.platform == "darwin" else 1024
    )
    cuda_active = str(config.device).startswith("cuda") and torch.cuda.is_initialized()
    profile["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(config.device)) if cuda_active else 0
    profile["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(config.device)) if cuda_active else 0
    save_json(path, profile)


def evaluate_held_out_test_split(
    prepared: PreparedRun,
    config: TrainConfig,
    *,
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not config.run_test_eval or config.test_structure_dir is None or config.test_summary_csv is None:
        return None
    structure_overlap_report = validate_held_out_structure_disjointness(config)

    current_state_dict = copy.deepcopy(prepared.model.state_dict())
    model_state_dict = (
        checkpoint["model_state_dict"] if checkpoint is not None else prepared.model.state_dict()
    )
    prepared.model.load_state_dict(model_state_dict)
    try:
        test_load_result = load_training_pockets_with_report_from_dir(
            load_workers=config.load_workers,
            structure_dir=config.test_structure_dir,
            require_full_labels=True,
            required_targets=required_targets_for_task(config.task),
            summary_csv=config.test_summary_csv,
            esm_dim=config.esm_dim,
            esm_embeddings_dir=config.esm_embeddings_dir,
            require_esm_embeddings=config.require_esm_embeddings,
            ring_features_dir=config.ring_features_dir,
            external_features_root_dir=config.external_features_root_dir,
            external_feature_source=config.external_feature_source,
            require_external_features=config.require_external_features,
            unsupported_metal_policy=config.unsupported_metal_policy,
            invalid_structure_policy=config.invalid_structure_policy,
            ec_label_depth=config.ec_label_depth,
            ec_label_to_index=prepared.ec_label_to_index,
            ec_class_vocabulary=config.ec_class_vocabulary,
            site_extraction=config.site_extraction,
            selected_site_pocket_grouping=config.selected_site_pocket_grouping,
            metal_example_unit=config.metal_example_unit,
        )
        if not test_load_result.pockets:
            raise ValueError("No held-out test pockets were loaded.")
        if task_predicts_ec(config.task):
            assign_ec_group_metadata(test_load_result.pockets, weighting_mode=config.ec_group_weighting)

        train_like_pockets = prepared.split.train_pockets + prepared.split.val_pockets
        overlap_report = train_test_overlap_report(train_like_pockets, test_load_result.pockets)
        overlap_report = enforce_held_out_overlap_policy(
            config,
            overlap_report,
            phase="loaded_pockets_before_inference",
        )

        test_graphs = build_graph_data_list(
            test_load_result.pockets,
            esm_dim=config.esm_dim,
            edge_radius=config.edge_radius,
            use_ring_edges=config.use_ring_edges,
            require_ring_edges=config.require_ring_edges,
            shell_role_source=config.shell_role_source,
            node_feature_set=config.node_feature_set,
            omit_node_features=config.omit_node_features,
            metal_node_mode=config.metal_node_mode,
        )
        test_loader = DataLoader(
            PocketGraphDataset(
                test_load_result.pockets,
                esm_dim=config.esm_dim,
                edge_radius=config.edge_radius,
                normalization_stats=prepared.normalization_stats,
                use_ring_edges=config.use_ring_edges,
                require_ring_edges=config.require_ring_edges,
                shell_role_source=config.shell_role_source,
                precomputed_data=test_graphs,
                node_feature_set=config.node_feature_set,
                omit_node_features=config.omit_node_features,
                metal_node_mode=config.metal_node_mode,
            ),
            batch_size=config.batch_size,
            shuffle=False,
            **dataloader_runtime_kwargs(config),
        )
        test_predictions = evaluate_epoch_with_predictions(prepared.model, test_loader, device=config.device)
        metrics = metrics_from_predictions(
            test_predictions,
            "test",
            task=config.task,
            ec_label_map=prepared.ec_labels,
            ec_label_depth=config.ec_label_depth,
            **ec_metric_options(config),
        )
        final_reporting_payload: dict[str, Any] = {
            "metrics": metrics,
            "calibrated_metrics": {},
            "fitted_temperatures": {},
            "temperature_scaling": {
                "enabled": False,
                "unavailable_reason": "no metal logits were available for final-test calibration",
            },
            "bootstrap_settings": {
                "method": "not_run",
                "n_bootstrap": int(config.final_test_bootstrap_resamples),
                "confidence_level": float(config.final_test_bootstrap_confidence_level),
                "seed": int(config.final_test_bootstrap_seed),
            },
            "calibration_settings": {
                "enabled": bool(config.final_test_enable_calibration),
                "n_equal_mass_bins": int(config.final_test_calibration_bins),
            },
            "calibration_plot_paths": {},
            "reliability_diagram_path": None,
            "confidence_histogram_path": None,
            "prediction_artifact_path": None,
        }
        if task_predicts_metal(config.task) and "metal_logits" in test_predictions and "metal_y" in test_predictions:
            metal_test_pockets = [
                pocket for pocket in test_load_result.pockets if getattr(pocket, "y_metal", None) is not None
            ]
            metal_sample_ids = [str(pocket.pocket_id) for pocket in metal_test_pockets]
            metal_pdb_ids = [parse_structure_identity(pocket.structure_id)[0] for pocket in metal_test_pockets]
            if len(metal_sample_ids) != int(test_predictions["metal_y"].numel()):
                raise RuntimeError(
                    "Cannot persist aligned held-out prediction IDs: metal pocket count does not match target count."
                )
            val_predictions = (
                evaluate_epoch_with_predictions(prepared.model, prepared.val_loader, device=config.device)
                if prepared.val_loader is not None
                else {}
            )
            final_reporting_payload = build_metal_final_reporting_payload(
                output_dir=prepared.run_dir,
                metrics=metrics,
                test_logits=test_predictions["metal_logits"],
                test_targets=test_predictions["metal_y"],
                val_logits=val_predictions.get("metal_logits") if isinstance(val_predictions, dict) else None,
                val_targets=val_predictions.get("metal_y") if isinstance(val_predictions, dict) else None,
                task=config.task,
                enable_calibration=config.final_test_enable_calibration,
                enable_temperature_scaling=config.final_test_enable_temperature_scaling,
                enable_bootstrap_ci=config.final_test_enable_bootstrap_ci,
                n_bins=config.final_test_calibration_bins,
                n_bootstrap=config.final_test_bootstrap_resamples,
                confidence_level=config.final_test_bootstrap_confidence_level,
                bootstrap_seed=config.final_test_bootstrap_seed,
                sample_ids=metal_sample_ids,
                pdb_ids=metal_pdb_ids,
            )
        split_identity = infer_split_identity(config)
        overlap_warning = overlap_report.get("overlap_warning") or split_identity.get("overlap_warning")
        return {
            "task": config.task,
            "final_test_primary_report": config.final_test_primary_report,
            "final_test_ensemble_mode": config.final_test_ensemble_mode,
            "final_test_result_role": config.final_test_result_role,
            "primary_final_report": config.final_test_result_role in {"primary_final_report", "primary_preselected"},
            "secondary_diagnostic_report": config.final_test_result_role in {"secondary_diagnostic_report", "exploratory_posthoc"},
            "primary_preselected": config.final_test_result_role == "primary_preselected",
            "exploratory_posthoc": config.final_test_result_role == "exploratory_posthoc",
            "selected_config_id": config.final_test_selected_config_id,
            "selected_run_id": (
                Path(config.final_test_source_run_dirs[0]).name
                if config.final_test_source_run_dirs
                else config.run_name
            ),
            "checkpoint_paths": list(config.final_test_checkpoint_paths),
            "seed_values": list(config.final_test_seed_values) or [int(config.seed)],
            "run_directories": list(config.final_test_source_run_dirs),
            "test_structure_dir": str(config.test_structure_dir),
            "test_summary_csv": str(config.test_summary_csv),
            "train_val_split_by": config.train_val_split_by,
            "split_by": config.train_val_split_by,
            "split_by_scope": "train_validation_only",
            "test_split_source": "explicit_test_structure_dir_and_test_summary_csv",
            "evaluation_protocol_id": config.evaluation_protocol_id,
            "held_out_overlap_policy": config.held_out_overlap_policy,
            "overlap_policy_decision": overlap_report.get("overlap_policy_decision"),
            "overlap_exception_applied": overlap_report.get("overlap_exception_applied", False),
            "validated_protocol_manifest": overlap_report.get("validated_protocol_manifest"),
            "metrics": final_reporting_payload["metrics"],
            "calibrated_metrics": final_reporting_payload["calibrated_metrics"],
            "fitted_temperatures": final_reporting_payload["fitted_temperatures"],
            "temperature_scaling": final_reporting_payload["temperature_scaling"],
            "bootstrap_settings": final_reporting_payload["bootstrap_settings"],
            "calibration_settings": final_reporting_payload["calibration_settings"],
            "calibration_plot_paths": final_reporting_payload["calibration_plot_paths"],
            "reliability_diagram_path": final_reporting_payload["reliability_diagram_path"],
            "confidence_histogram_path": final_reporting_payload["confidence_histogram_path"],
            "prediction_artifact_path": final_reporting_payload["prediction_artifact_path"],
            "n_test_pockets": len(test_load_result.pockets),
            "feature_load_report": test_load_result.feature_report,
            "ec_labels": prepared.ec_labels,
            "split_name": split_identity.get("split_name"),
            "split_type": split_identity.get("split_type"),
            "train_test_overlap_detected": overlap_report["train_test_overlap_detected"],
            "overlap_counts": overlap_report["overlap_counts"],
            "overlap_examples": overlap_report["overlap_examples"],
            "overlap_warning": overlap_warning,
            "pre_inference_structure_overlap_report": structure_overlap_report,
            "timestamp": utc_timestamp(),
            "git_commit": git_commit_hash(),
            "code_version": git_commit_hash(),
            "selection_policy_statement": (
                "No held-out test metric was used for model, hyperparameter, ensemble, "
                "temperature, threshold, seed, or checkpoint selection."
            ),
            "reporting_only_comparison_statement": (
                "Primary and secondary final-test reports are labels fixed before evaluation; "
                "test metrics must not be used to switch the primary result."
            ),
        }
    finally:
        prepared.model.load_state_dict(current_state_dict)


def _checkpoint_rule_fields(checkpoint: dict[str, Any] | None) -> dict[str, Any]:
    """Terminal-rule provenance (empty for the default rule, so default outputs are unchanged)."""
    if checkpoint is None or checkpoint.get("checkpoint_rule") != "terminal":
        return {}
    return {
        "checkpoint_rule": "terminal",
        "descriptive_best_metric": checkpoint.get("descriptive_best_metric"),
        "descriptive_best_epoch": checkpoint.get("descriptive_best_epoch"),
        "descriptive_best_metric_value": checkpoint.get("descriptive_best_metric_value"),
    }


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
        best_checkpoint_path = prepared.run_dir / selected_checkpoint_filename(best_checkpoint)
        torch.save(best_checkpoint, best_checkpoint_path)
        selected_checkpoint_path = str(best_checkpoint_path)
        selected_checkpoint_epoch = best_checkpoint.get("epoch")
        selected_metric_value = best_checkpoint.get("selection_metric_value")
    else:
        selected_checkpoint_path = str(checkpoint_path)
        selected_checkpoint_epoch = None
        selected_metric_value = None

    if test_report is not None:
        test_report = dict(test_report)
        if not test_report.get("checkpoint_paths"):
            test_report["checkpoint_paths"] = [selected_checkpoint_path]
        if not test_report.get("run_directories"):
            test_report["run_directories"] = [str(prepared.run_dir)]
        if not test_report.get("selected_config_id"):
            test_report["selected_config_id"] = prepared.config_payload.get("run_name") or prepared.run_dir.name

    run_metadata = {
        "config": prepared.config_payload,
        "runtime_environment": prepared.runtime_environment,
        "source_artifacts": prepared.source_artifacts,
        "source_control": prepared.source_control,
        "dataset_summary": prepared.dataset_summary,
        "esm_embedding_metadata": prepared.dataset_summary.get("runtime_preparation", {}).get(
            "esm_embedding_metadata"
        ),
        "metal_labels": METAL_TARGET_LABELS,
        "metal_label_scheme": prepared.config_payload.get("metal_label_scheme", ACTIVE_METAL_LABEL_SCHEME),
        "ec_labels": prepared.ec_labels,
        "normalization_stats": normalization_stats_payload(prepared.normalization_stats),
        "selection_metric": prepared.config_payload.get("selection_metric"),
        "selected_checkpoint": selected_checkpoint_path,
        "selected_checkpoint_epoch": selected_checkpoint_epoch,
        "selected_metric_value": selected_metric_value,
        **_checkpoint_rule_fields(best_checkpoint),
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
    run_metadata["fit_status"] = "completed" if len(history) == prepared.config_payload["epochs"] else "incomplete"
    metal_weights = getattr(prepared.model, "metal_class_weights", None)
    run_metadata["metal_class_weights"] = (
        {METAL_TARGET_LABELS[index]: float(value) for index, value in enumerate(metal_weights.tolist())}
        if isinstance(metal_weights, torch.Tensor) and metal_weights.numel() else None
    )
    if prepared.config_payload.get("campaign_run_identity"):
        run_metadata["campaign_run_identity"] = json.loads(prepared.config_payload["campaign_run_identity"])
    if prepared.config_payload.get("explicit_membership"):
        run_metadata["explicit_membership"] = prepared.config_payload["explicit_membership"]
        run_metadata["fit_status"] = "completed" if len(history) == prepared.config_payload["epochs"] else "incomplete"
        run_metadata["outer_observed_during_training"] = False

    run_metadata["optimizer_groups"] = finalize_optimizer_groups(
        prepared.run_dir, prepared.optimizer, len(history),
        amp_active=bool(prepared.config_payload.get("use_amp")) and str(prepared.config_payload.get("device", "")).startswith("cuda"),
    )
    save_json(
        prepared.run_dir / "run_config.json",
        {
            "config": prepared.config_payload,
            "runtime_environment": prepared.runtime_environment,
            "source_artifacts": prepared.source_artifacts,
            "source_control": prepared.source_control,
            "dataset_summary": prepared.dataset_summary,
            "esm_embedding_metadata": prepared.dataset_summary.get("runtime_preparation", {}).get(
                "esm_embedding_metadata"
            ),
            "metal_labels": METAL_TARGET_LABELS,
            "metal_label_scheme": prepared.config_payload.get("metal_label_scheme", ACTIVE_METAL_LABEL_SCHEME),
            "ec_labels": prepared.ec_labels,
            "normalization_stats": normalization_stats_payload(prepared.normalization_stats),
            "selection_metric": prepared.config_payload.get("selection_metric"),
            "selected_checkpoint": selected_checkpoint_path,
            "selected_checkpoint_epoch": selected_checkpoint_epoch,
            "selected_metric_value": selected_metric_value,
            **_checkpoint_rule_fields(best_checkpoint),
            "history": history,
            "test_report": test_report,
        },
    )
    save_json(prepared.run_dir / "run_metadata.json", run_metadata)
    if test_report is not None:
        save_json(prepared.run_dir / "test_report.json", test_report)

    print(f"Saved checkpoint to {checkpoint_path}")
    if best_checkpoint is not None:
        print(f"Saved selected checkpoint to {prepared.run_dir / selected_checkpoint_filename(best_checkpoint)}")
    print(f"Saved dataset summary to {prepared.run_dir / 'dataset_summary.json'}")
    print(f"Saved run config to {prepared.run_dir / 'run_config.json'}")
    print(f"Saved run metadata to {prepared.run_dir / 'run_metadata.json'}")
    if test_report is not None:
        print(f"Saved test report to {prepared.run_dir / 'test_report.json'}")


def run_training(config: TrainConfig) -> Path:
    configure_active_metal_label_scheme(config.metal_label_scheme)
    config = resolve_selection_metric(config)
    validate_training_configuration(config)
    if config.export_ec_predictions or config.ec_probe_structure_ids is not None:
        from training.ec_prediction_export import validate_ec_export_configuration

        validate_ec_export_configuration(config)
    validate_held_out_structure_disjointness(config)
    set_seed(config.seed, deterministic=config.deterministic)
    if config.omit_node_features:
        print("Omitting conservative node features:", ", ".join(config.omit_node_features))
    phase_started = perf_counter()
    prepared = prepare_run(config)
    _record_runtime_phase(prepared.run_dir, config, "prepare", phase_started, reset=True)
    history, best_checkpoint = train_and_select_checkpoint(prepared, config)
    test_report = (None if config.explicit_membership_manifest else
                   evaluate_held_out_test_split(prepared, config, checkpoint=best_checkpoint))
    phase_started = perf_counter()
    persist_run_outputs(prepared, history=history, best_checkpoint=best_checkpoint, test_report=test_report)
    _record_runtime_phase(prepared.run_dir, config, "checkpoint_save", phase_started)
    if config.export_validation_predictions:
        if best_checkpoint is None or prepared.val_loader is None:
            raise ValueError("--export-validation-predictions requires a validation split and selected checkpoint")
        phase_started = perf_counter()
        export_selected_validation_predictions(
            model=prepared.model,
            val_loader=prepared.val_loader,
            val_pockets=prepared.split.val_pockets,
            best_checkpoint=best_checkpoint,
            checkpoint_path=prepared.run_dir / selected_checkpoint_filename(best_checkpoint),
            history=history,
            config_payload=prepared.config_payload,
            run_dir=prepared.run_dir,
            device=config.device,
        )
        _record_runtime_phase(prepared.run_dir, config, "selected_export", phase_started)
    if config.export_ec_predictions:
        from training.ec_prediction_export import export_ec_run_predictions

        phase_started = perf_counter()
        export_ec_run_predictions(prepared, config, history=history, checkpoint=best_checkpoint)
        _record_runtime_phase(prepared.run_dir, config, "selected_export", phase_started)
    return prepared.run_dir


def evaluate_saved_checkpoint(config: TrainConfig, checkpoint_path: Path) -> Path:
    configure_active_metal_label_scheme(config.metal_label_scheme)
    config = resolve_selection_metric(config)
    validate_training_configuration(config)
    if not config.run_test_eval:
        raise ValueError("evaluate_saved_checkpoint requires config.run_test_eval=True.")
    if config.test_structure_dir is None or config.test_summary_csv is None:
        raise ValueError("evaluate_saved_checkpoint requires held-out test structure and summary paths.")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    validate_held_out_structure_disjointness(config)
    set_seed(config.seed, deterministic=config.deterministic)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain a model_state_dict: {checkpoint_path}")

    prepared = prepare_run(config)
    checkpoint_stats = checkpoint.get("normalization_stats")
    if isinstance(checkpoint_stats, dict) and checkpoint_stats.get("means") and checkpoint_stats.get("stds"):
        prepared = replace(prepared, normalization_stats=normalization_stats_from_payload(checkpoint_stats))

    prepared.model.load_state_dict(checkpoint["model_state_dict"])
    history = checkpoint.get("history", [])
    if not isinstance(history, list):
        history = []
    test_report = evaluate_held_out_test_split(prepared, config, checkpoint=checkpoint)
    persist_run_outputs(prepared, history=history, best_checkpoint=checkpoint, test_report=test_report)
    return prepared.run_dir


def evaluate_softmax_mean_checkpoint_ensemble(
    config: TrainConfig,
    *,
    prediction_artifact_paths: list[Path],
    single_checkpoint_report_paths: list[Path] | None = None,
) -> Path:
    """Write a Stage 7 softmax-mean ensemble report from five fixed prediction artifacts."""
    configure_active_metal_label_scheme(config.metal_label_scheme)
    config = resolve_selection_metric(config)
    if not config.run_test_eval:
        raise ValueError("evaluate_softmax_mean_checkpoint_ensemble requires config.run_test_eval=True.")
    if config.final_test_ensemble_mode != "softmax_mean_5_seeds":
        raise ValueError(
            "evaluate_softmax_mean_checkpoint_ensemble requires "
            "final_test_ensemble_mode='softmax_mean_5_seeds'."
        )
    if len(prediction_artifact_paths) != 5:
        raise ValueError(
            "softmax_mean_5_seeds requires exactly five prediction artifacts; "
            f"got {len(prediction_artifact_paths)}."
        )
    if len(config.final_test_source_run_dirs) != 5 or len(config.final_test_checkpoint_paths) != 5:
        raise ValueError("softmax_mean_5_seeds requires exactly five source run directories and checkpoints.")
    validate_training_configuration(config)

    config_payload = config_to_payload(config)
    config_payload.update(infer_split_identity(config))
    runtime_environment = runtime_environment_payload()
    source_artifacts = source_artifacts_payload(config)
    source_control = source_control_payload(Path(__file__).resolve().parents[2])
    config_payload["git_commit"] = source_control.get("commit")
    config_payload["git_dirty"] = source_control.get("dirty")
    run_dir = build_run_dir(config)
    ensemble_payload = build_softmax_mean_ensemble_payload(
        output_dir=run_dir,
        prediction_artifact_paths=prediction_artifact_paths,
        task=config.task,
        enable_calibration=config.final_test_enable_calibration,
        enable_bootstrap_ci=config.final_test_enable_bootstrap_ci,
        n_bins=config.final_test_calibration_bins,
        n_bootstrap=config.final_test_bootstrap_resamples,
        confidence_level=config.final_test_bootstrap_confidence_level,
        bootstrap_seed=config.final_test_bootstrap_seed,
    )
    split_identity = infer_split_identity(config)
    test_report = {
        "task": config.task,
        "final_test_primary_report": config.final_test_primary_report,
        "final_test_ensemble_mode": config.final_test_ensemble_mode,
        "final_test_result_role": config.final_test_result_role,
        "primary_final_report": config.final_test_result_role in {"primary_final_report", "primary_preselected"},
        "secondary_diagnostic_report": config.final_test_result_role in {"secondary_diagnostic_report", "exploratory_posthoc"},
        "primary_preselected": config.final_test_result_role == "primary_preselected",
        "exploratory_posthoc": config.final_test_result_role == "exploratory_posthoc",
        "selected_config_id": config.final_test_selected_config_id,
        "selected_run_id": config.final_test_selected_config_id,
        "checkpoint_paths": list(config.final_test_checkpoint_paths),
        "seed_values": list(config.final_test_seed_values),
        "run_directories": list(config.final_test_source_run_dirs),
        "test_structure_dir": str(config.test_structure_dir),
        "test_summary_csv": str(config.test_summary_csv),
        "metrics": ensemble_payload["metrics"],
        "calibrated_metrics": ensemble_payload["calibrated_metrics"],
        "fitted_temperatures": ensemble_payload["fitted_temperatures"],
        "temperature_scaling": ensemble_payload["temperature_scaling"],
        "bootstrap_settings": ensemble_payload["bootstrap_settings"],
        "calibration_settings": ensemble_payload["calibration_settings"],
        "calibration_plot_paths": ensemble_payload["calibration_plot_paths"],
        "reliability_diagram_path": ensemble_payload["reliability_diagram_path"],
        "confidence_histogram_path": ensemble_payload["confidence_histogram_path"],
        "prediction_artifact_path": ensemble_payload["prediction_artifact_path"],
        "prediction_artifact_paths": [str(path) for path in prediction_artifact_paths],
        "n_test_pockets": ensemble_payload.get("n_test_pockets"),
        "single_checkpoint_report_paths": [str(path) for path in (single_checkpoint_report_paths or [])],
        "secondary_diagnostic_reports": [
            {
                "report_type": "single_checkpoint",
                "report_role": "secondary_diagnostic_report",
                "test_report_path": str(path),
            }
            for path in (single_checkpoint_report_paths or [])[:1]
        ],
        "split_name": split_identity.get("split_name"),
        "split_type": split_identity.get("split_type"),
        "timestamp": utc_timestamp(),
        "git_commit": git_commit_hash(),
        "code_version": git_commit_hash(),
        "selection_policy_statement": (
            "No held-out test metric was used for model, hyperparameter, ensemble, "
            "temperature, threshold, seed, or checkpoint selection."
        ),
        "reporting_only_comparison_statement": (
            "The five checkpoint list, averaging rule, and primary report label were fixed "
            "before test evaluation; single-checkpoint comparisons are reporting-only."
        ),
    }
    run_config_payload = {
        "config": config_payload,
        "runtime_environment": runtime_environment,
        "source_artifacts": source_artifacts,
        "source_control": source_control,
        "selection_metric": config_payload.get("selection_metric"),
        "selected_checkpoint": None,
        "selected_checkpoint_epoch": None,
        "selected_metric_value": None,
        "test_report": test_report,
    }
    run_metadata = {
        "config": config_payload,
        "runtime_environment": runtime_environment,
        "source_artifacts": source_artifacts,
        "source_control": source_control,
        "selection_metric": config_payload.get("selection_metric"),
        "selected_checkpoint": None,
        "selected_checkpoint_epoch": None,
        "selected_metric_value": None,
        "split_name": split_identity.get("split_name"),
        "split_type": split_identity.get("split_type"),
        "test_report": test_report,
        "result_stage": "final-test ensemble evaluated",
    }
    save_json(run_dir / "run_config.json", run_config_payload)
    save_json(run_dir / "run_metadata.json", run_metadata)
    save_json(run_dir / "test_report.json", test_report)
    print(f"Saved softmax-mean ensemble test report to {run_dir / 'test_report.json'}")
    return run_dir
