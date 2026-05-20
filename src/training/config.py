from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from data_structures import (
    DEFAULT_EDGE_RADIUS,
    METAL_NODE_MODE_CHOICES,
    NODE_FEATURE_SET_CHOICES,
    STRUCTURAL_READOUT_SCOPE_CHOICES,
    validate_node_feature_omissions,
)
from label_schemes import (
    METAL_LABEL_SCHEME_ALIASES,
    METAL_LABEL_SCHEMES,
    active_metal_label_scheme_name,
    configure_active_metal_label_scheme,
    normalize_metal_label_scheme_name,
)
from model_variants import FUSION_MODE_CHOICES, MODEL_ARCHITECTURE_CHOICES
from model_variants.factory import normalize_fusion_mode, normalize_model_architecture
from training.defaults import DEFAULT_STRUCTURE_DIR, DEFAULT_TRAIN_SUMMARY_CSV
from training.esm_feature_loading import DEFAULT_ESMC_EMBED_DIM
from training.feature_paths import VALID_EXTERNAL_FEATURE_SOURCE_CHOICES

VALID_SPLIT_BY_CHOICES = ("pdbid", "pdbid_chain", "structure_id", "pocket_id")
VALID_UNSUPPORTED_METAL_POLICY_CHOICES = ("error", "skip")
VALID_INVALID_STRUCTURE_POLICY_CHOICES = ("error", "skip")
VALID_TASK_CHOICES = ("joint", "metal", "ec")
VALID_METAL_LABEL_SCHEME_CHOICES = tuple(METAL_LABEL_SCHEMES)
VALID_NODE_FEATURE_SET_CHOICES = NODE_FEATURE_SET_CHOICES
VALID_METAL_NODE_MODE_CHOICES = METAL_NODE_MODE_CHOICES
VALID_STRUCTURAL_READOUT_SCOPE_CHOICES = STRUCTURAL_READOUT_SCOPE_CHOICES
VALID_MODEL_ARCHITECTURE_CHOICES = MODEL_ARCHITECTURE_CHOICES
VALID_METAL_LOSS_FUNCTION_CHOICES = ("cross_entropy", "focal")
VALID_METAL_CLASS_WEIGHT_MODE_CHOICES = (
    "none",
    "manual",
    "inverse_frequency",
    "inverse_sqrt_frequency",
    "effective_number",
)
VALID_JOINT_LOSS_WEIGHTING_CHOICES = ("auto", "fixed", "uncertainty")
VALID_EC_GROUP_WEIGHTING_CHOICES = ("none", "structure_id", "pdbid_chain", "pdbid")
VALID_LR_SCHEDULE_CHOICES = ("fixed", "cosine", "step")
VALID_FUSION_MODE_CHOICES = FUSION_MODE_CHOICES
VALID_EARLY_ESM_SCOPE_CHOICES = ("all", "first_shell", "first_second_shell")
VALID_CROSS_ATTENTION_NEIGHBORHOOD_CHOICES = ("all", "first_shell", "first_second_shell")
VALID_FINAL_TEST_REPORT_CHOICES = ("single_checkpoint", "softmax_mean_5_seeds")
VALID_FINAL_TEST_ENSEMBLE_MODE_CHOICES = ("single_checkpoint", "softmax_mean_5_seeds")
VALID_FINAL_TEST_RESULT_ROLE_CHOICES = ("primary_final_report", "secondary_diagnostic_report")
VALID_SELECTION_METRIC_CHOICES = (
    "train_loss",
    "val_loss",
    "val_metal_acc",
    "val_ec_acc",
    "val_ec_group_acc",
    "val_ec_group_balanced_acc",
    "val_ec_group_macro_f1",
    "val_ec_group_level_1_acc",
    "val_ec_group_level_1_balanced_acc",
    "val_ec_group_level_1_macro_f1",
    "val_ec_group_level_2_acc",
    "val_ec_group_level_2_balanced_acc",
    "val_ec_group_level_2_macro_f1",
    "val_ec_group_level_3_acc",
    "val_ec_group_level_3_balanced_acc",
    "val_ec_group_level_3_macro_f1",
    "val_ec_group_level_4_acc",
    "val_ec_group_level_4_balanced_acc",
    "val_ec_group_level_4_macro_f1",
    "val_joint_balanced_acc",
    "val_joint_macro_f1",
    "val_metal_balanced_acc",
    "val_metal_min_recall",
    "val_metal_mn_recall",
    "val_metal_fe_recall",
    "val_metal_class_viii_recall",
    "val_metal_collapsed4_acc",
    "val_metal_collapsed4_balanced_acc",
    "val_metal_collapsed4_min_recall",
    "val_metal_collapsed4_macro_f1",
    "val_metal_collapsed4_mn_recall",
    "val_ec_balanced_acc",
    "val_metal_macro_f1",
    "val_ec_macro_f1",
)


@dataclass(frozen=True)
class TrainConfig:
    structure_dir: Path = DEFAULT_STRUCTURE_DIR
    summary_csv: Path = DEFAULT_TRAIN_SUMMARY_CSV
    esm_embeddings_dir: str | None = None
    ring_features_dir: str | None = None
    external_features_root_dir: str | None = None
    external_feature_source: str = "auto"
    runs_dir: str | None = None
    run_name: str | None = None
    test_structure_dir: Path | None = None
    test_summary_csv: Path | None = None
    run_test_eval: bool = False
    allow_train_loss_test_eval_debug: bool = False
    device: str = "cpu"
    deterministic: bool = False
    task: str = "joint"
    metal_label_scheme: str = "split_all_metals"
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 3e-4
    grad_clip_norm: float = 1.0
    use_amp: bool = False
    grad_accum_steps: int = 1
    num_workers: int = 0
    pin_memory: bool = False
    # `0.0` is useful for smoke runs; real training should usually use a nonzero validation split.
    val_fraction: float = 0.0
    esm_dim: int = DEFAULT_ESMC_EMBED_DIM
    model_architecture: str = "gvp"
    edge_radius: float = DEFAULT_EDGE_RADIUS
    weight_decay: float = 1e-4
    seed: int = 42
    split_seed: int | None = None
    hidden_s: int = 128
    hidden_v: int = 16
    edge_hidden: int = 64
    gvp_layers: int = 4
    esm_fusion_dim: int = 128
    head_mlp_layers: int = 2
    node_rbf_sigma: float = 0.75
    edge_rbf_sigma: float = 0.75
    node_rbf_use_raw_distances: bool = False
    classifier_pool_distance_cutoff: float = 0.0
    metal_node_mode: str = "none"
    structural_readout_scope: str = "residue_only"
    normalize_message_aggregation: bool = False
    position_noise_std: float = 0.0
    second_shell_dropout: float = 0.0
    node_feature_set: str = "conservative"
    omit_node_features: tuple[str, ...] = ()
    use_esm_branch: bool = True
    fusion_mode: str = "late_fusion"
    cross_attention_layers: int = 1
    cross_attention_heads: int = 4
    cross_attention_dropout: float = 0.1
    cross_attention_neighborhood: str = "all"
    cross_attention_bidirectional: bool = False
    use_early_esm: bool = False
    early_esm_dim: int = 32
    early_esm_dropout: float = 0.2
    early_esm_raw: bool = False
    early_esm_scope: str = "all"
    use_ring_edges: bool = False
    require_ring_edges: bool = False
    split_by: str = "pdbid"
    n_folds: int | None = None
    fold_index: int | None = None
    balance_metal_site_symbols: bool = False
    require_all_task_classes: bool = False
    mn_loss_multiplier: float = 1.0
    cu_loss_multiplier: float = 1.0
    zn_loss_multiplier: float = 1.0
    fe_loss_multiplier: float = 1.0
    co_loss_multiplier: float = 1.0
    ni_loss_multiplier: float = 1.0
    class_viii_loss_multiplier: float = 1.0
    joint_loss_weighting: str = "uncertainty"
    metal_loss_weight: float = 1.0
    ec_loss_weight: float = 1.0
    metal_class_weight_mode: str = "inverse_frequency"
    metal_loss_function: str = "cross_entropy"
    metal_focal_gamma: float = 2.0
    metal_label_smoothing: float = 0.0
    metal_collapsed_loss_weight: float = 0.0
    require_esm_embeddings: bool = True
    prepare_missing_esm_embeddings: bool = True
    require_external_features: bool = True
    prepare_missing_ring_edges: bool = True
    unsupported_metal_policy: str = "error"
    invalid_structure_policy: str = "skip"
    ec_label_depth: int = 1
    ec_group_weighting: str = "structure_id"
    ec_contrastive_weight: float = 0.0
    ec_contrastive_temperature: float = 0.1
    lr_schedule: str = "fixed"
    lr_step_size: int = 0
    lr_decay_gamma: float = 0.5
    save_epoch_checkpoints: bool = False
    log_per_class_metrics: bool = False
    selection_metric: str | None = None
    final_test_primary_report: str = "single_checkpoint"
    final_test_ensemble_mode: str = "single_checkpoint"
    final_test_result_role: str = "primary_final_report"
    final_test_selected_config_id: str | None = None
    final_test_source_run_dirs: tuple[str, ...] = ()
    final_test_checkpoint_paths: tuple[str, ...] = ()
    final_test_seed_values: tuple[int, ...] = ()
    final_test_enable_calibration: bool = True
    final_test_enable_temperature_scaling: bool = True
    final_test_calibration_bins: int = 15
    final_test_enable_bootstrap_ci: bool = True
    final_test_bootstrap_resamples: int = 1000
    final_test_bootstrap_confidence_level: float = 0.95
    final_test_bootstrap_seed: int = 20260518


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the pocket classifier on the catalytic-only MAHOMES summary table."
    )
    parser.add_argument("--structure-dir", type=Path, default=DEFAULT_STRUCTURE_DIR)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_TRAIN_SUMMARY_CSV)
    parser.add_argument("--esm-embeddings-dir", type=str, default=None)
    parser.add_argument("--ring-features-dir", type=str, default=None)
    parser.add_argument("--external-features-root-dir", type=str, default=None)
    parser.add_argument(
        "--external-feature-source",
        type=str,
        default="auto",
        choices=VALID_EXTERNAL_FEATURE_SOURCE_CHOICES,
    )
    parser.add_argument("--runs-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--test-structure-dir", type=Path, default=None)
    parser.add_argument("--test-summary-csv", type=Path, default=None)
    parser.add_argument("--run-test-eval", action="store_true")
    parser.add_argument(
        "--allow-train-loss-test-eval-debug",
        action="store_true",
        help=(
            "Debug/smoke override: allow held-out test evaluation without validation-selected "
            "checkpointing. Do not use for final or publication-quality reporting."
        ),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Enable extra deterministic PyTorch settings for reproducible debug/experiment "
            "runs. This can reduce training speed."
        ),
    )
    parser.add_argument("--task", type=str, default="joint", choices=VALID_TASK_CHOICES)
    parser.add_argument(
        "--metal-label-scheme",
        type=normalize_metal_label_scheme_name,
        default=None,
        choices=VALID_METAL_LABEL_SCHEME_CHOICES,
        help=(
            "Metal target scheme. split_all_metals/six_class keeps Mn, Cu, Zn, Fe, Co, Ni separate; "
            "five_class keeps Mn, Cu, Zn, Fe separate and groups Co/Ni; "
            "merge_fe_class_viii/four_class groups Fe/Co/Ni. "
            "If omitted, DEEPGM_METAL_LABEL_SCHEME is used, defaulting to split_all_metals. "
            f"Accepted aliases include: {', '.join(sorted(METAL_LABEL_SCHEME_ALIASES))}."
        ),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--esm-dim", type=int, default=DEFAULT_ESMC_EMBED_DIM)
    parser.add_argument(
        "--model-architecture",
        type=normalize_model_architecture,
        default="gvp",
        choices=VALID_MODEL_ARCHITECTURE_CHOICES,
    )
    parser.add_argument("--edge-radius", type=float, default=DEFAULT_EDGE_RADIUS)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Gradient clipping max norm. Values <= 0 disable clipping.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable CUDA automatic mixed precision for training. No-op on CPU; evaluation stays FP32.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Number of mini-batches to accumulate before each optimizer step. Must be >= 1.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes. Default 0 preserves single-process loading.",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Request pinned host memory for CUDA DataLoaders. Ignored on CPU.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help=(
            "Optional seed for train/validation splitting. Defaults to --seed. "
            "Use this to keep grouped fold definitions fixed while varying model initialization."
        ),
    )
    parser.add_argument("--hidden-s", type=int, default=128)
    parser.add_argument("--hidden-v", type=int, default=16)
    parser.add_argument("--edge-hidden", type=int, default=64)
    parser.add_argument("--gvp-layers", type=int, default=4)
    parser.add_argument("--esm-fusion-dim", type=int, default=128)
    parser.add_argument("--head-mlp-layers", type=int, default=2)
    parser.add_argument("--node-rbf-sigma", type=float, default=0.75)
    parser.add_argument("--edge-rbf-sigma", type=float, default=0.75)
    parser.add_argument("--node-rbf-use-raw-distances", action="store_true")
    parser.add_argument(
        "--classifier-pool-distance-cutoff",
        type=float,
        default=0.0,
        help=(
            "If > 0, pool only residues with CA-to-metal distance within this "
            "Angstrom cutoff before the final classifier head. 0 disables the filter."
        ),
    )
    parser.add_argument(
        "--metal-node-mode",
        type=str,
        default="none",
        choices=VALID_METAL_NODE_MODE_CHOICES,
        help=(
            "Opt-in GVP graph construction mode for adding generic metal-site nodes. "
            "'none' preserves residue-only graphs; 'per_metal' appends one generic "
            "metal node per metal coordinate and promotes metal-ligand edges into edge_index."
        ),
    )
    parser.add_argument(
        "--structural-readout-scope",
        type=str,
        default="auto",
        choices=VALID_STRUCTURAL_READOUT_SCOPE_CHOICES,
        help=(
            "Which structural nodes are pooled for the GVP classifier. 'auto' uses "
            "residue_only for residue-only graphs and residue_and_metal when "
            "--metal-node-mode per_metal is enabled."
        ),
    )
    parser.add_argument(
        "--gvp-normalize-message-aggregation",
        dest="normalize_message_aggregation",
        action="store_true",
        help=(
            "Normalize each GVP layer's summed incoming messages by destination node degree. "
            "Default off preserves existing validation comparisons."
        ),
    )
    parser.add_argument(
        "--position-noise-std",
        type=float,
        default=0.0,
        help=(
            "Training-only Gaussian coordinate noise standard deviation in Angstrom. "
            "Default 0.0 leaves train/validation/test graphs unchanged."
        ),
    )
    parser.add_argument(
        "--second-shell-dropout",
        type=float,
        default=0.0,
        help=(
            "Training-only probability of dropping second-shell residues from each "
            "in-memory graph. Default 0.0 disables the augmentation."
        ),
    )
    parser.add_argument(
        "--node-feature-set",
        type=str,
        default="conservative",
        choices=VALID_NODE_FEATURE_SET_CHOICES,
    )
    parser.add_argument(
        "--omit-node-features",
        type=str,
        default="",
        help=(
            "Comma-separated conservative node feature names to zero out for "
            "ablation runs. Leave blank to use the full node feature set."
        ),
    )
    parser.add_argument("--disable-esm-branch", action="store_true")
    parser.add_argument(
        "--fusion-mode",
        type=normalize_fusion_mode,
        default="late_fusion",
        choices=VALID_FUSION_MODE_CHOICES,
    )
    parser.add_argument("--cross-attention-layers", type=int, default=1)
    parser.add_argument("--cross-attention-heads", type=int, default=4)
    parser.add_argument("--cross-attention-dropout", type=float, default=0.1)
    parser.add_argument(
        "--cross-attention-neighborhood",
        type=str,
        default="all",
        choices=VALID_CROSS_ATTENTION_NEIGHBORHOOD_CHOICES,
    )
    parser.add_argument("--cross-attention-bidirectional", action="store_true")
    parser.add_argument("--use-early-esm", action="store_true")
    parser.add_argument("--early-esm-dim", type=int, default=32)
    parser.add_argument("--early-esm-dropout", type=float, default=0.2)
    parser.add_argument("--early-esm-raw", action="store_true")
    parser.add_argument("--early-esm-scope", type=str, default="all", choices=VALID_EARLY_ESM_SCOPE_CHOICES)
    parser.add_argument(
        "--use-ring-edges",
        action="store_true",
        help=(
            "Include precomputed or generated RING interaction edges when available. "
            "The raw CLI leaves this off unless requested; the notebook and playbook "
            "project defaults enable RING for graph runs."
        ),
    )
    parser.add_argument("--require-ring-edges", action="store_true")
    parser.add_argument("--allow-missing-esm-embeddings", action="store_true")
    parser.add_argument("--no-prepare-missing-esm-embeddings", action="store_true")
    parser.add_argument("--allow-missing-external-features", action="store_true")
    parser.add_argument("--prepare-missing-ring-edges", action="store_true")
    parser.add_argument(
        "--no-prepare-missing-ring-edges",
        action="store_true",
        help=(
            "Do not generate missing RING edge files during preflight. "
            "With --use-ring-edges or --require-ring-edges, missing files will be skipped or fail instead."
        ),
    )
    parser.add_argument("--lr-schedule", type=str, default="fixed", choices=VALID_LR_SCHEDULE_CHOICES)
    parser.add_argument("--lr-step-size", type=int, default=0)
    parser.add_argument("--lr-decay-gamma", type=float, default=0.5)
    parser.add_argument("--save-epoch-checkpoints", action="store_true")
    parser.add_argument(
        "--log-per-class-metrics",
        action="store_true",
        help=(
            "Expand each epoch log line with per-class recall/support summaries. "
            "Metrics are already saved to epoch_metrics.csv; this only changes stdout."
        ),
    )
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--n-folds", type=int, default=None)
    parser.add_argument("--fold-index", type=int, default=None)
    parser.add_argument(
        "--balance-metal-site-symbols",
        action="store_true",
        help=(
            "Use a weighted training sampler that balances metal classes and, within "
            "Class VIII, balances Co/Ni site symbols."
        ),
    )
    parser.add_argument(
        "--require-all-task-classes",
        action="store_true",
        help=(
            "Fail preflight if the training split is missing any label class required by "
            "the selected task."
        ),
    )
    parser.add_argument(
        "--mn-loss-multiplier",
        type=float,
        default=1.0,
        help=(
            "Additional multiplicative boost applied to the metal class weight for "
            "Mn."
        ),
    )
    parser.add_argument(
        "--zn-loss-multiplier",
        type=float,
        default=1.0,
        help=(
            "Additional multiplicative boost applied to the metal class weight for "
            "Zn. Values below 1.0 down-weight Zn."
        ),
    )
    parser.add_argument(
        "--cu-loss-multiplier",
        type=float,
        default=1.0,
        help="Additional multiplicative boost applied to the metal class weight for Cu.",
    )
    parser.add_argument(
        "--fe-loss-multiplier",
        type=float,
        default=1.0,
        help=(
            "Additional multiplicative boost applied to the metal class weight for "
            "Fe."
        ),
    )
    parser.add_argument(
        "--co-loss-multiplier",
        type=float,
        default=1.0,
        help="Additional multiplicative boost applied to the metal class weight for Co.",
    )
    parser.add_argument(
        "--ni-loss-multiplier",
        type=float,
        default=1.0,
        help="Additional multiplicative boost applied to the metal class weight for Ni.",
    )
    parser.add_argument(
        "--class-viii-loss-multiplier",
        type=float,
        default=1.0,
        help=(
            "Additional multiplicative boost applied to the metal class weight for "
            "Class VIII (the grouped Co/Ni class)."
        ),
    )
    parser.add_argument(
        "--joint-loss-weighting",
        type=str,
        default="auto",
        choices=VALID_JOINT_LOSS_WEIGHTING_CHOICES,
        help=(
            "How to combine metal and EC losses. 'auto' uses learned uncertainty "
            "weighting for joint runs and fixed weighting for single-task runs; "
            "'fixed' keeps --metal-loss-weight and --ec-loss-weight as manual "
            "task multipliers."
        ),
    )
    parser.add_argument(
        "--metal-loss-weight",
        type=float,
        default=1.0,
        help="Task-level multiplier for the metal classification loss.",
    )
    parser.add_argument(
        "--ec-loss-weight",
        type=float,
        default=1.0,
        help="Task-level multiplier for the EC classification loss.",
    )
    parser.add_argument(
        "--metal-class-weight-mode",
        type=str,
        default="inverse_frequency",
        choices=VALID_METAL_CLASS_WEIGHT_MODE_CHOICES,
        help=(
            "How to compute metal class weights from the training split. "
            "'manual' starts from weight 1.0 for every class so the per-class "
            "loss multipliers become the exact manual class weights. "
            "'inverse_frequency' preserves the previous default behavior."
        ),
    )
    parser.add_argument(
        "--metal-loss-function",
        type=str,
        default="cross_entropy",
        choices=VALID_METAL_LOSS_FUNCTION_CHOICES,
        help="Loss function for the metal head.",
    )
    parser.add_argument(
        "--metal-focal-gamma",
        type=float,
        default=2.0,
        help="Gamma value used when --metal-loss-function focal is active.",
    )
    parser.add_argument(
        "--metal-label-smoothing",
        type=float,
        default=0.0,
        help="Optional label smoothing applied to cross-entropy metal loss.",
    )
    parser.add_argument(
        "--metal-collapsed-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Experimental validation-only auxiliary collapsed-4 metal loss weight. "
            "0.0 disables it and preserves the standard six-class metal objective."
        ),
    )
    parser.add_argument(
        "--unsupported-metal-policy",
        type=str,
        default="error",
        choices=VALID_UNSUPPORTED_METAL_POLICY_CHOICES,
    )
    parser.add_argument(
        "--invalid-structure-policy",
        type=str,
        default="skip",
        choices=VALID_INVALID_STRUCTURE_POLICY_CHOICES,
    )
    parser.add_argument("--ec-label-depth", type=int, default=1)
    parser.add_argument(
        "--ec-group-weighting",
        type=str,
        default="structure_id",
        choices=VALID_EC_GROUP_WEIGHTING_CHOICES,
        help=(
            "EC-only per-sample weighting by the number of EC-supervised pockets "
            "in each group. Metal training is unchanged."
        ),
    )
    parser.add_argument("--ec-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--ec-contrastive-temperature", type=float, default=0.1)
    parser.add_argument(
        "--selection-metric",
        type=str,
        default=None,
        choices=VALID_SELECTION_METRIC_CHOICES,
    )
    parser.add_argument(
        "--final-test-primary-report",
        type=str,
        default="single_checkpoint",
        choices=VALID_FINAL_TEST_REPORT_CHOICES,
        help="Predeclared primary Stage 7 result before held-out test evaluation starts.",
    )
    parser.add_argument(
        "--final-test-ensemble-mode",
        type=str,
        default="single_checkpoint",
        choices=VALID_FINAL_TEST_ENSEMBLE_MODE_CHOICES,
        help="Stage 7 final-test reporting mode. Does not affect validation selection or HPO.",
    )
    parser.add_argument(
        "--final-test-result-role",
        type=str,
        default="primary_final_report",
        choices=VALID_FINAL_TEST_RESULT_ROLE_CHOICES,
        help="Whether this final-test report is the primary result or a secondary diagnostic.",
    )
    parser.add_argument("--final-test-selected-config-id", type=str, default=None)
    parser.add_argument(
        "--final-test-source-run-dirs",
        type=str,
        default="",
        help="Comma-separated validation source run directories recorded in test_report.json.",
    )
    parser.add_argument(
        "--final-test-checkpoint-paths",
        type=str,
        default="",
        help="Comma-separated checkpoint paths recorded in test_report.json.",
    )
    parser.add_argument(
        "--final-test-seed-values",
        type=str,
        default="",
        help="Comma-separated model seed values recorded in test_report.json.",
    )
    parser.set_defaults(
        final_test_enable_calibration=True,
        final_test_enable_temperature_scaling=True,
        final_test_enable_bootstrap_ci=True,
    )
    parser.add_argument(
        "--enable-final-test-calibration",
        "--final-test-calibration",
        dest="final_test_enable_calibration",
        action="store_true",
        help="Enable Stage 7 calibration metrics and plots.",
    )
    parser.add_argument(
        "--disable-final-test-calibration",
        "--no-final-test-calibration",
        dest="final_test_enable_calibration",
        action="store_false",
        help="Disable Stage 7 calibration metrics and plots.",
    )
    parser.add_argument(
        "--enable-final-test-temperature-scaling",
        "--final-test-temperature-scaling",
        dest="final_test_enable_temperature_scaling",
        action="store_true",
        help="Enable validation-fitted temperature scaling for Stage 7 reporting.",
    )
    parser.add_argument(
        "--disable-final-test-temperature-scaling",
        "--no-final-test-temperature-scaling",
        dest="final_test_enable_temperature_scaling",
        action="store_false",
        help="Disable validation-fitted temperature scaling for Stage 7 reporting.",
    )
    parser.add_argument("--final-test-calibration-bins", type=int, default=15)
    parser.add_argument(
        "--enable-final-test-bootstrap-ci",
        "--final-test-bootstrap-ci",
        dest="final_test_enable_bootstrap_ci",
        action="store_true",
        help="Enable Stage 7 bootstrap confidence intervals.",
    )
    parser.add_argument(
        "--disable-final-test-bootstrap-ci",
        "--no-final-test-bootstrap-ci",
        dest="final_test_enable_bootstrap_ci",
        action="store_false",
        help="Disable Stage 7 bootstrap confidence intervals.",
    )
    parser.add_argument("--final-test-bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--final-test-bootstrap-confidence-level", type=float, default=0.95)
    parser.add_argument("--final-test-bootstrap-seed", type=int, default=20260518)
    parser.add_argument(
        "--split-by",
        type=str,
        default="pdbid",
        choices=VALID_SPLIT_BY_CHOICES,
    )
    return parser


def parse_omit_node_features(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(token.strip() for token in value.split(",") if token.strip())


def parse_string_tuple(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    normalized = value.replace("\n", ",")
    return tuple(token.strip() for token in normalized.split(",") if token.strip())


def parse_int_tuple(value: str | None) -> tuple[int, ...]:
    parsed: list[int] = []
    for token in parse_string_tuple(value):
        parsed.append(int(token))
    return tuple(parsed)


def resolve_structural_readout_scope(metal_node_mode: str, requested_scope: str) -> str:
    if requested_scope != "auto":
        return requested_scope
    return "residue_and_metal" if metal_node_mode != "none" else "residue_only"


def parse_args(argv: Sequence[str] | None = None) -> TrainConfig:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    omit_node_features = validate_node_feature_omissions(
        args.node_feature_set,
        parse_omit_node_features(args.omit_node_features),
    )
    selection_metric = args.selection_metric
    if selection_metric is None:
        selection_metric = default_selection_metric_for_task(
            args.task,
            has_validation=args.val_fraction > 0.0 or args.n_folds is not None,
        )
    model_uses_esm_inputs = args.model_architecture != "only_gvp"
    joint_loss_weighting = resolve_joint_loss_weighting(args.task, args.joint_loss_weighting)
    metal_label_scheme = configure_active_metal_label_scheme(
        args.metal_label_scheme or active_metal_label_scheme_name()
    )
    structural_readout_scope = resolve_structural_readout_scope(
        args.metal_node_mode,
        args.structural_readout_scope,
    )
    return TrainConfig(
        structure_dir=args.structure_dir,
        summary_csv=args.summary_csv,
        esm_embeddings_dir=args.esm_embeddings_dir,
        ring_features_dir=args.ring_features_dir,
        external_features_root_dir=args.external_features_root_dir,
        external_feature_source=args.external_feature_source,
        runs_dir=args.runs_dir,
        run_name=args.run_name,
        test_structure_dir=args.test_structure_dir,
        test_summary_csv=args.test_summary_csv,
        run_test_eval=args.run_test_eval,
        allow_train_loss_test_eval_debug=args.allow_train_loss_test_eval_debug,
        device=args.device,
        deterministic=args.deterministic,
        task=args.task,
        metal_label_scheme=metal_label_scheme,
        epochs=args.epochs,
        batch_size=args.batch_size,
        esm_dim=args.esm_dim,
        model_architecture=args.model_architecture,
        edge_radius=args.edge_radius,
        learning_rate=args.learning_rate,
        grad_clip_norm=args.grad_clip_norm,
        use_amp=args.amp,
        grad_accum_steps=args.grad_accum_steps,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        weight_decay=args.weight_decay,
        seed=args.seed,
        split_seed=args.split_seed,
        hidden_s=args.hidden_s,
        hidden_v=args.hidden_v,
        edge_hidden=args.edge_hidden,
        gvp_layers=args.gvp_layers,
        esm_fusion_dim=args.esm_fusion_dim,
        head_mlp_layers=args.head_mlp_layers,
        node_rbf_sigma=args.node_rbf_sigma,
        edge_rbf_sigma=args.edge_rbf_sigma,
        node_rbf_use_raw_distances=args.node_rbf_use_raw_distances,
        classifier_pool_distance_cutoff=args.classifier_pool_distance_cutoff,
        metal_node_mode=args.metal_node_mode,
        structural_readout_scope=structural_readout_scope,
        normalize_message_aggregation=args.normalize_message_aggregation,
        position_noise_std=args.position_noise_std,
        second_shell_dropout=args.second_shell_dropout,
        node_feature_set=args.node_feature_set,
        omit_node_features=omit_node_features,
        use_esm_branch=model_uses_esm_inputs and not args.disable_esm_branch,
        fusion_mode=args.fusion_mode,
        cross_attention_layers=args.cross_attention_layers,
        cross_attention_heads=args.cross_attention_heads,
        cross_attention_dropout=args.cross_attention_dropout,
        cross_attention_neighborhood=args.cross_attention_neighborhood,
        cross_attention_bidirectional=args.cross_attention_bidirectional,
        use_early_esm=args.use_early_esm,
        early_esm_dim=args.early_esm_dim,
        early_esm_dropout=args.early_esm_dropout,
        early_esm_raw=args.early_esm_raw,
        early_esm_scope=args.early_esm_scope,
        use_ring_edges=args.use_ring_edges or args.require_ring_edges or args.prepare_missing_ring_edges,
        require_ring_edges=args.require_ring_edges,
        val_fraction=args.val_fraction,
        split_by=args.split_by,
        n_folds=args.n_folds,
        fold_index=args.fold_index,
        balance_metal_site_symbols=args.balance_metal_site_symbols,
        require_all_task_classes=args.require_all_task_classes,
        mn_loss_multiplier=args.mn_loss_multiplier,
        cu_loss_multiplier=args.cu_loss_multiplier,
        zn_loss_multiplier=args.zn_loss_multiplier,
        fe_loss_multiplier=args.fe_loss_multiplier,
        co_loss_multiplier=args.co_loss_multiplier,
        ni_loss_multiplier=args.ni_loss_multiplier,
        class_viii_loss_multiplier=args.class_viii_loss_multiplier,
        joint_loss_weighting=joint_loss_weighting,
        metal_loss_weight=args.metal_loss_weight,
        ec_loss_weight=args.ec_loss_weight,
        metal_class_weight_mode=args.metal_class_weight_mode,
        metal_loss_function=args.metal_loss_function,
        metal_focal_gamma=args.metal_focal_gamma,
        metal_label_smoothing=args.metal_label_smoothing,
        metal_collapsed_loss_weight=args.metal_collapsed_loss_weight,
        require_esm_embeddings=model_uses_esm_inputs and not args.allow_missing_esm_embeddings,
        prepare_missing_esm_embeddings=not args.no_prepare_missing_esm_embeddings,
        require_external_features=not args.allow_missing_external_features,
        prepare_missing_ring_edges=args.prepare_missing_ring_edges or not args.no_prepare_missing_ring_edges,
        unsupported_metal_policy=args.unsupported_metal_policy,
        invalid_structure_policy=args.invalid_structure_policy,
        ec_label_depth=args.ec_label_depth,
        ec_group_weighting=args.ec_group_weighting,
        ec_contrastive_weight=args.ec_contrastive_weight,
        ec_contrastive_temperature=args.ec_contrastive_temperature,
        lr_schedule=args.lr_schedule,
        lr_step_size=args.lr_step_size,
        lr_decay_gamma=args.lr_decay_gamma,
        save_epoch_checkpoints=args.save_epoch_checkpoints,
        log_per_class_metrics=args.log_per_class_metrics,
        selection_metric=selection_metric,
        final_test_primary_report=args.final_test_primary_report,
        final_test_ensemble_mode=args.final_test_ensemble_mode,
        final_test_result_role=args.final_test_result_role,
        final_test_selected_config_id=args.final_test_selected_config_id,
        final_test_source_run_dirs=parse_string_tuple(args.final_test_source_run_dirs),
        final_test_checkpoint_paths=parse_string_tuple(args.final_test_checkpoint_paths),
        final_test_seed_values=parse_int_tuple(args.final_test_seed_values),
        final_test_enable_calibration=args.final_test_enable_calibration,
        final_test_enable_temperature_scaling=args.final_test_enable_temperature_scaling,
        final_test_calibration_bins=args.final_test_calibration_bins,
        final_test_enable_bootstrap_ci=args.final_test_enable_bootstrap_ci,
        final_test_bootstrap_resamples=args.final_test_bootstrap_resamples,
        final_test_bootstrap_confidence_level=args.final_test_bootstrap_confidence_level,
        final_test_bootstrap_seed=args.final_test_bootstrap_seed,
    )


def required_targets_for_task(task: str) -> tuple[str, ...]:
    if task == "joint":
        return ("metal", "ec")
    if task == "metal":
        return ("metal",)
    if task == "ec":
        return ("ec",)
    raise ValueError(
        f"Unsupported training task {task!r}. "
        f"Expected one of: {', '.join(repr(choice) for choice in VALID_TASK_CHOICES)}."
    )


def resolve_joint_loss_weighting(task: str, mode: str) -> str:
    if mode not in VALID_JOINT_LOSS_WEIGHTING_CHOICES:
        raise ValueError(
            f"Unsupported --joint-loss-weighting value {mode!r}. "
            f"Expected one of: {', '.join(repr(choice) for choice in VALID_JOINT_LOSS_WEIGHTING_CHOICES)}."
        )
    if mode == "auto":
        return "uncertainty" if task == "joint" else "fixed"
    return mode


def default_selection_metric_for_task(task: str, *, has_validation: bool) -> str:
    if not has_validation:
        return "train_loss"
    if task == "joint":
        return "val_joint_balanced_acc"
    if task == "metal":
        return "val_metal_balanced_acc"
    if task == "ec":
        return "val_ec_group_balanced_acc"
    raise ValueError(
        f"Unsupported training task {task!r}. "
        f"Expected one of: {', '.join(repr(choice) for choice in VALID_TASK_CHOICES)}."
    )


def config_to_payload(config: TrainConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["structure_dir"] = str(config.structure_dir)
    payload["summary_csv"] = str(config.summary_csv)
    payload["test_structure_dir"] = str(config.test_structure_dir) if config.test_structure_dir is not None else None
    payload["test_summary_csv"] = str(config.test_summary_csv) if config.test_summary_csv is not None else None
    return payload
