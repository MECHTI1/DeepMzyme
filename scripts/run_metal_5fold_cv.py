#!/usr/bin/env python3
"""Generalized 5-Fold Cross-Validation & Benchmark Runner.

Supports training, validation, and held-out test evaluation on any DeepMzyme
dataset folder split at either the pocket level or the metal ion focused level:
- Clustered Pocket Level: --metal-example-unit pocket
- Disentangled Metal Ion Level: --metal-example-unit ion

Works with any dataset folder (e.g., exact_pinmymetal, zenodo_pmm_exact,
non_overlapped_pinmymetal, common_pdbid_70_30_pinmymetal, CLEAN_30_*, CARE_*,
or custom folders).

Maintains complete run isolation by tagging run names and output directories
with the dataset identity and example unit (pocket vs. ion).
"""

from __future__ import annotations

import os

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in [
    REPO_ROOT / "src",
    Path("/content/DeepMzyme/src"),
    Path(__file__).resolve().parent / "src",
    Path("/content/src"),
]:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from training.graph_dataset import PocketGraphDataset, build_graph_data_list
from label_schemes import configure_active_metal_label_scheme
from training.data import load_labeled_pockets_with_report_from_dir
from training.final_test_reporting import metal_metrics_from_probabilities
from training.run import evaluate_epoch_with_predictions

PINMYMETAL_FIG2A_CV = {
    "source": "PinMyMetal Fig 2a (5-fold CV published)",
    "collapsed4_balanced_acc": 0.7508,
    "recalls": {
        "Mn": 0.903,
        "Zn": 0.738,
        "Class VIII": 0.733,
        "Cu": 0.629,
    },
}

PINMYMETAL_FIG2B_TEST = {
    "source": "PinMyMetal Fig 2b (Held-Out Test Set published)",
    "collapsed4_balanced_acc": 0.6785,
    "recalls": {
        "Mn": 0.886,
        "Zn": 0.659,
        "Class VIII": 0.575,
        "Cu": 0.594,
    },
}

PINMYMETAL_FIG2C_METAL3D = {
    "source": "Metal3D on external dataset (Fig 2c)",
    "collapsed4_balanced_acc": 0.6170,
    "recalls": {
        "Mn": 0.638,
        "Zn": 0.644,
        "Class VIII": 0.616,
        "Cu": 0.570,
    },
}

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "benchmark_only_esm": {
        "short_name": "only_esm",
        "description": "Sequence-only ESM-C (50 epochs, lr=3e-5)",
        "architecture": "only_esm",
        "fusion_mode": None,
        "use_esm": True,
        "gvp_lr": None,
        "lr": "3e-5",
        "rbf_raw": False,
    },
    "benchmark_enhanced_only_gvp": {
        "short_name": "enhanced_only_gvp",
        "description": "Structure-only GVP (50 epochs, lr=3e-4, raw RBF)",
        "architecture": "only_gvp",
        "fusion_mode": None,
        "use_esm": False,
        "gvp_lr": None,
        "lr": "3e-4",
        "rbf_raw": True,
    },
    "benchmark_enhanced_gvp_esmc": {
        "short_name": "enhanced_gvp_esmc",
        "description": "Multimodal ESM-C + GVP Late Fusion (50 epochs, lr=3e-5, gvp-lr=3e-4)",
        "architecture": "gvp",
        "fusion_mode": "late_fusion",
        "use_esm": True,
        "gvp_lr": "3e-4",
        "lr": "3e-5",
        "rbf_raw": True,
    },
}

MODEL_ALIASES = {
    "enhanced_gvp_esmc": "benchmark_enhanced_gvp_esmc",
    "gvp_esmc": "benchmark_enhanced_gvp_esmc",
    "late_fusion": "benchmark_enhanced_gvp_esmc",
    "benchmark_enhanced_gvp_esmc": "benchmark_enhanced_gvp_esmc",
    "only_esm": "benchmark_only_esm",
    "esm": "benchmark_only_esm",
    "benchmark_only_esm": "benchmark_only_esm",
    "enhanced_only_gvp": "benchmark_enhanced_only_gvp",
    "only_gvp": "benchmark_enhanced_only_gvp",
    "gvp": "benchmark_enhanced_only_gvp",
    "benchmark_enhanced_only_gvp": "benchmark_enhanced_only_gvp",
}

CHECKPOINT_CANDIDATES = (
    "best_model_checkpoint.pt",
    "best_checkpoint.pt",
    "last_model_checkpoint.pt",
)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    tmp.replace(path)


def resolve_fold_checkpoint(run_dir: Path) -> Path | None:
    """Return the fold's selected-model checkpoint, or None when no weights exist."""
    for name in CHECKPOINT_CANDIDATES:
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    return None


def is_fold_complete(run_dir: Path, target_epochs: int = 50) -> bool:
    val_csv = run_dir / "val_metrics.csv"
    best_ckpt = resolve_fold_checkpoint(run_dir)
    if not (val_csv.exists() and best_ckpt is not None):
        return False
    try:
        with open(val_csv, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return len(rows) >= target_epochs
    except Exception:
        return False


def find_summary_csv(directory: Path, explicit_csv: Path | None = None) -> Path:
    """Find the best matching summary CSV in a dataset directory."""
    if explicit_csv is not None:
        csv_path = Path(explicit_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"Specified summary CSV does not exist: {csv_path}")
        return csv_path

    candidates = [
        directory / "final_data_summarazing_table_transition_metals_only_catalytic.csv",
        directory / "final_data_summarazing_table.csv",
        directory / "final_data_summarazing_table_transition_metals_only_catalytic_verified_biological_metal.csv",
    ]
    for c in candidates:
        if c.is_file():
            return c

    all_csvs = sorted(p for p in directory.glob("*.csv") if not p.name.endswith("manifest.csv"))
    if all_csvs:
        return all_csvs[0]

    raise FileNotFoundError(f"Could not find a valid summary CSV in {directory}")


def resolve_dataset_layout(
    data_root: Path,
    dataset_name_or_path: str,
    train_summary_csv: Path | None = None,
    test_summary_csv: Path | None = None,
) -> tuple[str, Path, Path, Path | None, Path | None]:
    """Resolve train/test structure directories and summary CSVs.

    Returns:
        (dataset_id, train_dir, train_csv, test_dir_or_none, test_csv_or_none)
    """
    candidates = [
        Path(dataset_name_or_path),
        data_root / dataset_name_or_path,
        REPO_ROOT / "DeepMzyme_Data" / dataset_name_or_path,
        Path("/media/mechti/Data1/DeepMzyme_Data") / dataset_name_or_path,
    ]
    dataset_root = None
    for cand in candidates:
        if cand.is_dir():
            dataset_root = cand.resolve()
            dataset_id = cand.name
            break

    if dataset_root is None:
        raise FileNotFoundError(
            f"Dataset '{dataset_name_or_path}' not found across candidates: {[str(c) for c in candidates]}"
        )

    # Check for train subdirectory
    train_sub = dataset_root / "train"
    if train_sub.is_dir():
        train_dir = train_sub
    else:
        train_dir = dataset_root

    train_csv = find_summary_csv(train_dir, train_summary_csv)

    # Check for test subdirectory
    test_sub = dataset_root / "test"
    test_dir: Path | None = None
    test_csv: Path | None = None
    if test_sub.is_dir():
        test_dir = test_sub
        try:
            test_csv = find_summary_csv(test_dir, test_summary_csv)
        except FileNotFoundError:
            test_csv = None

    return dataset_id, train_dir, train_csv, test_dir, test_csv


def build_fold_command(
    python_bin: str,
    model_key: str,
    fold_idx: int,
    n_folds: int,
    train_dir: Path,
    train_csv: Path,
    runs_dir: Path,
    run_name: str,
    feat_dir: Path,
    esm_dir: Path,
    metal_example_unit: str = "ion",
    metal_label_scheme: str = "five_class",
    train_val_split_by: str = "pocket_id",
    epochs: int = 50,
    batch_size: int = 16,
    device: str = "cuda",
    seed: int = 42,
    save_epoch_checkpoints: bool = False,
) -> list[str]:
    cfg = MODEL_CONFIGS[model_key]

    train_py = REPO_ROOT / "src" / "train.py"
    if not train_py.exists() and Path("/content/DeepMzyme/src/train.py").exists():
        train_py = Path("/content/DeepMzyme/src/train.py")

    cmd = [
        python_bin, "-u", str(train_py),
        "--task", "metal",
        "--metal-label-scheme", metal_label_scheme,
        "--metal-example-unit", metal_example_unit,
        "--structure-dir", str(train_dir),
        "--summary-csv", str(train_csv),
        "--external-feature-source", "updated",
        "--external-features-root-dir", str(feat_dir),
        "--allow-missing-external-features",
        "--allow-missing-esm-embeddings",
        "--runs-dir", str(runs_dir),
        "--run-name", run_name,
        "--model-architecture", cfg["architecture"],
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--learning-rate", str(cfg["lr"]),
        "--device", device,
        "--seed", str(seed),
        "--n-folds", str(n_folds),
        "--fold-index", str(fold_idx),
        "--train-val-split-by", train_val_split_by,
    ]

    if cfg["use_esm"]:
        cmd.extend(["--esm-embeddings-dir", str(esm_dir)])

    if cfg["fusion_mode"]:
        cmd.extend(["--fusion-mode", cfg["fusion_mode"]])

    if cfg["gvp_lr"]:
        cmd.extend(["--gvp-learning-rate", str(cfg["gvp_lr"])])

    if cfg["rbf_raw"]:
        cmd.append("--rbf-use-raw-distances")

    if save_epoch_checkpoints:
        cmd.append("--save-epoch-checkpoints")

    return cmd


def run_single_fold(
    python_bin: str,
    model_key: str,
    fold_idx: int,
    n_folds: int,
    train_dir: Path,
    train_csv: Path,
    runs_dir: Path,
    run_name: str,
    feat_dir: Path,
    esm_dir: Path,
    metal_example_unit: str = "ion",
    metal_label_scheme: str = "five_class",
    train_val_split_by: str = "pocket_id",
    epochs: int = 50,
    batch_size: int = 16,
    device: str = "cuda",
    seed: int = 42,
    skip_existing: bool = True,
    save_epoch_checkpoints: bool = False,
) -> tuple[int, float]:
    cmd = build_fold_command(
        python_bin=python_bin,
        model_key=model_key,
        fold_idx=fold_idx,
        n_folds=n_folds,
        train_dir=train_dir,
        train_csv=train_csv,
        runs_dir=runs_dir,
        run_name=run_name,
        feat_dir=feat_dir,
        esm_dir=esm_dir,
        metal_example_unit=metal_example_unit,
        metal_label_scheme=metal_label_scheme,
        train_val_split_by=train_val_split_by,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        seed=seed,
        save_epoch_checkpoints=save_epoch_checkpoints,
    )
    run_dir = runs_dir / run_name

    if skip_existing and is_fold_complete(run_dir, target_epochs=epochs):
        print(f"[RUNNER] Fold {fold_idx} ({run_name}) already completed at {run_dir}, skipping...", flush=True)
        return 0, 0.0
    elif run_dir.exists():
        print(f"[RUNNER] Cleaning up incomplete prior run at {run_dir}...", flush=True)
        shutil.rmtree(run_dir, ignore_errors=True)

    print("=" * 76, flush=True)
    print(f"[RUNNER] STARTING FOLD {fold_idx}/{n_folds-1}: {run_name} (unit={metal_example_unit})", flush=True)
    print("=" * 76, flush=True)

    started = time.time()
    child_env = os.environ.copy()
    child_env["MKL_THREADING_LAYER"] = "GNU"
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=child_env)
    assert proc.stdout is not None
    prefix = f"[{model_key[-8:]}:f{fold_idx}:{metal_example_unit[:3]}]"
    for line in proc.stdout:
        print(f"{prefix} {line}", end="", flush=True)
    return_code = proc.wait()
    elapsed = time.time() - started
    print(f"[RUNNER] Fold {fold_idx} finished in {elapsed:.1f}s with return code {return_code}", flush=True)
    return return_code, elapsed


def evaluate_test_set_for_fold(
    model_key: str,
    fold_idx: int,
    run_name: str,
    runs_dir: Path,
    test_dir: Path,
    test_csv: Path,
    feat_dir: Path,
    esm_dir: Path,
    metal_example_unit: str = "ion",
    metal_label_scheme: str = "five_class",
    device: str = "cuda",
    batch_size: int = 16,
) -> dict[str, Any] | None:
    """Evaluate a trained fold checkpoint on the held-out test set."""
    from model_variants import build_pocket_classifier
    from training.run import normalization_stats_from_payload

    run_dir = runs_dir / run_name
    ckpt_path = resolve_fold_checkpoint(run_dir)
    if ckpt_path is None:
        print(
            f"[TEST EVAL] No checkpoint found in {run_dir} (looked for: {', '.join(CHECKPOINT_CANDIDATES)})",
            flush=True,
        )
        return None

    pred_out = run_dir / "test_predictions.pt"
    report_out = run_dir / "test_report.json"
    if pred_out.exists() and report_out.exists():
        with open(report_out, encoding="utf-8") as h:
            return json.load(h)

    print(f"[TEST EVAL] Running held-out test inference for {run_name} ({metal_example_unit} mode)...", flush=True)
    configure_active_metal_label_scheme(metal_label_scheme)

    cfg = MODEL_CONFIGS[model_key]

    test_res = load_labeled_pockets_with_report_from_dir(
        structure_dir=test_dir,
        summary_csv=test_csv,
        required_targets=("metal",),
        esm_embeddings_dir=esm_dir if cfg["use_esm"] else None,
        require_esm_embeddings=False,
        external_features_root_dir=feat_dir,
        external_feature_source="updated",
        require_external_features=False,
        metal_example_unit=metal_example_unit,
    )

    test_pockets = [p for p in test_res.pockets if getattr(p, "y_metal", None) is not None]
    if not test_pockets:
        print("[TEST EVAL] Error: No valid test pockets found.", flush=True)
        return None

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    norm_stats = None
    if isinstance(checkpoint.get("normalization_stats"), dict):
        norm_stats = normalization_stats_from_payload(checkpoint["normalization_stats"])

    saved_cfg = checkpoint.get("config", {}) if isinstance(checkpoint.get("config"), dict) else {}
    esm_dim = saved_cfg.get("esm_dim", 1152)
    edge_radius = saved_cfg.get("edge_radius", 10.0)
    node_feat_set = saved_cfg.get("node_feature_set", "conservative")

    test_graphs = build_graph_data_list(
        test_pockets,
        esm_dim=esm_dim,
        edge_radius=edge_radius,
        node_feature_set=node_feat_set,
    )

    test_dataset = PocketGraphDataset(
        test_pockets,
        esm_dim=esm_dim,
        edge_radius=edge_radius,
        normalization_stats=norm_stats,
        precomputed_data=test_graphs,
        node_feature_set=node_feat_set,
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    arch = saved_cfg.get("model_architecture", cfg["architecture"])
    fusion = saved_cfg.get("fusion_mode", cfg["fusion_mode"] or "late_fusion")
    n_metal = len(checkpoint.get("metal_labels", [0, 1, 2, 3, 4]))
    task = saved_cfg.get("task", "metal")
    sd = checkpoint["model_state_dict"]

    model = build_pocket_classifier(
        model_architecture=arch,
        esm_dim=esm_dim,
        hidden_s=saved_cfg.get("hidden_s", 128),
        hidden_v=saved_cfg.get("hidden_v", 16),
        edge_hidden=saved_cfg.get("edge_hidden", 32),
        n_layers=saved_cfg.get("gvp_layers", 4),
        n_metal=n_metal,
        n_ec=max(1, len(checkpoint.get("ec_labels", {0: "ec1"}))),
        predict_metal=task in ("metal", "joint"),
        predict_ec=task in ("ec", "joint"),
        metal_class_weights=sd.get("metal_class_weights"),
        metal_collapsed4_class_weights=sd.get("metal_collapsed4_class_weights"),
        ec_class_weights=sd.get("ec_class_weights"),
        esm_fusion_dim=saved_cfg.get("esm_fusion_dim", 128),
        head_mlp_layers=saved_cfg.get("head_mlp_layers", 2),
        head_mlp_dropout=saved_cfg.get("head_mlp_dropout", 0.2),
        esm_graph_encoder_dropout=saved_cfg.get("esm_graph_encoder_dropout", 0.1),
        node_rbf_sigma=saved_cfg.get("node_rbf_sigma", 0.75),
        edge_rbf_sigma=saved_cfg.get("edge_rbf_sigma", 0.75),
        node_rbf_use_raw_distances=saved_cfg.get("node_rbf_use_raw_distances", cfg["rbf_raw"]),
        edge_rbf_use_raw_distances=saved_cfg.get("edge_rbf_use_raw_distances", cfg["rbf_raw"]),
        classifier_pool_distance_cutoff=saved_cfg.get("classifier_pool_distance_cutoff", 0.0),
        structural_readout_scope=saved_cfg.get("structural_readout_scope", "residue_only"),
        use_node_type_embedding=saved_cfg.get("metal_node_mode", "none") != "none",
        site_geometry_features=saved_cfg.get("site_geometry_features", "legacy"),
        use_esm_branch=saved_cfg.get("use_esm_branch", cfg["use_esm"]),
        fusion_mode=fusion,
        cross_attention_layers=saved_cfg.get("cross_attention_layers", 1),
        cross_attention_heads=saved_cfg.get("cross_attention_heads", 4),
        cross_attention_dropout=saved_cfg.get("cross_attention_dropout", 0.1),
        cross_attention_neighborhood=saved_cfg.get("cross_attention_neighborhood", "all"),
        cross_attention_bidirectional=saved_cfg.get("cross_attention_bidirectional", False),
        use_early_esm=saved_cfg.get("use_early_esm", False),
        early_esm_dim=saved_cfg.get("early_esm_dim", 32),
        early_esm_dropout=saved_cfg.get("early_esm_dropout", 0.2),
        early_esm_raw=saved_cfg.get("early_esm_raw", False),
        early_esm_scope=saved_cfg.get("early_esm_scope", "all"),
        normalize_message_aggregation=saved_cfg.get("normalize_message_aggregation", False),
        site_feature_dim=saved_cfg.get("site_feature_dim", 32),
        use_site_angle_features=saved_cfg.get("use_site_angle_features", False),
    )
    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    preds = evaluate_epoch_with_predictions(model, test_loader, device=device)
    if "metal_probabilities" in preds:
        probs = preds["metal_probabilities"].float().cpu()
    elif "metal_logits" in preds:
        probs = preds["metal_logits"].softmax(dim=-1).float().cpu()
    else:
        raise KeyError(f"Expected metal predictions in evaluate_epoch_with_predictions, found: {list(preds.keys())}")
    targets = preds["metal_y"].long().cpu()

    metrics = metal_metrics_from_probabilities(probs, targets, prefix="test")

    torch.save(
        {
            "metal_probabilities": probs,
            "metal_y": targets,
            "pocket_ids": [p.pocket_id for p in test_pockets],
            "structure_ids": [p.structure_id for p in test_pockets],
            "metal_example_unit": metal_example_unit,
        },
        pred_out,
    )

    report_payload = {
        "run_name": run_name,
        "fold_index": fold_idx,
        "metal_example_unit": metal_example_unit,
        "n_test_sites": len(test_pockets),
        "metrics": metrics,
    }
    write_json(report_out, report_payload)
    print(f"[TEST EVAL] Saved test report to {report_out}", flush=True)
    return report_payload


def _safe_float(val: Any, default: float | None = None) -> float | None:
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def summarize_single_fold(run_dir: Path, fold_idx: int, run_name: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "fold_index": fold_idx,
        "run_name": run_name,
        "run_dir": str(run_dir),
    }
    val_csv = run_dir / "val_metrics.csv"
    if val_csv.exists():
        with open(val_csv, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        out["completed_epochs"] = len(rows)
        if rows:
            best_5c = max(rows, key=lambda r: _safe_float(r.get("val_metal_balanced_acc"), 0.0) or 0.0)
            out["best_epoch_5class"] = int(_safe_float(best_5c.get("epoch"), 0.0) or 0.0)
            out["best_val_balanced_acc_5class"] = _safe_float(best_5c.get("val_metal_balanced_acc"))
            out["best_val_raw_acc_5class"] = _safe_float(best_5c.get("val_metal_acc"))

            best_c4 = max(rows, key=lambda r: _safe_float(r.get("val_metal_collapsed4_balanced_acc"), 0.0) or 0.0)
            out["best_epoch_collapsed4"] = int(_safe_float(best_c4.get("epoch"), 0.0) or 0.0)
            out["best_val_balanced_acc_collapsed4"] = _safe_float(best_c4.get("val_metal_collapsed4_balanced_acc"))
            out["best_val_collapsed4_mn_recall"] = _safe_float(best_c4.get("val_metal_collapsed4_mn_recall"))
            out["best_val_collapsed4_cu_recall"] = _safe_float(best_c4.get("val_metal_collapsed4_cu_recall"))
            out["best_val_collapsed4_zn_recall"] = _safe_float(best_c4.get("val_metal_collapsed4_zn_recall"))
            out["best_val_collapsed4_class_viii_recall"] = _safe_float(best_c4.get("val_metal_collapsed4_class_viii_recall"))

            final_row = rows[-1]
            out["final_epoch"] = int(_safe_float(final_row.get("epoch"), 0.0) or 0.0)
            out["final_val_balanced_acc_5class"] = _safe_float(final_row.get("val_metal_balanced_acc"))
            out["final_val_balanced_acc_collapsed4"] = _safe_float(final_row.get("val_metal_collapsed4_balanced_acc"))

    test_json = run_dir / "test_report.json"
    if test_json.exists():
        with open(test_json, encoding="utf-8") as handle:
            report = json.load(handle)
        metrics = report.get("metrics", {})
        out["test_metal_acc"] = metrics.get("test_metal_acc")
        out["test_metal_balanced_acc"] = metrics.get("test_metal_balanced_acc")
        out["test_metal_collapsed4_acc"] = metrics.get("test_metal_collapsed4_acc")
        out["test_metal_collapsed4_balanced_acc"] = metrics.get("test_metal_collapsed4_balanced_acc")
        out["test_metal_macro_f1"] = metrics.get("test_metal_macro_f1")
        out["test_metal_collapsed4_mn_recall"] = metrics.get("test_metal_collapsed4_mn_recall")
        out["test_metal_collapsed4_cu_recall"] = metrics.get("test_metal_collapsed4_cu_recall")
        out["test_metal_collapsed4_zn_recall"] = metrics.get("test_metal_collapsed4_zn_recall")
        out["test_metal_collapsed4_class_viii_recall"] = metrics.get("test_metal_collapsed4_class_viii_recall")

    return out


def evaluate_model_ensemble(
    model_key: str,
    runs_dir: Path,
    run_name_fn,
    n_folds: int = 5,
    metal_label_scheme: str = "five_class",
) -> dict[str, Any] | None:
    configure_active_metal_label_scheme(metal_label_scheme)

    fold_probs = []
    targets = None

    for fold_idx in range(n_folds):
        run_name = run_name_fn(fold_idx)
        pred_path = runs_dir / run_name / "test_predictions.pt"
        if not pred_path.exists():
            print(f"[ENSEMBLE] Notice: {pred_path} not found. Skipping ensemble for {model_key}.", flush=True)
            return None
        data = torch.load(pred_path, map_location="cpu", weights_only=False)
        fold_probs.append(data["metal_probabilities"].float())
        if targets is None:
            targets = data["metal_y"].long()
        else:
            if not torch.equal(targets, data["metal_y"].long()):
                raise ValueError(f"Target mismatch in fold {fold_idx}")

    ensemble_probs = torch.stack(fold_probs, dim=0).mean(dim=0)
    ensemble_metrics = metal_metrics_from_probabilities(ensemble_probs, targets, prefix="test_ensemble")

    result = {
        "model": model_key,
        "n_folds": n_folds,
        "n_test_sites": int(targets.size(0)),
        "ensemble_metrics": ensemble_metrics,
    }

    ens_path = runs_dir / f"{model_key}_5fold_ensemble_report.json"
    write_json(ens_path, result)
    print(f"[ENSEMBLE] Saved 5-fold ensemble report to {ens_path}", flush=True)
    return result


def compute_oof_cv_metrics(fold_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    keys_to_agg = [
        "best_val_balanced_acc_5class",
        "best_val_balanced_acc_collapsed4",
        "best_val_raw_acc_5class",
        "best_val_collapsed4_mn_recall",
        "best_val_collapsed4_cu_recall",
        "best_val_collapsed4_zn_recall",
        "best_val_collapsed4_class_viii_recall",
        "test_metal_acc",
        "test_metal_balanced_acc",
        "test_metal_collapsed4_acc",
        "test_metal_collapsed4_balanced_acc",
        "test_metal_collapsed4_mn_recall",
        "test_metal_collapsed4_cu_recall",
        "test_metal_collapsed4_zn_recall",
        "test_metal_collapsed4_class_viii_recall",
    ]
    cv_stats: dict[str, Any] = {}
    for key in keys_to_agg:
        vals = [f[key] for f in fold_summaries if key in f and f[key] is not None]
        if vals:
            cv_stats[f"mean_{key}"] = float(np.mean(vals))
            cv_stats[f"std_{key}"] = float(np.std(vals))
    return cv_stats


def format_comparison_table(
    results: dict[str, dict[str, Any]],
    dataset_name: str,
    metal_example_unit: str,
) -> str:
    def _pct(val: Any) -> float:
        if val is None:
            return 0.0
        try:
            return float(val) * 100.0
        except (ValueError, TypeError):
            return 0.0

    lines = [
        f"# 5-FOLD CV BENCHMARK COMPARISON ({dataset_name} | {metal_example_unit.upper()} LEVEL)",
        "",
        f"- **Dataset Split:** `{dataset_name}`",
        f"- **Example Granularity:** `{metal_example_unit}` (1 example per {metal_example_unit})",
        "",
        "## Table 1: 5-Fold Cross-Validation Performance (Validation Folds)",
        "",
        "| Architecture | 5-Fold CV Val Bal Acc (5-class) | 5-Fold CV Val Bal Acc (Collapsed-4) | Mn Recall | Zn Recall | Group VIII Recall | Cu Recall | Delta vs PMM Fig 2a (75.08%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **PinMyMetal Fig 2a Baseline** | - | **{PINMYMETAL_FIG2A_CV['collapsed4_balanced_acc']*100:.2f}%** | {PINMYMETAL_FIG2A_CV['recalls']['Mn']*100:.1f}% | {PINMYMETAL_FIG2A_CV['recalls']['Zn']*100:.1f}% | {PINMYMETAL_FIG2A_CV['recalls']['Class VIII']*100:.1f}% | {PINMYMETAL_FIG2A_CV['recalls']['Cu']*100:.1f}% | Baseline |",
    ]

    for model_key, res in results.items():
        short_name = MODEL_CONFIGS[model_key]["short_name"]
        cv_stats = res.get("cv_stats", {})

        v5_m = _pct(cv_stats.get("mean_best_val_balanced_acc_5class"))
        v5_s = _pct(cv_stats.get("std_best_val_balanced_acc_5class"))
        vc4_m = _pct(cv_stats.get("mean_best_val_balanced_acc_collapsed4"))
        vc4_s = _pct(cv_stats.get("std_best_val_balanced_acc_collapsed4"))

        mn_r = _pct(cv_stats.get("mean_best_val_collapsed4_mn_recall"))
        zn_r = _pct(cv_stats.get("mean_best_val_collapsed4_zn_recall"))
        viii_r = _pct(cv_stats.get("mean_best_val_collapsed4_class_viii_recall"))
        cu_r = _pct(cv_stats.get("mean_best_val_collapsed4_cu_recall"))

        c4_val = cv_stats.get("mean_best_val_balanced_acc_collapsed4")
        delta_pmm = (float(c4_val) - PINMYMETAL_FIG2A_CV["collapsed4_balanced_acc"]) * 100 if c4_val is not None else 0.0
        delta_str = f"**{delta_pmm:+.2f} pp**" if vc4_m > 0 else "-"

        lines.append(
            f"| **{short_name}** | "
            f"{v5_m:.2f}% ± {v5_s:.2f}% | "
            f"**{vc4_m:.2f}% ± {vc4_s:.2f}%** | "
            f"{mn_r:.1f}% | "
            f"{zn_r:.1f}% | "
            f"{viii_r:.1f}% | "
            f"{cu_r:.1f}% | "
            f"{delta_str} |"
        )

    lines.extend([
        "",
        "## Table 2: Held-Out Test Set Performance",
        "",
        "| Architecture | Test Mode | Test Bal Acc (Collapsed-4) | Mn Recall | Zn Recall | Group VIII Recall | Cu Recall | Delta vs PMM Fig 2b (67.85%) | Delta vs Metal3D Fig 2c (61.70%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **PinMyMetal Fig 2b Baseline** | Held-Out Test | **{PINMYMETAL_FIG2B_TEST['collapsed4_balanced_acc']*100:.2f}%** | {PINMYMETAL_FIG2B_TEST['recalls']['Mn']*100:.1f}% | {PINMYMETAL_FIG2B_TEST['recalls']['Zn']*100:.1f}% | {PINMYMETAL_FIG2B_TEST['recalls']['Class VIII']*100:.1f}% | {PINMYMETAL_FIG2B_TEST['recalls']['Cu']*100:.1f}% | Baseline | +6.15 pp |",
        f"| **Metal3D Fig 2c Baseline** | External Test | **{PINMYMETAL_FIG2C_METAL3D['collapsed4_balanced_acc']*100:.2f}%** | {PINMYMETAL_FIG2C_METAL3D['recalls']['Mn']*100:.1f}% | {PINMYMETAL_FIG2C_METAL3D['recalls']['Zn']*100:.1f}% | {PINMYMETAL_FIG2C_METAL3D['recalls']['Class VIII']*100:.1f}% | {PINMYMETAL_FIG2C_METAL3D['recalls']['Cu']*100:.1f}% | -6.15 pp | Baseline |",
    ])

    for model_key, res in results.items():
        short_name = MODEL_CONFIGS[model_key]["short_name"]
        cv_stats = res.get("cv_stats", {})
        ens = res.get("ensemble", {})
        ens_metrics = ens.get("ensemble_metrics", {}) if ens else {}

        t_bal4_m = _pct(cv_stats.get("mean_test_metal_collapsed4_balanced_acc"))
        t_bal4_s = _pct(cv_stats.get("std_test_metal_collapsed4_balanced_acc"))
        t_mn = _pct(cv_stats.get("mean_test_metal_collapsed4_mn_recall"))
        t_zn = _pct(cv_stats.get("mean_test_metal_collapsed4_zn_recall"))
        t_viii = _pct(cv_stats.get("mean_test_metal_collapsed4_class_viii_recall"))
        t_cu = _pct(cv_stats.get("mean_test_metal_collapsed4_cu_recall"))

        mean_test_bal = cv_stats.get("mean_test_metal_collapsed4_balanced_acc")
        delta_t_pmm = (float(mean_test_bal) - PINMYMETAL_FIG2B_TEST["collapsed4_balanced_acc"]) * 100 if mean_test_bal is not None else 0.0
        delta_t_m3d = (float(mean_test_bal) - PINMYMETAL_FIG2C_METAL3D["collapsed4_balanced_acc"]) * 100 if mean_test_bal is not None else 0.0

        lines.append(
            f"| **{short_name} (Per-Fold Mean)** | 5-Fold Mean | "
            f"**{t_bal4_m:.2f}% ± {t_bal4_s:.2f}%** | "
            f"{t_mn:.1f}% | {t_zn:.1f}% | {t_viii:.1f}% | {t_cu:.1f}% | "
            f"{delta_t_pmm:+.2f} pp | "
            f"{delta_t_m3d:+.2f} pp |"
        )

        if ens_metrics:
            e_bal4 = _pct(ens_metrics.get("test_ensemble_metal_collapsed4_balanced_acc"))
            recalls = ens_metrics.get("test_ensemble_metal_collapsed4_per_class_recall", {}) or {}
            mn_r = _pct(recalls.get("Mn"))
            zn_r = _pct(recalls.get("Zn"))
            viii_r = _pct(recalls.get("Class VIII"))
            cu_r = _pct(recalls.get("Cu"))

            ens_bal = ens_metrics.get("test_ensemble_metal_collapsed4_balanced_acc")
            delta_e_pmm = (float(ens_bal) - PINMYMETAL_FIG2B_TEST["collapsed4_balanced_acc"]) * 100 if ens_bal is not None else 0.0
            delta_e_m3d = (float(ens_bal) - PINMYMETAL_FIG2C_METAL3D["collapsed4_balanced_acc"]) * 100 if ens_bal is not None else 0.0

            lines.append(
                f"| **{short_name} (5-Fold Ensemble)** | 5-Fold Ensemble | "
                f"**{e_bal4:.2f}%** | "
                f"{mn_r:.1f}% | "
                f"{zn_r:.1f}% | "
                f"{viii_r:.1f}% | "
                f"{cu_r:.1f}% | "
                f"**{delta_e_pmm:+.2f} pp** | "
                f"**{delta_e_m3d:+.2f} pp** |"
            )

    return "\n".join(lines)


def run_campaign(
    models: list[str],
    folds: list[int],
    dataset_name: str,
    train_dir: Path,
    train_csv: Path,
    test_dir: Path | None,
    test_csv: Path | None,
    feat_dir: Path,
    esm_dir: Path,
    runs_dir: Path,
    metal_example_unit: str = "ion",
    metal_label_scheme: str = "five_class",
    train_val_split_by: str = "pocket_id",
    epochs: int = 50,
    batch_size: int = 16,
    device: str = "cuda",
    seed: int = 42,
    skip_existing: bool = True,
    save_epoch_checkpoints: bool = False,
    python_bin: str = sys.executable,
    evaluate_test: bool = True,
    n_folds: int = 5,
) -> dict[str, Any]:
    runs_dir.mkdir(parents=True, exist_ok=True)
    status_file = runs_dir / "runner_state.json"
    summary_file = runs_dir / "5fold_summary.json"

    print("=" * 76, flush=True)
    print("STARTING 5-FOLD BENCHMARK CAMPAIGN", flush=True)
    print(f"Dataset:     {dataset_name}", flush=True)
    print(f"Unit:        {metal_example_unit.upper()} LEVEL", flush=True)
    print(f"Train Dir:   {train_dir}", flush=True)
    print(f"Train CSV:   {train_csv}", flush=True)
    print(f"Test Dir:    {test_dir or 'None'}", flush=True)
    print(f"Test CSV:    {test_csv or 'None'}", flush=True)
    print(f"Split By:    {train_val_split_by}", flush=True)
    print(f"Runs Dir:    {runs_dir}", flush=True)
    print(f"Models:      {models}", flush=True)
    print(f"Folds:       {folds}", flush=True)
    print(f"Epochs:      {epochs}", flush=True)
    print(f"Device:      {device}", flush=True)
    print("=" * 76, flush=True)

    state: dict[str, Any] = {
        "status": "in_progress",
        "dataset_name": dataset_name,
        "metal_example_unit": metal_example_unit,
        "models": models,
        "folds": folds,
        "epochs": epochs,
        "batch_size": batch_size,
        "device": device,
        "runs_dir": str(runs_dir),
        "train_dir": str(train_dir),
        "train_csv": str(train_csv),
        "test_dir": str(test_dir) if test_dir else None,
        "test_csv": str(test_csv) if test_csv else None,
        "model_results": {},
        "start_time": time.time(),
    }
    write_json(status_file, state)

    overall_results: dict[str, Any] = {}

    for model_key in models:
        print("#" * 76, flush=True)
        print(f"# RUNNING MODEL: {model_key} [{metal_example_unit.upper()} LEVEL]", flush=True)
        print(f"# {MODEL_CONFIGS[model_key]['description']}", flush=True)
        print("#" * 76, flush=True)

        model_fold_summaries = []

        def fold_run_name(f_idx: int) -> str:
            return f"{model_key}_{metal_example_unit}_fold{f_idx}"

        for fold_idx in folds:
            run_name = fold_run_name(fold_idx)
            run_dir = runs_dir / run_name

            state["status"] = f"running_{run_name}"
            write_json(status_file, state)

            rc, elapsed = run_single_fold(
                python_bin=python_bin,
                model_key=model_key,
                fold_idx=fold_idx,
                n_folds=n_folds,
                train_dir=train_dir,
                train_csv=train_csv,
                runs_dir=runs_dir,
                run_name=run_name,
                feat_dir=feat_dir,
                esm_dir=esm_dir,
                metal_example_unit=metal_example_unit,
                metal_label_scheme=metal_label_scheme,
                train_val_split_by=train_val_split_by,
                epochs=epochs,
                batch_size=batch_size,
                device=device,
                seed=seed,
                skip_existing=skip_existing,
                save_epoch_checkpoints=save_epoch_checkpoints,
            )

            if rc != 0:
                print(f"[RUNNER] ERROR: {run_name} exited with code {rc}! Aborting.", flush=True)
                state["status"] = f"failed_{run_name}"
                write_json(status_file, state)
                sys.exit(rc)

            if evaluate_test and test_dir is not None and test_csv is not None:
                evaluate_test_set_for_fold(
                    model_key=model_key,
                    fold_idx=fold_idx,
                    run_name=run_name,
                    runs_dir=runs_dir,
                    test_dir=test_dir,
                    test_csv=test_csv,
                    feat_dir=feat_dir,
                    esm_dir=esm_dir,
                    metal_example_unit=metal_example_unit,
                    metal_label_scheme=metal_label_scheme,
                    device=device,
                    batch_size=batch_size,
                )

            summary = summarize_single_fold(run_dir, fold_idx, run_name)
            summary["elapsed_seconds"] = elapsed
            summary["return_code"] = rc
            model_fold_summaries.append(summary)

            state["model_results"].setdefault(model_key, {})["folds"] = model_fold_summaries
            write_json(status_file, state)

        cv_stats = compute_oof_cv_metrics(model_fold_summaries)
        print(f"[RUNNER] {model_key} completed all requested folds.", flush=True)
        if "mean_best_val_balanced_acc_collapsed4" in cv_stats:
            print(
                f"  OOF CV Val Collapsed-4 Bal Acc: "
                f"{cv_stats['mean_best_val_balanced_acc_collapsed4']*100:.2f}% ± "
                f"{cv_stats['std_best_val_balanced_acc_collapsed4']*100:.2f}%",
                flush=True,
            )

        ensemble_res = None
        if evaluate_test and test_dir is not None and len(folds) == n_folds:
            ensemble_res = evaluate_model_ensemble(
                model_key=model_key,
                runs_dir=runs_dir,
                run_name_fn=fold_run_name,
                n_folds=n_folds,
                metal_label_scheme=metal_label_scheme,
            )

        overall_results[model_key] = {
            "config": MODEL_CONFIGS[model_key],
            "folds": model_fold_summaries,
            "cv_stats": cv_stats,
            "ensemble": ensemble_res,
        }
        state["model_results"][model_key] = overall_results[model_key]
        write_json(status_file, state)

    state["status"] = "completed"
    state["end_time"] = time.time()
    state["total_duration_seconds"] = state["end_time"] - state["start_time"]
    write_json(status_file, state)
    write_json(summary_file, state)

    table_md = format_comparison_table(
        overall_results,
        dataset_name=dataset_name,
        metal_example_unit=metal_example_unit,
    )
    table_path = runs_dir / f"{dataset_name}_{metal_example_unit}_5fold_comparison_table.md"
    with open(table_path, "w", encoding="utf-8") as handle:
        handle.write(table_md + "\n")
    print("=" * 76, flush=True)
    print("FINAL BENCHMARK COMPARISON TABLE:", flush=True)
    print("=" * 76, flush=True)
    print(table_md, flush=True)
    print("=" * 76, flush=True)
    print(f"Saved to {table_path}", flush=True)

    return state


def parse_args() -> argparse.Namespace:
    colab_root = Path("/content/DeepMzyme_Data/DeepMzyme_Data")
    local_root = REPO_ROOT / "DeepMzyme_Data"
    default_data_dir = colab_root if colab_root.exists() else local_root

    parser = argparse.ArgumentParser(
        description="Generalized 5-Fold Cross-Validation Runner (Pocket & Metal-Ion Focused Levels)"
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="train_and_test_sets_structures_exact_pinmymetal",
        help=(
            "Dataset folder name (under data-root) or absolute path. "
            "Examples: train_and_test_sets_structures_exact_pinmymetal, "
            "train_and_test_sets_structures_zenodo_pmm_exact, "
            "train_and_test_sets_structures_non_overlapped_pinmymetal, "
            "CLEAN_30_train_test_split_0, etc."
        ),
    )
    parser.add_argument(
        "--metal-example-unit",
        "-u",
        choices=["pocket", "ion"],
        default="ion",
        help="Supervision granularity: 'pocket' (clustered 5Å centroid) or 'ion' (individual metal ion coordinates, default: ion)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["benchmark_enhanced_only_gvp", "benchmark_only_esm", "benchmark_enhanced_gvp_esmc"],
        help="List of models to run. Options: benchmark_enhanced_only_gvp, benchmark_only_esm, benchmark_enhanced_gvp_esmc",
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="Fold indices to run (default: 0 1 2 3 4)")
    parser.add_argument("--n-folds", type=int, default=5, help="Total number of cross-validation folds (default: 5)")
    parser.add_argument(
        "--train-val-split-by",
        type=str,
        default="pocket_id",
        choices=["pocket_id", "pdbid", "pdbid_chain", "structure_id"],
        help="Grouping strategy for CV split. 'pocket_id' groups sibling ions in ion mode (default: pocket_id)",
    )
    parser.add_argument(
        "--metal-label-scheme",
        type=str,
        default="five_class",
        help="Metal label scheme (default: five_class, with collapsed-4 evaluation)",
    )
    parser.add_argument("--data-root", type=Path, default=default_data_dir, help="Root path to DeepMzyme_Data directory")
    parser.add_argument("--runs-dir", type=Path, default=None, help="Root directory for run outputs (defaults to runs_<dataset>_<unit>)")
    parser.add_argument("--train-summary-csv", type=Path, default=None, help="Explicit path to train summary CSV")
    parser.add_argument("--test-summary-csv", type=Path, default=None, help="Explicit path to test summary CSV")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (default: cuda if available else cpu)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--no-skip-existing", action="store_true", help="Force re-running already completed folds")
    parser.add_argument(
        "--save-epoch-checkpoints",
        action="store_true",
        help="Write a checkpoint after every epoch (recommended on remote VMs to prevent loss upon preemption)",
    )
    parser.add_argument("--no-evaluate-test", action="store_true", help="Skip held-out test evaluation")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths, counts, and print execution plan and fold commands without running training",
    )
    parser.add_argument("--python-bin", type=str, default=sys.executable, help="Python interpreter binary path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve dataset paths
    dataset_id, train_dir, train_csv, test_dir, test_csv = resolve_dataset_layout(
        data_root=args.data_root,
        dataset_name_or_path=args.dataset,
        train_summary_csv=args.train_summary_csv,
        test_summary_csv=args.test_summary_csv,
    )

    feat_dir = args.data_root / "updated_feature_extraction"
    esm_dir = args.data_root / "esm_embeddings"

    # Default runs directory: segregated by dataset and unit
    if args.runs_dir is not None:
        runs_dir = args.runs_dir
    else:
        if Path("/content").exists():
            runs_root = Path(f"/content/runs/runs_{dataset_id}_{args.metal_example_unit}")
        elif Path("/media/mechti/Data1").exists():
            runs_root = Path(f"/media/mechti/Data1/DeepMzyme_Data/runs_{dataset_id}_{args.metal_example_unit}")
        else:
            runs_root = REPO_ROOT / "runs" / f"runs_{dataset_id}_{args.metal_example_unit}"
        runs_dir = runs_root

    resolved_models = []
    for m in args.models:
        clean = m.strip().lower()
        if clean in MODEL_ALIASES:
            resolved_models.append(MODEL_ALIASES[clean])
        elif m in MODEL_CONFIGS:
            resolved_models.append(m)
        else:
            raise ValueError(f"Unknown model architecture {m!r}. Choose from: {list(MODEL_CONFIGS.keys())}")

    if args.dry_run:
        print("=" * 76)
        print("DRY RUN VALIDATION: 5-FOLD BENCHMARK PLAN")
        print("=" * 76)
        print(f"Dataset ID:          {dataset_id}")
        print(f"Example Granularity: {args.metal_example_unit.upper()} LEVEL (--metal-example-unit {args.metal_example_unit})")
        print(f"Train Directory:     {train_dir} (exists: {train_dir.exists()})")
        print(f"Train CSV:           {train_csv} (exists: {train_csv.exists()})")
        print(f"Test Directory:      {test_dir} (exists: {test_dir.exists()})")
        print(f"Test CSV:            {test_csv} (exists: {test_csv.exists()})")

        def _count_csv(p: Path) -> int:
            try:
                with open(p, newline="", encoding="utf-8") as f:
                    return max(0, sum(1 for _ in csv.DictReader(f)))
            except Exception:
                return -1

        print(f"Train Ion Sites:     {_count_csv(train_csv):,}")
        print(f"Test Ion Sites:      {_count_csv(test_csv):,}")
        print(f"Output Directory:    {runs_dir}")
        print(f"Split Strategy:      --train-val-split-by {args.train_val_split_by} ({args.n_folds} folds)")
        print(f"Models ({len(resolved_models)}):      {resolved_models}")
        print(f"Folds to run:        {args.folds}")
        print(f"Epochs:              {args.epochs}")
        print(f"Batch Size:          {args.batch_size}")
        print(f"Device:              {args.device}")
        print("=" * 76)
        print("PLANNED FOLD COMMANDS:")
        for m in resolved_models:
            print(f"\n--- Model: {m} ---")
            for f in args.folds:
                run_name = f"{m}_{args.metal_example_unit}_fold{f}"
                cmd = build_fold_command(
                    python_bin=args.python_bin,
                    model_key=m,
                    fold_idx=f,
                    n_folds=args.n_folds,
                    train_dir=train_dir,
                    train_csv=train_csv,
                    runs_dir=runs_dir,
                    run_name=run_name,
                    feat_dir=feat_dir,
                    esm_dir=esm_dir,
                    metal_example_unit=args.metal_example_unit,
                    metal_label_scheme=args.metal_label_scheme,
                    train_val_split_by=args.train_val_split_by,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    device=args.device,
                    seed=args.seed,
                    save_epoch_checkpoints=args.save_epoch_checkpoints,
                )
                print(f"Fold {f}: {' '.join(cmd)}\n")
        print("=" * 76)
        print("Dry run validation successful: all paths, options, and commands verified.")
        return

    run_campaign(
        models=resolved_models,
        folds=args.folds,
        dataset_name=dataset_id,
        train_dir=train_dir,
        train_csv=train_csv,
        test_dir=test_dir,
        test_csv=test_csv,
        feat_dir=feat_dir,
        esm_dir=esm_dir,
        runs_dir=runs_dir,
        metal_example_unit=args.metal_example_unit,
        metal_label_scheme=args.metal_label_scheme,
        train_val_split_by=args.train_val_split_by,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        skip_existing=not args.no_skip_existing,
        save_epoch_checkpoints=args.save_epoch_checkpoints,
        python_bin=args.python_bin,
        evaluate_test=not args.no_evaluate_test,
        n_folds=args.n_folds,
    )


if __name__ == "__main__":
    main()
