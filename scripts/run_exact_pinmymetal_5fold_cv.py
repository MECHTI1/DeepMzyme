#!/usr/bin/env python3
"""5-Fold Cross-Validation Runner for PinMyMetal Benchmark.

Supports all three DeepMzyme architectures under identical 5-fold CV protocol:
1. Multimodal ESM+GVP Late Fusion: benchmark_enhanced_gvp_esmc
2. Sequence-only ESM: benchmark_only_esm
3. Structure-only GVP: benchmark_enhanced_only_gvp

Setup:
- Tabular pocket-level split: --train-val-split-by pocket_id --n-folds 5
- Training set: train_and_test_sets_structures_exact_pinmymetal/train (1,597 pockets)
- Held-out test set: train_and_test_sets_structures_exact_pinmymetal/test (352 pockets)
- External features: updated_feature_extraction
- Embeddings: esm_embeddings

Evaluates:
1. 5-Fold OOF Validation performance (5-class and collapsed-4 balanced accuracy).
2. Held-out Test Set performance per-fold and 5-fold soft-voting ensemble predictions.
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
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in [
    REPO_ROOT / "src",
    Path("/content/DeepMzyme/src"),
    Path(__file__).resolve().parent / "src",
    Path("/content/src"),
]:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from label_schemes import configure_active_metal_label_scheme
from training.final_test_reporting import metal_metrics_from_probabilities

PINMYMETAL_FIG2A_CV = {
    "source": "PinMyMetal Fig 2a (5-fold CV on 1,597 pockets)",
    "collapsed4_balanced_acc": 0.7508,
    "recalls": {
        "Mn": 0.903,
        "Zn": 0.738,
        "Class VIII": 0.733,
        "Cu": 0.629,
    },
}

PINMYMETAL_FIG2B_TEST = {
    "source": "PinMyMetal Fig 2b (Held-Out Test Set on 352 pockets)",
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


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    tmp.replace(path)


def resolve_default_paths() -> dict[str, Path]:
    colab_root = Path("/content/DeepMzyme_Data/DeepMzyme_Data")
    local_root = REPO_ROOT / "DeepMzyme_Data"
    data_dir = colab_root if colab_root.exists() else local_root

    colab_runs = Path("/content/runs/benchmark_exact_pinmymetal_5fold")
    local_runs = REPO_ROOT / "runs" / "benchmark_exact_pinmymetal_5fold"
    runs_dir = colab_runs if Path("/content").exists() else local_runs

    return {
        "data_dir": data_dir,
        "runs_dir": runs_dir,
    }


def is_fold_complete(run_dir: Path, target_epochs: int = 50) -> bool:
    val_csv = run_dir / "val_metrics.csv"
    test_json = run_dir / "test_report.json"
    if not (val_csv.exists() and test_json.exists()):
        return False
    try:
        with open(val_csv, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) < target_epochs:
            return False
        with open(test_json, encoding="utf-8") as handle:
            report = json.load(handle)
        metrics = report.get("metrics", {})
        return "test_metal_balanced_acc" in metrics and "test_metal_collapsed4_balanced_acc" in metrics
    except Exception:
        return False


def build_fold_command(
    python_bin: str,
    model_key: str,
    fold_idx: int,
    n_folds: int,
    data_root: Path,
    runs_dir: Path,
    epochs: int = 50,
    batch_size: int = 16,
    device: str = "cuda",
    seed: int = 42,
    metal_example_unit: str = "pocket",
) -> tuple[str, list[str]]:
    cfg = MODEL_CONFIGS[model_key]
    run_name = f"{model_key}_{metal_example_unit}_fold{fold_idx}" if metal_example_unit != "pocket" else f"{model_key}_fold{fold_idx}"

    train_dir = data_root / "train_and_test_sets_structures_exact_pinmymetal" / "train"
    train_csv = train_dir / "final_data_summarazing_table_transition_metals_only_catalytic.csv"
    test_dir = data_root / "train_and_test_sets_structures_exact_pinmymetal" / "test"
    test_csv = test_dir / "final_data_summarazing_table_transition_metals_only_catalytic.csv"
    feat_dir = data_root / "updated_feature_extraction"
    esm_dir = data_root / "esm_embeddings"

    train_py = REPO_ROOT / "src" / "train.py"
    if not train_py.exists() and Path("/content/DeepMzyme/src/train.py").exists():
        train_py = Path("/content/DeepMzyme/src/train.py")

    cmd = [
        python_bin, "-u", str(train_py),
        "--task", "metal",
        "--metal-label-scheme", "five_class",
        "--metal-example-unit", metal_example_unit,
        "--structure-dir", str(train_dir),
        "--summary-csv", str(train_csv),
        "--external-feature-source", "updated",
        "--external-features-root-dir", str(feat_dir),
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
        "--train-val-split-by", "pocket_id",
        "--test-structure-dir", str(test_dir),
        "--test-summary-csv", str(test_csv),
        "--run-test-eval",
        "--allow-final-refit-test-eval",
        "--evaluation-protocol-id", "metal_pinmymetal_shared_config_dual_v1",
        "--held-out-overlap-policy", "exact_pinmymetal_secondary_reference",
        "--final-test-result-role", "secondary_diagnostic_report",
        "--final-test-selected-config-id", f"{run_name}_exact_v1",
    ]

    if cfg["use_esm"]:
        cmd.extend(["--esm-embeddings-dir", str(esm_dir)])

    if cfg["fusion_mode"]:
        cmd.extend(["--fusion-mode", cfg["fusion_mode"]])

    if cfg["gvp_lr"]:
        cmd.extend(["--gvp-learning-rate", str(cfg["gvp_lr"])])

    if cfg["rbf_raw"]:
        cmd.append("--rbf-use-raw-distances")

    return run_name, cmd


def run_single_fold(
    python_bin: str,
    model_key: str,
    fold_idx: int,
    n_folds: int,
    data_root: Path,
    runs_dir: Path,
    epochs: int = 50,
    batch_size: int = 16,
    device: str = "cuda",
    seed: int = 42,
    skip_existing: bool = True,
    metal_example_unit: str = "pocket",
) -> tuple[int, float]:
    run_name, cmd = build_fold_command(
        python_bin=python_bin,
        model_key=model_key,
        fold_idx=fold_idx,
        n_folds=n_folds,
        data_root=data_root,
        runs_dir=runs_dir,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        seed=seed,
        metal_example_unit=metal_example_unit,
    )
    run_dir = runs_dir / run_name

    if skip_existing and is_fold_complete(run_dir, target_epochs=epochs):
        print(f"[RUNNER] Fold {fold_idx} ({run_name}) already completed at {run_dir}, skipping...", flush=True)
        return 0, 0.0
    elif run_dir.exists():
        print(f"[RUNNER] Cleaning up incomplete prior run at {run_dir}...", flush=True)
        shutil.rmtree(run_dir, ignore_errors=True)

    print("=" * 76, flush=True)
    print(f"[RUNNER] STARTING FOLD {fold_idx}/{n_folds-1}: {run_name}", flush=True)
    print("=" * 76, flush=True)

    started = time.time()
    child_env = os.environ.copy()
    child_env["MKL_THREADING_LAYER"] = "GNU"
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=child_env)
    assert proc.stdout is not None
    prefix = f"[{model_key[-8:]}:f{fold_idx}]"
    for line in proc.stdout:
        print(f"{prefix} {line}", end="", flush=True)
    return_code = proc.wait()
    elapsed = time.time() - started
    print(f"[RUNNER] Fold {fold_idx} finished in {elapsed:.1f}s with return code {return_code}", flush=True)
    return return_code, elapsed


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
        out["test_loss"] = metrics.get("test_loss")
        out["test_metal_acc"] = metrics.get("test_metal_acc")
        out["test_metal_balanced_acc"] = metrics.get("test_metal_balanced_acc")
        out["test_metal_collapsed4_acc"] = metrics.get("test_metal_collapsed4_acc")
        out["test_metal_collapsed4_balanced_acc"] = metrics.get("test_metal_collapsed4_balanced_acc")
        out["test_metal_macro_f1"] = metrics.get("test_metal_macro_f1")
        out["test_metal_per_class_recall"] = metrics.get("test_metal_per_class_recall")
        out["test_metal_collapsed4_per_class_recall"] = metrics.get("test_metal_collapsed4_per_class_recall")
        out["selected_checkpoint_epoch"] = report.get("selected_checkpoint_epoch")

    return out


def evaluate_model_ensemble(
    model_key: str,
    runs_dir: Path,
    n_folds: int = 5,
    metal_example_unit: str = "pocket",
) -> dict[str, Any] | None:
    configure_active_metal_label_scheme("five_class")

    fold_probs = []
    fold_cal_probs = []
    targets = None

    for fold_idx in range(n_folds):
        run_name = f"{model_key}_{metal_example_unit}_fold{fold_idx}" if metal_example_unit != "pocket" else f"{model_key}_fold{fold_idx}"
        pred_path = runs_dir / run_name / "test_predictions.pt"
        if not pred_path.exists():
            print(f"[ENSEMBLE] Warning: {pred_path} not found. Cannot evaluate complete ensemble.", flush=True)
            return None
        data = torch.load(pred_path, map_location="cpu")
        fold_probs.append(data["metal_probabilities"].float())
        if "metal_calibrated_probabilities" in data:
            fold_cal_probs.append(data["metal_calibrated_probabilities"].float())
        if targets is None:
            targets = data["metal_y"].long()
        else:
            if not torch.equal(targets, data["metal_y"].long()):
                raise ValueError(f"Target mismatch in fold {fold_idx}")

    ensemble_probs = torch.stack(fold_probs, dim=0).mean(dim=0)
    ensemble_metrics = metal_metrics_from_probabilities(ensemble_probs, targets, prefix="test_ensemble")

    calibrated_metrics = None
    if len(fold_cal_probs) == n_folds:
        cal_ensemble_probs = torch.stack(fold_cal_probs, dim=0).mean(dim=0)
        calibrated_metrics = metal_metrics_from_probabilities(cal_ensemble_probs, targets, prefix="test_calibrated_ensemble")

    result = {
        "model": model_key,
        "n_folds": n_folds,
        "metal_example_unit": metal_example_unit,
        "n_test_sites": int(targets.size(0)),
        "ensemble_metrics": ensemble_metrics,
        "calibrated_ensemble_metrics": calibrated_metrics,
    }

    ens_suffix = f"_{metal_example_unit}" if metal_example_unit != "pocket" else ""
    ens_path = runs_dir / f"{model_key}{ens_suffix}_5fold_ensemble_report.json"
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
    ]
    cv_stats: dict[str, Any] = {}
    for key in keys_to_agg:
        vals = [f[key] for f in fold_summaries if key in f and f[key] is not None]
        if vals:
            cv_stats[f"mean_{key}"] = float(np.mean(vals))
            cv_stats[f"std_{key}"] = float(np.std(vals))
    return cv_stats


def format_comparison_table(results: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# EXACT PINMYMETAL 5-FOLD CV BENCHMARK COMPARISON",
        "",
        "## Table 1: 5-Fold Cross-Validation Performance (Validation Folds vs PinMyMetal Fig 2a)",
        "",
        "| Architecture | 5-Fold CV Val Bal Acc (5-class) | 5-Fold CV Val Bal Acc (Collapsed-4) | Mn Recall | Zn Recall | Group VIII Recall | Cu Recall | Delta vs PMM Fig 2a (75.08%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **PinMyMetal Fig 2a Baseline** | - | **{PINMYMETAL_FIG2A_CV['collapsed4_balanced_acc']*100:.2f}%** | {PINMYMETAL_FIG2A_CV['recalls']['Mn']*100:.1f}% | {PINMYMETAL_FIG2A_CV['recalls']['Zn']*100:.1f}% | {PINMYMETAL_FIG2A_CV['recalls']['Class VIII']*100:.1f}% | {PINMYMETAL_FIG2A_CV['recalls']['Cu']*100:.1f}% | Baseline |",
    ]

    for model_key, res in results.items():
        short_name = MODEL_CONFIGS[model_key]["short_name"]
        cv_stats = res.get("cv_stats", {})

        v5_m = cv_stats.get("mean_best_val_balanced_acc_5class", 0) * 100
        v5_s = cv_stats.get("std_best_val_balanced_acc_5class", 0) * 100
        vc4_m = cv_stats.get("mean_best_val_balanced_acc_collapsed4", 0) * 100
        vc4_s = cv_stats.get("std_best_val_balanced_acc_collapsed4", 0) * 100

        mn_r = cv_stats.get("mean_best_val_collapsed4_mn_recall", 0) * 100
        zn_r = cv_stats.get("mean_best_val_collapsed4_zn_recall", 0) * 100
        viii_r = cv_stats.get("mean_best_val_collapsed4_class_viii_recall", 0) * 100
        cu_r = cv_stats.get("mean_best_val_collapsed4_cu_recall", 0) * 100

        delta_pmm = (cv_stats.get("mean_best_val_balanced_acc_collapsed4", 0) - PINMYMETAL_FIG2A_CV["collapsed4_balanced_acc"]) * 100
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
        "## Table 2: Held-Out Test Set Performance (352 Pockets vs PinMyMetal Fig 2b & Metal3D Fig 2c)",
        "",
        "| Architecture | Test Mode | Test Raw Acc | Test Bal Acc (5-class) | Test Bal Acc (Collapsed-4) | Mn Recall | Zn Recall | Group VIII Recall | Cu Recall | Delta vs PMM Fig 2b (67.85%) | Delta vs Metal3D Fig 2c (61.70%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **PinMyMetal Fig 2b Baseline** | Held-Out Test | - | - | **{PINMYMETAL_FIG2B_TEST['collapsed4_balanced_acc']*100:.2f}%** | {PINMYMETAL_FIG2B_TEST['recalls']['Mn']*100:.1f}% | {PINMYMETAL_FIG2B_TEST['recalls']['Zn']*100:.1f}% | {PINMYMETAL_FIG2B_TEST['recalls']['Class VIII']*100:.1f}% | {PINMYMETAL_FIG2B_TEST['recalls']['Cu']*100:.1f}% | Baseline | +6.15 pp |",
        f"| **Metal3D Fig 2c Baseline** | External Test | - | - | **{PINMYMETAL_FIG2C_METAL3D['collapsed4_balanced_acc']*100:.2f}%** | {PINMYMETAL_FIG2C_METAL3D['recalls']['Mn']*100:.1f}% | {PINMYMETAL_FIG2C_METAL3D['recalls']['Zn']*100:.1f}% | {PINMYMETAL_FIG2C_METAL3D['recalls']['Class VIII']*100:.1f}% | {PINMYMETAL_FIG2C_METAL3D['recalls']['Cu']*100:.1f}% | -6.15 pp | Baseline |",
    ])

    for model_key, res in results.items():
        short_name = MODEL_CONFIGS[model_key]["short_name"]
        cv_stats = res.get("cv_stats", {})
        ens = res.get("ensemble", {})
        ens_metrics = ens.get("ensemble_metrics", {}) if ens else {}

        t_raw_m = cv_stats.get("mean_test_metal_acc", 0) * 100
        t_raw_s = cv_stats.get("std_test_metal_acc", 0) * 100
        t_bal5_m = cv_stats.get("mean_test_metal_balanced_acc", 0) * 100
        t_bal5_s = cv_stats.get("std_test_metal_balanced_acc", 0) * 100
        t_bal4_m = cv_stats.get("mean_test_metal_collapsed4_balanced_acc", 0) * 100
        t_bal4_s = cv_stats.get("std_test_metal_collapsed4_balanced_acc", 0) * 100

        delta_t_pmm = (cv_stats.get("mean_test_metal_collapsed4_balanced_acc", 0) - PINMYMETAL_FIG2B_TEST["collapsed4_balanced_acc"]) * 100
        delta_t_m3d = (cv_stats.get("mean_test_metal_collapsed4_balanced_acc", 0) - PINMYMETAL_FIG2C_METAL3D["collapsed4_balanced_acc"]) * 100

        lines.append(
            f"| **{short_name} (Per-Fold Mean)** | 5-Fold Mean | "
            f"{t_raw_m:.2f}% ± {t_raw_s:.2f}% | "
            f"{t_bal5_m:.2f}% ± {t_bal5_s:.2f}% | "
            f"**{t_bal4_m:.2f}% ± {t_bal4_s:.2f}%** | "
            f"- | - | - | - | "
            f"{delta_t_pmm:+.2f} pp | "
            f"{delta_t_m3d:+.2f} pp |"
        )

        if ens_metrics:
            e_raw = ens_metrics.get("test_ensemble_metal_acc", 0) * 100
            e_bal5 = ens_metrics.get("test_ensemble_metal_balanced_acc", 0) * 100
            e_bal4 = ens_metrics.get("test_ensemble_metal_collapsed4_balanced_acc", 0) * 100

            recalls = ens_metrics.get("test_ensemble_metal_collapsed4_per_class_recall", {})
            mn_r = recalls.get("Mn", 0) * 100
            zn_r = recalls.get("Zn", 0) * 100
            viii_r = recalls.get("Class VIII", 0) * 100
            cu_r = recalls.get("Cu", 0) * 100

            delta_e_pmm = (ens_metrics.get("test_ensemble_metal_collapsed4_balanced_acc", 0) - PINMYMETAL_FIG2B_TEST["collapsed4_balanced_acc"]) * 100
            delta_e_m3d = (ens_metrics.get("test_ensemble_metal_collapsed4_balanced_acc", 0) - PINMYMETAL_FIG2C_METAL3D["collapsed4_balanced_acc"]) * 100

            lines.append(
                f"| **{short_name} (5-Fold Ensemble)** | 5-Fold Ensemble | "
                f"{e_raw:.2f}% | "
                f"{e_bal5:.2f}% | "
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
    data_root: Path,
    runs_dir: Path,
    epochs: int = 50,
    batch_size: int = 16,
    device: str = "cuda",
    seed: int = 42,
    skip_existing: bool = True,
    python_bin: str = sys.executable,
    metal_example_unit: str = "pocket",
) -> dict[str, Any]:
    runs_dir.mkdir(parents=True, exist_ok=True)
    status_file = runs_dir / "benchmark_5fold_status.json"
    summary_file = runs_dir / "benchmark_5fold_summary.json"

    state: dict[str, Any] = {
        "campaign": "exact_pinmymetal_5fold_cv",
        "metal_example_unit": metal_example_unit,
        "models": models,
        "folds": folds,
        "epochs": epochs,
        "batch_size": batch_size,
        "device": device,
        "seed": seed,
        "data_root": str(data_root),
        "runs_dir": str(runs_dir),
        "start_time": time.time(),
        "status": "in_progress",
        "model_results": {},
    }
    write_json(status_file, state)

    overall_results: dict[str, dict[str, Any]] = {}

    for model_key in models:
        print("#" * 76, flush=True)
        print(f"# RUNNING MODEL: {model_key} [{metal_example_unit.upper()} LEVEL]", flush=True)
        print(f"# {MODEL_CONFIGS[model_key]['description']}", flush=True)
        print("#" * 76, flush=True)

        model_fold_summaries = []

        for fold_idx in folds:
            run_name = f"{model_key}_{metal_example_unit}_fold{fold_idx}" if metal_example_unit != "pocket" else f"{model_key}_fold{fold_idx}"
            run_dir = runs_dir / run_name

            state["status"] = f"running_{run_name}"
            write_json(status_file, state)

            rc, elapsed = run_single_fold(
                python_bin=python_bin,
                model_key=model_key,
                fold_idx=fold_idx,
                n_folds=len(folds),
                data_root=data_root,
                runs_dir=runs_dir,
                epochs=epochs,
                batch_size=batch_size,
                device=device,
                seed=seed,
                skip_existing=skip_existing,
                metal_example_unit=metal_example_unit,
            )

            summary = summarize_single_fold(run_dir, fold_idx, run_name)
            summary["elapsed_seconds"] = elapsed
            summary["return_code"] = rc
            model_fold_summaries.append(summary)

            state["model_results"].setdefault(model_key, {})["folds"] = model_fold_summaries
            write_json(status_file, state)

            if rc != 0:
                print(f"[RUNNER] ERROR: {run_name} exited with code {rc}! Aborting.", flush=True)
                state["status"] = f"failed_{run_name}"
                write_json(status_file, state)
                sys.exit(rc)

        cv_stats = compute_oof_cv_metrics(model_fold_summaries)
        print(f"[RUNNER] {model_key} Completed all requested folds.", flush=True)
        if "mean_best_val_balanced_acc_collapsed4" in cv_stats:
            print(f"  OOF CV Val Collapsed-4 Bal Acc: {cv_stats['mean_best_val_balanced_acc_collapsed4']*100:.2f}% ± {cv_stats['std_best_val_balanced_acc_collapsed4']*100:.2f}%", flush=True)
        if "mean_test_metal_collapsed4_balanced_acc" in cv_stats:
            print(f"  Mean Per-Fold Test Collapsed-4 Bal Acc: {cv_stats['mean_test_metal_collapsed4_balanced_acc']*100:.2f}% ± {cv_stats['std_test_metal_collapsed4_balanced_acc']*100:.2f}%", flush=True)

        ensemble_res = evaluate_model_ensemble(
            model_key, runs_dir, n_folds=len(folds), metal_example_unit=metal_example_unit
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

    table_md = format_comparison_table(overall_results)
    table_path = runs_dir / "benchmark_5fold_comparison_table.md"
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
    paths = resolve_default_paths()
    parser = argparse.ArgumentParser(description="PinMyMetal Exact 5-Fold Cross-Validation Benchmark Runner")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["benchmark_only_esm", "benchmark_enhanced_only_gvp", "benchmark_enhanced_gvp_esmc"],
        help="List of models to run. Options: benchmark_only_esm, benchmark_enhanced_only_gvp, benchmark_enhanced_gvp_esmc",
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="Fold indices (default: 0 1 2 3 4)")
    parser.add_argument(
        "--metal-example-unit",
        choices=["pocket", "ion"],
        default="pocket",
        help="Supervision granularity: 'pocket' (clustered 5Å centroid, default) or 'ion' (individual metal ion coordinates)",
    )
    parser.add_argument("--data-root", type=Path, default=paths["data_dir"], help="Path to DeepMzyme_Data directory")
    parser.add_argument("--runs-dir", type=Path, default=paths["runs_dir"], help="Path to save model runs")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device (default: cuda if available else cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--no-skip-existing", action="store_true", help="Force re-running already completed folds")
    parser.add_argument("--python-bin", type=str, default=sys.executable, help="Python interpreter binary path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    resolved_models = []
    for m in args.models:
        clean = m.strip().lower()
        if clean in MODEL_ALIASES:
            resolved_models.append(MODEL_ALIASES[clean])
        elif m in MODEL_CONFIGS:
            resolved_models.append(m)
        else:
            raise ValueError(f"Unknown model architecture {m!r}. Choose from: {list(MODEL_CONFIGS.keys())}")

    run_campaign(
        models=resolved_models,
        folds=args.folds,
        data_root=args.data_root,
        runs_dir=args.runs_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        skip_existing=not args.no_skip_existing,
        python_bin=args.python_bin,
        metal_example_unit=args.metal_example_unit,
    )


if __name__ == "__main__":
    main()
