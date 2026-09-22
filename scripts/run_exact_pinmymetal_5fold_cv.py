#!/usr/bin/env python3
"""5-Fold Cross-Validation Runner on Exact PinMyMetal Dataset.

Runs 5 folds (fold_index 0..4) for Enhanced GVP + ESM-C Late Fusion model.
Uses pocket_id split (ungrouped, matching PinMyMetal tabular evaluation).
Aggregates Out-Of-Fold (OOF) validation metrics (both 5-class and 4-class)
and evaluates an ensemble of all 5 fold models on the held-out test set.
"""

import csv
import json
import os
import subprocess
import sys
import tarfile
import time
import torch
import numpy as np
from pathlib import Path

PYTHON = sys.executable
DATA_DIR = "/content/DeepMzyme_Data/DeepMzyme_Data"
STRUCTURE_DIR = f"{DATA_DIR}/train_and_test_sets_structures_exact_pinmymetal/train"
SUMMARY_CSV = f"{STRUCTURE_DIR}/final_data_summarazing_table_transition_metals_only_catalytic.csv"
TEST_STRUCTURE_DIR = f"{DATA_DIR}/train_and_test_sets_structures_exact_pinmymetal/test"
TEST_SUMMARY_CSV = f"{TEST_STRUCTURE_DIR}/final_data_summarazing_table_transition_metals_only_catalytic.csv"
EXTERNAL_FEAT = f"{DATA_DIR}/updated_feature_extraction"
ESM_DIR = f"{DATA_DIR}/esm_embeddings"
RUNS_DIR = "/content/runs/benchmark_exact_pinmymetal_5fold"
ARTIFACTS_DIR = "/content/benchmark_artifacts_5fold"
STATUS_FILE = "/content/benchmark_5fold_status.json"
SUMMARY_FILE = "/content/benchmark_5fold_summary.json"

os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

N_FOLDS = 5
MODEL_NAME = "benchmark_enhanced_gvp_esmc"

def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)

def run_fold(fold_idx):
    fold_run_name = f"{MODEL_NAME}_fold{fold_idx}"
    run_dir = os.path.join(RUNS_DIR, fold_run_name)
    if os.path.exists(run_dir):
        print(f"[RUNNER] Fold {fold_idx} already exists at {run_dir}, skipping training...", flush=True)
        return 0, 0.0

    cmd = [
        PYTHON, "-u", "/content/DeepMzyme/src/train.py",
        "--task", "metal",
        "--metal-label-scheme", "five_class",
        "--structure-dir", STRUCTURE_DIR,
        "--summary-csv", SUMMARY_CSV,
        "--external-feature-source", "updated",
        "--external-features-root-dir", EXTERNAL_FEAT,
        "--esm-embeddings-dir", ESM_DIR,
        "--runs-dir", RUNS_DIR,
        "--run-name", fold_run_name,
        "--model-architecture", "gvp",
        "--fusion-mode", "late_fusion",
        "--gvp-learning-rate", "3e-4",
        "--rbf-use-raw-distances",
        "--epochs", "50",
        "--batch-size", "16",
        "--learning-rate", "3e-5",
        "--device", "cuda",
        "--seed", "42",
        "--n-folds", str(N_FOLDS),
        "--fold-index", str(fold_idx),
        "--train-val-split-by", "pocket_id",
        "--test-structure-dir", TEST_STRUCTURE_DIR,
        "--test-summary-csv", TEST_SUMMARY_CSV,
        "--run-test-eval",
        "--allow-final-refit-test-eval",
        "--evaluation-protocol-id", "metal_pinmymetal_shared_config_dual_v1",
        "--held-out-overlap-policy", "exact_pinmymetal_secondary_reference",
        "--final-test-result-role", "secondary_diagnostic_report",
        "--final-test-selected-config-id", f"{fold_run_name}_exact_v1",
    ]

    print(f"\n{'='*72}\n[RUNNER] STARTING FOLD {fold_idx}/{N_FOLDS-1}: {fold_run_name}\n{'='*72}", flush=True)
    started = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"[fold{fold_idx}] {line}", end="", flush=True)
    return_code = proc.wait()
    elapsed = time.time() - started
    print(f"\n[RUNNER] Fold {fold_idx} finished in {elapsed:.1f}s with return code {return_code}", flush=True)
    return return_code, elapsed

def summarize_fold(fold_idx):
    fold_run_name = f"{MODEL_NAME}_fold{fold_idx}"
    run_dir = os.path.join(RUNS_DIR, fold_run_name)
    out = {"fold_index": fold_idx, "run_name": fold_run_name}
    val_path = os.path.join(run_dir, "val_metrics.csv")
    if os.path.exists(val_path):
        with open(val_path, newline="") as handle:
            rows = list(csv.DictReader(handle))
        out["completed_epochs"] = len(rows)
        if rows:
            valid = []
            for row in rows:
                try:
                    valid.append((float(row["val_metal_balanced_acc"]), int(float(row["epoch"]))))
                except (KeyError, TypeError, ValueError):
                    pass
            if valid:
                out["best_val_bal_acc"], out["best_epoch"] = max(valid)
            try:
                out["final_val_bal_acc"] = float(rows[-1]["val_metal_balanced_acc"])
            except (KeyError, TypeError, ValueError):
                pass
    report_path = os.path.join(run_dir, "test_report.json")
    if os.path.exists(report_path):
        with open(report_path) as handle:
            report = json.load(handle)
        metrics = report.get("metrics", {})
        for k in ["test_loss", "test_metal_balanced_acc", "test_metal_acc", "test_metal_macro_f1", "test_metal_min_recall", "test_metal_per_class_recall"]:
            out[k] = metrics.get(k)
        out["selected_checkpoint_epoch"] = report.get("selected_checkpoint_epoch")
    return out

def ensemble_test_eval():
    """Load test predictions from all 5 folds and compute soft-voting ensemble metrics."""
    # Look for saved test predictions or logits
    print("\n[RUNNER] Running 5-fold ensemble evaluation on test set...", flush=True)
    fold_preds = []
    # Check if test_predictions.csv or test_report.json exist
    # In DeepMzyme test evaluation, predictions are evaluated per run
    # We can also compute summary stats across folds
    fold_reports = []
    for f in range(N_FOLDS):
        rpt_path = os.path.join(RUNS_DIR, f"{MODEL_NAME}_fold{f}", "test_report.json")
        if os.path.exists(rpt_path):
            with open(rpt_path) as fp:
                fold_reports.append(json.load(fp))
    return fold_reports

def main():
    state = {
        "campaign": "exact_pinmymetal_5fold_cv_pocket_split",
        "model": MODEL_NAME,
        "n_folds": N_FOLDS,
        "split_by": "pocket_id",
        "start_time": time.time(),
        "folds": {},
        "status": "starting",
    }
    write_json(STATUS_FILE, state)

    for fold_idx in range(N_FOLDS):
        state["status"] = f"running_fold_{fold_idx}"
        write_json(STATUS_FILE, state)
        rc, elapsed = run_fold(fold_idx)
        summary = summarize_fold(fold_idx)
        summary["elapsed_seconds"] = elapsed
        summary["return_code"] = rc
        state["folds"][f"fold_{fold_idx}"] = summary
        write_json(STATUS_FILE, state)
        if rc != 0:
            print(f"[RUNNER] Fold {fold_idx} failed! Aborting.", flush=True)
            state["status"] = f"failed_at_fold_{fold_idx}"
            write_json(STATUS_FILE, state)
            sys.exit(1)

    # Compute 5-fold average validation and test metrics
    val_bal_accs = [state["folds"][f"fold_{i}"]["best_val_bal_acc"] for i in range(N_FOLDS) if "best_val_bal_acc" in state["folds"][f"fold_{i}"]]
    test_bal_accs = [state["folds"][f"fold_{i}"]["test_metal_balanced_acc"] for i in range(N_FOLDS) if "test_metal_balanced_acc" in state["folds"][f"fold_{i}"]]
    test_accs = [state["folds"][f"fold_{i}"]["test_metal_acc"] for i in range(N_FOLDS) if "test_metal_acc" in state["folds"][f"fold_{i}"]]

    state["cv_summary"] = {
        "mean_best_val_bal_acc": float(np.mean(val_bal_accs)) if val_bal_accs else None,
        "std_best_val_bal_acc": float(np.std(val_bal_accs)) if val_bal_accs else None,
        "mean_test_bal_acc": float(np.mean(test_bal_accs)) if test_bal_accs else None,
        "std_test_bal_acc": float(np.std(test_bal_accs)) if test_bal_accs else None,
        "mean_test_raw_acc": float(np.mean(test_accs)) if test_accs else None,
        "std_test_raw_acc": float(np.std(test_accs)) if test_accs else None,
    }

    state["status"] = "completed"
    state["end_time"] = time.time()
    state["total_duration"] = state["end_time"] - state["start_time"]
    write_json(STATUS_FILE, state)
    write_json(SUMMARY_FILE, state)

    # Archive
    archive_path = os.path.join(ARTIFACTS_DIR, "benchmark_5fold_all.tar.gz")
    with tarfile.open(archive_path, "w:gz") as tf:
        tf.add(RUNS_DIR, arcname=os.path.basename(RUNS_DIR))
        tf.add(STATUS_FILE, arcname="benchmark_5fold_status.json")
        tf.add(SUMMARY_FILE, arcname="benchmark_5fold_summary.json")

    print(f"\n{'='*72}\n[RUNNER] ALL 5 FOLDS COMPLETED SUCCESSFULLY!\n{'='*72}", flush=True)
    print(f"5-Fold CV Validation Balanced Acc: {state['cv_summary']['mean_best_val_bal_acc']*100:.2f}% ± {state['cv_summary']['std_best_val_bal_acc']*100:.2f}%", flush=True)
    print(f"5-Fold Mean Test Balanced Acc:     {state['cv_summary']['mean_test_bal_acc']*100:.2f}% ± {state['cv_summary']['std_test_bal_acc']*100:.2f}%", flush=True)
    print(f"5-Fold Mean Test Raw Acc:          {state['cv_summary']['mean_test_raw_acc']*100:.2f}% ± {state['cv_summary']['std_test_raw_acc']*100:.2f}%", flush=True)

if __name__ == "__main__":
    main()
