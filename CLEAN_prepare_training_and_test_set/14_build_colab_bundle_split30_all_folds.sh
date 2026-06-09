#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_common_split30_fold0.sh"

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CLEAN_ALL_FOLDS_WORK_ROOT="${CLEAN_ALL_FOLDS_WORK_ROOT:-/media/Data/clean_sets/split30/all_folds}"
export LOG_DIR="${CLEAN_ALL_FOLDS_LOG_DIR:-$CLEAN_ALL_FOLDS_WORK_ROOT/logs}"
export CLEAN_ALL_FOLDS_BUNDLE_DIR="${CLEAN_ALL_FOLDS_BUNDLE_DIR:-$CLEAN_ALL_FOLDS_WORK_ROOT/bundles}"
export CLEAN_ALL_FOLDS_BUNDLE_NAME="${CLEAN_ALL_FOLDS_BUNDLE_NAME:-DeepMzyme_Data_v7_clean${CLEAN_IDENTITY}_all5_full_esm.tar.zst}"
export CLEAN_ALL_FOLDS_BUNDLE_OUTPUT="${CLEAN_ALL_FOLDS_BUNDLE_OUTPUT:-$CLEAN_ALL_FOLDS_BUNDLE_DIR/$CLEAN_ALL_FOLDS_BUNDLE_NAME}"
mkdir -p "$LOG_DIR" "$CLEAN_ALL_FOLDS_BUNDLE_DIR"

run_logged "14_build_colab_bundle_split${CLEAN_IDENTITY}_all_folds" bash -c '
  set -euo pipefail
  dataset_args=()
  for fold in 0 1 2 3 4; do
    dataset_args+=(--dataset-root "DeepMzyme_Data/CLEAN_${CLEAN_IDENTITY}_train_test_split_${fold}")
  done
  "$PYTHON_BIN" src/build_colab_bundle.py \
    "${dataset_args[@]}" \
    --include-esm-embeddings \
    --output-bundle "$CLEAN_ALL_FOLDS_BUNDLE_OUTPUT"
  sha256sum "$CLEAN_ALL_FOLDS_BUNDLE_OUTPUT" | tee "$CLEAN_ALL_FOLDS_BUNDLE_OUTPUT.sha256"
  ls -lh "$CLEAN_ALL_FOLDS_BUNDLE_OUTPUT" "$CLEAN_ALL_FOLDS_BUNDLE_OUTPUT.sha256"
'
