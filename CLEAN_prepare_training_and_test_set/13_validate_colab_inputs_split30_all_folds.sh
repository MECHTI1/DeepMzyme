#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_common_split30_fold0.sh"

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CLEAN_ALL_FOLDS_WORK_ROOT="${CLEAN_ALL_FOLDS_WORK_ROOT:-/media/Data/clean_sets/split30/all_folds}"
export LOG_DIR="${CLEAN_ALL_FOLDS_LOG_DIR:-$CLEAN_ALL_FOLDS_WORK_ROOT/logs}"
mkdir -p "$LOG_DIR"

dataset_args=()
for fold in 0 1 2 3 4; do
  dataset_args+=(--dataset-root "DeepMzyme_Data/CLEAN_${CLEAN_IDENTITY}_train_test_split_${fold}")
done

run_logged "13_validate_colab_inputs_split${CLEAN_IDENTITY}_all_folds" \
  "$PYTHON_BIN" src/build_colab_bundle.py \
    "${dataset_args[@]}" \
    --include-esm-embeddings \
    --skip-bundle
