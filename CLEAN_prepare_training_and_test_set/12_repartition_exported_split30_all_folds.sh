#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_common_split30_fold0.sh"

export CLEAN_ALL_FOLDS_WORK_ROOT="${CLEAN_ALL_FOLDS_WORK_ROOT:-/media/Data/clean_sets/split30/all_folds}"
export LOG_DIR="${CLEAN_ALL_FOLDS_LOG_DIR:-$CLEAN_ALL_FOLDS_WORK_ROOT/logs}"
mkdir -p "$LOG_DIR"

run_logged "12_repartition_exported_split${CLEAN_IDENTITY}_all_folds" \
  "$PYTHON_BIN" CLEAN_prepare_training_and_test_set/clean_prepare.py repartition-exported-dataset \
    --identity "$CLEAN_IDENTITY" \
    --source-dataset-root "$PROJECT_ROOT/DeepMzyme_Data/CLEAN_${CLEAN_IDENTITY}_train_test_split_0" \
    --output-base "$PROJECT_ROOT/DeepMzyme_Data" \
    --fold 1 \
    --fold 2 \
    --fold 3 \
    --fold 4 \
    --overwrite
