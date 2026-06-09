#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_common_care_task1_30_clusterRes30.sh"
run_logged "01_audit_care_task1_30_clusterRes30" \
  "$PYTHON_BIN" CARE_prepare_training_and_test_set/care_prepare.py audit-care-task1 \
    --care-root "$CARE_ROOT" \
    --train-csv "$CARE_TRAIN_CSV" \
    --test-csv "$CARE_TEST_CSV" \
    --care-task "$CARE_TASK" \
    --care-split-name "$CARE_SPLIT_NAME" \
    --train-representative-column "$CARE_TRAIN_REPRESENTATIVE_COLUMN" \
    --work-root "$WORK_ROOT"
