#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_common_care_task1_30.sh"

ARGS=(
  "$PYTHON_BIN" CARE_prepare_training_and_test_set/care_prepare.py audit-care-task1
  --care-root "$CARE_ROOT"
  --care-task "$CARE_TASK"
  --care-split-name "$CARE_SPLIT_NAME"
  --work-root "$WORK_ROOT"
)

if [[ -n "${CARE_TRAIN_CSV:-}" ]]; then
  ARGS+=(--train-csv "$CARE_TRAIN_CSV")
fi

if [[ -n "${CARE_TEST_CSV:-}" ]]; then
  ARGS+=(--test-csv "$CARE_TEST_CSV")
fi

run_logged "01_audit_care_task1_30" "${ARGS[@]}"
