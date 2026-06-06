#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_common_split30_fold0.sh"
run_logged "01_audit_split30_fold0" \
  "$PYTHON_BIN" CLEAN_prepare_training_and_test_set/clean_prepare.py audit-split \
    --identity "$CLEAN_IDENTITY" \
    --fold "$CLEAN_FOLD" \
    --work-root "$WORK_ROOT"
