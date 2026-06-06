#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_common_split30_fold0.sh"
run_logged "06_summarize_mahomes_split30_fold0" \
  "$PYTHON_BIN" CLEAN_prepare_training_and_test_set/clean_prepare.py summarize-mahomes \
    --work-root "$WORK_ROOT" \
    --splits train test
