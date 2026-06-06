#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_common_split30_fold0.sh"
run_logged "05_run_mahomes_test_split30_fold0" \
  env WORK_ROOT="$WORK_ROOT" SPLIT=test N_JOBS="$MAHOMES_N_JOBS" CLEAN_JOB_DIRS=1 SKIP_COMPLETED_JOBS=0 \
    bash CLEAN_prepare_training_and_test_set/run_mahomes_clean.sh
