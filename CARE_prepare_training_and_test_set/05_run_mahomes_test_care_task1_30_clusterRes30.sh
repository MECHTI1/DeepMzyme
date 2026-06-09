#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_common_care_task1_30_clusterRes30.sh"
run_logged "05_run_mahomes_test_care_task1_30_clusterRes30" \
  env WORK_ROOT="$WORK_ROOT" SPLIT=test N_JOBS="$MAHOMES_N_JOBS" CARE_JOB_DIRS=1 SKIP_COMPLETED_JOBS=0 \
    bash CARE_prepare_training_and_test_set/run_mahomes_care.sh
