#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_common_split30_fold0.sh"
run_logged "02_fetch_alphafill_split30_fold0_full" \
  "$PYTHON_BIN" CLEAN_prepare_training_and_test_set/clean_prepare.py fetch-alphafill \
    --identity "$CLEAN_IDENTITY" \
    --fold "$CLEAN_FOLD" \
    --work-root "$WORK_ROOT" \
    --splits train test \
    --prefilter-uniprot-supported-metals
