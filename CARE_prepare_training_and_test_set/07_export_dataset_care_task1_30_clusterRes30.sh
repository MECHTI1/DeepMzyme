#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_common_care_task1_30_clusterRes30.sh"
run_logged "07_export_dataset_care_task1_30_clusterRes30" \
  "$PYTHON_BIN" CARE_prepare_training_and_test_set/care_prepare.py export-dataset \
    --work-root "$WORK_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --overwrite
