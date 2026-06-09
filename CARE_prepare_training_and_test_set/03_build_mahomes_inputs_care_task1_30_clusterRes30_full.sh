#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_common_care_task1_30_clusterRes30.sh"
run_logged "03_build_mahomes_inputs_care_task1_30_clusterRes30_full" \
  "$PYTHON_BIN" CARE_prepare_training_and_test_set/care_prepare.py build-mahomes-inputs \
    --work-root "$WORK_ROOT" \
    --splits train test \
    --min-alphafill-identity 0.30 \
    --min-alignment-length 85 \
    --site-dedup-distance 1.0 \
    --uniprot-metal-policy require_supported \
    --use-uniprot-annotation-cache \
    --no-fetch-missing-uniprot
