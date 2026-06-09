#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_common_care_task1_30.sh"
run_logged "02_fetch_alphafill_care_task1_30_smoke" \
  bash -c '
    set -euo pipefail
    "$PYTHON_BIN" CARE_prepare_training_and_test_set/care_prepare.py prefetch-uniprot-annotations \
      --work-root "$WORK_ROOT" \
      --splits train test \
      --limit-per-split 20 \
      --batch-size "$CARE_UNIPROT_BATCH_SIZE"
    "$PYTHON_BIN" CARE_prepare_training_and_test_set/care_prepare.py fetch-alphafill \
      --work-root "$WORK_ROOT" \
      --splits train test \
      --prefilter-uniprot-supported-metals \
      --skip-uniprot \
      --use-uniprot-annotation-cache \
      --download-cif-only-if-json-has-supported-candidate \
      --min-alphafill-identity 0.30 \
      --min-alignment-length 85 \
      --limit-per-split 20
  '
