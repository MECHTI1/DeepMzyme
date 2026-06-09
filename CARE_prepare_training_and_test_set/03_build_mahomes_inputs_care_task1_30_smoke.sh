#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_common_care_task1_30.sh"

# Smoke scope is controlled by the fetch step. This builds MAHOMES inputs from
# whichever smoke-limited AlphaFill/UniProt files are already present, using the
# same thresholds as the full preparation.
run_logged "03_build_mahomes_inputs_care_task1_30_smoke" \
  "$PYTHON_BIN" CARE_prepare_training_and_test_set/care_prepare.py build-mahomes-inputs \
    --work-root "$WORK_ROOT" \
    --splits train test \
    --limit-per-split 20 \
    --min-alphafill-identity 0.30 \
    --min-alignment-length 85 \
    --site-dedup-distance 1.0 \
    --uniprot-metal-policy require_supported \
    --use-uniprot-annotation-cache \
    --no-fetch-missing-uniprot
