#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_common_split30_fold0.sh"

echo "[INFO] Cleaning prior CLEAN-generated MAHOMES inputs/jobs for split30 fold0. AlphaFill/UniProt/RCSB caches are preserved."
rm -rf "$WORK_ROOT/mahomes_inputs/train" "$WORK_ROOT/mahomes_inputs/test" \
       "$WORK_ROOT/mahomes/train" "$WORK_ROOT/mahomes/test" \
       "$WORK_ROOT/mahomes_outputs/train" "$WORK_ROOT/mahomes_outputs/test"

run_logged "03_build_mahomes_inputs_split30_fold0_full" \
  "$PYTHON_BIN" CLEAN_prepare_training_and_test_set/clean_prepare.py build-mahomes-inputs \
    --identity "$CLEAN_IDENTITY" \
    --fold "$CLEAN_FOLD" \
    --work-root "$WORK_ROOT" \
    --splits train test \
    --min-alphafill-identity 0.30 \
    --min-alignment-length 85 \
    --site-dedup-distance 1.0 \
    --uniprot-metal-policy require_supported
