#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_common_split30_fold0.sh"

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CLEAN_BUNDLE_DIR="${CLEAN_BUNDLE_DIR:-$WORK_ROOT/bundles}"
export CLEAN_BUNDLE_NAME="${CLEAN_BUNDLE_NAME:-DeepMzyme_Data_v6_clean${CLEAN_IDENTITY}_split${CLEAN_FOLD}_full_esm.tar.zst}"
export CLEAN_BUNDLE_OUTPUT="${CLEAN_BUNDLE_OUTPUT:-$CLEAN_BUNDLE_DIR/$CLEAN_BUNDLE_NAME}"

run_logged "11_build_colab_bundle_split${CLEAN_IDENTITY}_fold${CLEAN_FOLD}" bash -c '
  set -euo pipefail
  mkdir -p "$(dirname "$CLEAN_BUNDLE_OUTPUT")"
  "$PYTHON_BIN" src/build_colab_bundle.py \
    --dataset-root "DeepMzyme_Data/CLEAN_${CLEAN_IDENTITY}_train_test_split_${CLEAN_FOLD}" \
    --include-esm-embeddings \
    --output-bundle "$CLEAN_BUNDLE_OUTPUT"
  sha256sum "$CLEAN_BUNDLE_OUTPUT" | tee "$CLEAN_BUNDLE_OUTPUT.sha256"
  ls -lh "$CLEAN_BUNDLE_OUTPUT" "$CLEAN_BUNDLE_OUTPUT.sha256"
'
