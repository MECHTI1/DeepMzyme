#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_common_split30_fold0.sh"

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

run_logged "10_validate_colab_inputs_split${CLEAN_IDENTITY}_fold${CLEAN_FOLD}" \
  "$PYTHON_BIN" src/build_colab_bundle.py \
    --dataset-root "DeepMzyme_Data/CLEAN_${CLEAN_IDENTITY}_train_test_split_${CLEAN_FOLD}" \
    --include-esm-embeddings \
    --skip-bundle
