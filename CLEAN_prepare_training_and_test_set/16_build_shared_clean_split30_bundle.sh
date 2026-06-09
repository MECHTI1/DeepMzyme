#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_common_split30_fold0.sh"

export CLEAN_SHARED_WORK_ROOT="${CLEAN_SHARED_WORK_ROOT:-/media/Data/clean_sets/split30/shared}"
export LOG_DIR="${CLEAN_SHARED_LOG_DIR:-$CLEAN_SHARED_WORK_ROOT/logs}"
export CLEAN_SHARED_ROOT="${CLEAN_SHARED_ROOT:-$PROJECT_ROOT/DeepMzyme_Data/CLEAN_${CLEAN_IDENTITY}_shared}"
export CLEAN_SHARED_BUNDLE_DIR="${CLEAN_SHARED_BUNDLE_DIR:-$CLEAN_SHARED_WORK_ROOT/bundles}"
export CLEAN_SHARED_BUNDLE_NAME="${CLEAN_SHARED_BUNDLE_NAME:-DeepMzyme_Data_v8_clean${CLEAN_IDENTITY}_shared_full_esm.tar.zst}"
export CLEAN_SHARED_BUNDLE_OUTPUT="${CLEAN_SHARED_BUNDLE_OUTPUT:-$CLEAN_SHARED_BUNDLE_DIR/$CLEAN_SHARED_BUNDLE_NAME}"
mkdir -p "$LOG_DIR" "$CLEAN_SHARED_BUNDLE_DIR"

run_logged "16_build_shared_clean_split${CLEAN_IDENTITY}_bundle" bash -c '
  set -euo pipefail
  if ! command -v zstd >/dev/null 2>&1; then
    echo "zstd is required to build the bundle" >&2
    exit 1
  fi
  test -d "$CLEAN_SHARED_ROOT"
  tar --use-compress-program="zstd -T0 -19" \
    -cf "$CLEAN_SHARED_BUNDLE_OUTPUT" \
    -C "$PROJECT_ROOT" \
    "DeepMzyme_Data/CLEAN_${CLEAN_IDENTITY}_shared" \
    DeepMzyme_Data/updated_feature_extraction \
    DeepMzyme_Data/RING_features \
    DeepMzyme_Data/ring-4.0 \
    DeepMzyme_Data/esm_embeddings
  sha256sum "$CLEAN_SHARED_BUNDLE_OUTPUT" | tee "$CLEAN_SHARED_BUNDLE_OUTPUT.sha256"
  ls -lh "$CLEAN_SHARED_BUNDLE_OUTPUT" "$CLEAN_SHARED_BUNDLE_OUTPUT.sha256"
'
