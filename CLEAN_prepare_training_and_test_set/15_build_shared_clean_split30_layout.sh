#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_common_split30_fold0.sh"

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CLEAN_SHARED_WORK_ROOT="${CLEAN_SHARED_WORK_ROOT:-/media/Data/clean_sets/split30/shared}"
export LOG_DIR="${CLEAN_SHARED_LOG_DIR:-$CLEAN_SHARED_WORK_ROOT/logs}"
export CLEAN_SHARED_ROOT="${CLEAN_SHARED_ROOT:-$PROJECT_ROOT/DeepMzyme_Data/CLEAN_${CLEAN_IDENTITY}_shared}"
mkdir -p "$LOG_DIR"

run_logged "15_build_shared_clean_split${CLEAN_IDENTITY}_layout" bash -c '
  set -euo pipefail
  "$PYTHON_BIN" CLEAN_prepare_training_and_test_set/clean_prepare.py build-shared-fold-layout \
    --identity "$CLEAN_IDENTITY" \
    --source-base "$PROJECT_ROOT/DeepMzyme_Data" \
    --output-root "$CLEAN_SHARED_ROOT" \
    --overwrite \
    --link-mode hardlink

  "$PYTHON_BIN" - <<PY
from pathlib import Path
from training.runtime_preparation import (
    discover_missing_esm_embeddings,
    discover_missing_ring_edges,
    discover_missing_updated_external_features,
)
root = Path("$CLEAN_SHARED_ROOT")
structures = sorted((root / "structures").glob("*.pdb"))
missing_esm = discover_missing_esm_embeddings(structures, Path("$PROJECT_ROOT/DeepMzyme_Data/esm_embeddings"))
missing_ring = discover_missing_ring_edges(structures, Path("$PROJECT_ROOT/DeepMzyme_Data/RING_features"))
missing_external = discover_missing_updated_external_features(
    structures,
    structure_root=root / "structures",
    external_features_root_dir=Path("$PROJECT_ROOT/DeepMzyme_Data/updated_feature_extraction"),
)
print(f"[COVERAGE] shared_structures={len(structures)} missing_esm={len(missing_esm)} missing_ring={len(missing_ring)} missing_external={len(missing_external)}")
if missing_esm or missing_ring or missing_external:
    raise SystemExit("Shared CLEAN feature coverage is incomplete.")
PY
'
