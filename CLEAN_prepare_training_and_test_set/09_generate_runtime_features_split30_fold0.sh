#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_common_split30_fold0.sh"

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export RING_EXE_PATH="${RING_EXE_PATH:-$PROJECT_ROOT/DeepMzyme_Data/ring-4.0/out/bin/ring}"
export RING_EDGE_JOBS="${RING_EDGE_JOBS:-4}"
export CLEAN_EXTERNAL_FEATURE_JOBS="${CLEAN_EXTERNAL_FEATURE_JOBS:-4}"

export STRUCTURE_LIST="$WORK_ROOT/manifests/clean_split${CLEAN_IDENTITY}_fold${CLEAN_FOLD}_exported_structures.txt"

run_logged "09_generate_runtime_features_split${CLEAN_IDENTITY}_fold${CLEAN_FOLD}" bash -c '
  set -euo pipefail
  mkdir -p "$(dirname "$STRUCTURE_LIST")"
  find "$PROJECT_ROOT/DeepMzyme_Data/CLEAN_${CLEAN_IDENTITY}_train_test_split_${CLEAN_FOLD}/train" \
       "$PROJECT_ROOT/DeepMzyme_Data/CLEAN_${CLEAN_IDENTITY}_train_test_split_${CLEAN_FOLD}/test" \
       -maxdepth 1 -type f -name "*.pdb" | sort > "$STRUCTURE_LIST"
  echo "[INFO] structures=$(wc -l < "$STRUCTURE_LIST")"
  echo "[INFO] RING_EXE_PATH=$RING_EXE_PATH"
  echo "[INFO] RING_EDGE_JOBS=$RING_EDGE_JOBS"
  echo "[INFO] CLEAN_EXTERNAL_FEATURE_JOBS=$CLEAN_EXTERNAL_FEATURE_JOBS"

  echo "[INFO] generating updated external features for train"
  "$PYTHON_BIN" src/feature_extraction/generate_features.py \
    --structure-dir "$PROJECT_ROOT/DeepMzyme_Data/CLEAN_${CLEAN_IDENTITY}_train_test_split_${CLEAN_FOLD}/train" \
    --output-root "$PROJECT_ROOT/DeepMzyme_Data/updated_feature_extraction" \
    --skip-existing \
    --jobs "$CLEAN_EXTERNAL_FEATURE_JOBS"

  echo "[INFO] generating updated external features for test"
  "$PYTHON_BIN" src/feature_extraction/generate_features.py \
    --structure-dir "$PROJECT_ROOT/DeepMzyme_Data/CLEAN_${CLEAN_IDENTITY}_train_test_split_${CLEAN_FOLD}/test" \
    --output-root "$PROJECT_ROOT/DeepMzyme_Data/updated_feature_extraction" \
    --skip-existing \
    --jobs "$CLEAN_EXTERNAL_FEATURE_JOBS"

  echo "[INFO] generating missing RING edge files"
  "$PYTHON_BIN" - <<PY
from pathlib import Path
from embed_helpers.Interaction_edge import create_ring_edges_batch
from training.runtime_preparation import discover_missing_ring_edges

structure_list = Path("$STRUCTURE_LIST")
structures = [Path(line.strip()) for line in structure_list.read_text(encoding="utf-8").splitlines() if line.strip()]
ring_dir = Path("$PROJECT_ROOT/DeepMzyme_Data/RING_features")
missing = discover_missing_ring_edges(structures, ring_dir)
print(f"[INFO] missing RING before={len(missing)}")
summary = create_ring_edges_batch(missing, dir_results=ring_dir, overwrite=False, jobs=int("$RING_EDGE_JOBS"))
print(summary)
if summary.get("failed_structures"):
    raise SystemExit(f"RING generation failed for {len(summary['failed_structures'])} structures")
PY

  echo "[INFO] generating missing ESMC embeddings"
  "$PYTHON_BIN" src/embed_helpers/esmc.py \
    --structure-list "$STRUCTURE_LIST" \
    --out-dir "$PROJECT_ROOT/DeepMzyme_Data/esm_embeddings"

  echo "[INFO] final coverage check"
  "$PYTHON_BIN" - <<PY
from pathlib import Path
from training.runtime_preparation import (
    discover_missing_esm_embeddings,
    discover_missing_ring_edges,
    discover_missing_updated_external_features,
)

root = Path("$PROJECT_ROOT/DeepMzyme_Data/CLEAN_${CLEAN_IDENTITY}_train_test_split_${CLEAN_FOLD}")
structures = sorted((root / "train").glob("*.pdb")) + sorted((root / "test").glob("*.pdb"))
esm_dir = Path("$PROJECT_ROOT/DeepMzyme_Data/esm_embeddings")
ring_dir = Path("$PROJECT_ROOT/DeepMzyme_Data/RING_features")
external_dir = Path("$PROJECT_ROOT/DeepMzyme_Data/updated_feature_extraction")
missing_esm = discover_missing_esm_embeddings(structures, esm_dir)
missing_ring = discover_missing_ring_edges(structures, ring_dir)
missing_external = discover_missing_updated_external_features(
    structures,
    structure_root=root,
    external_features_root_dir=external_dir,
)
print(f"[COVERAGE] structures={len(structures)} missing_esm={len(missing_esm)} missing_ring={len(missing_ring)} missing_external={len(missing_external)}")
if missing_esm or missing_ring or missing_external:
    raise SystemExit("Runtime feature coverage is incomplete.")
PY
'
