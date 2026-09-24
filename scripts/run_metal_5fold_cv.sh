#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

CONDA_PYTHON="/home/mechti/miniconda3/envs/DeepMzyme/bin/python"
VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"

if [ -f "$CONDA_PYTHON" ]; then
    PYTHON_BIN="$CONDA_PYTHON"
elif [ -f "$VENV_PYTHON" ]; then
    PYTHON_BIN="$VENV_PYTHON"
else
    PYTHON_BIN="python3"
fi

DATASET="${DATASET:-train_and_test_sets_structures_exact_pinmymetal}"
UNIT="${UNIT:-ion}"
FOLDS="${FOLDS:-0 1 2 3 4}"
EPOCHS="${EPOCHS:-50}"
DEVICE="${DEVICE:-cuda}"

echo "=== DeepMzyme Generalized 5-Fold CV Benchmark Runner ==="
echo "Python binary: $PYTHON_BIN"
echo "Dataset:       $DATASET"
echo "Unit:          $UNIT"
echo "Folds:         $FOLDS"
echo "Epochs:        $EPOCHS"
echo "Device:        $DEVICE"
echo "========================================================"

exec "$PYTHON_BIN" "${SCRIPT_DIR}/run_metal_5fold_cv.py" \
    --dataset "$DATASET" \
    --metal-example-unit "$UNIT" \
    --folds $FOLDS \
    --epochs "$EPOCHS" \
    --device "$DEVICE" \
    --python-bin "$PYTHON_BIN" \
    "$@"
