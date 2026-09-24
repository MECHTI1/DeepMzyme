#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    VENV_PYTHON="python3"
fi

RUNS_DIR="/media/mechti/Data1/DeepMzyme_Data/runs_zenodo_pmm_exact"
if [ ! -d "/media/mechti/Data1" ]; then
    RUNS_DIR="${REPO_ROOT}/runs/runs_zenodo_pmm_exact"
fi

mkdir -p "$RUNS_DIR"

echo "=== DeepMzyme Zenodo PinMyMetal Exact 5-Fold Benchmark Runner ==="
echo "Python binary: $VENV_PYTHON"
echo "Runs directory: $RUNS_DIR"

exec "$VENV_PYTHON" "${SCRIPT_DIR}/run_zenodo_pmm_exact_5fold_cv.py" \
    --runs-dir "$RUNS_DIR" \
    "$@"
