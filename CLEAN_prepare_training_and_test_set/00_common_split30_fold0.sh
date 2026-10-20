#!/usr/bin/env bash
# Shared settings for the CLEAN split30/fold0 preparation scripts.
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Default interpreter: the repo .venv (uv lock, CPU-only torch) if present, else python3.
# Export PYTHON_BIN to use another environment, e.g. a GPU env for the ESM C steps.
if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"; else PYTHON_BIN="python3"; fi
fi
export PYTHON_BIN
export CLEAN_IDENTITY="${CLEAN_IDENTITY:-30}"
export CLEAN_FOLD="${CLEAN_FOLD:-0}"
export WORK_ROOT="${WORK_ROOT:-/media/Data/clean_sets/split30/fold0}"
export LOG_DIR="${LOG_DIR:-$WORK_ROOT/logs}"
export MAHOMES_N_JOBS="${MAHOMES_N_JOBS:-4}"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

run_logged() {
    local step_name="$1"
    shift
    local log_path="$LOG_DIR/${step_name}.log"
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') starting $step_name"
    echo "[INFO] log: $log_path"
    "$@" 2>&1 | tee "$log_path"
    local status=${PIPESTATUS[0]}
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') finished $step_name with status $status"
    return "$status"
}
