#!/usr/bin/env bash
# Shared settings for the CARE Task 1 <30% identity preparation scripts.
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/home/mechti/PycharmProjects/DeepMzyme}"
export PYTHON_BIN="${PYTHON_BIN:-/home/mechti/miniconda3/envs/DeepMzyme/bin/python}"

export CARE_ROOT="${CARE_ROOT:-$PROJECT_ROOT/DeepMzyme_Data/CARE_dataset}"

# Optional explicit overrides. Leave empty by default and let care_prepare.py discover them.
export CARE_TRAIN_CSV="${CARE_TRAIN_CSV:-}"
export CARE_TEST_CSV="${CARE_TEST_CSV:-}"

export CARE_TASK="${CARE_TASK:-task1}"
export CARE_SPLIT_NAME="${CARE_SPLIT_NAME:-30_identity}"

export WORK_ROOT="${WORK_ROOT:-/media/Data/care_sets/task1_30}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/DeepMzyme_Data/CARE_task1_30_train_test_metallo}"

export LOG_DIR="${LOG_DIR:-$WORK_ROOT/logs}"
export MAHOMES_N_JOBS="${MAHOMES_N_JOBS:-4}"
export CARE_UNIPROT_BATCH_SIZE="${CARE_UNIPROT_BATCH_SIZE:-200}"
export CARE_ALPHAFILL_N_JOBS="${CARE_ALPHAFILL_N_JOBS:-1}"
export CARE_ALPHAFILL_TIMEOUT="${CARE_ALPHAFILL_TIMEOUT:-90}"
export CARE_ALPHAFILL_RETRIES="${CARE_ALPHAFILL_RETRIES:-4}"
export CARE_ALPHAFILL_SLEEP_SECONDS="${CARE_ALPHAFILL_SLEEP_SECONDS:-0.5}"

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
