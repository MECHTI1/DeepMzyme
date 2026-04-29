#!/usr/bin/env bash
# IMPORTANT:
# This script is for MAHOMES held-out test-set runs using the edited
# `pdb_updatedv2` structures as input.
# Note that `CLEAN_JOB_DIRS=1` and `SKIP_COMPLETED_JOBS=0` are the safe
# settings for a fresh rerun after source structures changed, because they
# prevent reuse of stale completed MAHOMES outputs.
# Use `CLEAN_JOB_DIRS=0` and `SKIP_COMPLETED_JOBS=1` only to resume an
# interrupted run when the source PDB files are unchanged.
# Save MAHOMES predictions for the held-out test-set structures.
set -euo pipefail

### SETTINGS ###
job_root="${JOB_ROOT:-/media/Data/pinmymetal_sets/mahomes/test_set}"
pdb_dir="${PDB_DIR:-/media/Data/pinmymetal_sets/test/pdb_updatedv2}"
N_JOBS="${N_JOBS:-4}"
MAHOMES_DIR="${MAHOMES_DIR:-/home/mechti/MAHOMES-II}"
VENV="${VENV:-$MAHOMES_DIR/venv/bin/activate}"
CLEAN_JOB_DIRS="${CLEAN_JOB_DIRS:-1}"
SKIP_COMPLETED_JOBS="${SKIP_COMPLETED_JOBS:-0}"
RUN_MODE="${RUN_MODE:-all}"
pdb_source_marker="$job_root/pdb_source_dir.txt"
pdbids_query_txt="$job_root/pdbids_query.txt"
################

if [[ ! -d "$pdb_dir" ]]; then
    echo "[ERROR] PDB directory not found: $pdb_dir"
    exit 1
fi

if [[ ! -d "$MAHOMES_DIR" ]]; then
    echo "[ERROR] MAHOMES directory not found: $MAHOMES_DIR"
    exit 1
fi

if [[ ! -f "$VENV" ]]; then
    echo "[ERROR] MAHOMES virtualenv activate script not found: $VENV"
    exit 1
fi

case "$RUN_MODE" in
    all|last|features|predict) ;;
    *)
        echo "[ERROR] Invalid RUN_MODE: $RUN_MODE"
        echo "Allowed values: all | last | features | predict"
        exit 1
        ;;
esac

if [[ "$CLEAN_JOB_DIRS" != "0" && "$CLEAN_JOB_DIRS" != "1" ]]; then
    echo "[ERROR] CLEAN_JOB_DIRS must be 0 or 1, got: $CLEAN_JOB_DIRS"
    exit 1
fi

if [[ "$SKIP_COMPLETED_JOBS" != "0" && "$SKIP_COMPLETED_JOBS" != "1" ]]; then
    echo "[ERROR] SKIP_COMPLETED_JOBS must be 0 or 1, got: $SKIP_COMPLETED_JOBS"
    exit 1
fi

mkdir -p "$job_root"
printf '%s\n' "$pdb_dir" > "$pdb_source_marker"

job_log_indicates_completion() {
    local dir_path="$1"
    local job_idx="$2"
    local job_log_path="$dir_path/job.log"

    [[ -f "$job_log_path" ]] || return 1
    grep -Fq "Finished job $job_idx" "$job_log_path"
}

job_matches_current_batch() {
    local dir_path="$1"
    local current_part_file="$2"
    local batch_input_path="$dir_path/batch_input.txt"

    [[ -f "$batch_input_path" ]] || return 1
    cmp -s "$current_part_file" "$batch_input_path"
}

job_predictions_fresh_for_current_sources() {
    local dir_path="$1"
    local current_part_file="$2"
    local current_pdb_dir="$3"
    local predictions_path="$dir_path/predictions.csv"

    [[ -f "$predictions_path" ]] || return 1

    python - <<'PY' "$predictions_path" "$current_part_file" "$current_pdb_dir"
from __future__ import annotations

from pathlib import Path
import sys

predictions_path = Path(sys.argv[1])
batch_input_path = Path(sys.argv[2])
pdb_dir = Path(sys.argv[3])
prediction_mtime = predictions_path.stat().st_mtime

with batch_input_path.open("r", encoding="utf-8", errors="replace") as handle:
    for raw_line in handle:
        struct_id = raw_line.rstrip()
        if not struct_id:
            continue
        pdb_path = pdb_dir / f"{struct_id}.pdb"
        if not pdb_path.exists():
            continue
        if pdb_path.stat().st_mtime > prediction_mtime:
            raise SystemExit(1)

raise SystemExit(0)
PY
}

echo "[INFO] PDB dir:        $pdb_dir"
echo "[INFO] Job root:       $job_root"
echo "[INFO] N_JOBS:         $N_JOBS"
echo "[INFO] RUN_MODE:       $RUN_MODE"
echo "[INFO] CLEAN_JOB_DIRS: $CLEAN_JOB_DIRS"
echo "[INFO] SKIP_COMPLETED_JOBS: $SKIP_COMPLETED_JOBS"
echo "[INFO] PDB source tag: $pdb_source_marker"

if [[ "$CLEAN_JOB_DIRS" == "1" && "$SKIP_COMPLETED_JOBS" == "0" ]]; then
    echo "[WARN] Existing job_* directories will be deleted and recreated before rerun."
elif [[ "$SKIP_COMPLETED_JOBS" == "1" ]]; then
    echo "[INFO] Existing completed job_* directories with matching batch_input.txt will be preserved and skipped."
else
    echo "[INFO] Existing job_* directories will be preserved so incomplete jobs can resume in place."
fi

find "$pdb_dir" -maxdepth 1 -type f -name "*.pdb" -printf '%f\n' \
    | sed 's/\.pdb$//' \
    | sort -u > "$pdbids_query_txt"

if [[ ! -s "$pdbids_query_txt" ]]; then
    echo "[ERROR] No .pdb files found in: $pdb_dir"
    exit 1
fi

echo "[INFO] Auto-generated IDs file: $pdbids_query_txt ($(wc -l < "$pdbids_query_txt") IDs)"

rm -f "$job_root"/batch_input_part_* 2>/dev/null || true
split -d -n "l/$N_JOBS" "$pdbids_query_txt" "$job_root/batch_input_part_"

job_index=0
declare -a pids
skipped_jobs=0

for part_file in "$job_root"/batch_input_part_*; do
    job_dir="$job_root/job_$job_index"

    if [[ "$SKIP_COMPLETED_JOBS" == "1" ]] \
        && [[ -d "$job_dir" ]] \
        && job_matches_current_batch "$job_dir" "$part_file" \
        && job_log_indicates_completion "$job_dir" "$job_index" \
        && job_predictions_fresh_for_current_sources "$job_dir" "$part_file" "$pdb_dir"; then
        echo "[SKIP] Job $job_index already finished according to job.log for current batch -> $job_dir"
        skipped_jobs=$((skipped_jobs + 1))
        job_index=$((job_index + 1))
        continue
    fi

    if [[ "$CLEAN_JOB_DIRS" == "1" ]]; then
        rm -rf "$job_dir"
    fi
    mkdir -p "$job_dir"

    (
        exec >> "$job_dir/job.log" 2>&1

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting job $job_index (PID $$)"
        echo "[INFO] Processing files from: $part_file"

        copied=0
        missing=0
        skipped_copy=0

        while IFS= read -r struct_id_raw; do
            struct_id="$(printf '%s' "$struct_id_raw" | sed 's/[[:space:]]*$//')"
            [[ -z "$struct_id" ]] && continue

            pdb_file="$pdb_dir/${struct_id}.pdb"
            target_pdb="$job_dir/${struct_id}.pdb"

            if [[ -f "$target_pdb" ]]; then
                echo "[SKIP COPY] PDB already present for ID: '$struct_id'"
                skipped_copy=$((skipped_copy + 1))
            elif [[ -f "$pdb_file" ]]; then
                cp "$pdb_file" "$target_pdb"
                copied=$((copied + 1))
            else
                echo "  [WARN] PDB not found for ID: '$struct_id' (raw: '$struct_id_raw') in $pdb_dir"
                missing=$((missing + 1))
            fi
        done < "$part_file"

        echo "[INFO] Copied $copied PDBs for this job; $skipped_copy copies skipped; $missing IDs missing PDBs."

        cp "$part_file" "$job_dir/batch_input.txt"

        source "$VENV"
        bash "$MAHOMES_DIR/driver.sh" "$job_dir" "$RUN_MODE"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished job $job_index"
    ) &

    pids[$job_index]=$!
    echo "[LAUNCHED] Job $job_index (PID ${pids[$job_index]}) -> Log: $job_dir/job.log"

    job_index=$((job_index + 1))
done

echo ""
echo "============================================"
echo "All $job_index jobs launched!"
echo "Skipped completed jobs: $skipped_jobs"
echo "PIDs: ${pids[*]}"
echo "============================================"
echo ""
echo "Monitor with: watch -n 2 'tail -n 3 $job_root/job_*/job.log'"

wait

echo ""
echo "[DONE] All parallel MAHOMES test jobs finished at $(date '+%Y-%m-%d %H:%M:%S')"
