#!/usr/bin/env bash
# Run MAHOMES on CLEAN-derived reduced PDB files without touching PinMyMetal jobs.
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/media/Data/clean_sets/split30/fold0}"
SPLIT="${SPLIT:-train}"
PDB_DIR="${PDB_DIR:-$WORK_ROOT/mahomes_inputs/$SPLIT}"
JOB_ROOT="${JOB_ROOT:-$WORK_ROOT/mahomes/$SPLIT}"
N_JOBS="${N_JOBS:-4}"
MAHOMES_DIR="${MAHOMES_DIR:-/home/mechti/MAHOMES-II}"
VENV="${VENV:-$MAHOMES_DIR/venv/bin/activate}"
CLEAN_JOB_DIRS="${CLEAN_JOB_DIRS:-1}"
SKIP_COMPLETED_JOBS="${SKIP_COMPLETED_JOBS:-0}"
RUN_MODE="${RUN_MODE:-all}"

case "$SPLIT" in
    train|test) ;;
    *) echo "[ERROR] SPLIT must be train or test, got: $SPLIT"; exit 1 ;;
esac
case "$RUN_MODE" in
    all|last|features|predict) ;;
    *) echo "[ERROR] RUN_MODE must be all, last, features, or predict; got: $RUN_MODE"; exit 1 ;;
esac
if [[ "$CLEAN_JOB_DIRS" != "0" && "$CLEAN_JOB_DIRS" != "1" ]]; then
    echo "[ERROR] CLEAN_JOB_DIRS must be 0 or 1, got: $CLEAN_JOB_DIRS"
    exit 1
fi
if [[ "$SKIP_COMPLETED_JOBS" != "0" && "$SKIP_COMPLETED_JOBS" != "1" ]]; then
    echo "[ERROR] SKIP_COMPLETED_JOBS must be 0 or 1, got: $SKIP_COMPLETED_JOBS"
    exit 1
fi
if [[ ! -d "$PDB_DIR" ]]; then
    echo "[ERROR] PDB_DIR not found: $PDB_DIR"
    exit 1
fi
if [[ ! -d "$MAHOMES_DIR" ]]; then
    echo "[ERROR] MAHOMES_DIR not found: $MAHOMES_DIR"
    exit 1
fi
if [[ ! -f "$VENV" ]]; then
    echo "[ERROR] MAHOMES virtualenv not found: $VENV"
    exit 1
fi

mkdir -p "$JOB_ROOT"
pdbids_query_txt="$JOB_ROOT/pdbids_query.txt"
pdb_source_marker="$JOB_ROOT/pdb_source_dir.txt"
printf '%s\n' "$PDB_DIR" > "$pdb_source_marker"

find "$PDB_DIR" -maxdepth 1 -type f -name "*.pdb" -printf '%f\n' \
    | sed 's/\.pdb$//' \
    | sort -u > "$pdbids_query_txt"

if [[ ! -s "$pdbids_query_txt" ]]; then
    echo "[ERROR] No .pdb files found in: $PDB_DIR"
    exit 1
fi

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
        if pdb_path.exists() and pdb_path.stat().st_mtime > prediction_mtime:
            raise SystemExit(1)
raise SystemExit(0)
PY
}

echo "[INFO] WORK_ROOT: $WORK_ROOT"
echo "[INFO] SPLIT:     $SPLIT"
echo "[INFO] PDB_DIR:   $PDB_DIR"
echo "[INFO] JOB_ROOT:  $JOB_ROOT"
echo "[INFO] N_JOBS:    $N_JOBS"
echo "[INFO] RUN_MODE:  $RUN_MODE"
echo "[INFO] IDs:       $(wc -l < "$pdbids_query_txt")"

rm -f "$JOB_ROOT"/batch_input_part_* 2>/dev/null || true
split -d -n "l/$N_JOBS" "$pdbids_query_txt" "$JOB_ROOT/batch_input_part_"

job_index=0
declare -a pids
skipped_jobs=0

for part_file in "$JOB_ROOT"/batch_input_part_*; do
    job_dir="$JOB_ROOT/job_$job_index"
    if [[ "$SKIP_COMPLETED_JOBS" == "1" ]] \
        && [[ -d "$job_dir" ]] \
        && job_matches_current_batch "$job_dir" "$part_file" \
        && job_log_indicates_completion "$job_dir" "$job_index" \
        && job_predictions_fresh_for_current_sources "$job_dir" "$part_file" "$PDB_DIR"; then
        echo "[SKIP] Job $job_index already complete for current batch"
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
        echo "[INFO] Processing IDs from: $part_file"
        copied=0
        missing=0
        while IFS= read -r struct_id_raw; do
            struct_id="$(printf '%s' "$struct_id_raw" | sed 's/[[:space:]]*$//')"
            [[ -z "$struct_id" ]] && continue
            pdb_file="$PDB_DIR/${struct_id}.pdb"
            target_pdb="$job_dir/${struct_id}.pdb"
            if [[ -f "$pdb_file" ]]; then
                cp -f "$pdb_file" "$target_pdb"
                copied=$((copied + 1))
            else
                echo "[WARN] Missing PDB for ID: '$struct_id'"
                missing=$((missing + 1))
            fi
        done < "$part_file"
        echo "[INFO] Copied $copied PDBs; missing $missing."
        cp "$part_file" "$job_dir/batch_input.txt"
        source "$VENV"
        bash "$MAHOMES_DIR/driver.sh" "$job_dir" "$RUN_MODE"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished job $job_index"
    ) &
    pids[$job_index]=$!
    echo "[LAUNCHED] Job $job_index (PID ${pids[$job_index]}) -> $job_dir/job.log"
    job_index=$((job_index + 1))
done

echo "[INFO] Launched jobs: $job_index; skipped completed jobs: $skipped_jobs"
echo "[INFO] Monitor: watch -n 2 'tail -n 3 $JOB_ROOT/job_*/job.log'"
wait

echo "[DONE] CLEAN MAHOMES $SPLIT jobs finished at $(date '+%Y-%m-%d %H:%M:%S')"
