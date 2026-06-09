#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_common_care_task1_30.sh"

for split in train test; do
    job_root="$WORK_ROOT/mahomes/$split"
    input_root="$WORK_ROOT/mahomes_inputs/$split"
    echo "=== CARE $split ==="
    if [[ -d "$input_root" ]]; then
        echo "input_pdbs=$(find "$input_root" -maxdepth 1 -type f -name '*.pdb' | wc -l)"
    else
        echo "input_root_missing=$input_root"
    fi
    if [[ ! -d "$job_root" ]]; then
        echo "job_root_missing=$job_root"
        continue
    fi
    echo "job_directories=$(find "$job_root" -maxdepth 1 -type d -name 'job_*' | wc -l)"
    echo "prediction_files=$(find "$job_root" -maxdepth 2 -type f -name predictions.csv | wc -l)"
    echo "job_root_size=$(du -sh "$job_root" | awk '{print $1}')"
    for log in "$job_root"/job_*/job.log; do
        [[ -f "$log" ]] || continue
        job_name="$(basename "$(dirname "$log")")"
        finished_tools="$(grep -c 'Finished third-party tools' "$log" || true)"
        timed_structures="$(grep -c ' time:' "$log" || true)"
        completed_jobs="$(grep -c 'Finished job' "$log" || true)"
        echo "$job_name finished_tools=$finished_tools timed_structures=$timed_structures completed_job_markers=$completed_jobs"
        tail -n 5 "$log" | sed "s/^/  tail: /"
    done
done

du -sh "$WORK_ROOT" 2>/dev/null || true
df -h /media/Data
