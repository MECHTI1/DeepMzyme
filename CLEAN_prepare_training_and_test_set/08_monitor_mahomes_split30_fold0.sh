#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_common_split30_fold0.sh"

for split in train test; do
    job_root="$WORK_ROOT/mahomes/$split"
    input_root="$WORK_ROOT/mahomes_inputs/$split"
    echo "=== $split ==="
    if [[ -d "$input_root" ]]; then
        echo "input_pdbs=$(find "$input_root" -maxdepth 1 -type f -name '*.pdb' | wc -l)"
    fi
    if [[ ! -d "$job_root" ]]; then
        echo "job_root_missing=$job_root"
        continue
    fi
    echo "job_root_size=$(du -sh "$job_root" | awk '{print $1}')"
    echo "prediction_files=$(find "$job_root" -maxdepth 2 -type f -name predictions.csv | wc -l)"
    for log in "$job_root"/job_*/job.log; do
        [[ -f "$log" ]] || continue
        job_name="$(basename "$(dirname "$log")")"
        finished_tools="$(grep -c 'Finished third-party tools' "$log" || true)"
        timed_structures="$(grep -c ' time:' "$log" || true)"
        completed_jobs="$(grep -c 'Finished job' "$log" || true)"
        echo "$job_name finished_tools=$finished_tools timed_structures=$timed_structures completed_job_markers=$completed_jobs"
        tail -n 3 "$log" | sed "s/^/  tail: /"
    done
done

df -h /media/Data
