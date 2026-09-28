#!/bin/bash
set -euo pipefail
cd /home/mechti/projects/DeepMzyme
PMM_DIAG=/home/mechti/deepmzyme_runs/pmm_ion_metal_v2_context/runtime/replay_diagnostic_20260928
PMM_PYTHON=/home/mechti/venvs/deepmzyme/bin/python
export MKL_THREADING_LAYER=GNU
test "$#" -eq 1
PMM_REMAINING=$(( $1 - $(date -u +%s) ))
# No parsing: 360-second recovery/evaluation forecast, 1.25 margin +900s closeout.
test "$PMM_REMAINING" -ge 1350
date -u +%FT%TZ > "$PMM_DIAG/recovery_worker_started.txt"
"$PMM_PYTHON" -u audit_pmm_replay.py recover \
  --failed-manifest "$PMM_DIAG/prepared/input_manifest.json" \
  --output-dir "$PMM_DIAG/prepared_recovered"
PMM_FAILED=0
for PMM_CONDITION in original strict; do
  for PMM_PROCESS in 1 2; do
    "$PMM_PYTHON" -u audit_pmm_replay.py evaluate \
      --prepared-dir "$PMM_DIAG/prepared_recovered" \
      --output-dir "$PMM_DIAG/${PMM_CONDITION}_${PMM_PROCESS}" \
      --condition "$PMM_CONDITION" --device cuda --repeats 5 || PMM_FAILED=1
  done
done
date -u +%FT%TZ > "$PMM_DIAG/recovery_worker_finished.txt"
exit "$PMM_FAILED"
