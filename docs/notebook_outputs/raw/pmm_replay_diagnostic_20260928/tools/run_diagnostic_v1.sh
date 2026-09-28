#!/bin/bash
set -euo pipefail
cd /home/mechti/projects/DeepMzyme
PMM_DIAG=/home/mechti/deepmzyme_runs/pmm_ion_metal_v2_context/runtime/replay_diagnostic_20260928
PMM_CAMPAIGN=/home/mechti/deepmzyme_runs/pmm_ion_metal_v2_context
PMM_PYTHON=/home/mechti/venvs/deepmzyme/bin/python
export MKL_THREADING_LAYER=GNU
export DEEPMZYME_PARSE_CACHE_DIR="$PMM_CAMPAIGN/parse_cache"
test "$#" -eq 1
PMM_REMAINING=$(( $1 - $(date -u +%s) ))
# 900-second execution forecast, existing 1.25 margin and 900-second closeout reserve.
test "$PMM_REMAINING" -ge 2025
date -u +%FT%TZ > "$PMM_DIAG/worker_started.txt"
"$PMM_PYTHON" -u audit_pmm_replay.py prepare \
  --run-dir "$PMM_CAMPAIGN/runs/only_gvp__six_class__none__fold0__seed42" \
  --cache-namespace "$PMM_CAMPAIGN/raw_graph_cache/117d3f2d78b3429930bcc78491821c7352c89d5e488b7df0ab3715b32c5c0f0f" \
  --cache-after 2026-09-27T18:14:00Z --cache-before 2026-09-27T18:54:00Z \
  --allow-ec-metadata-differences --output-dir "$PMM_DIAG/prepared"
PMM_FAILED=0
for PMM_CONDITION in original strict; do
  for PMM_PROCESS in 1 2; do
    "$PMM_PYTHON" -u audit_pmm_replay.py evaluate \
      --prepared-dir "$PMM_DIAG/prepared" \
      --output-dir "$PMM_DIAG/${PMM_CONDITION}_${PMM_PROCESS}" \
      --condition "$PMM_CONDITION" --device cuda --repeats 5 || PMM_FAILED=1
  done
done
date -u +%FT%TZ > "$PMM_DIAG/worker_finished.txt"
exit "$PMM_FAILED"
