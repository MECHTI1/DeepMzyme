#!/usr/bin/env bash
set -euo pipefail
PMM_REPO=/home/mechti/projects/DeepMzyme
PMM_C=/home/mechti/deepmzyme_runs/pmm_ion_metal_v2_context
PMM_D="$PMM_C/runtime/core_continuation_20260928"
PMM_PY=/home/mechti/venvs/deepmzyme/bin/python
test -f "$PMM_D/gvp5_diagnostic/completed_preparation_path.txt"
test "$(date -u +%s)" -lt 1790575822
# 06:10:22 UTC is the latest start allowing 600 seconds and 900 seconds reserve.
"$PMM_PY" "$PMM_D/reconcile_backup.py" > "$PMM_D/reconciled_backup.json"
cd "$PMM_REPO"
"$PMM_PY" -u benchmark_pmm_gpu_concurrency.py \
  --prepared-dir "$PMM_C/runtime/replay_diagnostic_20260928/prepared_recovered" \
  --checkpoint "$PMM_C/runs/only_gvp__six_class__none__fold0__seed42/best_model_checkpoint.pt" \
  --campaign-root "$PMM_C" --output-dir "$PMM_D/concurrency_probe"
