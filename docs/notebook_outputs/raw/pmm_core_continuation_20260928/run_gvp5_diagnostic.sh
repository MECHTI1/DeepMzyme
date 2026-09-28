#!/usr/bin/env bash
set -euo pipefail
PMM_PY=/home/mechti/venvs/deepmzyme/bin/python
PMM_REPO=/home/mechti/projects/DeepMzyme
PMM_CAMPAIGN=/home/mechti/deepmzyme_runs/pmm_ion_metal_v2_context
PMM_BASE="$PMM_CAMPAIGN/runtime/core_continuation_20260928"
PMM_DIAG="$PMM_BASE/gvp5_diagnostic"
cd "$PMM_REPO"
mkdir "$PMM_DIAG"
cp "$PMM_REPO/audit_pmm_replay.py" "$PMM_DIAG/audit_pmm_replay.py"
cp "$PMM_BASE/gvp5_diagnostic_protocol.md" "$PMM_DIAG/protocol.md"
cp "$PMM_BASE/run_gvp5_diagnostic.sh" "$PMM_DIAG/run_diagnostic.sh"
PMM_PREPARED="$PMM_DIAG/prepared"
if ! "$PMM_PY" -u audit_pmm_replay.py prepare \
  --run-dir "$PMM_CAMPAIGN/runs/only_gvp__five_class__none__fold0__seed42" \
  --cache-namespace "$PMM_CAMPAIGN/raw_graph_cache/7e9402a3ec2149bb4a722fe58999ae4d6309c87b4a36f6986184ee7b01b2785b" \
  --cache-after 2026-09-28T03:32:30Z --cache-before 2026-09-28T04:13:33Z \
  --output-dir "$PMM_PREPARED"; then
  cp "$PMM_PREPARED/input_manifest.json" "$PMM_DIAG/failed_input_manifest.json"
  "$PMM_PY" -u audit_pmm_replay.py recover \
    --failed-manifest "$PMM_PREPARED/input_manifest.json" \
    --output-dir "$PMM_DIAG/prepared_recovered"
  PMM_PREPARED="$PMM_DIAG/prepared_recovered"
fi
for PMM_CONDITION in original strict; do
  for PMM_PROCESS in 1 2; do
    "$PMM_PY" -u audit_pmm_replay.py evaluate --prepared-dir "$PMM_PREPARED" \
      --output-dir "$PMM_DIAG/${PMM_CONDITION}_${PMM_PROCESS}" \
      --condition "$PMM_CONDITION" --device cuda --repeats 5
  done
done
printf '%s\n' "$PMM_PREPARED" > "$PMM_DIAG/completed_preparation_path.txt"
