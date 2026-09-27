#!/bin/bash
# One existing runner unit followed by a verified host-pull acknowledgment.
set -euo pipefail
if [[ $# -ne 5 ]]; then
  echo 'Usage: pmm_run_one.sh FAMILY TARGET READOUT FOLD ESTIMATED_SECONDS' >&2
  exit 2
fi
case "$1" in only_gvp|only_esm|gvp_late_fusion) ;; *) exit 2;; esac
case "$2" in four_class|six_class) ;; *) exit 2;; esac
case "$3" in none|first_shell_bias) ;; *) exit 2;; esac
[[ "$4" =~ ^[0-4]$ && "$5" =~ ^[0-9]+$ ]] || exit 2
export CLOUDSDK_PYTHON=/home/mechti/miniconda3/envs/DeepMzyme/bin/python
export CLOUDSDK_PYTHON_SITEPACKAGES=1
PMM_LOCAL=/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v2_context
PMM_REMOTE=/home/mechti/deepmzyme_runs/pmm_ion_metal_v2_context
PMM_TRAIN=/home/mechti/deepmzyme_data/pmm/train_and_test_sets_structures_zenodo_pmm_exact/train
# Read the new allocation; never reuse expired timestamps from the old VM.
readarray -t PMM_SESSION_VALUES < <(/home/mechti/miniconda3/envs/DeepMzyme/bin/python - <<'PY'
import datetime, json
from pathlib import Path
state = json.loads(Path('/home/mechti/deepmzyme-vm/state/current_session.json').read_text())
config = dict(line.split('=', 1) for line in
              Path('/home/mechti/deepmzyme-vm/config.env').read_text().splitlines()
              if line and not line.startswith('#') and '=' in line)
assert state.get('active')
assert all(str(state[key]) == config[setting] for key, setting in (
    ('project', 'PROJECT_ID'), ('zone', 'ZONE'), ('vm_name', 'VM_NAME'),
    ('instance_id', 'SELECTED_INSTANCE_ID')))
stamp = lambda value: int(datetime.datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp())
assert stamp(state['session_start']) <= datetime.datetime.now(datetime.timezone.utc).timestamp()
assert stamp(state['termination_ts']) > datetime.datetime.now(datetime.timezone.utc).timestamp()
print(state['session_id'])
print(stamp(state['session_start']))
print(stamp(state['termination_ts']))
print(state['max_run_duration_s'])
PY
)
[[ ${#PMM_SESSION_VALUES[@]} -eq 4 ]] || exit 2
PMM_SESSION=${PMM_SESSION_VALUES[0]}
PMM_START=${PMM_SESSION_VALUES[1]}
PMM_STOP=${PMM_SESSION_VALUES[2]}
PMM_SECONDS=${PMM_SESSION_VALUES[3]}
ssh -n -F /home/mechti/deepmzyme-vm/state/ssh_config deepmzyme-vm \
  "cd /home/mechti/projects/DeepMzyme && /home/mechti/venvs/deepmzyme/bin/python -u scripts/run_metal_5fold_cv.py \
    --campaign-dir $PMM_REMOTE --train-dir $PMM_TRAIN --campaign-action run \
    --device cuda --load-workers 4 --families $1 --targets $2 --readouts $3 --folds $4 \
    --status-tag ${1}__${2}__${3}__fold${4}__seed42 \
    --session-id $PMM_SESSION --allocation-started $PMM_START \
    --execution-deadline $PMM_STOP --execution-max-seconds $PMM_SECONDS \
    --estimated-fit-seconds $5 --durable-root $PMM_LOCAL --persistence-mode host_pull"
/home/mechti/miniconda3/envs/DeepMzyme/bin/python \
  "$PMM_LOCAL/runtime/pmm_host_pull.py" --remote-root "$PMM_REMOTE"
