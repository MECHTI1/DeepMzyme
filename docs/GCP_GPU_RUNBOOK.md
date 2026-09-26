# GCP GPU VM runbook

Agents first read the project
[`gpu-use-skill`](../.agents/skills/gpu-use-skill/SKILL.md), which owns runtime
routing and the bounded capacity pass. Use `~/deepmzyme-vm/bin/*` for all VM
lifecycle actions. Read that controller's
`AGENTS.md` and current `config.env` before operating it. The controller owns
authorization, prices, quotas, daily/session accounting, locking and Google's
automatic STOP. The [PMM runtime guide](GPU_EXECUTION_CASCADE_PLAYBOOK.md) covers
measured admission and persistence on this route or [Colab](COLAB_GPU_RUNBOOK.md);
the [metal playbook](METAL_TRAINING_PIPELINE_PLAYBOOK.md) owns scientific commands.
Current progress belongs in [`EXPERIMENT_STATUS.md`](../EXPERIMENT_STATUS.md).

## Controller and resource contract

The initial machine is `deepmzyme-l4` in project `deepmzyme-gpu-vm`, using
`g2-standard-8`: 8 vCPU, 32 GiB RAM and one NVIDIA L4 with 24 GB GPU memory.
Read `config.env` for the currently selected name and zone after recovery.
Connection uses IAP SSH. Confirm actual hardware after creation.

Read live configuration for limits. Reviewed defaults are 4 hours/$6 per
session, 6 hours/$10 per UTC day, and $1.50/hour maximum verified gross rate.
The controller subtracts stop overhead, so the usual effective session is
237 minutes. Scientific work needs an additional closeout allowance within
that deadline. Costs include compute, IP and persistent storage, including
overlapping source/replacement disks and temporary snapshots during recovery;
credits never widen caps. A stopped VM still incurs storage charges.

```bash
~/deepmzyme-vm/bin/vm-status
~/deepmzyme-vm/bin/vm-report --no-ssh
~/deepmzyme-vm/bin/vm-preflight
~/deepmzyme-vm/bin/vm-price
```

Present the planned hard stop and maximum estimated gross charge before using
existing start authorization. First creation uses
`vm-create --hours H --authorize 'AUTHORIZE VM START'`; authorization must come
from the user, including an applicable earlier explicit instruction. Do not
manufacture authorization or request it again when already provided. An
existing VM starts through `vm-start --confirm` only when authorized. Never
bypass controller refusals, price checks or Google-side STOP.

## Bounded stockout handling

`ZONE_RESOURCE_POOL_EXHAUSTED` is a capacity failure, distinct from insufficient
quota, pricing refusal or authentication failure. Capacity changes; neither a
past failure nor positive quota establishes current availability
([Google documentation](https://docs.cloud.google.com/compute/docs/troubleshooting/troubleshooting-vm-creation)).

The GPU skill owns route selection; the controller enforces recovery limits.
An existing stopped VM uses the same-region recovery below. A fresh allocation
without a managed VM may use the skill's separately authorized cross-region
creation pass. These are distinct procedures. Never delete, rename or unlabel
a source to evade a guard. Reconcile uncertain requests against actual cloud
state before another attempt.

### Recover a stopped VM in the same region

Use `vm-fallback` after capacity exhaustion when the source and every other
managed GPU are positively confirmed `TERMINATED`. Running, starting, stopping,
suspended or unknown GPU states block allocation. The controller locks and
rechecks the inventory immediately before a create. One coordinator operates
the pass; an optional monitor only reads status and reports.

Preview before executing the already authorized recovery. The example uses
one requested hour and a provider-suggested zone; substitute the actual approved
duration and same-region hint. `--preferred-zone` is optional.

```bash
~/deepmzyme-vm/bin/vm-fallback --dry-run --hours 1 --preferred-zone us-central1-c
~/deepmzyme-vm/bin/vm-fallback --confirm --hours 1 --preferred-zone us-central1-c
```

Preview must not create cloud resources or change the selected configuration,
ledger or SSH files. Inspect its candidate order, storage estimate, effective
hard stop and maximum estimated gross cost. Unknown pricing blocks execution.
Authorization for this bounded pass covers its eligible candidates without
repeated per-zone questions; it does not widen spending caps or authorize
cross-region migration or another provider.

The controller snapshots the stopped source boot disk once and restores it into
the replacement zone. It prefers the eligible provider hint, then remaining
same-region L4 machine offerings, excluding the source. A durable receipt binds
the source, snapshot, replacement, original configuration, phase and attempt
history in `state/fallback.json`; the exact original config is saved as
`state/fallback-<pass_id>.config.env`. At most three destination creation attempts
may begin within ten minutes after snapshot preparation. Reconnects and agent changes retain those
limits. Advance only on confirmed capacity exhaustion; diagnose quota, price,
authentication, ownership or configuration failures. A timeout requires
reconciliation of the recorded candidate before retrying.

Provider automatic STOP, managed labels and scheduling safeguards must pass
verification before the controller selects the replacement and refreshes its
guardian and IAP SSH configuration. Session records bind project, zone, name
and immutable instance ID so old stopped resources cannot close a new session.
The active-disk charge is counted once; retained disks and the snapshot remain
chargeable until verified deletion, including after an unsuccessful pass.

Connecting, restoring and mounting are different operations. IAP SSH connects
to the selected VM. Snapshot restoration copies the old zonal boot filesystem
into the new zone; it does not attach the original zonal disk across zones.
A mounted durable destination or verified host-pull copy provides independent
artifact storage. Two paths on the restored boot disk do not provide a backup.
The snapshot/recreate route follows
[Google's zone-migration guidance](https://docs.cloud.google.com/compute/docs/instances/moving-instance-across-zones).

After connecting, verify the interpreter, a relevant CUDA operation, frozen
source/input identities, cached files and independent backup destination. Reuse
valid preparation and completed fits. Bind job admission to the replacement's
actual fresh allocation start and deadline, not the source's expired session.
Use `vm-pycharm-info` to inspect the selected connection; managed SSH changes
preserve unrelated user entries.

Cleanup requires a successful replacement fit, independent checkpoint replay
and host-side backup verification tied to that replacement session. Under
existing cleanup authorization, finalize with its evidence file:

```bash
~/deepmzyme-vm/bin/vm-fallback --finalize --dry-run --evidence /absolute/path/recovery_evidence.json
~/deepmzyme-vm/bin/vm-fallback --finalize --evidence /absolute/path/recovery_evidence.json --confirm
~/deepmzyme-vm/bin/vm-status
```

Use `~/deepmzyme-vm/README.md`'s "Verify and finalize recovery" evidence contract.
It binds the pass and immutable replacement/session IDs, UTC completion time,
completed fit and passed replay artifacts, and the independently verified local
backup's complete file/hash manifest. Fit/replay artifacts must be inside that
backup. The controller rehashes bytes; the coordinating operator verifies the
scientific meaning of those artifacts. A completion claim or an older baseline's
receipts are insufficient. Finalization stops and confirms the replacement
before deleting only the recorded superseded VM, old
disk and temporary snapshot. The working replacement and verified host backup
remain. Incomplete, mismatched or stale evidence preserves the old resources
and their ongoing charges. Ordinary session completion remains stop-only.
Snapshot deletion can retain a billable recycle-bin copy. The controller also
verifies permanent removal of that exact temporary snapshot and records its
separate one-hour minimum charge. It preserves unrelated recoverable snapshots
and does not change project retention settings.

### Fresh allocation in another region

The table below supplies the region mapping for eligible, authorized new
allocations. Virginia has a separately price-supported configuration.

| Candidate | Region | Existing subnet range | Compute SKUs (quantity) | Disk SKU | Reviewed compute $/h |
|---|---|---|---|---|---|
| `us-central1-a` | `us-central1` | `10.42.0.0/24` | `A88A-5A60-E821:1,D18E-3563-415F:8,F0A4-E3D3-33BD:32` | `6AE1-525F-8B80` | `0.853624271` |
| `us-east4-a` | `us-east4` | `10.45.0.0/24` | `1D4C-3419-0297:1,CF07-A027-C390:8,A15F-B56F-97FE:32` | `7D52-6D58-14FF` | `0.850829156` |
| `us-west1-a` | `us-west1` | `10.44.0.0/24` | same Americas SKUs as central | `6AE1-525F-8B80` | `0.853624271` |

These are reference mappings, not permission to skip live price verification.
The IP SKU is `C054-7F72-A02E`. With a 150 GB balanced disk the reviewed gross
rates are approximately $0.87917/hour for central/west and $0.87843/hour for
Virginia; the controller's current Catalog query is authoritative.

Before an authorized region change, save the exact bytes of `config.env`.
Update `ZONE`, `REGION`, `SUBNET_RANGE`, `PRICE_COMPUTE_SKUS`,
`PRICE_DISK_CAPACITY_SKU_ID` and `PRICE_EXPECTED_COMPUTE_USD_PER_HOUR` together.
Check the subnet exists; use `vm-infra --apply` only if an authorized required
subnet is absent. Run `vm-price` and `vm-preflight` for the candidate. Caps,
machine type and provisioning model remain unchanged.

Only a confirmed stockout permits moving to the next candidate. Check cloud
state after an interrupted or ambiguous request before any retry. On success,
retain that configuration while the VM exists. On exhausted/abandoned search,
including exceptions, restore the exact saved file only after confirming no
candidate is active or ambiguous. Do not run a background retry loop or
automatically restart a stopped VM.

## PMM campaign setup: exact source and train-only data

The generic controller's `env`/`data` stages assume a `main` checkout and a v12
data bundle. The PMM campaign uses its reviewed working-tree source archive
and train-only archive instead. Use:

```bash
~/deepmzyme-vm/bin/vm-setup --stages driver,ssh
```

Transfer `code_snapshot.tar.gz`, `train_side.tar.gz`, `campaign_frozen.tar.gz`
and `bundle_manifest.json`, generated and verified by
`src/benchmarking/pmm_campaign_bundle.py`. Preserve hard links, verify archive
hashes and allowed members before extraction, and compare the unpacked
source-tree hash to the manifest. The mixed train/test archive is not a
development input. Keep embeddings, caches and data outside IDE automatic
deployment. Do not overlay new source while reportable fits run.

The controller's actual paths are `~/projects/DeepMzyme`, `~/venvs/deepmzyme`,
`~/deepmzyme_cache/{parse,esm,ring}` and `~/deepmzyme_runs`; read `config.env`
if customized. Stage reviewed source at the project path before installing
requirements. Inside the protected VM, bootstrap the campaign environment
with these commands (the local workstation interpreter is separate):

```bash
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv build-essential rsync jq
mkdir -p "$HOME/venvs" "$HOME/deepmzyme_runs"
python3.12 -m venv "$HOME/venvs/deepmzyme"
PMM_PY="$HOME/venvs/deepmzyme/bin/python"
PMM_REPO="$HOME/projects/DeepMzyme"
"$PMM_PY" -m pip install 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cu128
printf 'torch==2.11.0\n' > /tmp/pmm-torch-constraint.txt
"$PMM_PY" -m pip install -r "$PMM_REPO/requirements/colab-overlay.txt" --constraint /tmp/pmm-torch-constraint.txt
"$PMM_PY" -m pip install --no-deps -r "$PMM_REPO/requirements/esmc-campaign-no-deps.txt"
"$PMM_PY" -m pip freeze > "$HOME/deepmzyme_runs/pmm-environment.txt"
```

The ESM overlay avoids dependency resolution that can replace PyTorch. It is a
pinned candidate environment, not a guarantee for every host. Record
installation output and dependency-check findings. Validate imports, the real
ESMC-600M model, representative sequence lengths and payload roundtrip before
expensive generation. Keep PMM's released-classifier dependencies in a
separate pinned environment.

```python
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

assert torch.cuda.is_available()
major, minor = torch.cuda.get_device_capability()
compiled_arches = torch.cuda.get_arch_list()
compiled_capabilities = [
    (int(arch[3:]) // 10, int(arch[3:]) % 10)
    for arch in compiled_arches if arch.startswith("sm_") and arch[3:].isdigit()
]
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())
print("capability", (major, minor), "compiled", compiled_arches)
assert any(m == major and n <= minor for m, n in compiled_capabilities), "No compatible compiled CUDA architecture"
assert torch.cuda.is_bf16_supported()
for dtype in (torch.float32, torch.bfloat16):
    x = torch.ones((32, 32), device="cuda", dtype=dtype)
    assert torch.isfinite(x @ x).all().item()
torch.cuda.synchronize()
```

Exact architecture-name membership is unnecessary: an `sm_86` cubin supports
L4's `sm_89` within the same compute-capability major version
([NVIDIA compatibility guide](https://docs.nvidia.com/cuda/ada-compatibility-guide/)).
Keep both the compatible-architecture check and actual CUDA kernel checks.

Observed GCP preflight on 2026-09-25: L4 compute capability 8.9 with PyTorch
`2.11.0+cu128` reported
`['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']`.
The former exact-membership assertion stopped before kernel/inference checks;
that failure did not establish an incompatible build. The corrected preflight
then passed FP32/BF16 CUDA operations and real ESMC-600M inference, including the
longest planned 3,715-residue sequence. The [campaign evidence](notebook_outputs/summaries/summary_pmm_ion_v2_context_20260926.md)
records the measured memory and timing; this establishes compatibility for
those inputs, not training throughput or universal freedom from OOM.

Use the campaign's ESM preflight for the actual longest-sequence test. Parameter
dtype alone does not establish embedding dtype or lossless casting. The probe
above is not a training smoke.

## Run, measure and close

Use exact campaign commands from the metal playbook. Full GPU training starts
only after the frozen cohort, feature inventory and nine-configuration smoke
pass. The fixed recipe does not imply AMP; preserve precision and batch size.
Measure preparation, training, validation, checkpointing and persistence
separately. Small graph models may be limited by CPU work or synchronization;
high utilization and universal AMP speedups are not promises. Tune workers or
add raw-graph caching only after measuring the bottleneck.

Use the existing runner's execution hooks for exclusive ownership,
complete-unit admission, a 1.25 forecast margin, 15-minute closeout reserve and
verified durable terminal artifacts. Record the real allocation start, including
installation/transfer time. Start with one training process. Failed fits stop
the queue after preserving evidence; interrupted fits restart from the original
seed. Completed fits require identity and checkpoint verification.

### Independent artifact transfer without a new mount

Use `--persistence-mode host_pull` when the workstation holds the independent
artifact copy. Pass its absolute destination as `--durable-root`; this is a
destination label on the VM, not a directory to create there. The runner stops
after a terminal unit and writes an immutable file manifest. The next unit is
blocked until the workstation downloads every listed file, verifies each
SHA-256 by reading the downloaded bytes, and uploads the resulting small
acknowledgment. Reuse the original campaign command after acknowledgment; its
completed-fit checks decide which scientific work can be reused.

Run the following on the workstation for each pending transfer. Set both roots
to the exact campaign and destination passed to the runner. The existing
controller supplies the IAP SSH configuration; these commands do not start a VM.

```bash
PMM_SSH_CONFIG="$HOME/deepmzyme-vm/state/ssh_config"
PMM_VM_ROOT=/home/mechti/deepmzyme_runs/pmm_ion_metal_v2_context
PMM_HOST_ROOT=/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v2_context/artifacts
PMM_TRANSFER=$(ssh -F "$PMM_SSH_CONFIG" deepmzyme-vm "jq -er '.pending_transfer.manifest_path' '$PMM_VM_ROOT/execution_state.json'")
mkdir -p "$PMM_HOST_ROOT/persistence_receipts"
scp -F "$PMM_SSH_CONFIG" "deepmzyme-vm:$PMM_VM_ROOT/$PMM_TRANSFER" "$PMM_HOST_ROOT/$PMM_TRANSFER"
jq -r '.files[].path' "$PMM_HOST_ROOT/$PMM_TRANSFER" > "$PMM_HOST_ROOT/.host_pull_files"
rsync -a --files-from="$PMM_HOST_ROOT/.host_pull_files" -e "ssh -F $PMM_SSH_CONFIG" "deepmzyme-vm:$PMM_VM_ROOT/" "$PMM_HOST_ROOT/"
PMM_ACK_REL=$(jq -er '.ack_path' "$PMM_HOST_ROOT/$PMM_TRANSFER")
PYTHONPATH=/home/mechti/PycharmProjects/DeepMzyme/src /home/mechti/miniconda3/envs/DeepMzyme/bin/python -c 'import sys; from pathlib import Path; from benchmarking.pmm_execution import verify_host_pull; verify_host_pull(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))' "$PMM_HOST_ROOT/$PMM_TRANSFER" "$PMM_HOST_ROOT" "$PMM_HOST_ROOT/$PMM_ACK_REL"
scp -F "$PMM_SSH_CONFIG" "$PMM_HOST_ROOT/$PMM_ACK_REL" "deepmzyme-vm:$PMM_VM_ROOT/$PMM_ACK_REL"
```

Stop immediately if any command fails; do not upload an old acknowledgment or
continue admission. A failed host or connection leaves the pending unit on the
VM's persistent disk and the admission gate closed. Keep the provider hard stop
active; recover the copy after connectivity returns through the existing
controller. Two directories on the same VM disk are not independent storage.
The transfer manifest also includes immutable execution-ledger snapshots.
Source archives and prepared embeddings need their own verified staging copy;
they are not recopied after every fit. With a verified independent mounted
destination, `--persistence-mode mounted` performs copy/readback directly.

On normal completion and exceptions, copy/read back artifacts and run:

```bash
~/deepmzyme-vm/bin/vm-stop
~/deepmzyme-vm/bin/vm-status
```

Confirm `TERMINATED` and record provider evidence. Keep Google's hard stop
active independently of the workstation. A finished worker, closed local
ledger or ended SSH connection does not stop billing by itself. Do not delete
the selected VM/disk for ordinary closeout. Recovery finalization is a separate,
authorized, evidence-gated cleanup of superseded resources.

For PyCharm details use `vm-pycharm-info --write-ssh-config`. Select the
configured remote interpreter, and exclude datasets, embeddings, caches and
run directories from automatic deployment.
