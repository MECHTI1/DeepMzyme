# Google Colab GPU Runbook: Browser and CLI

This runbook provides two first-class interfaces to one Colab VM:

- the browser notebook for interactive configuration, Drive authorization, and
  long visible runs;
- the `colab` CLI for provisioning, hardware verification, environment setup,
  execution, artifact transfer, monitoring, and teardown.

The canonical notebook remains
[`notebooks/DeepMzyme_training_colab.ipynb`](../notebooks/DeepMzyme_training_colab.ipynb).
Exact metal experiment values remain in the
[metal playbook](METAL_TRAINING_PIPELINE_PLAYBOOK.md). This document owns the
runtime connection and environment procedure, not scientific stage budgets.

When measured execution no longer fits a planned GPU ceiling, keep the owned
session safe and persist the current state. If the user has not already decided,
ask whether to stop and close out or continue with a specifically quantified
higher ceiling. An existing explicit continuation authorization is sufficient;
do not ask for it again. Record an approved increase in the campaign's durable
authorization ledger before the next affected launch. The increase changes
only cumulative compute permission: per-session shutdown reserves, one-process
ownership, scientific recipes, validation selection, and held-out protections
remain in force.

## Non-negotiable runtime rules

1. Preserve the PyTorch build already supplied by Colab when it imports
   cleanly. Install only `requirements/colab-overlay.txt`; do not install the
   Linux CPU lock or `src/requirements.txt` on the VM.
2. Print the PyTorch version, CUDA version, GPU compute capability, and compiled
   CUDA architecture list before any GPU workload.
3. Use an explicit CLI session name on every command.
4. After provisioning, verify the actually assigned accelerator with
   `colab status` before installing or running anything expensive.
5. Use a timeout greater than the CLI's approximately 30-second default for
   installs, downloads, notebook execution, or model initialization.
6. Stop every CLI-created session on success and failure. An idle session keeps
   consuming metered Colab resources.
7. Keep held-out evaluation off outside the approved Stage 7 workflow.
8. For `metal_single_gpu_20h_v2`, keep exactly one owned allocation and one
   training process active. Account for provisioning/setup/idle/recovery time,
   verify saved artifacts, then stop the real VM before closing its ledger.

## Single-GPU campaign runtime policy

The [GPU runtime efficiency plan](GPU_RUNTIME_EFFICIENCY_PLAN.md) separates the
implemented, CPU-tested host queue and pause/accounting controls from later
preparation caching and same-session budget rebinding. The maintained adapter
has not yet been validated on a live Colab segment; consult current status
before any resume. Existing scientific recipes and admission gates still apply.

### Maintained queue and durable pause

`scripts/colab_serial_metal_host.py control-status --output CAMPAIGN_DIR`
reports both the append-only control history and existing
`USER_REQUESTED_PAUSE*.json` receipts. A pause blocks allocation, fresh worker
receipts and new launches. The queue drains terminal artifacts within the
bounded closeout path and the independent watchdog still enforces shutdown.
An explicit user resume must be recorded with its request identity and
authorization; deleting an old pause file is not the resume procedure.

`reconcile-accounting --output CAMPAIGN_DIR` previews closed host-only intervals;
adding `--apply` imports only intervals with matching independent provider-stop
and hardware receipts. It is idempotent and remains available while paused.
Archive the before-state. Accounting reconciliation is not a new allocation.

After source/state have been frozen as a separate operational continuation and
one owned endpoint has been prepared, the maintained command is:

```bash
/home/mechti/miniconda3/envs/DeepMzyme/bin/python \
  scripts/colab_serial_metal_supervisor.py \
  --output "$CAMPAIGN_DIR" --session-id "$OWNED_SESSION" \
  --persistence-seconds "$MEASURED_PER_FIT_PERSISTENCE_SECONDS"
```

The three variables must name the reviewed continuation, its current owned
session and its measured transfer allowance. There is no invented persistence
default. This command does not provision a VM, clear a pause, install packages
or regenerate features. It verifies existing staged setup, persists completed
attempts/state with independent readback, advances the original queue, applies
its cost gates and dispatches each exact launch intent once. An exclusive
host lease prevents a second controller. A transport failure triggers bounded
same-endpoint reconciliation; ambiguous ownership or launch state stops work.
It never allocates a replacement automatically.

Do not hot-patch an archived worker. Operational continuation must preserve
the accepted science, attempts, folds, charges and immutable receipts. CPU
fake-provider tests and offline CLI-help checks do not establish live transfer
performance or guarantee shutdown after loss of the entire host. The host
watchdog remains necessary; record actual provider-stop evidence on closeout.

The [single-GPU playbook recipe](METAL_TRAINING_PIPELINE_PLAYBOOK.md#single-gpu-metal-campaign)
owns the new profile's exact total/phase/session budgets and commands. It is
separate from prior pilots and serious Optuna studies. The worker runner does not
provision or stop a Colab VM. The dedicated
`scripts/colab_serial_metal_host.py` wrapper owns allocation, the independent
watchdog, ownership checks, and verified teardown; its commands are in the
playbook. Export an untouched local plan for the worker's paths and interpreter
before binding the host manifest. CPU preparation must then pass on that worker.
Use the actual allocation-start timestamp when opening the campaign session,
including setup before the runner became available. Closing a ledger entry
does not demonstrate that billing or the VM has stopped.

Prefer G4 if available. Record the assigned GPU model, physical memory,
compute capability, PyTorch architecture support, peak training memory and
measured full-fit costs. G4 is not a synonym for 16 GB. Repeat hardware/cost
inspection after reconnecting to a new runtime. Colab Pro availability and
session duration are variable; a persistent output directory does not make the
VM persistent. The queue's complete-block forecast and shutdown reserve must
use the actual assignment, not a historical fit-time promise.

Do CPU preparation before provisioning when possible. On the VM, stage cached
inputs on local disk and keep verified durable copies of every completed fit,
the frozen campaign manifest, queue/attempt state and allocation ledger.
Authorize Drive in the same browser kernel before unattended execution when
using a mounted Drive destination. Do not start a second VM to work around a
transport failure while the first might still exist. First reconcile the
owned runtime through status/assignment evidence.

After interruption, completed verified fits remain reusable; an interrupted
fit restarts once from its original seed. The trainer does not implement exact
optimizer/RNG/epoch continuation. Preserve the original attempt, charge the
whole allocation, and record its linked retry. Before planned shutdown, stop
admitting training, verify transfers/checkpoints, stop the owned VM and retain
its actual stop receipt. A lost connection is not evidence of teardown.

## Browser-only route

Open the notebook directly from GitHub:

```text
https://colab.research.google.com/github/MECHTI1/DeepMzyme/blob/main/notebooks/DeepMzyme_training_colab.ipynb
```

Then:

1. Select the intended GPU runtime.
2. Run the notebook's runtime/environment cell.
3. Run the PyTorch/CUDA preflight below before a training or benchmark cell.
4. Paste the exact Stage 0 block from the metal playbook.
5. Let the notebook clone the repository and install its PyTorch-free Colab overlay.
6. Authorize Drive interactively if persistence is required.
7. Keep launch controls off until the planning table matches the intended
   stage.

Browser Colab is interactive. `google.colab.drive.mount(...)` may request user
authorization; that is expected in this route.

## Chat 4 standalone smoke with a working-code snapshot

This is the browser handoff for **Stage 0: environment/data readiness** and
the gated **Stage 1: 1-epoch smoke**. The first standalone block in the
[metal playbook](METAL_TRAINING_PIPELINE_PLAYBOOK.md#exact-standalone-notebook-block)
owns the scientific configuration. Only the first Only-GVP/direct-four smoke
is prepared by the optional Chat 4 cells; the full baseline campaign stays off.

A GitHub clone omits uncommitted changes. Build a current snapshot without
resetting, committing, pulling, pushing, or modifying the Git index:

```bash
/home/mechti/miniconda3/envs/DeepMzyme/bin/python scripts/prepare_colab_smoke_snapshot.py
```

The archive, SHA256 sidecar and file/commit/dirty-state manifest are written to
`DeepMzyme_Data/notebook_outputs/plans/standalone_v1/colab/`. Regenerate after
further edits. Data and Git internals are excluded; the data bundle is fetched
separately. **Main configuration is the single bundle selector**, including
its immutable URL and checksum; the Chat 4 cell preserves that selection.
See [DATASETS](DATASETS.md#main-colab-bundle-v12-current-hosted-release-care-complete)
for the current v12 release.

Use **File → Upload notebook** in Colab to open the current local
`notebooks/DeepMzyme_training_colab.ipynb`, save a copy in Drive, and select a
GPU under **Runtime → Change runtime type** (G4 when available). Run cells
individually in this order; do not use Run all:

1. **Runtime/environment checks**.
2. **Main planned training launch switch**: leave it `False`.
3. **Main configuration**: run once; the next cell applies the complete
   standalone playbook override to its task/model defaults.
4. **Chat 4: load prepared code and configure one metal smoke (optional)**:
   set `PREPARE_CHAT4_METAL_SMOKE = True`, paste the archive's SHA256 into
   `CHAT4_SNAPSHOT_SHA256`, then run and upload `deepmzyme-chat4-code.tar.gz`.
   It verifies the archive and every file, creates a fresh
   `/content/DeepMzyme_chat4_*` directory and sets `REPO_ROOT` to it.
5. **Build central CONFIG dictionary**.
6. **Clone or locate DeepMzyme repository and install dependencies**: verify
   it uses the snapshot directory. Git diagnostics will say it is not a Git
   repository; provenance is supplied by the snapshot manifest. The overlay
   preserves stock PyTorch; precomputed ESM needs no generation environment.
7. **Data source configuration and bundle/data setup**: authorize Google
   Drive interactively. The cell verifies the data archive checksum and
   extracts it to `/content/deepmzyme_bundle`.
8. **Detect dataset paths and validate inputs**.
9. **Chat 4: verify remote inputs, CUDA and persistent outputs**: must pass
   the actual-GPU forward/backward check, prepared metal CSV/structure hash
   checks, complete ESM/external coverage, ESMC-300m/960 metadata checks,
   train/test group-membership separation, and Drive write/read check.
   Existing CARE train caches are inventoried without regeneration or training.
10. **Build planned configuration commands**: inspect exactly one CUDA
    command, one epoch, Only-GVP, metal, `four_class`, with the playbook's
    `pdbid`/seed-42/15% validation split, `metal_site` stratification and
    `six_class` eligibility. The selection metric is
    `val_metal_balanced_acc`. No test input or evaluation arguments may appear.

**Stop at the normal launch gate.** After explicitly choosing to launch the
reviewed smoke, set **Main planned training launch switch** to `True`, rerun
only that switch cell, then run **Optional training execution**. The setup
override resets launch to false, so do not rerun it at this step. After the
run completes, run **Summarize completed runs**. Leave Stage 6/6B and final
held-out cells off.

Checkpoints and results write directly to
`/content/drive/MyDrive/DeepMzyme/notebook_outputs/runs/standalone_metal_common70_four_class_only_gvp_smoke_v1/`.
The readiness cell also saves `colab_smoke_readiness.json`,
`ec_storage_inventory.json`, the code archive/manifest and `prepared_notebook.ipynb`.
The playbook lists the standard command/config, checkpoint, metrics and summary
artifacts. Confirm they are visible in Drive before disconnecting/deleting the
runtime. Smoke accuracy is not model-selection evidence. EC training and the
24-run metal campaign remain off during this handoff.

## Install the host CLI

The audited host setup uses `google-colab-cli==0.6.0` in an isolated `uv` tool
environment. Keep `jupyter-kernel-client` below 1.0 for this CLI release:

```bash
uv tool install --force "google-colab-cli==0.6.0" --with "jupyter-kernel-client<1.0"
```

Verify authentication without creating a VM:

```bash
colab whoami
```

On the audited workstation OAuth2 is already configured. If `whoami` returns
401 or 403, stop and repair host authentication with the user; do not use
`colab auth`, because that command injects credentials into a VM and is not
host CLI login.

Valid GPU names for CLI 0.6.0 are exactly `T4`, `L4`, `G4`, `H100`, and
`A100`. Do not guess another spelling: an unrecognized value can silently fall
back to A100.

## Provision a G4 and verify the assignment

The examples use the literal session name `deepmzyme-g4` so that every command
targets the same VM:

```bash
colab new --gpu G4 -s deepmzyme-g4
colab status -s deepmzyme-g4
```

Read the status output. For this G4-specific procedure, confirm the assigned
hardware is G4-class before continuing. A 400 response means that the account
lacks entitlement for the requested accelerator. A separately planned
single-GPU campaign can use another available accelerator only after recording
the assignment, rerunning CUDA/memory preflight and replacing cost estimates;
do not silently reuse G4 timing assumptions.

To attach the browser UI to this exact CLI-created VM:

```bash
colab url -s deepmzyme-g4
```

Open the printed URL. Do not create a second browser runtime. The CLI and this
URL attach to the same persistent Jupyter kernel, so imports and variables set
through one interface are visible to the other.

An optional same-kernel check from the CLI is:

```bash
colab exec -s deepmzyme-g4 --timeout 120 <<'PY'
DEEPMZYME_SHARED_VM_MARKER = "deepmzyme-g4"
print(DEEPMZYME_SHARED_VM_MARKER)
PY
```

In the attached browser, evaluating `DEEPMZYME_SHARED_VM_MARKER` should print
the same value.

## Mandatory PyTorch/CUDA preflight

Run this before dependency installation and again afterward:

```bash
colab exec -s deepmzyme-g4 --timeout 120 <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("The assigned Colab runtime does not expose CUDA.")

major, minor = torch.cuda.get_device_capability(0)
device_arch = f"sm_{major}{minor}"
compiled_arches = torch.cuda.get_arch_list()
compiled_capabilities = [
    (int(arch[3:]) // 10, int(arch[3:]) % 10)
    for arch in compiled_arches if arch.startswith("sm_") and arch[3:].isdigit()
]
print("GPU:", torch.cuda.get_device_name(0))
print("compute capability:", f"{major}.{minor}")
print("device architecture:", device_arch)
print("compiled architectures:", compiled_arches)
if not any(m == major and n <= minor for m, n in compiled_capabilities):
    raise RuntimeError(
        f"PyTorch {torch.__version__} has no compatible compiled architecture for {device_arch}. "
        "Restart with the stock Colab PyTorch build; do not start GPU work."
    )
probe_dtypes = [torch.float32]
if torch.cuda.is_bf16_supported():
    probe_dtypes.append(torch.bfloat16)
for dtype in probe_dtypes:
    x = torch.ones((32, 32), device="cuda", dtype=dtype)
    assert torch.isfinite(x @ x).all().item()
torch.cuda.synchronize()
print("CUDA kernel checks passed:", probe_dtypes)
PY
```

Accept a compiled cubin with the same compute-capability major and an equal or
lower minor version; `sm_86` therefore supports L4's `sm_89`
([NVIDIA compatibility guide](https://docs.nvidia.com/cuda/ada-compatibility-guide/)).
This still rejects the observed Blackwell 12.0 build capped at `sm_90`.
Architecture metadata alone is insufficient: the CUDA operations above must
pass, followed by the workload's real model preflight and training smoke.

Observed compatibility evidence on 2026-08-22:

| Runtime | PyTorch/CUDA | Architecture result | Outcome |
|---|---|---|---|
| G4, NVIDIA RTX PRO 6000 Blackwell Server Edition, compute 12.0 | Stock Colab `2.11.0+cu128` | Included `sm_120` | Compatible |
| G4 after unfiltered project requirements | `2.5.1+cu124` | Compiled only through `sm_90` | Failed with `no kernel image is available for execution on the device` |
| A100-SXM4-40GB, compute 8.0 | Stock Colab `2.11.0+cu128` | Included `sm_80` | Compatible |

Stock Colab versions can change. The preflight output, not the historical
version number, is the acceptance gate.

## Clone the repository and install dependencies safely

This setup uses the same explicit PyTorch-free overlay as the notebook. The
guard below fails if a future edit accidentally adds a top-level PyTorch line.

```bash
colab exec -s deepmzyme-g4 --timeout 1200 <<'PY'
import subprocess
import sys
from pathlib import Path

repo_dir = Path("/content/DeepMzyme")
if not (repo_dir / "src" / "train.py").is_file():
    if repo_dir.exists():
        raise RuntimeError(f"{repo_dir} exists but is not a DeepMzyme checkout")
    subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            "main",
            "https://github.com/MECHTI1/DeepMzyme.git",
            str(repo_dir),
        ],
        check=True,
    )

subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, check=True)

try:
    import torch
except Exception as exc:
    raise RuntimeError(
        "Stock Colab PyTorch is not importable. Restart the runtime before installing dependencies."
    ) from exc
print("Preserving existing PyTorch:", torch.__version__)

overlay_path = repo_dir / "requirements" / "colab-overlay.txt"
overlay_lines = overlay_path.read_text(encoding="utf-8").splitlines()
normalized = [line.split("#", 1)[0].strip().lower() for line in overlay_lines]
if any(line == "torch" or line.startswith(("torch==", "torch>=", "torch<=")) for line in normalized):
    raise RuntimeError("The managed Colab overlay must not contain a PyTorch requirement")

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-r", str(overlay_path)],
    check=True,
)

print("Installed the DeepMzyme Colab overlay without replacing PyTorch.")
PY
```

Run the mandatory PyTorch/CUDA preflight again. If PyTorch was deliberately
changed for any reason, restart the kernel before importing `torch`,
`torch_geometric`, or DeepMzyme training modules.

The current hosted bundle includes complete audited CARE ESM/external/RING
caches; see the release identity and historical gaps in [DATASETS.md](DATASETS.md).
For feature-complete normal runs, keep
`PREPARE_MISSING_ESM_EMBEDDINGS = False` and
`AUTO_INSTALL_ESM_FOR_EMBEDDING_GENERATION = False`, as in the updated notebook.
This avoids installing the optional ESM generation package during dependency
setup, which otherwise runs before cache coverage is inspected. If missing
embeddings must be generated, the current notebook can install the pinned
`esm==3.2.3` package when its explicit
auto-install control is enabled; record the resulting environment as usual.

### ESM generation on a Python 3.13 Colab runtime

Check the actual runtime version first. The pinned `esm==3.2.3` requires
Python 3.12; its installation failed in the audited Python 3.13 runtime.
An **optional, tested workaround** is a separate Python 3.12 environment for
ESMC feature preparation, followed by training with precomputed caches in the
stock Colab interpreter. This is a suggested shortcut, not the only possible
solution or a required runtime downgrade. If the runtime already uses
compatible Python 3.12, skip the extra environment. Other compatible routes
may work, but were not verified here; do not silently change the scientific
model or bypass the package's Python-version constraint.

The CARE repair used this setup on the same G4 VM, retaining stock PyTorch for
the notebook training path:

```bash
python -m pip install uv
uv python install 3.12
uv venv --python 3.12 /content/deepmzyme-env
uv pip install --python /content/deepmzyme-env/bin/python \
  torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python /content/deepmzyme-env/bin/python \
  -r requirements/colab-overlay.txt esm==3.2.3
```

Recheck CUDA in that interpreter before generation. The pinned legacy
ESMC-300m model revision and per-structure sequence hashes belong in the
generation report. `src/complete_care_caches.py` implements inventory,
resumable external/RING/ESM generation, and a final alignment/feature audit.
It generates test-side features only; it never runs held-out model evaluation.
The [CARE repair evidence](notebook_outputs/summaries/summary_colab_care_cache_smoke_20260914.md)
records the exact environment and commands used. A complete filename inventory
alone is insufficient: verify PROPKA availability and exercise the ESM loader.

Two additional failures were confirmed and resolved during that run:

- Existing external JSON files concealed a failed PROPKA step (`pka:
  unavailable`, `No module named propka`). Installing PROPKA in the generation
  environment and regenerating the affected files resolved it. Check metadata,
  not just file existence; the cache utility's `--repair-audit` handles this case.
- Only-ESM failed on duplicate residue IDs because both exact and older
  EC-annotation filenames were loaded. The fix in
  `src/training/esm_feature_loading.py` prefers exact caches and uses the older
  alias only when no exact cache exists. Use that updated loader; the final
  three GPU smoke runs passed without deleting the older dataset's cache.

For large CLI downloads, split archives into approximately 64 MiB chunks on
the VM, download them separately, concatenate locally, and verify the original
whole-archive SHA256. This bounds local CLI memory use. Always retrieve caches,
run artifacts, and provenance before stopping the runtime.

Do not replace the code above with:

```bash
# Unsafe for this project on Colab G4:
colab install -s deepmzyme-g4 -r src/requirements.txt
```

## Data and Drive choices

### Ephemeral CLI or smoke work

Use the Hugging Face bundle and keep Drive mounting off:

```python
COLAB_DATA_SOURCE = "huggingface_link"
MOUNT_DRIVE = False
```

The notebook downloads the main archive to `/content`, verifies the configured
SHA256, installs `zstd` if required, and unpacks under
`/content/deepmzyme_bundle`. The VM is ephemeral; download every needed output
before stopping it.

### Persistent serious HPO

Persistent Drive SQLite storage is mandatory for metal Stage 4 and Stage 5.
The CLI's Drive mount command is interactive and must not be used unattended.
Instead:

1. Provision with the CLI and open `colab url -s deepmzyme-g4`.
2. In the attached browser, mount Drive once and complete the authorization.
3. Confirm `/content/drive` exists in that kernel.
4. In the notebook's editable main configuration, use the exact playbook stage
   block and Drive paths. If Drive is already mounted, set `MOUNT_DRIVE = False`
   to prevent a second interactive mount attempt.
5. Keep `OPTUNA_ALLOW_INCOMPATIBLE_STUDY_REUSE = False` for reportable HPO.

The current canonical notebook has `MOUNT_DRIVE = True` in its editable live
configuration. An unattended `colab exec -f` can therefore block at
`google.colab.drive.mount(...)` unless a working copy is configured for the
headless path or Drive was authorized in the same kernel first. This is tracked
as [`TECH-008`](FOLLOW_UP_TECHNICAL_ISSUES.md#tech-008--interactive-drive-mount-blocks-unattended-cli-execution).

## Running the notebook or a planned command

Use the browser notebook for the safest long staged workflow. It exposes
planning output, launch switches, Stage 6/6B gates, and the final-test guard in
one place.

For CLI notebook execution, first prepare a working copy whose main
configuration has the exact playbook block and no unresolved interactive Drive
prompt. Do not overwrite the canonical tracked notebook merely to create a run
copy. Then execute with an explicit timeout:

```bash
colab exec -s deepmzyme-g4 --timeout 1800 -f /path/to/DeepMzyme_training_colab_cli.ipynb
```

The CLI writes a new `DeepMzyme_training_colab_cli_output.ipynb` beside the
local input notebook. The approximately 30-second default timeout is too short
for bundle setup, model construction, or most notebook runs.

Do not hold one CLI transport call open for a multi-hour HPO batch. Either run
the long notebook cells visibly in the attached browser or launch the exact
shell-safe command printed by the notebook as a detached VM job that writes to
a persistent Drive run directory and a log. Keep the notebook-generated config
artifacts with the run.

Before any launch, confirm:

- the stage block came from the current playbook;
- `INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False`;
- the effective metal selection metric is `val_metal_balanced_acc`;
- normal validation uses `VAL_FRACTION = 0.15` and `SPLIT_BY = "pdbid"`;
- one `MODEL_PRESET` maps to one compatible Optuna study;
- serious HPO uses persistent Drive SQLite;
- the planned command and output directory are saved.

## Remote artifact compatibility

Do not upload a pickle containing a project-defined Python class and expect it
to deserialize in a remote runtime without the defining module on `sys.path`.
An audited attempt failed with `ModuleNotFoundError: No module named 'graph'`.

The hosted benchmark v1 artifact does not satisfy that rule: it contains
`graph.construction.PocketData` and is retained only as historical evidence.
The v2 contract serializes mappings, lists, scalars, and tensors, proves
`torch.load(..., weights_only=True)` succeeds, and reconstructs `PocketData`
inside the runner after safe loading. No v2 artifact has been generated or
uploaded yet. See [`bench/README.md`](../bench/README.md).

## Monitor, download, and stop

Inspect the session and recent events:

```bash
colab status -s deepmzyme-g4
colab log -s deepmzyme-g4 -n 30
```

Download a known artifact before teardown:

```bash
colab download -s deepmzyme-g4 /content/path/to/artifact.json ./artifact.json
```

When the work is complete, or after any failure:

```bash
colab stop -s deepmzyme-g4
colab sessions
```

Confirm that `deepmzyme-g4` is gone. Entries shown as `[?]` were created
outside this CLI and must not be targeted or stopped by name.

## Failure guide

| Symptom | Meaning | Action |
|---|---|---|
| `no kernel image is available for execution on the device` on G4 | Installed PyTorch lacks `sm_120` kernels | Stop the run, restart with stock Colab PyTorch, apply only the Colab overlay, rerun preflight |
| `colab exec` ends near 30 seconds | CLI transport timeout, not an OOM | Repeat only after raising `--timeout`, or use browser/detached execution for long work |
| `ModuleNotFoundError` while loading a pickle | Serialized project class is unavailable remotely | Use the tensor-only v2 schema and reconstruct the project graph class only after `weights_only=True` loading |
| Drive mount waits for input | Interactive authorization was triggered | Use the browser attached through `colab url`, authorize once, and avoid a second mount in CLI execution |
| 401/403 from CLI | Host CLI authentication problem | Run `colab whoami`, report the result, and stop; do not use VM-side `colab auth` as a repair |
| Requested GPU differs from status | Assignment mismatch | Do not start the workload; report the actual assignment |
| `colab new` returns 400 | No entitlement for that accelerator | Report it; do not silently retry on another accelerator |

## Completion record

For every CLI-managed run, record:

- CLI session name;
- accelerator actually assigned by `colab status`;
- PyTorch version, CUDA version, compute capability, and architecture list;
- repository commit;
- bundle filename and SHA256;
- stage/config artifact paths;
- local destination of downloaded outputs;
- explicit confirmation that the session was stopped.
