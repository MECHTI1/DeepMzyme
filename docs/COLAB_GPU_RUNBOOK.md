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

Read the status output. The assigned hardware must be G4-class before
continuing. A 400 response means that the account lacks entitlement for the
requested accelerator; do not silently substitute another GPU.

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
required_arch = f"sm_{major}{minor}"
compiled_arches = torch.cuda.get_arch_list()
print("GPU:", torch.cuda.get_device_name(0))
print("compute capability:", f"{major}.{minor}")
print("required architecture:", required_arch)
print("compiled architectures:", compiled_arches)
if required_arch not in compiled_arches:
    raise RuntimeError(
        f"PyTorch {torch.__version__} does not contain kernels for {required_arch}. "
        "Restart with the stock Colab PyTorch build; do not start GPU work."
    )
PY
```

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

The main v10 bundle already includes ESM embeddings. For normal runs, keep
`PREPARE_MISSING_ESM_EMBEDDINGS = False` and avoid installing the optional ESM
generation package. If missing embeddings must be generated, the current
notebook can install the pinned `esm==3.2.3` package when its explicit
auto-install control is enabled; record the resulting environment as usual.

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
