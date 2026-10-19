# Getting Started and Project Navigation

This is the shortest reliable route from a new checkout to an informed
DeepMzyme run. It explains what can be executed now, which setup claims are
reproducible, and where to look before choosing an experiment.

For live scientific progress, read
[`EXPERIMENT_STATUS.md`](../EXPERIMENT_STATUS.md). For exact metal-stage values,
use the [metal playbook](METAL_TRAINING_PIPELINE_PLAYBOOK.md); do not invent a
configuration from examples in this guide.

## Choose an execution path

Before GPU planning or execution, consult [`GCP_GPU_RUNBOOK.md`](GCP_GPU_RUNBOOK.md)
and [`GPU_EXECUTION_CASCADE_PLAYBOOK.md`](GPU_EXECUTION_CASCADE_PLAYBOOK.md). They cover reuse,
capacity fallback, connection, admission, optional monitoring and verified
shutdown through the provider tools.
For an existing GCP VM that cannot restart because of capacity, follow the
[same-region recovery procedure](GCP_GPU_RUNBOOK.md#recover-a-stopped-vm-in-the-same-region).
It preserves the stopped source until the replacement and independent backup
are verified, while enforcing one active GPU.

| Path | Use it when | Setup status |
|---|---|---|
| Fresh Linux x86_64 checkout | Standard local development, CLI verification, and CPU test execution | Python 3.12 and all resolved packages are pinned in `uv.lock`; follow the environment contract below |
| GPU workstation or cloud VM | Production training, HPO, screen campaigns, and benchmark runners via CLI | Primary GPU route; follow [`GCP_GPU_RUNBOOK.md`](GCP_GPU_RUNBOOK.md) |
| Colab browser notebook | Interactive staged workflow with Google Drive persistence | Supported fallback; follow [`COLAB_GPU_RUNBOOK.md`](COLAB_GPU_RUNBOOK.md) |

## Five-minute orientation

From the repository root, install the locked Python 3.12 CPU environment and verify the interpreter:

```bash
uv python install 3.12
uv sync --frozen
uv run --frozen --no-sync python -c "import sys; print(sys.executable)"
```

Inspect the unified training interface without loading a dataset:

```bash
uv run --frozen --no-sync python src/train.py --help
```

Then answer these questions in order:

1. What is the current campaign stage? Read
   [`EXPERIMENT_STATUS.md`](../EXPERIMENT_STATUS.md).
2. Is the intended dataset actually materialized and in the current bundle?
   Read [`DATASETS.md`](DATASETS.md).
3. Which experiment batches and results are trustworthy? Read the
   [experiment index](notebook_outputs/README.md), then the cited summary and
   raw/config evidence.
4. Which exact stage should be run? For metal, use the corresponding block in
   [`METAL_TRAINING_PIPELINE_PLAYBOOK.md`](METAL_TRAINING_PIPELINE_PLAYBOOK.md).
5. Are there known execution problems? Read
   [`FOLLOW_UP_TECHNICAL_ISSUES.md`](FOLLOW_UP_TECHNICAL_ISSUES.md).

For metal runs, also check the saved `metal_example_unit` before interpreting
sample counts: `PocketRecord` and `pocket_id` are used for both clustered
pockets and ion-centered examples. See
[metal example terminology](../Plan.md#metal-example-terminology).

## Local environment contracts

For a fresh Linux x86_64 development and CPU-test environment, follow
[`requirements/README.md`](../requirements/README.md):

```bash
uv python install 3.12
uv sync --frozen
uv run --frozen --no-sync python src/train.py --help
uv run --frozen --no-sync pytest
```

`pyproject.toml` and `uv.lock` pin Python and the complete resolved CPU
environment. The default groups include tests and reporting; ESMC generation is
an explicit optional group. This contract does not select a local CUDA wheel.

If using a dedicated local Conda or virtual environment prefix on a GPU workstation:

- record actual library versions in every serious validation or final-report
  run;
- do not install the Linux lock or `src/requirements.txt` in Colab. Use the
  PyTorch-free [Colab overlay](../requirements/colab-overlay.txt) and the
  [Colab runbook](COLAB_GPU_RUNBOOK.md).

`src/requirements.txt` remains a pinned compatibility file for older local pip
commands, not the transitive reconstruction authority:

```bash
python -m pip install -r src/requirements.txt
```

That command is an environment mutation, not a routine orientation step.

## Data required before training

Training needs more than the Git checkout. The large structures, embeddings,
RING features, external features, and run outputs live under the Git-ignored
`DeepMzyme_Data/` tree or in a Colab bundle.

Local structure bytes live once in the content-addressed
`DeepMzyme_Data/structure_store/`; split directories contain
`structure_manifest.csv` membership files. Training and the notebook resolve
them automatically, and older bundles with structure files directly inside
train/test directories remain supported. See [`STRUCTURE_STORE.md`](STRUCTURE_STORE.md)
for the schema, audit, and verification command.

The main cloud bundle currently contains exact PinMyMetal, Common-PDBID 70/30,
historical non-overlapped PinMyMetal, CLEAN30 material, CARE clusterRes30,
ESM embeddings, external features, and RING assets. Harsh PinMyMetal is absent.
CARE caches are incomplete; consult the cache-readiness table before running
feature-dependent CARE training. Exact filenames, sizes, SHA256 values, Hugging Face paths,
and contents are owned by [`DATASETS.md`](DATASETS.md).

Before a run, confirm all of the following:

- the selected dataset root exists;
- the site-level summary CSV is used for labels;
- train/validation groups use `pdbid` unless a documented stage says
  otherwise;
- required ESM, RING, and external-feature coverage passes preflight;
- the bundle identifier and SHA256 will be saved for a serious run;
- held-out evaluation remains off for every non-final stage.

## Supported entry points

The unified CLI supports all tasks:

```bash
uv run --frozen --no-sync python src/train.py --task metal --help
uv run --frozen --no-sync python src/train.py --task ec --help
uv run --frozen --no-sync python src/train.py --task joint --help
```

Thin wrappers are also present:

```bash
uv run --frozen --no-sync python src/train_metal.py --help
uv run --frozen --no-sync python src/train_ec.py --help
```

These help commands prove interface availability only. A training launch still
requires explicit dataset, split, feature, output, and experiment settings.

`src/training/smoke_test.py` is an internal dormant helper, not a supported CLI
entry point: it has no executable main or current caller. Use
`tests/smoke_checks.py` for the supported fast compatibility check.

## Notebook workflow

The implemented workflow is
[`notebooks/DeepMzyme_training_colab.ipynb`](../notebooks/DeepMzyme_training_colab.ipynb).
It supports metal, EC, and joint tasks, but its editable live values are resume
state rather than canonical stage defaults.

For a metal run:

1. Start with **Stage 0: environment/data readiness**.
2. Paste exactly one stage block from the metal playbook into the main
   configuration cell.
3. Keep all launch switches off while inspecting the planned commands.
4. Use **Stage 1: 1-epoch smoke** before any expensive run in a fresh runtime.
5. Continue according to `EXPERIMENT_STATUS.md`, not according to whichever
   values happen to be left in the notebook.

The EC playbook preserves important scientific intent, but affected blocks are
not certified executable against the current notebook. Read its compatibility
warning and [`TECH-002`](FOLLOW_UP_TECHNICAL_ISSUES.md#tech-002--ec-playbook-assignments-do-not-match-the-notebook-surface)
before attempting an EC stage.

## Current executable-readiness checks

| Check | Audited outcome on 2026-08-22 | Interpretation |
|---|---|---|
| Configured interpreter | Present at the documented absolute path | Existing workstation path is usable |
| `src/train.py --help` | Passes | Unified CLI can be imported and parsed |
| `tests/smoke_checks.py` | Compatibility wrapper now continues through all checks, reports optional local-data absence as a skip, and returns nonzero if any check fails | Also exposed as isolated pytest cases; see [`TECH-007`](FOLLOW_UP_TECHNICAL_ISSUES.md#tech-007--smoke-suite-references-a-removed-root-document) |
| Colab browser setup | Notebook clones the repository, preserves importable Colab PyTorch, and installs the PyTorch-free Colab overlay | Supported with interactive Drive behavior noted in `TECH-008` |
| Colab G4 PyTorch | Stock `2.11.0+cu128` was compatible with `sm_120`; the separate overlay omits PyTorch | Preserve stock Colab PyTorch and run the architecture preflight |
| EC staged recipe | Code supports EC, but the EC playbook has notebook-surface mismatches | Not certified end-to-end in affected stages |

The smoke wrapper now completes all checks even if an earlier check fails, and
the optional multi-metal check reports a skip when its local fixture data is
absent. Pytest exposes the same checks independently.

## Repository map

| Path | What it owns |
|---|---|
| `Plan.md` | Scientific design, architecture intent, selection and held-out-test policy |
| `EXPERIMENT_STATUS.md` | Current stage, results, blockers, and next actions |
| `docs/DATASETS.md` | Dataset, split, bundle, Hugging Face, and test-use inventory |
| `docs/notebook_outputs/` | Indexed experiment summaries and copied evidence |
| `docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md` | Exact executable metal stage blocks and gates |
| `docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md` | Stable notebook option meanings |
| `docs/EC_TRAINING_PIPELINE_PLAYBOOK.md` | EC recipe intent plus the current compatibility warning |
| `notebooks/DeepMzyme_training_colab.ipynb` | Implemented interactive execution surface |
| `src/training/` | CLI configuration, preflight, data, splitting, loop, and task dispatch |
| `src/model_variants/` and `src/model.py` | Model families and experimental model definitions |
| `prepare_training_and_test_set/`, `CLEAN_prepare_training_and_test_set/`, `CARE_prepare_training_and_test_set/` | Dataset preparation and tracked provenance |
| `bench/` | Compute-throughput evidence; not model-quality evidence |

## How to interpret results

Do not rank runs only by the largest number visible in an output file. Check the
task, target scheme, dataset, validation split, metric, model family, seeds,
folds, and evidence grade first.

Current anchors, evidence grades, the Stage 6B source and exact values live in
[`EXPERIMENT_STATUS.md`](../EXPERIMENT_STATUS.md); the primary final-test route
is unresolved ([DATASETS](DATASETS.md#final-reporting-status)).
