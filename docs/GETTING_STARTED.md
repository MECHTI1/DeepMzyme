# Getting Started and Project Navigation

This is the shortest reliable route from a new checkout to an informed
DeepMzyme run. It explains what can be executed now, which setup claims are
reproducible, and where to look before choosing an experiment.

For live scientific progress, read
[`EXPERIMENT_STATUS.md`](../EXPERIMENT_STATUS.md). For exact metal-stage values,
use the [metal playbook](METAL_TRAINING_PIPELINE_PLAYBOOK.md); do not invent a
configuration from examples in this guide.

## Choose an execution path

| Path | Use it when | Setup status |
|---|---|---|
| Existing project workstation | The configured DeepMzyme Conda environment is already present | Directly executable with the interpreter below |
| Colab browser notebook | You want the supported staged workflow and interactive Drive authorization | Recommended cloud entry point |
| Colab CLI plus browser | You want terminal provisioning, environment checks, transfers, or monitoring while retaining the notebook UI on the same VM | Follow [`COLAB_GPU_RUNBOOK.md`](COLAB_GPU_RUNBOOK.md) |
| Fresh local machine | You need a newly reconstructed environment | Not yet exactly reproducible: the repository has a dependency list, but no Python-version specification or solved lockfile |

## Five-minute orientation

From the repository root, verify the required interpreter:

```bash
/home/mechti/miniconda3/envs/DeepMzyme/bin/python -c "import sys; print(sys.executable)"
```

The expected output is:

```text
/home/mechti/miniconda3/envs/DeepMzyme/bin/python
```

Inspect the unified training interface without loading a dataset:

```bash
/home/mechti/miniconda3/envs/DeepMzyme/bin/python src/train.py --help
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

## Local environment contract

The current repository expects this existing interpreter:

```text
/home/mechti/miniconda3/envs/DeepMzyme/bin/python
```

[`src/requirements.txt`](../src/requirements.txt) is a lightweight dependency
list, not a complete environment lock. It currently pins `torch==2.5.1` and
`torch-geometric==2.7.0`, but it does not record the Python version, CUDA wheel
source, transitive versions, ESM/ESMC version, Optuna, NumPy, scikit-learn, or
the full notebook/reporting stack. Therefore:

- use the configured interpreter on the existing workstation;
- do not claim that a fresh local environment is bit-for-bit reproducible from
  `src/requirements.txt` alone;
- record actual library versions in every serious validation or final-report
  run;
- do not install the full requirements file unchanged in Colab. Its PyTorch
  pin can replace a GPU-compatible Colab build. Use the
  [Colab runbook](COLAB_GPU_RUNBOOK.md), which filters only the top-level
  `torch` requirement just as the notebook does.

If dependencies are intentionally being repaired in the existing configured
environment, use that interpreter explicitly rather than a bare `python` or
`pip`:

```bash
/home/mechti/miniconda3/envs/DeepMzyme/bin/python -m pip install -r src/requirements.txt
```

That command is an environment mutation, not a routine orientation step.

## Data required before training

Training needs more than the Git checkout. The large structures, embeddings,
RING features, external features, and run outputs live under the Git-ignored
`DeepMzyme_Data/` tree or in a Colab bundle.

The main cloud bundle currently contains exact PinMyMetal, Common-PDBID 70/30,
CLEAN30 material, CARE clusterRes30, ESM embeddings, external features, and
RING assets. It does not contain the historical non-overlapped or harsh
PinMyMetal roots. Exact filenames, sizes, SHA256 values, Hugging Face paths,
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
/home/mechti/miniconda3/envs/DeepMzyme/bin/python src/train.py --task metal --help
/home/mechti/miniconda3/envs/DeepMzyme/bin/python src/train.py --task ec --help
/home/mechti/miniconda3/envs/DeepMzyme/bin/python src/train.py --task joint --help
```

Thin wrappers are also present:

```bash
/home/mechti/miniconda3/envs/DeepMzyme/bin/python src/train_metal.py --help
/home/mechti/miniconda3/envs/DeepMzyme/bin/python src/train_ec.py --help
```

These help commands prove interface availability only. A training launch still
requires explicit dataset, split, feature, output, and experiment settings.

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
| `tests/smoke_checks.py` | 37 checks pass, then the suite stops on a removed root file path | The full smoke suite is not currently green; see [`TECH-007`](FOLLOW_UP_TECHNICAL_ISSUES.md#tech-007--smoke-suite-references-a-removed-root-document) |
| Colab browser setup | Notebook clones the repository, preserves importable Colab PyTorch, and filters the top-level torch requirement | Supported with interactive Drive behavior noted in `TECH-008` |
| Colab G4 PyTorch | Stock `2.11.0+cu128` was compatible with `sm_120`; installing the repository's `torch==2.5.1` resolved to `2.5.1+cu124` and failed on `sm_120` | Preserve stock Colab PyTorch and run the architecture preflight |
| EC staged recipe | Code supports EC, but the EC playbook has notebook-surface mismatches | Not certified end-to-end in affected stages |

The smoke failure is a stale documentation-path assertion, not evidence that
the preceding 37 checks failed. The final optional multi-metal data check is
not reached until `TECH-007` is fixed.

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

The current project has fixed-split metal anchors and exploratory joint
results, but no Grade-1 or Grade-2 grouped-fold result, no completed Stage 6B
final refit, and no approved primary final-test route. The concise current
matrix and exact values live in [`EXPERIMENT_STATUS.md`](../EXPERIMENT_STATUS.md).

