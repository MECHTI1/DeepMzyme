# Getting started

For metal prediction with the published weights, follow the [quick start](../README.md#quick-start).
For training, check the [current campaign](../EXPERIMENT_STATUS.md) and its linked README.

## Local setup

From the repository root on Linux x86_64, install the locked Python 3.12 CPU environment and run the smoke checks.

```bash
uv python install 3.12
uv sync --frozen
uv run --frozen --no-sync python tests/smoke_checks.py
uv run --frozen --no-sync python src/train.py --help
```

Run the full CPU suite with `uv run --frozen --no-sync pytest`. An optional smoke check skips when its local fixture is absent.
The [environment guide](../requirements/README.md) covers ESM-C, legacy pip installs and recording versions on a GPU workstation.

## Data required before training

Structures, embeddings and feature caches live under `DeepMzyme_Data/` or in a Colab bundle.
The [dataset inventory](DATASETS.md) lists bundle contents, checksums and cache coverage.

- Check the dataset root and site-level label CSV. Confirm ESM, RING and external-feature coverage for the selected model.
- Check the grouping key against the split protocol. For metal counts, read `metal_example_unit` to distinguish parent pockets from ion examples. See [metal example terminology](../Plan.md#metal-example-terminology).
- Record the bundle ID and SHA-256 for serious runs. Keep held-out evaluation off until Stage 7.

Split directories can contain structure files or a `structure_manifest.csv` pointing to the shared [structure store](STRUCTURE_STORE.md).

## Supported entry points

`src/train.py --task metal|ec|joint` selects the training task. `src/train_metal.py` and `src/train_ec.py` are standalone wrappers.
Use the [metal playbook](METAL_TRAINING_PIPELINE_PLAYBOOK.md) for stage settings and the [configuration guide](METAL_NOTEBOOK_CONFIGURATION_GUIDE.md) for option meanings.
For EC, check [playbook compatibility](EC_TRAINING_PIPELINE_PLAYBOOK.md) and [TECH-002](FOLLOW_UP_TECHNICAL_ISSUES.md#tech-002--ec-playbook-assignments-do-not-match-the-notebook-surface) before HPO or final evaluation.

The primary GPU route is the GCP VM. Follow the [GCP runbook](GCP_GPU_RUNBOOK.md) and [campaign execution guide](GPU_EXECUTION_CASCADE_PLAYBOOK.md) for admission, monitoring and shutdown.
If capacity blocks a restart, use the [stopped-VM recovery procedure](GCP_GPU_RUNBOOK.md#recover-a-stopped-vm-in-the-same-region).

## Notebook workflow

The [Colab notebook](../notebooks/DeepMzyme_training_colab.ipynb) supports all three tasks. Follow the [Colab runbook](COLAB_GPU_RUNBOOK.md) to keep the runtime's PyTorch installation and use the [Colab overlay](../requirements/colab-overlay.txt).
Saved notebook values reflect the previous session. For a fresh metal run, start at Stage 0, copy one playbook block into the configuration cell, and inspect the commands with launches disabled. Run the Stage 1 one-epoch smoke before longer fits.

## How to interpret results

Use the [experiment index](notebook_outputs/README.md) to find the run summary, folds, seeds, checkpoint metric and evidence grade.
Compare runs on the same task, reporting target and split. The [test-use ledger](DATASETS.md#final-reporting-status) records the open final-test decision.
The [documentation index](README.md) maps the training, model, data preparation and benchmarking code.
