# DeepMzyme

DeepMzyme is a deep-learning framework for predicting metalloenzyme metal type
and EC/function labels from protein structural pocket graphs, residue-level
features, and optional ESMC representations.

## What the project does

DeepMzyme supports:

1. metal-type classification;
2. EC/function classification;
3. joint metal + EC prediction.

Model families include structure-only GVP variants, ESM-only baselines, and
configurable graph/ESM fusion. Graph construction can use geometric radius edges
and optional RING interactions. The main training entry point supports all three
tasks; the Colab notebook provides the staged interactive workflow.

This README is the public overview, not a live experiment record or parameter
table.

## Where the project stands

Open [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md) for the current objective,
trusted validation anchor, challengers, blockers, dataset readiness, and exact
next actions.

Important current caveat:

> **Primary final-test route: unresolved scientific decision required before final reporting.**

The historical non-overlap PinMyMetal test was evaluated in seven early runs.
See [`docs/DATASETS.md`](docs/DATASETS.md) for the precise access record. Those
test values are not eligible current HPO or model-selection evidence.

## Quick start

Use the configured project interpreter from the repository root:

```bash
/home/mechti/miniconda3/envs/DeepMzyme/bin/python -c "import sys; print(sys.executable)"
```

Show the full training interface:

```bash
/home/mechti/miniconda3/envs/DeepMzyme/bin/python src/train.py --help
```

Task entry points:

```bash
# Unified interface
/home/mechti/miniconda3/envs/DeepMzyme/bin/python src/train.py --task metal --help
/home/mechti/miniconda3/envs/DeepMzyme/bin/python src/train.py --task ec --help
/home/mechti/miniconda3/envs/DeepMzyme/bin/python src/train.py --task joint --help

# Thin task-specific wrappers
/home/mechti/miniconda3/envs/DeepMzyme/bin/python src/train_metal.py --help
/home/mechti/miniconda3/envs/DeepMzyme/bin/python src/train_ec.py --help
```

Interactive workflow:
[`notebooks/DeepMzyme_training_colab.ipynb`](notebooks/DeepMzyme_training_colab.ipynb).

Do not copy mutable notebook cell values from this README. The notebook is the
implemented live surface; exact reportable metal stage recipes belong to the
metal playbook.

## Documentation map

| Question | Authority |
|---|---|
| Where am I and what should I do next? | [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md) |
| What are the scientific/design rules? | [`Plan.md`](Plan.md) |
| What datasets, splits, and bundles exist? | [`docs/DATASETS.md`](docs/DATASETS.md) |
| What has validation/HPO taught us? | [`docs/PARAMETER_FINDINGS.md`](docs/PARAMETER_FINDINGS.md) |
| Which experiment batches ran and where is their evidence? | [`docs/notebook_outputs/README.md`](docs/notebook_outputs/README.md) |
| What exact metal stage block should be used? | [`docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`](docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md) |
| What is the current EC recipe compatibility state? | [`docs/EC_TRAINING_PIPELINE_PLAYBOOK.md`](docs/EC_TRAINING_PIPELINE_PLAYBOOK.md) |
| Which verified technical issues remain unfixed? | [`docs/FOLLOW_UP_TECHNICAL_ISSUES.md`](docs/FOLLOW_UP_TECHNICAL_ISSUES.md) |
| Where is the complete documentation index? | [`docs/README.md`](docs/README.md) |

## Main repository areas

| Path | Purpose |
|---|---|
| `src/` | Training, models, graph construction, features, reporting |
| `notebooks/` | Unified Colab workflow |
| `prepare_training_and_test_set/` | PinMyMetal preparation and original membership |
| `CLEAN_prepare_training_and_test_set/` | CLEAN preparation and tracked provenance |
| `CARE_prepare_training_and_test_set/` | CARE preparation and tracked provenance |
| `CLEAN/` | CLEAN sequence-baseline workflow |
| `docs/notebook_outputs/` | Experiment index, summaries, and copied raw evidence |
| `DeepMzyme_Data/` | Local data, features, bundles, and runs; intentionally Git-ignored |

Avoid moving these directories casually: notebooks, preparation scripts,
generated provenance, bundle layouts, and local run records refer to their
current paths.

## Experiment principles

- Use validation evidence for model, checkpoint, architecture, fusion, and
  hyperparameter decisions.
- Keep task, label scheme, dataset, split, metric, seeds, and folds explicit.
- A single Optuna result is discovery evidence, not a confirmed best model.
- Preserve negative and incomplete experiments.
- Keep exact configs and raw outputs linked from summaries.
- Do not use the seven historical non-overlap test outcomes to derive parameter
  recommendations or current model rankings.

Detailed policy remains in [`Plan.md`](Plan.md).
