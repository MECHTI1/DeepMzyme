# DeepMzyme

DeepMzyme is a deep-learning framework with two primary missions: predicting
metalloenzyme transition-metal type and predicting EC/function labels from
protein structural pocket graphs, residue-level features, and optional ESMC
representations. Each task remains independently trainable and reportable;
shared metal-EC learning is an experimental challenger.

## What the project does

DeepMzyme supports:

1. direct four-class metal prediction: Mn, Cu, Zn, and Class VIII = Fe+Co+Ni;
2. EC/function classification, beginning scientifically at EC depth 1; and
3. experimental joint metal + EC prediction with independent heads and a
   shared learned representation.

The scientific plan requires a controlled comparison between direct four-class
training and six-class training evaluated after collapsing Fe/Co/Ni into Class
VIII. Historical six-class results retain their original target and cannot
substitute for that matched comparison; five-class results also remain
separately labeled. The current EC path is single-label at a selected hierarchy
depth and should not be described as full multi-label EC prediction.

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

For a fresh checkout, Colab, data setup, known readiness failures, and the
shortest route to the correct experiment stage, start with
[`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md).

On the existing project workstation, use the configured interpreter from the
repository root:

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

### Reproducing the Exact Zenodo PinMyMetal Benchmark

To replicate the 99.89% exact Zenodo PinMyMetal benchmark with ion-level supervision:

```bash
# 1-command end-to-end replication (downloads from HF, validates SHA-256, verifies, and runs):
bash scripts/reproduce_zenodo_pmm_benchmark.sh
```
Full documentation is in [`docs/ZENODO_PINMYMETAL_EXACT_ION_LEVEL_REPRODUCIBILITY.md`](docs/ZENODO_PINMYMETAL_EXACT_ION_LEVEL_REPRODUCIBILITY.md).

### Generalized 5-Fold Cross-Validation on Any Dataset Split (Ion & Pocket Levels)

DeepMzyme provides a unified orchestrator to run 5-fold cross-validation and benchmarking across any dataset folder split in both **metal ion-focused** (`--metal-example-unit ion`) and **pocket-centroid** (`--metal-example-unit pocket`) modes:

```bash
# Validate paths, site counts, and planned fold commands instantly without training:
python scripts/run_metal_5fold_cv.py --dataset train_and_test_sets_structures_exact_pinmymetal --metal-example-unit ion --dry-run

# Run full 5-fold CV across all 3 benchmark models:
python scripts/run_metal_5fold_cv.py --dataset train_and_test_sets_structures_exact_pinmymetal --metal-example-unit ion --epochs 50 --batch-size 16 --device cuda

# Or using the portable shell wrapper:
bash scripts/run_metal_5fold_cv.sh train_and_test_sets_structures_exact_pinmymetal ion cuda
```
Full documentation is in [`docs/GENERALIZED_METAL_5FOLD_CV_GUIDE.md`](docs/GENERALIZED_METAL_5FOLD_CV_GUIDE.md).

Interactive workflow:
[`notebooks/DeepMzyme_training_colab.ipynb`](notebooks/DeepMzyme_training_colab.ipynb).
For browser and terminal access to the same Colab GPU VM, including the required
PyTorch-preserving installation procedure, use
[`docs/COLAB_GPU_RUNBOOK.md`](docs/COLAB_GPU_RUNBOOK.md).

Fresh Linux x86_64 development and CPU-test setup is locked to Python 3.12 by
[`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock); exact commands and
the optional ESM group are in the [environment contract](requirements/README.md).
Colab uses the separate
[`requirements/colab-overlay.txt`](requirements/colab-overlay.txt), which
deliberately preserves the GPU-compatible stock PyTorch build.

Do not copy mutable notebook cell values from this README. The notebook is the
implemented live surface; exact reportable metal stage recipes belong to the
metal playbook.

## Documentation map

| Question | Authority |
|---|---|
| How do I get oriented and execute the project safely? | [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) |
| How do I reconstruct the development/test environment? | [`requirements/README.md`](requirements/README.md) |
| How do I reproduce or interpret compute benchmarks? | [`bench/README.md`](bench/README.md) |
| How do I use a Colab GPU through the browser and CLI? | [`docs/COLAB_GPU_RUNBOOK.md`](docs/COLAB_GPU_RUNBOOK.md) |
| Where am I and what should I do next? | [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md) |
| What are the scientific/design rules? | [`Plan.md`](Plan.md) |
| What datasets, splits, and bundles exist? | [`docs/DATASETS.md`](docs/DATASETS.md) |
| How are structure files deduplicated and resolved? | [`docs/STRUCTURE_STORE.md`](docs/STRUCTURE_STORE.md) |
| What has validation/HPO taught us? | [`docs/PARAMETER_FINDINGS.md`](docs/PARAMETER_FINDINGS.md) |
| Which experiment batches ran and where is their evidence? | [`docs/notebook_outputs/README.md`](docs/notebook_outputs/README.md) |
| What exact metal stage block should be used? | [`docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`](docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md) |
| What is the current EC recipe compatibility state? | [`docs/EC_TRAINING_PIPELINE_PLAYBOOK.md`](docs/EC_TRAINING_PIPELINE_PLAYBOOK.md) |
| Which verified technical issues remain unfixed? | [`docs/FOLLOW_UP_TECHNICAL_ISSUES.md`](docs/FOLLOW_UP_TECHNICAL_ISSUES.md) |
| How do I reproduce the exact Zenodo PinMyMetal ion-level benchmark? | [`docs/ZENODO_PINMYMETAL_EXACT_ION_LEVEL_REPRODUCIBILITY.md`](docs/ZENODO_PINMYMETAL_EXACT_ION_LEVEL_REPRODUCIBILITY.md) |
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
| `bench/` | G4/A100 compute-throughput evidence; not model-quality evidence |
| `DeepMzyme_Data/` | Local manifest-backed data, content-addressed structures, features, bundles, and runs; intentionally Git-ignored |

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
