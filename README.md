# DeepMzyme

DeepMzyme is a deep-learning framework for predicting metalloenzyme metal type and EC/function labels from protein structural pocket graphs, residue-level features, and optional ESMC embeddings.

## Main Tasks

1. Metal-type classification
2. EC/function classification
3. Joint metal + EC prediction

## Recommended Split

Use the non-overlapped PinMyMetal split for final held-out evaluation. Select checkpoints and model variants with validation metrics only; use the held-out test set for final reporting of the selected checkpoint.

## Quick Start

Use the project interpreter from the repository root:

```bash
/home/mechti/miniconda3/envs/DeepMzyme/bin/python -c "import sys; print(sys.executable)"
```

Show the training CLI:

```bash
/home/mechti/miniconda3/envs/DeepMzyme/bin/python src/train.py --help
```

Example trusted-split Only-GVP metal baseline:

```bash
PYTHONPATH=src /home/mechti/miniconda3/envs/DeepMzyme/bin/python src/train.py \
  --task metal \
  --model-architecture only_gvp \
  --structure-dir .data/train_and_test_sets_structures_non_overlapped_pinmymetal/train \
  --summary-csv .data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv \
  --test-structure-dir .data/train_and_test_sets_structures_non_overlapped_pinmymetal/test \
  --test-summary-csv .data/train_and_test_sets_structures_non_overlapped_pinmymetal/test/final_data_summarazing_table_transition_metals_only_catalytic.csv \
  --run-test-eval \
  --runs-dir .data/runs_baseline_first \
  --run-name metal_only_gvp_seed42 \
  --seed 42 \
  --val-fraction 0.15 \
  --split-by pdbid \
  --selection-metric val_metal_balanced_acc \
  --epochs 50 \
  --batch-size 8
```

More baseline-first commands are in `list_train_commands.md`. A Google Colab workflow is available at `notebooks/DeepMzyme_training_colab.ipynb`.

Optional reproducibility and joint-loss controls include `--deterministic`, `--metal-loss-weight`, and `--ec-loss-weight`.
