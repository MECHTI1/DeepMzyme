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

Build the default Colab bundle from the trusted non-overlapped split:

```bash
PYTHONPATH=src /home/mechti/miniconda3/envs/DeepMzyme/bin/python src/build_colab_bundle.py
```

The bundle includes the site-level MAHOMES summary CSVs used by training. It also includes structure-level CSV artifacts for inspection; structures with multiple catalytic metal sites are represented there with semicolon-joined metal labels such as `Co;Cu`.

Example trusted-split Only-GVP metal baseline:

```bash
PYTHONPATH=src /home/mechti/miniconda3/envs/DeepMzyme/bin/python src/train.py \
  --task metal \
  --model-architecture only_gvp \
  --structure-dir DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train \
  --summary-csv DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv \
  --test-structure-dir DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/test \
  --test-summary-csv DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/test/final_data_summarazing_table_transition_metals_only_catalytic.csv \
  --run-test-eval \
  --runs-dir DeepMzyme_Data/runs_baseline_first \
  --run-name metal_only_gvp_seed42 \
  --seed 42 \
  --val-fraction 0.15 \
  --split-by pdbid \
  --selection-metric val_metal_balanced_acc \
  --epochs 50 \
  --batch-size 8
```

Detailed baseline-first commands are in `list_train_commands.md`. The interactive workflow is in `notebooks/DeepMzyme_training_colab.ipynb`.

Generated ESM embeddings should normally live outside the Git repository, then be passed with `--esm-embeddings-dir`. Use validation metrics for model and hyperparameter choice; reserve the held-out test set for final reporting of the selected checkpoint. Previous DeepMzyme runs favored `learning_rate=3e-5` among tested values, while `1e-4` was also reasonable in some runs. Use `3e-5` as the project-specific serious baseline starting point so far, and use `1e-4` as the main follow-up confirmation value rather than treating any LR as universally best.

Optional reproducibility and joint-loss controls include `--deterministic`, `--metal-loss-weight`, and `--ec-loss-weight`.

## EC Group Weighting

Metal prediction remains a pocket/site-level task, so metal loss and metal metrics are computed per pocket. EC/function prediction is structure/protein/chain-level; by default `--ec-group-weighting structure_id` weights EC cross-entropy so multiple separated EC-supervised pockets from the same structure contribute one total EC unit per split group. True multinuclear pockets are not downweighted by raw metal atom count, because nearby metals are already represented as one pocket by the extraction logic.

Use `--ec-group-weighting none` to recover unweighted pocket-level EC loss. Validation and held-out test reports include both pocket-level EC metrics and EC group-level metrics based on mean logits per group.
