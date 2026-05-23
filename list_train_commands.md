# DeepMzyme baseline-first training commands

This file lists conservative, baseline-first direct CLI examples. The staged
notebook playbooks are the authoritative source for exact current budgets,
search spaces, Stage 6 confirmation, and Stage 7 final-test policy.

For the canonical validation/testing order and folder map, use `docs/README.md`.

The main policy is:

1. Use the legacy **Non-overlapped PinMyMetal** split for final held-out testing unless a planned comparison explicitly selects another named split.
2. Select checkpoints by **validation metrics**, not by the held-out test set.
3. Run simple baselines before complex fusion models.
4. Keep run names explicit so results can be compared later with `src/report_runs.py`.
5. The commands below are validation-only examples and intentionally omit
   `--run-test-eval`. Add test evaluation only for the final validation-selected
   configuration under the one-shot Stage 7 policy.

---

## 0. Environment and project location

Run commands from the repository root:

```bash
cd /home/mechti/PycharmProjects/DeepMzyme
```


Use the project Python interpreter explicitly:

```bash
/home/mechti/miniconda3/envs/DeepMzyme/bin/python -c "import sys; print(sys.executable)"
```


Expected output:

```text
/home/mechti/miniconda3/envs/DeepMzyme/bin/python
```

Recommended syntax checks before training:

```bash
/home/mechti/miniconda3/envs/DeepMzyme/bin/python -m py_compile src/model.py
/home/mechti/miniconda3/envs/DeepMzyme/bin/python -m py_compile src/training/run.py
/home/mechti/miniconda3/envs/DeepMzyme/bin/python -m py_compile src/report_runs.py
```

---

## 1. Shared paths

These commands assume the current trusted split, legacy **Non-overlapped PinMyMetal**, is located here:

```bash
TRAIN_DIR="DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train"
TEST_DIR="DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/test"
TRAIN_CSV="DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv"
TEST_CSV="DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/test/final_data_summarazing_table_transition_metals_only_catalytic.csv"
RUNS_DIR="DeepMzyme_Data/runs_baseline_first"
ESM_EMBEDDINGS_DIR="/media/Data/DeepMzyme_Data/esm_embeddings"
RING_FEATURES_DIR="DeepMzyme_Data/RING_features"
METAL_LABEL_SCHEME="six_class"
PYTHON="/home/mechti/miniconda3/envs/DeepMzyme/bin/python"
```

If your summary CSV has a different name, update `TRAIN_CSV` and `TEST_CSV` before running.
For ESM-enabled models, set `ESM_EMBEDDINGS_DIR` to your precomputed/generated ESM embedding directory and pass it with `--esm-embeddings-dir`. Missing ESM embeddings are generated only for structures whose embedding files are absent, unless `--no-prepare-missing-esm-embeddings` is passed. For graph runs, DeepMzyme's project default is RING-enabled plus strict updated external features: pass `--external-feature-source updated`, `--use-ring-edges`, `--ring-features-dir "${RING_FEATURES_DIR}"`, and `--prepare-missing-ring-edges`. Missing updated external features should fail unless you intentionally add `--allow-missing-external-features` for a debug/fallback ablation.

These examples use `learning_rate=3e-5` only as a simple fixed CLI value. For
current recommended grids, Optuna budgets, and stage-specific learning-rate
ranges, use the metal and EC playbooks. Choose LR by validation metrics, not by
repeatedly checking the held-out test set.

The examples use `METAL_LABEL_SCHEME="six_class"`, the default six separate
metal targets. For an explicitly labeled five-class validation comparison, set
`METAL_LABEL_SCHEME="five_class"`; this keeps Mn/Cu/Zn/Fe separate and groups
Co/Ni. Use separate run names and separate Optuna studies when changing it.

---

## 2. Metal classification: baseline-first order

Main metric for model selection:

```text
val_metal_balanced_acc
```

This is preferred over plain accuracy because metal classes may be imbalanced.

### 2.1 Only-GVP baseline

This tests what the structural pocket graph can learn without an ESM branch.

```bash
PYTHONPATH=src ${PYTHON} src/train.py \
  --task metal \
  --metal-label-scheme "${METAL_LABEL_SCHEME}" \
  --model-architecture only_gvp \
  --structure-dir "${TRAIN_DIR}" \
  --summary-csv "${TRAIN_CSV}" \
  --test-structure-dir "${TEST_DIR}" \
  --test-summary-csv "${TEST_CSV}" \
  --runs-dir "${RUNS_DIR}" \
  --run-name metal_only_gvp_seed42 \
  --seed 42 \
  --val-fraction 0.15 \
  --train-val-split-by pdbid \
  --selection-metric val_metal_balanced_acc \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 3e-5 \
  --weight-decay 1e-4 \
  --node-feature-set conservative \
  --external-feature-source updated \
  --use-ring-edges \
  --ring-features-dir "${RING_FEATURES_DIR}" \
  --prepare-missing-ring-edges
```

### 2.2 Only-ESM baseline

This tests how much signal comes from ESMC embeddings without graph message passing.

```bash
PYTHONPATH=src ${PYTHON} src/train.py \
  --task metal \
  --metal-label-scheme "${METAL_LABEL_SCHEME}" \
  --model-architecture only_esm \
  --structure-dir "${TRAIN_DIR}" \
  --summary-csv "${TRAIN_CSV}" \
  --test-structure-dir "${TEST_DIR}" \
  --test-summary-csv "${TEST_CSV}" \
  --runs-dir "${RUNS_DIR}" \
  --run-name metal_only_esm_seed42 \
  --seed 42 \
  --val-fraction 0.15 \
  --train-val-split-by pdbid \
  --selection-metric val_metal_balanced_acc \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 3e-5 \
  --weight-decay 1e-4 \
  --esm-embeddings-dir "${ESM_EMBEDDINGS_DIR}" \
  --node-feature-set conservative \
  --external-feature-source updated
```

### 2.3 GVP + simple late ESM fusion

This is the first combined graph + ESM model to test.

```bash
PYTHONPATH=src ${PYTHON} src/train.py \
  --task metal \
  --metal-label-scheme "${METAL_LABEL_SCHEME}" \
  --model-architecture gvp \
  --fusion-mode late_fusion \
  --structure-dir "${TRAIN_DIR}" \
  --summary-csv "${TRAIN_CSV}" \
  --test-structure-dir "${TEST_DIR}" \
  --test-summary-csv "${TEST_CSV}" \
  --runs-dir "${RUNS_DIR}" \
  --run-name metal_gvp_late_fusion_seed42 \
  --seed 42 \
  --val-fraction 0.15 \
  --train-val-split-by pdbid \
  --selection-metric val_metal_balanced_acc \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 3e-5 \
  --weight-decay 1e-4 \
  --esm-embeddings-dir "${ESM_EMBEDDINGS_DIR}" \
  --node-feature-set conservative \
  --external-feature-source updated \
  --use-ring-edges \
  --ring-features-dir "${RING_FEATURES_DIR}" \
  --prepare-missing-ring-edges
```

### 2.4 GVP + early residue-level ESM fusion

Run this only after the first three baselines are working and comparable.

```bash
PYTHONPATH=src ${PYTHON} src/train.py \
  --task metal \
  --metal-label-scheme "${METAL_LABEL_SCHEME}" \
  --model-architecture gvp \
  --fusion-mode early_fusion \
  --structure-dir "${TRAIN_DIR}" \
  --summary-csv "${TRAIN_CSV}" \
  --test-structure-dir "${TEST_DIR}" \
  --test-summary-csv "${TEST_CSV}" \
  --runs-dir "${RUNS_DIR}" \
  --run-name metal_gvp_early_fusion_seed42 \
  --seed 42 \
  --val-fraction 0.15 \
  --train-val-split-by pdbid \
  --selection-metric val_metal_balanced_acc \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 3e-5 \
  --weight-decay 1e-4 \
  --esm-embeddings-dir "${ESM_EMBEDDINGS_DIR}" \
  --node-feature-set conservative \
  --external-feature-source updated \
  --use-ring-edges \
  --ring-features-dir "${RING_FEATURES_DIR}" \
  --prepare-missing-ring-edges \
  --early-esm-dim 32 \
  --early-esm-dropout 0.2
```

---

## 3. Optional metal models after the baselines

Run these only if the simple baselines justify more complexity.

### 3.1 Node-level late fusion

```bash
PYTHONPATH=src ${PYTHON} src/train.py \
  --task metal \
  --metal-label-scheme "${METAL_LABEL_SCHEME}" \
  --model-architecture gvp \
  --fusion-mode node_level_late_fusion \
  --structure-dir "${TRAIN_DIR}" \
  --summary-csv "${TRAIN_CSV}" \
  --test-structure-dir "${TEST_DIR}" \
  --test-summary-csv "${TEST_CSV}" \
  --runs-dir "${RUNS_DIR}" \
  --run-name metal_gvp_node_level_late_fusion_seed42 \
  --seed 42 \
  --val-fraction 0.15 \
  --train-val-split-by pdbid \
  --selection-metric val_metal_balanced_acc \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 3e-5 \
  --weight-decay 1e-4 \
  --esm-embeddings-dir "${ESM_EMBEDDINGS_DIR}" \
  --node-feature-set conservative \
  --external-feature-source updated \
  --use-ring-edges \
  --ring-features-dir "${RING_FEATURES_DIR}" \
  --prepare-missing-ring-edges
```

### 3.2 Hybrid fusion

```bash
PYTHONPATH=src ${PYTHON} src/train.py \
  --task metal \
  --metal-label-scheme "${METAL_LABEL_SCHEME}" \
  --model-architecture gvp \
  --fusion-mode hybrid \
  --structure-dir "${TRAIN_DIR}" \
  --summary-csv "${TRAIN_CSV}" \
  --test-structure-dir "${TEST_DIR}" \
  --test-summary-csv "${TEST_CSV}" \
  --runs-dir "${RUNS_DIR}" \
  --run-name metal_gvp_hybrid_seed42 \
  --seed 42 \
  --val-fraction 0.15 \
  --train-val-split-by pdbid \
  --selection-metric val_metal_balanced_acc \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 3e-5 \
  --weight-decay 1e-4 \
  --esm-embeddings-dir "${ESM_EMBEDDINGS_DIR}" \
  --node-feature-set conservative \
  --external-feature-source updated \
  --use-ring-edges \
  --ring-features-dir "${RING_FEATURES_DIR}" \
  --prepare-missing-ring-edges \
  --early-esm-dim 32 \
  --early-esm-dropout 0.2
```

### 3.3 Cross-modal attention

```bash
PYTHONPATH=src ${PYTHON} src/train.py \
  --task metal \
  --metal-label-scheme "${METAL_LABEL_SCHEME}" \
  --model-architecture gvp \
  --fusion-mode cross_modal_attention \
  --structure-dir "${TRAIN_DIR}" \
  --summary-csv "${TRAIN_CSV}" \
  --test-structure-dir "${TEST_DIR}" \
  --test-summary-csv "${TEST_CSV}" \
  --runs-dir "${RUNS_DIR}" \
  --run-name metal_gvp_cross_modal_attention_seed42 \
  --seed 42 \
  --val-fraction 0.15 \
  --train-val-split-by pdbid \
  --selection-metric val_metal_balanced_acc \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 3e-5 \
  --weight-decay 1e-4 \
  --esm-embeddings-dir "${ESM_EMBEDDINGS_DIR}" \
  --node-feature-set conservative \
  --external-feature-source updated \
  --use-ring-edges \
  --ring-features-dir "${RING_FEATURES_DIR}" \
  --prepare-missing-ring-edges \
  --cross-attention-layers 1 \
  --cross-attention-heads 4 \
  --cross-attention-dropout 0.1 \
  --cross-attention-neighborhood all
```

---

## 4. EC classification commands

For EC prediction, start with EC level 1, then test deeper EC labels later.

Main metric for model selection:

```text
val_ec_group_balanced_acc
```

EC loss uses `--ec-group-weighting structure_id` by default so structures with multiple separated EC-supervised pockets are not over-counted.

### 4.1 EC Only-GVP baseline

```bash
PYTHONPATH=src ${PYTHON} src/train.py \
  --task ec \
  --model-architecture only_gvp \
  --ec-label-depth 1 \
  --structure-dir "${TRAIN_DIR}" \
  --summary-csv "${TRAIN_CSV}" \
  --test-structure-dir "${TEST_DIR}" \
  --test-summary-csv "${TEST_CSV}" \
  --runs-dir "${RUNS_DIR}" \
  --run-name ec_level1_only_gvp_seed42 \
  --seed 42 \
  --val-fraction 0.15 \
  --train-val-split-by pdbid \
  --selection-metric val_ec_group_balanced_acc \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 3e-5 \
  --weight-decay 1e-4 \
  --node-feature-set conservative \
  --external-feature-source updated \
  --use-ring-edges \
  --ring-features-dir "${RING_FEATURES_DIR}" \
  --prepare-missing-ring-edges
```

### 4.2 EC Only-ESM baseline

```bash
PYTHONPATH=src ${PYTHON} src/train.py \
  --task ec \
  --model-architecture only_esm \
  --ec-label-depth 1 \
  --structure-dir "${TRAIN_DIR}" \
  --summary-csv "${TRAIN_CSV}" \
  --test-structure-dir "${TEST_DIR}" \
  --test-summary-csv "${TEST_CSV}" \
  --runs-dir "${RUNS_DIR}" \
  --run-name ec_level1_only_esm_seed42 \
  --seed 42 \
  --val-fraction 0.15 \
  --train-val-split-by pdbid \
  --selection-metric val_ec_group_balanced_acc \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 3e-5 \
  --weight-decay 1e-4 \
  --esm-embeddings-dir "${ESM_EMBEDDINGS_DIR}" \
  --node-feature-set conservative \
  --external-feature-source updated
```

### 4.3 EC GVP + late ESM fusion

```bash
PYTHONPATH=src ${PYTHON} src/train.py \
  --task ec \
  --model-architecture gvp \
  --fusion-mode late_fusion \
  --ec-label-depth 1 \
  --structure-dir "${TRAIN_DIR}" \
  --summary-csv "${TRAIN_CSV}" \
  --test-structure-dir "${TEST_DIR}" \
  --test-summary-csv "${TEST_CSV}" \
  --runs-dir "${RUNS_DIR}" \
  --run-name ec_level1_gvp_late_fusion_seed42 \
  --seed 42 \
  --val-fraction 0.15 \
  --train-val-split-by pdbid \
  --selection-metric val_ec_group_balanced_acc \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 3e-5 \
  --weight-decay 1e-4 \
  --esm-embeddings-dir "${ESM_EMBEDDINGS_DIR}" \
  --node-feature-set conservative \
  --external-feature-source updated \
  --use-ring-edges \
  --ring-features-dir "${RING_FEATURES_DIR}" \
  --prepare-missing-ring-edges
```

---

## 5. Joint metal + EC commands

Joint training should come after separate metal and EC baselines are working.

Main metric for model selection:

```text
val_joint_balanced_acc
```

### 5.1 Joint Only-GVP baseline

```bash
PYTHONPATH=src ${PYTHON} src/train.py \
  --task joint \
  --model-architecture only_gvp \
  --ec-label-depth 1 \
  --structure-dir "${TRAIN_DIR}" \
  --summary-csv "${TRAIN_CSV}" \
  --test-structure-dir "${TEST_DIR}" \
  --test-summary-csv "${TEST_CSV}" \
  --runs-dir "${RUNS_DIR}" \
  --run-name joint_level1_only_gvp_seed42 \
  --seed 42 \
  --val-fraction 0.15 \
  --train-val-split-by pdbid \
  --selection-metric val_joint_balanced_acc \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 3e-5 \
  --weight-decay 1e-4 \
  --node-feature-set conservative \
  --external-feature-source updated \
  --use-ring-edges \
  --ring-features-dir "${RING_FEATURES_DIR}" \
  --prepare-missing-ring-edges
```

### 5.2 Joint GVP + late ESM fusion

```bash
PYTHONPATH=src ${PYTHON} src/train.py \
  --task joint \
  --model-architecture gvp \
  --fusion-mode late_fusion \
  --ec-label-depth 1 \
  --structure-dir "${TRAIN_DIR}" \
  --summary-csv "${TRAIN_CSV}" \
  --test-structure-dir "${TEST_DIR}" \
  --test-summary-csv "${TEST_CSV}" \
  --runs-dir "${RUNS_DIR}" \
  --run-name joint_level1_gvp_late_fusion_seed42 \
  --seed 42 \
  --val-fraction 0.15 \
  --train-val-split-by pdbid \
  --selection-metric val_joint_balanced_acc \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 3e-5 \
  --weight-decay 1e-4 \
  --esm-embeddings-dir "${ESM_EMBEDDINGS_DIR}" \
  --node-feature-set conservative \
  --external-feature-source updated \
  --use-ring-edges \
  --ring-features-dir "${RING_FEATURES_DIR}" \
  --prepare-missing-ring-edges
```

---

## 6. Repeating the strongest runs with multiple seeds

After the first pass, repeat only the most promising models with multiple seeds.
For reportable metal promotion, prefer the current playbook Stage 6 grouped-fold
confirmation. The direct CLI loop below is an exploratory validation-only
fallback, not a replacement for grouped-fold Stage 6.

Example exploratory seeds:

```text
42, 123, 777
```

Example for the GVP + late fusion metal model:

```bash
for SEED in 42 123 777; do
  PYTHONPATH=src ${PYTHON} src/train.py \
    --task metal \
    --metal-label-scheme "${METAL_LABEL_SCHEME}" \
    --model-architecture gvp \
    --fusion-mode late_fusion \
    --structure-dir "${TRAIN_DIR}" \
    --summary-csv "${TRAIN_CSV}" \
    --test-structure-dir "${TEST_DIR}" \
    --test-summary-csv "${TEST_CSV}" \
    --runs-dir "${RUNS_DIR}" \
    --run-name "metal_gvp_late_fusion_seed${SEED}" \
    --seed "${SEED}" \
    --val-fraction 0.15 \
    --train-val-split-by pdbid \
    --selection-metric val_metal_balanced_acc \
    --epochs 50 \
    --batch-size 8 \
    --learning-rate 3e-5 \
    --weight-decay 1e-4 \
    --esm-embeddings-dir "${ESM_EMBEDDINGS_DIR}" \
    --node-feature-set conservative \
    --external-feature-source updated \
    --use-ring-edges \
    --ring-features-dir "${RING_FEATURES_DIR}" \
    --prepare-missing-ring-edges
done
```

---

## 7. Summarize all runs

After training, summarize the run directories into one CSV:

```bash
${PYTHON} src/report_runs.py \
  --runs-dir "${RUNS_DIR}" \
  --out-csv "${RUNS_DIR}/baseline_first_summary.csv" \
  --out-figure "${RUNS_DIR}/baseline_first_summary.png"
```

The CSV is the main comparison table. Prefer comparing models by validation-selected metrics first, then inspect the held-out test metrics only for final reporting.

---

## 8. Optional: Metal Split PinMyMetal only as secondary/reference

Metal Split PinMyMetal is the exact PinMyMetal split for available supported structures. It should not be the main final held-out result if train/test overlap exists.

If it is used, label the run names clearly, for example:

```text
metal_exact_split_gvp_late_fusion_seed42_reference_only
```

Do not mix Metal Split PinMyMetal, Harsh Split PinMyMetal, and custom split runs in the same final comparison without clearly labeling the split type.
