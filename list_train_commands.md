# DeepMzyme baseline-first training commands

This file lists conservative, baseline-first commands for DeepMzyme experiments.

The main policy is:

1. Use the **non-overlapped PinMyMetal split** for final held-out testing.
2. Select checkpoints by **validation metrics**, not by the held-out test set.
3. Run simple baselines before complex fusion models.
4. Keep run names explicit so results can be compared later with `src/report_runs.py`.

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

These commands assume the main trusted split is located here:

```bash
TRAIN_DIR="DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train"
TEST_DIR="DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/test"
TRAIN_CSV="DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv"
TEST_CSV="DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/test/final_data_summarazing_table_transition_metals_only_catalytic.csv"
RUNS_DIR="DeepMzyme_Data/runs_baseline_first"
ESM_EMBEDDINGS_DIR="/media/Data/DeepMzyme_Data/embeddings"
RING_FEATURES_DIR="DeepMzyme_Data/RING_features"
PYTHON="/home/mechti/miniconda3/envs/DeepMzyme/bin/python"
```

If your summary CSV has a different name, update `TRAIN_CSV` and `TEST_CSV` before running.
For ESM-enabled models, set `ESM_EMBEDDINGS_DIR` to your precomputed/generated ESM embedding directory and pass it with `--esm-embeddings-dir`. Missing ESM embeddings are generated only for structures whose embedding files are absent, unless `--no-prepare-missing-esm-embeddings` is passed. For RING-enabled graph runs, pass `--ring-features-dir "${RING_FEATURES_DIR}"`; missing RING edge files are generated only when RING edges are enabled and the expected files are absent, unless `--no-prepare-missing-ring-edges` is passed.

Previous DeepMzyme runs favored `learning_rate=3e-5` among tested values, while `1e-4` was also reasonable in some runs. Use `3e-5` with `weight_decay=1e-4` as the serious baseline starting point so far. Use `1e-4` as the main follow-up confirmation value, or compare only `3e-5` and `1e-4` if time is limited. Reserve `3e-4` for an optional wider LR check or debug/default-code comparison after simpler runs are working. For larger or fusion-heavy models such as `hybrid` and `cross_modal_attention`, start conservatively with `3e-5` or `1e-4`, and reduce to `1e-5` if training is unstable. Choose LR by validation metrics, not by repeatedly checking the held-out test set.

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
  --model-architecture only_gvp \
  --structure-dir "${TRAIN_DIR}" \
  --summary-csv "${TRAIN_CSV}" \
  --test-structure-dir "${TEST_DIR}" \
  --test-summary-csv "${TEST_CSV}" \
  --run-test-eval \
  --runs-dir "${RUNS_DIR}" \
  --run-name metal_only_gvp_seed42 \
  --seed 42 \
  --val-fraction 0.15 \
  --split-by pdbid \
  --selection-metric val_metal_balanced_acc \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 3e-5 \
  --weight-decay 1e-4 \
  --node-feature-set conservative
```

### 2.2 Only-ESM baseline

This tests how much signal comes from ESMC embeddings without graph message passing.

```bash
PYTHONPATH=src ${PYTHON} src/train.py \
  --task metal \
  --model-architecture only_esm \
  --structure-dir "${TRAIN_DIR}" \
  --summary-csv "${TRAIN_CSV}" \
  --test-structure-dir "${TEST_DIR}" \
  --test-summary-csv "${TEST_CSV}" \
  --run-test-eval \
  --runs-dir "${RUNS_DIR}" \
  --run-name metal_only_esm_seed42 \
  --seed 42 \
  --val-fraction 0.15 \
  --split-by pdbid \
  --selection-metric val_metal_balanced_acc \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 3e-5 \
  --weight-decay 1e-4 \
  --esm-embeddings-dir "${ESM_EMBEDDINGS_DIR}" \
  --node-feature-set conservative
```

### 2.3 GVP + simple late ESM fusion

This is the first combined graph + ESM model to test.

```bash
PYTHONPATH=src ${PYTHON} src/train.py \
  --task metal \
  --model-architecture gvp \
  --fusion-mode late_fusion \
  --structure-dir "${TRAIN_DIR}" \
  --summary-csv "${TRAIN_CSV}" \
  --test-structure-dir "${TEST_DIR}" \
  --test-summary-csv "${TEST_CSV}" \
  --run-test-eval \
  --runs-dir "${RUNS_DIR}" \
  --run-name metal_gvp_late_fusion_seed42 \
  --seed 42 \
  --val-fraction 0.15 \
  --split-by pdbid \
  --selection-metric val_metal_balanced_acc \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 3e-5 \
  --weight-decay 1e-4 \
  --esm-embeddings-dir "${ESM_EMBEDDINGS_DIR}" \
  --node-feature-set conservative
```

### 2.4 GVP + early residue-level ESM fusion

Run this only after the first three baselines are working and comparable.

```bash
PYTHONPATH=src ${PYTHON} src/train.py \
  --task metal \
  --model-architecture gvp \
  --fusion-mode early_fusion \
  --structure-dir "${TRAIN_DIR}" \
  --summary-csv "${TRAIN_CSV}" \
  --test-structure-dir "${TEST_DIR}" \
  --test-summary-csv "${TEST_CSV}" \
  --run-test-eval \
  --runs-dir "${RUNS_DIR}" \
  --run-name metal_gvp_early_fusion_seed42 \
  --seed 42 \
  --val-fraction 0.15 \
  --split-by pdbid \
  --selection-metric val_metal_balanced_acc \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 3e-5 \
  --weight-decay 1e-4 \
  --esm-embeddings-dir "${ESM_EMBEDDINGS_DIR}" \
  --node-feature-set conservative \
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
  --model-architecture gvp \
  --fusion-mode node_level_late_fusion \
  --structure-dir "${TRAIN_DIR}" \
  --summary-csv "${TRAIN_CSV}" \
  --test-structure-dir "${TEST_DIR}" \
  --test-summary-csv "${TEST_CSV}" \
  --run-test-eval \
  --runs-dir "${RUNS_DIR}" \
  --run-name metal_gvp_node_level_late_fusion_seed42 \
  --seed 42 \
  --val-fraction 0.15 \
  --split-by pdbid \
  --selection-metric val_metal_balanced_acc \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 3e-5 \
  --weight-decay 1e-4 \
  --esm-embeddings-dir "${ESM_EMBEDDINGS_DIR}" \
  --node-feature-set conservative
```

### 3.2 Hybrid fusion

```bash
PYTHONPATH=src ${PYTHON} src/train.py \
  --task metal \
  --model-architecture gvp \
  --fusion-mode hybrid \
  --structure-dir "${TRAIN_DIR}" \
  --summary-csv "${TRAIN_CSV}" \
  --test-structure-dir "${TEST_DIR}" \
  --test-summary-csv "${TEST_CSV}" \
  --run-test-eval \
  --runs-dir "${RUNS_DIR}" \
  --run-name metal_gvp_hybrid_seed42 \
  --seed 42 \
  --val-fraction 0.15 \
  --split-by pdbid \
  --selection-metric val_metal_balanced_acc \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 3e-5 \
  --weight-decay 1e-4 \
  --esm-embeddings-dir "${ESM_EMBEDDINGS_DIR}" \
  --node-feature-set conservative \
  --early-esm-dim 32 \
  --early-esm-dropout 0.2
```

### 3.3 Cross-modal attention

```bash
PYTHONPATH=src ${PYTHON} src/train.py \
  --task metal \
  --model-architecture gvp \
  --fusion-mode cross_modal_attention \
  --structure-dir "${TRAIN_DIR}" \
  --summary-csv "${TRAIN_CSV}" \
  --test-structure-dir "${TEST_DIR}" \
  --test-summary-csv "${TEST_CSV}" \
  --run-test-eval \
  --runs-dir "${RUNS_DIR}" \
  --run-name metal_gvp_cross_modal_attention_seed42 \
  --seed 42 \
  --val-fraction 0.15 \
  --split-by pdbid \
  --selection-metric val_metal_balanced_acc \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 3e-5 \
  --weight-decay 1e-4 \
  --esm-embeddings-dir "${ESM_EMBEDDINGS_DIR}" \
  --node-feature-set conservative \
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
  --run-test-eval \
  --runs-dir "${RUNS_DIR}" \
  --run-name ec_level1_only_gvp_seed42 \
  --seed 42 \
  --val-fraction 0.15 \
  --split-by pdbid \
  --selection-metric val_ec_group_balanced_acc \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 3e-5 \
  --weight-decay 1e-4 \
  --node-feature-set conservative
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
  --run-test-eval \
  --runs-dir "${RUNS_DIR}" \
  --run-name ec_level1_only_esm_seed42 \
  --seed 42 \
  --val-fraction 0.15 \
  --split-by pdbid \
  --selection-metric val_ec_group_balanced_acc \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 3e-5 \
  --weight-decay 1e-4 \
  --esm-embeddings-dir "${ESM_EMBEDDINGS_DIR}" \
  --node-feature-set conservative
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
  --run-test-eval \
  --runs-dir "${RUNS_DIR}" \
  --run-name ec_level1_gvp_late_fusion_seed42 \
  --seed 42 \
  --val-fraction 0.15 \
  --split-by pdbid \
  --selection-metric val_ec_group_balanced_acc \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 3e-5 \
  --weight-decay 1e-4 \
  --esm-embeddings-dir "${ESM_EMBEDDINGS_DIR}" \
  --node-feature-set conservative
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
  --run-test-eval \
  --runs-dir "${RUNS_DIR}" \
  --run-name joint_level1_only_gvp_seed42 \
  --seed 42 \
  --val-fraction 0.15 \
  --split-by pdbid \
  --selection-metric val_joint_balanced_acc \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 3e-5 \
  --weight-decay 1e-4 \
  --node-feature-set conservative
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
  --run-test-eval \
  --runs-dir "${RUNS_DIR}" \
  --run-name joint_level1_gvp_late_fusion_seed42 \
  --seed 42 \
  --val-fraction 0.15 \
  --split-by pdbid \
  --selection-metric val_joint_balanced_acc \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 3e-5 \
  --weight-decay 1e-4 \
  --esm-embeddings-dir "${ESM_EMBEDDINGS_DIR}" \
  --node-feature-set conservative
```

---

## 6. Repeating the strongest runs with multiple seeds

After the first pass, repeat only the most promising models with multiple seeds.

Recommended seeds:

```text
42, 123, 777
```

Example for the GVP + late fusion metal model:

```bash
for SEED in 42 123 777; do
  PYTHONPATH=src ${PYTHON} src/train.py \
    --task metal \
    --model-architecture gvp \
    --fusion-mode late_fusion \
    --structure-dir "${TRAIN_DIR}" \
    --summary-csv "${TRAIN_CSV}" \
    --test-structure-dir "${TEST_DIR}" \
    --test-summary-csv "${TEST_CSV}" \
    --run-test-eval \
    --runs-dir "${RUNS_DIR}" \
    --run-name "metal_gvp_late_fusion_seed${SEED}" \
    --seed "${SEED}" \
    --val-fraction 0.15 \
    --split-by pdbid \
    --selection-metric val_metal_balanced_acc \
    --epochs 50 \
    --batch-size 8 \
    --learning-rate 3e-5 \
    --weight-decay 1e-4 \
    --esm-embeddings-dir "${ESM_EMBEDDINGS_DIR}" \
    --node-feature-set conservative
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

## 8. Optional: exact PinMyMetal split only as secondary/reference

The exact PinMyMetal split should not be the main final held-out result if train/test overlap exists.

If it is used, label the run names clearly, for example:

```text
metal_exact_split_gvp_late_fusion_seed42_reference_only
```

Do not mix exact-split runs and non-overlapped-split runs in the same final comparison without clearly labeling the split type.
