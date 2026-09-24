# Exact PinMyMetal 5-Fold Cross-Validation Reproducibility Playbook

**Date of Execution & Documentation:** September 22, 2026  
**Target Benchmark:** Nature Communications (2025) PinMyMetal Transition Metal Benchmark (Figure 2a)  
**Primary Architecture:** Multimodal Late Fusion — Enhanced Geometric Vector Perceptron (GVP) + ESM-C Embeddings (`benchmark_enhanced_gvp_esmc`)  
**Split Protocol:** 5-Fold Tabular Cross-Validation Stratified by `pocket_id` (`--train-val-split-by pocket_id --n-folds 5`)

---

## 1. Executive Summary & Purpose

This document provides exhaustive, deterministic reproduction specifications for running the exact **PinMyMetal 5-Fold Cross-Validation Benchmark** within DeepMzyme.

**Source-site audit qualification (2026-09-23):** Here “exact PinMyMetal”
names DeepMzyme's historical PDB-projection/catalytic-pocket dataset. It does
not mean the same PMM source rows, individual ion anchors, or published fold
membership. The [source audit](PMM_SOURCE_AUDIT.md) found 7,920/1,488 PMM
train/test source rows versus this document's 1,597/352 loaded pockets; PMM
row-fold IDs have not been recovered. The PMM Fig. 2a numbers below are
historical context, not a paired or direct reproduction comparison. Existing
test access remains documented in [DATASETS.md](DATASETS.md).

Any researcher or AI agent reading this repository can replicate the exact runs, evaluation protocols, and per-fold performance metrics reported below.

### Key Empirical Findings
1. **Historical numerical context from PinMyMetal (Figure 2a)**:
   - In *PinMyMetal* (Nature Communications 2025, Fig. 2a), the authors report a headline cross-validation balanced accuracy of **~75.08%** across 4 collapsed metal classes (Mn: 90.3%, Zn: 73.8%, Group VIII: 73.3%, Cu: 62.9%) derived from **5-fold cross-validation on their training dataset**.
   - In two recorded DeepMzyme folds on the PDB-projection cohort, collapsed-four validation balanced accuracy was **81.27% (Fold 0) and 75.88% (Fold 1), mean 78.58%**. Different row cohorts and unverified fold identity prevent a direct PMM baseline claim.
2. **Resolution of the "Generalization Gap"**:
   - In strict held-out test splits without pocket overlap, test performance is ~58–60% due to the complete absence of homology.
   - In pocket-stratified cross-validation (`pocket_id`), validation balanced accuracy reaches **~78.6%**, and individual-fold test set balanced accuracy reaches **74.2%–77.6%**. This confirms that the tabular pocket split preserves shared local geometric motifs between train and validation partitions.

---

## 2. Experimental Setup & Exact Architecture

### Model Configuration (`benchmark_enhanced_gvp_esmc`)
- **Backbone**: Enhanced GVP (Geometric Vector Perceptron) Graph Neural Network
- **Sequence Features**: ESM-C 600M residue-level embeddings (dimension 1152)
- **External Geometric Features**: Complete physicochemical & coordination features (`updated_feature_extraction`)
- **Fusion Mode**: Late Fusion (`--fusion-mode late_fusion`)
  - GVP Learning Rate: `3e-4`
  - Fusion / MLP Learning Rate: `3e-5`
  - Optimizer: AdamW with weight decay
  - Distance Cutoff: `8.0 Å`
  - Distance Feature Representation: Raw Euclidean distances with Gaussian RBF (`--rbf-use-raw-distances`)
- **Training Epochs**: `50` per fold
- **Batch Size**: `16`
- **Random Seed**: `42` (deterministic split seed and model initialization)

---

## 3. Dataset Paths & Data Verification

The historical DeepMzyme `exact_pinmymetal` PDB-projection dataset consists of transition metal binding catalytic pockets partitioned as follows:
- **Training Set Directory**: `DeepMzyme_Data/train_and_test_sets_structures_exact_pinmymetal/train`
  - Summary CSV: `final_data_summarazing_table_transition_metals_only_catalytic.csv` (1,597 total training pockets across 1,483 PDB structures)
  - Train distribution across 5 folds: 1,272 train pockets / 325 validation pockets per fold
- **Held-Out Test Set Directory**: `DeepMzyme_Data/train_and_test_sets_structures_exact_pinmymetal/test`
  - Summary CSV: `final_data_summarazing_table_transition_metals_only_catalytic.csv` (352 test pockets across 316 PDB structures)
- **ESM Embeddings Directory**: `DeepMzyme_Data/esm_embeddings` (2,946 pre-computed `.pt` tensors matching structure identifiers)
- **External Features Directory**: `DeepMzyme_Data/updated_feature_extraction`

---

## 4. Empirical Benchmark Results (Recorded September 22, 2026)

### Fold-by-Fold Performance Table

The PMM paper column is unpaired historical context. The mean uses only the
two completed DeepMzyme folds listed here, not all five folds.

| Metric Scope | Fold 0 | Fold 1 | **Mean (Folds 0–1)** | PinMyMetal (Fig 2a) |
| :--- | :---: | :---: | :---: | :---: |
| **Validation Balanced Acc (5-Class)** | **75.54%** | **69.51%** | **72.53%** | N/A (4-class only) |
| **Validation Balanced Acc (Collapsed-4)** | **81.27%** | **75.88%** | **78.58%** | **75.08%** |
| **Validation Best Epoch** | Epoch 41 | Epoch 46 | — | — |
| **Held-Out Test Balanced Acc (5-Class)** | **71.68%** | **69.67%** | **70.68%** | — |
| **Held-Out Test Balanced Acc (Collapsed-4)** | **77.62%** | **74.21%** | **75.92%** | — |
| **Held-Out Test Raw Accuracy** | **75.57%** | **73.58%** | **74.58%** | — |

### Per-Class Held-Out Test Recall (Collapsed-4: Mn, Zn, Group VIII, Cu)

| Class | Fold 0 Test Recall | Fold 1 Test Recall | **Mean Test Recall** | PinMyMetal Fig 2a CV |
| :--- | :---: | :---: | :---: | :---: |
| **Copper (Cu)** | 89.66% | 82.76% | **86.21%** | 62.9% |
| **Zinc (Zn)** | 85.54% | 82.53% | **84.04%** | 73.8% |
| **Group VIII (Fe / Co / Ni)** | 69.01% | 67.61% | **68.31%** | 73.3% |
| **Manganese (Mn)** | 66.28% | 63.95% | **65.12%** | 90.3% |

---

## 5. Exact Commands for Deterministic Reproduction

### Automated 5-Fold Cross-Validation Script
An automated orchestrator is provided at `scripts/run_exact_pinmymetal_5fold_cv.py`.

To run the complete 5-fold cross-validation pipeline locally or on a GPU cluster:

```bash
python scripts/run_exact_pinmymetal_5fold_cv.py \
    --data-root DeepMzyme_Data \
    --runs-dir runs/benchmark_exact_pinmymetal_5fold \
    --folds 0 1 2 3 4 \
    --epochs 50 \
    --batch-size 16 \
    --device cuda \
    --seed 42
```

### Running Individual Folds via Direct CLI
To reproduce any specific fold directly using `src/train.py`:

```bash
# Fold 0 Reproduction Command
python src/train.py \
    --task metal \
    --metal-label-scheme five_class \
    --structure-dir DeepMzyme_Data/train_and_test_sets_structures_exact_pinmymetal/train \
    --summary-csv DeepMzyme_Data/train_and_test_sets_structures_exact_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv \
    --external-feature-source updated \
    --external-features-root-dir DeepMzyme_Data/updated_feature_extraction \
    --esm-embeddings-dir DeepMzyme_Data/esm_embeddings \
    --runs-dir runs/benchmark_exact_pinmymetal_5fold \
    --run-name benchmark_enhanced_gvp_esmc_fold0 \
    --model-architecture gvp \
    --fusion-mode late_fusion \
    --gvp-learning-rate 3e-4 \
    --rbf-use-raw-distances \
    --epochs 50 \
    --batch-size 16 \
    --learning-rate 3e-5 \
    --device cuda \
    --seed 42 \
    --n-folds 5 \
    --fold-index 0 \
    --train-val-split-by pocket_id \
    --test-structure-dir DeepMzyme_Data/train_and_test_sets_structures_exact_pinmymetal/test \
    --test-summary-csv DeepMzyme_Data/train_and_test_sets_structures_exact_pinmymetal/test/final_data_summarazing_table_transition_metals_only_catalytic.csv \
    --run-test-eval \
    --allow-final-refit-test-eval \
    --evaluation-protocol-id metal_pinmymetal_shared_config_dual_v1 \
    --held-out-overlap-policy exact_pinmymetal_secondary_reference \
    --final-test-result-role secondary_diagnostic_report \
    --final-test-selected-config-id benchmark_enhanced_gvp_esmc_fold0_exact_v1
```

*(To run Fold 1, 2, 3, or 4, substitute `--fold-index <k>` and `--run-name benchmark_enhanced_gvp_esmc_fold<k>` accordingly).*

---

## 6. Output Artifacts and Logs
For each fold `$k$`, the training process generates the following artifacts in `runs/benchmark_exact_pinmymetal_5fold/benchmark_enhanced_gvp_esmc_fold$k/`:
- `val_metrics.csv`: Per-epoch training loss, validation loss, raw accuracy, balanced accuracy, and collapsed-4 balanced accuracy.
- `best_model_checkpoint.pt`: Checkpoint of model weights achieving highest validation balanced accuracy.
- `last_model_checkpoint.pt`: Checkpoint of model weights at Epoch 50.
- `test_report.json`: Comprehensive held-out test evaluation report including 5-class balanced accuracy, collapsed-4 balanced accuracy, raw accuracy, and per-class recalls.
- `dataset_summary.json`: Record of class distributions and split metadata.
- `run_metadata.json` & `run_config.json`: Full serialized runtime configuration for strict reproducibility.
