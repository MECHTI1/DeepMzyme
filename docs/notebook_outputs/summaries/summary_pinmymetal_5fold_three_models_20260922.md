# PinMyMetal Exact 5-Fold Cross-Validation Benchmark (Three Architectures)

> [!NOTE]
> **Superceded by Complete Benchmark Report**: The complete 5-fold cross-validation results across all three architectures (`benchmark_only_esm`, `benchmark_enhanced_only_gvp`, and `benchmark_enhanced_gvp_esmc`) are documented in [summary_pinmymetal_5fold_three_models_20260923.md](summary_pinmymetal_5fold_three_models_20260923.md).

**Date**: 2026-09-23  
**Protocol**: Nature Communications (2025) PinMyMetal Exact 5-Fold Cross-Validation Setup  
**Split Configuration**: `--train-val-split-by pocket_id --n-folds 5 --seed 42`  
**Dataset**:
- Training set: `DeepMzyme_Data/train_and_test_sets_structures_exact_pinmymetal/train` (1,597 pockets across 1,483 PDB structures)
- Held-out test set: `DeepMzyme_Data/train_and_test_sets_structures_exact_pinmymetal/test` (352 pockets across 316 PDB structures)
- External features: `DeepMzyme_Data/updated_feature_extraction`
- Embeddings: `DeepMzyme_Data/esm_embeddings` (ESM-C 300M, dim 960)

---

## Benchmark Results Summary

### Table 1: 5-Fold Cross-Validation Performance (Validation Folds vs PinMyMetal Fig 2a)

| Architecture | 5-Fold CV Val Bal Acc (5-class) | 5-Fold CV Val Bal Acc (Collapsed-4) | Mn Recall | Zn Recall | Group VIII Recall | Cu Recall | Delta vs PMM Fig 2a (75.08%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PinMyMetal Fig 2a Baseline** | - | **75.08%** | 90.3% | 73.8% | 73.3% | 62.9% | Baseline |
| **only_esm** | 74.89% ± 2.92% | **80.29% ± 3.26%** | 84.2% | 68.3% | 78.3% | 90.4% | **+5.21 pp** |
| **enhanced_only_gvp** | 67.40% ± 3.11% | **74.42% ± 2.39%** | 76.6% | 59.8% | 72.1% | 89.0% | **-0.66 pp** |
| **enhanced_gvp_esmc** | 74.51% ± 3.77% | **80.22% ± 3.31%** | 85.6% | 65.9% | 78.2% | 91.2% | **+5.14 pp** |

### Table 2: Held-Out Test Set Performance (352 Pockets vs PinMyMetal Fig 2b & Metal3D Fig 2c)

| Architecture | Test Mode | Test Raw Acc | Test Bal Acc (5-class) | Test Bal Acc (Collapsed-4) | Mn Recall | Zn Recall | Group VIII Recall | Cu Recall | Delta vs PMM Fig 2b (67.85%) | Delta vs Metal3D Fig 2c (61.70%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PinMyMetal Fig 2b Baseline** | Held-Out Test | - | - | **67.85%** | 88.6% | 65.9% | 57.5% | 59.4% | Baseline | +6.15 pp |
| **Metal3D Fig 2c Baseline** | External Test | - | - | **61.70%** | 63.8% | 64.4% | 61.6% | 57.0% | -6.15 pp | Baseline |
| **only_esm (Per-Fold Mean)** | 5-Fold Mean | 71.02% ± 2.50% | 67.57% ± 0.89% | **74.17% ± 1.90%** | - | - | - | - | +6.32 pp | +12.47 pp |
| **only_esm (5-Fold Ensemble)** | 5-Fold Ensemble | 75.57% | 71.48% | **77.37%** | 61.6% | 84.9% | 73.2% | 89.7% | **+9.52 pp** | **+15.67 pp** |
| **enhanced_only_gvp (Per-Fold Mean)** | 5-Fold Mean | 65.74% ± 4.30% | 61.92% ± 3.75% | **69.89% ± 2.66%** | - | - | - | - | +2.04 pp | +8.19 pp |
| **enhanced_only_gvp (5-Fold Ensemble)** | 5-Fold Ensemble | 74.72% | 68.68% | **78.03%** | 70.9% | 82.5% | 69.0% | 89.7% | **+10.18 pp** | **+16.33 pp** |
| **enhanced_gvp_esmc (Per-Fold Mean)** | 5-Fold Mean | 73.69% ± 1.64% | 70.12% ± 1.73% | **75.84% ± 1.46%** | - | - | - | - | +7.99 pp | +14.14 pp |
| **enhanced_gvp_esmc (5-Fold Ensemble)** | 5-Fold Ensemble | **77.84%** | **73.51%** | **79.92%** | 67.4% | **88.0%** | **74.6%** | **89.7%** | **+12.07 pp** | **+18.22 pp** |

See [summary_pinmymetal_5fold_three_models_20260923.md](summary_pinmymetal_5fold_three_models_20260923.md) for full breakdown.
