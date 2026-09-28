# Preserved STATUS history

Historical text, copied verbatim before the Job B rewrite. Old next actions,
permissions, counts and claims are dated evidence, not current instructions.
Current authority: [EXPERIMENT_STATUS.md](../../../../EXPERIMENT_STATUS.md).
Source ranges and hashes: [preservation manifest](../../../../docs/archive/consolidation_2026-09/inventory/job_b_status_preservation.json).
Literal blocks preserve original relative link spelling; use this folder’s
README for navigable evidence links. No historical error has been silently fixed.

## exact-001

Source: `3c0f80c:EXPERIMENT_STATUS.md:399-411`.

```text
**Exact PinMyMetal 5-Fold Cross-Validation Benchmark (2026-09-23, Exploratory):** Exploratory benchmark under `--train-val-split-by pocket_id --n-folds 5` across three comparative architectures: Sequence-Only ESM-C (`benchmark_only_esm`), Structure-Only Enhanced GVP (`benchmark_enhanced_only_gvp`), and Multimodal Late Fusion (`benchmark_enhanced_gvp_esmc`). *(Caveat: Not a like-for-like comparison with PinMyMetal; test set opened 2026-09-22/23; see [EXACT doc qualification](docs/EXACT_PINMYMETAL_5FOLD_CV_REPRODUCIBILITY.md) and [test-use ledger](docs/DATASETS.md#test-use-ledger)).*
- **Table 1 (5-Fold CV Val Bal Acc — Unadjusted Epoch Maxima vs PinMyMetal Fig 2a: 75.08%)**:
  *(Selected-checkpoint out-of-fold values: Late Fusion **79.84%**, ESM-C **78.98%**, Enhanced GVP **73.87%**; see [reporting and split audit](docs/agents_report/RESUMED_REVIEWS_HANDOFF_20260923.md#6-substantive-findings-already-established-cached-re-checkable))*
  - `benchmark_only_esm`: **80.29% ± 3.26%** (epoch max; OOF: 78.98%; historical arithmetic delta +5.21 pp)
  - `benchmark_enhanced_only_gvp`: **74.42% ± 2.39%** (epoch max; OOF: 73.87%; historical arithmetic delta -0.66 pp)
  - `benchmark_enhanced_gvp_esmc`: **80.22% ± 3.31%** (epoch max; OOF: 79.84%; historical arithmetic delta +5.14 pp)
- **Table 2 (Held-Out Test Set 352 Pockets vs PinMyMetal Fig 2b: 67.85% & Metal3D Fig 2c: 61.70%)**:
  - `benchmark_only_esm` 5-Fold Ensemble: **77.37%** (+9.52 pp vs PMM, +15.67 pp vs Metal3D)
  - `benchmark_enhanced_only_gvp` 5-Fold Ensemble: **78.03%** (+10.18 pp vs PMM, +16.33 pp vs Metal3D)
  - `benchmark_enhanced_gvp_esmc` 5-Fold Ensemble: **79.92%** (+12.07 pp vs PMM, +18.22 pp vs Metal3D)
  - Multimodal per-class recalls: Cu: **89.7%** (+30.3 pp vs PMM 59.4%), Zn: **88.0%** (+22.1 pp vs PMM 65.9%), Group VIII: **74.6%** (+17.1 pp vs PMM 57.5%).
Complete documentation is available in [`docs/notebook_outputs/summaries/summary_pinmymetal_5fold_three_models_20260923.md`](docs/notebook_outputs/summaries/summary_pinmymetal_5fold_three_models_20260923.md) and [`docs/EXACT_PINMYMETAL_5FOLD_CV_REPRODUCIBILITY.md`](docs/EXACT_PINMYMETAL_5FOLD_CV_REPRODUCIBILITY.md).

```

## exact-002

Source: `9a25f36^:EXPERIMENT_STATUS.md:12-13`.

```text
**Exact PinMyMetal 5-Fold Cross-Validation Benchmark (2026-09-22):** Direct replication of the Nature Communications (2025) PinMyMetal evaluation protocol (`--train-val-split-by pocket_id --n-folds 5`) using the multimodal Enhanced GVP + ESM-C architecture (`benchmark_enhanced_gvp_esmc`). On the exact PinMyMetal dataset (1,597 training pockets across 1,483 structures, 352 held-out test pockets across 316 structures), Fold 0 achieved **81.27%** collapsed-4 validation balanced accuracy (**77.62%** held-out test balanced accuracy) and Fold 1 achieved **75.88%** collapsed-4 validation balanced accuracy (**74.21%** held-out test balanced accuracy), achieving a two-fold mean validation balanced accuracy of **78.58%** (matching/exceeding PinMyMetal Fig 2a's reported ~75.08%). Complete deterministic CLI commands, hyperparameters, per-class recall tables, and reproduction instructions are documented in [`docs/EXACT_PINMYMETAL_5FOLD_CV_REPRODUCIBILITY.md`](docs/EXACT_PINMYMETAL_5FOLD_CV_REPRODUCIBILITY.md) and automated via [`scripts/run_exact_pinmymetal_5fold_cv.py`](scripts/run_exact_pinmymetal_5fold_cv.py).

```
