# Generalized Metal 5-Fold Cross-Validation & Benchmarking Guide

## 1. Overview & Purpose

DeepMzyme supports rigorous 5-fold cross-validation and benchmarking across **arbitrary dataset folder splits** using a single, unified orchestrator:
- **Python CLI Orchestrator:** [`scripts/run_metal_5fold_cv.py`](../scripts/run_metal_5fold_cv.py)
- **Convenience Shell Script:** [`scripts/run_metal_5fold_cv.sh`](../scripts/run_metal_5fold_cv.sh)

This orchestrator eliminates manual per-dataset scripting by providing automatic dataset directory resolution, summary CSV discovery, cross-validation fold splitting, model training execution, held-out test evaluation, out-of-fold (OOF) cross-validation aggregation, soft-voting ensemble prediction, and markdown report generation.

Crucially, it supports **both representation granularities** via clean, separate command-line flags:
1. **Ion-centered examples** (`--metal-example-unit ion`)
2. **Clustered-pocket examples** (`--metal-example-unit pocket`)

---

## 2. Representation Granularities: Ion vs. Pocket

Use [Plan.md's metal example terminology](../Plan.md#metal-example-terminology)
when interpreting either mode. The table describes the example unit, which is
separate from the label scheme and validation grouping.

| Feature | Ion-centered (`--metal-example-unit ion`) | Clustered-pocket (`--metal-example-unit pocket`) |
| :--- | :--- | :--- |
| **Example definition** | One example per eligible target ion | One example per eligible parsed metal cluster |
| **Multinuclear sites** | A retained three-ion cluster produces three examples, with potentially overlapping or identical residue neighborhoods | That cluster produces one record containing all three metal coordinates |
| **Extraction coordinates** | The target ion's coordinate | All metal coordinates in the cluster |
| **Residue neighborhood** | Any residue atom within 10 Å of the target ion | Any residue atom within 10 Å of any cluster ion; not a centroid sphere |
| **Metal target** | One class per ion under the active label scheme | One class per eligible cluster under the active label scheme; not a multi-label target |

Both modes use `PocketRecord` and `pocket_id`. Ion examples additionally retain
`parent_pocket_id`; `split_by="pocket_id"` groups siblings by that parent,
whereas `pdbid` groups all examples from a PDB. Inspect the saved
`metal_example_unit` and split configuration, and report ion-example,
parent-pocket, and PDB-group counts separately. The source-bound PMM ion
comparison uses frozen PDB-grouped folds; ion mode alone does not establish
paper-protocol equivalence.

---

## 3. Supported Datasets

The runner automatically resolves and parses any of the following dataset splits located under `DeepMzyme_Data/` or custom filesystem paths:

- `train_and_test_sets_structures_exact_pinmymetal` (Exact Nature Comms 2025 PinMyMetal dataset)
- `train_and_test_sets_structures_non_overlapped_pinmymetal` (Historical sequence non-overlapped split)
- `train_and_test_sets_structures_common_pdbid_70_30_pinmymetal` (Structure-redundancy 70/30 split)
- `train_and_test_sets_structures_zenodo_pmm_exact` (Reconstructed official Zenodo 9,398 ion sites)
- `CLEAN_30_train_test_split_0` through `CLEAN_30_train_test_split_4` (CLEAN 30% sequence identity splits)
- `CARE_30_*` (CARE benchmark splits)
- Any custom directory containing `train/` (and optional `test/`) with a valid summary CSV.

---

## 4. Evaluated Model Architectures

The runner benchmarks three core model families:

1. **Multimodal Late Fusion (`benchmark_enhanced_gvp_esmc`)**:
   - Enhanced Geometric Vector Perceptron (GVP) Graph Neural Network on structural pocket graph.
   - ESM-C 300M (dim 960) representations.
   - Late-fusion MLP combining geometric structure and protein language sequence embeddings.
2. **Sequence-Only Baseline (`benchmark_only_esm`)**:
   - Mean-pooled ESM-C embeddings passed through a classification MLP.
3. **Structure-Only Baseline (`benchmark_enhanced_only_gvp`)**:
   - Enhanced GVP GNN with raw Euclidean RBF distances and orientation vectors.

---

## 5. Ready-to-Use Commands

### A. Instant Dry-Run Validation (`--dry-run`)
Before launching long training jobs, use `--dry-run` to verify dataset layout discovery, summary CSV detection, ion site counts, and exact fold command strings:

```bash
# Verify metal ion-focused level:
python scripts/run_metal_5fold_cv.py \
  --dataset train_and_test_sets_structures_exact_pinmymetal \
  --metal-example-unit ion \
  --dry-run

# Verify clustered-pocket level:
python scripts/run_metal_5fold_cv.py \
  --dataset train_and_test_sets_structures_exact_pinmymetal \
  --metal-example-unit pocket \
  --dry-run

# Verify on any other dataset (e.g. Zenodo exact):
python scripts/run_metal_5fold_cv.py \
  --dataset train_and_test_sets_structures_zenodo_pmm_exact \
  --metal-example-unit ion \
  --dry-run
```

### B. Full 5-Fold Cross-Validation Campaign (All 3 Models)
Runs all 5 folds for all 3 models (15 training runs total), evaluates on held-out test, computes out-of-fold cross-validation metrics, evaluates soft-voting ensembles, and generates comparative markdown tables:

```bash
# Metal ion focused level
python scripts/run_metal_5fold_cv.py \
  --dataset train_and_test_sets_structures_exact_pinmymetal \
  --metal-example-unit ion \
  --epochs 50 \
  --batch-size 16 \
  --device cuda \
  --seed 42

# Clustered-pocket level
python scripts/run_metal_5fold_cv.py \
  --dataset train_and_test_sets_structures_exact_pinmymetal \
  --metal-example-unit pocket \
  --epochs 50 \
  --batch-size 16 \
  --device cuda \
  --seed 42
```

### C. Quick Single-Fold Validation (Smoke Test)
To quickly check training and evaluation on Fold 0 with 5 epochs:

```bash
python scripts/run_metal_5fold_cv.py \
  --dataset train_and_test_sets_structures_exact_pinmymetal \
  --metal-example-unit ion \
  --models benchmark_enhanced_only_gvp \
  --folds 0 \
  --epochs 5 \
  --batch-size 16 \
  --device cuda
```

### D. Convenience Shell Script Wrapper
A portable wrapper script is provided at `scripts/run_metal_5fold_cv.sh`:

```bash
# Syntax: bash scripts/run_metal_5fold_cv.sh <DATASET_NAME_OR_PATH> [ion|pocket] [cuda|cpu]

# Run ion-level on PinMyMetal exact:
bash scripts/run_metal_5fold_cv.sh train_and_test_sets_structures_exact_pinmymetal ion cuda

# Run pocket-level on PinMyMetal non-overlapped:
bash scripts/run_metal_5fold_cv.sh train_and_test_sets_structures_non_overlapped_pinmymetal pocket cuda
```

---

## 6. Output Artifacts and Reports

All outputs are saved to isolated, non-colliding directories:
`runs/benchmark_<dataset_id>_<metal_example_unit>_5fold/`

For example:
- `runs/benchmark_train_and_test_sets_structures_exact_pinmymetal_ion_5fold/`
- `runs/benchmark_train_and_test_sets_structures_exact_pinmymetal_pocket_5fold/`

### Generated Artifacts:
1. **Per-Fold Training Artifacts:**
   - `<model>_fold<k>/best_checkpoint.pt`: Best model weights evaluated on validation fold.
   - `<model>_fold<k>/val_metrics.csv`: Complete epoch-by-epoch loss and accuracy metrics.
   - `<model>_fold<k>/test_report.json`: Classification report and metrics on held-out test set.
   - `<model>_fold<k>/test_predictions.pt`: Raw prediction probabilities and true labels on held-out test.
2. **Out-of-Fold (OOF) Aggregate Cross-Validation:**
   - `<model>_cv_metrics.json`: Aggregated 5-fold CV metrics (5-class and collapsed-4 balanced accuracies, precision, recall, F1).
3. **5-Fold Soft-Voting Ensemble:**
   - `<model>_5fold_ensemble_report.json`: Ensemble metrics obtained by averaging softmax prediction probabilities from all 5 fold models on the test set.
4. **Summary Comparison Table:**
   - `benchmark_5fold_comparison_table.md`: Comprehensive markdown table presenting:
     - 5-Fold Cross-Validation Validation Balanced Accuracy
     - 5-Fold Ensemble Held-Out Test Balanced Accuracy
     - Per-class recalls for Manganese (Mn), Zinc (Zn), Group VIII (Fe/Co/Ni), and Copper (Cu)
     - Direct comparison deltas against published PinMyMetal and Metal3D baselines.

---

## 7. Built-In Robustness & Safety Guarantees

1. **Strict Run Isolation:** Output directories are segregated by both dataset name and example unit (`_ion_` vs `_pocket_`), preventing overwriting or mixing of results.
2. **PyTorch Geometric Compatibility:** Test evaluator batches graph objects with `torch_geometric.loader.DataLoader` to prevent collation type errors.
3. **MKL Environment Threading:** Sets `MKL_THREADING_LAYER=GNU` in python processes and subprocess environments to eliminate Intel MKL / GNU OpenMP runtime conflicts.
4. **Null-Safe Metrics Parsing:** Uses `_safe_float` and `_pct` helpers so runs with zero representations for rare classes in specific folds do not cause parsing failures.
5. **Multi-Location Dataset Resolution:** Checks repository root `DeepMzyme_Data/`, local working directories, and external drives automatically.
