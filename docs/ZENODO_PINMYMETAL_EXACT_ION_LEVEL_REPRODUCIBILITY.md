# Zenodo PinMyMetal Exact Dataset & 5-Fold Cross-Validation Reproducibility Guide

## 1. Overview & Motivation

The **PinMyMetal (PMM)** benchmark is the primary published standard for predicting catalytic metal ion binding sites (Manganese, Zinc, Copper, and Class VIII transition metals) in enzyme structures.

Historically, earlier dataset extractions merged nearby metal ions within proximity into single pocket clusters. However, **PinMyMetal and the official Zenodo `classmodel_train_set` / `classmodel_test_set` supervise each individual metal ion center (`(chain, resseq)`) separately**. In multinuclear sites (e.g. trinuclear zinc centers in alkaline phosphatase `1a0e`), each target has its own coordinate; the selected residue neighborhoods can overlap or be identical.

This document serves as the complete, authoritative guide to:
1. Replicating the exact Zenodo PinMyMetal dataset (99.89% exact reconstruction).
2. Understanding the **individual metal ion-level representation contract** (`--metal-example-unit ion`).
3. Verifying dataset integrity and non-destructive coexistence with existing workflows.
4. Executing the 5-fold cross-validation benchmark and held-out test evaluation on GPU (locally or on Google Colab).
5. Interpreting results against published PinMyMetal and Metal3D baselines.

---

## 2. Dataset Reconstruction & Provenance

### Source & High-Fidelity Recovery
- **Source Tables:** Extracted row-by-row from the official Zenodo archive (`classmodel_train_set.csv` and `classmodel_test_set.csv`).
- **Exact Recovery Rates:**
  - **Train set:** **7,911 / 7,920** source rows (**99.89%** fidelity) across 6,443 PDB structure files.
  - **Test set:** **1,487 / 1,488** source rows (**99.93%** fidelity) across 1,281 PDB structure files.
  - **Total:** **9,398 / 9,408** source rows (**99.89%** overall).
  - *Note on remaining 10 rows:* 9 rows in train and 1 row in test corresponded to obsolete/withdrawn PDB IDs or non-canonical/DNA-only complexes lacking protein coordinates.

### Class Distribution (Collapsed 4-Class & 5-Class)
| Metal Category | Train Count | Test Count | Zenodo Role |
| :--- | :--- | :--- | :--- |
| **Manganese (Mn)** | 2,586 | 167 | Target metal class |
| **Copper (Cu)** | 400 | 64 | Target metal class |
| **Zinc (Zn)** | 2,300 | 1,004 | Target metal class |
| **Class VIII (Fe, Co, Ni)** | 2,625 | 252 | Target metal class (Iron/Cobalt/Nickel) |
| **Total** | **7,911** | **1,487** | **9,398 Total Sites** |

### Hugging Face Distribution & Verification
The reconstructed dataset is bundled and hosted permanently on Hugging Face:
- **Repository:** [`GMBioinformatics/DeepMzyme`](https://huggingface.co/datasets/GMBioinformatics/DeepMzyme)
- **Direct Download URL:**
  ```text
  https://huggingface.co/datasets/GMBioinformatics/DeepMzyme/resolve/main/train_and_test_sets_structures_zenodo_pmm_exact.tar.gz
  ```
- **Archive Size:** 1.1 GB compressed (contains all 7,724 cleaned PDB structures and metadata manifests)
- **SHA-256 Checksum:**
  ```text
  24903f120462aacb90b43c4af97f4f08d61f1847d12636606fcc66e6351db296
  ```

---

## 3. Directory Layout & Manifest Contracts

The dataset is organized additively under `train_and_test_sets_structures_zenodo_pmm_exact`:

```
train_and_test_sets_structures_zenodo_pmm_exact/
├── train/
│   ├── structures/                       # 6,443 PDB structure files
│   ├── final_data_summarazing_table.csv  # Standard DeepMzyme summary table
│   ├── site_manifest.csv                 # Exact Zenodo site metadata & coordinates
│   └── structure_manifest.csv            # Structure checksums & sizes
├── test/
│   ├── structures/                       # 1,281 PDB structure files
│   ├── final_data_summarazing_table.csv  # Standard DeepMzyme summary table
│   ├── site_manifest.csv                 # Exact Zenodo site metadata & coordinates
│   └── structure_manifest.csv            # Structure checksums & sizes
├── site_crosswalk.csv                    # Complete audit mapping all 9,408 source rows
└── coverage.json                         # Exact reconstruction counts & metrics
```

### Manifest Columns
`site_manifest.csv` preserves the ground-truth Zenodo identity:
- `source_uid`: Unique identifier per source row (e.g. `train_row_000042`)
- `pdbid`: 4-letter PDB code (lowercase)
- `chain`: PDB chain identifier of the metal ion
- `resseq`: Residue sequence number of the metal ion
- `chain_resi`: Combined chain + residue (e.g. `A_501`)
- `element`: Element chemical symbol (`MN`, `CU`, `ZN`, `FE`, `CO`, `NI`)
- `metaltype`: Metal classification string
- `label_metal`: Canonical metal label
- `structure_name`: Standardized file name (`<pdbid>.pdb`)
- `residueid_ion`: Canonical residue identifier matching Zenodo

---

## 4. The Metal Ion-Level Representation Contract

### Pocket vs. Ion Representation

Canonical definitions and identifier semantics are in
[Plan.md's metal example terminology](../Plan.md#metal-example-terminology).

- **Pocket-Level (`--metal-example-unit pocket`):** Creates one example per eligible metal cluster, retaining its metal coordinates and one target class under the active label scheme. Residues are selected within 10 Å of any cluster ion, not from a centroid sphere. This is not a multi-label target.
- **Ion-Level (`--metal-example-unit ion`):** **MANDATORY FOR THIS BENCHMARK.**
  - Every retained target ion forms a separate example; sibling examples remain correlated and must stay in the same validation group.
  - In multinuclear sites (e.g., trinuclear zinc site in `1a0e`), three distinct `PocketRecord` examples are created:
    - Center 1: `(A, 501)` -> Zn
    - Center 2: `(A, 502)` -> Zn
    - Center 3: `(A, 503)` -> Zn
  - Each ion example includes residues with any atom within 10 Å of that ion's exact 3D Cartesian coordinate $(x, y, z)$.
  - The prediction unit is the individual ion. This alone does not certify the same cohort, inputs, folds, or metrics as the published benchmark.

`PocketRecord` and record `pocket_id` are reused for both modes. Ion examples
retain `parent_pocket_id` for their original cluster; the record ID identifies
the individual example. Inspect `metal_example_unit` in the saved run
configuration and distinguish ion-example counts from parent-pocket and PDB
counts. With `split_by="pocket_id"`, siblings share their parent's fold;
the source-bound PMM ion comparison requires the stricter frozen PDB-grouped
folds. Historical pocket-unit results retain their original meaning.

---

## 5. Automated Verification Suite

To verify that the dataset and parsing pipeline uphold the ion-level contract and conform to all expectations:

```bash
python scripts/verify_zenodo_pmm_ion_dataset.py \
  --data-dir /path/to/DeepMzyme_Data/train_and_test_sets_structures_zenodo_pmm_exact
```

### Verification Checks:
1. **Manifest Integrity:** Verifies `site_manifest.csv` contains all expected columns and valid rows.
2. **PDB Structure Existence:** Asserts that every PDB referenced in the manifest exists on disk.
3. **Multinuclear Disentanglement:** Inspects trinuclear zinc structure `1a0e` and validates that exactly 3 distinct examples are produced with distinct ion centers.
4. **Label Scheme Alignment:** Validates 5-class and collapsed-4 label mappings against Zenodo ground truth.

---

## 6. Execution Guide (Google Colab & Local GPU)

### One-Command Full Pipeline Replication
The easiest and most robust way to reproduce the benchmark on any environment (local GPU workstation or Colab) is via the master reproduction script:

```bash
bash scripts/reproduce_zenodo_pmm_benchmark.sh
```

This single command:
1. Automatically verifies whether the dataset is present; if not, fetches `train_and_test_sets_structures_zenodo_pmm_exact.tar.gz` from Hugging Face.
2. Cryptographically validates the archive SHA-256 (`24903f120462aacb90b43c4af97f4f08d61f1847d12636606fcc66e6351db296`).
3. Extracts and lays out the structures and manifests.
4. Executes `scripts/verify_zenodo_pmm_ion_dataset.py` to confirm 100% structure integrity and multinuclear site separation.
5. Launches the 5-fold cross-validation campaign across all comparative models.
6. Prints the benchmark comparison table comparing results directly against published PinMyMetal and Metal3D metrics.

To run only a single fold (e.g. Fold 0 for rapid validation) or custom epochs:
```bash
FOLDS="0" EPOCHS=50 bash scripts/reproduce_zenodo_pmm_benchmark.sh
```

### Manual Step-by-Step Launch on Google Colab
On an active Colab session with GPU (e.g. NVIDIA L4):

1. **Bootstrap & Download Dataset:**
   ```bash
   mkdir -p /content/DeepMzyme_Data/DeepMzyme_Data
   cd /content/DeepMzyme_Data/DeepMzyme_Data
   wget -q -c https://huggingface.co/datasets/GMBioinformatics/DeepMzyme/resolve/main/train_and_test_sets_structures_zenodo_pmm_exact.tar.gz
   tar -xzf train_and_test_sets_structures_zenodo_pmm_exact.tar.gz
   ```

2. **Run Verification:**
   ```bash
   cd /content/DeepMzyme
   python scripts/verify_zenodo_pmm_ion_dataset.py \
     --data-dir /content/DeepMzyme_Data/DeepMzyme_Data/train_and_test_sets_structures_zenodo_pmm_exact
   ```

3. **Launch the Benchmark Runner:**
   ```bash
   nohup python3 -u /content/DeepMzyme/scripts/run_zenodo_pmm_exact_5fold_cv.py \
     --models benchmark_enhanced_only_gvp benchmark_only_esm benchmark_enhanced_gvp_esmc \
     --folds 0 1 2 3 4 \
     --n-folds 5 \
     --data-root /content/DeepMzyme_Data/DeepMzyme_Data \
     --runs-dir /content/runs/runs_zenodo_pmm_exact \
     --epochs 50 \
     --batch-size 16 \
     --device cuda > /content/zenodo_5fold_execution.log 2>&1 &
   ```

4. **Monitor Progress:**
   ```bash
   tail -f /content/zenodo_5fold_execution.log
   ```

### Local Execution (on Host)
To run locally on `/media/mechti/Data1`:
```bash
bash scripts/run_zenodo_pmm_exact.sh
```
Or directly:
```bash
python scripts/run_zenodo_pmm_exact_5fold_cv.py \
  --models benchmark_enhanced_only_gvp benchmark_only_esm benchmark_enhanced_gvp_esmc \
  --folds 0 1 2 3 4 \
  --n-folds 5 \
  --data-root /media/mechti/Data1/DeepMzyme_Data \
  --runs-dir /media/mechti/Data1/DeepMzyme_Data/runs_zenodo_pmm_exact \
  --epochs 50 \
  --batch-size 16 \
  --device cuda
```

---

## 7. Model Architectures & Configurations

The runner executes 3 core models across all 5 folds:

| Model Key | Description | Architecture | Fusion | Features | Learning Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `benchmark_enhanced_only_gvp` | Structure-only GVP-GNN | `only_gvp` | None | Raw RBF distances + geometric vectors | `3e-4` |
| `benchmark_only_esm` | Sequence-only ESM-C | `only_esm` | None | Pre-extracted ESM-C embeddings | `3e-5` |
| `benchmark_enhanced_gvp_esmc` | Multimodal Late Fusion | `gvp` | `late_fusion` | GVP structure + ESM-C embeddings | `3e-5` (GVP `3e-4`) |

---

## 8. Benchmark Comparison Baselines

The runner automatically compiles fold-by-fold results, calculates OOF validation balanced accuracy, evaluates each fold model on the held-out test set, computes soft-voting ensemble predictions across all 5 folds, and compares against published literature baselines:

| Model / Benchmark | Source | Evaluation Type | Collapsed-4 Balanced Accuracy | Mn Recall | Zn Recall | Class VIII Recall | Cu Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PinMyMetal (Published)** | PinMyMetal Fig 2a | 5-Fold CV (OOF) | **75.08%** | 90.3% | 73.8% | 73.3% | 62.9% |
| **PinMyMetal (Published)** | PinMyMetal Fig 2b | Held-Out Test Set | **67.85%** | 88.6% | 65.9% | 57.5% | 59.4% |
| **Metal3D (Published)** | External Benchmark (Fig 2c) | Held-Out Test Set | **61.70%** | 63.8% | 64.4% | 61.6% | 57.0% |
| **DeepMzyme Enhanced GVP** | This Benchmark | 5-Fold CV & Test Ensemble | *Computed upon completion* | - | - | - | - |
| **DeepMzyme Only ESM** | This Benchmark | 5-Fold CV & Test Ensemble | *Computed upon completion* | - | - | - | - |
| **DeepMzyme GVP + ESM-C** | This Benchmark | 5-Fold CV & Test Ensemble | *Computed upon completion* | - | - | - | - |

---

## 9. Non-Destructive Coexistence Policy (Zero Clobbering)

As strictly mandated, all additions are 100% additive:
- **Legacy Datasets:** `train_and_test_sets_structures_exact_pinmymetal` and all existing datasets remain completely intact and untouched.
- **Legacy Scripts:** `scripts/run_exact_pinmymetal_5fold_cv.py` and all earlier training workflows remain unchanged.
- **Disk Safety:** Checkpoints and heavy assets are redirected to `/media/mechti/Data1` locally or `/content` on Colab, strictly protecting the host `/home` partition.

---

## 10. Computational Complexity & Runtime Timing Breakdown

Empirically measured on Google Colab (NVIDIA L4 GPU, 24GB VRAM, 2 vCPU):

| Scope | Stage | Operations | Hardware | Estimated Duration |
| :--- | :--- | :--- | :--- | :--- |
| **Initial Prep (Fold 0)** | Structure parsing & 3D k-NN graphs | 6,443 PDBs -> 6,328 train + 1,583 val graphs | CPU | ~9–11 minutes |
| **Training (Fold 0)** | 50 Epochs forward + backward passes | 396 batches/epoch * 50 epochs = 19,800 steps | GPU (L4) | ~30–33 minutes (~38s/epoch) |
| **Test Eval (Fold 0)** | Test set parsing & inference | 1,281 PDBs -> 1,487 test sites | CPU + GPU | ~2 minutes |
| **Total (1 Fold)** | **Fold 0 Complete + Held-Out Test Report** | **Full Fold 0 Pipeline** | **End-to-End** | **~42–46 minutes** |
| **Full 5-Fold (1 Model)** | Folds 0, 1, 2, 3, 4 + 5-Fold Soft Voting | 5 Folds * 50 Epochs + Ensemble Test Eval | GPU (L4) | **~3.2–3.5 hours** |
| **Full 3-Model Benchmark** | 3 Architectures * 5 Folds (15 total runs) | Complete Nature Communications Replication | GPU (L4) | **~9.5–10 hours** |

