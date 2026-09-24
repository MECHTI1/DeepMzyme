# DeepMzyme Cloud Backup & Data Provenance Inventory

This document serves as the authoritative, permanent inventory of all cloud-backed files, datasets, trained model weights, benchmark checkpoints, and repository branches for the **DeepMzyme** project.

Last updated: **2026-09-24**.

---

## 1. Cloud Backup Architecture Summary

| Destination | Primary Host | Role & Content |
|---|---|---|
| **Hugging Face** | [`GMBioinformatics/DeepMzyme`](https://huggingface.co/datasets/GMBioinformatics/DeepMzyme) | Core dataset bundles (PMM, CLEAN, CARE), precomputed ESM embeddings, RING features, 5-fold CV trained model checkpoints, and source audit crosswalks. |
| **GitHub** | [`MECHTI1/DeepMzyme`](https://github.com/MECHTI1/DeepMzyme) | Primary project code, configs, test suites, documentation, and `codex/pmm-source-audit` branch. |
| **GitHub** | [`MECHTI1/deepmzyme-vm-manager`](https://github.com/MECHTI1/deepmzyme-vm-manager) | Cloud GPU VM orchestration tools and execution environments. |
| **Google Drive** | `MyDrive/DeepMzyme/` (`1slTff0joKjL-gZJDhYSGbHzOPnYhGk6I`) | Colab campaign artifacts, intermediate pilot archives, and Optuna persistent SQLite databases. |
| **Zenodo Note** | Upstream Reference | DeepMzyme's input PMM data originated from Zenodo (`classmodel_train_set.csv` / `classmodel_test_set.csv`). DeepMzyme's own published cloud archives reside on Hugging Face. |

---

## 2. Hugging Face Datasets & Checkpoint Releases

Hosted on: **[`https://huggingface.co/datasets/GMBioinformatics/DeepMzyme`](https://huggingface.co/datasets/GMBioinformatics/DeepMzyme)**

### A. Core Dataset & Feature Bundles

| File Path in Repository | Size | SHA-256 Checksum | Description & Inclusions |
|---|---:|---|---|
| `DeepMzyme_Data_v12_manifest_exact_common70_nonoverlap_clean30_care30_complete_esm_ring_external.tar.gz` | 4.73 GB | `90c0899829e0ac5ca94a5ef34484b74ca1014d3e9b6b1fba90bd338ee3440dee` | **Canonical Colab & Training Bundle (v12).** Contains 3,211 deduplicated structure-store objects, exact PinMyMetal, Common-PDBID 70/30, non-overlapped PinMyMetal, CLEAN30 shared sources, complete CARE clusterRes30, full ESMC (300M) embeddings, RING features, and external features. |
| `train_and_test_sets_structures_zenodo_pmm_exact.tar.gz` | 1.14 GB | `24903f120462aacb90b43c4af97f4f08d61f1847d12636606fcc66e6351db296` | **Exact Ion-Level Zenodo PinMyMetal Benchmark Dataset.** 9,398 total ion-level supervised sites (7,911 train / 1,487 test across 7,724 PDBs) with exact coordinates, structure manifests, site manifests, and crosswalks. |
| `CLEAN_predictor_baselines_v2_clean30x5_single_donor_supported_metal_conservative_care30_sources.tar.zst` | 29.2 MB | `5124b0b514b49affc158df121a87f5389ec1e027d14e0cf0a53cfb13a602c0f0` | Pre-trained CLEAN baseline predictions and fold evaluation sources. |
| `benchmarks/gvp_esm_hybrid_realistic_subset_v1/realistic_subset.pt` | 51.8 MB | `84e7e039f1df5b3a7b32dc3d4ac1b8fa21bba2827679b4d3f1650d394e2754bf` | Historical v1 G4/A100 benchmark subset (240 pockets). |

### B. Benchmark Runs, Trained Checkpoints & Audit Crosswalks

Published to Hugging Face commit: [`d8372220f2a3d0ce2d098a19f4728fa2d9882f90`](https://huggingface.co/datasets/GMBioinformatics/DeepMzyme/commit/d8372220f2a3d0ce2d098a19f4728fa2d9882f90)

| File Path in Repository | Size | SHA-256 Checksum | Description & Inclusions |
|---|---:|---|---|
| [`benchmarks/DeepMzyme_runs_benchmark_exact_pinmymetal_5fold_and_pilots_20260924.tar.gz`](https://huggingface.co/datasets/GMBioinformatics/DeepMzyme/resolve/main/benchmarks/DeepMzyme_runs_benchmark_exact_pinmymetal_5fold_and_pilots_20260924.tar.gz) | 511 MB | `374c888132f99345ee5ca97a7a515b1a468ec4bdace69c77f72ecde4e5b23dfc` | **Full 5-Fold CV Benchmark Runs & Model Weights.** Contains `best_checkpoint.pt`, test predictions (`test_predictions.pt`), per-fold logs, and comparison tables for: Enhanced GVP+ESMC (Folds 0-4), Enhanced Only GVP (Folds 0-4), Only ESM (Folds 0-4), plus recovery and pilot runs. |
| [`benchmarks/DeepMzyme_pmm_source_audit_output_20260924.tar.gz`](https://huggingface.co/datasets/GMBioinformatics/DeepMzyme/resolve/main/benchmarks/DeepMzyme_pmm_source_audit_output_20260924.tar.gz) | 144 MB | `8ca96f245e4ae015080f0b18b156e04c87155e4b024101137752bb2bc653f6ac` | **PinMyMetal Source Audit & Crosswalk Data.** Deposition crosswalks v1–v3, full feature crosswalks, and upstream audit summaries. |
| [`benchmarks/benchmark_5fold_comparison_table.md`](https://huggingface.co/datasets/GMBioinformatics/DeepMzyme/raw/main/benchmarks/benchmark_5fold_comparison_table.md) | 2.4 KB | — | Markdown summary table of 5-fold cross-validation balanced accuracy and per-metal recalls across models. |

---

## 3. GitHub Repositories and Branches

### A. DeepMzyme Repository: [`git@github.com:MECHTI1/DeepMzyme.git`](https://github.com/MECHTI1/DeepMzyme)
- **`main`**: Canonical development branch. All source code (`src/`), tests (`tests/`), pipelines (`scripts/`), and documentation summaries (`docs/`) are tracked here.
- **`codex/pmm-source-audit`**: Source audit and crosswalk branch. Contains:
  - Crosswalk audit tools: `src/audit_pmm_deposition_crosswalk.py`, `src/audit_pmm_feature_crosswalk.py`
  - Site dataset builder: `src/build_pmm_site_dataset.py`
  - Review scripts: `src/review_pmm_feature_matches.py`, `src/summarize_pmm_feature_crosswalk.py`
  - Documentation: `docs/PMM_*.md`, `docs/VERY_EXACT_PMM_SETS_PLAN.md`
  - Unit tests: `tests/test_pmm_deposition_crosswalk.py`, `tests/test_pmm_site_dataset.py`

### B. VM Manager Repository: [`git@github.com:MECHTI1/deepmzyme-vm-manager.git`](https://github.com/MECHTI1/deepmzyme-vm-manager)
- Branch `main`: Provisioning, environment configuration, and execution tools for GCP GPU VMs (G4, L4, A100).

---

## 4. Google Drive Storage Reference

- **Campaign Folder ID:** `1slTff0joKjL-gZJDhYSGbHzOPnYhGk6I` (`MyDrive/DeepMzyme/`)
- **Key Artifacts Stored:**
  - `optuna/`: Persistent SQLite databases (`ec_only_gvp_optuna_*.db`, `ec_late_fusion_optuna_*.db`).
  - `campaigns/`: Transfer receipts and archives for `metal_architecture_pilot`, `metal_coordination_geometry_pilot`, and `metal_ring_pilot_v1_20260915`.
  - Stored archives have verified SHA-256 receipts tracked in `docs/notebook_outputs/raw/`.

---

## 5. Local Storage & Secondary Disk Hierarchy

For large-scale processing that exceeds local root partition limits:
- **Local Machine Root:** `/home/mechti/PycharmProjects/DeepMzyme`
- **Secondary HDD Partition (`/media/mechti/Data1`):**
  - `/media/mechti/Data1/DeepMzyme_Data/`: Overflow storage for runs and caches.
  - `/media/mechti/Data1/DeepMzyme_PMM_Zenodo_Exact_Dataset/`: Reconstructed uncompressed dataset structures.
  - `/media/mechti/Data1/clean_sets/`, `/media/mechti/Data1/care_sets/`, `/media/mechti/Data1/pinmymetal_sets/`: Raw upstream downloads.
  - `mgy_proteins_pfam.tsv.gz` (33 GB), `5-allmembers-repId-entryId-cluFlag-taxId.tar.xz` (528 MB): Source sequence and clan databases.

---

## 6. How to Restore & Verify Backups

### Restoring the 5-Fold Trained Models
```bash
# 1. Download from Hugging Face
wget https://huggingface.co/datasets/GMBioinformatics/DeepMzyme/resolve/main/benchmarks/DeepMzyme_runs_benchmark_exact_pinmymetal_5fold_and_pilots_20260924.tar.gz

# 2. Verify SHA-256
echo "374c888132f99345ee5ca97a7a515b1a468ec4bdace69c77f72ecde4e5b23dfc  DeepMzyme_runs_benchmark_exact_pinmymetal_5fold_and_pilots_20260924.tar.gz" | sha256sum -c

# 3. Extract to project root
tar -xzvf DeepMzyme_runs_benchmark_exact_pinmymetal_5fold_and_pilots_20260924.tar.gz -C /home/mechti/PycharmProjects/DeepMzyme/
```

### Restoring the Exact PinMyMetal Benchmark Dataset
```bash
# 1. Download from Hugging Face
wget https://huggingface.co/datasets/GMBioinformatics/DeepMzyme/resolve/main/train_and_test_sets_structures_zenodo_pmm_exact.tar.gz

# 2. Verify SHA-256
echo "24903f120462aacb90b43c4af97f4f08d61f1847d12636606fcc66e6351db296  train_and_test_sets_structures_zenodo_pmm_exact.tar.gz" | sha256sum -c

# 3. Extract to DeepMzyme_Data
mkdir -p DeepMzyme_Data/train_and_test_sets_structures_zenodo_pmm_exact
tar -xzvf train_and_test_sets_structures_zenodo_pmm_exact.tar.gz -C DeepMzyme_Data/
```
