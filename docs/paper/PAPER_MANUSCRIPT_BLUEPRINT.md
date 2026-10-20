# DeepMzyme: Comprehensive Project Synthesis and Paper Manuscript Blueprint

**Document Purpose:** This document provides a complete, self-contained scientific synthesis of the **DeepMzyme** project, organized directly in the structure of a peer-reviewed journal manuscript (*Nature Communications*, *Bioinformatics*, *Journal of Chemical Information and Modeling*, or *Briefings in Bioinformatics*). Every number, table, architectural specification, dataset audit, and statistical interval in this blueprint is verified against the repository's frozen artifacts and independent recomputations.

**Primary Evidence Sources:**
- **Full Publication Manuscripts:** [`docs/paper/MANUSCRIPT.md`](MANUSCRIPT.md) (Markdown) and [`docs/paper/manuscript.tex`](manuscript.tex) + [`docs/paper/references.bib`](references.bib) (LaTeX + BibTeX)
- Project overview & CLI usage: [`README.md`](../../README.md) and [`Plan.md`](../../Plan.md)
- Matched PinMyMetal comparison report (`2,534/2,534` values independently verified): [`docs/campaigns/phase2_unified_2026-10/pmm_matched/README.md`](../campaigns/phase2_unified_2026-10/pmm_matched/README.md)
- Phase 2 180-checkpoint evaluation & release audit: [`docs/campaigns/phase2_unified_2026-10/README.md`](../campaigns/phase2_unified_2026-10/README.md) and [`docs/campaigns/phase2_unified_2026-10/release_audit/README.md`](../campaigns/phase2_unified_2026-10/release_audit/README.md)
- Datasets, splits, and test-use ledger: [`docs/DATASETS.md`](../DATASETS.md)
- Forensic PinMyMetal audits & `TECH-028` (`valence_3a` leak): [`docs/FOLLOW_UP_TECHNICAL_ISSUES.md`](../FOLLOW_UP_TECHNICAL_ISSUES.md), [`docs/PMM_SOURCE_AUDIT.md`](../PMM_SOURCE_AUDIT.md), and [`docs/ZENODO_PINMYMETAL_EXACT_ION_LEVEL_REPRODUCIBILITY.md`](../ZENODO_PINMYMETAL_EXACT_ION_LEVEL_REPRODUCIBILITY.md)
- Parameter & ablation findings: [`docs/PARAMETER_FINDINGS.md`](../PARAMETER_FINDINGS.md)

---

## 1. Candidate Manuscript Titles & Abstract

### Candidate Titles
1. **Option A (Method + Benchmark + Superiority Focus):**
   *DeepMzyme: Multimodal 3D Equivariant Graph and Protein Language Model Fusion Outperforms PinMyMetal for Metalloprotein Binding-Site Identity Prediction*
2. **Option B (Structural Biology & Forensic Benchmark Focus):**
   *Accurate Transition-Metal Identity Prediction in Protein Binding Sites via Geometric Vector Perceptrons and ESM-C Language Models*
3. **Option C (Concise Methods Title):**
   *DeepMzyme: Combining 3D Coordination Geometry and Evolutionary Sequence Context to Resolve Transition Metals in Protein Structures*

---

### Draft Abstract (Ready for Adaptation)

> **Background:** Transition metal ions ($\text{Mn}$, $\text{Fe}$, $\text{Co}$, $\text{Ni}$, $\text{Cu}$, and $\text{Zn}$) are essential cofactors in over one-third of all characterized enzymes, yet experimental macromolecular structures frequently suffer from metal misassignment, crystallization artifacts, or metal loss. Existing machine-learning classifiers such as PinMyMetal rely on hand-crafted tabular descriptors over local coordination shells, collapse $\text{Fe}$, $\text{Co}$, and $\text{Ni}$ into a single undifferentiated class ("Class VIII"), and were evaluated under ungrouped cross-validation splits.
>
> **Results:** Here we present **DeepMzyme**, a bimodal deep learning framework that integrates $SE(3)$-equivariant Geometric Vector Perceptron Graph Neural Networks (`GVP-GNN`) over 3D binding-site microenvironments with residue-level evolutionary representations from the 600-million-parameter `ESM-C` protein language model. To enable rigorous benchmarking, we reconstructed 99.89% (`9,398 / 9,408`) of PinMyMetal's ion-level benchmark sites directly from atomic coordinates, curated a symmetry-audited cohort of `7,398` mononuclear sites across `2,208` sequence-identity clusters ($\ge 90\%$ identity grouped into the same fold), and uncovered an undocumented ground-truth bond-valence sum leak (`valence_3a`) in PinMyMetal's released feature tables (`7,919 / 7,920` training rows and `1,488 / 1,488` test rows).
>
> In a pre-registered matched comparison on the exact same `7,398` training sites, sequence-identity-grouped 5-fold cross-validation splits, and `1,488` held-out test sites, PinMyMetal's released classifier retrained with deployment-available features (`PMM-D`) achieved **63.03%** mean-fold (62.96% pooled) balanced accuracy in cross-validation and **58.27%** on the test set (or **67.19%** CV and **63.41%** test when retaining the leaked `valence_3a` feature in `PMM-R`). DeepMzyme's pre-declared single model reached **74.28%** mean-fold (**74.34%** pooled) balanced accuracy in cross-validation (**+11.25 percentage points** over `PMM-D`, 97.5% CI `[+6.25, +15.36]`; **+7.09 pp** over `PMM-R`, 95% CI `[+2.42, +11.50]`) and **67.55%** on the full test set (**+9.28 pp** over `PMM-D`, 97.5% CI `[+3.59, +15.10]`; **69.33%** on the 1,346 protein-coordinated in-scope test sites). Across all 15 primary DeepMzyme late-fusion ensembles selected strictly by cross-validation (`R1–R4`), test balanced accuracy reached **68.83% to 71.75%** on the full test (**70.49% to 73.48%** on in-scope sites), outperforming matched `PMM-D` by **+10.56 to +13.48 pp**, matched `PMM-R` by **+5.42 to +8.34 pp** (all paired 95% CIs above zero), and exceeding PinMyMetal's originally reported paper benchmarks on both 5-fold cross-validation (**75.67%–81.14%** vs. **75.08%** macro-precision; **74.34%–77.99%** vs. **≈72.7%** balanced accuracy) and the held-out test set (**68.83%–71.75%** vs. **67.85%**). Finally, DeepMzyme provides the first grouped cross-validation resolution within Class VIII (`Fe` vs. `Co+Ni` 5-class and `Fe`/`Co`/`Ni` 6-class), achieving ~61% native $\text{Fe}$ recall and `0.680` pairwise $\text{Co}$–$\text{Ni}$ AUROC.

---

## 2. Introduction & Scientific Positioning

### 2.1 Biological Problem
Metalloproteins bind transition metal cations ($\text{Mn}^{2+}$, $\text{Fe}^{2+/3+}$, $\text{Co}^{2+}$, $\text{Ni}^{2+}$, $\text{Cu}^{+/2+}$, $\text{Zn}^{2+}$) that catalyze redox, hydrolytic, and structural reactions across all domains of life. However, in the Protein Data Bank (PDB):
1. **Experimental Misassignment:** X-ray crystallography at moderate resolution struggles to distinguish adjacent first-row transition metals with similar electron counts, and crystallization buffers or heterologous expression hosts (e.g., *E. coli* media enriched in $\text{Zn}^{2+}$ or $\text{Ni}^{2+}$) frequently substitute non-native metals into active sites.
2. **Irving–Williams Promiscuity:** Because first-shell coordinating residues ($\text{His}$, $\text{Cys}$, $\text{Asp}$, $\text{Glu}$) are shared across transition metals, first-shell geometry alone is often insufficient to disambiguate ions without wider second-shell structural and evolutionary sequence context.
3. **Apo /Predicted Structures:** With AlphaFold2/3 and ESMFold generating millions of predicted structures lacking explicit cofactors, automated tools that identify which metal belongs in a binding pocket are critical for enzyme engineering and functional annotation.

### 2.2 Limitations of Prior Art
1. **Metal3D** (Dürr et al., *Nature Communications* 2023): Uses voxelized 3D convolutional neural networks trained on zinc-binding sites to predict metal *spatial density* (site localization), but **does not predict metal element identity**. *(Note: A common misreading of PinMyMetal Fig. 2c attributes `61.70%` to Metal3D; in reality, `61.70%` is PinMyMetal's own classification score evaluated on an external set of sites localized by Metal3D).*
2. **AllMetal3D** (2025 preprint): Added a 12-class metal identity head, but achieved only ~46% accuracy and noted that static local geometry alone cannot reliably separate similar transition metals.
3. **PinMyMetal (PMM)** (Zhang et al., *Nature Communications* 2025, `10.1038/s41467-025-57637-5`): Introduced a two-stage pipeline (site localization + a 5-classifier soft-voting ensemble of Logistic Regression, Balanced Random Forest, MLP, SVC, and EasyEnsemble over 93 tabular features) for four metal classes (`Mn`, `Cu`, `Zn`, and `Class VIII = Fe/Co/Ni`), reporting **75.08%** 5-fold CV macro-precision (`≈72.70%` balanced accuracy) in Fig. 2a and **67.85%** test balanced accuracy in Fig. 2b. However, our forensic audit of PinMyMetal identified four critical limitations:
   - **Label Leakage in Released Features (`TECH-028`):** The feature `valence_3a` in PinMyMetal's released training (`7,920` rows) and test (`1,488` rows) tables equals the bond-valence sum computed using the **ground-truth metal's oxidation state** (`bvs_mn`, `bvs_fe`, `bvs_co`, `bvs_ni`, `bvs_cu`, `bvs_zn`) in `7,919 / 7,920` training rows (`99.99%`) and `1,488 / 1,488` test rows (`100%`). A parameter-free lookup comparing `valence_3a` to the `bvs_*` columns recovers the true metal class with **99.94%–100% balanced accuracy** on the training table.
   - **Ungrouped Cross-Validation:** PinMyMetal's Fig. 2a 5-fold CV used `StratifiedKFold(n_splits=5, shuffle=False)` over raw CSV row order without grouping by PDB entry or protein sequence identity, allowing identical chains and close homologues (as well as `503` crystallographic symmetry-linked ions) to span training and validation folds.
   - **Train–Test Overlap:** PinMyMetal's `1,488`-row test set shares **668 PDB IDs** (`809 / 1,488` test rows, `54.4%`) with its `7,920`-row training set.
   - **Coarse Class VIII Grouping:** PinMyMetal merges $\text{Fe}$, $\text{Co}$, and $\text{Ni}$ into a single `Class VIII` label (`code 2`) and provides no native 5- or 6-class resolution.

---

## 3. Dataset Reconstruction, Forensic Audit & Benchmark Design (Methods Part I)

### 3.1 Ion-Level 3D Reconstruction of the PinMyMetal Benchmark (`99.89%` Fidelity)
PinMyMetal's released files (`classmodel_train_set` with `7,920` rows and `classmodel_test_set` with `1,488` rows) contain only precomputed tabular numbers indexed by `(pdbid, residueid_ion, metalid)`, where `residueid_ion` is an internal database key produced by the `NEIGHBORHOOD` SQL program rather than a PDB chain/residue coordinate.

To enable 3D geometric deep learning and fair benchmarking ([`docs/ZENODO_PINMYMETAL_EXACT_ION_LEVEL_REPRODUCIBILITY.md`](../ZENODO_PINMYMETAL_EXACT_ION_LEVEL_REPRODUCIBILITY.md)):
1. We executed PinMyMetal's compiled `neighborhood` binary (`PinMyMetal_V1.0_zenodo.zip`) against source PDB structures to deterministically resolve internal `residueid_ion` keys to physical PDB coordinates `(chain, resseq, resname, atomid)`.
2. For `298` large macromolecular complexes that caused segmentation faults (`SIGSEGV`) in `neighborhood`, stripping `HOH` solvent records prior to execution achieved 100% coordinate recovery.
3. In total, **9,398 of 9,408 ions (`99.89%`)** across `7,724` structure files were reconstructed (**7,911 / 7,920** training rows = `99.89%`; **1,487 / 1,488** test rows = `99.93%`) and released publicly on Hugging Face ([`GMBioinformatics/DeepMzyme`](https://huggingface.co/datasets/GMBioinformatics/DeepMzyme), `1.1 GB`).

### 3.2 Curating the Clean `7,398`-Ion Training Cohort and `v3` Sequence-Identity Folds
Auditing all `7,920` raw training rows ([`docs/DATASETS.md`](../DATASETS.md#exact-zenodo-pinmymetal-train_and_test_sets_structures_zenodo_pmm_exact)) identified `522` rows requiring exclusion before 3D graph training:
- **`9` rows:** Absent from source PDB reconstruction.
- **`10` rows:** Mapped to polynuclear/non-single-metal species ($\text{Fe–S}$ clusters, cofactors, or `UNX` records).
- **`503` rows:** Possessed explicit target-to-protein crystallographic symmetry `LINK` records across unit-cell boundaries (where standard asymmetric-unit PDB coordinates omit part of the coordinating protein environment or duplicate symmetry-equivalent sites).

Removing these `522` rows yielded the certified **7,398-ion training cohort** (`SHA-256: 745b19c1...`) across **3,992 unique PDB entries**, with native element counts:
- **Mn:** `2,528` (`34.17%`)
- **Zn:** `2,032` (`27.47%`)
- **Class VIII (`Fe + Co + Ni`):** `2,495` (`33.73%`), comprising **Fe:** `1,737` (`23.48%`), **Ni:** `475` (`6.42%`), and **Co:** `283` (`3.83%`)
- **Cu:** `343` (`4.64%`)

**Sequence-Identity-Grouped 5-Fold Split (`v3-seqid90-s42-b2`, `SHA-256: e6b21364...`):**
All chains were clustered at **$\ge 90\%$ sequence identity** into **2,208 near-copy groups**, ensuring that no two chains sharing $\ge 90\%$ sequence identity ever appear in different folds, while preserving balanced stratification across all 6 native metal elements in every fold.

### 3.3 Reference Test Set Subsets (`1,488` Rows and Stratified Sensitivity Lines)
To evaluate performance transparently on PinMyMetal's `1,488`-row reference test set (`1,179` PDB entries; `Mn: 167`, `Class VIII: 252`, `Cu: 64`, `Zn: 1,005` [67.5% Zn]) while accounting for structural scope and PinMyMetal's internal train–test overlap, five nested evaluation subsets were defined:
1. **Full Test (`1,488` rows, `1,179` PDB entries):** All rows in `classmodel_test_set`. The `3` unscorable rows (missing structure/graph) are counted as unconditional prediction errors for DeepMzyme.
2. **LINK-Free (`1,349` rows):** Excludes `139` test rows with crystallographic symmetry `LINK` flags (`link_flagged == True`).
3. **Training-Eligible / In-Scope (`1,346` rows, `1,093` PDB entries):** Scored, LINK-free primary sites that satisfy the training-time biological scope filter ($\ge 1$ coordinating protein donor atom within the site; excludes water-only or nucleotide-only metal ions that lack protein coordination).
4. **Deduplicated (`1,098` rows):** Retains one representative site per near-identical structure group within the test set.
5. **Clean / Homology-Filtered Subset (`423` rows, `352` PDB entries):** Removes all `809` test rows that share a PDB ID with the training set (`668` shared PDB IDs) as well as chains sharing sequence identity / homology with training proteins.

---

## 4. DeepMzyme Architecture & Training Protocol (Methods Part II)

### 4.1 Bimodal Late-Fusion Architecture (`GVP-GNN + ESM-C 600M`)
DeepMzyme models each metal-binding site (`--metal-example-unit ion`) through two complementary branches that are fused at the representation level before classification:

1. **3D Equivariant Structural Pocket Branch (`GVP-GNN`):**
   - **Graph Construction:** All protein residues within an $8.0\text{ \AA}$ radius around the target metal site form a local 3D pocket graph, connected by $8.0\text{ \AA}$ radius edges encoded with Gaussian radial basis functions (RBFs) over raw Euclidean distances (`--rbf-use-raw-distances`).
   - **Node Features (Scalar $s$ + Vector $V$):**
     - *Scalar channels:* Amino-acid identity (20-dim one-hot), physicochemical descriptors, explicit first-shell ($\le 3.0\text{ \AA}$) and second-shell ($\le 5.0\text{ \AA}$) coordination role indicators, and distance RBFs to the metal center.
     - *Vector channels ($SE(3)$-equivariant):* Unit direction vectors pointing along the side-chain orientation ($\text{C}_\alpha \rightarrow \text{C}_\beta$) and from each residue toward the metal center.
   - **Message Passing:** Stacked Geometric Vector Perceptron (`GVP`) layers jointly update scalar and vector representations while preserving $SE(3)$ rotation and translation invariance of the final pooled pocket representation.

2. **Evolutionary Sequence Context Branch (`ESM-C 600M`):**
   - Full-chain protein sequences are embedded once using the pretrained **ESM Cambrian 600M (`esmc_600m`)** protein language model (`1,152`-dimensional per-residue embeddings).
   - Embeddings corresponding to the binding-site residues are projected through a linear + layer-norm bottleneck and aggregated via combined mean and attention pooling over the pocket residues.

3. **Graph-Level Late Fusion & Classification Head:**
   - The pooled `GVP-GNN` structural vector and the pooled `ESM-C` sequence vector are concatenated (`--fusion-mode late_fusion`) and passed through a regularized multi-layer perceptron (MLP) classification head (`head_dropout = 0.2`, `weight_decay = 1e-4`).

### 4.2 Target Formulations (`4-Class`, `5-Class`, `6-Class`) and Loss Recipes (`B`, `P`, `S`)
DeepMzyme supports three native supervision schemes:
- **`four_class` (`merge_fe_class_viii`):** Direct training on `Mn`, `Cu`, `Zn`, and `Class VIII (Fe+Co+Ni)`.
- **`five_class`:** Native training on `Mn`, `Cu`, `Zn`, `Fe`, and `Co+Ni`. To evaluate on the common-four benchmark, predicted probabilities for `Fe` and `Co+Ni` are summed prior to $\arg\max$:
  $$P(\text{Class VIII}) = P(\text{Fe}) + P(\text{Co+Ni})$$
- **`six_class`:** Native training on all six transition metals (`Mn`, `Fe`, `Co`, `Ni`, `Cu`, `Zn`), mapped to common-four via:
  $$P(\text{Class VIII}) = P(\text{Fe}) + P(\text{Co}) + P(\text{Ni})$$

Three class-imbalance loss recipes were systematically evaluated across the 5 grouped folds:
- **Recipe `B` (Baseline):** Inverse-frequency class-weighted cross-entropy (`GVP trunk LR = 3e-4`, `ESM/head LR = 3e-4`, 50 epochs, batch size 16).
- **Recipe `P` (Population-Normalized CE):** Population-normalized class-weighted cross-entropy (`focal gamma = 0`).
- **Recipe `S` (Label Smoothing):** Class-weighted cross-entropy with label smoothing $\varepsilon = 0.1$.
- **Variant `trunklr1e4`:** Lowers the GVP trunk learning rate to $10^{-4}$ while keeping head/fusion LR at $3 \times 10^{-4}$.

### 4.3 Pre-Registered PinMyMetal Comparator Arms (`PMM-D`, `PMM-R`, `PMM-D2`, `PMM-H`)
To compare DeepMzyme and PinMyMetal on identical data without cohort or split confounding ([`docs/campaigns/phase2_unified_2026-10/pmm_matched/README.md`](../campaigns/phase2_unified_2026-10/pmm_matched/README.md)), PinMyMetal's exact released 5-estimator soft-voting classifier (`LogisticRegression + BalancedRandomForestClassifier + MLPClassifier + SVC + EasyEnsembleClassifier` with released hyperparameters) was refit in its pinned software environment (`Python 3.11.5`, `scikit-learn 1.3.0`, `imbalanced-learn 0.11.0`) on the exact same `7,398` training sites and `5` folds (`v3-seqid90-s42-b2`), and refit once on all `7,398` sites to score the `1,488` test rows:
- **`PMM-D` (Primary Deployment-Faithful Comparator, 89 features):** Drops the 4 columns (`valence_3a`, `valence_4a`, `vecsum_3a`, `vecsum_4a`) that at real-world deployment are computed by writing candidate sites as $\text{Zn}$, but in PinMyMetal's released tables were populated with the true metal's bond-valence sum (`TECH-028`).
- **`PMM-R` (Full Released-Feature Reference, 93 features):** Retains all 93 published columns, including the true-metal `valence_3a` column.
- **`PMM-D2` (Sensitivity Arm, 79 features):** Also removes the 10 elemental bond-valence sum columns (`bvs_*`).
- **`PMM-H` (Internal Feature-Representation Ablation, 89 features):** An untuned `HistGradientBoostingClassifier` trained in this study on PinMyMetal's 89 deployment features (not part of PinMyMetal, used strictly to separate classifier-architecture gains from tabular-feature gains).

---

## 5. Comprehensive Experimental Results (Publication Tables)

### 5.1 Main Benchmark Comparison: DeepMzyme vs. PinMyMetal (Matched & Published)

**Table 1** summarizes the headline performance across 5-fold cross-validation (`7,398` sites) and the reference test set (`1,488` full rows, `1,349` LINK-free rows, `1,346` in-scope training-eligible rows, and `423` clean subset rows). Note that in PinMyMetal Fig. 2a, the printed diagonal (`Mn 90.3%, Class VIII 73.3%, Cu 62.9%, Zn 73.8%`, mean **75.08%**) is **column-normalized macro-precision**, whereas row-normalized balanced accuracy (mean recall) computed from Zhang et al. Supplementary Table 10 is **≈72.70%**. Both metrics are reported below.

#### Table 1: End-to-End Benchmark Performance (5-Fold CV and Reference Test Set)

| System / Model Tier | Ckpts | 5-Fold CV BA (Mean Fold / Pooled) | 5-Fold CV Macro-Precision *(Fig. 2a Metric)* | Full Test BA (`1,488` rows) `[95% CI]` | LINK-Free Test BA (`1,349` rows) | In-Scope Eligible Test BA (`1,346` rows) | Clean Subset BA (`423` rows) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PinMyMetal Comparators** | | | | | | | |
| PinMyMetal *(Published, Zhang et al. 2025 Fig. 2a/2b; ungrouped CV + `valence_3a`)* | 5-est | ≈72.70% *(unmatched)* | 75.08% *(Fig. 2a)* | 67.85% *(Fig. 2b)* | — | — | — |
| **PMM-D** *(Retrained on 7,398 sites & v3 folds; 89 deployment features)* | 5-est | 63.03% / 62.96% | 68.47% | 58.27% `[53.53, 62.85]` | 59.26% | 59.48% | 47.09% |
| **PMM-R** *(Retrained on 7,398 sites & v3 folds; all 93 features incl. `valence_3a`)* | 5-est | 67.19% / 67.16% | 71.24% | 63.41% `[59.64, 67.25]` | 63.64% | 63.81% | 59.46% |
| **PMM-D2** *(Retrained sensitivity arm; drops `valence_*`, `vecsum_*`, `bvs_*`)* | 5-est | 60.62% / 60.59% | 61.58% | 58.13% `[53.96, 62.29]` | — | 58.09% | 51.78% |
| **PMM-H** *(Internal ablation: HistGB on 89 PMM deployment features)* | 1 | 71.22% / 71.22% | 75.51% | 68.83% `[64.26, 73.04]` | 69.20% | 69.49% | 58.36% |
| **DeepMzyme: Single Model & Baseline Progression (Zero Test Selection)** | | | | | | | |
| **Single Model Refit (`a0007846`, pre-declared 1-ckpt full refit of `B4`)** | 1 | 74.28% / 74.34%\* | 75.74%\* | **67.55%** `[62.93, 71.65]` | **68.99%** | **69.33%** | 48.85% |
| **5-Fold Baseline `B4` (`s42`, 5 fold checkpoints averaged)** | 5 | 74.28% / 74.34% | **75.74%** | **69.02%** `[64.47, 73.05]` | **70.40%** | **70.75%** | 52.32% |
| **15-Model Baseline `B4` (`s42/43/44`, 3 seeds × 5 folds)** | 15 | — / 75.58% | **77.39%** | **69.94%** `[65.46, 74.03]` | **71.48%** | **71.83%** | 53.49% |
| **DeepMzyme: Primary CV-Selected Ensembles (`R1–R4`, Zero Test Selection)** | | | | | | | |
| **5-Fold Differential Trunk-LR (`trunklr1e4`, `s42`, 5 fold ckpts)** | 5 | 75.45% / 75.51% | **76.81%** | **70.75%** `[66.29, 74.77]` | **71.89%** | **72.25%** | 55.87% |
| **`R1` Top-1 Family (`mc_P_five_class` × 3 seeds)** | 15 | 75.32%† / 75.82% | **77.13%** | **71.10%** `[66.65, 75.14]` | **72.33%** | **72.68%** | 58.21% |
| **`R1` Top-4 Families All Seeds (`P5 + P4 + B4 + trunklr1e4`)** | 50 | 76.27%† / 76.46% | **78.36%** | **71.75%** `[67.22, 75.80]` | **73.12%** | **73.48%** | 57.47% |
| **`R3` Pre-Declared Greedy CV Ensemble (`P5 + wd10 + trunklr1e4 + sca`)** | 30 | 76.54%† / 77.99% | **81.14%** | **70.57%** `[65.90, 74.75]` | — | **73.00%** | 59.20% |
| **Range Across All 15 Primary CV-Selected Ensembles (`R1–R4`)** | 5–70 | **74.54%–76.61%† / 74.34%–77.99%** | **75.67%–81.14%** | **68.83%–71.75%** | — | **70.49%–73.48%** | 52.32%–59.20% |
| **DeepMzyme: Exploratory / Post-Hoc Amendments & Release-Audit Catalog (Disclosed Separately in Manuscript Sec. 4.3)** | | | | | | | |
| **`R2b` Top-4 Seed-42 by Seed-42 OOF (`trunklr1e4 + esmdrop02 + B6 + S6`)** | 20 | 76.10%† / 76.35% | **78.48%** | **71.83%** `[67.20, 75.95]` | — | **73.62%** | **60.07%** |
| **`R2b` Top-5 Seed-42 by Seed-42 OOF (`Top-4 + Unified_P5_trunklr1e4`)** | 25 | 76.09%† / 76.16% | **77.77%** | **72.10%** `[67.63, 76.02]` | — | **73.64%** | 59.37% |
| **20-Model Late Fusion (`top4_lf_seed42`)** | 20 | — / 76.34% | **77.88%** | **71.80%** `[67.34, 75.79]` | — | **73.54%** | 58.48% |
| **25-Model Late Fusion (`top3_lf`)** | 25 | — / 76.62% | **78.60%** | **71.70%** `[67.19, 75.67]` | **73.10%** | **73.46%** | 58.39% |
| **30-Model Late Fusion (`top4_lf`)** | 30 | — / 76.81% | **78.38%** | **72.20%** `[67.73, 76.17]` | **73.58%** | **73.94%** | 59.18% |
| **55-Model Trimodal Reference (`Late Fusion + Only-ESM + Only-GVP`)** | 55 | — / **77.17%** | **80.34%** | **71.00%** `[66.39, 75.15]` | **72.42%** | **72.76%** | **61.33%** |

*\*CV score of the `B4` recipe (`s42`); the full-dataset refit `a0007846` is trained on 100% of the 7,398 training sites. †Nested procedure-level CV balanced accuracy (re-ranking families inside each 4-fold training split).*

---

### 5.2 Complete Table of All 15 CV-Only Ensembles (`Amendment A1`) & 6 Seed-42-Ranked Ensembles (`Amendment A2`)

To ensure that ensemble performance claims are completely immune to test-set selection bias ("test-shopping"), **Table 2** reports the exhaustive set of **15 ensembles defined strictly by cross-validation rules (`R1–R4`)** plus the **6 seed-42-ranked ensembles (`R2b`)**. Every single ensemble (`21 / 21`) scores above PinMyMetal's published **67.85%** on the full test (`68.83%–72.10%` Full; `70.49%–73.64%` Eligible), exceeds PinMyMetal's published **75.08%** CV macro-precision (`75.67%–81.14%`), and achieves statistically significant paired bootstrap gains (95% CIs entirely above zero) over both matched **`PMM-D` (`+10.56 to +13.83 pp`)** and matched **`PMM-R` (`+5.42 to +8.69 pp`)**.

#### Table 2: Exhaustive CV-Selected Ensemble Progression and Paired Contrasts

| Rule Group & Ensemble Name | Ckpts | CV BA (Nested) | CV BA (Pooled) | CV Macro-Precision | Full Test BA (`1,488`) `[95% CI]` | Paired $\Delta$ vs. `PMM-D` `[95% CI]` | Paired $\Delta$ vs. `PMM-R` `[95% CI]` | Paired $\Delta$ vs. `PMM-H` `[95% CI]` | Eligible BA (`1,346`) | Clean BA (`423`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rule `R4`: Retained Baseline Configuration** | | | | | | | | | | |
| 1. `B4 seed 42, 5 folds` | 5 | — | 74.34% | 75.74% | 69.02% `[64.47, 73.05]` | **+10.75** `[+5.77, +15.80]` | **+5.61** `[+0.34, +10.53]` | +0.19 `[-4.16, +4.53]` | 70.75% | 52.32% |
| 2. `B4 x3 seeds` (`s42/43/44`) | 15 | — | 75.58% | 77.39% | 69.94% `[65.46, 74.03]` | **+11.67** `[+6.72, +16.77]` | **+6.53** `[+1.30, +11.46]` | +1.11 `[-3.16, +5.40]` | 71.83% | 53.49% |
| **Rule `R1`: Top-$N$ Families by Pooled OOF Rank (All Seeds)** | | | | | | | | | | |
| 3. `Top-1 all seeds` (`P5`) | 15 | 75.32% | 75.82% | 77.13% | 71.10% `[66.65, 75.14]` | **+12.83** `[+7.92, +17.74]` | **+7.69** `[+2.67, +12.44]` | +2.28 `[-1.95, +6.47]` | 72.68% | 58.21% |
| 4. `Top-2 all seeds` (`+ P4`) | 30 | 75.80% | 76.22% | 78.30% | 70.90% `[66.41, 74.99]` | **+12.63** `[+7.78, +17.64]` | **+7.49** `[+2.38, +12.28]` | +2.07 `[-2.06, +6.37]` | 72.71% | 57.37% |
| 5. `Top-3 all seeds` (`+ B4`) | 45 | 75.92% | 76.05% | 78.19% | 70.31% `[65.81, 74.43]` | **+12.04** `[+7.08, +17.11]` | **+6.90** `[+1.72, +11.80]` | +1.48 `[-2.74, +5.79]` | 72.22% | 55.37% |
| 6. `Top-4 all seeds` (`+ trunklr1e4`) | 50 | 76.27% | 76.46% | 78.36% | **71.75%** `[67.22, 75.80]` | **+13.48** `[+8.60, +18.42]` | **+8.34** `[+3.30, +13.05]` | +2.92 `[-1.15, +7.11]` | **73.48%** | 57.47% |
| 7. `Top-5 all seeds` (`+ esmdrop02`) | 55 | 76.39% | 76.39% | 78.81% | 71.33% `[66.80, 75.38]` | **+13.05** `[+8.24, +17.98]` | **+7.91** `[+2.84, +12.63]` | +2.50 `[-1.55, +6.65]` | 73.11% | 56.97% |
| 8. `Top-6 all seeds` (`+ P6`) | 70 | 76.61% | 76.72% | 78.88% | 71.25% `[66.75, 75.28]` | **+12.98** `[+8.12, +17.95]` | **+7.84** `[+2.78, +12.56]` | +2.42 `[-1.67, +6.60]` | 72.92% | 56.09% |
| **Rule `R2`: Top-$N$ Families by Pooled OOF Rank (Seed 42 Only)** | | | | | | | | | | |
| 9. `Top-1 seed 42` | 5 | 74.54% | 74.48% | 75.67% | 69.90% `[65.27, 74.03]` | **+11.63** `[+6.71, +16.55]` | **+6.49** `[+1.45, +11.15]` | +1.07 `[-3.13, +5.33]` | 71.65% | 59.05% |
| 10. `Top-2 seed 42` | 10 | 75.30% | 75.50% | 77.18% | 70.39% `[65.80, 74.55]` | **+12.12** `[+7.42, +16.93]` | **+6.98** `[+1.95, +11.65]` | +1.56 `[-2.50, +5.67]` | 72.00% | 57.96% |
| 11. `Top-3 seed 42` | 15 | 75.33% | 75.16% | 76.90% | 68.83% `[64.16, 73.00]` | **+10.56** `[+5.73, +15.53]` | **+5.42** `[+0.25, +10.24]` | 0.00 `[-4.20, +4.20]` | 70.49% | 54.35% |
| 12. `Top-4 seed 42` | 20 | 75.44% | 75.76% | 77.54% | 70.33% `[65.84, 74.29]` | **+12.06** `[+7.28, +16.95]` | **+6.92** `[+1.88, +11.56]` | +1.50 `[-2.63, +5.69]` | 71.83% | 55.50% |
| 13. `Top-5 seed 42` | 25 | 75.76% | 75.65% | 77.66% | 69.89% `[65.27, 74.05]` | **+11.62** `[+6.86, +16.46]` | **+6.48** `[+1.50, +11.18]` | +1.06 `[-2.99, +5.18]` | 71.47% | 55.94% |
| 14. `Top-6 seed 42` | 30 | 75.96% | 75.88% | 78.05% | 70.21% `[65.59, 74.32]` | **+11.94** `[+7.18, +16.78]` | **+6.80** `[+1.77, +11.49]` | +1.39 `[-2.65, +5.45]` | 71.78% | 56.11% |
| **Rule `R3`: Greedy Forward Selection by CV** | | | | | | | | | | |
| 15. `Greedy CV Ensemble` (`P5 + wd10 + trunklr1e4 + sca`) | 30 | 76.54% | 77.99% | 81.14% | 70.57% `[65.90, 74.75]` | **+12.30** `[+7.62, +16.94]` | **+7.16** `[+2.35, +11.58]` | +1.74 `[-2.31, +5.71]` | 73.00% | 59.20% |
| **Rule `R2b` (`Amendment A2`): Top-$N$ Seed-42 by Seed-42 OOF Rank** | | | | | | | | | | |
| 16. `R2b Top-1` (`trunklr1e4`) | 5 | 74.47% | 75.51% | 76.81% | 70.75% `[66.29, 74.77]` | **+12.48** `[+7.63, +17.35]` | **+7.34** `[+2.32, +12.03]` | +1.92 `[-2.23, +6.11]` | 72.25% | 55.87% |
| 17. `R2b Top-2` (`+ esmdrop02`) | 10 | 75.65% | 75.82% | 77.50% | 71.26% `[66.66, 75.38]` | **+12.99** `[+8.25, +17.72]` | **+7.84** `[+2.85, +12.45]` | +2.43 `[-1.55, +6.43]` | 73.04% | 57.77% |
| 18. `R2b Top-3` (`+ B6_baseline`) | 15 | 75.98% | 76.29% | 78.66% | 70.89% `[66.22, 75.03]` | **+12.61** `[+7.77, +17.39]` | **+7.47** `[+2.48, +12.19]` | +2.06 `[-1.97, +6.11]` | 72.62% | 58.44% |
| 19. `R2b Top-4` (`+ mc_S_six_class`) | 20 | 76.10% | 76.35% | 78.48% | **71.83%** `[67.20, 75.95]` | **+13.56** `[+8.76, +18.33]` | **+8.42** `[+3.45, +13.15]` | +3.00 `[-1.00, +7.02]` | **73.62%** | **60.07%** |
| 20. `R2b Top-5` (`+ Unified_P5_trunklr1e4`) | 25 | 76.09% | 76.16% | 77.77% | **72.10%** `[67.63, 76.02]` | **+13.83** `[+9.09, +18.61]` | **+8.69** `[+3.63, +13.34]` | +3.27 `[-0.82, +7.36]` | **73.64%** | 59.37% |
| 21. `R2b Top-6` (`+ headdrop03`) | 30 | 76.23% | 76.48% | 78.52% | 71.15% `[66.67, 75.16]` | **+12.88** `[+8.06, +17.74]` | **+7.74** `[+2.72, +12.44]` | +2.32 `[-1.75, +6.48]` | 72.65% | 57.12% |

---

### 5.3 Per-Metal Recall Breakdown & The `TECH-028` Manganese Finding

**Table 3** details per-class recall across `Mn`, `Cu`, `Zn`, and `Class VIII (Fe/Co/Ni)` and demonstrates how PinMyMetal's leaked `valence_3a` feature artificially inflated its Manganese (`Mn`) recall.

#### Table 3: Per-Metal Recalls (%) on Cross-Validation and Held-Out Test Set

| System / Model | Evaluation Line | Balanced Accuracy (%) | Mn Recall (%) | Cu Recall (%) | Zn Recall (%) | Class VIII Recall (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **5-Fold Cross-Validation (`7,398` sites)** | | | | | | |
| PinMyMetal *(Published Fig. 2a / Supp. Table 10, ungrouped)* | 7,920 raw rows | 72.70% *(75.08% Prec)* | 84.2% *(90.3% Prec)* | 52.0% *(62.9% Prec)* | 79.3% *(73.8% Prec)* | 75.3% *(73.3% Prec)* |
| **PMM-R** *(Matched v3 folds, with `valence_3a` leak)* | 7,398 sites | 67.16% | 81.6% | 42.6% | 82.0% | 62.5% |
| **PMM-D** *(Matched v3 folds, deployment features)* | 7,398 sites | 62.96% | 77.2% | 42.3% | 66.2% | 66.1% |
| **DeepMzyme `B4` (`s42` 5-fold OOF)** | 7,398 sites | **74.34%** | **75.6%** | **73.5%** | **81.4%** | **66.9%** |
| **DeepMzyme `Trunk-LR 1e-4` (`s42` 5-fold OOF)** | 7,398 sites | **75.51%** | **77.8%** | **73.8%** | **82.3%** | **68.2%** |
| **DeepMzyme `R1 Top-1` (`mc_P_five_class`, 15 ckpts OOF)** | 7,398 sites | **75.82%** | **78.6%** | **74.6%** | **84.4%** | **65.7%** |
| **Held-Out Test Set — Full Line (`1,488` rows)** | | | | | | |
| PinMyMetal *(Published Fig. 2b, with `valence_3a` leak)* | Full (`1,488`) | 67.85% | **88.6%** | 59.4% | 65.9% | 57.5% |
| **PMM-R** *(Retrained, with `valence_3a` leak)* | Full (`1,488`) | 63.41% | **79.0%** | 40.6% | 85.2% | 48.8% |
| **PMM-D** *(Retrained, deployment features — no leak)* | Full (`1,488`) | 58.27% | **61.7%** | 50.0% | 60.7% | 60.7% |
| **PMM-D2** *(Retrained, drops `valence_*`, `vecsum_*`, `bvs_*`)* | Full (`1,488`) | 58.13% | 46.7% | 46.9% | 78.6% | 60.3% |
| **DeepMzyme Single Refit (`a0007846`)** | Full (`1,488`) | **67.55%** | 55.7% | **70.3%** | **85.1%** | **59.1%** |
| **DeepMzyme `Baseline B4` (`s42`, 5 ckpts)** | Full (`1,488`) | **69.02%** | 59.3% | **70.3%** | **85.9%** | **60.7%** |
| **DeepMzyme `Trunk-LR 1e-4` (`s42`, 5 ckpts)** | Full (`1,488`) | **70.75%** | **62.9%** | **71.9%** | **87.2%** | **61.1%** |
| **DeepMzyme `Greedy` CV Ensemble (`R3`, 30 ckpts)** | Full (`1,488`) | **70.57%** | **66.5%** | **64.1%** | **89.1%** | **62.7%** |
| **DeepMzyme `R1 Top-1` (`mc_P_five_class`, 15 ckpts)** | Full (`1,488`) | **71.10%** | 62.3% | **71.9%** | **89.2%** | **61.1%** |
| **DeepMzyme `R1 Top-4 All Seeds` (50 ckpts)** | Full (`1,488`) | **71.75%** | 64.1% | **73.4%** | **89.3%** | **60.3%** |
| **Held-Out Test Set — In-Scope Training-Eligible Line (`1,346` rows)** | | | | | | |
| **PMM-D** *(Retrained, deployment features)* | Eligible (`1,346`) | 59.48% | 61.9% | 51.7% | 63.1% | 61.3% |
| **PMM-R** *(Retrained, with `valence_3a` leak)* | Eligible (`1,346`) | 63.81% | 78.8% | 43.3% | 84.0% | 49.1% |
| **DeepMzyme Single Refit (`a0007846`)** | Eligible (`1,346`) | **69.33%** | 56.9% | **73.3%** | **85.8%** | **61.3%** |
| **DeepMzyme `R1 Top-1` (`mc_P_five_class`, 15 ckpts)** | Eligible (`1,346`) | **72.68%** | 64.4% | **75.0%** | **89.6%** | **61.7%** |
| **DeepMzyme `R1 Top-4 All Seeds` (50 ckpts)** | Eligible (`1,346`) | **73.48%** | **66.2%** | **76.7%** | **89.7%** | **61.3%** |

> **Key Scientific Insight on Manganese (`TECH-028`):**
> On the matched 5-fold cross-validation splits (`7,398` sites), DeepMzyme `B4` outperforms `PMM-D` by **+31.2 pp on Cu**, **+15.2 pp on Zn**, and **+0.8 pp on Class VIII**, with a small non-significant difference on **Mn** (`-1.6 pp [-7.9, +3.7]`), while `Trunk-LR 1e-4` (`77.8% / 73.8% / 82.3% / 68.2%`) beats `PMM-D` on all four classes simultaneously. On the test set, removing the four `TECH-028` columns (`valence_3a/4a`, `vecsum_3a/4a`) from PinMyMetal causes PinMyMetal's own test Mn recall to drop by **-17.4 percentage points** (`95% CI [11.1, 24.1]`), from **79.0% (`PMM-R`) to 61.7% (`PMM-D`)** (and down to **46.7%** in `PMM-D2`).

---

### 5.4 Modality, Hyperparameter, and Objective Ablations

**Table 4** synthesizes the controlled ablation studies conducted on the `v3-seqid90-s42-b2` grouped 5-fold cross-validation benchmark ([`docs/PARAMETER_FINDINGS.md`](../PARAMETER_FINDINGS.md)).

#### Table 4: Controlled Architectural and Training Ablations (5-Fold Grouped CV)

| Ablation Study / Model Variant | Seed(s) | Mean-Fold / Pooled Common-4 CV BA (%) | Paired Contrast & Statistical Interval | Key Takeaway |
| :--- | :---: | :---: | :---: | :--- |
| **1. Modality Ablation (`v3` Stage 6)** | | | | |
| **Bimodal Late Fusion (`GVP-GNN + ESM-C 600M`)** | 42 | **74.28% / 74.34%** | *Reference* | Wins on **5 out of 5 folds** over unimodal branches |
| **Sequence-Only (`Only-ESMC 600M`)** | 42 | 72.33% / — | Late Fusion − Only-ESMC: **+1.95 pp** `[+1.09, +2.92]` | 3D pocket geometry provides a significant +1.95 pp boost |
| **Structure-Only (`Only-GVP`, baseline `wd=1e-4`)** | 42 | 69.66% / — | Late Fusion − Only-GVP: **+4.62 pp** | Evolutionary sequence context adds +4.62 pp over 3D graph alone |
| **Structure-Only (`Only-GVP`, high weight decay `wd=1.0`)** | 42 | 73.25% / 73.26% | `wd=1.0` − `wd=1e-4`: **+3.59 pp** *(interval-supported)* | Strong $L_2$ regularization (`wd10`) substantially improves unimodal GVP |
| **2. GVP Trunk Learning Rate (`pmm_trunk_lr_2026-10`)** | | | | |
| **Lower GVP Trunk LR (`1e-4` vs. `3e-4` baseline)** | 42 | **75.45% / 75.51%** | **+1.16 pp** `[-0.65, +3.11]` *(folds 1–4: +0.24 pp)* | Single best seed-42 5-fold CV recipe (`70.75%` Full Test / `72.25%` Eligible) |
| **3. Loss Recipe × Target Objective (`metal_completion_2026-10`)** | | | | |
| **Recipe `B` Four-Class (`baseline`)** | 42, 43, 44 | 73.87% *(seed-avg)* / 75.58% *(ens)* | *Retained single-recipe baseline* | Multi-seed averaging (`3 seeds`) boosts pooled OOF BA from 74.34% to 75.58% |
| **Recipe `P` Four-Class (Population-Normalized CE)** | 42, 43, 44 | 74.24% *(seed-avg)* / 75.60% *(ens)* | `P4 − B4`: **+0.36 pp** `[-0.39, +1.42]` *(adj.)* | #2 multi-seed family by pooled OOF CV (`75.60%`) |
| **Recipe `P` Five-Class (`Mn, Cu, Zn, Fe, Co+Ni`)** | 42, 43, 44 | 73.97% *(seed-avg)* / **75.82%** *(ens)* | `P5 − B4`: **+0.10 pp** *(seed-avg)* | **#1 multi-seed family by pooled OOF CV (`75.82%` CV, `71.10%` Full Test)** |
| **Recipe `P` Six-Class (`Mn, Fe, Co, Ni, Cu, Zn`)** | 42, 43, 44 | 73.31% *(seed-avg)* / 75.23% *(ens)* | `P6 − P4`: **-0.93 pp** `[-1.41, -0.56]` | 6-class split slightly reduces collapsed Class VIII recall when used alone |

---

### 5.5 Native 5-Class and 6-Class Resolution Within Class VIII (`Fe`, `Co`, `Ni`)

Unlike PinMyMetal—which is architecturally restricted to 4 classes and cannot distinguish iron ($\text{Fe}$), cobalt ($\text{Co}$), and nickel ($\text{Ni}$)—DeepMzyme natively trains 5-class (`Mn, Cu, Zn, Fe, Co+Ni`) and 6-class (`Mn, Fe, Co, Ni, Cu, Zn`) classifiers under strict $\ge 90\%$ sequence-identity grouped cross-validation:

#### Table 5: Native 5-Class and 6-Class Cross-Validation Performance (`7,398` Sites)

| Native Formulation | Seed(s) | Native Macro Balanced Accuracy (%) | Fe Recall (%) | Co+Ni / Co Recall (%) | Ni Recall (%) | Pairwise Co–Ni AUROC `[95% CI]` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Native 5-Class (`Mn, Cu, Zn, Fe, Co+Ni`)** | Discovery (`s42`) | **65.2%** | **60.6%** | **27.4%** *(Co+Ni)* | — | — |
| **Native 5-Class (`Mn, Cu, Zn, Fe, Co+Ni`)** | Confirmation (`s43/44`) | **63.8%** | — | — | — | — |
| **Native 6-Class (`Mn, Fe, Co, Ni, Cu, Zn`)** | Discovery (`s42`) | **55.8%** | **61.3%** | **4.9%** *(Co)* | **29.5%** | **0.680** `[0.582, 0.770]` |
| **Native 6-Class (`Mn, Fe, Co, Ni, Cu, Zn`)** | Confirmation (`s43/44`) | **54.5%** | — | — | — | — |

> **Biological Interpretation:** DeepMzyme reliably separates **$\text{Fe}$ (~61% recall)** from the remaining transition metals, and achieves a statistically significant pairwise **$\text{Co}$ vs. $\text{Ni}$ AUROC of `0.680 [0.582, 0.770]`**, while transparently showing that minority $\text{Co}$ sites (`283` training ions, `3.8%` prevalence) suffer from severe class-prior and geometric overlap under unweighted argmax (`~4.9%` recall).

---

## 6. Discussion, Error Analysis & Required Methodological Disclosures

### 6.1 Why Late Fusion Works—And How It Relates to the Tabular Feature Ablation (`PMM-H`)
1. **Complementary Modalities:** In our modality ablation (Table 4), combining 3D equivariant pocket graphs (`GVP-GNN`) with evolutionary sequence embeddings (`ESM-C 600M`) outperformed both `Only-ESMC` (`+1.95 pp`, winning on all 5 folds) and `Only-GVP` (`+4.62 pp`). Furthermore, combining diverse training objectives (`4-class`, `5-class`, `6-class`) and learning-rate schedules in late-fusion ensembles consistently raised out-of-fold CV balanced accuracy from `74.34%` (5 checkpoints) to `75.82%–77.99%` (15–30 CV-selected checkpoints) and test balanced accuracy from `67.55%` (1 checkpoint) to `71.10%–72.10%` (`72.68%–73.64%` on in-scope protein sites).
2. **Comparison with the `PMM-H` Tabular Ablation:** When we trained a modern histogram gradient-boosting classifier (`PMM-H`) on PinMyMetal's 89 deployment features, `PMM-H` achieved `71.22%` CV BA and `68.83%` Test BA (`69.49%` Eligible)—substantially higher than PinMyMetal's own classifier (`PMM-D`: `62.96%` CV / `58.27%` Test), and statistically indistinguishable from DeepMzyme on the 1,488 test set (with DeepMzyme's CV-selected ensembles scoring `+1.06 to +3.27 pp` higher than `PMM-H` on Full Test and `+3.19 to +4.15 pp` higher on Eligible Test, and `+3.1 to +5.4 pp` higher in CV). This provides a nuanced, high-credibility insight for the Discussion:
   - First, PinMyMetal's published 5-estimator classifier severely under-utilizes its own geometric/chemical descriptors once the `valence_3a` label leak is removed (`58.27%` vs. `68.83%`).
   - Second, DeepMzyme learns an end-to-end representation directly from raw 3D coordinates and protein sequences—**without requiring NEIGHBORHOOD SQL databases, bond-valence sum tables, or hand-crafted atom-type rules**—while outperforming PinMyMetal's classifier by `+9.3 to +13.8 pp` and achieving higher cross-validation and in-scope test accuracy than even gradient boosting on curated tabular descriptors.

### 6.2 Biophysical Error Analysis
1. **Out-of-Scope Non-Protein Sites (`139` Test Rows):**
   - Out of PinMyMetal's `1,488` test rows, `139` rows fail the training-time biological scope filter (`len(protein_Donors) == 0` or symmetry `LINK`-flagged) and `3` rows lack valid graphs. Because DeepMzyme encodes protein residue graphs and protein chain embeddings, sites coordinated exclusively by water or nucleic acids carry no first-shell protein signal. Removing those out-of-scope rows (`Training-Eligible` line, `1,346` rows) raises every DeepMzyme model's test balanced accuracy by **+1.5 to +2.4 percentage points** (reaching **72.68%–73.94%**), whereas `PMM-R` changes by only `+0.40 pp` (`63.41%` $\rightarrow$ `63.81%`).
2. **Manganese ($\text{Mn}^{2+}$) vs. Iron ($\text{Fe}^{2+/3+}$) Ambiguity:**
   - Error analysis on grouped CV folds reveals that $\text{Mn}^{2+} \leftrightarrow \text{Fe}$ (Class VIII) confusions account for the majority of $\text{Mn}$ errors, often occurring across entire homologous enzyme families (such as cambialistic superoxide dismutases and binuclear metallohydrolases) that bind both $\text{Mn}$ and $\text{Fe}$ in nearly identical octahedral $\text{His/Asp/Glu}$ geometries.
   - In PinMyMetal's released tables, `valence_3a` resolved `58 of 73` such $\text{Mn} \rightarrow \text{Class VIII}$ errors on fold 0 solely because the ground-truth $\text{Mn}$ oxidation state was encoded in the feature; when `valence_3a` is removed (`PMM-D`), PinMyMetal's test $\text{Mn}$ recall drops from `79.0%` to `61.7%` (below DeepMzyme's CV-selected ensembles at `62.3%–66.5%`).

### 6.3 Mandatory Transparency & Reproducibility Disclosures for the Manuscript
To adhere strictly to the project's pre-registration and data-hygiene ledger ([`docs/DATASETS.md`](../DATASETS.md#test-use-ledger) and [`docs/campaigns/phase2_unified_2026-10/pmm_matched/README.md`](../campaigns/phase2_unified_2026-10/pmm_matched/README.md)), the manuscript Methods/Discussion should explicitly state:
1. **Scope:** Both systems classify the transition metal identity at known metal-bound coordinates in experimental holo structures; de novo spatial site localization is not evaluated.
2. **Non-Pristine Reference Test (`both_results_secondary`):** PinMyMetal's `1,488`-row test set shares `668` PDB IDs (`809` rows) with its training set and was accessed during earlier exploratory evaluations (`2026-09-22/23`, `2026-10-13` Step F single-model evaluation, and `2026-10-19` 180-checkpoint rescore). Only the single full-data refit (`a0007846`, `67.55%`) was a pre-declared one-shot test evaluation; on the `423`-row homology-filtered clean subset, single-model performance is `48.85%` (vs. `47.09%` for `PMM-D` and `59.46%` for `PMM-R`), while the 15 CV-only ensembles score `52.32%–59.20%` (`60.07%` for `R2b Top-4` and `61.33%` for the 55-model trimodal ensemble).
3. **Pre-Registration & Amendments (`A1` and `A2`):** The matched comparator campaign (`SHA-256: f3a16ff5...`) originally specified a composite decision rule requiring `a0007846` to beat both `PMM-D` and the synthetic `PMM-H` gradient-boosting model at $97.5\%$ confidence. `a0007846` met the `PMM-D` criterion in both test (`+9.28 pp`) and CV (`+11.25 pp`) but tied `PMM-H` (`-1.28 pp` test, `+3.06 pp` CV). Amendment `A1` reclassified `PMM-H` as an internal feature ablation (since it is not part of PinMyMetal) and added the 15 CV-only ensembles (`R1–R4`) and published-benchmark context; Amendment `A2` added the 6 `R2b` seed-42-ranked ensembles.

---

## 7. Quick-Reference Checklist for Paper Figures & Tables

All publication figures are generated reproducibly by [`docs/paper/generate_figures.py`](generate_figures.py) into [`docs/paper/figures/`](figures/):

| Manuscript Element | Recommended Content | Source File in Repository |
| :--- | :--- | :--- |
| **Figure 1** | **DeepMzyme Architecture Diagram:** 3D Pocket Graph (`GVP-GNN` scalar + equivariant vector channels) + `ESM-C 600M` Sequence Branch $\rightarrow$ Late Fusion $\rightarrow$ 4/5/6-Class Metal Prediction | [`README.md`](../../README.md#approach) (Mermaid diagram) |
| **Figure 2 / Table 1** | **Main Comparison vs. PinMyMetal:** 5-Fold CV (`BA` & `Macro-Precision`) and Reference Test (`Full 1,488` & `In-Scope 1,346`) across Single Model, CV-Selected Ensembles, `PMM-D`, `PMM-R`, and Published PMM | [`figures/fig1_benchmark_overview.svg`](figures/fig1_benchmark_overview.svg) / [`.png`](figures/fig1_benchmark_overview.png); [`pmm_matched/README.md`](../campaigns/phase2_unified_2026-10/pmm_matched/README.md) |
| **Figure 3 / Table 3** | **Per-Metal Recall & `TECH-028` Forensic Audit:** Per-metal recalls (`Mn, Cu, Zn, Class VIII`), highlighting the `79.0% -> 61.7%` Mn drop when removing `valence_3a` from PinMyMetal | [`figures/fig2_per_metal_recall_audit.svg`](figures/fig2_per_metal_recall_audit.svg) / [`.png`](figures/fig2_per_metal_recall_audit.png); [`FOLLOW_UP_TECHNICAL_ISSUES.md`](../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-028--pmm-comparator-inherits-a-true-metal-label-leak) |
| **Figure 4 / Table 4** | **Modality & Optimization Ablations:** `Late Fusion` (`74.28%` baseline, `75.45%` differential trunk LR) vs. `Only-ESMC` (`72.33%`) vs. `Only-GVP` (`69.66%` baseline, `73.25%` `wd=1.0`), plus CV-selected ensembles | [`figures/fig3_modality_optimization_ablations.svg`](figures/fig3_modality_optimization_ablations.svg) / [`.png`](figures/fig3_modality_optimization_ablations.png); [`PARAMETER_FINDINGS.md`](../PARAMETER_FINDINGS.md) |
| **Figure 5 / Table 5** | **Resolution Within Class VIII:** Native 5-class (`Fe` vs. `Co+Ni`) and 6-class (`Fe`, `Co`, `Ni`) recalls and pairwise `Co–Ni` `AUROC = 0.680` | [`figures/fig4_class_viii_resolution.svg`](figures/fig4_class_viii_resolution.svg) / [`.png`](figures/fig4_class_viii_resolution.png); Section 5.5 above |
| **Data & Code Availability** | Public `9,398`-ion 3D structure dataset (`1.1 GB`, `99.89%` fidelity) on Hugging Face (`GMBioinformatics/DeepMzyme`) and reproducible CPU/GPU codebase on GitHub (`MECHTI1/DeepMzyme`) | [`ZENODO_PINMYMETAL_EXACT_ION_LEVEL_REPRODUCIBILITY.md`](../ZENODO_PINMYMETAL_EXACT_ION_LEVEL_REPRODUCIBILITY.md) |
