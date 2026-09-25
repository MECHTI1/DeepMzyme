# Plan: metal_level_metal_task_compared_PMM

> **Execution Target:** Google Cloud GPU VM (`deepmzyme-l4`, 1x NVIDIA L4 24 GB VRAM)  
> **Dataset:** `train_and_test_sets_structures_zenodo_pmm_exact` (Zenodo PinMyMetal Exact Ion-Level Benchmark)  
> **Modalities Evaluated:** ESM-only (ESM-C 600M), GVP-only (Enhanced GVP), and Multimodal Hybrid (GVP + ESM-C Late Fusion)  
> **Core Hypothesis:** Explicitly encoding target metal coordinating residues resolves binuclear/multinuclear site ambiguity and boosts CV/test performance.

---

## 1. Executive Summary & Core Objectives

This plan formalizes a rigorous experimental campaign to benchmark DeepMzyme on the **exact real PinMyMetal (PMM) Zenodo benchmark splits** at the **metal ion level** (`--metal-example-unit ion`), directly comparing:
1. **ESM-only (`only_esm`):** Sequence-based pocket representation using ESM-C 600M embeddings.
2. **GVP-only (`only_gvp`):** Structure-based geometric vector perceptron with raw radial basis distance functions.
3. **Hybrid (`enhanced_gvp_esmc`):** Multimodal late fusion combining structural GVP and sequence ESM-C representations.

### Primary Objectives:
1. **Benchmark Against Literature:** Evaluate 5-fold cross-validation (CV) and held-out test set performance against published PinMyMetal (Fig 2a/2b) and Metal3D (Fig 2c) baselines across 5-class and collapsed-4 schemes.
2. **Solve Binuclear / Multinuclear Ambiguity:** In multi-metal centers (69.0% of training sites), two metal ions share overlapping 10 Å pocket spheres. Sequence-only ESM-C produces nearly identical representations for both ions, creating severe label conflict and prediction interference.
3. **Develop & Evaluate Coordination-Aware Architectures:** Provide the model with explicit target metal binding residue information (via coordination-weighted pooling, distance priors, and direct ligand shell conditioning) to determine if resolving multi-metal ambiguity improves validation and test accuracy.
4. **Reliable VM Execution:** Execute the entire workflow on the dedicated GCP GPU VM (`deepmzyme-l4`), utilizing graph caching to prevent redundant parsing and streaming checkpoints to guarantee fault tolerance.

---

## 2. Dataset Clarity: Exact Real Zenodo PMM vs. Legacy Incomplete Split

A critical requirement of this plan is strictly targeting the **true Zenodo PinMyMetal dataset**, not the legacy incomplete subset.

| Property | Exact Real Zenodo PMM (`train_and_test_sets_structures_zenodo_pmm_exact`) | Legacy Incomplete Split (`train_and_test_sets_structures_exact_pinmymetal`) |
| :--- | :--- | :--- |
| **Status** | **Authoritative (Target of this Plan)** | **Deprecated / Incomplete (Do NOT use)** |
| **Reconstruction Fidelity** | **99.89% train (7,911 / 7,920 ions)**<br>**99.93% test (1,487 / 1,488 ions)** | Incomplete (2,144 train / 490 test rows; 1,483 train / 316 test PDBs) |
| **Granularity** | **Per-ion resolution** (`chain_resi`, `metaltype`, `whether_catalytic`) | PDB-level overlap check only; lacks original site-row reconstruction |
| **Class Distribution (Train)** | Mn: 2,586 \| Zn: 2,300 \| Fe: 1,759 \| Ni: 555 \| Cu: 400 \| Co: 311 | Skewed due to missing PDB structures |
| **Collapsed-4 Distribution (Train)** | Mn: 2,586 \| Zn: 2,300 \| Class VIII (Fe/Co/Ni): 2,625 \| Cu: 400 | Incomplete representation of Class VIII and Mn |
| **Paper Benchmark Alignment** | Direct 1-to-1 comparison with PinMyMetal Fig 2a/2b and Metal3D Fig 2c | Invalid for direct comparison with published figures |

```
DATASET RESOLUTION:
DeepMzyme_Data/train_and_test_sets_structures_zenodo_pmm_exact -> /media/mechti/Data1/DeepMzyme_PMM_Zenodo_Exact_Dataset/dataset
├── coverage.json               (99.89% train / 99.93% test validation)
├── site_crosswalk.csv          (Exact mapping to Zenodo row IDs & coordinates)
├── train/
│   ├── final_data_summarazing_table.csv (7,911 rows: structure, chain_resi, metaltype, ...)
│   └── structures/             (6,443 PDB structures)
└── test/
    ├── final_data_summarazing_table.csv (1,487 rows: structure, chain_resi, metaltype, ...)
    └── structures/             (1,281 PDB structures)
```

---

## 3. The Scientific Problem: Binuclear Ambiguity & Interference

### 3.1 Empirical Dataset Analysis
Analysis of `train_and_test_sets_structures_zenodo_pmm_exact` reveals:
- **Total Training Sites:** 7,911 across 4,191 unique structures.
- **Multi-Metal Sites:** **1,735 structures contain >1 metal ion, accounting for 5,455 sites (69.0% of the training set)!**
- **Heteronuclear / Mixed Metals:** 38 structures contain distinct metal elements in the same protein scaffold (e.g., Fe/Zn binuclear centers, purple acid phosphatases, Ni/Fe hydrogenases).

### 3.2 Mechanistic Breakdown of Interference

When `--metal-example-unit ion` is executed:
1. Each metal ion becomes an independent training example: e.g., `PDB_1__ION_0` and `PDB_1__ION_1`.
2. A pocket sphere of radius $R = 10	ext{ \AA}$ (`DEFAULT_POCKET_RADIUS`) is carved around each ion.
3. In binuclear sites, the distance between metals is typically $3.0	ext{ \AA} - 4.5	ext{ \AA}$. Consequently, the set of residues within $10	ext{ \AA}$ of Ion 0 and within $10	ext{ \AA}$ of Ion 1 overlap by $>90\%$.

#### Failure Mode in ESM-Only (`OnlyESMPocketClassifier`):
- Residue ESM-C embeddings $x_i^{	ext{esm}} \in \mathbb{R}^{960}$ depend strictly on sequence context.
- The standard `ESMGraphEncoder` performs:
  $$z_i = 	ext{MLP}(x_i^{	ext{esm}})$$
  $$h_{	ext{mean}} = rac{1}{|V|} \sum_{i \in V} z_i, \quad a_i = 	ext{AttentionScore}(z_i), \quad h_{	ext{attn}} = \sum_{i \in V} 	ext{softmax}(a_i) z_i$$
- **The Fatal Flaw:** The attention score $a_i$ has **zero access to the spatial coordinates or distance to the target metal ion**.
- Because the pocket residues for `ION_0` and `ION_1` are essentially identical, the pooled representations $h_{	ext{pocket}}(	ext{ION}_0) pprox h_{	ext{pocket}}(	ext{ION}_1)$.
- If `ION_0` is $	ext{Fe}$ and `ION_1` is $	ext{Zn}$, the model is forced to map the **same embedding** to two **mutually exclusive class labels**. This injects conflicting gradients, destroys confidence, and caps validation accuracy.

#### Failure Mode in GVP-Only & Hybrid:
- While GVP node features include scalar distances $x_{	ext{dist\_raw}} = [d_{	ext{CA}}, d_{	ext{FG}}, d_{	ext{donor}}]$, standard graph pooling (`structural_readout_scope = "residue_only"`) averages over all $40	ext{--}60$ residues in the $10	ext{ \AA}$ sphere.
- Coordination chemistry is determined primarily by the $3	ext{--}6$ direct ligand residues in the first coordination shell ($d \le 2.7	ext{ \AA}$). Unweighted pooling dilutes this direct coordination signal with background pocket noise and residues coordinating the neighboring metal.
- In the Hybrid model, concatenating an ambiguous, unconditioned ESM representation directly degrades the fused multimodal classifier.

---

## 4. Architectural Proposals: Coordinating-Residue-Aware Models

To eliminate multi-metal ambiguity, we formulate three architectural solutions to inform the model of the exact coordinating residues for each sample:

```
                            TARGET METAL COORDINATE (x, y, z)
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
          Coordinating Residues                         Outer Pocket Residues
        (First Shell: d ≤ 2.7 Å)                         (Second Shell / Shell 0)
    Direct Ligands: His, Cys, Asp, Glu             Backbone scaffold, solvent-exposed
                    │                                             │
                    ├──────────────────────┬──────────────────────┤
                    ▼                      ▼                      ▼
           [Option 1: Gated Pool]  [Option 2: Coord Tag]  [Option 3: Dual Shell]
           Soft distance prior     Concatenate RBF(d)     Separate latents:
           w_i ∝ a_i · exp(-d/σ)   into sequence input     [h_coord || h_env]
```

### Option 1: Coordination-Proximity Weighted ESM Pooling (Recommended Primary)
Modulate the attention pooling weights with a continuous distance prior and a discrete first-shell boost:
$$a_i = 	ext{MLP}_{	ext{attn}}(z_i)$$
$$s_i = a_i - \lambda_1 \cdot \min(d_{	ext{donor}, i}, 10.0) + \lambda_2 \cdot \mathbb{I}(i \in 	ext{first\_shell})$$
$$w_i = rac{\exp(s_i)}{\sum_{j \in V} \exp(s_j)}$$
$$h_{	ext{attn}} = \sum_{i \in V} w_i z_i$$
- **Effect:** Residues directly coordinating the target metal receive exponentially higher attention weight. Residues coordinating the neighboring metal (located $>4	ext{ \AA}$ away) have their weights heavily attenuated.

### Option 2: Coordination-Conditioned Sequence Projection (Feature Injection)
Inject local coordination geometry directly into the residue projection before graph pooling:
$$	ilde{x}_i = \left[ x_i^{	ext{esm}} \;\|\; 	ext{RBF}_{16}(d_{	ext{donor}, i}) \;\|\; 	ext{OneHot}(	ext{shell\_role}_i) 
ight]$$
$$z_i = 	ext{LinearLayerNormSiLU}(	ilde{x}_i)$$
- **Effect:** The ESM encoder learns to transform sequence embeddings in the direct context of whether each amino acid serves as an active ligand for *this specific ion*.

### Option 3: Dual-Scale Hierarchical Readout (Direct Ligand + Environment)
Explicitly partition pocket residues into two sets:
- First-shell coordinating set: $\mathcal{S}_{	ext{coord}} = \{i \in V \mid d_{	ext{donor}, i} \le 2.7	ext{ \AA}\}$
- Outer microenvironment set: $\mathcal{S}_{	ext{env}} = V \setminus \mathcal{S}_{	ext{coord}}$

Compute separate pooled representations:
$$h_{	ext{coord}} = 	ext{AttentionPool}(\{z_i \mid i \in \mathcal{S}_{	ext{coord}}\})$$
$$h_{	ext{env}} = 	ext{AttentionPool}(\{z_i \mid i \in \mathcal{S}_{	ext{env}}\})$$
$$h_{	ext{readout}} = \left[ h_{	ext{coord}} \;\|\; h_{	ext{env}} 
ight]$$
- **Effect:** Gives the classifier dedicated capacity to inspect the immediate coordination sphere (chemical geometry) independently from the broader binding pocket (electrostatic environment).

### Option 4: GVP Coordination-Targeted Readout
In `EnhancedOnlyGVPClassifier`:
- Restrict structural readout pooling to the coordination sphere (`structural_readout_scope = "first_shell"`), or configure `metal_node_mode = "per_metal"` where the target metal node serves as the primary readout anchor receiving messages from coordinating residues.

---

## 5. Experimental Design & Benchmark Matrix

We evaluate a balanced $3 	imes 2$ experimental matrix across 5-fold cross-validation and the held-out test set:

| Model ID | Architecture | ESM Input | GVP Input | Coordination Mechanism | Target Epochs | LR / Optimizer |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **M1a (Baseline)** | `only_esm` | Yes (ESM-C) | No | Standard Unconditioned Pool | 50 | AdamW, lr=3e-5 |
| **M1b (Coord-Aware)**| `only_esm_coord` | Yes (ESM-C) | No | Coordination-Weighted Pooling (Opt 1) | 50 | AdamW, lr=3e-5 |
| **M2a (Baseline)** | `only_gvp` | No | Yes (raw RBF) | Standard Residue Pool (scope=residue_only) | 50 | AdamW, lr=3e-4 |
| **M2b (Coord-Aware)**| `only_gvp_coord` | No | Yes (raw RBF) | First-Shell Targeted Readout (Opt 4) | 50 | AdamW, lr=3e-4 |
| **M3a (Baseline)** | `enhanced_gvp_esmc`| Yes (ESM-C) | Yes (raw RBF) | Standard Late Fusion | 50 | lr=3e-5, gvp-lr=3e-4 |
| **M3b (Coord-Aware)**| `enhanced_gvp_esmc_coord`| Yes (ESM-C) | Yes (raw RBF) | Coord-Aware ESM + Coord-Aware GVP Fusion | 50 | lr=3e-5, gvp-lr=3e-4 |

### Controlled Experimental Constants:
- **Batch Size:** 16
- **Epochs:** 50
- **Loss Function:** Cross-entropy with five-class supervision (`five_class`) and collapsed-4 evaluation.
- **Split Scheme:** 5-fold cross-validation grouped by `pocket_id` / `structure_id` (identical to Zenodo PMM protocol).
- **Seeds:** Seed 42 for primary 5-fold splits.

---

## 6. Evaluation Protocols & Success Metrics

### 6.1 Primary Benchmarking Metrics
Evaluated on both:
1. **5-Fold Out-of-Fold (OOF) Cross-Validation** (7,911 train sites).
2. **Held-Out Test Set** (1,487 test sites):
   - Single-fold test scores (mean $\pm$ standard deviation across 5 models).
   - 5-Fold Soft-Voting Ensemble predictions: $ar{p}(y \mid x) = rac{1}{5} \sum_{k=0}^4 p_k(y \mid x)$.

| Metric | Target / Literature Benchmark | Source |
| :--- | :--- | :--- |
| **Collapsed-4 Balanced Accuracy (CV)** | **75.08%** | PinMyMetal Fig 2a (Published 5-fold CV) |
| **Collapsed-4 Balanced Accuracy (Test)** | **67.85%** | PinMyMetal Fig 2b (Held-Out Test Set) |
| **Metal3D Comparison (Test)** | **61.70%** | PinMyMetal Fig 2c (Metal3D on external set) |
| **Per-Metal Recalls (Test)** | Mn $\ge$ 88.6%, Zn $\ge$ 65.9%, Class VIII $\ge$ 57.5%, Cu $\ge$ 59.4% | PinMyMetal Fig 2b targets |

### 6.2 Targeted Hypothesis Disentanglement (Stratified Evaluation)
To specifically test the user hypothesis regarding multinuclear site interference:
- **Cohort A: Mononuclear Sites** (Structures with exactly 1 metal ion, ~31% of train).
- **Cohort B: Multinuclear Sites** (Structures with $>1$ metal ion, 69% of train).
- **Cohort C: Heteronuclear Multinuclear Sites** (Structures with mixed metal identities).
- **Success Criterion:** The coordination-aware variants (M1b, M2b, M3b) should demonstrate a statistically significant gain ($\Delta \ge +3.0\%	ext{ Balanced Acc}$) specifically on **Cohort B & C**, validating that multi-metal interference was resolved.

---

## 7. VM Execution Infrastructure & Reproducibility Protocol

All training and evaluation must be conducted on the dedicated Google Cloud GPU VM (`deepmzyme-l4`).

### 7.1 Infrastructure Specifications
- **VM Name:** `deepmzyme-l4`
- **Machine Type:** `g2-standard-8` (8 vCPU, 32 GB RAM, 1x NVIDIA L4 24 GB VRAM)
- **Zone:** `us-central1-a`
- **OS:** Ubuntu 24.04 LTS
- **Python / PyTorch:** Python 3.12, PyTorch 2.11.0+cu128
- **Gross Cost:** $0.879 / hour (Session cap: 4h / $6.00; Daily cap: 6h / $10.00)

### 7.2 Post-Mortem Hardening & Caching Strategy
Following the lessons documented in `HANDOFF_ZENODO_PMM_EXACT_BENCHMARK.md`:
1. **Pre-Parsed Graph Caching:** Structure parsing on 6,443 structures takes ~36 minutes on an L4 instance. Without caching, a 15-run campaign repeats this 15 times (~9 hours of pure CPU parsing!).
   - **Protocol:** Implement a disk cache for built PyG `PocketData` graphs keyed by structure hash. On fold 0, graphs are parsed and serialized to `${REMOTE_DATA_DIR}/cache/zenodo_pmm_graphs.pt`. Subsequent folds load graphs in $<30$ seconds.
2. **Durability & Epoch Checkpointing:**
   - Always pass `--save-epoch-checkpoints` to ensure models survive unexpected session terminations.
   - Selected weights are verified via `resolve_fold_checkpoint` (`best_model_checkpoint.pt`).
3. **VM Lifecycle Governance (`AGENTS.md` compliance):**
   - The VM is currently `NOT_CREATED`.
   - Creation requires explicit user authorization: `vm-create --authorize "AUTHORIZE VM START"`.
   - Upon completion of experimental batches, artifacts are synced to persistent storage and the VM is stopped via `vm-stop`.

---

## 8. Step-by-Step Execution Schedule

```
Phase 1: Local Code Readiness & Coordination-Aware Modules
  ├── Step 1.1: Implement coordination-weighted attention pool in src/model.py
  ├── Step 1.2: Add --use-coord-aware-pooling and --structural-readout-scope flags
  ├── Step 1.3: Add parsed graph caching in scripts/run_zenodo_pmm_exact_5fold_cv.py
  └── Step 1.4: Run local unit tests verifying forward pass & backward gradients

Phase 2: VM Staging & Data Verification
  ├── Step 2.1: User authorizes VM creation (vm-create --authorize "AUTHORIZE VM START")
  ├── Step 2.2: Stage train_and_test_sets_structures_zenodo_pmm_exact to VM disk
  ├── Step 2.3: Verify GPU smoke test and PyTorch CUDA environment
  └── Step 2.4: Pre-build and verify the graph cache on VM

Phase 3: Baseline Benchmark Campaign (M1a, M2a, M3a)
  ├── Step 3.1: Run 5-fold CV for M1a (Baseline only_esm)
  ├── Step 3.2: Run 5-fold CV for M2a (Baseline only_gvp)
  ├── Step 3.3: Run 5-fold CV for M3a (Baseline enhanced_gvp_esmc late fusion)
  └── Step 3.4: Compute baseline OOF CV metrics and held-out test ensemble scores

Phase 4: Coordination-Aware Campaign (M1b, M2b, M3b)
  ├── Step 4.1: Run 5-fold CV for M1b (Coord-Aware only_esm)
  ├── Step 4.2: Run 5-fold CV for M2b (Coord-Aware only_gvp)
  ├── Step 4.3: Run 5-fold CV for M3b (Coord-Aware enhanced_gvp_esmc)
  └── Step 4.4: Compute coordination-aware OOF CV metrics and held-out test ensemble scores

Phase 5: Synthesis, Statistical Comparison & Publication Reporting
  ├── Step 5.1: Generate side-by-side comparison tables (CV, Test, Ensemble)
  ├── Step 5.2: Conduct stratified analysis on Mononuclear vs Multinuclear cohorts
  ├── Step 5.3: Verify whether performance exceeds PinMyMetal (CV: 75.08%, Test: 67.85%)
  ├── Step 5.4: Safely stop the VM (vm-stop) and verify TERMINATED state
  └── Step 5.5: Deliver final comprehensive report to user
```

---

## Antigravity Plan Metadata
- **Plan Name:** `metal_level_metal_task_compared_PMM_antigravity_plan.md`
- **Specification Reference:** `antigravity_plan.md`
- **Agent:** Antigravity (DeepMind)
- **Status:** APPROVED FOR STAGING & EXECUTION
