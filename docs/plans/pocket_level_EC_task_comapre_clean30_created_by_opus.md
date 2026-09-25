# pocket_level_EC_task_comapre_clean30 (created by Opus)

Status: **planned**. Nothing here has been implemented, smoke-tested or
evaluated unless a line says so explicitly. Written 2026-09-25 from a
read-only inspection of the repository, data tree and prior evidence.

Two separate tasks:

- **Task A — DeepMzyme pocket-level EC on CLEAN_30 metalloenzymes.**
  Only-ESMC, Only-GVP and hybrid (GVP + ESMC), each with and without
  explicit metal-binding-residue information.
- **Task B — the official CLEAN predictor on CLEAN_30.**
  Regime 1 trains on all CLEAN_30 train proteins and is tested on (a) all
  test proteins and (b) the metalloenzyme test subset. Regime 2 trains on
  metalloenzyme train proteins only and is tested on the same test sets.

Both tasks end in one head-to-head table on the same metalloenzyme test
proteins.

---

## 0. Verified facts this plan relies on

All counts were measured on 2026-09-25 from the local data tree.

### 0.1 CLEAN_30 is a 5-fold benchmark with no validation split

`DeepMzyme_Data/CLEAN_all_train_valid_splits/split30/`
(`split30_train_split_{k}.csv`, `split30_test_split_{k}_curate.csv`; tab
separated, columns `Entry / EC number / Sequence`):

| Fold | Full train proteins | Full test proteins | Test proteins with >1 EC1 prefix |
|---:|---:|---:|---:|
| 0 | 8,115 | 1,648 | 28 |
| 1 | 8,138 | 1,614 | 35 |
| 2 | 8,170 | 1,574 | 27 |
| 3 | 8,173 | 1,600 | 31 |
| 4 | 8,212 | 1,534 | 37 |

- Train ∩ test = 0 in every fold. The five test sets are pairwise disjoint
  (7,970 test proteins in total; 10,202 proteins overall).
- **No validation file exists.** Any model selection must use an inner
  validation split carved from each fold's train set.
- No sequence is longer than 1,022 residues, so ESM-1b truncation is not an
  issue.
- About 6.8% of train proteins carry several EC numbers; 2.2% span more than
  one EC1 class.

### 0.2 The metalloenzyme cohort is a strict subset of each fold

Source: `DeepMzyme_Data/CLEAN_30_main` →
`CLEAN_30_shared_single_donor_supported_metal_conservative` (one AlphaFill
donor per UniProt target, deduplication within 2.0 Å, 740 structures).

| Fold | Metallo train proteins (pockets) | Ambiguous EC1 in train | Metallo test proteins (pockets) | Ambiguous EC1 in test |
|---:|---:|---:|---:|---:|
| 0 | 622 (743) | 19 | 118 (139) | 3 |
| 1 | 578 (698) | 17 | 109 (128) | 3 |
| 2 | 562 (668) | 17 | 121 (148) | 4 |
| 3 | 586 (696) | 18 | 98 (119) | 2 |
| 4 | 612 (723) | 17 | 99 (124) | 5 |

- Fold 0 check: 622/622 metallo train proteins are in the full train split,
  118/118 metallo test proteins are in the full test split, and every metallo
  `ecnumber` equals the CLEAN `EC number` string.
- **Pooled unambiguous EC1 metallo test proteins (5 folds): 528.** By class:
  EC1 111, EC2 85, EC3 264, EC4 37, EC5 7, EC6 20, EC7 4. EC5 and EC7 are
  too small for per-fold macro metrics; pooled reporting is mandatory.
- Train EC1 in fold 0 (proteins): EC3 275, EC1 164, EC2 95, EC4 45, EC6 21,
  EC5 19, EC7 3.
- Pockets per protein (fold 0 train): 1 → 511 proteins, 2 → 101,
  3 → 10. Because labels are protein-level, protein grouping matters.
- "Metalloenzyme" is an **operational** definition: UniProt-supported
  transition metal + AlphaFill transfer + MAHOMES-catalytic site. The
  complement of the full CLEAN_30 set is "not in the DeepMzyme metallo
  cohort", not a certified metal-free set. Metal assignments are
  computational (see `docs/DATASETS.md`).

### 0.3 Data-path trap

The materialized fold roots `DeepMzyme_Data/CLEAN_30_train_test_split_{k}/`
currently point to the **multi-donor** `CLEAN_30_shared` source. For example,
fold 0 train has 1,102 rows, not 743, and its `.clean_shared_materialized`
marker says `CLEAN_30_shared`. Passing those directories as they stand would
silently train on the wrong cohort. The fix is covered by Phase 0.

### 0.4 Features are already in place

- ESMC embeddings: 740/740 CLEAN_30_main structures have
  `esm_embeddings/<structure>_chain_A_esmc.pt`. The model is **ESMC-300m,
  960-dim**, embedded from the whole-sequence context of the AlphaFold model.
- EC labels are parsed from the structure file name (`__EC_...`).
  `src/training/labels.py::ec_label_token_from_numbers` returns `None`
  (excluded) when a protein spans more than one prefix at the chosen depth.
  EC1-ambiguous proteins are therefore dropped from DeepMzyme training and
  evaluation.
- The per-structure parse cache (commit `c3728b0`) means all arms can reuse
  one parse.

### 0.5 Binding-residue information: what already exists, and what helped before

| Mechanism | State in code | Family | Prior trained evidence |
|---|---|---|---|
| `is_first_shell`, `is_second_shell`, `ca_to_metal`, `fg_to_metal`, `min_donor_to_metal` node features | **Default ON** in the `conservative` feature set (`src/data_structures.py`) | GVP, hybrid | Always on in every GVP result; never ablated |
| `--omit-node-features` (removes the features above) | Implemented | GVP, hybrid | Not run |
| `--classifier-pool-distance-cutoff X` (pool only residues with Cα–metal ≤ X Å) | Implemented. Applies to Only-ESM pooling (`models.py:325`) and to **both** GVP and ESM branches in the fused model (`model.py:957`) | ESMC, GVP, hybrid | Every pilot ran with cutoff 0; never tested |
| `--site-geometry-features counts / counts_angles` | Implemented | GVP | Metal-level geometry pilot 2026-09-15: B−A +2.06 / −0.97 pp, C−B +0.91 / −1.26 pp across seeds. **No consistent benefit** |
| Explicit metal nodes (`--metal-node-mode per_metal`) | Implemented | GVP | Same pilot: −1.3 to −2.0 pp in both seeds. **Negative** |
| Two-view ESMC (wide 10 Å pool + first-shell pool) | **Not implemented** | ESMC, hybrid | Representation audit only (2026-09-24, ion level): wide-view cosine between sibling ions 0.990 vs 0.920 in the first-shell view. No training |
| `--early-esm-scope first_shell`, `--cross-attention-neighborhood first_shell` | Implemented | early fusion / cross-attention only | Not tested |

**Answer to "did it help before?"** No trained metal-level result shows that
binding-residue information improved performance. Counts and angles were
inconsistent; metal nodes hurt. The first-shell ESMC view is justified only
representationally. The GVP models already receive first-shell flags and
distances by default. This plan therefore tests the question directly for EC,
with a small predeclared set of arms (section 2.4), rather than assuming a
benefit.

### 0.6 A warning from the earlier EC run

On CARE30 metalloenzymes (EC1 v12, 2026-09-14) Only-ESM reached **0.97**
validation EC1 balanced accuracy on a random UniProt-grouped split of a
30%-identity training set, using only 42 validation proteins. Such a high
score is consistent with homologs leaking from train into validation.
CLEAN_30's 30% guarantee holds only between train and test, **not** inside
train. Inner validation here must therefore be homology-aware (section 2.2),
or it will choose checkpoints and arms on inflated signal.

---

## 1. Questions and primary endpoints

| ID | Question | Primary endpoint |
|---|---|---|
| Q1 | How do Only-ESMC, Only-GVP and hybrid compare on pocket-level EC1 for 30%-remote metalloenzymes? | Protein-level EC1 balanced accuracy on the pooled 5-fold metallo test (528 proteins) |
| Q2 | Does explicit binding-residue information improve EC1 for any family? | Paired Δ in inner-validation EC1 balanced accuracy (selection); the outer test is reported only for arms fixed in advance |
| Q3 | Does CLEAN trained on all enzymes do worse on metalloenzymes than on all enzymes? | CLEAN Regime 1: EC1–EC4 metrics on all test vs metallo test vs non-metallo complement |
| Q4 | Does CLEAN trained only on metalloenzymes beat CLEAN trained on everything, when tested on metalloenzymes? | Regime 2 vs Regime 1, same metallo test proteins, paired |
| Q5 | Pocket-structure model vs sequence-contrastive model on the same proteins | Head-to-head table (section 4) |

EC depth 1 is primary, per `Plan.md` / `AGENTS.md` §1c. CLEAN predicts full EC
numbers, so its EC2–EC4 results are reported as secondary. DeepMzyme EC2 is
an optional later extension, not part of this plan.

---

## 2. Task A — DeepMzyme pocket-level EC on CLEAN_30 metalloenzymes

### 2.1 Unit, labels and loss

- `--task ec --ec-label-depth 1`. The example unit is the **pocket**
  (clustered metal site); the ion unit is not used. `--metal-example-unit`
  does not apply.
- Labels are protein-level. Keep `--ec-group-weighting structure_id` (each
  protein contributes equally) and `--ec-class-weight-unit group`
  (inverse-frequency weights counted per protein).
- Validation and test are scored **at protein level**: average the softmax
  over a protein's pockets, then take the argmax. The validation side already
  exists as `val_ec_group_level_1_balanced_acc`, which is the selection
  metric. The test side needs the same aggregation (item I3).
- EC1-ambiguous proteins are excluded by the existing label rule. Their
  counts are logged per fold.
- No contrastive loss (`--ec-contrastive-weight 0.0`), no metal auxiliary
  loss, no RING edges, no augmentation. This matches the EC1 v12 recipe so
  families differ only in architecture.

### 2.2 Inner validation (homology-aware)

For each fold k:

1. Cluster fold k's **metallo train** sequences with MMseqs2 (already used by
   `scripts/run_remote_homology_search.py`):
   `easy-cluster --min-seq-id 0.3 -c 0.8 --cov-mode 0`. Only train sequences
   are used; the fold's test is never read.
2. Hold out about 15% of clusters as inner validation, stratified by EC1 as
   far as clusters allow. Use a fixed split seed of 42. Every EC1 class
   present in train must also appear in validation, except EC7, which has
   about 3 train proteins; log it as validation-absent.
3. Write the result as a frozen, hashed membership CSV per fold
   (`structure_id, cluster_id, role`).

This needs a small, additive code change (item I2): a
`--train-val-group-map <csv>` option in `src/training/splits.py` that groups
by the mapped cluster instead of `pdbid`. The existing
`--explicit-membership-manifest` accepts only metal label schemes, so it
cannot be reused as it stands.

If I2 is not approved, fall back to `--train-val-split-by pdbid` (UniProt
grouping), and label every validation number "optimistic: homologs possible
across train and validation".

### 2.3 Model families

| Family | Flags | Notes |
|---|---|---|
| **Only-ESMC** | `--model-architecture only_esm` | Mean + attention pooling of ESMC-300m residue embeddings over pocket residues, plus 4 site statistics |
| **Only-GVP** | `--model-architecture only_gvp` | Conservative features (already include first-shell flags and metal distances), residue edges 6 Å, extraction 10 Å, residue-only readout |
| **Hybrid (primary)** | `--model-architecture gvp --fusion-mode late_fusion` | GVP + ESMC graph-level late fusion. `AGENTS.md` §1c requires this as the first combined baseline |
| Hybrid (secondary, optional) | `--fusion-mode hybrid` | The repository's own "hybrid" fusion mode. Run only if the gate in 2.6 passes. Kept under a separate name so the two "hybrids" are never confused |

Shared settings: batch 4, weight decay `1e-4`, fixed LR schedule, 30 epochs,
deterministic, checkpoint chosen by inner validation only.

### 2.4 Binding-residue arms (predeclared; no grid)

The rule: at most one "focus" arm per family, the same mechanism across
families where possible, and one diagnostic ablation for GVP.

| Arm | Family | Change from baseline | Existing? |
|---|---|---|---|
| E0 | Only-ESMC | none (all pocket residues pooled) | yes |
| E1 | Only-ESMC | `--classifier-pool-distance-cutoff C*` (pool only the metal-proximal residues) | yes |
| E2 | Only-ESMC | Two-view pooling: wide pocket view ⊕ first-shell view (mask = `is_first_shell`), concatenated before the head | **no, item I4** |
| G0 | Only-GVP | none (first-shell flags and distances already present) | yes |
| G− | Only-GVP | `--omit-node-features is_first_shell,is_second_shell,ca_to_metal,fg_to_metal,min_donor_to_metal` (diagnostic: does GVP use binding-residue information at all?) | yes |
| G1 | Only-GVP | `--classifier-pool-distance-cutoff C*` | yes |
| H0 | Hybrid | none | yes |
| H1 | Hybrid | `--classifier-pool-distance-cutoff C*` (both branches) | yes |
| H2 | Hybrid | ESM branch uses two-view pooling (as E2) | **no, item I4** |

- **C\*** is fixed from data, not outcomes. On fold-0 metallo train only,
  compute the Cα–metal distance of every first-shell residue. C\* is the
  smallest 0.5 Å step that covers at least 95% of them (expected about
  7 Å). Also log the median number of pooled residues. The value is frozen
  before any training.
- Not repeated: `counts_angles` and metal nodes. Prior metal-level evidence
  was inconsistent or negative, and they change graph normalization. They can
  be added only as a separately named follow-up.
- If I4 is not approved, run E0/E1/G0/G−/G1/H0/H1 and record E2/H2 as not
  run.

### 2.5 Training matrix

| Stage | Fits | Purpose |
|---|---:|---|
| A1: LR screen | 3 families × LR {3e-5, 1e-4, 3e-4} × 5 folds × seed 42 = **45** | Choose one LR per family by mean inner-validation BA across folds (EC1 v12 favored 1e-4 for every family) |
| A2: binding arms | 9 arms × 5 folds × seeds {42, 43, 44} = 135, minus 15 baseline seed-42 fits reused from A1 = **120** | Q1 and Q2 on inner validation |
| A3: outer test | inference only | Section 2.7 |

About 165 fits in total. The data is small: about 700 pockets and 30 epochs
per fit.

### 2.6 Decision gates (inner validation only)

- **LR choice (A1):** the highest mean inner-validation EC1 BA over 5 folds.
  Ties within 0.5 pp go to the lower LR.
- **A binding arm "helps"** only if all three hold:
  1. mean paired Δ BA ≥ **+1.0 pp** over the 15 fold×seed pairs;
  2. the paired-bootstrap 95% CI (10,000 resamples of fold×seed pairs) lies
     above 0;
  3. no EC class with ≥ 10 pooled validation proteins loses more than 10 pp
     of pooled recall.

  Otherwise the family's baseline stays primary.
- **G− interpretation:** if G− is no worse than G0 (Δ within ±1 pp), GVP is
  not using binding-residue information. Report that as a finding; it is not
  a failure.
- **Secondary `--fusion-mode hybrid`:** run (5 folds × 3 seeds) only if
  hybrid late fusion beats both single modalities on inner validation
  (paired CI excluding 0 for at least one of them).
- All gates are evaluated and written to `decision_A2.json` **before** any
  outer-test inference.

### 2.7 Outer-test protocol

CLEAN folds are a CV benchmark, not a sealed one-shot test
(`docs/DATASETS.md`), but they are still protected.

- For each fold, family and arm, evaluate on the outer test only after
  `decision_A2.json` is frozen.
- **Primary report per family:** the gate-selected arm (baseline unless an
  arm passed), scored as a 3-seed softmax-mean ensemble of the inner-validation
  checkpoints. Every other arm is reported in a clearly labeled secondary
  table. Test numbers never change which arm is primary.
- There is no refit on train+validation. The checkpoint rule stays the
  inner-validation checkpoint, which avoids choosing an epoch count after
  the fact.
- Record every outer-test access in the `docs/DATASETS.md` test-use ledger.

---

## 3. Task B — the official CLEAN predictor on CLEAN_30

### 3.1 Code and environment

- Official implementation: `https://github.com/tttianhao/CLEAN`. **Pin the
  commit hash** in the run manifest.
- Use a separate conda env (Python 3.10, PyTorch ≥ 1.11, `fair-esm==1.0.2`).
  The DeepMzyme env must stay untouched.
- Start from the existing notebook `CLEAN/train_clean_predictor_baselines.ipynb`
  (implemented: table normalization, clone/install, ESM-1b embedding, orphan-EC
  mutation, distance maps, triplet training, maxsep inference, EC1/EC2
  scoring). It currently tests **only** on the metallo subset. Items I5/I6
  add the other test scopes.
- CLEAN is sequence-only. "Training CLEAN on the structures of CLEAN_30"
  means training on those same proteins' sequences. P0 checks that each
  metallo protein's CLEAN sequence equals the sequence of its AlphaFold
  model / ESMC source (`*_esmc.pt.json:source_sequence`).

### 3.2 Recipe (official, untuned)

- ESM-1b embeddings (1280-d) → triplet-margin model (128-d output) →
  max-separation inference.
- Train with `train-triplet.py --epoch 2500` (the official README's setting
  for 30% splits) at learning rate `5e-4` (the notebook's `TRIPLET_LR`; confirm
  it against the pinned commit's default).
- Orphan ECs go through `mutate_single_seq_ECs` as officially prescribed.
- There is no inner validation, because CLEAN's official recipe uses a fixed
  epoch count. This asymmetry with DeepMzyme is declared, not hidden.
- Seeds: Regime 1 uses 1 seed per fold (cost). Regime 2 uses 3 seeds per
  fold (cheap); report the mean and the per-seed spread.
- Optional: an official SupCon-Hard variant (`--epoch 1500 --n_pos 9
  --n_neg 30 -T 0.1`), only if Regime 1 triplet finishes within budget.
  Name it separately.

### 3.3 Regimes and test scopes

| Job | Train set (per fold k) | Test scopes scored |
|---|---|---|
| **R1: CLEAN-full** | `split30_train_split_k` (~8.1k proteins, all enzymes) | **1a** all test (`split30_test_split_k_curate`, ~1.5–1.65k); **1b** metallo test (98–121); **1c** non-metallo complement (diagnostic) |
| **R2: CLEAN-metallo** | Proteins of `CLEAN_30_main/folds/..._k_train.csv` (562–622), sequences and full EC strings taken from the split30 train file | **2a** metallo test (primary); **2b** all test (diagnostic: how far a metallo-only embedding space generalizes) |

- Scopes 1a/1b/1c come from **one** inference pass. The metallo and
  non-metallo scores are subsets of the same predictions, so 1a vs 1b is a
  pure subset contrast.
- R2 predicts only ECs present in its small training set. Expect low EC3/EC4
  coverage on 2b. Report the number of test ECs unseen in training for every
  scope.
- **Reproduction sanity check:** compare R1 scope 1a (full-EC precision,
  recall and F1, weighted) with the CLEAN paper's split30 figures. If the
  result is far off, investigate before trusting 1b.

### 3.4 CLEAN scoring

- For each protein, the top-1 prediction is maxsep's first (closest) EC.
- Levels EC1–EC4:
  - **any-true top-1 accuracy**: correct if the predicted prefix is in the
    protein's true prefix set (this handles multi-EC proteins);
  - **macro-F1** and **macro-recall** over classes present in the test scope;
  - per-class recall for EC1;
  - for EC1 on the common cohort (section 4), also single-label balanced
    accuracy.
- The support of every class is printed next to it.

---

## 4. Head-to-head comparison

**Common cohort:** the metallo test proteins with an unambiguous EC1 label,
pooled over 5 folds: **528 proteins**. Every model's per-fold predictions
are joined on UniProt ID. A protein missing from any model's output is
reported, not silently dropped.

| Model | Train scope | Input | EC1 BA | EC1 acc | EC1 macro-F1 | EC1..7 recall | EC2 acc (CLEAN only) |
|---|---|---|---|---|---|---|---|
| CLEAN-full (R1) | all enzymes | full sequence, ESM-1b | | | | | |
| CLEAN-metallo (R2) | metallo | full sequence, ESM-1b | | | | | |
| DeepMzyme Only-ESMC, primary arm | metallo | pocket ESMC-300m | | | | | — |
| DeepMzyme Only-GVP, primary arm | metallo | pocket graph | | | | | — |
| DeepMzyme hybrid, primary arm | metallo | pocket graph + ESMC | | | | | — |
| *(secondary)* every other A2 arm | metallo | … | | | | | — |
| *(optional)* C0: whole-protein ESMC control | metallo | mean-pooled full-chain ESMC → MLP | | | | | — |

- **Aggregation:** pooled over all 528 proteins, with a percentile bootstrap
  CI over proteins (10,000 resamples). Also report the per-fold mean ± SD.
  Macro metrics are computed on the pooled set only.
- **Paired tests** on identical proteins: paired bootstrap of Δ balanced
  accuracy and McNemar on correctness, for DeepMzyme-primary vs CLEAN-full,
  vs CLEAN-metallo, and CLEAN-metallo vs CLEAN-full (Q4).
- **Optional C0** (implementation item I7): the cheapest way to separate
  "pocket focus" from "ESMC vs ESM-1b". Same inner-validation protocol, 3
  seeds.
- **Caveats printed with the table:**
  1. DeepMzyme takes a known (AlphaFill-transferred) metal site as input;
     CLEAN needs only sequence.
  2. The language models differ (ESMC-300m vs ESM-1b), and neither
     pretraining set is controlled for test overlap.
  3. CLEAN-full sees about 13× more training proteins.
  4. CLEAN uses a fixed official recipe, while DeepMzyme is selected on inner
     validation.
  5. EC5 has 7 proteins and EC7 has 4, so their recalls are anecdotal.

---

## 5. Implementation items (code; all additive)

| ID | Item | Where | Test |
|---|---|---|---|
| I1 | Fold-view correctness: either re-materialize `CLEAN_30_train_test_split_k` from the conservative source (the notebook's `CLEAN_SHARED_SOURCE="main"` path), or pass `--summary-csv CLEAN_30_main/folds/..._k_{train,test}.csv` with the manifest-backed structure dir. Add a preflight assertion that the row count matches `split_metadata.json` (743, 139, …) | runner / preflight | unit test on fold 0 counts |
| I2 | `--train-val-group-map` (CSV structure → group) in the grouped split | `src/training/splits.py`, `config.py` | split-disjointness test; old behavior unchanged when the flag is unset |
| I3 | EC 5-fold outer runner: per-fold train with inner validation, then outer-test inference that exports **per-pocket probabilities + protein-level aggregation**. Generalize `scripts/run_metal_5fold_cv.py` with `--task ec`, or add `scripts/run_clean30_ec_5fold.py`. Supports `--dry-run` | `scripts/` | dry-run test like `tests/test_generalized_metal_5fold_cv.py` |
| I4 | Two-view ESMC pooling (`--esm-pool-views wide,first_shell`) for `only_esm` and the fused ESM branch; the default stays single-view | `src/model.py`, `models.py`, `config.py` | shape test, a test that the mask matches `is_first_shell`, and a test that the default produces an identical forward pass |
| I5 | CLEAN job matrix: add test scopes `all`, `metallo`, `non_metallo` and R2→`all`; emit per-protein prediction CSVs keyed by UniProt | `CLEAN/` notebook or `CLEAN/run_clean30_baselines.py` | table-count test against section 0 |
| I6 | Unified scorer: joins DeepMzyme and CLEAN predictions on the common cohort; computes metrics, bootstrap CIs and McNemar; writes the comparison CSV/MD | `scripts/score_clean30_ec_comparison.py` | synthetic-prediction unit test |
| I7 *(optional)* | Whole-protein ESMC control C0 | small script | smoke |

Per `AGENTS.md`, run `py_compile` and `tests/smoke_checks.py` after edits.
Local checks run under the 2-core / 3 GB `systemd-run` cap.

---

## 6. Phases, outputs and gates

| Phase | Where | Work | Gate to proceed |
|---|---|---|---|
| **P0: readiness** | local CPU (capped) | I1 counts; ambiguous-EC1 logs; CLEAN-vs-structure sequence equality; MMseqs2 clusters and inner-validation memberships for 5 folds (hashed); C\* computation; label-space check (all EC1 classes present per fold) | every count reconciles with section 0; memberships frozen |
| **P1: smokes** | GPU | 1-epoch fit per family on fold 0 (+ E2 if I4 done); CLEAN R2 fold 0 with about 20 epochs through to scoring; I6 on smoke outputs | all artifacts written; **measure** seconds/fit, parse time and CLEAN embedding time; set the budget ceiling |
| **P2: A1 LR screen** | GPU | 45 fits | `decision_A1.json` |
| **P3: A2 arms** | GPU | 120 fits (+ secondary hybrid if gated) | `decision_A2.json` frozen |
| **P4: CLEAN** | GPU (can run beside P2/P3) | ESM-1b for all ~10.2k sequences + mutated orphans; R1 × 5 folds; R2 × 5 folds × 3 seeds; inference | reproduction sanity check passes |
| **P5: outer test + comparison** | GPU → local | DeepMzyme outer-test inference; I6 scoring | — |
| **P6: report** | local | summary in `docs/notebook_outputs/summaries/summary_clean30_pocket_ec1_vs_clean_<date>.md`; status line in `EXPERIMENT_STATUS.md`; test-use ledger in `docs/DATASETS.md`; reusable recipe block in `docs/EC_TRAINING_PIPELINE_PLAYBOOK.md` | — |

**Run naming:** `clean30ec1_{family}_{arm}_lr{lr}_f{k}_s{seed}`, under
`DeepMzyme_Data/notebook_outputs/runs/pocket_level_EC_clean30/`. CLEAN
models are named `clean30_{R1full|R2metallo}_f{k}_s{seed}`, with outputs in
`CLEAN/work/`.

**Reproducibility:** each run stores `run_config.json` / `run_metadata.json`,
the membership CSV hash, the dataset root and hash, the pinned CLEAN commit,
and library versions.

### 6.1 Compute (estimate, to be replaced by the P1 measurement)

- DeepMzyme: about 165 fits × roughly 3–6 min on an L4 (scaled from about
  280 s per 50-epoch fit on 1,181 pockets on the G4 geometry pilot) comes to
  about 8–17 GPU-h.
- CLEAN: ESM-1b embedding of about 10–12k sequences plus 25 triplet trainings.
  Unmeasured; P1 measures it.
- Provisional ceiling: **20 GPU-h** (about $17.6 gross at the recorded L4 VM
  rate of $0.879/h). If P1's forecast exceeds this, stop and ask the user
  (stop vs raise ceiling), per `AGENTS.md`.
- Hardware: the `deepmzyme-l4` VM through the `~/deepmzyme-vm` controller
  (not yet created; L4 stockouts on 2026-09-24). Colab is the fallback, with
  `scripts/colab_artifact_streamer.py` and `--save-epoch-checkpoints`,
  because a lost Colab VM loses everything.

---

## 7. Out of scope

- DeepMzyme on non-metalloenzymes: there is no metal pocket to build a graph
  on.
- Metal-type prediction, joint metal+EC, and CARE30 (report separately if run
  later).
- EC depth ≥ 2 for DeepMzyme; multi-label EC.
- Any HPO beyond the 3-point LR screen.

---

## Appendix — differences from `docs/plans/pocket_level_EC_task_compare_clean30_antigravity_plan.md`

That plan was written by another agent, and this one is independent. Points
where the repository evidence differs from it:

1. It treats the fold directories as the conservative cohort. Locally, those
   directories are materialized from the multi-donor source (section 0.3).
2. It proposes a 15% UniProt-grouped inner validation with no homology
   control (see section 0.6).
3. Its "Arm C" (`--node-rbf-use-raw-distances false`) changes the RBF
   encoding of distances. It does not add binding-residue information.
4. It cites `src/graph/features.py` and `x_role` as the Arm B mechanism.
   That file does not exist. The first-shell flag is already a default node
   feature, so "adding" it is a no-op; ablating it (G−) is the informative
   test.
5. It says `PARAMETER_FINDINGS.md` supports counts/angles benefits. The
   geometry evidence lives in
   `summary_metal_coordination_geometry_pilot_20260915.md` and shows no
   consistent benefit.
6. It omits Regime 2 on the full test and the non-metallo complement, and
   uses per-fold macro metrics even though EC5/EC7 support is 0–4 per fold.
