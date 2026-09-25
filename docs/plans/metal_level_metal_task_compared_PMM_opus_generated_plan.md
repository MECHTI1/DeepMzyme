# Plan: metal_level_metal_task_compared_PMM (Opus-generated)

**Status: planned.** No code has been changed and no training has run.
**Date:** 2026-09-25. **Author:** Claude Opus 5.5, at the user's request.
**Companion file:** `docs/plans/metal_level_metal_task_compared_PMM.md` was
written separately by another agent. This file does not edit or replace it.
Section 11 lists the points where the two plans disagree.

Scientific policy comes from `Plan.md`. Dataset identity and the test-use
ledger live in `docs/DATASETS.md`. The VM is operated only through
`~/deepmzyme-vm/bin/*` (`docs/GCP_GPU_RUNBOOK.md`). This plan sits under those
documents and does not override them.

---

## 1. What this plan answers

1. **Baseline (paper protocol, metal/ion level).** On the exact real PMM sets,
   what are the 5-fold cross-validation and held-out test results for
   **Only-ESM (ESMC)**, **Only-GVP** and **Hybrid (GVP + ESMC)**? Each graph is
   built around one metal ion (`--metal-example-unit ion`).
2. **Binding-residue awareness.** If each model is told which residues bind
   *this particular ion*, does validation improve for each family? One way is
   to weight those residues differently. The motivating concern is that in
   multinuclear sites, and especially binuclear sites holding two different
   metals, the wide pocket view mixes the neighbouring ion's environment into
   the target ion's representation.
3. **Test transfer.** Do the effects selected on validation carry over to the
   exact PMM held-out test set? Results are compared with PinMyMetal Fig 2a
   (CV, 75.08%), Fig 2b (test, 67.85%) and Metal3D Fig 2c (61.70%).

---

## 2. Fixed definitions and guardrails

### 2.1 Dataset: exact real PMM, and only that

| | **Used in this plan** | **Must NOT be used** |
|---|---|---|
| Directory | `DeepMzyme_Data/train_and_test_sets_structures_zenodo_pmm_exact` | `DeepMzyme_Data/train_and_test_sets_structures_exact_pinmymetal` |
| Membership | PMM `classmodel_train_set` / `classmodel_test_set` source **rows** (one row = one ion) | PMM **PDB IDs**, re-expanded, catalytic/EC-filtered, MAHOMES-rebuilt, ions clustered into pockets |
| Size | 7,911 train ions / 6,443 chain files; 1,487 test ions / 1,281 chain files (99.89% of 9,408 source rows) | ~1.6k pockets / ~2.1k ions |
| Identity | HF bundle SHA-256 `24903f12…6db296`; `coverage.json`; `site_crosswalk.csv` | n/a |

**Hard guard (code, Phase 1).** Before any fold starts, the runner must fail
unless all of the following hold:

- the dataset folder name is `train_and_test_sets_structures_zenodo_pmm_exact`;
- `coverage.json` reports 7,911 / 1,487 resolved ions;
- `train/site_manifest.csv` and `test/site_manifest.csv` match SHA-256 values
  recorded in the campaign manifest;
- the loaded ion counts after skips match the frozen cohort counts (§2.4).

A run that resolves to `…exact_pinmymetal` must stop with an error.

### 2.2 Unit, labels and metrics

- **Unit:** `--metal-example-unit ion`. Each graph holds one target ion, with
  a 10 Å residue sphere around that ion's own coordinates
  (`src/training/metal_examples.py:57`). Shell roles and distances are
  computed from **that ion only**: `metal_coords=[coord]`, and
  `src/graph/shell_roles.py:27` measures donor distances to it. Other metal
  atoms do not appear in the graph at all.
- **Training objective:** `five_class` (Mn, Cu, Zn, Fe, Co+Ni), with
  deterministic collapsed-four evaluation. This matches the existing Zenodo
  runner and the user's earlier choice. It conflicts with `AGENTS.md` §1c,
  which asks for direct `four_class` plus a `six_class` challenger, so the
  conflict is recorded as open question Q2 rather than resolved silently.
- **Primary endpoint:** mean 5-fold **collapsed-four balanced accuracy**
  (`val_metal_collapsed4_balanced_acc`), the scale PMM reports.
- **Co-reported:** native five-class balanced accuracy, macro-F1, per-class
  recall for Mn, Cu, Zn and Class VIII (plus Fe/Co/Ni where the scheme
  allows), and ECE before and after temperature scaling.

### 2.3 Folds: PMM-protocol-like, not PMM's own

PMM describes 5-fold CV but did not publish fold IDs (`docs/VERY_EXACT_PMM_SETS_PLAN.md`,
"Validation-fold identity"). This plan therefore:

- generates **one** 5-fold partition of the train ions, grouped by
  `parent_pocket_id` (`--train-val-split-by pocket_id`) so sibling ions stay
  together, and stratified by active targets;
- pins `--split-seed 42`, separately from the model `--seed`, so every arm and
  every seed sees identical folds (fixes blocker B4 of the ion-unit audit);
- writes `folds/fold_membership.csv` plus a SHA-256 per fold, and makes the
  runner **refuse** any fit whose `split_diagnostics.json` hash differs;
- labels all CV numbers as **"paired DeepMzyme 5-fold CV on PMM train rows"**,
  never as a reproduction of PMM's Fig 2a folds.

**Known optimism.** Grouping by `pocket_id` keeps different chains of one PDB
entry apart in different folds. An optional pdbid-grouped check covers this
(§6, Phase D2).

### 2.4 Frozen cohort

After Phase 0, record the retained ion count per side, the skip reasons, the
class counts and the fold hashes in `campaign_manifest.json`. Every arm must
match it byte-for-byte.

### 2.5 Held-out test policy for this campaign

The user explicitly asked for test results for every arm. To stay consistent
with `Plan.md`, where validation selects and test only reports:

1. All CV fits for every arm finish with `--no-evaluate-test`.
2. **Freeze before opening the test set:** the arm list, the promotion
   decisions (§7), the ensemble rule (softmax mean of the 5 fold checkpoints)
   and the calibration rule (temperature fitted on each fold's validation
   split only).
3. Run **one** test pass covering every frozen arm. Report all of them.
   Test numbers never change a selection, a checkpoint, a hyperparameter or
   the primary report.
4. Record the pass in the `docs/DATASETS.md` test-use ledger. After that, the
   Zenodo test set counts as used for this model generation.

**Test-set caveat to print in every table.** PMM's own train and test sides
share **667 PDB IDs** (4,191 train PDB IDs, 1,179 test PDB IDs; measured
2026-09-25 from `site_crosswalk.csv`). The paper split is not
homology-disjoint. Numbers are comparable with PMM's, but they are not a
generalisation estimate.

---

## 3. Evidence already measured

### 3.1 On the Zenodo exact PMM sets (measured 2026-09-25, CPU, read-only)

Computed from `site_crosswalk.csv` ion coordinates. Only **labelled** ions are
counted. Other metal atoms in the file, such as Mg, Ca or unlabelled copies,
are not.

| cutoff between labelled ions | train ions with a labelled sibling | train hetero-metal sibling ions (pairs) | test ions with a labelled sibling | test hetero-metal sibling ions |
|---|---:|---:|---:|---:|
| 5 Å | 984 (12.4%) | 30 (15) | 68 (4.6%) | **0** |
| 7 Å | 1,158 (14.6%) | 42 (22) | 78 (5.2%) | **0** |
| 10 Å | 1,196 (15.1%) | 44 (23) | 87 (5.9%) | **0** |

PMM's own per-row feature `num_metal_4a > 0` (any metal atom within 4 Å) is
**21.4% train / 20.5% test**.

**What follows from this:**

- Two *different* labelled metals in one site is rare: 0.5% of train ions and
  **zero** test ions. A test-set gain therefore **cannot** come from fixing
  hetero-binuclear confusion.
- The mechanism that can matter on a larger scale is **dilution**. For about
  15% of train ions, and about 21% of all ions counting unlabelled metals,
  the 10 Å sphere overlaps a neighbouring metal's environment. More generally,
  the metal-identity signal sits in the few binding residues, while mean or
  attention pooling spreads it over about 20–40 residues.
- The hetero subset (42 train ions) is reported **descriptively only**. It is
  far too small for a gate.

### 3.2 From the earlier ion-unit audit (on `exact_pinmymetal`, a different cohort)

Source: `~/.claude/plans/five-class-is-training-smooth-snowflake.md`. These
results support the hypothesis but must be re-measured on Zenodo (Phase 0c):

- Raw ESMC mean-pool cosine between sibling ions: **0.990** for the wide 10 Å
  view against 0.765 for random pairs. For hetero-target siblings it is
  0.991 wide against 0.920 for the first shell, roughly **9×** more separation
  from the first-shell view.
- 21 first-shell composition counts alone reach **0.61 collapsed-four balanced
  accuracy** with a random forest. The signal is concentrated in the binding
  residues.
- 18 of 2,144 ions have **zero** first-shell residues at 3.0 Å (23 at 2.7 Å),
  so a first-shell-only pool needs a non-empty fallback.

### 3.3 What each family currently sees (code facts)

| family | sees which residues bind this ion? | how the residues are pooled |
|---|---|---|
| Only-ESM (`OnlyESMPocketClassifier`, `src/model_variants/models.py:243`) | **No.** Input is `x_esm` only. `site_metal_stats` is the constant `[0,1,0,0]` in ion mode. | Mean plus attention over **all** residues in 10 Å (`src/model.py:274`) |
| Only-GVP (`src/model.py:629`) | **Yes, as a node feature** (`x_role` first/second-shell flags plus distance RBF) | Mean plus attention over all residues (`_structural_pool_mask`) |
| Hybrid (`--fusion-mode hybrid`: early ESM injected into GVP nodes plus a late ESM branch, `src/model_variants/factory.py:82`) | GVP side yes; the late ESM branch no | GVP pool over all residues; ESM pool over all residues; `early_esm_scope` default `all` |

Only-ESM is the family most exposed to dilution. It is where the
binding-residue signal is most likely to help, and it is also the cheapest
family to screen.

---

## 4. Blockers found; fix before any GPU spend

| # | blocker | evidence | fix |
|---|---|---|---|
| **K1** | **No ESMC embeddings exist for any Zenodo structure.** 0 / 6,443 train and 0 / 1,281 test chain files match (`…__EC_0.0.0.0` stems vs. `…__EC_<real>` in `DeepMzyme_Data/esm_embeddings`). Both runners pass `--allow-missing-esm-embeddings`, so every ESM-using arm would **silently train on missing ESM input**. | `embedding_path_candidates`, `src/training/esm_feature_loading.py:220`; `scripts/run_metal_5fold_cv.py:284` | Generate ESMC-600M (1152-d) embeddings on the VM into a **new** directory, `DeepMzyme_Data/esm_embeddings_zenodo_pmm_exact/`. **Remove** `--allow-missing-esm-embeddings` for ESM-using arms so they fail closed. Preflight asserts 100% coverage of retained ions. |
| **K2** | **No external residue features exist for any Zenodo structure** (0 of 2,947 `updated_feature_extraction` folders). The runner passes `--allow-missing-external-features`, so GVP burial and electrostatics channels would be zero-filled. | `ls DeepMzyme_Data/updated_feature_extraction` | Decision Q4: generate the features on the VM, or run **every** arm without them and state that explicitly. The choice must be the same for all arms. |
| K3 | The runners define only `benchmark_only_esm`, `benchmark_enhanced_only_gvp` and late-fusion `benchmark_enhanced_gvp_esmc`. There is **no hybrid arm** and no variant arms. | `scripts/run_metal_5fold_cv.py:88-120` | Add explicit, named `MODEL_CONFIGS` entries (§5.5) instead of free-form flag passthrough. |
| K4 | Model seed and split seed are coupled, so seed 43 would reshuffle the folds. | ion-unit audit B4 | Pass `--split-seed 42` in every command (§2.3). |
| K5 | Ion-unit validation export and replay would fall back to pocket mode. | ion-unit audit B1, `src/export_validation_predictions.py:365` | Pass `metal_example_unit` from the saved run config into the reload. |
| K6 | New model flags must be dropped in every factory branch that does not use them, and threaded into the runner's checkpoint re-build used for test evaluation. | `src/model_variants/factory.py:111-158`; `evaluate_test_set_for_fold` | Add each new kwarg to the pop lists. Rebuild the model at test time from `checkpoint["config"]`, not from the runner's defaults. |
| K7 | The bootstrap resamples ions independently, so the confidence interval is too narrow for multi-ion sites. | ion-unit audit B7, `src/training/final_test_reporting.py:246` | Resample clusters by `parent_pocket_id` for every CI in this plan. Carry `parent_pocket_id` in the prediction artifacts (B9). |
| K8 | In fusion models, `gvp_attn_pool` and `gvp_fusion_proj` fall into the *head* optimizer group and train at `3e-5` rather than the GVP learning rate. | `src/training/run.py:1492-1512` | **Leave unchanged in this campaign.** It affects baseline and variant equally, so paired deltas stay fair. Record it; it is a candidate for a separate study. |
| K9 | The VM does not exist yet (`vm-status`: `NOT_CREATED`; L4 stockouts on 2026-09-24). Controller caps are 6 h and $10 gross per day. | `~/deepmzyme-vm/bin/vm-status` | See §8 and budget question Q5. |
| K10 | The local root disk is 98% full (6.6 GB free). | `df -h /` | Keep runs on the VM persistent disk. Mirror to the mounted 1.8 TB `/dev/sda1` rather than `/`. |

---

## 5. Proposed "binding-residue awareness" mechanisms

Design rules for every mechanism:

- **Ion-specific and label-free.** Mechanisms use only the target ion's
  coordinates and residue geometry. They never use the element or label of a
  neighbouring metal. (About 96% of labelled sibling pairs share a metal, so
  the neighbour's element would nearly leak the target.) A unit test enforces
  this by relabelling a sibling's element and asserting identical descriptors
  and logits.
- **Nested with the baseline.** Every new parameter is initialised so that at
  step 0 the model equals the baseline, with zero-initialised biases and gates.
  Any difference then comes from learning, not from a changed initial function.
- **One flag per mechanism**, stored in `run_config.json`, written as a plain
  `config.<attr>` in the `build_pocket_classifier(...)` call so the replay AST
  parser accepts it (ion-unit audit B5).
- **First-shell definition:** the project's live `DEFAULT_FIRST_SHELL_CUTOFF =
  2.7 Å` on side-chain donor atoms, unchanged. The earlier audit showed that
  2.7 vs 3.0 Å does not change the verdict, and changing the default would
  alter `x_role` for every baseline.
- Graphs with an empty first shell fall back through
  `ensure_nonempty_pool_mask` (`src/model.py:291`). Report the fallback rate.

### M0: distance-restricted pooling (no code change)

Existing flag `--classifier-pool-distance-cutoff 7.0`, where 7 Å is the CA
distance to the target ion. Pooling is restricted to residues near this ion;
for GVP it also restricts the structural readout (`_structural_pool_mask`).
This is a crude "only binding neighbourhood" baseline and the cheapest check
of the hypothesis. The 7 Å value is fixed in advance: first-shell CA atoms
typically sit 5–6.5 Å from the metal. It is not tuned.

### M1: two-view pooling (wide + first shell of this ion)

`--binding-residue-pooling two_view`. The model pools the same residue states
twice, once over all residues and once over this ion's first-shell mask
(`shell_mask_from_roles(x_role, "first_shell")`), then concatenates the two.

- **Only-ESM:** `ESMGraphEncoder` is called twice, and the input of
  `esm_fusion_proj` grows from `2·proj` to `4·proj`.
- **GVP / Hybrid:** the same applies to the GVP readout (`pool_graph_states`
  ×2, wider `gvp_fusion_proj`) and to the late ESM branch.
- Nesting: the first-shell half of the fusion input layer is zero-initialised.

### M2: role- and distance-biased attention ("different weights")

`--binding-residue-pooling attention_bias`. This is the user's idea of giving
binding residues their own weights, implemented directly: the logits of
`AttentionPool` (`src/model.py:242`) become

```
logit_i = score(h_i) + w_role · x_role[i, :3] + w_dist · RBF_8(d_i)
```

Here `d_i` is residue i's distance to the target ion, and `w_role` and
`w_dist` are zero-initialised vectors (about 11 parameters per pool). The mean
branch is replaced by the same weights applied softly:
`Σ softmax(w_role·role + w_dist·RBF)·h`. The model can then *learn* how much
more the binding residues count, instead of receiving a hard mask. The learned
`w_role` and `w_dist` are logged per fold as an interpretable result, for
example "first-shell residues weighted e^{w}× more".

### M3: ion-site descriptor (fills the dead `site_metal_stats` slot)

`--ion-site-descriptor first_shell_nuclearity`. This adds a fixed, label-free
per-ion vector to the site branch that every family already has:

- the 20 residue-type counts of this ion's first shell, plus shell size;
- donor-atom counts (N, O, S) within 2.7 Å;
- **element-blind nuclearity:** the number of other metal atoms within 4 Å
  and within 6 Å of this ion, and the distance to the nearest one. The
  neighbour's element is never used.

It is stored as a new graph field, and the site-encoder input widens only when
the flag is on, so baseline tensors stay byte-identical.

**Built-in correctness check against the paper's own per-ion features.** The
Spearman correlation of our recomputed CHED count, coordination number and
`n_metal_4Å` against `site_crosswalk.csv` columns `source_ched_count`,
`source_coordnum_inner` and `source_num_metal_4a` must be ≥ 0.8. If it is
not, the extraction is wrong and M3 does not run.

M3 is the closest analogue of what PMM itself feeds its classifier. Note that
M3 by itself cannot separate identical-environment siblings, which is M1 and
M2's job.

### M4: hybrid-only, first-shell early injection (no code change)

`--fusion-mode hybrid --early-esm-scope first_shell`. ESMC is injected into
GVP node features **only on this ion's binding residues** before message
passing. The option exists already (`src/model_variants/models.py:546`).

### Mechanism × family applicability

| | M0 | M1 | M2 | M3 | M4 |
|---|---|---|---|---|---|
| Only-ESM | ✓ | ✓ | ✓ | ✓ | n/a |
| Only-GVP | ✓ | ✓ (readout) | ✓ (readout) | ✓ | n/a |
| Hybrid | ✓ | ✓ (both pools) | ✓ (both pools) | ✓ | ✓ |

Only-GVP already receives `x_role` as a node feature, so its expected gain is
smaller. It gets its focus through the readout and M3.

### 5.5 Named arms (runner `MODEL_CONFIGS` keys)

`pmmz_ion_only_esm`, `pmmz_ion_only_gvp`, `pmmz_ion_hybrid`: the baselines.
Each variant appends a suffix: `__m0_dist7`, `__m1_twoview`, `__m2_attnbias`,
`__m3_sitedesc`, `__m4_early_fs`, or a combination such as `__m2m3`. Baseline
learning rates are carried over from the executed exact-PMM runs:
Only-ESM `3e-5`; Only-GVP `3e-4` with raw RBF; Hybrid `3e-5` with
`--gvp-learning-rate 3e-4` and raw RBF, `early_esm_dim 32` and dropout 0.2.
All arms use 50 epochs, batch size 16, `--save-epoch-checkpoints`,
`--split-seed 42` and model seed 42. Hybrid had no prior executed LR, so it
reuses the late-fusion LR pair; this is noted as untuned.

---

## 6. Phases and experiment matrix

All GPU work runs on the DeepMzyme L4 VM. Fit counts assume 5 folds.

### Phase 0: VM, data and inputs (GPU used only for ESMC)

- **0a.** Run `vm-create` with a capped duration, then
  `vm-setup --stages driver,env,smoke,ssh`, and check out the plan commit.
- **0b.** Download the HF bundle, verify its SHA-256, and run
  `scripts/verify_zenodo_pmm_ion_dataset.py`. Build the parse cache once
  (`DEEPMZYME_PARSE_CACHE_DIR` on the persistent disk, 8 parse workers).
- **0c.** Generate ESMC for all 7,724 chain files with
  `python -m embed_helpers.esmc --structure-dir … --out-dir …/esm_embeddings_zenodo_pmm_exact --device cuda`,
  then assert 100% coverage. Then make **CPU-only measurements** on Zenodo:
  sibling cosine (wide vs first shell); first-shell empty rate; the M3
  descriptor correlation against the PMM crosswalk; and a grouped-CV random
  forest on M3 descriptors, which serves as the floor any neural arm must
  clear.
- **0d.** K2 decision (external features) is executed consistently.
- **0e.** Freeze the cohort and folds (§2.3–2.4) in `campaign_manifest.json`.

**Gate 0:** 100% ESMC coverage, verifier passes, 5 fold hashes frozen, M3
correlation ≥ 0.8.

### Phase 1: code (local, capped at 2 cores / 3 GB), no GPU

Fix K1 and K3–K7. Implement M1, M2 and M3 behind flags, together with the
dataset guard (§2.1) and the fold-hash guard (§2.3). Tests:

- the existing ion-unit tests (`tests/test_metal_ion_examples.py`,
  `tests/test_generalized_metal_5fold_cv.py`) plus `tests/smoke_checks.py`;
- **new:** nesting (flag on with zero-init gives identical logits to the
  baseline on a fixed batch); a sibling-element relabel invariance test; an
  empty-first-shell fallback test; a factory pop-list test for all three
  architectures; checkpoint → test-time rebuild round-trip for every arm;
  and dataset-guard rejection of `exact_pinmymetal`.

**Gate 1:** all tests pass and a clean git tree is committed
(`git_dirty: false` in every fit).

### Phase A: baselines (15 fits)

`pmmz_ion_only_esm`, `pmmz_ion_only_gvp` and `pmmz_ion_hybrid` × 5 folds,
with `--no-evaluate-test`. Fold 0 of each family runs first as a timed smoke:
confirm `nvidia-smi` utilisation above 0 and the ion count, then measure
min/epoch. §9 is re-forecast from these measurements before the remaining
folds start.

### Phase B: mechanism screen on Only-ESM (20 fits)

`__m0_dist7`, `__m1_twoview`, `__m2_attnbias` and `__m3_sitedesc` × 5 folds.
Only-ESM is the cheapest family and the most exposed to dilution (§3.3). The
screen picks at most **two** mechanisms to carry forward (gate §7). If none
passes, Phase C still runs M3 and M4, because they are ESM-independent.

### Phase C: carry winners to GVP and Hybrid (up to 25 fits)

- Only-GVP + each carried mechanism (≤ 2 × 5 fits).
- Hybrid + each carried mechanism (≤ 2 × 5 fits).
- Hybrid + M4 (5 fits).
- If both carried mechanisms passed and are complementary (for example M2 and
  M3), add **one** combined arm in the family that gained most (5 fits).

### Phase D: confirmation (conditional; separate authorisation)

- **D1.** Model seed 43, same frozen folds, for baseline and the best variant
  in each family that passed §7 (≤ 6 × 5 = 30 fits). The mode is
  `group_kfold_seed_repeat` in spirit.
- **D2 (optional).** Re-run baseline and best variant for the best family
  with `--train-val-split-by pdbid` (10 fits), to show that the gain does not
  depend on chains of the same PDB leaking across folds.

### Phase E: one held-out test pass

For every frozen arm (A, B and C, plus D if it ran): all 5 fold checkpoints
are scored on the 1,487 test ions. Output includes per-fold metrics, the
5-fold softmax-mean ensemble, a temperature-scaled calibration view, and
cluster-bootstrap 95% CIs. Paired test deltas are reported for each family
against its baseline. Also reported: the **multi-ion vs mono-ion stratified
test** (78 test ions have a labelled sibling within 7 Å, so it is descriptive
only), and the comparison tables against PMM Fig 2a/2b and Metal3D.

---

## 7. Decision gates (validation only)

For a variant vs its own family baseline on the same folds and seed:

- **Primary:** Δ mean collapsed-four balanced accuracy over 5 folds.
- **Promote / carry forward** if all of the following hold:
  1. Δ ≥ **+1.0 pp**;
  2. the lower bound of the paired **cluster** bootstrap 95% CI on the pooled
     out-of-fold predictions exceeds 0 (10,000 resamples by
     `parent_pocket_id`);
  3. at least 4 of 5 folds show a positive delta;
  4. **rare-class protection:** no collapsed-four class recall drops by more
     than 3 pp, with Cu, the smallest class at 400 train ions, checked
     explicitly;
  5. native five-class balanced accuracy does not fall.
- **Tie-break** (Δ < 1 pp but CI > 0): prefer the arm with fewer added
  parameters, then the lower validation ECE.
- **Mechanistic check** (reported, not gated): the variant's gain should be
  larger on the multi-ion stratum (train ions with a labelled sibling within
  7 Å) than on mono-ion ions. If the gain appears only on mono-ion ions, the
  mechanism is "focus on binding residues" in general, not "de-interference".
  The write-up must say which.

These thresholds are proposed here, before any run, and are frozen in
`campaign_manifest.json` at Gate 0. They must not be changed after results
are seen.

---

## 8. VM execution protocol

- **Lifecycle** runs only through `~/deepmzyme-vm/bin`:
  `vm-create --hours N --authorize 'AUTHORIZE VM START'` (user phrase
  required), `vm-setup`, `vm-status`, `vm-stop`. The Google-side
  `--max-run-duration` STOP always stays on.
- **Durability** (lessons from the lost Colab fold 0):
  - runs write to the VM **persistent** disk, not tmpfs;
  - every fit uses `--save-epoch-checkpoints`;
  - a local mirror of the run directory is synced every 10 min to the mounted
    `/dev/sda1` (not `/`, which is 98% full);
  - the queue runs under `tmux` with `nohup`, so a laptop sleep does not
    matter;
  - resume is idempotent through the runner's `--skip-existing`, keyed by
    completed epoch count.
- **Liveness** is judged by `nvidia-smi` utilisation and new `val_metrics.csv`
  rows, never by CPU time.
- **Throughput:** the parse cache is built once. In the Phase A smoke, measure
  whether two Only-ESM fits in parallel on one L4 give ≥ 1.5× throughput; if
  they do, use pairs for Phase B.
- **Session sizing:** each VM session is capped below the remaining daily cap
  and ends with `vm-stop`. The queue is ordered so a session boundary always
  falls between fits.

---

## 9. Compute forecast (to be replaced by Phase A measurements)

No fit has been measured on the Zenodo ion cohort. That cohort has about
3.7× the train examples of the exact-PMM cohort, where Only-ESM took about
13 min per fold. The planning ranges are therefore **ESM 0.4–0.8 h, GVP
0.6–1.0 h and Hybrid 0.7–1.2 h per 50-epoch fit.**

| phase | fits | L4 hours (range) | gross $ at $0.879/h |
|---|---:|---:|---:|
| 0 (setup, ESMC, CPU audit) | none | 2–3 | 2–3 |
| A baselines | 15 | 8.5–15 | 7.5–13 |
| B ESM screen | 20 | 8–16 (4–8 if pairs run in parallel) | 3.5–14 |
| C carry-over | ≤ 25 | 15–27 | 13–24 |
| E test pass | none | 1–2 | 1–2 |
| **Total without D** | **≤ 60** | **~35–63** | **~$31–55** |
| D1 (conditional) | ≤ 30 | 20–30 | 18–26 |

This exceeds the controller's **6 h / $10 per day** cap. Under `AGENTS.md`
"Accepted-work budget escalation", the user must choose one of two options
before Phase A starts:

1. keep the daily cap and run over about 6–11 VM-days; or
2. record a larger ceiling.

The grid does not change with either choice (Q5).

---

## 10. Outputs

Everything goes under `runs_zenodo_pmm_exact_ion/metal_level_metal_task_compared_PMM/`:

- `campaign_manifest.json`: data SHAs, cohort, fold hashes, gates, arm list,
  git SHA and library versions;
- `<arm>_fold<k>/`: `run_config.json`, `val_metrics.csv`, best and last
  checkpoints, validation predictions with `parent_pocket_id`;
- `cv_summary.csv` and `cv_paired_deltas.csv`: per arm and per fold, with
  cluster CIs and gate verdicts;
- `stage_decisions.json`: the carried and promoted arms, frozen before
  Phase E;
- `test_summary.csv`, `test_paired_deltas.csv`, and the PMM / Metal3D
  comparison table (Markdown);
- `learned_binding_weights.csv`: M2 `w_role` and `w_dist` for each fold;
- a summary note at `docs/notebook_outputs/summaries/summary_pmmz_ion_binding_residue_<date>.md`,
  plus updates to `EXPERIMENT_STATUS.md` and the `docs/DATASETS.md` test ledger.

---

## 11. Where this plan differs from the companion plan

- **The "same embedding → conflicting labels" failure mode does not occur as
  stated.** Ion graphs are built around each ion's own coordinates, and no
  sibling pair had identical 10 Å residue sets in the earlier audit. The real
  effect is near-degeneracy (cosine about 0.99) combined with dilution.
- **Hetero-metal binuclear sites are rare.** They are 0.5% of train ions and
  0 test ions (§3.1), so they cannot drive a test gain and cannot support a
  gate.
- **The ESMC input is 1152-d** (ESMC-600M in this repo), not 960-d.
- **Missing embeddings and features.** K1 and K2 (no Zenodo ESMC embeddings,
  no external features, and silent `--allow-missing-*` flags) must be fixed
  first. Otherwise every ESM-arm result is invalid.

---

## 12. Open questions for the user

- **Q1. What "hybrid" means.** This plan uses `--fusion-mode hybrid` (early +
  late ESMC), as named. Should the published-benchmark late-fusion model
  (`benchmark_enhanced_gvp_esmc`) also run as a reference baseline (+5 fits)?
- **Q2. Training objective.** Keep `five_class` with collapsed-four reporting
  (the existing runner and your earlier choice), or switch to direct
  `four_class` to match PMM's four classes and `AGENTS.md` §1c?
- **Q3. pdbid-grouped check.** Should D2 run?
- **Q4. External features (K2).** Generate `updated_feature_extraction` for
  the 7,724 Zenodo chain files on the VM, or run all arms without them?
- **Q5. Budget (§9).** Keep 6 h / $10 per day and run over many days, or set a
  larger campaign ceiling? D1 needs its own authorisation either way.
