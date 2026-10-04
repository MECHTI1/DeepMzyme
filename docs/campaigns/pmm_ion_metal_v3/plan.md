# PMM ion metal v3 — approved plan

Approved by the user on 2026-10-04. Decisions and their dates are in the
[campaign log](log.md); scientific policy stays in [Plan](../../../Plan.md).
Each step ends with a results table, a dated log entry and a commit. Nothing
here authorizes a GPU start: every start needs the user's explicit OK within
the recorded budget ceiling.

## Fixed decisions

- Objectives: separate `four_class`, `five_class` and `six_class` arms for
  Only-ESMC, Only-GVP and graph-level late fusion, all evaluated on common-four.
- Checkpoint rule for every v3 arm, fold and refit: cosine learning-rate schedule
  to zero over 50 epochs and the terminal checkpoint
  ([Plan](../../../Plan.md#2-train-the-metal-classification-model)); best-epoch
  metrics are descriptive only. Chosen after the v2 fold-0 results were seen;
  its reason is the CV/refit mismatch of
  [TECH-027](../../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-027--cross-validation-and-final-refit-use-different-checkpoint-rules),
  which does not depend on which target leads.
- PMM comparison: published PinMyMetal numbers only, cited as reported (5-fold
  CV per-class Mn 90.3, Cu 62.9, Zn 73.8, Class VIII 73.3, mean 75.1; test-set
  values from the paper's Figure 2b). PMM is not retrained; its fold IDs were never
  released, and its features carry
  [TECH-028](../../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-028--pmm-comparator-inherits-a-true-metal-label-leak).
- v3 is fully separate from v2: its own fold-set identity, data root
  (`/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v3`), run names
  carrying fold-set, step, family, target, fold and seed, and its own commits.
  v2 folds, runs and the frozen `_code/pmm_core_scope_v2` checkout stay untouched.

## Steps

**A. CPU preparation (no cost).**
1. Strict folds: chains at ≥90% identity (over the shorter chain) form one group;
   groups are assigned to five folds in random order with a recorded seed,
   balancing native Mn, Fe, Co, Ni, Cu and Zn (Cu groups spread evenly). Fold
   file, SHA-256, balance report and builder version are frozen before any fit
   ([TECH-020](../../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-020--greedy-k-fold-assignment-concentrates-large-groups-in-fold-0)).
2. Code (off-by-default options, tests proving unchanged behaviour when off):
   v3 runner with guards and own hashes, 2–3 concurrent training workers,
   terminal-checkpoint export, improvement options 3 and 4 below. Land all code
   before building the v3 graph cache once.
3. Executable v3 recipe in the metal playbook; free space on `/`.

**B. Speed check (first GPU session).** Fresh VM setup through the controller,
then about 30–45 minutes on real data, a few epochs each: 1, 2 and 3 concurrent
workers, AMP off/on; record time per epoch, GPU and RAM. Adopt concurrency if
faster without resource pressure. Adopt AMP only if at least 1.3× faster and one
full run matches the FP32 result within normal run-to-run variation. Record the
chosen configuration; all later v3 runs use it.

**C. Baseline on new fold 0 (13 runs).** Nine runs (three families × four/five/six);
three v2-recipe four-class runs (fixed LR, best epoch) for continuity and as the
cosine comparison; one regression run of the new code with v2 settings on the
old v2 fold 0, which must reproduce 89.47 within normal variation. CPU checks of
labels, collapse and class weights per fold. Compare with PMM's reported CV.
Fold-0 results are exploratory, not confirmation.

**D. Improvements on new fold 0 (two seeds; Only-GVP and late fusion;
`four_class` development target).** Each idea alone against the baseline, in
this order (ranks of the [improvement plan](../../plans/gvp_and_esmc_evidence_ranked_improvement_plan.md)):

| Round | Candidate | Family |
|---|---|---|
| A | 2 mean message aggregation; 3 residual dropout 0.1; 4 structural LR group; 5 GVP auxiliary loss or ESM modality dropout | both / both / fusion / fusion |
| B | 6 coordination summaries; 9 coordinate noise or residue dropout; 10 vector normalization; 11 class-weighting variant | both / GVP / GVP / both |
| C | 12 concatenation instead of gate; 13 RING edges; 15 head width; 16 final-vector readout; seqsep encoding and label smoothing ([TECH-029](../../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-029--verified-input-and-training-behaviours-with-small-measured-effect)) | various |

Pass rule, fixed before the first D run: mean gain over two seeds of at least
1.5 common-four BA points and no class recall drop above 3 points. Stop
descending when gains stop; combine passing candidates and test the
combination once. Excluded: hybrid fusion, blends and ensembles, PMM features,
ESMC layer changes (separate re-embedding project).

**E. Five-fold confirmation.** Final recipe for three families × three targets
on all folds; pre-declared neutral four-versus-five/six analysis (paired
bootstrap plus a t-based guard; tie keeps four), rare-class recall protection.
The recipe was developed on `four_class`, which favours four; disclose it.

**F. Final model and the one test.** Stage 6 selection, Stage 6B full refit with
the same rule, then one evaluation of PMM's full test set (comparison with PMM's
published test results) and its clean subset (no ≥90% near-copy in training,
previously opened PDB IDs removed). Before F: settle the 2026-09-24 Zenodo
status and freeze the test usage in [DATASETS](../../DATASETS.md#test-use-ledger).

## Budget

Ceiling **$40 gross** for steps B–F (credit coverage confirmed on 2026-10-04;
[log](log.md#v3-003)). Controller session/daily caps and gross-cost accounting
are unchanged. Estimated total with concurrency: about $25–35. Before any step
would exceed the ceiling, ask the user to stop or to record a larger ceiling.
