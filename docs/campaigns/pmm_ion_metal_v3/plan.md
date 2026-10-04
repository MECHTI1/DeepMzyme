# PMM ion metal v3 — approved plan

Approved by the user on 2026-10-04 and revised the same day after a verified
external review ([log v3-004](log.md#v3-004)). Decisions and their dates are in
the [campaign log](log.md); scientific policy stays in [Plan](../../../Plan.md).
Each step ends with a results table, a dated log entry and a commit. Nothing
here authorizes a GPU start: every start needs the user's explicit OK within
the recorded budget ceiling. Values marked "(default)" were set under the
user's delegation; each may change only by a dated log entry before the first
run of the step it affects.

## Fixed decisions

- Objectives: separate `four_class`, `five_class` and `six_class` arms for
  Only-ESMC, Only-GVP and graph-level late fusion, all evaluated on common-four.
- Checkpoint rule for every v3 arm, fold and refit: cosine learning-rate schedule
  to zero over 50 epochs and the terminal checkpoint
  ([Plan](../../../Plan.md#2-train-the-metal-classification-model)); best-epoch
  values are descriptive only. Its reason is the CV/refit mismatch of
  [TECH-027](../../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-027--cross-validation-and-final-refit-use-different-checkpoint-rules);
  every report states that it was chosen after the v2 fold-0 post-hoc epoch
  analysis.
- Neutral four-versus-five/six test: the baseline recipe, frozen in the A4
  specification before any GPU run, on all five folds (step E). A comparison
  that uses a recipe developed on `four_class` in step D is only "conditional on
  four-class development" and is never reported as the neutral test.
- Decision statistics (user-approved): per comparison, 95% intervals from a
  paired fold-level bootstrap and from a t interval (4 degrees of freedom) on the
  five fold differences. "Better" needs both lower bounds above zero, "worse"
  both upper bounds below zero; otherwise "no clear difference". A target tie
  keeps `four_class`. A five/six arm with a zero mean native recall for any metal
  cannot be called better. Claims across all families need a
  multiplicity-adjusted bound. Details are in the A4 specification.
- PMM comparison: published PinMyMetal numbers only, cited as reported; PMM is
  not retrained. 5-fold CV per-class Mn 90.3, Cu 62.9, Zn 73.8, Class VIII 73.3
  (mean 75.1); test (Figure 2b) Mn 88.6, Cu 59.4, Zn 65.9, Class VIII 57.5
  (mean 67.85), rechecked against the paper before step C. PMM fold IDs were
  never released and its features carry
  [TECH-028](../../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-028--pmm-comparator-inherits-a-true-metal-label-leak).
  The comparison is descriptive, not matched (folds, eligible ions and inputs
  differ). Allowed wording: "higher (or lower) than PMM's published CV (or test)
  values; not a matched comparison". No superiority claim and no interval or
  p-value for the difference. The CV comparison uses the five-fold v3 result
  with folds 1–4 also shown; the full-test comparison states how many of PMM's
  1,488 test rows were scored; the clean subset is never compared with PMM.
- v3 is fully separate from v2: its own fold-set identity, data root
  (`/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v3`), run names
  carrying fold-set, step, family, target, fold and seed, and its own commits.
  v2 folds, runs, tools and the frozen `_code/pmm_core_scope_v2` checkout stay
  untouched.

## Steps

**A. CPU preparation (no cost).**

1. Strict folds, near-copy-disjoint
   ([TECH-020](../../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-020--greedy-k-fold-assignment-concentrates-large-groups-in-fold-0)):
   - Unit: the PDB entry; all ions of one PDB stay in one fold.
   - Sequences: the first-model ATOM-record sequence of every context chain,
     exactly the sequences hashed in the v2 ESMC plan (`sequence_sha256`, checked).
   - Near-copy: two chains of at least 30 residues whose global alignment with
     free end gaps (BLOSUM62, gap open -11, extend -1) has identical aligned
     residues of at least 90% of the shorter chain's length. A prefilter may skip
     only pairs it provably cannot miss.
   - Groups: PDBs linked by any near-copy pair are merged transitively; a group
     is never split.
   - Assignment: each of 200 seeded starts shuffles the groups (no size sorting)
     and places each group where it least increases an imbalance score (sum of
     squared relative deviations of fold ion counts, native Mn/Fe/Co/Ni/Cu/Zn ion
     counts and per-element group counts); the best start is kept.
   - Acceptance (default): every ion in exactly one fold; no PDB, group or
     near-copy pair across folds; every fold holds all six elements with at least
     15 groups each; fold ion counts within ±5% of one fifth; each element within
     ±15% of its fifth; Cu group counts within ±20% of their mean. The report
     lists group sizes and each element's largest single-group share per fold.
     If no start passes, stop and report to the user; never split a group.
   - Outputs: near-copy pair list, PDB-to-group table, `fold_membership.csv`
     (existing columns, group ID in `group_id`, plus `pdbid`), balance report and
     a receipt with builder version, seed, best start and SHA-256 values; a rerun
     must reproduce the same SHA-256.
   - Claim: no near-copy crosses folds; chains below 90% identity can, so this
     is near-copy separation, not homology separation.
2. Code, off by default, each with tests proving unchanged behaviour when off;
   all `src/` changes land before the first GPU run:
   - the fold builder of A1, with tests (transitive merge, chains under 30
     residues never link, a multi-chain PDB stays whole, a cross-fold near-copy
     fails the build, same seed gives the same SHA-256);
   - a v3 runner whose guards bind the cohort, exclusions, ion identities, fold
     file, feature inventory and graph contract, target scheme and code commit,
     and that refuses graph options outside the contract;
   - a trainer option that takes training/validation membership from the
     hash-bound v3 fold file, checks that no group crosses sides, and recomputes
     class weights on the v3 folds;
   - 2–3 concurrent training workers;
   - terminal-checkpoint selection carried through run metadata, the
     selected-checkpoint receipt, validation-prediction export, independent
     replay, the v3 assessor and the refit; in-process test evaluation refused;
     best-epoch values only as descriptive fields; a test run whose best epoch is
     earlier than its last must report its last epoch everywhere;
   - step-D options 3, 4, 5a, 5b and 10, built after the existing modules so the
     shared modules start with identical initial weights for the same seed (new
     modules are freshly initialized); tests include an optimizer-ownership dump
     for 4 and whole-channel vector dropout for 3;
   - a v3 replay tool and a v3 assessor as new files (the v2 tools stay frozen);
     the assessor is tested on synthetic data for bootstrap/t disagreement,
     ties, zero recall, incomplete grids and the Stage 6 tie-breakers;
   - a label-blind clean-subset and test-preparation builder that reuses the A1
     near-copy rule at PDB-entry level and a hashed list of previously opened PDB
     IDs from tracked provenance; tested on training-side or synthetic data only
     and run only in step F.
3. CPU regression gate: with every v3 option off, the new code replays the nine
   pinned v2 fold-0 checkpoints within the v2 replay tolerances
   (`pmm-core-replay-v1`: best checkpoints against the saved predictions, last
   checkpoints against the epoch-50 history), and a short deterministic CPU
   training check matches the frozen v2 checkout (losses and weights within 1e-6).
4. v3 assessment specification, frozen with its SHA-256 in the log before any
   GPU run and shown to the user first. It fixes:
   - identities: fold-set ID and SHA-256, cohort SHA-256, the baseline recipe
     (with AMP and worker count chosen by the step-B gate as declared below);
   - units: one paired difference per fold; seed 42 for all step-E contrasts
     (default); step-D seed-43 runs are screening evidence only;
   - metric: common-four balanced accuracy (BA) at the terminal checkpoint, with
     five/six probabilities summed into Class VIII; mean over folds primary,
     pooled out-of-fold secondary;
   - contrasts: per family five-vs-four and six-vs-four (neutral); per improved
     family final-vs-baseline `four_class`; others descriptive;
   - the interval and outcome rules of the fixed decisions above;
   - rare classes: no common-four class mean recall drop above 3 points, no
     missing or zero common-four recall; native Fe/Co/Ni and Fe/Co+Ni recalls
     reported;
   - Stage 6 selection (default): control = Only-ESMC baseline `four_class`; a
     candidate replaces it only if eligible, interval-supported and best by mean
     common-four BA; tie band 0.2 points; tie-breakers mean minimum recall, worst
     fold, SD, simpler family, baseline before improved recipe; refit seed 42; no
     calibration; no ensemble;
   - failed runs are rerun with the same seed and identity; no replacement
     after results are seen.
5. Pre-fit gate: check per fold and target the labels, common-four collapse,
   class weights and support, and that normalization is fitted on training
   graphs only; measure graph-build seconds per ion to forecast augmented runs
   (every training graph is rebuilt each epoch) and RING runs (cache bypassed;
   RING files must first be generated for the training structures; none exist).
6. Executable v3 recipe in the metal playbook; free space on `/`.

**B. Speed check (first GPU session).**

- Fresh VM through the controller, then short runs on real data (a few epochs
  each): 1, 2 and 3 concurrent workers, AMP off/on, batch size unchanged;
  record time per epoch, GPU memory, RAM and CPU.
- Speed means completed, replay-verified fits per hour, including preparation,
  replay and persistence.
- Concurrency is adopted if that rate rises at least 1.2× (default) without
  memory or CPU pressure, and same-seed concurrent runs differ from a serial run
  no more than two same-seed serial runs differ from each other.
- AMP is adopted only if at least 1.3× faster and one full 50-epoch AMP run (late
  fusion, `four_class`, new fold 0, seed 42) stays within 1.0 common-four BA
  point and 3 points per class recall (default) of its same-seed FP32
  counterpart. The run matching the chosen setting becomes that cell's step C
  baseline.
- Admission uses the slowest worker in a concurrent batch with the controller's
  safety margin (forecast × 1.25 plus 900 seconds). All later v3 runs use the
  chosen configuration except the regression run.

**C. Baseline on new fold 0 (13 runs).**

- First build the six cache sets (four/five/six, each with and without ESMC) in
  the step-B environment; a cache built locally does not match the VM's torch
  build.
- Nine runs: three families × four/five/six, seed 42, v3 rule. They are fold 0
  of the step E neutral grid.
- Three v2-recipe four-class runs (fixed learning rate) on the same fold and
  seed, each read at its best epoch (continuity only) and at its terminal epoch
  50. The schedule effect is fixed-terminal versus cosine-terminal; it is
  descriptive and does not reopen the checkpoint rule.
- One regression run of the new code with the v2 recipe on the old v2 fold 0 at
  v2 execution settings (FP32, one worker, v2 batch size, seed 42, fixed learning
  rate, best epoch). It passes only if all 50 epochs finish, its checkpoints
  replay under `pmm-core-replay-v1`, and its best-epoch and epochs 41–50 mean
  common-four BA are each within 3.0 points (default) of 89.47 and 85.78. The
  band is an engineering convention, not a statistical bound; the strict code
  check is A3. A failure stops v3 for diagnosis; the run is not repeated until it
  passes.
- Fold-0 values sit next to PMM's published CV as context only; they are
  exploratory, not confirmation.

**D. Improvements on new fold 0 (Only-GVP and late fusion; `four_class`; seeds
42 and 43).** Frozen before the first D run; each candidate changes exactly one
setting from its control; fold file, code commit, cache identity and the step-B
configuration stay fixed.

- Controls: the step C four-class baseline of each family (seed 42) plus one
  seed-43 baseline per family (2 runs).
- Score: terminal common-four BA and class recalls on fold 0; Δ = candidate
  minus control with the same seed.
- Screening gate (one-fold screen, not an improvement claim): mean Δ over the
  two seeds at least 1.5 points, Δ above zero for both seeds, and no
  Mn/Cu/Zn/Class VIII mean recall change below −3 points. A crashed run is
  repeated once unchanged; a second failure counts as not passing.

| Round | Candidate (plan rank; status; values are defaults) | Families | Runs |
|---|---|---|---|
| A | 2 mean message aggregation (`--gvp-normalize-message-aggregation`; exists) | GVP, fusion | 4 |
| A | 3 residual dropout 0.1 on scalar and vector updates (new) | GVP, fusion | 4 |
| A | 4 input-vector projection and structural pooling/projection at the GVP learning rate; ESM branch, gate and head unchanged (new) | fusion | 2 |
| A | 5a GVP-only auxiliary metal loss, weight 0.3, training only (new) | fusion | 2 |
| A | 5b ESM modality dropout, p = 0.2 per training example (new) | fusion | 2 |
| B | 6 `--site-geometry-features counts_angles` against a matched `none` control (exists) | GVP, fusion | 8 |
| B | 9a coordinate noise 0.1 Å (`--position-noise-std`); 9b outer-residue dropout 0.1 (`--outer-residue-dropout`); both exist, cost-gated | GVP | 4 |
| B | 10 vector normalization (new) | GVP | 2 |
| B | 11 `--metal-class-weight-mode inverse_sqrt_frequency` (exists) | GVP, fusion | 4 |

Plan ranks are those of the
[improvement plan](../../plans/gvp_and_esmc_evidence_ranked_improvement_plan.md).
Round A is 16 runs including the two seed-43 controls; Round B is 18.

- Round C (gate concatenation, RING edges, linear head and width 256,
  final-vector readout, sequence-separation encoding, label smoothing 0.1;
  [TECH-029](../../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-029--verified-input-and-training-behaviours-with-small-measured-effect))
  is outside this plan's code and budget. Running it needs a separate user
  decision after Round B, a new code drop, new cache sets and new seed-matched
  controls.
- Cost gate (default): a candidate whose measured per-fit forecast exceeds three
  normal fits needs the user's explicit OK; otherwise it is recorded "not tested
  (cost)". Augmentation (9a, 9b) falls under this gate.
- Stopping: rounds run in order and to completion; D ends after a complete
  round in which no candidate passes in either family. If a reforecast shows
  that B–F would exceed the ceiling, the user chooses between stopping and
  closing out at the current evidence or recording a larger ceiling. Untested
  candidates are "not tested"; failed ones "did not pass the one-fold screen",
  never "ineffective".
- Combination per family: no pass keeps the baseline; one pass adopts it; with
  two or more, keep the better of each alternative pair (5a/5b, 9a/9b), run that
  combination once with both seeds against the same controls, and adopt it only
  if it passes the gate with a larger mean Δ than the best single candidate;
  otherwise adopt the best single candidate. No other subsets are tested.
  Only-ESMC keeps the baseline recipe.
- Excluded: hybrid fusion, blends and ensembles, PMM features, ESMC layer
  changes (separate re-embedding project).

**E. Five-fold confirmation (seed 42; rules from A4).**

1. Neutral target test: the baseline recipe for three families × four/five/six
   on all five folds; fold 0 reuses the nine step C runs, folds 1–4 need 36 runs.
2. Improvement check, a separate claim: for each family that passed D, the final
   recipe with `four_class` on folds 1–4 (up to 8 runs), paired with the
   baseline `four_class` arms of (1). Its fold-0 pair is the step-D seed-42 run
   against the step C baseline and is reported separately as the development
   fold; the improvement must show a positive mean gain on folds 1–4. The
   five-fold result is development-validation evidence, not an unbiased
   estimate of generalization.
3. Not planned (default): the final recipe with five/six classes; if ever run,
   it is labelled conditional on four-class development.

**F. Final model and the one test.**

- Stage 6 selection by the A4 rule, Stage 6B full refit with the same checkpoint
  rule, then one prediction pass on PMM's full test set; the clean-subset
  summary is computed from the same predictions.
- Full test: compared with PMM's published test values and labelled "exact,
  possibly overlapped PMM test; includes rows from previously opened tests".
- Clean subset (no chain that is a near-copy of a training chain under the A1
  rule; previously opened PDB IDs removed): secondary by default, because the
  2026-09-24 Zenodo run may have scored the whole test set and test label counts
  were seen on 2026-10-03. It becomes the primary route only if the user records
  it so in Plan before step E, and it is then still labelled "not pristine".
- Before any test access, freeze the primary report, the test-row and
  clean-subset rules with membership hashes, the refit checkpoint and seed, no
  ensemble, and the calibration rule (none unless declared), and record them in
  the [DATASETS ledger](../../DATASETS.md#test-use-ledger). Reading test
  sequences and structures for this preparation is recorded there too.

## Open decisions

- Before step E: the label of the final test (clean subset as "primary, not
  pristine, history disclosed", or both results secondary), after the user's
  Drive/Colab check of the 2026-09-24 Zenodo run.
- Before step F: which PMM test rows count. Recommended: score every
  reconstructable row under the training input contract, flag rather than drop
  symmetry-LINK rows, count unscorable rows as errors, and report "N of 1,488
  scored", with the training-eligible subset as a sensitivity line.
- After Round B: whether Round C runs (see step D).

## Budget

Ceiling **$40 gross** for steps B–F ([log](log.md#v3-003)); controller session
and daily caps and gross-cost accounting are unchanged. Rough estimate from the
v2 forecast ($0.51 per warm serial fit including persistence, $0.78 with
repeated preparation): about $31–39 if concurrency gives close to the 1.5× of
the short 2026-09-28 probe and D stops after Round A or B; about $40–48 without
a concurrency gain. Re-forecast from measured fits per hour after B, after C,
and before each D round or cost-gated candidate. Before any step would exceed
the ceiling, ask the user to stop and close out at the current evidence or to
record a larger ceiling.
