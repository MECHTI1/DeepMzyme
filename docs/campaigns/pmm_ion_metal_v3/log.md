# PMM ion metal v3 — decision log

Dated user decisions and STATUS history for this campaign, newest first.
Current authority: [EXPERIMENT_STATUS.md](../../../EXPERIMENT_STATUS.md).

## v3-007

2026-10-05 — step A checkpoint; **step A not yet declared complete** (the A4
specification is still to be frozen). The user's four code-review findings and
a follow-up were verified against the code; all were real and are resolved
with tests (`0f92bf1`, `afa4767`, `8eef4ef`):

1. Step D now pools passes across completed rounds, blocks every decision
   while a required run is missing, stops after a complete round without a
   pass, and adopts a combination only if it passes and beats the best single
   candidate.
2. Runs refuse changed runner files or recipe definitions (manifest hashes)
   and require the frozen A4 specification whose cohort, fold and baseline
   identities match the campaign; the assessor checks the same before
   assessing, and rejects completed runs recorded with other runner files.
3. A run name is claimed atomically for all lanes; archiving a failed or
   interrupted attempt requires, on the same host, a stopped worker, a stopped
   training child and a free lane lock; a rerun must match the archived
   identity.
4. A3 gives an explicit verdict bound to the tested source-tree hash.

A3 **accepted** (`audits/a3_acceptance_20261005T122647Z`): the replay
(`a3_regression_20261005T102411Z`) passed all nine pinned v2 fold-0
checkpoints (best checkpoints within `pmm-core-replay-v1`, worst probability
difference 4.5e-6, all discrete fields equal; every epoch-50 checkpoint matches
its history), and the training check (`a3_regression_20261005T101042Z`) passed
all four configurations bit-identically. Tested source-tree hash
`5a120b2c514f3b03ceefb8e9f5b30836d8bbb47968cc62ba2557f885f71c584d`, unchanged
since both audits started (bound post hoc by `--accept`); frozen v2 tree
`adc95c42…` matches its pin. No `src/` change followed, so no A3 check needs
repeating. A5 was repeated on the final runner and passed
(`audits/a5_prefit_20261005T122704Z`). Full test suite: only the 2 known
failures and 32 known order-dependent errors.

Known limit: claims are local coordination files, not persisted; a unit
interrupted by the loss of its host cannot be archived without a user
decision, because its worker's stop cannot be verified.

## v3-006

2026-10-05 — plan step A2–A6 work, **step A not yet complete** (open: the A3
replay of all nine pinned checkpoints, the user's four code-review findings,
then freezing the A4 specification). Branch `v3-step-a` (not merged or pushed):
`2acef49`, `0a0c534`, `46c1515`, `335b9c0`, `2b4b220`, `22f4e29`, `ea086fb` and
this entry's commit.

- A2 code: v3 runner with lanes, independent replay, persistence and the step C
  regression run; one-time rerun of a failed unit (`archive-failed`); step-B
  probes and a write-once execution setting (AMP, lanes, reuse of the matching
  full run as the step C cell); v3 assessor with the approved statistics.
- Fix found in step A: each residue's ESMC row was a view of the whole float32
  chain matrix, so every loaded structure kept all its chain embeddings in RAM
  (a v2 late-fusion fit peaked at 8.0 GB; one fold-0 replay above 3 GB). Rows
  are now owned copies (one fold-0 replay peaks near 1.9 GB); values are
  unchanged.
- A3 training check **passed**: on a synthetic campaign, 3-epoch CPU fits of
  four configurations with the new code and with the frozen
  `_code/pmm_core_scope_v2` checkout are bit-identical (every history value
  and every weight; difference 0.0), after the fix
  (`audits/a3_regression_20261005T101042Z`). The real-data replay of the nine
  pinned v2 fold-0 checkpoints (best checkpoints against the saved predictions
  under `pmm-core-replay-v1`; epoch-50 checkpoints against the epoch-50
  history) is running on CPU; results go in the step A completion entry.
- A4: the user approved the four new default rules on 2026-10-05 with
  clarifications (eligibility is not an improvement claim; improved-recipe
  five-fold results are development-validation; the five-class tie preference
  is a selection convention). The specification is frozen only at step A
  completion. The approval covers the assessment rules, not a GPU start.
- A5 pre-fit gate **passed** on a scratch preparation
  (`audits/a5_prefit_20261005T103224Z`): all 82 planned units build valid
  commands; every native class is present in every fold (validation Co 55–58,
  Cu 67–73); class weights equalize the four common classes; the trainer applies
  them to five- and six-class targets (tests); normalization is fitted on the
  training fold only (test). Augmented recipes would rebuild graphs for about
  7.6–9.5 h per fit on this PC, so they fall under the cost gate.
- Deferred by the user's decision: the label-blind clean-subset and
  test-preparation builder is written just before step F, outside `src/`, and
  tested on synthetic or training-side data before any test access, keeping the
  agreed near-copy and previously-opened-PDB exclusion rules. The final-test
  reporting decision stays due before step E.
- A6: the executable v3 recipe is in the
  [metal playbook](../../METAL_TRAINING_PIPELINE_PLAYBOOK.md#pmm-ion-metal-v3-campaign-pmm_ion_metal_v3).
  Free space on `/` is about 6 GB; the user decides what to remove before step B.

## v3-005

2026-10-04 — plan step A1 complete: the v3 fold set `v3-seqid90-s42-b2` is
frozen under `/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v3/folds/`
(`fold_membership.csv` SHA-256 `e6b21364159da479dc3fb325a01ede36c62489184a77d858f71fefe6c401da7a`;
receipt `fold_receipt.json` with input, output and code identities; builder
`v3-near-copy-folds-2` at commit `3102c2a`, seed 42, 200 starts). Only training
inputs were read, under the read guard; every one of the 7,664 context-chain
sequences matched its frozen ESMC hash.

Result: 7,398 ions, 3,992 PDB entries, 4,972 unique chains, 11,609 near-copy
pairs, 2,208 groups. All acceptance checks pass (48 of 200 starts; start 198):
fold ions 1,471–1,500 (target 1,479.6); Cu 67–73 ions from 24–25 groups per
fold; at least 24 groups of every element in every fold. No near-copy under
the builder's rule crosses folds (the v2 folds had 50.9% of ions with a
cross-fold near-copy).

Refinement of the A1 near-copy rule: identity divides by the shorter chain's
length plus its number of internal insertion events, so the 5-mer prefilter is
provably lossless. A first build (`v3-seqid90-s42`, builder version 1) counted
every inserted residue instead; independent verification showed it split 25
plan-rule near-copy pairs (54 ions) across folds, so it is marked superseded and
must not be used. Under the plan's original wording, 9 borderline pairs
(identity 90.0–90.6% of the shorter chain) still cross the frozen folds.

Verification (read-only, independent methods): exhaustive screening of all
11.9 million chain pairs with a different lossless filter found no qualifying
pair the builder missed; 50,000+ skipped pairs aligned exactly stayed below 90%;
balance recomputed exactly; byte-identical rebuilds under different Python hash
seeds. Known limits: chains below 90% identity can cross folds (about 11% of
ions have an 80–90% cross-fold partner), so this is near-copy, not homology,
separation; and a few large families dominate one element in one fold (largest
single-group shares: Ni 0.51 and Cu 0.43 in fold 4, Mn 0.40 in fold 1).

## v3-004

2026-10-04 — an external review of the v3 plan (Codex, read-only) raised twelve
points; each was checked against the repository by five read-only verifiers
(no test files, no compute). All were accepted. Two were accepted only in part:
policy does not require two confirmation seeds, and the step-D parameter values
are a user choice rather than a wording fix. Minor inaccuracies (misplaced line
references, a non-binding source cited as binding) did not change any verdict.
The verifiers also found that the trainer cannot yet consume group-based v3
folds, that a locally built graph cache does not match the VM's torch build,
that augmented fits rebuild every training graph each epoch, and that no RING
files exist for the v3 cohort. The [plan](plan.md) was revised accordingly and
then checked once more for consistency; that check moved the A4 freeze before
any GPU run, made every budget stop a user choice, and moved Round C out of the
plan's code and budget pending a separate decision after Round B.

User decisions:

- Step E: the neutral four-versus-five/six test uses the step C baseline recipe,
  frozen before step D, on all five folds; improvements are confirmed separately
  in `four_class` on folds 1–4 against that baseline.
- Decision statistics: bootstrap and t intervals must agree; a target tie keeps
  `four_class`; a zero native recall blocks a five/six "better" call.
- Step D gate: both seeds must improve, in addition to the 1.5-point mean and the
  3-point class guard.
- The final-test label is decided before step E, after the user checks the
  2026-09-24 Zenodo run.

Defaults set by Claude under the user's instruction to accept or reject the
review's points (changeable by a dated entry before the affected step): seed 42
only in step E; step-D values (modality dropout 0.2, outer-residue dropout 0.1,
coordinate noise 0.1 Å, label smoothing 0.1, `counts_angles` against `none`);
the step-B adoption thresholds; the 3.0-point regression band; the fold
acceptance limits; the cost gate for augmentation; and the Stage 6 selection
rule (Only-ESMC baseline `four_class` control, 0.2-point tie band,
tie-breakers), which is shown to the user with the A4 specification before it
is frozen.

STATUS text replaced by this update, preserved verbatim:

```text
- Stage: v3 design; no folds, runs, Stage 6 confirmation, Stage 6B refit or Stage 7.
Next: user decisions on the v3 design listed in its README, starting with the
final metal test route. Stage 6 grouped-fold selection (or an explicitly labeled
fallback), a completed Stage 6B full non-test refit, frozen report/checkpoint
rules and a scientifically resolved final-test route must precede one-shot
Stage 7. No test-based tuning, ranking, promotion, rejection or checkpoint
choice; see [Plan](Plan.md#canonical-staged-metal-training-pipeline).
```

## v3-003

2026-10-04 — the user confirmed active Google Cloud free-trial credits on the
billing account (upgraded free trial; about ₪848 remaining, expiring
2026-12-24; Google reports they cover all eligible usage). Under the user's
rule from [v3-002](#v3-002), the v3 GPU budget ceiling is **$40 gross** for
steps B–F. Costs stay tracked at gross list price; credits never widen the
controller's session and daily caps or any limit, and usage is not described
as free. Every GPU start still needs the user's explicit OK.

STATUS line replaced by this update, preserved verbatim:

```text
- Authorized now: v3 CPU preparation only ([plan](docs/campaigns/pmm_ion_metal_v3/plan.md)); every GPU start needs the user's explicit OK within the [recorded ceiling](docs/campaigns/pmm_ion_metal_v3/log.md#v3-002); no refit or held-out evaluation.
```

## v3-002

2026-10-04 — user decisions after discussion:

- The [v3 plan](plan.md) (steps A–F) is approved.
- The 2026-09-28 rule "no file added or changed in `src/` or `scripts/`" is
  lifted for v3 work: changes use off-by-default options with tests. The frozen
  `_code/pmm_core_scope_v2` checkout stays untouched.
- Checkpoint rule: cosine learning-rate schedule to zero and the terminal
  checkpoint for every v3 arm, fold and refit (Plan updated the same day).
- PMM: compare with published PinMyMetal numbers only; no PMM retraining.
- Folds: chains at ≥90% identity grouped together, random group order, metals
  balanced. Improvement screening on one new fold with two seeds is accepted.
- GPU budget: $15 gross until the user confirms that free credits cover Compute
  Engine, then $40. No GPU start is authorized without the user's explicit OK.

STATUS lines replaced by this update, preserved verbatim:

```text
- Status: planned (2026-10-03 PMM core v2 closed at fold 0; v3 planning only)
- Last execution evidence: 2026-09-28. Documentation reconciliation: 2026-10-03 (closure and verified findings).
- Authorized now: nothing (no experiment, GPU work, final refit or held-out evaluation).
```

## v3-001

2026-10-03 — campaign opened in planning status. The user chose separate
`four_class`, `five_class` and `six_class` arms for Only-ESMC, Only-GVP and
graph-level late fusion. Predecessor:
[closed PMM campaign](../../archive/campaigns/pmm_ion_metal/README.md).
