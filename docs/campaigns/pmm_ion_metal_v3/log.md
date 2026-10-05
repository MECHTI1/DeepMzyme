# PMM ion metal v3 — decision log

Dated user decisions and STATUS history for this campaign, newest first.
Current authority: [EXPERIMENT_STATUS.md](../../../EXPERIMENT_STATUS.md).

## v3-009

2026-10-06 — **step B preparation completed on CPU** (nothing ran on a GPU; no
VM exists). After the step-B readiness audit, the user decided and asked for one
preparation pass (these decisions change step-B defaults before the first
step-B run, as plan.md allows):

1. Replay: every v3 fit, probe and the regression run is independently replayed
   under the explicit policy `pmm-v3-replay-1`: probabilities within 1e-5 (the
   pmm-core-replay-v1 tolerance), compared exactly on the saved 8-decimal values;
   identities, labels, native and common-four predicted classes and confusion
   matrices stay exact, and the balanced-accuracy reconciliation stays within
   1e-9 (only the tolerance comes from pmm-core-replay-v1, not its other row
   checks). It is an option of
   `campaign_runtime.replay_campaign_run` (default unchanged: 1e-6, same messages,
   same receipt), recorded in the campaign manifest, every replay receipt (with
   the largest observed difference) and every unit status. Evidence: A3 on
   2026-10-05 found 6 of 9 CPU replays of GPU-trained v2 checkpoints above 1e-6
   (worst 4.53e-6) with every discrete field equal.
2. Concurrency: fixed limits of 1.0 common-four BA point and 3 points per
   common-four class recall against the first serial run; never widened. If the
   two serial runs disagree beyond them, the report requires diagnosis and
   decides nothing; the only pre-declared closure (`step_b/diagnosis.json`,
   written after a dated user decision) records serial FP32 without concurrency
   or AMP. Probability and history differences are diagnostics.
3. CPU pressure: median, peak, longest and total overload (load1 per CPU above
   1.0) are reported; pressure if the median exceeds 1.0, one overload lasts
   longer than 300 s, or the steady training phase (all members training, first
   60 s skipped for the load1 lag, at least 60 s judged) is overloaded more than
   half the time; memory (at least 10% available) and GPU memory (at most 90%;
   missing GPU samples fail) limits unchanged. Throughput is end to end: pre-admission checks (recorded per
   unit), preparation, training, replay and measured host persistence; batch
   members are charged their own (contended) preparation.
4. Retries: completed fits are never rerun. Only a step-B timing batch (2 or 3
   lanes) may be retried, once, as a whole under `-retry` tags, and only when
   its first attempt is invalid under outcome-blind rules (a member failed or did
   not run, members not in distinct lanes, admissions more than 10% of the
   shortest elapsed time apart, or host samples not covering the batch window);
   the retry then decides. A retry after a valid first attempt makes the report
   refuse. All attempts are kept.
5. Costs: storage counts toward the $40 gross ceiling, at estimated actual
   charges (disk $15.00/month = $0.49/day while the restored disk exists;
   snapshot about $1.74/month = $0.06/day, from the retirement receipt), apart
   from the controller's conservative daily reservation (snapshot $7.50/month,
   $0.25/day). Running hours are costed at $0.859/h (the gross $0.879/h minus its
   disk share), so the disk is counted once. The snapshot is kept.
6. Disk: the two archives were moved to `/media/mechti/Data1/archive_from_home/`
   (SHA-256 equal on both sides, copies re-read with direct I/O, originals then
   removed; `SHA256SUMS` beside them). `/` has 16 GB free. Nothing else was
   deleted.

Also built (all tested): the fixed probe manifest `pmm_v3_probes.py` (short
probes 10 epochs, default, so a steady training phase can be judged), enforced
by the runner before admission (an archived batch member is never relaunched;
a failed serial probe or full run keeps its one rerun before the decision); the runner
hash now covers `pmm_v3_probes.py` and `pmm_v3_speed_report.py`, so the gates
are frozen at preparation, before any step-B data; `set-execution` records a
setting only if it equals a recomputed, decision-ready speed report; the
workstation launcher `pmm_v3_step_b.py` (never starts, stops or extends a VM;
detached launches; waits; pulls) and the per-lane host-pull tool
`pmm_v3_host_pull.py` (never re-stamps an acknowledgment; keeps a superseded
local copy when a rerun reuses a run name; no torch); the launcher refuses a
step while a lane awaits its host pull, holds a running or interrupted unit, or
an earlier launch has no exit code, and a failed pull fails the command;
evidence backup before every `vm-stop`; `pmm_v3_bundle.py build` refuses
without an A3 report that accepted exactly the bundled source tree; runner exit
codes 3 (blocked) and 4 (persistence).

An independent five-agent review (read-only) found one blocker (a batch could
start while a lane still awaited its host pull, wasting the single retry) and
six major points (A3 binding after the src change, failed probes losing their
rerun, no closure for a serial disagreement, archived batch members allowing
extra attempts, reruns overwriting the workstation copy, a weak sustained-CPU
check in short probes); all were fixed with tests, as were the cheap minor
points. Deferred as minor: a guard against deliberately launching one batch
member alone outside the launcher (the launcher always starts members
together).

Commits: `64f29de` (src replay option), `76395bf` (runner, gates, manifest,
tools, tests, plan and playbook) and this entry's commit. Checks:

- A3 repeated and **accepted** (`audits/a3_regression_20261005T194214Z`,
  `a3_report.json` SHA-256 `50ecf905…`): tested source tree
  `212db0e3865968701475e7172b2e4cad772e492b35855da293238437fd97c15c` (src at
  `64f29de`, unchanged in `76395bf`), unchanged from start to end, no uncommitted
  src change; the frozen v2 tree matches its pin. Replay: all nine pinned v2
  fold-0 checkpoints pass pmm-core-replay-v1 (worst probability difference
  4.53e-6; 6 of 9 above the strict 1e-6, every discrete field equal) and every
  epoch-50 checkpoint matches its history; training: all four configurations
  bit-identical to the frozen v2 checkout (difference 0.0). The default strict
  replay path is therefore unchanged on real data. **Correction to v3-007 and
  v3-008:** their A3 acceptance covers source tree `5a120b2c…` only and is
  superseded by this one; `pmm_v3_bundle.py build` now refuses any bundle whose
  source tree differs from an accepted A3 report.
- A5 repeated and **passed** (`audits/a5_prefit_20261005T204150Z`): 82 of 82
  planned units build valid commands on a scratch root whose manifest records the
  committed runner files (`pmm_v3_campaign.py` `ce6ddec8…`,
  `run_pmm_v3_campaign.py` `21b798ef…`, `pmm_v3_probes.py` `6c6a0d73…`,
  `pmm_v3_speed_report.py` `6e4c2997…`) and the replay policy; augmented recipes
  still need about 8.8 h (late fusion) to 12.4 h (Only-GVP) of graph rebuilding
  per fit on this PC, so they stay cost-gated.
- Tests: all 186 v3 tests pass. Full suite on `76395bf`: 1,167 passed, 19
  skipped, 2 failed, 32 errors, all known and unrelated to this pass. The 2
  failures fail identically on `f3510ba`:
  `test_explicit_membership.py::test_outer_loader_not_deserialized` (patches a
  `training.data.load_structure_pockets` that does not exist) and
  `test_generalized_metal_5fold_cv.py::test_dry_run_flag` (expects the local
  dataset folder `train_and_test_sets_structures_exact_pinmymetal`). The 32
  errors are every setup of `test_pmm_core_assessment.py` ("Scientific module
  already loaded from another tree: data_structures"), an order dependence: the
  same 40 tests pass when that file runs alone.

Step-B forecast (gross): one session of about 2.4–3.2 VM hours with 10-epoch
probes, $2.06–2.75 of compute and IP (about 5.5 h and $4.72 if the AMP
follow-ups, a retry or connection trouble need a second session), plus the
restored disk at $0.49/day and the snapshot at $0.06/day. Whole campaign B–F,
from the measured v2 fit times (late fusion 1,747 s, Only-GVP 1,619 s, Only-ESMC
1,125 s per warm fit including replay, persistence and pre-admission checks; five
further cold cache builds; about 25 minutes of overhead per session):

| Scenario | VM hours | Compute + IP | Disk | Snapshot (actual) | Total |
|---|---|---|---|---|---|
| 1.5× concurrency, D ends after Round A | 26 | $22.6 | $4.9 (10 days) | $0.6 | $28 |
| 1.5× concurrency, full D | 35 | $30.3 | $6.9 (14 days) | $0.8 | $38 |
| no concurrency, D ends after Round A | 37 | $32.0 | $6.9 (14 days) | $0.8 | $40 |
| no concurrency, full D | 51 | $43.6 | $10.4 (21 days) | $1.2 | $55 |

Augmented candidates (9a, 9b) stay cost-gated and are excluded. Calendar days
are an assumption; every extra day with the disk kept adds $0.49. Without a
concurrency gain the full plan exceeds the ceiling, so the re-forecast after
step B decides whether the user is asked to stop early or raise the ceiling.

Unchanged: the frozen A4 specification (SHA-256 `29694070…`), the fold set and
its caveat (near-copy, not homology, separation; about 11% of ions have an
80–90% cross-fold partner), and the open final-test label decision before
step E. No held-out data were read.

STATUS text replaced by this update, preserved verbatim:

```text
- Status: active (2026-10-05 v3 step A complete, CPU only; step B awaits the user's GPU OK)
- Last execution evidence: 2026-10-05 (v3 step A CPU audits). Documentation reconciliation: 2026-10-05.
- Current campaign: pmm_ion_metal_v3, step A complete (objectives four/five/six for Only-ESMC, Only-GVP and late fusion) — [README](docs/campaigns/pmm_ion_metal_v3/README.md)
- Stage: v3 step A done ([log v3-008](docs/campaigns/pmm_ion_metal_v3/log.md#v3-008)): strict folds, code, CPU audits, frozen assessment rules; no GPU runs, Stage 6 confirmation, Stage 6B refit or Stage 7.
- Authorized now: CPU work only ([plan](docs/campaigns/pmm_ion_metal_v3/plan.md)); every GPU start needs the user's explicit OK within the [$40 gross ceiling](docs/campaigns/pmm_ion_metal_v3/log.md#v3-003); no refit or held-out evaluation.
Next: v3 step B (GPU speed check) per its [plan](docs/campaigns/pmm_ion_metal_v3/plan.md), after the user's explicit GPU OK and freeing 10–15 GB on `/`.
```

## v3-008

2026-10-05 — **plan step A complete** (CPU only; nothing ran on a GPU). After
the v3-007 checkpoint the user said to continue, and the A4 specification was
frozen:

- [`assessment_spec.md`](assessment_spec.md) SHA-256
  `899005aaba036739519876860121fdf4e60241d55c393d435847aae56ece3667`;
  [`assessment_spec.json`](assessment_spec.json) SHA-256
  `2969407042dbb5858f15e1a114a2d602cc2fd764e788022ddaee1907ff17306c`, pinned as
  `FROZEN_SPEC_SHA256` in `pmm_v3_assessment.py`. Its fold, cohort and baseline
  identities match a real-data preparation of this campaign (the A5 scratch
  root). Every run and assessment now requires it.

Step A record: A1 strict folds (v3-005); A2 code (v3-006, v3-007), with the
clean-subset and test-preparation builder **deferred** by the user to just
before step F (outside `src/`, tested on synthetic or training-side data before
any test access, keeping the agreed near-copy and previously-opened-PDB rules);
A3 accepted (`audits/a3_acceptance_20261005T122647Z`, tested source tree
`5a120b2c…`); A4 frozen (above); A5 passed
(`audits/a5_prefit_20261005T122704Z`); A6 playbook recipe written. All 77 v3
tests pass; the full suite shows only the known 2 failures and 32
order-dependent errors.

Before step B: the user's explicit GPU OK (within the $40 gross ceiling),
about 10–15 GB freed on `/` (about 6 GB free), and the real campaign root
prepared once from the final code. Still due before step E: the final-test
reporting decision.

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
