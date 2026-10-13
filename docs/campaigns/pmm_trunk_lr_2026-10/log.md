# PMM trunk-learning-rate comparison: log

Dated entries, newest last. STATUS is overwritten; its previous text is preserved here first.

## tlr-001

2026-10-13 — **campaign opened by the user's decision; CPU preparation complete; no GPU start authorized.**

User decisions (2026-10-13):
- Training target: `four_class` only (collapsed-four reporting), the v3-selected late-fusion architecture.
- Single change: GVP-trunk learning rate 3e-4 → 1e-4. Base learning rate 3e-5, optimizer groups, 50-epoch cosine
  schedule, terminal checkpoint, data, features and seed 42 unchanged. Five candidate fits, one per v3 grouped fold,
  against the five verified v3 controls; separate campaign identity; v3 evidence and final model preserved.
- Rule, frozen here before any fit ([specification](trunklr_assessment_spec.json), SHA-256 `61728a8f…`):
  - **T1, folds 0–4 (primary).** Promising candidate only if all of these hold:
    - (a) complete and verified, including the optimizer groups;
    - (b) both the fold-bootstrap and the t (4 df) lower 95% bounds are above 0;
    - (c) the mean gain is above 0.2 points;
    - (d) no class mean recall drops by more than 3 points, and no fold has a missing or zero class recall;
    - (e) the folds 1–4 mean gain is positive.
  - **Folds 1–4:** reported as sensitivity.
  - **Failing:** the reference (74.28%) is retained. A failure is not evidence that the trunk learning rate cannot
    help.
- Budget (proposed, pending authorization): one L4 session, compute ceiling $2.50 (running charges), 2.5-hour hard stop.
- The structural-context investigation of recovered PMM sites is closed at its evidence boundary. Its closure note is
  on Data1 under `post_v3_diagnostics/`; no model input changed.

Tooling (additive; no v3 file changed):
- `run_pmm_trunklr.py` (VM runner). It swaps the frozen builder's single `--gvp-learning-rate 3e-4` pair for `1e-4`,
  recomputes the identity hash with the builder's own functions, and refuses any other difference from the control.
- `pmm_trunklr.py` (launcher; standard library). It keeps the 10-epoch comparison's gate chain and its records,
  evidence copies and assessments go to Data1 `campaigns/pmm_trunk_lr_2026-10/`. It adds a compute-only ceiling.
- `pmm_trunklr_assessment.py` (assessor).
- `tests/test_trunklr.py`.

Readiness evidence (CPU only; no held-out access):
- **Data1 and the VM.** Data1 is remounted read-write after the reboot (not in `/etc/fstab`). Live `vm-status` works
  again: `deepmzyme-l4` was TERMINATED at 15:53 UTC.
- **Records.** The v3 readiness report (`d_readiness_20261006T180939Z`, ready) and the extension record match this
  worktree's runner files. The in-memory manifest verifies. It binds the five controls by checkpoint and prediction
  hash; A4, E2 and the E2 manifest agree. Fold values: 71.62, 73.34, 72.96, 75.17 and 78.32 (mean 74.28).
- **Optimizer groups.** All five control checkpoints have a trunk group of 84 tensors at initial LR 3e-4 and a base
  group of 31 at 3e-5, with weight decay 1e-4.
- **Assessor dry run.** It ran on the 10-epoch comparison's evidence copy (`evidence_20261012T111709Z`):
  - the controls are verified, including durable receipts and replay;
  - five-fold parity with A4 is exact;
  - on the real frozen campaign files, each fold differs only in `gvp_learning_rate` (3e-4 → 1e-4) and the run name,
    and the identity differs only in `resolved_config_sha256` and `lr_extension`;
  - status is "incomplete" because no candidate fit exists yet.
- **Tests.** `tests/test_trunklr.py`: 27 passed. The opt-in synthetic fit (`DEEPMZYME_SYNTHETIC_FIT_TESTS=1`) passed in
  85 s: 50 epochs, terminal checkpoint, independent replay, persistence, optimizer groups (84 at 1e-4, 31 at 3e-5).
  `tests/smoke_checks.py`: 43 passed.

Estimate (from v3 measurements; not a guarantee):
- **Rounds.** Two rounds (3 fits, then 2) at 31–37 minutes each. With start, checks, deploy and evidence copying, about
  1.3–1.5 VM hours, $1.1–1.3 at $0.858624/h.
- **With one rerun.** About 2.2 hours, $1.9.
- **Admission.** v3's rule requires 1.25 × 2,400 s + 900 s left at each launch, so a 2.5-hour hard stop admits both
  rounds and one rerun.
- **Storage.** The kept disk and snapshots bill separately, at about $19 per month.

STATUS text replaced by this entry, preserved verbatim:

```text
# DeepMzyme Current Experiment Status

- Status: closed (2026-10-13 v3 closed by the user, log v3-052; no active campaign)
- Last execution evidence: 2026-10-13 (v3 step F test pass and report on the workstation CPU; [log v3-048](docs/archive/campaigns/pmm_ion_metal_v3/log.md#v3-048)).

## Current objective and stage

- Current campaign: none active (v3 closed on 2026-10-13: [archived README](docs/archive/campaigns/pmm_ion_metal_v3/README.md)); a new campaign starts only by the user's decision.
- Other open campaigns: [metal_single_gpu](docs/campaigns/metal_single_gpu/README.md) (paused); [zenodo_pmm_exact_2026-09-24](docs/campaigns/zenodo_pmm_exact_2026-09-24/README.md) (paused; test possibly opened)
- Stage: v3 complete. Stage 6 selection `gvp_late_fusion__four_class__baseline` (50 epochs), Stage 6B refit `a0007846…`, one-shot test: full-test common-four BA 67.55% [62.93, 71.65], label both_results_secondary ([completion summary](docs/notebook_outputs/summaries/summary_pmm_ion_metal_v3_20261013.md)).

**PMM core v2 was closed at fold 0 on 2026-10-03 (user decision):** nine fold-0 fits
are final validation-only evidence; 36 fits were not run; its neutral four-versus-five/six
test is unanswered; nothing is confirmed, promoted, refit or test-evaluated
([archived README](docs/archive/campaigns/pmm_ion_metal/README.md)).

## Anchor and evidence state

- Best validation result: none promoted. The closed v3 campaign's validation line (Grade 2; five grouped folds, seed 42) is in [findings](docs/PARAMETER_FINDINGS.md#pmm-ion-metal-v3-objective-test-and-stage-6-selection-closed-2026-10-13): late fusion four-class, mean common-four BA 74.28%, better than Only-ESMC on all five folds; no supported gain from five/six-class training for any family. The closed PMM core fold-0 fits stay Grade 5 exploratory evidence ([core summary](docs/notebook_outputs/summaries/summary_pmm_core_continuation_20260928.md)).
- Deferred and never run (untested, not disproven): the v3 Only-ESMC fairness arm and the cost-gated augmentations `posnoise01` and `outerdrop01`.
- EC: the [EC1 reference](docs/archive/campaigns/ec1_standalone_v12_2026-09-14/README.md) has twelve fixed-split runs (Grade 3), not promotion; cross-task holdout certification precedes auxiliary learning ([issues](docs/FOLLOW_UP_TECHNICAL_ISSUES.md)).

### Known caveats and open mismatches

- Cross-scheme ranking needs a campaign assessor on collapsed-four BA; the notebook Stage 6/6B route is single-scheme ([TECH-010](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-010--four-class-endpoint-and-paired-metal-target-recipes-are-not-reconciled)).
- v3 replaces the size-strata folds ([TECH-020](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-020--greedy-k-fold-assignment-concentrates-large-groups-in-fold-0)) and uses one terminal checkpoint rule in CV and refit ([TECH-027](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-027--cross-validation-and-final-refit-use-different-checkpoint-rules)); PMM comparator citations keep the label-leak caveat ([TECH-028](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-028--pmm-comparator-inherits-a-true-metal-label-leak)).

## Dataset and test readiness

The primary final-test route is unresolved; [DATASETS](docs/DATASETS.md#test-use-ledger) owns the ledger. Safety boundary:

- Non-overlap and exact PMM tests (same 316-structure set): opened 13 and 21 times; not eligible as an unopened test. Harsh and Common-PDBID 70/30 test IDs come from that opened set.
- Exact Zenodo PMM test: evaluated once by v3 step F on 2026-10-13 (frozen rules, secondary report); earlier possible evaluation, 2026-10-03 metadata read and 2026-10-11 byte scan disclosed in the ledger.
- No evaluation artifacts found for CLEAN30 or CARE clusterRes30; CARE had incidental test-metadata exposure on 2026-09-15. Absence of found artifacts is not proof of no outside run.

## Blockers and immediate next action

- Authorized now: nothing to run (no active campaign); no GPU start, training, embeddings or test predictions.
- GPU/VM: `deepmzyme-l4` TERMINATED; disk kept (about $0.49/day); snapshots `deepmzyme-paused-20261003` and `deepmzyme-v3-final-20261013` kept ([handoff section 13](docs/archive/campaigns/pmm_ion_metal_v3/handoff.md#13-closeout-proposal-2026-10-13-pending-the-users-approval)).

Next: the user's decision on deleting the VM and its disk (inventory and typed confirmation in handoff section 13); any new campaign needs the user's decision.
Final-test label `both_results_secondary` was recorded before the test pass ([log v3-022](docs/archive/campaigns/pmm_ion_metal_v3/log.md#v3-022)). Stage 6 selection, a completed Stage 6B refit and frozen report/checkpoint rules precede one-shot Stage 7; no test-based selection ([Plan](Plan.md#canonical-staged-metal-training-pipeline)).

## History and update rule

Overwrite this file after preserving dated changes in the campaign's `log.md`; keep the objective/stage and campaign lines, anchor, best result and grade, caveats, dataset/test readiness, authorization and GPU/VM state, blockers, next action and evidence links. Budgets stay in the playbooks, findings in [PARAMETER_FINDINGS](docs/PARAMETER_FINDINGS.md), policy in [Plan](Plan.md), batch evidence in the [index](docs/notebook_outputs/README.md); the [history map](docs/archive/consolidation_2026-09/MAP.md) holds the recovered 09-16/17 pause and removed policy.
```
