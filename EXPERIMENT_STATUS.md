# DeepMzyme Current Experiment Status

- Status: active (2026-10-13 v3: all steps complete; closeout recorded and backed up, log v3-050; closure is the user's decision)
- Last execution evidence: 2026-10-13 (v3 step F test pass and report on the workstation CPU; [log v3-048](docs/campaigns/pmm_ion_metal_v3/log.md#v3-048)).

## Current objective and stage

- Current campaign: pmm_ion_metal_v3, steps A–F complete (objectives four/five/six for Only-ESMC, Only-GVP and late fusion) — [README](docs/campaigns/pmm_ion_metal_v3/README.md)
- Other open campaigns: [metal_single_gpu](docs/campaigns/metal_single_gpu/README.md) (paused); [zenodo_pmm_exact_2026-09-24](docs/campaigns/zenodo_pmm_exact_2026-09-24/README.md) (paused; test possibly opened)
- Stage: v3 Stage 6 selection `gvp_late_fusion__four_class__baseline` (50 epochs), Stage 6B refit `a0007846…`, one-shot test done: full-test common-four BA 67.55% [62.93, 71.65], label both_results_secondary ([log v3-048](docs/campaigns/pmm_ion_metal_v3/log.md#v3-048)).

**PMM core v2 was closed at fold 0 on 2026-10-03 (user decision):** nine fold-0 fits
are final validation-only evidence; 36 fits were not run; its neutral four-versus-five/six
test is unanswered; nothing is confirmed, promoted, refit or test-evaluated
([archived README](docs/archive/campaigns/pmm_ion_metal/README.md)).

## Anchor and evidence state

- Best validation result: none promoted; the closed PMM core fold-0 fits are Grade 5 exploratory evidence (incomplete grid Grade 6) per the archived README above; results in the [core summary](docs/notebook_outputs/summaries/summary_pmm_core_continuation_20260928.md); its fold 0 is not representative ([findings](docs/PARAMETER_FINDINGS.md#pmm-ion-campaign-single-fold-target-and-readout-comparisons)).
- v3 steps E/E2: no supported gain from five/six-class training for any family (tie keeps `four_class`); late fusion four-class 74.28 mean common-four BA, "better" than Only-ESMC on all five folds; E2 (`E2_20261012T053919Z`): no candidate eligible to replace it; Only-GVP `wd10` is an interval-supported Only-GVP improvement (+3.59).
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

- Authorized now: nothing to run (step F complete); the $55 gross ceiling (about $41.8 spent); no Round C.
- GPU/VM: `deepmzyme-l4` TERMINATED (verified 2026-10-13 02:15 UTC); disk and archive snapshot kept (about $0.55/day).

Next: the user's closure decision (a compatibility solution for the archive move) and storage retirement A/B/C ([handoff section 13](docs/campaigns/pmm_ion_metal_v3/handoff.md#13-closeout-proposal-2026-10-13-pending-the-users-approval)); open: fairness arm, cost-gated augmentations.
Final-test label `both_results_secondary` is recorded ([log v3-022](docs/campaigns/pmm_ion_metal_v3/log.md#v3-022)). Stage 6 selection, a completed Stage 6B refit and frozen report/checkpoint rules precede one-shot Stage 7; no test-based selection ([Plan](Plan.md#canonical-staged-metal-training-pipeline)).

## History and update rule

Overwrite this file after preserving dated changes in the campaign's `log.md`; keep the objective/stage and campaign lines, anchor, best result and grade, caveats, dataset/test readiness, authorization and GPU/VM state, blockers, next action and evidence links. Budgets stay in the playbooks, findings in [PARAMETER_FINDINGS](docs/PARAMETER_FINDINGS.md), policy in [Plan](Plan.md), batch evidence in the [index](docs/notebook_outputs/README.md); the [history map](docs/archive/consolidation_2026-09/MAP.md) holds the recovered 09-16/17 pause and removed policy.
