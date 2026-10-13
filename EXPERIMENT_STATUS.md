# DeepMzyme Current Experiment Status

- Status: active (2026-10-13 trunk-LR comparison opened by the user; CPU preparation complete; GPU start not authorized)
- Last execution evidence: 2026-10-13 (v3 step F test pass on the workstation CPU, [log v3-048](docs/archive/campaigns/pmm_ion_metal_v3/log.md#v3-048); trunk-LR readiness checks, CPU only, [log tlr-001](docs/campaigns/pmm_trunk_lr_2026-10/log.md#tlr-001)).

## Current objective and stage

- Current campaign: [pmm_trunk_lr_2026-10](docs/campaigns/pmm_trunk_lr_2026-10/README.md): v3-selected four-class late fusion with GVP-trunk learning rate 1e-4 instead of 3e-4, five fits against the five v3 controls; objective `four_class` (user, 2026-10-13).
- Other open campaigns: [metal_single_gpu](docs/campaigns/metal_single_gpu/README.md) (paused); [zenodo_pmm_exact_2026-09-24](docs/campaigns/zenodo_pmm_exact_2026-09-24/README.md) (paused; test possibly opened)
- Stage: CPU preparation complete; rule frozen in the [specification](docs/campaigns/pmm_trunk_lr_2026-10/trunklr_assessment_spec.json); no candidate fit yet. Unchanged v3 result: selection `gvp_late_fusion__four_class__baseline`, refit `a0007846…`, one-shot test common-four BA 67.55% [62.93, 71.65], `both_results_secondary` ([summary](docs/notebook_outputs/summaries/summary_pmm_ion_metal_v3_20261013.md)).

PMM core v2 closed at fold 0 on 2026-10-03 (user decision): nine fold-0 fits only; nothing confirmed, promoted, refit or test-evaluated ([archived README](docs/archive/campaigns/pmm_ion_metal/README.md)).

## Anchor and evidence state

- Best validation result: none promoted. The closed v3 campaign's validation line (Grade 2; five grouped folds, seed 42) is in [findings](docs/PARAMETER_FINDINGS.md#pmm-ion-metal-v3-objective-test-and-stage-6-selection-closed-2026-10-13): late fusion four-class, mean common-four BA 74.28%, better than Only-ESMC on all five folds; no supported gain from five/six-class training. The PMM core fold-0 fits stay Grade 5 exploratory evidence ([core summary](docs/notebook_outputs/summaries/summary_pmm_core_continuation_20260928.md)).
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

- Authorized now: CPU preparation only; no GPU start, fit, embeddings or test predictions. Pending the user's approval: one L4 session, $2.50 compute ceiling, 2.5-hour hard stop.
- GPU/VM: `deepmzyme-l4` TERMINATED (live check 2026-10-13 15:53 UTC); disk and caches kept for this comparison; snapshots `deepmzyme-paused-20261003` and `deepmzyme-v3-final-20261013` kept ([handoff section 13](docs/archive/campaigns/pmm_ion_metal_v3/handoff.md#13-closeout-proposal-2026-10-13-pending-the-users-approval)).

Next: the user's GPU-start authorization and budget, then one session, the assessment and one comparison table and decision. Later: stronger EC standalone validation, then cross-task exclusion certification. No test-based selection ([Plan](Plan.md#canonical-staged-metal-training-pipeline)).

## History and update rule

Overwrite this file after preserving dated changes in the campaign's `log.md`; keep the objective/stage and campaign lines, anchor, best result and grade, caveats, dataset/test readiness, authorization and GPU/VM state, blockers, next action and evidence links. Budgets stay in the playbooks, findings in [PARAMETER_FINDINGS](docs/PARAMETER_FINDINGS.md), policy in [Plan](Plan.md), batch evidence in the [index](docs/notebook_outputs/README.md); the [history map](docs/archive/consolidation_2026-09/MAP.md) holds the recovered 09-16/17 pause and removed policy.
