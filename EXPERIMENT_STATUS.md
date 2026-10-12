# DeepMzyme Current Experiment Status

- Status: active (2026-10-12 v3: step F plan approved; the Stage 6B refit authorized, one GPU session)
- Last execution evidence: 2026-10-12 (v3 10-epoch comparison, 1 session, 37 min, $0.55 gross; [log v3-041](docs/campaigns/pmm_ion_metal_v3/log.md#v3-041)).

## Current objective and stage

- Current campaign: pmm_ion_metal_v3, steps A–E2 complete (objectives four/five/six for Only-ESMC, Only-GVP and late fusion) — [README](docs/campaigns/pmm_ion_metal_v3/README.md)
- Other open campaigns: [metal_single_gpu](docs/campaigns/metal_single_gpu/README.md) (paused); [zenodo_pmm_exact_2026-09-24](docs/campaigns/zenodo_pmm_exact_2026-09-24/README.md) (paused; test possibly opened)
- Stage: v3 steps A–E2 and the pre-step-F comparison done: Stage 6 selection `gvp_late_fusion__four_class__baseline`, 50 epochs (A4, retained by E2; 10 epochs not adopted); step F plan approved ([log v3-045](docs/campaigns/pmm_ion_metal_v3/log.md#v3-045)), refit in progress; no Stage 7.

**PMM core v2 was closed at fold 0 on 2026-10-03 (user decision):** nine fold-0 fits
are final validation-only evidence; 36 fits were not run; its neutral four-versus-five/six
test is unanswered; nothing is confirmed, promoted, refit or test-evaluated
([archived README](docs/archive/campaigns/pmm_ion_metal/README.md)).

## Anchor and evidence state

- Best validation result: none promoted; the closed PMM core fold-0 fits are Grade 5 exploratory evidence (incomplete grid Grade 6) per the archived README above; results in the [core summary](docs/notebook_outputs/summaries/summary_pmm_core_continuation_20260928.md); its fold 0 is not representative ([findings](docs/PARAMETER_FINDINGS.md#pmm-ion-campaign-single-fold-target-and-readout-comparisons)).
- v3 steps E/E2: no family gains from five/six-class training (tie keeps `four_class`); late fusion four-class 74.28 mean common-four BA, "better" than Only-ESMC on all five folds; E2 (`E2_20261012T053919Z`): no candidate eligible to replace it; Only-GVP `wd10` is an interval-supported Only-GVP improvement (+3.59).
- EC: the [EC1 reference](docs/archive/campaigns/ec1_standalone_v12_2026-09-14/README.md) has twelve fixed-split runs (Grade 3), not promotion; cross-task holdout certification precedes auxiliary learning ([issues](docs/FOLLOW_UP_TECHNICAL_ISSUES.md)).

### Known caveats and open mismatches

- Cross-scheme ranking needs a campaign assessor on collapsed-four BA; the notebook Stage 6/6B route is single-scheme ([TECH-010](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-010--four-class-endpoint-and-paired-metal-target-recipes-are-not-reconciled)).
- v3 replaces the size-strata folds ([TECH-020](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-020--greedy-k-fold-assignment-concentrates-large-groups-in-fold-0)) and uses one terminal checkpoint rule in CV and refit ([TECH-027](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-027--cross-validation-and-final-refit-use-different-checkpoint-rules)); PMM comparator citations keep the label-leak caveat ([TECH-028](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-028--pmm-comparator-inherits-a-true-metal-label-leak)).

## Dataset and test readiness

The primary final-test route is unresolved; [DATASETS](docs/DATASETS.md#test-use-ledger) owns the ledger. Safety boundary:

- Non-overlap and exact PMM tests (same 316-structure set): opened 13 and 21 times; not eligible as an unopened test. Harsh and Common-PDBID 70/30 test IDs come from that opened set.
- Exact Zenodo PMM test: unknown, possibly evaluated; test-file label/feature metadata read in aggregate on 2026-10-03, test-directory bytes scanned without display on 2026-10-11 (no evaluation).
- No evaluation artifacts found for CLEAN30 or CARE clusterRes30; CARE had incidental test-metadata exposure on 2026-09-15. Absence of found artifacts is not proof of no outside run.

## Blockers and immediate next action

- Authorized now: step F F1–F2 only (binding records and one refit GPU session, `--hours 2`, one unchanged rerun if it fits; [log v3-045](docs/campaigns/pmm_ion_metal_v3/log.md#v3-045)); the $55 gross ceiling (about $40.81 spent at 2026-10-12 18:31 UTC); no Round C.
- GPU/VM: `deepmzyme-l4` TERMINATED (verified 2026-10-12 16:16 UTC); disk and archive snapshot kept (about $0.55/day).

Next: the refit, its receipt and the Freeze 1 approval package; Freeze 1, Freeze 2 and the single test pass stay separately gated ([step F plan F7](docs/campaigns/pmm_ion_metal_v3/step_f_plan_draft.md#f7-decisions-for-the-user-none-inferred)); open: fairness arm, cost-gated augmentations.
Final-test label `both_results_secondary` is recorded ([log v3-022](docs/campaigns/pmm_ion_metal_v3/log.md#v3-022)). Stage 6 selection, a completed Stage 6B refit and frozen report/checkpoint rules precede one-shot Stage 7; no test-based selection ([Plan](Plan.md#canonical-staged-metal-training-pipeline)).

## History and update rule

Overwrite this file after preserving dated changes in the campaign's `log.md`; keep the objective/stage and campaign lines, anchor, best result and grade, caveats, dataset/test readiness, authorization and GPU/VM state, blockers, next action and evidence links. Budgets stay in the playbooks, findings in [PARAMETER_FINDINGS](docs/PARAMETER_FINDINGS.md), policy in [Plan](Plan.md), batch evidence in the [index](docs/notebook_outputs/README.md); the [history map](docs/archive/consolidation_2026-09/MAP.md) holds the recovered 09-16/17 pause and removed policy.
