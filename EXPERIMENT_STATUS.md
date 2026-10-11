# DeepMzyme Current Experiment Status

- Status: active (2026-10-11 v3 step E complete; review done; E2 built and checked; E2 authorized, running)
- Last execution evidence: 2026-10-11 (v3 step E sessions 8 and 9, 4 h 37 min, $4.06 gross).

## Current objective and stage

- Current campaign: pmm_ion_metal_v3, step E complete (objectives four/five/six for Only-ESMC, Only-GVP and late fusion) — [README](docs/campaigns/pmm_ion_metal_v3/README.md)
- Other open campaigns: [metal_single_gpu](docs/campaigns/metal_single_gpu/README.md) (paused); [zenodo_pmm_exact_2026-09-24](docs/campaigns/zenodo_pmm_exact_2026-09-24/README.md) (paused; test possibly opened)
- Stage: v3 steps A–E done ([log v3-031](docs/campaigns/pmm_ion_metal_v3/log.md#v3-031)): A4 Stage 6 selection `gvp_late_fusion__four_class__baseline` (five folds, development-validation); step E2 (36 fits; [E2 spec](docs/campaigns/pmm_ion_metal_v3/e2_assessment_spec.md)) built and CPU-checked ([log v3-033](docs/campaigns/pmm_ion_metal_v3/log.md#v3-033)), not run, required before F unless cancelled; no 6B refit or Stage 7.

**PMM core v2 was closed at fold 0 on 2026-10-03 (user decision):** nine fold-0 fits
are final validation-only evidence; 36 fits were not run; its neutral four-versus-five/six
test is unanswered; nothing is confirmed, promoted, refit or test-evaluated
([archived README](docs/archive/campaigns/pmm_ion_metal/README.md)).

## Anchor and evidence state

- Best validation result: none promoted; the closed PMM core fold-0 fits are Grade 5 exploratory evidence (incomplete grid Grade 6) per the archived README above; results in the [core summary](docs/notebook_outputs/summaries/summary_pmm_core_continuation_20260928.md); its fold 0 is not representative ([findings](docs/PARAMETER_FINDINGS.md#pmm-ion-campaign-single-fold-target-and-readout-comparisons)).
- v3 step E (A4 assessment `E_20261011T130128Z`): no family gains from five/six-class training (tie keeps `four_class`); `headdrop03`, `meanagg` not improvements; late fusion four-class 74.28 mean common-four BA, "better" than Only-ESMC (72.33) on all five folds.
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

- Authorized now: E2 GPU sessions ([log v3-034](docs/campaigns/pmm_ion_metal_v3/log.md#v3-034), user 2026-10-11 20:10 UTC) within the $55 ceiling recorded in E2's launch control (about $32.15 spent at 20:10 UTC; E→E2→F forecast $41.7–48.9, log v3-033) and the controller caps; the advance authorization through step F stands ([log v3-014](docs/campaigns/pmm_ion_metal_v3/log.md#v3-014)); step F after E2 with its own plan approval; no Round C.
- GPU/VM: `deepmzyme-l4` TERMINATED (verified 2026-10-11 13:03 UTC); disk and archive snapshot kept (about $0.55/day).

Next: E2 sessions with `pmm_v3_step_e2.py` (session 1 from 20:15 UTC, then after 00:00 UTC), then the workstation E2 assessment and log; step F after E2 ([draft](docs/campaigns/pmm_ion_metal_v3/step_f_plan_draft.md)); open: fairness arm, cost-gated augmentations.
Final-test label `both_results_secondary` is recorded ([log v3-022](docs/campaigns/pmm_ion_metal_v3/log.md#v3-022)). Stage 6 selection, a completed Stage 6B refit and frozen report/checkpoint rules precede one-shot Stage 7; no test-based selection ([Plan](Plan.md#canonical-staged-metal-training-pipeline)).

## History and update rule

Overwrite this file after preserving dated changes in the campaign's `log.md`; keep the objective/stage and campaign lines, anchor, best result and grade, caveats, dataset/test readiness, authorization and GPU/VM state, blockers, next action and evidence links. Budgets stay in the playbooks, findings in [PARAMETER_FINDINGS](docs/PARAMETER_FINDINGS.md), policy in [Plan](Plan.md), batch evidence in the [index](docs/notebook_outputs/README.md); the [history map](docs/archive/consolidation_2026-09/MAP.md) holds the recovered 09-16/17 pause and removed policy.
