# PMM ion-level metal comparison v3

Status: closed (2026-10-13 all steps A–F complete; closed by the user, log v3-052)

**Closed 2026-10-13 ([log v3-052](log.md#v3-052)).**
- **Results:** the validation findings, the secondary test result and the deferred, never-run experiments are in
  the [completion summary](../../../notebook_outputs/summaries/summary_pmm_ion_metal_v3_20261013.md).
- **Validation line:** [PARAMETER_FINDINGS](../../../PARAMETER_FINDINGS.md#pmm-ion-metal-v3-objective-test-and-stage-6-selection-closed-2026-10-13).
- **Move:** the folder was moved here byte-preserving.
  - `docs/campaigns/pmm_ion_metal_v3` stays as a registered alias for the frozen tools and records.
  - The six hash-bound specifications are pinned in [frozen_moves.json](../../frozen_moves.json).
  - Every tool, specification and record verifies together at tag `pmm-ion-metal-v3-final`.
- **Storage and backups:** [handoff section 13](handoff.md#13-closeout-proposal-2026-10-13-pending-the-users-approval).

Successor to the [closed PMM campaign](../pmm_ion_metal/README.md),
whose fold-0 results are never confirmatory evidence here.

Objectives (user decision 2026-10-03, before any run): separate `four_class`,
`five_class` and `six_class` arms for Only-ESMC, Only-GVP and graph-level late
fusion, all evaluated on common-four (Mn, Cu, Zn, Class VIII = Fe+Co+Ni) with
native five/six metrics kept. Comparing objectives is the
[neutral test](../../../../Plan.md#2-train-the-metal-classification-model).

The [approved plan](plan.md) fixes the order: CPU preparation and new strict
folds, a GPU speed check, a fold-0 baseline with a regression check, ranked
improvement screening on one fold, five-fold confirmation, then the final refit
and one evaluation of PMM's test set. Dated decisions, including the
checkpoint rule, the PMM comparison basis and the budget ceiling, are in the
[log](log.md).

Fold set: `v3-seqid90-s42-b2`, frozen 2026-10-04 ([log v3-005](log.md#v3-005)).
Assessment rules: [frozen A4 specification](assessment_spec.md) (2026-10-05).
How to run: [metal playbook](../../../METAL_TRAINING_PIPELINE_PLAYBOOK.md#pmm-ion-metal-v3-campaign-pmm_ion_metal_v3).
Runtime and run identities are recorded before use; v2 artifacts stay untouched.
The [regularization amendment](log.md#v3-017) is [implemented](log.md#v3-018).
[Step E2](plan.md#steps) ([log v3-025](log.md#v3-025)): the eight remaining step D passes
singly on folds 1–4 (36 fits, seed 42), before step F unless the user cancels it; rules
frozen in the [E2 specification](e2_assessment_spec.md) ([log v3-028](log.md#v3-028));
completed: no candidate replaced the A4 selection ([log v3-037](log.md#v3-037)). The pre-step-F
10-epoch schedule comparison ran and was not adopted ([handoff section 11](handoff.md#11-pre-step-f-review), [log v3-041](log.md#v3-041)).
Step F ran under the approved [plan](step_f_plan_draft.md) (refit, two freezes, one secondary test pass;
[log v3-048](log.md#v3-048)); results in the
[completion summary](../../../notebook_outputs/summaries/summary_pmm_ion_metal_v3_20261013.md).
The [handoff](handoff.md) keeps the final verified state and evidence paths. Nothing remains authorized
([STATUS](../../../../EXPERIMENT_STATUS.md)); the $55 ceiling closed at about $41.8 spent ([log v3-025](log.md#v3-025)).
