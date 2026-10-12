# PMM ion-level metal comparison v3

Status: active (2026-10-12 step F plan approved; the Stage 6B refit authorized; log v3-045)

Successor to the [closed PMM campaign](../../archive/campaigns/pmm_ion_metal/README.md),
whose fold-0 results are never confirmatory evidence here.

Objectives (user decision 2026-10-03, before any run): separate `four_class`,
`five_class` and `six_class` arms for Only-ESMC, Only-GVP and graph-level late
fusion, all evaluated on common-four (Mn, Cu, Zn, Class VIII = Fe+Co+Ni) with
native five/six metrics kept. Comparing objectives is the
[neutral test](../../../Plan.md#2-train-the-metal-classification-model).

The [approved plan](plan.md) fixes the order: CPU preparation and new strict
folds, a GPU speed check, a fold-0 baseline with a regression check, ranked
improvement screening on one fold, five-fold confirmation, then the final refit
and one evaluation of PMM's test set. Dated decisions, including the
checkpoint rule, the PMM comparison basis and the budget ceiling, are in the
[log](log.md).

Fold set: `v3-seqid90-s42-b2`, frozen 2026-10-04 ([log v3-005](log.md#v3-005)).
Assessment rules: [frozen A4 specification](assessment_spec.md) (2026-10-05).
How to run: [metal playbook](../../METAL_TRAINING_PIPELINE_PLAYBOOK.md#pmm-ion-metal-v3-campaign-pmm_ion_metal_v3).
Runtime and run identities are recorded before use; v2 artifacts stay untouched.
The [regularization amendment](log.md#v3-017) is [implemented](log.md#v3-018).
[Step E2](plan.md#steps) ([log v3-025](log.md#v3-025)): the eight remaining step D passes
singly on folds 1–4 (36 fits, seed 42), before step F unless the user cancels it; rules
frozen in the [E2 specification](e2_assessment_spec.md) ([log v3-028](log.md#v3-028));
completed: no candidate replaced the A4 selection ([log v3-037](log.md#v3-037)). The pre-step-F
10-epoch schedule comparison ran and was not adopted ([handoff section 11](handoff.md#11-pre-step-f-review), [log v3-041](log.md#v3-041)).
Step F tooling is ready ([handoff section 12](handoff.md#12-step-f-tooling-prepared-2026-10-12-log-v3-042)); its
[plan](step_f_plan_draft.md) awaits approval.
A new chat continues from the [handoff](handoff.md) (ready-to-paste prompt, verified state, evidence paths).
Execution follows the authorization and readiness state in
[STATUS](../../../EXPERIMENT_STATUS.md); the ceiling is $55 ([log v3-025](log.md#v3-025)).
