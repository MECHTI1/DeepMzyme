# PMM ion-level metal comparison v3

Status: planned (2026-10-04 plan approved; CPU preparation only)

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
Runtime and run identities are recorded here before use; v2 artifacts stay
untouched. Nothing in this folder authorizes a GPU start, refit or
held-out evaluation without the user's explicit OK.
