# PMM ion-level metal comparison v3

Status: planned (2026-10-03 planning only; no run authorized)

Successor to the [closed PMM campaign](../../archive/campaigns/pmm_ion_metal/README.md),
whose fold-0 results are never confirmatory evidence here.

User decision (2026-10-03, recorded before any run): train separate `four_class`,
`five_class` and `six_class` arms for Only-ESMC, Only-GVP and graph-level late
fusion, all evaluated on common-four (Mn, Cu, Zn, Class VIII = Fe+Co+Ni) with
native five/six metrics kept. Comparing objectives is the
[neutral test](../../../Plan.md#2-train-the-metal-classification-model); its
metric, collapse rule, checkpoint rule, decision and tie rules must be frozen
here before the first fit.

Open design decisions, all before any fit or GPU use:

- final metal test route ([ledger](../../DATASETS.md#test-use-ledger));
- new folds that are not size strata
  ([TECH-020](../../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-020--greedy-k-fold-assignment-concentrates-large-groups-in-fold-0));
- one checkpoint rule shared by CV and the final refit
  ([TECH-027](../../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-027--cross-validation-and-final-refit-use-different-checkpoint-rules));
- comparators without the PMM label leak
  ([TECH-028](../../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-028--pmm-comparator-inherits-a-true-metal-label-leak));
- fixed architecture target, seeds, budget ceiling, stop and retention rules,
  and the executable playbook recipe.

Nothing in this folder authorizes training, GPU use, refits or held-out
evaluation.
