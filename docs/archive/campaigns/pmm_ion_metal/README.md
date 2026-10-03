# PMM ion-level metal comparison

Status: closed (2026-10-03 user decision; closed at fold 0)

Scope: `pmm-core-v2`, campaign `pmm_ion_metal_v2_context`. The
[v2 manifest](../../../plans/pmm_core_scope_v2.json) retains nine ordinary-readout
fold-0 fits across Only-ESMC, Only-GVP and graph-level late fusion, each trained
separately with `four_class`, `five_class` and `six_class`; the other 36 fits were
not run. Every arm is evaluated on common-four (Mn, Cu, Zn, Class VIII =
Fe+Co+Ni); five/six arms also retain native metrics. The neutral
four-versus-five/six test is unanswered: not "no difference" and not "four wins"
([closure record](log.md#pmm-013);
[Plan](../../../../Plan.md#2-train-the-metal-classification-model)).

Dataset: frozen training-only PMM source cohort, 7,398 ions / 3,992 PDB groups;
`metal_example_unit=ion`. Five PDB-grouped folds frozen in `fold_membership.csv`
(`split_seed=42`, `split_stratify_by=metal_site`), model seed 42 only. The folds
are size strata, not exchangeable random folds
([TECH-020](../../../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-020--greedy-k-fold-assignment-concentrates-large-groups-in-fold-0);
[fold contract](../../../plans/metal_level_metal_task_compared_PMM_final_plan.md);
[fold regimes](../../../../Plan.md#fold-regimes)).

Validation: fold 0 only for the neural core; qualified fits are Grade 5, the
incomplete fivefold grid Grade 6. The released PMM comparator completed all five
folds (Grade 2; [index](../../../notebook_outputs/README.md#pmm-ion-level-comparison-with-corrected-protein-context))
with label-bearing upstream inputs
([TECH-028](../../../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-028--pmm-comparator-inherits-a-true-metal-label-leak)).
Seven original strict passes and GVP5/GVP6 strict failures remain intact; all
nine have separate retrospective `pmm-core-replay-v1` qualification, and the
original `1e-6` and retrospective `1e-5` checks are not interchangeable
([results and limits](../../../notebook_outputs/summaries/summary_pmm_core_continuation_20260928.md)).
No target/model promotion, final refit or held-out evaluation occurred.

| Sub-batch | Standing and evidence |
|---|---|
| v1 audit | Closed historical audit; [plan and preserved lineage](../../../plans/metal_level_metal_task_compared_PMM_final_plan.md) |
| Five-class screen v1 | Closed three-fit screen; [strict outcomes](../../../notebook_outputs/summaries/summary_pmm_five_class_screen_20260928.md) |
| Replay diagnostic | Closed; [separate v2.1 agreement](../../../notebook_outputs/raw/pmm_replay_diagnostic_20260928/screen_agreement_v2_1.md) applies to the earlier screen, not the later GVP5 fit |
| Core v1 | Superseded by v2; [four/six scope](../../../plans/pmm_core_scope_v1.json) preserved |
| Core v2 | Closed at fold 0; [execution contract](../../../plans/pmm_core_execution_v1.md), [core evidence](../../../notebook_outputs/raw/pmm_core_continuation_20260928/README.md) |
| Binding-aware arms | Closed; three trained arms remain separate exploratory evidence ([screen](../../../notebook_outputs/raw/pmm_ion_v2_context_fold0_20260927/runtime/screen_report.md)) |

Compute: GCP L4 via `~/deepmzyme-vm/bin/*` under
[gpu-use-skill](../../../../.agents/skills/gpu-use-skill/SKILL.md); the VM and disk
were retired to a snapshot on 2026-10-03 ([pmm-011](log.md#pmm-011)).

Files here: this overview, [verbatim STATUS history and closure record](log.md)
and the [storage retirement receipt](storage_retirement_20261003.json). Plans,
scopes, recipes and evidence keep their paths; all run artifacts and the frozen
`_code/pmm_core_scope_v2` checkout are preserved.

Closed: nothing here may be resumed or rerun. `pmm_core_scope_v2.json` keeps
`deferred_by_user` so the runner refuses execution. Never use this campaign's
fold-0 results as confirmatory evidence, and never retry GVP5/GVP6 for a chance
strict pass ([pmm-003](log.md#pmm-003), [pmm-006](log.md#pmm-006),
[pmm-007](log.md#pmm-007)). Successor planning:
[pmm_ion_metal_v3](../../../campaigns/pmm_ion_metal_v3/README.md).
