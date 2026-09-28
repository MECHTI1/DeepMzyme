# PMM ion-level metal comparison

Status: paused (2026-09-28)

Current scope: `pmm-core-v2`, campaign `pmm_ion_metal_v2_context`.
The [v2 manifest](../../plans/pmm_core_scope_v2.json) retains nine ordinary-readout
fold-0 fits across Only-ESMC, Only-GVP and graph-level late fusion, each trained
separately with `four_class`, `five_class` and `six_class`; 36 fits are deferred.
The common-four endpoint is Mn, Cu, Zn, Class VIII = Fe+Co+Ni; every arm is evaluated
on it, and five/six arms also retain native metrics. No training objective is primary. Five- or six-class
training followed by collapse differs from direct four-class training; comparing objectives is a neutral test
where better, no difference and worse are all valid ([Plan](../../../Plan.md#2-train-the-metal-classification-model)).

Dataset: frozen training-only PMM source cohort, 7,398 ions / 3,992 PDB groups;
`metal_example_unit=ion`. Five PDB-grouped folds are frozen once in
`fold_membership.csv`, `split_seed=42`, `split_stratify_by=metal_site`;
model seed 42 only. This is one-seed grouped-fold design, not seed-repeat
confirmation or the pocket-stratified 2026-09-22/23 benchmark
([fold contract](../../plans/metal_level_metal_task_compared_PMM_final_plan.md),
[fold regimes](../../../Plan.md#fold-regimes)).

Validation: fold 0 only for the neural core; individual qualified fits are Grade 5,
the incomplete fivefold grid Grade 6. The released PMM comparator completed all five
folds (Grade 2; [index](../../notebook_outputs/README.md#pmm-ion-level-comparison-with-corrected-protein-context)).
Seven original strict passes and GVP5/GVP6
strict failures remain intact. All nine have separate retrospective
`pmm-core-replay-v1` qualification; original absolute `1e-6` and retrospective
absolute `1e-5` checks are not interchangeable. Classes/metrics remain unchanged
([results and limits](../../notebook_outputs/summaries/summary_pmm_core_continuation_20260928.md)).
No target/model promotion, final refit or held-out evaluation occurred in this core.

| Sub-batch | Standing and evidence |
|---|---|
| v1 audit | Closed historical audit; [plan and preserved lineage](../../plans/metal_level_metal_task_compared_PMM_final_plan.md) |
| Five-class screen v1 | Closed three-fit screen; [strict outcomes](../../notebook_outputs/summaries/summary_pmm_five_class_screen_20260928.md) |
| Replay diagnostic | Closed; [separate v2.1 agreement](../../notebook_outputs/raw/pmm_replay_diagnostic_20260928/screen_agreement_v2_1.md) applies to the earlier screen, not the later GVP5 fit |
| Core v1 | Superseded by v2; [four/six scope](../../plans/pmm_core_scope_v1.json) preserved |
| Core v2 | Paused; [execution contract](../../plans/pmm_core_execution_v1.md), [core evidence](../../notebook_outputs/raw/pmm_core_continuation_20260928/README.md) |
| Binding-aware arms | Paused; three earlier trained arms remain separate exploratory evidence ([screen](../../notebook_outputs/raw/pmm_ion_v2_context_fold0_20260927/runtime/screen_report.md)) |

Compute route: CLI runners on GCP L4 via `~/deepmzyme-vm/bin/*` under
[gpu-use-skill](../../../.agents/skills/gpu-use-skill/SKILL.md); Colab is the
authorized fallback, notebook secondary ([route decision](../../archive/consolidation_2026-09/PLAN_v2.md#6-user-decisions)).
Production stays one training worker per GPU; the
[concurrency probe](../../plans/pmm_gpu_concurrency_probe.md) establishes no accuracy equivalence.
Last verified VM state and retained-disk cost: the `GPU/VM:` line in [current status](../../../EXPERIMENT_STATUS.md#blockers-and-immediate-next-action).

Files here: this overview and [verbatim STATUS history](log.md).
Plans, JSON scopes, recipes and immutable evidence keep their existing paths.
How to continue: no experiment or GPU action is authorized now. Resume requires
an explicit user request, refreshed budget authority and new readiness verification
under the [v2 scope](../../plans/pmm_core_scope_v2.json) and
[guarded recipe](../../METAL_TRAINING_PIPELINE_PLAYBOOK.md#active-core-only-continuation).
On any authorized resume, reuse all completed fits and run only missing
selectors; do not rerun completed units ([pmm-003](log.md#pmm-003),
[pmm-008](log.md#pmm-008)). An old scope or implementation hash cannot
authorize a run ([pmm-001](log.md#pmm-001)). Do not retry or retrain GVP5 or
GVP6, or repeat their replay, in the hope of a chance strict pass; their original
strict failures stay recorded ([pmm-003](log.md#pmm-003), [pmm-006](log.md#pmm-006),
[pmm-007](log.md#pmm-007)). The campaign plan's pre-declared contrasts and tie
rule (without clear improvement, keep direct four-class) govern its comparison. Old forecasts and historical next actions in the log
confer no authorization.
