# EC class versus CLEAN on the official CLEAN split30 benchmark

Status: active (2026-11-12; amendment 2 and execution v5 published and verified; launches paused)

Goal: predict the first EC digit and outperform retrained CLEAN on the official `split30` five-fold metalloenzyme
benchmark and its test proteins without a ≥30% identity hit to training. This is a secondary benchmark;
the EC Stage 7 final-test route stays unresolved
([Plan §8](../../../Plan.md#8-data-leakage-and-split-policy); [DATASETS](../../DATASETS.md#final-reporting-status)).
It supersedes the 2026-09-25 [CLEAN30 plan](../../plans/pocket_level_EC_task_compare_clean30_final_plan.md) by
reference; that file stays unchanged.

- [Design](design.md): benchmark, populations, systems, training rule, metric, claim rule, test access and licence.
- [Log](log.md): dated decisions, preparation evidence and execution attempts.
- [Recipe](../../EC_TRAINING_PIPELINE_PLAYBOOK.md#ec-class-versus-clean30-campaign-recipe): commands.
- [S1 amendment](s1_amendment/amendment.md): fixed policy, source pins and verification evidence.
- [Amendment 2](amendment2/amendment.md): the 2026-11-04 decisions for real execution, with their evidence.
- [Continuation](continuation.md) and [G2 readiness](g2_readiness.md): path to the first fold-0 decision.
- Outputs (secondary drive, NVMe mirror): `DeepMzyme_Data/campaigns/ec_class_vs_clean30_2026-10/`.

Official folds stay unchanged. The [design](design.md) records the published scientific policy; [amendment 2](amendment2/amendment.md)
implements the 2026-11-04 decisions ([evc-037](log.md#evc-037)). [Execution v5](execution_v5/README.md)
owns the current launch hold, execution requirements and provisional cost scenarios.

The recorded phases and historical estimates remain in the [continuation](continuation.md).
The amended fold-0 scope and cost ceiling must be fixed before G2; historical estimates do not price that scope.

Current state: [Freeze 1](freeze1.md) is published; Freeze 1b passed ([evc-011](log.md#evc-011)).
The [log](log.md) preserves capacity attempts and verified cache seeding.
[evc-034](log.md#evc-034) records Pass I: main-runtime checks passed, but filesystem and path constraints
blocked scientific admission. The pod is stopped and deleted; G2 remains 0/12.
The POSIX correction is retained inside v5. [evc-036](log.md#evc-036) records the CPU-checked dispatch hooks.
[evc-037](log.md#evc-037) records amendment 2.
Next: complete the cost forecast, approve the fold-0 ceiling and obtain separate explicit launch authorization.
No launch loop is active.
Root STATUS is stale (evc-009); [DATASETS](../../DATASETS.md#test-use-ledger) owns exposures and identity restrictions.
