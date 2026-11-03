# EC class versus CLEAN on the official CLEAN split30 benchmark

Status: active (2026-11-03; cache-backed bootstrap implemented; G2 host qualification pending)

Goal: predict the first EC digit and outperform retrained CLEAN on the official `split30` five-fold metalloenzyme
benchmark and its test proteins without a ≥30% identity hit to training. This is a secondary benchmark;
the EC Stage 7 final-test route stays unresolved
([Plan §8](../../../Plan.md#8-data-leakage-and-split-policy); [DATASETS](../../DATASETS.md#final-reporting-status)).
It supersedes the 2026-09-25 [CLEAN30 plan](../../plans/pocket_level_EC_task_compare_clean30_final_plan.md) by
reference; that file stays unchanged.

- [Design](design.md): benchmark, populations, systems, training rule, metric, claim rule, test access and licence.
- [Log](log.md): evc-001 (decisions), evc-002 (G0), evc-003 (G1 checks), evc-004 (G1 approval), evc-005 (Phase 2 start), evc-006 (inputs, round 1b), evc-007 (views, calibration), evc-008 (dry runs, Freeze-1 code).
- [Recipe](../../EC_TRAINING_PIPELINE_PLAYBOOK.md#ec-class-versus-clean30-campaign-recipe): commands.
- [S1 amendment](s1_amendment/amendment.md): fixed policy, source pins and verification evidence.
- [Continuation](continuation.md) and [G2 readiness](g2_readiness.md): path to the first fold-0 decision.
- Outputs (secondary drive, NVMe mirror): `DeepMzyme_Data/campaigns/ec_class_vs_clean30_2026-10/`.

Official folds stay unchanged. The [design](design.md) defines both co-primary populations, the EC1–6 balanced-accuracy
endpoint, paired homology-cluster inference, published CLEAN comparator and five-seed DeepMzyme refits.

Phases: 0 decisions and records; 1 infrastructure (CPU); 2 inputs and Freeze 1 (CPU); 3 measured GPU smoke (G2);
4 fold-0 development with a stop-or-continue gap gate; 5 folds 1–4; 6 refits and Freeze 2; 7 the single
scoring pass (test read at G5); 8 close-out. Estimate: GPU about $55–100 (three lanes) or $99–166 (serial), CLEAN about
160 CPU-hours run serially (measured in Phase 2; planned 95–135); the campaign GPU ceiling (G3) is set only after the Phase 3 measurement.

Current state: [Freeze 1](freeze1.md) is published; Freeze 1b passed ([evc-011](log.md#evc-011)).
The [log](log.md) preserves earlier exhausted capacity passes and Runpod closeout.
[evc-026](log.md#evc-026) records cap verification; [evc-027](log.md#evc-027) records the
isolated runtime repair, migration closeout and bounded replacement window.
[evc-028](log.md#evc-028) records the reproducible bootstrap and private artifact-cache extension.
Zero of twelve G2 events completed; Freeze 1c/G3 await measurements.
Root STATUS is stale (evc-009); [DATASETS](../../DATASETS.md#test-use-ledger) owns exposures and identity restrictions.
