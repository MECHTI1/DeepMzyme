# EC class versus CLEAN on the official CLEAN split30 benchmark

Status: active (2026-10-23; G0 approved, Phase 1 CPU infrastructure)

Goal: a DeepMzyme model for the EC class (the first digit of the EC number: 1 oxidoreductases to 7 translocases) that
outperforms a retrained CLEAN on the official CLEAN `split30` five-fold benchmark (metalloenzyme subset), and on its test proteins without a ≥30% identity hit to training. This is a
named secondary five-fold benchmark, not the EC Stage 7 final-test route, which stays unresolved
([Plan §8](../../../Plan.md#8-data-leakage-and-split-policy); [DATASETS](../../DATASETS.md#final-reporting-status)).
It supersedes the 2026-09-25 [CLEAN30 plan](../../plans/pocket_level_EC_task_compare_clean30_final_plan.md) by
reference; that file stays unchanged.

- [Design](design.md): benchmark, populations, systems, training rule, metric, claim rule, test access and licence.
- [Log](log.md): evc-001 (decisions and reasons), evc-002 (G0 approval).
- Outputs (secondary drive, NVMe mirror; not yet created): `DeepMzyme_Data/campaigns/ec_class_vs_clean30_2026-10/`.

In short: official folds unchanged; co-primary populations are all evaluable metallo test proteins (about 530) and
those with neither a ≥30% identity hit to their fold's full training file nor a shared selected AlphaFill donor. CLEAN
is retrained per fold with its published recipe. DeepMzyme selects per fold from its own training proteins and ends as
a 5-seed ensemble. The primary metric is balanced accuracy over EC classes 1–6, and a claim needs paired homology-cluster bootstrap
lower bounds above 0.

Phases: 0 decisions and records; 1 infrastructure (CPU); 2 inputs and Freeze 1 (CPU); 3 measured GPU smoke (G2, at
most $3); 4 fold-0 development with a stop-or-continue gap gate; 5 folds 1–4; 6 refits and Freeze 2; 7 the single
scoring pass (test read at G5); 8 close-out. Estimate: GPU about $55–100 (three lanes) or $99–166 (serial), CLEAN about
95–135 CPU-hours; the campaign GPU ceiling (G3) is set only after the Phase 3 measurement.

Current state: G0 approved on 2026-10-23; Phase 1 (CPU code and tests) in progress. Nothing installed, trained or
evaluated for the campaign; its test read (G5) has not happened. Incidental planning-time exposures, including one
deleted upstream download, are in the [test-use ledger](../../DATASETS.md#test-use-ledger).
