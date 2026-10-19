# DeepMzyme Current Experiment Status

- Status: active (2026-10-19 metal OOF-CV and held-out evaluation complete; `deepmzyme-l4` TERMINATED)
- Last execution evidence: 2026-10-19 Phase 2 Unified 5-fold L4 run (`1h 09m`, $1.02) and 180-checkpoint OOF-CV / test evaluation ([report](docs/campaigns/phase2_unified_2026-10/README.md); [log](docs/campaigns/phase2_unified_2026-10/log.md)).

## Current objective and stage

- Current campaign: [phase2_unified_2026-10](docs/campaigns/phase2_unified_2026-10/README.md) (complete; [metal_completion_2026-10](docs/campaigns/metal_completion_2026-10/README.md) complete)
- Objective: metal ion classification benchmark complete; EC classification is next when authorized.
- Other open campaigns: [EC independent support](docs/campaigns/ec1_independent_support_2026-10/README.md) (deferred); [metal_single_gpu](docs/campaigns/metal_single_gpu/README.md) (paused); [zenodo_pmm_exact_2026-09-24](docs/campaigns/zenodo_pmm_exact_2026-09-24/README.md) (paused).
- Stage: one-shot Step F refit (`67.55%` Full Test) and 180-checkpoint 5-Fold OOF-CV / held-out progression complete ([summary JSON](docs/campaigns/phase2_unified_2026-10/phase2_unified_summary.json)).

## Anchor and evidence state

- Best validation result: pre-declared single-model `B 4-class` baseline (`74.28%` mean / `74.34%` OOF CV; Step F refit `67.55%` Full, `69.33%` Eligible, `48.85%` Clean; [v3 summary](docs/notebook_outputs/summaries/summary_pmm_ion_metal_v3_20261013.md)). By 7,398-ion OOF CV ([report](docs/campaigns/phase2_unified_2026-10/README.md)): 5-Fold `trunklr1e4` (`75.51%` OOF $\rightarrow$ `70.75%` Full / `72.25%` Eligible); 15-Model `P 5-class` (`75.82%` OOF $\rightarrow$ `71.10%` Full / `72.68%` Eligible); 5-Fold Unified `P+5c+trunklr1e4` (`74.91%` OOF $\rightarrow$ `71.72%` Full / `73.08%` Eligible); `Top-4 LF` (`76.81%` OOF $\rightarrow$ **`72.20%` Full** / `73.94%` Eligible); `CV-Locked Tri-Modal` (**`77.17%` OOF** $\rightarrow$ **`61.33%` Clean**).
- EC stays at fixed-split Grade 3 evidence (deferred).

### Known caveats and open mismatches

- Cross-target ranking used the campaign assessor ([TECH-010](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-010--four-class-endpoint-and-paired-metal-target-recipes-are-not-reconciled)); post-Step-F multi-checkpoint evaluations carry `both_results_secondary` disclosure and PMM label-leak caveat ([TECH-028](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-028--pmm-comparator-inherits-a-true-metal-label-leak)).
- EC child-file guard ([TECH-030](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-030--discovered-ec-cache-files-can-bypass-the-training-path-guard)) and focal-loss label smoothing ([TECH-031](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-031--the-focal-metal-loss-silently-ignores-label-smoothing-and-uses-weighted-ce-confidence)) remain open.

## Dataset and test readiness

[DATASETS](docs/DATASETS.md#test-use-ledger) records both test evaluations (2026-10-13 Step F pass and 2026-10-19 180-checkpoint evaluation). External temporal PDB evaluation remains optional ([protocol](docs/campaigns/metal_completion_2026-10/evaluation.md)).

## Blockers and immediate next action

- Authorized now: nothing to run; no active GPU session.
- GPU/VM: `deepmzyme-l4` TERMINATED (2026-10-19); all artifacts backed up across 5 tiers ([inventory](docs/BACKUP_AND_DATA_INVENTORY.md)).
- Next: metal paper writing, or user decision to open standalone EC classification.

## History and update rule

Previous STATUS is preserved in [`docs/campaigns/phase2_unified_2026-10/log.md`](docs/campaigns/phase2_unified_2026-10/log.md). Policy: [Plan](Plan.md), [DATASETS](docs/DATASETS.md), [PARAMETER_FINDINGS](docs/PARAMETER_FINDINGS.md).
