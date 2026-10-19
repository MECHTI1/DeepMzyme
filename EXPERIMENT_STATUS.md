# DeepMzyme Current Experiment Status

- Status: active (2026-10-16 optimization complete; B four-class retained; Stage 6B source verified; independent evaluation not started)
- Last execution evidence: 2026-10-16 confirmation fits on the GCP L4 (69 campaign fits, seven sessions); no test evaluations.

## Current objective and stage

- Current campaign: [metal_completion_2026-10](docs/campaigns/metal_completion_2026-10/README.md)
- Objective: independently evaluate the selected metal model; EC is deferred (the user's metal-first choice).
- Other open campaigns: [EC independent support](docs/campaigns/ec1_independent_support_2026-10/README.md) (deferred); [metal_single_gpu](docs/campaigns/metal_single_gpu/README.md) (paused); [zenodo_pmm_exact_2026-09-24](docs/campaigns/zenodo_pmm_exact_2026-09-24/README.md) (paused).
- Stage: optimization complete (validation only): discovery (45 cells) froze recipe P; Stage 6 confirmation (40 cells) passed no challenger, so the unchanged B four-class v3 recipe is retained. Stage 6B source: v3 full non-test refit `a0007846…`, compatible (25/25 CPU checks), designated without retraining under the [evidence](docs/campaigns/metal_completion_2026-10/evidence.md#stage-6b-source-and-compatibility-2026-10-16) conditions. Stage 7 has not started; the [log](docs/campaigns/metal_completion_2026-10/log.md) records decisions.

## Anchor and evidence state

- Best validation result: retained v3 late-fusion direct-four, terminal 50-epoch cosine. Grade 2: five grouped folds, seed 42, common-four BA 74.28%; no supported benefit from five/six training ([v3](docs/PARAMETER_FINDINGS.md#pmm-ion-metal-v3-objective-test-and-stage-6-selection-closed-2026-10-13)) or from recipes P/S ([completion](docs/PARAMETER_FINDINGS.md#metal-completion-bps-loss-recipes-on-fourfivesix-objectives-executed-2026-10-1516)).
- Trunk LR unpromoted ([closed](docs/archive/campaigns/pmm_trunk_lr_2026-10/README.md)).
- EC stays at fixed-split Grade 3 evidence (deferred).

### Known caveats and open mismatches

- Seeds 43/44 reuse development folds; cross-target ranking used the campaign assessor ([TECH-010](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-010--four-class-endpoint-and-paired-metal-target-recipes-are-not-reconciled)). The refit already had one secondary held-out pass ([summary](docs/notebook_outputs/summaries/summary_pmm_ion_metal_v3_20261013.md)); a primary Stage 7 report from it conflicts with Plan's one-shot rule until the user approves an exception or amends Plan.
- EC child-file guard repair remains open ([TECH-030](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-030--discovered-ec-cache-files-can-bypass-the-training-path-guard)); the trainer's focal branch ignores label smoothing, unreachable here ([TECH-031](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-031--the-focal-metal-loss-silently-ignores-label-smoothing-and-uses-weighted-ce-confidence)).

## Dataset and test readiness

The primary independent final-test route remains unresolved; the reused refit is valid only for an external cohort disjoint from its training cohort, every opened test and their near-copies. [DATASETS](docs/DATASETS.md#test-use-ledger) owns historical exposure; opened PMM tests are not pristine. The proposed temporal PDB cohort is neither assembled nor certified; the [protocol](docs/campaigns/metal_completion_2026-10/evaluation.md#preparation-inventory-2026-10-16) owns its inventory and blockers.

## Blockers and immediate next action

- Authorized now: nothing to run; the 2026-10-15 GPU authorization is consumed. No GPU start, refit, test access, download, commit or push.
- GPU/VM: `deepmzyme-l4` TERMINATED (confirmed 2026-10-16T11:59Z) after seven sessions: 18 h 37 min, $16.37 gross running; $0.87917/h; storage ~$1/day. [Backup inventory](docs/BACKUP_AND_DATA_INVENTORY.md#5a-local-independent-backups-and-gcp-state-verified-2026-10-13).
- Next: the user's route and one-shot decisions, then approval of the label-free prior-use read ([blockers](docs/campaigns/metal_completion_2026-10/evaluation.md#remaining-blockers); [handoff](docs/campaigns/metal_completion_2026-10/handoff.md)).

## History and update rule

The exact previous STATUS is preserved in the [campaign log](docs/campaigns/metal_completion_2026-10/log.md). Overwrite this page only after preserving dated history. Scientific policy remains in Plan, recipes in the metal playbook, dataset/test access in DATASETS, and validation conclusions in PARAMETER_FINDINGS.
