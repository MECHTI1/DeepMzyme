# DeepMzyme Current Experiment Status

- Status: active (2026-10-14 metal completion CPU preparation reviewed, repaired and re-sealed; execution not authorized)
- Last execution evidence: 2026-10-13 EC1 inventory; no new model fits or test evaluations.

## Current objective and stage

- Current campaign: [metal_completion_2026-10](docs/campaigns/metal_completion_2026-10/README.md)
- Objective: finish a bounded metal optimization comparison across four/five/six objectives, then independently evaluate the frozen final model; EC work is deferred by the user's metal-first choice.
- Other open campaigns: [EC independent support](docs/campaigns/ec1_independent_support_2026-10/README.md) (deferred); [metal_single_gpu](docs/campaigns/metal_single_gpu/README.md) (paused); [zenodo_pmm_exact_2026-09-24](docs/campaigns/zenodo_pmm_exact_2026-09-24/README.md) (paused).
- Stage: successor CPU preparation independently reviewed (no blocker; majors repaired or pinned) and re-sealed; the discovery manifest is frozen. Next is validation-only discovery, then Stage 6 grouped-fold/seed confirmation; neither has run. [Evidence](docs/campaigns/metal_completion_2026-10/evidence.md) owns hashes, checks and control reuse; the [log](docs/campaigns/metal_completion_2026-10/log.md) records the review.

## Anchor and evidence state

- Best validation result: retained v3 late-fusion direct-four, terminal 50-epoch cosine training. Grade 2 validation evidence: five grouped folds, seed 42, common-four BA 74.28%; no supported benefit from five/six training. [Findings](docs/PARAMETER_FINDINGS.md#pmm-ion-metal-v3-objective-test-and-stage-6-selection-closed-2026-10-13).
- Lower trunk LR remains unpromoted. [Closed comparison](docs/archive/campaigns/pmm_trunk_lr_2026-10/README.md).
- Historical refit `a0007846…` and secondary test reports remain unchanged. [Preserved summary](docs/notebook_outputs/summaries/summary_pmm_ion_metal_v3_20261013.md). New code and synthetic checks are not accuracy evidence; the unexecuted successor is not a better model.
- EC remains at fixed-split Grade 3 evidence (deferred campaign).

### Known caveats and open mismatches

- Additional seeds reuse development folds. Cross-target ranking requires the campaign assessor ([TECH-010](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-010--four-class-endpoint-and-paired-metal-target-recipes-are-not-reconciled)); no new model has been evaluated.
- EC child-file guard repair remains open ([TECH-030](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-030--discovered-ec-cache-files-can-bypass-the-training-path-guard)); the trainer's focal branch ignores label smoothing, unreachable here ([TECH-031](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-031--the-focal-metal-loss-silently-ignores-label-smoothing-and-uses-weighted-ce-confidence)).

## Dataset and test readiness

The primary independent final-test route remains unresolved. [DATASETS](docs/DATASETS.md#test-use-ledger) owns historical exposure; opened PMM tests are not pristine. The [offline evaluation protocol](docs/campaigns/metal_completion_2026-10/evaluation.md) authorizes no candidate retrieval and establishes no support. Source dates, complete exclusions, near-copy checks and report/refit freezes precede scoring.

## Blockers and immediate next action

- Authorized now: nothing to run. No GPU start, real-data training/inference, download, new held-out access, commit or push.
- GPU/VM: `deepmzyme-l4` TERMINATED (verified 2026-10-14T15:51Z); gross rate $0.87917/h; retained disk and two snapshots about $1.00/day. [Backup inventory](docs/BACKUP_AND_DATA_INVENTORY.md#5a-local-independent-backups-and-gcp-state-verified-2026-10-13).
- Next: the user's scoped decision on the [costed proposal](docs/campaigns/metal_completion_2026-10/evidence.md#proposed-execution-ceiling--approval-pending); then the GPU-host [deployment prerequisite](docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md#cpu-preparation-block) (identical absolute paths) and the authorization/session/spending receipts before the first discovery unit. The [handoff](docs/campaigns/metal_completion_2026-10/handoff.md) is the next-session prompt.

## History and update rule

The exact previous STATUS is preserved in the [campaign log](docs/campaigns/metal_completion_2026-10/log.md). Overwrite this page only after preserving dated history. Scientific policy remains in Plan, recipes in the metal playbook, dataset/test access in DATASETS, and validation conclusions in PARAMETER_FINDINGS.
