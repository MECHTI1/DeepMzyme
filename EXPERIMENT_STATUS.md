# DeepMzyme Current Experiment Status

- Status: active (2026-10-14 metal completion CPU preparation complete; execution not authorized)
- Last execution evidence: 2026-10-13 EC1 inventory; no new model fits or test evaluations.

## Current objective and stage

- Current campaign: [metal_completion_2026-10](docs/campaigns/metal_completion_2026-10/README.md)
- Objective: finish a bounded metal optimization comparison across four/five/six objectives, then independently evaluate the frozen final model; EC work is deferred by the user's metal-first choice.
- Other open campaigns: [EC independent support](docs/campaigns/ec1_independent_support_2026-10/README.md) (deferred); [metal_single_gpu](docs/campaigns/metal_single_gpu/README.md) (paused); [zenodo_pmm_exact_2026-09-24](docs/campaigns/zenodo_pmm_exact_2026-09-24/README.md) (paused).
- Stage: successor CPU preparation verified; discovery manifest frozen. Next is validation-only discovery followed by Stage 6 grouped-fold/seed confirmation; neither has run. [Campaign evidence](docs/campaigns/metal_completion_2026-10/evidence.md) records checks and control reuse.

## Anchor and evidence state

- Best validation result: retained v3 late-fusion direct-four, terminal 50-epoch cosine training. Grade 2 validation evidence: five grouped folds, seed 42, common-four BA 74.28%; no supported benefit from five/six training. [Findings](docs/PARAMETER_FINDINGS.md#pmm-ion-metal-v3-objective-test-and-stage-6-selection-closed-2026-10-13).
- Lower trunk LR remains unpromoted; the completed comparison retained the reference. [Closed comparison](docs/archive/campaigns/pmm_trunk_lr_2026-10/README.md).
- Historical refit `a0007846…` and secondary test reports remain unchanged. [Preserved summary](docs/notebook_outputs/summaries/summary_pmm_ion_metal_v3_20261013.md). New code and synthetic checks are not evidence of improved accuracy.
- EC remains at fixed-split Grade 3 evidence; the completed support inventory and eligibility restrictions are preserved in its deferred campaign.

### Known caveats and open mismatches

- Additional seeds reuse development folds. Cross-target ranking requires the campaign assessor ([TECH-010](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-010--four-class-endpoint-and-paired-metal-target-recipes-are-not-reconciled)); no new model has been evaluated.
- EC child-file guard repair remains open before reuse ([TECH-030](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-030--discovered-ec-cache-files-can-bypass-the-training-path-guard)).

## Dataset and test readiness

The primary independent final-test route remains unresolved. [DATASETS](docs/DATASETS.md#test-use-ledger) owns historical exposure; opened PMM tests are not pristine. The new [offline evaluation protocol](docs/campaigns/metal_completion_2026-10/evaluation.md) does not authorize candidate retrieval or establish adequate support. Source dates, complete exclusions, near-copy checks and final report/refit freezes precede scoring. Additional optimizer seeds on reused development folds are not independent data.

## Blockers and immediate next action

- Authorized now: CPU implementation, synthetic checks, allowlisted historical development-evidence verification and preparation of a concrete costed proposal. No GPU start, real-data training/inference, download, new held-out access, commit or push.
- GPU/VM: last verified `deepmzyme-l4` TERMINATED on 2026-10-13; no fresh provider check in this task. Retained disks/snapshots continue to cost money. [Backup inventory](docs/BACKUP_AND_DATA_INVENTORY.md#5a-local-independent-backups-and-gcp-state-verified-2026-10-13).
- Next: review the staged draft and costed proposal; obtain scoped execution/budget authority before launch. Independent evaluation has separate provenance and data-access prerequisites. The [updated Claude handoff](docs/campaigns/metal_completion_2026-10/handoff.md) requests independent review of completed work.

## History and update rule

The exact previous STATUS is preserved in the [campaign log](docs/campaigns/metal_completion_2026-10/log.md). Overwrite this page only after preserving dated history. Scientific policy remains in Plan, recipes in the metal playbook, dataset/test access in DATASETS, and validation conclusions in PARAMETER_FINDINGS.
