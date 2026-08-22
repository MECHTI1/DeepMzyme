# DeepMzyme Documentation Index

Use this page to locate the owner of a fact. Do not copy mutable values between
documents when a link is sufficient.

## Start here

1. [`GETTING_STARTED.md`](GETTING_STARTED.md) — execution paths, environment
   limits, first checks, and repository navigation.
2. [`EXPERIMENT_STATUS.md`](../EXPERIMENT_STATUS.md) — current objective,
   anchors, blockers, and next actions.
3. [`DATASETS.md`](DATASETS.md) — datasets, splits, Hugging Face files,
   bundles, provenance, and
   historical test-use record.
4. [`PARAMETER_FINDINGS.md`](PARAMETER_FINDINGS.md) — validation/HPO findings
   with evidence grades.
5. [`notebook_outputs/README.md`](notebook_outputs/README.md) — experiment-batch
   index and links to summaries/configs/raw evidence.

Scientific policy is in [`Plan.md`](../Plan.md).

## Ownership

| Information | Authority | Not its role |
|---|---|---|
| Public overview and minimal quick start | [`README.md`](../README.md) | Live defaults, status, or experiment history |
| Executable orientation and local setup limits | [`GETTING_STARTED.md`](GETTING_STARTED.md) | Exact experiment budgets or mutable results |
| Colab browser/CLI connection and environment procedure | [`COLAB_GPU_RUNBOOK.md`](COLAB_GPU_RUNBOOK.md) | Scientific stage values or model selection |
| Current status and next action | [`EXPERIMENT_STATUS.md`](../EXPERIMENT_STATUS.md) | Long chronological diary |
| Scientific/design/test policy | [`Plan.md`](../Plan.md) | Dataset inventory or copied stage blocks |
| Dataset identity, readiness, bundles, test use | [`DATASETS.md`](DATASETS.md) | Preparation procedure |
| Empirical parameter/HPO knowledge | [`PARAMETER_FINDINGS.md`](PARAMETER_FINDINGS.md) | Future executable search-space prescription |
| Experiment batches and evidence links | [`notebook_outputs/README.md`](notebook_outputs/README.md) | Current status |
| Exact metal execution recipes | [`METAL_TRAINING_PIPELINE_PLAYBOOK.md`](METAL_TRAINING_PIPELINE_PLAYBOOK.md) | Measured results |
| EC recipe intent and compatibility warning | [`EC_TRAINING_PIPELINE_PLAYBOOK.md`](EC_TRAINING_PIPELINE_PLAYBOOK.md) | A claim that all affected blocks currently execute |
| Stable notebook option semantics | [`METAL_NOTEBOOK_CONFIGURATION_GUIDE.md`](METAL_NOTEBOOK_CONFIGURATION_GUIDE.md) | Live cell-value snapshot |
| Implemented notebook behavior | [`DeepMzyme_training_colab.ipynb`](../notebooks/DeepMzyme_training_colab.ipynb) | Scientific policy |
| Verified but unfixed workflow issues | [`FOLLOW_UP_TECHNICAL_ISSUES.md`](FOLLOW_UP_TECHNICAL_ISSUES.md) | Status or policy |
| Agent operating behavior | [`AGENTS.md`](../AGENTS.md) | Scientific evidence |
| Completed 2026-08-20 cleanup plan | [`archive/plans/PROJECT_CLEANUP_PLAN_2026-08-20.md`](archive/plans/PROJECT_CLEANUP_PLAN_2026-08-20.md) | Active project authority |

## Execution references

For the end-to-end entry path, start with
[`GETTING_STARTED.md`](GETTING_STARTED.md). For Colab provisioning, stock
PyTorch preservation, browser/CLI same-VM attachment, Drive authorization,
artifact transfer, and teardown, use
[`COLAB_GPU_RUNBOOK.md`](COLAB_GPU_RUNBOOK.md).

### Metal

- Use the metal playbook for exact stage blocks, budgets, seeds, ranges,
  expected artifacts, and gates.
- Use the configuration guide for option meaning, precedence, study reuse, and
  artifact interpretation.
- Read current status before choosing a stage.
- The primary final-test route is currently an unresolved scientific decision;
  the playbook warning links to the dataset record.

### EC

The EC playbook preserves scientifically important historical budgets, ranges,
label-depth progression, and contrastive-loss intent. Some variables and final
workflow values do not match the current notebook. Read its warning and
[`TECH-002`](FOLLOW_UP_TECHNICAL_ISSUES.md#tech-002--ec-playbook-assignments-do-not-match-the-notebook-surface)
before copying affected blocks. Reconciliation is a separate task.

## Evidence storage

- `notebook_outputs/summaries/`: immutable human-readable batch summaries.
- `notebook_outputs/raw/`: copied outputs, exact configs, recovered metadata,
  and historical test-access evidence.
- `archive/`: recoverable historical documents that are no longer current.
- CLEAN/CARE/PinMyMetal preparation `provenance/` directories: tracked
  lightweight copies of generated dataset metadata.

Read summary first, then exact raw/config evidence. If evidence is absent, use
`MISSING — recovery required`; do not reconstruct unsupported details.

## Coordination rules

- Validation evidence owns model and hyperparameter decisions.
- Historical held-out-test values are access evidence only.
- Keep trial IDs namespaced by family, study/batch, and storage identity.
- Update current status, parameter findings, experiment index, and dataset
  authority in the same change when a new result affects them.
- Preserve raw evidence and negative/incomplete findings.
- Treat the current dirty worktree as user-owned.
- Do not move working code/data paths as part of documentation cleanup.
