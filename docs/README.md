# DeepMzyme Documentation Index

Use this page to locate the owner of a fact. Do not copy mutable values between
documents when a link is sufficient.

## Start here

1. [`GETTING_STARTED.md`](GETTING_STARTED.md) — execution paths, environment
   limits, first checks, and repository navigation.
2. [`BACKUP_AND_DATA_INVENTORY.md`](BACKUP_AND_DATA_INVENTORY.md) — authoritative
   cloud backup locations, Hugging Face bundles, model checkpoints, and disk inventory.
3. [`EXPERIMENT_STATUS.md`](../EXPERIMENT_STATUS.md) — current objective,
   anchors, blockers, and next actions.
4. [`DATASETS.md`](DATASETS.md) — datasets, splits, Hugging Face files,
   bundles, provenance, and
   historical test-use record.
5. [`PARAMETER_FINDINGS.md`](PARAMETER_FINDINGS.md) — validation/HPO findings
   with evidence grades.
6. [`notebook_outputs/README.md`](notebook_outputs/README.md) — experiment-batch
   index and links to summaries/configs/raw evidence.

Scientific policy is in [`Plan.md`](../Plan.md).
Its [metal example terminology](../Plan.md#metal-example-terminology) defines
clustered pockets, ion-centered examples, record IDs, and validation groups.

## Task plans and reviews

Keep current task plans in `docs/plans/`. The two final plans below supersede
the four Antigravity/Opus proposals and link to their originals preserved in
Git commit `63d4cd52bf98b2448d42f4498364828431c962eb`:

- [Metal-level prediction and PMM comparison](plans/metal_level_metal_task_compared_PMM_final_plan.md).
- [Pocket-level EC prediction and CLEAN30 comparison](plans/pocket_level_EC_task_compare_clean30_final_plan.md).

These documents describe planned work; they do not replace `Plan.md` as the
scientific authority or establish implemented or evaluated behavior.

## Ownership

| Information | Authority | Not its role |
|---|---|---|
| Public overview and minimal quick start | [`README.md`](../README.md) | Live defaults, status, or experiment history |
| Cloud backup locations, Hugging Face bundles, checkpoint releases, and disk inventory | [`BACKUP_AND_DATA_INVENTORY.md`](BACKUP_AND_DATA_INVENTORY.md) | Mutable experiment progress |
| Executable orientation and local setup limits | [`GETTING_STARTED.md`](GETTING_STARTED.md) | Exact experiment budgets or mutable results |
| Locked Linux environment and Colab overlay boundary | [`../requirements/README.md`](../requirements/README.md) | Scientific stage policy |
| Benchmark files, schemas, commands, and interpretation | [`../bench/README.md`](../bench/README.md) | Model-quality evidence |
| Reproducibility remediation decisions and verification | [`REPRODUCIBILITY_REMEDIATION_PLAN.md`](REPRODUCIBILITY_REMEDIATION_PLAN.md) | Scientific stage policy |
| Optional PMM source-site benchmark implementation plan | [`VERY_EXACT_PMM_SETS_PLAN.md`](VERY_EXACT_PMM_SETS_PLAN.md) | Implemented dataset or validated comparison |
| PinMyMetal exact 5-fold CV reproducibility playbook | [`EXACT_PINMYMETAL_5FOLD_CV_REPRODUCIBILITY.md`](EXACT_PINMYMETAL_5FOLD_CV_REPRODUCIBILITY.md) | Multi-task or unrelated split policies |
| Zenodo PinMyMetal exact ion-level 5-fold CV reproducibility playbook | [`ZENODO_PINMYMETAL_EXACT_ION_LEVEL_REPRODUCIBILITY.md`](ZENODO_PINMYMETAL_EXACT_ION_LEVEL_REPRODUCIBILITY.md) | Non-destructive ion-level replication & benchmark guide |
| Generalized 5-fold CV runner guide (arbitrary splits, ion & pocket levels) | [`GENERALIZED_METAL_5FOLD_CV_GUIDE.md`](GENERALIZED_METAL_5FOLD_CV_GUIDE.md) | Single dataset-specific fixed scripts |
| PinMyMetal 5-fold three-architecture benchmark summary | [`notebook_outputs/summaries/summary_pinmymetal_5fold_three_models_20260923.md`](notebook_outputs/summaries/summary_pinmymetal_5fold_three_models_20260923.md) | Unverified pilot results |
| Colab browser/CLI connection and environment procedure | [`COLAB_GPU_RUNBOOK.md`](COLAB_GPU_RUNBOOK.md) | Scientific stage values or model selection |
| Required agent GPU entry point, bounded routing and optional monitor delegation | [`gpu-use-skill`](../.agents/skills/gpu-use-skill/SKILL.md) | New spending permission, a second controller or guaranteed GPU availability |
| GCP GPU VM provisioning, PyCharm remote setup, and batch execution | [`GCP_GPU_RUNBOOK.md`](GCP_GPU_RUNBOOK.md) | Scientific stage values or model selection |
| PMM application of GPU routing, measured admission and persistence | [`GPU_EXECUTION_CASCADE_PLAYBOOK.md`](GPU_EXECUTION_CASCADE_PLAYBOOK.md) | A second provisioning controller or a hardware speed guarantee |
| GPU orchestration rollout, preparation and performance gates | [`GPU_RUNTIME_EFFICIENCY_PLAN.md`](GPU_RUNTIME_EFFICIENCY_PLAN.md) | Launch authorization or new scientific recipes |
| Sequence-remoteness protocol, validation replay and support gates | [`REMOTE_HOMOLOGY_ADDENDUM.md`](REMOTE_HOMOLOGY_ADDENDUM.md) | Homology absence, new training authorization or model promotion |
| Current status, implementation/evidence state, and next action | [`EXPERIMENT_STATUS.md`](../EXPERIMENT_STATUS.md) | Long chronological diary |
| Scientific/design/test policy, primary task boundaries, and metal-EC auxiliary strategy | [`Plan.md`](../Plan.md) | Dataset inventory or copied stage blocks |
| Dataset identity, readiness, bundles, test use | [`DATASETS.md`](DATASETS.md) | Preparation procedure |
| Structure-store layout, manifests, deduplication audit, and maintenance | [`STRUCTURE_STORE.md`](STRUCTURE_STORE.md) | Scientific split policy |
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
[`GETTING_STARTED.md`](GETTING_STARTED.md). Agents must first use the project
[`gpu-use-skill`](../.agents/skills/gpu-use-skill/SKILL.md) for GPU operations.
For Colab provisioning, stock
PyTorch preservation, browser/CLI same-VM attachment, Drive authorization,
artifact transfer, and teardown, use
[`COLAB_GPU_RUNBOOK.md`](COLAB_GPU_RUNBOOK.md). For dedicated GCP GPU VM
execution with PyCharm remote development and gross commercial cost bounds, use
[`GCP_GPU_RUNBOOK.md`](GCP_GPU_RUNBOOK.md).

For the planned continuous-queue, transfer and preparation improvements, read
[`GPU_RUNTIME_EFFICIENCY_PLAN.md`](GPU_RUNTIME_EFFICIENCY_PLAN.md). It includes
measured overhead, implementation order, failure handling and performance gates;
current pause/resume authority remains in the current-status document.
