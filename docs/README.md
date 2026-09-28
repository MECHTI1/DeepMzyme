# DeepMzyme Documentation Index

Use this page to locate the owner of a fact. Do not copy mutable values between
documents when a link is sufficient.

## Agent default read path

[AGENTS](../AGENTS.md) → [STATUS](../EXPERIMENT_STATUS.md) → the active
campaign README linked from STATUS. Other owners below are read on demand;
read [`Plan.md`](../Plan.md) before experiment design, split, selection-metric or
test-set decisions and [`DATASETS.md`](DATASETS.md) before any test-set use.
[History map](archive/consolidation_2026-09/MAP.md) records the Job B section copies;
[relocations](MOVED.md) distinguishes these from whole-file moves.

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

- [Metal-level prediction and PMM comparison](plans/metal_level_metal_task_compared_PMM_final_plan.md):
  existing fold-0 comparison retained; remaining core fits are deferred for
  future work, and further binding-awareness work is paused.
  The [scope manifest](plans/pmm_core_scope_v2.json) and
  [execution recipe](METAL_TRAINING_PIPELINE_PLAYBOOK.md#active-core-only-continuation)
  define the four/five/six comparison. The additive
  [core execution contract](plans/pmm_core_execution_v1.md) owns replay
  qualification, readiness and the separate core assessment/refit bridge.
  The [five-class screen](plans/pmm_five_class_screen_v1.json) is its bounded
  exploratory execution route.
- [Single-GPU concurrency probe](plans/pmm_gpu_concurrency_probe.md): a bounded
  performance diagnostic; production remains one training process per GPU.
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
| GCP GPU provisioning, stopped-VM recovery, disk restoration, connection and cleanup | [`GCP_GPU_RUNBOOK.md`](GCP_GPU_RUNBOOK.md) | Scientific stage values or model selection |
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

## Repository navigation

Use this as a navigation map when a task touches the relevant area. Do not read
every file for every small request; inspect the applicable files before making a
claim or change.

#### Primary authority and status

- `Plan.md`: design authority for architecture, experiment policy, validation
  selection, and held-out test rules. Contains the document map for all
  related files.
- `docs/plans/metal_level_metal_task_compared_PMM_final_plan.md`: active scope
  of the known-ion PMM comparison. Resolve its scope manifest and the metal
  playbook's guarded entry point before resuming; the frozen low-level runner's
  full grid is not automatically the current authorized queue. Preserve old
  scientific identities and distinguish active, paused and historical arms.
- `docs/README.md`: top-level documentation index for validation/testing,
  notebook, playbook, copied-output documentation, Drive/local output
  handling, and copied-evidence placement rules.
- `EXPERIMENT_STATUS.md`: current experiment status, selected validation
  anchors, trusted evidence files, caveats, and next planned action.
- `docs/DATASETS.md`: authoritative dataset/split/bundle inventory, current
  availability, provenance links, and historical test-use ledger.
- `docs/plans/pmm_core_scope_v2.json`: current PMM ordinary-readout four/five/six
  scope; use the metal playbook for its bounded five-class fold-0 recipe.
  Preserve the v1 manifest and historical awareness screen.
- `docs/PARAMETER_FINDINGS.md`: validation/HPO findings with evidence grades;
  historical test metrics are excluded from parameter conclusions.
- `docs/FOLLOW_UP_TECHNICAL_ISSUES.md`: verified implementation/documentation
  problems that remain deliberately unfixed pending separate authorization.
- `README.md`: public-facing overview and minimal quick start. Good entry point
  for understanding what the project does; it does not own live defaults.
- `docs/GETTING_STARTED.md`: executable orientation, current local-environment
  limits, first checks, and the shortest navigation route from checkout to the
  correct stage/evidence owner.
- `docs/COLAB_GPU_RUNBOOK.md`: browser/CLI same-VM connection, Colab
  PyTorch-preserving dependency installation, CUDA architecture preflight,
  Drive authorization boundary, artifact transfer, and teardown.
  Read it before Colab GPU work: it records verified hurdle fixes and an
  optional Python 3.12 ESMC preparation route for Python 3.13 runtimes.
  That route is a tested convenience, not a requirement; skip the extra
  environment when the runtime already uses compatible Python 3.12.

#### Notebook workflow and training recipes

- `notebooks/DeepMzyme_training_colab.ipynb`: actual Colab planning, command
  expansion, run execution, skipping, capping, and reporting behavior. The
  single notebook supports all tasks (metal, EC, joint) via `TASK` selection.
- `docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md`: stable notebook workflow and
  option meaning for metal classification, including notebook execution order,
  Optuna behavior, and safety policy. It is not a live results table.
- `docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`: staged, copy-paste-ready notebook
  configuration blocks for metal classification. Use this as the practical
  execution recipe and exact parameter source for each training stage (smoke,
  baseline, Optuna, Stage 6 grouped-fold confirmation, final test). For
  historical G4-class GPU planning, this records the historical serious/custom Optuna
  budgets and search spaces.
- `docs/EC_TRAINING_PIPELINE_PLAYBOOK.md`: same staged structure as the metal
  playbook, covering EC-number classification. Covers EC label depth, group
  weighting, contrastive loss progression, and 200-trial Optuna examples.
  Read its compatibility warning before using affected HPO or final-test
  blocks; reconciliation with the current notebook remains an open technical
  task.
- `docs/archive/workflows/list_train_commands_legacy.md`: historical direct
  `src/train.py` command examples. Use `src/train.py --help` plus current
  playbooks for live execution.

#### Experiment evidence

- `docs/notebook_outputs/README.md`: authoritative experiment-batch index and
  copied-evidence contract; read this before browsing raw files.
- `docs/notebook_outputs/summaries/`: short human-readable run summaries and
  historical planning notes; read these before raw outputs when tracking
  experiment status.
- `docs/notebook_outputs/raw/`: copied notebook outputs used as portable
  evidence.
- `docs/archive/experiments/experiment_notes_legacy.md`: intact early
  learning-rate/epoch notes and historical test-access evidence; not a design
  or current parameter-selection document.
- `docs/archive/`: historical command references, audits, session notes,
  experiment snapshots, and changelogs. They remain recoverable but are not
  current authority.
- `DeepMzyme_Data/notebook_outputs/runs/`: local run outputs when present;
  treat these as measured evidence, not design intent.

#### Training source code

- `src/train.py`: main training entry point for all tasks (`--task metal`,
  `--task ec`, `--task joint`). Delegates to `src/training/config.py` and
  `src/training/task_entrypoint.py`.
- `src/train_metal.py`: thin task-specific wrapper that invokes the metal
  training path directly, bypassing joint-task dispatch.
- `src/train_ec.py`: thin task-specific wrapper that invokes the EC training
  path directly.
- `src/training/config.py`: CLI and training configuration parsing. The
  authoritative list of all CLI flags and their defaults.
- `src/training/task_entrypoint.py`: dispatches training to the correct task
  head after configuration is parsed.
- `src/training/run.py`: main training loop.
- `src/training/loop.py`: epoch-level training and validation logic.
- `src/training/data.py`: dataset loading and preparation.
- `src/training/splits.py`: train/validation split logic.
- `src/training/labels.py`: label extraction and EC depth handling.
- `src/training/preflight.py`: pre-training validation checks (paths, splits,
  feature availability, ESM/RING coverage).
- `src/training/defaults.py`: default values for training configuration.

#### Model source code

- `src/model.py`: model definitions; may contain experimental or non-final
  code. See the `src/model.py` caution in
  [AGENTS](../AGENTS.md#development-and-verification) before editing.
- `src/model_variants/factory.py`: model instantiation factory.
- `src/model_variants/models.py`: concrete model variant definitions.

#### Graph and feature extraction

- `src/graph/construction.py`: pocket graph construction from structure files.
- `src/graph/ring_edges.py`: RING interaction edge loading and generation.
- `src/graph/shell_roles.py`: first/second shell residue role assignment.
- `src/featurization.py`: residue-level featurization pipeline.
- `src/feature_extraction/`: PROPKA, physicochemical, and external feature
  extraction modules.
- `src/embed_helpers/esmc.py`: ESMC embedding generation and loading.
- `src/embed_helpers/Interaction_edge.py`: interaction edge helpers.
- `src/label_schemes.py`: metal and EC label scheme definitions.
- `src/data_structures.py`: shared data container types.

#### Data preparation utilities

- `src/build_dataset_csv.py`: builds site-level MAHOMES-format summary CSVs
  from PDB structures and PinMyMetal labels. Run this before training when
  creating a new split.
- `src/build_colab_bundle.py`: packs a Colab-ready `.tar.zst` data bundle from
  a specified split directory. Run this to produce the bundle uploaded to
  HuggingFace or used via Drive.
- `src/structure_store.py`: manifest schema and resolver for the shared
  content-addressed structure store.
- `src/manage_structure_store.py`: audits, migrates, and verifies structure
  storage without changing scientific split membership.
- `prepare_training_and_test_set/`: original split preparation scripts.
  Downloads PDB structures, creates non-redundant chain files, and runs MAHOMES
  activation to produce site-level summary CSVs. Scripts are named
  `step1a_...`, `step1b_...`, etc. for sequential execution.
  `prepare_training_and_test_set/pinmymetal_files` contains the original
  PinMyMetal train/test membership files.

#### Reporting

- `src/report_runs.py`: run-summary and comparison-table generation. Summarizes
  multiple run directories into a single CSV. Used by the notebook summary cell
  and can be run standalone.

#### Data directories

- `DeepMzyme_Data/structure_store/`: one object per distinct structure-file
  byte content, keyed by SHA-256. Split and CLEAN shared-structure directories
  use `structure_manifest.csv` to reference these objects; see
  `docs/STRUCTURE_STORE.md`.

- `DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/`:
  historical non-overlap split path. It is present locally in manifest-backed
  form as of 2026-09-14, and its test was evaluated in seven early runs plus six
  reports on 2026-09-18 ([ledger](DATASETS.md#test-use-ledger)), so it
  is not pristine.
  Do not recommend final reporting until the primary final-test route is
  resolved scientifically; see `docs/DATASETS.md`.
- `DeepMzyme_Data/train_and_test_sets_structures_exact_pinmymetal/`:
  exact PinMyMetal train/test PDB-ID membership for available supported
  structures; may contain train/test PDB-ID overlap.
- `DeepMzyme_Data/train_and_test_sets_structures_harsh_pinmymetal/`:
  harsh split where all common PDB IDs are assigned to test.
- `DeepMzyme_Data/train_and_test_sets_structures_common_pdbid_70_30_pinmymetal/`:
  custom comparison split where train-only PDB IDs stay train, test-only PDB
  IDs stay test, and only common exact-split PDB IDs are assigned as whole
  groups at a 70/30 train/test ratio; not the main final held-out split.
- `DeepMzyme_Data/esm_embeddings/`: precomputed ESMC residue embeddings.
  Pass this path via `--esm-embeddings-dir` or the notebook `ESM_EMBEDDINGS_DIR`
  variable. Do not commit embeddings to git.
- `DeepMzyme_Data/RING_features/`: precomputed RING interaction edge files.
  Pass this path via `--ring-features-dir` or `RING_FEATURES_DIR`.
- `DeepMzyme_Data/notebook_outputs/runs/`: local run output directories.
  Treat as measured evidence, not design intent.
- `DeepMzyme_Data/DeepMzyme_Colab_Bundles/`: built `.tar.zst` data bundles.

Current materialization and bundle inclusion can change. Verify them in
`docs/DATASETS.md` and the filesystem instead of assuming every named path
above exists.

#### Internal and staging

- No `internal/` workflow is part of the active pipeline. If an
  `internal/codex_suggested/` directory is reintroduced later, treat it as
  unreviewed staging material only, not production code.


**On-demand only** (large; do not bulk-load):
- `docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md` — only when editing or running
  the Colab metal workflow.
- `docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md` — only when setting up or
  executing a specific metal training stage.
- `docs/EC_TRAINING_PIPELINE_PLAYBOOK.md` — only when setting up or executing
  a specific EC training stage.
- `docs/archive/workflows/list_train_commands_legacy.md` — historical context
  only when recovering an old direct CLI invocation.
- Files under `docs/notebook_outputs/raw/` — only when a summary cites the
  file or when exact logs / run commands are needed.
- `notebooks/DeepMzyme_training_colab.ipynb` — only for notebook-behavior
  questions.
- `src/training/config.py` — only when checking the exact CLI flag name or
  default value.
- `DeepMzyme_Data/notebook_outputs/runs/*` — only when `EXPERIMENT_STATUS.md`
  names a specific run.

Read individual files under `docs/notebook_outputs/summaries/` by name; do not
bulk-load all summaries.

## Documentation coordination protocol

For future documentation edits:

- Identify the full coupled document set before editing, including stable docs,
  mutable status notes, run-evidence indexes, notebook context, and CLI examples
  that restate the same facts.
- Assign each fact to one owning document before changing it. Keep exact metal
  executable stage values in `docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`, EC
  equivalents in `docs/EC_TRAINING_PIPELINE_PLAYBOOK.md`, live notebook defaults
  in `notebooks/DeepMzyme_training_colab.ipynb`, notebook option meanings in
  `docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md`, research policy in `Plan.md`,
  mutable status in `EXPERIMENT_STATUS.md`, datasets/test-use in
  `docs/DATASETS.md`, validation/HPO findings in
  `docs/PARAMETER_FINDINGS.md`, and unresolved behavior problems in
  `docs/FOLLOW_UP_TECHNICAL_ISSUES.md`.
- Keep durable metal-EC auxiliary policy and phase order in `Plan.md`. Put an
  exact EC-primary auxiliary recipe in the EC playbook only after its split and
  leakage protocol is certified; put the optional reverse-direction recipe in
  the applicable playbook if that experiment is later approved.
- Update or re-point every cross-reference in the same change set instead of
  leaving duplicated stale text behind.
- Avoid copying current anchors, run IDs, transient trial numbers, local disk
  state, or mutable best-result notes into stable docs.
- Run a consistency sweep after edits for stage names, split policy, selection
  metrics, held-out-test rules, Stage 6/7 safeguards, seed/bootstrap/pruning
  wording, and notebook option names.
- Report unresolved conflicts explicitly instead of silently choosing one source
  when the repository evidence is insufficient.
