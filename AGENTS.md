# AGENTS.md — DeepMzyme

## Project purpose

This repository develops **DeepMzyme**, a deep-learning framework for metalloenzyme metal-type prediction and EC/function prediction.

The main goals are:

1. Predict metal type from protein structural pocket graphs and ESMC embeddings/models.
2. Predict enzyme class / EC-level labels from protein structural pocket information and ESMC embeddings/models.
3. Compare model variants fairly using validation metrics and held-out test metrics.
4. Keep the code reproducible, simple to run, and suitable for publication-quality experiments.

---

## Environment

Use this Python interpreter unless explicitly instructed otherwise:

/home/mechti/miniconda3/envs/DeepMzyme/bin/python

Before running scripts, verify the interpreter with:

/home/mechti/miniconda3/envs/DeepMzyme/bin/python -c "import sys; print(sys.executable)"

Do not assume that the shell's default python or python3 points to the correct environment.

---

## Main development guidance

### 1. Follow Plan.md as the main design authority

Plan.md is the main source of truth for the intended architecture, training logic, experiments, and project direction.

When there is a conflict between Plan.md and other scripts, prefer Plan.md, unless the existing code clearly contains newer working logic that should be preserved.

Before making major changes, inspect Plan.md and any script directly related to the requested task.

Do not make large architectural changes that contradict Plan.md unless explicitly requested.

---

### 1b. Read EXPERIMENT_STATUS.md for current progress

If `EXPERIMENT_STATUS.md` is present at the repository root, read it before suggesting
the next experiment, the next baseline, or the next hyperparameter sweep. It is a
lightweight, frequently updated summary of where the project currently stands
(stage, current best validation result, trusted split, next planned experiment,
known caveats, and test-set rules).

How to weigh these sources:

- For design intent and planned project direction, `Plan.md` remains the main
  design authority.
- `EXPERIMENT_STATUS.md` is a mutable current-progress note and is lower
  authority than `Plan.md`.
- Current source code under `src/` and run outputs under
  `DeepMzyme_Data/notebook_outputs/runs/` are evidence of actual implemented
  behavior and actual results, not statements of design intent.
- If code, run outputs, `Plan.md`, or `EXPERIMENT_STATUS.md` conflict, report
  the conflict clearly instead of silently choosing one.

`EXPERIMENT_STATUS.md` is the right place for short-lived experiment-status notes.
Do not move that kind of frequently changing state into `AGENTS.md` or `Plan.md`.

---

### 1c. Key project files and directories

Use this as a navigation map when a task touches the relevant area. Do not read
every file for every small request; inspect the applicable files before making a
claim or change.

#### Primary authority and status

- `Plan.md`: design authority for architecture, experiment policy, validation
  selection, and held-out test rules. Contains the document map for all
  related files.
- `EXPERIMENT_STATUS.md`: current experiment status, selected validation
  anchors, trusted evidence files, caveats, and next planned action.
- `README.md`: public-facing overview, quick-start commands, and split
  reference. Good entry point for understanding what the project does.

#### Notebook workflow and training recipes

- `notebooks/DeepMzyme_training_colab.ipynb`: actual Colab planning, command
  expansion, run execution, skipping, capping, and reporting behavior. The
  single notebook supports all tasks (metal, EC, joint) via `TASK` selection.
- `docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md`: stable notebook workflow and
  option meaning for metal classification. It is not a live results table.
- `docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`: staged, copy-paste-ready notebook
  configuration blocks for metal classification. Use this as the practical
  execution recipe for each training stage (smoke, baseline, Optuna,
  seed-repeat, final test).
- `docs/EC_TRAINING_PIPELINE_PLAYBOOK.md`: same staged structure as the metal
  playbook, covering EC-number classification. Covers EC label depth, group
  weighting, contrastive loss progression, and 200-trial Optuna examples.
- `list_train_commands.md`: baseline-first CLI command examples for
  direct `src/train.py` invocations outside the notebook.

#### Experiment evidence

- `docs/notebook_outputs/README.md`: index for copied notebook-output evidence;
  read this before browsing raw notebook output files.
- `docs/notebook_outputs/summaries/LEADERBOARD.md`: cross-family validation
  snapshot with reliability tiers (5-seed/50-epoch vs. partial); fastest entry
  point for comparing model families at the validation level.
- `docs/notebook_outputs/summaries/`: short human-readable run summaries and
  historical planning notes; read these before raw outputs when tracking
  experiment status.
- `docs/notebook_outputs/raw/`: copied notebook outputs used as portable
  evidence.
- `experiment_notes.md`: early experiment notes from initial learning-rate and
  epoch checks. Historical context only; not a design document.
- `Documenation/`: dated session planning notes (e.g. `14May_26.md`). These
  are informal scratch notes, not authoritative documents. Do not treat them
  as design decisions unless they are reflected in Plan.md.
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
  code. See the caution below before editing.
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

- `DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/`:
  legacy trusted final held-out split. Use for all reportable final evaluations
  unless an experiment explicitly switches to another named split.
- `DeepMzyme_Data/train_and_test_sets_structures_exact_pinmymetal/`:
  exact PinMyMetal split; may contain train/test PDB-ID overlap.
- `DeepMzyme_Data/train_and_test_sets_structures_harsh_pinmymetal/`:
  harsh split where all common PDB IDs are assigned to test.
- `DeepMzyme_Data/train_and_test_sets_structures_common_pdbid_70_30_pinmymetal/`:
  custom 70/30 comparison split; not the main final held-out split.
- `DeepMzyme_Data/esm_embeddings/`: precomputed ESMC residue embeddings.
  Pass this path via `--esm-embeddings-dir` or the notebook `ESM_EMBEDDINGS_DIR`
  variable. Do not commit embeddings to git.
- `DeepMzyme_Data/RING_features/`: precomputed RING interaction edge files.
  Pass this path via `--ring-features-dir` or `RING_FEATURES_DIR`.
- `DeepMzyme_Data/notebook_outputs/runs/`: local run output directories.
  Treat as measured evidence, not design intent.
- `DeepMzyme_Data/DeepMzyme_Colab_Bundles/`: built `.tar.zst` data bundles.

#### Internal and staging

- `internal/codex_suggested/`: staging area for code suggested by AI tools
  that has not yet been reviewed, tested, or merged into `src/`. Do not treat
  this as production code.

**Read by default** (small, high-leverage): `Plan.md`, `EXPERIMENT_STATUS.md`,
and `docs/notebook_outputs/README.md`.

**On-demand only** (large; do not bulk-load):
- `docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md` — only when editing or running
  the Colab metal workflow.
- `docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md` — only when setting up or
  executing a specific metal training stage.
- `docs/EC_TRAINING_PIPELINE_PLAYBOOK.md` — only when setting up or executing
  a specific EC training stage.
- `list_train_commands.md` — only when building or verifying CLI commands.
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

For experiment-status questions, first read `EXPERIMENT_STATUS.md`, then inspect
the specific evidence files it names. For notebook behavior questions, inspect
the notebook itself rather than relying only on the workflow guide.

---

### 2. Be careful with src/model.py

The current src/model.py may contain experimental, non-final, partially inconsistent, or not fully validated code.

Do not assume every implementation detail in src/model.py is final.

When editing src/model.py:

- Prefer additive, configurable changes over hard replacement.
- Keep backward compatibility with existing training scripts when possible.
- If Plan.md and src/model.py disagree, treat this as a design issue and resolve it conservatively. Plan.md should be higher in your hierarchy of decision.

---

### 3. Keep experiments fair and reproducible

When adding or modifying training/evaluation code:

- Use validation metrics for model selection.
- Keep the held-out test set for final reporting only.
- Avoid tuning directly on the test set.
- Save enough metadata to reproduce results, including:
  - model configuration
  - feature set
  - random seed
  - train/validation/test split
  - loss function
  - class weights or sampling strategy
  - learning rate and scheduler
  - checkpoint selection rule

Prefer clear experiment names and structured output directories.

---

### 4. Prefer clean, simple, publication-quality code

Code should be understandable and maintainable.

Prefer:

- explicit configuration over hidden constants
- clear function names
- small helper functions
- readable error messages
- comments only where they clarify non-obvious logic
- minimal duplication

Avoid:

- hardcoded absolute paths unless already part of the project convention
- silent failures
- changing unrelated files
- large rewrites when a focused patch is enough
- adding unnecessary dependencies

---

## Project-specific modeling notes

DeepMzyme may use several information sources, including:

- protein structural pocket graphs
- residue-level geometric features
- ESMC residue embeddings or sequence-derived representations
- optional early, late, or gated ESM fusion
- metal-type classification heads
- EC/function classification heads

When adding model options, make them configurable where reasonable.

For example, prefer command-line/config options such as:

- --use_esm
- --esm_fusion_mode
- --early_esm_dim
- --node_feature_set
- --loss_type
- --metal_loss_weight
- --ec_loss_weight

rather than hardcoding one experimental choice.

---

## Testing and validation

After code changes, run the smallest reasonable checks first.

Syntax checks:

```
/home/mechti/miniconda3/envs/DeepMzyme/bin/python -m py_compile src/model.py
/home/mechti/miniconda3/envs/DeepMzyme/bin/python -m py_compile src/train.py
/home/mechti/miniconda3/envs/DeepMzyme/bin/python -m py_compile src/training/config.py
/home/mechti/miniconda3/envs/DeepMzyme/bin/python -m py_compile src/training/run.py
```

Smoke test (fast, no GPU required):

```
/home/mechti/miniconda3/envs/DeepMzyme/bin/python tests/smoke_checks.py
```

If relevant, run a small smoke test before long training jobs.

Do not launch expensive full training runs unless explicitly requested.

Do not write temporary smoke-test files into `DeepMzyme_Data/` unless this is explicitly needed for the test. Prefer using a temporary directory outside the project data tree, and clean up any temporary files immediately after the test.

---

## Data and paths

Be careful with project-relative paths.

Prefer paths based on the repository root rather than paths relative to the currently running script.

For example, avoid assuming that DeepMzyme_Data/... is relative to src/.

Use robust path construction with pathlib.Path.

---

## Expected behavior

When working on this repository:

1. First inspect the relevant files.
2. Compare the requested change against Plan.md.
3. Make the smallest safe change that satisfies the request.
4. Preserve existing useful options.
5. Run syntax or smoke checks when possible.
6. Clearly summarize what changed and what was not changed.

When editing AGENTS.md itself, briefly summarize the changed sections in the
response so the user can review the policy delta without re-reading the whole
file.
