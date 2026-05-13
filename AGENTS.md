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

Do not assume that the shell’s default python or python3 points to the correct environment.

---

## Main development guidance

### 1. Follow Plan.md as the main design authority

Plan.md is the main source of truth for the intended architecture, training logic, experiments, and project direction.

When there is a conflict between Plan.md and other scripts, prefer Plan.md, unless the existing code clearly contains newer working logic that should be preserved.

Before making major changes, inspect:

- Plan.md

and any script directly related to the requested task.

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

- `Plan.md`: design authority for architecture, experiment policy, validation
  selection, and held-out test rules.
- `EXPERIMENT_STATUS.md`: current experiment status, selected validation
  anchors, trusted evidence files, caveats, and next planned action.
- `docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md`: stable notebook workflow and
  option meaning. It is not a live results table.
- `notebooks/DeepMzyme_training_colab.ipynb`: actual Colab planning, command
  expansion, run execution, skipping, capping, and reporting behavior.
- `src/train.py`: training entry point.
- `src/training/config.py`: CLI and training configuration parsing.
- `src/model.py`: model definitions; see the caution below before editing.
- `src/report_runs.py`: run-summary and comparison-table generation.
- `docs/notebook outputs/`: copied notebook outputs used as portable evidence.
- `DeepMzyme_Data/notebook_outputs/runs/`: local run outputs when present; treat
  these as measured evidence, not design intent.

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

Examples:

/home/mechti/miniconda3/envs/DeepMzyme/bin/python -m py_compile src/model.py
/home/mechti/miniconda3/envs/DeepMzyme/bin/python -m py_compile src/train.py

If relevant, run a small smoke test before long training jobs.

Do not launch expensive full training runs unless explicitly requested.

Do not write temporary smoke-test files into `DeepMzyme_Data/` unless this is explicitly needed for the test.
Prefer using a temporary directory outside the project data tree, and clean up any temporary files immediately after the test.---

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
