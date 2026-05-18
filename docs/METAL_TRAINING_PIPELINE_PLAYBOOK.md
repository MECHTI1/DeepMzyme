# Metal Training Pipeline Playbook

This playbook is the practical, notebook-ready pipeline for DeepMzyme metal
classification. It complements `Plan.md`, which remains the high-level research
and design authority. Current best validation evidence belongs in
`EXPERIMENT_STATUS.md` and `docs/notebook_outputs/`, not in this stable
playbook.

## Pipeline Overview At A Glance

This playbook is the operational pipeline for
`notebooks/DeepMzyme_training_colab.ipynb`. Each stage is a self-contained
notebook configuration block; you paste one block at a time into the Main
configuration cell. Stages 1, 3 are smoke checks. Stages 2, 4, 5, 6 are
validation-only. Stage 7 is the only stage that touches the held-out test set,
and it is run exactly once for the final validation-selected configuration.

| Stage | Purpose | G4 wall-time (approx.) | Decision produced | Block |
| --- | --- | --- | --- | --- |
| 0 | Setup, Drive, data bundle, RING+ESM preflight | 10-20 min | "Environment ready" | Stage 0 |
| 1 | 1-epoch smoke (Only-GVP, RING-on) | 5-15 min | "Notebook & data plumbing OK" | Stage 1 |
| 2A | Only-GVP validation baseline (50 ep x 3 LR x 3 seeds) | 6-10 h | Only-GVP validation anchor | Stage 2A |
| 2B | Baseline model comparison (Only-GVP, Only-ESM, GVP+late fusion) | 8-14 h | First cross-family ranking | Stage 2B |
| 3 | Debug Optuna (4 x 3 ep) | 20-40 min | "Optuna plumbing OK" | Stage 3 |
| 4 | Medium Optuna inside one preset (64 x 35 ep) | 18-28 h | Top 5 validation configs | Stage 4 |
| 5A-5F | Large 200-trial Optuna per model family | 36-60 h each | Top 3 validation configs | Stage 5 |
| 5G | RING radius-only ablation | optional 6-10 h | RING value evidence | Stage 5G |
| 6 | Top-K x 5-seed validation confirmation (50 ep) | 15-25 h | Final validation-selected config | Stage 6 |
| 7 | Held-out test (single shot) | 20-60 min | Final reportable test metrics | Stage 7 |

All configuration blocks below use variables that exist in
`notebooks/DeepMzyme_training_colab.ipynb` as of this repository state. To use a
block, edit the notebook's **Main configuration** cell directly or paste the
block at the end of that cell before running **Build central CONFIG
dictionary**.

## How To Use This Playbook

Use exactly one stage block at a time. After editing the notebook's **Main
configuration** cell, run the CONFIG/planning cells and inspect the resolved
commands before setting `LAUNCH_PLANNED_TRAINING_RUNS = True`.

Notebook execution order:

1. Setup/install and data-source cells.
2. Main configuration cell with one block from this playbook.
3. Build central `CONFIG`.
4. Planning/preflight cells.
5. Optional training execution cell.
6. Summarize/report cell for the current `RUN_BATCH_ID`.
7. For final testing only: select final run, preview final held-out test, then
   launch once.

For all comparison, HPO, and seed-repeat stages:

- Keep `INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False`.
- Keep `VAL_FRACTION = 0.15` and `SPLIT_BY = "pdbid"` unless a new split
  experiment is explicitly being labeled.
- Use validation metrics, usually `val_metal_balanced_acc`, for checkpoint,
  hyperparameter, architecture, and fusion decisions.
- Do not run the optional final held-out test cell until the final
  validation-selected configuration is fixed.
- If the user asks for a new check, new run, or fresh Optuna sweep without
  explicitly asking to rely on previous raws/results, use previous notebook
  outputs only as context and safety checks. Prefer the broadest sensible
  validation-only Optuna search within the selected `MODEL_PRESET`, with
  common-sense runtime and feature-availability limits.
- If the user explicitly asks to rely on previous running/results/raws, inspect
  the relevant copied evidence and use it to narrow, continue, or repeat that
  prior configuration.

## Common Defaults

Use these shared defaults unless a stage overrides them.

```python
TASK = "metal"
DATASET_NAME = "train_and_test_sets_structures_non_overlapped_pinmymetal"
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_metal_balanced_acc"
OPTUNA_SELECTION_METRIC = "val_metal_balanced_acc"
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False

RING_EDGE_MODE = "with_ring"
REQUIRE_RING_EDGES = False
PREPARE_MISSING_RING_EDGES = True

ALLOW_MISSING_EXTERNAL_FEATURES = False
PREPARE_MISSING_EXTERNAL_FEATURES = False
EXTERNAL_FEATURES_ROOT_DIR = ""

METAL_CLASS_WEIGHT_MODES_CSV = "inverse_frequency"
METAL_LOSS_FUNCTION = "cross_entropy"
METAL_LABEL_SMOOTHING = 0.0
BALANCE_METAL_SITE_SYMBOLS = False

COPY_OUTPUTS_TO_DRIVE = True
METAL_REPORT_VIEW = "both"

DEVICE = "cuda"
SKIP_EXISTING_RUNS = True
STOP_ON_FIRST_FAILURE = False
ALLOW_MODEL_PRESET_MISMATCH = False
ALLOW_SINGLE_MODE_TO_TRUNCATE_COMPARISON = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False

OPTUNA_DIRECTION = "maximize"
OPTUNA_TPE_MULTIVARIATE = True
OPTUNA_TPE_GROUP = True
OPTUNA_TPE_CONSTANT_LIAR = False
OPTUNA_AUTO_CONFIGURE_BUDGET = False
OPTUNA_USE_PRUNING = False
OPTUNA_PRUNER_TYPE = "none"
OPTUNA_TIMEOUT_MINUTES = 0
```

## G4-Class Optuna Policy

This project runs on a G4-class GPU (16 GB VRAM, persistent runtime). All
serious Optuna stages must use:

- `OPTUNA_INTENSITY = "custom"` - never rely on `first_useful`/`serious`
  notebook presets for reportable HPO.
- `OPTUNA_TPE_MULTIVARIATE = True`, `OPTUNA_TPE_GROUP = True`,
  `OPTUNA_TPE_CONSTANT_LIAR = False`.
- `OPTUNA_AUTO_CONFIGURE_BUDGET = False` (explicit budgets only).
- `OPTUNA_USE_PRUNING = False` (subprocess training cannot report intermediate
  metrics yet).
- Persistent SQLite storage in Drive:
  `sqlite:////content/drive/MyDrive/DeepMzyme/optuna/<study_name>.db`.
- Startup trials: `OPTUNA_N_STARTUP_TRIALS = max(20, 0.2 x N_OPTUNA_TRIALS)`.
  For 200-trial searches this is 40; for 64-trial it is 20.
- `OPTUNA_SPLIT_SEED = 42` for every study (so seed-repeat at Stage 6 is
  meaningful).
- `OPTUNA_SELECTION_METRIC = "val_metal_balanced_acc"`,
  `OPTUNA_DIRECTION = "maximize"`.

Forbidden in serious stages:

- Mixing `MODEL_PRESET` values inside one study (Optuna optimizes one family at
  a time).
- Held-out test evaluation inside trials
  (`INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False`).
- Letting `EPOCHS <= 3` reach Stage 4/5 (the short-training guard will block
  this; do not override).

Batch-size policy for serious stages:

- Use `4` only for smoke/debug runs or when a memory failure forces it.
- Use `8,16` as the default serious validation-only Optuna batch-size search
  space. This keeps the current validated `batch_size=8` anchor in scope while
  testing whether `16` improves minority-class stability and GPU utilization.
- Do not include `32` in the main Stage 4/5 search space. Test `32` only as an
  explicitly labeled validation-only ablation after the `8,16` search is
  stable, because it cuts optimizer updates per epoch substantially.

Recommended G4 budgets (canonical):

| Stage | `N_OPTUNA_TRIALS` | `MAX_EPOCHS_PER_TRIAL` | `OPTUNA_N_STARTUP_TRIALS` |
| --- | --- | --- | --- |
| Stage 3 (debug) | 4 | 3 | 4 |
| Stage 4 (medium per family) | 64 | 35 | 20 |
| Stage 5A (Only-GVP) | 200 | 50 | 40 |
| Stage 5B (Only-ESM) | 120 | 50 | 30 |
| Stage 5C (GVP+late) | 200 | 50 | 40 |
| Stage 5D (GVP+node-late) | 200 | 50 | 40 |
| Stage 5E (GVP+hybrid) | 200 | 50 | 40 |
| Stage 5F (GVP+cross-attn) | 120 | 50 | 30 |

## Canonical G4 Metal Training Route

Use this route when starting a clean, serious metal-classification campaign in
`notebooks/DeepMzyme_training_colab.ipynb`.

### Required order

1. Stage 0 - environment and data readiness.
2. Stage 1 - 1-epoch smoke test.
3. Stage 2A - Only-GVP validation baseline.
4. Stage 2B - ESM-ready baseline comparison, only after ESM coverage is valid.
5. Stage 3 - debug Optuna plumbing check.
6. Stage 5A - 200-trial Only-GVP capacity search.
7. Stage 5B - 120-trial Only-ESM search, only after ESM coverage is valid.
8. Stage 5C - 200-trial GVP + late-fusion search, only after simple baselines
   justify ESM fusion.
9. Stage 6 - top-K x 5-seed validation for the best candidates.
10. Stage 7 - one held-out test evaluation for the final validation-selected
    configuration.

Stage 4 is optional on a G4 GPU. Use it when you want a medium 64-trial check
before committing to a 120/200-trial search. For a serious fresh search on G4,
Stage 5 is preferred after Stage 3 passes.

### Advanced fusion gate

Do not launch Stage 5D, Stage 5E, or Stage 5F until Stage 5C has produced a
Stage 6 seed-repeat candidate that beats the Stage 2A Only-GVP anchor by at
least `0.01` mean `val_metal_balanced_acc` across the 5-seed list.

If this gate is not passed, stop advanced fusion escalation and continue with
the best validated simpler family.

### Final-selection rule

The final selected model must come from Stage 6 seed-repeat validation, not from
a single Optuna trial. Select by mean `val_metal_balanced_acc`, then inspect
standard deviation, minimum seed result, `val_metal_min_recall`,
`val_metal_macro_f1`, and collapsed-4 balanced accuracy as diagnostics.

The held-out test is used only once, in Stage 7, after this validation-based
selection is frozen.

## Optuna Study Naming And Storage

Study naming: `metal_<preset_slug>_<size>_<purpose>`, for example
`metal_only_gvp_200_capacity` or `metal_late_fusion_200_controlled`. Always use
lowercase, underscore-separated names.

Storage path template:
`sqlite:////content/drive/MyDrive/DeepMzyme/optuna/<study_name>.db`. Use one
file per study. Never share a study DB across different `MODEL_PRESET` values.

Resumption rule: re-running the notebook with the same `OPTUNA_STUDY_NAME` and
storage URL appends trials. To start fresh, change the study name; do not delete
the `.db` unless you mean to discard history.

If a run uses `Only-ESM` or any GVP + ESM fusion preset, set
`ESM_EMBEDDINGS_DIR` to the embeddings folder or set
`PREPARE_MISSING_ESM_EMBEDDINGS = True` deliberately. Do not use
`ALLOW_MISSING_ESM_EMBEDDINGS = True` for reportable runs.

## Stage 0 - Environment And Data Readiness

Purpose: confirm Drive is mounted, the bundle is present, RING/ESM/external
features coverage is acceptable, and `RUNS_DIR` resolves under Drive.

When to use it: first cell pass in a fresh Colab runtime, or after switching
data bundles.

Configuration block (paste into Main configuration cell):

```python
TASK = "metal"
RUN_MODE = "single"
RECOMMENDED_RUN_SET = "only_gvp_smoke"
MODEL_PRESET = "Only-GVP"
RUN_BATCH_ID = "stage0_environment_check"
SUMMARY_BASENAME = "stage0_environment_check"
RUN_NAME_PREFIX = "stage0_env"

EPOCHS = 1
BATCH_SIZES_CSV = "4"
LEARNING_RATES_CSV = "3e-5"
WEIGHT_DECAYS_CSV = "1e-4"
SEEDS_CSV = "42"

DATASET_NAME = "train_and_test_sets_structures_non_overlapped_pinmymetal"
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_metal_balanced_acc"

RING_EDGE_MODE = "with_ring"
REQUIRE_RING_EDGES = False
PREPARE_MISSING_RING_EDGES = True
ESM_EMBEDDINGS_DIR = ""
ALLOW_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_ESM_EMBEDDINGS = False
ALLOW_MISSING_EXTERNAL_FEATURES = False

INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
LAUNCH_PLANNED_TRAINING_RUNS = False   # planning only; do NOT train at Stage 0
```

Success criteria for Stage 0:

- `RUNS_DIR` resolves under `<DRIVE_ROOT>/notebook_outputs/runs`.
- Planning cell prints RING coverage >= 95% (or generation will run).
- Planning cell prints external-features coverage = 100%.
- No model preset mismatch warnings.

### Decision gate after Stage 0

Proceed to Stage 1 only if:

- All four Stage 0 success criteria are met.
- No held-out test files were created.
- `val_metal_balanced_acc` is the selection metric on the planned run.
- Diagnostics report every class present in both train and validation splits.

If gate fails: fix paths, Drive mounting, bundle selection, RING coverage, or
external-feature coverage before any training.

## Stage 1 - Smoke And Readiness Check

Purpose: verify Colab setup, data paths, CSV detection, graph construction, and
the training command path.

When to use it: first run in a fresh Colab/runtime, after changing the notebook,
or after changing data bundle paths.

Expected scale/runtime: smoke/debug, minutes.

Notebook configuration block:

```python
TASK = "metal"
RUN_MODE = "single"
RECOMMENDED_RUN_SET = "only_gvp_smoke"
MODEL_PRESET = "Only-GVP"
RUN_BATCH_ID = "metal_smoke_readiness"
SUMMARY_BASENAME = "metal_smoke_readiness"
RUN_NAME_PREFIX = "metal_smoke"

EPOCHS = 1
BATCH_SIZES_CSV = "4"
LEARNING_RATES_CSV = "3e-5"
WEIGHT_DECAYS_CSV = "1e-4"
SEEDS_CSV = "42"
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_metal_balanced_acc"

RING_EDGE_MODE = "with_ring"
ESM_EMBEDDINGS_DIR = ""
ALLOW_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_ESM_EMBEDDINGS = False
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
MAX_CONFIGURATION_RUNS = 1
```

In the **Optional training execution** cell:

```python
LAUNCH_PLANNED_TRAINING_RUNS = True
```

Expected outputs/files:

- `<RUNS_DIR>/<SUMMARY_BASENAME>_planned_runs.csv`
- `<RUNS_DIR>/<SUMMARY_BASENAME>_planned_run_dictionary.json`
- One run directory under `<RUNS_DIR>/`
- `run_metadata.json`, `run_config.json`, `split_diagnostics.json`
- No `test_report.json`

Success criteria:

- Planning prints one runnable Only-GVP command.
- Training completes without missing-path, split, feature, or CLI errors.
- Train and validation metal diagnostics are printed.
- No held-out test report is produced.

### Decision gate after Stage 1

Proceed to Stage 2A only if:

- The Stage 1 success criteria are met.
- No held-out test files were created.
- `val_metal_balanced_acc` is the selection metric on all completed runs.
- Diagnostics report every class present in both train and validation splits.

If gate fails: return to Stage 0 and fix paths, bundle setup, RING executable
configuration, structure parsing, ESM coverage, or feature availability before
running real comparisons. Ignore the 1-epoch metric as model-quality evidence.

## Stage 2 - Baseline Model Comparison

Purpose: establish clean validation baselines before adding complex fusion or
large HPO.

When to use it: after smoke passes and before Optuna or advanced fusion. If ESM
embeddings are not ready, run the Only-GVP block first. Once ESM embeddings are
ready, run the ESM-ready baseline block.

Expected scale/runtime: medium validation run, hours. Runtime depends on GPU,
ESM coverage, and whether embeddings must be prepared.

### 2A - Structure-Only Only-GVP Baseline

Notebook configuration block:

```python
TASK = "metal"
RUN_MODE = "manual_configurations"
RECOMMENDED_RUN_SET = "only_gvp_broad_comparison"
MODEL_PRESET = "Only-GVP"
RUN_BATCH_ID = "metal_only_gvp_baseline_lr_seed"
SUMMARY_BASENAME = "metal_only_gvp_baseline_lr_seed"
RUN_NAME_PREFIX = "metal_only_gvp_baseline"

EPOCHS = 50
BATCH_SIZES_CSV = "8"
LEARNING_RATES_CSV = "3e-5,1e-4,3e-4"
WEIGHT_DECAYS_CSV = "1e-4"
SEEDS_CSV = "42,43,44"
MAX_CONFIGURATION_RUNS = 9

HIDDEN_S_VALUES_CSV = "128"
HIDDEN_V_VALUES_CSV = "16"
EDGE_HIDDEN_VALUES_CSV = "64"
GVP_LAYERS_VALUES_CSV = "4"
HEAD_MLP_LAYERS_VALUES_CSV = "2"
EDGE_RADIUS_VALUES_CSV = "8.0"

RING_EDGE_MODE = "with_ring"
ESM_EMBEDDINGS_DIR = ""
ALLOW_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_ESM_EMBEDDINGS = False
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

### Decision gate after Stage 2A

Proceed to Stage 2B or Stage 4 only if:

- The Only-GVP validation baseline completes all planned runs or all failures
  are understood and documented.
- No held-out test files were created.
- `val_metal_balanced_acc` is the selection metric on all completed runs.
- Diagnostics report every class present in both train and validation splits.
- Seed variance is acceptable: if seed standard deviation or high-low spread
  suggests `val_metal_balanced_acc` variance above 0.04, rerun with 5 seeds
  before Stage 2B or Stage 4.

If gate fails: rerun Stage 2A with 5 seeds before any Stage 2B/4 decision, or
return to Stage 0/1 if the failure is path, feature, or split related.

### 2B - ESM-Ready Baseline Comparison

Run this only after ESM embeddings are available or after you intentionally allow
the notebook to prepare missing ESM embeddings.

Notebook configuration block:

```python
TASK = "metal"
RUN_MODE = "manual_configurations"
RECOMMENDED_RUN_SET = "baseline_model_comparison"
# MODEL_PRESET is overridden by baseline_model_comparison (runs Only-GVP, Only-ESM, GVP + late fusion)
RUN_BATCH_ID = "metal_baseline_model_comparison"
SUMMARY_BASENAME = "metal_baseline_model_comparison"
RUN_NAME_PREFIX = "metal_baseline"

EPOCHS = 50
BATCH_SIZES_CSV = "8"
LEARNING_RATES_CSV = "3e-5,1e-4"
WEIGHT_DECAYS_CSV = "1e-4"
SEEDS_CSV = "42,43"
MAX_CONFIGURATION_RUNS = 12

HIDDEN_S_VALUES_CSV = "128"
HIDDEN_V_VALUES_CSV = "16"
EDGE_HIDDEN_VALUES_CSV = "64"
GVP_LAYERS_VALUES_CSV = "4"
HEAD_MLP_LAYERS_VALUES_CSV = "2"
EDGE_RADIUS_VALUES_CSV = "8.0"
ESM_FUSION_DIM_VALUES_CSV = "128"
EARLY_ESM_DIM_VALUES_CSV = "32"

ESM_EMBEDDINGS_DIR = ""  # set to your embeddings folder when available
ALLOW_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_ESM_EMBEDDINGS = True
RING_EDGE_MODE = "with_ring"
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

In the **Optional training execution** cell:

```python
LAUNCH_PLANNED_TRAINING_RUNS = True
```

Expected outputs/files:

- Planned-run CSV and dictionary under `<RUNS_DIR>/`
- Completed run directories for each planned validation-only run
- `<SUMMARY_BASENAME>.csv` and `<SUMMARY_BASENAME>_completed_only.csv`
- `<SUMMARY_BASENAME>.png` when plotting succeeds
- No `test_report.json`

Success criteria:

- All planned runs complete or failures are understood and documented.
- Each run uses `selection_metric = val_metal_balanced_acc`.
- `split_diagnostics.json` shows usable train/validation class coverage.
- Comparison tables rank only validation or seed-repeat validation rows.

### Decision gate after Stage 2B

Proceed to Stage 3 or Stage 4 only if:

- The Stage 2B success criteria are met.
- No held-out test files were created.
- `val_metal_balanced_acc` is the selection metric on all completed runs.
- Diagnostics report every class present in both train and validation splits.

If gate fails: fix ESM coverage, rerun the affected baseline family, or fall
back to the Stage 2A Only-GVP anchor until ESM-ready runs are trustworthy. Choose
baseline anchors by validation evidence, not by held-out test, and prefer
stability across seeds over one high run.

## Stage 3 - Small Debug Optuna

Purpose: verify the controlled Optuna path, storage, command generation, and
search-space parsing without treating the result as model-selection evidence.

When to use it: first Optuna run in a new runtime or after editing Optuna
configuration fields.

Expected scale/runtime: smoke/debug, minutes to under an hour.

Notebook configuration block:

```python
TASK = "metal"
RUN_MODE = "controlled_hpo_optuna"
RECOMMENDED_RUN_SET = "custom"
MODEL_PRESET = "Only-GVP"
RUN_BATCH_ID = "metal_only_gvp_optuna_debug"
SUMMARY_BASENAME = "metal_only_gvp_optuna_debug"
RUN_NAME_PREFIX = "metal_only_gvp_optuna_debug"

EPOCHS = 10
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_metal_balanced_acc"

HIDDEN_S_VALUES_CSV = "128"
HIDDEN_V_VALUES_CSV = "16"
EDGE_HIDDEN_VALUES_CSV = "64"
GVP_LAYERS_VALUES_CSV = "4"
HEAD_MLP_LAYERS_VALUES_CSV = "2"
EDGE_RADIUS_VALUES_CSV = "8.0"
RING_EDGE_MODE = "with_ring"

OPTUNA_INTENSITY = "custom"
N_OPTUNA_TRIALS = 4
MAX_EPOCHS_PER_TRIAL = 3
OPTUNA_N_STARTUP_TRIALS = 4
OPTUNA_TPE_MULTIVARIATE = True
OPTUNA_TPE_GROUP = True
OPTUNA_AUTO_CONFIGURE_BUDGET = False
OPTUNA_USE_PRUNING = False
OPTUNA_PRUNER_TYPE = "none"
OPTUNA_SEARCH_PRESET = "first_useful_only_gvp_narrow"
OPTUNA_STUDY_NAME = "metal_only_gvp_optuna_debug"
OPTUNA_STORAGE = "sqlite:////content/drive/MyDrive/DeepMzyme/optuna/metal_only_gvp_optuna_debug.db"
OPTUNA_SPLIT_SEED = 42
OPTUNA_LEARNING_RATE_RANGE = "1e-5,3e-4"
OPTUNA_WEIGHT_DECAYS_CSV = "0.0,1e-5,1e-4"
OPTUNA_BATCH_SIZES_CSV = "4,8"
OPTUNA_METAL_CLASS_WEIGHT_MODES_CSV = "none,inverse_frequency,inverse_sqrt_frequency,effective_number"
OPTUNA_METAL_LOSS_FUNCTIONS_CSV = "cross_entropy"
OPTUNA_METAL_LABEL_SMOOTHING_VALUES_CSV = "0.0"
OPTUNA_BALANCE_METAL_SITE_SYMBOLS_CSV = "False"
RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = False

INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

In the **Optional training execution** cell:

```python
LAUNCH_PLANNED_TRAINING_RUNS = True
```

Expected outputs/files:

- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/all_trials.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/top_trials.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/best_trial.json`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/optuna_study_summary.md`
- `top_reevaluation_commands.txt`

Success criteria:

- Optuna launches and completes the debug trials.
- Search-space preview shows architecture fixed to Only-GVP.
- Trial commands omit held-out test evaluation.

### Decision gate after Stage 3

Proceed to Stage 4 only if:

- The Stage 3 success criteria are met.
- No held-out test files were created.
- `val_metal_balanced_acc` is the selection metric on all completed trials.
- Diagnostics report every class present in both train and validation splits.

If gate fails: fix Optuna storage, search-space parsing, command generation, or
feature paths before launching Stage 4. Do not choose hyperparameters from this
debug run.

## Stage 4 - Controlled Medium Optuna Search

Purpose: run a useful but bounded HPO pass inside one selected model family.

When to use it: after baseline behavior is understood and you have selected a
model family to tune, usually Only-GVP first.

Expected scale/runtime: useful serious run on a G4-class GPU, usually hours.

Notebook configuration block for first useful Only-GVP HPO:

```python
TASK = "metal"
RUN_MODE = "controlled_hpo_optuna"
RECOMMENDED_RUN_SET = "custom"
MODEL_PRESET = "Only-GVP"
RUN_BATCH_ID = "metal_only_gvp_optuna_medium"
SUMMARY_BASENAME = "metal_only_gvp_optuna_medium"
RUN_NAME_PREFIX = "metal_only_gvp_optuna_medium"

EPOCHS = 50
BATCH_SIZES_CSV = "8"
LEARNING_RATES_CSV = "3e-5"
WEIGHT_DECAYS_CSV = "1e-4"
SEEDS_CSV = "42"

HIDDEN_S_VALUES_CSV = "128"
HIDDEN_V_VALUES_CSV = "16"
EDGE_HIDDEN_VALUES_CSV = "64"
GVP_LAYERS_VALUES_CSV = "4"
HEAD_MLP_LAYERS_VALUES_CSV = "2"
EDGE_RADIUS_VALUES_CSV = "8.0"
RING_EDGE_MODE = "with_ring"

OPTUNA_INTENSITY = "custom"
N_OPTUNA_TRIALS = 64
MAX_EPOCHS_PER_TRIAL = 35
OPTUNA_N_STARTUP_TRIALS = 20
OPTUNA_TPE_MULTIVARIATE = True
OPTUNA_TPE_GROUP = True
OPTUNA_AUTO_CONFIGURE_BUDGET = False
OPTUNA_USE_PRUNING = False
OPTUNA_PRUNER_TYPE = "none"
OPTUNA_SEARCH_PRESET = "first_useful_only_gvp_narrow"
OPTUNA_STUDY_NAME = "metal_only_gvp_optuna_medium"
OPTUNA_STORAGE = "sqlite:////content/drive/MyDrive/DeepMzyme/optuna/metal_only_gvp_optuna_medium.db"
OPTUNA_SPLIT_SEED = 42
OPTUNA_LEARNING_RATE_RANGE = "1e-5,3e-4"
OPTUNA_WEIGHT_DECAYS_CSV = "0.0,1e-5,1e-4"
OPTUNA_BATCH_SIZES_CSV = "8,16"
OPTUNA_METAL_CLASS_WEIGHT_MODES_CSV = "none,inverse_frequency,inverse_sqrt_frequency,effective_number"
OPTUNA_METAL_LOSS_FUNCTIONS_CSV = "cross_entropy"
OPTUNA_METAL_LABEL_SMOOTHING_VALUES_CSV = "0.0,0.05"
OPTUNA_BALANCE_METAL_SITE_SYMBOLS_CSV = "False,True"
RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = False
TOP_K_CONFIGS_FOR_SEED_REPEAT = 3
REPEAT_SEEDS = "42,123,2026,43,44"

INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

Expected outputs/files:

- Optuna study directory under `<RUNS_DIR>/optuna/`
- `all_trials.csv`, `top_trials.csv`, `best_trial.json`
- `optuna_best_config.json`, `best_config_command.txt`
- `top_reevaluation_commands.txt`

Success criteria:

- The study completes enough trials for a meaningful ranking.
- The best-trial summary is based on `val_metal_balanced_acc`.
- Trial logs show validation-only runs, not final-test runs.
- Top candidates are plausible and not dominated by missing-class diagnostics.

### Decision gate after Stage 4

Proceed to Stage 5 or Stage 6 only if:

- The Stage 4 success criteria are met.
- No held-out test files were created.
- `val_metal_balanced_acc` is the selection metric on all completed runs.
- Diagnostics report every class present in both train and validation splits.
- Top candidates are meaningfully above the Stage 2A random/seed mean, not just
  isolated noisy trials.

If gate fails: check the search space; widen `OPTUNA_LEARNING_RATE_RANGE` or
open `OPTUNA_HIDDEN_S_VALUES_CSV`. Do not pick the final model from one Optuna
trial alone; run top-K seed-repeat validation before considering a configuration
stable.

## Stage 5 - Large Extensive Optuna Search

Purpose: perform a longer, controlled search after the simpler baseline and
medium HPO justify the model family and search axes.

When to use it: after at least one medium HPO or seed-repeat batch identifies
the model family and search axes worth expanding, or when the user asks for a
fresh broad Optuna check and does not explicitly ask to rely on previous raw
outputs.

Expected scale/runtime: large Optuna search, potentially very long or
overnight. A 200-trial run can be substantially longer than one night depending
on GPU and model.

Important scope rule: the notebook's Optuna mode optimizes within the selected
`MODEL_PRESET`. It does not freely search architectures or fusion modes. Choose
the model family explicitly, then search a controlled set of hyperparameters.

Advanced-fusion ordering rule: Stages 5D, 5E, 5F are only valid after Stage 5C
(GVP + late fusion) has produced a Stage 6 seed-repeat candidate that exceeds
the Stage 2A Only-GVP anchor by >= 0.01 `val_metal_balanced_acc` mean across
the 5-seed list. If Stage 5C does not clear that bar, do not launch 5D/5E/5F.

### 5A - 200-Trial Only-GVP Capacity Search

Notebook configuration block:

```python
TASK = "metal"
RUN_MODE = "controlled_hpo_optuna"
RECOMMENDED_RUN_SET = "custom"
MODEL_PRESET = "Only-GVP"
RUN_BATCH_ID = "metal_only_gvp_optuna_200_capacity"
SUMMARY_BASENAME = "metal_only_gvp_optuna_200_capacity"
RUN_NAME_PREFIX = "metal_only_gvp_optuna_200_capacity"

EPOCHS = 50
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_metal_balanced_acc"
RING_EDGE_MODE = "with_ring"

OPTUNA_INTENSITY = "custom"
N_OPTUNA_TRIALS = 200
MAX_EPOCHS_PER_TRIAL = 50
OPTUNA_N_STARTUP_TRIALS = 40
OPTUNA_TPE_MULTIVARIATE = True
OPTUNA_TPE_GROUP = True
OPTUNA_AUTO_CONFIGURE_BUDGET = False
OPTUNA_USE_PRUNING = False
OPTUNA_PRUNER_TYPE = "none"
OPTUNA_SEARCH_PRESET = "later_capacity"
OPTUNA_STUDY_NAME = "metal_only_gvp_optuna_200_capacity"
OPTUNA_STORAGE = "sqlite:////content/drive/MyDrive/DeepMzyme/optuna/metal_only_gvp_optuna_200_capacity.db"
OPTUNA_SPLIT_SEED = 42
OPTUNA_TIMEOUT_MINUTES = 0

OPTUNA_LEARNING_RATE_RANGE = "5e-6,3e-4"
OPTUNA_WEIGHT_DECAYS_CSV = "0.0,1e-6,1e-5,1e-4"
OPTUNA_BATCH_SIZES_CSV = "8,16"
OPTUNA_METAL_CLASS_WEIGHT_MODES_CSV = "none,inverse_frequency,inverse_sqrt_frequency,effective_number"
OPTUNA_METAL_LOSS_FUNCTIONS_CSV = "cross_entropy,focal"
OPTUNA_METAL_LABEL_SMOOTHING_VALUES_CSV = "0.0,0.03,0.05,0.1"
OPTUNA_BALANCE_METAL_SITE_SYMBOLS_CSV = "False,True"
OPTUNA_METAL_FOCAL_GAMMA_VALUES_CSV = "1.5,2.0,2.5"

OPTUNA_HIDDEN_S_VALUES_CSV = "128,256"
OPTUNA_HIDDEN_V_VALUES_CSV = "16,32"
OPTUNA_EDGE_HIDDEN_VALUES_CSV = "64,128"
OPTUNA_GVP_LAYERS_VALUES_CSV = "2,3,4"
OPTUNA_HEAD_MLP_LAYERS_VALUES_CSV = "1,2"
OPTUNA_EDGE_RADIUS_VALUES_CSV = "6.0,8.0,10.0"

RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = False
TOP_K_CONFIGS_FOR_SEED_REPEAT = 3
REPEAT_SEEDS = "42,123,2026,43,44"
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

### Decision gate after Stage 5A

Proceed to Stage 6 for Only-GVP candidates, or to Stage 5B/5C for family
comparison, only if:

- The study writes complete `all_trials.csv`, `top_trials.csv`, and
  `best_trial.json`.
- No held-out test files were created.
- `val_metal_balanced_acc` is the selection metric on all completed runs.
- Diagnostics report every class present in both train and validation splits.

If gate fails: do not advance to a more complex fusion family; revisit Stage 2A
and the Stage 5A search space.

### 5B - 120-Trial Only-ESM Search

Run this after ESM coverage is valid. It is the ESM-only baseline HPO; it does
not use graph/RING capacity fields even if those fields remain present in the
notebook.

Notebook configuration block:

```python
TASK = "metal"
RUN_MODE = "controlled_hpo_optuna"
RECOMMENDED_RUN_SET = "custom"
MODEL_PRESET = "Only-ESM"
RUN_BATCH_ID = "metal_only_esm_optuna_120_controlled"
SUMMARY_BASENAME = "metal_only_esm_optuna_120_controlled"
RUN_NAME_PREFIX = "metal_only_esm_optuna_120"

EPOCHS = 50
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_metal_balanced_acc"
RING_EDGE_MODE = "with_ring"

ESM_EMBEDDINGS_DIR = ""  # set to your embeddings folder when available
ALLOW_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_ESM_EMBEDDINGS = True

OPTUNA_INTENSITY = "custom"
N_OPTUNA_TRIALS = 120
MAX_EPOCHS_PER_TRIAL = 50
OPTUNA_N_STARTUP_TRIALS = 30
OPTUNA_TPE_MULTIVARIATE = True
OPTUNA_TPE_GROUP = True
OPTUNA_AUTO_CONFIGURE_BUDGET = False
OPTUNA_USE_PRUNING = False
OPTUNA_PRUNER_TYPE = "none"
OPTUNA_SEARCH_PRESET = "custom"
OPTUNA_STUDY_NAME = "metal_only_esm_optuna_120_controlled"
OPTUNA_STORAGE = "sqlite:////content/drive/MyDrive/DeepMzyme/optuna/metal_only_esm_optuna_120_controlled.db"
OPTUNA_SPLIT_SEED = 42
OPTUNA_TIMEOUT_MINUTES = 0

OPTUNA_LEARNING_RATE_RANGE = "5e-6,2e-4"
OPTUNA_WEIGHT_DECAYS_CSV = "0.0,1e-6,1e-5,1e-4"
OPTUNA_BATCH_SIZES_CSV = "8,16"
OPTUNA_METAL_CLASS_WEIGHT_MODES_CSV = "none,inverse_frequency,inverse_sqrt_frequency,effective_number"
OPTUNA_METAL_LOSS_FUNCTIONS_CSV = "cross_entropy"
OPTUNA_METAL_LABEL_SMOOTHING_VALUES_CSV = "0.0,0.03,0.05,0.1"
OPTUNA_BALANCE_METAL_SITE_SYMBOLS_CSV = "False,True"

OPTUNA_HIDDEN_S_VALUES_CSV = "128,256"
OPTUNA_HEAD_MLP_LAYERS_VALUES_CSV = "1,2,3"

RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = False
TOP_K_CONFIGS_FOR_SEED_REPEAT = 3
REPEAT_SEEDS = "42,123,2026,43,44"
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

### Decision gate after Stage 5B

Proceed to Stage 6 for Only-ESM candidates, or to Stage 5C, only if:

- ESM coverage is valid and no run used missing ESM embeddings as a reportable
  fallback.
- No held-out test files were created.
- `val_metal_balanced_acc` is the selection metric on all completed runs.
- Diagnostics report every class present in both train and validation splits.

If gate fails: fix ESM coverage or narrow the Only-ESM search before comparing
ESM-informed model families.

### 5C - 200-Trial GVP + Late-Fusion Search

Run this only after ESM coverage is valid and simpler baselines justify ESM
fusion.

Notebook configuration block:

```python
TASK = "metal"
RUN_MODE = "controlled_hpo_optuna"
RECOMMENDED_RUN_SET = "custom"
MODEL_PRESET = "GVP + late fusion"
RUN_BATCH_ID = "metal_late_fusion_optuna_200_controlled"
SUMMARY_BASENAME = "metal_late_fusion_optuna_200_controlled"
RUN_NAME_PREFIX = "metal_late_fusion_optuna_200"

EPOCHS = 50
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_metal_balanced_acc"
RING_EDGE_MODE = "with_ring"

ESM_EMBEDDINGS_DIR = ""  # set to your embeddings folder when available
ALLOW_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_ESM_EMBEDDINGS = True

OPTUNA_INTENSITY = "custom"
N_OPTUNA_TRIALS = 200
MAX_EPOCHS_PER_TRIAL = 50
OPTUNA_N_STARTUP_TRIALS = 40
OPTUNA_TPE_MULTIVARIATE = True
OPTUNA_TPE_GROUP = True
OPTUNA_AUTO_CONFIGURE_BUDGET = False
OPTUNA_USE_PRUNING = False
OPTUNA_PRUNER_TYPE = "none"
OPTUNA_SEARCH_PRESET = "custom"
OPTUNA_STUDY_NAME = "metal_late_fusion_optuna_200_controlled"
OPTUNA_STORAGE = "sqlite:////content/drive/MyDrive/DeepMzyme/optuna/metal_late_fusion_optuna_200_controlled.db"
OPTUNA_SPLIT_SEED = 42
OPTUNA_TIMEOUT_MINUTES = 0

OPTUNA_LEARNING_RATE_RANGE = "5e-6,2e-4"
OPTUNA_WEIGHT_DECAYS_CSV = "0.0,1e-6,1e-5,1e-4"
OPTUNA_BATCH_SIZES_CSV = "8,16"
OPTUNA_METAL_CLASS_WEIGHT_MODES_CSV = "inverse_frequency,inverse_sqrt_frequency,effective_number"
OPTUNA_METAL_LOSS_FUNCTIONS_CSV = "cross_entropy"
OPTUNA_METAL_LABEL_SMOOTHING_VALUES_CSV = "0.0,0.03,0.05"
OPTUNA_BALANCE_METAL_SITE_SYMBOLS_CSV = "False,True"

OPTUNA_HIDDEN_S_VALUES_CSV = "128,256"
OPTUNA_HIDDEN_V_VALUES_CSV = "16,32"
OPTUNA_EDGE_HIDDEN_VALUES_CSV = "64,128"
OPTUNA_GVP_LAYERS_VALUES_CSV = "2,3,4"
OPTUNA_HEAD_MLP_LAYERS_VALUES_CSV = "1,2"
OPTUNA_EDGE_RADIUS_VALUES_CSV = "6.0,8.0,10.0"
OPTUNA_ESM_FUSION_DIM_VALUES_CSV = "64,128,256"

RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = False
TOP_K_CONFIGS_FOR_SEED_REPEAT = 3
REPEAT_SEEDS = "42,123,2026,43,44"
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

### Decision gate after Stage 5C

Proceed to Stage 6 only if:

- The late-fusion study produces plausible top candidates and complete Optuna
  outputs.
- No held-out test files were created.
- `val_metal_balanced_acc` is the selection metric on all completed runs.
- Diagnostics report every class present in both train and validation splits.

Proceed to Stage 5D/5E/5F only after Stage 6 confirms a late-fusion candidate
that exceeds the Stage 2A Only-GVP anchor by >= 0.01
`val_metal_balanced_acc` mean across the 5-seed list.

If gate fails: no candidate from Stage 5C should trigger advanced fusion. Return
to Stage 2A/5A or revise the late-fusion search space.

### 5D - 200-Trial GVP + Node-Level Late-Fusion Search

Run this after the late-fusion baseline has a stable validation anchor.

Notebook configuration block:

```python
TASK = "metal"
RUN_MODE = "controlled_hpo_optuna"
RECOMMENDED_RUN_SET = "custom"
MODEL_PRESET = "GVP + node-level late fusion"
RUN_BATCH_ID = "metal_node_late_fusion_optuna_200_controlled"
SUMMARY_BASENAME = "metal_node_late_fusion_optuna_200_controlled"
RUN_NAME_PREFIX = "metal_node_late_fusion_optuna_200"

EPOCHS = 50
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_metal_balanced_acc"
RING_EDGE_MODE = "with_ring"

ESM_EMBEDDINGS_DIR = ""  # set to your embeddings folder when available
ALLOW_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_ESM_EMBEDDINGS = True

OPTUNA_INTENSITY = "custom"
N_OPTUNA_TRIALS = 200
MAX_EPOCHS_PER_TRIAL = 50
OPTUNA_N_STARTUP_TRIALS = 40
OPTUNA_TPE_MULTIVARIATE = True
OPTUNA_TPE_GROUP = True
OPTUNA_AUTO_CONFIGURE_BUDGET = False
OPTUNA_USE_PRUNING = False
OPTUNA_PRUNER_TYPE = "none"
OPTUNA_SEARCH_PRESET = "custom"
OPTUNA_STUDY_NAME = "metal_node_late_fusion_optuna_200_controlled"
OPTUNA_STORAGE = "sqlite:////content/drive/MyDrive/DeepMzyme/optuna/metal_node_late_fusion_optuna_200_controlled.db"
OPTUNA_SPLIT_SEED = 42
OPTUNA_TIMEOUT_MINUTES = 0

OPTUNA_LEARNING_RATE_RANGE = "5e-6,2e-4"
OPTUNA_WEIGHT_DECAYS_CSV = "0.0,1e-6,1e-5,1e-4"
OPTUNA_BATCH_SIZES_CSV = "8,16"
OPTUNA_METAL_CLASS_WEIGHT_MODES_CSV = "inverse_frequency,inverse_sqrt_frequency,effective_number"
OPTUNA_METAL_LOSS_FUNCTIONS_CSV = "cross_entropy"
OPTUNA_METAL_LABEL_SMOOTHING_VALUES_CSV = "0.0,0.03,0.05"
OPTUNA_BALANCE_METAL_SITE_SYMBOLS_CSV = "False,True"

OPTUNA_HIDDEN_S_VALUES_CSV = "128,256"
OPTUNA_HIDDEN_V_VALUES_CSV = "16,32"
OPTUNA_EDGE_HIDDEN_VALUES_CSV = "64,128"
OPTUNA_GVP_LAYERS_VALUES_CSV = "2,3,4"
OPTUNA_HEAD_MLP_LAYERS_VALUES_CSV = "1,2"
OPTUNA_EDGE_RADIUS_VALUES_CSV = "6.0,8.0,10.0"
OPTUNA_ESM_FUSION_DIM_VALUES_CSV = "64,128,256"

RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = False
TOP_K_CONFIGS_FOR_SEED_REPEAT = 3
REPEAT_SEEDS = "42,123,2026,43,44"
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

### Decision gate after Stage 5D

Proceed to Stage 6 only if:

- Stage 5C previously cleared the advanced-fusion ordering gate.
- No held-out test files were created.
- `val_metal_balanced_acc` is the selection metric on all completed runs.
- Diagnostics report every class present in both train and validation splits.

If gate fails: do not advance to Stage 5E/5F because no candidate beats the
Stage 2A anchor by >= 1% balanced accuracy across 5 seeds at Stage 6; revisit
Stage 2A or Stage 5C.

### 5E - 200-Trial GVP + Hybrid-Fusion Search

Run this only after early/late ESM evidence justifies injecting ESM before graph
message passing and also using late fusion.

Notebook configuration block:

```python
TASK = "metal"
RUN_MODE = "controlled_hpo_optuna"
RECOMMENDED_RUN_SET = "custom"
MODEL_PRESET = "GVP + hybrid fusion"
RUN_BATCH_ID = "metal_hybrid_fusion_optuna_200_controlled"
SUMMARY_BASENAME = "metal_hybrid_fusion_optuna_200_controlled"
RUN_NAME_PREFIX = "metal_hybrid_fusion_optuna_200"

EPOCHS = 50
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_metal_balanced_acc"
RING_EDGE_MODE = "with_ring"

ESM_EMBEDDINGS_DIR = ""  # set to your embeddings folder when available
ALLOW_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_ESM_EMBEDDINGS = True

OPTUNA_INTENSITY = "custom"
N_OPTUNA_TRIALS = 200
MAX_EPOCHS_PER_TRIAL = 50
OPTUNA_N_STARTUP_TRIALS = 40
OPTUNA_TPE_MULTIVARIATE = True
OPTUNA_TPE_GROUP = True
OPTUNA_AUTO_CONFIGURE_BUDGET = False
OPTUNA_USE_PRUNING = False
OPTUNA_PRUNER_TYPE = "none"
OPTUNA_SEARCH_PRESET = "custom"
OPTUNA_STUDY_NAME = "metal_hybrid_fusion_optuna_200_controlled"
OPTUNA_STORAGE = "sqlite:////content/drive/MyDrive/DeepMzyme/optuna/metal_hybrid_fusion_optuna_200_controlled.db"
OPTUNA_SPLIT_SEED = 42
OPTUNA_TIMEOUT_MINUTES = 0

OPTUNA_LEARNING_RATE_RANGE = "5e-6,1.5e-4"
OPTUNA_WEIGHT_DECAYS_CSV = "0.0,1e-6,1e-5,1e-4"
OPTUNA_BATCH_SIZES_CSV = "8,16"
OPTUNA_METAL_CLASS_WEIGHT_MODES_CSV = "inverse_frequency,inverse_sqrt_frequency,effective_number"
OPTUNA_METAL_LOSS_FUNCTIONS_CSV = "cross_entropy"
OPTUNA_METAL_LABEL_SMOOTHING_VALUES_CSV = "0.0,0.03,0.05"
OPTUNA_BALANCE_METAL_SITE_SYMBOLS_CSV = "False,True"

OPTUNA_HIDDEN_S_VALUES_CSV = "128,256"
OPTUNA_HIDDEN_V_VALUES_CSV = "16,32"
OPTUNA_EDGE_HIDDEN_VALUES_CSV = "64,128"
OPTUNA_GVP_LAYERS_VALUES_CSV = "2,3,4"
OPTUNA_HEAD_MLP_LAYERS_VALUES_CSV = "1,2"
OPTUNA_EDGE_RADIUS_VALUES_CSV = "6.0,8.0,10.0"
OPTUNA_ESM_FUSION_DIM_VALUES_CSV = "64,128,256"
OPTUNA_EARLY_ESM_DIM_VALUES_CSV = "16,32,64"
OPTUNA_EARLY_ESM_DROPOUT_VALUES_CSV = "0.0,0.1,0.2"

RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = False
TOP_K_CONFIGS_FOR_SEED_REPEAT = 3
REPEAT_SEEDS = "42,123,2026,43,44"
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

### Decision gate after Stage 5E

Proceed to Stage 6 only if:

- Stage 5C previously cleared the advanced-fusion ordering gate.
- No held-out test files were created.
- `val_metal_balanced_acc` is the selection metric on all completed runs.
- Diagnostics report every class present in both train and validation splits.

If gate fails: stop advanced fusion escalation and revisit the simpler
late-fusion or Only-GVP anchors before cross-attention.

### 5F - 120-Trial GVP + Cross-Modal Attention Search

Run this last among fusion models. Keep attention narrow at first because it has
more overfitting degrees of freedom.

Notebook configuration block:

```python
TASK = "metal"
RUN_MODE = "controlled_hpo_optuna"
RECOMMENDED_RUN_SET = "custom"
MODEL_PRESET = "GVP + cross-modal attention"
RUN_BATCH_ID = "metal_cross_attention_optuna_120_controlled"
SUMMARY_BASENAME = "metal_cross_attention_optuna_120_controlled"
RUN_NAME_PREFIX = "metal_cross_attention_optuna_120"

EPOCHS = 50
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_metal_balanced_acc"
RING_EDGE_MODE = "with_ring"
CROSS_ATTENTION_NEIGHBORHOOD = "first_second_shell"
CROSS_ATTENTION_BIDIRECTIONAL = False

ESM_EMBEDDINGS_DIR = ""  # set to your embeddings folder when available
ALLOW_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_ESM_EMBEDDINGS = True

OPTUNA_INTENSITY = "custom"
N_OPTUNA_TRIALS = 120
MAX_EPOCHS_PER_TRIAL = 50
OPTUNA_N_STARTUP_TRIALS = 30
OPTUNA_TPE_MULTIVARIATE = True
OPTUNA_TPE_GROUP = True
OPTUNA_AUTO_CONFIGURE_BUDGET = False
OPTUNA_USE_PRUNING = False
OPTUNA_PRUNER_TYPE = "none"
OPTUNA_SEARCH_PRESET = "custom"
OPTUNA_STUDY_NAME = "metal_cross_attention_optuna_120_controlled"
OPTUNA_STORAGE = "sqlite:////content/drive/MyDrive/DeepMzyme/optuna/metal_cross_attention_optuna_120_controlled.db"
OPTUNA_SPLIT_SEED = 42
OPTUNA_TIMEOUT_MINUTES = 0

OPTUNA_LEARNING_RATE_RANGE = "5e-6,1e-4"
OPTUNA_WEIGHT_DECAYS_CSV = "0.0,1e-6,1e-5,1e-4"
OPTUNA_BATCH_SIZES_CSV = "8,16"
OPTUNA_METAL_CLASS_WEIGHT_MODES_CSV = "inverse_frequency,inverse_sqrt_frequency,effective_number"
OPTUNA_METAL_LOSS_FUNCTIONS_CSV = "cross_entropy"
OPTUNA_METAL_LABEL_SMOOTHING_VALUES_CSV = "0.0,0.03,0.05"
OPTUNA_BALANCE_METAL_SITE_SYMBOLS_CSV = "False"

OPTUNA_HIDDEN_S_VALUES_CSV = "128,256"
OPTUNA_HIDDEN_V_VALUES_CSV = "16,32"
OPTUNA_EDGE_HIDDEN_VALUES_CSV = "64,128"
OPTUNA_GVP_LAYERS_VALUES_CSV = "2,3,4"
OPTUNA_HEAD_MLP_LAYERS_VALUES_CSV = "1,2"
OPTUNA_EDGE_RADIUS_VALUES_CSV = "6.0,8.0,10.0"
OPTUNA_CROSS_ATTENTION_LAYERS_CSV = "1"
OPTUNA_CROSS_ATTENTION_HEADS_CSV = "2,4"
OPTUNA_CROSS_ATTENTION_DROPOUT_VALUES_CSV = "0.0,0.1,0.2"

RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = False
TOP_K_CONFIGS_FOR_SEED_REPEAT = 3
REPEAT_SEEDS = "42,123,2026,43,44"
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

### Decision gate after Stage 5F

Proceed to Stage 6 only if:

- Stage 5C previously cleared the advanced-fusion ordering gate.
- No held-out test files were created.
- `val_metal_balanced_acc` is the selection metric on all completed runs.
- Diagnostics report every class present in both train and validation splits.
- Attention candidates justify their extra complexity against the Stage 6
  late-fusion candidate.

If gate fails: do not broaden cross-attention. Return to the best validated
simpler fusion family.

### 5G - Optional Radius-Only Ablation

Use only when you deliberately want to compare against the older radius-only
graph setting. This does not make Optuna sample RING on/off; it fixes the base
run to radius-only graph construction.

```python
RING_EDGE_MODE = "without_ring"
REQUIRE_RING_EDGES = False
PREPARE_MISSING_RING_EDGES = True
RING_FEATURES_DIR = ""
```

Expected outputs/files:

- Large persistent SQLite Optuna study in Drive
- Complete Optuna CSV/JSON/Markdown outputs under `<RUNS_DIR>/optuna/`
- Per-trial run directories and logs under `<RUNS_DIR>/`
- No held-out test report

Success criteria:

- Search space preview matches the intended controlled scope.
- No test-report files are created by the HPO runs.
- Top trials improve or clarify validation behavior without relying on one
  lucky seed.

### Decision gate after Stage 5G

Proceed to Stage 6 only if:

- The ablation was explicitly labeled radius-only and compared against the
  matching RING-enabled family.
- No held-out test files were created.
- `val_metal_balanced_acc` is the selection metric on all completed runs.
- Diagnostics report every class present in both train and validation splits.

If gate fails: do not use the ablation as model-selection evidence. Choose the
top 2-3 valid candidates for seed-repeat validation and do not finalize from a
raw Optuna ranking alone.

## Stage 6 - Top-K Seed-Repeat Validation

Purpose: confirm whether top HPO candidates are stable across random seeds.

When to use it: after a medium or large Optuna search has produced top
candidates.

Expected scale/runtime: serious run, long or overnight. Runtime is roughly
`TOP_K_CONFIGS_FOR_SEED_REPEAT x number_of_repeat_seeds x EPOCHS`.

Preferred notebook-integrated configuration: set these before launching the HPO
whose top trials should be repeated. The notebook writes top commands and then
runs seed repeats after the study finishes.

```python
RUN_MODE = "controlled_hpo_optuna"
RECOMMENDED_RUN_SET = "custom"
RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = True
TOP_K_CONFIGS_FOR_SEED_REPEAT = 3
REPEAT_SEEDS = "42,123,2026,43,44"
ALLOW_SEED_REPEAT_MODEL_PRESET_MISMATCH = False
RETRAIN_BEST_CONFIG_AFTER_HPO = False

EPOCHS = 50
SELECTION_METRIC = "val_metal_balanced_acc"
OPTUNA_SELECTION_METRIC = "val_metal_balanced_acc"
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

If the Optuna study is already complete and you did not enable
`RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION`, inspect:

```text
<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/top_reevaluation_commands.txt
```

Those commands intentionally omit held-out test evaluation. Run only the top-K
commands you predeclare, with the seed list you predeclare, and keep the results
as validation-only evidence.

Expected outputs/files:

- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/seed_repeat_results.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/seed_repeat_summary.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/seed_repeat_summary.json`
- One validation-only run directory per top-K/seed pair
- No `test_report.json`

Success criteria:

- All top-K/seed runs complete or failures are documented.
- The selected candidate has the best validation mean or an acceptable
  mean/variance tradeoff.
- Diagnostics do not show leakage, missing validation classes, or invalid
  feature coverage.

### Decision gate after Stage 6

Proceed to Stage 7 only if:

- The Stage 6 success criteria are met.
- No held-out test files were created.
- `val_metal_balanced_acc` is the selection metric on all completed runs.
- Diagnostics report every class present in both train and validation splits.
- One final configuration is selected using validation evidence only.

If gate fails: if the top-1 mean is within 1 std of top-2 or top-3, report all
three as candidates and pick by `val_metal_min_recall` tiebreak. Record the
mean validation score, variability, per-class diagnostics, split, seed list, and
epoch budget before any held-out test launch.

### Recommended Stage 6 candidate policy

For each completed Optuna study, repeat the top 3 candidates across:

```python
TOP_K_CONFIGS_FOR_SEED_REPEAT = 3
REPEAT_SEEDS = "42,123,2026,43,44"
RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = True
RETRAIN_BEST_CONFIG_AFTER_HPO = False
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
```

Use top 5 only when the top trials are tightly clustered or when the top 3
contain different model-capacity regimes that are scientifically important to
compare.

A candidate is considered stable enough for final selection only if:

- all 5 seed-repeat runs complete, or failures are explained and not biased;
- no held-out test report was created;
- all runs use `selection_metric = val_metal_balanced_acc`;
- no validation class is missing;
- the candidate has either the best mean validation balanced accuracy or a
  clearly better mean/variance/per-class-recall tradeoff.

## Stage 7 - Final Held-Out Test Evaluation

Purpose: report held-out test performance for the final validation-selected
configuration.

When to use it: only after model family, hyperparameters, checkpoint-selection
metric, seed-repeat interpretation, and final source run are fixed.

Expected scale/runtime: final reporting run, usually minutes to hours depending
on checkpoint loading and evaluation mode.

First run the **Select final run and show saved outputs** cell. Use:

```python
FINAL_RUN_SELECTION_MODE = "auto_best_validation"
FINAL_RUN_TABLE_INDEX = 1
FINAL_RUN_DIR = ""
FINAL_REPORT_BASENAME = "deepmzyme_final_selected_run"
```

Then run the **Optional final held-out test evaluation** cell in preview mode:

```python
FINAL_TEST_WORKFLOW = "preview_only"
LAUNCH_FINAL_HELD_OUT_TEST_EVAL = False
CONFIRM_ONE_SHOT_POLICY = False
FINAL_TEST_SOURCE_RUN_DIR = ""
FINAL_TEST_SOURCE_CHOICE_INDEX = 0
FINAL_TEST_BATCH_PARENT_DIR = ""
FINAL_TEST_BATCH_RUN_GLOB = "*"
FINAL_TEST_RUN_NAME_PREFIX = "final_test"
FINAL_TEST_BATCH_SUMMARY_BASENAME = "final_test_batch_summary"
FINAL_TEST_METAL_REPORT_VIEW = "use_METAL_REPORT_VIEW"
ALLOW_REPEAT_FINAL_TEST_EVAL = False
ALLOW_MIXED_FINAL_TEST_BATCH = False
```

Inspect the printed pre-flight checklist. If the selected source run is the
final validation-selected run and this is final reporting, switch to launch:

```python
FINAL_TEST_WORKFLOW = "evaluate_selected_checkpoint"
LAUNCH_FINAL_HELD_OUT_TEST_EVAL = True
CONFIRM_ONE_SHOT_POLICY = True
FINAL_TEST_SOURCE_RUN_DIR = ""
FINAL_TEST_SOURCE_CHOICE_INDEX = 0
FINAL_TEST_RUN_NAME_PREFIX = "final_test"
FINAL_TEST_METAL_REPORT_VIEW = "use_METAL_REPORT_VIEW"
ALLOW_REPEAT_FINAL_TEST_EVAL = False
ALLOW_MIXED_FINAL_TEST_BATCH = False
```

Expected outputs/files:

- A new final-test run folder under the resolved `RUNS_DIR`
- `test_report.json` in the final-test output folder
- Updated final-test summary CSV/PNG when plotting succeeds
- The source validation run remains unchanged

Success criteria:

- The source run has validation-selected checkpoint metadata.
- The final-test run uses `best_model_checkpoint.pt` or the explicitly selected
  validation checkpoint.
- The output folder is separate from the source validation run.
- The test report includes six-class metal metrics and collapsed-4 metrics.

### Decision gate after Stage 7

Final reporting is complete only if:

- The Stage 7 success criteria are met.
- The source run was the Stage 6 validation-selected configuration.
- `val_metal_balanced_acc` was the selection metric for the source run.
- The final-test output is a separate folder from the source validation run.

If gate fails: do not report the run as final. If the test completed, treat the
one-shot final-test result as already spent for that selection cycle. Do not
choose a different configuration because another tested candidate has a better
held-out test score; return to validation-only experiments for new development.

## Safety Guards To Check

Before any reportable comparison or HPO launch, confirm:

- `INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False`
- `FINAL_TEST_WORKFLOW = "preview_only"` or the final-test cell has not been
  run
- `VAL_FRACTION > 0` or a fold split is explicitly configured
- `SELECTION_METRIC = "val_metal_balanced_acc"` for metal model selection
- `RUN_BATCH_ID` and `SUMMARY_BASENAME` identify the experiment batch clearly
- `ALLOW_SHORT_TRAINING_FOR_DEBUG = False` for reportable runs
- `ALLOW_SEED_REPEAT_MODEL_PRESET_MISMATCH = False`
- `ALLOW_MIXED_FINAL_TEST_BATCH = False`
- `ALLOW_REPEAT_FINAL_TEST_EVAL = False`

The training code also blocks held-out test evaluation when there is no
validation split or when `train_loss` would be used for final-test checkpoint
selection, unless the explicit debug override is passed. Keep that override off
for reportable work.
