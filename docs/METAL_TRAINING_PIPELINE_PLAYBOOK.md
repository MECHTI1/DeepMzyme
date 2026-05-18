# Metal Training Pipeline Playbook

This playbook is the practical, notebook-ready pipeline for DeepMzyme metal
classification. It complements `Plan.md`, which remains the high-level research
and design authority. Current best validation evidence belongs in
`EXPERIMENT_STATUS.md` and `docs/notebook_outputs/`, not in this stable
playbook.


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

Pipeline stage map:

| Stage | Purpose | Exact block |
| --- | --- | --- |
| 0 | Runtime/data/output setup shared by all stages | Common Defaults |
| 1 | Smoke/readiness check | Stage 1 |
| 2 | Baseline validation comparisons | Stage 2A/2B |
| 3 | Debug Optuna plumbing check | Stage 3 |
| 4 | Medium useful Optuna | Stage 4 |
| 5 | Large model-family Optuna and optional RING ablation | Stage 5A-5G |
| 6 | Top-K multi-seed validation | Stage 6 |
| 7 | Final held-out test evaluation | Stage 7 |

G4 GPU policy: for real searches, prefer the explicit `custom` Optuna budgets
below over the notebook's small `first_useful` or `serious` presets. The
recommended serious profile is 50-epoch baselines, batch size 8 when memory is
stable, 64 x 35-epoch medium HPO, 200 x 50-epoch large HPO, and top-3 x 5-seed
validation confirmation.

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

If a run uses `Only-ESM` or any GVP + ESM fusion preset, set
`ESM_EMBEDDINGS_DIR` to the embeddings folder or set
`PREPARE_MISSING_ESM_EMBEDDINGS = True` deliberately. Do not use
`ALLOW_MISSING_ESM_EMBEDDINGS = True` for reportable runs.

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

Decision after this stage:

- If it fails, fix data paths, bundle setup, structure parsing, or feature
  availability before running real comparisons.
- If it succeeds, move to baseline model comparison. Ignore the 1-epoch metric
  as model-quality evidence.

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
RECOMMENDED_RUN_SET = "only_gvp_lr_seed"
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

Decision after this stage:

- Choose a baseline anchor by validation evidence, not by held-out test.
- Prefer stability across seeds over a single high run.
- If explicitly continuing from this baseline, use the selected simpler anchor
  to constrain the next HPO/fusion stage.
- If launching a fresh Optuna check, do not over-constrain it to prior raw
  outputs; search broadly within the selected model family/fusion mode.

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

OPTUNA_INTENSITY = "debug"
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
OPTUNA_STORAGE = ""
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

Decision after this stage:

- If debug Optuna works, move to medium controlled Optuna.
- Do not choose hyperparameters from this debug run.

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
OPTUNA_BATCH_SIZES_CSV = "8"
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

Decision after this stage:

- Do not pick the final model from one Optuna trial alone.
- Run top-K seed-repeat validation before considering a configuration stable.

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
OPTUNA_BATCH_SIZES_CSV = "4,8"
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
OPTUNA_BATCH_SIZES_CSV = "8"
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
OPTUNA_BATCH_SIZES_CSV = "4,8"
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
OPTUNA_BATCH_SIZES_CSV = "4,8"
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
OPTUNA_BATCH_SIZES_CSV = "4,8"
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
OPTUNA_BATCH_SIZES_CSV = "4,8"
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

Decision after this stage:

- Choose the top 2-3 candidates for seed-repeat validation.
- Do not finalize the model from the raw 200-trial ranking alone.

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

Decision after this stage:

- Select one final configuration using validation evidence only.
- Record why it was selected: mean validation score, variability, per-class
  diagnostics, split, seed list, and epoch budget.
- Only then move to final held-out test evaluation.

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

Decision after this stage:

- Report the final held-out test metrics.
- Do not choose a different configuration because another tested candidate has
  a better held-out test score. If more model development is needed, return to
  validation-only experiments and treat the final-test result as already spent
  for that selection cycle.

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
