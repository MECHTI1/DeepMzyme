# EC Training Pipeline Playbook

This playbook is the practical, notebook-ready pipeline for DeepMzyme EC-number
classification. It follows the same stage structure as
`docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`. Read that playbook for the full
rationale behind each stage; this document covers only the EC-specific
differences.

`Plan.md` remains the high-level research and design authority. Current best
validation evidence belongs in `EXPERIMENT_STATUS.md` and
`docs/notebook_outputs/`, not in this stable playbook.


All configuration blocks use variables that exist in
`notebooks/DeepMzyme_training_colab.ipynb` as of this repository state. Paste
any block at the end of the notebook's **Main configuration** cell before
running **Build central CONFIG dictionary**.

## EC-Specific Rules Before You Start

**Label depth.** Start with `EC_LABEL_DEPTHS_CSV = "1"` (first EC digit, seven
classes). Add depth-2 and deeper runs only after depth-1 behavior is stable.
Each depth level is a separate classification problem with its own class count
and difficulty.

**Group weighting.** Always keep `EC_GROUP_WEIGHTING = "structure_id"`. EC
supervision is at the protein/structure level. Multiple metal-pocket samples from
the same structure share the same EC annotation; group weighting prevents those
structures from dominating the loss.

**Contrastive loss.** Start with `EC_CONTRASTIVE_WEIGHTS_CSV = "0.0"` for all
first baselines. Contrastive loss is a secondary feature; enabling it before the
plain supervised baseline is stable adds a confound.

**Split policy.** The non-overlapped PinMyMetal split is mandatory for final
EC held-out testing. The exact PinMyMetal split must not be used as the main
final EC held-out split if train/test structures overlap (see Plan.md section 8).

**Held-out test policy.** Identical to metal: never use the held-out test set
for model comparison, Optuna HPO, or seed-repeat validation. Use only
validation metrics for all selection decisions.

**Fresh-check default.** If the user asks for a new check, new run, or fresh
Optuna sweep without explicitly asking to rely on previous raws/results, use
previous notebook outputs only as context and safety checks. Prefer the broadest
sensible validation-only Optuna search within the selected `MODEL_PRESET`, with
fixed EC depth, fixed selection metric, and common-sense runtime and
feature-availability limits. If the user explicitly asks to rely on previous
running/results/raws, inspect that evidence and use it to narrow, continue, or
repeat the prior configuration.

**Selection metric.** For depth-1 EC training use
`val_ec_group_level_1_balanced_acc`. This uses group weighting and balanced
accuracy, which is the right choice for potentially imbalanced EC first-digit
classes. Update the metric name when the label depth changes (e.g., use
`val_ec_group_level_2_balanced_acc` for depth-2 runs).

**ESM and model order.** For EC classification, sequence-level information
(ESM) may matter more than for metal. Still follow the baseline-first order:
1. Only-GVP first (structure-only baseline).
2. Only-ESM (sequence-only baseline, important reference for EC).
3. GVP + late fusion after both simple baselines are measured.
4. Advanced fusion only if simpler models justify the added complexity.

## Common Defaults

Use these shared defaults unless a stage overrides them.

```python
TASK = "ec"
DATASET_NAME = "train_and_test_sets_structures_non_overlapped_pinmymetal"
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"
OPTUNA_SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False

EC_LABEL_DEPTHS_CSV = "1"
EC_CONTRASTIVE_WEIGHTS_CSV = "0.0"
EC_GROUP_WEIGHTING = "structure_id"

RING_EDGE_MODE = "with_ring"
REQUIRE_RING_EDGES = False
PREPARE_MISSING_RING_EDGES = True

ALLOW_MISSING_EXTERNAL_FEATURES = False
PREPARE_MISSING_EXTERNAL_FEATURES = False
EXTERNAL_FEATURES_ROOT_DIR = ""

COPY_OUTPUTS_TO_DRIVE = True
```

If using `Only-ESM`, `GVP + late fusion`, or any ESM-requiring preset, provide
`ESM_EMBEDDINGS_DIR` or set `PREPARE_MISSING_ESM_EMBEDDINGS = True`
deliberately. Do not use `ALLOW_MISSING_ESM_EMBEDDINGS = True` for reportable
runs.

Update `SELECTION_METRIC` and `OPTUNA_SELECTION_METRIC` when running depth-2 or
deeper EC experiments — change `level_1` to the matching level number.

## Stage 1 — Smoke And Readiness Check

Purpose: verify Colab setup, data paths, EC CSV detection, graph construction,
and the EC training command path.

When to use it: first run in a fresh Colab/runtime, after changing the notebook,
or after changing data bundle paths.

Expected scale/runtime: smoke/debug, minutes.

Notebook configuration block:

```python
TASK = "ec"
RUN_MODE = "single"
RECOMMENDED_RUN_SET = "only_gvp_smoke"
MODEL_PRESET = "Only-GVP"
RUN_BATCH_ID = "ec_smoke_readiness"
SUMMARY_BASENAME = "ec_smoke_readiness"
RUN_NAME_PREFIX = "ec_smoke"

EPOCHS = 1
BATCH_SIZES_CSV = "4"
LEARNING_RATES_CSV = "3e-5"
WEIGHT_DECAYS_CSV = "1e-4"
SEEDS_CSV = "42"
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"

EC_LABEL_DEPTHS_CSV = "1"
EC_CONTRASTIVE_WEIGHTS_CSV = "0.0"
EC_GROUP_WEIGHTING = "structure_id"

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

- Planning prints one runnable Only-GVP EC command.
- Training completes without missing-path, split, EC CSV detection, or CLI
  errors.
- Train and validation EC diagnostics print (class counts at EC level 1).
- No held-out test report is produced.

Decision after this stage:

- If it fails, fix data paths, EC CSV format, bundle setup, or structure
  parsing before running real comparisons.
- If it succeeds, move to baseline model comparison. Ignore the 1-epoch metric.

## Stage 2 — Baseline Model Comparison

Purpose: establish clean EC validation baselines before adding fusion or HPO.

When to use it: after smoke passes and before Optuna or advanced fusion. Run
2A (Only-GVP) first. Run 2B (multi-model) once ESM embeddings are ready.

Expected scale/runtime: medium validation run, hours.

### 2A — Structure-Only Only-GVP Baseline

Notebook configuration block:

```python
TASK = "ec"
RUN_MODE = "manual_configurations"
RECOMMENDED_RUN_SET = "only_gvp_lr_seed"
MODEL_PRESET = "Only-GVP"
RUN_BATCH_ID = "ec_only_gvp_baseline_lr_seed"
SUMMARY_BASENAME = "ec_only_gvp_baseline_lr_seed"
RUN_NAME_PREFIX = "ec_only_gvp_baseline"

EPOCHS = 30
BATCH_SIZES_CSV = "4,8"
LEARNING_RATES_CSV = "3e-5,1e-4"
WEIGHT_DECAYS_CSV = "1e-4"
SEEDS_CSV = "42,43"
MAX_CONFIGURATION_RUNS = 8

HIDDEN_S_VALUES_CSV = "128"
HIDDEN_V_VALUES_CSV = "16"
EDGE_HIDDEN_VALUES_CSV = "64"
GVP_LAYERS_VALUES_CSV = "4"
HEAD_MLP_LAYERS_VALUES_CSV = "2"
EDGE_RADIUS_VALUES_CSV = "8.0"

EC_LABEL_DEPTHS_CSV = "1"
EC_CONTRASTIVE_WEIGHTS_CSV = "0.0"
EC_GROUP_WEIGHTING = "structure_id"
SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"

RING_EDGE_MODE = "with_ring"
ESM_EMBEDDINGS_DIR = ""
ALLOW_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_ESM_EMBEDDINGS = False
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

### 2B — ESM-Ready Baseline Comparison

Run this only after ESM embeddings are available or after you intentionally
allow the notebook to prepare missing embeddings. The
`baseline_model_comparison` preset overrides `MODEL_PRESET` and runs
`Only-GVP`, `Only-ESM`, and `GVP + late fusion`.

Note: for EC classification, the `Only-ESM` baseline is especially important
because EC function often correlates strongly with sequence. Do not skip it.

Notebook configuration block:

```python
TASK = "ec"
RUN_MODE = "manual_configurations"
RECOMMENDED_RUN_SET = "baseline_model_comparison"
# MODEL_PRESET is overridden by baseline_model_comparison (runs Only-GVP, Only-ESM, GVP + late fusion)
RUN_BATCH_ID = "ec_baseline_model_comparison"
SUMMARY_BASENAME = "ec_baseline_model_comparison"
RUN_NAME_PREFIX = "ec_baseline"

EPOCHS = 30
BATCH_SIZES_CSV = "4"
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

EC_LABEL_DEPTHS_CSV = "1"
EC_CONTRASTIVE_WEIGHTS_CSV = "0.0"
EC_GROUP_WEIGHTING = "structure_id"
SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"

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

- Completed run directories for each planned validation-only run
- `<SUMMARY_BASENAME>.csv` and `<SUMMARY_BASENAME>_completed_only.csv`
- No `test_report.json`

Success criteria:

- Each run uses `selection_metric = val_ec_group_level_1_balanced_acc`.
- `split_diagnostics.json` shows usable train/validation EC class coverage at
  depth 1.
- Comparison tables rank only validation rows.

Decision after this stage:

- Choose a baseline anchor by validation evidence, not by held-out test.
- Note whether `Only-ESM` outperforms `Only-GVP` — this is a key signal for EC.
- If Only-ESM is clearly stronger, the ESM-fusion models become the priority.
- If explicitly continuing from this baseline, use the selected simpler anchor
  to constrain the next HPO stage.
- If launching a fresh Optuna check, do not over-constrain it to prior raw
  outputs; search broadly within the selected model family/fusion mode while
  keeping EC depth fixed per study.

## Stage 3 — Small Debug Optuna

Purpose: verify the controlled Optuna path, storage, and search-space parsing
for EC without treating the result as model-selection evidence.

When to use it: first Optuna run in a new runtime or after editing Optuna
configuration fields for EC.

Expected scale/runtime: smoke/debug, minutes to under an hour.

Notebook configuration block:

```python
TASK = "ec"
RUN_MODE = "controlled_hpo_optuna"
RECOMMENDED_RUN_SET = "custom"
MODEL_PRESET = "Only-GVP"
RUN_BATCH_ID = "ec_only_gvp_optuna_debug"
SUMMARY_BASENAME = "ec_only_gvp_optuna_debug"
RUN_NAME_PREFIX = "ec_only_gvp_optuna_debug"

EPOCHS = 10
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"

EC_LABEL_DEPTHS_CSV = "1"
EC_CONTRASTIVE_WEIGHTS_CSV = "0.0"
EC_GROUP_WEIGHTING = "structure_id"

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
OPTUNA_SEARCH_PRESET = "first_useful_only_gvp_narrow"
OPTUNA_STUDY_NAME = "ec_only_gvp_optuna_debug"
OPTUNA_STORAGE = ""
OPTUNA_SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"
OPTUNA_LEARNING_RATE_RANGE = "1e-5,3e-4"
OPTUNA_WEIGHT_DECAYS_CSV = "0.0,1e-5,1e-4"
OPTUNA_BATCH_SIZES_CSV = "4,8"
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

- Optuna launches and completes the debug trials against the EC task.
- Trial commands omit held-out test evaluation.
- Best-trial summary uses `val_ec_group_level_1_balanced_acc`.

Decision after this stage:

- If debug Optuna works, move to medium controlled Optuna.
- Do not choose hyperparameters from this debug run.

## Stage 4 — Controlled Medium Optuna Search

Purpose: run a useful but bounded HPO pass for EC inside one selected model
family, with EC label depth and contrastive weight fixed.

When to use it: after baseline behavior is understood and a model family is
selected for EC HPO, usually Only-GVP first.

Expected scale/runtime: medium validation run to serious run, hours.

Notebook configuration block for first useful Only-GVP EC HPO:

```python
TASK = "ec"
RUN_MODE = "controlled_hpo_optuna"
RECOMMENDED_RUN_SET = "custom"
MODEL_PRESET = "Only-GVP"
RUN_BATCH_ID = "ec_only_gvp_optuna_medium"
SUMMARY_BASENAME = "ec_only_gvp_optuna_medium"
RUN_NAME_PREFIX = "ec_only_gvp_optuna_medium"

EPOCHS = 50
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"

EC_LABEL_DEPTHS_CSV = "1"
EC_CONTRASTIVE_WEIGHTS_CSV = "0.0"
EC_GROUP_WEIGHTING = "structure_id"

HIDDEN_S_VALUES_CSV = "128"
HIDDEN_V_VALUES_CSV = "16"
EDGE_HIDDEN_VALUES_CSV = "64"
GVP_LAYERS_VALUES_CSV = "4"
HEAD_MLP_LAYERS_VALUES_CSV = "2"
EDGE_RADIUS_VALUES_CSV = "8.0"
RING_EDGE_MODE = "with_ring"

OPTUNA_INTENSITY = "first_useful"
OPTUNA_SEARCH_PRESET = "first_useful_only_gvp_narrow"
OPTUNA_STUDY_NAME = "ec_only_gvp_optuna_medium"
OPTUNA_STORAGE = "sqlite:////content/drive/MyDrive/DeepMzyme/optuna/ec_only_gvp_optuna_medium.db"
OPTUNA_SPLIT_SEED = 42
OPTUNA_SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"
OPTUNA_LEARNING_RATE_RANGE = "1e-5,3e-4"
OPTUNA_WEIGHT_DECAYS_CSV = "0.0,1e-5,1e-4"
OPTUNA_BATCH_SIZES_CSV = "4,8"
RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = False
TOP_K_CONFIGS_FOR_SEED_REPEAT = 3
REPEAT_SEEDS = "42,123,2026"

INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

Expected outputs/files:

- Optuna study directory under `<RUNS_DIR>/optuna/`
- `all_trials.csv`, `top_trials.csv`, `best_trial.json`
- `optuna_best_config.json`, `best_config_command.txt`
- `top_reevaluation_commands.txt`

Success criteria:

- Best-trial summary uses `val_ec_group_level_1_balanced_acc`.
- Trial logs show validation-only runs.
- Top candidates are plausible and not dominated by missing-class diagnostics.

Decision after this stage:

- Do not pick the final model from one Optuna trial alone.
- Run top-K seed-repeat validation before considering a configuration stable.

## Stage 5 — Large Extensive Optuna Search

Purpose: perform a longer, controlled EC search after the simpler baseline and
medium HPO justify the model family and search axes.

When to use it: after at least one medium HPO or seed-repeat batch identifies
the model family and axes worth expanding, or when the user asks for a fresh
broad Optuna check and does not explicitly ask to rely on previous raw outputs.

Expected scale/runtime: large Optuna search, potentially very long or
overnight. A 200-trial run can be substantially longer than one night.

Important scope rule: fix `EC_LABEL_DEPTHS_CSV` to one depth for the full
study. Mixing depths inside a single study makes the metric comparison
meaningless. Fix the depth, complete the study, then start a new study for a
different depth.

### 5A — 200-Trial Only-GVP EC Capacity Search

Notebook configuration block:

```python
TASK = "ec"
RUN_MODE = "controlled_hpo_optuna"
RECOMMENDED_RUN_SET = "custom"
MODEL_PRESET = "Only-GVP"
RUN_BATCH_ID = "ec_only_gvp_optuna_200_capacity"
SUMMARY_BASENAME = "ec_only_gvp_optuna_200_capacity"
RUN_NAME_PREFIX = "ec_only_gvp_optuna_200_capacity"

EPOCHS = 50
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"
RING_EDGE_MODE = "with_ring"

EC_LABEL_DEPTHS_CSV = "1"
EC_CONTRASTIVE_WEIGHTS_CSV = "0.0"
EC_GROUP_WEIGHTING = "structure_id"

OPTUNA_INTENSITY = "custom"
N_OPTUNA_TRIALS = 200
MAX_EPOCHS_PER_TRIAL = 50
OPTUNA_SEARCH_PRESET = "later_capacity"
OPTUNA_STUDY_NAME = "ec_only_gvp_optuna_200_capacity"
OPTUNA_STORAGE = "sqlite:////content/drive/MyDrive/DeepMzyme/optuna/ec_only_gvp_optuna_200_capacity.db"
OPTUNA_SPLIT_SEED = 42
OPTUNA_TIMEOUT_MINUTES = 0
OPTUNA_SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"

OPTUNA_LEARNING_RATE_RANGE = "5e-6,3e-4"
OPTUNA_WEIGHT_DECAYS_CSV = "0.0,1e-6,1e-5,1e-4"
OPTUNA_BATCH_SIZES_CSV = "4,8"

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

### 5B — 200-Trial GVP + Late-Fusion EC Search

Run this only after ESM coverage is valid and simpler baselines justify ESM
fusion for EC.

Notebook configuration block:

```python
TASK = "ec"
RUN_MODE = "controlled_hpo_optuna"
RECOMMENDED_RUN_SET = "custom"
MODEL_PRESET = "GVP + late fusion"
RUN_BATCH_ID = "ec_late_fusion_optuna_200_controlled"
SUMMARY_BASENAME = "ec_late_fusion_optuna_200_controlled"
RUN_NAME_PREFIX = "ec_late_fusion_optuna_200"

EPOCHS = 50
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"
RING_EDGE_MODE = "with_ring"

EC_LABEL_DEPTHS_CSV = "1"
EC_CONTRASTIVE_WEIGHTS_CSV = "0.0"
EC_GROUP_WEIGHTING = "structure_id"

ESM_EMBEDDINGS_DIR = ""  # set to your embeddings folder when available
ALLOW_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_ESM_EMBEDDINGS = True

OPTUNA_INTENSITY = "custom"
N_OPTUNA_TRIALS = 200
MAX_EPOCHS_PER_TRIAL = 50
OPTUNA_SEARCH_PRESET = "custom"
OPTUNA_STUDY_NAME = "ec_late_fusion_optuna_200_controlled"
OPTUNA_STORAGE = "sqlite:////content/drive/MyDrive/DeepMzyme/optuna/ec_late_fusion_optuna_200_controlled.db"
OPTUNA_SPLIT_SEED = 42
OPTUNA_TIMEOUT_MINUTES = 0
OPTUNA_SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"

OPTUNA_LEARNING_RATE_RANGE = "5e-6,2e-4"
OPTUNA_WEIGHT_DECAYS_CSV = "0.0,1e-6,1e-5,1e-4"
OPTUNA_BATCH_SIZES_CSV = "4,8"

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

Expected outputs/files:

- Large persistent SQLite Optuna study in Drive
- Complete Optuna CSV/JSON/Markdown outputs under `<RUNS_DIR>/optuna/`
- Per-trial run directories and logs
- No held-out test report

Success criteria:

- Search space preview shows fixed EC label depth and group weighting.
- No test-report files are created by the HPO runs.
- Top trials improve or clarify EC validation behavior without relying on one
  lucky seed.

Decision after this stage:

- Choose the top 2-3 candidates for seed-repeat validation.
- Do not finalize the model from the raw 200-trial ranking alone.

## Stage 6 — Top-K Seed-Repeat Validation

Purpose: confirm whether top HPO candidates are stable across random seeds.

When to use it: after a medium or large Optuna search has produced top
candidates.

Expected scale/runtime: serious run, long or overnight.

```python
RUN_MODE = "controlled_hpo_optuna"
RECOMMENDED_RUN_SET = "custom"
RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = True
TOP_K_CONFIGS_FOR_SEED_REPEAT = 3
REPEAT_SEEDS = "42,123,2026,43,44"
ALLOW_SEED_REPEAT_MODEL_PRESET_MISMATCH = False
RETRAIN_BEST_CONFIG_AFTER_HPO = False

EPOCHS = 50
SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"
OPTUNA_SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

If the Optuna study is already complete and `RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION`
was not set, inspect:

```text
<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/top_reevaluation_commands.txt
```

Those commands omit held-out test evaluation. Run only the top-K commands you
predeclare, with the seed list you predeclare.

Expected outputs/files:

- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/seed_repeat_results.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/seed_repeat_summary.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/seed_repeat_summary.json`
- One validation-only run directory per top-K/seed pair
- No `test_report.json`

Success criteria:

- All top-K/seed runs complete or failures are documented.
- The selected candidate has the best validation mean or an acceptable
  mean/variance tradeoff on `val_ec_group_level_1_balanced_acc`.
- Diagnostics do not show missing validation EC classes or invalid split
  coverage.

Decision after this stage:

- Select one final configuration using validation evidence only.
- Record the mean validation score, variability, per-class diagnostics, split,
  seed list, depth, and epoch budget.
- Only then move to Stage 7.

## Stage 7 — Final Held-Out Test Evaluation

Purpose: report held-out test performance for the final validation-selected
EC configuration.

When to use it: only after model family, hyperparameters, EC label depth,
contrastive weight, checkpoint-selection metric, seed-repeat interpretation, and
final source run are fixed.

**EC split policy reminder**: the non-overlapped PinMyMetal split is mandatory
for final EC held-out reporting. Do not report the exact/possibly-overlapped
split result as the main final EC result.

First run the **Select final run and show saved outputs** cell:

```python
FINAL_RUN_SELECTION_MODE = "auto_best_validation"
FINAL_RUN_TABLE_INDEX = 1
FINAL_RUN_DIR = ""
FINAL_REPORT_BASENAME = "deepmzyme_ec_final_selected_run"
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
FINAL_TEST_RUN_NAME_PREFIX = "ec_final_test"
FINAL_TEST_BATCH_SUMMARY_BASENAME = "ec_final_test_batch_summary"
FINAL_TEST_METAL_REPORT_VIEW = "use_METAL_REPORT_VIEW"
ALLOW_REPEAT_FINAL_TEST_EVAL = False
ALLOW_MIXED_FINAL_TEST_BATCH = False
```

Inspect the pre-flight checklist. If the selected source run is the final
validation-selected run and this is final reporting, switch to launch:

```python
FINAL_TEST_WORKFLOW = "evaluate_selected_checkpoint"
LAUNCH_FINAL_HELD_OUT_TEST_EVAL = True
CONFIRM_ONE_SHOT_POLICY = True
FINAL_TEST_SOURCE_RUN_DIR = ""
FINAL_TEST_SOURCE_CHOICE_INDEX = 0
FINAL_TEST_RUN_NAME_PREFIX = "ec_final_test"
FINAL_TEST_METAL_REPORT_VIEW = "use_METAL_REPORT_VIEW"
ALLOW_REPEAT_FINAL_TEST_EVAL = False
ALLOW_MIXED_FINAL_TEST_BATCH = False
```

Expected outputs/files:

- A new final-test run folder under the resolved `RUNS_DIR`
- `test_report.json` in the final-test output folder
- Updated final-test summary CSV/PNG
- The source validation run remains unchanged

Success criteria:

- The source run has validation-selected checkpoint metadata.
- The final-test run uses `best_model_checkpoint.pt`.
- The output folder is separate from the source validation run.
- The test report includes EC level-1 metrics (and deeper levels when trained).
- The split in `dataset_summary.json` confirms non-overlapped PinMyMetal.

Decision after this stage:

- Report final held-out EC metrics: level-1 balanced accuracy, macro F1, and
  per-class recall across all seven EC first-digit classes.
- If deeper EC levels were trained, also report the depth-matched metrics.
- Do not choose a different configuration because another candidate has a
  better held-out test score. If more development is needed, return to
  validation-only experiments and treat this test result as already spent.

## EC Label Depth Progression

After the depth-1 baseline is stable and the final test result is reported, a
separate depth-2 experiment cycle starts from Stage 1 using the same structure
but:

```python
EC_LABEL_DEPTHS_CSV = "2"
SELECTION_METRIC = "val_ec_group_level_2_balanced_acc"
OPTUNA_SELECTION_METRIC = "val_ec_group_level_2_balanced_acc"
RUN_BATCH_ID = "ec_only_gvp_depth2_baseline_lr_seed"
```

Keep depth-1 and depth-2 run batches separate. Do not compare depth-1 and
depth-2 validation metrics directly.

## Contrastive Loss Exploration

After a clean depth-1 baseline is established with `EC_CONTRASTIVE_WEIGHTS_CSV = "0.0"`,
run a narrow controlled comparison with contrastive loss enabled. This is a
Stage 2-style manual comparison, not an Optuna stage:

```python
EC_CONTRASTIVE_WEIGHTS_CSV = "0.0,0.1,0.5"
EC_CONTRASTIVE_TEMPERATURE = 0.1
RUN_BATCH_ID = "ec_contrastive_comparison"
SUMMARY_BASENAME = "ec_contrastive_comparison"
```

Keep the architecture fixed to the best depth-1 Only-GVP or fusion anchor.
Compare by `val_ec_group_level_1_balanced_acc` only.

## Safety Guards To Check

Before any reportable comparison or HPO launch, confirm:

- `INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False`
- `FINAL_TEST_WORKFLOW = "preview_only"` or the final-test cell has not been run
- `VAL_FRACTION > 0` or a fold split is explicitly configured
- `SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"` (or matching depth level)
- `EC_GROUP_WEIGHTING = "structure_id"`
- `EC_LABEL_DEPTHS_CSV` is fixed to one depth value per study
- `ALLOW_SHORT_TRAINING_FOR_DEBUG = False` for reportable runs
- `ALLOW_SEED_REPEAT_MODEL_PRESET_MISMATCH = False`
- `ALLOW_MIXED_FINAL_TEST_BATCH = False`
- `ALLOW_REPEAT_FINAL_TEST_EVAL = False`
- Final test uses the non-overlapped PinMyMetal split
  (`DATASET_NAME = "train_and_test_sets_structures_non_overlapped_pinmymetal"`)
