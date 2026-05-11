# Metal Notebook Configuration Guide

This guide explains how to use `notebooks/DeepMzyme_training_colab.ipynb` to find a reliable metal-classification configuration before moving to broader DeepMzyme experiments.

The current project policy is baseline-first: use validation metrics for all model, checkpoint, hyperparameter, and architecture decisions; reserve the held-out test set for final reporting only.

## Starting Point

Use the non-overlapped PinMyMetal split:

- `DATASET_NAME = "train_and_test_sets_structures_non_overlapped_pinmymetal"`
- `TASK = "metal"`
- `VAL_FRACTION = 0.15`
- `SPLIT_BY = "pdbid"`
- `SELECTION_METRIC = ""`, which defaults to `val_metal_balanced_acc` for the metal task
- `INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False`

The trusted final split for metal is the non-overlapped PinMyMetal split. Exact PinMyMetal split results, if used later, must be labeled as secondary/possibly-overlapped reference results.

Do not choose configurations from old mixed run folders unless you have verified that every run in the folder belongs to the same comparison, same task, same split, same epoch budget, and compatible model family.

## What To Run First

### 1. Minimal smoke test

Use this to confirm that Colab setup, data paths, CSV detection, graph construction, and training execution work.

Set:

| Option | Value |
| --- | --- |
| `TASK` | `metal` |
| `RUN_MODE` | `single` |
| `RECOMMENDED_RUN_SET` | `only_gvp_smoke` |
| `MODEL_PRESET` | `Only-GVP` |
| `EPOCHS` | `1` |
| `BATCH_SIZES_CSV` | `4` |
| `LEARNING_RATES_CSV` | `3e-5` |
| `WEIGHT_DECAYS_CSV` | `1e-4` |
| `SEEDS_CSV` | `42` |
| `RING_EDGE_MODE` | `without_ring` |
| `INCLUDE_HELD_OUT_TEST_DURING_TRAINING` | `False` |

Then run the planning cells. In the optional training execution cell, set `LAUNCH_PLANNED_TRAINING_RUNS = True`.

Interpretation: a 1-epoch smoke run is not a performance result. It only proves the notebook and training loop run end to end.

### 2. First real baseline

After the smoke test succeeds, run a real Only-GVP radius-only baseline. Start with the notebook's baseline-first GVP settings:

| Option | Value |
| --- | --- |
| `RUN_MODE` | `manual_configurations` |
| `RECOMMENDED_RUN_SET` | `only_gvp_lr_seed` |
| `EPOCHS` | `30` or `50` |
| `BATCH_SIZES_CSV` | `4` initially in Colab; `8` if GPU memory is stable |
| `WEIGHT_DECAYS_CSV` | `1e-4` |
| `SELECTION_METRIC` | blank or `val_metal_balanced_acc` |
| `INCLUDE_HELD_OUT_TEST_DURING_TRAINING` | `False` |

`only_gvp_lr_seed` runs `Only-GVP`, radius-only, with learning rates `3e-5` and `1e-4` across seeds `42` and `43`.

If those runs are stable and time permits, run:

| Option | Value |
| --- | --- |
| `RECOMMENDED_RUN_SET` | `only_gvp_broad_comparison` |
| `EPOCHS` | `30` or `50` |

This expands to learning rates `3e-5`, `1e-4`, and `3e-4` across seeds `42`, `43`, and `44`. Treat this as the first useful validation ranking for the structure-only baseline.

### 3. Recommended model order

Use this order for metal:

1. `Only-GVP`, radius-only, no ESM.
2. `Only-ESM`, after ESM embeddings are present.
3. `GVP + late fusion`, after both structure-only and ESM-only baselines are stable.
4. `GVP + early fusion`, if late fusion or ESM-only looks promising.
5. Advanced fusion only if simple baselines justify it: `GVP + node-level late fusion`, `GVP + hybrid fusion`, `GVP + cross-modal attention`, and `SimpleGNN + ESM`.

The corresponding notebook presets are:

- `baseline_model_comparison`: `Only-GVP`, `Only-ESM`, `GVP + late fusion`.
- `esm_ready_comparison`: `Only-ESM`, `GVP + late fusion`, `GVP + early fusion`.
- `full_model_comparison`: all eight model presets. Use this only after cheaper comparisons are stable.

## Important Notebook Options

### Run mode

`RUN_MODE` controls the workflow:

- `single`: one resolved configuration. Use it for smoke tests and manually selected retrains.
- `manual_configurations`: controlled Cartesian product from CSV fields or from `RECOMMENDED_RUN_SET`. Use this for small planned comparisons.
- `controlled_hpo_optuna`: controlled Optuna within one selected base model. It does not run unrestricted AutoML and does not evaluate the held-out test set inside trials.

### `RECOMMENDED_RUN_SET` and `MODEL_PRESET`

`RECOMMENDED_RUN_SET` is the run-plan preset. Named run sets can override `MODEL_PRESET`, learning rates, seeds, RING mode, and architecture-grid values.

`MODEL_PRESET` is the architecture/fusion choice. It is active when `RECOMMENDED_RUN_SET = "custom"` and is the safest way to choose the one model optimized by `controlled_hpo_optuna`.

For Optuna, use:

| Option | Value |
| --- | --- |
| `RUN_MODE` | `controlled_hpo_optuna` |
| `RECOMMENDED_RUN_SET` | `custom` |
| `MODEL_PRESET` | one model only, usually `Only-GVP` first |

Do not set `RECOMMENDED_RUN_SET = "full_model_comparison"` and expect Optuna to optimize architectures. In the current notebook, Optuna uses one base configuration.

### Model architecture and fusion

`MODEL_PRESET` maps to CLI behavior:

| Preset | Meaning | ESM needed |
| --- | --- | --- |
| `Only-GVP` | structure graph only, `--model-architecture only_gvp` | No |
| `Only-ESM` | ESM embeddings only | Yes |
| `GVP + late fusion` | graph and ESM pooled separately, fused near head | Yes |
| `GVP + early fusion` | residue ESM injected before message passing | Yes |
| `GVP + node-level late fusion` | ESM merged into node states after message passing | Yes |
| `GVP + hybrid fusion` | both early and late ESM fusion | Yes |
| `GVP + cross-modal attention` | advanced graph/ESM attention fusion | Yes |
| `SimpleGNN + ESM` | non-GVP graph plus ESM comparison | Yes |

For `Only-GVP`, fusion fields are effectively irrelevant even if a saved config displays `fusion_mode = late_fusion`.

The main capacity fields are:

- `HIDDEN_S_VALUES_CSV`: scalar hidden width for GVP/GNN states. Keep `128` first.
- `HIDDEN_V_VALUES_CSV`: vector hidden width for GVP models. Keep `16` first.
- `EDGE_HIDDEN_VALUES_CSV`: edge-feature hidden width. Keep `64` first.
- `GVP_LAYERS_VALUES_CSV`: graph message-passing depth. Keep `4` first.
- `HEAD_MLP_LAYERS_VALUES_CSV`: classifier-head depth. Keep `2` first.
- `EDGE_RADIUS_VALUES_CSV`: graph radius cutoff. Keep `8.0` first.

Do not vary all capacity fields at once in the first baseline. Use `only_gvp_architecture_grid` or `only_gvp_geometry_grid` only after the simpler learning-rate and seed behavior is understood.

### ESM options

For the first metal baseline:

- Use `MODEL_PRESET = "Only-GVP"`.
- Leave `ESM_EMBEDDINGS_DIR` blank unless you already have embeddings.
- Keep `ALLOW_MISSING_ESM_EMBEDDINGS = False`.
- Keep `PREPARE_MISSING_ESM_EMBEDDINGS = False`.

For ESM or fusion runs:

- Provide `ESM_EMBEDDINGS_DIR`, or explicitly set `PREPARE_MISSING_ESM_EMBEDDINGS = True`.
- Do not set `ALLOW_MISSING_ESM_EMBEDDINGS = True` for reportable runs. It is a debug/ablation escape hatch.

`use_early_esm` is controlled by presets: early and hybrid fusion enable it; late fusion and `Only-ESM` do not.

In notebook language, `uses_esm` means the selected preset requires ESM residue embeddings. `use_early_esm` means residue-level ESM vectors are injected into node features before graph message passing. `Only-GVP` has both off.

### RING options

Default first baseline:

- `RING_EDGE_MODE = "without_ring"`
- `REQUIRE_RING_EDGES = False`
- `PREPARE_MISSING_RING_EDGES = True` is harmless for radius-only planning, but RING files are not needed unless `RING_EDGE_MODE = "with_ring"` or `ring_comparison` is selected.

Use `ring_comparison` only after the radius-only Only-GVP baseline is stable. If `with_ring` is used, make sure existing RING files or a working `RING_EXE_PATH` are available. If `REQUIRE_RING_EDGES = True`, incomplete RING coverage should fail instead of silently mixing graph types.

### Training hyperparameters

Recommended starting values:

| Option | Smoke | First real baseline |
| --- | --- | --- |
| `EPOCHS` | `1` | `30` or `50` |
| `BATCH_SIZES_CSV` | `4` | `4`, or `8` if memory is stable |
| `LEARNING_RATES_CSV` | `3e-5` | `3e-5,1e-4`; optionally add `3e-4` |
| `WEIGHT_DECAYS_CSV` | `1e-4` | `1e-4` first; later compare `0,1e-5,1e-4` |
| `LR_SCHEDULES_CSV` | `fixed` | `fixed` first |
| `SEEDS_CSV` | `42` | at least `42,43`; preferably `42,43,44` |

Do not compare 1-epoch runs as if they are model-quality evidence.

### Validation and selection metric

For metal, optimize:

- Primary: `val_metal_balanced_acc`
- Secondary diagnostics: `val_metal_macro_f1`, `val_metal_min_recall`, `val_metal_per_class_recall`, `val_metal_collapsed4_balanced_acc`

Use balanced accuracy because the metal classes are imbalanced. Plain accuracy can look good while failing rare metals.

Keep `VAL_FRACTION > 0` for reportable comparisons. If `VAL_FRACTION = 0`, the training CLI falls back to `train_loss`, which is not a valid basis for model selection.

### Metal class weighting

Current code supports:

- `METAL_CLASS_WEIGHT_MODES_CSV = "none"`
- `METAL_CLASS_WEIGHT_MODES_CSV = "inverse_frequency"`
- `METAL_CLASS_WEIGHT_MODES_CSV = "inverse_sqrt_frequency"`
- `METAL_CLASS_WEIGHT_MODES_CSV = "effective_number"`
- `BALANCE_METAL_SITE_SYMBOLS = True` or `False`
- `METAL_LOSS_FUNCTION = "cross_entropy"` or `"focal"`

Start cautiously:

1. Use the source-code/notebook default `inverse_frequency` for the first baseline, because existing DeepMzyme runs used it.
2. Compare `none,inverse_frequency,inverse_sqrt_frequency,effective_number` only after the baseline is stable.
3. Keep `METAL_LOSS_FUNCTION = "cross_entropy"` first.
4. Treat `focal` and per-class loss multipliers as later ablations, not first-line defaults.
5. Do not decide class weighting from one seed.

Class weights are computed from the training split only, which is correct. Still inspect whether weighting improves rare-class recall without destroying common-class performance.

### Optuna storage and seed repeats

For debug only:

- `OPTUNA_INTENSITY = "debug"`
- Blank `OPTUNA_STORAGE` is acceptable.

For useful Colab HPO:

- `OPTUNA_INTENSITY = "first_useful"` for the first meaningful pass.
- Use persistent SQLite storage in Drive, for example:
  `sqlite:////content/drive/MyDrive/DeepMzyme/optuna/deepmzyme_metal_only_gvp.db`
- Keep `OPTUNA_SELECTION_METRIC` blank or set it to `val_metal_balanced_acc`.
- Keep `OPTUNA_DIRECTION = "maximize"`.
- Use `RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = True` only after you are ready to rerun the top configurations across seeds.
- Use `REPEAT_SEEDS = "42,123,2026"` or another fixed, predeclared seed list.

Current intensity presets:

| `OPTUNA_INTENSITY` | Effective budget | Use |
| --- | --- | --- |
| `debug` | 4 trials x 3 epochs | setup check only |
| `first_useful` | 16 trials x 20 epochs | first meaningful HPO |
| `serious` | 40 trials x 40 epochs | longer Colab HPO with persistent storage |
| `custom` | uses visible `N_OPTUNA_TRIALS` and `MAX_EPOCHS_PER_TRIAL` | deliberate manual budget |

Recommended controlled first search:

| Option | Value |
| --- | --- |
| `RUN_MODE` | `controlled_hpo_optuna` |
| `RECOMMENDED_RUN_SET` | `custom` |
| `MODEL_PRESET` | `Only-GVP` |
| `OPTUNA_INTENSITY` | `first_useful` |
| `OPTUNA_LEARNING_RATE_RANGE` | `1e-5,3e-4` |
| `OPTUNA_WEIGHT_DECAYS_CSV` | `0.0,1e-5,1e-4` |
| `OPTUNA_BATCH_SIZES_CSV` | `4,8` if memory allows |
| `OPTUNA_METAL_CLASS_WEIGHT_MODES_CSV` | initially `inverse_frequency`, then optionally broader |

Avoid broad architecture/capacity HPO until the simple baseline behavior is clear. Short HPO trials mostly rank early-training behavior.

## Professional Configuration Search Strategy

Use this controlled sequence:

1. Smoke test `only_gvp_smoke` for 1 epoch. Ignore metrics except for obvious failures.
2. Run `only_gvp_lr_seed` for 30-50 epochs. Choose by `val_metal_balanced_acc`, not test.
3. Run `only_gvp_broad_comparison` if the first baseline is stable. Confirm learning-rate and seed sensitivity.
4. Optionally run controlled Optuna on `Only-GVP` only, with narrow ranges for learning rate, weight decay, batch size, and possibly a small set of capacity values.
5. Rerun the top 2-3 validation configurations across several seeds.
6. Once ESM embeddings are available, run `baseline_model_comparison`. Compare `Only-GVP`, `Only-ESM`, and `GVP + late fusion` using the same epoch budget and selection metric.
7. Only then test early or advanced fusion modes.
8. Select one final configuration by validation evidence.
9. Use the optional final held-out test cell once for that selected configuration.

When comparing configurations, keep the comparison clean:

- Same `TASK`.
- Same split policy.
- Same `EPOCHS`.
- Same seed-repeat list.
- Same final selection metric.
- Same test-set policy.

## Metal-Class Diagnostics

Inspect these before trusting a run ranking:

- Train/validation metal class counts printed by the planning cell.
- `missing_train_metal_classes` and `missing_val_metal_classes` in summaries.
- `split_diagnostics.json` in each run directory.
- `val_metal_per_class_recall` in `run_metadata.json` or `run_config.json` history.
- `val_metal_min_recall`, `val_metal_mn_recall`, `val_metal_fe_recall`, and `val_metal_class_viii_recall`.
- `val_metal_collapsed4_balanced_acc` as a secondary view, where Fe/Co/Ni are collapsed into `VIII`.

Weak rare classes should be interpreted carefully. If a class has few validation examples, a single correct or incorrect pocket can move recall sharply. Look for consistency across seeds and check whether improvement in a weak class is bought by large degradation in common classes.

Class weights should be tested, but cautiously. A class-weighting mode that improves `val_metal_min_recall` while reducing `val_metal_balanced_acc` and `val_metal_macro_f1` may not be better overall.

## Output Files To Inspect

After planning:

- `<RUNS_DIR>/<SUMMARY_BASENAME>_planned_runs.csv`: exact planned configurations.
- `<RUNS_DIR>/<SUMMARY_BASENAME>_planned_run_dictionary.json`: full planned configuration details.
- `<RUNS_DIR>/<SUMMARY_BASENAME>_metal_weight_diagnostics.csv`: planned class-weighting diagnostics for metal runs.

After training:

- Each run directory under `RUNS_DIR`.
- `<run_dir>/run_metadata.json`: selected metric, selected checkpoint, config, history, split/test metadata.
- `<run_dir>/run_config.json`: full saved config and history.
- `<run_dir>/dataset_summary.json`: dataset and split identity summary.
- `<run_dir>/split_diagnostics.json`: train/validation counts, grouping, overlap, missing classes.
- `<run_dir>/best_model_checkpoint.pt`: checkpoint selected by validation metric.
- `<run_dir>/last_model_checkpoint.pt`: final epoch checkpoint.
- `<run_dir>/prepare_status.json`: preparation/preflight status.
- `<run_dir>/test_report.json`: only present after held-out test evaluation; do not use this for model selection.

After the summarize/report cell:

- `<RUNS_DIR>/<SUMMARY_BASENAME>_completed_only.csv`: summary generated from completed run directories.
- `<RUNS_DIR>/<SUMMARY_BASENAME>.csv`: comparison table combining planned/executed run information.
- `<RUNS_DIR>/<SUMMARY_BASENAME>.png`: comparison figure when plotting succeeds.
- `<RUNS_DIR>/<SUMMARY_BASENAME>_execution_records.json`: execution status, logs, failures.

After Optuna:

- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/all_trials.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/optuna_trials.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/top_trials.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/best_trial.json`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/optuna_best_config.json`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/best_config_command.txt`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/top_reevaluation_commands.txt`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/seed_repeat_results.csv`, if seed repeats were run
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/seed_repeat_summary.csv`, if seed repeats were run
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/optuna_study_summary.md`

## How To Decide The Current Best Configuration

For a validation-only manual comparison:

1. Open `<SUMMARY_BASENAME>.csv` and filter to `status = completed`.
2. Filter to `result_stage = validation-only` or `seed-repeat validation`, not final-test rows.
3. Filter to the intended `task = metal`, split type, epoch budget, and model comparison group.
4. Rank by `selected_best_validation_metric_value` or `best_validation_metric_used_for_checkpoint_selection`, using `selection_metric = val_metal_balanced_acc`.
5. Check diagnostics for missing validation metal classes and train/validation overlap.
6. Check per-class recall and collapsed-4 balanced accuracy as secondary evidence.
7. Prefer a configuration that is stable across seeds over one single high seed.

For Optuna:

1. Inspect `top_trials.csv`.
2. Inspect `optuna_study_summary.md`.
3. Run top-k seed-repeat validation.
4. Select by seed-repeat mean and variability on `val_metal_balanced_acc`, not by a single trial.
5. Retrain or evaluate the final selected validation-best checkpoint only after selection is complete.

For final reporting:

1. Use the notebook's "Select final run and show saved outputs" cell to choose the validation-best run.
2. Use the "Optional final held-out test evaluation" cell with `LAUNCH_FINAL_HELD_OUT_TEST_EVAL = True`.
3. Prefer `FINAL_TEST_MODE = "evaluate_selected_checkpoint"` unless you intentionally want a retrain from the selected config.
4. Record `test_metal_balanced_acc`, `test_metal_macro_f1`, `test_metal_collapsed4_balanced_acc`, and test per-class diagnostics from `test_report.json`.
5. Do not go back and choose a different configuration because its test score is better.

## Mistakes To Avoid

- Do not select models based on held-out test metrics.
- Do not enable `INCLUDE_HELD_OUT_TEST_DURING_TRAINING` for comparison, HPO, or seed-repeat runs.
- Do not compare old mixed folders silently. The current local `DeepMzyme_Data/notebook_outputs/runs/` contains older Only-GVP metal runs with held-out test reports; treat them as historical unless deliberately included.
- Do not mix incompatible `MODEL_PRESET` values in seed-repeat evaluation.
- Do not set `ALLOW_SEED_REPEAT_MODEL_PRESET_MISMATCH = True` unless you are intentionally overriding the guard.
- Do not trust one lucky seed.
- Do not over-interpret 1-epoch or 3-epoch debug results.
- Do not let missing ESM embeddings silently define an ESM baseline.
- Do not present exact/possibly-overlapped split results as the main held-out result.
- Do not use `VAL_FRACTION = 0` for reportable model selection.

## Potential Notebook Improvements

These are documentation/UX issues found during inspection. They are not implemented here.

1. `EPOCHS` defaults to `1`, which is safe for smoke tests but easy to forget before real comparisons. A stronger warning near the launch cell could reduce accidental 1-epoch rankings.
2. `RECOMMENDED_RUN_SET` can override `MODEL_PRESET`; the notebook warns about this, but it remains a common user-error point. A more visible resolved-model summary could help.
3. `INCLUDE_HELD_OUT_TEST_DURING_TRAINING` still exists as a top-level option. It is guarded, but keeping final test evaluation only in the separate final-test cell would be safer.
4. `PREPARE_MISSING_RING_EDGES = True` appears near the default radius-only settings. It may confuse users even though `without_ring` does not need RING files.
5. `fusion_mode` may appear in saved `Only-GVP` configs even though it is irrelevant for `only_gvp`; reporting could display it as `none` for clarity.
6. Optuna can sample many capacity and graph parameters by default. For first useful metal HPO, the notebook could offer a narrower `Only-GVP_lr_wd_only` search-space preset.
7. The current default `OPTUNA_METAL_CLASS_WEIGHT_MODES_CSV` includes several weighting modes. That is useful later, but first HPO may be easier to interpret if class weighting is fixed initially.
8. Existing run folders can be summarized with new runs if the same `RUNS_DIR` is reused. The notebook warns about mixed summaries, but a run-batch identifier could make accidental mixing harder.
