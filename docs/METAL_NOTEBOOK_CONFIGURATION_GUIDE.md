# Metal Notebook Configuration Guide

This guide explains how to use `notebooks/DeepMzyme_training_colab.ipynb` to find a reliable metal-classification configuration before moving to broader DeepMzyme experiments.

The current project policy is baseline-first: use validation metrics for all model, checkpoint, hyperparameter, and architecture decisions; reserve the held-out test set for final reporting only.

## Scope Of This Guide

This guide describes stable workflow principles. Exact current experiment results and hyperparameters should not be duplicated here; those belong with the run evidence and current-status notes. For the current project state, read `EXPERIMENT_STATUS.md` if present. For raw copied notebook outputs, inspect `docs/notebook_outputs/raw/`; for concise run summaries, start with `docs/notebook_outputs/summaries/`.


For staged, copy-paste-ready notebook configuration blocks, use
`docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`. This guide explains option meaning
and safe workflow principles; the playbook is the practical execution recipe.

### Stage-to-option crosswalk

| Playbook stage | Notebook variables most relevant to the stage | This guide's section to read |
| --- | --- | --- |
| Stage 0 | `DATASET_NAME`, `RING_EDGE_MODE`, `RING_EXE_PATH`, `ESM_EMBEDDINGS_DIR`, `ALLOW_MISSING_EXTERNAL_FEATURES`, `RUNS_DIR` | "Starting Point", "RING options", "ESM options" |
| Stage 1 | `RUN_MODE`, `RECOMMENDED_RUN_SET`, `EPOCHS`, `LAUNCH_PLANNED_TRAINING_RUNS` | "Minimal smoke test" |
| Stage 2A | `RECOMMENDED_RUN_SET="only_gvp_broad_comparison"`, `EPOCHS`, `SEEDS_CSV`, `LEARNING_RATES_CSV` | "First real baseline" |
| Stage 2B | `RECOMMENDED_RUN_SET="baseline_model_comparison"` | "Recommended model order" |
| Stage 3/4/5 | `RUN_MODE="controlled_hpo_optuna"`, `MODEL_PRESET`, `OPTUNA_*` | "Optuna storage and seed repeats" |
| Stage 6 | `RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION`, `TOP_K_CONFIGS_FOR_SEED_REPEAT`, `REPEAT_SEEDS` | "Optuna storage and seed repeats" |
| Stage 7 | `FINAL_TEST_WORKFLOW`, `CONFIRM_ONE_SHOT_POLICY`, `FINAL_TEST_SOURCE_*` | "How To Decide The Current Best Configuration" -> "For final reporting" |

## Current Status Pointer

This guide is not the source of truth for the latest best run or next experiment. Read `EXPERIMENT_STATUS.md` first for the current recommendation, then use this guide only to configure the notebook safely.

Do not rerun old "first baseline" or Optuna examples just because they appear below. Those sections are stable workflow examples; the current metal-task recommendation may already be later in the sequence.

When planning a new check or fresh Optuna sweep, use previous raw outputs only
as context unless the user explicitly asks to rely on prior runs/results/raws.
For the practical fresh-run default, prefer the broadest sensible
validation-only Optuna search within one selected `MODEL_PRESET`; the current
copy-paste blocks for that are in `docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`.

## Exact Pipeline Source And Notebook Cell Order

Use the metal playbook for exact values. This guide explains how to interpret
the notebook controls and what must stay fixed for a fair comparison.

Recommended notebook usage:

1. Open `notebooks/DeepMzyme_training_colab.ipynb` and run setup, repo, and
   bundle cells.
2. Paste exactly one stage block from `METAL_TRAINING_PIPELINE_PLAYBOOK.md` into
   the Main configuration cell. Stages 0-7 cover the entire pipeline.
3. Run Build CONFIG, planning, and preflight cells. Inspect the resolved
   configuration table.
4. Set `LAUNCH_PLANNED_TRAINING_RUNS = True` only when the planned commands
   match the intended stage.
5. Run the summarize/report cell for the current `RUN_BATCH_ID`.
6. After Stage 5, inspect `top_trials.csv` and the Optuna summary, then run
   Stage 6 (top-K x 5-seed validation).
7. Only after Stage 6, run the Select final run cell, preview Stage 7 in
   `preview_only` mode, then launch Stage 7 with `CONFIRM_ONE_SHOT_POLICY =
   True` exactly once.

The notebook is intentionally staged. Smoke, baseline, HPO, seed-repeat, and
final-test settings should not be mixed in one batch folder unless the batch is
explicitly labeled as mixed and not used for model selection.

## G4-Oriented Training Profile

The exact G4-class Optuna budgets, sampler settings, storage URLs, and search
spaces live in `METAL_TRAINING_PIPELINE_PLAYBOOK.md` under "G4-Class Optuna
Policy". This guide does not duplicate them. The high-level posture is:
persistent SQLite Optuna in Drive, multivariate/group TPE, one `MODEL_PRESET`
per study, validation-only objective, and >= 5-seed confirmation before any
held-out test.

## Starting Point

Use the legacy **Non-overlapped PinMyMetal** split for current benchmark continuity:

- `DATASET_NAME = "train_and_test_sets_structures_non_overlapped_pinmymetal"`
- `TASK = "metal"`
- `VAL_FRACTION = 0.15`
- `SPLIT_BY = "pdbid"`
- `SELECTION_METRIC = ""`, which defaults to `val_metal_balanced_acc` for the metal task
- `INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False`
- Set a visible `RUN_BATCH_ID` for each real comparison batch. The notebook
  writes into that batch folder when `RUN_BATCH_ID` is set; use the stage block
  in the playbook for the canonical name.

The trusted final split for current metal evidence is the legacy Non-overlapped PinMyMetal split. Harsh Split PinMyMetal moves every common exact-split PDB ID to test as a whole group; use it only as an explicitly labeled new comparison. Metal Split PinMyMetal follows the exact PinMyMetal split for available supported structures; results from it, if used later, must be labeled as secondary/possibly-overlapped reference results. Common-PDBID 70/30 Split PinMyMetal is a custom comparison split, not the trusted final held-out split.

Do not choose configurations from old mixed run folders unless you have verified that every run in the folder belongs to the same comparison, same task, same split, same epoch budget, and compatible model family. The notebook summary prints whether it is scanning only the current `RUN_BATCH_ID` folder or a broader `RUNS_DIR`, and it warns strongly when old or mixed run directories may be present.

## What To Run First

### 1. Minimal smoke test

Use this to confirm that Colab setup, data paths, CSV detection, graph construction, and training execution work.

Use the Stage 1 block in `METAL_TRAINING_PIPELINE_PLAYBOOK.md`; the playbook is
the source of truth for exact smoke-test values. Then run the planning cells and
launch only after the planned command matches Stage 1.

Interpretation: a 1-epoch smoke run is not a performance result. It only proves the notebook and training loop run end to end. The notebook blocks accidental 1-3 epoch non-smoke launches unless `ALLOW_SHORT_TRAINING_FOR_DEBUG=True` is set deliberately.

### 2. First real baseline

After the smoke test succeeds, run a real Only-GVP baseline with the notebook's
default RING-enabled graph construction and strict updated external features.
Use Stage 2A in `METAL_TRAINING_PIPELINE_PLAYBOOK.md` for the exact baseline
block. Treat it as the first useful validation ranking for the structure-only
baseline, and do not copy older numeric examples from this guide.

### 3. Recommended model order

Use this order for metal:

1. `Only-GVP`, RING-enabled, no ESM.
2. `Only-ESM`, after ESM embeddings are present.
3. `GVP + late fusion`, after both structure-only and ESM-only baselines are stable.
4. `GVP + early fusion`, if late fusion or ESM-only looks promising.
5. Advanced fusion only if simple baselines justify it: `GVP + node-level late fusion`, `GVP + hybrid fusion`, and `GVP + cross-modal attention`.
6. `SimpleGNN + ESM` as an auxiliary architecture ablation, not as the first best-pipeline candidate.

The corresponding notebook presets are:

- `baseline_model_comparison`: `Only-GVP`, `Only-ESM`, `GVP + late fusion`.
- `esm_ready_comparison`: `Only-ESM`, `GVP + late fusion`, `GVP + early fusion`.
- `full_model_comparison`: all eight model presets. Use this only as a late-stage broad comparison after staged simpler comparisons justify advanced fusion.

### 4. Sequential anchor protocol

Do not treat each more complex model as a fresh, unrelated search. Use the best stable simpler model as the starting anchor for the next stage, then retune narrowly.

Recommended protocol:

1. Tune `Only-GVP` first using validation metrics only.
2. Select a stable `Only-GVP` anchor from validation evidence across seeds, not from a single lucky run.
3. Carry forward shared settings from that anchor when testing `GVP + late fusion`: split, epoch budget, seed list, graph radius, GVP capacity, class-weighting policy, and the selection metric.
4. Retune only the settings likely affected by adding ESM, such as learning rate, weight decay, dropout/fusion dimension, and possibly batch size.
5. Use the same idea for `GVP + early fusion`: start from the best validated late-fusion or ESM-informed baseline, then run a narrow validation-only comparison.
6. Move to advanced fusion only if simpler graph-plus-ESM models give a real validation benefit over the simple baselines.

This is a starting-anchor rule, not a freeze-everything rule. The best `Only-GVP` learning rate, regularization, or class weighting may change after ESM is added, so each added complexity still needs a small controlled validation search.

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

### Advanced fusion policy

`GVP + node-level late fusion`, `GVP + hybrid fusion`, and `GVP + cross-modal attention` are not recommended as part of the first best-pipeline search. Treat them as later ablations after simpler models have earned the extra complexity.

Run advanced fusion only when all of the following are true:

- `Only-GVP` has a stable validation baseline across multiple seeds.
- `Only-ESM` has been measured with valid ESM embeddings.
- `GVP + late fusion` has been compared against both simple baselines under the same split, epoch budget, seed-repeat list, and selection metric.
- Late fusion or another simple ESM-informed model gives enough validation benefit to justify testing more expressive fusion.

Suggested advanced-fusion order:

1. `GVP + node-level late fusion`: first advanced option, because it adds residue-level ESM/node-state interaction after graph message passing without changing graph inputs.
2. `GVP + hybrid fusion`: test only if early fusion or late fusion looks useful, because it combines early residue-level ESM injection with late graph-level ESM fusion.
3. `GVP + cross-modal attention`: most expressive and easiest to over-tune; keep it last unless there is a specific reason to test attention earlier.

Do not run `full_model_comparison` as the first serious architecture comparison. It mixes simple and advanced models before the simple anchor is established, which makes the result harder to interpret.

If cross-attention is tested, keep the first search narrow:

| Option | Starting value |
| --- | --- |
| `MODEL_PRESET` | `GVP + cross-modal attention` |
| `CROSS_ATTENTION_LAYERS_CSV` | `1` |
| `CROSS_ATTENTION_HEADS_CSV` | `4` |
| `CROSS_ATTENTION_DROPOUT` | `0.1` |
| `CROSS_ATTENTION_NEIGHBORHOOD` | `first_second_shell` or `all` |
| `CROSS_ATTENTION_BIDIRECTIONAL` | `False` first |

Compare cross-attention against the best validation-selected late-fusion model, not against an untuned or mismatched baseline. Use validation metrics only for the decision, and do not broaden layers, heads, bidirectionality, or neighborhoods until the narrow run beats the simpler model consistently across seeds.

For node-level late fusion and hybrid fusion, inherit the same graph anchor and comparison rules:

- Keep the validated `Only-GVP` graph settings fixed at first.
- Keep the same split, epoch budget, seed-repeat list, and selection metric as late fusion.
- Retune only a narrow set of ESM/fusion-sensitive settings first.
- Compare against the best validation-selected `GVP + late fusion`, not only against `Only-GVP`.

`SimpleGNN + ESM` is not an advanced GVP fusion mode. Use it later as an ablation to ask whether GVP vector geometry is helping compared with a simpler scalar graph + ESM model. It should not replace the `Only-GVP`, `Only-ESM`, and `GVP + late fusion` baseline sequence.

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

- `RING_EDGE_MODE = "with_ring"`
- `REQUIRE_RING_EDGES = False`
- `PREPARE_MISSING_RING_EDGES = True`
- `ALLOW_MISSING_EXTERNAL_FEATURES = False`

The notebook now starts from RING-enabled graph construction and strict updated
external features by default. Existing RING files are reused; missing files are
generated when a working `RING_EXE_PATH` is available. If `REQUIRE_RING_EDGES =
True`, incomplete RING coverage should fail instead of silently mixing graph
types. To run a radius-only ablation, set `RING_EDGE_MODE = "without_ring"`.

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

Keep `VAL_FRACTION > 0` for reportable comparisons. If `VAL_FRACTION = 0`, the training CLI falls back to `train_loss`, which is not a valid basis for model selection. The notebook prints a metal split diagnostic showing whether every metal class is present in train and validation; if any class is missing from validation, that run is not suitable for reportable model selection.

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

- Use `OPTUNA_INTENSITY = "custom"` when you want exact control over the
  budget. The playbook uses this for G4-oriented medium and 200-trial searches.
- Use persistent SQLite storage in Drive, for example:
  `sqlite:////content/drive/MyDrive/DeepMzyme/optuna/deepmzyme_metal_only_gvp.db`
- Keep `OPTUNA_SELECTION_METRIC` blank or set it to `val_metal_balanced_acc`.
- Keep `OPTUNA_DIRECTION = "maximize"`.
- Keep `OPTUNA_TPE_MULTIVARIATE = True` and `OPTUNA_TPE_GROUP = True` so TPE
  can model correlated parameters such as hidden width, vector width, graph
  depth, and fusion dimension.
- Set `OPTUNA_N_STARTUP_TRIALS` below `N_OPTUNA_TRIALS`. If startup trials are
  greater than or equal to total trials, the run is effectively random search.
  Use about `20` startup trials for a 64-trial medium search and about `40` for
  a 200-trial wide search.
- Keep `OPTUNA_AUTO_CONFIGURE_BUDGET = False` when using a playbook block with
  explicit trial counts. If enabled, the notebook may raise trial counts to the
  advisor's minimum recommendation.
- Keep `OPTUNA_USE_PRUNING = False` for now. The notebook launches
  `src/train.py` as subprocess trials and does not currently report
  intermediate values back to Optuna, so pruner settings are not an effective
  early-stopping mechanism in this path.
- Use `RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = True` only after you are ready to rerun the top configurations across seeds.
- Use the playbook Stage 6 `REPEAT_SEEDS` value for project-standard metal
  confirmation, unless a smaller exploratory check is explicitly labeled.

Numeric Optuna budgets are defined per stage in
`METAL_TRAINING_PIPELINE_PLAYBOOK.md`. Use `OPTUNA_INTENSITY = "custom"` for
every reportable run on the G4 GPU.

`OPTUNA_SEARCH_PRESET = "first_useful_only_gvp_narrow"` keeps architecture/capacity fixed and varies mainly learning rate, weight decay, batch size, and metal class-weight mode. Use it for the first controlled HPO path or for explicit anchor continuation. For a user-requested fresh broad Optuna check, use the playbook's large-search blocks and expand capacity/search axes within one selected model family instead of over-narrowing to old raw outputs. Short HPO trials mostly rank early-training behavior.

## Professional Configuration Search Strategy

Use this controlled sequence:

1. Run Stage 0 and Stage 1 from the playbook. Ignore smoke metrics except for
   obvious failures.
2. Run Stage 2A and Stage 2B as validation-only baselines. Choose by
   `val_metal_balanced_acc`, not test.
3. Run Stage 3 before the first Optuna batch in a new runtime.
4. Run Stage 4 or Stage 5 inside one selected `MODEL_PRESET`. Use narrower
   ranges only when deliberately continuing from an anchor; use the playbook's
   broader large-search blocks when the user asks for a fresh broad check.
5. Run Stage 6 seed-repeat validation before treating any HPO candidate as
   stable.
6. Advance to Stage 5D/5E/5F only if the playbook's advanced-fusion gate is
   satisfied.
7. Select one final configuration by validation evidence.
8. Use Stage 7 once for that selected configuration.

For the current metal task, check `EXPERIMENT_STATUS.md` before starting at any
numbered step. If the current validation anchor is already recorded there, use
the status file's next planned action rather than restarting from an older
workflow example in this guide.

When comparing configurations, keep the comparison clean:

- Same `TASK`.
- Same split policy.
- Same `EPOCHS`.
- Same seed-repeat list.
- Same final selection metric.
- Same test-set policy.
- Same anchor configuration for shared settings when comparing a simple model to the next more complex model.

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

In Colab, the notebook now mounts Google Drive by default and resolves a blank
`RUNS_DIR` to `<DRIVE_ROOT>/notebook_outputs/runs` when Drive is available. The
final held-out test cell uses the same resolved `RUNS_DIR`, so final-test output
folders are saved beside the validation/HPO/seed-repeat run folders. If Drive
cannot be mounted, the notebook prints a warning and falls back to local Colab
storage.

The "Optional final held-out test evaluation" cell has two different folder
roles:

- **Source run folder**: the validation run that supplies the saved config and
  `best_model_checkpoint.pt`. In single-checkpoint mode, a blank
  `FINAL_TEST_SOURCE_RUN_DIR` means "use `selected_final_run_dir` from the
  previous Select final run cell" when `FINAL_TEST_SOURCE_CHOICE_INDEX = 0`. If
  `FINAL_TEST_SOURCE_RUN_DIR` is filled, the cell uses that folder instead. If
  `FINAL_TEST_SOURCE_CHOICE_INDEX` is positive, the cell uses that numbered row
  from its printed source-run picker table.
- **Batch parent folder**: the folder scanned in `evaluate_all_seeds_batch`
  mode. A filled `FINAL_TEST_BATCH_PARENT_DIR` is used directly. If it is blank,
  the cell uses the parent folder of the selected source run.
- **Final-test output folder**: a new run folder written under the same
  `RUNS_DIR`, named from `FINAL_TEST_RUN_NAME_PREFIX` plus the source run name.
  The original validation run folder is not overwritten.

The final-test cell prints a pre-flight checklist before launch. For a single
source run, it checks that validation selection exists, the checkpoint exists,
the held-out test paths exist, repeat policy is satisfied, and the output folder
is separate from the source run. For batch mode, it also checks that all source
runs share the same `task`, `model_architecture`, `fusion_mode`, and
`selection_metric`. Leave `ALLOW_MIXED_FINAL_TEST_BATCH = False` unless a mixed
folder is being evaluated deliberately.

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

For metal reports, `METAL_REPORT_VIEW` controls which already-computed metrics
are emphasized in the notebook tables: `six_class`, `collapsed4`, or `both`.
The final held-out test cell also has `FINAL_TEST_METAL_REPORT_VIEW`; its
default `use_METAL_REPORT_VIEW` follows the main notebook setting. These are
display/reporting controls only. They do not change the model targets, training
loss, checkpoint-selection metric, or held-out test policy.

`FINAL_TEST_BATCH_METRICS` controls only which metric columns are emphasized in
batch final-test summaries and plots. It does not change which metrics are
computed or saved in `test_report.json`.

Metal evaluation always keeps the six-class prediction problem
`Mn`, `Cu`, `Zn`, `Fe`, `Co`, `Ni`. For every metal or joint test report, the
code also computes collapsed-4 metrics by merging `Fe`, `Co`, and `Ni` into
`Class VIII`, giving `Mn`, `Cu`, `Zn`, and `Class VIII`. Use the toggle to choose
which view is emphasized in notebook output, not to rerun a different test.

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
3. Prefer `FINAL_TEST_WORKFLOW = "evaluate_selected_checkpoint"` unless you intentionally want a retrain from the selected config.
4. Record `test_metal_balanced_acc`, `test_metal_macro_f1`, `test_metal_per_class_recall`, `test_metal_collapsed4_balanced_acc`, and `test_metal_collapsed4_per_class_recall` from `test_report.json`.
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
4. RING is now the default first graph setting. Use the playbook's radius-only ablation block only when you intentionally want to reproduce the older graph construction.
5. `fusion_mode` may appear in saved `Only-GVP` configs even though it is irrelevant for `only_gvp`; reporting could display it as `none` for clarity.
6. Optuna can sample many capacity and graph parameters by default. For first useful metal HPO, the notebook could offer a narrower `Only-GVP_lr_wd_only` search-space preset.
7. The current default `OPTUNA_METAL_CLASS_WEIGHT_MODES_CSV` includes several weighting modes. That is useful later, but first HPO may be easier to interpret if class weighting is fixed initially.
8. Existing run folders can be summarized with new runs if the same `RUNS_DIR` is reused. The notebook warns about mixed summaries, but a run-batch identifier could make accidental mixing harder.
