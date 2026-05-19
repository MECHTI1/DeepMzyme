# Metal Notebook Configuration Guide

This guide explains how to use `notebooks/DeepMzyme_training_colab.ipynb` to find a reliable metal-classification configuration before moving to broader DeepMzyme experiments.

The current project policy is baseline-first: use validation metrics for all model, checkpoint, hyperparameter, and architecture decisions; reserve the held-out test set for final reporting only.

For the cross-document validation/testing order, output-folder map, and Drive
copying policy, read `docs/README.md` first.

## Scope Of This Guide

This guide describes stable workflow principles. Exact current experiment results and hyperparameters should not be duplicated here; those belong with the run evidence and current-status notes. For the current project state, read `EXPERIMENT_STATUS.md` if present. For raw copied notebook outputs, inspect `docs/notebook_outputs/raw/`; for concise run summaries, start with `docs/notebook_outputs/summaries/`.



For staged, copy-paste-ready notebook configuration blocks, use
`docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`. This guide explains option meaning
and safe workflow principles; the playbook is the practical execution recipe.

### Stage-to-option crosswalk


| Playbook stage | Notebook variables most relevant to the stage | This guide's section to read |
| --- | --- | --- |
| Stage 0: environment/data readiness | Data source, bundle, Drive, `RUNS_DIR`, RING/ESM/external-feature path controls | "Starting Point", "RING options", "ESM options" |
| Stage 1: 1-epoch smoke | `RUN_MODE`, `RECOMMENDED_RUN_SET`, launch toggle, smoke/debug guards | "Minimal smoke test" |
| Stage 2A: Only-GVP validation anchor | Manual-comparison controls, Only-GVP preset, split/selection controls | "First real baseline", "Validation and selection metric" |
| Stage 2B: baseline family comparison | Baseline run-set controls, ESM readiness controls, comparison hygiene | "Recommended model order", "ESM options" |
| Stage 3: Optuna plumbing debug | `RUN_MODE="controlled_hpo_optuna"`, study name/storage, sampler controls, debug budget controls | "Optuna storage and Stage 6 confirmation" |
| Stage 4: medium per-family Optuna, optional on G4 | One `MODEL_PRESET`, custom Optuna settings, persistent storage, validation-only objective | "Optuna storage and Stage 6 confirmation", "Professional Configuration Search Strategy" |
| Stage 5A: serious Only-GVP HPO | One `MODEL_PRESET`, serious Optuna search controls, graph capacity controls, imbalance controls | "Model architecture and fusion", "Metal class weighting", "Optuna storage and Stage 6 confirmation" |
| Stage 5B: Only-ESM HPO | ESM path/generation controls, ESM-only preset, serious Optuna controls | "ESM options", "Optuna storage and Stage 6 confirmation" |
| Stage 5C: GVP + late fusion HPO | ESM fusion controls, serious Optuna controls, validation gate for advanced fusion | "Advanced fusion policy", "Optuna storage and Stage 6 confirmation" |
| Stage 5D: GVP + node-level late fusion HPO | Node-level fusion preset, ESM path controls, advanced-fusion gate | "Advanced fusion policy", "ESM options" |
| Stage 5E: GVP + hybrid fusion HPO | Early+late ESM controls, early ESM dimensions/dropout, advanced-fusion gate | "Advanced fusion policy", "ESM options" |
| Stage 5F: GVP + cross-attention HPO | Cross-attention controls, ESM path controls, advanced-fusion gate | "Advanced fusion policy", "Model architecture and fusion" |
| Stage 5G: RING/radius-only ablation | `RING_EDGE_MODE`, RING requirement/preparation flags, ablation labeling | "RING options" |
| Stage 6: top-K seed/split confirmation | `RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION`, `TOP_CONFIG_REEVALUATION_MODE="group_kfold"`, top-K controls, `SEED_REPEAT_N_FOLDS`, `SEED_REPEAT_SPLIT_SEED`, repeat model seed, mismatch guard | "Optuna storage and Stage 6 confirmation", "How To Decide The Current Best Configuration" |
| Stage 7: one-shot held-out test | final-run selector, preview/evaluate workflow, one-shot confirmation, repeat/mixed-batch guards | "Output Files To Inspect", "How To Decide The Current Best Configuration" |

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

For the current notebook execution order, use the playbook's "How To Use This
Playbook" section. The invariant is: paste exactly one stage block, inspect the
resolved planning table, launch only when the planned commands match the stage,
summarize the current `RUN_BATCH_ID`, run Stage 6 before any final selection,
and use Stage 7 only once after validation selection is frozen.

The notebook is intentionally staged. Smoke, baseline, HPO, grouped-fold
confirmation, and final-test settings should not be mixed in one batch folder
unless the batch is explicitly labeled as mixed and not used for model
selection.

## How This Guide Relates To Exact Stage Blocks

This guide explains notebook controls, common mistakes, and safe interpretation.
It does not own the exact current values for serious experiments.

For any real run, use the exact block from
`docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`. If a numeric value in this guide
appears to conflict with the Playbook, the Playbook wins.

Before launching a run, verify these resolved notebook values:

| Check | Required value for reportable metal runs |
| --- | --- |
| Task | `TASK = "metal"` |
| Metal label scheme | `METAL_LABEL_SCHEME = "six_class"` for the default reportable target; use `"five_class"` only for explicitly labeled validation-only comparisons |
| Split | `DATASET_NAME = "train_and_test_sets_structures_non_overlapped_pinmymetal"` |
| Validation split | `VAL_FRACTION = 0.15` |
| Split grouping | `SPLIT_BY = "pdbid"` |
| Selection metric | `SELECTION_METRIC = "val_metal_balanced_acc"` |
| Held-out test during training | `INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False` |
| Device on G4 | `DEVICE = "cuda"` |
| RING default | `RING_EDGE_MODE = "with_ring"` |
| Serious Optuna intensity | `OPTUNA_INTENSITY = "custom"` |
| Serious Optuna storage | persistent Drive SQLite `OPTUNA_STORAGE` |
| Serious Optuna sampler | `OPTUNA_TPE_MULTIVARIATE = True`, `OPTUNA_TPE_GROUP = True` |
| Serious Optuna pruning | optional; if enabled, use real per-epoch metrics and `OPTUNA_PRUNING_MIN_EPOCH >= 8` for 50-epoch HPO |
| Collapsed-4 auxiliary loss | `METAL_COLLAPSED_LOSS_WEIGHT = 0.0` unless running an explicitly labeled validation-only objective probe |
| Multi-objective Optuna | `OPTUNA_MULTIOBJECTIVE = False` unless running an explicitly labeled validation-only Pareto probe |
| Final test | Stage 7 only, after Stage 6 grouped-fold validation |

Stage 6 grouped-fold confirmation is the exception to the `VAL_FRACTION`
default: it sets `VAL_FRACTION = 0.0` and uses `SEED_REPEAT_N_FOLDS = 5` with
`SPLIT_BY = "pdbid"`.

Keep this guide explanatory. Do not paste full Stage 0-7 blocks here.

## Glossary

- `EPOCHS`: maximum training epochs for normal manual runs, grouped-fold
  confirmation runs, and final retraining/evaluation workflows.
- `MAX_EPOCHS_PER_TRIAL`: per-trial epoch cap used only inside
  `RUN_MODE = "controlled_hpo_optuna"`. It can be lower than `EPOCHS` for
  debug or medium HPO, but then the Optuna ranking reflects early-training
  behavior rather than a full training budget.
- `seed_repeat*`: historical notebook naming for top-config reevaluation. In
  reportable metal Stage 6, these variables now drive grouped-fold validation:
  `TOP_CONFIG_REEVALUATION_MODE = "group_kfold"`, one fixed `REPEAT_SEEDS`
  model seed, `SEED_REPEAT_N_FOLDS = 5`, and a fixed
  `SEED_REPEAT_SPLIT_SEED`. The legacy `seed_repeat` mode remains exploratory.
- `active_run_config.json` / `active_run_config.md`: notebook-generated records
  of the resolved live configuration before a subprocess starts. Completed
  runs still rely on `run_config.json` and `run_metadata.json`.
- `five-class`: optional metal target scheme where `Mn`, `Cu`, `Zn`, and `Fe`
  stay separate while `Co` and `Ni` share the fifth class. This changes the
  model output classes; use a separate run batch and Optuna study when enabling
  it.
- `collapsed-4`: supplemental metal-reporting view where Fe, Co, and Ni are
  merged into `Class VIII`. Six-class metal classification remains primary.

## G4-Oriented Training Profile

The exact G4-class Optuna budgets, sampler settings, storage URLs, and search
spaces live in `METAL_TRAINING_PIPELINE_PLAYBOOK.md` under "G4-Class Optuna
Policy". This guide does not duplicate them. The high-level posture is:
persistent SQLite Optuna in Drive, multivariate/group TPE, one `MODEL_PRESET`
per study, validation-only objective, and predeclared grouped-fold confirmation
before any held-out test. The playbook owns the exact trial counts, startup
trial counts, epoch budgets, learning-rate ranges, class-weight/loss ranges,
batch-size search space, and seed list.

## Starting Point

Use the legacy **Non-overlapped PinMyMetal** split for current benchmark continuity:

- Use the playbook stage block for the exact dataset, task, validation split,
  grouping, and selection metric values.
- For reportable metal runs, the resolved configuration must use the trusted
  non-overlapped PinMyMetal split, grouped validation splitting, metal balanced
  accuracy for selection, and no held-out test during training.
- Set a visible `RUN_BATCH_ID` for each real comparison batch. The notebook
  writes into that batch folder when `RUN_BATCH_ID` is set; use the stage block
  in the playbook for the canonical name.

For detailed split-variant definitions and final split policy, use Plan.md
section 8. This guide only repeats the operational rule: reportable metal runs
use the trusted non-overlapped PinMyMetal split unless a new split experiment is
explicitly labeled.

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

`GVP + early fusion` and `SimpleGNN + ESM` are supported notebook presets, but
the metal playbook does not currently define standalone serious HPO stage
blocks for them. Treat them as optional manual comparisons or future candidate
stages until exact executable blocks are added to the playbook.

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
- `GVP + late fusion` has been compared against both simple baselines under
  the same split, epoch budget, Stage 6 fold plan, and selection metric.
- Late fusion or another simple ESM-informed model gives enough validation benefit to justify testing more expressive fusion.

Suggested advanced-fusion order:

1. `GVP + node-level late fusion`: first advanced option, because it adds residue-level ESM/node-state interaction after graph message passing without changing graph inputs.
2. `GVP + hybrid fusion`: test only if early fusion or late fusion looks useful, because it combines early residue-level ESM injection with late graph-level ESM fusion.
3. `GVP + cross-modal attention`: most expressive and easiest to over-tune; keep it last unless there is a specific reason to test attention earlier.

Do not run `full_model_comparison` as the first serious architecture comparison. It mixes simple and advanced models before the simple anchor is established, which makes the result harder to interpret.

If cross-attention is tested, use the playbook's Stage 5F block for the exact
first serious search space. Compare cross-attention against the best
validation-selected late-fusion model, not against an untuned or mismatched
baseline. Use validation metrics only for the decision, and do not broaden
attention settings until the playbook gate is passed.

For node-level late fusion and hybrid fusion, inherit the same graph anchor and comparison rules:

- Keep the validated `Only-GVP` graph settings fixed at first.
- Keep the same split, epoch budget, Stage 6 fold plan, and selection metric
  as late fusion.
- Retune only a narrow set of ESM/fusion-sensitive settings first.
- Compare against the best validation-selected `GVP + late fusion`, not only against `Only-GVP`.

`SimpleGNN + ESM` is not an advanced GVP fusion mode. Use it later as an ablation to ask whether GVP vector geometry is helping compared with a simpler scalar graph + ESM model. It should not replace the `Only-GVP`, `Only-ESM`, and `GVP + late fusion` baseline sequence.

The main capacity fields are:

- `HIDDEN_S_VALUES_CSV`: scalar hidden width for GVP/GNN states.
- `HIDDEN_V_VALUES_CSV`: vector hidden width for GVP models.
- `EDGE_HIDDEN_VALUES_CSV`: edge-feature hidden width.
- `GVP_LAYERS_VALUES_CSV`: graph message-passing depth.
- `HEAD_MLP_LAYERS_VALUES_CSV`: classifier-head depth.
- `EDGE_RADIUS_VALUES_CSV`: graph radius cutoff.

Do not vary all capacity fields at once in the first baseline. Use the playbook
for the exact first baseline and HPO search spaces; use
`only_gvp_architecture_grid` or `only_gvp_geometry_grid` only after simpler
learning-rate and seed behavior is understood.

### ESM options

For the first metal baseline:

- Use `MODEL_PRESET = "Only-GVP"`.
- Leave `ESM_EMBEDDINGS_DIR` blank unless you already have embeddings.
- Keep `ALLOW_MISSING_ESM_EMBEDDINGS = False`.
- Keep `PREPARE_MISSING_ESM_EMBEDDINGS = False`.

For ESM or fusion runs:

- Provide `ESM_EMBEDDINGS_DIR`, or explicitly set `PREPARE_MISSING_ESM_EMBEDDINGS = True`.
- Do not set `ALLOW_MISSING_ESM_EMBEDDINGS = True` for reportable runs. It is a debug/ablation escape hatch.
- Current canonical embeddings are ESMC `esmc_300m` with embedding dimension
  `960`. Newly generated embeddings write a `*.pt.json` sidecar recording the
  ESM model/checkpoint name, embedding dimension, generation time, code
  version, and source structure/sequence identifier. Older embedding files may
  lack this sidecar; label those as `unknown_in_older_embeddings` instead of
  inferring the variant.

`use_early_esm` is controlled by presets: early and hybrid fusion enable it; late fusion and `Only-ESM` do not.

In notebook language, `uses_esm` means the selected preset requires ESM residue embeddings. `use_early_esm` means residue-level ESM vectors are injected into node features before graph message passing. `Only-GVP` has both off.

### RING options

Default first baseline:

- `RING_EDGE_MODE = "with_ring"`
- `REQUIRE_RING_EDGES = False`
- `PREPARE_MISSING_RING_EDGES = True`
- `RING_FEATURES_DIR = ""`
- `RING_EXE_PATH = "DeepMzyme_Data/ring-4.0/out/bin/ring"`
- `ALLOW_MISSING_EXTERNAL_FEATURES = False`
- `PREPARE_MISSING_EXTERNAL_FEATURES = True`

The notebook now starts from RING-enabled graph construction and strict updated
external features by default. Existing RING files are reused; missing files are
generated when the corresponding preparation tool is available. The default
empty `RING_FEATURES_DIR` resolves to the dataset-adjacent `RING_features/`
directory, and the default `RING_EXE_PATH` resolves to the bundled/local
`ring-4.0/out/bin/ring` executable. Missing updated external features are not
allowed to fall back to default-filled values. If `REQUIRE_RING_EDGES = True`,
incomplete RING coverage should fail instead of silently mixing graph types. To
run a radius-only ablation, set `RING_EDGE_MODE = "without_ring"`.

### Training hyperparameters

The playbook owns the exact values for `EPOCHS`, `BATCH_SIZES_CSV`,
`LEARNING_RATES_CSV`, `WEIGHT_DECAYS_CSV`, `LR_SCHEDULES_CSV`, and `SEEDS_CSV`
for each stage. Use this guide only to understand what those fields mean:

- `EPOCHS` controls the maximum training duration for normal/manual runs and
  Stage 6/final workflows; Optuna trial duration is controlled by
  `MAX_EPOCHS_PER_TRIAL`.
- `BATCH_SIZES_CSV`, `LEARNING_RATES_CSV`, and `WEIGHT_DECAYS_CSV` define
  manual comparison grids when `RUN_MODE = "manual_configurations"`.
- `LR_SCHEDULES_CSV` controls the learning-rate schedule for manual runs.
- `SEEDS_CSV` controls both initialization and the validation split when using
  the simple grouped split path.

Do not compare 1-epoch runs as if they are model-quality evidence.

### Training-only graph augmentation

`POSITION_NOISE_STD` and `SECOND_SHELL_DROPOUT` map to
`--position-noise-std` and `--second-shell-dropout`. Both default to `0.0`.
They are applied only by the training dataset when nonzero; validation,
grouped-fold confirmation, and held-out test inference use unaugmented
coordinates and graph membership.

For serious Stage 5A-and-later HPO, the playbook may opt into
`OPTUNA_POSITION_NOISE_STDS_CSV = "0.0,0.05,0.1"` and
`OPTUNA_SECOND_SHELL_DROPOUTS_CSV = "0.0,0.1,0.2"`. Keep `0.0` in every
augmentation search so the unaugmented baseline remains directly comparable.

### Joint-loss weighting caution

`--joint-loss-weighting auto` resolves to learned uncertainty weighting for
joint metal+EC runs. This can be useful, but it can also collapse one task's
effective gradient if one loss becomes much smaller than the other. Prefer
fixed task weights for reportable joint experiments unless uncertainty
weighting has its own validation evidence. Record `--joint-loss-weighting`,
`--metal-loss-weight`, `--ec-loss-weight`, and the learned task-loss scales in
`run_metadata.json` / per-epoch metric CSVs.

### Validation and selection metric

For metal, optimize:

- Primary: `val_metal_balanced_acc`
- Secondary diagnostics: `val_metal_macro_f1`, `val_metal_min_recall`, `val_metal_per_class_recall`, `val_metal_collapsed4_balanced_acc`

Use balanced accuracy because the metal classes are imbalanced. Plain accuracy can look good while failing rare metals.

Keep `VAL_FRACTION > 0` for single-split reportable comparisons. Stage 6
grouped-fold confirmation is the planned exception: it sets
`VAL_FRACTION = 0` together with `N_FOLDS`/`FOLD_INDEX`, so validation still
exists. If `VAL_FRACTION = 0` without a fold split, the training CLI falls
back to `train_loss`, which is not a valid basis for model selection. The
notebook prints a metal split diagnostic showing whether every metal class is
present in train and validation; if any class is missing from validation, that
run is not suitable for reportable model selection.

### Metal class weighting

Current code supports:

- `METAL_CLASS_WEIGHT_MODES_CSV = "none"`
- `METAL_CLASS_WEIGHT_MODES_CSV = "manual"`
- `METAL_CLASS_WEIGHT_MODES_CSV = "inverse_frequency"`
- `METAL_CLASS_WEIGHT_MODES_CSV = "inverse_sqrt_frequency"`
- `METAL_CLASS_WEIGHT_MODES_CSV = "effective_number"`
- `BALANCE_METAL_SITE_SYMBOLS = True` or `False`
- `METAL_LOSS_FUNCTION = "cross_entropy"` or `"focal"`
- `METAL_COLLAPSED_LOSS_WEIGHT = 0.0` by default

Start cautiously:

1. Use the source-code/notebook default `inverse_frequency` for the first baseline, because existing DeepMzyme runs used it.
2. Compare `none,inverse_frequency,inverse_sqrt_frequency,effective_number` only after the baseline is stable.
3. Keep `METAL_LOSS_FUNCTION = "cross_entropy"` first.
4. Treat `focal` and per-class loss multipliers as later ablations, not first-line defaults.
5. Do not decide class weighting from one seed.

Class weights are computed from the training split only, except for `manual`.
With `manual`, the base weight for every active metal class is `1.0`, so the
per-class `*_LOSS_MULTIPLIER` notebook values are the exact metal loss weights.
With computed modes, those same values multiply the training-split-derived
weights. Still inspect whether weighting improves rare-class recall without
destroying common-class performance.

### Collapsed-4 Auxiliary Loss

`METAL_COLLAPSED_LOSS_WEIGHT` is an experimental metal-only objective option.
The default `0.0` preserves the existing six-class loss. Nonzero values add an
auxiliary collapsed-4 cross-entropy term where `Fe`, `Co`, and `Ni` are merged
into `Class VIII` only for that auxiliary view.

Use it only as a validation-only probe after the initial baselines are stable.
The playbook's optional-objective section owns the exact first-use values and
the Stage 5A overlay. Keep this out of initial baselines unless that playbook
block is being run deliberately.

Do not use collapsed-4 loss in initial Stage 2 baselines, during final held-out
test reporting, or as a reason to repeatedly inspect held-out test performance.
Six-class reporting remains primary; collapsed-4 reporting is supplemental and
must not hide Fe/Co/Ni failures.

### Optuna Storage And Stage 6 Confirmation

For debug only:

- `OPTUNA_INTENSITY = "debug"`
- Blank `OPTUNA_STORAGE` is acceptable.

For useful Colab HPO:

- Use `OPTUNA_INTENSITY = "custom"` when you want exact control over the
  budget. The playbook uses this for G4-oriented serious searches.
- Use persistent SQLite storage in Drive. The playbook gives the exact storage
  path for each serious stage.
- Keep `OPTUNA_SELECTION_METRIC` blank or set it to `val_metal_balanced_acc`.
- Keep `OPTUNA_DIRECTION = "maximize"`.
- Keep `OPTUNA_MULTIOBJECTIVE = False` for the normal single-objective path.
  When set to `True`, the notebook creates a validation-only multi-objective
  study over `val_metal_balanced_acc` and active metal-scheme
  `val_metal_min_recall`.
- Keep `OPTUNA_TPE_MULTIVARIATE = True` and `OPTUNA_TPE_GROUP = True` so TPE
  can model correlated parameters such as hidden width, vector width, graph
  depth, and fusion dimension.
- Set `OPTUNA_N_STARTUP_TRIALS` below `N_OPTUNA_TRIALS`. `N_OPTUNA_TRIALS` is
  the target number of completed trials in the study; with persistent storage,
  reruns launch only the remaining trials needed to reach that target. If
  startup trials are greater than or equal to the completed-trial target, the
  run is effectively random search. The playbook owns the exact startup-trial
  value for each stage.
- Keep `OPTUNA_AUTO_CONFIGURE_BUDGET = False` when using a playbook block with
  explicit trial counts. If enabled, the notebook may raise trial counts to the
  advisor's minimum recommendation.
- If the final launch-time `OPTUNA_N_STARTUP_TRIALS` or completed-trial target
  `N_OPTUNA_TRIALS` is below the advisor recommendation, the notebook asks for
  terminal-style confirmation with `input()`. Type `Y` to continue an
  intentionally under-budgeted smoke/debug run, or `N`/Enter to stop before
  Optuna launches.
- `OPTUNA_USE_PRUNING` is now real but optional: the notebook monitors
  `val_metrics.csv` / `train_metrics.csv` in each trial run directory, calls
  `trial.report(...)`, and terminates the trial subprocess process group when
  Optuna prunes it. Use `OPTUNA_PRUNING_MIN_EPOCH >= 8` for serious 50-epoch
  HPO; lower values are debug/plumbing-only.
- Use `OPTUNA_LR_SCHEDULES_CSV` to search `fixed,cosine` where the playbook
  enables LR-schedule search. Do not include `step` in Optuna schedule search
  unless step size and gamma are also explicitly searched.
- Keep `OPTUNA_ALLOW_INCOMPATIBLE_STUDY_REUSE = False`. The notebook records
  model preset, task, split, metric, search-space hash, sampler seed, pruning
  settings, batch-size choices, LR schedule choices, class-weight choices, and
  run batch ID in study metadata, then stops if a persistent study is reused
  incompatibly.
- Use `RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = True` only after you are ready
  to rerun the top configurations through the Stage 6 confirmation plan.
- Use `TOP_CONFIG_REEVALUATION_MODE = "group_kfold"` for project-standard
  metal confirmation. The playbook Stage 6 block uses one model seed,
  `SEED_REPEAT_N_FOLDS = 5`, and a fixed `SEED_REPEAT_SPLIT_SEED` so every
  candidate sees the same `pdbid` folds.
- Use `TOP_CONFIG_REEVALUATION_MODE = "seed_repeat"` only for explicitly
  labeled exploratory checks; it measures combined initialization and split
  variance.

Numeric Optuna budgets, sampler seeds, split seeds, storage URLs,
learning-rate ranges, class-weight/loss ranges, and batch-size search spaces are
defined per stage in `METAL_TRAINING_PIPELINE_PLAYBOOK.md`. Use
`OPTUNA_INTENSITY = "custom"` and persistent Drive-backed storage for every
reportable run on the G4 GPU.

In multi-objective mode, Optuna uses minimum recall over the active metal label
scheme for rare-class protection. For default reportable runs that is six-class
minimum recall; for explicitly labeled `five_class` runs it is five-class
minimum recall over `Mn`, `Cu`, `Zn`, `Fe`, and grouped Co/Ni. Collapsed-4
minimum recall is reported as supplemental information, not as the default
second objective. If pruning is incompatible with the multi-objective study,
the notebook disables pruning and warns before launch. Inspect Pareto
candidates as review inputs, then run Stage 6 grouped-fold confirmation before
promoting any candidate.

`OPTUNA_SEARCH_PRESET = "first_useful_only_gvp_narrow"` keeps architecture/capacity fixed and varies mainly learning rate, LR schedule when enabled, weight decay, batch size, and metal class-weight mode. Use it for the first controlled HPO path or for explicit anchor continuation. For a user-requested fresh broad Optuna check, use the playbook's large-search blocks and expand capacity/search axes within one selected model family instead of over-narrowing to old raw outputs. Short HPO trials mostly rank early-training behavior.

Stage 3 is plumbing/debug only. If `N_OPTUNA_TRIALS` equals
`OPTUNA_N_STARTUP_TRIALS`, all trials are startup trials and the run is
effectively random search. Do not treat Stage 3 rankings as model-selection
evidence.

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
5. Run Stage 6 grouped-fold validation before treating any HPO candidate as
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
- Same Stage 6 fold plan or explicitly labeled seed-repeat plan.
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
- `<run_dir>/run_metadata.json`: selected metric, selected checkpoint, config,
  ESM embedding metadata summary, history, split/test metadata.
- `<run_dir>/run_config.json`: full saved config and history.
- `<run_dir>/dataset_summary.json`: dataset and split identity summary.
- `<run_dir>/split_diagnostics.json`: train/validation counts, grouping, overlap, missing classes.
- `<run_dir>/best_model_checkpoint.pt`: checkpoint selected by validation metric.
- `<run_dir>/last_model_checkpoint.pt`: final epoch checkpoint.
- `<run_dir>/prepare_status.json`: preparation/preflight status.
- `<run_dir>/test_report.json`: only present after held-out test evaluation; do not use this for model selection.
- `<run_dir>/test_predictions.pt`: final-test logits/probabilities used for
  reporting artifacts, only present after held-out test evaluation.
- `<run_dir>/test_reliability_diagram.png` and
  `<run_dir>/test_confidence_histogram.png`: calibration plots when matplotlib
  is available.

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

`METAL_LABEL_SCHEME` is different: it changes the training targets before the
commands are built. `six_class` is the default, `five_class` groups only Co/Ni,
and `four_class` groups Fe/Co/Ni. Changing this field creates a different
prediction problem and must use a separately named validation batch/study.

`FINAL_TEST_BATCH_METRICS` controls only which metric columns are emphasized in
batch final-test summaries and plots. It does not change which metrics are
computed or saved in `test_report.json`.

Stage 7 also has explicit final-reporting controls:

- `FINAL_TEST_PRIMARY_REPORT`: predeclares the primary final result before the
  held-out test is opened. Use `single_checkpoint` for the selected checkpoint
  or `softmax_mean_5_seeds` for the optional five-checkpoint ensemble.
- `FINAL_TEST_ENSEMBLE_MODE`: use `single_checkpoint` for current behavior or
  `softmax_mean_5_seeds` to average probabilities from exactly five fixed
  Stage 6 source checkpoints.
- `FINAL_TEST_ENSEMBLE_SOURCE_RUN_DIRS`: comma- or newline-separated list of
  the five source run directories for ensemble mode. This list must be fixed in
  preview before launch.
- `FINAL_TEST_SELECTED_CONFIG_ID`: optional human-readable identifier for the
  validation-selected configuration.
- `FINAL_TEST_CALIBRATION_BINS`, `FINAL_TEST_ENABLE_TEMPERATURE_SCALING`,
  `FINAL_TEST_ENABLE_BOOTSTRAP_CI`, `FINAL_TEST_BOOTSTRAP_RESAMPLES`, and
  `FINAL_TEST_BOOTSTRAP_SEED`: reporting controls for calibration plots,
  validation-fitted temperature scaling, and bootstrap confidence intervals.

Temperature scaling is fitted only on validation logits from the already fixed
configuration. In ensemble mode, the fixed rule is one validation-fitted
temperature per checkpoint, then averaging calibrated softmax probabilities.
The notebook records primary versus secondary/diagnostic reports in
`test_report.json`; do not change the primary report after viewing held-out
metrics. The full Stage 7 policy and executable blocks live in
`docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`.

Metal evaluation normally keeps the default six-class prediction problem
`Mn`, `Cu`, `Zn`, `Fe`, `Co`, `Ni`. If `METAL_LABEL_SCHEME = "five_class"` is
selected, the active metal metrics are five-class metrics over
`Mn`, `Cu`, `Zn`, `Fe`, and grouped `Co/Ni`. For every metal or joint test
report, the code also computes collapsed-4 metrics by merging `Fe`, `Co`, and
`Ni` into `Class VIII`, giving `Mn`, `Cu`, `Zn`, and `Class VIII`. Use
`METAL_REPORT_VIEW` to choose which view is emphasized in notebook output, not
to rerun a different test.

After Optuna:

- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/all_trials.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/optuna_trials.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/top_trials.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/best_trial.json`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/optuna_study_metadata.json`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/active_run_config.json`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/active_run_config.md`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/optuna_best_config.json`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/best_config_command.txt`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/top_reevaluation_commands.txt`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/pareto_front.csv`, when
  `OPTUNA_MULTIOBJECTIVE = True`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/pareto_candidates.csv`, when
  `OPTUNA_MULTIOBJECTIVE = True`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/pareto_candidates_ranked_for_review.csv`, when
  `OPTUNA_MULTIOBJECTIVE = True`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/seed_repeat_results.csv`, if Stage 6 top-config reevaluation was run
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/seed_repeat_summary.csv`, if Stage 6 top-config reevaluation was run
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/seed_repeat_pairwise_bootstrap.csv`, if Stage 6 top-config reevaluation was run
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/seed_repeat_pairwise_bootstrap.json`, if Stage 6 top-config reevaluation was run
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/optuna_study_summary.md`

Each launched run directory also gets `active_run_config.json` and
`active_run_config.md` before the subprocess starts. Completed runs still write
`run_config.json`, `run_metadata.json`, and per-epoch
`train_metrics.csv` / `val_metrics.csv`.

## How To Decide The Current Best Configuration

For a validation-only manual comparison:

1. Open `<SUMMARY_BASENAME>.csv` and filter to `status = completed`.
2. Filter to `result_stage = validation-only`, `group-kfold validation`, or
   `seed-repeat validation`, not final-test rows.
3. Filter to the intended `task = metal`, `metal_label_scheme`, split type,
   epoch budget, and model comparison group.
4. Rank by `selected_best_validation_metric_value` or `best_validation_metric_used_for_checkpoint_selection`, using `selection_metric = val_metal_balanced_acc`.
5. Check diagnostics for missing validation metal classes and train/validation overlap.
6. Check per-class recall and collapsed-4 balanced accuracy as secondary evidence.
7. Prefer a configuration supported across Stage 6 folds over one single high
   Optuna trial.

For Optuna:

1. Inspect `top_trials.csv` for single-objective studies, or
   `pareto_candidates_ranked_for_review.csv` for multi-objective studies.
2. Inspect `optuna_study_summary.md`.
3. Run top-k grouped-fold validation.
4. Select by grouped-fold mean, paired bootstrap CI, and rare-class recall on
   `val_metal_balanced_acc`, not by a single trial.
5. Retrain or evaluate the final selected validation-best checkpoint only after selection is complete.

For final reporting:

1. Use the notebook's "Select final run and show saved outputs" cell to choose the validation-best run.
2. In preview, set `FINAL_TEST_PRIMARY_REPORT` and, if using the ensemble,
   list exactly five fixed source runs in `FINAL_TEST_ENSEMBLE_SOURCE_RUN_DIRS`.
3. Use the "Optional final held-out test evaluation" cell with
   `LAUNCH_FINAL_HELD_OUT_TEST_EVAL = True` only after the preview is correct.
4. Prefer `FINAL_TEST_WORKFLOW = "evaluate_selected_checkpoint"` unless you
   intentionally want a retrain from the selected config.
5. Record `test_metal_balanced_acc`, `test_metal_macro_f1`,
   `test_metal_per_class_recall`, `test_metal_collapsed4_balanced_acc`,
   calibration metrics, plot paths, bootstrap CI fields, and
   `calibrated_metrics` from `test_report.json`.
6. Do not go back and choose a different configuration, ensemble subset,
   temperature, or primary report because its test score is better.

## Mistakes To Avoid

- Do not select models based on held-out test metrics.
- Do not enable `INCLUDE_HELD_OUT_TEST_DURING_TRAINING` for comparison, HPO,
  grouped-fold confirmation, or seed-repeat runs.
- Do not run reportable Optuna with notebook preset budgets; use the playbook's
  `OPTUNA_INTENSITY = "custom"` stage blocks.
- Do not enable `METAL_COLLAPSED_LOSS_WEIGHT > 0` in initial baselines or final
  held-out test workflows.
- Do not use multi-objective Pareto review as a substitute for Stage 6
  grouped-fold confirmation.
- Do not run serious Optuna with missing or nonpersistent `OPTUNA_STORAGE`.
- Do not reuse one persistent Optuna study for multiple `MODEL_PRESET` values
  metal label schemes, or incompatible search spaces; let the default
  study-compatibility guard stop the run.
- Do not compare old mixed folders silently. Treat local run directories as
  historical evidence unless the current task explicitly includes them; use
  `EXPERIMENT_STATUS.md` and `docs/notebook_outputs/` to identify trusted
  evidence.
- Do not mix incompatible `MODEL_PRESET` values in Stage 6 top-config
  reevaluation.
- Do not set `ALLOW_SEED_REPEAT_MODEL_PRESET_MISMATCH = True` unless you are intentionally overriding the guard.
- Do not launch Stage 7 from raw Optuna trial folders. Use a Stage 6
  validation-selected source run.
- Do not trust one lucky seed.
- Do not over-interpret 1-epoch or 3-epoch debug results.
- Do not let missing ESM embeddings silently define an ESM baseline.
- Do not present exact/possibly-overlapped split results as the main held-out result.
- Do not use `VAL_FRACTION = 0` for reportable model selection unless Stage 6
  grouped-fold validation is explicitly configured with `N_FOLDS`/`FOLD_INDEX`.

## Notebook Behavior Notes

These are stable usage cautions, not an implementation backlog:

- The notebook's visible default `EPOCHS = 1` is smoke-safe. Every real stage
  should paste the playbook block so the resolved training budget is explicit.
- `RECOMMENDED_RUN_SET` may override `MODEL_PRESET`; inspect the resolved
  planning table before launch.
- `INCLUDE_HELD_OUT_TEST_DURING_TRAINING` remains visible for compatibility, but
  reportable workflows must keep it `False` and use Stage 7 only after
  validation selection is fixed.
- RING-enabled graph construction is the normal first graph setting. Use the
  playbook's radius-only ablation only when intentionally testing that ablation.
- Saved `Only-GVP` configs may show a fusion-mode field even though ESM fusion
  is irrelevant for `only_gvp`; interpret that as a display artifact.
