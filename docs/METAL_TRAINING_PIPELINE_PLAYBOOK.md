# Metal Training Pipeline Playbook

This playbook is the practical, notebook-ready pipeline for DeepMzyme metal
classification. It complements `Plan.md`, which remains the high-level research
and design authority. Current best validation evidence belongs in
`EXPERIMENT_STATUS.md` and `docs/notebook_outputs/`, not in this stable
playbook.

For the cross-document run order and output-folder map, see `docs/README.md`.


## Quick-Paste Stage Selector

This playbook is the operational pipeline for
`notebooks/DeepMzyme_training_colab.ipynb`. Each stage is a self-contained
notebook configuration block; you paste one block at a time into the Main
configuration cell. Stages 1, 3 are smoke checks. Stages 2, 4, 5, 6 are
validation-only. Stage 6B is the validation-to-final-refit bridge: it applies
promotion gates to Stage 6 evidence and trains/refits one final model on the
full non-test training set. Stage 7 is the only stage that touches the held-out
test set, and it is run exactly once for that frozen Stage 6B final-refit run.

Use these stage names exactly when planning, documenting, or asking an agent
what to run next:

- Stage 0: environment/data readiness
- Stage 1: 1-epoch smoke
- Stage 2A: Only-GVP validation anchor
- Stage 2B: baseline family comparison
- Stage 3: Optuna plumbing debug
- Stage 4: medium per-family Optuna, optional on G4
- Stage 5A: serious Only-GVP HPO
- Stage 5B: Only-ESM HPO
- Stage 5C: GVP + late fusion HPO
- Stage 5D: GVP + node-level late fusion HPO
- Stage 5E: GVP + hybrid fusion HPO
- Stage 5F: GVP + cross-attention HPO
- Stage 5G: RING/radius-only ablation
- Stage 6: top-K seed/split confirmation
- Stage 6B: promotion gates and final full-train refit
- Stage 7: one-shot held-out test

## Pipeline Overview At A Glance

| Stage | Purpose | Owns exact budget? | G4 wall-time (approx.) | Pass/fail decision gate | Required outputs |
| --- | --- | --- | --- | --- | --- |
| Stage 0 | Environment, Drive, data bundle, RING/ESM/external-feature readiness | Yes, planning-only | 10-20 min | Planned config resolves under Drive, coverage diagnostics pass, no test artifacts | Planned-run CSV/dictionary, optional metal-weight diagnostics |
| Stage 1 | 1-epoch smoke to prove notebook and training path | Yes, smoke budget | 5-15 min | One validation-only run completes; no missing paths/classes; no test artifacts | Planned files, one run dir, `run_config.json`, `run_metadata.json`, `split_diagnostics.json` |
| Stage 2A | Only-GVP validation anchor | Yes | 10-16 h | All planned validation runs complete and rare-class diagnostics are usable | Planned files, run dirs, summary CSV/PNG, no `test_report.json` |
| Stage 2B | Baseline family comparison after ESM is ready | Yes | 8-14 h | All planned validation runs complete and ESM coverage is valid | Planned files, run dirs, summary CSV/PNG, no `test_report.json` |
| Stage 3 | Optuna plumbing debug | Yes, debug only | 20-40 min | Four complete validation-only trials and valid persistent-storage plumbing | Optuna `all_trials.csv`, `top_trials.csv`, `best_trial.json`, study summary |
| Stage 4 | Medium per-family Optuna, optional on G4 | Yes | 8-16 h | Sixty-four complete validation-only trials in one `MODEL_PRESET` | Optuna CSV/JSON/Markdown outputs and per-trial run dirs |
| Stage 5A | Serious Only-GVP HPO | Yes | 36-60 h | Two hundred complete validation-only trials in the Only-GVP study | Optuna CSV/JSON/Markdown outputs and per-trial run dirs |
| Stage 5B | Only-ESM HPO | Yes | 24-48 h | One hundred twenty complete validation-only trials with valid ESM coverage | Optuna CSV/JSON/Markdown outputs and per-trial run dirs |
| Stage 5C | GVP + late fusion HPO | Yes | 36-60 h | Two hundred complete validation-only trials with valid ESM coverage | Optuna CSV/JSON/Markdown outputs and per-trial run dirs |
| Stage 5D | GVP + node-level late fusion HPO | Yes | 36-60 h | Stage 5C gate passed, then two hundred complete validation-only trials | Optuna CSV/JSON/Markdown outputs and per-trial run dirs |
| Stage 5E | GVP + hybrid fusion HPO | Yes | 36-60 h | Stage 5C gate passed, then two hundred complete validation-only trials | Optuna CSV/JSON/Markdown outputs and per-trial run dirs |
| Stage 5F | GVP + cross-attention HPO | Yes | 30-55 h | Stage 5C gate passed, then one hundred twenty complete validation-only trials | Optuna CSV/JSON/Markdown outputs and per-trial run dirs |
| Stage 5G | RING/radius-only ablation | Yes, ablation budget | 6-10 h | Matching radius-only validation runs complete and are labeled as ablation | Planned files, run dirs, summary CSV/PNG, no `test_report.json` |
| Stage 6 | Top-K seed/split confirmation | Yes | 15-25 h for one seed; more with extra seeds | All predeclared top-K x fold x active-seed validation runs complete; one candidate selected by paired validation evidence | `seed_repeat_results.csv`, `seed_repeat_summary.csv`, `seed_repeat_summary.json`, `seed_repeat_pairwise_bootstrap.csv`, `seed_repeat_pairwise_bootstrap.json`, `stage6_ranked_candidates.csv`, `stage6_selected_final_candidate.json`, run dirs |
| Stage 6B | Promotion gates and final full-train refit | Yes | One final training run | Stage 6 candidate passes configured paired-CI, rare-recall, and tie-breaker gates; final refit completes with no test report | `stage6b_ranked_candidates.csv`, `stage6b_decision.json`, `stage6b_final_refit_command.txt`, `stage6b_selected_final_refit_candidate.json`, final-refit run dir |
| Stage 7 | One-shot held-out test | Yes, final only | 20-60 min | Source is the frozen Stage 6B final-refit run and one-shot policy is confirmed | Separate final-test run dir, `test_report.json`, final-test summary |

All configuration blocks below use variables that exist in
`notebooks/DeepMzyme_training_colab.ipynb` as of this repository state. For
ordinary planned training and HPO stages, edit the notebook's **Main
configuration** cell directly or paste the block at the end of that cell before
running **Build central CONFIG dictionary**. Stage 6 grouped-fold confirmation
uses the dedicated **Stage 6 controls and existing Optuna/HPO reuse** panel; for
an already completed HPO directory, Stage 6 can run in standalone existing-HPO
mode independently of the Main configuration cell.

## How To Use This Playbook

Use exactly one stage block at a time. For ordinary planned training or HPO,
after editing the notebook's **Main configuration** cell, run the CONFIG/planning
cells and inspect the resolved commands before setting
`LAUNCH_PLANNED_MAIN_TRAINING_RUNS = True` in the dedicated launch-switch cell.
For Stage 6 from an already completed HPO directory, keep that switch `False`,
fill the Stage 6 controls, and use the dedicated Stage 6 launch cell. It is now
safe to use Colab **Run all** for this path: setup/clone/data cells run, the
ordinary main training/HPO cell no-ops, and the Stage 6 launch cell switches to
standalone existing-HPO mode when an old HPO source is configured. If no Stage 6
source is configured, the Stage 6 launch cell no-ops with a message instead of
trying to import from an empty current run folder.

Notebook execution order:

1. Setup/install and data-source cells.
2. Main planned-training launch switch. Keep it `False` while planning or
   loading helpers; set it `True` only before ordinary planned runs or HPO.
3. For ordinary planned runs/HPO: Main configuration cell with one block from
   this playbook.
4. For ordinary planned runs/HPO: Build central `CONFIG`.
5. For ordinary planned runs/HPO: Planning/preflight cells.
6. For ordinary planned runs/HPO: Optional training execution cell.
7. Summarize/report cell for the current `RUN_BATCH_ID` when relevant.
8. For Stage 6 only: Stage 6 controls/checklist, then
   **Launch Stage 6 top-K grouped-fold confirmation**.
9. For Stage 6B: apply promotion gates, preview the final-refit command, then
   launch one final full-train refit from the selected configuration without
   changing model-selection choices.
10. For final testing only: select that Stage 6B final-refit run, preview final
    held-out test, then launch once.

For all comparison, HPO, and Stage 6 confirmation stages:

- Keep `INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False`.
- Keep `VAL_FRACTION = 0.15` and `SPLIT_BY = "pdbid"` unless a new split
  experiment is explicitly being labeled. `pdbid` grouping is stricter than
  `pdbid_chain` grouping: it keeps all chains and pockets from one PDB entry on
  one side, so binuclear or repeated same-chain metal sites cannot leak across
  train/validation. Stage 6 grouped-fold confirmation is the planned exception:
  it sets `VAL_FRACTION = 0.0`, `SPLIT_BY = "pdbid"`,
  `SEED_REPEAT_N_FOLDS = 5`, a fixed `SEED_REPEAT_SPLIT_SEED`, and the
  predeclared `REPEAT_SEEDS` model-seed list. Stage 6B is the final-refit
  exception: it uses `VAL_FRACTION = 0.0` because the selected configuration is
  retrained on the full non-test training set after validation/CV selection.
- Use validation metrics, usually `val_metal_balanced_acc`, for checkpoint,
  hyperparameter, architecture, and fusion decisions.
- Do not run the optional final held-out test cell until the final
  validation-selected configuration is fixed and its Stage 6B final-refit run
  has been completed and frozen.
- If the user asks for a new check, new run, or fresh Optuna sweep without
  explicitly asking to rely on previous raws/results, use previous notebook
  outputs only as context and safety checks. Prefer the broadest sensible
  validation-only Optuna search within the selected `MODEL_PRESET`, with
  common-sense runtime and feature-availability limits.
- If the user explicitly asks to rely on previous running/results/raws, inspect
  the relevant copied evidence and use it to narrow, continue, or repeat that
  prior configuration.

## Run Tiers And Reproducibility Records

| Tier | Playbook stages | Selection/reporting status | Required record |
| --- | --- | --- | --- |
| Debug | Stage 0, Stage 1, Stage 3 | Not model-selection evidence | Resolved notebook config, planned commands, run logs, and any failure context |
| Serious validation | Stage 2, Stage 4, Stage 5, Stage 6 | Validation-only model-selection evidence if the stage gate passes | Full run config, split/fold identity, seeds, Optuna study metadata, dataset bundle ID/checksum, git commit, and key library versions |
| Final refit | Stage 6B | Final training from validation-selected configuration; no held-out test | Stage 6 evidence, Stage 6B decision JSON, final-refit command, final-refit run config/metadata/checkpoint |
| Final test | Stage 7 | One-shot held-out reporting only | Stage 6/6B selection evidence, Stage 6B final-refit source run/checkpoint, primary report declaration, calibration/CI settings, dataset bundle ID/checksum, git commit, key library versions, and no-test-selection statement |

Serious validation and final-test records should capture key library versions
when available: PyTorch, torch-geometric, ESM/ESMC, Optuna, NumPy, and
scikit-learn. This repository currently has no checked-in environment spec, so
per-run version records are required until an environment file is added.

Limited-compute fallback: use Stage 4 instead of Stage 5 for candidate
discovery, or stop after Stage 2 with a clearly labeled provisional
validation-only result. Do not launch Stage 7 from a provisional result. A final
held-out report still requires one fixed validation-selected configuration, one
frozen Stage 6B final-refit run derived from it, and the one-shot Stage 7
policy.

`EPOCHS` and `MAX_EPOCHS_PER_TRIAL` have different roles:

- `EPOCHS` is the normal training budget for manual comparison runs, Stage 6
  grouped-fold confirmation runs, and final retraining/evaluation workflows.
- `MAX_EPOCHS_PER_TRIAL` is the per-trial cap only inside
  `RUN_MODE = "controlled_hpo_optuna"`.
- If `MAX_EPOCHS_PER_TRIAL < EPOCHS`, Optuna ranks early-training behavior.
  Stage 6 must then confirm candidates at the full validation budget before any
  final selection.

Bootstrap counts intentionally differ by stage. Stage 6 uses 10,000 paired
bootstrap resamples over shared fold-level differences for candidate-promotion
decisions. Stage 7 uses 1,000 stratified bootstrap resamples by default for
held-out-test reporting uncertainty. Do not use Stage 7 CIs to change the
selected model.

Pruning is now enabled by default in the canonical reportable metal Stage 4,
5A, 5C, 5D, 5E, and 5F blocks using `MedianPruner` with
`OPTUNA_PRUNING_MIN_EPOCH = 25`. The notebook monitors real per-epoch metric
CSVs, reports intermediate values to Optuna, and terminates pruned subprocess
process groups. Pruning can bias the TPE trajectory toward early-learning
behavior, so keep the pruner type and minimum epoch fixed within a study and
record pruned-attempt counts separately from completed trials. Consequence:
`OPTUNA_TARGET_COMPLETE_TRIALS` counts only non-pruned completions, so total
trial attempts will be larger than the target -- plan compute accordingly.
Stage 3 may lower the minimum epoch only for plumbing/debug.

Supported presets without canonical serious HPO blocks:

- `GVP + early fusion` is implemented in the notebook/model preset map and may
  be used in ESM-ready manual comparisons. This playbook does not currently own
  a standalone serious HPO block for it.
- `SimpleGNN + ESM` is implemented as an auxiliary scalar-graph ablation. This
  playbook does not currently own a standalone serious HPO block for it.

Do not present either preset as a required metal HPO stage unless an exact
executable block is added here.

## Common Defaults

Use these shared defaults unless a stage overrides them.

```python
TASK = "metal"
METAL_LABEL_SCHEME = "six_class"
DATASET_NAME = "train_and_test_sets_structures_exact_pinmymetal"
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_metal_balanced_acc"
OPTUNA_SELECTION_METRIC = "val_metal_balanced_acc"
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False

RING_EDGE_MODE = "with_ring"
REQUIRE_RING_EDGES = False
PREPARE_MISSING_RING_EDGES = True
RING_FEATURES_DIR = ""
RING_EXE_PATH = "DeepMzyme_Data/ring-4.0/out/bin/ring"

ALLOW_MISSING_EXTERNAL_FEATURES = False
PREPARE_MISSING_EXTERNAL_FEATURES = True
EXTERNAL_FEATURES_ROOT_DIR = ""

CLASSIFIER_POOL_DISTANCE_CUTOFF_VALUES_CSV = "0.0"
POSITION_NOISE_STDS_CSV = "0.0"
SECOND_SHELL_DROPOUTS_CSV = "0.0"  # Fixed off for canonical HPO; use outer-residue dropout instead.
OUTER_RESIDUE_DROPOUTS_CSV = "0.0"

METAL_CLASS_WEIGHT_MODES_CSV = "inverse_frequency"
METAL_LOSS_FUNCTIONS_CSV = "cross_entropy"
METAL_LABEL_SMOOTHING_VALUES_CSV = "0.0"
METAL_COLLAPSED_LOSS_WEIGHTS_CSV = "0.0"
BALANCE_METAL_SITE_SYMBOLS_CSV = "False"

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
OPTUNA_TPE_CONSTANT_LIAR = True
OPTUNA_PARALLEL_WORKERS = 1
OPTUNA_PARALLEL_STARTUP_STAGGER_SECONDS = 10.0
OPTUNA_STOP_ON_PARALLEL_CUDA_OOM = True
OPTUNA_SAMPLER_SEED = None
OPTUNA_AUTO_CONFIGURE_BUDGET = False
OPTUNA_USE_PRUNING = False
OPTUNA_PRUNER_TYPE = "none"
OPTUNA_PRUNING_MIN_EPOCH = 20
OPTUNA_TIMEOUT_MINUTES = 0
OPTUNA_MULTIOBJECTIVE = False
```

`DATASET_NAME` chooses the external train/test dataset split. The notebook
default for new serious metal runs is the exact PinMyMetal split, and the
current Colab bundle exposes exact PinMyMetal plus Common-PDBID 70/30
PinMyMetal choices. Non-overlapped and harsh split roots remain named variants
in Plan.md, but they require supplying a bundle or Drive folder that contains
those dataset roots. `SPLIT_BY` controls only the internal train/validation
grouping inside the selected external train split and is emitted to the CLI as
`--train-val-split-by`; it never changes the external test directory or test
CSV. When the exact split is requested, the notebook must stop if that dataset
root is missing; it must not silently fall back to the non-overlapped split.

Summary CSV/PNG basenames are generated from the live provenance by default:
task, metal label scheme, model preset or run set, `DATASET_NAME`,
`RUN_BATCH_ID`, `SPLIT_BY`, and validation mode. A manual `SUMMARY_BASENAME`
override is still allowed, but the notebook warns and records metadata when
the manual name appears inconsistent with the resolved dataset, batch, or split
policy.

ESM/fusion stages currently assume canonical ESMC `esmc_300m` residue
embeddings with `embedding_dim=960`. Newly generated embeddings write a
`*.pt.json` sidecar with model name, embedding dimension, generation time, code
version, and source structure/sequence metadata. Older embeddings without
sidecars must be labeled as `unknown_in_older_embeddings` in run metadata and
status notes rather than guessed.

## G4-Class Optuna Policy

This project runs on a G4-class GPU (16 GB VRAM, persistent runtime). All
serious Optuna stages must use:

- `OPTUNA_INTENSITY = "custom"` - never rely on `first_useful`/`serious`
  notebook presets for reportable HPO.
- `OPTUNA_TPE_MULTIVARIATE = True`, `OPTUNA_TPE_GROUP = True`,
  `OPTUNA_TPE_CONSTANT_LIAR = True` so shared-storage studies support multiple
  parallel workers without duplicate/in-flight TPE suggestions.
- `OPTUNA_PARALLEL_WORKERS = 1` is the canonical default and preserves
  historical serial trial execution. On a G4/T4 16 GB GPU, `2` is an optional
  validation-only acceleration override after Stage 3 or another short debug
  study confirms there is CUDA memory headroom for the active model family,
  batch-size range, and feature set. Keep `OPTUNA_TPE_CONSTANT_LIAR = True`,
  keep persistent storage enabled, and keep
  `OPTUNA_STOP_ON_PARALLEL_CUDA_OOM = True` when using more than one worker.
- `OPTUNA_SAMPLER_SEED = None` unless deliberately re-exploring the same split
  with a different Optuna trajectory. With `None`, the sampler seed follows
  `OPTUNA_SPLIT_SEED`.
- `OPTUNA_AUTO_CONFIGURE_BUDGET = False` (explicit budgets only).
- `OPTUNA_USE_PRUNING = True` by default in the canonical reportable metal
  Stage 4/5A/5C/5D/5E/5F blocks, with `OPTUNA_PRUNER_TYPE = "median"` and
  `OPTUNA_PRUNING_MIN_EPOCH = 25`. The notebook monitors real per-epoch metric
  CSVs from each trial run directory, reports intermediate values to Optuna, and
  terminates pruned subprocess process groups. Keep the pruner fixed within a
  study because pruner decisions can bias the TPE trajectory. Consequence:
  `OPTUNA_TARGET_COMPLETE_TRIALS` counts only non-pruned completions, so total
  trial attempts will be larger than the target -- plan compute accordingly.
- Persistent SQLite storage in Drive:
  `sqlite:////content/drive/MyDrive/DeepMzyme/optuna/<study_name>.db`.
- Startup trials: use the stage table below. The default rule is at least
  `max(20, 0.2 x OPTUNA_TARGET_COMPLETE_TRIALS)`; the 120-trial Only-ESM study
  uses 30 startup trials to cover its conditional space.
- `OPTUNA_SPLIT_SEED = 42` for every study. Stage 6 uses a separate fixed
  `SEED_REPEAT_SPLIT_SEED` for grouped-fold definitions, so every compared
  candidate sees the same validation folds.
- `OPTUNA_SELECTION_METRIC = "val_metal_balanced_acc"`,
  `OPTUNA_DIRECTION = "maximize"`.
- When `RUN_MODE = "controlled_hpo_optuna"`, active Optuna categorical choices
  reuse the normal CSV fields where those fields exist. This includes
  `LR_SCHEDULES_CSV`, `BATCH_SIZES_CSV`, `METAL_CLASS_WEIGHT_MODES_CSV`, loss
  mode fields, booleans, and `CROSS_ATTENTION_HEADS_CSV`.
- Numeric Optuna fields can optionally use explicit `OPTUNA_*_RANGE`
  overrides. Blank range fields keep the CSV behavior. Nonblank float ranges
  use `low,high`; nonblank integer ranges use `low,high,step`; weight decay is
  log-sampled. Use a fresh Optuna study name for any range-enabled experiment.
  `OPTUNA_SEARCH_PRESET` still decides whether each model-capacity field is
  sampled or fixed.
- `OPTUNA_MULTIOBJECTIVE = False` by default. Optional multi-objective studies
  are validation-only Stage 5A experiments and use NSGA-II over
  `val_metal_balanced_acc` and active metal-scheme `val_metal_min_recall`; they
  do not replace the normal single-objective path.
- Record both the split seed and the sampler seed in the notebook output, study
  summary, and per-run artifacts. If the sampler seed is `None`, record the
  effective sampler seed as the split seed.
- Record `OPTUNA_PARALLEL_WORKERS`, startup stagger seconds, and CUDA-OOM stop
  behavior in the study metadata. Parallel trial order is inherently
  nondeterministic, so Stage 6 grouped-fold confirmation remains mandatory
  before promotion.
- Record `DATALOADER_NUM_WORKERS` and `DATALOADER_PIN_MEMORY` in the generated
  run configuration. These are runtime-throughput controls, not model-selection
  knobs; keep them fixed within a comparable study unless a run is explicitly
  labeled as a DataLoader throughput/debug check.

Forbidden in serious stages:

- Mixing `MODEL_PRESET` values inside one study (Optuna optimizes one family at
  a time).
- Reusing a persistent study DB for a different model preset, architecture,
  fusion mode, split, task, selection metric, or search-space hash. The notebook
  hard-stops incompatible persistent-study reuse unless
  `OPTUNA_ALLOW_INCOMPATIBLE_STUDY_REUSE = True`; leave that override false for
  reportable HPO.
- Held-out test evaluation inside trials
  (`INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False`).
- Using collapsed-4 metrics or multi-objective Pareto review to select,
  inspect, or repeat the held-out test.
- Letting `EPOCHS <= 3` reach Stage 4/5 (the short-training guard will block
  this; do not override).
- Reportable HPO with `OPTUNA_INTENSITY != "custom"` or blank/nonpersistent
  `OPTUNA_STORAGE`.
- Reportable HPO with `OPTUNA_PARALLEL_WORKERS > 1` and blank/nonpersistent
  `OPTUNA_STORAGE`, or with `OPTUNA_TPE_CONSTANT_LIAR = False`.
- `ALLOW_MISSING_ESM_EMBEDDINGS = True` for ESM or fusion stages.

Batch-size policy for serious stages:

- Use `4` only for smoke/debug runs or when a memory failure forces it.
- Use `8,16` as the default serious validation-only Optuna batch-size search
  space. This keeps the current validated `batch_size=8` anchor in scope while
  testing whether `16` improves minority-class stability and GPU utilization.
- Stage 5A may add `32` as an exploratory Only-GVP value because it does not
  carry the ESM/fusion memory footprint. Watch every `batch_size=32` trial for
  CUDA OOM and for degraded minority-class recall from fewer optimizer updates
  per epoch.
- Do not include `32` in fusion stages unless a separate memory/quality ablation
  explicitly justifies it.

Recommended G4 budgets (canonical):

| Stage | `OPTUNA_TARGET_COMPLETE_TRIALS` | `MAX_EPOCHS_PER_TRIAL` | `OPTUNA_N_STARTUP_TRIALS` |
| --- | --- | --- | --- |
| Stage 3 (debug) | 4 | 3 | 4 |
| Stage 4 (medium per family) | 64 | 35 | 20 |
| Stage 5A (Only-GVP) | 200 | 50 | 40 |
| Stage 5B (Only-ESM) | 120 | 50 | 30 |
| Stage 5C (GVP+late) | 200 | 50 | 40 |
| Stage 5D (GVP+node-late) | 200 | 50 | 40 |
| Stage 5E (GVP+hybrid) | 200 | 50 | 40 |
| Stage 5F (GVP+cross-attn) | 120 | 50 | 30 |

Serious learning-rate ranges:

| Stage | `OPTUNA_LEARNING_RATE_RANGE` |
| --- | --- |
| Stage 3 | `1e-5,3e-4` |
| Stage 4 | `1e-5,3e-4` |
| Stage 5A | `5e-6,3e-4` |
| Stage 5B | `5e-6,2e-4` |
| Stage 5C | `5e-6,2e-4` |
| Stage 5D | `5e-6,2e-4` |
| Stage 5E | `5e-6,1.5e-4` |
| Stage 5F | `5e-6,1e-4` |

Serious LR schedule choices:

| Stage | `LR_SCHEDULES_CSV` |
| --- | --- |
| Stage 5A | `fixed,cosine` |
| Stage 5C | `fixed,cosine` |
| Stage 5D | `fixed,cosine` |
| Stage 5E | `fixed,cosine` |

Do not add `step` to Optuna LR-schedule search until `lr_step_size` and
`lr_decay_gamma` are also part of the Optuna search space. The notebook exposes
manual step-decay controls, but Optuna currently searches only `fixed` and
`cosine`. TODO: warmup is not currently a training CLI/config option, so do not
add warmup choices until a real warmup implementation exists.

Serious class-weight and loss search ranges:

| Stage | Class weighting | Losses | Label smoothing | Sampling balance |
| --- | --- | --- | --- | --- |
| Stage 5A | `none,inverse_frequency,inverse_sqrt_frequency,effective_number` | `cross_entropy,focal` | `0.0,0.03,0.05,0.1` | `False,True` |
| Stage 5B | `none,inverse_frequency,inverse_sqrt_frequency,effective_number` | `cross_entropy` | `0.0,0.03,0.05,0.1` | `False,True` |
| Stage 5C | `inverse_frequency,inverse_sqrt_frequency,effective_number` | `cross_entropy` | `0.0,0.03,0.05,0.1` | `False,True` |
| Stage 5D | `inverse_frequency,inverse_sqrt_frequency,effective_number` | `cross_entropy` | `0.0,0.03,0.05,0.1` | `False,True` |
| Stage 5E | `inverse_frequency,inverse_sqrt_frequency,effective_number` | `cross_entropy` | `0.0,0.03,0.05,0.1` | `False,True` |
| Stage 5F | `inverse_frequency,inverse_sqrt_frequency,effective_number` | `cross_entropy` | `0.0,0.03,0.05,0.1` | `False` |

Serious capacity/search-space policy:

- Stage 5A searches narrowed-from-the-top GVP capacity, edge radius, head
  dropout, training-only regularization/augmentation, class weighting, focal
  loss, and batch size inside `MODEL_PRESET = "Only-GVP"`.
- Stage 5B searches ESM-only classifier capacity and metal imbalance settings
  inside `MODEL_PRESET = "Only-ESM"`.
- Stage 5C/5D search narrowed-from-the-top GVP capacity, edge radius, ESM fusion
  dimension, head dropout, ESM graph encoder dropout, training-only
  regularization/augmentation, class weighting, and batch size inside their
  single fusion preset.
- Stage 5E additionally searches early-ESM bottleneck and dropout.
- Stage 5F keeps attention narrow: one layer, limited heads/dropout, no
  bidirectionality in the first serious search.
- Common training-only graph augmentation defaults remain off. Canonical Stage
  4, 5A, 5C, 5D, 5E, and 5F blocks explicitly sample position noise and
  outer-residue dropout inside one model-family study. Augmentation never runs
  for validation or held-out test inference.

## Conservative First-Pass Anti-Overfitting GVP Profile

Use this profile as a recommended conservative starting point for GVP-based
metal-focused runs when the goal is to reduce overfitting risk before a wider
second-stage expansion. It is not a universal optimum, not held-out-test
selected evidence, and not a replacement for Stage 6 confirmation.

The current GVP input is already information-rich. Node scalar inputs include
amino-acid chemistry, hydrophobicity, donor/acceptor/aromatic/acidic/basic
flags, shell role, distance/RBF-derived terms, and burial/SASA/electrostatics/
PROPKA-like features where available. The graph also has explicit residue
vector channels plus edge scalar, RING, and radius features. Because the
dataset is modest, first-stage capacity should stay conservative.

Main capacity knobs:

- `HIDDEN_S_VALUES_CSV`
- `HIDDEN_V_VALUES_CSV`
- `EDGE_HIDDEN_VALUES_CSV`
- `GVP_LAYERS_VALUES_CSV`
- `EDGE_RADIUS_VALUES_CSV`
- `ESM_FUSION_DIM_VALUES_CSV`
- `EARLY_ESM_DIM_VALUES_CSV`
- `HEAD_MLP_LAYERS_VALUES_CSV`

Optional range overrides for a deliberately labeled range-search experiment:

- `OPTUNA_WEIGHT_DECAY_RANGE`
- `OPTUNA_HIDDEN_S_RANGE`, `OPTUNA_HIDDEN_V_RANGE`, `OPTUNA_EDGE_HIDDEN_RANGE`
- `OPTUNA_GVP_LAYERS_RANGE`, `OPTUNA_HEAD_MLP_LAYERS_RANGE`
- `OPTUNA_EDGE_RADIUS_RANGE`, `OPTUNA_CLASSIFIER_POOL_DISTANCE_CUTOFF_RANGE`
- `OPTUNA_HEAD_MLP_DROPOUT_RANGE`, `OPTUNA_ESM_GRAPH_ENCODER_DROPOUT_RANGE`
- `OPTUNA_POSITION_NOISE_STD_RANGE`, `OPTUNA_SECOND_SHELL_DROPOUT_RANGE`,
  `OPTUNA_OUTER_RESIDUE_DROPOUT_RANGE`
- `OPTUNA_ESM_FUSION_DIM_RANGE`, `OPTUNA_EARLY_ESM_DIM_RANGE`
- `OPTUNA_CROSS_ATTENTION_LAYERS_RANGE`, `OPTUNA_EARLY_ESM_DROPOUT_RANGE`,
  `OPTUNA_CROSS_ATTENTION_DROPOUT_RANGE`
- `OPTUNA_METAL_LABEL_SMOOTHING_RANGE`, `OPTUNA_METAL_FOCAL_GAMMA_RANGE`
- `OPTUNA_METAL_COLLAPSED_LOSS_WEIGHT_RANGE`,
  `OPTUNA_METAL_LOSS_WEIGHT_RANGE`, `OPTUNA_EC_LOSS_WEIGHT_RANGE`

Leave these blank for the canonical CSV-based stage blocks below unless the run
is explicitly named and documented as a validation-only range-search variant.

Notebook profile:

```python
RING_EDGE_MODE = "with_ring"
METAL_NODE_MODE = "per_metal"
STRUCTURAL_READOUT_SCOPE = "auto"

CLASSIFIER_POOL_DISTANCE_CUTOFF_VALUES_CSV = "0.0"
HIDDEN_S_VALUES_CSV = "128"
HIDDEN_V_VALUES_CSV = "8,16"
EDGE_HIDDEN_VALUES_CSV = "64"
GVP_LAYERS_VALUES_CSV = "2,3"
HEAD_MLP_LAYERS_VALUES_CSV = "1,2"
EDGE_RADIUS_VALUES_CSV = "6,8"
ESM_FUSION_DIM_VALUES_CSV = "64,128"
EARLY_ESM_DIM_VALUES_CSV = "32,48"

HEAD_MLP_DROPOUT_VALUES_CSV = "0.2"
ESM_GRAPH_ENCODER_DROPOUT_VALUES_CSV = "0.1"
EARLY_ESM_DROPOUT_VALUES_CSV = "0.05"  # 0.1 is also acceptable for the first pass.
CROSS_ATTENTION_DROPOUT_VALUES_CSV = "0.1"

POSITION_NOISE_STDS_CSV = "0.0,0.03,0.05"
SECOND_SHELL_DROPOUTS_CSV = "0.0"
OUTER_RESIDUE_DROPOUTS_CSV = "0.0,0.1"
```

Rationale:

- `hidden_s=128`, `hidden_v=8/16`, `edge_hidden=64`, and 2-3 GVP layers are
  appropriate low-capacity starting values for roughly one thousand samples.
- `edge_radius=6/8` keeps the radius graph local; radius `10` or higher is a
  second-stage option.
- `esm_fusion_dim=256`, `hidden_s>=192`, `hidden_v>=24`,
  `edge_hidden>=128`, and `gvp_layers>=4` are higher-capacity options and
  should not be first-stage anti-overfitting defaults.
- Position noise and residue dropout are training-only robustness tools. Keep
  coordinate noise mild for metal-site geometry. If using AlphaFold structures,
  mild training-only coordinate noise can be considered, but validation and
  held-out test graphs must remain unchanged.
- Do not claim that coordinate noise or residue dropout improves performance
  without validation evidence.

Budget tiers:

| Profile | `OPTUNA_TARGET_COMPLETE_TRIALS` | `MAX_EPOCHS_PER_TRIAL` / `OPTUNA_SEARCH_HPO_TRIAL_EPOCHS` | `OPTUNA_N_STARTUP_TRIALS` |
| --- | --- | --- | --- |
| Conservative first pass | 64 or 80 | 35-40 | 15-20 |
| Strong controlled | 100 | 50 | 20 |
| Extended serious | Use the canonical Stage 5 table above | Use the canonical Stage 5 table above | Use the canonical Stage 5 table above |

Two hundred complete trials is an extended serious search, not a simple
first-pass anti-overfitting search. Two-hundred-trial studies are acceptable
only when followed by predeclared Stage 6 top-K grouped-fold/seed
confirmation. Do not interpret one validation split or the best single Optuna
trial as conclusive.

This profile applies broadly to GVP-based metal-focused DeepMzyme runs. It is
not specific to `TASK = "joint"`, `METAL_LABEL_SCHEME = "five_class"`,
`MODEL_PRESET = "GVP + hybrid fusion"`, or
`SELECTION_METRIC = "val_metal_balanced_acc"`. If `TASK = "joint"` and
`SELECTION_METRIC = "val_metal_balanced_acc"`, model selection is primarily
metal-optimized and the EC branch is auxiliary. If the goal is EC prediction,
use an EC validation metric instead.

Feature-omission ablations use notebook `OMIT_NODE_FEATURE_SETS`; the CLI flag
is `--omit-node-features`. Do not invent additional notebook omission
variables.

## Optional Objective Experiments

These objective variants are experimental validation-only tools. They are not
defaults, must not be used in Stage 2 baselines, and must not change the
one-shot held-out test policy.

### Optional collapsed-4 auxiliary metal loss

`METAL_COLLAPSED_LOSS_WEIGHTS_CSV` maps to the CLI flag
`--metal-collapsed-loss-weight`; in single mode the first CSV value is used.
The default is `0.0`, which preserves the
standard six-class objective exactly. When enabled with cross-entropy metal
loss, the training objective is:

```text
L_total = (1 - alpha) * CE_6class + alpha * CE_4class
```

The collapsed view is deterministic: `Mn`, `Cu`, `Zn`, and `Class VIII`, where
`Class VIII = Fe + Co + Ni`. The collapsed logits are computed by log-sum-exp
marginalization from the six-class logits. Six-class metal classification
remains the primary task and primary report; collapsed-4 metrics are
supplemental and must not hide Fe/Co/Ni failures.

First-use rule: test this only against the current validation baseline before
using it broadly. For a Stage 5A-only validation experiment, use:

```python
METAL_COLLAPSED_LOSS_WEIGHTS_CSV = "0.0,0.3,0.5"
METAL_LOSS_FUNCTIONS_CSV = "cross_entropy"
```

Do not add this search axis automatically to Stage 5B-5F. Add it there only
after a Stage 5A validation comparison and Stage 6 confirmation show that it
improves six-class balanced accuracy without rare-class recall collapse.

### Optional five-class metal target scheme

The default reportable metal target is `METAL_LABEL_SCHEME = "six_class"`:
`Mn`, `Cu`, `Zn`, `Fe`, `Co`, and `Ni`.

For an explicitly labeled validation-only comparison, use:

```python
METAL_LABEL_SCHEME = "five_class"
```

This changes the training target to five classes: `Mn`, `Cu`, `Zn`, `Fe`, and a
grouped Co/Ni class. It is not a display toggle and is not the same as
`METAL_REPORT_VIEW = "collapsed4"`. When using it, create a new
`RUN_BATCH_ID`, `SUMMARY_BASENAME`, `OPTUNA_STUDY_NAME`, and persistent storage
file so five-class evidence cannot mix with six-class evidence. The notebook
auto-derives the run-name prefix from `RUN_BATCH_ID`. Keep
`SELECTION_METRIC = "val_metal_balanced_acc"`; that metric is then balanced
accuracy over the active five-class target.

Do not use five-class validation numbers to replace or rank six-class anchors
without a separately documented comparison goal. Stage 6, Stage 6B, and Stage 7
source runs must all use the same `METAL_LABEL_SCHEME`.

#### Five-class joint hybrid metal-target overlay

Use this overlay for an explicitly labeled pocket-level validation diagnostic
of the joint-task GVP + hybrid fusion family while selecting checkpoints by
metal balanced accuracy. It keeps the five-class target separate from six-class
evidence and applies additional metal-loss multipliers to Fe and Mn on top of
the selected training-split metal class-weight mode.

Notebook configuration block:

```python
TASK = "joint"
METAL_LABEL_SCHEME = "five_class"
RUN_MODE = "single"
RECOMMENDED_RUN_SET = "custom"
MODEL_PRESET = "GVP + hybrid fusion"
RUN_BATCH_ID = "joint_fiveclass_hybrid_metal_target_fe1p7_mn1p5_splitpocket_single"
SUMMARY_BASENAME = ""  # auto from provenance

EPOCHS = 50
VAL_FRACTION = 0.15
SPLIT_BY = "pocket_id"
SELECTION_METRIC = "val_metal_balanced_acc"
RING_EDGE_MODE = "with_ring"

ESM_EMBEDDINGS_DIR = ""  # set to your embeddings folder when available
ALLOW_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_ESM_EMBEDDINGS = True

LEARNING_RATES_CSV = "3.705631497756492e-05"
WEIGHT_DECAYS_CSV = "3e-07"
BATCH_SIZES_CSV = "12"
SEEDS_CSV = "42"
LR_SCHEDULES_CSV = "fixed"

HIDDEN_S_VALUES_CSV = "320"
HIDDEN_V_VALUES_CSV = "16"
EDGE_HIDDEN_VALUES_CSV = "192"
GVP_LAYERS_VALUES_CSV = "4"
HEAD_MLP_LAYERS_VALUES_CSV = "2"
EDGE_RADIUS_VALUES_CSV = "7.0"
ESM_FUSION_DIM_VALUES_CSV = "256"
EARLY_ESM_DIM_VALUES_CSV = "48"
EARLY_ESM_DROPOUT_VALUES_CSV = "0.05"

METAL_CLASS_WEIGHT_MODES_CSV = "effective_number"
BALANCE_METAL_SITE_SYMBOLS_CSV = "False"
METAL_LOSS_FUNCTIONS_CSV = "cross_entropy"
METAL_LABEL_SMOOTHING_VALUES_CSV = "0.0"
METAL_COLLAPSED_LOSS_WEIGHTS_CSV = "0.0"
MN_LOSS_MULTIPLIER = 1.5
FE_LOSS_MULTIPLIER = 1.7
CU_LOSS_MULTIPLIER = 1.0
ZN_LOSS_MULTIPLIER = 1.0
CO_LOSS_MULTIPLIER = 1.0
NI_LOSS_MULTIPLIER = 1.0
CLASS_VIII_LOSS_MULTIPLIER = 1.0

METAL_LOSS_WEIGHT_VALUES_CSV = "2.0"
EC_LOSS_WEIGHT_VALUES_CSV = "0.25"
EC_LABEL_DEPTHS_CSV = "1"
EC_CONTRASTIVE_WEIGHTS_CSV = "0.0"
EC_GROUP_WEIGHTING = "structure_id"

INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

Expected outputs/files:

- One validation-only run directory under `<RUNS_DIR>/<RUN_BATCH_ID>/...`.
- `best_model_checkpoint.pt`, `metrics_history.csv`, `run_config.json`,
  `run_metadata.json`, and split/validation artifacts in the run directory.
- Notebook-generated `active_run_config.json` and `active_run_config.md` before
  launch.
- Summary artifacts using `SUMMARY_BASENAME` after the notebook summary cells.

Decision gate:

- Treat this as a one-off validation-only pocket-level diagnostic, not a
  replacement anchor.
- Proceed to Stage 6 only if the run completes without held-out-test output,
  uses `METAL_LABEL_SCHEME = "five_class"`, `TASK = "joint"`,
  `MODEL_PRESET = "GVP + hybrid fusion"`,
  `SELECTION_METRIC = "val_metal_balanced_acc"`, `VAL_FRACTION = 0.15`,
  `SPLIT_BY = "pocket_id"`, and the saved configs show
  `MN_LOSS_MULTIPLIER = 1.5` and `FE_LOSS_MULTIPLIER = 1.7`.
- Do not compare this pocket-level validation value directly against grouped
  `pdbid` validation anchors, because `SPLIT_BY = "pocket_id"` is a different
  and less conservative split policy.
- Do not compare the five-class validation value directly against six-class
  anchors. Any promotion requires same-scheme Stage 6 grouped-fold
  confirmation with shared folds, paired bootstrap confidence intervals, and
  rare-class recall protection.

### Optional multi-objective Optuna

`OPTUNA_MULTIOBJECTIVE = True` creates a validation-only multi-objective Optuna
study with objectives:

- maximize `val_metal_balanced_acc`
- maximize `val_metal_min_recall`

The second objective is minimum recall across active metal-scheme validation
classes with support > 0. For default reportable runs this is six-class
minimum recall; for explicitly labeled `five_class` runs it is five-class
minimum recall over `Mn`, `Cu`, `Zn`, `Fe`, and grouped Co/Ni. Do not use
`val_metal_collapsed4_min_recall` as the default rare-class objective, because
it can hide Fe/Co/Ni failures in six-class runs. Collapsed-4 minimum recall is
still reported as supplemental context.

Multi-objective HPO writes Pareto review files:

- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/pareto_front.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/pareto_candidates.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/pareto_candidates_ranked_for_review.csv`

The ranked file is a review convenience only. It does not replace Stage 6
grouped-fold confirmation. If pruning is incompatible with the active Optuna
multi-objective study, the notebook disables pruning and prints a warning.

Recommended Stage 5A overlay:

```python
OPTUNA_MULTIOBJECTIVE = True
OPTUNA_SELECTION_METRIC = "val_metal_balanced_acc"
OPTUNA_USE_PRUNING = False
OPTUNA_PRUNER_TYPE = "none"
```

## Canonical G4 Metal Training Route

Use this route when starting a clean, serious metal-classification campaign in
`notebooks/DeepMzyme_training_colab.ipynb`.

### Required order

Recommended linear G4 route:

Stage 0 -> Stage 1 -> Stage 2A -> Stage 2B if ESM is ready -> Stage 3 ->
Stage 5A -> Stage 6 -> Stage 5B/5C/5D/5E/5F only if their gates pass ->
Stage 6B -> Stage 7.

Interpretation:

1. Stage 0: environment/data readiness.
2. Stage 1: 1-epoch smoke.
3. Stage 2A: Only-GVP validation anchor.
4. Stage 2B: baseline family comparison, only after ESM coverage is valid.
5. Stage 3: Optuna plumbing debug.
6. Stage 5A: serious Only-GVP HPO.
7. Stage 6: top-K grouped-fold confirmation for the Only-GVP HPO candidates.
8. Stage 5B and Stage 5C: run only if ESM coverage and baseline gates pass.
9. Stage 5D, Stage 5E, and Stage 5F: run only after the advanced-fusion gate
   below passes.
10. Stage 6 again for every HPO family that may become the final selected
    configuration.
11. Stage 6B: apply promotion gates and train/refit the single selected
    configuration on the full non-test training set.
12. Stage 7: one-shot held-out test for the frozen Stage 6B final-refit run.

Stage 4 is optional on a G4 GPU and mainly for sanity HPO, search-space
debugging at useful scale, or limited-compute campaigns. For a serious fresh
G4 search, Stage 5 is preferred after Stage 3 passes.

### Advanced fusion gate

Do not launch Stage 5D, Stage 5E, or Stage 5F until Stage 5C has produced a
Stage 6 grouped-fold candidate that clears the paired validation-improvement
threshold defined in this playbook's Stage 5C decision gate.


If this gate is not passed, stop advanced fusion escalation and continue with
the best validated simpler family.

### Final-selection rule


The final selected model must come from Stage 6 grouped-fold validation plus
Stage 6B promotion gates, not from a single Optuna trial. Stage 6B ranks by mean
`val_metal_balanced_acc`, promotes only when the paired bootstrap CI and
rare-recall gates pass, then uses configured tie-breakers such as standard
deviation, worst fold, and model simplicity.

Stage 6 selects the configuration; Stage 6B creates the final test source. Keep
`MODEL_PRESET`, model hyperparameters, feature policy, `METAL_LABEL_SCHEME`,
training budget, fixed final-refit seed, checkpoint rule, calibration rule, and
any ensemble rule fixed from validation evidence before the Stage 6B final
refit starts. Do not choose any of these from held-out test results.

The held-out test is used only once, in Stage 7, after this validation-based
selection and the Stage 6B final-refit run are frozen.

## Optuna Study Naming And Storage

Study naming: `metal_<preset_slug>_<size>_<purpose>`, for example
`metal_only_gvp_200_capacity` or `metal_late_fusion_200_controlled`. Always use
lowercase, underscore-separated names.

Storage path template:
`sqlite:////content/drive/MyDrive/DeepMzyme/optuna/<study_name>.db`. Use one
file per study. Never share a study DB across different `MODEL_PRESET` values.

Resumption rule: re-running the notebook with the same `OPTUNA_STUDY_NAME` and
storage URL resumes the persistent study and launches only the remaining trials
needed to reach `OPTUNA_TARGET_COMPLETE_TRIALS` completed trials. `N_OPTUNA_TRIALS`
is still accepted as a backward-compatible alias in older snippets. To start
fresh, change the study name; do not delete the `.db` unless you mean to discard
history.

Resume policy for reportable HPO:

- Resume only into the same `MODEL_PRESET`, task, split policy, selection
  metric, metal label scheme, search space, and storage URL.
- If any of those values changed, use a new `OPTUNA_STUDY_NAME` and new SQLite
  file.
- If a study was interrupted, resume until the requested number of `COMPLETE`
  trials in the stage gate is reached. Failed/pruned/incomplete trials do not
  count toward the required completed-trial count, so they may make the stored
  total trial count exceed `OPTUNA_TARGET_COMPLETE_TRIALS` while the completed-trial count
  reaches the target.
- The notebook records compatibility metadata and a search-space hash in each
  study. Reusing a persistent study with incompatible metadata stops with a
  clear error unless `OPTUNA_ALLOW_INCOMPATIBLE_STUDY_REUSE = True`.

If a run uses `Only-ESM` or any GVP + ESM fusion preset, set
`ESM_EMBEDDINGS_DIR` to the embeddings folder or set
`PREPARE_MISSING_ESM_EMBEDDINGS = True` deliberately. Do not use
`ALLOW_MISSING_ESM_EMBEDDINGS = True` for reportable runs.

## Exact Run-Configuration Artifacts

Every launched training run should produce a machine-readable record of the
configuration that actually ran. The current training code writes:

- `<run_dir>/run_config.json`
- `<run_dir>/run_metadata.json`
- `<run_dir>/active_run_config.json`
- `<run_dir>/active_run_config.md`

These artifacts are the authoritative record for completed training and
validation runs. They include the resolved config, selection metric, selected
checkpoint metadata, history, metal label scheme, split identity, and embedded
test information when test evaluation was requested. `active_run_config.json` and
`active_run_config.md` are written by the notebook before launch from the live
notebook variables and command configuration; they are especially useful for
failed or pruned subprocess trials that may not reach `run_config.json`.

For serious validation and Stage 7 final-test runs, verify that the run record
also captures the dataset bundle filename/checksum when a bundle is used, the
git commit, and key library versions. If a current artifact lacks one of these
fields, note the gap in the run summary or `EXPERIMENT_STATUS.md` rather than
inferring it later.

For Optuna studies, the notebook also writes
`<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/optuna_study_metadata.json` plus
study-level `active_run_config.json` / `active_run_config.md`.
When `OPTUNA_MULTIOBJECTIVE = True`, it also writes `pareto_front.csv`,
`pareto_candidates.csv`, and `pareto_candidates_ranked_for_review.csv`. These
Pareto files are validation-review artifacts only and do not authorize held-out
test evaluation without Stage 6 confirmation.

## Stage 0 - Environment/Data Readiness

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
SUMMARY_BASENAME = ""  # auto from provenance

EPOCHS = 1
BATCH_SIZES_CSV = "4"
LEARNING_RATES_CSV = "3e-5"
WEIGHT_DECAYS_CSV = "1e-4"
SEEDS_CSV = "42"

DATASET_NAME = "train_and_test_sets_structures_exact_pinmymetal"
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_metal_balanced_acc"

RING_EDGE_MODE = "with_ring"
REQUIRE_RING_EDGES = False
PREPARE_MISSING_RING_EDGES = True
RING_FEATURES_DIR = ""
RING_EXE_PATH = "DeepMzyme_Data/ring-4.0/out/bin/ring"
ESM_EMBEDDINGS_DIR = ""
ALLOW_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_ESM_EMBEDDINGS = False
ALLOW_MISSING_EXTERNAL_FEATURES = False
PREPARE_MISSING_EXTERNAL_FEATURES = True

INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
```

In the dedicated **Main planned training launch switch** cell:

```python
LAUNCH_PLANNED_MAIN_TRAINING_RUNS = False   # planning/helper loading only; do NOT train at Stage 0
```

Success criteria for Stage 0:

- `RUNS_DIR` resolves under `<DRIVE_ROOT>/notebook_outputs/runs`.
- Planning cell prints RING coverage >= 95% (or generation will run).
- Planning cell prints external-features coverage = 100%.
- No model preset mismatch warnings.

Expected outputs/files:

- `<RUNS_DIR>/<SUMMARY_BASENAME>_planned_runs.csv`
- `<RUNS_DIR>/<SUMMARY_BASENAME>_planned_run_dictionary.json`
- `<RUNS_DIR>/<SUMMARY_BASENAME>_metal_weight_diagnostics.csv`, when metal
  diagnostics are generated
- Printed resolved configuration, feature coverage, split diagnostics, and
  shell-safe command preview
- No training run directory and no `test_report.json`

Exact configuration record:

- Planning-only stage: use the planned-run CSV/dictionary and printed
  configuration summary.
- `active_run_config.json` / `active_run_config.md` are generated from the live
  notebook configuration before launch.

### Decision gate after Stage 0

Proceed to Stage 1 only if:

- All four Stage 0 success criteria are met.
- The planned-run table contains exactly one planned Stage 0 row and zero
  launched training runs.
- The expected planned-run CSV and dictionary exist.
- No held-out test files were created.
- `val_metal_balanced_acc` is the selection metric on the planned run.
- Diagnostics report every active metal-scheme class present in both train and validation splits.
- Rare-class protection passes at the split level: no active metal-scheme class is missing
  from train or validation diagnostics.

If gate fails: fix paths, Drive mounting, bundle selection, RING coverage, or
external-feature coverage before any training.

## Stage 1 - 1-Epoch Smoke

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
SUMMARY_BASENAME = ""  # auto from provenance

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

In the dedicated **Main planned training launch switch** cell, then run
**Optional training execution**:

```python
LAUNCH_PLANNED_MAIN_TRAINING_RUNS = True
```

Expected outputs/files:

- `<RUNS_DIR>/<SUMMARY_BASENAME>_planned_runs.csv`
- `<RUNS_DIR>/<SUMMARY_BASENAME>_planned_run_dictionary.json`
- One run directory under `<RUNS_DIR>/`
- `run_metadata.json`, `run_config.json`, `split_diagnostics.json`
- `dataset_summary.json`, `prepare_status.json`
- No `test_report.json`

Exact configuration record:

- `<run_dir>/run_config.json` and `<run_dir>/run_metadata.json`.
- `active_run_config.json` / `active_run_config.md` are generated from the live
  notebook configuration before launch.

Success criteria:

- Planning prints one runnable Only-GVP command.
- Training completes without missing-path, split, feature, or CLI errors.
- Train and validation metal diagnostics are printed.
- No held-out test report is produced.

### Decision gate after Stage 1

Proceed to Stage 2A only if:

- The Stage 1 success criteria are met.
- Exactly one Stage 1 validation-only run directory was launched and completed.
- The expected planned files and run-level JSON files exist.
- No held-out test files were created.
- `val_metal_balanced_acc` is the selection metric on all completed runs.
- Diagnostics report every active metal-scheme class present in both train and validation splits.
- Rare-class protection passes at the split level: no active metal-scheme class is missing
  from train or validation diagnostics. Do not use the 1-epoch recall values as
  model-quality evidence.

If gate fails: return to Stage 0 and fix paths, bundle setup, RING executable
configuration, structure parsing, ESM coverage, or feature availability before
running real comparisons. Ignore the 1-epoch metric as model-quality evidence.

## Stage 2 - Baseline Validation

Purpose: establish clean validation baselines before adding complex fusion or
large HPO.

When to use it: after smoke passes and before Optuna or advanced fusion. If ESM
embeddings are not ready, run the Only-GVP block first. Once ESM embeddings are
ready, run the ESM-ready baseline block.

Expected scale/runtime: medium validation run, hours. Runtime depends on GPU,
ESM coverage, and whether embeddings must be prepared.

### Stage 2A - Only-GVP Validation Anchor

Notebook configuration block:

```python
TASK = "metal"
RUN_MODE = "manual_configurations"
RECOMMENDED_RUN_SET = "only_gvp_broad_comparison"
MODEL_PRESET = "Only-GVP"
RUN_BATCH_ID = "metal_only_gvp_baseline_lr_seed"
SUMMARY_BASENAME = ""  # auto from provenance

EPOCHS = 50
BATCH_SIZES_CSV = "8"
LEARNING_RATES_CSV = "3e-5,1e-4,3e-4"
WEIGHT_DECAYS_CSV = "1e-4"
SEEDS_CSV = "42,123,2026,7,2718"
MAX_CONFIGURATION_RUNS = 15

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

In the dedicated **Main planned training launch switch** cell, then run
**Optional training execution**:

```python
LAUNCH_PLANNED_MAIN_TRAINING_RUNS = True
```

Expected outputs/files:

- `<RUNS_DIR>/<SUMMARY_BASENAME>_planned_runs.csv`
- `<RUNS_DIR>/<SUMMARY_BASENAME>_planned_run_dictionary.json`
- Fifteen completed validation-only run directories under `<RUNS_DIR>/`
- Each completed run directory contains `run_config.json`,
  `run_metadata.json`, `split_diagnostics.json`, `dataset_summary.json`, and
  `prepare_status.json`
- `<RUNS_DIR>/<SUMMARY_BASENAME>.csv`
- `<RUNS_DIR>/<SUMMARY_BASENAME>_completed_only.csv`
- `<RUNS_DIR>/<SUMMARY_BASENAME>.png`, when plotting succeeds
- No `test_report.json`

Exact configuration record:

- Per run: `<run_dir>/run_config.json` and `<run_dir>/run_metadata.json`.
- Batch plan: planned-run CSV/dictionary.
- `active_run_config.json` / `active_run_config.md` are generated from the live
  notebook configuration before launch.

### Decision gate after Stage 2A

Proceed to Stage 2B or Stage 4 only if:

- The Only-GVP validation baseline completes all 15 planned validation-only
  runs.
- The expected planned files, summary files, and run-level JSON files exist.
- No held-out test files were created.
- `val_metal_balanced_acc` is the selection metric on all completed runs.
- Diagnostics report every active metal-scheme class present in both train and validation splits.
- Rare-class recall protection passes: `val_metal_min_recall` and per-class
  recall are available in the run artifacts, and no candidate is promoted if a
  metal class has zero recall across the completed validation runs.
- Stage 2A anchor reliability is sufficient: the standard G4 block uses five
  seeds, `42,123,2026,7,2718`. If a compute-constrained run uses fewer seeds,
  mark the Stage 2A anchor as provisional and record the reason in
  `EXPERIMENT_STATUS.md`.
- Seed variance is acceptable: if seed standard deviation or high-low spread
  suggests `val_metal_balanced_acc` variance above 0.04, rerun with the
  recommended five-seed list before Stage 2B or Stage 4.

If gate fails: rerun Stage 2A with the recommended five-seed list before any
Stage 2B/4 decision, or return to Stage 0/1 if the failure is path, feature, or
split related.

### Stage 2B - Baseline Family Comparison

Run this only after ESM embeddings are available or after you intentionally allow
the notebook to prepare missing ESM embeddings.

Notebook configuration block:

```python
TASK = "metal"
RUN_MODE = "manual_configurations"
RECOMMENDED_RUN_SET = "baseline_model_comparison"
# MODEL_PRESET is overridden by baseline_model_comparison (runs Only-GVP, Only-ESM, GVP + late fusion)
RUN_BATCH_ID = "metal_baseline_model_comparison"
SUMMARY_BASENAME = ""  # auto from provenance

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

In the dedicated **Main planned training launch switch** cell, then run
**Optional training execution**:

```python
LAUNCH_PLANNED_MAIN_TRAINING_RUNS = True
```

Expected outputs/files:

- Planned-run CSV and dictionary under `<RUNS_DIR>/`
- Twelve completed validation-only run directories for the exact block above
- `<SUMMARY_BASENAME>.csv` and `<SUMMARY_BASENAME>_completed_only.csv`
- `<SUMMARY_BASENAME>.png` when plotting succeeds
- Each completed run directory contains `run_config.json`,
  `run_metadata.json`, `split_diagnostics.json`, `dataset_summary.json`, and
  `prepare_status.json`
- No `test_report.json`

Exact configuration record:

- Per run: `<run_dir>/run_config.json` and `<run_dir>/run_metadata.json`.
- Batch plan: planned-run CSV/dictionary.
- `active_run_config.json` / `active_run_config.md` are generated from the live
  notebook configuration before launch.

Success criteria:

- All planned runs complete.
- Each run uses `selection_metric = val_metal_balanced_acc`.
- `split_diagnostics.json` shows usable train/validation class coverage.
- Comparison tables rank only validation, group-kfold validation, or
  explicitly labeled seed-repeat validation rows.

### Decision gate after Stage 2B

Proceed to Stage 3 or Stage 4 only if:

- The Stage 2B success criteria are met.
- The expected planned files, summary files, and run-level JSON files exist.
- No held-out test files were created.
- `val_metal_balanced_acc` is the selection metric on all completed runs.
- Diagnostics report every active metal-scheme class present in both train and validation splits.
- Rare-class recall protection passes: per-class recall is available for each
  completed run, and no family is promoted if its seed mean has zero recall for
  any metal class.

If gate fails: fix ESM coverage, rerun the affected baseline family, or fall
back to the Stage 2A Only-GVP anchor until ESM-ready runs are trustworthy. Choose
baseline anchors by validation evidence, not by held-out test, and prefer
stability across seeds over one high run.

## Stage 3 - Optuna Plumbing Debug

Purpose: verify the controlled Optuna path, storage, command generation, and
search-space parsing without treating the result as model-selection evidence.

When to use it: first Optuna run in a new runtime or after editing Optuna
configuration fields.

Expected scale/runtime: smoke/debug, minutes to under an hour.

Stage 3 caveat: this is a plumbing/debug Optuna run. With the canonical
`OPTUNA_TARGET_COMPLETE_TRIALS = 4` and `OPTUNA_N_STARTUP_TRIALS = 4`, every trial is a TPE
startup trial, so Stage 3 is effectively random search. Stage 3 results are not
model-selection evidence; serious model-selection evidence comes from serious
HPO, validation-only comparison, and Stage 6 grouped-fold confirmation.

Notebook configuration block:

```python
TASK = "metal"
RUN_MODE = "controlled_hpo_optuna"
RECOMMENDED_RUN_SET = "custom"
MODEL_PRESET = "Only-GVP"
RUN_BATCH_ID = "metal_only_gvp_optuna_debug"
SUMMARY_BASENAME = ""  # auto from provenance

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
OPTUNA_TARGET_COMPLETE_TRIALS = 4
MAX_EPOCHS_PER_TRIAL = 3
OPTUNA_N_STARTUP_TRIALS = 4
OPTUNA_TPE_MULTIVARIATE = True
OPTUNA_TPE_GROUP = True
OPTUNA_TPE_CONSTANT_LIAR = True
OPTUNA_PARALLEL_WORKERS = 1
OPTUNA_PARALLEL_STARTUP_STAGGER_SECONDS = 10.0
OPTUNA_STOP_ON_PARALLEL_CUDA_OOM = True
OPTUNA_AUTO_CONFIGURE_BUDGET = False
OPTUNA_USE_PRUNING = False
OPTUNA_PRUNER_TYPE = "none"
OPTUNA_PRUNING_MIN_EPOCH = 2
OPTUNA_SEARCH_PRESET = "first_useful_only_gvp_narrow"
OPTUNA_STUDY_NAME = "metal_only_gvp_optuna_debug"
OPTUNA_STORAGE = "sqlite:////content/drive/MyDrive/DeepMzyme/optuna/metal_only_gvp_optuna_debug.db"
OPTUNA_SPLIT_SEED = 42
OPTUNA_SAMPLER_SEED = None
OPTUNA_LEARNING_RATE_RANGE = "1e-5,3e-4"
WEIGHT_DECAYS_CSV = "0.0,1e-5,1e-4"
BATCH_SIZES_CSV = "4,8"
METAL_CLASS_WEIGHT_MODES_CSV = "none,inverse_frequency,inverse_sqrt_frequency,effective_number"
METAL_LOSS_FUNCTIONS_CSV = "cross_entropy"
METAL_LABEL_SMOOTHING_VALUES_CSV = "0.0"
BALANCE_METAL_SITE_SYMBOLS_CSV = "False"
RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = False

INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

In the dedicated **Main planned training launch switch** cell, then run
**Optional training execution**:

```python
LAUNCH_PLANNED_MAIN_TRAINING_RUNS = True
```

Expected outputs/files:

- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/all_trials.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/top_trials.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/best_trial.json`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/optuna_study_metadata.json`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/active_run_config.json`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/active_run_config.md`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/optuna_study_summary.md`
- `top_reevaluation_commands.txt`
- Four per-trial validation-only run directories under `<RUNS_DIR>/`
- Per-trial `active_run_config.json`, `active_run_config.md`,
  `run_config.json`, `run_metadata.json`, and `split_diagnostics.json`
- No `test_report.json`

Exact configuration record:

- Study level: Optuna CSV/JSON/Markdown outputs listed above.
- Per trial: `<run_dir>/active_run_config.json`,
  `<run_dir>/active_run_config.md`, `<run_dir>/run_config.json`, and
  `<run_dir>/run_metadata.json`.

Success criteria:

- Optuna launches and completes the debug trials.
- Search-space preview shows architecture fixed to Only-GVP.
- Trial commands omit held-out test evaluation.

### Decision gate after Stage 3

Proceed to Stage 4 or Stage 5A only if:

- The Stage 3 success criteria are met.
- `all_trials.csv` has exactly 4 trials and all 4 are `COMPLETE`.
- The expected Optuna files and per-trial run-level JSON files exist.
- No held-out test files were created.
- `val_metal_balanced_acc` is the selection metric on all completed trials.
- One `MODEL_PRESET` is used in the study: `Only-GVP`.
- Diagnostics report every active metal-scheme class present in both train and validation splits.
- Rare-class protection passes at the split/diagnostic level. Do not promote
  any Stage 3 hyperparameter result as model-selection evidence.

If gate fails: fix Optuna storage, search-space parsing, command generation, or
feature paths before launching Stage 4. Do not choose hyperparameters from this
debug run.

## Stage 4 - Medium Per-Family Optuna, Optional On G4

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
SUMMARY_BASENAME = ""  # auto from provenance

EPOCHS = 50
BATCH_SIZES_CSV = "8"
LEARNING_RATES_CSV = "3e-5"
WEIGHT_DECAYS_CSV = "1e-5,1e-4,1e-3"
SEEDS_CSV = "42"

HIDDEN_S_VALUES_CSV = "128"
HIDDEN_V_VALUES_CSV = "8,16"
EDGE_HIDDEN_VALUES_CSV = "64"
GVP_LAYERS_VALUES_CSV = "2,3"
HEAD_MLP_LAYERS_VALUES_CSV = "2"
EDGE_RADIUS_VALUES_CSV = "8.0"
RING_EDGE_MODE = "with_ring"
REQUIRE_RING_EDGES = False
PREPARE_MISSING_RING_EDGES = True
RING_FEATURES_DIR = ""
RING_EXE_PATH = "DeepMzyme_Data/ring-4.0/out/bin/ring"
ALLOW_MISSING_EXTERNAL_FEATURES = False
PREPARE_MISSING_EXTERNAL_FEATURES = True
EXTERNAL_FEATURES_ROOT_DIR = ""

OPTUNA_INTENSITY = "custom"
OPTUNA_TARGET_COMPLETE_TRIALS = 64
MAX_EPOCHS_PER_TRIAL = 35
OPTUNA_N_STARTUP_TRIALS = 20
OPTUNA_TPE_MULTIVARIATE = True
OPTUNA_TPE_GROUP = True
OPTUNA_TPE_CONSTANT_LIAR = True
OPTUNA_PARALLEL_WORKERS = 1
OPTUNA_PARALLEL_STARTUP_STAGGER_SECONDS = 10.0
OPTUNA_STOP_ON_PARALLEL_CUDA_OOM = True
OPTUNA_AUTO_CONFIGURE_BUDGET = False
OPTUNA_USE_PRUNING = True
OPTUNA_PRUNER_TYPE = "median"
OPTUNA_PRUNING_MIN_EPOCH = 25
OPTUNA_SEARCH_PRESET = "first_useful_only_gvp_narrow"
OPTUNA_STUDY_NAME = "metal_only_gvp_optuna_medium"
OPTUNA_STORAGE = "sqlite:////content/drive/MyDrive/DeepMzyme/optuna/metal_only_gvp_optuna_medium.db"
OPTUNA_SPLIT_SEED = 42
OPTUNA_SAMPLER_SEED = None
OPTUNA_LEARNING_RATE_RANGE = "1e-5,3e-4"
WEIGHT_DECAYS_CSV = "1e-5,1e-4,1e-3"
BATCH_SIZES_CSV = "8,16"
METAL_CLASS_WEIGHT_MODES_CSV = "none,inverse_frequency,inverse_sqrt_frequency,effective_number"
METAL_LOSS_FUNCTIONS_CSV = "cross_entropy"
METAL_LABEL_SMOOTHING_VALUES_CSV = "0.0,0.03,0.05,0.1"
BALANCE_METAL_SITE_SYMBOLS_CSV = "False,True"
HEAD_MLP_DROPOUT_VALUES_CSV = "0.1,0.2,0.3"
POSITION_NOISE_STDS_CSV = "0.0,0.05,0.1"
OUTER_RESIDUE_DROPOUTS_CSV = "0.0,0.1,0.2"
RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = False
# Run Stage 6 later from the dedicated Stage 6 launch cell after HPO completes.
TOP_K_CONFIGS_FOR_SEED_REPEAT = "auto"
REPEAT_SEEDS = "42,123,2026,43,44"

INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

Expected outputs/files:

- Optuna study directory under `<RUNS_DIR>/optuna/`
- `all_trials.csv`, `top_trials.csv`, `best_trial.json`
- `optuna_best_config.json`, `best_config_command.txt`
- `top_reevaluation_commands.txt`
- `optuna_study_summary.md`
- Sixty-four per-trial validation-only run directories under `<RUNS_DIR>/`
- Per-trial `active_run_config.json`, `active_run_config.md`,
  `run_config.json`, `run_metadata.json`, and `split_diagnostics.json`
- No `test_report.json`

Exact configuration record:

- Study level: Optuna CSV/JSON/Markdown outputs listed above.
- Per trial: `<run_dir>/active_run_config.json`,
  `<run_dir>/active_run_config.md`, `<run_dir>/run_config.json`, and
  `<run_dir>/run_metadata.json`.
- `active_run_config.json` / `active_run_config.md` are generated from the live
  notebook configuration before launch.

Success criteria:

- The study completes the requested 64-trial count.
- The best-trial summary is based on `val_metal_balanced_acc`.
- Trial logs show validation-only runs, not final-test runs.
- Top candidates have finite selected validation metrics and no missing-class
  diagnostics.

### Decision gate after Stage 4

Proceed to Stage 5 or Stage 6 only if:

- The Stage 4 success criteria are met.
- `all_trials.csv` contains at least 64 `COMPLETE` trials for this
  `MODEL_PRESET`; resume the same study until that count is reached.
- The expected Optuna files and per-trial run-level JSON files exist.
- No held-out test files were created.
- `val_metal_balanced_acc` is the selection metric on all completed runs.
- One `MODEL_PRESET` is used in the study: `Only-GVP`.
- Diagnostics report every active metal-scheme class present in both train and validation splits.
- Rare-class recall protection passes: top candidates have available
  per-class recall, and no candidate is promoted if any metal class has zero
  recall in its validation artifact.
- At least one top candidate exceeds the Stage 2A seed mean on
  `val_metal_balanced_acc`; otherwise Stage 4 may be used only as
  search-space diagnosis, not as a promotion gate.

If gate fails: check the search space; widen `OPTUNA_LEARNING_RATE_RANGE` or
open `HIDDEN_S_VALUES_CSV`. Do not pick the final model from one Optuna
trial alone; run top-K grouped-fold confirmation before considering a
configuration stable.

## Stage 5 - Serious Per-Family HPO

Purpose: perform a longer, controlled search after the simpler baseline and
medium HPO justify the model family and search axes.

When to use it: after at least one medium HPO or Stage 6 confirmation batch
identifies the model family and search axes worth expanding, or when the user
asks for a fresh broad Optuna check and does not explicitly ask to rely on
previous raw outputs.

Expected scale/runtime: large Optuna search, potentially very long or
overnight. A 200-trial run can be substantially longer than one night depending
on GPU and model.

Important scope rule: the notebook's Optuna mode optimizes within the selected
`MODEL_PRESET`. It does not freely search architectures or fusion modes. Choose
the model family explicitly, then search a controlled set of hyperparameters.

Advanced-fusion ordering rule: Stages 5D, 5E, 5F are only valid after Stage 5C
(GVP + late fusion) has produced a Stage 6 grouped-fold candidate that exceeds
the Stage 2A Only-GVP anchor by at least `0.01` mean
`val_metal_balanced_acc`, and the paired bootstrap 95% CI for that improvement
excludes zero. If Stage 5C does not clear that bar, do not launch 5D/5E/5F.

### Shared Stage 5 Output, Config-Record, And Gate Template

This template applies to Stage 5A-5F unless a substage states an addition or
stricter rule.

Expected outputs/files:

- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/all_trials.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/top_trials.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/best_trial.json`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/optuna_study_metadata.json`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/active_run_config.json`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/active_run_config.md`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/optuna_best_config.json`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/best_config_command.txt`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/top_reevaluation_commands.txt`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/optuna_study_summary.md`
- One complete validation-only per-trial run directory for every required
  completed trial in the substage.
- Per-trial `active_run_config.json`, `active_run_config.md`,
  `run_config.json`, `run_metadata.json`, and `split_diagnostics.json`.
- No `test_report.json`.

Exact configuration record:

- Study level: Optuna CSV/JSON/Markdown outputs listed above.
- Per trial: `<run_dir>/active_run_config.json`,
  `<run_dir>/active_run_config.md`, `<run_dir>/run_config.json`, and
  `<run_dir>/run_metadata.json`.

Common decision-gate requirements:

- The expected Optuna files and per-trial run-level JSON files exist.
- `all_trials.csv` contains the required number of `COMPLETE` trials for the
  substage and one `MODEL_PRESET`; resume the same compatible study until that
  count is reached.
- No held-out test files were created.
- `val_metal_balanced_acc` is the selection metric on all completed runs.
- Diagnostics report every active metal-scheme class present in both train and validation splits.
- Rare-class recall protection passes: top candidates have available per-class
  recall, and no candidate is promoted if any metal class has zero recall in
  its validation artifact.
- Top candidates remain review-only until Stage 6 grouped-fold confirmation.

### Stage 5A - Serious Only-GVP HPO

Notebook configuration block:

```python
TASK = "metal"
RUN_MODE = "controlled_hpo_optuna"
RECOMMENDED_RUN_SET = "custom"
MODEL_PRESET = "Only-GVP"
RUN_BATCH_ID = "metal_only_gvp_optuna_200_capacity"
SUMMARY_BASENAME = ""  # auto from provenance

EPOCHS = 50
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_metal_balanced_acc"
RING_EDGE_MODE = "with_ring"
REQUIRE_RING_EDGES = False
PREPARE_MISSING_RING_EDGES = True
RING_FEATURES_DIR = ""
RING_EXE_PATH = "DeepMzyme_Data/ring-4.0/out/bin/ring"
ALLOW_MISSING_EXTERNAL_FEATURES = False
PREPARE_MISSING_EXTERNAL_FEATURES = True
EXTERNAL_FEATURES_ROOT_DIR = ""

OPTUNA_INTENSITY = "custom"
OPTUNA_TARGET_COMPLETE_TRIALS = 200
MAX_EPOCHS_PER_TRIAL = 50
OPTUNA_N_STARTUP_TRIALS = 40
OPTUNA_TPE_MULTIVARIATE = True
OPTUNA_TPE_GROUP = True
OPTUNA_TPE_CONSTANT_LIAR = True
OPTUNA_PARALLEL_WORKERS = 1
OPTUNA_PARALLEL_STARTUP_STAGGER_SECONDS = 10.0
OPTUNA_STOP_ON_PARALLEL_CUDA_OOM = True
OPTUNA_AUTO_CONFIGURE_BUDGET = False
OPTUNA_USE_PRUNING = True
OPTUNA_PRUNER_TYPE = "median"
OPTUNA_PRUNING_MIN_EPOCH = 25
OPTUNA_SEARCH_PRESET = "later_capacity"
OPTUNA_STUDY_NAME = "metal_only_gvp_optuna_200_capacity"
OPTUNA_STORAGE = "sqlite:////content/drive/MyDrive/DeepMzyme/optuna/metal_only_gvp_optuna_200_capacity.db"
OPTUNA_SPLIT_SEED = 42
OPTUNA_SAMPLER_SEED = None
OPTUNA_TIMEOUT_MINUTES = 0
OPTUNA_MULTIOBJECTIVE = False

OPTUNA_LEARNING_RATE_RANGE = "5e-6,3e-4"
LR_SCHEDULES_CSV = "fixed,cosine"
WEIGHT_DECAYS_CSV = "1e-5,1e-4,1e-3"
BATCH_SIZES_CSV = "8,16,32"
METAL_CLASS_WEIGHT_MODES_CSV = "none,inverse_frequency,inverse_sqrt_frequency,effective_number"
METAL_LOSS_FUNCTIONS_CSV = "cross_entropy,focal"
METAL_LABEL_SMOOTHING_VALUES_CSV = "0.0,0.03,0.05,0.1"
METAL_COLLAPSED_LOSS_WEIGHTS_CSV = "0.0"
BALANCE_METAL_SITE_SYMBOLS_CSV = "False,True"
METAL_FOCAL_GAMMA_VALUES_CSV = "1.5,2.0,2.5"
HEAD_MLP_DROPOUT_VALUES_CSV = "0.1,0.2,0.3"
POSITION_NOISE_STDS_CSV = "0.0,0.05,0.1"
OUTER_RESIDUE_DROPOUTS_CSV = "0.0,0.1,0.2"

HIDDEN_S_VALUES_CSV = "128"
HIDDEN_V_VALUES_CSV = "8,16"
EDGE_HIDDEN_VALUES_CSV = "64"
GVP_LAYERS_VALUES_CSV = "2,3"
HEAD_MLP_LAYERS_VALUES_CSV = "1,2"
EDGE_RADIUS_VALUES_CSV = "6.0,8.0,10.0"
CLASSIFIER_POOL_DISTANCE_CUTOFF_VALUES_CSV = "0.0"

RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = False
# Run Stage 6 later from the dedicated Stage 6 launch cell after HPO completes.
TOP_K_CONFIGS_FOR_SEED_REPEAT = "auto"
REPEAT_SEEDS = "42,123,2026,43,44"
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

Optional Stage 5A validation-only objective overlay:

```python
# Collapsed-4 auxiliary-loss probe; keep this out of initial baselines.
METAL_COLLAPSED_LOSS_WEIGHTS_CSV = "0.0,0.3,0.5"
METAL_LOSS_FUNCTIONS_CSV = "cross_entropy"

# Optional rare-class-protection Pareto search.
OPTUNA_MULTIOBJECTIVE = True
OPTUNA_SELECTION_METRIC = "val_metal_balanced_acc"
OPTUNA_USE_PRUNING = False
OPTUNA_PRUNER_TYPE = "none"
```

Use either part of this overlay only for an explicitly labeled validation-only
Stage 5A experiment. Do not enable it for Stage 2 baselines or Stage 7 final
held-out testing.

`batch_size=32` in this Stage 5A block is exploratory for Only-GVP only. Treat
CUDA OOM as a failed trial, not a prune, and inspect whether the larger batch
hurts rare-class recall before promoting any candidate.

Expected outputs/files:

- All shared Stage 5 Optuna and per-trial outputs.
- If `OPTUNA_MULTIOBJECTIVE = True`: `pareto_front.csv`,
  `pareto_candidates.csv`, and `pareto_candidates_ranked_for_review.csv`
- Two hundred complete per-trial validation-only run directories.

Exact configuration record:

- Use the shared Stage 5 exact-configuration record template.

### Decision gate after Stage 5A

Proceed to Stage 6 for Only-GVP candidates, or to Stage 5B/5C for family
comparison, only if:

- The shared Stage 5 decision-gate requirements pass for
  `MODEL_PRESET = "Only-GVP"` and at least 200 `COMPLETE` trials.
- For `OPTUNA_MULTIOBJECTIVE = True`, the study also writes complete
  `pareto_front.csv`, `pareto_candidates.csv`, and
  `pareto_candidates_ranked_for_review.csv`, and any convenience-ranked
  candidate is treated as review-only until Stage 6.
- Select top candidates for Stage 6 only if they do not degrade
  `val_metal_min_recall` by more than 0.05 versus the Stage 2A anchor, unless
  explicitly marked as exploratory.

If gate fails: do not advance to a more complex fusion family; revisit Stage 2A
and the Stage 5A search space.

### Stage 5B - Only-ESM HPO

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
SUMMARY_BASENAME = ""  # auto from provenance

EPOCHS = 50
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_metal_balanced_acc"
RING_EDGE_MODE = "with_ring"

ESM_EMBEDDINGS_DIR = ""  # set to your embeddings folder when available
ALLOW_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_ESM_EMBEDDINGS = True

OPTUNA_INTENSITY = "custom"
OPTUNA_TARGET_COMPLETE_TRIALS = 120
MAX_EPOCHS_PER_TRIAL = 50
OPTUNA_N_STARTUP_TRIALS = 30
OPTUNA_TPE_MULTIVARIATE = True
OPTUNA_TPE_GROUP = True
OPTUNA_TPE_CONSTANT_LIAR = True
OPTUNA_PARALLEL_WORKERS = 1
OPTUNA_PARALLEL_STARTUP_STAGGER_SECONDS = 10.0
OPTUNA_STOP_ON_PARALLEL_CUDA_OOM = True
OPTUNA_AUTO_CONFIGURE_BUDGET = False
OPTUNA_USE_PRUNING = False
OPTUNA_PRUNER_TYPE = "none"
OPTUNA_PRUNING_MIN_EPOCH = 20
OPTUNA_SEARCH_PRESET = "custom"
OPTUNA_STUDY_NAME = "metal_only_esm_optuna_120_controlled"
OPTUNA_STORAGE = "sqlite:////content/drive/MyDrive/DeepMzyme/optuna/metal_only_esm_optuna_120_controlled.db"
OPTUNA_SPLIT_SEED = 42
OPTUNA_SAMPLER_SEED = None
OPTUNA_TIMEOUT_MINUTES = 0

OPTUNA_LEARNING_RATE_RANGE = "5e-6,2e-4"
WEIGHT_DECAYS_CSV = "0.0,1e-6,1e-5,1e-4"
BATCH_SIZES_CSV = "8,16"
METAL_CLASS_WEIGHT_MODES_CSV = "none,inverse_frequency,inverse_sqrt_frequency,effective_number"
METAL_LOSS_FUNCTIONS_CSV = "cross_entropy"
METAL_LABEL_SMOOTHING_VALUES_CSV = "0.0,0.03,0.05,0.1"
BALANCE_METAL_SITE_SYMBOLS_CSV = "False,True"

HIDDEN_S_VALUES_CSV = "128,256"
HEAD_MLP_LAYERS_VALUES_CSV = "1,2,3"

RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = False
# Run Stage 6 later from the dedicated Stage 6 launch cell after HPO completes.
TOP_K_CONFIGS_FOR_SEED_REPEAT = "auto"
REPEAT_SEEDS = "42,123,2026,43,44"
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

Expected outputs/files:

- All shared Stage 5 Optuna and per-trial outputs.
- One hundred twenty complete per-trial validation-only run directories.

Exact configuration record:

- Use the shared Stage 5 exact-configuration record template.

### Decision gate after Stage 5B

Proceed to Stage 6 for Only-ESM candidates, or to Stage 5C, only if:

- The shared Stage 5 decision-gate requirements pass for
  `MODEL_PRESET = "Only-ESM"` and at least 120 `COMPLETE` trials.
- ESM coverage is valid and no run used missing ESM embeddings as a reportable
  fallback.

If gate fails: fix ESM coverage or narrow the Only-ESM search before comparing
ESM-informed model families.

### Stage 5C - GVP + Late Fusion HPO

Run this only after ESM coverage is valid and simpler baselines justify ESM
fusion.

Notebook configuration block:

```python
TASK = "metal"
RUN_MODE = "controlled_hpo_optuna"
RECOMMENDED_RUN_SET = "custom"
MODEL_PRESET = "GVP + late fusion"
RUN_BATCH_ID = "metal_late_fusion_optuna_200_controlled"
SUMMARY_BASENAME = ""  # auto from provenance

EPOCHS = 50
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_metal_balanced_acc"
RING_EDGE_MODE = "with_ring"

ESM_EMBEDDINGS_DIR = ""  # set to your embeddings folder when available
ALLOW_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_ESM_EMBEDDINGS = True

OPTUNA_INTENSITY = "custom"
OPTUNA_TARGET_COMPLETE_TRIALS = 200
MAX_EPOCHS_PER_TRIAL = 50
OPTUNA_N_STARTUP_TRIALS = 40
OPTUNA_TPE_MULTIVARIATE = True
OPTUNA_TPE_GROUP = True
OPTUNA_TPE_CONSTANT_LIAR = True
OPTUNA_PARALLEL_WORKERS = 1
OPTUNA_PARALLEL_STARTUP_STAGGER_SECONDS = 10.0
OPTUNA_STOP_ON_PARALLEL_CUDA_OOM = True
OPTUNA_AUTO_CONFIGURE_BUDGET = False
OPTUNA_USE_PRUNING = True
OPTUNA_PRUNER_TYPE = "median"
OPTUNA_PRUNING_MIN_EPOCH = 25
OPTUNA_SEARCH_PRESET = "custom"
OPTUNA_STUDY_NAME = "metal_late_fusion_optuna_200_controlled"
OPTUNA_STORAGE = "sqlite:////content/drive/MyDrive/DeepMzyme/optuna/metal_late_fusion_optuna_200_controlled.db"
OPTUNA_SPLIT_SEED = 42
OPTUNA_SAMPLER_SEED = None
OPTUNA_TIMEOUT_MINUTES = 0

OPTUNA_LEARNING_RATE_RANGE = "5e-6,2e-4"
LR_SCHEDULES_CSV = "fixed,cosine"
WEIGHT_DECAYS_CSV = "1e-5,1e-4,1e-3"
BATCH_SIZES_CSV = "8,16"
METAL_CLASS_WEIGHT_MODES_CSV = "inverse_frequency,inverse_sqrt_frequency,effective_number"
METAL_LOSS_FUNCTIONS_CSV = "cross_entropy"
METAL_LABEL_SMOOTHING_VALUES_CSV = "0.0,0.03,0.05,0.1"
BALANCE_METAL_SITE_SYMBOLS_CSV = "False,True"
HEAD_MLP_DROPOUT_VALUES_CSV = "0.1,0.2,0.3"
ESM_GRAPH_ENCODER_DROPOUT_VALUES_CSV = "0.0,0.1,0.2"
POSITION_NOISE_STDS_CSV = "0.0,0.05,0.1"
OUTER_RESIDUE_DROPOUTS_CSV = "0.0,0.1,0.2"

HIDDEN_S_VALUES_CSV = "128"
HIDDEN_V_VALUES_CSV = "8,16"
EDGE_HIDDEN_VALUES_CSV = "64"
GVP_LAYERS_VALUES_CSV = "2,3"
HEAD_MLP_LAYERS_VALUES_CSV = "1,2"
EDGE_RADIUS_VALUES_CSV = "6.0,8.0,10.0"
CLASSIFIER_POOL_DISTANCE_CUTOFF_VALUES_CSV = "0.0"
ESM_FUSION_DIM_VALUES_CSV = "64,128,256"

RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = False
# Run Stage 6 later from the dedicated Stage 6 launch cell after HPO completes.
TOP_K_CONFIGS_FOR_SEED_REPEAT = "auto"
REPEAT_SEEDS = "42,123,2026,43,44"
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

Expected outputs/files:

- All shared Stage 5 Optuna and per-trial outputs.
- Two hundred complete per-trial validation-only run directories.

Exact configuration record:

- Use the shared Stage 5 exact-configuration record template.

### Decision gate after Stage 5C

Proceed to Stage 6 only if:

- The shared Stage 5 decision-gate requirements pass for
  `MODEL_PRESET = "GVP + late fusion"` and at least 200 `COMPLETE` trials.
- Top candidates have finite selected validation metrics.

Proceed to Stage 5D/5E/5F only after Stage 6 confirms a late-fusion candidate
that beats the Stage 2A Only-GVP anchor by at least 0.01 mean
`val_metal_balanced_acc`, and the paired bootstrap 95% CI for the improvement
excludes zero.

If gate fails: no candidate from Stage 5C should trigger advanced fusion. Return
to Stage 2A/5A or revise the late-fusion search space.

### Stage 5D - GVP + Node-Level Late Fusion HPO

Run this after the late-fusion baseline has a stable validation anchor.

Notebook configuration block:

```python
TASK = "metal"
RUN_MODE = "controlled_hpo_optuna"
RECOMMENDED_RUN_SET = "custom"
MODEL_PRESET = "GVP + node-level late fusion"
RUN_BATCH_ID = "metal_node_late_fusion_optuna_200_controlled"
SUMMARY_BASENAME = ""  # auto from provenance

EPOCHS = 50
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_metal_balanced_acc"
RING_EDGE_MODE = "with_ring"

ESM_EMBEDDINGS_DIR = ""  # set to your embeddings folder when available
ALLOW_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_ESM_EMBEDDINGS = True

OPTUNA_INTENSITY = "custom"
OPTUNA_TARGET_COMPLETE_TRIALS = 200
MAX_EPOCHS_PER_TRIAL = 50
OPTUNA_N_STARTUP_TRIALS = 40
OPTUNA_TPE_MULTIVARIATE = True
OPTUNA_TPE_GROUP = True
OPTUNA_TPE_CONSTANT_LIAR = True
OPTUNA_PARALLEL_WORKERS = 1
OPTUNA_PARALLEL_STARTUP_STAGGER_SECONDS = 10.0
OPTUNA_STOP_ON_PARALLEL_CUDA_OOM = True
OPTUNA_AUTO_CONFIGURE_BUDGET = False
OPTUNA_USE_PRUNING = True
OPTUNA_PRUNER_TYPE = "median"
OPTUNA_PRUNING_MIN_EPOCH = 25
OPTUNA_SEARCH_PRESET = "custom"
OPTUNA_STUDY_NAME = "metal_node_late_fusion_optuna_200_controlled"
OPTUNA_STORAGE = "sqlite:////content/drive/MyDrive/DeepMzyme/optuna/metal_node_late_fusion_optuna_200_controlled.db"
OPTUNA_SPLIT_SEED = 42
OPTUNA_SAMPLER_SEED = None
OPTUNA_TIMEOUT_MINUTES = 0

OPTUNA_LEARNING_RATE_RANGE = "5e-6,2e-4"
LR_SCHEDULES_CSV = "fixed,cosine"
WEIGHT_DECAYS_CSV = "1e-5,1e-4,1e-3"
BATCH_SIZES_CSV = "8,16"
METAL_CLASS_WEIGHT_MODES_CSV = "inverse_frequency,inverse_sqrt_frequency,effective_number"
METAL_LOSS_FUNCTIONS_CSV = "cross_entropy"
METAL_LABEL_SMOOTHING_VALUES_CSV = "0.0,0.03,0.05,0.1"
BALANCE_METAL_SITE_SYMBOLS_CSV = "False,True"
HEAD_MLP_DROPOUT_VALUES_CSV = "0.1,0.2,0.3"
ESM_GRAPH_ENCODER_DROPOUT_VALUES_CSV = "0.0,0.1,0.2"
POSITION_NOISE_STDS_CSV = "0.0,0.05,0.1"
OUTER_RESIDUE_DROPOUTS_CSV = "0.0,0.1,0.2"

HIDDEN_S_VALUES_CSV = "128"
HIDDEN_V_VALUES_CSV = "8,16"
EDGE_HIDDEN_VALUES_CSV = "64"
GVP_LAYERS_VALUES_CSV = "2,3"
HEAD_MLP_LAYERS_VALUES_CSV = "1,2"
EDGE_RADIUS_VALUES_CSV = "6.0,8.0,10.0"
CLASSIFIER_POOL_DISTANCE_CUTOFF_VALUES_CSV = "0.0"
ESM_FUSION_DIM_VALUES_CSV = "64,128,256"

RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = False
# Run Stage 6 later from the dedicated Stage 6 launch cell after HPO completes.
TOP_K_CONFIGS_FOR_SEED_REPEAT = "auto"
REPEAT_SEEDS = "42,123,2026,43,44"
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

Expected outputs/files:

- All shared Stage 5 Optuna and per-trial outputs.
- Two hundred complete per-trial validation-only run directories.

Exact configuration record:

- Use the shared Stage 5 exact-configuration record template.

### Decision gate after Stage 5D

Proceed to Stage 6 only if:

- Stage 5C previously cleared the advanced-fusion ordering gate.
- The shared Stage 5 decision-gate requirements pass for
  `MODEL_PRESET = "GVP + node-level late fusion"` and at least 200 `COMPLETE`
  trials.

After Stage 6, promote a node-level late-fusion candidate only if it beats the
current best confirmed comparator by at least 0.005 mean
`val_metal_balanced_acc`, and the paired bootstrap 95% CI for the improvement
excludes zero.

If the Stage 5D launch gate or the later Stage 6 promotion gate fails, do not
advance to Stage 5E/5F; revisit Stage 2A or Stage 5C.

### Stage 5E - GVP + Hybrid Fusion HPO

Run this only after early/late ESM evidence justifies injecting ESM before graph
message passing and also using late fusion.

Notebook configuration block:

```python
TASK = "metal"
RUN_MODE = "controlled_hpo_optuna"
RECOMMENDED_RUN_SET = "custom"
MODEL_PRESET = "GVP + hybrid fusion"
RUN_BATCH_ID = "metal_hybrid_fusion_optuna_200_controlled"
SUMMARY_BASENAME = ""  # auto from provenance

EPOCHS = 50
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_metal_balanced_acc"
RING_EDGE_MODE = "with_ring"

ESM_EMBEDDINGS_DIR = ""  # set to your embeddings folder when available
ALLOW_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_ESM_EMBEDDINGS = True

OPTUNA_INTENSITY = "custom"
OPTUNA_TARGET_COMPLETE_TRIALS = 200
MAX_EPOCHS_PER_TRIAL = 50
OPTUNA_N_STARTUP_TRIALS = 40
OPTUNA_TPE_MULTIVARIATE = True
OPTUNA_TPE_GROUP = True
OPTUNA_TPE_CONSTANT_LIAR = True
OPTUNA_PARALLEL_WORKERS = 1
OPTUNA_PARALLEL_STARTUP_STAGGER_SECONDS = 10.0
OPTUNA_STOP_ON_PARALLEL_CUDA_OOM = True
OPTUNA_AUTO_CONFIGURE_BUDGET = False
OPTUNA_USE_PRUNING = True
OPTUNA_PRUNER_TYPE = "median"
OPTUNA_PRUNING_MIN_EPOCH = 25
OPTUNA_SEARCH_PRESET = "custom"
OPTUNA_STUDY_NAME = "metal_hybrid_fusion_optuna_200_controlled"
OPTUNA_STORAGE = "sqlite:////content/drive/MyDrive/DeepMzyme/optuna/metal_hybrid_fusion_optuna_200_controlled.db"
OPTUNA_SPLIT_SEED = 42
OPTUNA_SAMPLER_SEED = None
OPTUNA_TIMEOUT_MINUTES = 0

OPTUNA_LEARNING_RATE_RANGE = "5e-6,1.5e-4"
LR_SCHEDULES_CSV = "fixed,cosine"
WEIGHT_DECAYS_CSV = "1e-5,1e-4,1e-3"
BATCH_SIZES_CSV = "8,16"
METAL_CLASS_WEIGHT_MODES_CSV = "inverse_frequency,inverse_sqrt_frequency,effective_number"
METAL_LOSS_FUNCTIONS_CSV = "cross_entropy"
METAL_LABEL_SMOOTHING_VALUES_CSV = "0.0,0.03,0.05,0.1"
BALANCE_METAL_SITE_SYMBOLS_CSV = "False,True"
HEAD_MLP_DROPOUT_VALUES_CSV = "0.1,0.2,0.3"
ESM_GRAPH_ENCODER_DROPOUT_VALUES_CSV = "0.0,0.1,0.2"
POSITION_NOISE_STDS_CSV = "0.0,0.05,0.1"
OUTER_RESIDUE_DROPOUTS_CSV = "0.0,0.1,0.2"

HIDDEN_S_VALUES_CSV = "128"
HIDDEN_V_VALUES_CSV = "8,16"
EDGE_HIDDEN_VALUES_CSV = "64"
GVP_LAYERS_VALUES_CSV = "2,3"
HEAD_MLP_LAYERS_VALUES_CSV = "1,2"
EDGE_RADIUS_VALUES_CSV = "6.0,8.0,10.0"
CLASSIFIER_POOL_DISTANCE_CUTOFF_VALUES_CSV = "0.0"
ESM_FUSION_DIM_VALUES_CSV = "64,128,256"
EARLY_ESM_DIM_VALUES_CSV = "16,32,64"
EARLY_ESM_DROPOUT_VALUES_CSV = "0.0,0.1,0.2"

RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = False
# Run Stage 6 later from the dedicated Stage 6 launch cell after HPO completes.
TOP_K_CONFIGS_FOR_SEED_REPEAT = "auto"
REPEAT_SEEDS = "42,123,2026,43,44"
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

Expected outputs/files:

- All shared Stage 5 Optuna and per-trial outputs.
- Two hundred complete per-trial validation-only run directories.

Exact configuration record:

- Use the shared Stage 5 exact-configuration record template.

### Decision gate after Stage 5E

Proceed to Stage 6 only if:

- Stage 5C previously cleared the advanced-fusion ordering gate.
- The shared Stage 5 decision-gate requirements pass for
  `MODEL_PRESET = "GVP + hybrid fusion"` and at least 200 `COMPLETE` trials.

After Stage 6, promote a hybrid-fusion candidate only if it beats the current
best confirmed comparator by at least 0.005 mean `val_metal_balanced_acc`, and
the paired bootstrap 95% CI for the improvement excludes zero.

If the Stage 5E launch gate or the later Stage 6 promotion gate fails, stop
advanced fusion escalation and revisit the simpler late-fusion or Only-GVP
anchors before cross-attention.

### Stage 5F - GVP + Cross-Attention HPO

Run this last among fusion models. Keep attention narrow at first because it has
more overfitting degrees of freedom.

Notebook configuration block:

```python
TASK = "metal"
RUN_MODE = "controlled_hpo_optuna"
RECOMMENDED_RUN_SET = "custom"
MODEL_PRESET = "GVP + cross-modal attention"
RUN_BATCH_ID = "metal_cross_attention_optuna_120_controlled"
SUMMARY_BASENAME = ""  # auto from provenance

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
OPTUNA_TARGET_COMPLETE_TRIALS = 120
MAX_EPOCHS_PER_TRIAL = 50
OPTUNA_N_STARTUP_TRIALS = 30
OPTUNA_TPE_MULTIVARIATE = True
OPTUNA_TPE_GROUP = True
OPTUNA_TPE_CONSTANT_LIAR = True
OPTUNA_PARALLEL_WORKERS = 1
OPTUNA_PARALLEL_STARTUP_STAGGER_SECONDS = 10.0
OPTUNA_STOP_ON_PARALLEL_CUDA_OOM = True
OPTUNA_AUTO_CONFIGURE_BUDGET = False
OPTUNA_USE_PRUNING = True
OPTUNA_PRUNER_TYPE = "median"
OPTUNA_PRUNING_MIN_EPOCH = 25
OPTUNA_SEARCH_PRESET = "custom"
OPTUNA_STUDY_NAME = "metal_cross_attention_optuna_120_controlled"
OPTUNA_STORAGE = "sqlite:////content/drive/MyDrive/DeepMzyme/optuna/metal_cross_attention_optuna_120_controlled.db"
OPTUNA_SPLIT_SEED = 42
OPTUNA_SAMPLER_SEED = None
OPTUNA_TIMEOUT_MINUTES = 0

OPTUNA_LEARNING_RATE_RANGE = "5e-6,1e-4"
WEIGHT_DECAYS_CSV = "1e-5,1e-4,1e-3"
BATCH_SIZES_CSV = "8,16"
METAL_CLASS_WEIGHT_MODES_CSV = "inverse_frequency,inverse_sqrt_frequency,effective_number"
METAL_LOSS_FUNCTIONS_CSV = "cross_entropy"
METAL_LABEL_SMOOTHING_VALUES_CSV = "0.0,0.03,0.05,0.1"
BALANCE_METAL_SITE_SYMBOLS_CSV = "False"
HEAD_MLP_DROPOUT_VALUES_CSV = "0.1,0.2,0.3"
POSITION_NOISE_STDS_CSV = "0.0,0.05,0.1"
OUTER_RESIDUE_DROPOUTS_CSV = "0.0,0.1,0.2"

HIDDEN_S_VALUES_CSV = "128"
HIDDEN_V_VALUES_CSV = "8,16"
EDGE_HIDDEN_VALUES_CSV = "64"
GVP_LAYERS_VALUES_CSV = "2,3"
HEAD_MLP_LAYERS_VALUES_CSV = "1,2"
EDGE_RADIUS_VALUES_CSV = "6.0,8.0,10.0"
CLASSIFIER_POOL_DISTANCE_CUTOFF_VALUES_CSV = "0.0"
CROSS_ATTENTION_LAYERS_CSV = "1"
CROSS_ATTENTION_HEADS_CSV = "2,4"
CROSS_ATTENTION_DROPOUT_VALUES_CSV = "0.0,0.1,0.2"

RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = False
# Run Stage 6 later from the dedicated Stage 6 launch cell after HPO completes.
TOP_K_CONFIGS_FOR_SEED_REPEAT = "auto"
REPEAT_SEEDS = "42,123,2026,43,44"
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

Expected outputs/files:

- All shared Stage 5 Optuna and per-trial outputs.
- One hundred twenty complete per-trial validation-only run directories.

Exact configuration record:

- Use the shared Stage 5 exact-configuration record template.

### Decision gate after Stage 5F

Proceed to Stage 6 only if:

- Stage 5C previously cleared the advanced-fusion ordering gate.
- The shared Stage 5 decision-gate requirements pass for
  `MODEL_PRESET = "GVP + cross-modal attention"` and at least 120 `COMPLETE`
  trials.
- Attention candidates justify their extra complexity against the Stage 6
  late-fusion candidate.

After Stage 6, promote a cross-attention candidate only if it beats the current
best confirmed comparator by at least 0.005 mean `val_metal_balanced_acc`, and
the paired bootstrap 95% CI for the improvement excludes zero.

If the Stage 5F launch gate or the later Stage 6 promotion gate fails, do not
broaden cross-attention. Return to the best validated simpler fusion family.

### Stage 5G - RING/Radius-Only Ablation

Use only when you deliberately want to compare against the older radius-only
graph setting. This does not make Optuna sample RING on/off; it fixes the base
run to radius-only graph construction. This standalone block mirrors Stage 2A's
Only-GVP validation anchor while changing only the graph-edge mode and labels
the output as a radius-only ablation.

```python
TASK = "metal"
RUN_MODE = "manual_configurations"
RECOMMENDED_RUN_SET = "only_gvp_broad_comparison"
MODEL_PRESET = "Only-GVP"
RUN_BATCH_ID = "metal_only_gvp_radius_only_ablation"
SUMMARY_BASENAME = ""  # auto from provenance

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

RING_EDGE_MODE = "without_ring"
REQUIRE_RING_EDGES = False
PREPARE_MISSING_RING_EDGES = True
RING_FEATURES_DIR = ""

ESM_EMBEDDINGS_DIR = ""
ALLOW_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_ESM_EMBEDDINGS = False
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_metal_balanced_acc"
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

In the dedicated **Main planned training launch switch** cell, then run
**Optional training execution**:

```python
LAUNCH_PLANNED_MAIN_TRAINING_RUNS = True
```

Expected outputs/files:

- `<RUNS_DIR>/<SUMMARY_BASENAME>_planned_runs.csv`
- `<RUNS_DIR>/<SUMMARY_BASENAME>_planned_run_dictionary.json`
- Nine completed validation-only radius-only run directories under
  `<RUNS_DIR>/`
- Each completed run directory contains `run_config.json`,
  `run_metadata.json`, `split_diagnostics.json`, `dataset_summary.json`, and
  `prepare_status.json`
- `<RUNS_DIR>/<SUMMARY_BASENAME>.csv`
- `<RUNS_DIR>/<SUMMARY_BASENAME>_completed_only.csv`
- `<RUNS_DIR>/<SUMMARY_BASENAME>.png`, when plotting succeeds
- No `test_report.json`

Exact configuration record:

- Per run: `<run_dir>/run_config.json` and `<run_dir>/run_metadata.json`.
- Batch plan: planned-run CSV/dictionary.
- `active_run_config.json` / `active_run_config.md` are generated from the live
  notebook configuration before launch.

Success criteria:

- Planning table shows `RING_EDGE_MODE = "without_ring"` / radius-only graph
  mode for every planned run.
- No test-report files are created by the ablation runs.
- The ablation clarifies validation behavior without relying on one lucky seed.

### Decision gate after Stage 5G

Proceed to Stage 6 only if:

- The ablation was explicitly labeled radius-only and compared against the
  matching RING-enabled family.
- All 9 planned validation-only ablation runs complete.
- The expected planned files, summary files, and run-level JSON files exist.
- No held-out test files were created.
- `val_metal_balanced_acc` is the selection metric on all completed runs.
- Diagnostics report every active metal-scheme class present in both train and validation splits.
- Rare-class recall protection passes: per-class recall is available for each
  completed run, and the ablation is not promoted if any metal class has zero
  seed-mean recall.

If gate fails: do not use the ablation as model-selection evidence. Choose the
top 2-3 valid candidates for grouped-fold confirmation and do not finalize from
a raw single-batch ranking alone.

## Stage 6 - Top-K Seed/Split Confirmation

Purpose: confirm whether top HPO candidates are stable across validation data
partitions and model-initialization seeds. The standard fold-plus-seed Stage 6
design uses `TOP_CONFIG_REEVALUATION_MODE = "group_kfold_seed_repeat"`:
grouped 5-fold validation by `pdbid` crossed with the configured
`REPEAT_SEEDS`; every compared candidate uses the same fold definitions and
seed list. Candidate ranking uses one primary score, the mean selected
validation metric over all completed fold x seed runs. Paired comparisons
remain conservative by averaging common seeds within each fold and bootstrapping
fold-level differences.

When to use it: after a medium or large Optuna search has produced top
candidates, and before any candidate is treated as final-selection evidence.

Expected scale/runtime: serious run, long or overnight. Training count is
roughly `TOP_K_CONFIGS_FOR_SEED_AND_CROSS_FOLD_REPEAT x SEED_REPEAT_N_FOLDS x
len(REPEAT_SEEDS)` in `group_kfold_seed_repeat` mode, or top-K x folds in
plain `group_kfold` mode; runtime then scales with that count and `EPOCHS`.
Optional Stage 6 parallelism is candidate-scoped: for each ranked top-K
candidate, the notebook can run up to
`STAGE6_PARALLEL_CROSS_VALIDATION_PROCESSES` fold/seed validation subprocesses
at once, waits for that candidate's units to finish, and then moves to the
next ranked candidate. The effective worker count is capped by
`SEED_REPEAT_N_FOLDS` and by the number of remaining units for that candidate.
Keep the default `1` for serial/reproducible behavior; set it to
`SEED_REPEAT_N_FOLDS` only after confirming GPU memory headroom for the active
model.

Preferred same-runtime configuration: run the exact Stage 5 HPO block first
with `RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = False`. After HPO finishes, use
the dedicated **Stage 6 controls and existing Optuna/HPO reuse** panel plus the
**Launch Stage 6 top-K grouped-fold confirmation** cell to import the previous
validation-only HPO candidates, write the top-K commands, and optionally launch
grouped-fold confirmation. Keep the Stage 5 block's `MODEL_PRESET`,
`OPTUNA_STUDY_NAME`, `OPTUNA_STORAGE`, search space, and persistent-storage
settings unchanged.

Existing-HPO standalone configuration: when the HPO directory already exists
from a previous Colab/runtime, keep `LAUNCH_PLANNED_MAIN_TRAINING_RUNS = False`,
set the old HPO source in the Stage 6 controls, and use the Stage 6 launch cell.
You can press **Run all** so the required setup/clone cells execute before Stage
6; ordinary main training/HPO remains skipped. This imports saved run metadata
from the old directory and launches only the new Stage 6 fold/seed runs. When
resuming an interrupted Stage 6, keep the values identical and keep
`SKIP_EXISTING_RUNS = True` so completed folds are reused while missing folds
continue. During an incomplete run, Stage 6 may write a provisional
`stage6_partial/` report. That report is safe to inspect and safe for Stage 6B
preview mode, but it is not promotion evidence and does not replace the
canonical Stage 6 files.

Set these values in the dedicated Stage 6 controls panel:

```python
RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = True
TOP_CONFIG_REEVALUATION_MODE = "group_kfold_seed_repeat"
TOP_K_CONFIGS_FOR_SEED_AND_CROSS_FOLD_REPEAT = "auto"
REPEAT_SEEDS = "42"
SEED_REPEAT_N_FOLDS = 5
STAGE6_PARALLEL_CROSS_VALIDATION_PROCESSES = 1
SEED_REPEAT_SPLIT_SEED = 42
STAGE6_RAW_IMPROVEMENT_THRESHOLD = 0.0
ALLOW_SEED_REPEAT_MODEL_PRESET_MISMATCH = False
SKIP_EXISTING_RUNS = True
WRITE_STAGE6_PARTIAL_PROGRESS_REPORTS = False
USE_EXISTING_OPTUNA_TRIALS_FOR_STAGE6 = False
EXISTING_OPTUNA_TRIALS_BASE_RUNS_DIR = "/content/drive/MyDrive/DeepMzyme/notebook_outputs/runs"
EXISTING_OPTUNA_TRIALS_RUN_BATCH_ID = ""
STAGE6_OUTPUT_RUNS_DIR = ""      # blank = sibling <OLD_HPO_RUN_BATCH_ID>_stage6 directory
STAGE6_OVERWRITE_OUTPUT = False  # False = continue compatible output or create; True = replace incompatible output
STAGE6_EPOCHS = 50
STAGE6_DEVICE = "auto"
STAGE6_SELECTION_METRIC = "val_metal_balanced_acc"
```

For same-runtime HPO, leave `USE_EXISTING_OPTUNA_TRIALS_FOR_STAGE6 = False` so
the Stage 6 launch cell imports from the current notebook run directory. For a
previous HPO directory, set `USE_EXISTING_OPTUNA_TRIALS_FOR_STAGE6 = True` plus
`EXISTING_OPTUNA_TRIALS_BASE_RUNS_DIR / EXISTING_OPTUNA_TRIALS_RUN_BATCH_ID`.
If reuse is enabled but the old-HPO source is blank, the controls cell should
warn without crashing; the Stage 6 launch still refuses to import candidates
until a concrete validation HPO source path is provided.

For standalone existing-HPO Stage 6, leave `STAGE6_OUTPUT_RUNS_DIR` blank to
write a sibling output directory named from the old HPO directory. If that
output already contains a matching `stage6_manifest.json`, the notebook can
continue it and reuse completed fold/seed units when `SKIP_EXISTING_RUNS=True`.
If the output exists but the manifest does not match the requested source,
top-K, folds, seeds, metric, epochs, or selected candidates, the launch stops
unless `STAGE6_OVERWRITE_OUTPUT=True`.

In the **Launch Stage 6 top-K grouped-fold confirmation** cell, first preview
the imported candidates and generated commands:

```python
LAUNCH_STAGE6_TOP_K_CONFIRMATION = False
```

Preview mode does not launch missing folds. If compatible Stage 6 fold/seed
run directories already exist and `WRITE_STAGE6_PARTIAL_PROGRESS_REPORTS=True`,
it scans those completed runs and refreshes the `stage6_partial/` progress
report. The current default is `False`: Stage 6 writes canonical files only
when the declared Stage 6 grid is complete, and Stage 6B can reconstruct a
preview-only partial table from existing CV run folders when complete Stage 6
files are absent. After confirming the import report and generated top-K
commands are correct, launch or resume the missing folds with:

```python
LAUNCH_STAGE6_TOP_K_CONFIRMATION = True
```

Notes:

- `TOP_CONFIG_REEVALUATION_MODE = "group_kfold_seed_repeat"` is the explicit
  reportable fold-plus-seed Stage 6 mode.
- `TOP_CONFIG_REEVALUATION_MODE = "group_kfold"` is grouped-fold confirmation
  with only the first `REPEAT_SEEDS` value. Use it for one-seed confirmation or
  backward-compatible reruns where seed crossing was not intended.
- `TOP_K_CONFIGS_FOR_SEED_AND_CROSS_FOLD_REPEAT = "auto"` resolves from
  completed Optuna trials: fewer than 50 completed trials repeats up to 5
  candidates, fewer than 150 repeats up to 10, and 150 or more repeats up to
  20. A predeclared integer is allowed, including 20, but it should be chosen
  before Stage 6 launches. The older notebook variable
  `TOP_K_CONFIGS_FOR_SEED_REPEAT` remains a backward-compatible alias.
- `REPEAT_SEEDS = "42"` is the comma-separated model-initialization seed list.
  In `group_kfold_seed_repeat`, every seed is crossed with every grouped fold.
  In `group_kfold`, only the first listed seed is used. Add more seeds only
  when the resulting `top_k x folds x seeds` training count is practical and
  predeclared.
- `SEED_REPEAT_SPLIT_SEED = 42` fixes the grouped fold definitions. Keep it
  identical for every candidate in the same comparison.
- `STAGE6_PARALLEL_CROSS_VALIDATION_PROCESSES = 1` preserves the original
  serial Stage 6 launch. Values above `1` run fold/seed units in parallel
  within the current ranked candidate only; all units for top-1 finish before
  top-2 starts. On a single G4/T4-class GPU, set this no higher than
  `SEED_REPEAT_N_FOLDS`; the launcher caps the effective worker count at the
  fold count and only a short launch should confirm CUDA memory headroom.
- The legacy `TOP_CONFIG_REEVALUATION_MODE = "seed_repeat"` mode remains
  available for exploratory checks only. It measures combined initialization
  and split variance, not isolated initialization variance.

If the Optuna study is already complete, use the Stage 6 launch cell in preview
mode first and inspect:

```text
<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/top_reevaluation_commands.txt
```

Those commands intentionally omit held-out test evaluation. Launch only the
top-K commands you predeclare, with the same fold definitions for every
compared candidate, and keep the results as validation-only evidence.

Expected outputs/files:

- `<RUNS_DIR>/stage6_manifest.json` for standalone existing-HPO Stage 6
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/stage6_manifest.json` for standalone existing-HPO Stage 6
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/stage6_existing_trials_import_report.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/stage6_existing_trials_import_report.json`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/top_reevaluation_commands.txt`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/seed_repeat_results.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/seed_repeat_summary.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/seed_repeat_summary.json`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/seed_repeat_pairwise_bootstrap.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/seed_repeat_pairwise_bootstrap.json`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/stage6_ranked_candidates.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/stage6_selected_final_candidate.json`
- One validation-only run directory per top-K/fold/seed unit
- Per-fold `run_config.json`, `run_metadata.json`, and
  `split_diagnostics.json`
- No `test_report.json`

When `WRITE_STAGE6_PARTIAL_PROGRESS_REPORTS=True` and Stage 6 is still
incomplete, provisional progress files are written under:

- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/stage6_partial/stage6_partial_manifest.json`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/stage6_partial/stage6_partial_results.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/stage6_partial/stage6_partial_ranked_candidates.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/stage6_partial/stage6_partial_pairwise_bootstrap.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/stage6_partial/stage6_partial_report.md`

These files are explicitly provisional. They include completion counts and are
ignored by Stage 6 resume planning. With the default
`WRITE_STAGE6_PARTIAL_PROGRESS_REPORTS=False`, these provisional files are not
created by Stage 6; use Stage 6B's reconstruction option if a preview table is
needed before all declared seeds complete. Canonical
`stage6_ranked_candidates.csv` and `stage6_selected_final_candidate.json` are
written only after the planned candidate/fold/seed units are complete.

Stage 6 result rows include:

- candidate identifier
- source Optuna study, top rank, trial number, and source run directory
- model seed
- split seed
- fold index / validation unit
- fold unit and model seed when grouped-fold confirmation uses multiple seeds
- validation balanced accuracy
- validation minimum per-class recall
- per-class recall when available
- collapsed-4 balanced accuracy when available
- run directory
- selected checkpoint path

Exact configuration record:

- Grouped-fold summary: CSV/JSON files listed above.
- Stage 6 ranking and frozen final-candidate selection:
  `stage6_ranked_candidates.csv` and
  `stage6_selected_final_candidate.json`.
- Per repeated run: `<run_dir>/run_config.json` and
  `<run_dir>/run_metadata.json`.
- `active_run_config.json` / `active_run_config.md` are generated from the live
  notebook configuration before launch.

Success criteria:

- All predeclared top-K/fold/active-seed runs complete.
- The number of completed validation-only runs equals
  the resolved `TOP_K_CONFIGS_FOR_SEED_AND_CROSS_FOLD_REPEAT` value times
  `SEED_REPEAT_N_FOLDS` times `len(REPEAT_SEEDS)` for
  `group_kfold_seed_repeat`, or times one active model seed for `group_kfold`.
- Every candidate has the same fold definitions and the same active model-seed
  list.
- No held-out test files were created.
- Candidate ranking uses validation/CV metrics only: highest mean
  `val_metal_balanced_acc` over all completed fold x seed runs, then higher
  mean `val_metal_min_recall`, then lower fold-to-fold standard deviation of
  seed-averaged `val_metal_balanced_acc`, then a simpler/smaller model if still
  tied.
- `stage6_selected_final_candidate.json` records the selected configuration
  ID, selected source run directories, model preset, selected hyperparameters,
  and ranking metrics. It is the selection evidence for the required final
  training/refit run; a raw Stage 6 fold checkpoint is not the preferred primary
  final-test source.
- Pairwise comparisons use paired bootstrap over fold-level differences with
  10,000 resamples. When multiple seeds are configured, the notebook averages
  common seeds within each fold before bootstrapping. Candidate A beats
  candidate B only if mean A-B is positive, the 95% CI excludes zero on the
  positive side, and the raw improvement meets the applicable threshold.
- Diagnostics do not show leakage, missing active metal-scheme validation classes, or invalid
  feature coverage.

### Decision gate after Stage 6

Proceed to Stage 6B, and only then Stage 7, if:

- The Stage 6 success criteria are met.
- `stage6_existing_trials_import_report.csv`,
  `stage6_existing_trials_import_report.json`, and
  `top_reevaluation_commands.txt` exist and point to validation-only HPO
  candidates.
- `seed_repeat_results.csv`, `seed_repeat_summary.csv`,
  `seed_repeat_summary.json`, `seed_repeat_pairwise_bootstrap.csv`,
  `seed_repeat_pairwise_bootstrap.json`, `stage6_ranked_candidates.csv`, and
  `stage6_selected_final_candidate.json` exist.
- The selected candidate has all planned fold/active-seed units completed.
- No held-out test files were created anywhere in the validation chain.
- `val_metal_balanced_acc` is the selection metric on all completed runs.
- Diagnostics report every active metal-scheme class present in both train and validation splits.
- Rare-class recall protection passes: the selected candidate has available
  mean per-class recall across folds and acceptable `val_metal_min_recall`; no
  metal class has zero mean recall.
- Any claimed improvement over a comparator is supported by the paired
  bootstrap 95% CI and the relevant raw-improvement threshold.
- One final configuration is selected using validation/CV evidence only.
- The exact Stage 6 source run directories/checkpoints and selected
  configuration are recorded before Stage 6B launch.
- Stage 6B promotion/refit policy is declared before running it: paired-CI
  thresholds, rare-recall thresholds, tie-breakers, fixed final-refit seed,
  epoch/checkpoint rule, and output folder. None of these may be changed after
  held-out test metrics are seen.

If gate fails: report the top candidates, paired bootstrap rows,
`val_metal_min_recall`, per-class recall, split seed, fold indices, and epoch
budget. Do not launch held-out test evaluation.

## Stage 6B - Promotion Gates And Final Full-Train Refit

Purpose: convert Stage 6 validation/CV evidence into one frozen final model.
Stage 6B ranks candidates by mean `val_metal_balanced_acc`, applies the
predeclared paired-CI, rare-class recall, and tie-breaker policy, then
optionally trains the selected configuration once on the full non-test training
set. Stage 6B does not open the held-out test set.

When to use it: only after Stage 6 has completed and the Stage 6 decision gate
above passes.

Set these values in the dedicated **Stage 6B - promotion gates and final
full-train refit** cell:

```python
RUN_STAGE6B_FINAL_SELECTION = True
LAUNCH_STAGE6B_FINAL_REFIT = False

# Stage 6B uses STAGE6_OUTPUT_RUNS_DIR from the Stage 6 controls.
# Blank = current RUNS_DIR in same-runtime mode, or auto sibling
# <OLD_HPO_RUN_BATCH_ID>_stage6 in standalone existing-HPO mode.
# Backward-compatible explicit optuna/<study> override:
STAGE6B_STAGE6_OPTUNA_DIR = ""

STAGE6B_ALLOW_PARTIAL_STAGE6_PREVIEW = False
STAGE6B_PARTIAL_STAGE6_REPORT_DIR = ""  # blank = resolved Stage 6 optuna dir/stage6_partial

STAGE6B_RECONSTRUCT_PARTIAL_FROM_COMPLETE_SEED_CV = False
STAGE6B_RECONSTRUCT_PARTIAL_SOURCE_RUNS_DIR = ""  # blank = infer Stage 6 output root from STAGE6_OUTPUT_RUNS_DIR/current source

STAGE6B_RANK_BY_METRIC = "mean_val_metal_balanced_acc"
STAGE6B_TIE_EPSILON = 0.002
STAGE6B_TIE_BREAKERS = "mean_val_metal_min_recall_desc,min_validation_metric_desc,std_val_metal_balanced_acc_asc,model_complexity_proxy_asc"

STAGE6B_REQUIRE_PAIRED_CI_IMPROVEMENT = True
STAGE6B_MIN_RAW_IMPROVEMENT = 0.0
STAGE6B_MIN_CI_LOWER_BOUND = 0.0
STAGE6B_ALLOW_SINGLE_CANDIDATE_WITHOUT_CI = False
STAGE6B_ALLOW_TIE_BREAK_WITHOUT_POSITIVE_CI = False

STAGE6B_BLOCK_ON_MISSING_RARE_RECALL = True
STAGE6B_MIN_MEAN_MIN_RECALL = 0.0
STAGE6B_MIN_WORST_MIN_RECALL = 0.0
STAGE6B_MIN_PER_CLASS_MEAN_RECALL = 0.0
STAGE6B_MAX_MEAN_MIN_RECALL_DROP_VS_COMPARATOR = 0.03

STAGE6B_FINAL_REFIT_EPOCHS = 50
STAGE6B_FINAL_REFIT_SEED = 42  # fixed protocol seed; not a Colab UI input
STAGE6B_FINAL_REFIT_DEVICE = "auto"
STAGE6B_FINAL_REFIT_RUN_NAME_PREFIX = "stage6b_final_refit"
STAGE6B_REUSE_EXISTING_REFIT_RUN = True
```

First run Stage 6B in preview mode with `LAUNCH_STAGE6B_FINAL_REFIT = False`.
Inspect the ranked table, `stage6b_decision.json`, and
`stage6b_final_refit_command.txt`. If the candidate is approved and the command
matches the selected configuration, rerun the same cell with:

```python
LAUNCH_STAGE6B_FINAL_REFIT = True
```

If Stage 6 is incomplete but you want to inspect the provisional ordering, set
`STAGE6B_ALLOW_PARTIAL_STAGE6_PREVIEW = True` and keep
`LAUNCH_STAGE6B_FINAL_REFIT = False`. This reads `stage6_partial/` and writes
only preview files:

- `stage6b_partial_preview_ranked_candidates.csv`
- `stage6b_partial_preview_decision.json`
- `stage6b_partial_preview_final_refit_command_BLOCKED.txt`

Partial preview never writes `stage6b_decision.json`,
`stage6b_final_refit_command.txt`, or
`stage6b_selected_final_refit_candidate.json`, and it cannot launch the final
refit. Resume and complete Stage 6 before reportable Stage 6B.

If an old Stage 6 CV run directory predates canonical/partial table writing,
Stage 6B can reconstruct the tables from completed Stage 6 CV run folders. Set
`STAGE6B_RECONSTRUCT_PARTIAL_FROM_COMPLETE_SEED_CV = True` while keeping
`LAUNCH_STAGE6B_FINAL_REFIT = False` for a first preview. If canonical completed
Stage 6 files are missing, Stage 6B scans
`STAGE6B_RECONSTRUCT_PARTIAL_SOURCE_RUNS_DIR`, or the Stage 6 output root from
`STAGE6_OUTPUT_RUNS_DIR` when that source is blank. For a standalone
existing-HPO Stage 6 run with blank `STAGE6_OUTPUT_RUNS_DIR`, Stage 6B infers
the auto sibling `<OLD_HPO_RUN_BATCH_ID>_stage6` output root.

When a Stage 6 manifest exists and reconstruction proves every manifest-declared
candidate x fold x active-seed unit is present, Stage 6B writes the canonical
Stage 6 files (`seed_repeat_results.csv`, `stage6_ranked_candidates.csv`,
`stage6_selected_final_candidate.json`, and related JSON/CSV summaries), then
normal Stage 6B promotion/refit may proceed. If the manifest is missing or any
declared unit is missing, reconstruction writes only `stage6_partial/` preview
files. Partial preview includes only candidate/seed blocks that have every
detected grouped-CV fold; candidate/seed blocks missing any fold are dropped.
Partial reconstruction is not promotion evidence and cannot launch the final
refit.

Promotion policy:

- Primary ranking metric is `mean_val_metal_balanced_acc`.
- Paired-CI promotion is required by default. The selected candidate must beat
  the comparator with paired mean improvement at least
  `STAGE6B_MIN_RAW_IMPROVEMENT`, and the paired 95% CI lower bound must be
  greater than `STAGE6B_MIN_CI_LOWER_BOUND`.
- Rare-class recall protection blocks promotion when required recall values are
  missing, below configured absolute minima, or when mean
  `val_metal_min_recall` drops by more than
  `STAGE6B_MAX_MEAN_MIN_RECALL_DROP_VS_COMPARATOR` versus the comparator.
- Tie-breakers are predeclared by `STAGE6B_TIE_BREAKERS`; the default order is
  higher mean minimum recall, higher worst-fold validation metric, lower
  fold-to-fold standard deviation, then simpler model.
- `STAGE6B_ALLOW_TIE_BREAK_WITHOUT_POSITIVE_CI` should stay `False` for the
  primary report. Turning it on is a labeled policy choice for statistically
  tied candidates; it does not claim a CI-supported improvement.

Final refit policy:

- The refit uses `VAL_FRACTION = 0.0`, no k-fold split, and
  `selection_metric = "train_loss"` because it trains on the full non-test
  training set.
- The checkpoint rule is fixed before launch: use the best train-loss
  checkpoint from the final refit. The held-out test is not evaluated during
  Stage 6B.
- For reportable runs, the final refit seed is not a policy choice and is not
  exposed as a Colab input. The notebook uses the fixed, predeclared integer
  `STAGE6B_FINAL_REFIT_SEED = 42`, so the final model seed is known before
  refit and cannot be chosen after looking at held-out metrics.

Expected outputs/files:

- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/stage6b_ranked_candidates.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/stage6b_decision.json`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/stage6b_final_refit_command.txt`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/stage6b_selected_final_refit_candidate.json`
  after a completed or reused final refit
- `<stage6b_final_refit_run_dir>/active_run_config.json`
- `<stage6b_final_refit_run_dir>/active_run_config.md`
- `<stage6b_final_refit_run_dir>/run_config.json`
- `<stage6b_final_refit_run_dir>/run_metadata.json`
- `<stage6b_final_refit_run_dir>/best_model_checkpoint.pt`
- No `test_report.json`

Decision gate after Stage 6B:

- `stage6b_decision.json` status is `selected_for_final_refit`.
- Stage 6B used canonical completed Stage 6 artifacts, not `stage6_partial/`
  preview artifacts.
- `stage6b_selected_final_refit_candidate.json` exists and records
  `protocol_stage = "Stage 6B"`, `selected_before_held_out_test_evaluation =
  True`, `held_out_test_metrics_used = False`, and
  `final_training_refit.status` as `completed` or `existing`.
- The selected final-refit run directory exists and contains
  `run_config.json`, `run_metadata.json`, and a checkpoint.
- The final-refit run used the selected Stage 6 configuration, full non-test
  training set, fixed final-refit seed, fixed epoch budget, and no held-out test
  evaluation.
- Stage 7 points to `stage6b_selected_final_refit_candidate.json` for the
  primary report; if that file is absent, Stage 7 must not be launched as the
  reportable primary route.

If gate fails: stop at validation evidence. Do not open the held-out test set.

### Recommended Stage 6 candidate policy

For each completed Optuna study, repeat the auto-selected top candidates across:

```python
TOP_CONFIG_REEVALUATION_MODE = "group_kfold_seed_repeat"
TOP_K_CONFIGS_FOR_SEED_AND_CROSS_FOLD_REPEAT = "auto"
REPEAT_SEEDS = "42"
SEED_REPEAT_N_FOLDS = 5
STAGE6_PARALLEL_CROSS_VALIDATION_PROCESSES = 1
SEED_REPEAT_SPLIT_SEED = 42
RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = True
RETRAIN_BEST_CONFIG_AFTER_HPO = False
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
```

The default auto rule repeats up to 5 candidates below 50 completed trials, up
to 10 below 150 completed trials, and up to 20 for 150 or more completed
trials. Use a fixed integer only when the top-K count is predeclared before
launch; integer 20 is allowed for serious large HPO and with the default single
seed implies `20 x 5 x 1 = 100` extra validation-only runs.

A candidate is considered stable enough for final selection only if:

- all planned grouped-fold/active-seed runs complete, or failures are explained
  and not biased;
- no held-out test report was created;
- all source runs use the same `METAL_LABEL_SCHEME`;
- all runs use `selection_metric = val_metal_balanced_acc`;
- no active metal-scheme validation class is missing;
- rare-class recall is acceptable;
- paired bootstrap comparisons support the improvement over the relevant
  comparator.

## Stage 7 - One-Shot Held-Out Test

Purpose: report held-out test performance for the frozen Stage 6B final
full-train refit produced from the Stage-6-selected configuration. Stage 7 is
the only stage that may open the held-out test set. The primary final report
must be declared before test evaluation starts.

When to use it: only after model family, hyperparameters, Stage 6
interpretation, Stage 6B promotion decision, final-refit run, and final source
checkpoint are fixed.

Expected scale/runtime: final reporting run, usually minutes to hours depending
on checkpoint loading and evaluation mode.

First run the **Select final run and show saved outputs** cell. The selected
source must be the Stage 6B final full-train refit derived from the Stage 6
selected configuration. Do not use a raw Optuna trial or an arbitrary Stage 6
fold checkpoint as the primary final-test source. Use:

```python
FINAL_RUN_SELECTION_MODE = "stage6_selected_candidate"
FINAL_RUN_TABLE_INDEX = 1
FINAL_RUN_DIR = ""
FINAL_RUN_STAGE6_SELECTED_CANDIDATE_JSON = ""
FINAL_REPORT_BASENAME = "deepmzyme_final_selected_run"
```

The notebook resolves `stage6b_selected_final_refit_candidate.json` first when
it exists, then falls back to `stage6_selected_final_candidate.json` for legacy
preview. If the current notebook resolves the primary source to a Stage 6
fold/seed checkpoint instead of the Stage 6B final refit, stop before launching
Stage 7. That direct CV-fold evaluation is not the primary reportable path.

Then run the **Optional final held-out test evaluation** cell with launch still
disabled:

```python
FINAL_TEST_WORKFLOW = "evaluate_stage6_selected_candidate"
LAUNCH_FINAL_HELD_OUT_TEST_EVAL = False
```

Inspect the printed pre-flight checklist. The final-test cell supports exactly
two workflow values:

- `evaluate_stage6_selected_candidate`: primary serious final-test mode. It
  requires `stage6b_selected_final_refit_candidate.json` as final-source
  evidence for the primary route and must resolve to the frozen Stage 6B
  final-refit run for the Stage-6-selected rank #1 configuration. The output
  role is `primary_preselected`.
- `exploratory_evaluate_all_stage6_ranked_candidates`: optional post-hoc
  diagnostic mode. It requires both `stage6_ranked_candidates.csv` and
  `stage6_selected_final_candidate.json`, evaluates candidates in Stage 6 rank
  order, labels rank #1 as `primary_preselected`, labels every other row as
  `exploratory_posthoc`, and cannot be used to select or replace the primary
  model after held-out metrics are seen.

For the single-checkpoint primary report, if the Stage 6 selection evidence,
Stage 6B decision, and Stage 6B final-refit run are all frozen and point to the
same selected configuration, switch to launch:

```python
FINAL_TEST_WORKFLOW = "evaluate_stage6_selected_candidate"
LAUNCH_FINAL_HELD_OUT_TEST_EVAL = True
```

For optional exploratory diagnostics after the primary selected model is fixed:

```python
FINAL_TEST_WORKFLOW = "exploratory_evaluate_all_stage6_ranked_candidates"
LAUNCH_FINAL_HELD_OUT_TEST_EVAL = True
```

The exploratory mode prints and saves a strong warning. The primary model
remains the Stage 6B final-refit candidate regardless of exploratory held-out
test scores.

Expected outputs/files:

- Primary mode: a new final-test run folder under the resolved `RUNS_DIR`
- Exploratory all-candidates mode: one new final-test run folder per Stage-6
  ranked candidate, evaluated in Stage 6 rank order
- `run_config.json` and `run_metadata.json` in the final-test output folder
- `test_report.json` in the final-test output folder
- `<final_run_dir>/test_predictions.pt`
- `<final_run_dir>/test_temperature_validation_predictions.pt`, when
  validation logits are available for temperature fitting
- `<final_run_dir>/test_reliability_diagram.png`
- `<final_run_dir>/test_confidence_histogram.png`
- `<final_run_dir>/test_temperature_scaled_reliability_diagram.png`, when
  temperature scaling is available
- `<final_run_dir>/test_temperature_scaled_confidence_histogram.png`, when
  temperature scaling is available
- Exploratory mode additionally writes
  `<RUNS_DIR>/exploratory_final_test_all_stage6_ranked_candidates.csv`,
  `<RUNS_DIR>/exploratory_final_test_all_stage6_ranked_candidates.json`, and
  `<RUNS_DIR>/exploratory_final_test_warning.txt`
- Updated final-test summary CSV/PNG when plotting succeeds
- The source Stage 6B final-refit run remains unchanged

Exact configuration record:

- Stage 6/6B selection evidence: `stage6_selected_final_candidate.json`,
  `stage6_ranked_candidates.csv`, paired-bootstrap outputs,
  `stage6b_decision.json`, `stage6b_ranked_candidates.csv`, and
  `stage6b_selected_final_refit_candidate.json`.
- Source Stage 6B final-refit run: `<source_run_dir>/run_config.json` and
  `<source_run_dir>/run_metadata.json`.
- Final-test output: `<final_run_dir>/run_config.json`,
  `<final_run_dir>/run_metadata.json`, and `<final_run_dir>/test_report.json`.
- `active_run_config.json` / `active_run_config.md` are generated from the live
  notebook configuration before launch.

`test_report.json` schema additions:

- `task`
- `final_test_primary_report`
- `final_test_ensemble_mode`
- `final_test_result_role`
- `selected_config_id` / `selected_run_id`
- `checkpoint_paths`
- `seed_values`
- `run_directories`
- `test_structure_dir` / `test_summary_csv`
- `metrics`
- `calibrated_metrics`
- `fitted_temperatures`
- `temperature_scaling`
- `bootstrap_settings`
- CI fields such as `test_metal_balanced_acc_ci95`,
  `test_metal_collapsed4_balanced_acc_ci95`, and per-class recall CI fields
- `calibration_settings`
- `calibration_plot_paths`
- `reliability_diagram_path`
- `confidence_histogram_path`
- `prediction_artifact_path`
- `timestamp`
- `git_commit` / `code_version`
- explicit `selection_policy_statement` that no test metric was used for
  selection

Calibration and temperature scaling:

- Overall ECE uses predicted-class confidence with 15 equal-mass bins.
- Class-wise ECE uses one-vs-rest class probabilities with equal-mass bins.
- NLL is reported when metal probabilities are available.
- Temperature scaling fits one scalar only on validation logits from the
  validation-selected configuration or Stage 6B final-refit source run, then
  applies that temperature to held-out test logits.
- Ensemble temperature scaling uses the fixed rule: fit one scalar temperature
  per fixed checkpoint on that checkpoint's validation logits, apply it to that
  checkpoint's test logits, then average the calibrated softmax probabilities.
- No temperature, calibration method, seed subset, ensemble weight, threshold,
  checkpoint, or primary report can be selected using held-out test metrics.

Bootstrap confidence intervals:

- Default: 1000 stratified bootstrap resamples, 95% confidence intervals,
  `FINAL_TEST_BOOTSTRAP_SEED = 20260518`.
- Stratification is by true class, preserving every class present in the
  original held-out test labels.
- Report CIs for active-scheme balanced accuracy, per-class recall, collapsed-4
  balanced accuracy and collapsed per-class recall, plus ECE when available.

Success criteria:

- The source run is recorded in
  `stage6b_selected_final_refit_candidate.json` with Stage 6B validation/CV
  promotion evidence.
- The final-test run uses `best_model_checkpoint.pt` from the Stage 6B final
  full-train refit, or an explicitly selected fixed checkpoint recorded before
  Stage 7.
- The source run is the Stage 6B final-refit run derived from the Stage 6
  selected configuration, not a raw Optuna trial or arbitrary Stage 6 fold.
- The output folder is separate from the source Stage 6B final-refit run.
- Primary mode loads the selected candidate from
  `stage6b_selected_final_refit_candidate.json` and does not inspect or
  evaluate other candidates.
- Exploratory mode loads `stage6_ranked_candidates.csv` and
  `stage6_selected_final_candidate.json`, preserves Stage 6 rank order, labels
  rank #1 as `primary_preselected`, labels all other rows as
  `exploratory_posthoc`, and does not modify either Stage 6 file.
- The test report includes active metal-scheme metrics and collapsed-4 metrics.
- Primary and exploratory/post-hoc reports are labeled clearly.

### Decision gate after Stage 7

Final reporting is complete only if:

- The Stage 7 success criteria are met.
- The source run was the frozen Stage 6B final-refit run derived from the Stage
  6 validation-selected configuration.
- `val_metal_balanced_acc` was the Stage 6/6B selection metric; the Stage 6B
  full-train refit may use `train_loss` only as the predeclared checkpoint rule
  for the full non-test training run.
- The final-test output is a separate folder from the source final
  training/refit run.
- `test_report.json`, `run_config.json`, and `run_metadata.json` exist in the
  final-test output folder.
- `test_report.json` records `final_test_result_role` / `role`, selected
  configuration identity, checkpoint path(s), seed values, run directories,
  calibration settings, bootstrap settings, plot paths, and the
  no-test-selection policy statement.
- Temperature scaling, if present, was fitted only on validation logits from
  the fixed validation-selected configuration.
- Bootstrap CI fields are present for the requested final-test metrics, or the
  report explicitly records why CIs were disabled.
- This is the first and only Stage 7 launch for this validation-selected
  configuration and Stage 6B final-refit run; keep
  `ALLOW_REPEAT_FINAL_TEST_EVAL = False` unless explicitly documenting a
  non-reportable rerun.
- The Stage 7 result is not used to pick a different checkpoint, seed,
  hyperparameter set, model family, fusion mode, ensemble subset, ensemble
  weight, calibration method, temperature, threshold, or primary report.

If gate fails: do not report the run as final. If the test completed, treat the
one-shot final-test result as already spent for that selection cycle. Do not
choose a different configuration because another tested candidate has a better
held-out test score; return to validation-only experiments for new development.

## Safety Guards To Check

Before any reportable comparison or HPO launch, confirm:

- `INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False`
- `FINAL_TEST_WORKFLOW = "evaluate_stage6_selected_candidate"` for primary
  final reporting, or the final-test cell has not been run
- `LAUNCH_FINAL_HELD_OUT_TEST_EVAL = False` until the separate Stage 7 cell is
  intentionally launched
- `stage6_selected_final_candidate.json` exists as Stage 6 evidence, and
  `stage6b_selected_final_refit_candidate.json` exists as the completed Stage
  6B final-refit source before primary Stage 7 launch
- Exploratory all-candidates mode is explicitly selected only for post-hoc
  diagnostics and cannot change the primary selected model
- `VAL_FRACTION > 0` or a fold split is explicitly configured for validation
  stages; Stage 6B is the only reportable full-train refit path with
  `VAL_FRACTION = 0.0`
- `SELECTION_METRIC = "val_metal_balanced_acc"` for metal model selection
- `RUN_BATCH_ID` identifies the experiment batch clearly, and the default
  `SUMMARY_BASENAME` is derived from live provenance rather than stale manual
  labels
- `ALLOW_SHORT_TRAINING_FOR_DEBUG = False` for reportable runs
- `ALLOW_SEED_REPEAT_MODEL_PRESET_MISMATCH = False`
- `ALLOW_MIXED_FINAL_TEST_BATCH = False`
- `ALLOW_REPEAT_FINAL_TEST_EVAL = False`

The training code blocks ordinary held-out test evaluation when there is no
validation split or when `train_loss` would be used for final-test checkpoint
selection. The only reportable exception is the Stage 6B final-refit path,
where validation/CV evidence has already fixed the configuration and the
full-train checkpoint rule before Stage 7.
