# EXPERIMENT_STATUS.md - DeepMzyme

This is the short, mutable status note for current experiments. Keep exact run
evidence in saved outputs, not in stable workflow guides.

## Authority And Evidence Rules


- `Plan.md` remains the design authority for intended architecture, training
  logic, experiment policy, and project direction.
- Source code under `src/` is evidence of implemented behavior.
- Run outputs, saved configs, and notebook outputs are evidence of measured
  results.
- `EXPERIMENT_STATUS.md` is only a current-orientation note and may lag behind
  recent runs or code changes.
- For metal-training stages, use `docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md` as
  the only source of exact executable values, budgets, search spaces, expected
  outputs, and gates. Use `docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md` for
  option meanings and workflow explanation.
- As of the Stage 6 statistical-validation update, reportable Stage 6
  confirmation uses top-K 5-fold grouped validation by `pdbid`, paired
  fold-level bootstrap confidence intervals, and rare-class recall protection.
  Older 5-seed validation-only repeats remain historical evidence, not the
  current Stage 6 standard for new promotions.
- If this file conflicts with `Plan.md`, source code, or run outputs, report the
  conflict instead of silently trusting this file.
- Do not invent missing values or exact experiment numbers.
- Default for new checks: unless the user explicitly asks to rely on previous
  running/results/raw outputs, treat prior raw outputs as context only and plan
  a fresh validation-only Optuna search with the largest sensible search space
  for the chosen task/model family.
- Explicit prior-run mode: when the user asks to rely on previous raws/results,
  inspect the cited evidence and use it directly to narrow, continue, or repeat
  that previous configuration.



## Current Stage

- Current task focus: metal classification, with exploratory joint
  (metal + EC) hybrid-fusion and hybrid+RING checks now also on record.
- Current default configuration policy for new notebook/playbook graph runs:
  use updated external features strictly and use RING-enabled graph
  construction by default. In notebook/configuration terms this means
  `RING_EDGE_MODE="with_ring"`, `ALLOW_MISSING_EXTERNAL_FEATURES=False`,
  `PREPARE_MISSING_EXTERNAL_FEATURES=True`, `external_feature_source=updated`,
  `--use-ring-edges`, `--ring-features-dir`, and
  `--prepare-missing-ring-edges`. Radius-only graph construction is now an
  explicit ablation or raw low-level/CLI default, not the recommended first
  graph setting.
- Trusted ESM embedding variant for new ESM/fusion work: ESMC `esmc_300m` with
  `embedding_dim=960`. Newly generated embeddings should have `*.pt.json`
  sidecar metadata. Older copied embeddings without sidecars are
  `unknown_in_older_embeddings`; do not infer their exact checkpoint from
  path/name alone.
- Stage: metal Only-ESM baseline confirmed; GVP + late fusion has completed a
  historical 5-seed, 50-epoch validation-only seed repeat for the top Round 3
  Optuna candidates; GVP + node-level late fusion has also completed a 5-seed,
  50-epoch validation-only check initialized from the selected late-fusion
  anchor; GVP + ESM hybrid fusion has completed an exploratory Round 1
  Optuna + 3-seed top-K repeat for the joint task.
- Trusted split policy for current evidence: legacy Non-overlapped PinMyMetal
  train/test split, with validation split by `pdbid` and `VAL_FRACTION=0.15`
  for model selection. New Harsh Split PinMyMetal and Common-PDBID 70/30 Split
  PinMyMetal variants exist for future comparisons but are not the basis of the
  current reported anchors.
- Test-set policy: held-out test remains unused for model, checkpoint,
  hyperparameter, architecture, and fusion decisions. Use it once after the
  validation-selected anchor is fixed.
- Selected stable Only-ESM anchor: confirmed original `3e-5` +
  `inverse_frequency` configuration from 5-seed validation evidence.
- Selected GVP + late-fusion anchor: trial `49`, selected from the historical
  Round 4 5-seed, 50-epoch validation-only seed repeat. Under the current
  Stage 6 policy, any new replacement/promotion should be confirmed with
  shared grouped folds and paired bootstrap CIs.
- Node-level late-fusion status: Round 1 did not replace the selected GVP +
  late-fusion trial `49` anchor.
- Hybrid fusion status: Round 1 is an exploratory joint-task batch only and
  has not been confirmed under the current Stage 6 grouped-fold, paired-
  bootstrap validation standard. Its best single-seed `val_joint_balanced_acc`
  (trial 17 = `0.748343`) is not a metal anchor; the corresponding
  metal-side `val_metal_balanced_acc` was `0.672077`.
- Hybrid+RING status: the `joint_hybrid_ring_optuna_50epoch_wide_v1`
  continuation log is also exploratory single-seed validation evidence. In the
  copied trials 105-176 artifact, trial `114` reached
  `val_metal_balanced_acc=0.7303469775006777`, but this has not been confirmed
  under the current Stage 6 grouped-fold, paired-bootstrap validation standard
  and is not a replacement anchor.

For a cross-family snapshot of validation results and reliability tiers,
see `docs/notebook_outputs/summaries/LEADERBOARD.md`.

## Notebook Output File Map

- Current experiment evidence:
  - `docs/notebook_outputs/summaries/summary_run_gvp_late_fusion_round4_top3_seedrepeat_50epoch.md`
    summarizes the historical 15-run seed-repeat batch for GVP + late-fusion trials
    `49`, `32`, and `15`, using 5 seeds and 50 epochs.
  - `docs/notebook_outputs/raw/GVP + late fusion/metal_late_fusion_optuna_top3_seedrepeat_50epoch_v1/`
    contains copied run artifacts for that Round 4 seed-repeat batch.
  - `docs/notebook_outputs/summaries/summary_run_gvp_node_level_late_fusion_round1_from_latefusion_trial49_seedrepeat_50epoch.md`
    summarizes the node-level late-fusion 5-seed, 50-epoch validation-only
    check from the selected trial `49` anchor.
  - `docs/notebook_outputs/raw/GVP + node-level late fusion/Round1_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1.output_cell_notebook.md`
    contains the copied raw notebook output for the node-level late-fusion
    check.
  - `docs/notebook_outputs/raw/GVP + late fusion/Round3_late_fusion_optuna_50_v1.output_cell_notebook.md`
    remains useful as candidate-discovery evidence for the Round 3 50-trial
    Optuna run. Its generated top-3 seed-repeat commands used `--epochs 1`, so
    those generated reruns are smoke/debug evidence only.
  - `docs/notebook_outputs/raw/Only-ESM/Round1_Rerun validation-only Only-ESM on full ESM coverage.output_cell_notebook`
    contains the original 5-seed validation-only Only-ESM anchor evidence.
  - `docs/notebook_outputs/raw/Only-ESM/Round2_ESMonly.output_cell_notebook.md`
    contains the narrow Only-ESM learning-rate, weight-decay, and class-weight
    screen. It intended `36` runs but only `24` ran; `learning_rate=5e-5` was
    not run because the notebook planned/executed only the first `24` Cartesian
    product rows, consistent with `MAX_CONFIGURATION_RUNS=24`.
  - `docs/notebook_outputs/raw/Only-ESM/Round3_ESMonly_add_seeds43_44_5seed_confirmation.output_cell_notebook.md`
    contains the Round 3 confirmation run adding seeds `43` and `44` for the
    Round 2 finalist settings.
  - `docs/notebook_outputs/raw/Hybrid/Round1_hybrid_fusion_optuna_plus_top3_seedrepeat.output_cell_notebook.md`
    contains the GVP + ESM hybrid fusion Round 1 Optuna + top-3 seed-repeat
    for the joint task. The batch id in the raw output is `debug_smoke` and
    the raw output flags mixed/missing `RUN_BATCH_ID` values; treat it as
    exploratory evidence only.
  - `docs/notebook_outputs/raw/Hybrid/Round2_joint_hybrid_ring_optuna_50epoch_wide_v1_trials105_176_partial_trial177.output_cell_notebook.md`
    contains the copied continuation log for the GVP + ESM hybrid fusion +
    RING Optuna study, with completed trials `105`-`176` and an incomplete
    started trial `177`. Treat it as exploratory single-seed validation
    evidence only.
- Current summaries / planning notes:
  - Concise run summaries are under `docs/notebook_outputs/summaries/`.
  - Cross-family validation snapshot:
    `docs/notebook_outputs/summaries/LEADERBOARD.md`.
  - Current late-fusion anchor summary:
    `docs/notebook_outputs/summaries/summary_run_gvp_late_fusion_round4_top3_seedrepeat_50epoch.md`.
  - Current node-level late-fusion summary:
    `docs/notebook_outputs/summaries/summary_run_gvp_node_level_late_fusion_round1_from_latefusion_trial49_seedrepeat_50epoch.md`.
  - Hybrid fusion Round 1 exploratory summary:
    `docs/notebook_outputs/summaries/summary_run_hybrid_round1_optuna_plus_top3_seedrepeat.md`.
  - Hybrid+RING exploratory continuation summary:
    `docs/notebook_outputs/summaries/summary_run_hybrid_ring_round2_optuna_50epoch_wide_v1_trials105_176.md`.
- Stable usage guide:
  - `docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md` should stay focused on stable
    notebook usage principles and should point here for current status.

## Latest Trusted Evidence

All numbers below are validation metrics from copied notebook outputs or copied
run artifacts. They are not held-out test results.

### GVP + Late Fusion Round 4 Top-3 Seed Repeat

Round 4 was the historical validation-only seed-repeat confirmation for the top
3 GVP + late-fusion Optuna candidates from Round 3:

- Source Optuna trials: `49`, `32`, `15`
- Seeds: `42,123,2026,43,44`
- Intended and completed runs: `3` trial configs x `5` seeds = `15`
- Epochs: `50`
- Batch size: `8`
- Split: `pdbid`
- Validation fraction: `0.15`
- Selection metric: `val_metal_balanced_acc`
- Metal class weighting: `inverse_frequency`
- Metal loss: `cross_entropy`
- Metal label smoothing: `0.0`
- Learning-rate schedule: `fixed`
- Node feature set: `conservative`
- RING edges: disabled
- Held-out test during training: disabled
- `test_report`: absent / null for all runs

Aggregate validation-balanced-accuracy results:

| Rank by mean | Source trial | n seeds | Mean | Sample std | Min | Max |
|---:|---:|---:|---:|---:|---:|---:|
| `1` | `49` | `5` | `0.635468206972` | `0.043023727308` | `0.597794518922` | `0.688000505242` |
| `2` | `32` | `5` | `0.630032719914` | `0.051725380573` | `0.550812772796` | `0.686374815178` |
| `3` | `15` | `5` | `0.629927684340` | `0.052550289215` | `0.563584164877` | `0.699275572298` |

Decision: select GVP + late fusion trial `49` as the current validation-selected
late-fusion metal anchor. Trial `49` has the best mean, lowest standard
deviation among the three late-fusion candidates, and the best worst-seed
result in the Round 4 evidence.

Selected GVP + late-fusion trial `49` anchor hyperparameters:

- `learning_rate=1.6801503587890522e-05`
- `weight_decay=1e-05`
- `batch_size=8`
- `hidden_s=256`
- `hidden_v=32`
- `gvp_layers=4`
- `edge_hidden=128`
- `edge_radius=6.0`
- `head_mlp_layers=1`
- `esm_fusion_dim=64`
- `metal_class_weight_mode=inverse_frequency`
- `metal_loss_function=cross_entropy`
- `metal_label_smoothing=0.0`
- `selection_metric=val_metal_balanced_acc`

The selected trial `49` anchor result is:

- mean `val_metal_balanced_acc=0.635468206972`
- sample std `0.043023727308`
- min `0.597794518922`
- max `0.688000505242`
- n seeds `5`

Comparison to the confirmed Only-ESM anchor from prior 5-seed validation
evidence:

- Only-ESM mean approximately `0.6253`
- Only-ESM sample std approximately `0.0314`
- Only-ESM min approximately `0.5902`
- Only-ESM max approximately `0.6722`

Interpretation: GVP + late fusion trial `49` narrowly improves over the
confirmed Only-ESM anchor by mean and worst-seed value, but the improvement is
modest and the late-fusion seed variance is higher.

### GVP + Node-Level Late Fusion Round 1

Round 1 tested GVP + node-level late fusion using the selected GVP +
late-fusion trial `49` anchor settings:

- Task: `metal`
- Model preset: `GVP + node-level late fusion`
- Batch id:
  `metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1`
- Run mode: `manual_configurations`
- Planned runs: `5`
- Completed runs: `5`
- Failed runs: none
- Epochs per run: `50`
- Seeds: `42, 123, 2026, 43, 44`
- Validation fraction: `0.15`
- Split: `pdbid`
- Selection metric: `val_metal_balanced_acc`
- Held-out test during training: disabled
- Held-out test results present in copied output: false

Aggregate validation result:

| Metric | Value |
|---|---:|
| mean `val_metal_balanced_acc` | `0.606599196822` |
| sample std | `0.023404449951` |
| min | `0.574873163235` |
| max | `0.633163185699` |
| n | `5` |

Decision: do not replace the selected GVP + late-fusion trial `49` anchor with
this node-level late-fusion configuration. The node-level late-fusion mean is
below both the selected GVP + late-fusion trial `49` anchor and the Only-ESM
anchor, and its best seed result is below the late-fusion trial `49` mean.

### GVP + Late Fusion Round 3 Optuna

Round 3 remains candidate-discovery evidence for the GVP + late-fusion model
family:

- `TASK=metal`
- `MODEL_PRESET=GVP + late fusion`
- `MODEL_ARCHITECTURE=gvp`
- `FUSION_MODE=late_fusion`
- `N_TRIALS=50`
- `MAX_EPOCHS_PER_TRIAL=40`
- fixed HPO split/model seed `42`
- `SPLIT_BY=pdbid`
- `VAL_FRACTION=0.15`
- `SELECTION_METRIC=val_metal_balanced_acc`
- no held-out test during training

Top single-seed Optuna candidates from Round 3:

| Rank | Trial | Validation balanced accuracy | Key settings |
|---:|---:|---:|---|
| 1 | `49` | `0.6750130535709283` | `lr=1.6801503587890522e-05`, `wd=1e-05`, `hidden_s=256`, `hidden_v=32`, `edge_hidden=128`, `gvp_layers=4`, `edge_radius=6.0`, `esm_fusion_dim=64`, `head_mlp_layers=1`, `class_weight=inverse_frequency` |
| 2 | `32` | `0.6585119076580177` | `lr=5.4715836015281065e-05`, `wd=0.001`, `hidden_s=128`, `hidden_v=32`, `edge_hidden=128`, `gvp_layers=2`, `edge_radius=6.0`, `esm_fusion_dim=64`, `head_mlp_layers=1`, `class_weight=inverse_frequency` |
| 3 | `15` | `0.6550963478857217` | `lr=7.032630334240692e-05`, `wd=0.001`, `hidden_s=128`, `hidden_v=32`, `edge_hidden=128`, `gvp_layers=2`, `edge_radius=6.0`, `esm_fusion_dim=64`, `head_mlp_layers=1`, `class_weight=inverse_frequency` |

The generated Round 3 top-3 seed-repeat commands used `--epochs 1`. Those
generated reruns should be recorded as smoke/debug results only and should not
be used to reject or select a late-fusion anchor. The Round 4 seed-repeat above
is now the trusted late-fusion anchor-selection evidence.

### Confirmed Only-ESM Anchor

Only-ESM Round 1 ran the original full-coverage 50-epoch validation-only
seed-repeat batch across seeds `42,123,2026,43,44`, with:

- `TASK=metal`
- `MODEL_PRESET=Only-ESM`
- `EPOCHS=50`
- `BATCH_SIZES_CSV=8`
- `LEARNING_RATES_CSV=3e-5`
- `WEIGHT_DECAYS_CSV=1e-4`
- `SPLIT_BY=pdbid`
- `VAL_FRACTION=0.15`
- `SELECTION_METRIC=val_metal_balanced_acc`
- `METAL_CLASS_WEIGHT_MODES_CSV=inverse_frequency`
- `HEAD_MLP_LAYERS_VALUES_CSV=2`
- `METAL_LOSS_FUNCTION=cross_entropy`
- `METAL_LABEL_SMOOTHING=0.0`
- no held-out test during training

Confirmed Only-ESM anchor summary from prior 5-seed validation evidence:

- mean approximately `0.6253`
- sample std approximately `0.0314`
- min approximately `0.5902`
- max approximately `0.6722`

The confirmed Only-ESM anchor remains the stable ESM-only baseline.

## Current Recommendation

- Keep the confirmed Only-ESM anchor as the stable ESM baseline.
- Keep GVP + late fusion trial `49` as the current validation-selected metal
  anchor.
- Do not replace the selected GVP + late-fusion trial `49` anchor with the
  tested node-level late-fusion configuration.
- Do not promote hybrid fusion Round 1 to anchor status. Before treating
  hybrid fusion as a candidate metal anchor, run the current Stage 6 top-K
  grouped-fold confirmation at the full validation epoch budget, and compare
  its metal-side balanced accuracy against the confirmed Only-ESM and GVP +
  late-fusion (trial `49`) anchors with paired bootstrap CIs and rare-class
  recall protection.
- Do not run held-out test yet unless the validation architecture search is
  explicitly declared complete.
- Do not spend another broad Only-ESM search now.
- For any new validation-side metal Optuna search, use the playbook's updated
  serious batch-size policy: search `8,16`, keep `4` for smoke/debug or memory
  fallback, and test `32` only as a separately labeled validation-only
  ablation. Keep final reporting tied to the selected run's own batch size
  unless a validation-only batch-size ablation selects a replacement.

## Recommended Next Notebook Action

Decide whether validation-side architecture selection is complete. If it is
complete, the next notebook action should be an explicit final-reporting step
using the held-out test exactly once for the selected anchors. If it is not
complete, run only narrowly scoped validation-side ablations; do not use the
held-out test for those decisions.

Current selected validation anchor for metal:

- GVP + late fusion trial `49`

Current non-selected recent architecture check:

- GVP + node-level late fusion Round 1 from trial `49`

## Decision Rule

Choose model and fusion anchors by Stage 6 grouped-fold mean, paired bootstrap
comparison, and per-class diagnostics, not by one lucky seed or raw single-run
delta.

Use, at minimum:

- `val_metal_balanced_acc`
- `val_metal_macro_f1`
- `val_metal_min_recall`
- `val_metal_per_class_recall`
- `val_metal_collapsed4_balanced_acc`

## Test-Set Rule

- Held-out test is for final reporting only.
- Do not use held-out test to choose model, hyperparameters, checkpoint,
  architecture, fusion mode, or seed.
- The copied Round 4 GVP + late-fusion summary reports `test_report` absent /
  null for all runs.
- The copied node-level late-fusion summary reports held-out test during
  training disabled and held-out test results present in copied output as false.

## Next Stage

- Next validation-only stage: optional narrow validation-side ablations only if
  architecture selection is not yet declared complete.
- Current validation-selected metal anchor: GVP + late fusion trial `49`.
- RING should be a later small side ablation, not mixed into the first
  ESM/fusion comparison.

## Caveats

- Round 4 GVP + late-fusion, node-level late-fusion Round 1, and hybrid
  fusion Round 1 are validation-only results. They are not held-out test
  results.
- Hybrid fusion Round 1 used a 3-seed top-K repeat (seeds `42,123,2026`),
  not the current Stage 6 grouped-fold confirmation standard, its raw output is
  tagged with a
  `debug_smoke` batch id and a mixed-batch warning, and its selection metric
  is `val_joint_balanced_acc`, not pure metal balanced accuracy. Treat it as
  exploratory evidence only.
- Held-out test remains postponed until the validation-side model/fusion
  selection is explicitly finalized.
- The GVP + late-fusion trial `49` improvement over the confirmed Only-ESM
  anchor is modest, and trial `49` has higher seed variance than the Only-ESM
  anchor.
- Per-class diagnostic aggregates are not clearly available in the copied Round
  4 late-fusion folder artifacts; the Round 4 summary is based on the saved
  selected validation metric from `run_metadata.json`.
- The copied earlier notebook-output file
  `Round4_late_fusion_optuna_top3_seedrepeat_50epoch_v1.output_cell_notebook.md`
  showed only one normal planned run and should not be used as the main Round 4
  evidence now that the full run-artifact folder is available.
- The generated Round 3 top-3 seed-repeat table used only `1` epoch per seed;
  do not compare those values to 40-epoch Optuna trials, 50-epoch seed-repeat
  baselines, or Only-ESM anchor evidence.
- Saved/displayed `fusion=late_fusion` may appear in some Only-ESM tables, but
  for `only_esm` the effective fusion mode is no graph/ESM fusion.

## Update Checklist

After each real batch:

- Update current stage.
- Update raw evidence source.
- Update selected anchor, if any.
- Update next planned batch.
- Update caveats.
- Keep detailed run evidence in notebook-output files or saved run summaries,
  not in `docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md`.
