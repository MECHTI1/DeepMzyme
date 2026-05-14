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
- If this file conflicts with `Plan.md`, source code, or run outputs, report the
  conflict instead of silently trusting this file.
- Do not invent missing values or exact experiment numbers.

## Current Stage

- Current task focus: metal classification.
- Stage: metal Only-ESM baseline confirmed; GVP + late-fusion candidate search
  has now advanced through a 50-trial validation-only Optuna run. The latest
  late-fusion seed-repeat block is not usable for anchor selection because the
  generated reruns used only `1` epoch.
- Trusted split policy: non-overlapped PinMyMetal train/test split, with
  validation split by `pdbid` and `VAL_FRACTION=0.15` for model selection.
- Test-set policy: held-out test remains unused for model, checkpoint,
  hyperparameter, architecture, and fusion decisions. Use it once after the
  validation-selected anchor is fixed.
- Latest copied notebook evidence: GVP + late fusion Round 3 Optuna output under
  `docs/notebook_outputs/raw/GVP + late fusion/Round3_late_fusion_optuna_50_v1.output_cell_notebook.md`.
- Selected stable Only-ESM anchor: confirmed original `3e-5` +
  `inverse_frequency` configuration from 5-seed validation evidence.
- Selected GVP + late-fusion anchor: not selected yet. Round 3 identified
  candidates, but the seed-repeat evidence is smoke/debug only.

## Notebook Output File Map

- Current experiment evidence:
  - `docs/notebook_outputs/raw/GVP + late fusion/Round3_late_fusion_optuna_50_v1.output_cell_notebook.md`
    contains the latest controlled 50-trial GVP + late-fusion Optuna run and
    its generated top-3 seed-repeat block. The seed-repeat commands used
    `--epochs 1`, so those reruns are not model-quality comparisons.
  - `docs/notebook_outputs/raw/GVP + late fusion/Round2_late_fusion_from_confirmed_only_esm_anchor.output_cell_notebook.md`
    contains the fixed five-seed late-fusion check using the confirmed Only-ESM
    training settings.
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
- Current summaries / planning notes:
  - Concise run summaries are under `docs/notebook_outputs/summaries/`.
  - Current late-fusion summary:
    `docs/notebook_outputs/summaries/summary_run_gvp_late_fusion_round3_optuna_50_v1.md`.
  - Older Only-GVP planning notes and outputs remain useful historical context.
- Stable usage guide:
  - `docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md` should stay focused on stable
    notebook usage principles and should point here for current status.

## Latest Trusted Evidence

All numbers below are validation metrics from copied notebook outputs. They are
not held-out test results.

### GVP + Late Fusion Round 3 Optuna

Round 3 ran a controlled Optuna search inside the GVP + late-fusion model
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

Raw-output check against
`docs/notebook_outputs/summaries/summary_run_gvp_late_fusion_round3_optuna_50_v1.md`:

- the raw file contains `50` completed Optuna trial-finished records;
- best Optuna trial is trial `49` with
  `val_metal_balanced_acc=0.6750130535709283`;
- the best trial command used `--epochs 40` and selected epoch `37`;
- the generated seed-repeat commands used `--epochs 1`;
- the raw output explicitly states that `1-3` epoch runs are smoke/debug only;
- no `test_report.json` was created for the selected run;
- failed run directories were reported as `[]`.

Top single-seed Optuna candidates:

| Rank | Trial | Validation balanced accuracy | Key settings |
|---:|---:|---:|---|
| 1 | `49` | 0.6750130535709283 | `lr=1.6801503587890522e-05`, `wd=1e-05`, `hidden_s=256`, `hidden_v=32`, `edge_hidden=128`, `gvp_layers=4`, `edge_radius=6.0`, `esm_fusion_dim=64`, `head_mlp_layers=1`, `class_weight=inverse_frequency` |
| 2 | `32` | 0.6585119076580177 | `lr=5.4715836015281065e-05`, `wd=0.001`, `hidden_s=128`, `hidden_v=32`, `edge_hidden=128`, `gvp_layers=2`, `edge_radius=6.0`, `esm_fusion_dim=64`, `head_mlp_layers=1`, `class_weight=inverse_frequency` |
| 3 | `15` | 0.6550963478857217 | `lr=7.032630334240692e-05`, `wd=0.001`, `hidden_s=128`, `hidden_v=32`, `edge_hidden=128`, `gvp_layers=2`, `edge_radius=6.0`, `esm_fusion_dim=64`, `head_mlp_layers=1`, `class_weight=inverse_frequency` |

The generated top-3 seed-repeat table reported means of `0.3349`, `0.3290`,
and `0.3099` for trials `32`, `15`, and `49`, respectively, but those values
come from `1`-epoch reruns. They should be recorded as smoke/debug results and
should not be used to reject or select a late-fusion anchor.

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

Round 2 screened Only-ESM `learning_rate`, `weight_decay`, and
`metal_class_weight_mode` values at seeds `42,123,2026`. The intended grid was
`3` learning rates (`2e-5,3e-5,5e-5`) x `2` weight decays (`1e-4,1e-5`) x `2`
class-weight modes x `3` seeds = `36` runs, but only `24` runs completed. The
`5e-5` learning-rate rows are absent from the copied evidence.

Round 3 added seeds `43` and `44` for the Cartesian product of:

- `LEARNING_RATES_CSV=3e-5,2e-5`
- `WEIGHT_DECAYS_CSV=1e-4`
- `METAL_CLASS_WEIGHT_MODES_CSV=inverse_sqrt_frequency,inverse_frequency`
- `HEAD_MLP_LAYERS_VALUES_CSV=2`

Combined Round 2 + Round 3 5-seed validation-balanced-accuracy summary for the
matching `weight_decay=1e-4`, `head_mlp_layers=2` configurations:

| Learning rate | Class weight | Mean | Sample std | Min | Max | Current interpretation |
|---:|---|---:|---:|---:|---:|---|
| `3e-5` | `inverse_frequency` | 0.6253 | 0.0314 | 0.5902 | 0.6722 | Confirmed best Only-ESM anchor; same setting as Round 1. |
| `3e-5` | `inverse_sqrt_frequency` | 0.6219 | 0.0499 | 0.5546 | 0.6930 | Round 2's apparent winner did not remain best after seeds `43` and `44`. |
| `2e-5` | `inverse_frequency` | 0.6199 | 0.0278 | 0.5800 | 0.6524 | Most stable among the Round 2 + Round 3 grid, but lower mean than the anchor. |
| `2e-5` | `inverse_sqrt_frequency` | 0.6072 | 0.0402 | 0.5492 | 0.6605 | Lower mean and weaker worst-seed result. |

The confirmed Only-ESM anchor is:

- `learning_rate=3e-5`
- `weight_decay=1e-4`
- `metal_class_weight_mode=inverse_frequency`
- `head_mlp_layers=2`
- `batch_size=8`
- `metal_loss_function=cross_entropy`
- `metal_label_smoothing=0.0`
- `EPOCHS=50`
- `SELECTION_METRIC=val_metal_balanced_acc`
- no held-out test during training

## Current Recommendation

- Keep the confirmed Only-ESM anchor above as the current stable ESM baseline.
- Do not select a GVP + late-fusion anchor from Round 3 yet. The valid evidence
  is single-seed 40-epoch Optuna candidate discovery; the seed-repeat block is
  1-epoch smoke/debug evidence.
- Do not run held-out test yet. The held-out test remains postponed until the
  validation-only model/fusion comparison is complete.
- Do not spend another broad Only-ESM search now.

## Recommended Next Notebook Action

Run a proper validation-only seed-repeat confirmation of the top late-fusion
Optuna candidates from Round 3. Use a new batch id such as
`metal_late_fusion_optuna_top3_seedrepeat_50epoch_v1`.

Recommended settings shared by all reruns:

- `TASK=metal`
- `RUN_MODE=manual_configurations`
- `RECOMMENDED_RUN_SET=custom`
- `MODEL_PRESET=GVP + late fusion`
- `RUN_BATCH_ID=metal_late_fusion_optuna_top3_seedrepeat_50epoch_v1`
- `EPOCHS=50`
- `SEEDS_CSV=42,123,2026,43,44`
- `BATCH_SIZES_CSV=8`
- use the three fixed candidate rows from trials `49`, `32`, and `15` listed in
  the table above
- `METAL_CLASS_WEIGHT_MODES_CSV=inverse_frequency`
- `HEAD_MLP_LAYERS_VALUES_CSV=1`
- `METAL_LOSS_FUNCTION=cross_entropy`
- `METAL_LABEL_SMOOTHING=0.0`
- `SELECTION_METRIC=val_metal_balanced_acc`
- `INCLUDE_HELD_OUT_TEST_DURING_TRAINING=False`
- `MAX_CONFIGURATION_RUNS=15` is sufficient for `3` candidates x `5` seeds.
- `SPLIT_BY=pdbid`
- `VAL_FRACTION=0.15`

The simplest execution path is to reuse the generated top-reevaluation commands
from Round 3 but change `--epochs 1` to `--epochs 50`, write to the new
`RUN_BATCH_ID`, and keep held-out test disabled.

## Decision Rule

Choose model and fusion anchors by seed-repeat mean, stability, and per-class
diagnostics, not by one lucky seed.

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
- The copied Only-ESM outputs inspected here do not show created
  `test_report.json` files for the confirmed anchor runs.

## Next Stage

- Next validation-only stage: full-budget seed-repeat confirmation for late
  fusion Optuna trials `49`, `32`, and `15`.
- RING should be a later small side ablation, not mixed into the first
  ESM/fusion comparison.

## Caveats

- Exact hyperparameters and result numbers must be parsed from raw notebook
  outputs or saved configs before being used in a publication table.
- GVP + late-fusion Round 3 includes a notebook-selected best single Optuna
  trial. Treat it as candidate discovery, not a project-level anchor decision.
- GVP + late-fusion Round 3's top-3 seed-repeat table used only `1` epoch per
  seed; do not compare those values to 40-epoch Optuna trials, 50-epoch
  seed-repeat baselines, or Only-ESM anchor evidence.
- The copied Only-ESM outputs do not provide a compact combined per-class
  diagnostic table across all four confirmed configs. Aggregate validation
  balanced accuracy is sufficient for the current anchor decision, but per-class
  diagnostics should be inspected before publication reporting.
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
