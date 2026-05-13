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
- Stage: metal Only-ESM baseline confirmed; next planned comparison is a
  validation-only tiny GVP + late-fusion stability check.
- Trusted split policy: non-overlapped PinMyMetal train/test split, with
  validation split by `pdbid` and `VAL_FRACTION=0.15` for model selection.
- Test-set policy: held-out test remains unused for model, checkpoint,
  hyperparameter, architecture, and fusion decisions. Use it once after the
  validation-selected anchor is fixed.
- Latest copied notebook evidence: Only-ESM Round 3 confirmation output under
  `docs/notebook outputs/Only-ESM/Round3_ESMonly_add_seeds43_44_5seed_confirmation.output_cell_notebook.md`.
- Selected stable Only-ESM anchor: confirmed original `3e-5` +
  `inverse_frequency` configuration from 5-seed validation evidence.

## Notebook Output File Map

- Current experiment evidence:
  - `docs/notebook outputs/Only-ESM/Round1_Rerun validation-only Only-ESM on full ESM coverage.output_cell_notebook`
    contains the original 5-seed validation-only Only-ESM anchor evidence.
  - `docs/notebook outputs/Only-ESM/Round2_ESMonly.output_cell_notebook.md`
    contains the narrow Only-ESM learning-rate, weight-decay, and class-weight
    screen. It intended `36` runs but only `24` ran; `learning_rate=5e-5` was
    not run because the notebook planned/executed only the first `24` Cartesian
    product rows, consistent with `MAX_CONFIGURATION_RUNS=24`.
  - `docs/notebook outputs/Only-ESM/Round3_ESMonly_add_seeds43_44_5seed_confirmation.output_cell_notebook.md`
    contains the Round 3 confirmation run adding seeds `43` and `44` for the
    Round 2 finalist settings.
- Current summaries / planning notes:
  - Older Only-GVP planning notes and outputs remain useful historical context,
    but the current metal-task status is now governed by the Only-ESM evidence
    listed above and the next planned late-fusion validation check.
- Stable usage guide:
  - `docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md` should stay focused on stable
    notebook usage principles and should point here for current status.

## Latest Trusted Evidence

All numbers below are validation metrics from copied notebook outputs. They are
not held-out test results.

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

- Select the confirmed Only-ESM anchor above. It confirms the original Round 1
  `3e-5` + `inverse_frequency` configuration rather than replacing it with the
  attempted Round 2 winner.
- Do not run held-out test yet. The held-out test remains postponed until the
  validation-only model/fusion comparison is complete.
- Do not spend another broad Only-ESM search now. The next useful comparison is
  a tiny validation-only GVP + late-fusion stability check using the confirmed
  Only-ESM training settings.

## Recommended Next Notebook Action

Run a validation-only tiny GVP + late-fusion stability check. Recommended
settings:

- `TASK=metal`
- `RUN_MODE=manual_configurations`
- `RECOMMENDED_RUN_SET=custom`
- `MODEL_PRESET=GVP + late fusion`
- `RUN_BATCH_ID=metal_late_fusion_from_confirmed_only_esm_anchor_v1`
- `EPOCHS=50`
- `SEEDS_CSV=42,123,2026,43,44`
- `BATCH_SIZES_CSV=8`
- `LEARNING_RATES_CSV=3e-5`
- `WEIGHT_DECAYS_CSV=1e-4`
- `METAL_CLASS_WEIGHT_MODES_CSV=inverse_frequency`
- `HEAD_MLP_LAYERS_VALUES_CSV=2`
- `METAL_LOSS_FUNCTION=cross_entropy`
- `METAL_LABEL_SMOOTHING=0.0`
- `SELECTION_METRIC=val_metal_balanced_acc`
- `INCLUDE_HELD_OUT_TEST_DURING_TRAINING=False`
- `MAX_CONFIGURATION_RUNS=24` is sufficient because this grid is expected to
  produce only `5` runs.
- `SPLIT_BY=pdbid`
- `VAL_FRACTION=0.15`

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

- Next validation-only stage: tiny GVP + late-fusion stability check using the
  confirmed Only-ESM training settings.
- RING should be a later small side ablation, not mixed into the first
  ESM/fusion comparison.

## Caveats

- Exact hyperparameters and result numbers must be parsed from raw notebook
  outputs or saved configs before being used in a publication table.
- Round 2's copied output includes a notebook-selected best single seed. Treat
  that as within-batch checkpoint selection, not as a project-level anchor
  decision by itself.
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
