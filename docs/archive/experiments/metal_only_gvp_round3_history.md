# Metal Only-GVP Round 3 History

> Status note, 2026-05-13: this file is consolidated historical context. It
> merges the earlier "next batch plan" and "decision next steps" notes for
> Round 3. Both notes predate the newer Round 6 finalist comparison in
> `docs/notebook_outputs/raw/Only-GVP/round6_three_Trials_comparisons.output_cell_notebook.md`.
> Use `EXPERIMENT_STATUS.md` for the current recommendation before launching
> any new notebook runs. `Plan.md` remains the design authority.

## Objective

Following the Optuna HPO from Round 2, three top-performing Only-GVP trial
configurations were identified (Trials 7, 12, and 13). Round 3 evaluated these
3 original configs plus a narrow `gvp_layers=3` ablation for each, across 5
fixed seeds. This resulted in exactly 30 validation-only runs to determine the
most stable Only-GVP anchor.

## Extracted Hyperparameters

From `docs/notebook_outputs/raw/Only-GVP/round2_results_onlyGVP_Optuna.output_cell_notebook`:

### Trial 7
- `learning_rate`: 6.464669746492395e-05
- `weight_decay`: 0.001
- `batch_size`: 8
- `hidden_s`: 128
- `head_mlp_layers`: 1
- `edge_hidden`: 128
- `gvp_layers`: 4
- `edge_radius`: 6.0
- `hidden_v`: 32
- `metal_class_weight_mode`: 'inverse_sqrt_frequency'

### Trial 12
- `learning_rate`: 4.735385769610685e-05
- `weight_decay`: 0.0
- `batch_size`: 8
- `hidden_s`: 128
- `head_mlp_layers`: 1
- `edge_hidden`: 128
- `gvp_layers`: 2
- `edge_radius`: 6.0
- `hidden_v`: 32
- `metal_class_weight_mode`: 'inverse_sqrt_frequency'

### Trial 13
- `learning_rate`: 6.817779343845317e-05
- `weight_decay`: 0.001
- `batch_size`: 8
- `hidden_s`: 128
- `head_mlp_layers`: 1
- `edge_hidden`: 128
- `gvp_layers`: 2
- `edge_radius`: 10.0
- `hidden_v`: 32
- `metal_class_weight_mode`: 'inverse_sqrt_frequency'

## Target Configurations (30 Runs Total)

For each trial, both the base config and a `gvp_layers=3` ablation are run.
All use 5 seeds: `42, 123, 2026, 43, 44`.

| Base Trial | lr | wd | gvp_layers | edge_radius | Target runs |
|---|---|---|---|---|---|
| Trial 7 Base | 6.46e-05 | 0.001 | 4 | 6.0 | 5 seeds |
| Trial 7 Ablation | 6.46e-05 | 0.001 | 3 | 6.0 | 5 seeds |
| Trial 12 Base | 4.74e-05 | 0.0 | 2 | 6.0 | 5 seeds |
| Trial 12 Ablation | 4.74e-05 | 0.0 | 3 | 6.0 | 5 seeds |
| Trial 13 Base | 6.82e-05 | 0.001 | 2 | 10.0 | 5 seeds |
| Trial 13 Ablation | 6.82e-05 | 0.001 | 3 | 10.0 | 5 seeds |

*Shared settings across all 30 runs:*
- `batch_size`: 8
- `hidden_s`: 128, `hidden_v`: 32, `edge_hidden`: 128
- `head_mlp_layers`: 1
- `metal_class_weight_mode`: 'inverse_sqrt_frequency'
- `TASK`: 'metal'
- `MODEL_PRESET`: 'Only-GVP'
- No ESM, No RING
- `SPLIT_BY`: 'pdbid'
- `VAL_FRACTION`: 0.15

## Colab Widget Settings

Due to the Cartesian product nature of the `manual_configurations` mode,
running this exactly as 30 runs without combinatorial explosion requires either
launching them in 3 separate batches or adding a custom `RECOMMENDED_RUN_SET`
in the notebook code.

To run them cleanly as batches, use the following settings for the 3 sequential
notebook executions:

### Common Settings
- `TASK` = "metal"
- `RUN_MODE` = "manual_configurations"
- `RECOMMENDED_RUN_SET` = "custom"
- `MODEL_PRESET` = "Only-GVP"
- `MAX_CONFIGURATION_RUNS` = 24
- `BATCH_SIZES_CSV` = "8"
- `SEEDS_CSV` = "42,123,2026,43,44"
- `METAL_CLASS_WEIGHT_MODES_CSV` = "inverse_sqrt_frequency"
- `ADVANCED_MODE` = True (to expose arch controls)
- `HIDDEN_S_VALUES_CSV` = "128"
- `HIDDEN_V_VALUES_CSV` = "32"
- `EDGE_HIDDEN_VALUES_CSV` = "128"
- `HEAD_MLP_LAYERS_VALUES_CSV` = "1"

### Execution 1: Trial 7 & Ablation (10 runs)
- `RUN_BATCH_ID` = "metal_only_gvp_round2_trial7_ablation"
- `LEARNING_RATES_CSV` = "6.46e-05"
- `WEIGHT_DECAYS_CSV` = "0.001"
- `EDGE_RADIUS_VALUES_CSV` = "6.0"
- `GVP_LAYERS_VALUES_CSV` = "3,4"

### Execution 2: Trial 12 & Ablation (10 runs)
- `RUN_BATCH_ID` = "metal_only_gvp_round2_trial12_ablation"
- `LEARNING_RATES_CSV` = "4.74e-05"
- `WEIGHT_DECAYS_CSV` = "0.0"
- `EDGE_RADIUS_VALUES_CSV` = "6.0"
- `GVP_LAYERS_VALUES_CSV` = "2,3"

### Execution 3: Trial 13 & Ablation (10 runs)
- `RUN_BATCH_ID` = "metal_only_gvp_round2_trial13_ablation"
- `LEARNING_RATES_CSV` = "6.82e-05"
- `WEIGHT_DECAYS_CSV` = "0.001"
- `EDGE_RADIUS_VALUES_CSV` = "10.0"
- `GVP_LAYERS_VALUES_CSV` = "2,3"

## Execution Status (2026-05-12)

Evidence files:

- `docs/notebook_outputs/raw/Only-GVP/round3_results_onlyGVP_Optuna.output_cell_notebook`
- `docs/notebook_outputs/raw/Only-GVP/round4_results_onlyGVP_Optuna.output_cell_notebook`
- `docs/notebook_outputs/raw/Only-GVP/round5_Trial_12_batch.output_cell_notebook`
- `docs/notebook_outputs/raw/Only-GVP/round5_Trial_13_batch.output_cell_notebook`

The copied outputs evidence two related validation-only comparisons across
seeds `42,123,2026,43,44`: a 50-epoch confirmation batch in the round3 output
and a later 30-epoch split batch in the round4/round5 outputs. Held-out test
evaluation was not present in these copied outputs, which is correct for model
selection.

50-epoch validation-balanced-accuracy summary from the round3 output:

| Config | Runs | Mean val balanced acc | Sample std | Min | Max |
|---|---:|---:|---:|---:|---:|
| Trial 7 base, `gvp_layers=4`, radius `6.0` | 5 | 0.6074 | 0.0424 | 0.5584 | 0.6559 |
| Trial 12 ablation, `gvp_layers=3`, radius `6.0` | 5 | 0.6071 | 0.0224 | 0.5671 | 0.6184 |
| Trial 7 ablation, `gvp_layers=3`, radius `6.0` | 5 | 0.6010 | 0.0273 | 0.5560 | 0.6302 |
| Trial 12 base, `gvp_layers=2`, radius `6.0` | 5 | 0.5986 | 0.0204 | 0.5785 | 0.6243 |
| Trial 13 base, `gvp_layers=2`, radius `10.0` | 5 | 0.5960 | 0.0257 | 0.5704 | 0.6316 |
| Trial 13 ablation, `gvp_layers=3`, radius `10.0` | 5 | 0.5809 | 0.0287 | 0.5488 | 0.6211 |

30-epoch validation-balanced-accuracy summary from the round4/round5 outputs:

| Config | Runs | Mean val balanced acc | Sample std | Min | Max |
|---|---:|---:|---:|---:|---:|
| Trial 12 ablation, `gvp_layers=3`, radius `6.0` | 5 | 0.6005 | 0.0301 | 0.5484 | 0.6216 |
| Trial 7 ablation, `gvp_layers=3`, radius `6.0` | 5 | 0.5935 | 0.0230 | 0.5560 | 0.6106 |
| Trial 7 base, `gvp_layers=4`, radius `6.0` | 5 | 0.5909 | 0.0239 | 0.5584 | 0.6240 |
| Trial 13 base, `gvp_layers=2`, radius `10.0` | 5 | 0.5908 | 0.0237 | 0.5704 | 0.6316 |
| Trial 12 base, `gvp_layers=2`, radius `6.0` | 5 | 0.5792 | 0.0150 | 0.5612 | 0.5985 |
| Trial 13 ablation, `gvp_layers=3`, radius `10.0` | 5 | 0.5673 | 0.0339 | 0.5374 | 0.6243 |

Interpretation at the time: the 50-epoch batch is more promising than the
30-epoch batch. Trial 7 with `gvp_layers=4` has the highest 50-epoch mean
validation balanced accuracy, but Trial 12 with `gvp_layers=3` is almost tied
and more stable. Trial 12 with `gvp_layers=2` has the best 50-epoch
worst-seed/min value. The final Only-GVP anchor should be chosen after checking
per-class recall, macro-F1, and min-recall diagnostics, not from a single best
seed.

## Evidence To Trust First

Use the 50-epoch confirmation batch from:

- `docs/notebook_outputs/raw/Only-GVP/round3_results_onlyGVP_Optuna.output_cell_notebook`

Use the 30-epoch split batch from round4/round5 only as supporting evidence,
because it has a different epoch budget and lower aggregate validation metrics.

Do not use held-out test results for this decision.

## Candidate Ranking

| Candidate | Mean val balanced acc | Std | Min | Interpretation |
|---|---:|---:|---:|---|
| Trial7, `gvp_layers=4`, radius `6.0` | 0.6074 | 0.0424 | 0.5584 | Highest mean and best single run, but high variance. |
| Trial12, `gvp_layers=3`, radius `6.0` | 0.6071 | 0.0224 | 0.5671 | Nearly tied mean, more stable. Inspect first. |
| Trial12, `gvp_layers=2`, radius `6.0` | 0.5986 | 0.0204 | 0.5785 | Lower mean, best worst-seed robustness. |

Trial7 `gvp_layers=3` and Trial13 configs are secondary unless their per-class
diagnostics reveal a specific advantage.

## Decision Rule

1. Build a config-level diagnostic table for the 50-epoch batch with:
   validation balanced accuracy, macro-F1, min recall, per-class recall, and seed.
2. Prefer Trial12 `gvp_layers=3` if its rare-metal recall and macro-F1 are not
   clearly worse than Trial7 `gvp_layers=4`.
3. Pick Trial7 `gvp_layers=4` only if per-class diagnostics show a meaningful
   improvement that justifies the higher seed variance.
4. Pick Trial12 `gvp_layers=2` only if worst-seed or rare-class robustness is
   the main priority.

## After Anchor Selection

After the Only-GVP anchor is fixed by validation metrics:

1. Run held-out test evaluation once for final reporting of that fixed anchor.
2. Record the selected anchor and held-out metrics in `EXPERIMENT_STATUS.md`.
3. Then move to the next baseline-first stage: Only-ESM, then GVP + late fusion.
4. Keep RING as a later side ablation, not part of the first ESM/fusion stage.
