# Metal Only-GVP Round 3 Next Batch Plan

> Status note, 2026-05-13: this plan is historical context. The current copied
> evidence includes a newer Round 6 finalist comparison in
> `round6_three_Trials_comparisons.output_cell_notebook.md`. Use
> `EXPERIMENT_STATUS.md` for the current recommendation before launching any new
> notebook runs.

## Objective
Following the Optuna HPO from Round 2, we have identified three top-performing Only-GVP trial configurations (Trials 7, 12, and 13). We will evaluate these 3 original configs plus a narrow `gvp_layers=3` ablation for each, across 5 fixed seeds. This results in exactly 30 validation-only runs to determine the most stable Only-GVP anchor.

## Execution Status - 2026-05-12

Evidence files:

- `docs/notebook outputs/round3_results_onlyGVP_Optuna.output_cell_notebook`
- `docs/notebook outputs/round4_results_onlyGVP_Optuna.output_cell_notebook`
- `docs/notebook outputs/round5_Trial_12_batch.output_cell_notebook`
- `docs/notebook outputs/round5_Trial_13_batch.output_cell_notebook`

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

Current interpretation: the 50-epoch batch is more promising than the 30-epoch
batch. Trial 7 with `gvp_layers=4` has the highest 50-epoch mean validation
balanced accuracy, but Trial 12 with `gvp_layers=3` is almost tied and more
stable. Trial 12 with `gvp_layers=2` has the best 50-epoch worst-seed/min value.
The final Only-GVP anchor should be chosen after checking per-class recall,
macro-F1, and min-recall diagnostics, not from a single best seed.

Next action: produce a config-level diagnostic table for the completed 50-epoch
runs, then select the stable Only-GVP anchor using validation metrics only. Use
the 30-epoch batch as supporting evidence. Run the held-out test only after the
anchor is fixed.

## Extracted Hyperparameters
From `docs/notebook outputs/round2_results_onlyGVP_Optuna.output_cell_notebook`:

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
For each trial, we run the base config and a `gvp_layers=3` ablation. All use 5 seeds: `42, 123, 2026, 43, 44`.

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
Due to the Cartesian product nature of the `manual_configurations` mode, running this exactly as 30 runs without combinatorial explosion requires either launching them in 3 separate batches or adding a custom `RECOMMENDED_RUN_SET` in the notebook code.

To run them cleanly as batches, use the following settings for the 3 sequential notebook executions:

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
