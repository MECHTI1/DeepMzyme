# Summary Run: Only-GVP Round 5 Trial 13 Batch

## Source File
`docs/notebook_outputs/raw/Only-GVP/round5_Trial_13_batch.output_cell_notebook`.

## Purpose
Run a 30-epoch validation-only batch for the Only-GVP Trial 13 configuration and a `gvp_layers=3` ablation.

## Configuration
- Task: metal
- Run mode: manual_configurations, validation-only
- Model preset / architecture: Only-GVP / only_gvp
- Fusion mode: none
- ESM usage: ESM disabled
- HPO/Optuna settings: Fixed HPO-derived Trial 13 candidate, not a new Optuna search
- Number of trials/runs: 10 planned configurations
- Seeds: Not clearly available in source file
- Selection metric: val_metal_balanced_acc

## Best Result
- Best validation metric: val_metal_balanced_acc = 0.6316031249930583
- Best trial/run: deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_round3_trial13_plus_gvp3_seedrepeat_2026_05_12_metal_only_gvp_archonly_gvp_fusion_22b2a9d2
- Best hyperparameters: learning_rate=6.817779343845317e-05, weight_decay=0.001, edge_radius=10.0, gvp_layers=2 or 3, metal_class_weight_mode=inverse_sqrt_frequency; other exact values are Not clearly available in source file
- Selected epoch/checkpoint if available: Not clearly available in source file

## Main Findings
- The best single Trial 13 batch run reached 0.6316031249930583 validation balanced accuracy.
- Trial 13 used a larger edge radius of 10.0 than the Trial 7/12 candidates.
- No held-out test result was present.

## Caveats
- Seeds and exact best-row layer count are not clearly available in the visible header.
- This is a 30-epoch validation-only batch.
- Held-out test metrics are absent.

## Recommended Next Step
Use Trial 13 as secondary evidence unless per-class diagnostics show a specific advantage over Trial 7 or Trial 12.
