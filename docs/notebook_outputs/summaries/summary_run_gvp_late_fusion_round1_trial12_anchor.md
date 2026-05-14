# Summary Run: GVP Late Fusion Round 1 Trial12 Anchor

## Source File
`docs/notebook_outputs/raw/GVP + late fusion/Round1_results_gvp_plus_latefusion_Optuna.output_cell_notebook` (renamed from `docs/notebook outputs/GVP + late fusion/Round1_results_gvp_plus_latefusion_Optuna.output_cell_notebook`).

## Purpose
Evaluate a GVP + late ESM fusion configuration derived from an Only-GVP Trial 12/GVP3 anchor across five validation-only seeds.

## Configuration
- Task: metal
- Run mode: manual_configurations, validation-only
- Model preset / architecture: GVP + late fusion / gvp
- Fusion mode: late_fusion
- ESM usage: ESM embeddings used
- HPO/Optuna settings: Not an Optuna search in this file; uses a fixed candidate configuration
- Number of trials/runs: 5 planned configurations
- Seeds: Not clearly available in source file
- Selection metric: val_metal_balanced_acc

## Best Result
- Best validation metric: val_metal_balanced_acc = 0.6817577959565789
- Best trial/run: deepmzyme_nonoverlap_baseline_batchmetal_gvp_late_fusion_trial12_gvp3_anchor_validation_50epoch_seedrepeat_metal_gvp_+_late_fusion_c7bf6e6e
- Best hyperparameters: learning_rate=4.752317377508605e-05, weight_decay=0.0, edge_radius=6.0, gvp_layers=3, hidden_s=128, hidden_v=32, edge_hidden=128, head_mlp_layers=1, metal_class_weight_mode=inverse_sqrt_frequency
- Selected epoch/checkpoint if available: Not clearly available in source file

## Main Findings
- The best single validation run reached 0.6817577959565789 balanced accuracy.
- The summary reports late_fusion as the top fusion mode.
- No held-out test result was present.

## Caveats
- The file name mentions Optuna, but the visible output is a fixed candidate seed-repeat run.
- This is validation-only evidence.
- Held-out test metrics are absent.

## Recommended Next Step
Compare this fixed late-fusion candidate against the confirmed Only-ESM anchor using seed-repeat validation aggregates, not a single best run.
