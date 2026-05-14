# Summary Run: Only-GVP Round 4 Top3 Plus GVP3

## Source File
`docs/notebook_outputs/raw/Only-GVP/round4_results_onlyGVP_Optuna.output_cell_notebook` (renamed from `docs/notebook outputs/Only-GVP/round4_results_onlyGVP_Optuna.output_cell_notebook`).

## Purpose
Evaluate a 30-epoch split batch for top Only-GVP candidates and GVP3 ablations.

## Configuration
- Task: metal
- Run mode: manual_configurations, validation-only
- Model preset / architecture: Only-GVP / only_gvp
- Fusion mode: none
- ESM usage: ESM disabled
- HPO/Optuna settings: HPO-derived fixed candidates, not a new Optuna search
- Number of trials/runs: 10 completed rows in the visible summary table
- Seeds: Not clearly available in source file
- Selection metric: val_metal_balanced_acc

## Best Result
- Best validation metric: val_metal_balanced_acc = 0.6239786072103145
- Best trial/run: metal_only_gvp_round3_trial7_plus_gvp3_batchmetal_only_gvp_round3_top3_plus_gvp3_seedrepeat_2026_05_12_metal_only_gvp_archonly_gvp_d45d79f4
- Best hyperparameters: learning_rate=6.464669746492395e-05, weight_decay=0.001, edge_radius=6.0, gvp_layers=3 or 4 in this split batch; exact best row layer count is not clearly available in source file
- Selected epoch/checkpoint if available: Not clearly available in source file

## Main Findings
- The best single validation run reached 0.6239786072103145.
- The output is a validation-only follow-up to the top HPO candidates.
- No held-out test result was present.

## Caveats
- This appears to be a 30-epoch split batch, not the final 50-epoch comparison.
- Exact best-row hyperparameters are partly inferred from the run family and commands.
- Held-out test metrics are absent.

## Recommended Next Step
Use this as supporting evidence only; prefer the 50-epoch Round 6/decision-note evidence for anchor selection.
