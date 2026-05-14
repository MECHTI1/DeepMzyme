# Summary Run: Only-GVP Round 5 Trial 12 Batch

## Source File
`docs/notebook_outputs/raw/Only-GVP/round5_Trial_12_batch.output_cell_notebook` (renamed from `docs/notebook outputs/Only-GVP/round5_Trial_12_batch.output_cell_notebook`).

## Purpose
Run a 30-epoch validation-only batch for the Only-GVP Trial 12 configuration and a `gvp_layers=3` ablation.

## Configuration
- Task: metal
- Run mode: manual_configurations, validation-only
- Model preset / architecture: Only-GVP / only_gvp
- Fusion mode: none
- ESM usage: ESM disabled
- HPO/Optuna settings: Fixed HPO-derived Trial 12 candidate, not a new Optuna search
- Number of trials/runs: 10 planned configurations
- Seeds: 42,123,2026,43,44
- Selection metric: val_metal_balanced_acc

## Best Result
- Best validation metric: val_metal_balanced_acc = 0.6215952953717157
- Best trial/run: deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_round3_trial12_plus_gvp3_seedrepeat_2026_05_12_metal_only_gvp_archonly_gvp_fusion_e42ac546
- Best hyperparameters: learning_rate=4.735385769610685e-05, weight_decay=0.0, edge_radius=6.0, gvp_layers=2 or 3, hidden_s=128, hidden_v=32, edge_hidden=128, head_mlp_layers=1, metal_class_weight_mode=inverse_sqrt_frequency
- Selected epoch/checkpoint if available: Not clearly available in source file

## Main Findings
- The best single Trial 12 batch run reached 0.6215952953717157 validation balanced accuracy.
- The batch compares base and GVP3 ablation settings over the five listed seeds.
- No held-out test result was present.

## Caveats
- The best row does not clearly expose whether it was the base `gvp_layers=2` or ablation `gvp_layers=3` run.
- This is a 30-epoch validation-only batch.
- Held-out test metrics are absent.

## Recommended Next Step
Use the Trial 12 evidence with Round 6 aggregate comparisons to decide whether Trial 12's stability justifies choosing it over Trial 7.
