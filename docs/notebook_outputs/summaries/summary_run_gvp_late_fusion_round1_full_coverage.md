# Summary Run: GVP Late Fusion Round 1 Full Coverage

## Source File
`docs/notebook_outputs/raw/GVP + late fusion/Round1_Rerun validation-only GVP + late fusion on full ESM coverage.output_cell_notebook` (renamed from `docs/notebook outputs/GVP + late fusion/Round1_Rerun validation-only GVP + late fusion on full ESM coverage.output_cell_notebook`).

## Purpose
Rerun the GVP + late fusion candidate on full ESM coverage as a validation-only five-seed comparison.

## Configuration
- Task: metal
- Run mode: manual_configurations, validation-only
- Model preset / architecture: GVP + late fusion / gvp
- Fusion mode: late_fusion
- ESM usage: ESM embeddings used
- HPO/Optuna settings: Fixed candidate; no Optuna settings visible
- Number of trials/runs: 5 planned configurations
- Seeds: Not clearly available in source file
- Selection metric: val_metal_balanced_acc

## Best Result
- Best validation metric: val_metal_balanced_acc = 0.6817577959565789
- Best trial/run: deepmzyme_nonoverlap_baseline_batchmetal_gvp_late_fusion_full_coverage_trial12_gvp3_50epoch_seedrepeat_metal_gvp_+_late_fusion_arch_e3c73340
- Best hyperparameters: learning_rate=4.752317377508605e-05, weight_decay=0.0, edge_radius=6.0, gvp_layers=3, hidden_s=128, hidden_v=32, edge_hidden=128, head_mlp_layers=1, metal_class_weight_mode=inverse_sqrt_frequency
- Selected epoch/checkpoint if available: Not clearly available in source file

## Main Findings
- Results match the Trial12/GVP3 late-fusion anchor output.
- The best single validation run reached 0.6817577959565789 balanced accuracy.
- No held-out test result was present.

## Caveats
- Validation-only status limits conclusions.
- Seeds are visible in run commands but not summarized in the header.
- Held-out test metrics are absent.

## Recommended Next Step
Use this as supporting late-fusion evidence and compare against Only-ESM with the same validation-first policy.
