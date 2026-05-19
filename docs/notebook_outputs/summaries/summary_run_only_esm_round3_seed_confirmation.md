# Summary Run: Only-ESM Round 3 Seed Confirmation

## Source File
`docs/notebook_outputs/raw/Only-ESM/Round3_ESMonly_add_seeds43_44_5seed_confirmation.output_cell_notebook.md`.

## Purpose
Add seeds 43 and 44 for Round 2 finalist Only-ESM settings to confirm whether the apparent winner is stable.

## Configuration
- Task: metal
- Run mode: manual_configurations, validation-only
- Model preset / architecture: Only-ESM / only_esm
- Fusion mode: none
- ESM usage: ESM embeddings used
- HPO/Optuna settings: Manual confirmation grid, not Optuna
- Number of trials/runs: 8 planned configurations
- Seeds: 43,44
- Selection metric: val_metal_balanced_acc

## Best Result
- Best validation metric: val_metal_balanced_acc = 0.6183848812144958 within this copied Round 3 output
- Best trial/run: deepmzyme_nonoverlap_baseline_batchmetal_only_esm_round2_confirm_5seed_metal_only_esm_archonly_esm_fusionnone_ringno_esmyes_mwinver_37e499fb
- Best hyperparameters: learning_rate=3e-5 or 2e-5 grid, weight_decay=1e-4, class_weight_mode=inverse_sqrt_frequency or inverse_frequency, head_mlp_layers=2; exact best row fields beyond the run name are Not clearly available in source file
- Selected epoch/checkpoint if available: Not clearly available in source file

## Main Findings
- This output adds seed coverage for the Round 2 finalists.
- The project status note combines this evidence with Round 2 and keeps the original 3e-5 + inverse_frequency Only-ESM anchor.
- No held-out test result was present.

## Caveats
- This raw output alone covers only seeds 43 and 44; combined 5-seed conclusions require Round 2 plus Round 3.
- The first-line configuration is dense and the best-row hyperparameters are not fully visible.
- Held-out test metrics are absent.

## Recommended Next Step
Use the confirmed Only-ESM anchor as the baseline for the next validation-only GVP + late-fusion comparison.
