# Summary Run: Only-ESM Round 2 LR/WD/Class-Weight Screen

## Source File
`docs/notebook_outputs/raw/Only-ESM/Round2_ESMonly.output_cell_notebook.md` (renamed from `docs/notebook outputs/Only-ESM/Round2_ESMonly.output_cell_notebook.md`).

## Purpose
Screen Only-ESM learning rate, weight decay, and metal class-weight mode values using validation metrics.

## Configuration
- Task: metal
- Run mode: manual_configurations, validation-only
- Model preset / architecture: Only-ESM / only_esm
- Fusion mode: late_fusion appears in the completed-run table, but run names indicate fusionnone for only_esm
- ESM usage: ESM embeddings used
- HPO/Optuna settings: Manual grid, not Optuna
- Number of trials/runs: 24 completed; intended grid described as 36 runs
- Seeds: 42,123,2026
- Selection metric: val_metal_balanced_acc

## Best Result
- Best validation metric: val_metal_balanced_acc = 0.6930 shown as 0.692962 in the table
- Best trial/run: deepmzyme_nonoverlap_baseline_batchmetal_only_esm_weight_lr_wd_narrow_v1_metal_only_esm_archonly_esm_fusionnone_ringno_esmyes_mwinv_94d353ae
- Best hyperparameters: learning_rate=3e-5, weight_decay shown in top rows as 1e-5 or 1e-4 duplicates, batch_size=8, head_mlp_layers=2; class-weight mode for selected run is truncated in the copied table and not clearly available in source file
- Selected epoch/checkpoint if available: epoch 44

## Main Findings
- The best single completed validation run reached about 0.693 validation balanced accuracy.
- Only 24 completed directories were found.
- The copied output says no `test_report.json` existed for the selected run.

## Caveats
- The intended 36-run grid was not fully run; `EXPERIMENT_STATUS.md` notes the 5e-5 learning-rate rows are absent.
- Some selected-run details are truncated in the copied table.
- Held-out test metrics are absent.

## Recommended Next Step
Use Round 3 seed confirmation before replacing the original Only-ESM anchor.
