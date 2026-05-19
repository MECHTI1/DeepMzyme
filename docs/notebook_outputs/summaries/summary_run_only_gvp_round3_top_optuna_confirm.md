# Summary Run: Only-GVP Round 3 Top Optuna Confirmation

## Source File
`docs/notebook_outputs/raw/Only-GVP/round3_results_onlyGVP_Optuna.output_cell_notebook`.

## Purpose
Confirm top Only-GVP Optuna candidates and ablations across validation-only seed-repeat runs.

## Configuration
- Task: metal
- Run mode: manual seed-repeat validation-only comparison after Optuna
- Model preset / architecture: Only-GVP / only_gvp
- Fusion mode: none in run names; summary output also prints late_fusion
- ESM usage: ESM disabled
- HPO/Optuna settings: Uses top HPO-derived candidates; not a new Optuna search in the visible summary
- Number of trials/runs: 30 planned configurations
- Seeds: Not clearly available in source file
- Selection metric: val_metal_balanced_acc

## Best Result
- Best validation metric: val_metal_balanced_acc = 0.6559072690667597
- Best trial/run: deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_top_optuna_confirm_layer3_2026_05_12_metal_only_gvp_archonly_gvp_fusionnone_ringn_d1ea4f0f
- Best hyperparameters: HPO-derived Only-GVP candidate; exact full best row is not clearly available in source file
- Selected epoch/checkpoint if available: Not clearly available in source file

## Main Findings
- The best single validation run reached 0.6559072690667597.
- The output planned a 30-run validation-only comparison of top candidates.
- Later short notes summarize this round as favoring Trial 7 `gvp_layers=4` by mean but with stability caveats.

## Caveats
- The raw output warns about mixed or missing RUN_BATCH_ID values, including `debug_smoke`.
- Treat single-run best values cautiously; aggregate validation diagnostics are needed.
- Held-out test metrics are absent.

## Recommended Next Step
Read `summary_run_only_gvp_round6_three_trial_comparison.md` and the existing Only-GVP decision notes before selecting an Only-GVP anchor.
