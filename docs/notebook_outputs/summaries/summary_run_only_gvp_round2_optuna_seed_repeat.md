# Summary Run: Only-GVP Round 2 Optuna Seed Repeat

## Source File
`docs/notebook_outputs/raw/Only-GVP/round2_results_onlyGVP_Optuna.output_cell_notebook` (renamed from `docs/notebook outputs/Only-GVP/round2_results_onlyGVP_Optuna.output_cell_notebook`).

## Purpose
Continue the Only-GVP Optuna workflow and launch seed-repeat validation for top HPO candidates.

## Configuration
- Task: metal
- Run mode: controlled Optuna HPO plus seed-repeat validation-only runs
- Model preset / architecture: Only-GVP / only_gvp
- Fusion mode: none in selected seed-repeat config; summary output also prints late_fusion
- ESM usage: ESM disabled
- HPO/Optuna settings: in-memory study, 16 trials, 20 trial epochs, normal/final retrain epochs=50
- Number of trials/runs: 16 HPO trials plus top-candidate seed-repeat runs; summary table shows 27 rows including one planned row
- Seeds: 42,123,2026 for selected seed-repeat reruns shown in source
- Selection metric: val_metal_balanced_acc

## Best Result
- Best validation metric: val_metal_balanced_acc = 0.64772364969639 in the completed-run summary
- Best trial/run: top3_trial7_deepmzyme_controlled_hpo_seed2026_deepmzyme_nonoverlap_baseline_metal_only_gvp_archonly_gvp_fusionnone_ringno_esmno_mwinverse_sqr_a03e8029
- Best hyperparameters: source Trial 7 configuration: learning_rate=6.464669746492395e-05, weight_decay=0.001, batch_size=8, hidden_s=128, head_mlp_layers=1, edge_hidden=128, gvp_layers=4, edge_radius=6.0, hidden_v=32, metal_class_weight_mode=inverse_sqrt_frequency
- Selected epoch/checkpoint if available: Not clearly available in source file

## Main Findings
- The best single seed-repeat validation run came from the Trial 7 configuration.
- The output also reports HPO best trial number 13 later in the file, which differs from the top completed seed-repeat run.
- No held-out test result was present.

## Caveats
- The raw output warns that the summary may mix old runs from the same RUNS_ROOT.
- HPO best-trial reporting and seed-repeat best-run reporting are not the same decision.
- The summary output's `Top fusion mode: late_fusion` conflicts with the Only-GVP no-ESM configuration.

## Recommended Next Step
Use Round 3/6 candidate comparisons to choose a stable Only-GVP anchor by validation aggregates and per-class diagnostics.
