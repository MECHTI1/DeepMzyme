# Summary Run: Only-GVP Round 1 Optuna HPO

## Source File
`docs/notebook_outputs/raw/Only-GVP/round1_results_onlyGVP_Optuna.output_cell_notebook` (renamed from `docs/notebook outputs/Only-GVP/round1_results_onlyGVP_Optuna.output_cell_notebook`).

## Purpose
Run an initial controlled Optuna HPO search within the Only-GVP model family.

## Configuration
- Task: metal
- Run mode: controlled Optuna HPO plus completed-run summary scan
- Model preset / architecture: Only-GVP / only_gvp
- Fusion mode: none in the Optuna base configuration; summary output also prints late_fusion
- ESM usage: ESM disabled
- HPO/Optuna settings: in-memory study `deepmzyme_controlled_hpo`, 16 trials, 20 trial epochs, fixed seed 42, split_by=pdbid, val_fraction=0.15
- Number of trials/runs: 16 completed HPO trials plus one planned row in the summary table
- Seeds: fixed HPO split/model seed 42
- Selection metric: val_metal_balanced_acc

## Best Result
- Best validation metric: val_metal_balanced_acc = 0.554291323653437
- Best trial/run: Optuna trial 7, `optuna_deepmzyme_controlled_hpo_trial0007_deepmzyme_nonoverlap_baseline_metal_only_gvp_archonly_gvp_fusionnone_ringno_esmno_mwinverse_sqrt_fr_254d3669`
- Best hyperparameters: learning_rate=6.464669746492395e-05, weight_decay=0.001, batch_size=8, hidden_s=128, head_mlp_layers=1, edge_hidden=128, gvp_layers=4, edge_radius=6.0, hidden_v=32, metal_class_weight_mode=inverse_sqrt_frequency
- Selected epoch/checkpoint if available: Not clearly available in source file

## Main Findings
- Trial 7 was the best initial HPO trial.
- The output recommends top-K seed-repeat validation after HPO.
- No held-out test result was present.

## Caveats
- The raw output warns that the summary may mix old runs from the same RUNS_ROOT.
- Optuna storage was in-memory and described as temporary/debug only.
- The summary output's `Top fusion mode: late_fusion` conflicts with the Only-GVP base configuration showing fusion none.

## Recommended Next Step
Use the best HPO candidates only as candidates for seed-repeat validation, not as final model selections.
