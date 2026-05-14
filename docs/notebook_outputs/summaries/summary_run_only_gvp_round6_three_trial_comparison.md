# Summary Run: Only-GVP Round 6 Three-Trial Comparison

## Source File
`docs/notebook_outputs/raw/Only-GVP/round6_three_Trials_comparisons.output_cell_notebook.md` (renamed from `docs/notebook outputs/Only-GVP/round6_three_Trials_comparisons.output_cell_notebook.md`).

## Purpose
Compare three finalist Only-GVP validation-only configurations: Trial12 GVP3, Trial7 GVP4, and Trial12 GVP2.

## Configuration
- Task: metal
- Run mode: manual_configurations, validation-only
- Model preset / architecture: Only-GVP / only_gvp
- Fusion mode: none
- ESM usage: ESM disabled
- HPO/Optuna settings: Fixed HPO-derived finalist comparison, not a new Optuna search
- Number of trials/runs: 15 expected from three five-seed batches
- Seeds: 42,123,2026,43,44
- Selection metric: val_metal_balanced_acc

## Best Result
- Best validation metric: val_metal_balanced_acc = 0.6559072690667597 for the Trial7 GVP4 detailed block
- Best trial/run: deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_ring_4db78203
- Best hyperparameters: Trial7 GVP4: learning_rate=6.464669746492395e-05, weight_decay=0.001, edge_radius=6.0, gvp_layers=4, hidden_s=128, hidden_v=32, edge_hidden=128, head_mlp_layers=1, metal_class_weight_mode=inverse_sqrt_frequency
- Selected epoch/checkpoint if available: Not clearly available in source file

## Main Findings
- The best single validation result came from Trial7 GVP4.
- Existing short notes report Trial7 GVP4 as highest 50-epoch mean, while Trial12 GVP3 is nearly tied and more stable.
- No held-out test result was present.

## Caveats
- This summary highlights the best single run; final selection should also inspect mean, variance, min recall, macro-F1, and per-class recall.
- The source contains multiple detailed result sections, so configuration-level aggregation is easier to read in the existing decision notes.
- Held-out test metrics are absent.

## Recommended Next Step
Select an Only-GVP anchor using validation aggregate and per-class diagnostics, then evaluate the fixed anchor on the held-out test set only once if final reporting is needed.
