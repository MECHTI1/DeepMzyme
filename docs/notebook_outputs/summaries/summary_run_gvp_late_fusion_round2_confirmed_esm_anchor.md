# Summary Run: GVP Late Fusion Round 2 Confirmed ESM Anchor

## Source File
`docs/notebook_outputs/raw/GVP + late fusion/Round2_late_fusion_from_confirmed_only_esm_anchor.output_cell_notebook.md`.

## Purpose
Run a validation-only GVP + late fusion check using the confirmed Only-ESM training settings as the anchor.

## Configuration
- Task: metal
- Run mode: manual_configurations, validation-only
- Model preset / architecture: GVP + late fusion / gvp
- Fusion mode: late_fusion
- ESM usage: ESM embeddings used
- HPO/Optuna settings: Fixed five-seed comparison, not Optuna
- Number of trials/runs: 5 planned configurations
- Seeds: 42,123,2026,43,44 inferred from run commands; header contains a stale Only-ESM seed line
- Selection metric: val_metal_balanced_acc

## Best Result
- Best validation metric: val_metal_balanced_acc = 0.6774914740431982
- Best trial/run: deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_from_confirmed_only_esm_anchor_v1_metal_gvp_+_late_fusion_archgvp_fusionlate_f_026c9c00
- Best hyperparameters: learning_rate=3e-5, weight_decay=1e-4, batch_size=8, hidden_s=128, hidden_v=16, edge_hidden=64, gvp_layers=4, edge_radius=8.0, head_mlp_layers=2, metal_class_weight_mode=inverse_frequency
- Selected epoch/checkpoint if available: Not clearly available in source file

## Main Findings
- The best single late-fusion run reached 0.6774914740431982 validation balanced accuracy.
- The output reports no held-out test results.
- The run aligns with the planned late-fusion follow-up from `EXPERIMENT_STATUS.md`.

## Caveats
- The first line appears stale and says `MODEL_PRESET = "Only-ESM"` with an Only-ESM batch id, while the scanned run and commands are GVP + late fusion.
- Validation-only evidence should not be used as final test performance.
- Held-out test metrics are absent.

## Recommended Next Step
Compare seed-repeat aggregate metrics for this late-fusion check against the confirmed Only-ESM anchor before any final test evaluation.
