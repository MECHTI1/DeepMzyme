# Summary Run: Only-ESM Round 1 Full Coverage

## Source File
`docs/notebook_outputs/raw/Only-ESM/Round1_Rerun validation-only Only-ESM on full ESM coverage.output_cell_notebook` (renamed from `docs/notebook outputs/Only-ESM/Round1_Rerun validation-only Only-ESM on full ESM coverage.output_cell_notebook`).

## Purpose
Validate the original full-coverage Only-ESM metal baseline across five seeds.

## Configuration
- Task: metal
- Run mode: manual_configurations, validation-only
- Model preset / architecture: Only-ESM / only_esm
- Fusion mode: none in the summary output
- ESM usage: ESM embeddings used
- HPO/Optuna settings: Not clearly available in source file
- Number of trials/runs: 5 planned configurations
- Seeds: 42,123,2026,43,44
- Selection metric: val_metal_balanced_acc

## Best Result
- Best validation metric: val_metal_balanced_acc = 0.6722436454687976
- Best trial/run: deepmzyme_nonoverlap_baseline_batchmetal_only_esm_full_coverage_50epoch_seedrepeat_metal_only_esm_archonly_esm_fusionnone_ringno_es_d215e1b6
- Best hyperparameters: learning_rate=3e-5, weight_decay=1e-4, batch_size=8, head_mlp_layers=2, metal_class_weight_mode=inverse_frequency, metal_loss_function=cross_entropy, metal_label_smoothing=0.0, epochs=50
- Selected epoch/checkpoint if available: Not clearly available in source file

## Main Findings
- The copied output reports a validation-only five-seed Only-ESM baseline.
- The best single run reached 0.6722436454687976 validation balanced accuracy.
- No held-out test result was present.

## Caveats
- This is validation-only evidence.
- The best single run is not the same as a seed-repeat aggregate decision.
- Held-out test metrics are absent.

## Recommended Next Step
Use this as the original Only-ESM anchor evidence and compare against the Round 2 and Round 3 validation summaries before any held-out test evaluation.
