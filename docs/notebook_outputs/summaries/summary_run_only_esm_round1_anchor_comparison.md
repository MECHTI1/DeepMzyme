# Summary Run: Only-ESM Round 1 Anchor Comparison

## Source File
`docs/notebook_outputs/raw/Only-ESM/Round1_results_only_esm_Optuna.output_cell_notebook`.

## Purpose
Record a validation-only Only-ESM anchor comparison batch on full ESM coverage.

## Configuration
- Task: metal
- Run mode: manual_configurations, validation-only
- Model preset / architecture: Only-ESM / only_esm
- Fusion mode: none in the summary output
- ESM usage: ESM embeddings used
- HPO/Optuna settings: File name says Optuna, but the visible configuration is a manual five-seed anchor comparison
- Number of trials/runs: 5 planned configurations
- Seeds: 42,123,2026,43,44
- Selection metric: val_metal_balanced_acc

## Best Result
- Best validation metric: val_metal_balanced_acc = 0.6722436454687976
- Best trial/run: deepmzyme_nonoverlap_baseline_batchmetal_only_esm_anchor_comparison_validation_50epoch_seedrepeat_metal_only_esm_archonly_esm_fusio_4b6a2856
- Best hyperparameters: learning_rate=3e-5, batch_size=8 if memory stable otherwise 4, epochs=50; other exact values are Not clearly available in source file
- Selected epoch/checkpoint if available: Not clearly available in source file

## Main Findings
- The best validation value matches the full-coverage Only-ESM rerun evidence.
- The output reinforces the same validation-only Only-ESM anchor.
- No held-out test result was present.

## Caveats
- Some configuration fields are described as notebook guidance rather than exact executed values.
- File naming mentions Optuna, but the visible run mode is manual_configurations.
- Held-out test metrics are absent.

## Recommended Next Step
Treat this as supporting evidence for the Only-ESM anchor, not as a separate HPO result.
