# Summary Run: GVP Node-Level Late Fusion Round 1 From Late-Fusion Trial 49 Seed Repeat 50 Epoch

## Source

Raw copied notebook output:

`docs/notebook_outputs/raw/GVP + node-level late fusion/Round1_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1.output_cell_notebook.md`

## Purpose

Validation-only seed repeat for **GVP + node-level late fusion**, initialized from the selected GVP + late-fusion Optuna trial 49 anchor settings.

This run tests whether moving from graph-level late fusion to node-level late fusion improves validation performance before any held-out test use.

## Run Setup

- Task: `metal`
- Model preset: `GVP + node-level late fusion`
- Batch id: `metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1`
- Run mode: `manual_configurations`
- Planned runs: 5
- Completed runs: 5
- Failed runs: none
- Epochs per run: 50
- Seeds: `42, 123, 2026, 43, 44`
- Validation fraction: `0.15`
- Split: `pdbid`
- Selection metric: `val_metal_balanced_acc`
- Held-out test during training: disabled
- Held-out test results present in copied output: false

## Fixed Hyperparameters

These are the late-fusion trial 49 anchor hyperparameters reused for node-level late fusion:

- learning_rate: `1.6801503587890522e-05`
- weight_decay: `1e-05`
- batch_size: `8`
- hidden_s: `256`
- hidden_v: `32`
- gvp_layers: `4`
- edge_hidden: `128`
- edge_radius: `6.0`
- head_mlp_layers: `1`
- esm_fusion_dim: `64`
- metal_class_weight_mode: `inverse_frequency`
- metal_loss_function: `cross_entropy`
- metal_label_smoothing: `0.0`
- lr_schedule: `fixed`
- ring_edge_mode: `without_ring`

## Per-Seed Validation Results

| Seed | Run suffix | val_metal_balanced_acc |
|---:|---|---:|
| 42 | `539ea463` | 0.614296708984 |
| 123 | `a7759f1f` | 0.619760953849 |
| 2026 | `da157306` | 0.633163185699 |
| 43 | `fcc23cc2` | 0.590901972342 |
| 44 | `c8a9d9c9` | 0.574873163235 |

## Aggregate Validation Result

| Metric | Value |
|---|---:|
| mean val_metal_balanced_acc | 0.606599196822 |
| sample std | 0.023404449951 |
| min | 0.574873163235 |
| max | 0.633163185699 |
| n | 5 |

## Comparison

Confirmed GVP + late-fusion trial 49 anchor:

- mean: approximately `0.635468206972`
- sample std: approximately `0.043023727308`
- min: approximately `0.597794518922`
- max: approximately `0.688000505242`

Only-ESM anchor:

- mean: approximately `0.6253`
- std: approximately `0.0314`
- min: approximately `0.5902`
- max: approximately `0.6722`

The node-level late-fusion mean is below both the selected GVP + late-fusion trial 49 anchor and the Only-ESM anchor. Its best seed result, `0.633163185699`, is also below the late-fusion trial 49 mean and below the late-fusion trial 49 maximum.

## Decision

Do **not** replace the selected GVP + late-fusion trial 49 anchor with this node-level late-fusion configuration.

Current validation evidence favors keeping **GVP + late fusion, trial 49** as the stronger late-fusion metal anchor.

Do **not** use held-out test results for this decision. Held-out test remains postponed until validation-side model/fusion selection is finalized.

## Next Action

Keep the selected GVP + late-fusion trial 49 anchor as the current validation-selected metal model. Do not move to held-out test unless the validation architecture search is explicitly declared complete.
