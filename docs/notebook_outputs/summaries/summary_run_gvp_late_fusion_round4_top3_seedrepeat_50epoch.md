# Summary Run: GVP Late Fusion Round 4 Top3 Seed Repeat 50 Epoch

## Source Folder
`docs/notebook_outputs/raw/GVP + late fusion/metal_late_fusion_optuna_top3_seedrepeat_50epoch_v1/`.

This folder contains copied run artifacts for the explicit validation-only 15-run seed-repeat batch.

## Purpose
Run full-budget validation-only seed repeats for the top 3 GVP + late-fusion Optuna candidates from Round 3:

- trial `49`
- trial `32`
- trial `15`
- seeds `42,123,2026,43,44`
- total intended runs: `3` trial configs x `5` seeds = `15`

The goal was to decide whether a GVP + late-fusion validation anchor should be selected before moving to node-level late fusion or any held-out test.

## Batch Configuration

Shared settings:

- Task: `metal`
- Model architecture: `gvp`
- Fusion mode: `late_fusion`
- ESM branch: enabled
- Epochs: `50`
- Batch size: `8`
- Split: `pdbid`
- Validation fraction: `0.15`
- Selection metric: `val_metal_balanced_acc`
- Metal class weighting: `inverse_frequency`
- Metal loss: `cross_entropy`
- Metal label smoothing: `0.0`
- Learning-rate schedule: `fixed`
- Node feature set: `conservative`
- RING edges: disabled
- Held-out test during training: disabled
- `test_report`: absent / null for all runs

The run artifacts show `run_test_eval=false` for all inspected runs.

## Candidate Configurations

| Source Optuna trial | Original Optuna val_metal_balanced_acc | learning_rate | weight_decay | hidden_s | hidden_v | gvp_layers | edge_hidden | edge_radius | head_mlp_layers | esm_fusion_dim |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `49` | `0.6750130535709283` | `1.6801503587890522e-05` | `1e-05` | `256` | `32` | `4` | `128` | `6.0` | `1` | `64` |
| `32` | `0.6585119076580177` | `5.4715836015281065e-05` | `0.001` | `128` | `32` | `2` | `128` | `6.0` | `1` | `64` |
| `15` | `0.6550963478857217` | `7.032630334240692e-05` | `0.001` | `128` | `32` | `2` | `128` | `6.0` | `1` | `64` |

## Per-Run Results

All values are validation metrics, not held-out test metrics.

| Source trial | Seed | Run name | Selected epoch | val_metal_balanced_acc |
|---:|---:|---|---:|---:|
| `49` | `42` | `top1_trial49_seed42_50epoch` | `37` | `0.6750130535709283` |
| `49` | `123` | `top1_trial49_seed123_50epoch` | `46` | `0.597794518921881` |
| `49` | `2026` | `top1_trial49_seed2026_50epoch` | `49` | `0.6880005052418845` |
| `49` | `43` | `top1_trial49_seed43_50epoch` | `28` | `0.6177941320943349` |
| `49` | `44` | `top1_trial49_seed44_50epoch` | `25` | `0.5987388250319284` |
| `32` | `42` | `top2_trial32_seed42_50epoch` | `22` | `0.6585119076580177` |
| `32` | `123` | `top2_trial32_seed123_50epoch` | `20` | `0.6418107461024083` |
| `32` | `2026` | `top2_trial32_seed2026_50epoch` | `27` | `0.6863748151780605` |
| `32` | `43` | `top2_trial32_seed43_50epoch` | `14` | `0.6126533578359136` |
| `32` | `44` | `top2_trial32_seed44_50epoch` | `29` | `0.5508127727955314` |
| `15` | `42` | `top3_trial15_seed42_50epoch` | `22` | `0.6550963478857217` |
| `15` | `123` | `top3_trial15_seed123_50epoch` | `37` | `0.635906286755011` |
| `15` | `2026` | `top3_trial15_seed2026_50epoch` | `24` | `0.6992755722978847` |
| `15` | `43` | `top3_trial15_seed43_50epoch` | `21` | `0.5957760498835549` |
| `15` | `44` | `top3_trial15_seed44_50epoch` | `37` | `0.5635841648772684` |

## Seed-Repeat Summary

| Rank by mean | Source trial | n seeds | Mean val_metal_balanced_acc | Sample std | Min | Max |
|---:|---:|---:|---:|---:|---:|---:|
| `1` | `49` | `5` | `0.635468206972` | `0.043023727308` | `0.597794518922` | `0.688000505242` |
| `2` | `32` | `5` | `0.630032719914` | `0.051725380573` | `0.550812772796` | `0.686374815178` |
| `3` | `15` | `5` | `0.629927684340` | `0.052550289215` | `0.563584164877` | `0.699275572298` |

Trial `49` has the best mean, lowest standard deviation among the three late-fusion candidates, and the best worst-seed result.

## Comparison Against Confirmed Only-ESM Anchor

Confirmed Only-ESM anchor from prior 5-seed validation evidence:

- mean approximately `0.6253`
- sample std approximately `0.0314`
- min approximately `0.5902`
- max approximately `0.6722`

Comparison:

- Trial `49` late fusion mean is approximately `0.0102` above the Only-ESM mean.
- Trial `49` late fusion min, `0.5978`, is slightly above the Only-ESM min, `0.5902`.
- Trial `49` late fusion max, `0.6880`, is above the Only-ESM max, `0.6722`.
- Trial `49` late fusion sample std, `0.0430`, is higher than the Only-ESM sample std, `0.0314`.

Interpretation: GVP + late fusion trial `49` is the validation-leading late-fusion candidate and narrowly improves over the confirmed Only-ESM anchor by mean and worst-seed value, but the improvement is modest and the late-fusion seed variance is higher.

## Caveats

- These are validation-only results. They are not held-out test results.
- Held-out test remains unused and should stay postponed until the validation-side architecture decision is fixed.
- Per-class diagnostic aggregates are not clearly available in the copied folder artifacts; the summary is based on the saved selected validation metric from `run_metadata.json`.
- The copied earlier notebook-output file `Round4_late_fusion_optuna_top3_seedrepeat_50epoch_v1.output_cell_notebook.md` showed only one normal planned run and should not be used as the main Round 4 evidence now that the full run-artifact folder is available.

## Decision

Select GVP + late fusion trial `49` as the current validation-selected late-fusion anchor:

- `learning_rate=1.6801503587890522e-05`
- `weight_decay=1e-05`
- `batch_size=8`
- `hidden_s=256`
- `hidden_v=32`
- `gvp_layers=4`
- `edge_hidden=128`
- `edge_radius=6.0`
- `head_mlp_layers=1`
- `esm_fusion_dim=64`
- `metal_class_weight_mode=inverse_frequency`
- `metal_loss_function=cross_entropy`
- `metal_label_smoothing=0.0`
- `selection_metric=val_metal_balanced_acc`

Do not select by the best single Optuna trial alone; this decision is based on the 5-seed validation repeat.

## Recommended Next Step

Proceed to the next validation-only architecture stage: a narrow node-level late-fusion check using trial `49` as the late-fusion anchor reference.

Keep the held-out test disabled. Do not run final held-out test yet.

The next run should compare node-level late fusion against this selected late-fusion anchor using validation metrics only, preferably with a small and controlled search or fixed 5-seed confirmation rather than a broad Cartesian grid.
