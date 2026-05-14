# Artifact-Derived Raw Results: metal_late_fusion_optuna_top3_seedrepeat_50epoch_v1

This file consolidates the copied run-artifact folder:

`docs/notebook_outputs/raw/GVP + late fusion/metal_late_fusion_optuna_top3_seedrepeat_50epoch_v1/`

It is reconstructed from the saved `run_metadata.json`, `run_config.json`,
`dataset_summary.json`, `split_diagnostics.json`, and checkpoint files in each
run directory. The Colab stdout/stderr epoch logs were not copied into this
folder, so this file records the artifact-derived raw run records rather than
full notebook cell stdout.

## Evidence Scope

- Batch id / runs directory name: `metal_late_fusion_optuna_top3_seedrepeat_50epoch_v1`
- Number of copied run directories: `15`
- Intended design: top 3 GVP + late-fusion Optuna configs x 5 seeds
- Seeds: `42,123,2026,43,44`
- Task: `metal`
- Model architecture: `gvp`
- Fusion mode: `late_fusion`
- ESM branch: enabled
- Epoch budget: `50`
- Split: `pdbid`
- Validation fraction: `0.15`
- Selection metric: `val_metal_balanced_acc`
- Held-out test during training: disabled in all copied run metadata
- `test_report`: null / absent in all copied run metadata
- Git commit recorded in run metadata: `a4718ab34e37ac1f9ed1fa5f6a7125ce50095441`

## Shared Data / Split Diagnostics

The seed `42` split diagnostics record:

- train pockets: `1031`
- validation pockets: `182`
- train groups by `pdbid`: `1001`
- validation groups by `pdbid`: `93`
- train/validation overlap counts: `pocket_id=0`, `structure_id=0`, `pdbid_chain=0`, `pdbid=0`
- train metal distribution: `Mn=475`, `Cu=60`, `Zn=156`, `Fe=223`, `Co=67`, `Ni=50`
- validation metal distribution: `Mn=85`, `Cu=13`, `Zn=31`, `Fe=33`, `Co=13`, `Ni=7`
- missing train metal classes: none
- missing validation metal classes: none

The copied run metadata consistently records:

- structure dir: `/content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train`
- summary CSV: `/content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv`
- ESM embeddings dir: `/content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings`
- invalid structure policy: `skip`
- unsupported metal policy: `error`

## Candidate Configurations

### Source Trial 49

- Original Round 3 Optuna validation score: `0.6750130535709283`
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
- `lr_schedule=fixed`

### Source Trial 32

- Original Round 3 Optuna validation score: `0.6585119076580177`
- `learning_rate=5.4715836015281065e-05`
- `weight_decay=0.001`
- `batch_size=8`
- `hidden_s=128`
- `hidden_v=32`
- `gvp_layers=2`
- `edge_hidden=128`
- `edge_radius=6.0`
- `head_mlp_layers=1`
- `esm_fusion_dim=64`
- `metal_class_weight_mode=inverse_frequency`
- `metal_loss_function=cross_entropy`
- `metal_label_smoothing=0.0`
- `lr_schedule=fixed`

### Source Trial 15

- Original Round 3 Optuna validation score: `0.6550963478857217`
- `learning_rate=7.032630334240692e-05`
- `weight_decay=0.001`
- `batch_size=8`
- `hidden_s=128`
- `hidden_v=32`
- `gvp_layers=2`
- `edge_hidden=128`
- `edge_radius=6.0`
- `head_mlp_layers=1`
- `esm_fusion_dim=64`
- `metal_class_weight_mode=inverse_frequency`
- `metal_loss_function=cross_entropy`
- `metal_label_smoothing=0.0`
- `lr_schedule=fixed`

## Per-Run Raw Records

Columns:

- `run_name`
- `source_trial`
- `seed`
- `epochs`
- `selected_checkpoint_epoch`
- `selected_metric`
- `selected_metric_value`
- `run_test_eval`
- `test_report_present`

| run_name | source_trial | seed | epochs | selected_checkpoint_epoch | selected_metric | selected_metric_value | run_test_eval | test_report_present |
|---|---:|---:|---:|---:|---|---:|---|---|
| `top1_trial49_seed42_50epoch` | `49` | `42` | `50` | `37` | `val_metal_balanced_acc` | `0.6750130535709283` | `false` | `false` |
| `top1_trial49_seed123_50epoch` | `49` | `123` | `50` | `46` | `val_metal_balanced_acc` | `0.597794518921881` | `false` | `false` |
| `top1_trial49_seed2026_50epoch` | `49` | `2026` | `50` | `49` | `val_metal_balanced_acc` | `0.6880005052418845` | `false` | `false` |
| `top1_trial49_seed43_50epoch` | `49` | `43` | `50` | `28` | `val_metal_balanced_acc` | `0.6177941320943349` | `false` | `false` |
| `top1_trial49_seed44_50epoch` | `49` | `44` | `50` | `25` | `val_metal_balanced_acc` | `0.5987388250319284` | `false` | `false` |
| `top2_trial32_seed42_50epoch` | `32` | `42` | `50` | `22` | `val_metal_balanced_acc` | `0.6585119076580177` | `false` | `false` |
| `top2_trial32_seed123_50epoch` | `32` | `123` | `50` | `20` | `val_metal_balanced_acc` | `0.6418107461024083` | `false` | `false` |
| `top2_trial32_seed2026_50epoch` | `32` | `2026` | `50` | `27` | `val_metal_balanced_acc` | `0.6863748151780605` | `false` | `false` |
| `top2_trial32_seed43_50epoch` | `32` | `43` | `50` | `14` | `val_metal_balanced_acc` | `0.6126533578359136` | `false` | `false` |
| `top2_trial32_seed44_50epoch` | `32` | `44` | `50` | `29` | `val_metal_balanced_acc` | `0.5508127727955314` | `false` | `false` |
| `top3_trial15_seed42_50epoch` | `15` | `42` | `50` | `22` | `val_metal_balanced_acc` | `0.6550963478857217` | `false` | `false` |
| `top3_trial15_seed123_50epoch` | `15` | `123` | `50` | `37` | `val_metal_balanced_acc` | `0.635906286755011` | `false` | `false` |
| `top3_trial15_seed2026_50epoch` | `15` | `2026` | `50` | `24` | `val_metal_balanced_acc` | `0.6992755722978847` | `false` | `false` |
| `top3_trial15_seed43_50epoch` | `15` | `43` | `50` | `21` | `val_metal_balanced_acc` | `0.5957760498835549` | `false` | `false` |
| `top3_trial15_seed44_50epoch` | `15` | `44` | `50` | `37` | `val_metal_balanced_acc` | `0.5635841648772684` | `false` | `false` |

## Seed-Repeat Aggregate From Raw Records

| source_trial | n | mean_val_metal_balanced_acc | sample_std | min | max |
|---:|---:|---:|---:|---:|---:|
| `49` | `5` | `0.635468206972` | `0.043023727308` | `0.597794518922` | `0.688000505242` |
| `32` | `5` | `0.630032719914` | `0.051725380573` | `0.550812772796` | `0.686374815178` |
| `15` | `5` | `0.629927684340` | `0.052550289215` | `0.563584164877` | `0.699275572298` |

## Artifact Inventory

Each copied run directory contains:

- `best_model_checkpoint.pt`
- `last_model_checkpoint.pt`
- `run_config.json`
- `run_metadata.json`
- `dataset_summary.json`
- `split_diagnostics.json`
- `prepare_status.json`

No copied per-run stdout/stderr logs were present in the artifact folder.

## Run Directories

- `docs/notebook_outputs/raw/GVP + late fusion/metal_late_fusion_optuna_top3_seedrepeat_50epoch_v1/top1_trial49_seed42_50epoch/`
- `docs/notebook_outputs/raw/GVP + late fusion/metal_late_fusion_optuna_top3_seedrepeat_50epoch_v1/top1_trial49_seed123_50epoch/`
- `docs/notebook_outputs/raw/GVP + late fusion/metal_late_fusion_optuna_top3_seedrepeat_50epoch_v1/top1_trial49_seed2026_50epoch/`
- `docs/notebook_outputs/raw/GVP + late fusion/metal_late_fusion_optuna_top3_seedrepeat_50epoch_v1/top1_trial49_seed43_50epoch/`
- `docs/notebook_outputs/raw/GVP + late fusion/metal_late_fusion_optuna_top3_seedrepeat_50epoch_v1/top1_trial49_seed44_50epoch/`
- `docs/notebook_outputs/raw/GVP + late fusion/metal_late_fusion_optuna_top3_seedrepeat_50epoch_v1/top2_trial32_seed42_50epoch/`
- `docs/notebook_outputs/raw/GVP + late fusion/metal_late_fusion_optuna_top3_seedrepeat_50epoch_v1/top2_trial32_seed123_50epoch/`
- `docs/notebook_outputs/raw/GVP + late fusion/metal_late_fusion_optuna_top3_seedrepeat_50epoch_v1/top2_trial32_seed2026_50epoch/`
- `docs/notebook_outputs/raw/GVP + late fusion/metal_late_fusion_optuna_top3_seedrepeat_50epoch_v1/top2_trial32_seed43_50epoch/`
- `docs/notebook_outputs/raw/GVP + late fusion/metal_late_fusion_optuna_top3_seedrepeat_50epoch_v1/top2_trial32_seed44_50epoch/`
- `docs/notebook_outputs/raw/GVP + late fusion/metal_late_fusion_optuna_top3_seedrepeat_50epoch_v1/top3_trial15_seed42_50epoch/`
- `docs/notebook_outputs/raw/GVP + late fusion/metal_late_fusion_optuna_top3_seedrepeat_50epoch_v1/top3_trial15_seed123_50epoch/`
- `docs/notebook_outputs/raw/GVP + late fusion/metal_late_fusion_optuna_top3_seedrepeat_50epoch_v1/top3_trial15_seed2026_50epoch/`
- `docs/notebook_outputs/raw/GVP + late fusion/metal_late_fusion_optuna_top3_seedrepeat_50epoch_v1/top3_trial15_seed43_50epoch/`
- `docs/notebook_outputs/raw/GVP + late fusion/metal_late_fusion_optuna_top3_seedrepeat_50epoch_v1/top3_trial15_seed44_50epoch/`

## Raw-Evidence Notes

- These records are validation-only.
- No held-out test result is present.
- The selected checkpoint for each run is recorded in its `run_metadata.json`.
- This file should be used as the single-file raw evidence index for the folder batch.
- The corresponding human-readable summary is:
  `docs/notebook_outputs/summaries/summary_run_gvp_late_fusion_round4_top3_seedrepeat_50epoch.md`.
