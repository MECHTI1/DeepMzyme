# Summary Run: GVP Late Fusion Round 3 Optuna 50 V1

## Source File
`docs/notebook_outputs/raw/GVP + late fusion/Round3_late_fusion_optuna_50_v1.output_cell_notebook.md`.

## Purpose
Run a controlled Optuna hyperparameter search within the GVP + late fusion model family for metal classification, then generate top-K seed-repeat commands/results from the best Optuna candidates.

## Configuration
- Task: metal
- Run mode: controlled Optuna HPO plus top-K seed-repeat evaluation
- Model preset / architecture: GVP + late fusion / gvp
- Fusion mode: late_fusion
- ESM usage: ESM embeddings used
- HPO/Optuna settings: in-memory study `deepmzyme_controlled_hpo`, 50 trials, 40 epochs per trial, fixed HPO split/model seed 42, split_by=pdbid, val_fraction=0.15
- Search preset: custom
- Search space: batch_size=8; learning_rate=1e-5 to 1e-4; weight_decay in 0.0, 1e-5, 1e-4, 1e-3; hidden_s in 128,256; hidden_v in 16,32; edge_hidden in 64,128; edge_radius in 6.0,8.0; gvp_layers in 2,3,4; head_mlp_layers in 1,2,3; esm_fusion_dim in 64,128,256; metal_class_weight_mode in inverse_frequency,inverse_sqrt_frequency
- Selection metric: val_metal_balanced_acc
- Held-out test during training: no held-out test results present

## Best Single Optuna Trial
- Best validation metric: val_metal_balanced_acc = 0.6750130535709283
- Best trial/run: trial 49, `optuna_deepmzyme_controlled_hpo_trial0049_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d32997f7`
- Best hyperparameters: learning_rate=1.6801503587890522e-05, weight_decay=1e-05, batch_size=8, hidden_s=256, hidden_v=32, edge_hidden=128, gvp_layers=4, edge_radius=6.0, esm_fusion_dim=64, head_mlp_layers=1, metal_class_weight_mode=inverse_frequency
- Selected epoch/checkpoint if available: selected_epoch=37 for the best trial; best_model_checkpoint.pt saved

## Top Single-Trial Results
| Rank | Trial | Validation balanced accuracy | Key hyperparameters |
|---:|---:|---:|---|
| 1 | 49 | 0.6750130535709283 | selected_epoch=37, lr=1.6801503587890522e-05, wd=1e-05, hidden_s=256, hidden_v=32, edge_hidden=128, gvp_layers=4, edge_radius=6.0, esm_fusion_dim=64, head_mlp_layers=1, class_weight=inverse_frequency |
| 2 | 32 | 0.6585119076580177 | selected_epoch=22, lr=5.4715836015281065e-05, wd=0.001, hidden_s=128, hidden_v=32, edge_hidden=128, gvp_layers=2, edge_radius=6.0, esm_fusion_dim=64, head_mlp_layers=1, class_weight=inverse_frequency |
| 3 | 15 | 0.6550963478857217 | selected_epoch=22, lr=7.032630334240692e-05, wd=0.001, hidden_s=128, hidden_v=32, edge_hidden=128, gvp_layers=2, edge_radius=6.0, esm_fusion_dim=64, head_mlp_layers=1, class_weight=inverse_frequency |

## Seed-Repeat Section
The notebook selected the top 3 Optuna candidates and ran seeds 42, 123, 2026, 43, and 44. However, the generated seed-repeat commands used `--epochs 1`, and the raw output explicitly says 1-3 epoch runs are smoke/debug only and not model-quality comparisons.

| Seed-repeat rank | Source Optuna rank | Trial | Seeds completed | Mean validation balanced accuracy | Std | Min | Max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 32 | 5 | 0.33490255121673934 | 0.031422094392314186 | 0.30906417975383493 | 0.3724438959733078 |
| 2 | 3 | 15 | 5 | 0.3289684151101894 | 0.03928018169981929 | 0.27270718617467843 | 0.3744046802870333 |
| 3 | 1 | 49 | 5 | 0.3098742052523086 | 0.039841555042331014 | 0.24705882352941178 | 0.35392106297278714 |

Interpretation: these seed-repeat numbers should not be used to reject the top Optuna candidates because they are 1-epoch smoke/debug reruns, not comparable to the 40-epoch Optuna trials or prior 50-epoch validation baselines.

## Main Findings
- The best 40-epoch single-seed Optuna trial reached 0.6750130535709283 validation balanced accuracy.
- The notebook selected trial 49 as the final validation-selected run with selected_epoch=37.
- The top single-trial candidates all used `inverse_frequency` metal class weighting.
- The strongest single-trial region was late fusion with edge_radius=6.0, edge_hidden=128, hidden_v=32, esm_fusion_dim=64, and low to mid learning rates.
- The raw output reports no held-out test results.
- Failed run directories were reported as an empty list.

## Caveats
- The top-K seed-repeat section used only 1 epoch per seed, so it is smoke/debug evidence only.
- The first completed-run summary table warns that mixed or missing RUN_BATCH_ID values were found. Later seed-repeat source lines say the top-K source used the just-completed Optuna study rather than scanning old mixed directories.
- This HPO is single-seed model-selection evidence until the top candidates are rerun with a realistic epoch budget across seeds.
- Validation-only evidence should not be used as final held-out test performance.

## Recommended Next Step
Rerun the top Optuna candidates, especially trials 49, 32, and 15, with a proper validation-only seed-repeat budget such as the established 40-50 epochs before choosing a late-fusion anchor or comparing against the confirmed Only-ESM baseline.
