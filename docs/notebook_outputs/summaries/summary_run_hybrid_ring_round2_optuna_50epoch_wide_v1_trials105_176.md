# Summary Run: Hybrid RING Round 2 Optuna 50-Epoch Wide V1 Trials 105-176

## Source File
`docs/notebook_outputs/raw/Hybrid/Round2_joint_hybrid_ring_optuna_50epoch_wide_v1_trials105_176_partial_trial177.output_cell_notebook.md`.

## Purpose
Continuation log for the existing `joint_hybrid_ring_optuna_50epoch_wide_v1`
Optuna study. This run explores GVP + ESM hybrid fusion with RING edges enabled
for the joint task, while selecting trials by metal validation balanced
accuracy.

## Configuration
- Task: joint
- Model preset / architecture: GVP + hybrid fusion / `gvp`
- Fusion mode: `hybrid`
- RING edges: enabled with `--use-ring-edges`, `--require-ring-edges`, and `--prepare-missing-ring-edges`
- ESM usage: ESM embeddings enabled
- External features: updated source, missing external features allowed
- Optuna study: `joint_hybrid_ring_optuna_50epoch_wide_v1`
- Search preset: custom, wide search space
- Source log trial range: completed trials 105-176; trial 177 starts but is not completed in this copied output
- Completed trials in this source file: 72
- Trial epochs: 50
- Fixed HPO split/model seed: 42
- Validation split: `split_by=pdbid`, `val_fraction=0.15`
- Selection metric: `val_metal_balanced_acc`
- Held-out test during training: no held-out test results present in this source file

## Best Completed Trial In This Source File
- Best trial: 114
- Best validation metric: `val_metal_balanced_acc=0.7303469775006777`
- Selected epoch by source epoch logs: 37
- Companion epoch-37 validation metrics:
  - `val_metal_acc=0.8000`
  - `val_ec_acc=0.7611`
  - `val_ec_bal_acc=0.5814`
  - `val_ec_group_bal_acc=0.6196`
  - `val_joint_bal_acc=0.6559`
  - `val_metal_min_recall=0.3333`
  - `val_fe_recall=0.6471`

Best trial 114 hyperparameters:

| Parameter | Value |
|---|---:|
| learning_rate | `3.705631497756492e-05` |
| weight_decay | `3e-07` |
| batch_size | `12` |
| hidden_s | `320` |
| hidden_v | `16` |
| gvp_layers | `4` |
| edge_hidden | `192` |
| edge_radius | `7.0` |
| head_mlp_layers | `2` |
| esm_fusion_dim | `256` |
| early_esm_dim | `48` |
| early_esm_dropout | `0.05` |
| metal_loss_weight | `2.0` |
| ec_loss_weight | `0.25` |
| metal_class_weight_mode | `effective_number` |

## Top Completed Trials In This Source File

| Rank | Trial | val_metal_balanced_acc | Selected epoch | val_joint_bal_acc at selected epoch | val_ec_bal_acc at selected epoch | Notes |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 114 | `0.7303469775006777` | 37 | `0.6559` | `0.5814` | Best metal-selected trial in copied source |
| 2 | 124 | `0.7298725941989699` | 37 | `0.7504` | `0.7709` | One of many near-tied metal trials |
| 3 | 125 | `0.7298725941989699` | 37 | `0.7504` | `0.7709` | Near-tied metal trial |
| 4 | 126 | `0.7298725941989699` | 37 | `0.7504` | `0.7709` | Near-tied metal trial |
| 5 | 128 | `0.7298725941989699` | 37 | `0.7504` | `0.7709` | Near-tied metal trial |

Additional trials with the same `0.7298725941989699` value appear later in the
source file, including trials 130, 136, 137, 144, 147, 152, 154, 155, 162, 164,
167, 172, 174, and 176.

## Main Findings
- Trial 114 slightly improved the study best in this copied continuation,
  replacing the previous best trial 84 value of `0.725445016716364` with
  `0.7303469775006777`.
- The strongest region in this copied source is narrow: batch size 12,
  `hidden_s=320`, `hidden_v=16`, `gvp_layers=4`, `edge_hidden=192`,
  `edge_radius=7.0`, `esm_fusion_dim=256`, `early_esm_dim=48`, and
  `metal_class_weight_mode=effective_number`.
- Many near-tied trials use `metal_loss_weight=1.0`, `ec_loss_weight=0.75`,
  `weight_decay=0.001`, and `early_esm_dropout` of either `0.05` or `0.3`.
- Trial 114 is best by the configured metal-selection metric, but several
  near-tied metal trials have higher companion `val_joint_bal_acc` at their
  selected epoch. Do not reinterpret this as a joint-metric selection result.

## Caveats
- This is single-seed validation evidence with fixed seed 42, not the project
  standard 5-seed, 50-epoch confirmation.
- The source file is a partial continuation log: trial 177 starts but does not
  finish in the copied output.
- The notebook reports `Trials: 120`, but because the Optuna study already
  existed, this source file starts at trial 105 and does not contain a complete
  fresh study from trial 0.
- The held-out test set is not evaluated here and must remain reserved for final
  reporting after validation-side model selection is fixed.

## Recommended Next Step
Treat trial 114 and the near-tied trial family as exploratory hybrid+RING
candidates. Before changing the project anchor, run a validation-only seed
repeat on the best distinct candidates using the project-standard seeds
42, 123, 2026, 43, and 44, then compare against the confirmed Only-ESM and
GVP + late-fusion anchors by validation metrics only.
