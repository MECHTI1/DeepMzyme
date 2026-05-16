# Summary Run: Hybrid Round 1 Optuna Plus Top-3 Seed Repeat

## Source File
`docs/notebook_outputs/raw/Hybrid/Round1_hybrid_fusion_optuna_plus_top3_seedrepeat.output_cell_notebook.md`.

## Purpose
First controlled Optuna hyperparameter search within the GVP + ESM hybrid fusion model family for the joint (metal + EC) task, followed by a top-3 seed-repeat of the best Optuna candidates against seeds 42, 123, and 2026.

## Configuration
- Task: joint
- Model preset / architecture: GVP + ESM hybrid / gvp
- Fusion mode: hybrid
- ESM usage: ESM embeddings used (hybrid fusion combines GVP graph features with ESM)
- Run mode: Optuna HPO plus top-K seed-repeat
- Optuna study: `optuna_deepmzyme_joint_hybridfusion_serious_v1`
- Optuna trials completed: 40 (trial0000–trial0039 plus a few additional fixed-seed Optuna runs)
- Top-K seed repeat: top 3 Optuna trials (trial 17, trial 32, trial 24), each rerun for seeds 42, 123, 2026
- Total completed runs: 49 in batch directory; 1 additional planned row in the summary table
- Seeds (seed-repeat block): 42, 123, 2026
- Selection metric: val_joint_balanced_acc
- Metric direction: higher_is_better
- Class weight mode (best run): inverse_frequency
- Held-out test during training: no held-out test results present
- Run batch id: `debug_smoke`
- Runs root: `/content/drive/MyDrive/DeepMzyme/notebook_outputs/runs/debug_smoke`
- Epochs per trial: Not clearly available in source file (selected_epoch values up to 40 suggest a ~40-epoch budget)

## Best Single Optuna Trial
- Best validation metric: val_joint_balanced_acc = 0.748343
- Best trial / run: trial 17 → `optuna_deepmzyme_joint_hybridfusion_serious_v1_trial0017_..._archgvp_fus_0b71fd0c`
- Best hyperparameters: learning_rate=3.975e-05, weight_decay=1e-05, seed=42, class_weight=inverse_frequency
- Selected epoch / checkpoint: selected_epoch=31; best_model_checkpoint.pt saved
- Companion metal-side metrics at the selected epoch: val_metal_balanced_acc=0.672077, val_metal_collapsed4_balanced_acc=0.733259

## Top Single-Trial Results (single seed = 42)
| Rank | Trial | val_joint_balanced_acc | val_metal_balanced_acc | val_metal_collapsed4_balanced_acc | learning_rate | weight_decay | selected_epoch |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17 | 0.748343 | 0.672077 | 0.733259 | 3.975e-05 | 1e-05 | 31 |
| 2 | 32 | 0.720896 | 0.671772 | 0.720515 | 3.1e-05  | 1e-05 | 39 |
| 3 | 24 | 0.717782 | 0.646874 | 0.718244 | 2.9e-05  | 1e-05 | 37 |

## Top-3 Seed Repeat (val_joint_balanced_acc)
| Top rank | Trial | Seed 42 | Seed 2026 | Seed 123 | Mean | Min | Max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17 | 0.748343 | 0.682223 | 0.661561 | 0.697376 | 0.661561 | 0.748343 |
| 2 | 32 | 0.720896 | 0.705898 | 0.637834 | 0.688209 | 0.637834 | 0.720896 |
| 3 | 24 | 0.717782 | 0.670281 | 0.672888 | 0.686984 | 0.670281 | 0.717782 |

## Seed-Repeat Per-Seed Metal-Side Metrics
| Top rank | Trial | Seed | val_metal_balanced_acc | val_metal_collapsed4_balanced_acc | selected_epoch |
|---:|---:|---:|---:|---:|---:|
| 1 | 17 | 42   | 0.672077 | 0.733259 | 31 |
| 1 | 17 | 2026 | 0.709319 | 0.770346 | 21 |
| 1 | 17 | 123  | 0.658546 | 0.694643 | 19 |
| 2 | 32 | 42   | 0.671772 | 0.720515 | 39 |
| 2 | 32 | 2026 | 0.688095 | 0.741991 | 36 |
| 2 | 32 | 123  | 0.601800 | 0.711688 | 33 |
| 3 | 24 | 42   | 0.646874 | 0.718244 | 37 |
| 3 | 24 | 2026 | 0.668771 | 0.722024 | 33 |
| 3 | 24 | 123  | 0.652320 | 0.722511 | 37 |

## Main Findings
- The best single-seed Optuna trial for the GVP + ESM hybrid fusion + joint task reached val_joint_balanced_acc = 0.748343 at trial 17 (lr ≈ 4e-05, wd = 1e-05, selected_epoch = 31).
- All top-3 Optuna trials used `inverse_frequency` metal class weighting and shared a similar low–mid learning-rate region (≈ 2.9e-05 to 4e-05) with weight_decay = 1e-05.
- The notebook's automatic interpretation flagged "middle around 1e-4 (lr ≈ 3.975e-05)" as the best learning-rate region.
- Top-3 seed-repeat mean val_joint_balanced_acc landed in the 0.687–0.697 range across the three candidates, lower than the seed-42 Optuna numbers; trial 17 still has the highest single-seed peak (0.748343 at seed 42).
- Metal-side companion metrics (val_metal_balanced_acc, val_metal_collapsed4_balanced_acc) are reported alongside the joint-selection metric but were not the selection target.
- Held-out test results are not present in the raw output.
- No failed runs were reported.
- Best Only-GVP configuration: not available in this run (no Only-GVP rows under this batch).
- Best ESM-based configuration coincides with the overall best: trial 17.
- RING vs non-RING comparison: not available in this run.

## Caveats
- The run batch id is `debug_smoke`, and the raw output contains a STRONG WARNING that "Mixed or missing RUN_BATCH_ID values were found in the summary table: ['', 'debug_smoke']". This means the scanned table may mix rows from different intents — treat the comparison table as round-1 evidence for this batch only, not as a cross-batch comparison.
- Despite the `debug_smoke` batch id, the Optuna study name is `..._serious_v1` and selected_epoch values reach up to 40, so these are not 1-epoch smoke runs. The exact configured epoch budget is not clearly available in source file.
- Selection metric is `val_joint_balanced_acc` (joint task), not pure metal balanced accuracy, so direct comparison against Only-ESM / GVP + late-fusion metal-only summaries should use the metal-side columns rather than the joint-selection value.
- Top-K seed-repeat used three seeds (42, 123, 2026); seeds 43 and 44 used elsewhere in the project for 5-seed repeats are not part of this run.
- Validation-only evidence; held-out test must remain reserved until the validation-side hybrid-fusion architecture and HPO decisions are finalized.

## Recommended Next Step
Per the raw output's automatic interpretation: "run top-K seed-repeat validation for the best Optuna configurations". Before committing the hybrid-fusion family as a candidate metal-anchor architecture, extend the top-3 seed-repeat to the project's standard 5-seed set (42, 123, 2026, 43, 44) at the full validation epoch budget, and compare both the joint-selection metric and the metal-side balanced accuracies against the confirmed Only-ESM and GVP + late-fusion (trial 49) anchors. Do not move to held-out test evaluation on this basis alone.
