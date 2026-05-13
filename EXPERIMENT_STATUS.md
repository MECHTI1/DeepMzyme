# EXPERIMENT_STATUS.md - DeepMzyme

This is the short, mutable status note for current experiments. Keep exact run
evidence in saved outputs, not in stable workflow guides.

## Authority And Evidence Rules

- `Plan.md` remains the design authority for intended architecture, training
  logic, experiment policy, and project direction.
- Source code under `src/` is evidence of implemented behavior.
- Run outputs, saved configs, and notebook outputs are evidence of measured
  results.
- `EXPERIMENT_STATUS.md` is only a current-orientation note and may lag behind
  recent runs or code changes.
- If this file conflicts with `Plan.md`, source code, or run outputs, report the
  conflict instead of silently trusting this file.
- Do not invent missing values or exact experiment numbers.

## Current Stage

- Current task focus: metal classification.
- Stage: metal Only-GVP, radius-only, no ESM, no RING.
- Trusted split policy: non-overlapped PinMyMetal train/test split, with
  validation split by `pdbid` and `VAL_FRACTION=0.15` for model selection.
- Test-set policy: held-out test remains unused for model, checkpoint,
  hyperparameter, architecture, and fusion decisions. Use it once after the
  validation-selected anchor is fixed.
- Latest copied notebook evidence: Round 6 finalist confirmation output under
  `docs/notebook outputs/round6_three_Trials_comparisons.output_cell_notebook.md`.
- Selected stable Only-GVP anchor: not yet formally recorded, pending final
  config-level/per-class diagnostic review.

## Notebook Output File Map

- Current experiment evidence:
  - `docs/notebook outputs/round6_three_Trials_comparisons.output_cell_notebook.md`
    is the latest copied output. It contains three 50-epoch validation-only
    seed-repeat batches for finalist Only-GVP anchor candidates.
  - `docs/notebook outputs/round3_results_onlyGVP_Optuna.output_cell_notebook`
    contains the broader 50-epoch confirmation evidence for the top Round 2
    Only-GVP configs plus `gvp_layers=3` ablations.
  - `docs/notebook outputs/round4_results_onlyGVP_Optuna.output_cell_notebook`
    plus `round5_Trial_12_batch.output_cell_notebook` and
    `round5_Trial_13_batch.output_cell_notebook` contain a later 30-epoch split
    batch. Treat this as supporting evidence because the epoch budget differs.
  - `docs/notebook outputs/round1_results_onlyGVP_Optuna.output_cell_notebook`
    and `round2_results_onlyGVP_Optuna.output_cell_notebook` are earlier Optuna
    and top-k seed-repeat evidence.
- Current summaries / planning notes:
  - `docs/notebook outputs/metal_only_gvp_round3_next_batch_plan.md` documents
    the Round 3 plan and a 2026-05-12 status summary. It is now partly
    superseded by Round 6.
  - `docs/notebook outputs/metal_only_gvp_round3_decision_next_steps.md`
    captures the pre-Round-6 decision rule. It remains useful context but is not
    the latest status.
- Stable usage guide:
  - `docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md` should stay focused on stable
    notebook usage principles and should point here for current status.

## Latest Trusted Evidence

All numbers below are validation metrics from copied notebook outputs. They are
not held-out test results.

Round 6 re-ran the three finalist 50-epoch seed-repeat anchor candidates across
seeds `42,123,2026,43,44`, with:

- `TASK=metal`
- `MODEL_PRESET=Only-GVP`
- `EPOCHS=50`
- `BATCH_SIZES_CSV=8`
- `SPLIT_BY=pdbid`
- `VAL_FRACTION=0.15`
- `SELECTION_METRIC=val_metal_balanced_acc`
- `METAL_CLASS_WEIGHT_MODES_CSV=inverse_sqrt_frequency`
- no ESM, no RING, no held-out test during training

Round 6 validation-balanced-accuracy summary:

| Candidate | Mean | Sample std | Min | Max | Current interpretation |
|---|---:|---:|---:|---:|---|
| Trial7, `gvp_layers=4`, radius `6.0` | 0.6107 | 0.0415 | 0.5584 | 0.6559 | Highest mean and best single seed, but highest seed spread. |
| Trial12, `gvp_layers=3`, radius `6.0` | 0.6071 | 0.0224 | 0.5671 | 0.6184 | Nearly tied mean with much better stability. |
| Trial12, `gvp_layers=2`, radius `6.0` | 0.5986 | 0.0204 | 0.5785 | 0.6243 | Lower mean, best worst-seed value among the three finalists. |

Earlier Round 3 evidence also included Trial7 `gvp_layers=3` and Trial13
`gvp_layers=2/3`; these are currently secondary because the latest Round 6
finalist confirmation focused on Trial7 `gvp_layers=4`, Trial12 `gvp_layers=3`,
and Trial12 `gvp_layers=2`.

## Current Recommendation

- Do not rerun the broad top-Round-2-plus-`gvp_layers=3` validation batch unless
  the copied output evidence is found to be incomplete. That validation-only
  plan appears complete in the current outputs.
- Preferred anchor to inspect and likely select: Trial12 with `gvp_layers=3`,
  radius `6.0`. It is essentially tied with Trial7 `gvp_layers=4` on mean
  validation balanced accuracy while being much more stable across seeds.
- Higher-risk candidate: Trial7 with `gvp_layers=4`, radius `6.0`. Select it
  only if per-class recall, macro-F1, and min-recall diagnostics show a real
  advantage that justifies the higher variance.
- Conservative robustness candidate: Trial12 with `gvp_layers=2`, radius `6.0`.
  Select it only if worst-seed or rare-metal robustness is the priority and its
  lower mean validation balanced accuracy is acceptable.
- Do not move to Only-ESM or GVP + late fusion until the Only-GVP anchor is
  explicitly fixed.

## Recommended Next Notebook Action

First, use the notebook/reporting outputs to finish validation-only anchor
selection. No new training run is required if the Round 6 evidence is complete.

Recommended decision settings to preserve in the selected anchor record:

- `TASK=metal`
- `MODEL_PRESET=Only-GVP`
- no ESM
- no RING
- `SPLIT_BY=pdbid`
- `VAL_FRACTION=0.15`
- `SELECTION_METRIC=val_metal_balanced_acc`
- `EPOCHS=50`
- fixed seeds `42,123,2026,43,44`
- no held-out test during model selection

If the per-class diagnostics do not contradict the aggregate evidence, select
Trial12 `gvp_layers=3` as the stable Only-GVP anchor. The best validation run in
the Round 6 copied output is seed `123`, selected at epoch `20`, with
`val_metal_balanced_acc=0.6184115476458212`; this is a checkpoint candidate for
final reporting after the config-level anchor decision is recorded.

After the anchor is fixed, run the notebook's optional final held-out test
evaluation once for the selected checkpoint/configuration. That step is not a
validation experiment; it is final reporting. Do not use held-out test metrics to
switch to another seed, checkpoint, or config.

The next model-comparison experiment after the Only-GVP anchor and final test
report should be validation-only Only-ESM and then GVP + late fusion, carrying
forward the trusted split, seed list, epoch budget, selection metric, and shared
graph/training settings where applicable. It should remain validation-only
because these are still model/fusion choices.

## Decision Rule

Choose the stable Only-GVP anchor by seed-repeat mean, stability, and per-class
diagnostics, not by one lucky seed.

Use, at minimum:

- `val_metal_balanced_acc`
- `val_metal_macro_f1`
- `val_metal_min_recall`
- `val_metal_per_class_recall`
- `val_metal_collapsed4_balanced_acc`

## Test-Set Rule

- Held-out test is for final reporting only.
- Do not use held-out test to choose model, hyperparameters, checkpoint,
  architecture, fusion mode, or seed.
- The copied notebook outputs inspected here do not show created
  `test_report.json` files for the finalist runs.

## Next Stage

- After the stable Only-GVP anchor: compare Only-ESM and GVP + late fusion using
  validation metrics only.
- RING should be a later small side ablation, not mixed into the first
  ESM/fusion comparison.

## Caveats

- Exact hyperparameters and result numbers must be parsed from raw notebook
  outputs or saved configs before being used in a publication table.
- Round 6 includes per-batch "selected final run" notebook output. Treat those
  as within-batch best-validation checkpoints, not as a project-level anchor
  decision by themselves.
- The copied Round 6 output does not provide a compact combined per-class
  diagnostic table across the three finalist configs. Inspect run metadata or
  generated summary CSVs before declaring the anchor final.
- Saved/displayed `fusion=late_fusion` may appear for `Only-GVP` runs; for
  `only_gvp`, fusion is effectively irrelevant and should be reported as no ESM
  fusion.
- The 30-epoch Round 4/Round 5 batch uses a different epoch budget and should
  not override the 50-epoch finalist evidence.

## Update Checklist

After each real batch:

- Update current stage.
- Update raw evidence source.
- Update selected anchor, if any.
- Update next planned batch.
- Update caveats.
- Keep detailed run evidence in notebook-output files or saved run summaries,
  not in `docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md`.
