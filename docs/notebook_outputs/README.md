# Notebook Outputs

This directory keeps copied notebook outputs as experiment evidence while making
the important results easier to scan.

For the full validation/testing run order and folder ownership map, read
`docs/README.md`.

## Layout

- `raw/` preserves long copied notebook-output files. Treat these as the source
  evidence for measured validation results, run commands, warnings, and copied
  notebook logs.
- `summaries/` contains short human-readable summaries. Read these first when
  deciding what happened in a run, then inspect the matching raw file when you
  need exact output details.

Default reading order: scan the relevant summary by name; load a raw file only
when a summary cites it or when exact log lines or run commands are needed. Do
not bulk-load the `raw/` tree.

Planning rule: raw outputs are evidence for what already happened, not the
default recipe for the next fresh experiment. When planning a new check or new
Optuna sweep, rely on raw outputs heavily only if the user explicitly asks to
use previous running/results/raws. Otherwise use summaries and status as
context, then prefer a fresh, broad-but-sensible validation-only search space.
The live notebook default snapshot is tracked in the root `README.md`; as of
the current notebook variables, new default notebook runs use
`VAL_FRACTION = 0.18`, `METAL_LABEL_SCHEME = "five_class"`, and
`METAL_NODE_MODE = "per_metal"`. Older copied outputs that used
`VAL_FRACTION = 0.15`, six-class labels, or residue-only graphs remain valid
historical evidence, but they are not the current default recipe and must not
be merged into the same comparison without labeling those differences.

Canonical path rule: this folder is `docs/notebook_outputs/` with an
underscore. Do not create or use a parallel `docs/notebook outputs/` folder with
a space. Copied raw evidence should go under `raw/<model-family>/`; concise
summaries should go under `summaries/`.

## Experiment-Tracking Rules

- Use validation metrics for model, checkpoint, hyperparameter, architecture,
  and HPO decisions.
- Keep the held-out test set reserved for final reporting after a
  validation-selected configuration is fixed and the corresponding Stage 6B
  final-refit run has been completed and frozen.
- Record `VAL_FRACTION`, `METAL_LABEL_SCHEME`, `METAL_NODE_MODE`, and
  `SPLIT_BY` in copied summaries, because changing any of these values changes
  the comparison identity.
- Do not change or reinterpret experiment results when reorganizing these files.
- If a summary field is unclear in the raw output, it should say
  `Not clearly available in source file`.

## Cross-Family Snapshot

- `summaries/LEADERBOARD.md` — single-page validation snapshot across model
  families with reliability tiers (5-seed/50-epoch vs. partial). Start here
  before scanning individual summaries. Held-out test is not yet evaluated.

## Current Summary Files

- `summaries/summary_run_only_esm_round1_full_coverage.md`
- `summaries/summary_run_only_esm_round1_anchor_comparison.md`
- `summaries/summary_run_only_esm_round2_lr_wd_weight_screen.md`
- `summaries/summary_run_only_esm_round3_seed_confirmation.md`
- `summaries/summary_run_gvp_late_fusion_round1_trial12_anchor.md`
- `summaries/summary_run_gvp_late_fusion_round1_full_coverage.md`
- `summaries/summary_run_gvp_late_fusion_round2_confirmed_esm_anchor.md`
- `summaries/summary_run_gvp_late_fusion_round3_optuna_50_v1.md`
- `summaries/summary_run_gvp_late_fusion_round4_top3_seedrepeat_50epoch.md`
- `summaries/summary_run_gvp_node_level_late_fusion_round1_from_latefusion_trial49_seedrepeat_50epoch.md`
- `summaries/summary_run_only_gvp_round1_optuna_hpo.md`
- `summaries/summary_run_only_gvp_round2_optuna_seed_repeat.md`
- `summaries/summary_run_only_gvp_round3_top_optuna_confirm.md`
- `summaries/summary_run_only_gvp_round4_top3_plus_gvp3.md`
- `summaries/summary_run_only_gvp_round5_trial12_batch.md`
- `summaries/summary_run_only_gvp_round5_trial13_batch.md`
- `summaries/summary_run_only_gvp_round6_three_trial_comparison.md`
- `summaries/summary_run_hybrid_round1_optuna_plus_top3_seedrepeat.md`
- `summaries/summary_run_hybrid_ring_round2_optuna_50epoch_wide_v1_trials105_176.md`

Older short Only-GVP planning notes are also kept in `summaries/` as historical
context.
