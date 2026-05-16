# Notebook Outputs

This directory keeps copied notebook outputs as experiment evidence while making
the important results easier to scan.

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

## Experiment-Tracking Rules

- Use validation metrics for model, checkpoint, hyperparameter, architecture,
  and HPO decisions.
- Keep the held-out test set reserved for final reporting after a validation
  selected configuration is fixed.
- Do not change or reinterpret experiment results when reorganizing these files.
- If a summary field is unclear in the raw output, it should say
  `Not clearly available in source file`.

## Current Summary Files

- `summaries/summary_run_only_esm_round1_full_coverage.md`
- `summaries/summary_run_only_esm_round1_anchor_comparison.md`
- `summaries/summary_run_only_esm_round2_lr_wd_weight_screen.md`
- `summaries/summary_run_only_esm_round3_seed_confirmation.md`
- `summaries/summary_run_gvp_late_fusion_round1_trial12_anchor.md`
- `summaries/summary_run_gvp_late_fusion_round1_full_coverage.md`
- `summaries/summary_run_gvp_late_fusion_round2_confirmed_esm_anchor.md`
- `summaries/summary_run_gvp_late_fusion_round3_optuna_50_v1.md`
- `summaries/summary_run_only_gvp_round1_optuna_hpo.md`
- `summaries/summary_run_only_gvp_round2_optuna_seed_repeat.md`
- `summaries/summary_run_only_gvp_round3_top_optuna_confirm.md`
- `summaries/summary_run_only_gvp_round4_top3_plus_gvp3.md`
- `summaries/summary_run_only_gvp_round5_trial12_batch.md`
- `summaries/summary_run_only_gvp_round5_trial13_batch.md`
- `summaries/summary_run_only_gvp_round6_three_trial_comparison.md`
- `summaries/summary_run_hybrid_round1_optuna_plus_top3_seedrepeat.md`

Older short Only-GVP planning notes are also kept in `summaries/` as historical
context.
