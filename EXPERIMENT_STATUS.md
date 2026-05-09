# EXPERIMENT_STATUS.md — DeepMzyme

This file is the lightweight, frequently updated status note for the *current*
experimental state of DeepMzyme.

It is intentionally separate from:

- `Plan.md` — long-term design authority and experiment policy.
- `AGENTS.md` — stable instructions for AI coding agents.
- `experiment_notes.md` — historical record / lab-notebook-style notes from past runs.
- run output directories — saved configs, metrics, checkpoints, and reports.

This file is mutable and may lag behind recent runs or code changes. Treat it as
a quick orientation hint, not as proof.

When in doubt:

1. Use `Plan.md` as the design authority for intended architecture, training
   logic, experiment policy, and project direction.
2. Use committed source code under `src/` as evidence of actual implemented
   behavior.
3. Use actual run outputs in configured output directories, such as
   `DeepMzyme_Data/notebook_outputs/runs/`, `/tmp/...`, or a configured
   `RUNS_DIR`, as evidence of measured results.
4. Use this file only to answer: "Where are we right now?"

If this file conflicts with `Plan.md`, source code, or run outputs, report the
conflict clearly instead of silently trusting this file.

Use placeholders such as `unknown`, `not yet run`, or `to be filled manually`
for any field that is not confidently known. Placeholders are not current-status
claims and are not current best results. Do not invent experiment numbers.

---

## Project Stage

- Stage: to be filled manually.
  Examples: Only-GVP metal baseline tuning; controlled Optuna HPO for Only-GVP;
  ESM baseline comparison; GVP + ESM late-fusion comparison; EC head bring-up;
  joint metal + EC bring-up.

- Current focus: to be filled manually.

- Supporting evidence:
  - `experiment_notes.md` contains historical exploratory baseline notes.
  - Treat those notes as historical unless this file explicitly confirms them as
    the current trusted baseline.
  - Confirm the current stage against recent run outputs and recent commits
    before relying on it.

---

## Baseline Status

- Metal classification, 6-class view (`Mn`, `Fe`, `Zn`, `Cu`, `Co`, `Ni`):
  to be filled manually.

- Metal classification, collapsed-4 view (`Mn`, `Zn`, `Cu`, `VIII`, where
  `VIII = Fe + Co + Ni`):
  to be filled manually.

- EC classification:
  not yet run / to be filled manually.

- Joint metal + EC classification:
  not yet run / to be filled manually.

Notes:

- An early Only-GVP learning-rate sweep may be recorded in `experiment_notes.md`.
  Treat such values as historical exploratory results unless manually confirmed
  here as the current trusted baseline.
- Do not copy weak or outdated exploratory numbers into this section as if they
  are current best results.

---

## Best Validation Result

Fill this section only when a run is manually confirmed as the current trusted
best according to the selected validation metric.

- Task: to be filled manually.
  Examples: `metal_6_class`, `metal_collapsed_4`, `ec_level_1`, `joint`.

- Model / fusion:
  to be filled manually.

- Selection metric:
  to be filled manually.
  Examples: `val_metal_balanced_acc`, `val_ec_group_balanced_acc`,
  `val_joint_balanced_acc`.

- Validation score:
  to be filled manually.

- Held-out test score:
  to be filled manually.
  Use only for final reporting. Do not use held-out test metrics for model,
  checkpoint, hyperparameter, architecture, or fusion-mode selection.

- Run directory / config path:
  to be filled manually.

- Seed(s):
  to be filled manually.

Update this section whenever a newer run beats the previous best on the chosen
validation selection metric. Do not leave outdated weak numbers here; overwrite
them or move them to `experiment_notes.md`.

Historical reference:

- `experiment_notes.md` may record earlier Only-GVP `metal_6_class` learning-rate
  sweeps or other exploratory runs.
- Those numbers are kept there as an audit trail.
- Do not copy them into this section unless they are manually confirmed as the
  current trusted best.

---

## Trusted Split

- Per `Plan.md`, the non-overlapped PinMyMetal split is the main trusted split
  for final held-out evaluation for both metal-type and EC tasks.

- The exact PinMyMetal split may be reported as a secondary metal-testing mode
  only, with an explicit warning that train/test overlap is possible.

- The split used by the most recent run should be verified against each run's
  saved config:
  to be filled manually.

- Before final reporting, validate train/test overlap by:
  - full structure filename
  - PDB ID
  - PDB-chain, when available
  - pocket ID, when available

---

## Next Planned Experiment

- Active next experiment:
  to be filled manually.

Candidate next steps suggested by `Plan.md` and the notebook workflow. Do not
assume these are active unless confirmed above.

1. Lock in a reproducible Only-GVP metal baseline with multiple seeds.
   Example notebook preset: `only_gvp_broad_comparison`.
2. Run narrow controlled Optuna HPO for Only-GVP after the simple baseline is
   stable.
3. Compare Only-ESM once ESM embeddings are prepared.
4. Compare GVP + late ESM fusion as the first structure-plus-sequence model.
   Example notebook preset: `baseline_model_comparison`.
5. Bring up EC classification with clear EC label-depth and group-weighting
   rules.
6. Test advanced fusion modes only after simpler baselines justify them:
   early fusion, hybrid fusion, node-level late fusion, cross-modal attention.

---

## Known Caveats

- Validation metrics are used for checkpoint selection, hyperparameter choice,
  architecture choice, and fusion-mode choice.

- Held-out test metrics are for final reporting only. Do not repeatedly inspect
  held-out test performance to choose model settings.

- Short HPO trials may rank early-training behavior rather than final full-training
  behavior. Retrain selected configurations with full epochs and, when possible,
  multiple seeds.

- If validation splits are poorly balanced or missing rare metal classes, HPO or
  model selection can be misleading. Inspect split diagnostics before trusting
  the best run.

- EC supervision is structure/protein/chain level. EC loss should use group
  weighting so structures with multiple extracted pockets are not over-counted.

- `src/model.py` may contain experimental or partially validated code. Treat
  conflicts between `Plan.md` and implementation as design issues to resolve
  conservatively.

- Notebook output artifacts such as checkpoints, run directories, CSVs, and PNGs
  are usually not committed to git. Reproducibility depends on committed source
  code plus saved per-run configs and summaries.

- Some configuration options listed in the Colab notebook may be advanced,
  experimental, or partially supported. Confirm against `Plan.md`, `src/train.py`,
  and the relevant implementation before relying on them.

---

## Test Set Rules

These rules come from `Plan.md` and override convenience shortcuts.

- Use only validation or cross-validation for checkpoint selection, hyperparameter
  choices, model architecture choices, fusion-mode choices, and HPO trial ranking.

- Evaluate the selected validation-best checkpoint on the held-out test set once
  for final reporting.

- Do not iterate on the held-out test set.

- Do not choose a model because it looks better on the held-out test set.

- Always record which split was used:
  non-overlapped PinMyMetal split, exact PinMyMetal split, or custom split.

- Label exact-split metal results as secondary/possibly-overlapped reference
  results when relevant.

- The `--allow-train-loss-test-eval-debug` flag is debug-only. It must not be
  used to produce reportable held-out test numbers.

---

## Manual Update Checklist

When updating this file after a new trusted run:

1. Update `Project Stage`.
2. Update `Baseline Status`.
3. Update `Best Validation Result` only if the run is manually confirmed as the
   current best by validation metric.
4. Update `Trusted Split` if the split changed.
5. Update `Next Planned Experiment`.
6. Move old or superseded values to `experiment_notes.md` rather than leaving
   them here as current status.
7. Do not invent missing values. Use `to be filled manually` when unsure.
