# EXPERIMENT_STATUS.md — DeepMzyme

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

- Stage: metal Only-GVP, radius-only, no ESM, no RING.
- Raw evidence source: copied notebook outputs under `docs/notebook outputs/`.
- Exact hyperparameters should be parsed from those notebook-output files, not
  manually recopied into `docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md`.
- Selected stable anchor: not yet selected.

## Candidate Configs

- Current candidate configs: the best round2 Only-GVP Optuna-derived configs,
  especially trial7, trial12, and trial13 if supported by parsed notebook
  outputs.
- Also test a narrow `gvp_layers=3` ablation for each selected candidate config.
- Treat exact trial settings as parsed evidence from notebook outputs, not as
  hand-copied status-file facts.

## Planned Next Batch

- Run selected top configs plus their 3-layer ablations across a fixed seed list
  such as `42,123,2026,43,44`.
- Expected size if using 3 top configs plus 3 layer-3 ablations: 6 configs × 5
  seeds = 30 validation-only runs.

Required fixed settings:

- `TASK=metal`
- `MODEL_PRESET=Only-GVP`
- no ESM
- no RING
- `SPLIT_BY=pdbid`
- `VAL_FRACTION=0.15`
- `SELECTION_METRIC=val_metal_balanced_acc`
- no held-out test

## Decision Rule

Choose the stable Only-GVP anchor by seed-repeat mean, stability, and per-class
diagnostics, not by one lucky seed.

## Test-Set Rule

- Held-out test is for final reporting only.
- Do not use held-out test to choose model, hyperparameters, checkpoint,
  architecture, or fusion mode.

## Next Stage

- After the stable Only-GVP anchor: compare Only-ESM and GVP + late fusion.
- RING should be a later small side ablation, not mixed into the first
  ESM/fusion comparison.

## Caveats

- Exact hyperparameters and result numbers must be parsed from raw notebook
  outputs or saved configs before being used.
- Candidate trial labels such as trial7, trial12, and trial13 are current
  orientation hints, not proof by themselves.
- If parsed outputs do not support one of those candidates, drop it from the
  next batch.

## Update Checklist

After each real batch:

- Update current stage.
- Update raw evidence source.
- Update selected anchor, if any.
- Update next planned batch.
- Update caveats.
- Move old or superseded notes to `experiment_notes.md` if needed.
