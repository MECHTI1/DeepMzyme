# DeepMzyme Docs

This is the coordination page for documentation. It should stay short; exact
commands, budgets, search spaces, and results belong in the files below.

## Start Here

1. `../Plan.md` - research policy, split policy, validation selection, and
   held-out-test rules.
2. `../EXPERIMENT_STATUS.md` - current status, trusted anchors, caveats, and
   next planned action.
3. `METAL_TRAINING_PIPELINE_PLAYBOOK.md` or
   `EC_TRAINING_PIPELINE_PLAYBOOK.md` - exact notebook stage blocks and gates.
4. `notebook_outputs/README.md` - copied experiment evidence and reading order.

Use `METAL_NOTEBOOK_CONFIGURATION_GUIDE.md` only when you need notebook option
meanings, not exact stage values.

## Ownership

| Content | Owner |
| --- | --- |
| Policy and held-out-test rules | `../Plan.md` |
| Current mutable experiment status | `../EXPERIMENT_STATUS.md` |
| Current live notebook-default snapshot | `../README.md` |
| Exact metal stage blocks | `METAL_TRAINING_PIPELINE_PLAYBOOK.md` |
| Conservative first-pass GVP/HPO profile | `METAL_TRAINING_PIPELINE_PLAYBOOK.md` |
| Exact EC stage blocks | `EC_TRAINING_PIPELINE_PLAYBOOK.md` |
| Notebook option meanings | `METAL_NOTEBOOK_CONFIGURATION_GUIDE.md` |
| Copied raw outputs | `notebook_outputs/raw/<model-family>/` |
| Copied run summaries | `notebook_outputs/summaries/` |
| Validation leaderboard | `notebook_outputs/summaries/LEADERBOARD.md` |
| Local live run outputs | `../DeepMzyme_Data/notebook_outputs/runs/` |

## Coordination Rules

- Do not use held-out test data before Stage 7.
- After Stage 6/cross-validation, choose exactly one best configuration from
  validation evidence, run Stage 6B promotion gates, train/refit the final
  model with that frozen configuration, and only then launch Stage 7.
- Stage 7 must use the fixed Stage 6B final-refit run derived from the
  validation-selected configuration and a separate final-test output folder.
- Keep current notebook defaults, conservative first-pass GVP/HPO profiles,
  and canonical extended G4 HPO budgets distinct. Current defaults are a live
  launch surface; stage budgets and decision gates belong in the playbooks.
- Treat `METAL_LABEL_SCHEME`, `VAL_FRACTION`, and the effective selection metric
  as part of the experiment identity. The live notebook currently defaults to
  `METAL_LABEL_SCHEME = "five_class"`, `VAL_FRACTION = 0.18`, and
  `SELECTION_METRIC = "task_default"`; with the current `TASK = "joint"` launch
  surface, that resolves to `val_joint_balanced_acc`. Older six-class,
  `0.15`, or metal-selected validation evidence must use separate run/study
  names and must not be mixed with the current notebook-default evidence.
- Treat `METAL_NODE_MODE = "per_metal"` as part of the current notebook default
  configuration. It changes the graph/readout contract and should be recorded
  explicitly in summaries and run names when possible.
- Do not create a second notebook-output folder with a space in its name.
- When copying Drive/Colab evidence into the repo, add raw output under
  `notebook_outputs/raw/`, add a short summary under `notebook_outputs/summaries/`,
  then update `notebook_outputs/README.md`.
