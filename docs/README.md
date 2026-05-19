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
| Exact metal stage blocks | `METAL_TRAINING_PIPELINE_PLAYBOOK.md` |
| Exact EC stage blocks | `EC_TRAINING_PIPELINE_PLAYBOOK.md` |
| Notebook option meanings | `METAL_NOTEBOOK_CONFIGURATION_GUIDE.md` |
| Copied raw outputs | `notebook_outputs/raw/<model-family>/` |
| Copied run summaries | `notebook_outputs/summaries/` |
| Validation leaderboard | `notebook_outputs/summaries/LEADERBOARD.md` |
| Local live run outputs | `../DeepMzyme_Data/notebook_outputs/runs/` |

## Coordination Rules

- Do not use held-out test data before Stage 7.
- Stage 7 must use a fixed validation-selected configuration and a separate
  final-test output folder.
- Treat `METAL_LABEL_SCHEME` as part of the experiment identity. The default is
  six-class; five-class Co/Ni-grouped runs must use separate run/study names and
  must not be mixed with six-class evidence.
- Do not create a second notebook-output folder with a space in its name.
- When copying Drive/Colab evidence into the repo, add raw output under
  `notebook_outputs/raw/`, add a short summary under `notebook_outputs/summaries/`,
  then update `notebook_outputs/README.md`.
