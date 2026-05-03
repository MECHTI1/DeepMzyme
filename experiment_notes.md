# DeepMzyme Experiment Notes

## Only-GVP metal_6_class learning-rate checks

These checks were run through the training notebook command-building/reporting path, using validation metal balanced accuracy for model selection. Held-out test metrics are reported for the selected checkpoints only. The result is specific to the Only-GVP `metal_6_class` baseline and should not be generalized to other model presets.

Artifacts are under `DeepMzyme_Data/notebook_outputs/runs/`. Generated checkpoints, run directories, CSVs, PNGs, and other notebook output artifacts should not be added to git.

### Experiment 1: 10 epochs

- Task: `metal_6_class`
- Model: `Only-GVP`
- Batch size: `8`
- Epochs: `10`
- Learning rates: `1e-4`, `3e-4`, `1e-3`
- Best by validation metal balanced accuracy: `1e-4`

### Experiment 2: 30 epochs

- Task: `metal_6_class`
- Model: `Only-GVP`
- Batch size: `8`
- Epochs: `30`
- Learning rates: `1e-5`, `3e-5`, `1e-4`, `3e-4`
- Selection metric: validation metal balanced accuracy
- Held-out test evaluation: enabled
- Best by validation metal balanced accuracy: `3e-5`

Selected `3e-5` metrics:

- Selected epoch: `25`
- Validation balanced accuracy: `0.5105`
- Held-out test accuracy: `0.4830`
- Held-out test balanced accuracy: `0.3971`
- Held-out test macro F1: `0.3612`
- Collapsed-4 balanced accuracy: `0.5389`

The `1e-4` run was slightly higher for held-out test macro F1 (`0.3651` versus `0.3612`), but `3e-5` was selected by the predefined validation metric and was stronger on held-out test accuracy, held-out test balanced accuracy, and collapsed-4 balanced accuracy. The selected `3e-5` checkpoint was epoch 25 of 30, so 30 epochs looked sufficient for this baseline check.
