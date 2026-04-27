# AGENTS.md — DeepMzyme

## Project purpose

This repository develops DeepMzyme, a deep-learning framework for metalloenzyme metal-type prediction and EC/function prediction.

The main goals are:

1. Predict metal type from protein structural pocket graphs and ESMC embeddings/models.
2. Predict enzyme class / EC-level labels from protein structural pocket information and ESMC embeddings/models.
3. Compare model variants fairly using validation metrics and held-out test metrics.
4. Keep the code reproducible, simple to run, and suitable for publication-quality experiments.

## Environment

Always use this Python interpreter unless explicitly instructed otherwise:

```bash
/home/mechti/miniconda3/envs/DeepMzyme/bin/python
```
## Things to be aware
1) The current (at least at 27April 16:30 Israel time) script src/model.py may include the non-final or well defined or completely correct code. Be aware, if there is some misalignment between my Plan.md and the code there 
