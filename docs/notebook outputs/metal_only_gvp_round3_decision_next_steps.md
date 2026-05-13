# Metal Only-GVP Round 3 Decision Next Steps

> Status note, 2026-05-13: this note predates the newer Round 6 finalist
> comparison in `round6_three_Trials_comparisons.output_cell_notebook.md`. Use
> `EXPERIMENT_STATUS.md` for the current recommendation before launching any new
> notebook runs.

## Purpose

This note gives the immediate direction after reviewing the copied round3,
round4, and round5 notebook outputs. It is a short working note; `Plan.md`
remains the design authority and `EXPERIMENT_STATUS.md` remains the current
status summary.

## Evidence To Trust First

Use the 50-epoch confirmation batch from:

- `docs/notebook outputs/round3_results_onlyGVP_Optuna.output_cell_notebook`

Use the 30-epoch split batch from round4/round5 only as supporting evidence,
because it has a different epoch budget and lower aggregate validation metrics.

Do not use held-out test results for this decision.

## Candidate Ranking

| Candidate | Mean val balanced acc | Std | Min | Interpretation |
|---|---:|---:|---:|---|
| Trial7, `gvp_layers=4`, radius `6.0` | 0.6074 | 0.0424 | 0.5584 | Highest mean and best single run, but high variance. |
| Trial12, `gvp_layers=3`, radius `6.0` | 0.6071 | 0.0224 | 0.5671 | Nearly tied mean, more stable. Inspect first. |
| Trial12, `gvp_layers=2`, radius `6.0` | 0.5986 | 0.0204 | 0.5785 | Lower mean, best worst-seed robustness. |

Trial7 `gvp_layers=3` and Trial13 configs are currently secondary unless their
per-class diagnostics reveal a specific advantage.

## Decision Rule

1. Build a config-level diagnostic table for the 50-epoch batch with:
   validation balanced accuracy, macro-F1, min recall, per-class recall, and seed.
2. Prefer Trial12 `gvp_layers=3` if its rare-metal recall and macro-F1 are not
   clearly worse than Trial7 `gvp_layers=4`.
3. Pick Trial7 `gvp_layers=4` only if per-class diagnostics show a meaningful
   improvement that justifies the higher seed variance.
4. Pick Trial12 `gvp_layers=2` only if worst-seed or rare-class robustness is the
   main priority.

## After Anchor Selection

After the Only-GVP anchor is fixed by validation metrics:

1. Run held-out test evaluation once for final reporting of that fixed anchor.
2. Record the selected anchor and held-out metrics in `EXPERIMENT_STATUS.md`.
3. Then move to the next baseline-first stage: Only-ESM, then GVP + late fusion.
4. Keep RING as a later side ablation, not part of the first ESM/fusion stage.
