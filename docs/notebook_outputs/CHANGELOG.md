# Notebook Output And Pipeline Changelog

## 2026-05-19

- Added `docs/README.md` as the documentation-folder entry point, including
  the run-order and folder-ownership map for validation, Stage 6 confirmation,
  and Stage 7 held-out testing.
- Re-pointed the main documentation indexes to that single coordination file.
- Removed stale historical path mentions with the old space-separated
  notebook-output folder name from copied run summaries; canonical copied
  evidence remains under
  `docs/notebook_outputs/`.

## 2026-05-18

- Documentation source-of-truth cleanup: `docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`
  is the owner of exact metal notebook stage values; `Plan.md`,
  `AGENTS.md`, and `docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md` now describe
  policy, behavior, and option meanings without owning full executable blocks.
- Stage 6 statistical update: grouped 5-fold confirmation by `pdbid` is the
  reportable default, with shared fold definitions, fixed split seed,
  paired-bootstrap comparison outputs, rare-class recall protection, and
  validation-only final-candidate promotion.
- Optuna pruning/search-space update: Optuna studies now persist study metadata
  including study/storage identity, model preset, task, selection metric,
  split/sampler seeds, pruning settings, and search-space hash; incompatible
  persistent-study reuse is guarded. Optional pruning monitors per-epoch metric
  CSVs and can terminate pruned subprocess trials.
- Final-test reporting update: Stage 7 supports predeclared single-checkpoint
  and `softmax_mean_5_seeds` reporting, calibration/ECE outputs,
  validation-fitted temperature scaling, bootstrap confidence intervals,
  prediction artifacts, and explicit no-test-selection policy metadata.
- Loss/objective update: optional collapsed-4 auxiliary metal loss and
  validation-only multi-objective Optuna review are implemented while keeping
  six-class metal metrics and `val_metal_balanced_acc` as the reportable
  selection path.
- Augmentation/metadata update: training-only position noise and second-shell
  dropout controls are available with defaults off; run artifacts now include
  active notebook config snapshots, per-epoch metric CSVs, and ESM embedding
  metadata summaries when available.
- Final consistency pass: notebook prose was aligned with the implemented
  pruning behavior so the markdown/comments no longer describe pruning as
  future-only.
