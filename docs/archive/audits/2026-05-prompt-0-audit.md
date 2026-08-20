# Prompt 0 Audit Report

> Historical note: this report records the repository state at the time of the
> audit. Several recommendations may have been implemented or superseded later.
> For current policy and executable stage values, use `Plan.md`,
> `EXPERIMENT_STATUS.md`, and the task-specific training playbooks.

## A. Executive Summary


No source or documentation files were changed during the audit. The safest first changes are documentation-only: clean up documentation ownership drift, add missing Playbook output sections, and make stage naming/searchability consistent. After that, improve traceability by adding `active_run_config.json` / `active_run_config.md` and per-run command/config snapshots before touching ML behavior.


Main risks found:

- The Playbook owns exact stage values, but Stage 0, Stage 2A, and Stage 5A-F are missing explicit per-stage or per-substage expected-output lists.
- The Guide still contains some numeric "starting value" material that duplicates Playbook territory.
- The notebook/training code writes `run_config.json` and `run_metadata.json`, but not `active_run_config.json` / `active_run_config.md`.
- Optuna runs subprocess trials and does not report intermediate values, so pruning is currently ineffective.
- Stage 6 repeats `seed`, which changes both initialization and validation split. It does not isolate initialization-seed variance from split variance.
- Stage 7 supports single-checkpoint and batch checkpoint evaluation, not a true deep ensemble, calibration, or bootstrap CI report.

## B. File-By-File Current State

| File / area | Current state | Risk / gap |
|---|---|---|
| `Plan.md` | Correctly declares Playbook as exact-value owner and forbids full Stage 0-7 blocks in Plan. Contains policy-level numeric references like 5-seed, 200-trial, and `0.01` advanced-fusion gate. | No full stage blocks, but some numeric policy values remain. Keep only if treated as governance, not executable config. |
| `AGENTS.md` | Contains behavioral instructions and required answer format. No full stage blocks. Includes exact safety values like `VAL_FRACTION = 0.15`, `SPLIT_BY = "pdbid"`. | Acceptable as agent guardrails, but not an executable config source. |
| `EXPERIMENT_STATUS.md` | Mutable status note. Current selected metal anchor is GVP + late fusion trial `49`; held-out test still unused. | Contains many run numbers and evidence references by design. Do not move into stable docs. |
| `docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md` | Main exact-value source. Contains Common Defaults, G4 policy, and blocks for Stage 0, 1, 2A, 2B, 3, 4, 5A-G, 6, 7. | Missing explicit expected outputs for Stage 0 and Stage 2A; Stage 5A-F lack per-substage output lists; 5G is override-only, not a full notebook block. |
| `docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md` | Correctly says Playbook wins for exact values. Explains workflow, options, outputs. | Duplicates numeric starting values in training hyperparameter and Optuna guidance sections. |
| `notebooks/DeepMzyme_training_colab.ipynb` | Supports planning, subprocess training, Optuna, seed-repeat, summaries, final-test preview/evaluation. Writes planned CSV/dictionary and Optuna study artifacts. | No `active_run_config.*`; no intermediate Optuna reporting; no hard study-level guard against reusing one study DB across model presets. |
| `src/train.py` | Thin parser/entry wrapper into `training.run.run_training`. | No direct issue found. |
| `src/training/config.py` | CLI supports LR schedules, folds, metal losses, class weights, joint loss weighting, ESM/RING options. | No flags for calibration, bootstrap CI, augmentation, or active notebook config export. |
| `src/training/run.py` | Writes `prepare_status.json`, `split_diagnostics.json`, `dataset_summary.json`, `run_config.json`, `run_metadata.json`, and `test_report.json` when test eval runs. | Per-epoch metrics only embedded in JSON `history`; no `val_metrics.csv`. Test report lacks calibration/CI. |
| `src/training/loop.py` | Computes class weights from labels; metrics include accuracy, balanced accuracy, macro F1, per-class recall. | No calibration/ECE or prediction export for bootstrap/ensembles. |
| `src/report_runs.py` | Summarizes run dirs from JSON artifacts into CSV/figure. | No bootstrap, paired tests, or calibration columns. Minor code smells visible but not changed. |
| `src/summarize_runs.py` | Missing. `wc`/`rg` reported no such file. | References to this file should be corrected or the file created in a future implementation. |

### Stage Coverage In Playbook

| Stage | Exact block | Expected outputs | Decision gate | Safety notes |
|---|---:|---:|---:|---:|
| Stage 0 | Yes | Missing explicit list | Yes | Yes |
| Stage 1 | Yes | Yes | Yes | Yes |
| Stage 2A | Yes | Missing explicit list | Yes | Yes |
| Stage 2B | Yes | Yes | Yes | Yes |
| Stage 3 | Yes | Yes | Yes | Yes |
| Stage 4 | Yes | Yes | Yes | Yes |
| Stage 5A | Yes | Common/implicit only | Yes | Yes |
| Stage 5B | Yes | Common/implicit only | Yes | Yes |
| Stage 5C | Yes | Common/implicit only | Yes | Yes |
| Stage 5D | Yes | Common/implicit only | Yes | Yes |
| Stage 5E | Yes | Common/implicit only | Yes | Yes |
| Stage 5F | Yes | Common/implicit only | Yes | Yes |
| Stage 5G | Override-only | Yes | Yes | Yes |
| Stage 6 | Overlay block | Yes | Yes | Yes |
| Stage 7 | Yes | Yes | Yes | Yes |

## C. ML Issue Table

| Issue | Current status | Risk | Recommended prompt |
|---|---|---|---|
| Optuna intermediate-value pruning | Missing | Wasted trials; pruning controls are misleading because no intermediate values are reported. | P3 |
| Stage 6 split/fold variance | Partial/confounded | Seed changes split and initialization together, so true split variance is not isolated. | P5 |
| Bootstrap CI gates for candidate promotion | Missing | Candidate promotion can overfit noisy validation differences. | P5 |
| Cosine LR schedule in Optuna search | Partial | CLI supports cosine, but Optuna does not sample LR schedule. | P4 |
| Hierarchical collapsed-4 metal loss | Missing | Collapsed metrics exist, but the loss is not hierarchical/collapsed-aware. | P7 |
| Multi-objective Optuna for balanced accuracy + minimum recall | Missing | Search cannot optimize balanced accuracy and rare-class protection jointly. | P4 |
| Stage 7 deep ensemble | Partial | Batch evaluation aggregates checkpoints but does not ensemble predictions. | P6 |
| ECE / class-wise ECE / temperature scaling | Missing | No calibration reporting or post-hoc calibration. | P6 |
| Test-set bootstrap confidence intervals | Missing | Final report lacks uncertainty intervals. | P6 |
| Position-noise augmentation | Missing | No robustness augmentation path. | P7 |
| Second-shell dropout | Missing | Shell roles exist, but no dropout augmentation uses them. | P7 |
| Wider Stage 2A seed list | Partial | Stage 2A uses 3 seeds; 5 seeds are only conditional if variance is high. | P1 |
| Batch size 32 search / memory policy | Partial | Documentation says `32` is separate ablation only; no automated memory policy. | P1 |
| ESM model variant metadata | Partial | ESM dimension/path are saved; ESM model name/checksum are not persisted. | P8 |
| Rare-class recall protection gates | Partial | Metrics/docs exist; no automatic gate. | P5 |
| Stage 3 random-search caveat | Implemented | Notebook warns when startup trials cover all trials. | P1 |
| Class weights computed from training fold only | Implemented | Uses `split.train_pockets` for class weights. | P4 |
| Joint-loss auto-weighting caution | Partial | Implemented and lightly documented; methodology caution is not central. | P8 |
| Sampler seed recording | Partial | Exposed/printed and stored in some user attrs, but not fully persisted as exact run/study metadata. | P4 |
| Pipeline design trade-offs documentation | Partial | Present, but scattered across Plan, Guide, Playbook, and status notes. | P1 |

## D. Missing Files / Sections

- Missing file: `src/summarize_runs.py`.
- Missing generated artifacts: `active_run_config.json`, `active_run_config.md`.
- Missing metric files: `val_metrics.csv`, `train_metrics.csv` writers.
- Missing Playbook sections:
  - explicit expected outputs for Stage 0;
  - explicit expected outputs for Stage 2A;
  - per-substage expected outputs for Stage 5A-F;
  - full standalone Stage 5G block if it is meant to be runnable alone.
- Missing hard Optuna study guard: persisted `MODEL_PRESET` / architecture / fusion compatibility check when resuming an existing storage URL.
- Missing final-report methods: calibration/ECE, temperature scaling, bootstrap CIs, and true ensemble prediction report.
- Missing Stage 6 methods: split/fold variance design, paired comparisons, bootstrap CI promotion gate, and automated rare-class recall gate.

## E. Recommended Implementation Order

1. **P1 - Documentation cleanup only.**
   Add Playbook stage-output gaps, reduce Guide numeric duplication, and make Stage 2A/5A headings and references consistent.

2. **P2 - Exact active configuration traceability.**
   Add `active_run_config.json` and `active_run_config.md` from the notebook/planning path, and embed exact command/notebook snapshot references into run dirs.

3. **P3 - Per-epoch metrics and Optuna pruning.**
   Write `val_metrics.csv` / `train_metrics.csv`, then wire Optuna `trial.report(...)` and `trial.should_prune()`.

4. **P4 - Optuna metadata and ranking discipline.**
   Add Optuna study compatibility metadata, sampler seed persistence, LR schedule search, and optional multi-objective or gated ranking.

5. **P5 - Stage 6 methodology.**
   Redesign Stage 6 to separate initialization seeds from split/fold variance and add candidate-promotion gates.

6. **P6 - Stage 7 reporting.**
   Add calibration, temperature scaling, bootstrap CIs, and true deep-ensemble support.

7. **P7 - Method experiments.**
   Add hierarchical collapsed-4 loss and structural augmentations such as position-noise and second-shell dropout.

8. **P8 - Reproducibility metadata.**
   Persist ESM model variant, package versions, data bundle SHA, git dirty state, and methodology caveats.

## Additional Audit Notes

- The notebook currently uses subprocess trials for Optuna via `subprocess.Popen`.
- `OPTUNA_TPE_MULTIVARIATE`, `OPTUNA_TPE_GROUP`, `OPTUNA_SAMPLER_SEED`, `OPTUNA_SPLIT_SEED`, and `OPTUNA_STORAGE` are exposed in the notebook.
- The notebook warns that pruning is configured-but-no-effect when enabled, because subprocess trials do not call `trial.report(...)`.
- The notebook uses the first planned base configuration for Optuna. It prints that Optuna is within one selected base `MODEL_PRESET`, but it does not hard-stop an existing persistent study DB from being reused with a different preset.
- `run_config.json` contains full training config, dataset summary, labels, normalization stats, selection metric, selected checkpoint, history, and optional embedded test report.
- `run_metadata.json` contains config, dataset summary, labels, normalization stats, selection metric, selected checkpoint, selected metric value, split identity, overlap fields, and optional embedded test report.
- `split_diagnostics.json` is written in each run directory during `prepare_run`.
- `test_report.json` is written only when held-out test evaluation is requested.
- `test_report.json` currently contains normal task metrics, collapsed-4 metal metrics, feature load report, split identity, overlap checks, and EC labels. It does not include calibration, ECE, temperature scaling, bootstrap intervals, or ensemble fields.
