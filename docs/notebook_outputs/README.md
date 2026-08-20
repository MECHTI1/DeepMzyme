# Notebook Output Evidence and Experiment Index

This is the authoritative index of meaningful DeepMzyme experiment batches and
the storage contract for copied run evidence. Current project state belongs in
[`EXPERIMENT_STATUS.md`](../../EXPERIMENT_STATUS.md); empirical parameter
conclusions belong in [`PARAMETER_FINDINGS.md`](../PARAMETER_FINDINGS.md).

## How to read the evidence

1. Find the experiment ID below.
2. Read its short summary.
3. Follow the raw/config link for exact commands, trials, trajectories, warnings,
   per-class values, and incomplete-state evidence.
4. Treat the recorded decision as the decision at that time unless current
   status explicitly promotes it.

Raw outputs describe what happened. They are not automatically the recipe for a
new run. Exact future metal and EC recipes remain in the task playbooks.

## Evidence grades

| Grade | Design |
|---:|---|
| 1 | Grouped folds × seeds with paired CI |
| 2 | Grouped folds |
| 3 | Fixed validation split across seeds |
| 4 | HPO discovery on one validation split |
| 5 | Single-seed validation |
| 6 | Exploratory, smoke, partial, or incomplete |
| 7 | Superseded historical evidence |

No copied model-family experiment currently has Grade-1 or Grade-2 confirmation.
Do not present fixed-split seed evidence as grouped-fold confirmation.

## Comparison identity

Never compare or rank rows without checking:

- task and selection metric;
- dataset/split;
- metal label scheme;
- node mode and feature requirements;
- validation fractions and fold definitions;
- seed count;
- epoch budget;
- model family and fusion mode.

Most May 2026 metal batches use the legacy non-overlap training root,
six-class labels, `pdbid` grouping, validation fraction `0.15`, and
`val_metal_balanced_acc`. Where copied evidence does not explicitly retain a
field, the row says so instead of guessing.

## Only-ESM batches

| Experiment ID | Dataset/labels | Validation design | Main result | Grade and historical decision | Test access | Evidence |
|---|---|---|---|---|---|---|
| `metal/only-esm/r1/original-anchor` | Legacy non-overlap; historical six-class | Five fixed-split seeds `42,123,2026,43,44`; 50 epochs | mean `0.625325230595`, SD `0.031449451169`; best single `0.6722436454687976` | 3 — retained Only-ESM anchor | No report present | [summary](summaries/summary_run_only_esm_round1_anchor_comparison.md); [raw](<raw/Only-ESM/Round1_results_only_esm_Optuna.output_cell_notebook>) |
| `metal/only-esm/r1/full-coverage-replication` | Same historical context | Five fixed-split seeds; 50 epochs | reproduces best single `0.6722436454687976` | 3 — supporting replication evidence | No report present | [summary](summaries/summary_run_only_esm_round1_full_coverage.md); [raw](<raw/Only-ESM/Round1_Rerun validation-only Only-ESM on full ESM coverage.output_cell_notebook>) |
| `metal/only-esm/r2/lr-wd-weight-grid` | Same historical context | Manual grid; 24/36 rows completed; seeds `42,123,2026` | best single `0.692962`; LR `5e-5` never ran | 6 — incomplete screen; did not replace anchor | No report present | [summary](summaries/summary_run_only_esm_round2_lr_wd_weight_screen.md); [raw](raw/Only-ESM/Round2_ESMonly.output_cell_notebook.md) |
| `metal/only-esm/r3/add-seeds43-44` | Same historical context | Adds seeds `43,44` to Round-2 finalists | combined evidence retains original `3e-5` inverse-frequency anchor | 3 when combined with R1/R2; this file alone is partial | No report present | [summary](summaries/summary_run_only_esm_round3_seed_confirmation.md); [raw](raw/Only-ESM/Round3_ESMonly_add_seeds43_44_5seed_confirmation.output_cell_notebook.md) |

## Only-GVP batches

| Experiment ID | Dataset/labels | Validation design | Main result | Grade and historical decision | Test access | Evidence |
|---|---|---|---|---|---|---|
| `metal/only-gvp/r1/inmemory-hpo16` | Legacy non-overlap; historical six-class | 16 trials, 20 epochs, seed 42, one split | trial 7 BA `0.554291323653437` | 4 — discovery candidate only | No report present | [summary](summaries/summary_run_only_gvp_round1_optuna_hpo.md); [raw](raw/Only-GVP/round1_results_onlyGVP_Optuna.output_cell_notebook) |
| `metal/only-gvp/r2/hpo-plus-seed-repeat` | Same | New in-memory study plus seed repeats `42,123,2026` | best completed repeat `0.64772364969639`; study-best number differs | 4/6 — mixed identity; candidate evidence | No report present | [summary](summaries/summary_run_only_gvp_round2_optuna_seed_repeat.md); [raw](raw/Only-GVP/round2_results_onlyGVP_Optuna.output_cell_notebook) |
| `metal/only-gvp/r3/top-candidate-confirm` | Same | Planned 30-run fixed-split comparison; mixed batch warning | best single `0.6559072690667597` | 6 — aggregate identity requires historical note | No report present | [summary](summaries/summary_run_only_gvp_round3_top_optuna_confirm.md); [raw](raw/Only-GVP/round3_results_onlyGVP_Optuna.output_cell_notebook) |
| `metal/only-gvp/r4/top3-plus-gvp3-30epoch` | Same | Ten visible 30-epoch fixed-candidate rows | best single `0.6239786072103145` | 6 — supporting short-budget evidence | No report present | [summary](summaries/summary_run_only_gvp_round4_top3_plus_gvp3.md); [raw](raw/Only-GVP/round4_results_onlyGVP_Optuna.output_cell_notebook) |
| `metal/only-gvp/r5/trial12-30epoch` | Same | Ten planned rows; five seeds; GVP2/GVP3 | best single `0.6215952953717157` | 6 — short-budget supporting evidence | No report present | [summary](summaries/summary_run_only_gvp_round5_trial12_batch.md); [raw](raw/Only-GVP/round5_Trial_12_batch.output_cell_notebook) |
| `metal/only-gvp/r5/trial13-30epoch` | Same | Ten planned rows; seed list not clear in summary; GVP2/GVP3 | best single `0.6316031249930583` | 6 — short-budget supporting evidence | No report present | [summary](summaries/summary_run_only_gvp_round5_trial13_batch.md); [raw](raw/Only-GVP/round5_Trial_13_batch.output_cell_notebook) |
| `metal/only-gvp/r6/three-config-50epoch` | Same | Three configs × five seeds `42,123,2026,43,44` | trial7/GVP4 best single `0.6559072690667597`; historical aggregate identity conflict remains | 3 — validation anchor comparison, not grouped folds | No report present | [summary](summaries/summary_run_only_gvp_round6_three_trial_comparison.md); [raw](raw/Only-GVP/round6_three_Trials_comparisons.output_cell_notebook.md); [history](../archive/experiments/metal_only_gvp_round3_history.md) |

## Graph-level and node-level fusion batches

| Experiment ID | Dataset/labels | Validation design | Main result | Grade and historical decision | Test access | Evidence |
|---|---|---|---|---|---|---|
| `metal/late-fusion/r1/trial12-anchor` | Legacy non-overlap; historical six-class | Five fixed-split seeds; 50 epochs | best single `0.6817577959565789` | 3 — early fixed candidate | No report present | [summary](summaries/summary_run_gvp_late_fusion_round1_trial12_anchor.md); [raw](<raw/GVP + late fusion/Round1_results_gvp_plus_latefusion_Optuna.output_cell_notebook>) |
| `metal/late-fusion/r1/full-coverage-replication` | Same | Five fixed-split runs; seeds not summarized in header | reproduces best single `0.6817577959565789` | 3 — supporting replication | No report present | [summary](summaries/summary_run_gvp_late_fusion_round1_full_coverage.md); [raw](<raw/GVP + late fusion/Round1_Rerun validation-only GVP + late fusion on full ESM coverage.output_cell_notebook>) |
| `metal/late-fusion/r2/esm-anchor-settings` | Same | Fixed five-seed comparison | mean `0.630750019320`; best single `0.6774914740431982` | 3 — supporting comparison | No report present | [summary](summaries/summary_run_gvp_late_fusion_round2_confirmed_esm_anchor.md); [raw](<raw/GVP + late fusion/Round2_late_fusion_from_confirmed_only_esm_anchor.output_cell_notebook.md>) |
| `metal/late-fusion/r3/inmemory-hpo50` | Same | 50 trials × 40 epochs, seed 42; generated repeats were one epoch | trial49 discovery BA `0.6750130535709283` | 4 for HPO; 6 for smoke repeats | No report present | [summary](summaries/summary_run_gvp_late_fusion_round3_optuna_50_v1.md); [raw](<raw/GVP + late fusion/Round3_late_fusion_optuna_50_v1.output_cell_notebook.md>) |
| `metal/late-fusion/r4/top3-50epoch` | Same | Trials 49/32/15 × five shared seeds; 50 epochs | means `0.635468206972`, `0.630032719914`, `0.629927684340` | 3 — trial49 selected historically; no grouped folds | No report present | [summary](summaries/summary_run_gvp_late_fusion_round4_top3_seedrepeat_50epoch.md); [recovered JSON](<raw/GVP + late fusion/metal_late_fusion_optuna_top3_seedrepeat_50epoch_v1/>); [captured manifest](<raw/GVP + late fusion/Round4_late_fusion_optuna_top3_seedrepeat_50epoch_v1.output_cell_notebook.md>) |
| `metal/node-late-fusion/r1/trial49-derived` | Same | Five fixed-split seeds; 50 epochs | mean `0.606599196822`, SD `0.023404449951` | 3 — rejected relative to graph-level trial49 | No report present | [summary](summaries/summary_run_gvp_node_level_late_fusion_round1_from_latefusion_trial49_seedrepeat_50epoch.md); [raw](<raw/GVP + node-level late fusion/Round1_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1.output_cell_notebook.md>) |

## Joint Hybrid batches

These use a joint-task metric or a different search context and are not ranked
directly against metal-only anchors.

| Experiment ID | Dataset/labels | Validation design | Main result | Grade and historical decision | Test access | Evidence |
|---|---|---|---|---|---|---|
| `joint/hybrid/r1/optuna-plus-top3` | Legacy batch context; exact dataset/label metadata incomplete | HPO plus trials 17/32/24 × seeds `42,123,2026`; mixed `debug_smoke` batch | trial17 joint `0.748343`; three-seed joint mean `0.697376` | 6 — exploratory; full config/search space missing | No report present | [summary](summaries/summary_run_hybrid_round1_optuna_plus_top3_seedrepeat.md); [raw](raw/Hybrid/Round1_hybrid_fusion_optuna_plus_top3_seedrepeat.output_cell_notebook.md) |
| `joint/hybrid-ring/r2/study-trials105-177` | Joint historical context; six-class metal metrics in log | Persistent-study continuation; completed trials 105–176, trial177 partial; seed 42 | trial114 metal BA `0.7303469775006777` | 5/6 — exploratory; RING causal effect inconclusive | No report present | [summary](summaries/summary_run_hybrid_ring_round2_optuna_50epoch_wide_v1_trials105_176.md); [raw](raw/Hybrid/Round2_joint_hybrid_ring_optuna_50epoch_wide_v1_trials105_176_partial_trial177.output_cell_notebook.md) |

## Historical test access and incomplete local work

| Experiment ID | Design | Result role | Grade | Held-out access | Evidence |
|---|---|---|---:|---|---|
| `metal/only-gvp/legacy-lr-epoch-test-sweep/2026-05-01-03` | Seven single-seed LR/epoch configurations on the non-overlap split | Historical access ledger only; test metrics excluded from parameter/model decisions | 7 | **Yes — seven reports, 352 pockets each** | [tracked raw package](raw/legacy_nonoverlap_test_access/); [archived notes](../archive/experiments/experiment_notes_legacy.md); [dataset ledger](../DATASETS.md#test-use-ledger) |
| `joint/hybrid-ring/fiveclass-local-smoke/prepare-only` | Planned one epoch, seed 42; preparation completed | No training result; older ESM metadata sidecars missing | 6 | No | [prepare status](raw/local_smoke/fiveclass_joint_hybrid_metal_target_fe1p7_mn1p5_seed42_1epoch/prepare_status.json) |

## Artifact completeness

- Late-fusion Round 4 now has 75 byte-identical JSON artifacts recovered from
  commit `783acae`. Thirty checkpoint binaries were not restored; their Git blob
  identities are recorded in
  [`MISSING_CHECKPOINT_GIT_BLOBS.tsv`](<raw/GVP + late fusion/metal_late_fusion_optuna_top3_seedrepeat_50epoch_v1/MISSING_CHECKPOINT_GIT_BLOBS.tsv>).
- Hybrid Round 1 full architecture/search-space configuration:
  **MISSING — recovery required**.
- Hybrid+RING Optuna database and trials 0–104:
  **MISSING — recovery required**.
- Older ESM embedding model identity:
  `unknown_in_older_embeddings`.
- Historical run records generally lack proof tying them to the current bundle
  checksum. Do not retroactively assign a modern bundle hash.

## Storage contract

- `raw/<family>/` stores exact copied outputs, run configs, metadata, and
  portable provenance. Do not shorten or rewrite these files.
- `summaries/` stores one human-readable account per meaningful batch.
- Historical snapshots and superseded narratives live under `docs/archive/`.
- Use canonical `docs/notebook_outputs/`; do not create a parallel directory
  with a space.
- New serious batches should preserve `run_config.json`, `run_metadata.json`,
  selection artifacts, split identity, seeds/folds, dataset bundle checksum,
  and exact raw-output path.
- If a value is unclear, write `Not clearly available in source file`.
- If an artifact is expected but absent, write `MISSING — recovery required`.
- Historical “Recommended Next Step” sections in immutable summaries describe
  the decision at that time; current action always comes from
  [`EXPERIMENT_STATUS.md`](../../EXPERIMENT_STATUS.md).
