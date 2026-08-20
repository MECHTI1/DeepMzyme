# DeepMzyme Parameter and HPO Findings

This document is the authoritative synthesis of what validation and HPO have
taught the project about model and parameter choices. Exact run evidence remains
in [`notebook_outputs/`](notebook_outputs/), and historical narrative remains in
[`docs/archive/experiments/`](archive/experiments/).

This is not a table of live notebook values and not a replacement for the exact
future search spaces in the metal and EC playbooks.

## Evidence grades

| Grade | Evidence design | Permitted interpretation |
|---:|---|---|
| 1 | Grouped folds × seeds with paired CI | Strong promotion evidence under the declared folds/seeds |
| 2 | Grouped folds | Fold-confirmed, but model-seed uncertainty remains |
| 3 | Fixed validation split across seeds | Seed-stability evidence for that one split |
| 4 | HPO discovery on one validation split | Candidate discovery only |
| 5 | Single-seed validation | Exploratory direction only |
| 6 | Exploratory, smoke, partial, or incomplete | Diagnostic/context only |
| 7 | Superseded historical evidence | Preserve to avoid repetition; do not treat as current |

No copied model-family result currently qualifies for Grade 1 or 2. The leading
historical anchors are Grade 3 at best.

## Evidence exclusion: historical test access

The legacy non-overlap PinMyMetal test was opened in seven early runs. Its exact
metrics remain in the [test-use evidence package](notebook_outputs/raw/legacy_nonoverlap_test_access/)
and [archived notes](archive/experiments/experiment_notes_legacy.md).

Those test values are excluded from:

- the findings and recommendations below;
- model ranking and promotion/rejection;
- useful/harmful parameter conclusions;
- future HPO search-space choices.

The early LR batch contributes only its validation-side observation, explicitly
labeled Grade 5/7. See [`DATASETS.md`](DATASETS.md) for the access record.

## Current validation anchors and challengers

| Namespaced experiment/configuration | Main validation result | Grade | Current interpretation |
|---|---:|---:|---|
| `metal/only-esm/round1+round3/original-anchor/fixed-split-5seed` | mean `0.625325230595`, SD `0.031449451169`, min `0.5902137453861592`, max `0.6722436454687976` | 3 | Stable historical ESM-only anchor |
| `metal/late-fusion/round4/trial49/fixed-split-5seed` | mean `0.635468206972`, SD `0.043023727308`, min `0.597794518922`, max `0.688000505242` | 3 | Historical validation-leading metal anchor; not grouped-fold confirmed |
| `metal/node-late-fusion/round1/trial49-derived/fixed-split-5seed` | mean `0.606599196822`, SD `0.023404449951` | 3 | Tested node-level variant rejected relative to graph-level trial 49 |
| `joint/hybrid/round1/trial17/fixed-split-3seed` | joint mean `0.697376`; best single joint `0.748343` | 6 | Exploratory; different task/metric and incomplete full-config provenance |
| `joint/hybrid-ring/round2/trial114/single-split-seed42` | metal BA `0.7303469775006777` | 5 | Strong exploratory candidate only; no causal RING conclusion |

“Current anchor” here means the best preserved historical validation anchor,
not a final model and not a license to skip grouped-fold confirmation.

## Only-ESM

### Original five-seed anchor

Namespaced ID:
`metal/only-esm/round1+round3/lr3e-5-inverse-frequency/fixed-split-5seed`

Configuration:

- task `metal`, six-class label scheme;
- model `only_esm`;
- seeds `42,123,2026,43,44`;
- learning rate `3e-5`;
- weight decay `1e-4`;
- batch size `8`;
- classifier head layers `2`;
- inverse-frequency metal class weighting;
- cross-entropy loss;
- label smoothing `0`;
- 50 epochs;
- selection metric `val_metal_balanced_acc`.

Per-seed validation balanced accuracies:

`0.6722436454687976`, `0.6380923946579973`,
`0.6183848812144958`, `0.607691486249361`,
`0.5902137453861592`.

Finding: this remains the most clearly reproducible fixed-split ESM-only anchor.
Its best single seed must not replace the five-seed aggregate.

Evidence:

- [anchor summary](notebook_outputs/summaries/summary_run_only_esm_round1_anchor_comparison.md)
- [full-coverage summary](notebook_outputs/summaries/summary_run_only_esm_round1_full_coverage.md)
- [seed-confirmation summary](notebook_outputs/summaries/summary_run_only_esm_round3_seed_confirmation.md)

### Round-2 LR/WD/class-weight screen

Namespaced ID:
`metal/only-esm/round2/lr-wd-weight-manual-grid/fixed-split-partial`

- Intended rows: 36.
- Completed rows: 24.
- Learning rate `5e-5` was not run because of the cap.
- Best single result: `0.692962`, seed `2026`, LR `3e-5`, WD `1e-5`,
  inverse-sqrt-frequency weighting, selected epoch `44`.
- Some WD `1e-5`/`1e-4` rows have identical selected metrics.

Finding: the single inverse-sqrt-frequency result did not displace the original
five-seed inverse-frequency anchor. Identical WD outcomes are local
observations, not evidence that weight decay is generally irrelevant.
`5e-5` is untested here, not a negative result.

Evidence:

- [summary](notebook_outputs/summaries/summary_run_only_esm_round2_lr_wd_weight_screen.md)
- [raw output](notebook_outputs/raw/Only-ESM/Round2_ESMonly.output_cell_notebook.md)

Grade: 6 for the incomplete grid; individual rows are Grade 5.

## Only-GVP

Trial numbers below are namespaced because independent in-memory studies reused
`deepmzyme_controlled_hpo`.

### Round-1 discovery study

Namespaced study:
`metal/only-gvp/round1/deepmzyme_controlled_hpo/in-memory-16trial`

Search space:

- batch size `2,4,8`;
- hidden scalar `64,128,256`;
- hidden vector `8,16,32`;
- edge hidden `32,64,128`;
- radius `6,8,10`;
- GVP layers `2,4,6`;
- head layers `1,2,3`;
- LR `1e-5` to `3e-4`;
- WD `0,1e-5,1e-4,1e-3`;
- class weighting none, inverse-frequency, inverse-sqrt-frequency, or
  effective-number.

Best discovery trial:

- trial `7`;
- validation BA `0.554291323653437`;
- LR `6.464669746492395e-05`;
- WD `0.001`;
- batch `8`;
- hidden `128/32`;
- edge hidden `128`;
- four GVP layers;
- radius `6`;
- head layers `1`;
- inverse-sqrt-frequency weighting.

Grade: 4.

Evidence:
[summary](notebook_outputs/summaries/summary_run_only_gvp_round1_optuna_hpo.md)
and
[raw output](notebook_outputs/raw/Only-GVP/round1_results_onlyGVP_Optuna.output_cell_notebook).

### Later candidates and identity caveats

Namespaced candidates:

- `metal/only-gvp/round2/deepmzyme_controlled_hpo/trial13`: discovery BA
  `0.569839524736432`, LR `6.817779343845317e-05`, WD `0.001`,
  hidden `128/32`, edge hidden `128`, two layers, radius `10`.
- `metal/only-gvp/round2/deepmzyme_controlled_hpo/trial12`: LR
  `4.735385769610685e-05`, WD `0`, two layers, radius `6`.
- `metal/only-gvp/round1-or-round2/trial7`: later best single repeat
  `0.64772364969639`.

A separate Round-6/late-fusion “trial12” configuration uses LR
`4.752317377508605e-05`. It must not be silently merged with the actual HPO
trial-12 LR `4.735385769610685e-05`.

Historical five-seed, 50-epoch aggregates:

| Namespaced configuration | Mean | Sample SD | Interpretation |
|---|---:|---:|---|
| `only-gvp/trial7/gvp4/radius6` | `0.6074` | `0.0424` | Highest recorded mean among these six, but higher variance |
| `only-gvp/trial12/gvp3/radius6` | `0.6071` | `0.0224` | Nearly tied and more stable |
| `only-gvp/trial7/gvp3` | `0.6010` | Preserved in archive | Secondary ablation |
| `only-gvp/trial12/gvp2/radius6` | `0.5986` | Preserved in archive | Below GVP3 variant |
| `only-gvp/trial13/gvp2/radius10` | `0.5960` | Preserved in archive | Radius-10 family weaker in this comparison |
| `only-gvp/trial13/gvp3/radius10` | `0.5809` | Preserved in archive | Weakest of the six |

Round 6 reports a trial-7 mean `0.610711876419`, whereas the earlier historical
table reports `0.6074`. Treat these as separate batch identities until their
run membership is reconciled.

Negative knowledge: the tested radius-10/trial-13 family did not justify
promotion over the radius-6 finalists. This does not establish that radius 10
is universally harmful.

Evidence:

- [Round-2 summary](notebook_outputs/summaries/summary_run_only_gvp_round2_optuna_seed_repeat.md)
- [Round-6 summary](notebook_outputs/summaries/summary_run_only_gvp_round6_three_trial_comparison.md)
- [archived decision history](archive/experiments/metal_only_gvp_round3_history.md)

Grades: discovery results Grade 4; fixed-split seed aggregates Grade 3.

## GVP plus graph-level late fusion

### Round-3 discovery

Namespaced study:
`metal/late-fusion/round3/deepmzyme_controlled_hpo/in-memory-50trial`

Search design:

- 50 trials, 40 epochs per trial;
- HPO seed/split seed `42`;
- `pdbid` grouping and validation fraction `0.15`;
- batch `8`;
- LR `1e-5` to `1e-4`;
- WD `0,1e-5,1e-4,1e-3`;
- hidden scalar `128,256`;
- hidden vector `16,32`;
- edge hidden `64,128`;
- radius `6,8`;
- GVP layers `2,3,4`;
- head layers `1,2,3`;
- ESM fusion dimension `64,128,256`;
- inverse-frequency or inverse-sqrt-frequency weighting.

| Trial | Discovery validation BA | LR | WD | Hidden S/V | Layers | Edge hidden/radius | Head/fusion |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `49` | `0.6750130535709283` | `1.6801503587890522e-05` | `1e-5` | `256/32` | 4 | `128/6` | `1/64` |
| `32` | `0.6585119076580177` | `5.4715836015281065e-05` | `0.001` | `128/32` | 2 | `128/6` | `1/64` |
| `15` | `0.6550963478857217` | `7.032630334240692e-05` | `0.001` | `128/32` | 2 | `128/6` | `1/64` |

All three used inverse-frequency weighting. Trial 49 selected epoch `37`.

The generated top-K repeats in this Round-3 output used one epoch. Their low
means are Grade-6 smoke evidence and must not be used to reject the candidates.

Evidence:
[summary](notebook_outputs/summaries/summary_run_gvp_late_fusion_round3_optuna_50_v1.md)
and
[raw output](<notebook_outputs/raw/GVP + late fusion/Round3_late_fusion_optuna_50_v1.output_cell_notebook.md>).

### Round-4 fixed-split confirmation

Namespaced batch:
`metal/late-fusion/round4/top3-50epoch/fixed-split-5seed`

Seeds: `42,123,2026,43,44`.

| Trial | Mean | Sample SD | Min | Max |
|---:|---:|---:|---:|---:|
| `49` | `0.635468206972` | `0.043023727308` | `0.597794518922` | `0.688000505242` |
| `32` | `0.630032719914` | `0.051725380573` | `0.550812772796` | `0.686374815178` |
| `15` | `0.629927684340` | `0.052550289215` | `0.563584164877` | `0.699275572298` |

Finding: trial 49 was selected historically by mean, standard deviation, and
worst-seed result. Trial 15, not trial 49, produced the largest single run
(`0.6992755722978847`). The aggregate difference is modest and remains
fixed-split-only evidence.

Evidence:

- [summary](notebook_outputs/summaries/summary_run_gvp_late_fusion_round4_top3_seedrepeat_50epoch.md)
- [recovered exact JSON artifacts](<notebook_outputs/raw/GVP + late fusion/metal_late_fusion_optuna_top3_seedrepeat_50epoch_v1/>)

Grade: 3.

## Node-level late fusion

Namespaced batch:
`metal/node-level-late-fusion/round1/trial49-derived/fixed-split-5seed`

- Trial-49-derived fixed configuration.
- Seeds `42,123,2026,43,44`.
- 50 epochs.
- Mean `0.606599196822`.
- Sample SD `0.023404449951`.
- Min `0.574873163235`.
- Max `0.633163185699`.

Finding: the tested node-level fusion did not replace graph-level late fusion
trial 49 and fell below the Only-ESM historical mean. Repeating the identical
configuration is not justified without a specific methodological change.

This conclusion applies to this tested implementation/configuration; it does
not establish that every node-level fusion design is inferior.

Evidence:
[summary](notebook_outputs/summaries/summary_run_gvp_node_level_late_fusion_round1_from_latefusion_trial49_seedrepeat_50epoch.md)
and
[raw output](<notebook_outputs/raw/GVP + node-level late fusion/Round1_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1.output_cell_notebook.md>).

Grade: 3.

## Hybrid fusion

Namespaced study:
`joint/hybrid/round1/optuna-plus-top3/debug_smoke`

- Task `joint`.
- Selection metric `val_joint_balanced_acc`, not the metal-only anchor metric.
- Best single trial `17`: joint `0.748343`, metal `0.672077`, collapsed-4
  `0.733259`, LR approximately `3.975e-5`, WD `1e-5`.
- Three-seed joint means: trial 17 `0.697376`, trial 32 `0.688209`, trial 24
  `0.686984`.
- The copied batch has mixed/missing batch warnings.
- Full architecture settings and the HPO search space are absent from the
  tracked artifact; only Drive paths survive.

Finding: this is promising exploratory joint-task evidence, but it is not
directly comparable to metal-only anchors and is not fully reproducible from
the repository.

Evidence:
[summary](notebook_outputs/summaries/summary_run_hybrid_round1_optuna_plus_top3_seedrepeat.md)
and
[raw output](notebook_outputs/raw/Hybrid/Round1_hybrid_fusion_optuna_plus_top3_seedrepeat.output_cell_notebook.md).

Grade: 6.

## Hybrid plus RING

Namespaced study:
`joint/hybrid-ring/round2/joint_hybrid_ring_optuna_50epoch_wide_v1/trials105-177`

The copied continuation contains completed trials `105` through `176`; trial `177`
begins but is incomplete. The “120 trials” setting is a target on an
existing persistent study, not proof of a new independent 120-trial batch.

Best copied trial `114`:

| Parameter | Value |
|---|---:|
| `val_metal_balanced_acc` | `0.7303469775006777` |
| selected epoch | `37` |
| learning rate | `3.705631497756492e-05` |
| weight decay | `3e-7` |
| batch size | `12` |
| hidden scalar/vector | `320/16` |
| GVP layers | `4` |
| edge hidden | `192` |
| radius | `7` |
| head layers | `2` |
| ESM fusion dimension | `256` |
| early ESM dimension/dropout | `48/0.05` |
| metal/EC loss weights | `2.0/0.25` |
| class weighting | `effective_number` |

Previous study best trial `84` was `0.725445016716364`. Several trials are
near-tied at exactly `0.7298725941989699`.

Findings:

- Trial 114 is a high single-seed validation result, not a confirmed best
  model.
- The Optuna SQLite database and trials `0–104` are not tracked.
- RING's causal contribution is **inconclusive** because there is no matched
  no-RING control with the same task, search space, folds, seeds, and metric.

Evidence:
[summary](notebook_outputs/summaries/summary_run_hybrid_ring_round2_optuna_50epoch_wide_v1_trials105_176.md)
and
[raw continuation](notebook_outputs/raw/Hybrid/Round2_joint_hybrid_ring_optuna_50epoch_wide_v1_trials105_176_partial_trial177.output_cell_notebook.md).

Grade: 5 for completed individual trials; Grade 6 for the incomplete study
record and causal RING question.

## Parameter-domain conclusions

| Domain | Evidence-supported statement | What must not be inferred |
|---|---|---|
| Learning rate | Useful candidates occurred at `1.6801503587890522e-05`, `3e-5`, and `3.705631497756492e-05` in different families/tasks | No universal best range has been grouped-fold confirmed |
| Batch size | Historical anchors commonly used 8; Hybrid+RING trial 114 used 12 | Neither value is confirmed generally superior |
| Weight decay | Successful candidates span `0`, `1e-5`, `1e-4`, `0.001`, and trial 114's `3e-7` | Duplicate local results do not establish irrelevance |
| Class weighting | Inverse-frequency supports the ESM and late-fusion anchors; inverse-sqrt supported Only-GVP; effective-number appears in trial 114 | Cross-family/task comparisons cannot isolate weighting effects |
| Loss | Historical metal anchors generally use cross-entropy and zero smoothing; joint trial 114 uses metal/EC weights `2.0/0.25` | No matched study isolates loss or joint-weight effects |
| GVP capacity | Late-fusion trial 49 uses `256/32`, four layers, edge hidden 128; Only-GVP stability evidence favors smaller `128/32` candidates | Larger capacity is not confirmed better outside its model context |
| Radius | Radius 6 appears in selected Only-GVP/late-fusion candidates; radius-10 trial-13 variants were weaker in their confirmation | Radius 10 is not universally harmful |
| Fusion | Graph-level late fusion outperformed the tested node-level trial-49-derived variant | Other node-fusion designs remain untested |
| ESM | Only-ESM is a strong fixed-split anchor; graph late fusion gives a modest fixed-split mean improvement | Neither has grouped-fold promotion evidence |
| RING | Trial 114 is promising | Causal benefit is unestablished |
| Regularization/augmentation | Current records contain candidate values but no clean matched confirmation | Do not claim dropout/noise settings helped or hurt without new evidence |

## Settings and conclusions not to repeat incorrectly

- Do not use a single Optuna trial as a confirmed best model.
- Do not interpret the one-epoch late-fusion repeats as rejection evidence.
- Do not repeat the identical tested node-level-fusion configuration as a
  promotion candidate without a stated methodological reason.
- Do not call Only-ESM LR `5e-5` harmful; it did not run in Round 2.
- Do not merge reused trial numbers across studies or the two trial-12 LR
  identities.
- Do not compare joint `val_joint_balanced_acc` directly with metal-only
  `val_metal_balanced_acc`.
- Do not claim RING benefit without a matched ablation.
- Do not promote trial 49, trial 17, or trial 114 as grouped-fold confirmed.
- Do not use the historical PinMyMetal test metrics to support any parameter
  statement.

## Provenance gaps affecting parameter confidence

- Hybrid Round-1 full architecture/search-space configs:
  **MISSING — recovery required**.
- Hybrid+RING Optuna SQLite database and trials `0–104`:
  **MISSING — recovery required**.
- Older ESM embedding sidecars/model identity:
  `unknown_in_older_embeddings`.
- Checkpoint binaries for recovered late-fusion Round 4:
  omitted; JSON configs/metadata are restored and checkpoint Git blob identities
  are recorded.

See [`FOLLOW_UP_TECHNICAL_ISSUES.md`](FOLLOW_UP_TECHNICAL_ISSUES.md) for the
open recovery and workflow tasks.
