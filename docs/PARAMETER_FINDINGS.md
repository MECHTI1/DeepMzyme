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

## Target-scheme and cross-task evidence boundary

The intended primary metal reporting endpoint is Mn, Cu, Zn, and Class VIII
= Fe+Co+Ni. Completed direct-four evidence now includes the eight-run A1/A2
architecture screen, the four direct-four selected-LR repeats, the separate
15-run coordination-geometry comparison, and the 16-run matched RING comparison
summarized below. Historical six-class metal anchors retain that original
identity. Neither their native metrics nor post-hoc collapsed-four reports
are evidence from direct-four training, and their values must not be ranked
against the new direct-four scores.

The initial matched target-formulation pilot is complete across Only-GVP,
Only-ESM and graph-level late fusion: four/five/six targets, two seed-42 LRs
and native-selected-LR seed-43 repeats. This supplies Grade-3 evidence on one
validation split, not grouped-fold confirmation or target promotion. Compare
the common-four metric from each native-BA-selected checkpoint, with
probability aggregation before argmax. Both epoch and repeat-LR selection
retain their native metric; do not substitute a collapsed-score maximum.
Historical unmatched six-class maxima remain separate evidence.

Likewise, the historical joint Hybrid and Hybrid+RING runs show that a joint
code path was exercised in exploratory settings. They do not answer the
controlled EC-primary question “Does metal supervision improve EC prediction?”
because no matched EC-only versus EC-plus-auxiliary-metal comparison is indexed.
Do not infer auxiliary-learning benefit or promotion from their raw maxima.

## Current validation anchors and challengers

### EC1 standalone v12 campaign: initial fixed-split evidence

All twelve CARE30 EC1 standalone runs completed under the matched two-LR,
two-seed, 30-epoch protocol. Selection uses
`val_ec_group_level_1_balanced_acc`; held-out metrics were neither generated
nor used. Scores below are means and sample SDs across seeds 42/43 on the
same 42-protein validation partition. This is Grade 3, not promotion.

| Family | LR | Seed 42 | Seed 43 | Mean ± sample SD | Minimum |
|---|---:|---:|---:|---:|---:|
| Only-GVP | 3e-05 | 0.4190 | 0.3679 | 0.3935 ± 0.0362 | 0.3679 |
| Only-GVP | 0.0001 | 0.5560 | 0.5393 | 0.5476 ± 0.0118 | 0.5393 |
| Only-ESM | 3e-05 | 0.8393 | 0.9036 | 0.8714 ± 0.0455 | 0.8393 |
| Only-ESM | 0.0001 | 0.9821 | 0.9571 | 0.9696 ± 0.0177 | 0.9571 |
| GVP + late fusion | 3e-05 | 0.8571 | 0.8571 | 0.8571 ± 0.0000 | 0.8571 |
| GVP + late fusion | 0.0001 | 0.8571 | 0.9929 | 0.9250 ± 0.0960 | 0.8571 |


The largest observed mean is Only-ESM at `0.0001`. Retain it
as an initial standalone reference candidate, without inferring a general
architecture advantage or auxiliary-learning benefit. Every family received
the same LR, seed and epoch opportunities. Only-GVP retains zero seed-mean
recall for some active classes; late fusion at `3e-5` misses EC5 in both
seeds. EC5 and EC7 each have only one validation protein, limiting rare-class
conclusions. Grouped folds and paired confidence intervals remain absent.
See the [completed summary](notebook_outputs/summaries/summary_ec1_standalone_v12_20260914.md)
for per-class recalls, per-seed selected epochs and full provenance. That EC
campaign contributes no metal results; the separate metal evidence follows.

### Direct-four architecture screen: completed A1/A2

Namespaced ID: `metal/nonoverlap/architecture-pilot-continuation/2026-09-15`.
All eight full fits completed: four families × two learning rates, model seed
42, 50 epochs each. This is **Grade 5**, on one shared validation partition.
This is the initial discovery component of the completed 30-fit original pilot.
Neither this screen nor the geometry comparison uses held-out inference or
metrics.

The cohort comprises 1,181 training and 208 validation pockets, grouped into
1,151 and 110 PDB IDs with no overlap, from non-overlap PinMyMetal training
membership only. Split seed 42 and validation fraction 0.15 are fixed.
Each checkpoint is selected by native `val_metal_balanced_acc`. The executed
recipe holds conservative features, the certified PROPKA overlay, radius-6
edges, extraction radius 10, all-residue pooling at cutoff 0, no explicit
metal nodes, no RING, and no augmentation fixed.

All BA values below are percentages. The last column refers to the
single-seed run at that family's selected learning rate.

| Family | LR 3e-5 BA % | LR 1e-4 BA % | Selected LR | Minimum class recall % |
|---|---:|---:|---:|---:|
| Only-GVP | 64.711 | 70.897 | 1e-4 | 34.375 |
| Only-ESM | 72.123 | 71.748 | 3e-5 | 59.375 |
| GVP + early fusion | 69.367 | 68.239 | 3e-5 | 46.875 |
| GVP + graph-level late fusion | 71.789 | 72.312 | 1e-4 | 56.250 |

Finding: learning-rate response differs across families. Moving from 3e-5
to 1e-4 changes BA by +6.186 points for GVP, −0.375 for ESM, −1.128 for
early fusion, and +0.524 for late fusion. This supports checking more than
one reasonable learning rate before dropping a family; it does not identify
a generally optimal range. GVP's BA improvement accompanies a Zn recall
decrease from 40.625% to 34.375%.

Late fusion's largest observed BA exceeds Only-ESM's by only 0.189 points.
There is no established family winner or promotion. Early fusion has now
been experimentally evaluated in a matched direct-four screen. Its best BA
is 1.530 points below GVP's best, so hybrid is deferred under the pilot's
prespecified priority gate. That scheduling result does not reject early
fusion or predict hybrid performance. Paired grouped-fold confidence
intervals and rare-class recall protection are still required for promotion.

Evidence: [continuation summary](notebook_outputs/summaries/summary_metal_architecture_pilot_continuation_20260915.md),
[completed A1/A2 screen](notebook_outputs/raw/metal_architecture_pilot_20260915/continuation/finalization/architecture_screen.csv),
[selected-checkpoint metrics](notebook_outputs/raw/metal_architecture_pilot_20260915/continuation/finalization/validation_results.json),
and [verified copy inventory](notebook_outputs/raw/metal_architecture_pilot_20260915/continuation/portable_copy_manifest.tsv).
The final aggregate includes both the screen and the completed repeats.

### Original target and architecture repeats: completed fixed-split evidence

All ten native-selected recipes completed training seeds 42/43. The table
reports **common-four BA mean ± sample SD, in percent**, at the LR selected
separately for each family/target by seed-42 native BA. Native metrics and
selected epochs remain in the exact analysis; do not compare native BA across
different target sizes. This is Grade 3 on the same validation proteins.

| Family | Direct-four | Five-class, common-four view | Six-class, common-four view |
|---|---:|---:|---:|
| Only-GVP | 72.073 ± 1.663 | 65.612 ± 5.640 | 66.832 ± 0.624 |
| Only-ESM | 74.342 ± 3.138 | 72.878 ± 6.490 | 72.650 ± 4.602 |
| GVP + graph-level late fusion | 72.437 ± 0.177 | 74.718 ± 2.486 | 72.300 ± 1.099 |

Direct-four early fusion has mean **65.926 ± 4.866%**, with a worst Zn recall
of 18.750%. The tested configuration does not support automatic hybrid
escalation; neither that result nor the scheduling gate rejects early/hybrid
architectures generally.

Findings:

- In the original architecture pilot, Only-ESM has the largest direct-four
  selected-recipe mean. Its advantage over GVP appears
  in both seeds, but its mean Cu and VIII recalls are lower by 20.000 and
  10.938 points. Late fusion's tiny seed-42 lead over ESM reverses in seed
  43; the initial maximum did not predict the largest two-seed mean.
- GVP's five/six selected recipes score below direct-four in both seeds.
  ESM's five-class difference reverses sign, while six-class ESM scores
  below direct-four in both seeds. No target formulation wins across families.
- Late-five exceeds late-four by 0.648 and 3.914 common-four BA points in
  the two seeds, a mean gain of 2.281 points. It is an exploratory challenger:
  mean VIII recall improves by 11.719 points, but mean Zn and Mn recalls
  decrease by 1.563 and 1.031 points. Late-six's difference reverses sign
  and its mean is nearly tied with late-four.
- These are selected-recipe comparisons. GVP-six and both late target
  contrasts use different selected LRs from their direct-four references;
  they do not isolate a target effect at a fixed LR. The complete seed-42
  screen separately shows target × LR interactions.
- Native-six Co recall remains weak: mean/worst recall is 7.692/7.692% for
  GVP, 11.538/7.692% for ESM, and 15.385/7.692% for late fusion, on only
  13 Co sites. This limits fine-metal claims, without invalidating the
  correctly evaluated coarse four-class endpoint.

The two seeds share all 208 validation sites and 110 PDB groups. Sample SD
does not measure uncertainty across unseen proteins and is not a paired
confidence interval. LR selection used seed 42, so these means remain
conditional on that choice. No model/target promotion, held-out evaluation,
grouped-fold confirmation or original-pilot case-level error pairing follows.

Evidence: [final two-seed analysis](notebook_outputs/raw/metal_architecture_pilot_20260915/continuation/analysis/original_final_two_seed_analysis.json),
[recall/metric CSV](notebook_outputs/raw/metal_architecture_pilot_20260915/continuation/analysis/original_final_two_seed_analysis.csv),
[selected-checkpoint hash audit](notebook_outputs/raw/metal_architecture_pilot_20260915/continuation/analysis/original_final_checkpoint_and_early_audit.json),
and [completed summary](notebook_outputs/summaries/summary_metal_architecture_pilot_continuation_20260915.md).

### Coordination-geometry pilot: completed bounded comparison

Namespaced ID: `metal/nonoverlap/coordination-geometry-pilot/2026-09-15`.
All 15 full 50-epoch fits completed: five direct-four Only-GVP arms at two
learning rates with seed 42, followed by seed 43 at each arm's selected rate.
All arms selected 1e-4. The selected-LR comparison is **Grade 3** across two
training seeds on the same 208 validation sites and 110 PDB groups; the
initial LR screen is Grade 5. A second training seed is not new-data
confirmation. No arm is promoted.

All five arms use identical input dimensions and generic node-type machinery.
A masks the eight added coordination-count/angle slots. It retains original
GVP geometry and four base site statistics: multinuclear flag, metal count,
and minimum/mean intermetal distances. It is a fresh matched control,
distinct from the legacy Only-GVP A1/A2 implementation. Its score must not
be used as a direct replication of the earlier GVP baseline.

BA and recall values are percentages. SD is the sample standard deviation
across the two training seeds at selected LR 1e-4; the last column is the
lowest class recall in either of those runs.

| Arm | Added features | Metal nodes | BA mean ± sample SD | Worst class recall |
|---|---|---|---:|---:|
| A | All added slots masked | No | 69.641 ± 1.652 | 40.625 |
| B | Candidate counts | No | 70.186 ± 0.487 | 34.375 |
| C | Candidate counts and angles | No | 70.007 ± 2.022 | 40.625 |
| D | Candidate counts | Yes | 68.865 ± 1.558 | 31.250 |
| E | Candidate counts and angles | Yes | 68.030 ± 2.051 | 15.625 |

Findings under this recipe:

- Increasing LR from 3e-5 to 1e-4 improves seed-42 BA by 7.6–10.4 points
  across all five arms. The arms differed by only 0.250 points at the lower
  LR. A narrow low-LR screen alone would therefore have obscured the larger
  training-parameter effect; higher BA still does not guarantee every class
  improves.
- B−A (counts) changes from +2.057 to −0.967 points between seeds. C−B
  (angles without metal nodes) changes from +0.906 to −1.264 points.
  Neither establishes a consistent incremental benefit. B's largest observed
  mean comes with lower worst-class recall than A.
- D−B (metal nodes with counts) is −0.564 and −2.078 points; E−D (angles
  with metal nodes) is −0.487 and −1.184. These observations support lower
  exploration priority for those exact configurations under the bounded
  recipe, not universal rejection of metal nodes or angular features.
- E's seed-43 Zn recall is 15.625% (5/32). Mean BA alone would conceal that
  weakness. Validation support is Mn 97, Cu 15, Zn 32, and Class VIII 64;
  direct-four results provide no separate Fe/Co/Ni recall.

Metal nodes also change connectivity and train-fitted edge normalization.
A/B/C share one normalization hash, and D/E share another. The node
contrasts therefore test that combined representation, connectivity, and
normalization change; C−B and E−D keep graph construction and normalization
fixed. Candidate counts and pooled angular summaries are heuristic features,
not certified coordination numbers or complete coordination-shape labels.
They omit waters/cofactors/noncanonical residues, can use a centroid
fallback, and pool angles across metal centers.

Verified selected-checkpoint predictions and paired errors add descriptive
detail, not an independent validation sample or a causal explanation. For
example, C−B's unchanged Zn recall in seed 42 hides four corrected sites and
four new errors. The two-seed means are conditional on LR selection using
seed 42; their sample SD is not a confidence interval for unseen proteins.

Evidence: [completed geometry summary](notebook_outputs/summaries/summary_metal_coordination_geometry_pilot_20260915.md),
[exact two-seed analysis](notebook_outputs/raw/metal_coordination_geometry_pilot_20260915/geometry_two_seed_analysis.json),
[normalization controls](notebook_outputs/raw/metal_coordination_geometry_pilot_20260915/geometry_normalization_controls.json),
[paired-case audit](notebook_outputs/raw/metal_coordination_geometry_pilot_20260915/geometry_paired_case_analysis.json),
and [verified copy inventory](notebook_outputs/raw/metal_coordination_geometry_pilot_20260915/portable_copy_manifest.tsv).

### Matched RING pilot: completed bounded comparison

Namespaced ID: `metal/nonoverlap/ring-pilot/2026-09-15`.
All four one-epoch smokes and 16 full 50-epoch fits completed, with verified
selected-checkpoint bindings, terminal capture and actual GPU teardown.
Only-GVP and graph-level late fusion each received RING off/on at both
LRs `3e-5`/`1e-4` and model seeds 42/43. Every control was freshly trained;
all learning rates remain in the report. This is **Grade 3** on one shared
validation partition, with no held-out evaluation or model promotion.

The direct-four, native-six-eligible cohort remains 1,181 training pockets /
1,151 PDB groups and 208 validation pockets / 110 PDB groups. Split seed 42,
`pdbid` grouping, validation fraction 0.15, conservative features, radius-6
edges, legacy site inputs and no augmentation remain fixed. Both edge arms
use geometry-derived shell annotations. Each checkpoint is selected by
`val_metal_balanced_acc` within its own complete 50-epoch history.

BA values are percentages; differences are percentage points, RING on minus
off. SD is the sample standard deviation across the two training seeds.

| Family | LR | Off mean ± SD | On mean ± SD | Δ seed 42 | Δ seed 43 | Mean paired Δ |
|---|---:|---:|---:|---:|---:|---:|
| Only-GVP | 3e-5 | 67.609 ± 4.098 | 69.034 ± 3.551 | +1.812 | +1.039 | +1.426 |
| Only-GVP | 1e-4 | 72.073 ± 1.663 | 72.459 ± 2.176 | +0.024 | +0.749 | +0.387 |
| GVP + graph-level late fusion | 3e-5 | 73.538 ± 2.475 | 73.538 ± 2.475 | 0.000 | 0.000 | 0.000 |
| GVP + graph-level late fusion | 1e-4 | 72.437 ± 0.177 | 72.437 ± 0.177 | 0.000 | 0.000 | 0.000 |

Findings under this recipe:

- Only-GVP's selected BA increases in all four matched pairs, but the mean
  Class VIII recall falls by 3.125 points at `3e-5` and 4.688 points at
  `1e-4`. Mean Zn recall increases by 6.250 and 4.688 points, respectively;
  that mean does not protect every seed. Cu recall stays 14/15 in every GVP fit.
- At `1e-4`, seed 42's +0.024-point BA difference accompanies Mn −9,
  Zn +4 and Class VIII −2 correctly classified sites. Seed 43 has Mn +12,
  Zn −1 and Class VIII −4; its minimum class recall falls from 46.875% to
  43.750%. These are differences in marginal counts, not identified
  case-level corrections or new errors.
- All four late-fusion pairs tie in selected BA and every selected class
  recall. This does not establish identical models or site predictions:
  the reviewed `3e-5`, seed-42 pair has different training losses in every
  epoch and different validation metrics in 22 epochs. No prediction export
  or inference replay was performed. The ties do not establish universal
  RING ineffectiveness or statistical equivalence.
- The complete RING input audit finds no added undirected pairs. Instead,
  34,573 training and 6,057 validation pairs already present in radius graphs
  receive interaction annotations. Node/site inputs and shared-radius geometry
  are unchanged, and all 16 full fits have identical fitted normalization
  objects. The observed intervention is existing-edge interaction features,
  with no topology expansion or on/off normalization difference. The general
  recipe permits added edges and separately fitted normalization; this cohort
  does not exercise those changes. Raw RING `Angle` is not consumed, and the
  experiment adds neither coordination-angle summaries nor metal nodes.

Validation support is Mn 97, Cu 15, Zn 32 and Class VIII 64. The same sites
and PDB groups occur in every run; within-PDB sites are not independent.
Two training seeds and their sample SD provide no new-data confirmation,
paired grouped-fold CI, or rare-class promotion guarantee. Direct-four
training provides no separate Fe/Co/Ni recalls. Stage 6 confirmation and its
paired-CI/rare-class gates remain outstanding; no architecture is promoted.

The new low-LR late-fusion seed-43 run was not an opportunity in the original
selected-LR-repeat pilot. Keep that original pilot's Only-ESM reference claim
scoped to its own compared recipes. Do not rank families using unmatched
historical maxima or pool these two LRs into an architecture score.

Evidence: [completed RING summary](notebook_outputs/summaries/summary_metal_ring_pilot_20260915.md),
[post-stop analysis](notebook_outputs/raw/metal_ring_pilot_20260915/analysis/ring_post_stop_analysis.json),
[paired metrics and seed summaries](notebook_outputs/raw/metal_ring_pilot_20260915/analysis/ring_post_stop_analysis.csv),
[whole-cohort input audit](notebook_outputs/raw/metal_ring_pilot_20260915/finalization/ring_input_audit.json),
and [verified terminal checkpoint bindings](notebook_outputs/raw/metal_ring_pilot_20260915/finalization/final_capture_receipt.json).

### Historical metal anchors

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

Finding: this remains the most clearly reproducible historical six-class
fixed-split ESM-only anchor.
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

## Required comparison conclusions still open

The direct four-class metal research plan requires four controlled conclusions
that the current evidence cannot yet support:

| Comparison | Current limitation |
|---|---|
| Direct four-class training vs six-class training with collapsed-four evaluation | Initial matched four/five/six screen and selected-LR repeats are complete; target preference depends on family/recipe. No grouped-fold paired CI or class-recall promotion gate is satisfied |
| Early vs late vs hybrid ESMC fusion | Matched direct-four early/late two-LR screens and selected-LR repeats completed; hybrid was deferred. No full three-way grouped-fold/seed comparison or paired CI establishes an advantage |
| Only-ESM (using ESMC) vs Only-GVP vs combined GVP+ESMC | Completed direct-four screen and selected-LR repeats provide Grade-3 evidence; no shared grouped-fold Stage 6 comparison or paired confidence interval exists |
| GVP with vs without RING | The completed direct-four, two-LR/two-seed GVP and late-fusion comparison supplies Grade-3 existing-edge annotation evidence, with no observed topology or on/off normalization difference. Shared grouped-fold paired CIs and rare-class promotion gates remain outstanding |

Treat numerical differences as hypotheses until the candidates use the same
dataset, label scheme, folds, active seeds, metric, and comparable training/HPO
budgets. Publication claims of an advantage require the Stage 6 paired-CI and
rare-class-recall policy in `Plan.md` and the metal playbook.

## Parameter-domain conclusions

| Domain | Evidence-supported statement | What must not be inferred |
|---|---|---|
| Learning rate | A1/A2's 3e-5→1e-4 change helped GVP by 6.186 BA points, with different responses in other families; the geometry pilot gained 7.6–10.4 points across all five seed-42 arms | Neither two-point screen establishes a universal best LR or range; per-class recall can worsen |
| Batch size | Historical anchors commonly used 8; Hybrid+RING trial 114 used 12 | Neither value is confirmed generally superior |
| Weight decay | Successful candidates span `0`, `1e-5`, `1e-4`, `0.001`, and trial 114's `3e-7` | Duplicate local results do not establish irrelevance |
| Class weighting | Inverse-frequency supports the ESM and late-fusion anchors; inverse-sqrt supported Only-GVP; effective-number appears in trial 114 | Cross-family/task comparisons cannot isolate weighting effects |
| Loss | Historical metal anchors generally use cross-entropy and zero smoothing; joint trial 114 uses metal/EC weights `2.0/0.25` | No matched study isolates loss or joint-weight effects |
| GVP capacity | Late-fusion trial 49 uses `256/32`, four layers, edge hidden 128; Only-GVP stability evidence favors smaller `128/32` candidates | Larger capacity is not confirmed better outside its model context |
| Radius | Radius 6 appears in selected Only-GVP/late-fusion candidates; radius-10 trial-13 variants were weaker in their confirmation | Radius 10 is not universally harmful |
| Fusion | Direct-four early has mean BA65.926%; late72.437%; the tested early recipe does not justify automatic hybrid escalation. Historical graph late outperformed one node-late variant | No general early/hybrid rejection, three-way promotion, or benefit from every fusion design is established |
| ESM | In the original architecture pilot, direct-four ESM mean 74.342% exceeds the selected late recipe's 72.437%; late's small first-seed lead reverses in seed 43. Historical six-class evidence remains separate | This is the original selected-recipe comparison; it does not rank later LR/seed opportunities. Mean advantage does not protect every class or supply grouped-fold promotion evidence |
| Target formulation | Late-five improves common-four BA over late-four in both seeds with class tradeoffs; GVP and ESM respond differently | No universal target winner; some selected target recipes use different LRs and native class weighting |
| Coordination counts/angles | B−A and C−B reverse direction across the two high-LR training seeds; E−D is negative in both | Added geometric summaries have no consistent benefit established by this bounded pilot; GVP already receives geometric inputs |
| Explicit metal nodes | D−B has lower BA in both high-LR seeds; metal-node arms have substantial Zn/other-class tradeoffs | This tests added representation, connectivity, and fitted edge normalization together, not topology alone or every metal-node design |
| RING | The matched GVP pairs show mean BA gains of 1.426/0.387 points at the two LRs, with Class VIII recall losses; all four late-fusion pairs tie in selected BA/recalls | Existing-edge annotation evidence on one split does not establish universal benefit, identical predictions, equivalence, or Stage 6 promotion; added-edge effects were not exercised |
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
- Do not generalize the bounded RING annotation findings beyond the matched
  recipe or interpret late-fusion metric ties as identical predictions or
  universal ineffectiveness. Stage 6 paired-CI and rare-class gates still apply.
- Do not call the geometry A control “no geometry” or replace it with the
  earlier legacy GVP run; its masked inputs and node-type machinery are matched
  to B–E.
- Do not describe seed-43 repeats on the same validation proteins as independent
  data confirmation, or report a higher BA without the class-recall tradeoffs.
- Do not treat the completed bounded architecture, geometry and RING pilots
  as proof that the broader hybrid comparison or grouped-fold promotion is complete.
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
