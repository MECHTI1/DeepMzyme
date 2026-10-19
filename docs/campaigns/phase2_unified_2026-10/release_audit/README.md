# Metal ensemble release audit

This 2026-10-19 audit re-checks the Phase 2 metal ensembles from saved development artifacts: it rebuilds their
out-of-fold (OOF) scores, repeats the ensemble choice inside cross-validation, replays the predictor on development
sites and times CPU inference. Nothing was trained, refit or calibrated, and no test-set file was read. Most model
weights are not public.

## Ensembles

Every ensemble contains GVP-GNN + ESM-C late-fusion models trained with the recipes of the
[metal completion campaign](../../metal_completion_2026-10/README.md): B is the four-class baseline, P uses
population-normalized class weights, S adds label smoothing 0.1, and `trunklr1e4` is B with a GVP-trunk learning rate
of 1e-4. The 30-model averages P 5-class (three seeds), `trunklr1e4`, B 6-class and S 5-class; the 20-model is the
same with one P seed, and the 25-model drops S 5-class. The 55-model is a weighted average of 35 late-fusion
checkpoints (the 30-model plus a five-class P recipe with `trunklr1e4`), 10 Only-ESM and 10 Only-GVP checkpoints.

## Out-of-fold comparison

[`comparison.csv`](comparison.csv) and [`audit_summary.json`](audit_summary.json) rebuild pooled four-class balanced
accuracy (BA) on 7,398 ion sites; each checkpoint scores only the fold it did not train on. Differences are paired.
The 95% intervals resample the 2,208 near-copy groups (chains linked by 90% or higher sequence identity) that define
the folds; the summary also gives a fold-level bootstrap.

| Ensemble | Checkpoints | Checkpoints per OOF site | Pooled OOF BA | vs 55-model, pp [95% CI] | Similarity bar |
|---|---:|---:|---:|---:|:---:|
| Single model, B 4-class¹ | 1 | 1 | 74.34% | −2.83 [−4.55, −1.25] | no |
| `trunklr1e4` | 5 | 1 | 75.51% | −1.66 [−2.97, −0.38] | no |
| P 5-class, three seeds | 15 | 3 | 75.82% | −1.35 [−2.49, −0.21] | no |
| Late fusion, 20-model | 20 | 4 | 76.34% | −0.83 [−1.93, +0.32] | no |
| Late fusion, 25-model | 25 | 5 | 76.62% | −0.55 [−1.61, +0.51] | yes |
| Late fusion, 30-model | 30 | 6 | 76.81% | −0.35 [−1.36, +0.63] | yes |
| 55-model reference | 55 | 11 | 77.17% | — | — |

¹ The full-data refit has no OOF score; this row is the five-fold score of the same recipe.

The similarity bar (one-sided 95% lower bound ≥ −1.5 pp, no class recall more than 3 pp below the 55-model) was set
after these results were seen. It describes the results and does not select a model.

## Decision

[Plan §7](../../../../Plan.md#7-baseline-first-model-comparison-policy) moves to a more complex model only when
validation evidence justifies it. Here a step is justified when the paired 95% interval of the larger ensemble's gain
over the current choice excludes zero. `simplicity_rule` in the summary walks this rule from both five-checkpoint
models. The primary reading uses the near-copy-group interval (`related_group_interval`), which respects related-site
dependence; the fold-level interval rests on only five folds.

- Near-copy-group interval: from B the walk moves to the 15-model (P 5-class, three seeds) and then the 25-model;
  from `trunklr1e4` it moves to the 20-model. Neither walk continues.
- Fold-level interval: both walks move to the 20-model and then the 30-model. Requiring both intervals stops at the
  20-model.

No reading reaches the 55-model, whose gain over the 20-, 25- and 30-model ensembles neither interval supports. It
stays the best-CV reference and the predictor's code default, and no deployment default is chosen yet; the compact
candidates are the 20-, 25- and 30-model ensembles. The default should be chosen and frozen on OOF evidence; an
independent cohort can then evaluate it once.

[`nested_selection.json`](nested_selection.json) repeats the ensemble choice inside cross-validation, selecting on
four folds and scoring the fifth, from 18 late-fusion, 3 Only-ESM and 6 Only-GVP recipes with complete OOF. Nested
Top-N late fusion (N ≤ 6) scores 76.56% against the fixed 30-model's 76.81% (−0.25 pp [−0.91, +0.39]); nested
selection of the 55-model's three blocks scores 76.54% against 77.17% (−0.63 pp [−1.53, +0.26]). Late fusion keeps
its level under OOF-only selection. The 55-model's extra margin is not reproduced: on top of nested late fusion, the
Only-ESM and Only-GVP blocks add −0.02 pp [−1.16, +1.08]. This check has three limits. The selection folds' OOF
predictions come from models that trained on the scored fold, the usual approximation. The procedure (Top-N by OOF
rank, recipe-equal blocks, a weight grid that includes the historical 0.95) was specified after the historical
results were known, so it re-specifies the choice rather than replaying it. Multi-seed recipes average more
checkpoints per site, which raises their rank.

Limits:

- **Membership may be test-informed.** The 20-, 25- and 30-model recipes are the three or four highest full-test
  recipes before the Unified run in the [campaign log](../log.md). By OOF, P 5-class, `trunklr1e4`, B 6-class and
  S 5-class rank 1st, 4th, 7th and 15th of 18 late-fusion recipes (`families` in
  [`nested_selection.json`](nested_selection.json)). Nested late fusion reaches a similar level from OOF alone, so
  the evidence supports the late-fusion level rather than this particular membership.
- **OOF favours larger ensembles.** They average more checkpoints per site, while deployment averages every fold's
  checkpoints. OOF calibration error (ECE: 55-model 0.053, 30-model 0.075, 20-model 0.082, 25-model 0.126) depends
  on composition as well as size and is a rough guide only.

## Prediction parity and cost

Parity and timing were recorded with the first catalog, before the 15- and 25-model entries were added. Of the source
files they hash, only `catalog.py` and `manifest.py` have changed since; `predict.py` has not. The manifest, fixtures
and raw outputs they cite stay private, and the summaries keep their SHA-256 hashes.

[`parity_summary.json`](parity_summary.json): six sites in five development structures reproduced the saved
validation predictions of all 55 reference checkpoints (66 comparisons) with identical classes and a maximum
probability difference of `6.85e-8` (tolerance `1e-5`).

[`benchmark_summary.json`](benchmark_summary.json): median wall time for one site with cached embeddings, in separate
CPU processes (4 threads, Intel Core i7-1165G7), including checkpoint verification and loading:

| Ensemble | Checkpoints | Seconds | Peak RAM, GiB |
|---|---:|---:|---:|
| Single model (full-data refit) | 1 | 4.46 | 0.70 |
| `trunklr1e4` | 5 | 5.85 | 0.74 |
| Late fusion, 20-model | 20 | 8.27 | 0.85 |
| Late fusion, 30-model | 30 | 11.36 | 0.92 |
| 55-model reference | 55 | 16.72 | 1.14 |

With ESM-C embeddings computed in the same call, the 55-model took 43 s and 5.2 GiB.

## Reproducing

`bind` writes a manifest of local checkpoint paths and SHA-256 hashes; the other commands read it. Only 5 of the 55
checkpoints are public, so these commands need your own training runs (`python src/metal_release_cli.py --help`):

```bash
python src/metal_release_cli.py bind --run-root <run-root> --fold-membership <fold_membership.csv> --output <release-manifest.json>
python src/metal_release_cli.py audit --manifest <release-manifest.json> --output-dir <new-dir>
python src/metal_release_cli.py nested-selection --manifest <release-manifest.json> --run-root <run-root> --output <nested_selection.json>
python src/metal_release_cli.py predict --manifest <release-manifest.json> --structure <protein.pdb> \
  --atom A:402:_:ZN:_ --embedding-cache <esmc600m-cache> --output <result.json>
```

`predict` returns uncalibrated four-class scores for a site you supply (`--candidate` selects a smaller ensemble). It
assumes the site binds a metal and does not test whether it does. Earlier test-set use is recorded in
[DATASETS](../../../DATASETS.md#test-use-ledger).
