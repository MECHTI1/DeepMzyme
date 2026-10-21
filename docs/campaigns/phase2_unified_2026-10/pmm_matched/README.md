# Matched PinMyMetal comparison (2026-10-20)

Part of [Phase 2](../README.md). Pre-registered comparison of DeepMzyme with PinMyMetal's released metal-type
classifier, retrained on the same training sites and folds and scored on the same 1,488 test rows. Test label:
`both_results_secondary` (opened, non-pristine test; [test-use ledger](../../../DATASETS.md#test-use-ledger)).

**Design.** A pre-registration (SHA-256 `f3a16ff53edda3ab57fc71a580045b3a62a7d1b835810bde9901f8d2be6936ba`) fixed the
arms, lines, statistics and wording before any PinMyMetal model was fit. PinMyMetal's released classifier recipe
(soft-voting LR, balanced random forest, MLP, SVC and EasyEnsemble; released hyperparameters) was refit in its pinned
environment (Python 3.11.5, scikit-learn 1.3.0, imbalanced-learn 0.11.0) on the same 7,398 training sites and five
folds as DeepMzyme, then fit once on all 7,398 sites and applied label-blind to the 1,488 test rows. PMM-D drops
`valence_3a`, `valence_4a`, `vecsum_3a` and `vecsum_4a` (89 features): at deployment PinMyMetal writes every candidate
site as Zn, so these are Zn-based there but computed with the true metal in the released tables (verified for
`valence_3a`, TECH-028). PMM-R keeps all 93 published
features. Test intervals are paired PDB-entry bootstraps (10,000 resamples, 1,179 entries, the same draws for every
system); CV balanced-accuracy intervals are paired fold bootstraps and t intervals (df 4); CV recall intervals are
near-copy-group bootstraps (2,208 groups, 10,000 resamples). An independent recomputation from the frozen inputs
reproduced every pre-registered, amendment A1 and amendment A2 value (2,534 of 2,534, bit-identical), and every table
cell on this page matches it.

| Comparison | Test (full, 1,488 rows) | Cross-validation (mean over folds) |
|---|---|---|
| Single model − PMM-D (pre-registered, 97.5% intervals) | 67.55% vs 58.27%: **+9.28** [3.59, 15.10] | 74.28% vs 63.03%: **+11.25** [6.25, 15.36]; t [2.88, 19.62] |
| Single model − PMM-R (95%) | +4.14 [−1.12, 9.00]; Mn −23.4 [−32.9, −13.9] | +7.09 [2.42, 11.50]; t [0.14, 14.05]; Mn −6.0 [−12.9, −0.5] |
| Pre-declared CV-only ensemble (greedy, 30 checkpoints) − PMM-D (95%) | 70.57% vs 58.27%: +12.30 [7.62, 16.94] | nested 76.48% vs 63.03%: +13.45 [8.46, 18.28]; t [5.63, 21.28] |
| Pre-declared CV-only ensemble − PMM-R (95%) | +7.16 [2.35, 11.58]; Mn −12.6 [−21.1, −4.4] | +9.29 [4.27, 14.20]; t [1.30, 17.28]; Mn −2.9 (pooled, no interval) |
| Single model − gradient-boosting ablation (97.5%) | −1.28 [−6.28, 3.65] | +3.06 [−2.43, 6.56]; t [−5.99, 12.11] |

- **Result.** Under matched retraining, DeepMzyme outperformed PinMyMetal's released classifier with
  deployment-available features, on the test set and in cross-validation. This rests on amendment A1; the
  pre-registered criterion was not met (see Amendments). All 15 ensembles chosen by cross-validation
  alone (Top-N by out-of-fold rank with all seeds or seed 42, N = 1–6; the greedy ensemble; B four-class with one and
  three seeds) scored 68.83–71.75% on the full test, above PMM-D by 10.56–13.48 pp and above PMM-R by 5.42–8.34 pp
  (unadjusted paired 95% intervals above zero), with lower Mn recall than PMM-R in every case.
- **Published values (unpaired context).** These ensembles lie above PinMyMetal's published 67.85% by point estimate
  (+0.98 to +3.90 pp), but every two-sided 95% interval includes it; the single model is 0.30 pp below it. PMM-R did not
  reproduce the published value on the same test (63.41%), and the cause was not determined. On the project's
  90%-sequence-identity folds DeepMzyme's pre-declared recipe reached 75.74% CV macro-precision and 74.34% pooled
  balanced accuracy, and PMM-R 71.24% / 67.16%; PinMyMetal reported 75.08% (≈72.7% BA) on its own folds and rows
  (unmatched context, not a comparison).
- **Mn ([TECH-028](../../../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-028--pmm-comparator-inherits-a-true-metal-label-leak)).** In the matched refits, removing the four columns lowered PinMyMetal's test Mn recall from
  79.0% (PMM-R) to 61.7% (PMM-D), 17.4 pp [11.1, 24.1]. This is consistent with label-bearing inputs contributing to
  PinMyMetal's high Mn recall; neither refit reproduces the published 88.6%, so that contribution is not quantified.
  DeepMzyme's Mn recall (55.7%) is below PMM-D's as well.
- **Ablation.** A fixed, untuned gradient-boosting classifier on PinMyMetal's 89 deployment-available features
  (68.83% test, 71.22% CV, 75.51% CV macro-precision) was not distinguishable from DeepMzyme's single model or from
  any of the CV-selected ensembles (test −1.28 to +2.92 pp, all intervals include zero). DeepMzyme's demonstrated
  advantage is over PinMyMetal's released classifier, not over its feature representation.
- **Clean subset (423 rows, descriptive).** The single model was not distinguishable from PMM-D (+1.76 [−9.64, 11.64])
  and was below PMM-R (59.46%) and the ablation (58.36%) by point estimate.
- **Amendments.** The pre-registered criterion for "outperforms PinMyMetal" also required beating the
  gradient-boosting model and was met in neither setting. After the results were known, at the author's direction, amendment
  A1 reclassified that model as an internal ablation, because it is not PinMyMetal, so the Result claim rests on the
  pre-registered T1 and C1 contrasts under a decision rule re-scoped post hoc; A1 also added the published-value context, the CV-only
  ensemble table and CV macro-precision. Amendment A2 added a Top-N family ranked by seed-42 out-of-fold score at the
  author's request after its Top-4 and Top-5 test scores were known (Top-1 to Top-6: 70.75–72.10% full test); it is outside the
  15-candidate set and not promoted.
- **Scope and status.** Both systems classify the metal at known metal-bound sites in experimental holo structures;
  PinMyMetal's site localization is not evaluated. The test set was opened before this comparison and is not pristine
  (668 PDB IDs on both sides of PinMyMetal's own split; 809 test rows share a PDB ID with the training cohort); the
  result is secondary ([test-use ledger](../../../DATASETS.md#test-use-ledger)). DeepMzyme's recipe was screened on the
  same folds, while the PinMyMetal-feature models use released or fixed hyperparameters, and DeepMzyme also uses structure
  context and pretrained ESM C embeddings. PMM-D slightly under-powers deployed PinMyMetal, which would keep Zn-based
  `valence_4a`, `vecsum_3a` and `vecsum_4a` (the Zn write is inferred from source code). Sites are centred on deposited
  metal positions for both systems, and DeepMzyme's three unscorable test rows count as errors. Ensemble test
  probabilities come from the 2026-10-19 rescore, bound by key name (parity with the frozen single-model predictions
  1.6e-6, identical predicted classes). PMM-R was trained on the 7,398-site cohort, not on PinMyMetal's full
  7,920-row released table.

## Interpreting the HistGB comparison

The unresolved reference-test contrasts do not imply that all CV contrasts are unresolved.
The existing `matched_report.json` entry `cv.contrasts["DeepMzyme-greedy-nested - PMM-H"]`
reports **+5.32 pp** pooled BA (76.54% versus 71.22%), with a paired near-copy-group
95% interval **[2.41, 8.22]**. This is a secondary, descriptive comparison conditional
on the fitted models, as specified in the pre-registration. The mean-fold difference
is **+5.26 pp**, with a fold-bootstrap 95% interval **[−0.37, 9.46]** and a t interval
**[−2.73, 13.25]**. The fold-based uncertainty therefore remains unresolved.
These estimates concern the greedy selection procedure, not every ensemble or the
public candidate. Saved-OOF selection also reuses underlying fits that trained on the
outer scored fold; it is not fully retrained nested CV ([release-audit limits](../release_audit/README.md#decision)).

### Coordination environment and transfer to predicted structures

PMM-D and HistGB share 89 tabular features: 42 hydrophobicity/solvation,
24 chemical-class counts and 23 NEIGHBORHOOD coordination/geometry and elemental bond-valence
features. These include `coordnum_inner`, `geometry_type`, `distance_min/avg/max`,
`num_metal_4a` and ten `bvs_*` columns. Coordinating waters and other non-protein ligands
can complete the first coordination sphere: their donor atoms affect coordination number,
angular geometry and bond-valence sums. This supplies explicit coordination information
that is absent from DeepMzyme's standard-residue graph. The existing
[input audit](../../../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-029--verified-input-and-training-behaviours-with-small-measured-effect)
records water-inclusive coordination numbers of at least six for 63% of Mn sites and
48% of Class VIII sites, versus 6% of Zn and 2% of Cu sites in the training folds.
These are descriptive class associations, not a measurement of the benefit of adding waters.

[FindGeo](https://doi.org/10.1093/bioinformatics/bts246) characterizes the arrangement of
coordinating atoms by comparison with ideal geometries, including tetrahedral and octahedral
templates. It also supports geometries with a vacant coordination position; an incomplete
sphere can retain useful geometric information, and a vacant position does not prove a
missing ligand. FindGeo illustrates the importance of ligand completeness here; the
PinMyMetal feature generator is NEIGHBORHOOD, not FindGeo.

Standard AlphaFold2 predictions lack explicit waters, metal ions and other non-protein
ligands ([AlphaFill](https://doi.org/10.1038/s41592-022-01685-y)). Consequently, descriptors
recomputed from protein atoms around a proposed site need not represent the same coordination
environment as those from an experimental holo structure. Recovering missing partners
requires additional modeling or transfer, with its own uncertainty. This is an important
difference in available structural information; the matched benchmark does not isolate
its contribution to classification accuracy. Removing all 23 NEIGHBORHOOD columns would
also remove protein-derived geometry and would not constitute a water-only ablation.

DeepMzyme uses protein residues, pretrained ESM-C representations and a supplied site
coordinate, avoiding a requirement for experimentally resolved non-protein partners.
Its training structures nevertheless retain protein conformations shaped by metal binding;
omitting the partners does not turn a holo structure into an apo or predicted structure.
Performance on AlphaFold2-derived sites remains unvalidated.
Candidate-element `bvs_*` columns are distinct from the verified true-label-bearing
`valence_3a` column; ligand-inclusive geometry is not itself label leakage.

## Ensembles chosen by cross-validation only (amendment A1)

All 15 members of four rule families are listed: R1 Top-N families by out-of-fold rank with all seeds, R2 the same with
seed 42, R3 the pre-declared greedy ensemble, R4 B four-class with one and three seeds. Several members' test scores
were public from the 2026-10-19 rescore; none is promoted. CV macro-precision is pooled and non-nested, so it is
optimistic for R1 to R3. Paired intervals are descriptive, unadjusted and overlapping. The ablation is the
gradient-boosting model on PinMyMetal's features. "B4 seed 42, 5 folds" averages the five fold checkpoints; it is not
a0007846, the full-data refit.

| Ensemble | Checkpoints | CV BA, nested | CV BA, pooled | CV macro-precision | Full-test BA [95% CI] | − PMM-D, pp [95%] | − PMM-R, pp [95%] | Mn − PMM-R [95%] | − ablation, pp [95%] | Eligible BA | Clean BA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B4 seed 42, 5 folds | 5 | — | 74.34 | 75.74 | 69.02 [64.47, 73.05] | +10.75 [5.77, 15.80] | +5.61 [0.34, 10.53] | [-30.56, -11.39] | +0.19 [-4.16, 4.53] | 70.75 | 52.32 |
| B4 x3 seeds | 15 | — | 75.58 | 77.39 | 69.94 [65.46, 74.03] | +11.67 [6.72, 16.77] | +6.53 [1.30, 11.46] | [-29.07, -10.55] | +1.11 [-3.16, 5.40] | 71.83 | 53.49 |
| Top-1 all seeds | 15 | 75.32 | 75.82 | 77.13 | 71.10 [66.65, 75.14] | +12.83 [7.92, 17.74] | +7.69 [2.67, 12.44] | [-25.95, -7.93] | +2.28 [-1.95, 6.47] | 72.68 | 58.21 |
| Top-2 all seeds | 30 | 75.80 | 76.22 | 78.30 | 70.90 [66.41, 74.99] | +12.63 [7.78, 17.64] | +7.49 [2.38, 12.28] | [-28.40, -10.12] | +2.07 [-2.06, 6.37] | 72.71 | 57.37 |
| Top-3 all seeds | 45 | 75.92 | 76.05 | 78.19 | 70.31 [65.81, 74.43] | +12.04 [7.08, 17.11] | +6.90 [1.72, 11.80] | [-29.05, -10.64] | +1.48 [-2.74, 5.79] | 72.22 | 55.37 |
| Top-4 all seeds | 50 | 76.27 | 76.46 | 78.36 | 71.75 [67.22, 75.80] | +13.48 [8.60, 18.42] | +8.34 [3.30, 13.05] | [-24.82, -6.67] | +2.92 [-1.15, 7.11] | 73.48 | 57.47 |
| Top-5 all seeds | 55 | 76.39 | 76.39 | 78.81 | 71.33 [66.80, 75.38] | +13.05 [8.24, 17.98] | +7.91 [2.84, 12.63] | [-26.06, -7.78] | +2.50 [-1.55, 6.65] | 73.11 | 56.97 |
| Top-6 all seeds | 70 | 76.61 | 76.72 | 78.88 | 71.25 [66.75, 75.28] | +12.98 [8.12, 17.95] | +7.84 [2.78, 12.56] | [-25.97, -7.80] | +2.42 [-1.67, 6.60] | 72.92 | 56.09 |
| Top-1 seed 42 | 5 | 74.54 | 74.48 | 75.67 | 69.90 [65.27, 74.03] | +11.63 [6.71, 16.55] | +6.49 [1.45, 11.15] | [-25.29, -7.30] | +1.07 [-3.13, 5.33] | 71.65 | 59.05 |
| Top-2 seed 42 | 10 | 75.30 | 75.50 | 77.18 | 70.39 [65.80, 74.55] | +12.12 [7.42, 16.93] | +6.98 [1.95, 11.65] | [-26.16, -8.72] | +1.56 [-2.50, 5.67] | 72.00 | 57.96 |
| Top-3 seed 42 | 15 | 75.33 | 75.16 | 76.90 | 68.83 [64.16, 73.00] | +10.56 [5.73, 15.53] | +5.42 [0.25, 10.24] | [-29.45, -11.41] | 0.00 [-4.20, 4.20] | 70.49 | 54.35 |
| Top-4 seed 42 | 20 | 75.44 | 75.76 | 77.54 | 70.33 [65.84, 74.29] | +12.06 [7.28, 16.95] | +6.92 [1.88, 11.56] | [-28.14, -10.37] | +1.50 [-2.63, 5.69] | 71.83 | 55.50 |
| Top-5 seed 42 | 25 | 75.76 | 75.65 | 77.66 | 69.89 [65.27, 74.05] | +11.62 [6.86, 16.46] | +6.48 [1.50, 11.18] | [-28.14, -10.37] | +1.06 [-2.99, 5.18] | 71.47 | 55.94 |
| Top-6 seed 42 | 30 | 75.96 | 75.88 | 78.05 | 70.21 [65.59, 74.32] | +11.94 [7.18, 16.78] | +6.80 [1.77, 11.49] | [-26.95, -9.36] | +1.39 [-2.65, 5.45] | 71.78 | 56.11 |
| Greedy (pre-registered secondary) | 30 | 76.54 | 77.99 | 81.14 | 70.57 [65.90, 74.75] | +12.30 [7.62, 16.94] | +7.16 [2.35, 11.58] | [-21.14, -4.40] | +1.74 [-2.31, 5.71] | 73.00 | 59.20 |

## Seed-42 ranking variant (amendment A2, post hoc)

Top-N late-fusion families ranked by their own seed-42 out-of-fold score, seed 42 checkpoints. Added at the author's
request after the Top-4 and Top-5 test scores were known; outside the 15-candidate set and not promoted. The Unified
recipe (rank 5) has no recorded validation-only reason for its selection. Ranking pool: the 18 late-fusion families with
complete five-fold seed-42 OOF; families without it cannot be ranked. Top six (seed-42 pooled OOF BA):
`gvp_late_fusion__four_class__trunklr1e4` 75.51, `gvp_late_fusion__four_class__esmdrop02` 75.26,
`gvp_late_fusion__six_class__baseline` 74.98, `mc_S_six_class` 74.93, `unified__P_five_class_trunklr1e4` 74.91,
`gvp_late_fusion__four_class__headdrop03` 74.85.

| Ensemble | Checkpoints | CV BA, nested | CV BA, pooled | CV macro-precision | Full-test BA [95% CI] | − PMM-D, pp [95%] | − PMM-R, pp [95%] | Mn − PMM-R [95%] | − ablation, pp [95%] | Eligible BA | Clean BA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Top-1 seed 42, seed-42 rank | 5 | 74.47 | 75.51 | 76.81 | 70.75 [66.29, 74.77] | +12.48 [7.63, 17.35] | +7.34 [2.32, 12.03] | [-27.78, -9.56] | +1.92 [-2.23, 6.11] | 72.25 | 55.87 |
| Top-2 seed 42, seed-42 rank | 10 | 75.65 | 75.82 | 77.50 | 71.26 [66.66, 75.38] | +12.99 [8.25, 17.72] | +7.84 [2.85, 12.45] | [-25.66, -8.28] | +2.43 [-1.55, 6.43] | 73.04 | 57.77 |
| Top-3 seed 42, seed-42 rank | 15 | 75.98 | 76.29 | 78.66 | 70.89 [66.22, 75.03] | +12.61 [7.77, 17.39] | +7.47 [2.48, 12.19] | [-26.70, -8.55] | +2.06 [-1.97, 6.11] | 72.62 | 58.44 |
| Top-4 seed 42, seed-42 rank | 20 | 76.10 | 76.35 | 78.48 | 71.83 [67.20, 75.95] | +13.56 [8.76, 18.33] | +8.42 [3.45, 13.15] | [-25.33, -7.51] | +3.00 [-1.00, 7.02] | 73.62 | 60.07 |
| Top-5 seed 42, seed-42 rank | 25 | 76.09 | 76.16 | 77.77 | 72.10 [67.63, 76.02] | +13.83 [9.09, 18.61] | +8.69 [3.63, 13.34] | [-23.98, -6.29] | +3.27 [-0.82, 7.36] | 73.64 | 59.37 |
| Top-6 seed 42, seed-42 rank | 30 | 76.23 | 76.48 | 78.52 | 71.15 [66.67, 75.16] | +12.88 [8.06, 17.74] | +7.74 [2.72, 12.44] | [-25.75, -8.19] | +2.32 [-1.75, 6.48] | 72.65 | 57.12 |
