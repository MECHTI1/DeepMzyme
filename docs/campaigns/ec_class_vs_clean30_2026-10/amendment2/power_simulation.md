# Amendment 2: power simulation of the claim rule

Pre-data evidence for [amendment 2](amendment_draft.md), decision A2-D2. Completed on 2026-11-04, before any CLEAN
embedding farm, F6 sequence-mean cache, G2 measurement, model prediction, score, selection record, or CLEAN inner
fit or export existed; the certified views' label-free ESMC chain-mean caches predate the amendment.

**What this is.** A synthetic Monte Carlo of the frozen claim rule. No campaign model, prediction, label table or
test file was used. Every population, cluster structure, accuracy level and gain pattern is an assumption (table
below), so every number here is conditional on those assumptions and describes the procedure, not DeepMzyme or
CLEAN. It cannot change any rule; it is the recorded rationale for choosing the primary metric before data.

**Notation.** Margins are in points. "vs seed mean" is DeepMzyme minus the CLEAN R1 seed mean; the margin against
the R1 ensemble is that minus 1 point (assumption). BA is balanced accuracy over EC classes 1-6, with class-7
proteins excluded and predictions into class 7 counted wrong. "acc" is accuracy over the proteins of EC classes
1-6. P1 is B1-all and P2 is B1-NH30; both use the Holm-adjusted intersection-union test over both R1 components.
L and H are the low- and high-accuracy configurations; "s0.5" and "s1.0" are the shared cluster random-effect
standard deviations.

## Method

- **Frozen engine, called directly:** `build_cluster_count_table` (class support and cluster order), the stratum
  gate functions `assign_cluster_strata` and `stratum_members`, `stratified_multiplicities` (frozen seed,
  1,000-draw batches), the contrast matrix, `BootstrapResult`, the p-values, `intersection_union_test` and
  `co_primary_holm` (Freeze-1 engine, `stratified_k_of_k_v1`).
- **Replicated instead of called:** the resampled confusion tensor (only its diagonal and row sums, which BA and
  accuracy read) and the metric arithmetic, in the frozen order. All counts are integers, so the sums are exact.
- **Validation:** 11 units, covering every configuration and including 6 with a gated B1-NH30, gave draws
  bit-identical to the frozen engine. The only setting that differs from the frozen engine is 2,000 bootstrap
  draws instead of 10,000.
- **Replicates:** R = 500 for the main configurations and 300 for the sensitivity configurations. All
  configurations share the same populations and random numbers, so their results are correlated, not
  independent. Records carry a hash of the simulation code, and records of another code version are refused.
  An earlier interim run was discarded unread; no number below comes from it.
- **Margin grid:** 0, 1, 2, 4, 6, 8, 10, 12, 14, 16 and 20 points. Minimum detectable effects (MDEs) are
  interpolated on each metric's own true margin. Where the grid falls short, a probit fit gives the 80% crossing
  with a Monte Carlo interval; crossings beyond twice the grid are suppressed. The BA/accuracy ratio is paired by
  replicate.

No unit aborted and B1-all was never gated. B1-NH30 was gated in 1.4% of main-configuration replicates, 1.0-1.3% in
the sensitivities and 34% under rare-class concentration.

## Assumptions

| Quantity | Value | Source |
|---|---|---|
| B1-all size | 530 | approximate design-time figure ([design](../design.md#benchmark-systems-and-claims)) |
| EC class 5 / 6 in B1-all | 13 / 18 | approximate design-time figure (design-time label exposure, disclosed in the design) |
| Class 7 in B1-all | 3 | assumed |
| EC classes 1-4 shares (of the other 496 proteins) | 0.25 / 0.22 / 0.40 / 0.13 | assumed |
| B1-all components | 307, largest 39 | recorded for the 740-protein cohort ([evc-011](../log.md#evc-011)), used as an assumption for B1-all; sensitivity with 240 |
| Other cluster sizes | truncated zeta law over 1-25 | assumed |
| Rare-class placement | single-class clusters of at most 4 (rare-concentration configuration: at most 8, weighted by size squared) | assumed |
| Mixed clusters | 15% of multi-member clusters swap 20% of members | assumed |
| B1-NH30 construction | cluster weight size^-0.5, keep probability 0.75, target at least 215 | assumed; gives 216 proteins in 171 clusters, EC class 5 = 6.4 proteins in 5.7 clusters, EC class 6 = 8.7 in 7.8 |
| Seed accuracy per class (EC classes 1-4 / 5-6) | L 0.80 / 0.70; H 0.88 / 0.80 | assumed |
| CLEAN ensemble | seed + 1 point on EC classes 1-6 | assumed |
| Seed–ensemble disagreement | 5%, constant flip probabilities | assumed; the ensemble component binds the claim |
| DeepMzyme vs ensemble discordance | 0.20 (L), 0.15 (H); sensitivities 0.15 and 0.25 | assumed |
| Cluster random effects | shared 0.5 / 1.0; family-specific 0 (main), 0.5 or 1.0 (sensitivities) | assumed |
| Wrong predictions | 5% into class 7 | assumed |

## Type-I error at the null (DeepMzyme equal to the CLEAN ensemble; main configurations, R = 500, Monte Carlo SE about 0.01)

| | BA | acc | Nominal |
|---|---|---|---|
| Holm P1 | 0.042-0.052 | 0.028-0.032 | ≤ 0.05 |
| Holm P2 | 0.044-0.052 | 0.020-0.036 | ≤ 0.05 |
| Any claim | **0.068-0.076** | 0.038-0.048 | ≤ 0.05 |
| P(p_ens ≤ 0.025) at margin 1, all / NH30 | 0.040-0.050 / 0.040-0.048 | 0.026-0.030 / 0.016-0.032 | 0.025 |
| P(p_ens ≤ 0.05) at margin 1, all / NH30 | 0.070 / 0.068-0.080 | 0.038-0.060 / 0.042-0.064 | 0.05 |
| P(p_sm ≤ 0.025) at margin 0, all / NH30 | 0.028-0.046 / 0.042-0.052 | 0.028-0.032 / 0.020-0.038 | 0.025 |
| P(p_sm ≤ 0.05) at margin 0, all / NH30 | 0.052-0.076 / 0.068-0.082 | 0.054-0.062 / 0.046-0.068 | 0.05 |
| Bootstrap SE / sampling SD, all / NH30 | 0.93-0.95 / **0.88-0.93** | 0.97-1.00 / 0.95-1.01 | 1 |

In the sensitivity configurations the BA any-claim rate is 0.053-0.083 and accuracy's is 0.040-0.063.

The frozen BA test is anti-conservative, most of all on B1-NH30, because its bootstrap under-disperses when rare
EC classes sit in few clusters. BA's power is therefore measured at an above-nominal size, so the nominal-alpha
comparison of BA with accuracy favours BA. The size-matched alpha is 0.010-0.0165 for BA and 0.016-0.036 for
accuracy.

## MDE under per-class-uniform gains (80% power, true margin vs seed mean, ± Monte Carlo SD)

| Configuration | BA P1 | BA P2 | acc P1 | acc P2 | BA/acc ratio P1 | BA/acc ratio P2 |
|---|---|---|---|---|---|---|
| L_s0.5 | 10.3 ± 0.3 | 13.4 ± 0.4 | 6.4 ± 0.1 | 8.8 ± 0.2 | 1.59 [1.51, 1.69] | 1.53 [1.44, 1.63] |
| L_s1.0 | 10.3 ± 0.2 | 12.9 ± 0.4 | 6.4 ± 0.1 | 8.8 ± 0.2 | 1.60 [1.52, 1.68] | 1.47 [1.38, 1.56] |
| H_s0.5 | 9.2 ± 0.2 | 11.2 ± 0.2 | 5.5 ± 0.1 | 7.3 ± 0.1 | 1.68 [1.62, 1.75] | 1.53 [1.47, 1.59] |
| H_s1.0 | 9.1 ± 0.2 | 11.1 ± 0.2 | 5.5 ± 0.1 | 7.3 ± 0.1 | 1.67 [1.60, 1.73] | 1.52 [1.46, 1.58] |

- **Single-population intersection-union test at 0.025, vs seed mean (all / NH30):** BA 10.5-10.6 / 14.4-14.6 (L)
  and 9.3-9.4 / 11.9-12.0 (H); accuracy 6.5 / 9.6-9.7 (L) and 5.5 / 7.8-7.9 (H).
- **Check against the design figure:** H measured against the ensemble gives 8.3-8.4 / 10.9-11.0, which
  reproduces the design's analytic 8 / 11 BA points.
- **Size-matched test (all / NH30):** BA 11.3-11.5 / 15.4-15.9 (L) and 9.7-10.3 / 12.4-12.6 (H); accuracy
  6.8-6.9 / 9.6-10.1 (L) and 5.5-5.6 / 7.6-7.8 (H); BA/accuracy ratio 1.64-1.86 / 1.53-1.67.
- **P2 power ceiling:** P2 power levels off at 0.986, because 1.4% of replicates are gated.
- **Probit cross-check:** within 0.5 points of the interpolation.

## Other gain patterns (each metric's own true margin)

- **Data gap (BA margin d/2, accuracy margin about 0.91d).**
  - BA is not reached within the headroom: largest margin 9.7 (L) and 5.7 (H). BA power there is 0.69-0.70 (P1)
    and 0.51 (P2) for L, and 0.39-0.40 (P1) for H. BA probit crossings: P1 10.8-11.0 and P2 13.7-13.9 (L); P1 9.4 (H).
  - Accuracy: P1 6.4-6.5 and P2 8.7-8.9 (L); P1 5.3-5.5 and P2 7.0-7.2 (H), that is d = 7.0-7.1 (L) and 5.8-6.0 (H).
  - In true-margin units each metric is close to its uniform-gain MDE, so the difference between the metrics
    reflects what each measures, not test efficiency.
- **Rare-class gain only (+d on EC classes 5-6, 0 on 1-4; BA margin d/3).**
  - BA P1 needs 9.3-9.4 (L), that is d of about 28.5, raising EC classes 5-6 from 0.70 to about 0.98. H is not
    reached (largest margin 6.5, power 0.61, probit 8.1-8.2). BA P2 is not reached in any configuration (power 0.63
    at 9.8; probit 12.1).
  - Accuracy has no usable power for this pattern: margin at most 1.7, power at most 0.074.

## Cluster sensitivity (uniform gains, replicates 0-299, MDE P1 / P2)

| Configuration | BA | acc | Ratio P1 / P2 | BA SD all / NH30 | acc SD all / NH30 |
|---|---|---|---|---|---|
| L_s1.0, first 300 replicates | 10.4 / 13.0 | 6.2 / 8.7 | 1.68 / 1.50 | 3.41 / 5.50 | 1.95 / 3.08 |
| family-specific 0.5 | 10.2 / 12.8 | 6.7 / 8.2 | 1.53 / 1.56 | 3.59 / 5.40 | 2.15 / 3.18 |
| family-specific 1.0 | 9.8 / 12.0 | 7.5 / 8.9 | 1.30 / 1.35 | 3.89 / 5.61 | 2.66 / 3.51 |
| 240 components | 10.3 / 12.6 | 6.6 / 8.7 | 1.56 / 1.46 | 3.24 / 4.58 | 1.90 / 2.73 |
| family-specific 1.0 + 240 components | 10.5 / 12.2 | 8.0 / 9.0 | 1.32 / 1.36 | 3.98 / 5.07 | 3.06 / 3.56 |
| discordance 0.15 | 9.3 / 12.1 | 5.6 / 7.8 | 1.66 / 1.56 | 2.95 / 4.51 | 1.67 / 2.68 |
| discordance 0.25 | 11.2 / 13.6 | 6.9 / 9.3 | 1.62 / 1.46 | 3.79 / 6.00 | 2.19 / 3.52 |
| rare-class concentration | 10.2 / not reached | 6.6 / not reached | 1.55 / n/a | 3.42 / 6.87 | 1.91 / 3.06 |

- **The main MDEs are close to no-cluster values and include no cluster inflation.** The accuracy SD matches the
  no-cluster value sqrt(discordance/n): 1.93-1.94 against 1.95 (L), 1.62-1.67 against 1.69 (H), and 3.05-3.11
  against 3.05 on B1-NH30.
- **Family-specific effects inflate accuracy more than BA.** Accuracy SD rises 37% (family-specific 1.0) to 57%
  (with 240 components); its P1 MDE goes from 6.2 to 7.5 and 8.0. BA SD rises 11-17% and its MDE stays within
  about 2 Monte Carlo SD of the reference. The ratio falls from about 1.6 to about 1.3; accuracy keeps the smaller
  MDE.
- **Rare-class concentration:** P2 power levels off at about 0.62 (BA) and 0.66 (accuracy) because 34% of
  replicates are gated. Probit values for P2 there are fits to a curve that never reaches 80% and are not MDEs.
- **If B1-all is more clustered than assumed,** the accuracy MDEs could be optimistic by up to about 1.5-2 points;
  BA moves much less.

## Conclusions (conditional on the assumptions)

- **Uniform gains:** accuracy over EC classes 1-6 needs about 0.6-0.65 times the BA margin (L: P1 6.4 vs 10.3,
  P2 8.8 vs about 13). The ratio grows under size matching (1.6-1.9) and shrinks under strong family-specific
  clustering (about 1.3). This holds only for per-class-uniform gains and is partly built in: EC classes 5-6 carry a
  third of BA's weight on about 6% of the proteins.
- **The endpoint is also the estimand.** Gains on EC classes 5-6 only are detectable only by BA, and only with
  about +28 points on those classes. A trade of rare-class recall for common-class recall is detected only by
  accuracy.
- **Test validity:** the frozen BA test is anti-conservative (any claim 0.068-0.076 against 0.05); accuracy is about
  nominal (0.038-0.048).
- **B1-NH30:** a BA claim there needs about 13 points over the seed mean for L (about 12 over the ensemble), or
  about 11 for H.

## How amendment 2 uses this

- It is the pre-data rationale for A2-D2 (accuracy over the proteins of EC classes 1-6 as primary metric).
- BA over EC classes 1-6 stays the first secondary metric and is reported with every result. Its bootstrap
  under-disperses when rare-class clusters are sparse, so the small-stratum correction of rule A2-STAT-2 (Rao-Wu
  k−1 of k resampling) applies to BA as well, and its effect on BA is reported (bounds under the Freeze-1 engine
  beside those under engine `stratified_rao_wu_v2`, descriptively).
- The estimand caveat is stated in every report: an accuracy claim weights the large EC classes, and a gain
  confined to EC classes 5-6 would not be detected by the primary metric.

## Evidence

The simulation code, results and validation are kept with the maintainer's private campaign records (not in the
repository). SHA-256 of the files as kept on 2026-11-04:

| File | SHA-256 |
|---|---|
| `sim.py` | `ac956cf942d90db711e1df0d0233beecda8eff6113a3ebdf8ae8f159f7d67a83` |
| `results.json` | `64279d7a0d5f69c4cc311f5086f708c0bb5231a5669dec44993be79700e8f053` |
| `validation.json` | `643855257d50b155c5fb3bb3c62a4ddc9345a79aeffdd731212218cae5952dd5` |
| `tables.md` | `15e71823a48b76201db5c80dda8d25d2d4cb8539c3fcf37b58d2e11ef0bc1ae2` |
