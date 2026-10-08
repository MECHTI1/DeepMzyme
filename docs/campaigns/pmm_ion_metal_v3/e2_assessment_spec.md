# PMM ion metal v3 — step E2 assessment specification

Status: **frozen on 2026-10-08** (user decisions of 2026-10-08, Option B with
clarifications; [log v3-028](log.md#v3-028) records the SHA-256 of this page and
of [`e2_assessment_spec.json`](e2_assessment_spec.json)). The A4 specification
([`assessment_spec.md`](assessment_spec.md), [`assessment_spec.json`](assessment_spec.json))
is unchanged and stays the authority for the original step E assessment. The E2
assessor, once built, pins the E2 JSON hash and refuses a different one; the
JSON holds the numbers. This page covers the E2 assessment rules only: it is not
an implementation, it authorizes no GPU start, and nothing in it changes the
original step E results or the frozen A4 behaviour. The numerical rules below
were frozen before any E2 result existed and are not changed after results are
seen.

## 0. Interpretation (reported verbatim with every E2 result)

Step E2 is an exploratory extension that selects a promising configuration
from reused development-validation evidence: one training seed (42), four
folds (1–4), validation folds that were also used to choose the original step
E selection, and candidates chosen on fold 0. Passing its replacement rule does
not establish independently confirmed superiority or robustness across
training seeds. Step F stays one-shot with the recorded
`both_results_secondary` label and its prior-exposure caveats; neither the
exploratory label nor a possibly opened test removes selection bias. The
illustrative calculations in the log (a family-wise figure of about 0.18 for
eight independent null comparisons; a power table under independent normal
fold differences) describe the behaviour of the rule under stated assumptions
and are not estimates of this campaign's false-replacement rate or power.

## 1. Scope (fixed)

- Candidates (eight), each an existing recipe that changes exactly one setting
  from its control; `four_class`; folds 1–4; seed 42; 50-epoch cosine schedule;
  terminal checkpoint; the recorded step-B execution setting: Only-GVP `wd10`
  and `sitecountsangles`; late fusion `invsqrtw`, `meanagg`, `gvpaux03`,
  `resdrop01`, `esmdrop02` and `wd10`.
- Fits: 32 candidate fits plus four Only-GVP `sitenone` controls on folds 1–4,
  36 in total. No extra seed, combination, strength, target or schedule change;
  the cost-gated augmentations stay "not tested (cost)".
- Matched control: the completed step E four-class baseline of the candidate's
  family on folds 1–4, reused by checked identity (unit name, fold, seed,
  checkpoint and predictions SHA-256, replay verification). For
  `sitecountsangles` the matched control is `sitenone`, and the Only-GVP
  baseline is an additional required comparator.
- Original selection: `stage6_selection.selected` of the completed original
  step E assessment (all six neutral contrasts complete, both improvement
  checks complete, a non-null Stage 6 selection, `held_out_test_accessed`
  false), bound by that assessment file's SHA-256 in the E2 manifest. A
  projected winner is never substituted for the actual completed selection.
  The selection may be a four-, five- or six-class cell of any family.

## 2. Metric and evidence

- Metric: common-four balanced accuracy of the terminal checkpoint's
  validation predictions on each fold; five/six-class predictions are
  collapsed as in A4 (Fe, Co, Ni or Fe and Co+Ni probabilities summed into
  Class VIII before the argmax). The native metrics of a five/six-class
  selection (native BA; Fe, Co, Ni or Fe and Co+Ni recalls) are preserved and
  reported; native BAs are never compared across vocabularies.
- Evidence required for every unit used (candidate, control, selection):
  training completed at epoch 50, terminal checkpoint selected, worker exit
  code 0, host-pull manifest verified, acknowledgment uploaded, independent
  replay under `pmm-v3-replay-1` matching, durable copy on Data1 with SHA-256
  checked. Missing, duplicated, mismatched, unreplayed or unverified evidence
  makes the comparison "incomplete" and the candidate ineligible; nothing
  fails silently.

## 3. Paired differences and intervals (the A4 methods, unchanged)

- One paired difference per fold (candidate minus comparator), folds 1–4,
  seed 42: four differences per comparison.
- Fold bootstrap: 10,000 resamples of the four differences with replacement,
  bootstrap seed 42, percentile interval; Student t interval with 3 degrees of
  freedom; both at the unadjusted 95% level. Bonferroni-adjusted intervals
  (95% split over the 17 declared comparisons: 1 − 0.05/17 each) are reported
  for every comparison as sensitivity evidence; they are not the gate.
- Recorded qualification: with four folds the exact resampling distribution
  puts mass 1/256 on each endpoint, but the finite 10,000-resample estimate
  need not return the exact minimum at every adjusted quantile. The bootstrap
  and t bounds are two views of the same four numbers, not independent
  confirmations.
- Verdict per comparison, as A4 (`_outcome` and gates): "better" when both
  unadjusted lower bounds are above zero and the recall gates pass; "worse"
  when both upper bounds are below zero; "blocked by a recall gate" when the
  intervals say better but a gate fails; otherwise "no clear difference".

## 4. Declared comparison family (17; fixed before any E2 result)

- C1–C8: each candidate against its matched control (Only-GVP `wd10` against
  the Only-GVP baseline; `sitecountsangles` against `sitenone`; the six
  late-fusion candidates against the late-fusion baseline).
- C9: `sitecountsangles` against the Only-GVP baseline.
- C10–C17: each candidate against the original step E selection.
- A comparison whose comparator is the same cell as another comparison's
  (for example a late-fusion candidate when the original selection is the
  late-fusion baseline) is marked "coincident" in the report; the family and
  the adjustment divisor (17) do not change after results are seen.

## 5. Own-model conclusion (per candidate, against its matched control)

Reported with the numerical fold differences, mean, SD, both intervals
(unadjusted and adjusted), per-class recalls and gates, in the A4
improvement-check vocabulary:

- "improvement interval-supported on folds 1–4": verdict "better" against the
  matched control (both unadjusted lower bounds above zero, all recall gates
  passed);
- "positive mean gain on folds 1–4, not interval-supported" (promising): mean
  difference above zero, gates passed, not both lower bounds above zero;
- "no positive mean gain on folds 1–4": mean difference at or below zero (gates
  still reported);
- "blocked by a recall gate against the matched control": a positive mean (or
  interval support) with a failed gate; the comparator, class and magnitude
  are named, and this overrides any "promising" description.

For `sitecountsangles` the conclusion against the Only-GVP baseline is reported
alongside, in the same vocabulary.

## 6. Replacement conclusion (per candidate, against the original selection)

A candidate is eligible to replace the original selection only when all of
the following hold:

- (a) complete evidence (section 2) for the candidate, its matched control(s)
  and the original selection on folds 1–4;
- (b) both unadjusted 95% lower bounds (bootstrap and t) above zero against
  the matched control and against the original selection; for
  `sitecountsangles` also against the Only-GVP baseline;
- (c) estimated mean gain over the original selection on folds 1–4 strictly
  greater than 0.002 (0.2 percentage points); this applies to the estimated
  mean, while the interval lower bounds must exceed zero;
- (d) recall gates against every required comparator (matched control,
  original selection, and the Only-GVP baseline for `sitecountsangles`): no
  missing or zero common-four mean recall of the candidate, and no common-four
  class mean recall decrease greater than 0.03 (3 percentage points) relative
  to that comparator.

Among eligible candidates the highest mean common-four BA on folds 1–4 is
selected; within 0.002 the order is higher mean minimum common-four recall,
higher worst-fold BA, lower SD over folds 1–4, simpler family (Only-GVP before
late fusion), then the stable recipe name. If no candidate is eligible, the
original selection is retained. Labels: "selected"; "eligible, not best";
"not eligible", followed by every failed condition with its comparator, class
and magnitude. A failure against the original selection blocks replacement
but does not alter the own-model conclusion; a failed recall gate against the
matched control overrides "promising". Failure to replace is not evidence of
no effect.

## 7. Report and outputs

- Per candidate: fold values on folds 1–4 for the candidate and each
  comparator; per-fold differences, mean and SD against each comparator;
  unadjusted and Bonferroni-adjusted bootstrap and t intervals with verdicts;
  per-class common-four recalls; gates by comparator with magnitudes; the
  own-model conclusion; the replacement conclusion with reasons; the fold-0
  seed-42 development-fold difference shown separately and never inside an
  interval; five-fold summaries descriptive only; coincident comparisons
  flagged.
- The original selection: cell, assessment SHA-256, native and common-four
  metrics, unchanged.
- A machine-readable final selection: selected cell, whether it replaced the
  original, the original selection, the eligible candidates, the reasons, the
  E2 specification and manifest SHA-256 values and the evidence references;
  `held_out_test_accessed` false.
- The interpretation of section 0, verbatim.

## 8. Failed runs and completion

- One unchanged rerun after a failed or interrupted fit (the first attempt is
  archived, never deleted); a second failure blocks the final assessment for
  diagnosis.
- E2 is complete when all 36 fits have verified evidence, all 17 comparisons
  are reported and one final selection exists. Step F takes the E2 final
  selection, or the unchanged A4 selection if the user cancels E2 by a dated
  log entry.
