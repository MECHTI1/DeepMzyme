# PMM ion metal v3 — assessment specification (plan step A4)

Status: **frozen on 2026-10-05 at the close of step A** (approved by the user
the same day). The log records the SHA-256 of this page and of
[`assessment_spec.json`](assessment_spec.json), and the assessor
([`pmm_v3_assessment.py`](../../../pmm_v3_assessment.py)) pins the JSON hash.
Every run and every assessment refuses a different specification or one whose
cohort, fold and baseline identities differ from the campaign's. The JSON holds the numbers;
the assessor refuses a JSON whose numbers differ from the tested code.

Rules marked **(new default)** were not spelled out in the approved
[plan](plan.md); the user approved all four on 2026-10-05 with the
clarifications written into sections 3 and 4. This approval covers the
assessment rules only, not a GPU start.

## 1. What is compared

- Fold set `v3-seqid90-s42-b2`, `fold_membership.csv` SHA-256
  `e6b21364159da479dc3fb325a01ede36c62489184a77d858f71fefe6c401da7a`
  (builder `v3-near-copy-folds-2`); cohort SHA-256
  `745b19c1644ee17ec24e760b81916ff97d11d3c6017e791d610bca0b6bedc3bd`
  (7,398 ions).
- Baseline recipe: `pmm_v3_campaign.PROFILE` and the `baseline` recipe (cosine
  learning rate to zero over 50 epochs, terminal checkpoint, v2 model settings,
  class weights recomputed on the v3 folds). Mixed precision (AMP) and the number
  of concurrent workers are chosen only by the step-B rule of the plan; the
  chosen setting is logged before step C and then used by every v3 run except
  the regression run.
- Score: common-four balanced accuracy (BA: the mean of the Mn, Cu, Zn and
  Class VIII recalls) of the terminal checkpoint; five- and six-class
  probabilities are summed into Class VIII before the argmax. The mean over
  folds is the primary number; the pooled out-of-fold value is reported as a
  secondary, descriptive number.
- Every number is recomputed from the saved validation predictions of a
  completed, independently replayed run whose recorded identity equals the
  planned one.

## 2. How a difference is judged

- One paired difference per fold (same fold, same seed 42); step-D seed-43 runs
  are screening evidence only.
- Two 95% intervals on the fold differences: a fold bootstrap (10,000
  resamples, seed 42) and a Student t interval (degrees of freedom = folds − 1).
- **Better**: both lower bounds above zero. **Worse**: both upper bounds below
  zero. Otherwise **no clear difference** (also when the two methods disagree).
- A challenger cannot be "better" if any common-four class has a missing or zero
  mean recall, if any common-four class mean recall drops more than 3 points
  against the control, or (five/six-class) if any native class has a zero mean
  recall. Native Fe, Co, Ni (six-class) and Fe, Co+Ni (five-class) recalls are
  always reported.
- A claim across several contrasts (for example "five/six-class training helps
  in all three families") also needs both Bonferroni-adjusted lower bounds
  (95% split over the contrasts) above zero.

## 3. Contrasts

- **Neutral target test (step E)**: per family, five-class vs four-class and
  six-class vs four-class, baseline recipe, all five folds (six contrasts;
  Bonferroni over six for the cross-family claim). A tie keeps `four_class`.
  - **(new default)** If both five- and six-class are "better" in one family,
    the larger mean gain wins; within 0.2 points `five_class` wins (fewer
    classes). This tie preference is a selection convention, not evidence that
    five-class beats six-class.
- **Improvement check (step E)**: per family that passed step D, its final
  recipe vs the baseline, `four_class`, on folds 1–4 only (fold 0 was used to
  choose the recipe; its difference is reported separately as the development
  fold).
  - **(new default)** Folds 1–4 give 4 differences, so the t interval has 3
    degrees of freedom; Bonferroni is over the number of improved families.
    Wording: "improvement interval-supported on folds 1–4" when "better",
    "positive mean gain on folds 1–4, not interval-supported" when only the mean
    is positive. An improvement is claimed only in the first case: both lower
    bounds on folds 1–4 above zero.
  - The five-fold result of an improved recipe is labelled
    development-validation evidence, because fold 0 influenced recipe
    selection.
- Everything else (family vs family, five vs six) is descriptive.

## 4. Choosing the final model (Stage 6)

- Control: Only-ESMC, baseline, `four_class`. It is kept unless a candidate is
  eligible, interval-supported against the control on all five folds ("better"
  under section 2) and best by mean common-four BA.
- Eligible: complete on all five folds; a five/six-class arm must be "better"
  than its own family's `four_class` arm; an improved recipe must have a positive
  mean gain on folds 1–4 without a recall-gate failure (**new default**: the plan
  says "positive mean gain"; it does not require interval support here).
  Eligibility is kept separate from scientific claims: a positive mean gain
  permits consideration in Stage 6 but is not an improvement claim (section 3).
- Ties within 0.2 points are broken by, in order: higher mean minimum class
  recall, higher worst-fold BA, lower SD over folds, simpler family (Only-ESMC,
  Only-GVP, late fusion), baseline before an improved recipe.
- Refit: seed 42, same checkpoint rule, no calibration, no ensemble.

## 5. Step D one-fold screen (not an improvement claim)

- Score per seed: Δ = candidate minus control on new fold 0, same seed.
  The control is the family baseline (seed 42 from step C, seed 43 from Round A);
  `counts_angles` site geometry is compared with a matched `none` control.
- Pass: mean Δ over seeds 42 and 43 at least 1.5 points, Δ above zero for both
  seeds, and no Mn/Cu/Zn/Class VIII mean recall change below −3 points.
- Combination: as in the plan (better of each alternative pair, one combined run
  with both seeds, adopted only if it passes with a larger mean Δ than the best
  single candidate).

## 6. Failed runs

- A failed or interrupted run is rerun once, unchanged (same seed and identity);
  its first attempt is archived, never deleted (`archive-failed`).
- **(new default)** A second failure: in step D the candidate "did not pass the
  one-fold screen"; in steps C and E the step stops for diagnosis and the user
  decides. No run is replaced or repeated after its results are seen.

## 7. PMM context (descriptive only)

PMM published values, cited as reported (to be rechecked against the paper
before step C): 5-fold CV Mn 90.3, Cu 62.9, Zn 73.8, Class VIII 73.3 (mean
75.1); test (Figure 2b) Mn 88.6, Cu 59.4, Zn 65.9, Class VIII 57.5 (mean 67.85).
Allowed wording: "higher (or lower) than PMM's published CV (or test) values;
not a matched comparison". No interval or p-value for the difference.
