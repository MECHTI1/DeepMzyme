# PMM ion metal v3 — decision log

Dated user decisions and STATUS history for this campaign, newest first.
Current authority: [EXPERIMENT_STATUS.md](../../../EXPERIMENT_STATUS.md).

## v3-027

2026-10-08 — **proposed resolution of the pending E2 review, recorded for the
user's decision** (documentation only; CPU; no code change, fit, GPU action,
inference or test access; nothing here freezes the E2 specification, changes
the plan's drafted rules or authorizes a GPU start). Context recovered from
[v3-026](#v3-026); `deepmzyme-l4` verified TERMINATED at 16:11 UTC (today
5 h 44 min / $7.40 of the 12 h / $18 daily caps; nothing running; worktree
clean at df054b4 = `origin/v3-step-a`). The user asked to continue from the
recorded stopping point; the recorded next task is this review, so the
options below are prepared for the discussion. The eight candidates, the
36-fit scope, seed 42 and the $55 gross total planning ceiling are unchanged.

**Facts the proposal rests on (verified today from the frozen A4
specification, `pmm_v3_assessment.py`, and the assessments
`D-B_20261007T201334Z` and `E_20261008T121456Z`):**

- A4 decides every verdict ("better") on the **unadjusted** 95% intervals
  (both lower bounds above zero) plus the recall gates (`compare`, `_outcome`);
  the Bonferroni-adjusted intervals are reported as `adjusted_claim_supported`
  and decide nothing except the explicit multi-contrast claim of the neutral
  test ([spec](assessment_spec.md) section 2, last bullet). The drafted E2
  rule 3(b) (adjusted lower bounds over 17) is therefore stricter than any A4
  verdict in the campaign.
- Critical t values with 3 degrees of freedom: unadjusted 95% 3.18;
  Bonferroni over 2: 4.18; over 8: 6.90; over 17: 8.95. The review's example
  (+1, +2, +3, +4 points; mean +2.50, SE 0.65, t 3.87): unadjusted t lower
  bound +0.45 (passes); adjusted over 8: −1.95; over 17: −3.28 (both fail).
- With four folds the percentile bootstrap cannot reach beyond the fold
  minimum or maximum: each tail has a floor of 1/256 = 0.0039, so every
  adjusted bootstrap bound (tail 0.0031 for 8, 0.0015 for 17) equals the
  minimum fold difference (example: +1.00) and the adjusted bootstrap passes
  whenever all four folds improve. The t interval is the binding one; the
  bootstrap adds no protection at n = 4.
- Power of "t lower bound above zero" (four folds): a true gain of +2 points
  with a fold SD of 1.0 / 1.5 / 2.0 → unadjusted 0.75 / 0.45 / 0.29,
  Bonferroni over 8: 0.22 / 0.09 / 0.05, over 17: 0.11 / 0.04 / 0.02; a true
  gain of +3 with SD 1.5 → 0.75 / 0.22 / 0.11. Under a null, "all four folds
  positive" is implied by the t rule (joint 0.025 = t alone), so sign
  consistency adds no guard. The family-wise probability of a false
  replacement under the unadjusted rule over eight null candidates is about
  0.18 (independent nulls; the candidates are not null-exchangeable, each
  passed the two-seed fold-0 screen).
- Observed fold SD of the paired differences in step E: 0.66 (`headdrop03`)
  and 2.5 (`meanagg`). The eight E2 candidates' two-seed fold-0 mean gains
  were +2.1 to +2.9 (fusion `invsqrtw` +2.87, `meanagg` +2.54, `gvpaux03`
  +2.51, `resdrop01` +2.26, `esmdrop02` +2.21, `wd10` +2.11; Only-GVP `wd10`
  +2.88, `sitecountsangles` +2.90 against `sitenone`), below the +3.58
  (`headdrop03`) and +4.24 (`meanagg`) of the two chosen recipes, whose fold
  1–4 mean changes were −0.33 and +0.36 ([v3-024](#v3-024)).
- Projection, not a result: the four-class late-fusion baseline leads
  Only-ESMC on every fold (+3.68 / +1.53 / +2.04 / +0.45 / +2.06; mean +1.95;
  t lower bound with 4 df about +0.5) with no recall-gate failure, so unless
  a five/six-class cell is "better" than its four-class arm and exceeds
  74.28 mean common-four BA, the A4 Stage 6 selection will be
  `gvp_late_fusion__four_class__baseline`. The matched control and the
  original selection then coincide for the six fusion candidates, the 17
  comparisons reduce to 11 distinct ones, and the control-target-aware
  comparison is needed only if a five/six-class cell is selected (the
  requirement of [v3-025](#v3-025) stays).

**Proposed options (the user chooses; nothing is frozen):**

1. Threshold (review point 1).
   - Option A, as drafted: both Bonferroni-adjusted (over 17) lower bounds
     above zero for replacement. Honest, but 2–11% power at realistic
     effects: E2 becomes a descriptive study whose replacement rule cannot
     realistically fire.
   - **Option B (recommended):** apply the frozen A4 verdict rule unchanged
     to E2: both unadjusted 95% lower bounds above zero and the recall gates,
     with the Bonferroni-adjusted intervals over the predeclared family
     reported as a sensitivity line (`adjusted_claim_supported`), exactly as
     A4 does. Replacement additionally needs a mean gain over the original
     selection above 0.2 points, a "better" verdict against both comparators
     (own control and original selection; `sitecountsangles` also against the
     Only-GVP baseline) and the recall gates against both. Power 29–75% at +2
     points; the family-wise false-replacement rate of about 0.18 over eight
     nulls is disclosed in the report; the exploratory label and the single
     one-shot step F test remain the real safeguard.
   - Option C: Option B plus Bonferroni over the replacement comparisons only
     (at most 8, six if the selection is the fusion baseline): t critical
     6.9 → 5–22% power at +2 points.
2. Two conclusions (review point 2): accepted as a requirement. Every
   candidate carries `own_model_conclusion` in the A4 improvement-check
   vocabulary ("improvement interval-supported on folds 1–4"; "positive mean
   gain on folds 1–4, not interval-supported" = promising; "no positive mean
   gain on folds 1–4"; "blocked by a recall gate", naming the class and the
   drop) and `replacement_conclusion` ("selected"; "eligible, not best";
   "not eligible: reason"). A failed recall gate is printed in both and
   overrides "promising"; "not eligible to replace" is never summarized as
   "no effect".
3. Documentation consistency (review point 3): the handoff was reconciled in
   v3-026; the step F draft is reconciled when the step F plan is written,
   after the user approves that plan; the E2 specification names its output
   to step F explicitly. No decision needed.
4. Interpretation (review point 4): accepted. The E2 report header states
   "one seed, four folds, development-validation evidence; validation folds
   reused from the original selection; exploratory under Plan". Optional and
   outside the recorded scope unless the user adds it: a seed-43 repeat on
   folds 1–4 of a selected replacement only (four fits, about $1.6) before
   step F, reported as initialization sensitivity, never as confirmation.

**Decisions requested:** (1) Option A, B or C; (2) the two-conclusion
vocabulary; (3) no extra seed (default) or the optional seed-43 repeat;
(4) the next GPU start and its window for the 20 remaining step E fits
(forecast 4.2 lane-hours, about $3.9 plus storage: one 4 h session and one
short session; at 16:11 UTC the daily allowance left was 6 h 16 min /
$10.60). After the decisions: update plan.md step E2 items 2–3, freeze the
E2 specification (SHA-256 in the log), then the E2 CPU tooling of
[v3-025](#v3-025). The deferred 10-epoch review stays deferred until step E
is complete (its completeness checks need the 20 pending fits).

STATUS text replaced by this update, preserved verbatim:

```text
Next: read the saved review ([handoff §9](docs/campaigns/pmm_ion_metal_v3/handoff.md), log v3-026), discuss the unresolved assessment tradeoffs with the user, agree any changes, then update the documentation before implementation or training; then the 20 remaining E fits and the original E assessment; open: E2 go-ahead, step F plan, fairness arm, fallback leftovers, cost-gated augmentations, 10-epoch review, v3-024 corrections.
```

## v3-026

2026-10-08 — **pending review of the E2 assessment rules saved; the E2
documentation committed and pushed** (documentation only; no code change, fit,
GPU action, inference or test access). The user, closing the workstation, asked
that the review below be saved durably, that the documentation work of
[v3-025](#v3-025) be committed and pushed, and that no further investigation or
implementation begin. Step E2 stays planned, not implemented and not
experimentally evaluated; its assessment and replacement rules
([plan step E2](plan.md#steps), items 2 and 3) are **pending review** and not
frozen. The eight candidates, the 36-fit scope, seed 42 and the $55 gross total
planning ceiling are unchanged; the planning ceiling authorizes no GPU start.

**Pending review (user, 2026-10-08; unresolved; to be discussed before the E2
specification is frozen or any implementation or training starts; also in
[handoff section 9](handoff.md#9-pending-review-of-the-e2-assessment-rules-saved-2026-10-08-unresolved)):**

1. Promotion threshold. Bonferroni correction across 17 comparisons with only
   four folds creates a very demanding replacement gate, approximately 99.7%
   confidence intervals per comparison. A hypothetical improvement of +1, +2,
   +3 and +4 percentage points averages +2.5 points but still fails the
   adjusted t-interval check. Recorded as an unresolved design tradeoff: the
   threshold, comparison family, folds, seeds and budget are not changed
   silently, are discussed with the user before the specification is frozen,
   and are never changed after E2 results are seen to obtain a preferred
   outcome.
2. Separate experiment conclusions. Each candidate needs two separate
   conclusions: whether the change helps its own model compared with its
   matched control, and whether it qualifies to replace the overall selected
   model. Failure to replace the overall winner must not be interpreted as
   "the change is useless"; positive average gains must not conceal failed
   class-recall checks.
3. Documentation consistency. The handoff's old E → F sequence and reading
   list are reconciled with E → E2 → F and v3-025 in this update. The step F
   draft (`audits/session_notes_20261008/step_f_plan_DRAFT.md` on the campaign
   data root) takes the A4 selection as its only input and is marked as
   requiring reconciliation before implementation: step F must consume the
   final E2 selection, or the original A4 selection if the user explicitly
   cancels E2.
4. Interpretation. The development-validation/exploratory label is preserved.
   Four folds and one seed do not establish robustness across training seeds,
   and the adjusted intervals do not remove the uncertainty caused by reused
   validation data and overlapping cross-validation training sets.

**Next task (recorded):** "Read the saved review, discuss the unresolved
assessment tradeoffs with the user, agree any changes, then update the
documentation before implementation or training." The scientific questions
are not resolved here.

**Commit and push (user authorization of 2026-10-08, replacing the earlier
instruction not to commit or push):** the documentation changes of v3-025 and
this entry (plan, README, log, handoff, STATUS, playbook, improvement-plan
pointer) are committed with the campaign hook as "docs: record E2 draft and
pending review" and pushed to `origin/v3-step-a` without force; no other file
is included; training code, executable spending controls, frozen
specifications, manifests and evidence are untouched; no GPU resource,
training, inference or test evaluation was started. The commit ID and the
push verification are in the session's closeout report and in `git log`.

STATUS text replaced by this update, preserved verbatim:

```text
- Status: active (2026-10-08 v3 step E in progress: 24 of 44 fits done; step E2 amendment planned, not implemented)
- Stage: v3 steps A–D done; step E 24 of 44 fits ([log v3-024](docs/campaigns/pmm_ion_metal_v3/log.md#v3-024)); step E2 (the eight remaining step D passes on folds 1–4, 36 fits) planned, not implemented, required before step F unless the user cancels it ([log v3-025](docs/campaigns/pmm_ion_metal_v3/log.md#v3-025)); no Stage 6 confirmation, 6B refit or Stage 7.
- Best validation result: none promoted; the closed PMM core fold-0 fits are Grade 5 exploratory evidence (incomplete grid Grade 6) per the archived README above; results in the [core summary](docs/notebook_outputs/summaries/summary_pmm_core_continuation_20260928.md).
- Fold 0 of the closed campaign (multi-ion PDB groups, six Cu groups) is not representative ([findings](docs/PARAMETER_FINDINGS.md#pmm-ion-campaign-single-fold-target-and-readout-comparisons)).
- EC: the [EC1 reference](docs/archive/campaigns/ec1_standalone_v12_2026-09-14/README.md) has twelve fixed-split runs (Grade 3), not promotion; reconciliation and cross-task holdout certification precede auxiliary learning ([issues](docs/FOLLOW_UP_TECHNICAL_ISSUES.md)).
- Exact Zenodo PMM test: unknown, possibly evaluated; its source test file's label/feature metadata was read in aggregate on 2026-10-03 (no evaluation).
- Authorized now: advance GPU authorization through step F ([log v3-014](docs/campaigns/pmm_ion_metal_v3/log.md#v3-014)) within the recorded [$40 gross ceiling](docs/campaigns/pmm_ion_metal_v3/log.md#v3-003) including storage (about $12.9 left; E+F forecast $33.5–36.4); confirm the next GPU start with the user. The $55 planning ceiling selected for E2 (log v3-025) is in no execution control and authorizes no start. E2 only after its tooling, frozen specification and manifest, a refreshed forecast and the user's go-ahead; step F after E2 (unless cancelled) and its gates; no Round C.
- GPU/VM: `deepmzyme-l4` TERMINATED (verified 2026-10-08 12:17 UTC); disk and snapshot kept; failed-fallback leftovers (two disks, one snapshot, about $1.06/day) await the user's cleanup decision ([v3-019](docs/campaigns/pmm_ion_metal_v3/log.md#v3-019)).
Next: continue from the [handoff](docs/campaigns/pmm_ion_metal_v3/handoff.md): the 20 remaining five/six-class fits, the original E assessment, then the E2 CPU implementation and the deferred 10-epoch review; open decisions: E2 go-ahead, step F plan, fairness arm, fallback leftovers, cost-gated augmentations, v3-024 corrections.
Overwrite this file after preserving dated changes in the campaign's `log.md`; keep the objective/stage and campaign lines, anchor, best result and grade, caveats, dataset/test readiness, authorization and GPU/VM state, blockers, next action and evidence links. Budgets stay in the playbooks, findings in [PARAMETER_FINDINGS](docs/PARAMETER_FINDINGS.md), policy in [Plan](Plan.md), batch evidence in the [index](docs/notebook_outputs/README.md); the [history map](docs/archive/consolidation_2026-09/MAP.md) holds the recovered 09-16/17 pause and removed policy; historical next actions do not resume work.
```

## v3-025

2026-10-08 — **step E2 amendment recorded: the eight remaining step D passes,
tested one at a time on folds 1–4** (documentation only; CPU; no code change,
fit, GPU action, inference or test access; nothing in this entry authorizes a
GPU start). Requested by the user after the fold 1–4 results of
[v3-024](#v3-024) had been examined. The attached plan is recorded in the
[campaign plan, step E2](plan.md#steps), the [README](README.md), the
[handoff](handoff.md), [STATUS](../../../EXPERIMENT_STATUS.md) and the
[playbook](../../METAL_TRAINING_PIPELINE_PLAYBOOK.md#v3-step-e2-individual-confirmation-amendment-planned-not-implemented).
E2 is **planned**: not implemented, not smoke-tested, not experimentally
evaluated.

**User decisions of 2026-10-08 (this update):**

- Insert step E2 between E and F. After the remaining 20 original step E fits
  and the unchanged original A4 assessment (whose Stage 6 selection is kept as
  the reference), 36 additional fits on folds 1–4, seed 42, direct
  `four_class`, existing recipes only: 32 candidate fits (Only-GVP `wd10` and
  `sitecountsangles`; late fusion `invsqrtw`, `meanagg`, `gvpaux03`,
  `resdrop01`, `esmdrop02` and `wd10`) and four `sitenone` geometry controls.
  These are the eight step D passes that were not chosen as final recipes
  ([v3-022](#v3-022)); nothing is combined, re-tuned, re-seeded or extended to
  five/six-class training; the 10-epoch schedule proposal stays separate.
- Eight separate comparisons, each against its matched control (the complete
  step E four-class baselines, reused by checked identity; `sitenone` for
  geometry, plus the Only-GVP baseline) and each against the original step E
  selection: a fixed family of 17 comparisons with ordinary and
  Bonferroni-adjusted 95% intervals from the A4 methods (four paired
  differences, 10,000 bootstrap resamples, bootstrap seed 42, t with 3 df).
  Repeatability on folds 1–4 is reported per fold with per-class recalls; the
  fold-0 seed-42 value is the development fold; five-fold summaries are
  descriptive.
- Proposed replacement rule, to be frozen in a separate E2 specification
  before the first E2 fit with the A4 specification unchanged: both adjusted
  lower bounds above zero against the matched control and the original
  selection (geometry also against the Only-GVP baseline), a mean gain over
  the original selection above 0.2 points, no missing or zero common-four mean
  recall and no class mean recall drop above 3 points; the highest mean BA on
  folds 1–4 among eligible candidates, ties within 0.2 points by mean minimum
  recall, worst fold, SD, simpler family, stable recipe name; otherwise the
  original selection stands.
- **Step F requires E2 completion** (36 fits with verified evidence, the eight
  comparisons reported, one final selection) unless the user cancels E2 by a
  dated log entry; step F then takes the A4 selection. The selected
  configuration is step F's refit input (seed 42, terminal checkpoint, no
  ensemble, no calibration; the `both_results_secondary` label stands).
- Planning ceiling: **$55 gross in total**, storage included, for the campaign
  expanded by E2, in place of the [$40](#v3-003) figure of the original plan
  (the E+F forecast of [v3-024](#v3-024), $33.5–36.4, still fits $40, so the
  increase funds E2 only). Planning figures: about 9–12 additional VM hours,
  total about $44–51; consistent with the measured 31–37 min per three-lane
  fit of [v3-022](#v3-022) (36 fits about 6.2–7.4 three-lane hours before
  session overhead, admission slack and one rerun allowance). One L4 VM, at
  most three concurrent fits, the controller's session and daily caps
  unchanged. **The budget decision authorizes no GPU start**: the next start
  needs the user's confirmation (STATUS), and E2 fits need in addition the
  tooling, the frozen E2 specification and manifest, a refreshed forecast,
  the ceiling recorded in execution controls and the user's go-ahead.
- Documentation only: no training resumed, no VM started, no inference, no
  test evaluation, no change to training code, executable spending controls
  or frozen artifacts; the edits are staged, not committed or pushed.

**Requirements the user added during the update:**

- The E2 assessor must support an original step E selection trained with
  four, five or six classes. The existing `compare` helper summarizes its
  control as `four_class` (`summarize(control, "four_class")`) and takes only
  `challenger_target`, so it is reused unchanged for the eight
  candidate-versus-control comparisons only; "candidate versus the original
  selection" with a five- or six-class selection needs a control-target-aware
  comparison (common-four BA from the collapsed predictions on matched folds,
  the selection's native recalls under its own scheme, the same interval
  rule).
- Correction of an over-claim in the first draft of the playbook section: not
  every execution path blocks E2 today. The workstation launcher refuses E2
  units (`--step E` admits only the original 44); the VM runner's
  `--action run` checks fold, ESMC directory, extension record, hashes and
  retry identity but not step membership, so an E2 unit name passed to it
  directly would not be refused. This is a gap, closed only by the E2 manifest
  gate once built and, until then, by the rule never to call the VM runner
  directly.

**Labels and policy notes (reported, not resolved silently):**

- E2 was proposed after fold 1–4 results had been examined, so every E2
  result is development-validation evidence and, in the neutral-test wording
  of [Plan](../../../Plan.md#2-train-the-metal-classification-model) and
  AGENTS, a post-hoc exploratory arm. The original neutral four/five/six
  conclusions and the A4 Stage 6 selection are preserved separately; an E2
  replacement, if any, carries this label into step F's report.
- The frozen A4 specification is untouched: `assessment_spec.md` SHA-256
  `899005aa…` and `assessment_spec.json` SHA-256 `29694070…` equal the values
  frozen in [v3-008](#v3-008) (verified today). A4 §4 keeps its selection
  rule; E2 adds a second, stricter rule in a separate specification that may
  override that selection. This is the material change the user accepted by
  attaching the plan; the campaign's own mechanism applies (a dated log entry
  before the first run of the affected step), and the E2 specification must be
  frozen before any E2 fit.
- "Candidate versus the original step E selection" is a cross-family or
  cross-target comparison on common-four BA whenever that selection is not an
  Only-GVP or late-fusion `four_class` cell (the A4 control is Only-ESMC
  `four_class`); Plan allows ranking across targets on collapsed-four BA only,
  which this respects.
- AGENTS asks that an approved ceiling increase be recorded in campaign
  evidence and executable controls. The evidence record is this entry and the
  plan's Budget section. No executable control encodes the campaign ceiling
  (the controller has session and daily caps and an advisory `vm-budget`
  amount only), and this task excluded changes to spending controls, so that
  recording is pending and precedes any authorized E2 launch.
- Handoff correction: the ready-to-paste prompt still said "about $16.8 left on
  2026-10-08 02:30 UTC"; it now carries the [v3-024](#v3-024) figure (about
  $12.9 left at 12:17 UTC). No other number changed.

**Implementation status: none.** Exists: the trainer options and the eight
recipe definitions in the hash-frozen runner; the eight reusable controls; the
fold-0 seed-42/43 runs of every candidate and of `sitenone` (step D); the A4
interval, summary, tie-key and artifact-validation helpers; the shared lane,
admission, wait, pull, recovery and evidence logic. Refuses today: the step E
launcher (not the VM runner, see above); `extend` once step D has started (a
chained extension record is needed, as the fairness proposal noted in
[v3-022](#v3-022)); the A4 assessor's pinned specification hash. To build, in
order, before any E2 fit: the frozen E2 specification; the chained amendment
manifest (36 units, reused controls by identity, the original selection by
assessment hash, the rules, source/configuration/evidence hashes; refusals for
differing provenance, an incomplete original E assessment, missing
authorization and persistence failures); the E2 launcher with its own launch
and evidence directories and the admission check the VM runner lacks; the
separate E2 assessor accepting a four-, five- or six-class original selection,
with a machine-readable final selection; tests for the required CPU checks
(exact unit counts, one-setting overrides, geometry comparisons and baseline
reuse, refusal of missing, duplicated, mismatched or unreplayed evidence,
interval adjustment, recall gates, ties, incomplete results, retention of the
original selection, launch admission, recovery, unchanged original E
behaviour); then the existing launcher and assessment tests, the CPU smoke
checks and the documentation-contract check, a refreshed forecast and the
recorded ceiling. The playbook section lists these and contains no E2 command.

Order of work: the remaining 20 five/six-class E fits → `--step E assess` and
`evidence` (the original assessment, unchanged) → E2 CPU implementation and
checks → the user's go-ahead → E2 fits → E2 assessment and final selection →
step F. The step F draft (`audits/session_notes_20261008/step_f_plan_DRAFT.md`
on the campaign data root) still names the A4 selection as its input and is
not updated here.

Checks: `tools/check_docs_contract.py` passed after the edits (0 strict
failures; the 11 pre-existing warnings only). Commits: committed together with
[v3-026](#v3-026) as "docs: record E2 draft and pending review" and pushed to
`origin/v3-step-a` under the authorization recorded there.

STATUS text replaced by this update, preserved verbatim (every line of the
previous file that this overwrite changed or removed):

```text
- Status: active (2026-10-08 v3 step E in progress: 24 of 44 confirmation fits done; four-class baselines and improvement fits done on folds 1–4; five/six-class started)
- Stage: v3 steps A–D done; step E 24 of 44 fits ([log v3-024](docs/campaigns/pmm_ion_metal_v3/log.md#v3-024)); no Stage 6 confirmation, Stage 6B refit or Stage 7.
**The PMM core v2 campaign was closed at fold 0 on 2026-10-03 (user decision).**
Its nine fold-0 fits are final validation-only evidence; 36 fits were not run and
its neutral four-versus-five/six test is unanswered. No fivefold confirmation,
model/target promotion, final refit or held-out evaluation is claimed ([archived README](docs/archive/campaigns/pmm_ion_metal/README.md)).
- Best validation result: none promoted; the closed PMM core fold-0 fits are Grade 5 exploratory evidence (incomplete grid Grade 6) per the [archived PMM README](docs/archive/campaigns/pmm_ion_metal/README.md); results in the [core summary](docs/notebook_outputs/summaries/summary_pmm_core_continuation_20260928.md).
- v3 step E (descriptive): the two selected recipes show no clear gain on folds 1–4; the late-fusion baseline is above Only-ESMC on every fold so far ([log v3-024](docs/campaigns/pmm_ion_metal_v3/log.md#v3-024)).
Fold 0 of that campaign holds only multi-ion PDB groups and six Cu groups, so it is
not representative ([findings](docs/PARAMETER_FINDINGS.md#pmm-ion-campaign-single-fold-target-and-readout-comparisons)).
The [EC1 reference](docs/archive/campaigns/ec1_standalone_v12_2026-09-14/README.md)
retains twelve completed fixed-split runs (Grade 3), not promotion. EC workflow
reconciliation and cross-task holdout certification precede auxiliary learning
([issues](docs/FOLLOW_UP_TECHNICAL_ISSUES.md)).
- Cross-scheme ranking: the notebook Stage 6/6B route is single-scheme; comparing
  target schemes needs a campaign assessor on collapsed-four balanced accuracy
  ([TECH-010](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-010--four-class-endpoint-and-paired-metal-target-recipes-are-not-reconciled)).
- v3 replaces the size-strata folds
  ([TECH-020](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-020--greedy-k-fold-assignment-concentrates-large-groups-in-fold-0))
  and uses one terminal checkpoint rule in CV and refit
  ([TECH-027](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-027--cross-validation-and-final-refit-use-different-checkpoint-rules));
  PMM comparator citations keep the label-leak caveat
  ([TECH-028](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-028--pmm-comparator-inherits-a-true-metal-label-leak)).
The primary final-test route remains unresolved. [DATASETS](docs/DATASETS.md#test-use-ledger)
owns the ledger; this reminder preserves the default-read safety boundary:
- Non-overlap and exact PMM tests (same 316-structure set): opened 13 and 21
  times; not eligible as an unopened test. Harsh and Common-PDBID 70/30 test IDs
  come from that opened set.
- Exact Zenodo PMM test: unknown, possibly evaluated; its source test file's
  label/feature metadata was read in aggregate on 2026-10-03 (no evaluation).
- No evaluation artifacts found for CLEAN30 or CARE clusterRes30; CARE had
  incidental test-metadata exposure on 2026-09-15. Absence of found artifacts is
  not proof of no outside run.
- Authorized now: today's single-session arrangement has ended; the advance GPU authorization through step F ([log v3-014](docs/campaigns/pmm_ion_metal_v3/log.md#v3-014)) stands within the [$40 gross ceiling](docs/campaigns/pmm_ion_metal_v3/log.md#v3-003) including storage (about $12.9 left; forecast $33.5–36.4), but confirm the next GPU start with the user. Step F only after its plan and gates; no Round C.
- GPU/VM: `deepmzyme-l4` TERMINATED (verified 2026-10-08 12:16:58 UTC); disk and snapshot kept; failed-fallback leftovers (two disks, one snapshot, about $1.06/day) await the user's cleanup decision ([v3-019](docs/campaigns/pmm_ion_metal_v3/log.md#v3-019)).
Next: a new chat continues from the [handoff](docs/campaigns/pmm_ion_metal_v3/handoff.md): the remaining 20 five/six-class fits (about 4.3 h), then the E assessment and the deferred 10-epoch-schedule review; user decisions open: step F plan (draft in the campaign audits folder), fairness arm, fallback leftovers, cost-gated augmentations, log v3-024 corrections.
Overwrite this file after preserving dated changes in the campaign's `log.md`.
Keep the objective/stage and campaign lines, anchor, best validation
result and grade, caveats and mismatches, dataset/test readiness,
authorization and GPU/VM state, blockers, next action and evidence links. Exact budgets stay in the playbooks; parameter
findings stay in [PARAMETER_FINDINGS](docs/PARAMETER_FINDINGS.md), scientific
policy in [Plan](Plan.md), and batch evidence in the [index](docs/notebook_outputs/README.md).
[History map](docs/archive/consolidation_2026-09/MAP.md) includes the recovered
09-16/17 pause and earlier removed policy; historical next actions do not resume work.
```

## v3-024

2026-10-08 — **step E session 7: the improvement fits, the last four-class baselines and
the first five-class fits** (validation data only; folds 1–4; seed 42). Continuation of the
authorized plan ([v3-014](#v3-014)) under the user's go-ahead of 2026-10-08 ~07:45 UTC
(recorded below); no new authority was used.

**User decisions of 2026-10-08 (morning, this session):**

- ~07:45 UTC: the next GPU start confirmed; up to three capped sessions today (4 h + 4 h +
  about 1 h 45 min) within the remaining daily allowance, with persistence and a verified VM
  shutdown completed by 17:30 UTC; stop earlier if step E finishes; caps, reserves and the $40
  ceiling unchanged; the remaining 34 fits and the E assessment only; frozen settings unchanged;
  the existing launcher; `status` before every refill; no scratch drivers without reviewed
  corrections; the Only-ESMC fairness arm stays separate; step F, the refit and any test access
  need their own plan and approval; closeout to show baseline versus improved results, per-class
  recalls, uncertainty, completed/pending work and spending, confirmed versus inconclusive.
- ~08:55 UTC: two additional read-only reviewers authorized (completion tracking; reporting
  accuracy), consolidated below; review-driven changes await the user's decision.
- ~09:35 UTC: the workstation closes about 13:07 UTC, so session 7 is the only session today;
  commit **and push** at every closeout to keep the record complete (push authorized for the
  closeout commits of today); the arrangement is for today only, the standing rules stay.
- ~08:55 and ~11:05 UTC: a **deferred** read-only review request (a fixed 10-epoch cosine
  schedule as a possible amendment; four questions added at 11:05) starts only after this
  closeout; it authorizes no implementation, experiment, spending, refit or test access. Copies:
  `audits/session_notes_20261008/DEFERRED_amendment_review_request.md` on the campaign data root.

**Pre-flight (read-only, before the start).** A seven-agent workflow re-verified the handoff
against disk and ledger: all 10 completed E units complete in the four senses and replayed;
scores equal to the v3-023 table; ledger sums $20.48 compute; about $16.8 left at 02:30 UTC;
daily cap 9 h 45 min / $13.60 left at 07:31 UTC. Two corrections to the handoff: (1) the
launcher forecasts five/six-class units as WARM (`cache_warm` ignores folds and the step C
fold-0 records exist), so the cold cost is the parse only (6–15 min, not 2,000 s) and a cold
builder must be launched by session minute ~160; (2) `status` needs an active controller
session, not only SSH. Also: `--step` defaults to D, so every E command carries `--step E`.

**Start.** Eight same-VM retries hit the L4 stockout in `us-central1-a` (07:58, 08:05,
08:11, 08:17, 08:23, 08:28, 08:34, 08:40 UTC; nothing allocated or charged); the ninth
succeeded: `session-20261008T084732Z-3bbc7d8d` started 08:47:32 UTC with `--hours 4`
(Google-side hard stop 12:43:40 UTC, at most $3.47 gross); smoke check passed (L4, torch
2.11.0+cu128); `--step E status` reconciled 10 completed units, no lane awaiting a pull.

**Fits.** Units were launched with the launcher directly, lanes refilled after a **named**
`wait --any` (every running unit named, so a unit that ends before the first snapshot is
reported; reviewer 1's rule), `status` before every refill. Batches `e-20261008T084956Z`,
`…T093135Z`, `…T101219Z`, `…T105137Z`, `…T113642Z`, `…T114328Z`. 14 fits completed,
each at its first attempt, replayed under `pmm-v3-replay-1`, pulled and acknowledged
(24 of 44 complete; 20 pending: 9 five-class and 11 six-class baseline fits on folds 1–4; the Only-GVP six_class parse cache is the one cold set left). One refusal at 09:29 UTC
("lane0: an ended unit awaits its verified host pull") was resolved with `pull --lanes 0`
(no idle lane beyond 2 min). Terminal common-four BA and recalls (descriptive; the paired
contrasts are the assessor's):

| Unit (seed 42) | BA | Mn | Cu | Zn | Class VIII |
|---|---|---|---|---|---|
| Only-GVP baseline fold 4 | 72.76 | 66.7 | 70.1 | 78.8 | 75.4 |
| Late fusion baseline fold 4 | 78.32 | 78.6 | 74.6 | 82.5 | 77.6 |
| Late fusion headdrop03 fold 1 / 2 / 3 / 4 | 73.44 / 71.77 / 74.66 / 78.58 | 81.1 / 82.3 / 71.7 / 80.2 | 76.1 / 51.5 / 77.9 / 73.1 | 87.6 / 81.3 / 78.2 / 84.2 | 48.9 / 71.9 / 70.9 / 76.8 |
| Only-GVP meanagg fold 1 / 2 / 3 / 4 | 74.97 / 70.87 / 67.64 / 71.56 | 81.3 / 80.8 / 69.3 / 73.9 | 80.6 / 55.9 / 60.3 / 61.2 | 86.2 / 85.6 / 77.2 / 80.2 | 51.7 / 61.3 / 63.8 / 70.8 |
| Late fusion five_class fold 1 | 72.06 | 81.1 | 74.6 | 88.1 | 44.4 (native Fe 43.0; native BA 62.8) |
| Only-GVP five_class fold 1 | 71.40 | 82.1 | 76.1 | 84.8 | 42.6 (native Fe 36.5; native BA 62.0) |
| Only-ESMC five_class fold 1 | 71.86 | 79.4 | 79.1 | 84.3 | 44.6 (native Fe 41.5; native BA 63.4) |
| Only-ESMC six_class fold 1 | 71.19 | 78.2 | 77.6 | 86.0 | 43.0 (native Fe 39.2; native BA 53.9) |

Measured, not interpreted (paired differences on folds 1–4, recipe minus same-fold baseline):
late fusion headdrop03 +0.10 / −1.19 / −0.51 / +0.26 (mean −0.34); Only-GVP meanagg
+0.90 / +3.77 / −2.04 / −1.20 (mean +0.36). The fold-0 screening gains (+3.58 and +4.24
over two seeds) do not reappear as clear gains on the confirmation folds; under A4 §3 neither
can be "improvement interval-supported on folds 1–4" (the assessor's verdict is below).
Family contrast, baseline recipes, folds 1–4: late fusion minus Only-ESMC +1.53 / +2.04 /
+0.45 / +2.06 (fold 0 +3.68); Only-GVP minus Only-ESMC +2.26 / −3.82 / −5.04 / −3.50.
**Assessor (A4, folds 1–4, both improvement checks complete; copy `step_e_evidence/assessments/E_20261008T121456Z`, SHA-256 `2ab142eb…`):** late fusion `headdrop03` — mean difference −0.33 points, bootstrap interval [−0.87, +0.14] and t interval (3 df) [−1.71, +1.04] unadjusted, "no clear difference", status **"no positive mean gain on folds 1–4"**, no improvement claim, not Stage 6 matched; Only-GVP `meanagg` — mean +0.36 points, bootstrap [−1.62, +2.53] and t [−5.05, +5.76] unadjusted, "no clear difference", status **"positive mean gain on folds 1–4, not interval-supported"**, no improvement claim, Stage 6 matched pass only (eligibility, not a claim); recall gates passed in both. Under A4 §3 neither recipe is an improvement; the baseline recipes stay the step E reference. Every neutral contrast (five/six vs four per family) is still incomplete and no Stage 6 selection exists.

**Reviewer reports (user request, read-only; copies in
`audits/session_notes_20261008/` on the campaign data root).** Reviewer 1 (completion
tracking): no integrity or persistence risk; the idle-lane race of v3-023 is real (bare
`wait --any` takes `awaited` from the first snapshot, pmm_v3_step_c.py:463-466) and not
handled by code; closed procedurally by naming every running unit in `wait --any` (used from
this session on); optional code change (add lanes awaiting a pull to `awaited`) and a
playbook/handoff wording fix await the user's decision. Reviewer 2 (reporting accuracy):
every number of v3-021/v3-022/v3-023 and the E assessment reproduces from the terminal
validation predictions; two sentences of v3-023 need correction — "3–11 points above fold 0"
was computed against the seed-43 baselines (against the paired seed-42 fold-0 baselines the
range is +1.3 to +9.4) and "fold 0 holds the multi-ion groups [v3-005]" describes the v2
folds (v3 folds are size-balanced; fold 0 has 2.05 ions per entry against 1.67–1.93) —
plus stale STATUS lines (fixed in this update), a one-session-stale README line, the
v3-022 fairness-proposal wording ("gain" → "fold-0 lead, unconfirmed"; "better" is A4's
reserved word) and a handoff restatement of the E rule that omits the bootstrap interval.
The v3-023 sentences are corrected here, not edited in place: against the paired seed-42 fold-0 baselines the range is +1.3 to +9.4, and fold 0 is the development fold of size-balanced folds (v3-005); whether v3-023 is annotated in place is the user's decision.

**Session and spending.** `session-20261008T084732Z-3bbc7d8d`: 08:47:32–12:16:54 UTC,
**3 h 29 min, $3.07 gross** (stopped by `vm-stop` after the evidence copy;
TERMINATED verified 12:16:58 UTC). Daily use 2026-10-08 after this session:
5 h 44 min / $7.40 of 12 h / $18. Compute to date **$23.55 gross** (B $1.88, C $3.64,
D $12.98, E $5.05); storage about $3.5 accrued. **About $27.1 spent; about
$12.9 remains** of the $40 ceiling. Reforecast: 20 E fits about 4.3 h
three-lane plus overhead, about $3.8; step F about 1.5 h, $1.3; storage about $2.0 per
day reserved while the leftovers exist. **Total about $33.5–36.4 of $40.**
Evidence copy `step_e_evidence/evidence_20261008T121514Z` (591 files, SHA-256 checked on
both ends; `safe_to_stop` true); assessment copy `step_e_evidence/assessments/E_20261008T121456Z`
(SHA-256 `2ab142eb…`, both improvement checks complete, all six neutral contrasts incomplete, no Stage 6 selection; `held_out_test_accessed` false).

Open and unchanged: the Only-ESMC fairness arm ([v3-022](#v3-022)), the fallback leftovers,
the cost-gated augmentations, the step F plan (a draft for the user's approval is in
`audits/session_notes_20261008/step_f_plan_DRAFT.md`; nothing built), the reviewer
corrections above, the deferred schedule-amendment review.

Commits: this entry (log, STATUS, handoff refresh); pushed to `origin/v3-step-a` as the user asked.

STATUS text replaced by this update, preserved verbatim:

```text
- Status: active (2026-10-08 v3 step E in progress: 10 of 44 confirmation fits done, four-class baselines on folds 1–4; final D recipes late fusion `headdrop03`, Only-GVP `meanagg`)
- Last execution evidence: 2026-10-08 (v3 step E session 6, 2 h 15 min, $1.98 gross). Docs reconciliation: 2026-10-08.
- Current campaign: pmm_ion_metal_v3, step D running (objectives four/five/six for Only-ESMC, Only-GVP and late fusion) — [README](docs/campaigns/pmm_ion_metal_v3/README.md)
- Stage: v3 steps A–D done; step E 10 of 44 fits ([log v3-023](docs/campaigns/pmm_ion_metal_v3/log.md#v3-023)); no Stage 6 confirmation, Stage 6B refit or Stage 7.
- Authorized now: the user's 2026-10-07/08 working window has ended; the advance GPU authorization through step F ([log v3-014](docs/campaigns/pmm_ion_metal_v3/log.md#v3-014)) stands within the [$40 gross ceiling](docs/campaigns/pmm_ion_metal_v3/log.md#v3-003) including storage (about $16.8 left; forecast $33.3–37.1), but confirm the next GPU start with the user. Step F only after its plan and gates; no Round C.
- GPU/VM: `deepmzyme-l4` TERMINATED (verified 2026-10-08 02:23 UTC); disk and snapshot kept; failed-fallback leftovers (two disks, one snapshot, about $1.06/day) await the user's cleanup decision ([v3-019](docs/campaigns/pmm_ion_metal_v3/log.md#v3-019)).
Next: a new chat continues from the [handoff](docs/campaigns/pmm_ion_metal_v3/handoff.md): the remaining step E fits ($16.8_E: four-class baselines first, then the 8 improvement fits, then five/six-class, cache sets cold once), then the step E assessment; open user decisions: Only-ESMC fairness arm, fallback leftovers, cost-gated augmentations, step F plan.
Still open before step E: the final metal test label, after the user's check of
the 2026-09-24 Zenodo run. Stage 6 grouped-fold selection (or a labeled
fallback), a completed Stage 6B full non-test refit, frozen report/checkpoint
rules and a resolved final-test route must precede one-shot Stage 7. No test-based tuning, ranking, promotion, rejection or
checkpoint choice; see [Plan](Plan.md#canonical-staged-metal-training-pipeline).
```

## v3-023

2026-10-08 — **step E session 6 (first confirmation fits) and the end of the
user's working window** (validation data only; folds 1–4; seed 42). Continuation
of the authorized plan ([v3-014](#v3-014)) under the user's window decision of
[v3-022](#v3-022) (finish by 03:38 UTC, eight hours maximum).

**Start.** The first start at 00:00:50 UTC hit the L4 stockout in
`us-central1-a` (nothing allocated or charged); bounded same-VM retries at
five-minute spacing followed, with the session length computed so that the
Google-side hard stop stayed at or before 03:05 UTC. The first retry succeeded: `session-20261008T000829Z-f23d379e` started
00:08:29 UTC with `--hours 2.94` (Google-side hard stop 03:00:42 UTC, at most
$2.53 gross); smoke check passed (L4, torch 2.11.0+cu128); the step E gates
(readiness, extension record, final D assessment, recorded label) all passed on
the VM. Units were launched with the launcher directly (no scratch driver), one
lane refilled at a time after `wait --any`, `status` checked before each refill
from 01:09 UTC on.

**Fits.** Ten of the twelve four-class baseline fits of the neutral test (seed 42,
folds 1–4) completed, each at its first attempt, replayed under
`pmm-v3-replay-1`, pulled and acknowledged (batches `e-20261008T001047Z`,
`…T004448Z`, `…T004942Z`, `…T010949Z`, `…T012249Z`, `…T014046Z`,
`…T014856Z`). Late fusion fold 4 was refused by the admission rule at 01:58:29
UTC (3,742 s left, 3,960 s needed; nothing started) and Only-GVP fold 4 was
never launched: both stay pending with the 24 five/six-class and 8 improvement
units. Terminal common-four BA and recalls (descriptive; the paired contrasts
are the assessor's, and no fold-1–4 contrast is complete yet):

| Family | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|
| Only-ESMC | 71.81 | 70.92 | 74.72 | 76.26 |
| Only-GVP | 74.07 | 67.10 | 69.68 | — |
| Late fusion | 73.34 | 72.96 | 75.17 | — |

Measured, not interpreted: the confirmation folds score 3–11 points above fold
0 for every family (fold 0 holds the multi-ion groups, [v3-005](#v3-005)), Cu
recall on fold 2 is 50.0 / 50.0 / 55.9, and Class VIII recall on fold 1 is
47.5 / 48.3 / 49.1 for Only-ESMC / Only-GVP / late fusion. Phase times
14–31 min per fit. Step E assessment copy
`step_e_evidence/assessments/E_20261008T022125Z` (SHA-256 `477d173a…`,
incomplete: every contrast has missing folds; `held_out_test_accessed` false).
Evidence copy `step_e_evidence/evidence_20261008T022159Z` (457 files, SHA-256
checked on both ends; `safe_to_stop` true). One operational note: Only-GVP
fold 1 ended between a launch and the next `wait --any` snapshot, so its lane
was pulled but not reported and sat idle for 19 minutes (00:50–01:09 UTC; the
manual variant of the audit's concern 2); from then on `status` was checked
before every refill.

**Spending.** `session-20261008T000829Z-f23d379e`: 00:08:29–02:23:45 UTC, **2 h 15 min,
$1.98 gross** (stopped by `vm-stop` after the evidence copy; TERMINATED
verified 02:23:45 UTC). Daily use 2026-10-08 after this session: 2 h 15 min,
$4.40 of 12 h / $18. Compute to date **$20.48 gross** (B $1.88, C $3.64, D
$12.98, E $1.98); storage about $2.7 accrued (controller reservation about
$2.00 per day in total, of which the failed-fallback leftovers about $1.23).
**About $23.2 spent; about $16.8 remains** of the $40 ceiling. Reforecast:
step E 34 remaining fits (2 four-class baselines, 24 five/six-class with four
cold cache sets, 8 improvement) about 6.8 three-lane hours plus overhead, about
7.5 h, $6.6; step F about 1.5 h, $1.3; storage for about three more days $2.2
without the leftovers, $6.0 with them. **Total about $33.3–37.1 of $40**; the
optional Only-ESMC arm would add about $1.0.

**Closeout at the window's end.** The window's work ended with this entry: STATUS overwritten, the campaign
[handoff](handoff.md) refreshed to the verified state (ready-to-paste prompt
for a new chat), committed with the campaign hook and pushed to
`origin/v3-step-a` as the user asked; evidence on Data1 verified separately
from git (both evidence copies, both assessment copies, every pulled run and
acknowledgment); the audit folder `audits/driver_audit_20261007T193000Z/` holds
the session notes as well. The working-window
authorization of [v3-022](#v3-022) has ended; the standing advance
authorization of [v3-014](#v3-014) still covers the approved plan within the
ceiling, but the user asked that the next GPU start be confirmed with them.

Open and unchanged: the Only-ESMC fairness arm (proposal in [v3-022](#v3-022)),
the fallback leftovers, the cost-gated augmentations, the step F plan.

Commits: this entry (log, STATUS, handoff refresh); pushed to `origin/v3-step-a` at the user's request.

STATUS text replaced by this update, preserved verbatim:

```text
- Status: active (2026-10-07 v3 step D complete: 60 + 4 combination fits done; final recipes late fusion `headdrop03`, Only-GVP `meanagg`; step E awaits the final-test label)
- Last execution evidence: 2026-10-07 (v3 step D session 5, 3 h 15 min, $2.86 gross). Docs reconciliation: 2026-10-07.
- Stage: v3 steps A–D done ([log v3-022](docs/campaigns/pmm_ion_metal_v3/log.md#v3-022)): one-fold screen closed, final recipes named by the final D-B assessment; step E not started; no Stage 6 confirmation, Stage 6B refit or Stage 7.
- Authorized now: advance GPU authorization through step F ([log v3-014](docs/campaigns/pmm_ion_metal_v3/log.md#v3-014)); no new start request. Within the [$40 gross ceiling](docs/campaigns/pmm_ion_metal_v3/log.md#v3-003) including storage (about $19.3 left; forecast $32.6–36.4). Step E only after the user's final-test label is recorded (`record-e-gate`); step F only after its gates; no Round C.
- GPU/VM: `deepmzyme-l4` TERMINATED (verified 2026-10-07 20:16 UTC); disk and snapshot kept; failed-fallback leftovers (two disks, one snapshot, about $1.06/day) await the user's cleanup decision ([v3-019](docs/campaigns/pmm_ion_metal_v3/log.md#v3-019)).
Next: the user records the final-test label (then `pmm_v3_step_d.py record-e-gate --label … --log-entry v3-022`) and decides the Only-ESMC fairness arm ([log v3-022](docs/campaigns/pmm_ion_metal_v3/log.md#v3-022)); then step E (36 neutral-test + 8 improvement fits, final recipes; five/six-class cache sets build cold once); cost-gated augmentations stay "not tested (cost)".
```

## v3-022

2026-10-07 — **step D session 5: the eight site-geometry Round B fits and the
complete D-B assessment** (validation data only; fold 0; exploratory
development evidence). Continuation of the authorized step D work
([v3-014](#v3-014), [v3-018](#v3-018)); no new authority was used.

**Resume.** The workstation rebooted at 15:48 UTC, which unmounted the Data1
drive; it was remounted before any action (the worktree and campaign root were
unchanged, `git status` clean at `dca385e` = `origin/v3-step-a`). The start at
16:53 UTC hit the L4 stockout in `us-central1-a` (the fourth of the day, after
13:01, 13:06 and 13:12; nothing allocated or charged); a bounded same-VM retry
loop (five-minute spacing, at most twelve attempts, no fallback) succeeded at
the first retry.

**Session 5** `session-20261007T170135Z-a208a882`, `deepmzyme-l4` in
`us-central1-a`, started 17:01:35 UTC, hard stop 20:57:46 UTC; smoke check
passed (L4, torch 2.11.0+cu128). Units were launched by refilling each lane as
soon as its unit ended (`wait --any`), seed-43 units only after the seed-42
unit of the same cache set had completed, so that each of the four site-geometry
cache sets was built once; every launch went through the launcher's admission
and gate checks. First batch `d-20261007T170429Z` at 17:04 UTC: Only-GVP
`sitenone` 42 (lane 0), late fusion `sitenone` 42 (lane 1), Only-GVP
`sitecountsangles` 42 (lane 2), all cold.

Batches: `d-20261007T170429Z` (three cold sets, 17:04 UTC; all three completed
by 17:43 UTC, about 37 min each: the parse cache was warm, 0.6 min for 3,992
structures, and each site-geometry graph set built inside the 2-minute prepare
phase, so the 2,000 s cold allowance was not needed), `d-20261007T174535Z`,
`…T174547Z`, `…T174559Z` (late fusion `sitecountsangles` 42, Only-GVP
`sitenone` 43, late fusion `sitenone` 43; 17:45–17:46 UTC, completed 18:25 UTC),
then Only-GVP and late fusion `sitecountsangles` 43 together (18:28 UTC,
completed 18:59 UTC). Every fit completed at its first attempt, replayed under
`pmm-v3-replay-1`, pulled and acknowledged. The workstation-side refill script
(scratchpad only, not part of the repository; it only calls the launcher's
`status`, `pull`, `launch --no-wait` and `wait --any`) stopped twice on timing
edges — a routine "lane awaits its verified host pull" refusal at 17:42 UTC and
all three lanes ending within the same status snapshot at 18:25 UTC — costing
about 3 minutes of idle lanes each time; the launcher's own lane, pull,
completed-unit and admission refusals were never bypassed, and no unit was
launched twice. Assessment copies: `step_d_evidence/assessments/D-B_20261007T190153Z`
(SHA-256 `17721dda…`, "blocked: a combination run is still missing", names the
combinations) and the **final** `D-B_20261007T201334Z` (SHA-256 `bba7c728…`,
status `final`): **final step D recipes — late fusion `headdrop03`, Only-GVP
`meanagg`; Only-ESMC baseline.** Evidence copy `step_d_evidence/evidence_20261007T201431Z` (682 files, SHA-256 checked on both ends, none mismatched; `safe_to_stop` true).

**Round B complete (14 of 18 run; `posnoise01`, `outerdrop01` not tested (cost)).**
Terminal common-four BA, Δ against the same-seed control, seeds 42 / 43, mean
recall changes in points:

| Candidate | Only-GVP | Late fusion |
|---|---|---|
| `sitecountsangles` (`--site-geometry-features counts_angles` against the matched `none` control) | +0.50 / +5.29, **mean +2.90**, Cu +7.5, Class VIII +3.4, Mn −0.6: **passed** | −0.91 / −0.44, mean −0.68: not passed |

Terminal common-four BA of the eight fits (seed 42 / 43): Only-GVP `sitenone`
61.95 / 62.63, `sitecountsangles` 62.45 / 67.92; late fusion `sitenone`
73.71 / 71.78, `sitecountsangles` 72.80 / 71.34. The `sitenone` controls differ
from the baseline (whose setting is `legacy`) by −2.75 / −0.59 (Only-GVP) and
+2.09 / +3.20 (late fusion) points: descriptive only, within the 1.5–3 point
seed spread of the controls, and not a screened candidate. Round B passes:
`invsqrtw` in late fusion ([v3-021](#v3-021)) and `sitecountsangles` in
Only-GVP; `vecnorm`, Only-GVP `invsqrtw` and late-fusion `sitecountsangles`
did not pass; `posnoise01` and `outerdrop01` are "not tested (cost)".

**Combinations named by the complete D-B assessment**
(`step_d_evidence/assessments/D-B_20261007T190153Z`, SHA-256 `17721dda…`,
status "blocked: a combination run is still missing"): late fusion
`combo-gvpaux03+headdrop03+invsqrtw+meanagg+resdrop01+wd10` (best single
`headdrop03`, +3.58) and Only-GVP `combo-meanagg+sitecountsangles+wd10` (best
single `meanagg`, +4.24), each with seeds 42 and 43 against the baseline
controls. Neither combination changes learning-rate group membership (no
`structlr`), so the [v3-018](#v3-018) decay audit covers them.

**Combination fits** (batch `d-20261007T190314Z` at 19:03 UTC, three lanes;
the fourth at 19:44 UTC after the lanes were pulled, completed 20:12 UTC; terminal common-four BA,
Δ against the same-seed baseline control):

| Family | Combination | Seed 42 | Seed 43 | Mean Δ | Best single | Final recipe (assessor) |
|---|---|---|---|---|---|---|
| Late fusion | `combo-gvpaux03+headdrop03+invsqrtw+meanagg+resdrop01+wd10` | 74.44 (+2.82) | 72.27 (+3.69) | +3.25 | `headdrop03` +3.58 | **`headdrop03`** (single: the combination passed the gate but gained less) |
| Only-GVP | `combo-meanagg+sitecountsangles+wd10` | 62.74 (−1.96) | 69.51 (+6.29) | +2.17 | `meanagg` +4.24 | **`meanagg`** (single: the combination failed the gate at seed 42) |

Measured, not explained: stacking every passing late-fusion change gains less
than head dropout 0.3 alone, and the Only-GVP stack is below its baseline at
seed 42 (recalls against the seed-42 baseline: Mn +3.4, Cu -0.0, Zn +0.7, Class VIII -12.0 points). Under the plan's combination rule the best single
candidate is adopted when the combination does not beat it. All of this is a
one-fold screen with 1.5–3 point seed noise; nothing is an improvement until
folds 1–4 confirm it in step E.

**Session 5 and spending.** `session-20261007T170135Z-a208a882`: 17:01:35–20:16:42
UTC, **3 h 15 min, $2.86 gross** (stopped by `vm-stop` after the evidence copy;
TERMINATED verified 20:16:42 UTC). Twelve fits in one session (eight site-geometry,
four combination), every one at its first attempt. Daily use 2026-10-07 after
this session: 10 h 13 min, $11.25 of 12 h / $18. Compute to date **$18.50 gross**
(B $1.88, C $3.64, D $12.98 over five sessions). Storage: the campaign disk and
paused snapshot plus the failed-fallback leftovers are reserved by the controller
at about $2.00 per day in total (two of three disks and one of two snapshots are
the leftovers, about $1.23 per day of it); about $2.2 accrued to date. **About
$20.7 spent; about $19.3 remains** of the $40 ceiling.

Reforecast with the measured fit times (31–37 min wall per Only-GVP or
late-fusion fit, about 28 min per Only-ESMC fit, site-geometry and five/six-class
cache sets build inside the 2-minute prepare phase): step E 44 fits (36 neutral
+ 8 improvement) about 8.8 three-lane hours plus overhead, about 9.5 h, $8.4;
step F about 1.5 h, $1.3; storage for about three more days $2.2 without the
leftovers, $6.0 with them. **Total about $32.6–36.4 of $40**; the optional
Only-ESMC arm would add about $1.0.

**Only-ESMC fairness arm: proposal (user request of 2026-10-07, [v3-021](#v3-021)); kept separate for the user's decision, neither approved nor rejected; no fit runs without approval.**

The final late-fusion recipe is `headdrop03` alone (final D assessment above), so BRANCH B applies; BRANCH A is kept only as the record of what a combination recipe would have implied.

Question answered: does the late-fusion gain over Only-ESMC survive when Only-ESMC
receives the same *shared* training settings that the selected fusion recipe
carries, i.e. is the gain "structure adds information" rather than "fusion got
regularization that Only-ESMC did not"?

Shared settings in the selected late-fusion recipe (transferable: weight decay,
classifier-head dropout, class weighting; never transferred: `meanagg`,
`resdrop01`, `gvpaux03`, `esmdrop02/04`, `structlr`, any GVP-side change):

| Selected fusion recipe | Shared settings | Adjusted Only-ESMC recipe |
|---|---|---|
| BRANCH A: `combo-gvpaux03+headdrop03+invsqrtw+meanagg+resdrop01+wd10` adopted | `headdrop03` (head dropout 0.3), `wd10` (AdamW decay 1.0), `invsqrtw` (inverse-square-root class weights) | `only_esm__four_class__esmfair-headdrop03+invsqrtw+wd10` |
| BRANCH B: best single `headdrop03` adopted (combination not better) | `headdrop03` only | `only_esm__four_class__esmfair-headdrop03` |

Decay note: decay coefficient 1.0 removes 25% of a no-gradient weight over the
schedule at the base rate 3e-5 ([v3-018](#v3-018) audit) and Only-ESMC trains
entirely at that rate, so the same coefficient gives the fusion ESM-branch
strength, not Only-GVP's 94%. Head dropout acts on the same classifier module in
both models; `invsqrtw` changes only the loss weights.

Fits: 5 (seed 42, folds 0–4; the Only-ESMC baseline on folds 0–4 is already
part of step E). Time: about 1,800 s warm each, about 1.1 three-lane hours
(one session segment). Cost: about $1.0 gross at $0.879/h. Reuses every
completed run; no hidden-dimension or exhaustive search.

Comparison rules, recorded here BEFORE any fold 1–4 result is inspected:
(1) comparison arm only; the primary contrasts of A4 (Only-ESMC baseline vs
fusion baseline/final recipe; four/five/six neutral test) are unchanged;
(2) two descriptive paired per-fold contrasts under the A4 interval method:
adjusted Only-ESMC vs original Only-ESMC baseline, and final fusion recipe vs
adjusted Only-ESMC; fold 0 shown separately; (3) labelled "conditional on
four-class development" because the shared settings were selected on fold 0;
(4) original and adjusted Only-ESMC results are reported separately and the
original baseline is never replaced; (5) no transfer to five/six-class
training; (6) no checkpoint-rule, data, fold or test-policy change.

Implementation: a chained extension 2 of the frozen runner (one recipe, two
assessor contrasts, tests; the current `extend` is refused once step D has
started, so the extension tool needs a chained-record path), about 2–3 h CPU
before step E, then the 5 fits inside step E's sessions.

Recommendation for the applicable BRANCH B (`only_esm__four_class__esmfair-headdrop03`, 5 fits, about 1.1 three-lane hours, about $1.0 gross, plus 2–3 h CPU for the chained runner extension): low-to-moderate value. Head dropout 0.3 is the only shared setting; Only-ESMC already carries head dropout 0.2 at baseline, its best logged epoch is 3 and the audits show head dropout mainly damps the early validation peak, so the expected effect on the terminal-epoch Only-ESMC score is small. The arm would still remove the objection that the fusion gain came from a classifier-regularization difference. Skipping is defensible; running it is cheap. Decision: the user's; nothing is prepared or launched until then.

**Audit of the session's scratch refill tooling** (user request; read-only
inspection and CPU mocks, no change applied): report, mock test and the tools
themselves are kept under `audits/driver_audit_20261007T193000Z/` on the
campaign data root and summarized in [handoff.md](handoff.md). Findings in
short: both premature stops were snapshot races on temporary launcher refusals
("lane awaits its verified host pull"), the launcher's lane, pull, completed-unit
and admission guards were never bypassed, no unit ran twice, and proposed
corrections await the user's review before any such tool is reused.

**User decisions of 2026-10-07 (evening, this session):**

- **Final-test label: `both_results_secondary`.** Both step F reports — the
  full PMM test ("exact, possibly overlapped PMM test; includes rows from
  previously opened tests") and the clean subset — are secondary results; no
  primary, pristine held-out claim is made. Recorded uncertainty: the
  2026-09-24 Zenodo PMM run on Colab may have scored the whole PMM test set
  (its artifacts were lost with the Colab VM and no evaluation record exists
  either way), the non-overlap/exact PMM test set was opened 13 and 21 times in
  earlier campaigns, and PMM test label counts were read in aggregate on
  2026-10-03 without evaluation ([DATASETS ledger](../../DATASETS.md#test-use-ledger)).
  The decision changes no step E or F prerequisite: Stage 6 selection by A4,
  the Stage 6B full refit, the frozen test-row rule ([v3-012](#v3-012)), the
  clean-subset membership hashes, the refit checkpoint and seed, no ensemble
  and the calibration rule must still be recorded in the DATASETS ledger before
  any test access. Recorded for the launcher with `record-e-gate --label
  both_results_secondary --log-entry v3-022`.
- **Working window:** continue the approved plan until 03:38 UTC on 2026-10-08
  (06:38 Israel time) at the latest, eight hours from 19:38 UTC, including
  verified persistence, VM shutdown, documentation, commit and push; work that
  does not fit safely stays pending. The controller's 4 h / $6 session and
  12 h / $18 daily caps and the $40 gross ceiling are unchanged. Nothing can run
  between this session's stop and 00:00 UTC (about 1 h 10 min of today's daily
  cap remain, less than one admitted fit), so step E starts after midnight UTC
  if its gates pass, in one session planned backward from the 03:38 deadline.
- **Only-ESMC fairness arm:** kept separate for the user's decision (proposal
  below); neither approved nor rejected; it replaces no planned work.
- **Push:** the user asked for the closeout commits to be pushed to
  `origin/v3-step-a`.

Open and unchanged: fallback leftovers; cost-gated augmentations ("not tested
(cost)"); Round C (out of scope).

Commits: this entry (log, STATUS, the decided item in plan.md, the campaign handoff `handoff.md`); pushed to `origin/v3-step-a` at the user's request.

STATUS text replaced by this update, preserved verbatim:

```text
- Status: active (2026-10-07 v3 step D: Rounds A and R done, Round B 6 of 18; `headdrop03` and `invsqrtw` passed in late fusion)
- Last execution evidence: 2026-10-07 (v3 step D session 4, 3 h 24 min, $2.99 gross). Docs reconciliation: 2026-10-07.
- Stage: v3 steps A–C done; step D Rounds A and R complete, Round B 6 of 18 ([log v3-021](docs/campaigns/pmm_ion_metal_v3/log.md#v3-021)); no Stage 6 confirmation, Stage 6B refit or Stage 7.
- Authorized now: advance GPU authorization through step F ([log v3-014](docs/campaigns/pmm_ion_metal_v3/log.md#v3-014)), amended step D included; no new start request. Within the [$40 gross ceiling](docs/campaigns/pmm_ion_metal_v3/log.md#v3-003) including storage (about $22.7 left; forecast $35.3–39.5, the upper value with the fallback leftovers). Step F only after its gates and the final-test label decision; no Round C.
- GPU/VM: `deepmzyme-l4` TERMINATED (verified 2026-10-07 12:39 UTC); disk and snapshot kept; failed-fallback leftovers (two disks, one snapshot, about $1.06/day) await the user's cleanup decision ([v3-019](docs/campaigns/pmm_ion_metal_v3/log.md#v3-019)).
Next: the 8 site-geometry Round B fits, then one combination per family from the complete D-B assessment ([playbook](docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md#v3-step-d-regularization-amendment)); cost-gated augmentations stay "not tested (cost)" without the user's OK; the proposed Only-ESMC fairness arm ([log v3-021](docs/campaigns/pmm_ion_metal_v3/log.md#v3-021)) awaits the user's decision, recorded before any fold 1–4 result.
```

## v3-021

2026-10-07 — **step D session 4: Round R complete, the two CPU audits, Round B
started** (validation data only; fold 0; exploratory development evidence).
Continuation of the authorized step D work ([v3-014](#v3-014), [v3-018](#v3-018));
no new authority was used.

**Session 4** `session-20261007T091548Z-fab4dad3`, `deepmzyme-l4` in
`us-central1-a`, started 09:15:48 UTC at the first attempt (no stockout),
stopped 12:39:54 UTC and verified TERMINATED: **3 h 24 min, $2.99 gross**.
Five batches of three, all first attempts, every fit completed, replayed under
`pmm-v3-replay-1`, pulled and acknowledged: `d-20261007T091821Z`,
`…T095751Z`, `…T103657Z` (Round R), `…T111854Z`, `…T115750Z` (Round B).
Evidence copy `step_d_evidence/evidence_20261007T123827Z` (565 files, SHA-256
checked on both ends, none mismatched). Assessment copies
`step_d_evidence/assessments/D-R_20261007T111811Z` (SHA-256 `baae64a8…`,
"provisional: run D-B next") and `D-B_20261007T123755Z` (SHA-256 `ec8ed981…`,
blocked: the eight site-geometry fits are missing; `posnoise01` and
`outerdrop01` recorded "not tested (cost)").

**Round R complete (26 of 26).** Terminal common-four BA, Δ against the
same-seed control, seeds 42 / 43, mean recall changes in points:

| Candidate | Only-GVP | Late fusion |
|---|---|---|
| `headdrop03` (head dropout 0.3) | −2.27 / −1.04, mean −1.65, Class VIII −9.3: not passed | +3.74 / +3.43, **mean +3.58**, Cu +8.9, Class VIII +4.8: **passed** |
| `resdrop02` (residual dropout 0.2) | −4.50 / +4.58, mean +0.04, Class VIII −7.1: not passed | −0.33 / +2.10, mean +0.89: not passed |
| `esmdrop04` (modality dropout 0.4) | — | −1.87 / +4.46, mean +1.30: not passed |

With [v3-020](#v3-020): Round R passes are `wd10` in both families and
`headdrop03` in late fusion; `wd001`, `wd01`, `headdrop01`, `resdrop02` and
`esmdrop04` did not pass. Residual dropout disagrees between seeds in Only-GVP
at both strengths (−2.14 / +5.57 and −4.50 / +4.58).

**Round B, 6 of 18.** `vecnorm` (Only-GVP) +0.68 / −2.34, mean −0.83, Cu −6.8:
not passed. `invsqrtw` Only-GVP +0.56 / −0.64, mean −0.04, Class VIII −7.3:
not passed; late fusion +1.74 / +3.99, **mean +2.87**, Mn +4.1, Cu +5.5,
Class VIII +2.0: **passed**. Remaining: the eight site-geometry fits
(`sitenone` controls and `sitecountsangles`, four cold cache sets), then the
combinations. Provisional combinations named by the D-R assessment, before
Round B: late fusion `combo-gvpaux03+headdrop03+meanagg+resdrop01+wd10`
(`gvpaux03` over `esmdrop02`, +2.51 against +2.21; best single `headdrop03`),
Only-GVP `combo-meanagg+wd10` (best single `meanagg`); the complete D-B
assessment will add `invsqrtw` to the late-fusion set and may add site geometry.

**Audits before Round B** (CPU; exploratory; no rule, checkpoint or A4 change):

- Early stopping, `audits/early_stopping_audit_20261007T111826Z`, 51 cosine
  runs: every run stops early under the simulated patience-10 rule (median
  selected epoch 8), the selected epoch beats the terminal one by +3.4 points
  on average (one exception, Only-GVP `wd10` seed 43, −3.1), and 11 runs have
  a later improvement after the stop. The passing recipes reduce but do not
  remove the early peak: late fusion `headdrop03` +1.7, `resdrop01` +1.6,
  `wd10` +2.3, `esmdrop02` +2.2, `gvpaux03` +3.1, `meanagg` +4.7 against the
  baseline's +4.8; Only-GVP `wd10` −0.5 and `meanagg` +2.9 against +4.8. The
  v3-018 conclusion stands: no rule is adopted; the selected epochs were never
  saved and the margin is selection-optimistic.
- Train/validation gap, `audits/train_val_gap_20261007T111847Z`
  (`audit_v3_train_val_gap.py`, commit `7c1b299`; 40 paired units): every
  control fits the training fold almost perfectly (training BA 98.1–99.9
  against validation 63.2–71.6; gaps 27.4–36.3 points). The late-fusion passes
  shrink the gap by 1.7–3.4 points almost entirely through validation BA
  (training BA changes −0.8 to +0.9 points, all saturated above 97). Only-GVP `wd10` is the one recipe that fits the training fold
  less (training BA −11.5, gap −14.3, validation +2.9); `resdrop01`/`resdrop02`
  lower it by 2.2 / 1.5 without passing; `headdrop01` and `wd001` widen the gap
  by about 3 points in Only-GVP. The gap is an identity in the two deltas and
  says nothing about generalization.

**User guidance received during the session (recorded, no action taken):**
fusion is being tuned while Only-ESMC keeps its original settings, so once the
selected fusion recipe is known the user wants a small fairness assessment:
list the *shared* settings that changed (weight decay, classifier dropout,
class weighting; never ESM-branch dropout or the GVP auxiliary loss), judge
whether one adjusted Only-ESMC recipe with those settings would strengthen the
comparison, keep the original Only-ESMC baseline and the four/five/six-class
comparisons unchanged and report original and adjusted results separately. The
proposal (exact settings, fits, time, cost, question) precedes any fit;
additional training awaits the user's approval; the recipe and comparison
rules must be recorded before any fold 1–4 result is inspected; no automatic
transfer of four-class improvements to five/six-class training.

Preliminary proposal, to be finalized after the combination fits: the shared
settings in the provisional late-fusion recipe are `headdrop03` and `wd10`,
now joined by `invsqrtw`. A coefficient of 1.0 shrinks the base-rate group
(3e-5) by 25% over the schedule ([v3-018](#v3-018) decay audit), and Only-ESMC
trains entirely at that rate, so the same coefficient gives the fusion
ESM-branch strength, not Only-GVP's 94%. One recipe, Only-ESMC `four_class`
baseline plus head dropout 0.3, weight decay 1.0 and inverse-square-root class
weights, seed 42, folds 0–4 (5 fits, about 1,800 s each warm, about 1.1 three-lane
hours, about $1.0 gross), as a comparison arm only: paired per-fold contrasts
against the Only-ESMC baseline and against the final fusion recipe under the
A4 intervals, labelled conditional on four-class development, next to the
unchanged primary contrast. It would need a chained extension of the frozen
runner (the current `extend` is refused once step D has started), the recipe,
two descriptive assessor contrasts and tests, about 2–3 h of CPU work before
step E. Expected value moderate: Only-ESMC overfits earliest (training BA
97.6, validation 67.9, best logged epoch 3) and head dropout acts on the same
head module. Decision pending with the user.

**Spending and reforecast.** Compute $15.64 gross (B $1.88, C $3.64, D $10.12
over four sessions); storage about $1.6 (campaign disk and snapshot about
$1.0, failed-fallback leftovers about $0.6). **About $17.3 spent; about $22.7
remains** of the $40 ceiling. Remaining work: Round B 8 cold fits (about 3.2 h
in three rounds), combinations up to 4 (1.3 h), step E 36 + up to 8 fits
(8.8 h plus about 0.6 h for the four cold five/six-class cache sets), step F
1.5 h: about 15.4 fit-hours, with session overhead and a rerun allowance about
18 h, $15.8. Storage for about four more days: $2.2, or $6.4 while the
leftovers exist. **Total about $35.3–39.5 of $40.** The optional Only-ESMC
arm (+$1.0) fits only if the leftovers are deleted soon; otherwise it needs a
ceiling decision. Daily use on 2026-10-07 after this session: 6 h 58 min,
$8.46 of 12 h / $18, so one more 4-hour session is possible today.

Commits: `7c1b299` (gap audit tool and tests, playbook sentence); this entry.

STATUS text replaced by this update, preserved verbatim (the dated lines; in
the same update the file's other wording was shortened under its 6,000-byte
cap without changing any fact, date or number):

```text
- Status: active (2026-10-07 v3 step D: Round A done, Round R 17 of 26; `meanagg`, `wd10` passed)
- Last execution evidence: 2026-10-07 (v3 step D sessions 1–3, 8 h 06 min, $7.13 gross). Documentation reconciliation: 2026-10-07.
- Stage: v3 steps A–C done; step D Round A complete, Round R 17 of 26 ([log v3-020](docs/campaigns/pmm_ion_metal_v3/log.md#v3-020)); Round B follows; no Stage 6 confirmation, Stage 6B refit or Stage 7.
- Authorized now: advance GPU authorization for the remaining approved v3 plan through step F ([log v3-014](docs/campaigns/pmm_ion_metal_v3/log.md#v3-014)), amended step D included; no new start request needed. Within the [$40 gross ceiling](docs/campaigns/pmm_ion_metal_v3/log.md#v3-003) including storage (about $26.3 left; forecast $34.6–37.7). Step F only after its gates and the final-test label decision; no Round C.
- GPU/VM: `deepmzyme-l4` TERMINATED (verified 2026-10-07 03:37 UTC); disk and snapshot kept; failed-fallback leftovers (two disks, one snapshot, about $1.06/day) await the user's cleanup decision ([v3-019](docs/campaigns/pmm_ion_metal_v3/log.md#v3-019)).
Next: the 9 remaining Round R fits, then Round B (14) and the combinations, per the [playbook](docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md#v3-step-d-regularization-amendment); the user's work window ended 2026-10-07 04:10 UTC. Reforecast before Round B; cost-gated augmentations need the user's OK.
```

## v3-020

2026-10-07 — **step D sessions 2 and 3; closeout of the user's work window**
(validation data only; fold 0; exploratory development evidence). The window
recorded in [v3-019](#v3-019) ended at 04:10 UTC; the VM was stopped and
verified TERMINATED at 03:37 UTC.

Sessions, all on `deepmzyme-l4` in `us-central1-a` after same-VM retries
(each failed start is free; the fallback pass of v3-019 stays spent):

| Session | UTC | Running | Gross | Fits |
|---|---|---|---|---|
| 1 `session-20261006T191031Z-2d26fe96` | 19:10–22:23 | 3 h 13 min | $2.83 | 12 |
| 2 `session-20261006T224120Z-22b521f0` | 22:41–01:32 | 2 h 51 min | $2.51 | 12 |
| 3 `session-20261007T013440Z-cb560726` | 01:34–03:37 | 2 h 02 min | $1.79 | 9 |

33 fits, 8 h 06 min, **$7.13 gross**; every fit completed on its first
attempt, replayed under `pmm-v3-replay-1`, pulled and acknowledged; evidence
copies `step_d_evidence/evidence_20261006T222239Z`, `…T013116Z` and
`…T033532Z` (423 files in the last, SHA-256 checked on both ends);
assessment copies under `step_d_evidence/assessments/` (latest
`D-R_20261007T033512Z`, blocked: nine Round R fits missing).

**Round A complete (16 of 16).** Passed: `meanagg` in both families
(+4.24 Only-GVP, +2.54 late fusion), and in late fusion `resdrop01` (+2.26),
`gvpaux03` (+2.51) and `esmdrop02` (+2.21; alternatives with `gvpaux03`).
Not passed: `resdrop01` in Only-GVP (−2.14 / +5.57), `structlr` (+0.91).

**Round R, 17 of 26** (terminal common-four BA, Δ against the same-seed
control; mean over seeds 42 and 43):

| Candidate | Only-GVP | Late fusion |
|---|---|---|
| `wd001` (decay 0.01) | −5.22 / −0.66, mean −2.94, Class VIII −12.3: not passed | −0.01 / +1.16, mean +0.57: not passed |
| `wd01` (decay 0.1) | +0.17 / −0.35, mean −0.09: not passed | −0.25 / +1.76, mean +0.75: not passed |
| `wd10` (decay 1.0) | +2.05 / +3.70, **mean +2.88**, Mn +10.1, Class VIII −2.8: **passed** | +1.18 / +3.05, **mean +2.11**: **passed** |
| `headdrop01` (0.1) | −3.39 / −2.02, mean −2.70: not passed | −2.56 / +0.26, mean −1.15: not passed |
| `headdrop03` (0.3) | not run | seed 42 done, seed 43 missing |
| `resdrop02`, `esmdrop04` | not run | not run |

Measured, not explained: of the three decay strengths only the strongest
passes, in both families; the smaller strengths are within seed noise or
harmful; a lower head dropout hurts both families. Step D is on the full
path: Round B is entered after Round R, and each family will have a
combination run (late fusion: `meanagg`, `resdrop01`, the better of
`gvpaux03`/`esmdrop02`, `wd10` and whatever else passes; Only-GVP: `meanagg`,
`wd10`, …). All of this is a one-fold screen with seed noise of 1.5–3
points; nothing is an improvement until folds 1–4 confirm it in step E.

Remaining D–F work: Round R 9 fits (`headdrop03` ×3, `resdrop02` ×4,
`esmdrop04` ×2), Round B 14 runnable fits (the four cost-gated augmentations
default to "not tested (cost)"), at most 4 combination fits, step E 36 + up
to 8 fits, step F.

**Spending.** Compute to date $12.65 gross (B $1.88, C $3.64, D $7.13).
Storage: the campaign disk and the paused snapshot about $0.8 to date; the
failed-fallback leftovers (two 150 GB recovery disks and snapshot
`deepmzyme-fallback-873d08b26968`, 44.7 GiB) add about $1.06 per day while
they exist. **About $13.7 spent; about $26.3 remains.** Reforecast of the
remaining work: about 18 fit hours plus session overhead, slack and one
rerun allowance, about 22 h, $19.2; storage for about three more days $1.7,
or $4.8 if the leftovers stay. **Total about $34.6–37.7 of $40.** The
leftovers' cleanup is the user's decision (controller `vm-fallback
--finalize` is for a successful recovery; these need a separate deletion
decision); they are not needed, every result being on the workstation.

The parse cache of the new code directory now holds the four-class sets;
five- and six-class sets (step E) parse cold once each.

Still open: the final-test label (before step E); the cost-gated
augmentations (Round B; default not tested); cleanup of the fallback
leftovers.

STATUS text replaced by this update, preserved verbatim:

```text
- Status: active (2026-10-06 v3 step D running: Round A 12 of 16 fits done; `meanagg` passed)
- Current campaign: pmm_ion_metal_v3, step D running (objectives four/five/six for Only-ESMC, Only-GVP and late fusion) — [README](docs/campaigns/pmm_ion_metal_v3/README.md)
- Stage: v3 steps A–C done; step D Round A 12 of 16 fits ([log v3-019](docs/campaigns/pmm_ion_metal_v3/log.md#v3-019)): `meanagg` passed in both families; Round R next; no Stage 6 confirmation, Stage 6B refit or Stage 7.
- Authorized now: advance GPU authorization for the remaining approved v3 plan through step F ([log v3-014](docs/campaigns/pmm_ion_metal_v3/log.md#v3-014)), amended step D included; no new start request needed. Within the [$40 gross ceiling](docs/campaigns/pmm_ion_metal_v3/log.md#v3-003) including storage (about $30.9 left before session 2; forecast about $36.5). Step F only after its gates and the final-test label decision; no Round C.
- GPU/VM: `deepmzyme-l4` RUNNING (session 2 from 2026-10-06 22:41 UTC, hard stop 02:11 UTC); disk and snapshot kept; failed-fallback leftovers (two disks, one snapshot) await the user's cleanup decision ([v3-019](docs/campaigns/pmm_ion_metal_v3/log.md#v3-019)).
Next: finish Round A (4 fits), then Round R (26) per the [playbook](docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md#v3-step-d-regularization-amendment), in sessions of at most 4 h until 2026-10-07 04:10 UTC. Reforecast before Round B; cost-gated augmentations need the user's OK.
```

## v3-019

2026-10-06 — **step D session 1, the evening's user decisions and the second
start** (validation data only; fold 0; exploratory development evidence).

User decisions, each confirmed in the session after a pasted message:

- About 19:05 UTC: the controller's daily caps were raised to 12 h and $18 per
  UTC day (`MAX_DAILY_RUNNING_HOURS`, `MAX_DAILY_ESTIMATED_COST_USD`; backup
  `~/deepmzyme-vm/state/config.env.before_daily_caps_20261006`). The 4 h / $6
  session cap and the campaign's $40 gross ceiling are unchanged. A bounded
  same-region `vm-fallback` pass (4 h) was authorized for the `us-central1-a`
  stockout, including archiving the completed 2026-09-28 receipt. Commits
  `43d26ff`…`88c6d9d` were pushed to `origin/v3-step-a`.
- 22:10 UTC: the work window runs until **2026-10-07 04:10 UTC (07:10 IDT)**,
  as back-to-back sessions of at most 4 h; no change to the ceiling, the
  experiments or the test gates. Hidden-dimension tuning was assessed on
  existing evidence and deferred (widths were never tuned for the ion task;
  the results point at overfitting and inputs, not capacity; a width change
  alters several components at once).

Starts: four same-VM attempts between 18:13 and 19:03 UTC hit the L4 stockout
in `us-central1-a` (nothing started or charged); the attempt at 19:09 UTC
succeeded.

**Session 1** `session-20261006T191031Z-2d26fe96`, 19:10:31–22:23 UTC:
**3 h 13 min, $2.83 gross**; stopped and verified TERMINATED at 22:24 UTC.
Bundle `43d26ff` was applied into `/home/mechti/projects/DeepMzyme_v3_ext1`
(fold set reused after a byte-level check), and the extension was recorded at
19:13:33 UTC (SHA-256 `ade3f022…`, the worktree's runner hashes; all 12
completed units resolve unchanged; copy under `step_d_evidence/extension/`).
Twelve fits completed, replayed (largest probability difference 5e-6), pulled
and acknowledged; evidence `step_d_evidence/evidence_20261006T222239Z` (220
files, SHA-256 checked on both ends). The first round parsed structures cold
(14.8 min): the parse-cache key contains the default RING path, which is
derived from the code directory; RING is off, so the loaded data are
unchanged. The graph cache was warm (5,927 and 1,471 hits, no miss) and later
rounds parsed in 0.6 min. One-time cost: about 13 min per cache set, four
more sets in step E.

Terminal epoch-50 common-four BA, fold 0; Δ against the same-seed control:

| Candidate | Family | Seed 42 | Seed 43 | Mean Δ | Screen |
|---|---|---|---|---|---|
| baseline (control) | Only-GVP | 64.70 (step C) | 63.22 | — | — |
| baseline (control) | late fusion | 71.62 (step C) | 68.58 | — | — |
| `meanagg` | Only-GVP | 67.54 (+2.84) | 68.85 (+5.64) | +4.24 | passed |
| `meanagg` | late fusion | 73.30 (+1.69) | 71.98 (+3.39) | +2.54 | passed (Mn −2.1) |
| `resdrop01` | Only-GVP | 62.56 (−2.14) | 68.78 (+5.57) | +1.71 | not passed (seed 42) |
| `resdrop01` | late fusion | 73.54 (+1.93) | 71.17 (+2.59) | +2.26 | passed |
| `structlr` | late fusion | 71.71 (+0.10) | 70.31 (+1.73) | +0.91 | not passed |

`gvpaux03` and `esmdrop02` run in session 2. Assessment copies:
`step_d_evidence/assessments/D-A_20261006T222219Z` (blocked, four fits
missing). The seed spread of the controls is 1.5 (Only-GVP) and 3.0 (late
fusion) points, so the single-seed target differences of step C ([v3-016](#v3-016))
lie within seed noise. Round B will be entered, since Round A has passes.
Every gain is a one-fold screen; an improvement is claimed only after folds
1–4 in step E.

**Second start.** 22:24 and 22:26 UTC: stockout. 22:28–22:38 UTC: fallback
pass `873d08b26968` (authorized above): snapshot
`deepmzyme-fallback-873d08b26968` created; `us-central1-b` stocked out; the
`us-central1-c` create also failed with `ZONE_RESOURCE_POOL_EXHAUSTED`
(operation record; no instance exists), but the operator's 10-minute command
limit cut the controller before it recorded that, so the receipt
(`state/fallback.json`) still shows that attempt as requested and refuses
further allocation; the attempt window has expired, and the controller state
was not edited by hand. **Leftovers, kept pending the user's cleanup
decision and counted toward the ceiling:** recovery disks
`deepmzyme-l4-recovery-873d08b26968` in `us-central1-b` and `us-central1-c`
(150 GB each, about $0.49 per day each) and the snapshot (47.9 GB, about
$0.08 per day). At 22:41 UTC the same VM started in `us-central1-a`:
`session-20261006T224120Z-22b521f0`, hard stop 02:11:13 UTC; the smoke check
passed and `gvpaux03` (both seeds) and `esmdrop02` (seed 42) were launched at
22:43 UTC. Spending to date: compute $8.35 (B $1.88, C $3.64, D session 1
$2.83) plus storage; about $30.9 remained before session 2, before the
leftover storage above.

STATUS text replaced by this update, preserved verbatim:

```text
- Status: active (2026-10-06 v3 amended step D ready on CPU: Round R extension, launcher, audits; D fits next)
- Last execution evidence: 2026-10-06 (v3 step C session 2, 58 min, $0.86 gross). Documentation reconciliation: 2026-10-06.
- Current campaign: pmm_ion_metal_v3, step D prepared (objectives four/five/six for Only-ESMC, Only-GVP and late fusion) — [README](docs/campaigns/pmm_ion_metal_v3/README.md)
- Stage: v3 steps A–C done; amended step D (A → R → conditional B) is CPU-verified ([log v3-018](docs/campaigns/pmm_ion_metal_v3/log.md#v3-018)), no D fit yet; no Stage 6 confirmation, Stage 6B refit or Stage 7.
- Authorized now: advance GPU authorization for the remaining approved v3 plan through step F ([log v3-014](docs/campaigns/pmm_ion_metal_v3/log.md#v3-014)), amended step D included; no new start request needed. Within the [$40 gross ceiling](docs/campaigns/pmm_ion_metal_v3/log.md#v3-003) including storage (about $33.8 left; forecast $29–37). Step F only after its gates and the final-test label decision; no Round C.
- GPU/VM: `deepmzyme-l4` TERMINATED (verified 2026-10-06 10:08 UTC; 150 GB disk kept, about $0.49/day gross); snapshot `deepmzyme-paused-20261003` kept (about $1.74/month).
Next: step D Round A (16 fits), then Round R (26), per the [playbook](docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md#v3-step-d-regularization-amendment); the first session records the extension on the VM. Reforecast before Round B; cost-gated augmentations need the user's OK.
```

## v3-018

2026-10-06 — **amended step D: CPU integration, readiness gate, early-stopping
audit and cost reforecast** (CPU only: no fit, GPU action or test access).
Scope, from the user's continuation request of the same day: implement and
execute the amended step D, Round R included, within the existing ceiling. It
grants no larger budget, Round C, new model family, new training objective or
other checkpoint policy. Commits `43d26ff` and `d03c047` (code and tests).

**Implemented** (planned in [v3-017](#v3-017), now code):

- Extension 1, `v3-ext1-round-r`: seven Round R recipes (`wd001`, `wd01`,
  `wd10`, `headdrop01`, `headdrop03`, `resdrop02`, `esmdrop04`), groups of
  alternative strengths, round order A → R → B and the 26-fit Round R list.
  `--action extend` records `campaign_extension.json` once, before the first
  step D fit. It binds the prepared manifest, runner, recipe, profile, source,
  step-B setting and A4 hashes and every completed run to the new runner
  files. It is refused if step D has started or if the new code rebuilds any
  completed unit's identity differently. Later runs bind the record; earlier
  runs are reused by a checked identity and are never relabelled.
- Assessor: Round R, the amended stopping rule, one strength per setting in
  the single combination, and cost-gated candidates recorded as "not tested
  (cost)". The numerical A4 gates are unchanged.
- Bundle `43d26ff`, bound to the first bundle `20d5b06` (same source tree, A4
  specification and folds). Four files differ: the two changed runner files,
  the assessor and the bundle tool. It is applied into its own code directory;
  the first one and the fold set on the VM are not touched.
- Launcher `pmm_v3_step_d.py` for steps D and E, reusing the step-C lane, wait,
  pull, recovery and evidence logic. Its gates are listed in the
  [playbook](../../METAL_TRAINING_PIPELINE_PLAYBOOK.md#v3-step-d-regularization-amendment).
- CPU audits: effective decay, readiness and early stopping (below).

**Readiness gate: ready** (`audits/d_readiness_20261006T180939Z`, SHA-256
`a69fb126…`; 13 checks, all passed):

- source tree, prepared recipes, profile, replay policy and A4 unchanged; only
  `pmm_v3_campaign.py` and `run_pmm_v3_campaign.py` differ from the prepared
  runner files;
- unit counts 16 + 26 + 18, at most 4 combination fits, 36 + at most 8 for E;
- each of the 58 candidates resolves to its control plus exactly its own
  setting (Round R: one key, the intended value; `gvp_weight_decay` unset);
- all 12 completed step C units keep their recorded identity and argv under
  the new code; both seed-42 controls verify on the workstation (checkpoint,
  predictions, replay);
- alternatives cannot combine; R follows A without a pass; D ends after R
  without any pass; a pass in R enters B;
- screen thresholds are the frozen A4 values; none of 96 planned commands
  names held-out data; the launcher refuses before readiness;
- 256 tests passed in the gate (270 with the two v3 test files outside it); 43 CPU
  smoke checks passed.

A dry run on a workstation shadow of the real campaign root (evidence copy,
frozen inputs, pulled runs; both bundles applied as on the VM) recorded the
extension, reproduced the stored step C assessment exactly under the new
code, reported Round A as 16 missing fits and refused a changed runner file.

Limits: the real extension record is written on the VM in the first step D
session, where `extend` repeats these checks. The decay audit uses terminal
checkpoint weights (initial weights are not saved) and this workstation's
torch 2.11.0; the VM has the same version with another CUDA build. Step E
launcher gates are covered by tests only. Step F tooling (refit, test pass)
is not part of this extension. The first readiness report was "not ready"
for a tool reason (its test subprocess inherited an MKL setting) and is kept
as `audits/not_ready_d_readiness_20261006T175949Z_tool_env_error`.

**Effective weight decay** (`audits/d_decay_audit_20261006T175933Z`, SHA-256
`db1e6ad2…`; real AdamW and cosine schedule, zero gradients, 18,550 steps =
371 per epoch × 50, from the real runs' step counters; 5,927 training ions).
Cumulative factor on a weight that receives no gradient signal, ideal / FP32:

| Family, group (parameter tensors) | Initial LR | 0.0001 (baseline) | 0.01 | 0.1 | 1.0 |
|---|---|---|---|---|---|
| Only-GVP, all (115) | 3e-4 | 0.99972 / 0.99990 | 0.9720 / 0.9720 | 0.7529 / 0.7529 | 0.0585 / 0.0585 |
| Late fusion, GVP rate (84) | 3e-4 | 0.99972 / 0.99990 | 0.9720 / 0.9720 | 0.7529 / 0.7529 | 0.0585 / 0.0585 |
| Late fusion, base rate (31) | 3e-5 | 0.99997 / 1.00000 | 0.9972 / 0.9971 | 0.9720 / 0.9720 | 0.7529 / 0.7529 |

- The baseline coefficient is numerically inert, as
  [TECH-029](../../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-029--verified-input-and-training-behaviours-with-small-measured-effect)
  states: 0.01% in the GVP-rate group and exactly nothing in FP32 in the
  base-rate group.
- One coefficient acts about ten times more strongly on the GVP-rate group
  than on the base-rate group (ESM branch, gate, head). In late fusion,
  `1.0` removes 94% against 25%, `0.1` 25% against 2.8%, `0.01` 2.8% against
  0.3%.
- For every tested strength the median FP32 factor is within 0.0002 of the
  ideal one. No parameter is exempt; parameters without a gradient are never
  updated (15 tensors in Only-GVP, 3 in late fusion).
- `structlr` moves 9 tensors to the GVP rate (93 / 22). A combination of
  `structlr` with a decay strength is audited again before it runs.

This measures the decay operation only. It predicts neither trained weight
norms nor accuracy.

**Early-stopping audit: exploratory, separate from the official results**
(`audits/early_stopping_audit_20261006T174022Z`). No trainer, checkpoint or A4
change. Rule simulated sequentially on logged validation common-four BA:
patience 10, `min_delta` 0, at most 50 epochs, earliest epoch on a tie, best
epoch before the stop. Nine cosine fold-0 runs (seed 42):

| Run | Min val-loss epoch | Best BA epoch, full history (retrospective) | Simulated stop | Simulated selection: epoch, BA | Official epoch 50 BA | Difference |
|---|---|---|---|---|---|---|
| Only-ESMC four | 6 | 3 | 13 | 3, 72.1 | 67.9 | +4.2 |
| Only-ESMC five | 8 | 8 | 18 | 8, 72.2 | 69.6 | +2.6 |
| Only-ESMC six | 8 | 8 | 18 | 8, 72.4 | 69.4 | +3.1 |
| Only-GVP four | 12 | 33 | 43 | 33, 69.8 | 64.7 | +5.1 |
| Only-GVP five | 7 | 7 | 17 | 7, 69.7 | 66.1 | +3.6 |
| Only-GVP six | 7 | 7 | 17 | 7, 68.7 | 62.0 | +6.6 |
| Late fusion four | 7 | 4 | 14 | 4, 75.2 | 71.6 | +3.6 |
| Late fusion five | 7 | 9 | 19 | 9, 76.0 | 71.8 | +4.3 |
| Late fusion six | 7 | 34 | 15 | 5, 75.7 | 74.1 | +1.5 |

- The rule stops all nine runs early (epochs 13–43, a mean of 31 epochs
  saved). Its selected epoch is above epoch 50 in all nine: +1.5 to +6.6
  points, mean +3.8. The selected-minus-terminal recall differences average
  Mn +7.9, Cu +6.2, Zn +0.7 and Class VIII +0.5 points; single runs range
  from −6.5 (Zn) to +20.0 (Mn).
- One later improvement would have been missed (late fusion six-class,
  epoch 34, under 0.1 point).
- **None of the nine selected checkpoints was saved.** Every score in the
  "simulated selection" column is a logged hypothetical, not a recovered or
  replayed model.
- The difference is optimistic: the epoch is chosen and scored on the same
  validation fold. This is the selection luck of
  [TECH-027](../../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-027--cross-validation-and-final-refit-use-different-checkpoint-rules),
  which the terminal rule was chosen to avoid. It is not an estimate of the
  gain on new data.
- Stopping a 50-epoch cosine run at epoch t is not a t-epoch cosine run, so
  a shorter schedule is not measured here. The validation-loss minimum
  (epochs 6–12) is not the BA optimum in every run.
- Fixed-LR runs, kept apart (four runs, the old-fold regression included):
  +3.1 to +11.3 points, on histories that oscillate late ([v3-016](#v3-016)).

No rule is adopted. The audit is repeated on the two-seed step D histories;
regularization may change the useful duration. If earlier stopping still
looks useful then, the proposal will be a separate prospective comparison of
a shorter fixed duration, frozen before confirmation and used in the final
refit as well.

**Forecast** (measured warm three-lane times of [v3-016](#v3-016); the
controller's gross rate of $0.879 per hour; storage $0.55 per day):

| Work | Fits | Three-lane hours |
|---|---|---|
| D Round A | 16 | 3.4 |
| D Round R | 26 | 5.5 |
| D Round B, if entered | 14 runnable | 3.7 (four cold cache builds) |
| D combination | at most 4 | at most 1.3 |
| E neutral | 36 | 7.1 |
| E improvement | at most 8 | at most 1.7 |
| F refit and test pass | — | 1.5 |

- Spent by 2026-10-07 00:00 UTC: $5.52 compute (B $1.88, C $3.64) plus about
  $0.69 storage = **about $6.2; about $33.8 remains**.
- Full path (B entered, both families improved): 24.2 fit hours, about 3.6 h
  of session overhead over about 12 sessions, 1.2 h of lane and admission
  slack and 1.0 h for reruns: about 30 h, $26.4. Storage for about 7 more
  days: $3.9. **Total about $36.5 of $40; margin about $3.5.**
- If no candidate passes in A or R (no B, combination or improvement runs):
  about $28.6 in total.
- Reserved for confirmation and the final refit (E neutral, F, their sessions
  and storage): about $10.3. Step D is not allowed to consume it.
- The four cost-gated augmentation fits are outside this forecast. Each
  rebuilds its graphs every epoch (7.6–9.5 h of graph building per fit on
  this PC, [playbook](../../METAL_TRAINING_PIPELINE_PLAYBOOK.md#pmm-ion-metal-v3-campaign-pmm_ion_metal_v3)),
  so they do not fit the ceiling. Without the user's recorded OK they are
  "not tested (cost)".
- Reforecast before Round B and before step E. A zone stockout that delays
  sessions adds storage days; the margin covers about six.

Session plan: per UTC day one session at the 4-hour cap and one of about
2 hours (6-hour daily cap). Each launches three units per round and refills
lanes as units end; the admission rule (1.25 × forecast + 900 s) is
unchanged. The first session applies the bundle and records the extension
before any fit.

Open decisions (asked of the user; work that does not depend on them
continues): the final-test label, due before step E; the cost-gated
augmentations, due at Round B (default "not tested (cost)").

STATUS text replaced by this update, preserved verbatim:

```text
- Status: active (2026-10-06 v3 step C complete: all 12 fold-0 cells, regression gate passed; step D next)
- Current campaign: pmm_ion_metal_v3, step C complete (objectives four/five/six for Only-ESMC, Only-GVP and late fusion) — [README](docs/campaigns/pmm_ion_metal_v3/README.md)
- Stage: v3 steps A–C done ([log v3-016](docs/campaigns/pmm_ion_metal_v3/log.md#v3-016)): fold-0 baselines and diagnostic closeout (exploratory); no Stage 6 confirmation, Stage 6B refit or Stage 7.
- Authorized now: advance GPU authorization for the remaining approved v3 plan through step F ([log v3-014](docs/campaigns/pmm_ion_metal_v3/log.md#v3-014)); no new start request needed. Within the [$40 gross ceiling](docs/campaigns/pmm_ion_metal_v3/log.md#v3-003) including storage (about $34.2 left 2026-10-06). Step F only after its gates and the final-test label decision; no Round C.
Next: CPU integration and checks for amended D (A → regularization R → conditional B) and the D/E launcher; then execution within recorded scope. The [planning amendment](docs/campaigns/pmm_ion_metal_v3/log.md#v3-017) is not yet implemented; it preserves A4 and requires a fresh D–F cost forecast before fits.
```

## v3-017

2026-10-06 — **regularization planning amendment**, requested by the user after
the step C discussion. Documentation only: no code change, new fit, GPU action
or test access. This amendment follows the step C observations; it is not a
claim that the new candidates were predeclared before those observations.

The existing D plan tested one residual-dropout strength and one modality-dropout
strength, with no weight-decay or head-dropout screen. The user asked to update
the plan if this was insufficient. Add a bounded Round R between A and B;
the [metal playbook](../../METAL_TRAINING_PIPELINE_PLAYBOOK.md#v3-step-d-regularization-amendment)
owns its exact strengths, counts and readiness outputs. The
[campaign plan](plan.md#steps) owns the amended stopping and combination rules.
The historical improvement ranking now points to this current route.

Rationale: step C shows a training/validation gap, and TECH-029 identifies
near-inert baseline decay at the configured learning rates. These justify a
controlled regularization screen, not a predicted accuracy gain. Use the same
D families, target, seeds and metric. Keep the numerical A4 gates, baseline
neutral target test, terminal checkpoint policy and final-refit/test rules.
Different strengths of the same setting are alternatives; combine only passing
choices, once per family, before confirmation on folds 1–4. Only-ESMC remains
the original baseline, so this is not equal-budget tuning of all three families.

Implementation status: the trainer flags exist. The frozen campaign runner,
manifest registry and assessor still support the original A/B recipes and
stopping rule. A versioned, hash-bound extension and CPU verification are
required before the amended D; no frozen manifest, A4 specification or existing
evidence is rewritten by this amendment. The user's advance authorization in
[v3-014](#v3-014) remains recorded for its approved scope. This request updates
plans; it launches no work and grants no budget increase.

**Provisional forecast, not measured R runtime:** 26 additional warm fits,
about 6–8 VM hours including lane/session overhead, inferred from the measured
GVP/fusion times in [v3-016](#v3-016). Allow about $6–9 extra including 1–2
additional storage days. Reserving roughly $1–2 for up to four combination fits
not itemized in that earlier forecast gives approximately **$35–39 total**
against the unchanged $40 gross ceiling. This assumes the original cost-gated
augmentations remain unrun and no prolonged retries or storage delay. Before
execution, reforecast all accepted D–F work, combinations, recovery allowance
and accrued storage. If it no longer fits, use the existing stop/close-out or
increase-ceiling decision; never silently omit confirmation to fund tuning.

STATUS next-action text replaced by this amendment, preserved verbatim:

```text
Next: extend the workstation launcher to step D/E units (CPU), then step D Round A (16 warm fits, about two sessions) under [log v3-014](docs/campaigns/pmm_ion_metal_v3/log.md#v3-014).
```

## v3-016

2026-10-06 — **step C complete; diagnostic closeout** (validation data only;
fold 0, seed 42; exploratory, not confirmation).

GPU session 2 (`session-20261006T090942Z-90d418df`) ran 09:09–10:07 UTC:
**58 min, $0.86 gross**. Its start needed five same-VM retries over 43 min
because of a zone stockout. The VM was stopped and verified TERMINATED. It ran
the three v2-recipe runs and Only-ESMC six-class; all completed and replayed
(largest probability difference 1.22e-6). All 12 step C cells exist:
11 new fits plus the reused step-B cell. The frozen assessor
(`assessments/C_20261006T100621Z`, SHA-256 `4d1867c1…`, copied and checked) and
the evidence (`step_c_evidence/evidence_20261006T100646Z`, 174 files) are on
the workstation.

Common-four BA, fold 0 (cosine = v3 baseline at its terminal epoch 50;
fixed = v2 recipe; best epoch descriptive only):

| Family | Four, cosine | Five, cosine | Six, cosine | Four, fixed LR, epoch 50 | Fixed LR, best epoch |
|---|---|---|---|---|---|
| Only-ESMC | 67.9 | 69.6 | 69.4 | 67.6 | 72.1 (epoch 3) |
| Only-GVP | 64.7 | 66.1 | 62.0 | 58.9 | 70.2 (epoch 25) |
| Late fusion | 71.6 | 71.8 | **74.1** | 71.0 | 74.2 (epoch 48) |

Only-ESMC `four_class`, the likely Stage 6 control, has recalls Mn 56.5,
Cu 69.9, Zn 72.2 and Class VIII 73.2. These are higher or lower than PMM's
published values only descriptively, and not a matched comparison
([v3-011](#v3-011)).

Diagnostic answers. They come from existing histories, predictions and replay
receipts, with no new fits. Script and outputs:
`step_c_evidence/diagnostic_20261006/`.

1. **Old-fold, old-recipe reproduction: passed its gate.** Best epoch 89.11
   against 89.47, and the epochs 41–50 mean 85.70 against 85.78. The code
   reproduces v2.
2. **Fixed LR versus cosine at epoch 50** (cosine minus fixed): Only-ESMC
   +0.3, late fusion +0.6, Only-GVP **+5.8**. Separately, fixed LR leaves the
   Mn/Class VIII balance oscillating at the end. Over epochs 41–50, Mn recall
   spans 49.7–78.2 (late fusion) and 23.4–63.5 (Only-GVP) under fixed LR,
   against standard deviations of 1.5–3.5 points under cosine. A fixed-LR
   best epoch (late fusion 74.2 at epoch 48) is therefore a noisy peak, not a
   stable level. This is best-versus-terminal reporting, kept apart from the
   epoch-50 contrast.
3. **The decrease is shared across families**, largest in Only-GVP. Two
   decreases must be kept apart:
   - From v2's 89.5 to about 72 on the new fold: the same code and recipe give
     89.1 on the old fold, so this is a fold effect. The near-copy-disjoint,
     size-balanced fold is harder; it is not a code regression.
   - From the best to the terminal epoch: 1.5–6.6 points under cosine
     (Only-ESMC 2.6–4.2, late fusion 1.5–4.3, Only-GVP 3.6–6.6), and 3.2–11.3
     points under fixed LR.
4. **Mn explains most of it, through Mn→Class VIII.** Mn recall at epoch 50
   is 45–69 (cosine) in every family, against 84 on the old fold. 17–42% of
   Mn ions are predicted Class VIII, plus a share predicted Zn. Class VIII
   recall is 59–74. These errors are systematic: 79 Mn ions are wrong in at
   least 5 of 7 cosine models and 28 in all 7, concentrated in a few PDB
   groups (`4uxa` alone has 20 ions, about 4 recall points). Fe→Mn (3–21% of
   346 Fe ions; highest in late fusion, lowest in Only-ESMC) is mostly
   model-specific: 72 of those ions are wrong in only one model and 12 in at
   least 5. Five- and six-class training reduces the same errors only in late
   fusion and Only-ESMC:
   - late fusion: Mn→VIII 17–20% against 25%, Fe→Mn 14–17% against 21%;
     six-class 74.1 against 71.6;
   - Only-ESMC: Mn 59–61 against 57;
   - Only-GVP six-class is worse.

   One fold and one seed, so this is not the neutral test.
5. **Curves: overfitting in every run.** Validation loss is lowest at epochs
   6–14, then rises 1.5–4× by epoch 50. Training common-four BA at epoch 50
   is 94.6–100%. Validation BA falls far less than validation loss, so the
   late epochs mostly add overconfidence. Cosine stabilizes the terminal point
   but does not prevent the overfitting.

**Most likely explanation.** The new fold has no near-copies across the
split. Mn and Class VIII (mostly Fe) sites that are not near-copies of
training sites are hard to separate with the current inputs, and every family
overfits the training sites after about epoch 10. The fixed-LR schedule adds
end-of-training oscillation of the Mn/Class VIII boundary; cosine removes the
oscillation but not the overfitting.

**Evidence.** The regression run reproduces v2 on the old fold. The pattern is
the same in all three families and both schedules. Mn→VIII errors are
concentrated on the same ions and groups across models. Fe→Mn errors move
between models. The late-epoch recall spread is large under fixed LR and small
under cosine.

**Uncertain.** One fold and one seed: seed variability is unmeasured until
step D's seed-43 controls. Fold 0's group composition can shift Mn recall
(one group holds about 4 points). Whether earlier stopping or a lower learning
rate would help is not tested; the terminal rule is fixed by the plan. Why
`4uxa`-like Mn sites look Class-VIII-like is not examined (no new
investigation).

**One additional targeted check is not necessary.** The focused replay of the
fixed-LR late-fusion epoch-50 checkpoint would compare one snapshot of a run
whose Mn recall spans 49.7–78.2 over epochs 41–50, so its per-ion
Fe→Mn overlap would mostly measure that oscillation. The aggregates and the
cross-model overlap already show Fe→Mn to be model-specific. The step D
seed-43 controls update this interpretation.

**Forecast update** (measured). Warm three-lane fits take about 2,330 s (late
fusion), 2,220 s (Only-GVP) and 1,190–1,870 s (Only-ESMC) end to end, about
4.6 fits per hour, which matches the step-B projection for warm fits. Cold
cache builds add about 2,000–2,200 s each; none remain for the fold-0 recipes
of Round A.

Remaining approved work:

| Work | Fits | Three-lane hours |
|---|---|---|
| D Round A | 16 | about 4.3 |
| D Round B | 14 runnable (+4 cost-gated) | about 4 |
| E neutral (36) and improvement (up to 8) | 36–44 | about 9.5 |
| F refit and test pass | — | about 1.5 |

With about 0.3 h of overhead per session over roughly 9 sessions (the 6 h
daily cap), that is about 22 VM hours, or $19 compute, plus about $3 of storage
over 5–6 days. **About $28 of the $40 ceiling in total; about $12 margin.**

Spending: step B $1.88, step C $2.78 + $0.86, storage about $0.28.
**About $34.2 remains.**

STATUS text replaced by this update, preserved verbatim:

```text
- Status: active (2026-10-06 v3 step C session 1 done: regression gate passed, 8 of 12 cells; 4 units remain)
- Last execution evidence: 2026-10-06 (v3 step C session 1, 3 h 09 min, $2.78 gross). Documentation reconciliation: 2026-10-06.
- Current campaign: pmm_ion_metal_v3, step C running (objectives four/five/six for Only-ESMC, Only-GVP and late fusion) — [README](docs/campaigns/pmm_ion_metal_v3/README.md)
- Stage: v3 steps A and B done; step C session 1 done ([log v3-015](docs/campaigns/pmm_ion_metal_v3/log.md#v3-015)): regression gate passed, 7 new fold-0 cells plus the reused one; no Stage 6 confirmation, Stage 6B refit or Stage 7.
- GPU/VM: `deepmzyme-l4` TERMINATED (verified 2026-10-06 08:13 UTC; 150 GB disk kept, about $0.49/day gross); snapshot `deepmzyme-paused-20261003` kept (about $1.74/month).
Next: step C session 2 (the three v2-recipe runs and Only-ESMC six-class) with `pmm_v3_step_c.py`, then `assess`, the step C diagnostic closeout ([log v3-014](docs/campaigns/pmm_ion_metal_v3/log.md#v3-014)) and step D Round A.
```

## v3-015

2026-10-06 — **step C, GPU session 1** (under the advance authorization of
[v3-014](#v3-014); validation data only). The first `vm-start` at 04:51 UTC
hit a zone stockout (`ZONE_RESOURCE_POOL_EXHAUSTED`, us-central1-a). Nothing
started. A same-VM retry at 05:02 succeeded; no zone move was made, because a
new `vm-fallback` pass needs separate authorization. Session
`session-20261006T050327Z-157a0187` ran 05:03–08:12 UTC: **3 h 09 min, $2.78
gross**. The VM was stopped and verified TERMINATED. `vm-setup --stages
ssh,smoke` passed; inputs, `load_workers` 4 and the VM's runner hashes were
verified before any launch.

**Regression gate passed** (old v2 fold 0, v2 recipe and execution settings,
alone in lane 0; cold cache, 2,112 s):

- all 50 epochs completed, replay confirmed;
- best-epoch common-four BA 89.11 (reference 89.47, difference −0.35);
- epochs 41–50 mean 85.70 (reference 85.78, difference −0.08);
- band ±3.0 points.

Then seven units ran, refilled three at a time; all completed and passed
replay under `pmm-v3-replay-1`:

- late fusion five- and six-class;
- Only-GVP four-, five- and six-class;
- Only-ESMC four- and five-class.

The largest probability difference was 9.96e-6 (Only-GVP five-class, inside
the 1e-5 tolerance). Every lane was pulled and acknowledged.

Measured end-to-end times under three lanes:

| Units | Fit + replay | Preflight |
|---|---|---|
| Late fusion, cold | 4,507–4,526 s | 31 s |
| Only-GVP, cold | 4,064–4,380 s | 32–61 s |
| Only-ESMC, warm | 1,739–1,849 s | 46–60 s |

Cold parsing took 15 min under contention, against 6.1 min serially. The
Only-GVP cold forecast (4,300 s) undershot by about 170 s and the Only-ESMC
warm forecast (1,800 s) by about 100 s. Later launches used measured values
(`--fit-seconds 4600` and `2000`); the 1.25 × + 900 s rule was unchanged.

Measured rate: 7 units in the three-lane phase (05:43–08:11, with five cold
cache builds) is 2.8 fits per hour. With the regression run and setup in the
3.15-hour session it is 2.5 per hour. The step-B projection of 4.61 per hour
does not hold for mixed families with cold caches; the warm rate is
re-measured in session 2.

Not yet run: Only-ESMC six-class and the three v2-recipe runs. No unit fit the
admission rule after 08:11; 2,930 s remained before the hard stop and the
shortest unit needed about 3,400 s.

Evidence:

- `step_c_evidence/evidence_20261006T081147Z` (133 files, SHA-256 checked on
  both ends);
- workstation copies of every unit under `durable/lane0..2`.

Spending toward the $40 ceiling: $40 − ($1.88 step B + $2.78 session 1 +
storage about $0.25 to date) ≈ **$35.1 remains**. Results are reported after
step C completes (the closeout of v3-014).

STATUS text replaced by this update, preserved verbatim:

```text
- Status: active (2026-10-06 v3 step C prepared on CPU; step C awaits the user's typed GPU authorization)
- Last execution evidence: 2026-10-06 (v3 step B GPU session, 2 h 08 min, $1.88 gross). Documentation reconciliation: 2026-10-06.
- Current campaign: pmm_ion_metal_v3, step C prepared (objectives four/five/six for Only-ESMC, Only-GVP and late fusion) — [README](docs/campaigns/pmm_ion_metal_v3/README.md)
- Stage: v3 steps A and B done ([log v3-010](docs/campaigns/pmm_ion_metal_v3/log.md#v3-010)); step C prepared on CPU (launcher, PMM paper recheck, test-row rule; [log v3-013](docs/campaigns/pmm_ion_metal_v3/log.md#v3-013)); no step C runs, Stage 6 confirmation, Stage 6B refit or Stage 7.
- GPU/VM: `deepmzyme-l4` TERMINATED (verified 2026-10-06 03:09 UTC; 150 GB disk kept, about $0.49/day gross); snapshot `deepmzyme-paused-20261003` kept (about $1.74/month).
Next: v3 step C per the [metal playbook](docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md#pmm-ion-metal-v3-campaign-pmm_ion_metal_v3) after the user's typed authorization: `vm-start --confirm`, the regression fit alone, then the other 11 fits three at a time (about two sessions, [log v3-013](docs/campaigns/pmm_ion_metal_v3/log.md#v3-013)).
```

## v3-014

2026-10-06 — **user decision: advance, conditional GPU authorization for the
remaining approved plan.** It is recorded here and in STATUS so that later
sessions do not ask again for routine start authorization. Across three
messages the user said, in substance:

- Do not start the VM yet. First finish the CPU preparation, the review, the
  necessary corrections and the execution plan, then present the session plan
  and the cost estimate.
- Once that work is complete and the required checks pass, the GPU start and
  the automatic execution of the remaining approved plan are authorized,
  without asking again. This replaces the earlier rule that the user types
  AUTHORIZE VM START before each routine start.
- The authorization covers steps C, D, E and F. Step F starts only after its
  existing gates pass (Stage 6 selection by the A4 rule; the report,
  checkpoint, test-row and clean-subset rules frozen in the
  [DATASETS ledger](../../DATASETS.md#test-use-ledger) before any test access)
  and after the outstanding final-test label decision is resolved.
- The $40 gross ceiling for the whole campaign is unchanged; it includes
  earlier spending and storage. All scientific and test gates stay. Round C
  stays outside scope.
- Ask the user only for a genuine blocker, a necessary budget increase or an
  unresolved scientific decision. Under the plan these are:
  - the final-test label, due before step E;
  - any cost-gated candidate (9a, 9b, or any candidate forecast above three
    normal fits);
  - a failed regression gate, or a second failure in step C or E (stop for
    diagnosis);
  - a forecast above the ceiling (stop and close out, or record a larger
    ceiling).
- Keep a lightweight diagnostic closeout of step C. It answers five questions:
  - whether the old-fold, old-recipe regression passed its gate;
  - fixed learning rate versus cosine at epoch 50 per family, kept separate
    from best-versus-terminal reporting;
  - whether the decrease is shared by Only-ESMC, Only-GVP and late fusion, and
    how the four-, five- and six-class arms compare on common-four;
  - which class recalls explain the decrease, especially Mn and Class VIII;
  - what the training and validation curves suggest about overfitting.

  It reuses the assessor, histories, replay receipts and predictions, with no
  new fits and no new reporting framework. Only if the aggregates leave the
  Fe-to-Mn problem unexplained, it adds one focused CPU replay of the fixed-LR
  late-fusion epoch-50 checkpoint against the cosine run on the same validation
  ions. The interpretation is updated when step D's seed-43 controls exist.

Unchanged operating rules: one active GPU; every session ends with the
evidence copy, `vm-stop` and a verified TERMINATED; the disk and the snapshot
are kept; every step is logged and committed; no merge or push without the
user's OK.

## v3-013

2026-10-06 — **step C prepared on CPU** (no GPU; the VM stayed TERMINATED).
The user decided the same day that nothing new is added to the approved plan.
The four/five/six-class arms stay as they are. A shorter-schedule candidate, a
PMM-parity refit on 7,901 sites and Mn-specific features were discussed and not
added. The two items the plan foresees are recorded in
[v3-011](#v3-011) (paper recheck) and [v3-012](#v3-012) (test-row rule).

State check (03:10 UTC). Data1 is mounted with 88 GB free, and the worktree is
clean at `71fd9bc`. `vm-status` reports TERMINATED. `vm-report --no-ssh`
(gross rate $0.87917/h; session caps 4 h and $6.00; daily caps 6 h and $10)
shows 1 h 05 min and an estimated $2.17 used today (UTC). Spending toward the
$40 ceiling: step B $1.88, snapshot since 2026-10-03 $0.15, stopped disk since
01:05 UTC $0.05. **$37.92 remains** (03:33 UTC).

Throughput note: the 4.61 completed, replay-verified fits per hour with three
lanes ([v3-010](#v3-010)) is a **projection**. It extrapolates 50-epoch
throughput from 10-epoch concurrent probes; the only measured 50-epoch fit ran
serially. The measured step C rate replaces it in the step C closeout
(end to end, mixed families, cold cache builds included), and every later
forecast uses the measured rate.

Step C session plan and forecast (gross). It is simulated from the step-B
measurements and the launcher's admission rule:

- **Session 1** (at most 237 min; today's UTC cap has 4 h 55 min left):
  - setup and checks, about 15 min;
  - the regression fit alone, about 30 min (about 52 min if its cache set is
    cold);
  - then the 11 units, refilled three at a time, longest and cold-cache units
    first. Each family's measured end-to-end time replaces its forecast after
    its first fit; the 1.25 × + 900 s margin stays.
  - Expected: all 11 or 10 of them. Up to 4 remain if the regression cache is
    cold and the fits run 15% slower.
- **Session 2**, only if units remain: 0.8–1.6 h, on the next UTC day if
  today's cap is short.
- **Cost:** session 1 at most $3.47 (237 min × $0.879/h; the controller's
  estimate with its 15% margin is $3.99); session 2 $0.7–1.4; storage about
  $0.55 per day. Step C totals about $4–6, leaving about $32–34.

Launcher: `pmm_v3_step_c.py` (new, workstation, standard library only). It
reuses the step-B launcher's session, SSH, lane-readiness, host-pull and
evidence helpers. Commands:

- `units`, `session`, `status`: read-only views;
- `regression`: runs alone in lane 0;
- `launch UNIT@LANE ...`: one to three units, each detached under `setsid`;
- `wait [--any]`, `pull`: recovery and per-lane pulls;
- `recover-lane K`, `archive-failed`, `assess`, `evidence`.

A launch is refused in any of these cases:

- the regression run has not passed its gate (a failed gate prints STOP);
- any lane awaits its verified host pull;
- the target lane holds a running or interrupted unit, a live child, a held
  lock, or a launch without an exit code;
- the unit is outside step C, is the reused late-fusion `four_class` cell, is
  completed, claimed or already launched, or is listed twice;
- the lane is outside the recorded three;
- the session is expired or lacks 1.25 × forecast + 900 s (+60 s).

`wait` never launches anything. It pulls every lane whose unit ended
(retrying a pull up to three times) and reports outcomes from the VM statuses.
It reports a launch from before a VM stop or reboot (boot ID changed), or one
whose detached shell never started, as lost instead of waiting for it.

`archive-failed` is refused while any lane awaits its pull. `recover-lane K`
runs the lane's own recovery code (unchanged `pmm_execution`) for a runner that
died mid-unit, then a verified pull. `evidence` always copies, but fails while
a unit still runs or a lane awaits its pull or recovery.

Admission forecasts come from step B: late fusion 2,400 s, Only-GVP 2,300 s,
Only-ESMC 1,800 s, plus 2,000 s while the unit's cache set has no completed
fit. The cache set is the target with or without ESMC; the parse and graph
caches are keyed by label scheme and ESMC directory. The 2,000 s is the
measured cold-minus-warm gap of 1,508 s × 1.3 for overlapping cold builds. The
regression run gets 3,800 s.

The step-B launcher gained two shared helpers (`allocation_args`, and
`cmd_evidence` patterns) with unchanged step-B behaviour. Nothing in `src/`,
`scripts/` or the four hashed runner files changed. Their SHA-256 still begin
`ce6ddec8`, `21b798ef`, `6c6a0d73` and `6e4c2997`, so the prepared VM root stays
valid. The VM needs no new bundle, because the launcher runs on the
workstation.

Tests and checks:

- `tests/test_v3_step_c_launcher.py` covers each required protection,
  including a real local run of the detached launch script and state probe.
- A rehearsal on a copy of the step-B evidence: all three lanes are ready, the
  regression run is admitted alone in lane 0, the units are refused until its
  gate passes, and the reused cell shows as completed.
- An independent read-only review (three lenses, each finding checked by a
  skeptic, mutation testing on scratch copies) confirmed these issues. All
  were fixed with tests:
  - one blocker: `archive-failed` while a lane awaited its pull would have
    broken that pull for good;
  - two major gaps: no recovery for a lane whose runner died mid-unit (now
    `recover-lane`), and a running unit whose PID record is missing could
    read as lost, so `evidence` would have called a vm-stop safe (a held
    lane lock or live child now counts as running);
  - untested protections: "never relaunch" on every wait path, the evidence
    SHA comparison, admission by the slowest unit of a batch, and the
    command-line wiring;
  - minor points: hints from the VM statuses, pull retries, `evidence`
    refusing an unsafe stop, a contention allowance on cold caches, a grace
    period for a launch that never started, and exit-code labels.

  Three findings were refuted. The step-B refactor was checked equivalent on
  every step-B command.
- All 239 v3 tests pass (186 before this step), and the docs contract has no
  strict failure.
- The playbook documents the step C commands, and its illustrative
  `set-execution` example now shows the recorded three-lane setting.

STATUS text replaced by this update, preserved verbatim:

```text
- Status: active (2026-10-06 v3 step B complete: FP32, three concurrent lanes; step C awaits the user's spending authorization)
- Current campaign: pmm_ion_metal_v3, step B complete (objectives four/five/six for Only-ESMC, Only-GVP and late fusion) — [README](docs/campaigns/pmm_ion_metal_v3/README.md)
- Stage: v3 steps A and B done ([log v3-010](docs/campaigns/pmm_ion_metal_v3/log.md#v3-010)); execution setting recorded (FP32, three lanes); no step C runs, Stage 6 confirmation, Stage 6B refit or Stage 7.
- Authorized now: CPU work only; step C needs the user's spending authorization within the [$40 gross ceiling](docs/campaigns/pmm_ion_metal_v3/log.md#v3-003) including storage; no refit or held-out evaluation.
- GPU/VM: `deepmzyme-l4` restored from the snapshot 2026-10-05, TERMINATED after step B (150 GB disk kept, about $0.49/day gross); snapshot `deepmzyme-paused-20261003` kept (about $1.74/month).
Next: v3 step C per the [metal playbook](docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md#pmm-ion-metal-v3-campaign-pmm_ion_metal_v3) after the user authorizes its spending (about two sessions, [log v3-010](docs/campaigns/pmm_ion_metal_v3/log.md#v3-010)).
```

## v3-012

2026-10-06 — **user decision: which PMM test rows count** (the plan's open
decision before step F; asked and answered before step C; no test data were
read). The user chose the recommended rule:

- every reconstructable row is scored under the training input contract;
- symmetry-LINK rows are flagged, not dropped;
- rows that cannot be scored count as errors;
- the report states "N of 1,488 scored", with the training-eligible subset as
  a sensitivity line.

Under its default context policy, the PMM cohort code
(`src/benchmarking/pmm_ion_cohort.py`, the `missing_protein_symmetry_context`
branch) marks symmetry-LINK rows unresolved. The deferred step-F
test-preparation builder must therefore apply this rule itself. It lives
outside `src/`, because any `src/` change makes every run on the prepared
campaign root refuse, including the Stage 6B refit. It is tested on synthetic
or training-side data before any test access, and its membership hashes are
frozen in the [DATASETS ledger](../../DATASETS.md#test-use-ledger).

Unchanged: the clean subset is never compared with PMM; the final-test label
decision is still due before step E.

## v3-011

2026-10-06 — **PMM paper recheck before step C**, required by the plan's fixed
decisions (CPU only; no test data). Source: PinMyMetal, Nat. Commun. 2025,
16:3043, main PDF page 4, Figure 2. Two independent readers and a reconciler
read 300–600 dpi renders, and all 32 cells of panels a and b agree.

- **Panel a** (5-fold CV; rows "True metal", columns "Predicted metal"):
  columns sum to 1.000/1.000/1.001/1.001 and rows to 1.077/1.104/0.704/1.117,
  so it is **column-normalized**. Under the printed axes, the diagonal (Mn
  90.3, Class VIII 73.3, Cu 62.9, Zn 73.8; mean 75.1, a number the paper never
  prints) is per-class precision, although the text calls these values
  "accuracies". The cited numbers are correct, but 75.1 is not a balanced
  accuracy.
- **Inference, not a figure value:** back-solving panel a with the oversampled
  training counts of Supplementary Table 10 (Mn 2,590, Class VIII 2,629, Cu
  400, Zn 2,301; these 7,920 rows match PMM's released `classmodel_train_set`)
  gives implied CV recalls Mn 84.2, Class VIII 75.3, Cu 52.0 and Zn 79.3. That
  is a **BA-equivalent of about 72.7** (72.5–72.9 under rounding). It assumes
  the CV matrix pooled that set once and the axes were not swapped when
  plotted; the figure alone cannot exclude a swap, and PMM's released code has
  no CV or plotting step.
- **Panel b** (test, 1,488 rows): rows sum to 1.000, and the rows times the
  Supplementary Table 10 test counts (Mn 167, Class VIII 252, Cu 64, Zn 1,005)
  give whole numbers. It is **row-normalized**: Mn 88.6, Class VIII 57.5, Cu
  59.4 and Zn 65.9 are recalls, so **67.85 is a valid mean recall** (balanced
  accuracy).

Consequence for reporting (no plan change): step F's comparison target stays
67.85 (Fig. 2b), worded as the plan fixes. PMM's CV values are cited as
per-class precisions of a column-normalized matrix, never as a balanced
accuracy. Any CV context line shows 72.7 labelled "BA-equivalent inferred from
Supplementary Table 10 counts". The frozen A4 specification is unchanged; its
`pmm_published` numbers appear in the step C assessment as context only.

Evidence: renders, transcriptions, solver and reconciled verdicts in
`/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v3/pmm_paper_recheck_20261006/`
(`SHA256SUMS`). The 2026-10-06 outlook review reached the same reading
independently.

## v3-010

2026-10-06 — **plan step B complete** (one GPU session, authorized by the user
with AUTHORIZE VM START for step B only). `vm-restore` recreated `deepmzyme-l4`
from snapshot `deepmzyme-paused-20261003` (new instance `8553555212712383956`,
restore receipt in `~/deepmzyme-vm/state/`); session
`session-20261005T225707Z-26ada0e5` ran 2 h 08 min, **$1.88 gross**, and the VM
was stopped and verified TERMINATED. The 150 GB disk and the snapshot are kept.

Setup: `vm-setup --stages ssh,smoke` passed (L4, torch 2.11.0+cu128). The
restored VM had no loader setting, so `~/.config/deepmzyme/runtime.json` was set
to the documented `load_workers` 4 before any probe (speed only; not part of
any run identity). The bundle `bundles/20d5b06` was applied with every hash
re-verified (source tree `212db0e3…`, the A3-accepted tree), and the campaign
root was prepared once (fold set `e6b21364…`, replay policy `pmm-v3-replay-1`).

Probes (gvp_late_fusion four_class baseline fold 0 seed 42; short probes 10
epochs; every completed probe passed the independent replay under the v3 policy
and was pulled and acknowledged on the workstation):

| Probe | Lanes | Per epoch | Elapsed (fit + replay) | Projected 50-epoch fit, end to end |
|---|---|---|---|---|
| w1-fp32-r1 (cold caches, prepare 1,400 s) | 1 | 24.4 s | 2,155 s | 1,756 s (warm caps) |
| w1-fp32-r2 | 1 | 24.0 s | 647 s | 1,722 s |
| w2-fp32-a/b | 2 | 27.1 s | 699 / 697 s | 1,923 / 1,922 s |
| w3-fp32-a/b/c | 3 | 34.1–34.3 s | 793–795 s | 2,336–2,344 s |
| full-fp32 (50 epochs) | 1 | 24.0 s | 1,664 s | 1,719 s (actual 1,727 s incl. checks and pull) |

Measured completed, replay-verified fits per hour (end to end, including
pre-admission checks of 31–36 s and host persistence of 32–60 s): serial FP32
2.07; two lanes 3.74 (1.81×); **three lanes 4.61 (2.23×)**. Gates (as frozen in
`76395bf`): serial pair agrees (common-four BA difference 0.20 points, largest
class recall difference 0.6 points); every batch member agrees with w1-fp32-r1
(BA within 0.46 points; largest class recall difference 2.74 points, Cu, two
ions); host: memory available at least 24% (three lanes), GPU memory at most 8%,
median load1 per CPU 0.55 and peak 0.81 (three lanes), no CPU overload at all.
Peak process memory 7.4 GiB per lane.

AMP is **rejected**: `w1-amp-r1` failed in its first training batch with
`RuntimeError: index_add_(): self (Float) and source (Half) must have the same
scalar type` (GVP message aggregation, `src/model.py:654`, under autocast). Its
one unchanged rerun (after `archive-failed`; attempt 1 kept on the VM and on the
workstation under `superseded/`) failed identically, so the failure is final
and the AMP follow-ups were not required. Making AMP work would need a `src/`
change, a new A3 and a new preparation; with the concurrency gain it is not
needed for the budget.

Decision (`speed_report.json` SHA-256 `8fccd2e1…`, recomputed and matched by
`set-execution`): **FP32, three concurrent lanes**; the full-fp32 run is reused
as step C cell `gvp_late_fusion__four_class__baseline__fold0__seed42`
(terminal common-four BA 0.716 on the new fold 0; best epoch 4 at 0.752 is
descriptive only; replay largest difference 1.1e-6). The new fold 0 is not
comparable with the v2 fold 0 (near-copy-disjoint, size-balanced); step C
reports fold-0 values as exploratory context.

Evidence: workstation copies of every attempt under
`/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v3/durable/lane0..2`
(10 acknowledged transfers); 75 evidence files (manifest, settings, claims,
`step_b/` with host samples and reports, lane states and events, statuses,
acknowledgments, archived attempt) in `step_b_evidence/evidence_20261006T010332Z`,
SHA-256 checked on both ends; dated report copies in `step_b_evidence/reports/`.

Spending toward the $40 gross ceiling: step B $1.88, snapshot storage since
2026-10-03 about $0.14; about **$37.98 remains**. Forecast for C–F from the
measured rate (late fusion 1,739 s serial; Only-GVP and Only-ESMC scaled by the
v2 family ratios; 2.23× with three lanes; five further cold cache builds and the
serial regression run in C; 15% packing loss at session ends; 0.4 h overhead per
session; disk $0.49 and snapshot $0.06 per calendar day):

| Scenario | Remaining VM h | Compute + IP | Disk | Snapshot | B–F total |
|---|---|---|---|---|---|
| D ends after Round A | 20.5 | $17.6 | $4.0 (8 days) | $0.5 | **$24** |
| Full D (A, B, combinations, improvement check) | 27.6 | $23.7 | $5.9 (12 days) | $0.7 | **$32** |

Step C alone: about 4.3 h of work, about 5.8 VM hours over two sessions (the
237-minute session limit), about $5.0 of compute plus disk days. Augmented
candidates (9a, 9b) remain cost-gated. Unchanged: the frozen A4 specification,
the fold caveat (near-copy, not homology, separation) and the final-test label
decision due before step E. No held-out data were read.

STATUS text replaced by this update, preserved verbatim:

```text
- Status: active (2026-10-06 v3 step B prepared, CPU only; awaits the user's typed GPU authorization)
- Last execution evidence: 2026-10-06 (v3 A3/A5 repeated, CPU). Documentation reconciliation: 2026-10-06.
- Current campaign: pmm_ion_metal_v3, step B prepared (objectives four/five/six for Only-ESMC, Only-GVP and late fusion) — [README](docs/campaigns/pmm_ion_metal_v3/README.md)
- Stage: v3 step A done ([log v3-008](docs/campaigns/pmm_ion_metal_v3/log.md#v3-008)) and step B prepared ([log v3-009](docs/campaigns/pmm_ion_metal_v3/log.md#v3-009)); no GPU runs, Stage 6 confirmation, Stage 6B refit or Stage 7.
- Authorized now: CPU work only ([plan](docs/campaigns/pmm_ion_metal_v3/plan.md)); a GPU start needs the user to type AUTHORIZE VM START, within the [$40 gross ceiling](docs/campaigns/pmm_ion_metal_v3/log.md#v3-003) that includes campaign storage ([log v3-009](docs/campaigns/pmm_ion_metal_v3/log.md#v3-009)); no refit or held-out evaluation.
- GPU/VM: retired; provider inventory verified no VMs/disks at 2026-10-03 13:48:51 UTC. One restore-checked standard snapshot remains (34.77 GiB, approximately $1.74/month gross). [Receipt and recovery boundary](docs/archive/campaigns/pmm_ion_metal/storage_retirement_20261003.json); [storage decision](docs/archive/campaigns/pmm_ion_metal/log.md#pmm-011).
Next: v3 step B (GPU speed check) per the [metal playbook](docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md#pmm-ion-metal-v3-campaign-pmm_ion_metal_v3) (`vm-restore` after the user's typed authorization, then `pmm_v3_step_b.py`); `/` has 16 GB free.
```

## v3-009

2026-10-06 — **step B preparation completed on CPU** (nothing ran on a GPU; no
VM exists). After the step-B readiness audit, the user decided and asked for one
preparation pass (these decisions change step-B defaults before the first
step-B run, as plan.md allows):

1. Replay: every v3 fit, probe and the regression run is independently replayed
   under the explicit policy `pmm-v3-replay-1`: probabilities within 1e-5 (the
   pmm-core-replay-v1 tolerance), compared exactly on the saved 8-decimal values;
   identities, labels, native and common-four predicted classes and confusion
   matrices stay exact, and the balanced-accuracy reconciliation stays within
   1e-9 (only the tolerance comes from pmm-core-replay-v1, not its other row
   checks). It is an option of
   `campaign_runtime.replay_campaign_run` (default unchanged: 1e-6, same messages,
   same receipt), recorded in the campaign manifest, every replay receipt (with
   the largest observed difference) and every unit status. Evidence: A3 on
   2026-10-05 found 6 of 9 CPU replays of GPU-trained v2 checkpoints above 1e-6
   (worst 4.53e-6) with every discrete field equal.
2. Concurrency: fixed limits of 1.0 common-four BA point and 3 points per
   common-four class recall against the first serial run; never widened. If the
   two serial runs disagree beyond them, the report requires diagnosis and
   decides nothing; the only pre-declared closure (`step_b/diagnosis.json`,
   written after a dated user decision) records serial FP32 without concurrency
   or AMP. Probability and history differences are diagnostics.
3. CPU pressure: median, peak, longest and total overload (load1 per CPU above
   1.0) are reported; pressure if the median exceeds 1.0, one overload lasts
   longer than 300 s, or the steady training phase (all members training, first
   60 s skipped for the load1 lag, at least 60 s judged) is overloaded more than
   half the time; memory (at least 10% available) and GPU memory (at most 90%;
   missing GPU samples fail) limits unchanged. Throughput is end to end: pre-admission checks (recorded per
   unit), preparation, training, replay and measured host persistence; batch
   members are charged their own (contended) preparation.
4. Retries: completed fits are never rerun. Only a step-B timing batch (2 or 3
   lanes) may be retried, once, as a whole under `-retry` tags, and only when
   its first attempt is invalid under outcome-blind rules (a member failed or did
   not run, members not in distinct lanes, admissions more than 10% of the
   shortest elapsed time apart, or host samples not covering the batch window);
   the retry then decides. A retry after a valid first attempt makes the report
   refuse. All attempts are kept.
5. Costs: storage counts toward the $40 gross ceiling, at estimated actual
   charges (disk $15.00/month = $0.49/day while the restored disk exists;
   snapshot about $1.74/month = $0.06/day, from the retirement receipt), apart
   from the controller's conservative daily reservation (snapshot $7.50/month,
   $0.25/day). Running hours are costed at $0.859/h (the gross $0.879/h minus its
   disk share), so the disk is counted once. The snapshot is kept.
6. Disk: the two archives were moved to `/media/mechti/Data1/archive_from_home/`
   (SHA-256 equal on both sides, copies re-read with direct I/O, originals then
   removed; `SHA256SUMS` beside them). `/` has 16 GB free. Nothing else was
   deleted.

Also built (all tested): the fixed probe manifest `pmm_v3_probes.py` (short
probes 10 epochs, default, so a steady training phase can be judged), enforced
by the runner before admission (an archived batch member is never relaunched;
a failed serial probe or full run keeps its one rerun before the decision); the runner
hash now covers `pmm_v3_probes.py` and `pmm_v3_speed_report.py`, so the gates
are frozen at preparation, before any step-B data; `set-execution` records a
setting only if it equals a recomputed, decision-ready speed report; the
workstation launcher `pmm_v3_step_b.py` (never starts, stops or extends a VM;
detached launches; waits; pulls) and the per-lane host-pull tool
`pmm_v3_host_pull.py` (never re-stamps an acknowledgment; keeps a superseded
local copy when a rerun reuses a run name; no torch); the launcher refuses a
step while a lane awaits its host pull, holds a running or interrupted unit, or
an earlier launch has no exit code, and a failed pull fails the command;
evidence backup before every `vm-stop`; `pmm_v3_bundle.py build` refuses
without an A3 report that accepted exactly the bundled source tree; runner exit
codes 3 (blocked) and 4 (persistence).

An independent five-agent review (read-only) found one blocker (a batch could
start while a lane still awaited its host pull, wasting the single retry) and
six major points (A3 binding after the src change, failed probes losing their
rerun, no closure for a serial disagreement, archived batch members allowing
extra attempts, reruns overwriting the workstation copy, a weak sustained-CPU
check in short probes); all were fixed with tests, as were the cheap minor
points. Deferred as minor: a guard against deliberately launching one batch
member alone outside the launcher (the launcher always starts members
together).

Commits: `64f29de` (src replay option), `76395bf` (runner, gates, manifest,
tools, tests, plan and playbook) and this entry's commit. Checks:

- A3 repeated and **accepted** (`audits/a3_regression_20261005T194214Z`,
  `a3_report.json` SHA-256 `50ecf905…`): tested source tree
  `212db0e3865968701475e7172b2e4cad772e492b35855da293238437fd97c15c` (src at
  `64f29de`, unchanged in `76395bf`), unchanged from start to end, no uncommitted
  src change; the frozen v2 tree matches its pin. Replay: all nine pinned v2
  fold-0 checkpoints pass pmm-core-replay-v1 (worst probability difference
  4.53e-6; 6 of 9 above the strict 1e-6, every discrete field equal) and every
  epoch-50 checkpoint matches its history; training: all four configurations
  bit-identical to the frozen v2 checkout (difference 0.0). The default strict
  replay path is therefore unchanged on real data. **Correction to v3-007 and
  v3-008:** their A3 acceptance covers source tree `5a120b2c…` only and is
  superseded by this one; `pmm_v3_bundle.py build` now refuses any bundle whose
  source tree differs from an accepted A3 report.
- A5 repeated and **passed** (`audits/a5_prefit_20261005T204150Z`): 82 of 82
  planned units build valid commands on a scratch root whose manifest records the
  committed runner files (`pmm_v3_campaign.py` `ce6ddec8…`,
  `run_pmm_v3_campaign.py` `21b798ef…`, `pmm_v3_probes.py` `6c6a0d73…`,
  `pmm_v3_speed_report.py` `6e4c2997…`) and the replay policy; augmented recipes
  still need about 8.8 h (late fusion) to 12.4 h (Only-GVP) of graph rebuilding
  per fit on this PC, so they stay cost-gated.
- Tests: all 186 v3 tests pass. Full suite on `76395bf`: 1,167 passed, 19
  skipped, 2 failed, 32 errors, all known and unrelated to this pass. The 2
  failures fail identically on `f3510ba`:
  `test_explicit_membership.py::test_outer_loader_not_deserialized` (patches a
  `training.data.load_structure_pockets` that does not exist) and
  `test_generalized_metal_5fold_cv.py::test_dry_run_flag` (expects the local
  dataset folder `train_and_test_sets_structures_exact_pinmymetal`). The 32
  errors are every setup of `test_pmm_core_assessment.py` ("Scientific module
  already loaded from another tree: data_structures"), an order dependence: the
  same 40 tests pass when that file runs alone.

Step-B forecast (gross): one session of about 2.4–3.2 VM hours with 10-epoch
probes, $2.06–2.75 of compute and IP (about 5.5 h and $4.72 if the AMP
follow-ups, a retry or connection trouble need a second session), plus the
restored disk at $0.49/day and the snapshot at $0.06/day. Whole campaign B–F,
from the measured v2 fit times (late fusion 1,747 s, Only-GVP 1,619 s, Only-ESMC
1,125 s per warm fit including replay, persistence and pre-admission checks; five
further cold cache builds; about 25 minutes of overhead per session):

| Scenario | VM hours | Compute + IP | Disk | Snapshot (actual) | Total |
|---|---|---|---|---|---|
| 1.5× concurrency, D ends after Round A | 26 | $22.6 | $4.9 (10 days) | $0.6 | $28 |
| 1.5× concurrency, full D | 35 | $30.3 | $6.9 (14 days) | $0.8 | $38 |
| no concurrency, D ends after Round A | 37 | $32.0 | $6.9 (14 days) | $0.8 | $40 |
| no concurrency, full D | 51 | $43.6 | $10.4 (21 days) | $1.2 | $55 |

Augmented candidates (9a, 9b) stay cost-gated and are excluded. Calendar days
are an assumption; every extra day with the disk kept adds $0.49. Without a
concurrency gain the full plan exceeds the ceiling, so the re-forecast after
step B decides whether the user is asked to stop early or raise the ceiling.

Unchanged: the frozen A4 specification (SHA-256 `29694070…`), the fold set and
its caveat (near-copy, not homology, separation; about 11% of ions have an
80–90% cross-fold partner), and the open final-test label decision before
step E. No held-out data were read.

STATUS text replaced by this update, preserved verbatim:

```text
- Status: active (2026-10-05 v3 step A complete, CPU only; step B awaits the user's GPU OK)
- Last execution evidence: 2026-10-05 (v3 step A CPU audits). Documentation reconciliation: 2026-10-05.
- Current campaign: pmm_ion_metal_v3, step A complete (objectives four/five/six for Only-ESMC, Only-GVP and late fusion) — [README](docs/campaigns/pmm_ion_metal_v3/README.md)
- Stage: v3 step A done ([log v3-008](docs/campaigns/pmm_ion_metal_v3/log.md#v3-008)): strict folds, code, CPU audits, frozen assessment rules; no GPU runs, Stage 6 confirmation, Stage 6B refit or Stage 7.
- Authorized now: CPU work only ([plan](docs/campaigns/pmm_ion_metal_v3/plan.md)); every GPU start needs the user's explicit OK within the [$40 gross ceiling](docs/campaigns/pmm_ion_metal_v3/log.md#v3-003); no refit or held-out evaluation.
Next: v3 step B (GPU speed check) per its [plan](docs/campaigns/pmm_ion_metal_v3/plan.md), after the user's explicit GPU OK and freeing 10–15 GB on `/`.
```

## v3-008

2026-10-05 — **plan step A complete** (CPU only; nothing ran on a GPU). After
the v3-007 checkpoint the user said to continue, and the A4 specification was
frozen:

- [`assessment_spec.md`](assessment_spec.md) SHA-256
  `899005aaba036739519876860121fdf4e60241d55c393d435847aae56ece3667`;
  [`assessment_spec.json`](assessment_spec.json) SHA-256
  `2969407042dbb5858f15e1a114a2d602cc2fd764e788022ddaee1907ff17306c`, pinned as
  `FROZEN_SPEC_SHA256` in `pmm_v3_assessment.py`. Its fold, cohort and baseline
  identities match a real-data preparation of this campaign (the A5 scratch
  root). Every run and assessment now requires it.

Step A record: A1 strict folds (v3-005); A2 code (v3-006, v3-007), with the
clean-subset and test-preparation builder **deferred** by the user to just
before step F (outside `src/`, tested on synthetic or training-side data before
any test access, keeping the agreed near-copy and previously-opened-PDB rules);
A3 accepted (`audits/a3_acceptance_20261005T122647Z`, tested source tree
`5a120b2c…`); A4 frozen (above); A5 passed
(`audits/a5_prefit_20261005T122704Z`); A6 playbook recipe written. All 77 v3
tests pass; the full suite shows only the known 2 failures and 32
order-dependent errors.

Before step B: the user's explicit GPU OK (within the $40 gross ceiling),
about 10–15 GB freed on `/` (about 6 GB free), and the real campaign root
prepared once from the final code. Still due before step E: the final-test
reporting decision.

## v3-007

2026-10-05 — step A checkpoint; **step A not yet declared complete** (the A4
specification is still to be frozen). The user's four code-review findings and
a follow-up were verified against the code; all were real and are resolved
with tests (`0f92bf1`, `afa4767`, `8eef4ef`):

1. Step D now pools passes across completed rounds, blocks every decision
   while a required run is missing, stops after a complete round without a
   pass, and adopts a combination only if it passes and beats the best single
   candidate.
2. Runs refuse changed runner files or recipe definitions (manifest hashes)
   and require the frozen A4 specification whose cohort, fold and baseline
   identities match the campaign; the assessor checks the same before
   assessing, and rejects completed runs recorded with other runner files.
3. A run name is claimed atomically for all lanes; archiving a failed or
   interrupted attempt requires, on the same host, a stopped worker, a stopped
   training child and a free lane lock; a rerun must match the archived
   identity.
4. A3 gives an explicit verdict bound to the tested source-tree hash.

A3 **accepted** (`audits/a3_acceptance_20261005T122647Z`): the replay
(`a3_regression_20261005T102411Z`) passed all nine pinned v2 fold-0
checkpoints (best checkpoints within `pmm-core-replay-v1`, worst probability
difference 4.5e-6, all discrete fields equal; every epoch-50 checkpoint matches
its history), and the training check (`a3_regression_20261005T101042Z`) passed
all four configurations bit-identically. Tested source-tree hash
`5a120b2c514f3b03ceefb8e9f5b30836d8bbb47968cc62ba2557f885f71c584d`, unchanged
since both audits started (bound post hoc by `--accept`); frozen v2 tree
`adc95c42…` matches its pin. No `src/` change followed, so no A3 check needs
repeating. A5 was repeated on the final runner and passed
(`audits/a5_prefit_20261005T122704Z`). Full test suite: only the 2 known
failures and 32 known order-dependent errors.

Known limit: claims are local coordination files, not persisted; a unit
interrupted by the loss of its host cannot be archived without a user
decision, because its worker's stop cannot be verified.

## v3-006

2026-10-05 — plan step A2–A6 work, **step A not yet complete** (open: the A3
replay of all nine pinned checkpoints, the user's four code-review findings,
then freezing the A4 specification). Branch `v3-step-a` (not merged or pushed):
`2acef49`, `0a0c534`, `46c1515`, `335b9c0`, `2b4b220`, `22f4e29`, `ea086fb` and
this entry's commit.

- A2 code: v3 runner with lanes, independent replay, persistence and the step C
  regression run; one-time rerun of a failed unit (`archive-failed`); step-B
  probes and a write-once execution setting (AMP, lanes, reuse of the matching
  full run as the step C cell); v3 assessor with the approved statistics.
- Fix found in step A: each residue's ESMC row was a view of the whole float32
  chain matrix, so every loaded structure kept all its chain embeddings in RAM
  (a v2 late-fusion fit peaked at 8.0 GB; one fold-0 replay above 3 GB). Rows
  are now owned copies (one fold-0 replay peaks near 1.9 GB); values are
  unchanged.
- A3 training check **passed**: on a synthetic campaign, 3-epoch CPU fits of
  four configurations with the new code and with the frozen
  `_code/pmm_core_scope_v2` checkout are bit-identical (every history value
  and every weight; difference 0.0), after the fix
  (`audits/a3_regression_20261005T101042Z`). The real-data replay of the nine
  pinned v2 fold-0 checkpoints (best checkpoints against the saved predictions
  under `pmm-core-replay-v1`; epoch-50 checkpoints against the epoch-50
  history) is running on CPU; results go in the step A completion entry.
- A4: the user approved the four new default rules on 2026-10-05 with
  clarifications (eligibility is not an improvement claim; improved-recipe
  five-fold results are development-validation; the five-class tie preference
  is a selection convention). The specification is frozen only at step A
  completion. The approval covers the assessment rules, not a GPU start.
- A5 pre-fit gate **passed** on a scratch preparation
  (`audits/a5_prefit_20261005T103224Z`): all 82 planned units build valid
  commands; every native class is present in every fold (validation Co 55–58,
  Cu 67–73); class weights equalize the four common classes; the trainer applies
  them to five- and six-class targets (tests); normalization is fitted on the
  training fold only (test). Augmented recipes would rebuild graphs for about
  7.6–9.5 h per fit on this PC, so they fall under the cost gate.
- Deferred by the user's decision: the label-blind clean-subset and
  test-preparation builder is written just before step F, outside `src/`, and
  tested on synthetic or training-side data before any test access, keeping the
  agreed near-copy and previously-opened-PDB exclusion rules. The final-test
  reporting decision stays due before step E.
- A6: the executable v3 recipe is in the
  [metal playbook](../../METAL_TRAINING_PIPELINE_PLAYBOOK.md#pmm-ion-metal-v3-campaign-pmm_ion_metal_v3).
  Free space on `/` is about 6 GB; the user decides what to remove before step B.

## v3-005

2026-10-04 — plan step A1 complete: the v3 fold set `v3-seqid90-s42-b2` is
frozen under `/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v3/folds/`
(`fold_membership.csv` SHA-256 `e6b21364159da479dc3fb325a01ede36c62489184a77d858f71fefe6c401da7a`;
receipt `fold_receipt.json` with input, output and code identities; builder
`v3-near-copy-folds-2` at commit `3102c2a`, seed 42, 200 starts). Only training
inputs were read, under the read guard; every one of the 7,664 context-chain
sequences matched its frozen ESMC hash.

Result: 7,398 ions, 3,992 PDB entries, 4,972 unique chains, 11,609 near-copy
pairs, 2,208 groups. All acceptance checks pass (48 of 200 starts; start 198):
fold ions 1,471–1,500 (target 1,479.6); Cu 67–73 ions from 24–25 groups per
fold; at least 24 groups of every element in every fold. No near-copy under
the builder's rule crosses folds (the v2 folds had 50.9% of ions with a
cross-fold near-copy).

Refinement of the A1 near-copy rule: identity divides by the shorter chain's
length plus its number of internal insertion events, so the 5-mer prefilter is
provably lossless. A first build (`v3-seqid90-s42`, builder version 1) counted
every inserted residue instead; independent verification showed it split 25
plan-rule near-copy pairs (54 ions) across folds, so it is marked superseded and
must not be used. Under the plan's original wording, 9 borderline pairs
(identity 90.0–90.6% of the shorter chain) still cross the frozen folds.

Verification (read-only, independent methods): exhaustive screening of all
11.9 million chain pairs with a different lossless filter found no qualifying
pair the builder missed; 50,000+ skipped pairs aligned exactly stayed below 90%;
balance recomputed exactly; byte-identical rebuilds under different Python hash
seeds. Known limits: chains below 90% identity can cross folds (about 11% of
ions have an 80–90% cross-fold partner), so this is near-copy, not homology,
separation; and a few large families dominate one element in one fold (largest
single-group shares: Ni 0.51 and Cu 0.43 in fold 4, Mn 0.40 in fold 1).

## v3-004

2026-10-04 — an external review of the v3 plan (Codex, read-only) raised twelve
points; each was checked against the repository by five read-only verifiers
(no test files, no compute). All were accepted. Two were accepted only in part:
policy does not require two confirmation seeds, and the step-D parameter values
are a user choice rather than a wording fix. Minor inaccuracies (misplaced line
references, a non-binding source cited as binding) did not change any verdict.
The verifiers also found that the trainer cannot yet consume group-based v3
folds, that a locally built graph cache does not match the VM's torch build,
that augmented fits rebuild every training graph each epoch, and that no RING
files exist for the v3 cohort. The [plan](plan.md) was revised accordingly and
then checked once more for consistency; that check moved the A4 freeze before
any GPU run, made every budget stop a user choice, and moved Round C out of the
plan's code and budget pending a separate decision after Round B.

User decisions:

- Step E: the neutral four-versus-five/six test uses the step C baseline recipe,
  frozen before step D, on all five folds; improvements are confirmed separately
  in `four_class` on folds 1–4 against that baseline.
- Decision statistics: bootstrap and t intervals must agree; a target tie keeps
  `four_class`; a zero native recall blocks a five/six "better" call.
- Step D gate: both seeds must improve, in addition to the 1.5-point mean and the
  3-point class guard.
- The final-test label is decided before step E, after the user checks the
  2026-09-24 Zenodo run.

Defaults set by Claude under the user's instruction to accept or reject the
review's points (changeable by a dated entry before the affected step): seed 42
only in step E; step-D values (modality dropout 0.2, outer-residue dropout 0.1,
coordinate noise 0.1 Å, label smoothing 0.1, `counts_angles` against `none`);
the step-B adoption thresholds; the 3.0-point regression band; the fold
acceptance limits; the cost gate for augmentation; and the Stage 6 selection
rule (Only-ESMC baseline `four_class` control, 0.2-point tie band,
tie-breakers), which is shown to the user with the A4 specification before it
is frozen.

STATUS text replaced by this update, preserved verbatim:

```text
- Stage: v3 design; no folds, runs, Stage 6 confirmation, Stage 6B refit or Stage 7.
Next: user decisions on the v3 design listed in its README, starting with the
final metal test route. Stage 6 grouped-fold selection (or an explicitly labeled
fallback), a completed Stage 6B full non-test refit, frozen report/checkpoint
rules and a scientifically resolved final-test route must precede one-shot
Stage 7. No test-based tuning, ranking, promotion, rejection or checkpoint
choice; see [Plan](Plan.md#canonical-staged-metal-training-pipeline).
```

## v3-003

2026-10-04 — the user confirmed active Google Cloud free-trial credits on the
billing account (upgraded free trial; about ₪848 remaining, expiring
2026-12-24; Google reports they cover all eligible usage). Under the user's
rule from [v3-002](#v3-002), the v3 GPU budget ceiling is **$40 gross** for
steps B–F. Costs stay tracked at gross list price; credits never widen the
controller's session and daily caps or any limit, and usage is not described
as free. Every GPU start still needs the user's explicit OK.

STATUS line replaced by this update, preserved verbatim:

```text
- Authorized now: v3 CPU preparation only ([plan](docs/campaigns/pmm_ion_metal_v3/plan.md)); every GPU start needs the user's explicit OK within the [recorded ceiling](docs/campaigns/pmm_ion_metal_v3/log.md#v3-002); no refit or held-out evaluation.
```

## v3-002

2026-10-04 — user decisions after discussion:

- The [v3 plan](plan.md) (steps A–F) is approved.
- The 2026-09-28 rule "no file added or changed in `src/` or `scripts/`" is
  lifted for v3 work: changes use off-by-default options with tests. The frozen
  `_code/pmm_core_scope_v2` checkout stays untouched.
- Checkpoint rule: cosine learning-rate schedule to zero and the terminal
  checkpoint for every v3 arm, fold and refit (Plan updated the same day).
- PMM: compare with published PinMyMetal numbers only; no PMM retraining.
- Folds: chains at ≥90% identity grouped together, random group order, metals
  balanced. Improvement screening on one new fold with two seeds is accepted.
- GPU budget: $15 gross until the user confirms that free credits cover Compute
  Engine, then $40. No GPU start is authorized without the user's explicit OK.

STATUS lines replaced by this update, preserved verbatim:

```text
- Status: planned (2026-10-03 PMM core v2 closed at fold 0; v3 planning only)
- Last execution evidence: 2026-09-28. Documentation reconciliation: 2026-10-03 (closure and verified findings).
- Authorized now: nothing (no experiment, GPU work, final refit or held-out evaluation).
```

## v3-001

2026-10-03 — campaign opened in planning status. The user chose separate
`four_class`, `five_class` and `six_class` arms for Only-ESMC, Only-GVP and
graph-level late fusion. Predecessor:
[closed PMM campaign](../../archive/campaigns/pmm_ion_metal/README.md).
