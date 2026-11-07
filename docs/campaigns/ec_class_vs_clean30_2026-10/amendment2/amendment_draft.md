# Amendment 2: pre-G2 science amendment

**Record status: draft for the maintainer's review (2026-11-04); not in force.** This is the reviewed document of
amendment 2. The canonical record `amendment.json` does not exist yet. The amendment-2 builder writes it once, after
the repeated dry runs, binds the decisions file and evidence by SHA-256, reads the rule text below from this file
and writes the rendering `amendment.md` from the record, with that rule text unchanged; this file is never
overwritten. The record takes effect only when its SHA-256 is entered in the campaign log and it is published on
`origin/main` (rule A2-GOV-1). Until then Freeze 1, Freeze 1b and the S1 amendment stay in force and G2 stays
paused.

Campaign [README](../README.md) · [design section](../design.md#amendment-2-2026-11-04-before-any-g2-or-fold-0-data)
· [draft record and scope list](amendment_draft.json) · [chain-verifier contract](interface_contract.md) ·
[power simulation](power_simulation.md)

## Timing and exposure

- **Timing.** Accepted by the maintainer on 2026-11-04. Recorded before any CLEAN embedding farm, F6
  sequence-mean cache, G2 measurement, model prediction, score, selection record, or CLEAN inner fit or export
  existed; the certified views' label-free ESMC chain-mean caches predate the amendment (G2 completion 0/12,
  [evc-033](../log.md#evc-033)).
- **Design-time knowledge.** The class composition and the approximate sizes of the test populations were known at
  design time (design-time label exposure, already disclosed in the design).
- **Exposure.** About two thirds of the fold-0 metallo training proteins are curated test proteins of folds 1-4, so
  fold-0 development exposes folds 1-4. No rule below depends on any fold-0 result.
- **Labels.** A2-D1 to A2-D8 name the maintainer's decisions of 2026-11-04. Rule labels are A2-GOV, A2-SEL, A2-TRN,
  A2-F6, A2-MET, A2-STAT, A2-CLN, A2-FUT, A2-DIS, A2-SUP and A2-BUD. Bare D-numbers are Freeze-1 decisions.
- **Unchanged identities.** Freeze 1 (`28077f4c…3a3c0c`), Freeze 1b (`fc6f0328…cbdb`) and the S1 amendment
  (`d07dbb6a…a242ea`) are unchanged and stay replayable.

## Decisions of 2026-11-04

A2-D1 to A2-D5 are science decisions implemented by this amendment. A2-D6 to A2-D8 are execution decisions; the
execution v5 record implements and declares them, and this record lists them for completeness.

| ID | Decision | Reason | Rules |
|---|---|---|---|
| A2-D1 | **S1 rule.** Each outer fold selects, among F2, F3, F4, F5 and the new F6, the family with the highest pooled inner seed-mean score on the primary metric (2 screening seeds × 4 inner folds). Each family's learning rate is chosen by the same score. A zero-recall harm guard against the anchor F2 applies. Within 0.5 points of the maximum a pre-declared simplicity order decides. There is no significance test and no promotion test. Controls are reported and never selectable. The historical rules stay replayable. | Under the S1 amendment rule, the recorded characterization selected F4 in only 3.0–9.5% of runs at a true +2 BA points and 7.5–16.5% at +3 ([evc-012](../log.md#evc-012)), so the pocket-only F2 would be deployed in most folds even when a whole-protein family is better. A best-inner rule uses the inner evidence directly. It selects a model and supports no family-superiority statement. | A2-SEL-1 to A2-SEL-9 |
| A2-D2 | **Primary metric.** Protein-level accuracy over the proteins of EC classes 1-6. Class-7 proteins are excluded from the population, and predictions into class 7 count wrong. Balanced accuracy over EC classes 1-6 becomes the first secondary metric. The co-primary populations (B1-all, B1-NH30), the paired homology-cluster bootstrap, the intersection-union test against both the CLEAN R1 seed mean and the R1 ensemble, and Holm across the two populations are unchanged. | The pre-data [power simulation](power_simulation.md), run with the frozen statistics engine on synthetic populations: under per-class-uniform gains accuracy needs about 0.6–0.65 times the BA margin (80%-power MDE against the seed mean, low-accuracy configuration: B1-all 6.4 vs 10.3 points, B1-NH30 8.8 vs about 13). The frozen BA claim test is anti-conservative (any-claim rate 0.068–0.076 at the true null against a nominal 0.05; bootstrap SE over sampling SD 0.88–0.93 on B1-NH30), while accuracy is about nominal (0.038–0.048). The endpoint also fixes the estimand: gains on EC classes 5-6 alone are detectable only by BA (about +28 points there), and a trade of rare-class for common-class recall only by accuracy. | A2-MET-1 to A2-MET-5, A2-STAT |
| A2-D3 | **Candidate family F6.** F5 whose whole-protein branch is first trained on the fold's full official training file (the same file CLEAN R1 uses, EC class = first EC digit), with proteins of more than one EC class excluded as in the cohort rule. Inner-validation proteins and their identity neighbours are held out of that pretraining for inner fits. Embeddings are ESMC-300m means with the recipe of the existing whole-protein cache, computed per official sequence (deviation 4), built on the GPU pod (A2-D7) and reused for the CLEAN ESMC-input control only if the recipe is byte-identical. This reverses Freeze-1 decision U3 for this branch only. | The largest capability gap is training data: 545–603 metallo training proteins per fold ([evc-007](../log.md#evc-007)) against about 8.1K enzymes for CLEAN R1. A whole-protein branch trained on the same file tests the most direct remedy without changing the pocket model, the outer folds or the populations. Non-metalloenzymes are used only as whole-protein pretraining data. | A2-F6-1 to A2-F6-8 |
| A2-D4 | **Search and training.** S2 keeps only the no-fit L9 logit adjustment (τ chosen as before; see C2). S3 is dropped. The raw-distance RBF (the distance part of L2) becomes the default for the GVP families. Every fit uses 30 epochs (cosine, terminal checkpoint, as before). F1 runs in fold 0 only. | The fitted S2 levers would spend 48–72 of a fold's 149–181 fits on levers whose recorded promotion power at a true +2-point gain is low (Freeze-1 D1). The GVP radial basis (centres 0–12 Å) currently receives z-scored distances clamped at ±5, and the 999-Å no-donor sentinel collapses the donor distance; the metal recipe behind these families used raw distances. F1 is a never-selectable control; fold 0 keeps the structure-only reference. | A2-TRN-1 to A2-TRN-4, A2-BUD |
| A2-D5 | **Futility stop after fold 0.** From the fold-0 S1 inner predictions of the selected DeepMzyme recipe and CLEAN R1-inner on closure-eligible inner proteins, a pre-declared projected probability that the final claim succeeds is computed. Folds 1-4 run only if it is at least 0.20; otherwise the campaign stops and closes out as "not shown to outperform". The Freeze-1 −3-point gap record is kept as a descriptive record. | The claim's power is limited (power simulation). Stopping for futility cannot create a false "Outperforms" result, and it saves the 404 fits of folds 1-4 when success is unlikely. | A2-FUT-1 to A2-FUT-9 |
| A2-D6 | **Execution route.** Runpod only; G2 and fold 0 run in one pod session. | GCP L4 capacity was unavailable in every authorized zone attempt ([evc-016](../log.md#evc-016) to [evc-023](../log.md#evc-023)). One session keeps fold-0 state on one host. | execution v5 |
| A2-D7 | **GPU work on the pod.** CLEAN fits and the ESM-1b and ESMC embeddings run on the pod GPU with TF32 off. | A CPU CLEAN fit takes about 2.9 h (Freeze-1 F8), and the ESM-1b farm does not fit the local memory cap. TF32 off keeps float32 numerics (rule A2-GOV-3). | A2-GOV-3, A2-F6-4, execution v5 |
| A2-D8 | **Ceiling and time-box.** A pre-approved fold-0 spending ceiling, and a time-box of at most two further failed capacity windows, then a route change or a pause. The amount is recorded in the execution v5 record before G2. | Bounds cost and calendar time before the first productive session. | execution v5, A2-GOV-8 |

Audit items implemented with these decisions: P9 stratum gate (A2-STAT), P8 CLEAN outcome sanity anchor (A2-CLN-1
to A2-CLN-4), P12 post-fold-0 change rule (A2-GOV-7), P15 claim-scope wording (A2-MET-5), P17 tie counts
(A2-CLN-5) and P18 SupCon-Hard disclosure (A2-CLN-6).

## Choices awaiting the maintainer's confirmation

These are interpretations of the decisions above. The record is built only after the maintainer confirms or changes
each one in the decisions file; until then each stays a recommended default.

| Choice | Recommended default | Note |
|---|---|---|
| C1 Harm-guard threshold | k_F2(c) ≥ 1 | The literal reading of "non-zero" (A2-SEL-4), and the conservative direction. |
| C2 L9 τ metric | The primary metric; the BA-scored τ choice is descriptive | Grid, tie order, α 0.01 and the 1.0-point gain are kept. "As before" could also be read as BA. |
| C3 Within-stratum resampling | Rao-Wu, k−1 of k clusters (A2-STAT-2) | The pre-declared small-stratum handling of P9; the bootstrap type is unchanged. |
| C4 Futility override | None toward continuing; a stop takes precedence over any rerun (A2-GOV-7(d)) | A2-D5 says "otherwise stop and close out". |
| C5 Analysis completion | The narrowed class (a2) of A2-GOV-7 | Only four post-fit analysis files, refusals and descriptive outputs. |
| C6 CLEAN fit recorded as "proceed, disclosed" | Every comparison it enters reads "No superiority inference" (A2-MET-2) | Alternative: a mandatory wording qualifier only. The default can only remove claims. |
| C6b Held-out CLEAN arm recorded as "proceed, disclosed" | The held-out record has one status per arm and each failed arm its own disposition. An R1-inner disposition makes both co-primaries and every secondary hypothesis against R1 unavailable (R1-inner stands for the R1 recipe in fold 0), and FUT then returns `stop_futility` with 0 successes; an R2-inner disposition makes only hypotheses 3 and 4 unavailable (A2-MET-2, A2-CLN-3) | Alternative: no consequence for held-out arms (A2-MET-2 limited to final-arm fits). The default can only remove claims and avoids spending folds 1-4 on a claim that cannot be made. |
| C7 Futility DeepMzyme arm | The S1 configuration with the S2 τ applied (A2-FUT-3) | The selected recipe includes τ. |
| C8 F6 inner exclusion | One hop (the R1-inner list) ∪ exact sequence ∪ case alias ∪ shared selected donor (A2-F6-3) | Interpretation of "their identity components"; equals the CLEAN R1-inner and B1-NH30 one-hop rules. |

## Rule text

### A2-GOV: governance

**A2-GOV-1 Chain.**
1. **Links.** Amendment 2 is link 2 of the Freeze-1 amendment chain. Link 1 is the S1 amendment
   `s1-fixed-anchor-holm-v1` (SHA-256 `d07dbb6a68de7f441a0030661f110f2de9dabd4786a702a1e23d92a3a7a242ea`, commit
   `3154a7f3b0ad44e8dffbc66072990cfb7d9c9030`).
2. **Authenticated chain.** That SHA-256 followed by the SHA-256 of `amendment2/amendment.json`.
3. **Form and publication.** Every link is canonical JSON, written once, rendered by its own code and committed with
   its SHA-256 in the campaign log. It is published on `origin/main` before any execution depends on it.
4. **Authentication.** Each link is authenticated by the verifier code published with it, run in a checkout of that
   link's published source commit. Only the tip is compared with the live tree; there, execution files and post-fit
   analysis files may differ under A2-GOV-7.
5. **Shorter prefixes.** Real selection, futility, scoring and execution use the full published chain. A record
   bound to a shorter prefix is accepted only for replay under that prefix's rule and is never combined with records
   of the full chain.

**A2-GOV-2 Files and identities.** Every tracked `src/` and `scripts/` file belongs to exactly one of three closed
sets.

- **Execution files.** A closed, sorted list, recorded in amendment 2 by the SHA-256 of the sorted name list;
  execution manifest v5 pins their contents. It holds only orchestration, transport, provider, persistence and
  dispatch code. It never holds campaign, command, view, provenance, identity, cohort, transfer-list, selection,
  selector, freeze, amendment-chain, amendment-rule, runtime-precision, futility, statistics, scoring, CLEAN-sanity,
  trainer, model, embedding or comparator code. Two functions inside execution files implement frozen gates:
  `phase3.lane_gate` (G3) and the CPU replay comparison named in the contract fixture. Their source, including the
  module-level helpers they reference, is pinned in amendment 2, and an execution change never alters it.
- **Post-fit analysis files.** Exactly `src/ec_clean30/scoring.py`, `src/ec_clean30/baseline.py`,
  `scripts/run_ec_clean30_baseline.py` and `scripts/score_ec_clean30_comparison.py`. No other file joins this set
  without a scientific amendment.
- **Scientific files.** Every other tracked `src/` and `scripts/` file.

**Channels into analysis code.** Scientific code reaches post-fit analysis code only through `scoring.parse_label_table`
(the CLEAN comparator runner and `cohort.py`), `baseline.read_official_ec_numbers` (the comparator runner),
`scoring.parse_transfer_list` (`transfers.py`) and the scoring and baseline constants read by `freeze.live_constants`.
The source of `scoring._load_comparator`, `scoring.parse_label_table`, `scoring.parse_transfer_list` and
`baseline.read_official_ec_numbers`, with the module-level names each references transitively within its module, is
pinned, and so is the output of `freeze.live_constants()`. The selector's gap path loads the comparator through
`clean_sanity.load_comparator` instead of the scorer.

**Scientific identity.** The SHA-256 of canonical JSON holding: the campaign, Freeze-1 and Freeze-1b hashes; the
**fit chain** (the chain up to its last link that changed a scientific file); the composite pins of the scientific
files; the pinned execution functions; the v1 and v2 plan hashes; the Freeze-1 view records; the amendment-2
dry-run receipts; the precision policy (A2-GOV-3); and the specifications of deferred inputs. It holds no
execution-file pin, execution-manifest hash, commit, provider, hardware or session identifier.

**Analysis pins.** The composite pins of the post-fit analysis files. Freeze 2 and the scoring claim bind them,
together with the full chain.

**What the identities bind.** G2 measurements, Freeze 1c v2, the numerical G3 authority, every unit result, every
step record and the futility record bind the scientific identity and the fit chain. A later execution manifest is
accepted for these records only if it is published, verifies and has the same scientific identity. No scientific
module loads an execution file. No fit, selection, futility or CLEAN-sanity module loads a post-fit analysis file
except through the channels above.

**A2-GOV-3 Precision and replay.**
- **TF32.** TF32 is off for every GPU DeepMzyme fit, embedding, CLEAN fit, export and farm child. Each records the
  policy through `runtime_precision.configure_and_record`. Scientific verifiers (`commands.verify_unit`, the CLEAN
  sanity verifier and `pretrain-corpus attach`) refuse any receipt whose recorded policy differs.
- **ESMC whole-protein embeddings.** Float32 parameters, set after loading. The forward runs directly rather than
  through the SDK's logits wrapper, with no autocast and no flash attention. The emitted dtypes are verified as
  float32.
- **Replay.** A CUDA-trained model's logits are replayed on CPU at an absolute tolerance of 1e-5 (the Freeze-1
  cross-device value, G1). Near ties are checked at 2e-5, and argmax agreement is required.
- **Lane gate.** G3 keeps its frozen balanced-accuracy limits. The primary-metric difference is recorded
  descriptively.
- **Changes.** A change to this policy is a scientific change.

**A2-GOV-4 Dry runs.**
- **Why they repeat.** Amendment 2 changes protected command, view and trainer code, so the five dry runs are
  repeated once, on a clean tree at the source commit named in the record.
- **Receipts.** The repeat writes separate receipts, bound by SHA-256. It does not rewrite the in-view receipts that
  Freeze 1 binds.
- **Scope of each run.** Each repeated run statically builds every v2 plan command and every v1 command. F6 commands
  are built against the declared pretraining-input stub. Each run probes as before and has no static failure.
- **Rule `r_cmd_v2` from that commit on.** The Freeze-1 R-CMD allowlist files and the execution files may change
  without a new dry run. Added modules must not be loaded by the protected importers. The pinned runner functions
  stay unchanged. The v1 and v2 plans stay equal to their records.
- **What still needs a link.** A change to a scientific file still needs an amendment link, even when `r_cmd_v2`
  needs no new dry run. Any other change needs a new amendment and new dry runs.

**A2-GOV-5 Replay.**
- The five v1 fold plans keep their Freeze-1 SHA-256 values.
- Every v1 unit keeps its resolved-configuration hash: new trainer options leave the configuration payload while
  they are at their defaults.
- `freeze.live_constants()` and `campaign.campaign_constants()` keep their values at `c163995`.
- The historical escalation (empty chain) and the fixed-anchor rule (S1-amendment chain, authenticated at its own
  commit) stay executable on their original records.
- No pinned scientific file is deleted.

**A2-GOV-6 Precedence.** When amendment 2 was built, its list of fold-0 records was empty. Every record accepted
under the amendment-2 chain carries that chain and was created after the commit that published amendment 2.

**A2-GOV-7 Changes after fold-0 data exist** (the post-fold-0 change rule, audit item P12).

1. **When fold-0 data exist.** Under any chain, manifest or host, fold-0 data exist from the first moment any of
   these exists: a started G2 timing unit; a started fold-0 DeepMzyme fit; a fold-0 CLEAN inner fit, export or inner
   prediction. These carry predictions for curated test proteins of folds 1-4, so fold-0 development exposes folds
   1-4. Label-free embeddings are not fold-0 data.
2. **Classes.** From that moment the scientific identity is fixed. Every later change states its class.
   - **(a) Execution change.** Only execution files change, and the pinned execution functions stay identical. A
     new execution manifest records it. There is no rerun and no exposure label.
   - **(a2) Analysis completion.** A link of sequence 3 or higher, published after the fold-0 FUT record and before
     Freeze 2. It may change only the four post-fit analysis files and may only add refusals and descriptive
     outputs; it never removes or relaxes a check. The pinned functions and `freeze.live_constants()` stay
     identical. Every amendment-2 refusal fixture is still refused, and every amendment-2 decision fixture gives
     identical decision fields. It changes no decision rule, metric, population, threshold, seed, draw count, family
     composition or decision sentence. A test-input builder qualifies only if every byte it gives a checkpoint is
     verified by an unchanged scientific verifier against a Freeze-1 or amendment-2 recipe pin. The scientific
     identity and the fit chain do not move.
   - **(b) Defect amendment** (sequence 3 or higher). It corrects a demonstrated defect: the code does not implement
     the published rule, or a crash or refusal blocks a pre-registered step. This includes defects in the CLEAN
     comparator.
     - A synthetic or training-only test fails before the fix and passes after it.
     - It adds, removes or retunes no family, lever, grid value, threshold, metric, population, seed, epoch count,
       futility parameter or claim rule.
     - Its record gives the UTC time, the SHA-256 of every fold-0 record existing then, the record through which
       the defect was found, and whether any fold-0 inner score had been viewed before the defect was identified.
     - Every affected fold-0 step is rerun in full under the corrected code before folds 1-4 start or continue.
       Superseded records are kept and reported.
     - If the correction changes the behaviour of any fold 1-4 unit, step or CLEAN arm relative to amendment 2, the
       results of folds 1-4 and every pooled result are labelled "development-exposed (amendment N)" in the
       decision-sentence note, report headings and summaries. The claim rule is otherwise unchanged.
   - **(c) Any other change** is outside this campaign: stop and close out, or start a successor campaign under a
     new Freeze 1, in which every outer fold decides from its own training data and folds 1-4 are labelled
     development-contaminated.
   - **(d) Futility precedence.** A futility decision is never overridden toward continuing. Once a `stop_futility`
     record exists in this campaign, under any chain, no class-(b) amendment may rerun S1, S2 or FUT; a defect found
     after a stop goes only into the close-out record or a successor campaign. After a `continue`, a class-(b)
     rerun writes a new fold-0 chain under its new fit chain; if that rerun says `stop_futility`, the stop is
     binding.
3. **Required fields.** A record of sequence 3 or higher is refused without its class, timestamps and fold-0 record
   list.

**A2-GOV-8 Stops and budgets.** Budget ceilings follow the project's two-choice escalation (stop and close out, or
record a larger ceiling) and never change a scientific rule. The maintainer may stop at any time. Only a stop under
A2-FUT yields "not shown to outperform"; any other stop before the scoring pass yields no comparative result.

### A2-SEL: selection (A2-D1, A2-D4)

**A2-SEL-1 Pipeline.** The selection pipeline is `amendment2_selection_v1`: S1 `best_inner_harm_guarded_v1`; S2
`l9_tau_promote_primary_v1` (L9 only); no S3; FUT in fold 0 only (A2-FUT); S4 `refit_spec_v2`. The plan format is
`deepmzyme_ec_clean30_fold_plan_v2`; step records use `deepmzyme_ec_clean30_step_record_v2` and the FUT record
`deepmzyme_ec_clean30_futility_record_v1`. The step order is S1, S2, FUT, S4 in fold 0 and S1, S2, S4 in folds 1-4.
The canonical `selection_policy_record_v2()` is hashed into amendment 2, and every step record carries that SHA-256
and the fit chain.

**A2-SEL-2 Primary inner score.** For one configuration (family and learning-rate setting) of an outer fold, each
screening seed (42, 43) pools the out-of-fold predictions of its four inner fits; every certified inner-validation
protein appears once.
- **Seed score.** Protein accuracy over the pooled proteins of EC classes 1-6, in points.
- **Class 7.** Class-7 proteins are outside the population. A class-7 prediction for a protein of EC classes 1-6 is
  wrong.
- **Prediction.** The arg-max of seven logits; exact ties go to the lowest class index.
- **Configuration score.** The mean of the two seed scores.
- **Descriptive metrics.** Balanced accuracy over EC classes 1-6, seven-class balanced accuracy, macro-F1 over EC
  classes 1-6 and per-class recall are recorded and do not decide. Per-class rows stay in private step records; the
  public log prints totals only.

**A2-SEL-3 Learning rate.** Each family's setting is chosen by the A2-SEL-2 score over its unchanged two-setting
grid: the first-listed setting within 0.5 points of the family maximum (inclusive, tolerance 1e-9). F6 uses F5's
grid (1e-4, 3e-4).

**A2-SEL-4 Harm guard.** Compare a candidate X with the anchor F2, each at its chosen setting. For each EC class c of
1-6 with at least one pooled inner protein, let k_X(c) and k_F2(c) be the correct predictions for proteins of class c,
summed over both screening seeds. X is ineligible if, for some such c, k_X(c) = 0 and k_F2(c) ≥ 1. Class 7 is exempt,
and F2 is always eligible. The historical recall-guard rows (a drop of at least 5 points and at least 3 net proteins,
for classes with at least 30 training proteins) are recorded against F2 and do not decide.

**A2-SEL-5 Family choice.** The candidates in every outer fold are F2, F3, F4, F5 and F6. Let M be the highest
A2-SEL-2 score among eligible candidates. The selected family is the first in the order F2, F5, F3, F4, F6 that is
eligible and scores at least M − 0.5 points (tolerance 1e-9). No significance test, Holm family or minimum gain
applies. The rule selects a model; it does not establish that any family is better than another. The S1 record
states that F6 and C-GF pretraining may use non-metalloenzyme training proteins two or more homology hops from an
inner-validation protein, which F2-F5 never see (A2-F6-3).

**A2-SEL-6 Controls.** Scored and recorded, never selected or refit: F1 (fold 0 only); C-G (every fold); C-GF (every
fold), the F6 pretraining classifier scored on the sequence-mean vectors of the inner-validation proteins it did not
see (A2-F6-7).

**A2-SEL-7 Failure handling.** S1 runs no bootstrap. The step is refused, no record is written and no family is
dropped if a required fit is missing, duplicated, foreign or unverifiable, a configuration does not pool, a candidate
family is missing or a score is not finite. A failed fit is rerun only as the identical planned unit. If a family
cannot be produced, the fold stops for a maintainer decision under A2-GOV-7.

**A2-SEL-8 L9.**
- **Candidates.** On the selected configuration's pooled predictions, every row is adjusted with the add-one smoothed
  training priors of its own inner fit, for τ in {−1, −0.5, 0.5, 1}.
- **Challenger.** The τ with the highest A2-SEL-2 score; equal scores go to the smaller |τ|, then to the negative
  value.
- **Adoption.** Against τ = 0, all of: a gain of at least 1.0 point; a one-sided paired inner-component bootstrap
  lower bound above 0 at α = 0.01 (10,000 draws, seed 20261023, unstratified, engine `stratified_rao_wu_v2`, at most
  1% skipped draws); the historical recall guard passes; and no EC class 1-6 seed-mean recall falls to zero.
- **Abort.** A bootstrap abort means not adopted (τ = 0), and the abort is recorded.
- **Descriptive.** The balanced-accuracy-scored τ choice is reported.

**A2-SEL-9 S4 refit specification.** For each outer fold an S4 record is appended; in fold 0 only after a verified
FUT = `continue`. It binds the S1, S2 and, in fold 0, FUT record hashes, and fixes: the family and learning-rate
setting; 30 epochs; raw-distance RBFs for GVP families; τ from S2 (0 if not adopted); seeds 42-46; and normalisation,
class weights and priors recomputed on the fold's full metallo training view. If F6 is selected, each refit
pretrains on the whole corpus (A2-F6-6). Mechanism: `refit-recipe` writes each fold's verified refit recipe (with
the fold-0 binding `continue`) and `verify-amendment-chain` the chain receipt; the scoring manifest declares both by
SHA-256 and the scoring pass reads them as data. Freeze 2 and the scoring pass require every refit identity and the
scoring τ to equal this record. The final system is the mean of the five refit logits, then L9 with the refit priors.

### A2-TRN: fixed training settings (A2-D4)

**A2-TRN-1 Schedule.** 30 epochs, a per-epoch cosine schedule and the terminal checkpoint. Batch 4, AdamW with weight
decay 1e-4, gradient clipping 1.0, 6-Å edges, metal nodes disabled. The final epoch's learning rate is still
positive, which stays disclosed.

**A2-TRN-2 Raw-distance RBFs.** Under amendment 2, F1, F3 and F4 expand node and edge distances in raw ångström with
the existing 16 Gaussian RBFs (centres 0-12 Å, σ 0.75 Å).
- **Distances.** Node distances: CA, functional-group centroid and nearest donor atom to the nearest selected metal.
  Edge distances: contact and CA-CA.
- **Sentinel.** The 999-Å "no donor atom" sentinel is kept. Every distance of 20 Å or more gives an exactly zero RBF
  row in float32, and that row is the mask.
- **Refusal.** The trainer refuses a raw-distance unit whose batches lack the raw distance tensors.
- **Unchanged.** No graph is rebuilt. F2, F5, F6 and C-G are unchanged.
- **Unit IDs.** Raw-distance units carry `_rbfraw` in their configuration name.

**A2-TRN-3 Levers and steps.** L1-L8 are not screened, and no S2 or S3 fit is run. L9 is the only screened lever.
Every lever stays defined, unchanged, for replay.

**A2-TRN-4 Families per fold.** Fold 0: F1, F2, F3, F4, F5, F6 and C-G, plus C-GF. Folds 1-4: F2, F3, F4, F5, F6 and
C-G, plus C-GF.

### A2-F6: candidate family F6 (A2-D3, A2-D7)

**A2-F6-1 Family.** F6 has the F5 architecture: the pocket ESMC encoder, the four site statistics, the whole-protein
ESMC branch (LayerNorm(960) → Linear(960→128) → GELU → Dropout 0.2) and the seven-way EC-class head. The branch is
pretrained (A2-F6-5) and then frozen; branch dropout stays active in training mode. All other parameters train
exactly as F5, under A2-TRN-1 and A2-SEL-3.

**A2-F6-2 Pretraining data.** For outer fold K the corpus is every accession of the official training file
`split30_train_split_K.csv`, each labelled with its EC class (first EC digit) by the campaign's EC-class rule.
Proteins whose EC numbers span more than one EC class are excluded, and any untrusted annotation stops the build.
Class-7 proteins are kept. No protein outside fold K's training file is used. Identical sequences under different
accessions stay separate rows.

**A2-F6-3 Inner exclusion** (declared interpretation of A2-D3, choice C8). For inner fold j of outer fold K, these
proteins are removed from the corpus: (a) the inner-validation proteins of j; (b) every protein with a D3 hit to or
from an inner-validation protein in the pinned metallo-training search of fold K (the CLEAN R1-inner exclusion list
of K and j); (c) every protein whose sequence equals an inner-validation protein's sequence, or whose accession
differs from one only in letter case; (d) every corpus protein that shares a selected AlphaFill donor with an
inner-validation protein. Rule (b) is one homology hop, deliberately: it is the CLEAN R1-inner rule and the one-hop
rule that defines B1-NH30. Metallo proteins further away are already in the same inner fold through the transitive
metallo components. Non-metalloenzyme proteins two or more hops away may remain in F6 pretraining, and the S1 record
states this asymmetry (A2-SEL-5). The lists of folds 1-4 are built before publication in a scratch copy of the
overlay's training-side identity files and bound in the record; the overlay is never written. Counts are reported
for fold 0 only; the other folds' counts stay in private receipts. S4 refits use the whole corpus.

**A2-F6-4 Embeddings.**
- **Recipe.** ESMC-300m (the Freeze-1 pinned weights), final normalised layer, mean over all residue tokens
  excluding BOS and EOS, accumulated in float64 and stored in float32, from the frozen sequence map without
  truncation.
- **Precision.** As A2-GOV-3. TF32 is off and the emitted dtypes must be float32.
- **One cache.** One sequence-keyed union cache serves F6. It serves CLEAN's ESMC-input base embeddings only if that
  recipe is byte-identical.
- **Order.** The union cache, check (i), attach and check (ii) are completed for fold 0 before G2 v2, and Freeze 1c
  v2 binds their receipt.
- **Checks before any F6 unit.** (i) A GPU build matches a CPU build made on the same pod on a label-free sample:
  the 64 lexicographically first sequence SHA-256 keys of the sequence map plus its 8 longest sequences, with cosine
  at least 0.9999 and maximum absolute difference at most 1e-4. (ii) For every fold training structure whose chain
  sequence equals its official sequence, the sequence mean matches the view's final-layer chain mean within the same
  limits; proteins whose sequences differ are counted.
- **Failures.** A failed check is a defect to repair. Tolerances never change.
- **Per-view copy.** Each view receives a fresh manifest holding only its corpus keys, with the SHA-256 of the sorted
  corpus keys as its allowlist record, no labels field and no parent allowlist.

**A2-F6-5 Pretraining procedure** (fixed, not tuned). Pretraining runs inside each F6 fit, before the F6 model is
built.
- **Model.** The C-G whole-protein branch of A2-F6-1 plus a 128→128→7 head with dropout 0.2.
- **Loss.** Cross-entropy with inverse-frequency class weights over the split's accessions, normalised to mean 1 over
  the classes present.
- **Optimiser.** AdamW at learning rate 1e-3, weight decay 1e-4, clip 1.0; batch 64, 30 epochs, a cosine schedule
  stepped per epoch, terminal weights.
- **Execution.** CPU, float32, one thread, on an isolated random-number stream seeded from the fit seed plus a fixed
  offset. All other F6 initial tensors, the batch order and the dropout draws equal F5's for the same seed.
- **What carries over.** Only the branch weights are transferred and frozen. The head serves only C-GF.
- **Same-host check.** The two learning-rate settings of one (inner fold, seed) must give equal branch digests when
  the recorded CPU identity and thread count match; otherwise the difference is recorded.

**A2-F6-6 Refit and test.** Each S4 refit pretrains on the whole corpus with its own refit seed (42-46). The frozen
branch in every checkpoint equals, bit for bit, the branch recorded at pretraining. The label-blind test pass uses
only checkpoints and test chain means; it never pretrains or reopens pretraining paths.

**A2-F6-7 Control C-GF.** The pretraining classifier of every F6 inner fit predicts that fit's inner-validation
proteins from their sequence-mean vectors in the attached corpus cache. Pooled inner scores come from the first F6
learning-rate setting of each (inner fold, seed). C-GF is reported beside C-G, F6 and CLEAN R1-inner on the same
closure-eligible inner proteins. It has no refit and no test pass.

**A2-F6-8 Disclosure.** Whenever F6 is evaluated or selected, reports state that F6's whole-protein branch was
trained on all single-EC-class enzymes of the fold's official training file, including non-metalloenzymes, which
reverses Freeze-1 decision U3 for that branch only, and that the branch therefore does not share the fold 1-4 metallo
screening gap. "DeepMzyme trains only on metalloenzymes" then applies to the pocket components and the other
families.

### A2-MET: metrics, claims and wording (A2-D2, P15)

**A2-MET-1 Primary metric.** Protein-level accuracy over the proteins of EC classes 1-6, from seven-way predictions.
Class-7 proteins are excluded from the population. A class-7 prediction, an explicit failure, a CLEAN distance tie,
and a baseline no-hit or tie all count as errors. A protein listed in several folds keeps a total weight of one.
Balanced accuracy over the EC classes 1-6 present becomes the first secondary metric. Seven-class balanced
accuracy, accuracy over all proteins and macro-F1 over EC classes 1-6 are reported.

**A2-MET-2 Co-primary claims.** Unchanged except for the metric. P1 is B1-all and P2 is B1-NH30, both pooled over the
five folds and restricted to proteins of EC classes 1-6. Each population is tested by intersection-union of two
contrasts: DeepMzyme-final minus the CLEAN-R1 seed mean, and DeepMzyme-final minus the CLEAN-R1 ensemble. Each
contrast takes a one-sided lower bound from one paired homology-cluster bootstrap (10,000 draws, seed 20261023,
engine A2-STAT). Holm runs across P1 and P2: the smaller p is tested at 0.025, and the other at 0.05 only if the first
passes. The EC class 5 and EC class 6 removal reruns stay at each claim's Holm level and only qualify the wording.
**CLEAN sanity consequence.** If the R1 seed mean or ensemble includes a fit recorded as "proceed, disclosed"
(A2-CLN-3), both co-primaries are unavailable with the reason "a CLEAN sanity check failed without an identified
defect". The same holds for any secondary hypothesis whose CLEAN arm includes such a fit. For the fold-0 held-out
check (A2-CLN-2), which has one status per arm (choice C6b): an R1-inner arm recorded as "proceed, disclosed" makes
both co-primaries and every secondary hypothesis against R1 unavailable with the same reason, because R1-inner
stands for the R1 recipe in fold 0; an R2-inner arm so recorded makes only the hypotheses against the R2 seed mean
(secondary hypotheses 3 and 4) unavailable. The scoring pass and A2-FUT apply this one rule
(`clean_sanity.held_out_consequence`).

**A2-MET-3 Secondary family.** Seven hypotheses under Holm at family α 0.05, in this fixed order, which also breaks
ties between equal p-values: (1) P1, balanced accuracy over EC classes 1-6, against R1 (intersection-union over the
seed mean and ensemble); (2) P2, the same; (3) P1, primary metric, against the R2 seed mean; (4) P2, the same; (5) P1,
macro-F1 over EC classes 1-6, against R1 (intersection-union); (6) P2, the same; (7) the similarity difference in
differences on the primary metric: (R1 seed mean on the [30,50] and >50 bins minus on B1-NH30) minus (DeepMzyme-final,
same strata).

**A2-MET-4 Descriptive results** (no claim). The primary metric and balanced accuracy on B1-NH30-seq (also the P2
template check), B1-D2, the identity bins, per fold and without listed EC-class transfers; accuracy over all proteins
and seven-class balanced accuracy; class-7 outcomes; C-ESMC, the MMseqs2 nearest-hit baseline and B3; the DeepMzyme
seed mean against the R1 seed mean; DeepMzyme-final at τ = 0 whenever a fold adopted τ ≠ 0; per-class recall; CLEAN
tie counts and the tie rescore; the CLEAN sanity records; and the effect of the small-stratum correction on balanced
accuracy (A2-STAT-7).

**A2-MET-5 Wording and scope.** The scope phrase SCOPE is: "a conservative metalloenzyme subset of the official CLEAN
split30 benchmark (AlphaFold models with AlphaFill-transplanted, MAHOMES-catalytic metal sites; EC class = first EC
digit; accuracy over the proteins of EC classes 1-6)", with any qualifiers added inside the parentheses.
- **P1 passing:** "Outperforms retrained CLEAN in EC class prediction on SCOPE."
- **P2 passing:** "Outperforms retrained CLEAN in EC class prediction on the proteins of SCOPE that have no ≥30%
  identity hit to their fold's training file and no shared selected AlphaFill donor."
- **Not passing:** "Not shown to outperform retrained CLEAN in EC class prediction on" SCOPE.
- **Unavailable:** "No superiority inference on" SCOPE, followed by the reason and "results are descriptive".
- **A fold subset:** "No decision". **Never:** "equivalent".
- **Qualifiers:** the classes evaluated; dependence on EC class 5 or 6; template dependence; and "CLEAN distance ties
  counted as errors", added when either R1 contrast's point estimate under the 1/m tie rescore (A2-CLN-5) is ≤ 0.
- **Estimand statement.** Every report states that the primary metric weights the large EC classes, so a gain
  confined to EC classes 5-6 would not be detected by it, and that balanced accuracy over EC classes 1-6 is the
  first secondary metric.
- **Scope statement.** Every report states the cohort (the conservative metalloenzyme cohort of the official folds:
  AlphaFold models whose metals AlphaFill transplanted from one selected donor entry, keeping only the sites MAHOMES
  predicts as catalytic; 118 of the 1,648 curated fold-0 test proteins), one EC class (first EC digit) per protein,
  and the comparator (CLEAN retrained on each fold's full official training file with its published recipe,
  compared as the five-seed mean and the five-seed ensemble). The decisions concern this benchmark and procedure;
  they do not attribute a difference to the language model, the structure input or the training data.

### A2-STAT: bootstrap engine v2 (P9)

**A2-STAT-1 Engine.** Engine `stratified_rao_wu_v2` serves every amendment-2 bootstrap, including balanced accuracy
and macro-F1. The Freeze-1 engine `stratified_k_of_k_v1` stays for replay, with its metric list unchanged. Clusters
are cohort components restricted to the population. Each cluster joins the stratum of the rarest EC class 1-6 it
contains (rarity is the total protein weight in the population; ties go to the lower class number). Clusters with
only class-7 proteins form their own stratum.

**A2-STAT-2 Small strata.** For a stratum of k ≥ 2 clusters, each draw picks k − 1 clusters uniformly with
replacement and multiplies each pick's count by k/(k − 1). A one-cluster stratum keeps multiplicity 1 and uses no
random numbers. Strata are drawn in fixed order, in batches of 1,000.

**A2-STAT-3 Single-cluster merge** (primary metric only). Strata of EC classes 1-6 are visited from rarest to most
common. A one-cluster stratum gives its cluster to the next more common non-empty stratum; the most common gives it
to the next rarer non-empty stratum instead. Merges are recorded. EC classes are never merged for balanced accuracy
or macro-F1.

**A2-STAT-4 Valid draws.** For the primary metric every draw is valid. For balanced accuracy and macro-F1, a draw is
valid only if it contains every EC class 1-6 present in the population.

**A2-STAT-5 Support gate.** Inference is unavailable for the primary metric when fewer than two homology clusters
hold a protein of EC classes 1-6. For balanced accuracy and macro-F1 it is unavailable when (a) a stratum of an EC
class 1-6 holds a single cluster; (b) for macro-F1 only, a single-cluster class-7-only stratum is predicted by a
compared system as an EC class 1-6 present in the population; or (c) an EC class 1-6 present in the population is
not guaranteed in every draw. The exact absence probability is recorded: the product over strata of k ≥ 2 clusters
of ((k − m)/k)^(k−1), where m of the k clusters contain the class; a one-cluster stratum without the class contributes
a factor of 1. No class or cluster is removed to restore eligibility. Support, stratum sizes, merges and absence
probabilities are reported.

**A2-STAT-6 Unavailable.** A gated hypothesis, or one whose bootstrap exceeds 1% skipped draws, is unavailable: it
keeps its point estimates, takes p = 1, stays in its Holm family and never passes. For a co-primary, the other
population is still tested at 0.025. The scoring pass continues.

**A2-STAT-7 Effect of the correction on balanced accuracy** (descriptive). The power simulation found the Freeze-1
balanced-accuracy bootstrap under-dispersed when rare-class clusters are sparse. For every balanced-accuracy
hypothesis the report gives, beside the engine-v2 bound, the bound under the Freeze-1 k-of-k resampling with the
same seed and draws. This comparison never changes a decision.

### A2-CLN: CLEAN checks (P8, P17, P18)

**A2-CLN-1 In-sample check.** For every CLEAN fit (inner, R1, R2, C-ESMC, all folds), after its training receipt and
the TF32-off policy record are verified and before the fit enters any record.
- **Sample.** The fit's own training proteins that have a single EC class among EC classes 1-6 and whose every full
  EC number has at least two training proteins, ordered by the SHA-256 of fold, regime, inner fold (or "-") and
  accession joined by colons; the first 256 are taken.
- **Export.** A validation-mode distance export with in-sample queries allowed, for this check only.
- **Anchor.** A raw-embedding anchor on the same queries: centres are the per-EC-number means of the fit's own base
  input embeddings, distances are Euclidean, with CLEAN's projection and tie rule.
- **Pass.** All of: (IS1) primary metric ≥ 0.80; (IS2) primary metric ≥ the anchor's − 0.05; (IS3) ties ≤ 5% of
  queries; (IS4) the export records exactly the sample as its in-sample queries.

**A2-CLN-2 Held-out check (fold 0).** For the four R1-inner and four R2-inner fits, computed as soon as their eight
validation exports exist; it and any disposition must exist before the fold-0 S1 record is built. Queries are
inner-validation proteins with one EC class among EC classes 1-6 whose full EC numbers all occur among the R1-inner
training centres of their inner fold. Pooled per arm, all of: (HO1) primary metric ≥ the anchor's − 0.10; (HO2,
R1-inner only) primary metric ≥ the accuracy of predicting the training table's most frequent EC class + 0.10; (HO3)
each fit's ties ≤ 5%; (HO4) each fit's primary metric ≥ its own anchor's − 0.20. The S1 and FUT records bind this
record's SHA-256.

**A2-CLN-3 Failure response.** A failure gives the status `comparator_diagnosis_required`, and the fit enters no
record. The maintainer looks for defects in the inputs, embedding farms, EC column order, checkpoint identity,
distance map, projection and receipts. An input or receipt defect is fixed and recorded, and the same pinned recipe
and seeds are refit and rechecked. A defect in a scientific file found after fold-0 data exist is a class-(b)
amendment (A2-GOV-7), with the "development-exposed" label when fold 1-4 arms change. Recipe, epochs, loss, widths,
learning rate and seeds never change; refits are never chosen by score, and no fit is dropped for scoring low. If
no defect is found, the maintainer records "proceed, disclosed" or "stop"; "proceed, disclosed" has the fixed
consequence of A2-MET-2. A disposition of the held-out record names its failed arm; each failed arm takes exactly one
disposition and a passed arm none (choice C6b).

**A2-CLN-4 Thresholds and reporting.** Thresholds are fixed here. Loss trends are not used, because the recipe resets
the optimiser every 100 epochs. Published CLEAN values are context only. Records print totals only. The scoring pass
refuses a CLEAN checkpoint without a passed record or a recorded disposition.

**A2-CLN-5 Ties.** The rule is unchanged: a query whose nearest EC centres within 1e-6 belong to different EC classes
is a tie and counts as an error. Per CLEAN arm and co-primary population, reports give the tie counts, the number of
ties whose true EC class is among the tied classes, and a descriptive 1/m rescore over the m distinct tied EC
classes (points only). Exact DeepMzyme arg-max ties go to the lowest class index, while CLEAN distance ties always
count as errors; this asymmetry is disclosed. Before G2, the structural tie exposure is counted from the official
training files and the R2 lists only (EC-number groups with identical training-member sets that span more than one
EC class); only fold 0's count may be stated.

**A2-CLN-6 SupCon-Hard.** CLEAN is trained with its published triplet-margin recipe. The upstream repository also
provides a SupCon-Hard objective, which its documentation reports as better on small training sets. It is not run,
because the comparison uses the published recipe without local tuning (U12), and the metallo-only R2 arm uses the
same recipe.

### A2-FUT: fold-0 futility rule (A2-D5)

**A2-FUT-1 When and where.** Once per fit chain, on the pod CPU in the fold-0 session; after the fold-0 S1 and S2
records and after a passed A2-CLN-2 record or a recorded disposition; before any fold-0 S4 refit, any fit of folds
1-4 and any final-arm CLEAN fit. Only a binding record decides: a real run that could not bind (without the
A2-FUT-2 exclusion inputs, or with any test setting) is refused before anything is written, a smoke run writes
neither a pointer file nor a FUT step record, and a second run of the same fit chain is refused. The R1-inner and
R2-inner exports must be the fits the held-out record checked (equal checkpoint SHA-256 per arm and inner fold),
and the S1 record must bind that held-out record. If the held-out consequence (A2-MET-2, C6b) leaves no co-primary
available, the decision is `stop_futility` with 0 successes.

**A2-FUT-2 Population U.** Fold-0 inner-validation proteins of EC classes 1-6 whose full EC numbers all occur among
the R1-inner training centres of their inner fold, excluding proteins whose sequence equals a protein kept in their
inner fold's R1-inner training set. The fold-0 count of excluded proteins is reported.

**A2-FUT-3 Arms.** DeepMzyme: the mean of the two screening-seed logits of the selected S1 configuration, adjusted
with the S2 τ and that inner fit's training priors, then arg-max (choice C7). CLEAN: R1-inner of the protein's inner
fold (one seed), nearest-centre EC class; a distance tie counts as wrong. This arm stands for both the R1 seed mean
and the R1 ensemble.

**A2-FUT-4 Units.** The inner identity components of the fold-0 overlay copy, restricted to U. Each component's
stratum is its rarest EC class 1-6, by protein count in U.

**A2-FUT-5 Projection.** Targets N1 = 526 and N2 = 226 proteins of EC classes 1-6: design-time constants (about 530
evaluable B1-all proteins and about 230 B1-NH30 proteins, each minus the at most 4 class-7 test proteins stated in
the design). For r = 0, …, 1,999, with a generator seeded by (20261104, r):
- (a) resample U's components with replacement within each stratum, to that stratum's component count k_s;
- (b) synthetic B1-all: from each stratum of the step-(a) sample, draw n1_s = max(1, ⌊k_s·N1/n_U + 0.5⌋) components
  with replacement, in draw order (n_U is the number of proteins in U); each drawn copy becomes its own cluster;
- (c) synthetic B1-NH30 (nested): in each stratum take the first n2_s = max(1, ⌊n1_s·N2/N1 + 0.5⌋) copies of step
  (b), so that every B1-NH30 cluster is also a B1-all cluster;
- (d) score both synthetic populations as the scoring pass scores a co-primary: primary metric; engine
  `stratified_rao_wu_v2`, 10,000 draws, seed 20261023, at most 1% skipped draws; the A2-STAT gate (a gated or
  aborted population is unavailable with p = 1); intersection-union over the two R1 components; Holm at 0.025 then
  0.05;
- (e) replicate r succeeds if at least one co-primary reads "Outperforms".

**A2-FUT-6 Decision.** `continue` if at least 400 of the 2,000 replicates succeed (projected probability ≥ 0.20);
otherwise `stop_futility`. The comparison is between integers.

**A2-FUT-7 Binding stop.** After `stop_futility` no further campaign fit is run and no test protein is scored. The
result reads: "On a conservative metalloenzyme subset of the official CLEAN split30 benchmark (AlphaFold models with
AlphaFill-transplanted, MAHOMES-catalytic metal sites; EC class = first EC digit), DeepMzyme was not shown to
outperform retrained CLEAN in EC class prediction. The campaign stopped for futility after fold 0 under its
pre-declared rule; no test protein was scored." A2-GOV-7(d) applies. Stopping for futility cannot create a false
"Outperforms" result.

**A2-FUT-8 Descriptive content** (does not decide; totals only, no protein identifier). Projected probabilities per
population, for both and for either, with Wilson 95% intervals; the same projection with independent (non-nested)
draws, 500 replicates; the observed inner primary difference and its component-bootstrap bound; conditional power
at the observed difference (step (a) omitted), 500 replicates; targets divided by 1.25 and by 1.5, 500 replicates
each; `historical_rule_outcome`, the Freeze-1 gap computation (EC classes 1-6 balanced-accuracy gap against −3.0,
seven-class and R2-inner gaps, denominators); and C-GF beside CLEAN R1-inner and the selected configuration on U.
The synthetic operating-characteristic table `futility_oc.json` (action `futility-oc`: a declared synthetic U of 460
proteins of EC classes 1-6 at true margins of 0, 2, 4, 6 and 8 points, 200 replicates each; projected success
probability and the decision the rule would take) is computed before publication, bound by SHA-256 and cited.

**A2-FUT-9 Limitations disclosed with the record.** Inner-validation proteins have no D3 edge to their inner training
proteins, so B1-all is projected from an NH30-like population. Inner components are finer than the claim clusters.
Inner fits use three quarters of the training proteins and two seeds, against five full refits. One R1-inner seed
stands for the R1 ensemble. The selected configuration's inner score carries selection optimism. The projection is a
stop rule, not an estimate of test performance.

### A2-DIS: disclosures added or changed

- The primary-metric change, its date and its motivation (power; accuracy weights EC classes 5 and 6 less than
  balanced accuracy does), with the estimand statement of A2-MET-5.
- The anti-conservative Freeze-1 balanced-accuracy bootstrap found by the power simulation, and the correction
  applied to balanced accuracy (A2-STAT-2, A2-STAT-7).
- Design-time label exposure, and that fold-0 development exposes the curated test proteins of folds 1-4.
- The claim scope (A2-MET-5).
- The CLEAN tie rule, the tie asymmetry and the rescore (A2-CLN-5).
- The CLEAN sanity anchors, any disposition and its consequence (A2-CLN-1 to A2-CLN-3).
- SupCon-Hard not used (A2-CLN-6).
- The F6 training-data statement and screening-gap exception, per fold (A2-F6-8).
- The F6 and C-GF inner hold-out asymmetry (A2-F6-3).
- The uncharacterized operating characteristics of the S1 rule.
- The positive final-epoch learning rate.
- The futility limitations (A2-FUT-9).

### A2-SUP: superseded and unchanged

**Superseded for real execution under the amendment-2 chain** (each stays replayable on its original records):
- U8, which becomes A2-MET-1;
- U9 epoch selection and the U10 lever screen;
- Freeze-1 D1 and the S1 amendment rule, as the decision rule for S1;
- U3, for the F6 branch only;
- SA1 and SA1-c, which A2-STAT extends;
- the composition of E1 (its size and α are kept);
- C-8, the continue-at-−3 gap gate, which becomes descriptive;
- C-5, and C-1 to C-4 together with the L3 sub-ceiling and cost machinery, which become inactive;
- the presence-skip convention for the primary metric;
- the S1 cap of 96 and the per-fold cap of 181.

**Unchanged:**
- the official folds, the co-primary populations B1-all and B1-NH30, intersection-union against both R1 components,
  Holm across the two populations, the bootstrap seeds and draw counts;
- the contrast zero tolerance;
- E3, F1 and E5;
- the identity rules, cohort, views and overlay;
- the CLEAN recipe (upstream commit, ESM-1b, triplet loss, learning rate, epochs, seeds, ensemble rule);
- the G3 limits;
- the write-once claim and the single scoring pass after Freeze 2;
- every root file (C-10).

### A2-BUD: fit caps

| Step | Fold 0 | Folds 1-4, each |
|---|---|---|
| S1 (16 fits per family: 2 learning rates × 4 inner folds × 2 seeds) | 112 (F1-F6, C-G) | 96 (F2-F6, C-G) |
| S2, S3 | 0 | 0 |
| S4 (seeds 42-46) | 5 | 5 |
| Cap per fold | 117 | 101 |

Campaign DeepMzyme fits: 521, against 745-905 under Freeze 1; a futility stop spends 112. F6 pretraining runs inside
each F6 fit, and C-GF adds no fit. CLEAN keeps its 63 fits and adds 63 in-sample exports and one held-out record. The
G2 replay anchor is unchanged: F2, learning rate 1e-4, inner fold 0, seed 42. Per-fit costs are unmeasured while G2
is 0/12.

## Policy deviations

Citations are to tracked files; line numbers are at `c163995`.

1. **Primary metric (A2-D2).** Deviates from Freeze-1 U8 (`freeze1.md:144`), the design's primary-metric bullet
   (`design.md:46-48`), the note at `scoring.py:342-345`, and the EC playbook's selection metric and its rationale
   for imbalanced EC classes (`EC_TRAINING_PIPELINE_PLAYBOOK.md:650-653`), now replaced in S1 and L9. Motivation:
   statistical power ([power simulation](power_simulation.md)). Balanced accuracy over EC classes 1-6 is the first
   secondary metric and is reported with every result. Still met: `Plan.md:613-614` (final level-1 balanced
   accuracy, macro-F1 and per-class recall). The class composition of the test populations was known at design time.
2. **S1 rule without a significance test (A2-D1).** Deviates from Freeze-1 D1 (`freeze1.md:79`), the S1
   amendment, and the statistical methodology of `Plan.md:1116-1122` (Stage 6B promotes a candidate only after
   fold-level paired confidence intervals, rare-class recall protection and predeclared tie-breakers) and
   `Plan.md:347-356` ("Raw validation deltas alone are not sufficient promotion evidence"). The maintainer
   explicitly requested this change for this campaign (decision A2-D1); `Plan.md` stays unedited. Rare-class protection becomes the zero-recall harm guard, balanced
   accuracy as first secondary, the EC class 5/6 removal reruns and the descriptive recall guard. The rule's
   operating characteristics are uncharacterized.
3. **Non-metalloenzyme training data in F6 (A2-D3).** Deviates from Freeze-1 U3 (`freeze1.md:139`), U5 (`:141`) and
   U10 (`:132`), and from the design disclosure "DeepMzyme trains only on metalloenzymes" (`design.md:86`).
   Admissible because no fold-0 data exist. The mission framing of `Plan.md:3-5` is campaign-scoped, and `Plan.md`
   stays unedited. The fold 1-4 screening-gap disclosure is qualified for F6.
4. **F6 vectors.** Sequence means of official sequences, not structure chain means, with parity check (ii). The
   exclusion of proteins with more than one EC class follows A2-D3 and is not a deviation.
5. **One-hop F6 inner hold-out** (interpretation of A2-D3, choice C8). Equals the CLEAN R1-inner and B1-NH30 rules,
   adds the exact-sequence, alias and donor rules; the asymmetry is disclosed.
6. **F6 pretraining rule.** Deviates from the design's training rule (batch 4, `design.md:36-40`). It is a separate
   fixed rule, identical in selection and refit.
7. **Epochs and levers (A2-D4).** Deviates from U9 (`freeze1.md:145`), U10, the design's "each fold chooses 12, 20 or
   30 epochs" (`design.md:39`), and C-1 to C-4 together with the L3 sub-ceiling. These become inactive but replayable.
8. **Raw-distance RBF default (P14).** In v2 only, F1, F3 and F4 encode distances in raw ångström; the distance part
   of L2 moves into the base recipe. The v1 family definitions stay replayable.
9. **F1 in fold 0 only.** The baseline-first order of `Plan.md:1148-1163` and `EC_TRAINING_PIPELINE_PLAYBOOK.md:655-660`
   is met in fold 0 only.
10. **Binding futility stop (A2-D5).** Replaces C-8 (`freeze1.md:62`), C-5 (`:59`), the gate's maintainer-decision
    branch (`selector.py:116-119`) and the continuation's gap-gate step. It is a pre-declared scientific stop rule,
    distinct from the project's budget-ceiling escalation.
11. **Stratum gate and engine (P9).** Extends SA1 and SA1-c (`freeze1.md:129-130`) and the skip convention at
    `stats.py:18-20`. A co-primary abort becomes "unavailable", replacing `scoring.py:1971-1973`. Resampling becomes
    Rao-Wu k−1 of k; the bootstrap stays paired, cluster-level and stratified.
12. **Secondary family.** Freeze-1 E1 (`freeze1.md:86`) changes composition; its size and α are kept.
13. **In-sample queries.** Freeze-1 F5 (`freeze1.md:100`) gets a scoped exemption for the CLEAN sanity export only.
    The gap path and futility keep refusing in-sample exports.
14. **Fit caps.** `STEP_FIT_CAPS` and `MAX_FITS_PER_FOLD = 181` (`campaign.py:52,58`) are replaced in v2 only.
15. **Governance.** The single-link verifier (`freeze.py:1980-2040`) is complemented by the general chain verifier,
    which runs each link at its own commit. The R-CMD allowlist becomes `r_cmd_v2`. Per-unit equality with the
    Freeze-1 pins becomes scientific-identity binding. Class (a2) is added. No root file is edited (C-10,
    `freeze1.md:51`).
16. **CLEAN on GPU and the comparator CUDA path (A2-D7, execution v5).** Deviates from Freeze-1 M3, M4 and F8
    (`freeze1.md:103,117,118`). The comparator is a scientific file in the scope list.
17. **ESMC on GPU.** The SDK's CUDA path is bf16 (`esmc_chain_means.py:533-534`). The float32 path and the parity
    checks implement Freeze-1 C9 (`freeze1.md:77`) and F4 (`:99`).
18. **C2** (`freeze1.md:70`). Met by a fresh per-view manifest, never by copying a cache subset with its parent
    allowlist.
19. **CLEAN sanity consequence.** A "proceed, disclosed" fit removes superiority inference for the comparisons it
    enters. This adds to the claim rule of the design and can only remove claims.
20. **Execution route (A2-D6, declared by execution v5).** Runpod only, against the primary route of
    `Plan.md:310-312` (GCP L4, Colab fallback) and the project's operating policy.

## Source scope

The [draft record](amendment_draft.json) lists every changed `src/` and `scripts/` file with its SHA-256 at
`c163995` (equal to the composite Freeze-1 and S1-amendment pin) and of the staged content, its R-CMD class, its set
and the rules it implements. The refusal and decision fixtures of class (a2)
(`tests/fixtures/ec_clean30_amendment2_refusals.json`, `tests/fixtures/ec_clean30_amendment2_decisions.json`) are
bound there by SHA-256. Summary:

| Set | Files |
|---|---|
| Scientific, modified | `scripts/run_ec_clean30_campaign.py` (new actions only), `src/ec_clean30/campaign.py`, `commands.py`, `selection.py`, `selector.py`, `stats.py`, `views.py`, `src/embed_helpers/esmc_chain_means.py`, `src/training/config.py`, `ec_class_metrics.py`, `ec_prediction_export.py`, `run.py` |
| Scientific, new | `scripts/run_ec_clean30_clean_sanity.py`, `src/ec_clean30/amendment_rules.py`, `amendment_chain.py`, `amendment2.py`, `clean_sanity.py`, `futility.py`, `pretrain_corpus.py`, `src/training/global_esm_pretraining.py` |
| Post-fit analysis, modified | `src/ec_clean30/scoring.py` |
| Scientific, written by execution v5 | `src/ec_clean30/runtime_precision.py` (new), `scripts/run_clean_split30_comparator.py` (modified) |

Explicitly unchanged: `src/model.py`, `src/model_variants/`, `src/training/graph_dataset.py`, `src/featurization.py`,
`src/training/global_esm_features.py`, `src/training/loop.py`, and `freeze.py`, `provenance.py`, `identity.py`,
`overlay.py`, `cohort.py`, `transfers.py`, `logit_adjustment.py`, `baseline.py`, `simulation.py`, `s1_simulation.py`
and `r2_lists.py` in `src/ec_clean30/`.
