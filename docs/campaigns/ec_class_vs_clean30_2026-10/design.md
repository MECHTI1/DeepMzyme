# EC class versus CLEAN30: design

Campaign [README](README.md); decisions and reasons: [log evc-001](log.md#evc-001).

## Benchmark, systems and claims

- **Benchmark.** The five official split30 train/test folds, unchanged, restricted to the prepared conservative metallo
  cohort: 740 proteins, 545 test-structure occurrences, about 530 evaluable after excluding proteins annotated with
  more than one EC class ([cohort and its fold 1–4 screening gap](../../DATASETS.md#clean30-conservativecurrent-main)).
  Structures are AlphaFold models whose metals were transplanted by AlphaFill from one selected donor PDB entry per
  protein; only sites that MAHOMES predicts as catalytic are kept, so metal sites are computational.
- **Populations (co-primary).** B1-all: every evaluable official metallo test protein, pooled over folds. B1-NH30: those
  with no hit of ≥30% identity over ≥50 aligned residue pairs (E ≤ 1e-3, any coverage, pinned MMseqs2 18-8cc5c) to any
  protein of their fold's full training file, and no shared selected AlphaFill donor. split30's nominal 30% is an
  MMseqs2 clustering threshold whose coverage setting was not published (the MMseqs2 default, also used by CARE, is ≥80%
  of both sequences). Within fold-0 training, 5.5% of proteins have a ≥30% partner at that coverage, 42% at ≥50%
  coverage of the shorter sequence and 56% under the rule above. Secondary: a sequence-only no-hit set, identity bins
  and a ≥50%-coverage subset. All are analysis subsets of the official folds, not a re-split.
- **DeepMzyme.** Only-GVP and whole-protein-only controls; candidates Only-ESM (pocket), late fusion, and both with an
  added whole-protein ESMC branch. In every fold one pre-registered algorithm chooses the learning rate, family and
  levers from that fold's training proteins only (4 inner folds under the same identity rule, 2 screening seeds). The
  final model is a 5-seed refit ensemble. *[Amendment 2](#amendment-2-2026-11-04-before-any-g2-or-fold-0-data) adds
  the candidate F6, screens only the no-fit L9 lever and runs Only-GVP (F1) in fold 0 only.*
- **S1 amendment (2026-11-02, before any model comparison).** The approved versioned selector compares F3, F5
  and F4 directly with F2 in one fixed three-hypothesis Holm family at alpha 0.04. All three remain in the family
  when a gain or recall guard fails. The existing gain floor and recall/zero-recall guards apply against F2.
  Select the highest pooled seed-mean BA among passing candidates; within 0.5 points of that maximum prefer F5,
  then F3, then F4. If none passes, retain F2. F1 and C-G remain controls. The two-LR choice is unchanged.
  This selects a model; it does not establish an incremental modality benefit over another passing candidate
  or protect that candidate's recall gains. Historical escalation remains available for replay. The amendment
  requires authenticated source/evidence and publication before real execution; [the log](log.md#evc-012) owns
  verification and characterization results. Original Freeze 1/1b and prepared inputs remain unchanged.
  *Superseded as the S1 decision rule by amendment 2 (best inner score with a harm guard); it stays replayable.*
- **Inner selection** (declared deviation from the EC playbook's class-support rule,
  [launch gate](../../EC_TRAINING_PIPELINE_PLAYBOOK.md#readiness-outputs-and-explicit-launch-gate)): pooled inner
  out-of-fold BA over EC classes 1–6, 2 seeds. Class 7 (2–4 training proteins in at most 3 components per fold) may be absent from an
  inner split and is exempt from the zero-recall promotion guard; its recall is reported descriptively.
  *Amendment 2: the inner score is accuracy over the proteins of EC classes 1–6; BA is recorded descriptively.*
- **Training rule** (declared deviation from [Plan §7](../../../Plan.md#7-baseline-first-model-comparison-policy)
  checkpoint selection by validation metrics, whose fixed-schedule allowance is written for metal, and from the EC
  playbook default of a fixed LR and the best validation epoch): cosine to zero, terminal checkpoint, the same rule in
  selection and refit, batch 4; each fold chooses 12, 20 or 30 epochs
  ([TECH-027](../../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-027--cross-validation-and-final-refit-use-different-checkpoint-rules)).
  *Amendment 2: every fit uses 30 epochs; the epoch choice is superseded.*
- **CLEAN.** Upstream commit `f2bf2a4`, published recipe, not re-tuned here (ESM-1b, triplet loss, LR 5e-4, 2,500
  epochs; that epoch count was chosen upstream on these same folds). R1, the primary comparator: the fold's full
  training file (about 8.1K enzymes), 5 seeds, scored as the seed mean and as a 5-seed ensemble. R2: the fold's prepared
  metallo training subset, including the proteins with more than one EC class that DeepMzyme excludes, 3 seeds. Control: ESMC input instead of
  ESM-1b, 3 seeds. Baseline: MMseqs2 nearest-hit transfer.
- **Primary metric** (declared deviation from the seven-class EC BA default): protein-level balanced accuracy over EC classes 1–6.
  All seven-way predictions count; class 7 (translocases: at most about 4 test proteins, one family) is left out of the
  average. *Superseded by amendment 2: accuracy over the proteins of EC classes 1–6 is primary, and this BA is the
  first secondary metric.*
- **Claim rule.** "Outperforms CLEAN" needs one-sided paired homology-cluster bootstrap lower bounds above 0 against both
  the R1 seed mean and the R1 ensemble, Holm-corrected over B1-all and B1-NH30. Expected detectable margins: at least
  about 8 BA points (B1-all) and 11 (B1-NH30), before homology-cluster inflation. Without a pass the wording is "not
  shown to outperform"; every outcome is reported. *Amendment 2 keeps this rule with the new metric and adds the
  claim scope; its [power simulation](amendment2/power_simulation.md) replaces the margin figures.*
- **Analysis amendment (2026-10-30, before Freeze 1).** *Extended by amendment 2 (bootstrap engine v2); the
  statements below about the primary BA apply to BA as the first secondary metric.*
  - *Stratum support.* The bootstrap stratifies homology clusters by the rarest EC class they contain. If a stratum of
    EC classes 1–6 holds a single cluster, every resample draws it unchanged, so between-family uncertainty cannot be
    estimated. That population is then reported descriptively (point estimates, no bounds), and its superiority
    inference is marked unavailable. A class-7-only stratum does not trigger this for the primary BA, which reads only
    proteins of EC classes 1–6. No class or cluster is removed to recover eligibility. Class and cluster support and
    the stratum sizes are reported for every population. Two clusters only avoid this degeneracy; they do not
    establish adequate power.
  - *Secondary metrics (clarification, 2026-11-01).* The gate is applied per metric. Class-7 proteins enter accuracy
    directly, and they enter macro-F1 over EC classes 1–6 as false positives when a compared system assigns them to an
    EC class 1–6 present in the population. A single-cluster class-7-only stratum therefore also makes the accuracy
    comparison descriptive, and the macro-F1 comparison when such false positives occur. Those hypotheses keep their
    place in the secondary Holm family, are marked unavailable and keep their point estimates. The primary BA exemption
    is unchanged.
  - *Class scope.* If a population lacks some of EC classes 1–6, its BA averages the classes present (the k-class
    fallback). For both P1 and P2, the decision sentence, headings and summaries then name the classes evaluated
    and state that the six-class endpoint was not evaluated. A narrower result is never presented as meeting the
    six-class endpoint.
  - *Multiplicity.* An unavailable population stays in the two-hypothesis Holm family. The other population is
    still tested at 0.025, never at 0.05.

## Amendment 2 (2026-11-04, before any G2 or fold-0 data)

The maintainer accepted these decisions on 2026-11-04, before any CLEAN embedding farm, F6 sequence-mean cache,
G2 measurement, model prediction, score, selection record, or CLEAN inner fit or export existed; the certified
views' label-free ESMC chain-mean caches predate the amendment. The [amendment record](amendment2/amendment_draft.md) owns the rule text,
the maintainer's confirmations and the declared deviations. It is a review draft until its canonical record is
written, logged and published; until then Freeze 1, Freeze 1b and the S1 amendment stay in force. Superseded rules
stay replayable on their original records.

- **Selection (A2-D1).** Each fold selects, among F2, F3, F4, F5 and F6, the family with the highest pooled inner
  seed-mean score on the primary metric, with each family's learning rate chosen the same way. A candidate is
  ineligible if it has zero correct predictions in an EC class 1–6 where F2 has at least one. Within 0.5 points of
  the maximum the first of F2, F5, F3, F4, F6 wins. There is no significance test, and the choice supports no
  family-superiority statement. F1 (fold 0 only), C-G and the full-data whole-protein control C-GF are reported and
  never selectable. This is a declared deviation from `Plan.md:1116-1122` and `Plan.md:347-356` (promotion only
  after fold-level paired confidence intervals; raw validation deltas are not sufficient promotion evidence),
  which the maintainer explicitly requested for this campaign.
- **Primary metric (A2-D2).** Protein-level accuracy over the proteins of EC classes 1–6; class-7 proteins are left
  out of the population and predictions into class 7 count wrong. BA over EC classes 1–6 is the first secondary
  metric. Populations, the paired homology-cluster bootstrap, the intersection-union test against both R1 arms and
  Holm across B1-all and B1-NH30 are unchanged. Pre-data rationale (synthetic, assumption-driven
  [power simulation](amendment2/power_simulation.md)): under per-class-uniform gains accuracy needs about 0.6–0.65
  times the BA margin (80%-power margins over the R1 seed mean of about 6.4 points on B1-all and 8.8 on B1-NH30,
  against 10.3 and about 13 for BA), and the frozen BA test was anti-conservative (any-claim rate 0.068–0.076 against
  0.05) while accuracy was about nominal. The endpoint is also the estimand: accuracy weights the large EC classes, so
  a gain confined to EC classes 5–6 would be visible only in BA. BA's bootstrap under-disperses when rare-class
  clusters are few, so the small-stratum correction below applies to it too and its effect is reported.
- **Candidate F6 (A2-D3).** F5 whose whole-protein ESMC branch is first trained on the fold's full official training
  file (single-EC-class enzymes, EC class = first EC digit), then frozen. For inner fits the inner-validation
  proteins, their one-hop homologs, identical sequences, case aliases and shared-donor proteins are held out of that
  pretraining. This reverses U3 for this branch only: non-metalloenzymes serve only as whole-protein pretraining data.
- **Search and training (A2-D4).** S2 keeps only the L9 logit adjustment (no fits); there is no S3. Every fit uses
  30 epochs. The GVP families (F1, F3, F4) use raw-distance RBFs.
- **Futility (A2-D5).** After fold-0 S1 and S2, a pre-declared projection from the inner predictions of the selected
  recipe and CLEAN R1-inner estimates the probability that the final claim succeeds. Folds 1–4 run only if it is at
  least 0.20; otherwise the campaign closes as "not shown to outperform", with no test protein scored. The Freeze-1
  −3-point gap computation becomes descriptive.
- **Statistics (bootstrap engine v2).** Every class not guaranteed in every draw is gated; strata of two or more
  clusters resample k−1 of k clusters (Rao-Wu); one-cluster strata are merged for the primary metric only; a
  co-primary that cannot be tested is "unavailable" (p = 1, kept in its Holm slot) instead of stopping the scoring
  pass.
- **CLEAN checks.** Training-only sanity anchors for every CLEAN fit and a held-out anchor for the fold-0 inner fits.
  A failure means "comparator diagnosis required": defects are fixed and the same recipe refit, never retuned. The
  held-out record has one status per arm; under choice C6b an R1-inner arm recorded as "proceed, disclosed" makes
  both co-primaries unavailable (and futility then stops), an R2-inner arm only the R2 hypotheses. Tie counts and a
  descriptive 1/m tie rescore are reported, and SupCon-Hard (not run) is disclosed.
- **Claim scope.** Every decision sentence names the conservative metalloenzyme subset of the official split30
  benchmark (AlphaFold models with AlphaFill-transplanted, MAHOMES-catalytic metal sites; EC class = first EC digit).
- **Changes after fold-0 data.** Fold-0 development exposes the curated test proteins of folds 1–4. Afterwards only
  execution changes, narrowly bounded analysis completions and demonstrated-defect amendments are admissible; a
  defect fix that changes fold 1–4 behaviour labels those results "development-exposed". A futility stop is never
  reversed.
- **Execution (A2-D6 to A2-D8).** Runpod only, G2 and fold 0 in one pod session, CLEAN fits and embeddings on the pod
  GPU with TF32 off, a pre-approved fold-0 ceiling and a time-box; declared with execution v5.
- **Fits.** 117 DeepMzyme fits in fold 0 and 101 in each of folds 1–4 (521 in all, against 745–905); a futility stop
  spends 112. CLEAN keeps its 63 fits.

## Test access, disclosures and licence

- One scoring pass after every model and analysis script is frozen and hashed. Before it, no curated test file or
  curated test membership is read and no label is used in a test role. The identity audit after Freeze 1 queries each
  fold's complement (proteins outside its training file, all present in other folds' training files); for fold 0 these
  include its metallo test proteins. Earlier incidental exposures: [test-use ledger](../../DATASETS.md#test-use-ledger).
- Design-time label exposure: each fold's test proteins are labelled training proteins of the other four folds; the
  historical CARE development set behind the LR evidence and the family priors contains at least 48 of the 118 fold-0
  metallo test proteins; the planning pools (1,188 and 1,291 proteins) mix fold-test proteins. ESM-1b and ESMC were
  pretrained on UniRef.
- Disclosed with every result: the official tests are closed to full EC numbers seen in each fold's training (upstream
  removed the others), which favours CLEAN's nearest-centre design; metal sites are computational (AlphaFill/MAHOMES);
  DeepMzyme trains only on metalloenzymes; the cohort's fold 1–4 screening gap; CLEAN is not re-tuned here.
  *Amendment 2: F6's whole-protein branch is the exception to the metalloenzyme-only statement and to the screening
  gap; fold-0 development exposes the curated test proteins of folds 1–4; the endpoint caveat and the claim scope
  above are disclosed with every result.*
- CLEAN is used under its University of Illinois research-use licence (non-profit research). CLEAN code, patches and
  CLEAN weights are never distributed, only privately backed up. The repository holds the maintainer's own runner,
  setup steps, a plain-language description of each local change, and predictions. Every report carries the licence's
  acknowledgement (§2.7.1) and is notified to the University of Illinois (§2.7.2).
