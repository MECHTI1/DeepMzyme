# PMM ion metal v3 — continuation handoff (written 2026-10-08 12:17 UTC; step E2 amendment added ~13:00 UTC, its review saved ~13:20 UTC and resolved ~17:00 UTC with the E2 specification frozen; fallback leftovers removed and step F draft corrected ~18:40 UTC; step E completed and the deferred review item added 2026-10-11 ~13:15 UTC, log v3-031; step E2 run and completed 2026-10-11/12, log v3-034–v3-037; the section 10 review completed and the E2 tooling built, checked and its manifest and ceiling recorded 2026-10-11 ~14:30 UTC, log v3-032/v3-033; the pre-step-F review, its recommendation (section 11, PENDING) and step F draft v3 added 2026-10-12 ~09:30 UTC, log v3-038; the user chose the comparison and its CPU preparation was completed 2026-10-12 ~10:30 UTC, log v3-039; it ran and was not adopted 2026-10-12 11:19 UTC, log v3-040/v3-041; the user chose to finalize with the 50-epoch selection and the step F tooling was prepared and CPU-checked 2026-10-12, log v3-042, section 12, and corrected in one CPU round the same day, log v3-043)

Self-contained handoff for a new Claude Code chat. It depends on no earlier
conversation and on no temporary file; every path below is durable (git,
`/media/mechti/Data1`, `~/deepmzyme-vm`). Newest state first; the
[log](log.md) holds the dated history and [STATUS](../../../EXPERIMENT_STATUS.md)
the current authority.

## Ready-to-paste continuation prompt

```text
Continue the DeepMzyme PMM ion metal v3 campaign from the verified stopping point recorded in
docs/campaigns/pmm_ion_metal_v3/handoff.md. STEP F IS COMPLETE (log v3-045 to v3-048): the selection
gvp_late_fusion__four_class__baseline (50 epochs, seed 42, terminal checkpoint) was refit on the full non-test cohort
(checkpoint a00078466cf6…), Freeze 1 (54a724be…) and Freeze 2 (949bc899…) were recorded, the label-blind preparation
and the test ESMC ran on the workstation CPU, and the single prediction pass (claim d097add2…, predictions cd8231d9…)
and the frozen report (report.json 240ca831…) were produced: full-test common-four BA 67.55% [62.93, 71.65], label
both_results_secondary, all disclosures. The exact Zenodo PinMyMetal test is now evaluated once; no further test use.

Work only in the worktree /media/mechti/Data1/DeepMzyme_worktrees/v3 (branch v3-step-a; commits 0e845d8 and 7bd7884
are local and not pushed; the v3-046 to v3-048 records are staged, not committed). If /media/mechti/Data1 is missing:
`udisksctl mount -b /dev/sda1`. Python: /home/mechti/miniconda3/envs/DeepMzyme/bin/python.

Read: AGENTS.md; EXPERIMENT_STATUS.md; docs/campaigns/pmm_ion_metal_v3/README.md; log.md v3-048 back to v3-042.

Open user decisions (never infer them): closing the v3 campaign (closure procedure in AGENTS: Status closed, git mv to
docs/archive/campaigns, one PARAMETER_FINDINGS line, STATUS); the commit of the staged v3-046 to v3-048 records and
the push of the branch; retiring or keeping the stopped VM disk ($0.55/day); the Only-ESMC fairness arm and the
cost-gated augmentations (never run). No GPU start, retraining or further test use without the user.
```

## 1. Where to work and what to read

| Item | Value |
|---|---|
| Worktree | `/media/mechti/Data1/DeepMzyme_worktrees/v3` (branch `v3-step-a`; HEAD = `origin/v3-step-a`, see `git log -1`) — not `/home/mechti/PycharmProjects/DeepMzyme` (older state) |
| Python | `/home/mechti/miniconda3/envs/DeepMzyme/bin/python` (verify with `-c "import sys; print(sys.executable)"`); local CPU runs capped with `systemd-run --user --scope -p CPUQuota=200% -p MemoryMax=3G` |
| Campaign | `pmm_ion_metal_v3`; docs `docs/campaigns/pmm_ion_metal_v3/` (README, plan.md, assessment_spec.md, log.md, this file); data `/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v3` (the Data1 drive is `/dev/sda1`, label `Data`; after a reboot mount it with `udisksctl mount -b /dev/sda1`) |
| Reading order | AGENTS.md → EXPERIMENT_STATUS.md → docs/campaigns/pmm_ion_metal_v3/README.md → handoff.md sections 12, 11, 3, 5 and 8 → log.md v3-044 back to v3-031 → plan.md step F (and the pre-step-F comparison note) → step_f_plan_draft.md (F3, F6, F7 and the implementation notes) and step_f_spec.json → docs/DATASETS.md test-use ledger → docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md "v3 step F: Stage 6B refit and the one test pass" → .agents/skills/gpu-use-skill/SKILL.md → ~/deepmzyme-vm/AGENTS.md and config.env (before any GPU action) |
| Launchers | `pmm_v3_step_d.py` (steps D and E: units, gates, status, launch, wait, pull, recover-lane, archive-failed, assess, record-cost-gate, record-e-gate, evidence); `pmm_v3_step_e2.py` (step E2 only: units, gates, build-manifest, record-ceiling, record-authorization, spend, status, launch, wait, pull, recover-lane, archive-failed, evidence) and `pmm_v3_e2_assessment.py` (workstation E2 assessment from Data1); `pmm_v3_step_c.py`/`pmm_v3_step_b.py` (shared lane logic); never edit `pmm_v3_campaign.py` or `run_pmm_v3_campaign.py` (hash-frozen by the extension record), and not the E2 or reused modules pinned by the E2 manifest; step F: `pmm_v3_step_f.py` (workstation: F1 record, manifest, ceiling, authorization, deploy, the lane logic, `refit-receipt`, the freeze records) and `run_pmm_v3_step_f.py` (VM refit; workstation-CPU `prepare-test`, `test-embeddings`, `predict`), with `pmm_v3_test_prep.py` and `pmm_v3_step_f_report.py` (section 12); the 10-epoch comparison's `pmm_v3_sched10*.py` (done, not adopted) |
| VM code dir | `/home/mechti/projects/DeepMzyme_v3_ext1` (extension bundle 43d26ff); campaign root `/home/mechti/deepmzyme_runs/pmm_ion_metal_v3` (lanes/lane0..2, step_d/launch, step_e/launch, assessments) |

## 2. Approved objective, pipeline and frozen settings

- **Objective (user decision 2026-10-03, before any run):** separate `four_class`, `five_class` and `six_class`
  arms for Only-ESMC, Only-GVP and graph-level late fusion on PMM ion-level metal data, all evaluated on
  common-four (Mn, Cu, Zn, Class VIII = Fe+Co+Ni) with native five/six metrics kept; the objective comparison
  is the neutral test of Plan.md section 2; one final refit and one evaluation of PMM's test set (step F).
- **Pipeline (plan.md, approved):** A CPU preparation and new strict folds → B speed check → C fold-0
  baselines with regression gate → D one-fold improvement screen (Round A → R → B → one combination per
  family) → E five-fold confirmation (seed 42: 36 neutral-test fits on folds 1–4 + up to 8 improvement fits)
  → E2 (planned 2026-10-08, log v3-025: the eight remaining step D passes singly on folds 1–4, 36 fits, seed 42;
  assessment rules frozen 2026-10-08, e2_assessment_spec.md, log v3-028) → F Stage 6 selection (the final E2 selection, or the
  original A4 selection if the user explicitly cancels E2), Stage 6B full refit, one test pass. Steps A–E2 and the
  pre-step-F comparison are complete (log v3-031, v3-037, v3-041); step F is complete (log v3-045 to v3-048).
- **Frozen identities:** fold set `v3-seqid90-s42-b2` (log v3-005); A4 assessment specification
  `assessment_spec.md` (frozen 2026-10-05); first bundle `20d5b06`, extension bundle `43d26ff`
  (`v3-ext1-round-r`, record SHA-256 `ade3f022…` on the VM campaign root; runner files hash-frozen);
  step-B execution setting FP32, 3 lanes, AMP off, `load_workers` 4; replay policy `pmm-v3-replay-1`
  (independent replay of every completed fit); test-row rule v3-012 (score all reconstructable rows, flag
  symmetry-LINK rows, report "N of 1,488 scored").
- **Training settings fixed for every fit:** 50-epoch cosine schedule, terminal-epoch checkpoint (no early
  stopping; the early-stopping and train/validation-gap audits are descriptive only), fold 0 / seeds 42 and 43 for
  D, seed 42 for E and F, class weights, loss and learning rates of the baseline; each D candidate changes exactly
  one setting from its control.
- **Gates:** D screen = mean Δ over seeds 42/43 ≥ 1.5 points, Δ > 0 for both seeds, no Mn/Cu/Zn/Class VIII mean
  recall change below −3 points; a combination is adopted only if it passes and beats the best single candidate;
  E neutral test = five/six vs four per family on five folds (paired t interval, Bonferroni over six; tie keeps
  four; within 0.2 points five wins); E improvement = final recipe vs baseline on folds 1–4 only (t interval with
  3 df; "improvement interval-supported" only if both lower bounds are above zero; fold 0 reported separately as
  the development fold); Stage 6 control = Only-ESMC baseline four_class, kept unless a candidate is eligible and
  interval-supported on all five folds; refit seed 42, same checkpoint rule, no calibration, no ensemble.
- **Selection and test access:** validation-only selection; no held-out use before step F. Approved in step F draft
  v3 (log v3-038, approved v3-045) and executed (v3-047, v3-048): two dated freezes in docs/DATASETS.md's test-use ledger. Before any test-side read:
  the primary report, the test-row and clean-subset rules, the preparation code hashes, the refit checkpoint, seed
  and epoch count, ensemble (none), calibration (none) and the disclosures. After the label-blind preparation and
  before any prediction or label read: the derived memberships by hash. This replaces the earlier wording "rules with
  membership hashes before any test access", which could not be met literally. The step F tooling was prepared and
  CPU-checked on 2026-10-12 under a preparation-only authorization and corrected in one CPU round (log v3-042,
  v3-043; section 12); it ran under the approvals of log v3-045, v3-047 and v3-048.
- **Step E2 amendment (planned 2026-10-08, log v3-025; tooling built and CPU-checked 2026-10-11, log v3-033; completed 2026-10-12, log v3-037: no candidate replaced the selection):** after the original E assessment, the
  eight step D passes not chosen as final recipes (Only-GVP `wd10`, `sitecountsangles`; late fusion `invsqrtw`,
  `meanagg`, `gvpaux03`, `resdrop01`, `esmdrop02`, `wd10`) each on folds 1–4, seed 42, `four_class`, 50-epoch cosine,
  terminal checkpoint, against the reused step E baselines (`sitenone` × 4 as the geometry control): 36 fits, 17
  declared comparisons (unadjusted A4 intervals decide; Bonferroni-adjusted ones are sensitivity evidence), the
  Option B replacement rule frozen 2026-10-08 in e2_assessment_spec.md/.json (A4 unchanged; log v3-028); the E2
  assessor must accept a four-, five- or six-class original selection (the existing `compare` assumes a four-class
  control). Exploratory development-validation evidence, proposed after fold 1–4 results were seen; passing the
  rule is not independently confirmed superiority. Step F requires E2 completion unless the user cancels E2. Rules:
  plan.md step E2 and the E2 specification; tooling gaps: the playbook's E2 section.
- **Persistence contract:** every fit → run_status written → host-pull manifest (SHA-256 per file) → worker exit
  code → host `pull` verifies and uploads the acknowledgment (lane closed until then) → `evidence` copies the step
  evidence to Data1 with SHA-256 checked on both ends; `assess` keeps a checked copy of each assessment. A unit is
  "complete" only when all four happened: training completed, worker exited 0, artifacts persisted, host acknowledged.

## 3. Work state (verified 2026-10-11 13:05 UTC; rechecked 2026-10-12 after a reboot, log v3-043)

**Step D: complete (final assessment `D-B_20261007T201334Z`, status `final`).** Every listed unit below is complete
in all four senses (training completed at epoch 50, worker exit code 0, artifacts persisted by host pull with
SHA-256 manifests, host acknowledgment uploaded) and replayed under `pmm-v3-replay-1`; the pulled copies are in
`$DATA/durable/lane*/runs/<unit>/`. Nothing is running; no lane awaits a pull; no unit failed in steps D or E;
the VM is TERMINATED (verified 2026-10-11 13:03 UTC and again 2026-10-12 16:16 UTC).

| Round | Units | State |
|---|---|---|
| Controls | step C seed-42 baselines (3 families × 3 targets, 9 runs, `fold0`) + Round A seed-43 four-class baselines (`only_gvp__four_class__baseline__fold0__seed43`, `gvp_late_fusion__four_class__baseline__fold0__seed43`) | complete |
| D-A (16) | `{only_gvp,gvp_late_fusion}__four_class__{meanagg,resdrop01}__fold0__seed{42,43}`, `gvp_late_fusion__four_class__{structlr,gvpaux03,esmdrop02}__fold0__seed{42,43}` + the two seed-43 controls | complete; passed: meanagg (both), fusion resdrop01/gvpaux03/esmdrop02 |
| D-R (26) | `{only_gvp,gvp_late_fusion}__four_class__{wd001,wd01,wd10,headdrop01,headdrop03,resdrop02}__fold0__seed{42,43}`, `gvp_late_fusion__four_class__esmdrop04__fold0__seed{42,43}` | complete; passed: wd10 (both), fusion headdrop03 |
| D-B (14 of 18) | `{only_gvp,gvp_late_fusion}__four_class__{sitenone,sitecountsangles,invsqrtw}__fold0__seed{42,43}`, `only_gvp__four_class__vecnorm__fold0__seed{42,43}` | complete; passed: fusion invsqrtw, Only-GVP sitecountsangles (vs sitenone) |
| D-B cost-gated (4) | `only_gvp__four_class__{posnoise01,outerdrop01}__fold0__seed{42,43}` | never run: "not tested (cost)" (user OK never given) |
| D combination (4) | `gvp_late_fusion__four_class__combo-gvpaux03+headdrop03+invsqrtw+meanagg+resdrop01+wd10__fold0__seed{42,43}` (+2.82/+3.69, mean +3.25 < headdrop03 +3.58), `only_gvp__four_class__combo-meanagg+sitecountsangles+wd10__fold0__seed{42,43}` (−1.96/+6.29, mean +2.17, fails the both-seeds rule) | complete; **final recipes: late fusion `headdrop03`, Only-GVP `meanagg`, Only-ESMC baseline** |

**Step E: complete, 44 of 44 units** (sessions 6–9 of 2026-10-08 and 2026-10-11, log v3-023/v3-024/v3-031; gate
recorded: final-test label `both_results_secondary`, `step_e_evidence/step_e_gate.json`, log v3-022). Final A4
assessment `step_e_evidence/assessments/E_20261011T130128Z/assessment.json` (SHA-256 `d6d27c0c2bf4668cae2f03170faf06e3ab99e90654a0279f171260bab2681a45`;
`held_out_test_accessed` false), evidence `step_e_evidence/evidence_20261011T130145Z` (786 files). Saved unchanged.

| Group | Units (seed 42, folds 1–4) | State |
|---|---|---|
| E neutral, four/five/six_class (36) | `{only_esm,only_gvp,gvp_late_fusion}__{four,five,six}_class__baseline__fold{1,2,3,4}__seed42` | complete (all 36) |
| E improvement (8) | `gvp_late_fusion__four_class__headdrop03__fold{1,2,3,4}__seed42`, `only_gvp__four_class__meanagg__fold{1,2,3,4}__seed42` | complete (all 8) |

Results (five folds; details and intervals in log v3-031): **neutral test answered — no family shows a supported
gain from five- or six-class training on common-four BA; the tie keeps `four_class` for all three families**;
`headdrop03` "no positive mean gain on folds 1–4", `meanagg` "positive mean gain on folds 1–4, not
interval-supported" — neither is an improvement. **Stage 6 selection (the original step E selection):
`gvp_late_fusion__four_class__baseline`**, mean common-four BA 74.28 (Mn 75.6, Cu 73.4, Zn 81.3, Class VIII 66.8),
"better" than the Only-ESMC four-class control (72.33) on all five folds (mean +1.95; unadjusted bootstrap
[+1.09, +2.92], t [+0.50, +3.39]); refit seed 42, terminal checkpoint, no calibration, no ensemble. Development-
validation evidence. Native rare classes stay weak in five/six-class cells (six-class Co 2.8–5.3, Ni 25.3–31.2).

Fold 0 of every E contrast reuses the step C / step D runs (never rerun). **Step F (log v3-045 to v3-048): complete: the
refit (checkpoint `a0007846…`), both freezes, the single test pass and the report (full-test BA 67.55%).** Earlier: tooling prepared
and CPU-checked 2026-10-12 (log v3-042) and corrected in one CPU round (log v3-043; section 12; no record written,
nothing run on test data); the execution draft is `step_f_plan_draft.md` (draft v3 with implementation notes, log
v3-038/v3-042/v3-043; approved in v3-045; v2 of log v3-030 superseded). It
takes the final E2 selection (A4 only after a dated E2 cancellation) and a schedule decision (50 epochs unless the
section 11 comparison is approved, run and adopted), and makes a validation-free terminal refit. Test access is
gated by two ledger freezes. The 08:45 UTC session draft on the data root is kept unchanged as history.
**Pre-step-F review (2026-10-12, log v3-038): done. The user chose the 10-epoch comparison (log v3-039); readiness
and authorization log v3-040; it ran 4 of 4 fits and was NOT adopted (log v3-041): S1 mean +1.99, bootstrap
[−0.39, +5.90], t [−4.28, +8.26]; the 50-epoch selection stands (E_sel = 50).**
**Step E2: complete, 36 of 36 fits** (log v3-035–v3-037; sessions 10–12 on 2026-10-11/12, 8 h 51 min, $7.80
gross; two first-attempt replay-tolerance failures rerun once and passed). Final E2 assessment
`step_e2_evidence/assessments/E2_20261012T053919Z/assessment.json` (SHA-256 `6e804796…`; workstation, from the Data1
copies; 100 units verified; A4 parity exact): **no candidate eligible; the original selection
`gvp_late_fusion__four_class__baseline` is retained**. Only-GVP `wd10` is an interval-supported Only-GVP improvement
(+3.59) but not better than the selection; late fusion `esmdrop02` +0.53 not interval-supported; the other fusion
candidates show no positive mean gain; `sitecountsangles` is blocked by recall gates. Latest evidence
`step_e2_evidence/evidence_20261012T053726Z`; VM TERMINATED 05:39 UTC. Records: manifest `37c2ff05…`, ceiling
`a4c31da1…`, authorization (log v3-034).

## 4. User decisions and authorization boundaries

| Date (UTC) | Decision (actual, by the user) | Where recorded |
|---|---|---|
| 2026-10-03 | Objectives four/five/six for all three families; v3 campaign opened; PMM core v2 closed at fold 0 | README, log v3-001…003 |
| 2026-10-05/06 | A4 rules frozen; checkpoint rule terminal epoch; $40 gross ceiling incl. storage; advance GPU authorization for the approved plan through F (no routine start prompts); test-row rule | log v3-003, v3-012, v3-014 |
| 2026-10-06 | Regularization amendment (Round R) and its implementation; daily controller caps raised to 12 h / $18; bounded fallback pass authorized (failed; leftovers kept pending the user's cleanup decision) | log v3-017…v3-019 |
| 2026-10-07 | Only-ESMC fairness assessment requested as a proposal only (training awaits approval) | log v3-021 |
| 2026-10-08 morning | Next GPU start confirmed; up to three capped sessions with closeout by 17:30 UTC, then (PC closing) one session only; commit and push at every closeout (today); two read-only reviewers; a deferred read-only review of a 10-epoch schedule amendment after the next closeout (no implementation or spending); the arrangement is for 2026-10-08 only | log v3-024 |
| 2026-10-08 midday | Step E2 amendment recorded as planned, not implemented: the eight remaining step D passes tested separately on folds 1–4 (36 fits, seed 42), proposed selection rules kept separate from A4, E2 completion required before step F unless the user cancels E2, a $55 gross total planning ceiling for the expanded campaign; documentation only, no GPU start authorized, execution controls unchanged | log v3-025, plan.md step E2 |
| 2026-10-08 afternoon | E2 review resolved: Option B with clarifications (unadjusted A4 verdict rule, 0.2-point minimum mean gain, both comparators, recall gates by comparator; adjusted intervals as sensitivity only), two separate conclusions per candidate, no extra seed, exploratory label with no claim of independent protection; E2 specification frozen; v3-024/v3-027 corrected; no GPU start authorized | log v3-028, e2_assessment_spec.md |
| 2026-10-08 evening | Priority: the strongest defensible model; accepted order kept (20 original E fits → A4 assessment → read-only schedule/error review → E2 CPU → E2 GPU request → F); controller abandonment addition and deletion of exactly the failed-fallback disks and snapshot authorized and executed; step F draft corrected; no GPU start authorized | log v3-030 |
| 2026-10-11 | Next GPU start confirmed: both sessions that day for the 20 remaining E fits (stockout: spaced same-VM retries only); commit and push at every closeout of that window; two messages adding the deferred read-only review item of section 10 (PENDING; prerequisite met at 13:03 UTC); after the review, the E2 CPU implementation and checks continue without another approval stop and the findings, refreshed cost and remaining effort are presented at the E2 GPU-start gate | log v3-031 |
| 2026-10-11 afternoon | Continue: complete the section 10 review, then the E2 CPU implementation and checks without another approval stop, then present at the E2 GPU-start gate; E2 unchanged; no GPU start, experiment, budget increase, refit or test access | log v3-032, v3-033 |
| 2026-10-11 evening | E2 GPU sessions authorized; the $55 ceiling kept; push of the three pending commits; no optional items | log v3-034 |
| 2026-10-12 | Pre-step-F CPU review: improved-GVP complementarity against the selected fusion, then at most one recommendation among the 10-epoch schedule comparison, one selective GVP weight-decay recipe or step F directly, with a concrete proposal, step F changes and the recommendation recorded here; no training, inference, refit, test access, GPU work, change to frozen E/E2 records or budget increase | log v3-038, section 11 |
| 2026-10-12 ~10:00 | The 4-fit 10-epoch comparison of section 11 is the next experiment, exactly as specified (no other candidate or sweep); CPU-only preparation of its isolated runner extension and checks authorized; record the reconciliation of an independent report, separating measured observations from causal hypotheses and deferring the other proposals without calling them disproven; no GPU start, training, inference, step F, held-out access, commit or push; step F approval pending | log v3-039 |
| 2026-10-12 ~10:35 | Readiness first (synthetic CPU fit and smoke checks explicitly authorized; command comparisons do not replace them); if they pass, one GPU session for the comparison exactly as specified (2 h stop, existing caps, $55 ceiling, same-VM stockout retries ≤ 12 at 5 min, no fallback, one unchanged rerun); evidence, stop, assessment and documentation; no other experiment, step F, held-out test, commit or push | log v3-040, v3-041 |
| 2026-10-12 ~11:40 | Finalize v3 with the selected 50-epoch model; keep the 10-epoch comparison recorded as not adopted under its frozen gates; leave the other optimization hypotheses deferred and untested, not disproven; implement the step F tooling and its CPU readiness checks on synthetic or training-side data only (tiny synthetic CPU training, replay and persistence checks explicitly authorized); follow draft v3's two freezes, bind the selection and the 50-epoch terminal rule, keep `both_results_secondary` and the prior-exposure disclosures, prepare the freeze-record templates and make the clean-subset rules explicit; preparation only (the earlier step F GPU authorization is not activated); no GPU start, real refit, test-side read, test embedding, test inference, commit or push | log v3-042, section 12 |
| 2026-10-12 ~16:45 | After the post-reboot verification: one focused CPU-only correction round using the reviewed findings (campaign-wide one-shot prediction protection, freeze-record validation and code-hash bindings, an explicit Freeze 1 amendment procedure, a label-independent `ion_key` with uniqueness and row-accounting checks); keep the A1 clean-subset policy, fix missed identical chains of at least 30 residues (including the non-standard-residue case), keep and disclose the short-chain limitation (no short-chain exclusion); use the workstation-CPU test ESMC route and state its requirements; align documentation, approval sequence, costs and readiness evidence (original results preserved, corrections by a saved script); targeted regression tests and the authorized synthetic CPU fits; stage for review; no GPU start, real refit, test-data access, real selection binding, freeze record, commit or push | log v3-043 |
| 2026-10-12 ~18:00 | The CPU ESMC compatibility check on three training chains fixed in advance (weights download authorized), smallest correction if needed, documents staged | log v3-044 |
| 2026-10-12 ~18:30 | Step F plan approved as the design (clean-subset rule, CPU ESMC route, two freezes, amendment procedure, both_results_secondary disclosures); commit of v3-038 to v3-044 plus the approval documents (no push); bind the selection, manifest, ceiling and authorization; one GPU session (`--hours 2`, unchanged safeguards, $55 ceiling, one unchanged rerun only if it fits; no second start, extension, migration or fallback); verify, persist, stop; stop before Freeze 1 and prepare its approval package | log v3-045 |
| 2026-10-07 evening | Final-test label `both_results_secondary`; continue the approved plan until 03:38 UTC 2026-10-08 at the latest (eight hours maximum, not a target), step E after midnight UTC if gates pass, caps and ceiling unchanged; fairness arm kept separate for the user's decision; audit of the scratch drivers delivered before any correction or driver reuse; closeout commits pushed to `origin/v3-step-a`; a complete handoff for a new chat | log v3-022 |

**Authorization boundaries:** the operative ceiling is $55 gross, storage included (log v3-025, v3-034; about $40.8
spent at 2026-10-12 17:30 UTC). The advance step F GPU authorization of v3-014 was given within the old $40 ceiling,
which spending now exceeds, and it is not activated (log v3-042): the refit's GPU start needs the user's explicit,
logged authorization (step_f_plan_draft.md F7 item 4), recorded with `pmm_v3_step_f.py record-authorization`.
Controller caps: 4 h / $6 per session, 12 h / $18 per UTC day; one active GPU, controller
only via `~/deepmzyme-vm/bin`; every session ends with `evidence`, `vm-stop` and a verified TERMINATED. Never:
Round C, cost-gated augmentations (`posnoise01`, `outerdrop01`) without a recorded OK, `vm-fallback`, deleting cloud
resources, editing the frozen runner files, editing controller caps, pushing without the user's OK
(granted for the 2026-10-07/08 and 2026-10-11 closeouts and the v3-034 commits only), step F before its gates and plan.
**Decisions still open:** the six items of step_f_plan_draft.md F7, in order (section 12): plan approval (with the
clean-subset rule, the label-blind reads, the CPU test ESMC route, the amendment procedure and the F4 claim), the
CPU ESMC check (done, log v3-044), commit/push of the v3-038 to v3-044 work, the refit GPU start, the plan.md
alignment, then the Freeze 1, Freeze 2 and single-test-pass go-aheads;
Only-ESMC fairness arm; cost-gated augmentations. E2 is complete (its GPU-start authorization of v3-034 is used up).
The failed-fallback leftovers were deleted on 2026-10-08 under the user's authorization through `vm-fallback
--abandon` (log v3-030). Ask before any GPU start: the advance authorization of v3-014 covers approved step F work
only, not a pre-step-F experiment.

## 5. VM, spending and limits (verified 2026-10-12 16:16 UTC)

| Item | Value (`vm-status` 2026-10-12 16:16 UTC; the controller ledger) |
|---|---|
| VM | `deepmzyme-l4`, project `deepmzyme-gpu-vm`, zone `us-central1-a`, g2-standard-8 + 1× L4, **TERMINATED** (verified 2026-10-12 20:06 UTC after the Stage 6B refit session `session-20261012T185958Z-c995e05c`, 1 h 06 min, $0.97 gross, started after five stockouts; before that verified 2026-10-12 11:18:48 UTC after the 10-epoch comparison session `session-20261012T104112Z-c41caba4`, 37 min, $0.55 gross; before that 05:39:06 UTC after E2 session 12 `session-20261012T025638Z-baee022b`, 2 h 42 min, $2.38 gross; E2 sessions 10–12 ran 8 h 51 min, $7.80 gross); 150 GB boot disk kept (`deepmzyme-l4-from-deepmzyme-paused-20261003`) |
| Spending | 2026-10-12 20:07 UTC (log v3-046, offline `pmm_v3_step_f.py spend`): about **$41.80** after the refit session; **$13.20** left under $55; the rest of step F runs on the workstation CPU (storage only). 2026-10-12 17:30 UTC (log v3-043, an estimate; no VM use since 11:18:47 UTC): about **$40.8**; to finish about **$42.8–48.2** (the refit $0.9–1.9, storage for 2–10 days; the test side on the workstation CPU costs no VM time). 2026-10-12 11:20 UTC (log v3-041): about **$40.65** after the comparison session (37 min, $0.55 gross by the ledger); to finish about **$42.6–48.1** (step F 50-epoch path plus storage). 10:24 UTC (log v3-039, offline `pmm_v3_sched10.py spend`): about **$40.09**; the comparison's launch control projects $46.09 with its $6 reserve against $55; the comparison adds about 0.8–1.1 VM h ($0.7–0.95, up to $1.4 with a rerun) plus 1–2 storage days. 2026-10-12 09:02 UTC (log v3-038; `vm-status` TERMINATED, today 5 h 33 min / $6.01): about **$40.06 spent**; to finish about **$42.1–47.5** straight to step F or **$42.8–49.7** with the section 11 comparison (A 0.6–1.0 VM h, $0.5–0.9, plus 1–2 storage days). After E2 (log v3-037, 2026-10-12 05:40 UTC): about **$39.98 spent**; to finish (step F 1.0–2.25 VM h, $0.9–1.9, plus storage $0.55/day) about **$42.0–47.4**, within the recorded $55. Before E2: refreshed in log v3-033 (14:10 UTC): about $32.00 spent; E2 8.3–9.4 VM h central (stress 10.1), $7.1–8.9; E → E2 → F about **$41.7–48.9**; step F alone after a dated E2 cancellation about $33.5–36.7; the **$55 ceiling is recorded in E2's launch control** (`e2_ceiling.json`, estimated charges with the running session to its hard stop plus a $6 reserve). Earlier: one dated calculation, log v3-031 (2026-10-11 13:10 UTC; estimated charges, disk counted once): running B–E $26.96 (31.40 h × $0.8586/h); campaign disk $2.76; archive snapshot $0.32; leftovers final $1.94; **$31.98 spent, $8.02 of the $40 gross ceiling left**; forecast E → E2 → F $42.4–48.8 including storage (E2 8.6–9.6 h, F 1.0–2.25 h VM wall time; storage $0.55/day for 4–12 days), above $40 and within the user's $55 E2 planning ceiling, which must be recorded in the E2 execution controls before any E2 GPU fit (not start permission); step F alone after a dated E2 cancellation $33.9–36.7 |
| Retained storage | campaign disk `deepmzyme-l4-from-deepmzyme-paused-20261003` ($0.49/day) and archive snapshot `deepmzyme-paused-20261003` ($0.06/day): $0.55/day in estimated charges; the controller reserves $0.75/day (a guard, not a charge). The failed-fallback disks and snapshot were deleted on 2026-10-08 (log v3-030; receipt archived as `state/fallback-873d08b26968.abandoned.json`) |
| Limits | controller caps 4 h / $6 per session, 12 h / $18 per UTC day (`~/deepmzyme-vm/config.env`; never edit without the user's instruction); 2026-10-12 used 6 h 11 min / $6.55 of running time (2026-10-11: 4 h 37 min / $5.20); a new UTC day resets the daily cap; measured 2026-10-11: about 15 Only-GVP/late-fusion fits per 4 h session (about 40 min per fit at three lanes; Only-ESMC 30–35 min) |
| Stockouts | L4 `ZONE_RESOURCE_POOL_EXHAUSTED` in us-central1-a is frequent (five on 2026-10-07, one at 00:01 and nine in a row 07:58:51–08:41:04 on 2026-10-08 by the ledger; none on 2026-10-11); the handoff rule is spaced same-VM retries only (5-minute spacing, bounded), never `vm-fallback` |

## 6. Audit of the session-5 scratch drivers (delivered 2026-10-07)

Full report: `/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v3/audits/driver_audit_20261007T193000Z/driver_audit_report.md`
(with `mock_drivers.py`, the CPU mock that reproduces every finding, and `scratch_tools/` holding both driver and
watcher versions and their logs). The drivers were workstation-side refill scripts that only called the launcher's
`status`, `pull`, `launch --no-wait` and `wait --any`; they are not repository code and were not used after 18:26 UTC
on 2026-10-07; all later launches were made directly with the launcher.

| Concern | Finding | Occurred? |
|---|---|---|
| 1 Completion vs lane ownership | Handled by the launcher (lane lock, "no exit code yet", "awaits pull" refusals; status→rc gap measured 1 s); v1 treated the temporary refusal as final (defect); nonzero rc with a completed status would be missed by both drivers (unresolved, alert-only) | Yes, 17:42 UTC (3 min idle lanes) |
| 2 Exit before the wait snapshot | v1: benign variant occurred (exit_code null via completed_units); silent no-progress loop if the unit has no status record (not occurred). v2: re-reads status, but would relaunch a unit that exited with no status and no claim (pre-admission runner refusal; launcher does not block it) | Partly (benign) |
| 3 SSH error after launch | v1 logs "nothing started" wrongly; v2 stops launching and may report "driver done" while the unit runs; persistence still safe through launcher `wait`/`evidence` | No |
| 4 Watcher matching | v1 regex needed an exact closing quote: the 17:42 refusal raised no alert until "driver done" at 17:44; v2 prefix regex matches all its messages; neither detects a driver crash or stall (harness exit notification covers crashes) | Yes (delayed alert); first watcher fired a false alarm immediately |
| 5 Malformed responses | v1 loops silently on `{"raw": …}`; v2 crashes with KeyError (loud) | No |
| 6 Recovery / closeout | Launcher path (`status`, `wait`, idempotent `pull`, `evidence`, vm-stop) reconciles without a second scheduler or duplicate launches; v2's "nothing active and nothing launchable" rule stopped while a lane awaited its pull (defect) | Yes, 18:25 UTC (3 min idle lanes); recovered manually, no duplicate, no lost persistence |

Proposed smallest corrections (NOT applied; for the user's review before any driver is reused): bounded retry of
temporary refusals ("awaits pull", "no exit code yet", "holds the lane lock"); explicit tracking of launched units
with no relaunch and `wait` by name; reconcile with `status` after an error containing "may have started"; validate
response keys and stop after N invalid/no-progress cycles; add a staleness alert (no log line for 70 min while units
run). Harm of applying them: none to submission or persistence (the launcher's guards stay); they only remove
launches and add alerts. The normal-transition mock shows three initial launches and one refill per ended unit.

## 7. Evidence locations and recovery

All under `DATA=/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v3` unless noted (verify separately
from git: the Data1 drive must be mounted; each `evidence_*` copy was SHA-256 checked on both ends when written):

| What | Where |
|---|---|
| Pulled runs (per lane; run dir = config, metrics CSVs, terminal + selected checkpoints, validation predictions, `independent_validation_replay/`, `.log`) | `$DATA/durable/lane{0,1,2}/runs/<unit>/`; host-pull manifests and acknowledgments in `$DATA/durable/lane*/persistence_receipts/` |
| Step D evidence copies (statuses, launch records, lane state, events) | `$DATA/step_d_evidence/evidence_<UTC>/` (latest D: `evidence_20261007T201431Z`, 682 files) |
| Step D assessments (checked copies; SHA-256 in the launcher output and the log) | `$DATA/step_d_evidence/assessments/D-A_*/`, `D-R_*/`, `D-B_*/assessment.json` (final: `D-B_20261007T201334Z`, SHA-256 `bba7c728…`) |
| Extension record copy, readiness and decay audits, early-stopping and train/val-gap audits | `$DATA/step_d_evidence/extension/`, `$DATA/audits/d_readiness_20261006T180939Z/`, `d_decay_audit_20261006T175933Z/`, `early_stopping_audit_20261007T111826Z/`, `train_val_gap_20261007T111847Z/` |
| Scratch-driver audit (report, mock, tools, logs, interim drafts) | `$DATA/audits/driver_audit_20261007T193000Z/` |
| Logged-epoch contrast, folds 1–4 (log v3-029; `audit_v3_epoch_contrast.py`) | `$DATA/audits/epoch_contrast_20261008T171843Z/` |
| Step E gate record (label), evidence and assessments | `$DATA/step_e_evidence/step_e_gate.json`; `$DATA/step_e_evidence/evidence_<UTC>/` (final `evidence_20261011T130145Z`, 786 files), assessments `$DATA/step_e_evidence/assessments/` (**final `E_20261011T130128Z`, SHA-256 `d6d27c0c…`**; earlier E_20261008T022125Z, E_20261008T121456Z, E_20261011T115055Z are incomplete interim copies) |
| Section 10 review (done, log v3-032) | requests: `$DATA/audits/session_notes_20261008/DEFERRED_amendment_review_request.md`; scripts and outputs: `$DATA/audits/schedule_error_review_20261011T131706Z/` (`r1_weighting` … `r6_throughput`, `verify/`) |
| 10-epoch comparison records, evidence and assessment (log v3-039–v3-041) | `$DATA/step_s10_evidence/s10_manifest.json` (`a9779c05…`), `s10_ceiling.json` (`3a6710a6…`), `s10_authorization.json` (`43465211…`), `evidence_20261012T111709Z/` (763 files), **`assessments/S10_20261012T111859Z/assessment.json` (`8fbce69b…`; not adopted)**; pulled runs `$DATA/durable/lane{0,1,2}/runs/gvp_late_fusion__four_class__sched10__fold{1..4}__seed42/`; readiness `$DATA/audits/s10_readiness_20261012T102309Z/` (`real_config_check.*`, two dry runs, `readiness_manifest.json` `e57b4a7e…`) |
| Pre-step-F review (done, log v3-038; section 11) | `$DATA/audits/pre_f_gvp_complementarity_20261012T085520Z/` (`gvp_complementarity_review.py`, `complementarity_results.json`, `complementarity_tables.md`, `verify/`, `review_manifest.json` `1fd779b5…`) |
| Step F readiness (log v3-042, v3-043) | original run `$DATA/audits/f_readiness_20261012T122926Z/` (`readiness_checks.json` `561190f8…`, preserved unchanged; its stage C pass was a recorded reassessment, not as run); an aborted rerun `f_readiness_20261012T165246Z/` (`ABORTED.txt`; not evidence); the scripted rerun on the final tools `f_readiness_20261012T170111Z/` (`step_f_readiness.py` `e73c09c5…`, `readiness_checks.json` `5b1ce5f0…`: A, B, D passed, C failed by the script's scope error); the corrected stage C rerun `f_readiness_20261012T174408Z/` (`step_f_readiness.py` `b73ece28…`, `readiness_checks.json` `7e61a339…`: C passed, combined ready true) |
| Pre-shutdown worktree backup (2026-10-12 14:32 UTC) | `$DATA/backups/v3_worktree_uncommitted_20261012T143217Z/` (staged and unstaged patches, untracked-file tarball, `SHA256SUMS`, HEAD `f5c7f89`): the v3-042 state before the v3-043 round |
| CPU ESMC check (log v3-044) | `$DATA/audits/esmc_cpu_check_20261012T180322Z/` (attempt 1, failed on the spec defect before any result; `ATTEMPT1_FAILED.txt`) and `$DATA/audits/esmc_cpu_check_20261012T180915Z/` (attempt 2: `selection.json` `e3843acb…`, `criteria.json` `d1bd7a78…`, `cpu_esmc_check.py` `89b97cc8…`, `esmc_cpu_check.json` `e9c3c3d4…`, payloads in `tool_bf16/` and `fp32/`, `*_time.txt`); readiness stage A recheck `$DATA/audits/f_readiness_20261012T181114Z/` (`b99e9c2c…`); weights in `~/.cache/huggingface/hub/models--EvolutionaryScale--esmc-600m-2024-12` |
| Step F records and the refit (log v3-045, v3-046) | `$DATA/step_f_evidence/` (`f1_selection.json` `e8a2e8cd…`, `f_manifest.json` `d37ba100…`, `f_ceiling.json` `bdb044f7…`, `f_authorization.json` `27956449…`, `evidence_20261012T200435Z/` 751 files); the pulled refit `$DATA/durable/lane0/runs/stage6b_refit__gvp_late_fusion__four_class__baseline__e50__seed42/` (terminal checkpoint `a0007846…`, `refit_reload_check.json` `45759713…`) and its status `durable/lane0/run_status_stage6b_refit_…json` (`f48e5d24…`); VM-side originals under `/home/mechti/deepmzyme_runs/pmm_ion_metal_v3/lanes/lane0/` and `step_f/launch/` |
| Step E2 records and readiness | `$DATA/step_e2_evidence/e2_manifest.json` (`37c2ff05…`; superseded first manifest kept as `e2_manifest.superseded_20261011T141045Z.json`), `e2_ceiling.json` (`a4c31da1…`), later `e2_authorization.json`, `evidence_<UTC>/`, `assessments/E2_<UTC>/`; readiness dry run `$DATA/audits/e2_readiness_20261011T142030Z/assessment.json` (`ea0c6dbd…`; the earlier `…T141052Z` run used the superseded manifest) |
| Bundles (source + extension) and folds | `$DATA/bundles/20d5b06/`, `$DATA/bundles/43d26ff/`; `$DATA/folds/` |
| Steps B and C evidence | `$DATA/step_b_evidence/`, `$DATA/step_c_evidence/` |
| VM-side originals (kept on the stopped 150 GB disk) | `/home/mechti/deepmzyme_runs/pmm_ion_metal_v3/{lanes,step_d,step_e,assessments,campaign_extension.json}`; code `/home/mechti/projects/DeepMzyme_v3_ext1`; caches `/home/mechti/deepmzyme_cache/{parse,esm,ring}` |
| Controller state and ledger | `~/deepmzyme-vm/state/usage_ledger.jsonl`, `current_session.json`, `fallback-873d08b26968.abandoned.json` (abandoned 2026-10-08; no active fallback receipt), `config.env` |
| Fallback cleanup records | `$DATA/audits/fallback_abandon_20261008T181921Z/` (evidence, dry-run, confirmed run, post-verification) |
| Git | worktree `/media/mechti/Data1/DeepMzyme_worktrees/v3`, branch `v3-step-a`, remote `origin` (same repository as `/home/mechti/PycharmProjects/DeepMzyme`) |

Recovery: if the workstation was rebooted, mount Data1 (`udisksctl mount -b /dev/sda1`), confirm the worktree with
`git -C /media/mechti/Data1/DeepMzyme_worktrees/v3 status`, then read STATUS and this file. If a VM session was cut
(hard stop, SSH loss): start the VM under authorization, run `P pmm_v3_step_d.py --step E status`; `wait` pulls every
lane awaiting acknowledgment; `recover-lane K` records an interrupted unit (then `archive-failed UNIT` and one unchanged
rerun); never relaunch a unit that has a launch record without an exit code. Run `evidence` before every `vm-stop`.

## 8. Next actions and commands

**Step F is complete (log v3-045 to v3-048).** No campaign step remains; the closeout decisions are in
[section 13](#13-closeout-proposal-2026-10-13-pending-the-users-approval). The procedure below is the historical GPU
session procedure of steps D–F.

Items 1–5 below describe the GPU session procedure used for step E.

Reconcile first (read-only): `cd /media/mechti/Data1/DeepMzyme_worktrees/v3; git status; git log -3`;
`~/deepmzyme-vm/bin/vm-status` and `vm-report --no-ssh`; if the VM is RUNNING unexpectedly, do not start anything,
run `P pmm_v3_step_d.py --step E status` (needs SSH to a running VM) and `wait` (never relaunch), then `evidence`
and `vm-stop`. If TERMINATED, the campaign root on the VM disk and the Data1 evidence copies are the truth; the
launcher's `status` after the next start lists completed units; completed units are never rerun (launcher refusal),
a failed unit gets one unchanged rerun via `archive-failed UNIT` then `launch`.

GPU session procedure as used for step E (P = /home/mechti/miniconda3/envs/DeepMzyme/bin/python, in the worktree):
1. `vm-status`; `vm-report --no-ssh`; `vm-start --confirm [--hours H]` (on a stockout: spaced same-VM retries only);
   `vm-setup --stages ssh,smoke`.
2. `P pmm_v3_step_d.py --step E status` (gates: readiness, extension, label record, final D assessment).
3. Rounds of three: `P pmm_v3_step_d.py --step E launch U@0 U@1 U@2` (waits and pulls; admission 1.25×forecast+960 s;
   five/six-class units are forecast warm by the launcher; a cold parse adds 6–15 min, so launch a cold builder by
   session minute ~160), or `--no-wait` then `wait --any <every running unit by name>` and refill after `status`;
   if a launch is refused with "awaits its verified host pull", run `pull --lanes K` and retry.
   All parse caches of steps D and E are now warm on the VM (the Only-GVP six_class set was built on 2026-10-11).
   Admission is per unit forecast: late fusion 2,400 s, Only-GVP 2,300 s, Only-ESMC 1,800 s, so the last late-fusion
   launch of a 237-minute session falls about 66 min before its hard stop.
4. After every session: `P pmm_v3_step_d.py --step E assess` (allowed at any time; incomplete cells are reported with
   their missing units), `P pmm_v3_step_d.py --step E evidence`, `vm-stop`, `vm-status` (TERMINATED), `vm-report --no-ssh`.
5. Log a dated entry (newest first in log.md), overwrite STATUS (≤ 6,000 bytes; preserve replaced dated text in the log),
   `P tools/check_docs_contract.py`, `git add -- <paths>`, `git -c core.hooksPath=.githooks commit`, push only with the user's OK.
6. Step E2 (CPU part done 2026-10-11, log v3-033): `P pmm_v3_step_e2.py gates` and `spend` (offline) show the
   records; after the user's logged authorization `P pmm_v3_step_e2.py record-authorization --log-entry v3-NNN`; each
   session `P pmm_v3_step_e2.py status`, `launch U@0 U@1 U@2 --no-wait`, `wait --any <every running unit>`,
   `evidence`, vm-stop; afterwards `systemd-run --user --scope -p CPUQuota=200% -p MemoryMax=3G P
   pmm_v3_e2_assessment.py` (newest E2 evidence copy). Never run E2 units through `--step E` or the VM runner. Step F
   waits for E2 unless the user cancels E2 by a dated log entry.

<a id="9-pending-review-of-the-e2-assessment-rules-saved-2026-10-08-unresolved"></a>
## 9. Review of the E2 assessment rules (saved 2026-10-08; resolved by the user the same day, log v3-028)

The user's review of 2026-10-08 (also [log v3-026](log.md#v3-026)) raised four points; a proposed resolution with
options was recorded in [log v3-027](log.md#v3-027) and the user decided on Option B with clarifications
([log v3-028](log.md#v3-028)). The result is frozen in [`e2_assessment_spec.md`](e2_assessment_spec.md) and
[`e2_assessment_spec.json`](e2_assessment_spec.json) (SHA-256 in v3-028) and is not changed after E2 results are seen.
E2 is planned and not implemented; the eight candidates, 32 candidate fits plus four `sitenone` controls, seed 42,
folds 1–4, the 50-epoch cosine schedule, the terminal checkpoint and the $55 gross planning ceiling are unchanged; no
GPU start is authorized.

1. **Promotion threshold → decided.** Replacement uses the A4 verdict rule unchanged: both unadjusted 95% lower bounds
   (10,000 fold-bootstrap resamples with seed 42; t with 3 degrees of freedom) above zero against the matched control
   and against the original completed E selection (`sitecountsangles` also against the Only-GVP baseline), an
   estimated mean gain over the original selection strictly greater than 0.2 points, and the recall gates against every
   required comparator. The 17 declared contrasts are reported with unadjusted and Bonferroni-adjusted intervals; the
   adjusted results are sensitivity evidence, not the gate; coincident comparisons are identified without changing
   the declared family. The illustrative figures of v3-027 (about 0.18 over eight independent nulls; the power table)
   are calculations under stated assumptions, not estimates of this campaign's false-replacement rate or power.
2. **Separate conclusions → decided.** Each candidate carries an own-model conclusion against its matched control
   (numerical differences, interval verdict, recall gates; A4 improvement-check vocabulary; a matched-control recall
   failure overrides "promising") and a replacement conclusion with the exact reason when it does not qualify; every
   failed gate is named by comparator, class and magnitude; a failure against the original selection blocks
   replacement without erasing a supported own-control finding; failure to replace is not evidence of no effect.
3. **Documentation consistency → done.** The handoff reads E → E2 → F; the step F draft was corrected as
   `step_f_plan_draft.md` (log v3-030): step F consumes the final E2 selection, or the original A4 selection if the
   user explicitly cancels E2.
4. **Interpretation → decided.** E2 is an exploratory extension that selects a promising configuration from reused
   development-validation evidence; passing its rule does not establish independently confirmed superiority or
   robustness across training seeds; step F stays one-shot with `both_results_secondary` and the prior-exposure
   caveats; neither the exploratory label nor a possibly opened test removes selection bias. No extra seed.

Corrections recorded in v3-028: the v3-024 t intervals labelled "unadjusted" were the adjusted-over-two intervals
(unadjusted: `headdrop03` t [−1.38, +0.71], `meanagg` t [−3.76, +4.48]; verdicts unchanged), and the bootstrap
explanation of v3-027 is qualified (exact endpoint mass 1/256; the finite estimate need not return the exact minimum;
bootstrap and t bounds are not independent confirmations).

<a id="10-deferred-review"></a>
## 10. Deferred read-only review: schedule, errors and minibatch weighting (DONE 2026-10-11, log v3-032)

**Status: DONE** ([log v3-032](log.md#v3-032), decision report; scripts and outputs in
`$DATA/audits/schedule_error_review_20261011T131706Z/`). Requested by the user on 2026-10-08 and 2026-10-11
(request and addendum in `$DATA/audits/session_notes_20261008/DEFERRED_amendment_review_request.md`; the
2026-10-11 additions in [log v3-031](log.md#v3-031)). Read-only, CPU, saved validation evidence; every terminal
value reconciled with `E_20261011T130128Z`. One boundary incident is disclosed in v3-032 and in the
[DATASETS test-use ledger](../../DATASETS.md#test-use-ledger) (a recursive search read bytes under the exact
Zenodo test directory; nothing was displayed or used).

Conclusions (details, numbers and corrections in v3-032):

- **Minibatch weighting:** confirmed for the actual configuration: Cu gets 20.13–20.27% of the expected per-step
  loss coefficients instead of 25%, and 46.4–47.4% of full batches contain no Cu. The effect on accuracy is
  unknown. Population-normalized weighting is available without a source change (`focal`, gamma 0), but it needs a
  new runner extension. A 4-fit matched ablation is possible and not currently justified.
- **Fe→Mn in `v3g_1q0c`:** verified. Correcting the group's 105 Fe adds +1.07 points to the five-fold mean; with its
  8 Co, +1.44 on the folds 1–4 mean, so one group can pass or block a 3-point recall gate. Whole-group failures
  recur in every fold and family, and mixed-metal near-copy groups err more; no input or label problem is
  established. Co is the weakest native class (six-class 2.8–5.3%).
- **Training versus validation:** validation loss rises late in 55 of 55 units, but validation BA does not decline
  recurrently. The decline from logged epoch 10 is concentrated in fold 1 (Class VIII→Mn, largely `1q0c`), and
  seed noise is large. A logged epoch is not a schedule.
- **"Structural information missed by fusion":** the counts are verified (562; fusion recovers 171), but the premise
  is not established: two seeds of the same Only-GVP model disagree almost as much as Only-GVP and Only-ESMC do.
- **10-epoch schedule:** the seven complete 10-epoch fold-0 runs (76.0–76.6) are above all 35 fold-0 50-epoch
  late-fusion units (at most 75.35), but this is one development fold and one seed. A narrow comparison (4 fits on
  folds 1–4) needs a new runner extension and a separate decision.
- **Decision:** no small additional GPU intervention is currently justified; E2 proceeds unchanged. Optional
  proposals for separate decisions are listed in v3-032.

<a id="11-pre-step-f-review"></a>
## 11. Pre-step-F review and recommendation (2026-10-12, log v3-038): option A chosen, run, NOT ADOPTED (log v3-041)

**Result (log v3-041): DONE, not adopted; the 50-epoch selection stands (E_sel = 50).**
- 4 of 4 fits completed on the first attempt in one session (37 min, $0.55 gross; VM TERMINATED 11:18:48 UTC).
- 10-epoch minus 50-epoch common-four BA on folds 1–4: +7.82, +0.15, −0.78 and +0.77 (mean +1.99, SD 3.94).
- Unadjusted bootstrap [−0.39, +5.90] and t [−4.28, +8.26]: gate (b) fails. Gates (a), (c) and (d) pass.
- Mean recalls: Mn +0.33, Cu +2.61, Zn +0.04, Class VIII +4.97.
- Fold 1's gain is mostly the whole-group switch of `v3g_1q0c` (+5.36). Without each fold's top group the
  differences are +2.45, −0.30, −1.15 and −0.22.
- Assessment `S10_20261012T111859Z` (`8fbce69b…`). Readiness and authorization: log v3-040.

**Status before the run (log v3-039):**
- The user chose option A below on 2026-10-12, exactly as specified.
- Its CPU preparation is DONE: extension `v3-ext2-sched10`, the frozen
  [specification](sched10_assessment_spec.json) (`14c4b572…`), and the manifest (`a9779c05…`) and ceiling
  (`3a6710a6…`) records. The checks are in log v3-039.
- The GPU-start authorization was given and recorded in log v3-040. Commands:
  [playbook](../../METAL_TRAINING_PIPELINE_PLAYBOOK.md#v3-pre-step-f-10-epoch-schedule-comparison-completed-2026-10-12-not-adopted).
- Required CPU check 9 (the tiny synthetic CPU fit) and `tests/smoke_checks.py` ran under the user's authorization of
  log v3-040 and passed. The fit showed 10-epoch execution, the terminal checkpoint, the replay and persistence.
- Reconciliation with an independent report (log v3-039):
  - Measured: LayerNorm-gain shrinkage under `wd10`; lower validation confidence and NLL for Only-GVP `wd10` and the
    fold-0 10-epoch fit; the fold-0 breadth of the 10-epoch gain compared with the candidates that failed.
  - Not established: that any of these causes the Only-GVP gain.
  - Deferred, untested and not disproven: trunk learning rate 1e-4, three GVP layers, ESM dropout after pooling, a
    LayerNorm decay exemption, population-normalized weighting and label smoothing.

The original proposal follows unchanged. The review was requested by the user on 2026-10-12. It used CPU only and
saved validation evidence: no fit, inference, refit, GPU action or test access. The findings, the optimizer
verification and the cost refresh are in [log v3-038](log.md#v3-038). Scripts, outputs and an independent cross-check
are in `$DATA/audits/pre_f_gvp_complementarity_20261012T085520Z/` (manifest `1fd779b5…`). Nothing below is
authorized until the user decides.

Findings in brief:

- **Improved Only-GVP (`wd10`) against the selected fusion, folds 1–4, seed 42.**
  - It corrects 618 fusion errors (9.27 points per fold) and loses 668 (9.73): net −0.46 (C10).
  - Only fold 1 is net positive. 5.36 of its 6.77 points come from one whole-group switch: Class VIII for all 121
    ions of `v3g_1q0c`, its 8 Mn included.
  - Without each fold's top group the net is +1.41, −3.38, −3.70 and −5.09.
  - Half of its +3.59 gain over the original Only-GVP falls on examples fusion already gets right; outside fold 1,
    most of it does.
  - About half of its differing answers change with the seed (fold 0, Jaccard 0.51).
- **Option B, checked against the implementation.** The evaluated global fusion `wd10` already decayed the GVP trunk
  (84 tensors at learning rate 3e-4) by exactly the Only-GVP `wd10` factor (0.0585). It did not move fusion toward the
  improved GVP's answers (12.5% against a 10.5% base rate), and its result was −0.61 (C8). A selective
  `--gvp-weight-decay 1.0` would only remove the 0.753 decay on the 31 base-rate tensors. Reaching the GVP pooling,
  projection and site encoder at Only-GVP strength needs a source change or a learning-rate change.
- **Option A, on the development fold.** The 10-epoch fits gain +4.42 to +4.99 (one seed). The gain is broad: all four
  classes, about 14 effective groups, +2.70 to +3.28 without the top group, and no whole-group switch. The 10-epoch
  fit also recovers 111 of the 174 examples where Only-GVP `wd10` is right and the 50-epoch fusion wrong; fusion `wd10`
  recovers 63.

### Recommended: option A, the 10-epoch schedule comparison (one experiment, 4 fits)

- **Question.** Does a complete 10-epoch cosine schedule with a terminal checkpoint improve the selected cell
  `gvp_late_fusion__four_class__baseline` on folds 1–4 over its verified 50-epoch units?
- **Evidence for it.** The fold-0 development signal above and in v3-032: the only remaining lead with a broad,
  consistent signal.
- **Remaining uncertainty.** Whether the fold-0 gain transfers to folds 1–4 (fold 0 is the selection's worst fold);
  whether it holds across seeds; whether it survives the full-cohort refit; and its effect on the test.
- **Exact comparison.**
  - Candidates: four new fits `gvp_late_fusion__four_class__sched10__fold{1,2,3,4}__seed42`.
  - Each command equals the selected cell's command for the same fold except `--epochs 10` and the run name. The
    cosine runs over 10 epochs from the same peaks (GVP trunk 3e-4, rest 3e-5) with a terminal checkpoint at epoch
    10. Weight decay 1e-4, batch 16, FP32, three lanes, the frozen data, folds, per-fold class weights, inputs and
    model are all unchanged.
  - Controls: the four 50-epoch units `gvp_late_fusion__four_class__baseline__fold{1,2,3,4}__seed42`, bound by
    their E2 unit-evidence checkpoint and prediction hashes and never rerun.
  - Fold 0: the existing step-B probe `probe-w1-fp32-r1…` (76.14) is reported separately as the development fold. It
    is never inside an interval, and probes are not reused as evidence or as a namespace.
- **Frozen gates.** They are written into an assessment specification whose SHA-256 is logged before any fit, and
  they are not changed after results. Comparison S1 is `sched10` minus `baseline`, the common-four BA difference per
  fold at the terminal checkpoints. The 10-epoch schedule is adopted for step F only if all four gates hold:
  - (a) Completeness: the 4 fits complete in all four senses, replayed under `pmm-v3-replay-1`, evidence copied, and
    the control hashes match.
  - (b) Paired interval: both unadjusted 95% lower bounds above zero, using the A4/E2 methods unchanged (fold
    bootstrap with 10,000 resamples and seed 42; t interval with 3 df).
  - (c) Minimum gain: a mean gain strictly greater than 0.2 points (the A4/E2 tie band; a tie keeps 50 epochs).
  - (d) Class recalls: against the control, no common-four class mean recall drops by more than 3 points, and no
    common-four recall is missing or zero in any fold. Fold supports and zero-recall events are reported.

  Otherwise the result is "not adopted", with every failed gate named by class and magnitude. Reported without gating:
  per-fold differences, class recalls, a Bonferroni t interval over the 18 post-A4 comparisons on folds 1–4 (the 17
  of E2 and S1; sensitivity only), a labelled five-fold descriptive mean with the probe, and the v3-038 breakdown.
- **Labels.** Exploratory and post hoc: the hypothesis came from fold-0 probes already seen. Folds 1–4 are reused
  after A4 and E2 used them. There is one training seed. Passing does not establish independently confirmed
  superiority or robustness across seeds. The 50-epoch control was selected on these folds, so its values may be
  optimistic; this works against the challenger. The step F test stays one-shot and `both_results_secondary`.
- **Required CPU checks before any fit.**
  1. A versioned runner extension `v3-ext2-sched10` with its own namespace, manifest, launcher and assessor. It binds
     the original manifest, extension 1, the E2 manifest and assessment and the source tree by hash. The hash-frozen
     runner, E2 modules and `src/` stay unchanged.
  2. A command diff for each fold: only `--epochs` and the run name differ, and the trainer's parser accepts the
     command.
  3. Schedule and optimizer: the per-epoch learning rates equal the recorded probe schedule. Both AdamW groups match
     the baseline in membership, learning rate and decay (the decay-audit group dump). The receipt fields require
     `planned = completed = selected = 10` and the terminal rule.
  4. Identity: `sched10` is distinct from every 50-epoch unit and probe, and the identity carries the epochs.
  5. Control binding: the controls are refused unless their hashes equal the E2 unit evidence.
  6. Replay and persistence: the existing lane, pull, acknowledgment and evidence contract.
  7. Assessor tests on synthetic data: all-pass, an interval failure, a minimum-gain failure, a recall-gate failure,
     zero recall and an incomplete set. It writes a machine-readable schedule decision with the resolved epochs and
     the hashes.
  8. Launch controls: the write-once $55 ceiling record and a user GPU-start authorization record. The admission rule
     is the measured short-fit forecast × 1.25 + 900 s.
  9. A tiny CPU smoke through the extension, `tests/smoke_checks.py` and the docs-contract check.
- **Execution.** After the checks, one GPU session (three fits, then one) with `--hours 2`, under the unchanged
  controller caps and only after the user's explicit GPU-start authorization. Approving the experiment does not start
  a GPU, and the advance authorization of v3-014 covers approved step F work only.
- **Cost (09:02 UTC).**
  - Option A alone: about 0.6–1.0 VM hours, $0.5–0.9 (up to $1.1 with a rerun), plus 1–2 storage days at
    $0.55/day.
  - Campaign total with A: about $42.8–49.7, against $42.1–47.5 going straight to F.
  - Both are within the recorded $55; no increase is requested.
- **Stopping rule.**
  - Exactly these four fits; a failed fit gets one unchanged rerun.
  - A second failure blocks the decision, and the 50-epoch selection stands unless the user decides otherwise.
  - No other epoch counts, seeds, folds, recipes or families.
  - One assessment, after all four fits.
  - Whatever the result, the next step is step F; no further pre-step-F experiment follows from this one.
- **Fallback if it fails.** Step F proceeds on draft v3's default path: the selection unchanged, 50 epochs. The
  10-epoch result is reported as exploratory and never as evidence of no effect.
- **If adopted.** Step F refits the same cell with a 10-epoch cosine schedule and a terminal checkpoint at epoch 10.
  The refit identity and every binding carry the schedule-decision SHA-256 and the epoch count. The plan's fixed
  "50 epochs" decision is amended for the step F refit only, by a dated log entry.

### Not recommended

- **Option B** (selective GVP weight decay inside fusion): its premise is not supported (see above), and it costs
  about twice A's GPU time (1.5–1.8 VM hours, $1.3–1.6).
- **Both options, a combination, a sweep or a warm start:** excluded by the request.
- **Step F directly** is the defensible alternative if the user prefers no further reuse of folds 1–4.

### Decisions for the user (none inferred)

1. Decided 2026-10-12 (log v3-039): option A; authorized (log v3-040); run and not adopted (log v3-041).
2. Superseded by section 12 (approval 1 of step_f_plan_draft.md F7).
3. Superseded by section 12 (commit and push of the v3-038 to v3-044 work, F7 item 3).

## 12. Step F tooling prepared (2026-10-12, log v3-042)

**Update 2026-10-13 (log v3-047, v3-048): STEP F COMPLETE.**
- Freeze 1 `54a724be…` was recorded, then the label-blind preparation (1,485 of 1,488 scored, clean subset 423) and
  the test ESMC (1,918 payloads, verified).
- Freeze 2 `949bc899…` followed, then the single prediction pass (claim `d097add2…`, predictions `cd8231d9…`).
- The frozen report (`step_f_test/report_20261013T012534Z`) gives a full-test common-four BA of **67.55%** [62.93,
  71.65], with Mn 55.69, Cu 70.31, Zn 85.07 and Class VIII 59.13. The LINK-free line is 68.99 and the clean subset
  48.85 (secondary; not pristine). The label is both_results_secondary, with all disclosures; the comparison with
  PMM's published 67.85 is lower and not matched.
- Next: the user's decisions on closure, commit and push, and the VM disk.

**Update 2026-10-12 (log v3-045, v3-046): the plan is approved; F1–F2 are complete, and the refit is receipted (checkpoint
`a0007846…`). The Freeze 1 approval package is under "Freeze 1 approval package" below.**

**State.** The user chose to finalize v3 with `gvp_late_fusion__four_class__baseline` at 50 epochs (log v3-042). The
10-epoch comparison stays recorded as not adopted, and the other optimization hypotheses are deferred and untested,
not disproven. The step F tooling was prepared and CPU-checked under a preparation-only authorization (log v3-042),
verified after a reboot and corrected in one CPU-only round (log v3-043). The earlier step F GPU authorization of
v3-014 is **not** activated. Nothing ran on test data. No step F record is written: the F1 selection record was
dry-run only, and there is no manifest, ceiling, authorization, freeze record or prediction claim. The commands are in
the [playbook](../../METAL_TRAINING_PIPELINE_PLAYBOOK.md#v3-step-f-stage-6b-refit-and-the-one-test-pass-tooling-prepared-2026-10-12-not-approved-not-run);
the implementation choices and the explicit clean-subset rule are in the
[step F draft's implementation notes](step_f_plan_draft.md#implementation-notes-2026-10-12-log-v3-042).

| File (worktree, untracked) | Role |
|---|---|
| `docs/campaigns/pmm_ion_metal_v3/step_f_spec.json` | the rules as data; SHA-256 `04d22cc4…` (log v3-044), pinned in `pmm_v3_test_prep.py` and `pmm_v3_step_f.py` |
| `run_pmm_v3_step_f.py` (`29427650…`) | VM refit (`plan-refit`, `refit` with its in-process reload check, `reload-check`); workstation CPU `prepare-test` and `test-embeddings` (recorded Freeze 1 required, closed after Freeze 2) and `predict` (Freeze 1 and 2 verified, the F4 go-ahead, one campaign-wide claim) |
| `pmm_v3_test_prep.py` (`2df11024…`) | label-blind dispositions, the label-independent `ion_key`, row accounting, aliases, LINK flags, clean subset C1–C3, the test ESMC plan and bfloat16 payloads, and the record checks shared with the runner and the report |
| `pmm_v3_step_f_report.py` (`b4c7dacd…`) | the only component that uses a label value; the claimed predictions only; four lines, PDB-entry bootstrap, PMM wording, disclosures; byte-reproducible |
| `pmm_v3_step_f.py` (`a91e0db3…`) | workstation (standard library): F1 record, manifest, ceiling, authorization, deploy (three files), step C lane logic, `refit-receipt`, Freeze 1 (bound to the refit manifest's code; `--amends`) and Freeze 2 records and their ledger text; records that name a commit need the bound files committed |
| `tests/test_v3_step_f.py` (`89468a62…`) | 39 tests, two of them opt-in synthetic CPU fits (run as authorized) |

**Checks** (details in [log v3-043](log.md#v3-043)):
- The 39 step F tests pass, including the two authorized synthetic fits. Smoke checks: 43 passed; the docs contract has 0 strict failures.
- CPU ESMC check (log v3-044): three training chains fixed in advance, embedded on two cores under a 6 GB cap and
  compared with their stored GPU embeddings. Mean per-residue cosine 0.99979–0.99982 (minimum 0.99941), mean absolute
  difference 6.2–7.0e-4, peak memory 5.3 GB, 3.4–14.9 s per chain; suitable. Its first attempt exposed a
  specification defect (`storage_dtype`), now fixed and tested; readiness stage A was rechecked on the fixed
  specification (`$DATA/audits/f_readiness_20261012T181114Z/`).
- The original readiness run `$DATA/audits/f_readiness_20261012T122926Z/` (`561190f8…`) is preserved; its stage C
  passed by a recorded reassessment, and stage B ran with an earlier runner (`9e663f11…`). The scripted rerun on the
  final tools is `$DATA/audits/f_readiness_20261012T170111Z/` (`readiness_checks.json` `5b1ce5f0…`). There A, B and D passed. C failed as run because of a scope error in the script's sequence-count check (every row-level check passed); the corrected script reran stage C in `$DATA/audits/f_readiness_20261012T174408Z/` (`step_f_readiness.py` `b73ece28…`, `readiness_checks.json` `7e61a339…`) and it passed: all 1,049 fold-1 context sequences of at least 30 residues are identical to a reference and flagged (1,048 in the first run; 4uer chain 9 is now caught), all 1,498 rows with such a chain are excluded by C2, and only the two short-chain rows (5wlm, 6wbe) remain. That script combines the earlier A, B and D (verified by SHA-256 and identical tool hashes) and computes ready: true.

**Freeze 1 approval package** (historical: approved in log v3-047 and executed; prepared 2026-10-12 from the completed refit and existing records; no test data opened):
- **What Freeze 1 records** (`pmm_v3_step_f.py freeze1-template` prints it; `record-freeze1 --log-entry v3-NNN` writes
  it once):
  - specification `04d22cc4…`, selection record `e8a2e8cd…` and manifest `d37ba100…`;
  - the refit checkpoint `a0007846…` with its run config `8db9c5e2…`, metadata `f9b442c4…`, status `f48e5d24…` and
    reload check (passed); seed 42, 50 epochs, terminal checkpoint; no calibration, no ensemble;
  - cohort `745b19c1…`; tool hashes equal to the refit manifest's (`pmm_v3_step_f.py` `a91e0db3…`,
    `run_pmm_v3_step_f.py` `29427650…`, `pmm_v3_test_prep.py` `2df11024…`, `pmm_v3_step_f_report.py` `b4c7dacd…`);
    git commit `7bd7884` with every bound file committed;
  - the test source CSV `ec6427ad…` (1,488 rows) and 316 previously opened structure files;
  - the report specification, the clean-subset rule, the CPU test ESMC settings, the 9 disclosures and the freeze
    order; label `both_results_secondary`; `test_side_read_before_this_record` false.
- **Ledger paragraph** for docs/DATASETS.md (printed by the command, with the go-ahead entry filled in): "2026-10-12 v3
  step F Freeze 1 (exact Zenodo PinMyMetal test, before any test-side read): specification `04d22cc432de`; selected
  model `gvp_late_fusion__four_class__baseline` (50 epochs, terminal checkpoint, seed 42; refit checkpoint
  `a00078466cf6`); tools …; report specification, clean-subset rules and disclosures as in the specification (label
  both_results_secondary). User go-ahead: log v3-NNN. Next: label-blind preparation and test-side ESMC generation on
  the workstation CPU (no label value used)."
- **What the go-ahead permits next (F3b, workstation CPU only):**
  - `prepare-test` reads the exact Zenodo test directory's structure and site manifests (allowlisted identifier
    columns), its structures, the source CSV (allowlisted `pdbid` and `residueid_ion`; label cells parsed and
    discarded unused), and the file names of the previously opened set
    (`/home/mechti/PycharmProjects/DeepMzyme/DeepMzyme_Data/train_and_test_sets_structures_exact_pinmymetal/test`,
    which must hold exactly 316 structure files);
  - `test-embeddings --device cpu` embeds the planned context chains (ESMC-600M cached; 6 GB cap; about 1.5–3 hours;
    keep about 6 GB free);
  - no label value, prediction or report.
- **Not permitted by the Freeze 1 go-ahead:** Freeze 2, the prediction pass and the report, each separately gated.
- **Cost:** no VM time; storage about $0.55/day until the disk is retired; $13.20 left under $55.

**The approval list of log v3-043/v3-044** (step_f_plan_draft.md F7; items 1–5 decided in log v3-045, item 6 pending):
1. Approve or amend the step F plan. It includes:
   - the clean-subset rule as implemented: A1 with the identical-chain clause, the short-chain limitation kept and
     disclosed;
   - the label-blind test-side reads;
   - the workstation-CPU test ESMC route in bfloat16 and its requirements (plan F6: weights cached, a 6 GB memory
     cap with a measured peak of 5.3 GB, about 1.5–3 hours on two cores);
   - the Freeze 1 amendment procedure, the F4 go-ahead and campaign-wide claim, and the plan.md alignment.
2. Done: the training-side CPU ESMC check passed (log v3-044); the bfloat16 CPU route stays.
3. Commit (and decide whether to push) the v3-038 to v3-044 work. `bind-selection`, `build-manifest` and
   `record-freeze1` refuse until the bound files are committed.
4. Authorize the refit's GPU start: one session, `vm-start --confirm --hours 2`; one 50-epoch full-cohort fit
   (admission forecast 3,300 s) with its reload check. That is about 1.0–1.5 VM hours, $0.9–1.3 at $0.8586/h, or up
   to 2.25 h ($1.9) with the one unchanged rerun, plus storage at $0.55/day. Before launch: `bind-selection`,
   `build-manifest`, `record-ceiling` and, after the logged decision, `record-authorization`.
5. The plan.md alignment (F7 item 5), on approval.
6. Later, three logged go-aheads:
   - Freeze 1, then F3b on the workstation CPU;
   - Freeze 2;
   - the single test pass (`predict --log-entry`) and the report.

## 13. Closeout proposal (2026-10-13, pending the user's approval)

Prepared read-only ([log v3-049](log.md#v3-049)). Approved and done since ([log v3-050](log.md#v3-050)): the NVMe backup,
the summary wording corrections, the commit of the closeout records and the tag `pmm-ion-metal-v3-final`. Not done:
closure, archive move, new snapshot or deletion.

### Backup coverage (verified 2026-10-13 about 02:10 UTC; NVMe copy added 02:32 UTC)

The 16 key step F files on Data1 match their recorded SHA-256 values (refit checkpoint, configuration, metadata, reload
check and status; the F1, manifest, ceiling, authorization, Freeze 1, Freeze 2 and claim records; predictions, report
and F3b verification). The other copies:

| Artifact | Data1 (`/dev/sda1`) | VM disk (GCP) | git | Independent copies |
|---|---|---|---|---|
| Refit run folder, 37 MB (checkpoint `a0007846…`, run config, metadata, reload check) | yes, verified | yes, the original (the host-pull manifest `cc768bdb…` matched at the pull) | hashes only | 2, until the VM disk is retired |
| Refit status, command and launch files | yes (durable copy and `evidence_20261012T200435Z`) | yes | hashes only | 2 |
| F1, manifest, ceiling and authorization records | yes | no | hashes only | **1** |
| Freeze 1, Freeze 2 and the prediction claim | yes | no | hashes in log and DATASETS | **1** |
| Prepared memberships (1.2 MB), `f3b_verification.json` | yes | no | no | **1** |
| Test ESMC payloads (2.3 GB) | yes | no | no | **1** (regenerating them would be a new test-side read and need not be bit-identical) |
| Predictions (448 KB), report (24 KB) | yes | no | no | **1** |
| Steps B–E2 and S10 runs, evidence and assessments | yes (durable copies, `evidence_*`) | yes, the originals | no | 2, until the VM disk is retired |

**Missing coverage:** everything written on the workstation after the refit has a single copy on Data1, one HDD that is
96% full (79 GB free). The VM disk is the only second copy of the refit and of the earlier runs.

**Backup done (log v3-050):** every Data1 item in the table also has a verified copy on the workstation's NVMe disk
(`/dev/nvme0n1p6`, ext4) at `/home/mechti/deepmzyme_backups/pmm_ion_metal_v3_closeout_20261013T023038Z/`:
- the backup holds 4,836 files and 2,639,602,614 bytes;
- the content is `step_f_evidence/`, `step_f_test/` (both test ESMC folders included), the refit run folder with its
  log, status, command file and host-pull receipt `cc768bdb…`, and the step F readiness and CPU ESMC audits;
- every source file was hashed before the copy, and every copy then passed `sha256sum -c --strict` against that
  `SHA256SUMS` (`b9b6d44f…`);
- the 16 key files match the hashes recorded in the step F records;
- the receipt is `BACKUP_RECEIPT.json` (`ddbec5de…`), with a copy and the manifest on Data1 under
  `backups/nvme_closeout_20261013T023038Z/`;
- the originals are unchanged.

**Remaining gap:** both copies are on the same workstation, so there is no off-site copy. The VM disk stays the third
copy of the refit and the only second copy of the step B–E2 run originals.

### Campaign closure (the AGENTS procedure: `Status: closed`, `git mv` to `docs/archive/campaigns/`, one PARAMETER_FINDINGS line, STATUS)

**Under revision (log v3-050):** the user requires the original files, receipts and hashes to stay valid, without
weakening the documentation contract or closing in place silently. The proposed sequence below changes frozen tool
bytes and breaks the relative links of moved Markdown, so it is not approved; a compatibility solution is prepared
separately for approval.

**Dependencies found before any move:**
- Six code modules read their specifications by path from `docs/campaigns/pmm_ion_metal_v3/`: `pmm_v3_test_prep.py`
  (fails at import if the file is missing), `pmm_v3_step_f.py`, `pmm_v3_step_e2.py`, `pmm_v3_sched10.py`,
  `pmm_v3_assessment.py` and `pmm_v3_bundle.py`; two test files do the same. A plain `git mv` breaks them and the v3
  test suites.
- The four hash-frozen runner files only mention the path in a docstring and stay unchanged.
- 39 link mentions in 6 files outside the folder point into it: STATUS, DATASETS, the metal playbook, the archived v2
  campaign README and log, and the protected `docs/plans/gvp_and_esmc_evidence_ranked_improvement_plan.md`.
- The step F records bind the specification path and the tool hashes, so they verify together only at the closing
  commit.

**Proposed sequence:**
1. Commit the staged v3-046 to v3-049 records with the hook.
2. Tag that commit `pmm-ion-metal-v3-final`: the state where every tool, specification and record verifies together.
3. In one closure commit:
   - `git mv` the folder to `docs/archive/campaigns/pmm_ion_metal_v3` and set `Status: closed`;
   - point the six path constants and the two tests at the archive (no logic change; the pinned specification hashes
     stay the same, while the tool file hashes change by design after the tag);
   - repair the 39 links, using `docs/MOVED.md` where protected or archived files must stay unedited;
   - add the PARAMETER_FINDINGS line below and overwrite STATUS.
4. Rerun the v3 CPU test suites (no fits), the smoke checks and the documentation contract.

The smaller alternative is to close in place (`Status: closed`, no move) as an explicit exception to the AGENTS move
rule, which avoids the code and link changes.

**Draft PARAMETER_FINDINGS line** (validation-only): "PMM ion metal v3 (closed 2026-10-13; Grade 2, five grouped
folds, seed 42): five- or six-class training gave no supported common-four gain over `four_class` for any family;
four-class graph-level late fusion (mean common-four BA 74.28%) beat Only-ESMC (72.33%) on all five folds; the step D
recipes, the E2 candidates and a 10-epoch schedule were not adopted. Secondary test result and caveats: the
[completion summary](../../notebook_outputs/summaries/summary_pmm_ion_metal_v3_20261013.md)." (link
relative to this file; written relative to `docs/` when applied)

### Storage retirement (controller; dry run only)

**Last verified state** (`vm-status`, `vm-report --no-ssh`, 2026-10-13 about 02:00 UTC):
- VM `deepmzyme-l4` TERMINATED;
- disk `deepmzyme-l4-from-deepmzyme-paused-20261003`, 150 GB pd-balanced, $15.00/month (about $0.49/day);
- snapshot `deepmzyme-paused-20261003`, 34.8 GiB stored: about $1.74/month actual, with $7.50/month reserved by the
  controller;
- about $0.55/day, or $16.5/month, while kept.

**Options:**
- **A, archive then retire** (recommended if a restore may ever be wanted):
  1. `vm-archive --name deepmzyme-v3-final-20261013 --max-monthly-usd 10 --confirm`. The dry run showed a standard
     snapshot at $0.05/GiB-month: $7.50/month conservatively, likely about $2–3/month compressed, plus one temporary
     restore disk for minutes. It never starts or deletes the VM.
  2. Review its receipt, then `vm-delete --delete-disk --confirm 'DELETE deepmzyme-l4'`, a phrase the user types.
  3. Ongoing about $2–3/month, plus $1.74/month for the 2026-10-03 snapshot until its removal is decided; the
     controller has no command for that.
- **B, retire without a new archive:** `vm-delete --delete-disk` keeps only the 2026-10-03 snapshot (about
  $1.74/month). The VM-side originals and caches are lost; every pulled run output is on Data1.
- **C, keep everything:** about $0.55/day.

The independent backup above must come first for A and B, because the VM disk is currently the only second copy.

### Decisions for the user (none inferred)
1. **Backup:** done locally (above); an off-site copy is optional.
2. **Commit, tag and push:** approved in log v3-050.
3. **Closure:** the compatibility solution (pending; the sequence above is not approved).
4. **Storage:** option A, B or C (the archive storage authorization, and the delete phrase typed by the user).
