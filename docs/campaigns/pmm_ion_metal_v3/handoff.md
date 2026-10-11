# PMM ion metal v3 — continuation handoff (written 2026-10-08 12:17 UTC; step E2 amendment added ~13:00 UTC, its review saved ~13:20 UTC and resolved ~17:00 UTC with the E2 specification frozen; fallback leftovers removed and step F draft corrected ~18:40 UTC; step E completed and the deferred review item added 2026-10-11 ~13:15 UTC, log v3-031; the section 10 review completed and the E2 tooling built, checked and its manifest and ceiling recorded 2026-10-11 ~14:30 UTC, log v3-032/v3-033)

Self-contained handoff for a new Claude Code chat. It depends on no earlier
conversation and on no temporary file; every path below is durable (git,
`/media/mechti/Data1`, `~/deepmzyme-vm`). Newest state first; the
[log](log.md) holds the dated history and [STATUS](../../../EXPERIMENT_STATUS.md)
the current authority.

## Ready-to-paste continuation prompt

```text
Continue the DeepMzyme PMM ion metal v3 campaign from the verified stopping point recorded in
docs/campaigns/pmm_ion_metal_v3/handoff.md. Steps A–E are complete (final A4 assessment E_20261011T130128Z,
Stage 6 selection gvp_late_fusion__four_class__baseline; log v3-031). The read-only section 10 review is done
(log v3-032: no small extra GPU intervention justified; E2 unchanged; optional later proposals named there). The E2
CPU tooling is built and checked (pmm_v3_step_e2.py, pmm_v3_e2_assessment.py; log v3-033); the write-once E2
manifest (SHA-256 37c2ff05…) and the $55 ceiling record (a4c31da1…) are recorded in step_e2_evidence on the
campaign data root; NO E2 authorization record exists. The E2 GPU-start gate is reached: the user's decision is
pending (authorize E2 with a window, keep or change the $55 ceiling, or cancel E2 by a dated log entry; optional
items of v3-032 only by separate decision). Do not start a GPU without that recorded decision.

Work only in the worktree /media/mechti/Data1/DeepMzyme_worktrees/v3 (branch v3-step-a; check `git status` and
`git log -3`; the 2026-10-11 commits may be unpushed). Not in /home/mechti/PycharmProjects/DeepMzyme. If
/media/mechti/Data1 is missing, mount it first: `udisksctl mount -b /dev/sda1`. Python:
/home/mechti/miniconda3/envs/DeepMzyme/bin/python (verify with -c "import sys; print(sys.executable)"). Cap local CPU
runs: systemd-run --user --scope -p CPUQuota=200% -p MemoryMax=3G.

Read in order: AGENTS.md; EXPERIMENT_STATUS.md; docs/campaigns/pmm_ion_metal_v3/README.md; handoff.md (sections 8
and 10 first); log.md v3-033, v3-032, v3-031; plan.md steps E2 and F; e2_assessment_spec.md; the playbook section
"v3 step E2 individual-confirmation amendment" (commands); .agents/skills/gpu-use-skill/SKILL.md,
~/deepmzyme-vm/AGENTS.md and config.env before any GPU action.

After the user's authorization is logged: `pmm_v3_step_e2.py record-authorization --log-entry v3-NNN`, then E2
sessions with `pmm_v3_step_e2.py` only (status before every refill; name every running unit in `wait --any`;
`evidence` before every vm-stop; verified TERMINATED), then `pmm_v3_e2_assessment.py` on the workstation (no VM).
Three sessions are forecast (8.3–9.4 VM hours, stress 10.1; about $7.1–8.9). Never call run_pmm_v3_campaign.py
directly; never edit the hash-frozen runner files or any E2/reused module after the manifest (a change refuses
every launch until a new manifest and a new authorization are recorded).

Rules that stay fixed: 50-epoch cosine, terminal checkpoint; A4 and E2 rules unchanged; seed 42; one active GPU;
controller only via ~/deepmzyme-vm/bin; session cap 4 h / $6, daily cap 12 h / $18 (UTC day); campaign ceiling as
recorded ($55 gross including storage in E2's control; spent about $32.00 at 14:10 UTC 2026-10-11; E → E2 → F
forecast $41.7–48.9). Every GPU session ends with evidence, vm-stop and a verified TERMINATED; log + STATUS +
commit after each session or step with `git -c core.hooksPath=.githooks commit`; push only with the user's OK.
Never run vm-fallback, delete resources, raise controller caps, run the cost-gated augmentations or Round C, or
start step F without the user. No test-set access before step F (record any incidental exposure in the DATASETS
test-use ledger, as done in v3-032).

Open user decisions (never infer them): the E2 GPU-start authorization and window (or a dated E2 cancellation);
the recorded $55 ceiling; optional v3-032 items (narrow 10-epoch comparison, weighting ablation, CPU provenance
checks, a non-decision-bearing per-group E2 display declared before results); Only-ESMC fairness arm; cost-gated
augmentations; step F plan approval; push of v3-step-a. At each closeout report completed/pending work, spending
and forecast, evidence copies, verified VM state, the exact next action and any decision the user must make.
```

## 1. Where to work and what to read

| Item | Value |
|---|---|
| Worktree | `/media/mechti/Data1/DeepMzyme_worktrees/v3` (branch `v3-step-a`; HEAD = `origin/v3-step-a`, see `git log -1`) — not `/home/mechti/PycharmProjects/DeepMzyme` (older state) |
| Python | `/home/mechti/miniconda3/envs/DeepMzyme/bin/python` (verify with `-c "import sys; print(sys.executable)"`); local CPU runs capped with `systemd-run --user --scope -p CPUQuota=200% -p MemoryMax=3G` |
| Campaign | `pmm_ion_metal_v3`; docs `docs/campaigns/pmm_ion_metal_v3/` (README, plan.md, assessment_spec.md, log.md, this file); data `/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v3` (the Data1 drive is `/dev/sda1`, label `Data`; after a reboot mount it with `udisksctl mount -b /dev/sda1`) |
| Reading order | AGENTS.md → EXPERIMENT_STATUS.md → docs/campaigns/pmm_ion_metal_v3/README.md → handoff.md section 10 (pending review) and section 9 (E2 rules, resolved) → plan.md (steps D–F, E2 included) → assessment_spec.md → e2_assessment_spec.md → log.md entries v3-031 back to v3-017 → docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md "v3 step D regularization amendment" and "v3 step E2 individual-confirmation amendment" → .agents/skills/gpu-use-skill/SKILL.md → ~/deepmzyme-vm/AGENTS.md and config.env (before any GPU action) |
| Launchers | `pmm_v3_step_d.py` (steps D and E: units, gates, status, launch, wait, pull, recover-lane, archive-failed, assess, record-cost-gate, record-e-gate, evidence); `pmm_v3_step_e2.py` (step E2 only: units, gates, build-manifest, record-ceiling, record-authorization, spend, status, launch, wait, pull, recover-lane, archive-failed, evidence) and `pmm_v3_e2_assessment.py` (workstation E2 assessment from Data1); `pmm_v3_step_c.py`/`pmm_v3_step_b.py` (shared lane logic); never edit `pmm_v3_campaign.py` or `run_pmm_v3_campaign.py` (hash-frozen by the extension record), and not the E2 or reused modules pinned by the E2 manifest |
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
  original A4 selection if the user explicitly cancels E2), Stage 6B full refit, one test pass. Steps A–E are
  complete (log v3-031).
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
- **Selection and test access:** validation-only selection; no held-out use before step F; before any test
  access the primary report, test-row and clean-subset rules with membership hashes, refit checkpoint and seed,
  ensemble (none) and calibration rule (none) are recorded in docs/DATASETS.md's test-use ledger. Step F tooling
  (refit, label-blind clean-subset/test-preparation builder) does not exist yet and needs planning and user approval.
- **Step E2 amendment (planned 2026-10-08, log v3-025; tooling built and CPU-checked 2026-10-11, log v3-033; not run):** after the original E assessment, the
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

## 3. Work state (verified 2026-10-11 13:05 UTC)

**Step D: complete (final assessment `D-B_20261007T201334Z`, status `final`).** Every listed unit below is complete
in all four senses (training completed at epoch 50, worker exit code 0, artifacts persisted by host pull with
SHA-256 manifests, host acknowledgment uploaded) and replayed under `pmm-v3-replay-1`; the pulled copies are in
`$DATA/durable/lane*/runs/<unit>/`. Nothing is running; no lane awaits a pull; no unit failed in steps D or E;
the VM is TERMINATED (verified 2026-10-11 13:03 UTC).

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

Fold 0 of every E contrast reuses the step C / step D runs (never rerun). **Step F:** not started; tooling absent;
the corrected execution draft is `step_f_plan_draft.md` (draft v2, log v3-030; not approved): it takes the final E2
selection (A4 only after a dated E2 cancellation) and a validation-free terminal refit; the 08:45 UTC session draft
on the data root is kept unchanged as history.
**Step E2:** authorized (log v3-034) and running: **13 of 36 fits complete after E2 session 1** (log v3-035; Only-GVP `wd10`, `sitecountsangles` and `sitenone` folds 1–4, late fusion `invsqrtw` fold 1; two `sitenone` replay-tolerance failures rerun once and passed; no unit has a second failure; latest evidence `step_e2_evidence/evidence_20261011T233244Z`; VM TERMINATED 23:34 UTC). Tooling: log v3-033, commits `4c65474`, `e4ce93c`. Records in
`$DATA/step_e2_evidence/`: manifest `e2_manifest.json` (SHA-256 `37c2ff05…`; binds the original assessment `d6d27c0c…`
and its selection, the final D assessment, the extension record, runner, E2 tooling and reused modules, and 19 reused
units), ceiling `e2_ceiling.json` (`a4c31da1…`; $55, reserve $6), **no `e2_authorization.json`** (every E2 launch is
refused until the user's decision is recorded). Readiness dry run `audits/e2_readiness_20261011T142030Z/` (A4 result
re-derived exactly from the Data1 copies; reused units verified). The six late-fusion candidates' comparisons against
the selection coincide with their matched-control comparisons (C12–C17 = C3–C8; divisor 17 unchanged). Step F waits
for E2 unless the user cancels E2 by a dated log entry.

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
| 2026-10-07 evening | Final-test label `both_results_secondary`; continue the approved plan until 03:38 UTC 2026-10-08 at the latest (eight hours maximum, not a target), step E after midnight UTC if gates pass, caps and ceiling unchanged; fairness arm kept separate for the user's decision; audit of the scratch drivers delivered before any correction or driver reuse; closeout commits pushed to `origin/v3-step-a`; a complete handoff for a new chat | log v3-022 |

**Authorization boundaries:** GPU starts for the approved D–F work are pre-authorized within the $40 gross ceiling
(storage included) and the controller caps (4 h / $6 per session, 12 h / $18 per UTC day); one active GPU, controller
only via `~/deepmzyme-vm/bin`; every session ends with `evidence`, `vm-stop` and a verified TERMINATED. Never:
Round C, cost-gated augmentations (`posnoise01`, `outerdrop01`) without a recorded OK, `vm-fallback`, deleting cloud
resources, editing the frozen runner files, editing controller caps, pushing without the user's OK
(granted for the 2026-10-07/08 and 2026-10-11 closeouts only), step F before its gates and plan. **Decisions still open:** Only-ESMC
fairness arm; cost-gated augmentations; step F plan approval; the E2 GPU-start authorization after the accepted CPU
implementation (or a dated cancellation of E2). The failed-fallback leftovers were deleted on 2026-10-08 under the
user's authorization through `vm-fallback --abandon` (log v3-030).
The 2026-10-11 arrangement ended with session 9's closeout (step E complete): the next GPU use is E2, which needs
the user's separate E2 GPU-start authorization; ask before any GPU start. Push authorization was given for the
2026-10-07/08 and 2026-10-11 closeouts only.

## 5. VM, spending and limits (verified 2026-10-11 13:03 UTC)

| Item | Value (verified 2026-10-11 13:03 UTC by `vm-status` and `vm-report --no-ssh`) |
|---|---|
| VM | `deepmzyme-l4`, project `deepmzyme-gpu-vm`, zone `us-central1-a`, g2-standard-8 + 1× L4, **TERMINATED** (ledger STOPPED 2026-10-11 13:03:24 UTC, session `session-20261011T115703Z-cf28c51f`, 1 h 06 min, $0.97 gross; before it `session-20261011T082141Z-2e94653e`, 3 h 31 min, $3.09); 150 GB boot disk kept (`deepmzyme-l4-from-deepmzyme-paused-20261003`) |
| Spending | refreshed in log v3-033 (14:10 UTC): about **$32.00 spent**; E2 8.3–9.4 VM h central (stress 10.1), $7.1–8.9; E → E2 → F about **$41.7–48.9**; step F alone after a dated E2 cancellation about $33.5–36.7; the **$55 ceiling is recorded in E2's launch control** (`e2_ceiling.json`, estimated charges with the running session to its hard stop plus a $6 reserve). Earlier: one dated calculation, log v3-031 (2026-10-11 13:10 UTC; estimated charges, disk counted once): running B–E $26.96 (31.40 h × $0.8586/h); campaign disk $2.76; archive snapshot $0.32; leftovers final $1.94; **$31.98 spent, $8.02 of the $40 gross ceiling left**; forecast E → E2 → F $42.4–48.8 including storage (E2 8.6–9.6 h, F 1.0–2.25 h VM wall time; storage $0.55/day for 4–12 days), above $40 and within the user's $55 E2 planning ceiling, which must be recorded in the E2 execution controls before any E2 GPU fit (not start permission); step F alone after a dated E2 cancellation $33.9–36.7 |
| Retained storage | campaign disk `deepmzyme-l4-from-deepmzyme-paused-20261003` ($0.49/day) and archive snapshot `deepmzyme-paused-20261003` ($0.06/day): $0.55/day in estimated charges; the controller reserves $0.75/day (a guard, not a charge). The failed-fallback disks and snapshot were deleted on 2026-10-08 (log v3-030; receipt archived as `state/fallback-873d08b26968.abandoned.json`) |
| Limits | controller caps 4 h / $6 per session, 12 h / $18 per UTC day (`~/deepmzyme-vm/config.env`; never edit without the user's instruction); 2026-10-11 used 4 h 37 min / $5.20 of running time; a new UTC day resets the daily cap; measured 2026-10-11: about 15 Only-GVP/late-fusion fits per 4 h session (about 40 min per fit at three lanes; Only-ESMC 30–35 min) |
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

**Section 10 review done (log v3-032); E2 tooling built and checked, manifest and ceiling recorded (log v3-033).**
Next in order: (1) the user's decision at the E2 GPU-start gate (authorize E2 with a window, keep or change the $55
ceiling, or cancel E2 by a dated log entry); (2) after a logged authorization, `record-authorization`, then the E2
GPU sessions with `pmm_v3_step_e2.py` (procedure below, with `pmm_v3_step_e2.py` in place of `pmm_v3_step_d.py
--step E`; commands in the playbook E2 section); (3) the workstation E2 assessment (`pmm_v3_e2_assessment.py`, CPU,
after `evidence`) and its log entry; (4) step F tooling, plan approval and step F under its own approval. Items 1–5
below describe the GPU session procedure used for step E.

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
