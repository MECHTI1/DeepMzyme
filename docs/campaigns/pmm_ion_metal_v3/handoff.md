# PMM ion metal v3 — continuation handoff (written 2026-10-08 12:17 UTC; step E2 amendment added 2026-10-08 ~13:00 UTC and the pending E2 review saved ~13:20 UTC, documentation only)

Self-contained handoff for a new Claude Code chat. It depends on no earlier
conversation and on no temporary file; every path below is durable (git,
`/media/mechti/Data1`, `~/deepmzyme-vm`). Newest state first; the
[log](log.md) holds the dated history and [STATUS](../../../EXPERIMENT_STATUS.md)
the current authority.

## Ready-to-paste continuation prompt

```text
Continue the DeepMzyme PMM ion metal v3 campaign (step E, five-fold confirmation; step E2 planned, its rules pending
review) from the verified
stopping point recorded in docs/campaigns/pmm_ion_metal_v3/handoff.md.

Work only in the worktree /media/mechti/Data1/DeepMzyme_worktrees/v3 (branch v3-step-a; HEAD must equal
origin/v3-step-a; `git status` clean). Not in /home/mechti/PycharmProjects/DeepMzyme. If /media/mechti/Data1 is
missing, mount it first: `udisksctl mount -b /dev/sda1`. Python: /home/mechti/miniconda3/envs/DeepMzyme/bin/python
(verify with -c "import sys; print(sys.executable)"). Cap local CPU runs: systemd-run --user --scope -p CPUQuota=200%
-p MemoryMax=3G.

Read in order: AGENTS.md; EXPERIMENT_STATUS.md; docs/campaigns/pmm_ion_metal_v3/README.md; handoff.md (all
sections, section 9 first); plan.md steps E, E2 and F; assessment_spec.md; log.md entries v3-026 back to v3-017;
docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md "v3 step D regularization amendment"; .agents/skills/gpu-use-skill/SKILL.md;
~/deepmzyme-vm/AGENTS.md and config.env before any GPU action.

State: steps A–D complete; final step D recipes late fusion `headdrop03`, Only-GVP `meanagg`, Only-ESMC baseline
(final assessment D-B_20261007T201334Z). Step E: 36 neutral-test fits (seed 42, folds 1–4, three families × four/
five/six) + 8 improvement fits (gvp_late_fusion__four_class__headdrop03__fold{1-4}__seed42,
only_gvp__four_class__meanagg__fold{1-4}__seed42); 24 of 44 are complete (section 3 lists them); the rest are pending, none failed. Final-test label
both_results_secondary is recorded (record-e-gate done). Step F has no tooling yet: plan it and get the user's
approval before any refit or test access. Step E2 (log v3-025; plan.md step E2) is PLANNED ONLY: the eight
remaining step D passes tested one at a time on folds 1–4 (36 fits, seed 42; original E assessment first); no
launcher, assessor, manifest or specification exists and nothing has run. Step F requires E2 completion unless the
user cancels E2 by a dated log entry. `--step E` refuses E2 units, but the VM runner's `--action run` does not check
step membership: never call run_pmm_v3_campaign.py directly.

Rules that stay fixed: 50-epoch cosine, terminal checkpoint; A4 gates unchanged; seed 42 for E; one active GPU;
controller only via ~/deepmzyme-vm/bin (vm-status, vm-report --no-ssh, vm-start --confirm [--hours H], vm-setup
--stages ssh,smoke, vm-stop, vm-status); session cap 4 h / $6, daily cap 12 h / $18 (UTC day); campaign ceiling $40
gross including storage (about $12.9 left on 2026-10-08 12:17 UTC; see handoff.md section 5); the $55 gross
planning ceiling selected for E2 (log v3-025) is recorded in the plan and log only, not in any execution control,
and authorizes no GPU start;
every session ends with `pmm_v3_step_d.py --step E evidence`, vm-stop and a verified TERMINATED; log + STATUS +
commit after each session with `git -c core.hooksPath=.githooks commit`; push only with the user's OK. Never edit
pmm_v3_campaign.py or run_pmm_v3_campaign.py (hash-frozen). Never run vm-fallback, delete the failed-fallback
leftovers, raise controller caps, run the cost-gated augmentations or Round C, or start step F without the user.

Authorization to check first: the user's single-session arrangement of 2026-10-08 (log v3-024) ended with that
session's closeout. The standing advance authorization (log v3-014) covers the approved plan through F within the
ceiling, but ask the user to confirm the next GPU start and its time window before starting the VM. A deferred
read-only review (10-epoch schedule amendment; audits/session_notes_20261008/ on the campaign data root) is due
after the next closeout, with no implementation or spending. E2 fits need, beyond the standing authorization,
the CPU tooling, a frozen E2 specification and chained manifest, a refreshed forecast, the ceiling recorded in
execution controls and the user's explicit go-ahead.

Next task (user, 2026-10-08): "Read the saved review, discuss the unresolved assessment tradeoffs with the user, agree
any changes, then update the documentation before implementation or training." The review is handoff.md section 9
and log v3-026. Nothing of E2 (specification freeze, implementation, training) starts before that discussion; the
eight candidates, 36-fit scope, seed 42 and the $55 planning ceiling stay as recorded. A proposed resolution with
options (A as drafted, B recommended, C in between) is recorded in log v3-027 and section 9; the user's choice is pending.

Session procedure: vm-status; vm-report --no-ssh; vm-start --confirm; on a stockout, spaced same-VM retries only;
vm-setup --stages ssh,smoke; `pmm_v3_step_d.py --step E status` (reconciles: completed units are never rerun;
a lane awaiting its pull is pulled by `wait`/`pull --lanes K`; an interrupted lane needs `recover-lane K` then
`archive-failed UNIT` and one unchanged rerun); launch units three at a time with
`pmm_v3_step_d.py --step E launch U@0 U@1 U@2` (waits and pulls) or `--no-wait` + `wait --any <every running unit by
name>` (a bare `wait --any` misses a unit that ended before its first snapshot; log v3-024), admission
1.25 × forecast + 960 s before the hard stop; do not reuse the scratch refill drivers unless the user has reviewed
the audit corrections (handoff.md section 6). After every session: `--step E assess`, `--step E evidence`,
vm-stop, vm-status, vm-report --no-ssh; then log entry (newest first), STATUS (≤ 6,000 bytes; preserve the replaced
dated text in the log), tools/check_docs_contract.py, explicit `git add -- <paths>`, commit with the hook.

Open user decisions (never infer them): Only-ESMC fairness arm (proposal in log v3-022; neither approved nor
rejected); cleanup of the fallback leftovers; cost-gated augmentations; step F plan and test-access ledger entries;
the go-ahead for the E2 implementation and, later, for its GPU fits (or a dated cancellation of E2).
At each closeout report completed/pending units, spending and forecast, evidence copies, verified VM state, the
exact next action and any decision the user must make.
```

## 1. Where to work and what to read

| Item | Value |
|---|---|
| Worktree | `/media/mechti/Data1/DeepMzyme_worktrees/v3` (branch `v3-step-a`; HEAD = the commit of log v3-024 (see `git log -1`); `origin/v3-step-a` = same) — not `/home/mechti/PycharmProjects/DeepMzyme` (older state) |
| Python | `/home/mechti/miniconda3/envs/DeepMzyme/bin/python` (verify with `-c "import sys; print(sys.executable)"`); local CPU runs capped with `systemd-run --user --scope -p CPUQuota=200% -p MemoryMax=3G` |
| Campaign | `pmm_ion_metal_v3`; docs `docs/campaigns/pmm_ion_metal_v3/` (README, plan.md, assessment_spec.md, log.md, this file); data `/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v3` (the Data1 drive is `/dev/sda1`, label `Data`; after a reboot mount it with `udisksctl mount -b /dev/sda1`) |
| Reading order | AGENTS.md → EXPERIMENT_STATUS.md → docs/campaigns/pmm_ion_metal_v3/README.md → handoff.md section 9 (pending E2 review) → plan.md (steps D–F, E2 included) → assessment_spec.md → log.md entries v3-026 back to v3-017 → docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md "v3 step D regularization amendment" → .agents/skills/gpu-use-skill/SKILL.md → ~/deepmzyme-vm/AGENTS.md and config.env (before any GPU action) |
| Launchers | `pmm_v3_step_d.py` (steps D and E: units, gates, status, launch, wait, pull, recover-lane, archive-failed, assess, record-cost-gate, record-e-gate, evidence); `pmm_v3_step_c.py`/`pmm_v3_step_b.py` (shared lane logic); never edit `pmm_v3_campaign.py` or `run_pmm_v3_campaign.py` (hash-frozen by the extension record) |
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
  assessment rules pending review, log v3-026 and section 9) → F Stage 6 selection (the final E2 selection, or the
  original A4 selection if the user explicitly cancels E2), Stage 6B full refit, one test pass. Steps A–D are
  complete.
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
- **Step E2 amendment (planned 2026-10-08, log v3-025; not implemented):** after the original E assessment, the
  eight step D passes not chosen as final recipes (Only-GVP `wd10`, `sitecountsangles`; late fusion `invsqrtw`,
  `meanagg`, `gvpaux03`, `resdrop01`, `esmdrop02`, `wd10`) each on folds 1–4, seed 42, `four_class`, 50-epoch cosine,
  terminal checkpoint, against the reused step E baselines (`sitenone` × 4 as the geometry control): 36 fits, 17
  predeclared comparisons (ordinary and Bonferroni-adjusted A4 intervals), a proposed replacement rule kept in a
  separate E2 specification (A4 unchanged); the E2 assessor must accept a four-, five- or six-class original
  selection (the existing `compare` assumes a four-class control). Development-validation evidence, proposed after
  fold 1–4 results were seen. Step F requires E2 completion unless the user cancels E2. Rules: plan.md step E2;
  tooling gaps: the playbook's E2 section.
- **Persistence contract:** every fit → run_status written → host-pull manifest (SHA-256 per file) → worker exit
  code → host `pull` verifies and uploads the acknowledgment (lane closed until then) → `evidence` copies the step
  evidence to Data1 with SHA-256 checked on both ends; `assess` keeps a checked copy of each assessment. A unit is
  "complete" only when all four happened: training completed, worker exited 0, artifacts persisted, host acknowledged.

## 3. Work state (verified 2026-10-08 12:17 UTC)

**Step D: complete (final assessment `D-B_20261007T201334Z`, status `final`).** Every listed unit below is complete
in all four senses (training completed at epoch 50, worker exit code 0, artifacts persisted by host pull with
SHA-256 manifests, host acknowledgment uploaded) and replayed under `pmm-v3-replay-1`; the pulled copies are in
`$DATA/durable/lane*/runs/<unit>/`. Nothing is running; no lane awaits a pull; no unit failed in steps D or E.

| Round | Units | State |
|---|---|---|
| Controls | step C seed-42 baselines (3 families × 3 targets, 9 runs, `fold0`) + Round A seed-43 four-class baselines (`only_gvp__four_class__baseline__fold0__seed43`, `gvp_late_fusion__four_class__baseline__fold0__seed43`) | complete |
| D-A (16) | `{only_gvp,gvp_late_fusion}__four_class__{meanagg,resdrop01}__fold0__seed{42,43}`, `gvp_late_fusion__four_class__{structlr,gvpaux03,esmdrop02}__fold0__seed{42,43}` + the two seed-43 controls | complete; passed: meanagg (both), fusion resdrop01/gvpaux03/esmdrop02 |
| D-R (26) | `{only_gvp,gvp_late_fusion}__four_class__{wd001,wd01,wd10,headdrop01,headdrop03,resdrop02}__fold0__seed{42,43}`, `gvp_late_fusion__four_class__esmdrop04__fold0__seed{42,43}` | complete; passed: wd10 (both), fusion headdrop03 |
| D-B (14 of 18) | `{only_gvp,gvp_late_fusion}__four_class__{sitenone,sitecountsangles,invsqrtw}__fold0__seed{42,43}`, `only_gvp__four_class__vecnorm__fold0__seed{42,43}` | complete; passed: fusion invsqrtw, Only-GVP sitecountsangles (vs sitenone) |
| D-B cost-gated (4) | `only_gvp__four_class__{posnoise01,outerdrop01}__fold0__seed{42,43}` | never run: "not tested (cost)" (user OK never given) |
| D combination (4) | `gvp_late_fusion__four_class__combo-gvpaux03+headdrop03+invsqrtw+meanagg+resdrop01+wd10__fold0__seed{42,43}` (+2.82/+3.69, mean +3.25 < headdrop03 +3.58), `only_gvp__four_class__combo-meanagg+sitecountsangles+wd10__fold0__seed{42,43}` (−1.96/+6.29, mean +2.17, fails the both-seeds rule) | complete; **final recipes: late fusion `headdrop03`, Only-GVP `meanagg`, Only-ESMC baseline** |

**Step E: in progress, 24 of 44 units complete** (sessions 6–7 of 2026-10-08, log v3-023/v3-024; gate recorded: final-test label `both_results_secondary`, `step_e_evidence/step_e_gate.json`, log v3-022; final D assessment `D-B_20261007T201334Z` is final). Order used so far: the four-class baselines of all three families on folds 1–4, the 8 improvement fits, then five/six-class; the launcher forecasts five/six-class units as warm (fold-0 records exist), only the parse is cold (6–15 min); the five+ESMC, five-noESMC and six+ESMC parse caches are built, six-noESMC (Only-GVP six_class) is not.

| Group | Units (seed 42, folds 1–4) | State |
|---|---|---|
| E neutral, four_class (12) | `{only_esm,only_gvp,gvp_late_fusion}__four_class__baseline__fold{1,2,3,4}__seed42` | complete (all 12) |
| E neutral, five_class (12) | `{only_esm,only_gvp,gvp_late_fusion}__five_class__baseline__fold{1,2,3,4}__seed42` | complete: gvp_late_fusion fold1, only_gvp fold1, only_esm fold1; the rest not started |
| E neutral, six_class (12) | `{only_esm,only_gvp,gvp_late_fusion}__six_class__baseline__fold{1,2,3,4}__seed42` | complete: only_esm fold1; the rest not started |
| E improvement (8) | `gvp_late_fusion__four_class__headdrop03__fold{1,2,3,4}__seed42`, `only_gvp__four_class__meanagg__fold{1,2,3,4}__seed42` | complete (all 8); descriptive paired differences on folds 1–4: headdrop03 +0.10/−1.19/−0.51/+0.26, meanagg +0.90/+3.77/−2.04/−1.20 (the assessor decides; log v3-024) |

Fold 0 of every E contrast reuses the step C / step D runs (never rerun). **Step F:** not started; tooling absent;
the draft `audits/session_notes_20261008/step_f_plan_DRAFT.md` (campaign data root) takes the A4 selection as its
only input and REQUIRES RECONCILIATION with E2 before implementation (final E2 selection, or A4 if E2 is cancelled).
**Step E2:** planned only (log v3-025): 0 of 36 fits; no launcher, assessor, manifest, specification or evidence
directory exists; `pmm_v3_step_d.py --step E` refuses E2 units (the VM runner alone would not); it starts only after
the original E assessment, the CPU implementation and checks, a refreshed forecast, the recorded ceiling and the
user's go-ahead, and step F waits for it unless the user cancels E2.

## 4. User decisions and authorization boundaries

| Date (UTC) | Decision (actual, by the user) | Where recorded |
|---|---|---|
| 2026-10-03 | Objectives four/five/six for all three families; v3 campaign opened; PMM core v2 closed at fold 0 | README, log v3-001…003 |
| 2026-10-05/06 | A4 rules frozen; checkpoint rule terminal epoch; $40 gross ceiling incl. storage; advance GPU authorization for the approved plan through F (no routine start prompts); test-row rule | log v3-003, v3-012, v3-014 |
| 2026-10-06 | Regularization amendment (Round R) and its implementation; daily controller caps raised to 12 h / $18; bounded fallback pass authorized (failed; leftovers kept pending the user's cleanup decision) | log v3-017…v3-019 |
| 2026-10-07 | Only-ESMC fairness assessment requested as a proposal only (training awaits approval) | log v3-021 |
| 2026-10-08 morning | Next GPU start confirmed; up to three capped sessions with closeout by 17:30 UTC, then (PC closing) one session only; commit and push at every closeout (today); two read-only reviewers; a deferred read-only review of a 10-epoch schedule amendment after the next closeout (no implementation or spending); the arrangement is for 2026-10-08 only | log v3-024 |
| 2026-10-08 midday | Step E2 amendment recorded as planned, not implemented: the eight remaining step D passes tested separately on folds 1–4 (36 fits, seed 42), proposed selection rules kept separate from A4, E2 completion required before step F unless the user cancels E2, a $55 gross total planning ceiling for the expanded campaign; documentation only, no GPU start authorized, execution controls unchanged | log v3-025, plan.md step E2 |
| 2026-10-07 evening | Final-test label `both_results_secondary`; continue the approved plan until 03:38 UTC 2026-10-08 at the latest (eight hours maximum, not a target), step E after midnight UTC if gates pass, caps and ceiling unchanged; fairness arm kept separate for the user's decision; audit of the scratch drivers delivered before any correction or driver reuse; closeout commits pushed to `origin/v3-step-a`; a complete handoff for a new chat | log v3-022 |

**Authorization boundaries:** GPU starts for the approved D–F work are pre-authorized within the $40 gross ceiling
(storage included) and the controller caps (4 h / $6 per session, 12 h / $18 per UTC day); one active GPU, controller
only via `~/deepmzyme-vm/bin`; every session ends with `evidence`, `vm-stop` and a verified TERMINATED. Never:
Round C, cost-gated augmentations (`posnoise01`, `outerdrop01`) without a recorded OK, `vm-fallback`, deleting the
failed-fallback leftovers, editing the frozen runner files, editing controller caps, pushing without the user's OK
(granted for the 2026-10-07/08 closeouts only), step F before its gates and plan. **Decisions still open:** fallback
leftovers cleanup (two 150 GB recovery disks `deepmzyme-l4-recovery-873d08b26968` in us-central1-b/c and snapshot
`deepmzyme-fallback-873d08b26968`, about $1.06/day); Only-ESMC fairness arm; cost-gated augmentations; step F plan;
the E2 implementation go-ahead and, later, its GPU go-ahead (or a dated cancellation of E2).
The 2026-10-08 single-session arrangement ended with session 7's closeout: any later execution needs the user's
renewed go-ahead (the standing advance authorization of v3-014 covers the approved plan, but the user asked to be
told what must be renewed, so ask before the next GPU start). Push authorization was given for the 2026-10-08
closeouts only.

## 5. VM, spending and limits (verified 2026-10-08 12:17 UTC)

| Item | Value (verified 2026-10-07 20:16:45 UTC by `vm-status`/`vm-report --no-ssh`) |
|---|---|
| VM | `deepmzyme-l4`, project `deepmzyme-gpu-vm`, zone `us-central1-a`, g2-standard-8 + 1× L4, **TERMINATED** (ledger STOPPED 2026-10-08 12:16:58 UTC, session `session-20261008T084732Z-3bbc7d8d`, 3 h 29 min, $3.07 gross); 150 GB boot disk kept (`deepmzyme-l4-from-deepmzyme-paused-20261003`) |
| Spending | compute to date $23.55 gross (B $1.88, C $3.64, D $12.98, E $5.05); storage about $3.5 accrued; **about $27.1 spent, about $12.9 of the $40 gross ceiling left**; forecast to finish E and F $33.5–36.4 (upper value while the leftovers exist); E2 planning (log v3-025): about 9–12 additional VM hours, total about $44–51 against the selected $55 gross planning ceiling (recorded in plan and log only; no execution control; no start authorized) |
| Retained storage | campaign disk $15/month + snapshot `deepmzyme-paused-20261003` (reserved $7.50/month) + failed-fallback leftovers: disks `deepmzyme-l4-recovery-873d08b26968` in us-central1-b and -c ($15/month each) and snapshot `deepmzyme-fallback-873d08b26968` ($7.50/month reserved) → controller reservation about $2.00/day in total; leftovers cleanup is the user's decision (never delete them yourself; `state/fallback.json` is a stuck receipt, do not edit) |
| Limits | controller caps 4 h / $6 per session, 12 h / $18 per UTC day (`~/deepmzyme-vm/config.env`; never edit without the user's instruction); 2026-10-08 used 5 h 44 min / $7.40 (the day is over for GPU work: the PC closes about 13:07 UTC) |
| Stockouts | L4 `ZONE_RESOURCE_POOL_EXHAUSTED` in us-central1-a is frequent (five on 2026-10-07, one at 00:01 and eight in a row 07:58–08:40 on 2026-10-08); the handoff rule is spaced same-VM retries only (5-minute spacing, bounded), never `vm-fallback` |

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
| Step D evidence copies (statuses, launch records, lane state, events) | `$DATA/step_d_evidence/evidence_<UTC>/` (latest D: `evidence_20261007T201431Z`, 682 files; E: `evidence_20261008T022159Z (457 files)`) |
| Step D assessments (checked copies; SHA-256 in the launcher output and the log) | `$DATA/step_d_evidence/assessments/D-A_*/`, `D-R_*/`, `D-B_*/assessment.json` (final: `D-B_20261007T201334Z`, SHA-256 `bba7c728…`) |
| Extension record copy, readiness and decay audits, early-stopping and train/val-gap audits | `$DATA/step_d_evidence/extension/`, `$DATA/audits/d_readiness_20261006T180939Z/`, `d_decay_audit_20261006T175933Z/`, `early_stopping_audit_20261007T111826Z/`, `train_val_gap_20261007T111847Z/` |
| Scratch-driver audit (report, mock, tools, logs, interim drafts) | `$DATA/audits/driver_audit_20261007T193000Z/` |
| Step E gate record (label) and future E evidence | `$DATA/step_e_evidence/step_e_gate.json`; `$DATA/step_e_evidence/evidence_<UTC>/` (latest `evidence_20261008T022159Z (457 files)`), assessments `$DATA/step_e_evidence/assessments/` |
| Bundles (source + extension) and folds | `$DATA/bundles/20d5b06/`, `$DATA/bundles/43d26ff/`; `$DATA/folds/` |
| Steps B and C evidence | `$DATA/step_b_evidence/`, `$DATA/step_c_evidence/` |
| VM-side originals (kept on the stopped 150 GB disk) | `/home/mechti/deepmzyme_runs/pmm_ion_metal_v3/{lanes,step_d,step_e,assessments,campaign_extension.json}`; code `/home/mechti/projects/DeepMzyme_v3_ext1`; caches `/home/mechti/deepmzyme_cache/{parse,esm,ring}` |
| Controller state and ledger | `~/deepmzyme-vm/state/usage_ledger.jsonl`, `current_session.json`, `fallback.json` (stuck pass 873d08b26968, do not edit), `config.env` |
| Git | worktree `/media/mechti/Data1/DeepMzyme_worktrees/v3`, branch `v3-step-a`, remote `origin` (same repository as `/home/mechti/PycharmProjects/DeepMzyme`) |

Recovery: if the workstation was rebooted, mount Data1 (`udisksctl mount -b /dev/sda1`), confirm the worktree with
`git -C /media/mechti/Data1/DeepMzyme_worktrees/v3 status`, then read STATUS and this file. If a VM session was cut
(hard stop, SSH loss): start the VM under authorization, run `P pmm_v3_step_d.py --step E status`; `wait` pulls every
lane awaiting acknowledgment; `recover-lane K` records an interrupted unit (then `archive-failed UNIT` and one unchanged
rerun); never relaunch a unit that has a launch record without an exit code. Run `evidence` before every `vm-stop`.

## 8. Next actions and commands

**Next task (user, 2026-10-08):** "Read the saved review, discuss the unresolved assessment tradeoffs with the user,
agree any changes, then update the documentation before implementation or training." The review is section 9 (also
log v3-026). The E2 assessment rules are not frozen; nothing of E2 starts before that discussion. The steps below
describe the original step E work and the later E2 route.

Reconcile first (read-only): `cd /media/mechti/Data1/DeepMzyme_worktrees/v3; git status; git log -3`;
`~/deepmzyme-vm/bin/vm-status` and `vm-report --no-ssh`; if the VM is RUNNING unexpectedly, do not start anything,
run `P pmm_v3_step_d.py --step E status` (needs SSH to a running VM) and `wait` (never relaunch), then `evidence`
and `vm-stop`. If TERMINATED, the campaign root on the VM disk and the Data1 evidence copies are the truth; the
launcher's `status` after the next start lists completed units; completed units are never rerun (launcher refusal),
a failed unit gets one unchanged rerun via `archive-failed UNIT` then `launch`.

Step E session (P = /home/mechti/miniconda3/envs/DeepMzyme/bin/python, in the worktree):
1. `vm-status`; `vm-report --no-ssh`; `vm-start --confirm [--hours H]` (on a stockout: spaced same-VM retries only);
   `vm-setup --stages ssh,smoke`.
2. `P pmm_v3_step_d.py --step E status` (gates: readiness, extension, label record, final D assessment).
3. Rounds of three: `P pmm_v3_step_d.py --step E launch U@0 U@1 U@2` (waits and pulls; admission 1.25×forecast+960 s;
   five/six-class units are forecast warm by the launcher; a cold parse adds 6–15 min, so launch a cold builder by
   session minute ~160), or `--no-wait` then `wait --any <every running unit by name>` and refill after `status`;
   if a launch is refused with "awaits its verified host pull", run `pull --lanes K` and retry.
4. After every session: `P pmm_v3_step_d.py --step E assess` (allowed at any time; incomplete cells are reported with
   their missing units), `P pmm_v3_step_d.py --step E evidence`, `vm-stop`, `vm-status` (TERMINATED), `vm-report --no-ssh`.
5. Log a dated entry (newest first in log.md), overwrite STATUS (≤ 6,000 bytes; preserve replaced dated text in the log),
   `P tools/check_docs_contract.py`, `git add -- <paths>`, `git -c core.hooksPath=.githooks commit`, push only with the user's OK.
6. Step E2, only after the original E assessment and on CPU first (no GPU): build what the playbook's E2 section lists
   (frozen E2 specification, chained amendment manifest, launcher with the admission check the VM runner lacks,
   assessor accepting a four-/five-/six-class original selection, tests), run the existing launcher and assessment
   tests, the smoke checks and the docs-contract check, refresh the forecast, record the ceiling in execution
   controls, then ask the user for the go-ahead. No E2 command exists yet; never run E2 units through `--step E` or
   the VM runner. Step F waits for E2 unless the user cancels E2 by a dated log entry.

## 9. Pending review of the E2 assessment rules (saved 2026-10-08; unresolved)

Saved in substance from the user's request of 2026-10-08 (also [log v3-026](log.md#v3-026)). The eight separate
experiments and the 36-fit E2 design remain the intended scope (eight candidates, seed 42, $55 gross total planning
ceiling; the planning ceiling does not authorize a GPU start). E2 is planned, not implemented and not experimentally
evaluated; its assessment rules (plan.md step E2, items 2 and 3) are pending this review and are not frozen. The
scientific questions below are NOT resolved; they are discussed with the user before the E2 specification is frozen
or any implementation or training starts.

1. **Promotion threshold.** Bonferroni correction across 17 comparisons with only four folds creates a very demanding
   replacement gate, approximately 99.7% confidence intervals per comparison. A hypothetical improvement of +1, +2, +3
   and +4 percentage points averages +2.5 points but still fails the adjusted t-interval check. This is recorded as an
   unresolved design tradeoff. Do not silently change the threshold, comparison family, folds, seeds or budget; discuss
   it with the user before freezing the specification, and never change it after seeing E2 results to obtain a
   preferred outcome.
2. **Separate experiment conclusions.** Each candidate needs two separate conclusions: (a) does this change help its
   own model compared with its matched control? (b) does it qualify to replace the overall selected model? Failure to
   replace the overall winner must not be interpreted as "the change is useless". Positive average gains must not
   conceal failed class-recall checks.
3. **Documentation consistency.** The handoff's old E → F sequence and reading list are reconciled with E → E2 → F and
   log v3-025 (sections 1, 2 and 8 above). The existing step F draft (`audits/session_notes_20261008/step_f_plan_DRAFT.md`
   on the campaign data root) takes the A4 selection as its only input and is marked as requiring reconciliation before
   implementation: step F must consume the final E2 selection, or the original A4 selection if the user explicitly
   cancels E2.
4. **Interpretation.** Preserve the development-validation/exploratory label. Four folds and one seed do not establish
   robustness across training seeds, and the adjusted intervals do not remove the uncertainty caused by reused
   validation data and overlapping cross-validation training sets.

Next task: "Read the saved review, discuss the unresolved assessment tradeoffs with the user, agree any changes, then
update the documentation before implementation or training."

**Proposed resolution (recorded 2026-10-08 about 16:30 UTC in [log v3-027](log.md#v3-027); for the user's decision;
nothing frozen).** Verified basis: A4 decides every verdict on the unadjusted 95% intervals plus the recall gates and
only reports the Bonferroni-adjusted ones, so the drafted E2 rule 3(b) is stricter than any A4 verdict; with four folds
the adjusted t critical value is 6.9 (over 8) or 8.95 (over 17) against 3.18 unadjusted, the percentile bootstrap cannot
reach beyond the fold minimum, and the power of the adjusted rules at a +2-point gain is 2–22% against 29–75% unadjusted.
Options for point 1: A (as drafted, Bonferroni over 17), **B (recommended: the A4 verdict rule unchanged, adjusted
intervals reported as sensitivity, plus the 0.2-point minimum gain, both comparators and the recall gates; family-wise
false-replacement rate about 0.18 disclosed)**, C (B plus Bonferroni over the replacement comparisons only). Point 2:
two labelled conclusions per candidate in the A4 vocabulary, recall-gate failures printed in both. Point 3: no decision
needed (step F draft reconciled with the step F plan). Point 4: label accepted; optional seed-43 repeat of a selected
replacement only (four fits) if the user adds it. Decisions requested: option A/B/C, the vocabulary, the extra seed,
and the next GPU start and window for the 20 remaining step E fits.
