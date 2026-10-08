# PMM ion metal v3 — continuation handoff (written 2026-10-08 02:30 UTC)

Self-contained handoff for a new Claude Code chat. It depends on no earlier
conversation and on no temporary file; every path below is durable (git,
`/media/mechti/Data1`, `~/deepmzyme-vm`). Newest state first; the
[log](log.md) holds the dated history and [STATUS](../../../EXPERIMENT_STATUS.md)
the current authority.

## Ready-to-paste continuation prompt

```text
Continue the DeepMzyme PMM ion metal v3 campaign (step E, five-fold confirmation) from the verified
stopping point recorded in docs/campaigns/pmm_ion_metal_v3/handoff.md.

Work only in the worktree /media/mechti/Data1/DeepMzyme_worktrees/v3 (branch v3-step-a; HEAD must equal
origin/v3-step-a; `git status` clean). Not in /home/mechti/PycharmProjects/DeepMzyme. If /media/mechti/Data1 is
missing, mount it first: `udisksctl mount -b /dev/sda1`. Python: /home/mechti/miniconda3/envs/DeepMzyme/bin/python
(verify with -c "import sys; print(sys.executable)"). Cap local CPU runs: systemd-run --user --scope -p CPUQuota=200%
-p MemoryMax=3G.

Read in order: AGENTS.md; EXPERIMENT_STATUS.md; docs/campaigns/pmm_ion_metal_v3/README.md; handoff.md (all
sections); plan.md steps E and F; assessment_spec.md; log.md entries v3-023 back to v3-017;
docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md "v3 step D regularization amendment"; .agents/skills/gpu-use-skill/SKILL.md;
~/deepmzyme-vm/AGENTS.md and config.env before any GPU action.

State: steps A–D complete; final step D recipes late fusion `headdrop03`, Only-GVP `meanagg`, Only-ESMC baseline
(final assessment D-B_20261007T201334Z). Step E: 36 neutral-test fits (seed 42, folds 1–4, three families × four/
five/six) + 8 improvement fits (gvp_late_fusion__four_class__headdrop03__fold{1-4}__seed42,
only_gvp__four_class__meanagg__fold{1-4}__seed42); 10 of 44 are complete (section 3 lists them); the rest are pending, none failed. Final-test label
both_results_secondary is recorded (record-e-gate done). Step F has no tooling yet: plan it and get the user's
approval before any refit or test access.

Rules that stay fixed: 50-epoch cosine, terminal checkpoint; A4 gates unchanged; seed 42 for E; one active GPU;
controller only via ~/deepmzyme-vm/bin (vm-status, vm-report --no-ssh, vm-start --confirm [--hours H], vm-setup
--stages ssh,smoke, vm-stop, vm-status); session cap 4 h / $6, daily cap 12 h / $18 (UTC day); campaign ceiling $40
gross including storage (about $16.8 left on 2026-10-08 02:30 UTC; see handoff.md section 5);
every session ends with `pmm_v3_step_d.py --step E evidence`, vm-stop and a verified TERMINATED; log + STATUS +
commit after each session with `git -c core.hooksPath=.githooks commit`; push only with the user's OK. Never edit
pmm_v3_campaign.py or run_pmm_v3_campaign.py (hash-frozen). Never run vm-fallback, delete the failed-fallback
leftovers, raise controller caps, run the cost-gated augmentations or Round C, or start step F without the user.

Authorization to check first: the user's working window of 2026-10-07/08 ended at 03:38 UTC on 2026-10-08. The
standing advance authorization (log v3-014) covers the approved plan through F within the ceiling, but ask the
user to confirm the next GPU start and its time window before starting the VM.

Session procedure: vm-status; vm-report --no-ssh; vm-start --confirm; on a stockout, spaced same-VM retries only;
vm-setup --stages ssh,smoke; `pmm_v3_step_d.py --step E status` (reconciles: completed units are never rerun;
a lane awaiting its pull is pulled by `wait`/`pull --lanes K`; an interrupted lane needs `recover-lane K` then
`archive-failed UNIT` and one unchanged rerun); launch units three at a time with
`pmm_v3_step_d.py --step E launch U@0 U@1 U@2` (waits and pulls) or `--no-wait` + `wait --any`, admission
1.25 × forecast + 960 s before the hard stop; do not reuse the scratch refill drivers unless the user has reviewed
the audit corrections (handoff.md section 6). After every session: `--step E assess`, `--step E evidence`,
vm-stop, vm-status, vm-report --no-ssh; then log entry (newest first), STATUS (≤ 6,000 bytes; preserve the replaced
dated text in the log), tools/check_docs_contract.py, explicit `git add -- <paths>`, commit with the hook.

Open user decisions (never infer them): Only-ESMC fairness arm (proposal in log v3-022; neither approved nor
rejected); cleanup of the fallback leftovers; cost-gated augmentations; step F plan and test-access ledger entries.
At each closeout report completed/pending units, spending and forecast, evidence copies, verified VM state, the
exact next action and any decision the user must make.
```

## 1. Where to work and what to read

| Item | Value |
|---|---|
| Worktree | `/media/mechti/Data1/DeepMzyme_worktrees/v3` (branch `v3-step-a`; HEAD = the commit of log v3-023 (see `git log -1`); `origin/v3-step-a` = same) — not `/home/mechti/PycharmProjects/DeepMzyme` (older state) |
| Python | `/home/mechti/miniconda3/envs/DeepMzyme/bin/python` (verify with `-c "import sys; print(sys.executable)"`); local CPU runs capped with `systemd-run --user --scope -p CPUQuota=200% -p MemoryMax=3G` |
| Campaign | `pmm_ion_metal_v3`; docs `docs/campaigns/pmm_ion_metal_v3/` (README, plan.md, assessment_spec.md, log.md, this file); data `/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v3` (the Data1 drive is `/dev/sda1`, label `Data`; after a reboot mount it with `udisksctl mount -b /dev/sda1`) |
| Reading order | AGENTS.md → EXPERIMENT_STATUS.md → docs/campaigns/pmm_ion_metal_v3/README.md → plan.md (steps D–F) → assessment_spec.md → log.md entries v3-023 back to v3-017 → docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md "v3 step D regularization amendment" → .agents/skills/gpu-use-skill/SKILL.md → ~/deepmzyme-vm/AGENTS.md and config.env (before any GPU action) |
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
  → F Stage 6 selection by A4, Stage 6B full refit, one test pass. Steps A–D are complete.
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
- **Persistence contract:** every fit → run_status written → host-pull manifest (SHA-256 per file) → worker exit
  code → host `pull` verifies and uploads the acknowledgment (lane closed until then) → `evidence` copies the step
  evidence to Data1 with SHA-256 checked on both ends; `assess` keeps a checked copy of each assessment. A unit is
  "complete" only when all four happened: training completed, worker exited 0, artifacts persisted, host acknowledged.

## 3. Work state (verified 2026-10-08 02:30 UTC)

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

**Step E: in progress, 10 of 44 units complete** (session 6 of 2026-10-08, log v3-023; gate recorded: final-test label `both_results_secondary`, `step_e_evidence/step_e_gate.json`, log v3-022; final D assessment `D-B_20261007T201334Z` is final). Order used so far: the four-class baselines of all three families on folds 1–4 (the paired baselines of the improvement check); next the remaining four-class baselines, then the 8 improvement fits, then five/six-class (cache sets cold once).

| Group | Units (seed 42, folds 1–4) | State |
|---|---|---|
| E neutral, four_class (12) | `{only_esm,only_gvp,gvp_late_fusion}__four_class__baseline__fold{1,2,3,4}__seed42` | complete: only_esm fold1, only_esm fold2, only_esm fold3, only_esm fold4, only_gvp fold1, only_gvp fold2, only_gvp fold3, gvp_late_fusion fold1, gvp_late_fusion fold2, gvp_late_fusion fold3 |
| E neutral, five_class (12) | `{only_esm,only_gvp,gvp_late_fusion}__five_class__baseline__fold{1,2,3,4}__seed42` (cache sets cold once) | not started |
| E neutral, six_class (12) | `{only_esm,only_gvp,gvp_late_fusion}__six_class__baseline__fold{1,2,3,4}__seed42` (cache sets cold once) | not started |
| E improvement (8) | `gvp_late_fusion__four_class__headdrop03__fold{1,2,3,4}__seed42`, `only_gvp__four_class__meanagg__fold{1,2,3,4}__seed42` | not started |

Fold 0 of every E contrast reuses the step C / step D runs (never rerun). **Step F:** not started; tooling absent.

## 4. User decisions and authorization boundaries

| Date (UTC) | Decision (actual, by the user) | Where recorded |
|---|---|---|
| 2026-10-03 | Objectives four/five/six for all three families; v3 campaign opened; PMM core v2 closed at fold 0 | README, log v3-001…003 |
| 2026-10-05/06 | A4 rules frozen; checkpoint rule terminal epoch; $40 gross ceiling incl. storage; advance GPU authorization for the approved plan through F (no routine start prompts); test-row rule | log v3-003, v3-012, v3-014 |
| 2026-10-06 | Regularization amendment (Round R) and its implementation; daily controller caps raised to 12 h / $18; bounded fallback pass authorized (failed; leftovers kept pending the user's cleanup decision) | log v3-017…v3-019 |
| 2026-10-07 | Only-ESMC fairness assessment requested as a proposal only (training awaits approval) | log v3-021 |
| 2026-10-07 evening | Final-test label `both_results_secondary`; continue the approved plan until 03:38 UTC 2026-10-08 at the latest (eight hours maximum, not a target), step E after midnight UTC if gates pass, caps and ceiling unchanged; fairness arm kept separate for the user's decision; audit of the scratch drivers delivered before any correction or driver reuse; closeout commits pushed to `origin/v3-step-a`; a complete handoff for a new chat | log v3-022 |

**Authorization boundaries:** GPU starts for the approved D–F work are pre-authorized within the $40 gross ceiling
(storage included) and the controller caps (4 h / $6 per session, 12 h / $18 per UTC day); one active GPU, controller
only via `~/deepmzyme-vm/bin`; every session ends with `evidence`, `vm-stop` and a verified TERMINATED. Never:
Round C, cost-gated augmentations (`posnoise01`, `outerdrop01`) without a recorded OK, `vm-fallback`, deleting the
failed-fallback leftovers, editing the frozen runner files, editing controller caps, pushing without the user's OK
(granted for the 2026-10-07/08 closeouts only), step F before its gates and plan. **Decisions still open:** fallback
leftovers cleanup (two 150 GB recovery disks `deepmzyme-l4-recovery-873d08b26968` in us-central1-b/c and snapshot
`deepmzyme-fallback-873d08b26968`, about $1.06/day); Only-ESMC fairness arm; cost-gated augmentations; step F plan.
The working-window authorization expires at 03:38 UTC on 2026-10-08: any execution after that needs the user's
renewed go-ahead (the standing advance authorization of v3-014 covers the approved plan, but the user asked to be
told what must be renewed, so ask before the next GPU start).

## 5. VM, spending and limits (verified 2026-10-08 02:30 UTC)

| Item | Value (verified 2026-10-07 20:16:45 UTC by `vm-status`/`vm-report --no-ssh`) |
|---|---|
| VM | `deepmzyme-l4`, project `deepmzyme-gpu-vm`, zone `us-central1-a`, g2-standard-8 + 1× L4, **TERMINATED** (ledger STOPPED 2026-10-08 02:23:45 UTC, session `session-20261008T000829Z-f23d379e`, 2 h 15 min, $1.98 gross); 150 GB boot disk kept (`deepmzyme-l4-from-deepmzyme-paused-20261003`) |
| Spending | compute to date $20.48 gross (B $1.88, C $3.64, D $12.98, E $1.98); storage about $2.7 accrued; **about $23.2 spent, about $16.8 of the $40 gross ceiling left**; forecast to finish E and F $33.3–37.1 (upper value while the leftovers exist) |
| Retained storage | campaign disk $15/month + snapshot `deepmzyme-paused-20261003` (reserved $7.50/month) + failed-fallback leftovers: disks `deepmzyme-l4-recovery-873d08b26968` in us-central1-b and -c ($15/month each) and snapshot `deepmzyme-fallback-873d08b26968` ($7.50/month reserved) → controller reservation about $2.00/day in total; leftovers cleanup is the user's decision (never delete them yourself; `state/fallback.json` is a stuck receipt, do not edit) |
| Limits | controller caps 4 h / $6 per session, 12 h / $18 per UTC day (`~/deepmzyme-vm/config.env`; never edit without the user's instruction); 2026-10-08 used 2 h 15 min / $4.40 so far (9 h 45 min left today) |
| Stockouts | L4 `ZONE_RESOURCE_POOL_EXHAUSTED` in us-central1-a is frequent (four on 2026-10-07 before 17:00); the handoff rule is spaced same-VM retries only (5-minute spacing, bounded), never `vm-fallback` |

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
   five/six-class cache sets build cold once, forecast +2000 s, measured about +2 min), or `--no-wait` then
   `wait --any` and refill; if a launch is refused with "awaits its verified host pull", run `pull --lanes K` and retry.
4. After every session: `P pmm_v3_step_d.py --step E assess` (allowed only when all E units are complete? — run it; it
   reports missing units), `P pmm_v3_step_d.py --step E evidence`, `vm-stop`, `vm-status` (TERMINATED), `vm-report --no-ssh`.
5. Log a dated entry (newest first in log.md), overwrite STATUS (≤ 6,000 bytes; preserve replaced dated text in the log),
   `P tools/check_docs_contract.py`, `git add -- <paths>`, `git -c core.hooksPath=.githooks commit`, push only with the user's OK.
