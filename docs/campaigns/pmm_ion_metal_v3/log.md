# PMM ion metal v3 — decision log

Dated user decisions and STATUS history for this campaign, newest first.
Current authority: [EXPERIMENT_STATUS.md](../../../EXPERIMENT_STATUS.md).

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
