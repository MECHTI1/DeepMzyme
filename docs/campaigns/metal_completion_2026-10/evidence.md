# CPU preparation evidence and execution proposal

Prepared 2026-10-14; independently reviewed and re-sealed the same day, receipts closed late on 2026-10-14 UTC
(2026-10-15 local). This page records preparation and a proposed ceiling; it grants no execution authority.

## Verified reusable material

Local preparation root: Data1 `campaigns/metal_completion_2026-10/preparation_20261014T164111Z/` under
`/media/mechti/Data1/DeepMzyme_Data/`. `code/` holds the verified v3 source plus the reviewed completion modules;
`parent/` holds hash-identical frozen campaign inputs, folds, weights and execution settings (14 files re-hashed
against the first package's `package_preparation.json`). The first package `preparation_20261014/` is superseded
history after the review ([log](log.md)); it was not deleted or rewritten. Original source, campaign evidence,
checkpoints and caches were not rewritten. `src/` and `scripts/` retain source SHA-256
`212db0e3865968701475e7172b2e4cad772e492b35855da293238437fd97c15c`.

`evidence/control_records.json` SHA-256 `5a39f394599a6ec934774182a3b342f01e3311a0a762a7fdecc228f49bf26345`
(byte-identical to the first package) contains 16 verified late-fusion B controls: all three targets × five folds at
seed 42, plus four-class fold 0 at seed 43. The four-class fold-0 seed-42 control is the step-B FP32 probe run
(`probe-full-fp32__…__fold0__seed42`) that v3 recorded as that unit's fit in `parent/execution_settings.json`; its
identity, replay and bytes are verified through the same frozen path as the others. Verification used the frozen v3
collector, actual terminal checkpoint/prediction/replay hashes, exact fold membership, target/probability-collapse
validation and recomputed metric reconciliation. `evidence/host_control_integration.json` SHA-256
`6c4d9f8e2622d85f2cfcdc01439ff7dd529141b4d68ce06b4112cf54cb07bfa6` records the reviewed host code re-verifying all
16 against bytes on 2026-10-14. This saves 15 discovery fits and one confirmation fit. Reuse is checked again
against actual bytes during final collection.

`evidence/control_runtime_summary.json` binds each measured duration to its status file. The 16 fits took
1,663.8–4,526.1 seconds each, median 2,378.6, including replay; their sum is 42,019.1 process-seconds.
These are concurrent-process durations, not billable GPU hours. Frozen Step B selected FP32 with three lanes;
its speed report SHA-256 `8fccd2e1fa10961a8ecf346226ed9d86d334d0a48e5d80083e7b1d900098e082`
records 4.608 fits/hour versus 2.070 serial. A changed runtime still needs its relevant readiness checks.

Preparation on the re-sealed package (2026-10-14T16:43–16:53Z) passed all 6,443 training-structure content hashes,
the four frozen metadata hashes and all 7,664 ESM payload/sidecar pairs. `execution/manifest_discovery.json` SHA-256
`d405974eeb4fd0c029922ccfcd01adebef8fd85876c220d92953557753ce98f8` binds the seven executing modules (the six
completion modules and `pmm_v3_assessment.py`, which must also equal the assessor hash frozen by the parent
extension), the parent inputs, the specification (`2bc2243d…`) and the 45 prescribed units. `evidence/readiness.json`
SHA-256 `4f602b80de5243ac797d14d4a29a2745b40cd7fb3d46614c4b19aec045991b0f` records this verification and the 45
command previews (`discovery_command_preview.json`, built by the package's command builder from the hash-bound
manifest); `evidence/plan_preview_P_four_class_fold0.json` is the one full CLI `plan` run (nine minutes, exit 0),
which re-verified the whole package on this host. No discovery decision exists: `collect` refuses the incomplete grid
(the 15 discovery controls without the 30 new fits; `evidence/readiness_cli_collect.log`). Local absolute paths are
bound; the package must be deployed with the documented path invariant before future execution.

## Proposed execution ceiling — approval pending

The exact scientific grid and decision gates are in the [canonical recipe](../../METAL_TRAINING_PIPELINE_PLAYBOOK.md#metal-completion-successor-cpu-preparation-and-execution-gates)
and [frozen assessment specification](assessment_spec.json). Discovery needs 30 new fits. Confirmation needs 29
new fits if B advances, or 39 if P/S advances: **59–69 total**, before any failed-attempt recovery or final refit.
No additional recipes, targets, seeds, folds, epochs or model families follow from the ceiling.

For the existing GCP L4 route with the verified three-lane FP32 setting, plan approximately **16–23 billed GPU
hours**, including preparation/transfer/stop allowances and runtime variation. This is a forecast, not a guarantee;
serial execution would require a larger ceiling and must be reviewed before launch. The
[canonical execution gates](../../METAL_TRAINING_PIPELINE_PLAYBOOK.md#execution-evidence-and-transition-gates)
own the proposed operational ceilings (24 aggregate GPU hours, $30 gross, eight sessions of at most four hours).
Admission retains the measured per-unit forecast ×1.25 plus 900 seconds for persistence/closeout. If the accepted
work will exceed a ceiling, apply AGENTS' two-choice escalation at a safe boundary; do not silently shrink the grid.

Cost basis is the controller's cached price verified **2026-10-13 16:20:17 UTC**, read again on 2026-10-14T15:51Z
with `vm-status --no-reconcile` and `vm-report --no-ssh` (read-only; VM `deepmzyme-l4` TERMINATED): gross running
rate $0.87917/hour = compute $0.85362 + IP $0.0050 + disk $0.02055. Retained storage while TERMINATED, by the
controller's conservative full-disk reservation, is about $30.00/month (disk $15.00 plus two snapshots at $7.50
each), about **$1.00/day**; the stored snapshot sizes (34.8 and 52.7 GiB) imply a smaller actual charge, but the
conservative figure is the planning basis. At 24 running hours ($21.10) plus seven calendar days of retained storage
($7.00) the proposed $30 envelope leaves about $1.90 of reserve; the first proposal counted the disk only and
overstated the reserve as $5.94. Approving $32 gross would restore about $4 of reserve under the same forecast;
the choice is the user's. Fresh price verification, actual prior spending and provider automatic STOP are mandatory.
No migration or larger hardware is proposed.

The installed controller was inspected read-only: session caps 4 hours/$6; daily caps 12 hours/$18, with its
existing safety margin and stop reserve. These were not changed. Campaign limits remain additional constraints.
No GPU start was performed.

## Verification and remaining boundaries

The 2026-10-14 independent review (six read-only reviewer agents, two adversarial verifiers per finding; structured
record `evidence/independent_review_findings.json`) found no blocker and three confirmed majors, all closed: the
executing v3 assessor is now manifest-bound; the wrapper/exit-file contract, persistence layout and deployment
prerequisite are pinned in the playbook. The combined focused suite now passes **141 synthetic tests** (nine added:
absolute run maps, trainer-recorded configuration, reuse directory binding, assessor binding, discovery-era
specification binding, pre-start snapshots, symlinked lane children, future upper dates, receipt fields), covering
actual loss/gradient mathematics, target/seed identities, complete-grid selection, paired inference, rare-class guards,
frozen-source refusals, execution authorization, clock/budget/session limits, actual training-file binding, evidence
collection and offline evaluation preparation. CPU smoke passed 42 checks with one absent-local-fixture skip (the
runner prints “PASSED 43 smoke checks”). Docs contract has zero strict failures and 11 pre-existing warnings.
`evidence/cpu_verification.json` binds code/tests/logs and library versions; the [log](log.md) records the review and
closeout. Synthetic success is not an accuracy result or a real GPU lifecycle certification.

The local source package and all 16 controls are verified. A launch still requires the GPU host to realise the
identical absolute paths ([deployment prerequisite](../../METAL_TRAINING_PIPELINE_PLAYBOOK.md#cpu-preparation-block)),
a passing `plan` on that host, train/ESM re-verification there, fresh controller/session/spending records, explicit
authority, the wrapper that records each unit's exit file, and tested host-pull acknowledgement with the documented
placement of pulled lanes and root-level records. No authorization record has been fabricated. The worker never
allocates compute or retries automatically.

`evidence/evaluation_provenance_template.json` SHA-256
`88b87be64682ed533c9641e5eb8a25f5ebef12a41a5dfc5c0cd556e6b3f55ab9` is deliberately blocked: no real source
dates or verified-completeness claims were supplied. The [evaluation protocol](evaluation.md) is offline tooling,
not a certified independent cohort. Candidate retrieval, precise support/quality rules, final refit/report freeze
and one-shot scoring remain separate work. Final refit and independent-cohort construction are outside this
validation-fit cost proposal and need estimates once selection and feasibility are known.
