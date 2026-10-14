# CPU preparation evidence and execution proposal

Prepared 2026-10-14. This page records preparation and a proposed ceiling; it grants no execution authority.

## Verified reusable material

Local preparation root: Data1 `campaigns/metal_completion_2026-10/preparation_20261014/` under
`/media/mechti/Data1/DeepMzyme_Data/`. `code/` holds the verified v3 source plus additive completion modules;
`parent/` holds hash-identical frozen campaign inputs, folds, weights and execution settings. Original source,
campaign evidence, checkpoints and caches were not rewritten. `src/` and `scripts/` retain source SHA-256
`212db0e3865968701475e7172b2e4cad772e492b35855da293238437fd97c15c`.

`evidence/control_records.json` SHA-256 `5a39f394599a6ec934774182a3b342f01e3311a0a762a7fdecc228f49bf26345`
contains 16 verified late-fusion B controls: all three targets × five folds at seed 42, plus four-class fold 0 at
seed 43. Verification used the frozen v3 collector, actual terminal checkpoint/prediction/replay hashes, exact
fold membership, target/probability-collapse validation and recomputed metric reconciliation. This saves 15
discovery fits and one confirmation fit. Reuse is checked again against actual bytes during final collection.

`evidence/control_runtime_summary.json` binds each measured duration to its status file. The 16 fits took
1,663.8–4,526.1 seconds each, median 2,378.6, including replay; their sum is 42,019.1 process-seconds.
These are concurrent-process durations, not billable GPU hours. Frozen Step B selected FP32 with three lanes;
its speed report SHA-256 `8fccd2e1fa10961a8ecf346226ed9d86d334d0a48e5d80083e7b1d900098e082`
records 4.608 fits/hour versus 2.070 serial. A changed runtime still needs its relevant readiness checks.

Actual preparation passed all 6,443 training-structure content hashes, the four frozen metadata hashes, and all
7,664 ESM payload/sidecar pairs. `execution/manifest_discovery.json` SHA-256
`e5905d8ef3855c0dd0814b09c8777520293b63d9e7ead8b817f65907cfc9a5f2` binds the code, inputs, specification
and 45 prescribed units. `evidence/readiness.json` records this verification and the 45 parsed command previews;
`evidence/host_control_integration.json` records successful verification of all 16 controls by the new host.
No discovery decision exists: assessment correctly refuses the 30 missing new fits. Local absolute paths are bound;
the package must be deployed with the documented path invariant before future execution.

## Proposed execution ceiling — approval pending

The exact scientific grid and decision gates are in the [canonical recipe](../../METAL_TRAINING_PIPELINE_PLAYBOOK.md#metal-completion-successor-cpu-preparation-and-execution-gates)
and [frozen assessment specification](assessment_spec.json). Discovery needs 30 new fits. Confirmation needs 29
new fits if B advances, or 39 if P/S advances: **59–69 total**, before any failed-attempt recovery or final refit.
No additional recipes, targets, seeds, folds, epochs or model families follow from the ceiling.

For the existing GCP L4 route with the verified three-lane FP32 setting, plan approximately **16–23 billed GPU
hours**, including preparation/transfer/stop allowances and runtime variation. This is a forecast, not a guarantee;
serial execution would require a larger ceiling and must be reviewed before launch. The
[canonical execution gates](../../METAL_TRAINING_PIPELINE_PLAYBOOK.md#execution-evidence-and-transition-gates)
own the proposed operational ceilings.
Admission retains the measured per-unit forecast ×1.25 plus 900 seconds for persistence/closeout. If the accepted
work will exceed a ceiling, apply AGENTS' two-choice escalation at a safe boundary; do not silently shrink the grid.

Cost basis is the locally cached price verified **2026-10-13 16:20:17 UTC**, not a fresh quote:
running compute/IP $0.858624/hour and the existing 150-GiB disk $0.020548/hour (about $0.493/day).
At 24 running hours and seven calendar days of retained disk, those components total about $24.06; the proposed
$30 envelope leaves about $5.94 for retained snapshots, additional persistence and reserve. Actual retained storage
and snapshot charges must be measured by the controller before authorization/launch. Fresh price verification,
actual prior spending and provider automatic STOP are mandatory. No migration or larger hardware is proposed.

The installed controller was inspected read-only: session caps 4 hours/$6; daily caps 12 hours/$18, with its
existing safety margin and stop reserve. These were not changed. Campaign limits remain additional constraints.
No GPU start or fresh provider-state query was performed; the last recorded stopped state is 2026-10-13.

## Verification and remaining boundaries

The final combined suite passed **132 synthetic tests** covering actual loss/gradient mathematics, target/seed
identities, complete-grid selection, paired inference, rare-class guards, frozen-source refusals, execution
authorization, clock/budget/session limits, actual training-file binding, evidence collection and offline evaluation
preparation. CPU smoke passed 42 checks with one absent-local-fixture skip. Docs contract has zero strict failures
and 11 pre-existing warnings. `evidence/cpu_verification.json` binds code/tests/logs and library versions; the
[log](log.md) records closeout. Synthetic success is not an accuracy result or a real GPU lifecycle certification.

The local source package and all 16 controls are verified. A launch still requires a verified deployment with the
same absolute paths or a separately reviewed relocation mechanism, train/ESM re-verification on that host, fresh
controller/session/spending records, explicit authority and tested host-pull acknowledgement. Worker exit codes,
root claims, authorization/session records and lane artifacts must be preserved together. No authorization record
has been fabricated. The worker never allocates compute or retries automatically.

`evidence/evaluation_provenance_template.json` SHA-256
`88b87be64682ed533c9641e5eb8a25f5ebef12a41a5dfc5c0cd556e6b3f55ab9` is deliberately blocked: no real source
dates or verified-completeness claims were supplied. The [evaluation protocol](evaluation.md) is offline tooling,
not a certified independent cohort. Candidate retrieval, precise support/quality rules, final refit/report freeze
and one-shot scoring remain separate work. Final refit and independent-cohort construction are outside this
validation-fit cost proposal and need estimates once selection and feasibility are known.
