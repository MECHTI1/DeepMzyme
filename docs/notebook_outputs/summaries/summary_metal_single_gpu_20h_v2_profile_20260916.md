# `metal_single_gpu_20h_v2` measured admission profile — 2026-09-16

## Purpose and scope

This batch resumed the paused, validation-only `metal_single_gpu_20h_v2`
campaign far enough to measure its active-G4 cost gate. It did not admit or run
any 50-epoch discovery or confirmation fit. Held-out evaluation stayed disabled.

The executed worker identity used source `23d8ee4575a05c29622800b57300877ce1229225`,
campaign-manifest SHA-256
`9965982f7e4feababeb0337beb0e5a365db3812f76d2859937025fa77d4b5b4c`,
fold-plan SHA-256
`afaf7f7c8ee21baeef176978c402f79eb08d61ac531b72e6aa94787a275d1406`,
input SHA-256
`a5eca9f223df61b11f52851352a8007642e2cee29ef229098773ab7cd74c92cb`,
and cohort SHA-256
`dbc15c7013d3895b04ff6e5bca7207b01ff95d485769f614bbbd8feb7272fc1`.
Both resumed workers were G4 RTX PRO 6000 Blackwell allocations using stock
PyTorch 2.11.0+cu128 with CUDA 12.8 and compiled `sm_120` support.

## Profiling outcome

Twelve of the thirteen initially required one-epoch architecture probes
completed. They covered matched four/six-class Only-GVP, Only-ESM and
graph-level late fusion; direct-four early and hybrid fusion; reference and
large late-fusion capacities; the fixed five-class diagnostic; and RING-off.
A later same-hardware reference-GVP anchor also completed after cross-session
recovery. These one-epoch results are timing probes only and are not model
selection evidence.

The RING-on probe failed preflight because its frozen command omitted the
bundle's `RING_features` root, reporting 1,304 missing structures. Its one
allowed retry supplied that intended cache root operationally, but the Colab
provider dropped the owned session before terminal status could be observed.
Provider-verified teardown reconciled the retry as interrupted and charged
106.864 seconds. No result was invented and the probe did not count as
completed timing evidence.

The first resumed attempt also exposed a controller/trainer integration defect:
the controller created `execution.log` before training, while the trainer
rejected that file as foreign prelaunch output. Commit `23d8ee4` permits that
specific controller log and adds a subprocess regression test. The later local
fixes pass an explicit RING cache root to every RING run and let an exhausted
operations probe yield to a fresh-session hardware anchor. The complete focused
suite passed 165 tests.

## Measured admission decision

Full training is **not admitted**.

- Discovery projects to 21,454.135 raw seconds and 26,817.669 seconds
  (**7.449 hours**) with the mandatory 25% margin, exceeding the six-hour cap
  by 5,217.669 seconds (**1.449 hours**).
- Future operations project to 9,200 raw seconds and 11,500 seconds
  (**3.194 hours**) with margin. The authoritative host ledger closed at
  5,684.279 cumulative allocated seconds, leaving 8,715.721 seconds
  (**2.421 hours**) of the four-hour operations cap. The deficit is
  2,784.279 seconds (**0.773 hours**).
- Confirmation pricing is also incomplete because all ten grouped-fold/seed
  RING-on cells lack a compatible completed timing probe. Known raw costs are
  22,301.068 seconds for core, 4,188.659 for the fixed five-class block,
  8,321.233 for early/hybrid fusion, and 4,364.030 for the RING-off half of the
  RING block.

The worker-side forecast omitted 1,149.645 seconds from two earlier profiling
allocations after the execution-fix campaign identity was created. The host
allocation ledger retains that time and is authoritative. This reconciliation
makes the operations rejection stricter; it does not change the decision.

## Closeout and next action

Four owned allocation intervals associated with this profiling lineage total
5,684.279 seconds (**1.578966 hours**). Both execution-fix sessions have
provider-verified stop receipts, the final server listing reported no active
sessions, and the final closed worker-state archive is retained locally with
SHA-256
`dafd38f7aadf604d8c4ef259f5318996b76f5acc5baa5cce5ad8c5ccee5075b6`.

Do not start the frozen campaign's discovery fits. A future attempt needs a
new campaign identity with an explicitly revised grid or budget policy; fixing
the RING path alone cannot make the measured discovery and operations budgets
pass. No Stage 6B, Stage 7, held-out inference, model promotion, or scientific
architecture ranking occurred.

Portable evidence is indexed in
[`raw/metal_single_gpu_20h_v2_profile_20260916/`](../raw/metal_single_gpu_20h_v2_profile_20260916/README.md).
