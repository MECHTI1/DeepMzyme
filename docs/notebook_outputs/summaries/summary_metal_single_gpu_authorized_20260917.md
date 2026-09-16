# Authorized single-GPU metal continuation — 2026-09-17

Status at this evidence boundary: **in progress**. The user authorized additional
Colab GPU time to finish the accepted campaign and requested explicit
stop/continue choices when forecasts require higher ceilings. The current
recorded allowance is 36 cumulative allocation hours: 9 operations, 9 discovery
and 18 confirmation. This is a ceiling, not a spending target. Per-session
limits remain four hours with fifteen minutes reserved for closeout.

The original 20-hour profile's evidence and allocation charges are retained.
The continuation uses frozen source manifest
`ecf1799abb451ba286c56de9164128bead6b54dbcd02b526f2375f0929b86ee8`.
The source snapshot includes budget-authorization controls and the repaired
RING path. Later documentation edits do not rewrite that worker snapshot.

The repaired RING probe completed in 106.190 seconds and the new GVP anchor in
100.590 seconds. These are Grade-6 operational measurements. The first full
50-epoch compact Only-GVP/direct-four fit, LR `1e-5`, seed 42, completed in
236.399 seconds. Its native-selected epoch is 42, validation balanced accuracy
is 0.6521880369, and minimum class recall is 0.28125. This individual result is
Grade 5 and is not a model-family ranking or promotion.

All three attempts' artifacts were downloaded, checksummed, independently read
back and verified before the next launch. Session
`deepmzyme-metal-v2-authorized-20260917-8` was assigned G4 and is provider-verified
stopped. The complete allocation lineage through that stop totals
7,657.208562 seconds (2.127002 hours), including setup, failures, recovery,
transfers and idle time. Session 9 is a separate continuation; consult current
status rather than interpreting this closed receipt as its shutdown.

Remaining at this boundary: 67 mandatory discovery fits, triggered optional
blocks within their declared limits, candidate freezing and the 110-cell
shared-fold/seed confirmation matrix. No held-out data, Stage 6B refit or
Stage 7 evaluation is authorized by this campaign.

[Portable evidence](../raw/metal_single_gpu_authorized_20260917/README.md) preserves
configuration, selected-checkpoint binding, split diagnostics, timing and budget
provenance. The [metal playbook](../../METAL_TRAINING_PIPELINE_PLAYBOOK.md#single-gpu-metal-campaign)
owns the executable recipe; [current status](../../../EXPERIMENT_STATUS.md) owns
what to do next.
