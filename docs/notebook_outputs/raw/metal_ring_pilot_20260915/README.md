# Matched RING pilot: exact copied evidence

Experiment: `metal/nonoverlap/ring-pilot/2026-09-15`, profile
`metal_ring_pilot_v1`. Read the
[human-readable summary](../../summaries/summary_metal_ring_pilot_20260915.md)
before inspecting individual artifacts.

All four smokes and sixteen 50-epoch fits completed. The queue reached true
`completed`; checkpoint binding and finalization passed; the owned runtime
stopped at 2026-09-15 16:20:52.944619 UTC. Colab reported no active sessions.
Original cumulative allocation is 7.312911 hours, with 2.687089 hours remaining
under the original ten-hour cap. No budget was reset.

## Evidence map

- [Finalization](finalization/): frozen manifest, commands, result/coverage
  tables, input/cache audits, admission gates, attempt ledger, all full-fit
  configurations/metadata/epoch histories, copied frozen source/operators,
  checkpoint hashes/bindings and the terminal capture receipt.
- [Post-stop analysis JSON](analysis/ring_post_stop_analysis.json) and
  [CSV](analysis/ring_post_stop_analysis.csv): eight complete LR/seed off/on
  pairs, four family/LR summaries, every class recall and actual closed budget.
  The [analysis helper](analysis/analyze_completed_ring.py) requires verified
  terminal capture and actual stop, checks input hashes and checkpoint
  bindings, and performs no training or inference.
- [Actual stop](host_closeout_allocation3/session_stopped.json) and
  [post-stop receipt](host_closeout_allocation3/post_stop_receipt.json):
  owned endpoint identity, server absence, all three closed allocation
  intervals and final-archive preservation. The finalization capture itself
  correctly says `terminal_metadata_verified_gpu_not_stopped`; it is not
  substituted for this actual stop proof.
- [Drive readbacks](drive_readbacks/) and
  [attempt archive descriptors](attempt_archive_descriptors/): every attempt
  archive plus the finalization archive, with local checksum and Drive
  file-ID/byte-size/parent verification. The
  [post-stop archive readback](host_closeout_allocation3_drive_receipt.json)
  separately verifies preservation of actual teardown evidence.
- [Preparation](preparation/): pre-execution review, its Drive receipt and
  the existing local CPU audit, which completed without a duplicate restart.
  The pre-execution review describes its historical preparation state;
  finalization and actual closeout own the completed state.

## Copy contract

[portable_evidence_inventory.json](portable_evidence_inventory.json)
records SHA-256, bytes and local source path for **287 byte-identical copied
files**, totaling **52,832,742 bytes**. It inventories exact copied files,
not this authored README or the human-readable summary. Copied outputs are
not shortened, rewritten, or silently reconciled. The
[collector source](analysis/collect_portable_evidence.py) is preserved.

Checkpoint/archive binaries remain in the ignored local campaign
`DeepMzyme_Data/notebook_outputs/campaigns/metal_ring_pilot_v1_20260915/`
and in the
[authorized Drive folder](https://drive.google.com/drive/folders/1slTff0joKjL-gZJDhYSGbHzOPnYhGk6I).
They are not committed here. Local SHA-256 verification and Drive metadata
verification are distinct; no Drive-side SHA verification is claimed.

The evidence is Grade 3 on one fixed validation partition with two training
seeds. It does not contain grouped-fold CIs, promotion, held-out evaluation,
auxiliary-training results or case-level RING prediction export. All four
late-fusion pairs tie in selected metrics; this does not prove identical
models, per-site predictions or universal RING ineffectiveness.
