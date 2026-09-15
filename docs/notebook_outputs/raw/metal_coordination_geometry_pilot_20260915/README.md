# Coordination-geometry evidence — 2026-09-15

Read the [summary](../../summaries/summary_metal_coordination_geometry_pilot_20260915.md)
before interpreting these files. This package contains exact, SHA256-verified
copies from the local `metal_coordination_geometry_pilot_v1_20260915` campaign.

- `runs/`: all five smokes and 15 full fits, with exact configuration,
  metadata, epoch metrics, and validation metrics. Separate large
  `dataset_summary.json` files are omitted; retained identities are in the
  shared split and per-run metadata.
- Root files: source/manifest, readiness, cohort, input/normalization audits,
  admission and LR selection, attempt ledger, and aggregate result snapshots.
- `transfer_receipts/`: all 21 readiness/training attempt receipts.
- `geometry_two_seed_analysis.json`: descriptive matched-seed calculations
  bound to the recorded aggregate and run-artifact hashes.
- `geometry_paired_case_analysis.json`: verified site-level error comparisons
  from the later selected-checkpoint prediction export.
- `analysis/`: final checked figure, SVG, plotted values, and provenance.
- `validation_export_evidence/`: separate post-training replay helpers and
  receipts, 15 selected-checkpoint prediction CSVs, and 12 descriptive paired
  error comparisons. No training, checkpoint reselection, or test evaluation.
- `finalization/`: separate genuine completed queue/coverage and final run
  provenance, with the verified shared capture receipt. This preserves the
  original analysis-bound aggregate rather than overwriting it.

`geometry_validation_results.json` retains the last archived training
snapshot, including an embedded `awaiting_archive_transfer` status. Completed
run coverage and transfer verification are supported by their respective
artifacts; do not treat this status as the final queue state. The genuine
terminal snapshot is now in the separate `finalization/` directory, so the
aggregate hash used by the analysis remains unchanged.

`portable_copy_manifest.tsv` records the relative path, SHA256, and size of
each exact copied artifact. This README and the inventory itself are generated
documentation, not original run artifacts. No checkpoint, cache, or archive
binary is copied into this package. The capture predates teardown; the later
[shared session-stop receipt](../metal_architecture_pilot_20260915/continuation/closeout_allocation2/post_stop/session_stopped.json)
and [closed allocation report](../metal_architecture_pilot_20260915/continuation/closeout_allocation2/post_stop/post_stop_closeout.json)
verify shutdown separately from the historical first-allocation closeout.
