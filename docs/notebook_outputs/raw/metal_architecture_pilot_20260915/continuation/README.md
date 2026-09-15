# Original metal pilot continuation evidence

This package supports the
[continuation summary](../../../summaries/summary_metal_architecture_pilot_continuation_20260915.md).
**All 30 original full fits and seven model smokes are complete and verified.**
The genuine terminal capture is in `finalization/`; the subsequent shared
GPU teardown and closed ledger are in `closeout_allocation2/post_stop/`.
The earlier numerical snapshots through attempts 021/029 preserve their
historical partial states, distinct from this final evidence.

- `snapshots/attempt_021/` contains seven exact aggregate files extracted from
  that attempt's SHA-verified archive, preserving the earlier 12-run state.
- `snapshots/attempt_029/` contains seven exact aggregate files from the
  verified 20-run archive. Both snapshots retain their archived pending-transfer
  state preceding the subsequent receipt; neither is final closeout evidence.
- `cross_session_recovery/` contains exact recovery readiness and interrupted
  attempt reconciliation. The original manifest/cohort/feature identities
  were revalidated, and the complete first allocation remains charged.
- `runs/` contains each of 27 completed continuation fits once:
  completed A1 late retry, all four A2 fits, all 12 T1/T2 fits, and nine
  core-family seed-43 repeats plus the early-fusion repeat. Each has exact
  `run_config.json`, `run_metadata.json`, `epoch_metrics.csv`, and
  `val_metrics.csv`, checked against its corresponding verified archive.
  The original three A1 fits remain in [the earlier package](../partial_a1/runs/).
- `transfer_receipts/` contains exact receipts for attempts 013–039. Each
  certifies local archive SHA verification and Drive verification against
  the original campaign manifest.
- `analysis/target_formulation_snapshot_attempt_029.json` and `.csv` are exact
  copies of the independent completed-screen audit. It verifies all 18 core
  target runs against individual configurations, metadata, selected histories,
  archived scores, shared cohort/normalization, and the native-LR repeat rule.
  These derived files retain their own source hashes and exclude seed 43.
- `analysis/original_final_two_seed_analysis.json` and `.csv` are bound to
  the genuine terminal aggregate and capture. They verify all 30 full fits
  and summarize nine core target pairs plus the early-fusion pair. The
  supplemental checkpoint/early audit independently verifies selected
  checkpoint hashes and supporting files.
- `finalization/` contains genuine completed states, final run/source
  provenance, metrics and verified local/Drive capture receipts. The capture
  occurred before teardown and preserves `gpu_stopped=false` correctly.
- `closeout_allocation2/post_stop/` contains the later verified stop receipt,
  closed cumulative ledger, persistence receipts and shared capture manifest.
  It is the current-session shutdown proof used by both pilot summaries.
- `portable_copy_manifest.tsv` is a generated copy inventory, with relative
  path, SHA256, and byte count for every exact evidence file. This README and
  the inventory describe the copy; they are not original runtime outputs.

Per-run files are retained once for this final report. Do not overwrite the
partial snapshots with newer aggregate state,
or duplicate every completed configuration into another snapshot. Raw files
retain their original remote paths and labels. In particular, native-five
`Class VIII` means Co+Ni, while collapsed-four `Class VIII` means Fe+Co+Ni.

Checkpoints, archives, feature caches, and redundant standalone dataset
summaries are retained outside Git. The earlier
[first-allocation closeout](../closeout/) certifies only that earlier runtime's
teardown; the second session's stop proof is in `closeout_allocation2/post_stop/`.
