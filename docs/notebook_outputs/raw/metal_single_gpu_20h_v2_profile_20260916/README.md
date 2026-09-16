# Portable evidence: `metal_single_gpu_20h_v2` measured admission profile

This directory contains the small, reviewable closeout evidence for the
2026-09-16 G4 profiling run. Large checkpoints, per-attempt transfer archives,
and the final closed-state tar remain under the ignored local runtime root:

`DeepMzyme_Data/notebook_outputs/plans/metal_single_gpu_20h_v2_execfix_23d8ee4_runtime_g4/`

Files:

- `attempt_status.json`: sanitized status, timing, result summary and artifact
  inventory for attempts 1–15. One-epoch scores are operational diagnostics.
- `budget_forecast_blocked.json` / `.md`: measured worker-side rejection and
  missing RING-on timing cells.
- `host_budget_reconciliation.json`: authoritative reconciliation with all
  carried host allocation time.
- `host_allocations.json`: sanitized four-interval owned allocation and stop ledger; local-only watchdog and host identifiers are omitted.
- `session_5_stopped.json` and `session_6_stopped.json`: provider-verified
  stop receipts for the two execution-fix sessions.
- `closed_state_receipt.json` and `closed_worker_state.tar.gz.sha256`: binding
  to the final local closed worker-state archive.
- `SHA256SUMS`: hashes for this portable package.

The batch performed no 50-epoch fit and no held-out evaluation. Full training
was rejected by both discovery and operations caps before scientific admission.
