# Authorized metal campaign continuation — 2026-09-17

This is a partial evidence package, updated as verified batches finish. The
[current status](../../../../EXPERIMENT_STATUS.md) owns the active execution state;
the [batch summary](../../summaries/summary_metal_single_gpu_authorized_20260917.md)
explains the evidence. The original closed 20-hour profile remains separate.

The authorization ledger preserves the original limits, the initial 33-hour
extension and the subsequent 36-hour ceiling. Additional operations time funds
measured save/readback overhead. The scientific grid, fold/seed definitions,
validation selection and held-out protections are unchanged.

- `attempt_00016`: repaired RING-on one-epoch timing probe, completed.
- `attempt_00017`: fresh allocation GVP timing anchor, completed.
- `attempt_00018`: first full 50-epoch compact Only-GVP/direct-four discovery fit,
  LR `1e-5`, seed 42, completed and independently persisted.
- `session8_stopped.json`: provider-verified stop and cumulative allocation time.
- `session9_restore_manifest.json`: checksums of the next session's state and
  prior-attempt archives, including every earlier allocation and attempt.

Copied JSON/CSV artifacts retain their original bytes. Checkpoints and full
archives remain in ignored local runtime storage under
`DeepMzyme_Data/notebook_outputs/plans/metal_single_gpu_33h_v2_authorized_runtime_g4/`.
`SHA256SUMS` covers the copied evidence files; descriptive Markdown is excluded.
No held-out evaluation, final refit or promotion is part of this package.
