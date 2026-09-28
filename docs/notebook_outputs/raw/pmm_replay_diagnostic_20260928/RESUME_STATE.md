# PMM ion campaign handoff — 2026-09-28

The bounded fold-0 replay investigation is complete. No worker or GPU is active.
Do not use the earlier running handoff or resubmit any completed fit.

- Campaign: `pmm_ion_metal_v2_context`; canonical root is this file's parent
  directory's parent. Scientific source SHA-256 remains
  `adc95c42261448dc9d35572a138a3b8349de79124d9708618f511c27a848dd23`.
- Nine configurations completed 50 epochs on fold 0, seed 42; eight pass the
  original replay contract. All nine satisfy the explicitly retrospective
  v2.1 output-agreement policy. These are separate counts, not nine legacy passes.
  The complete campaign still has 36 untrained fits out of 45.
- GVP6 diagnosis: preserved cache/fresh predictive inputs match for 1,492 ions
  in 94 batches. Inactive `y_ec` differs; CPU logits/loss equality was checked
  on the first 16-example batch, with source branch guards audited separately.
  Ten original-setting evaluations vary by at most 2.324581e-6 pairwise;
  ten strict deterministic passes are bitwise identical across two processes.
  All class predictions match. No training was repeated.
- VM `deepmzyme-l4`, zone `us-central1-c`, project `deepmzyme-gpu-vm`, immutable
  instance ID `3144565200755786222` is **TERMINATED**. Session
  `session-20260927T234132Z-8191aea8` stopped at 00:04:22 UTC, independently
  verified at 00:05:21 UTC; the 00:17:33 read-only report also confirms shutdown.
  The requested 0.75-hour session used 1,371 seconds / $0.3347 estimated running
  gross. All 42 remote files were hash-verified and acknowledged before shutdown.
- Cumulative campaign use: 9.3061 VM hours / $8.1815 estimated running gross,
  excluding additional retained-storage charges. The retained recovery disk is
  150 GB, approximately $15/month. The old false-empty inventory report is
  preserved. Controller fix `1fe09d7` is tested, committed and pushed; the new
  report lists the disk. No resources were deleted or caps changed.
- The pending user choice is stop at this evidence boundary, or authorize the
  proposed **30 VM-hour / $34 total ceiling**. Forecast for the 36 remaining
  fits: 17.37 additional VM hours / $15.27 running gross, plus storage. Forecast
  excludes final refits/held-out work. **No budget increase has been received.**
  Existing 4h/$6 session and 6h/$10 UTC-day caps remain in force.
- Broader execution also needs explicit integration of the versioned agreement
  policy: the frozen runner and `completed_run_receipt` still enforce the old
  gate and cannot consume the supplemental report. Do not forge receipts,
  retry GVP6 until a chance pass, or change frozen training bytes. No promotion,
  final refit, held-out access, or 45-fit assessment occurred.

Evidence: `runtime/replay_diagnostic_20260928/` contains the complete diagnosis,
all tensor outputs, hashes, provider logs and `verified_closeout.json`.
`runtime/screen_agreement_v2_1_20260928/` preserves the initial v2.1 CPU report;
`runtime/screen_agreement_v2_1_checked_20260928/` holds the report after the
consumer's input-semantic enforcement review. Portable evidence lives in the
repository at `docs/notebook_outputs/raw/pmm_replay_diagnostic_20260928/`.

Before any authorized GPU continuation, read the project gpu-use-skill and
recheck current provider state through the installed controller. Reuse all
verified preparation, checkpoints, embeddings and caches. One coordinator owns
allocation/submission/shutdown; a focused auditor has no lifecycle ownership.
