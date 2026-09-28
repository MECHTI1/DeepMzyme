# PMM five-class screen: preparation and execution

All three five-class fits completed 50 epochs. Only-ESMC and graph-level late
fusion passed original strict replay; GVP5 failed and remains provisional.
All 212 terminal artifact entries were independently backed up and verified.
The GPU is stopped and superseded-resource cleanup is complete.

- [Results and five-versus-six comparison](../../summaries/summary_pmm_five_class_screen_20260928.md).
- [Strict report](execution/five_class_screen.md), [JSON](execution/five_class_screen.json)
  and [CSV](execution/five_class_screen.csv); failed GVP5 is excluded.
- [Separate GVP5 failure diagnostic](execution/only_gvp__five_class__none__fold0__seed42/failed_replay_diagnostic.md),
  including its provisional point estimate and unchanged replay threshold.
- [Final execution record](execution/execution.json) and
  [verified closeout](execution/verified_closeout.json), with provider reports,
  controller receipts and hashes. The retained working disk still bills.

Successful units contain exact small artifacts and explicit metadata excerpts
bound to original hashes. The failure folder preserves the unsuccessful replay
and carries a separate failure manifest. Full metadata and binary checkpoints
remain in the canonical verified backup; they are not copied into Git.

The initial preparation/capacity snapshot below is preserved. Preparation
JSON files at this directory's root remain the original snapshot; later outputs
are under [`execution/`](execution/). See
[current status](../../../../EXPERIMENT_STATUS.md) for the next authorized step.

## Initial preparation and capacity block

At this initial snapshot, no new campaign fit had started. The three-family five-class fold-0 screen is
implemented and CPU-verified at commit `b80d164`. The existing L4 could not
start in `us-central1-c` because of provider capacity exhaustion. Provider
state was confirmed **TERMINATED at 2026-09-28 01:59:48 UTC**. No additional
GPU compute use was recorded; the existing disk remains billable.

- [Protocol](../../../plans/pmm_five_class_screen_v1.json): Mn/Cu/Zn/Fe/Co+Ni,
  native-BA checkpoint selection, probability-first common-four reporting,
  original strict independent replay, no held-out access.
- [Validation](validation.json): 32 core-scope tests and 40 adapter tests,
  including a synthetic CPU fit and independent replay. These are engineering
  checks, not scientific campaign results.
- [Three-command preview](screen_preview.json) and
  [core-scope preview](core_scope_v2_preview.json) bind the prepared commands.
  The latter describes 39 candidate units, not a completion audit.
- [Execution record](execution.json) and
  [controller ledger excerpt](controller_ledger_excerpt.json) preserve the
  failed start and the pending new-recovery decision. The controller refused
  another recovery pass while the completed previous pass's receipt remains;
  a separately authorized pass is required before archiving that receipt.

Canonical artifacts are under
`/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v2_context/runtime/five_class_screen_20260928/`.
The isolated frozen checkout is its sibling campaign code directory
`/media/mechti/Data1/DeepMzyme_Data/campaigns/_code/pmm_core_scope_v2`.
Historical fits and replay receipts are unchanged. Full-grid assessment,
promotion and refit remain gated by TECH-023/025; binding-aware work stays paused.
