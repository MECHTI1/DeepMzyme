# PMM five-class screen: preparation and capacity block

No new campaign fit has started. The three-family five-class fold-0 screen is
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
