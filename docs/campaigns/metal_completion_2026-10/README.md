# Metal completion successor

Status: active (2026-10-16 optimization complete; B four-class retained; Stage 6B source verified; Stage 7 not started)

## Scope and authority

The maintainer chose all three metal targets and independent final evaluation before EC.
Preserve the frozen v3 `gvp_late_fusion` trainer, ion cohort, five grouped folds and terminal epoch-50
checkpoint rule. Structural-context and trunk-LR investigations stay closed.

The approved scientific design is owned by [assessment_spec.json](assessment_spec.json):

- **B:** existing manually weighted cross-entropy, smoothing zero.
- **P:** population-normalized weighted CE via focal gamma zero, smoothing zero.
- **S:** weighted CE with native-target label smoothing 0.1.
- Discovery: B/P/S × four/five/six × five folds, seed 42 (45 prescribed cells); reuse verified B controls.
- Confirmation: one shared recipe × all targets × five folds × seeds 43/44, plus B-four controls;
  deduplicate when B wins (30 cells for B, otherwise 40).

Keep `commonfour_equalized` manual weights. Selection is validation-only on probability-summed common-four
BA, with native metrics/recall protection and fixed contrasts/ties. Repeated seeds are not independent validation.

## Implementation and evidence

[evidence.md](evidence.md) owns verified reuse/checks/progress; [log.md](log.md) owns dated decisions.
Successor identities/output paths are separate. Frozen parent source, manifests and evidence remain read-only.
The 2026-10-14 review re-sealed the package; the first package is superseded history.

Use the [canonical metal playbook](../../METAL_TRAINING_PIPELINE_PLAYBOOK.md#metal-completion-successor-cpu-preparation-and-execution-gates)
for executable preparation and execution gates. Future fits require a forecast, approved ceiling and
bound policy/session/spending receipts; preparation grants no compute authority.

[evaluation.md](evaluation.md) owns the test proposal, inventory and blockers; the Stage 6B source is in
[evidence.md](evidence.md#stage-6b-source-and-compatibility-2026-10-16). No certified test cohort exists.

[handoff.md](handoff.md) is the continuation checklist. [Plan](../../../Plan.md) owns science;
[DATASETS](../../DATASETS.md#final-reporting-status) owns test history. EC remains deferred; repair
[TECH-030](../../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-030--discovered-ec-cache-files-can-bypass-the-training-path-guard).
