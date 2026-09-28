# PMM fold-0 replay diagnosis and retrospective agreement

No training was repeated. All nine existing fold-0 configurations satisfy the
explicit **v2.1 output-agreement policy**; **eight pass the original replay
contract**. These are different checks. Historical failures and receipts are
unchanged. The full grid still has 36 untrained fits; no promotion, final refit
or held-out access occurred.

## What was established

- Rebuilding the 1,492 validation graphs initially exposed one differing field
  in every example: the unused EC target `y_ec` was `0` in the training cache
  and `-1` in replay. Every other raw graph field, including metal targets,
  matched its unique original-window cache candidate exactly.
- The [v1.1 amendment](../../../plans/pmm_replay_diagnostic_v1_1.md) preserved
  that failed audit and recovered the independently captured fresh tensor
  contents from verified cache bytes. All 94 normalized/collated batches match
  in predictive fields. The model has its EC head disabled; source inspection
  verifies the guard, and CPU logits/loss were bitwise equal for the first
  complete 16-example batch. That empirical CPU check does not cover all ions.
- Ten original-setting evaluations, split across two fresh processes, produced
  small probability differences on identical inputs: maximum pairwise native
  difference **2.324581e-6**, common-four **2.205372e-6**. All 45 pairs differed.
  Ten strict deterministic evaluations were bitwise identical across both
  processes. All native/common-four class predictions stayed unchanged across
  all 20 evaluations and the three preserved original/replay exports.
- The largest original-setting difference against any preserved export was
  **3.068287e-6**. Model state and input hashes remained unchanged. This supports
  CUDA numerical nondeterminism as the replay discrepancy's explanation; it
  does not identify one specific CUDA operation or guarantee future stability.

The [agreement report](screen_agreement_v2_1.md) applies the same absolute
`1e-5` replay bound to every native/common-four probability and every preserved
attempt across all nine configurations. Predictions, identities, checkpoint
selection, confusion matrices and metrics must match exactly (existing metric
reconciliation retains `1e-9`). This is a disclosed retrospective engineering
acceptance rule, not proof of calibrated probabilities or a model-error bound.

The initial v2 consumer also introduced a redundant `2e-7` probability-sum
check, which rejected a previously certified ESM6 export because FP32 softmax
sums need not equal one exactly. Its failed report is preserved in
`initial_v2_failure/`. [V2.1](../../../plans/pmm_screen_agreement_v2_1.json)
reuses the pre-existing frozen campaign validator's `2e-6` simplex allowance;
the `1e-5` replay bound, `2e-7` collapse check and exact discrete/metric gates
remain unchanged. Both policy versions and consumer hashes are retained.

The final CPU consumer review added explicit enforcement of the known v1.1
schema, the sole allowed `y_ec` exclusion, disabled EC supervision, exact CPU
equivalence, and preserved diagnostic source/input links. All nine configurations
passed again with that checked consumer. `initial_v2_1/` preserves the earlier
report; the policy and probability bounds did not change. Validation comprises
17 diagnostic tests, 52 consumer tests and six controller report tests; see
[validation record](validation.json) and the test logs.

## Evidence and limits

- [Diagnostic summary](diagnostic_summary.json), four `evaluations/` reports,
  [input audit excerpt](input_manifest_excerpt.json), original failed-input
  excerpt, and worker logs preserve the measured checks and their provenance.
- [Agreement JSON](screen_agreement_v2_1.json) binds all nine checkpoints,
  original exports, preserved replay attempts, diagnostic evidence and policy.
  It retains native six-class metrics and Fe/Co/Ni recalls.
- [Verified closeout](verified_closeout.json), execution record, controller
  logs, 42-file remote manifest and host acknowledgment document persistence
  and provider shutdown. The retained 150-GB disk remains billable; direct
  provider evidence corrects a controller inventory report that printed none.
- [Controller fix](controller_fix.json) records the pushed fix; the
  [post-fix report](vm_report_after_inventory_fix.log) independently confirms
  the VM is stopped and correctly lists the retained disk.
- `tools/` preserves executed helper versions, launchers and the summary tool.
  Production entry points are the repository-root `audit_pmm_replay.py` and
  `audit_pmm_screen_agreement.py`; neither changes frozen training source.

Canonical complete artifacts, including input manifests, the tensor snapshot
and every forward-pass tensor, remain under
`/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v2_context/runtime/replay_diagnostic_20260928`.
Binary tensors are not copied into Git. Excerpts bind omitted originals by
SHA-256 and size; the 42-file transfer manifest describes the canonical remote
layout, not this selective portable layout. `SHA256SUMS` describes this batch.

Original in-memory training inputs were not fingerprinted contemporaneously;
the cache comparison is retrospective. Only GVP6 received this tensor and
repeatability diagnosis. The other eight fits received the uniform stored-export
agreement recheck. All results remain single-fold/single-seed evidence. The
frozen runner still enforces the original replay gate and cannot consume these
new reports; do not broadly resume it over GVP6 or claim 45-fit completion.
