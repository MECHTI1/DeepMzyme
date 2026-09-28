# PMM core execution and replay contract v1

This additive implementation carries out the user-approved
[core v2 scope](pmm_core_scope_v2.json): three families, native four/five/six
targets, ordinary readout, five frozen PDB-grouped folds, model seed 42 and
50 epochs. The nine trained core fold-0 units are preserved; continuation
contains 36 missing folds-1–4 fits. The three earlier awareness fits remain
historical exploratory evidence and twelve further awareness fits stay paused.
The active 45-fit grid differs from the original awareness-containing 45-fit
grid. This contract does not declare either grid complete.

The root runner, replay consumer and assessor/refit-preview bridge are
implemented and CPU-tested. Historical reuse requires the complete diagnostic
and readiness evidence described below. Current results, diagnostic completion
and resource state belong in
[`EXPERIMENT_STATUS.md`](../../EXPERIMENT_STATUS.md). Exact commands and outputs
belong in the [metal playbook](../METAL_TRAINING_PIPELINE_PLAYBOOK.md#active-core-only-continuation).

## Preservation and execution

`run_pmm_core_campaign.py`, `pmm_core_replay.py` and `pmm_core_assessment.py`
are outside the frozen scientific hash. They verify the unchanged training
tree `adc95c42261448dc9d35572a138a3b8349de79124d9708618f511c27a848dd23`
before scientific imports or execution. Historical source/configuration
identities, checkpoints, normalization, predictions, receipts, failed attempts,
cohort, folds, caches, ESMC embeddings and completed PMM folds are retained.
Shared-checkout source changes require a compatible isolated checkout;
they are not silently substituted into this campaign.

The runner previews 36 fixed selectors without inspecting fit results or
starting a child. Execution admits one explicitly selected new fold-1–4 unit
through the existing `CampaignExecution` controls and frozen trainer. Existing
run/log/command artifacts are refused. One coordinator owns allocation,
submission and shutdown; one training worker remains the default. A disposable
concurrency timing probe does not enable concurrent production fits.

Session/day limits, measured admission margins and independent backup/hash
acknowledgment remain intact. A readiness certificate is scientific evidence,
not compute authorization. No larger budget, automatic retry, checkpoint
reselection, altered cohort, HPO or awareness work is authorized here.

## Prospective replay qualification

Policy ID: `pmm-core-replay-v1`. The canonical serializable contract is
`POLICY` in [`pmm_core_replay.py`](../../pmm_core_replay.py); its SHA-256 at
adoption is `fa008bd06147544058fb36d185f8e2593e4483e53ccdeb1da0424f956e9bfa39`.
New fit identities bind both policy ID and hash before training.

Every target/family uses absolute probability tolerance `1e-5`, relative
tolerance zero, exact native/common-four predictions and confusion matrices,
and selected-checkpoint metrics reconciled within `1e-9`. Probabilities must
be finite and in range; the pre-existing simplex allowance is `2e-6` and
the separately declared native-to-common-four serialization allowance is
`2e-7`. These auxiliary checks do not enlarge the replay bound. Native-five
index 4 is Co+Ni; common-four index 3 is Fe+Co+Ni. Native-six Fe/Co/Ni remain
separate. Sum native probabilities before common-four argmax.

The consumer independently checks source, resolved configuration, complete
50-epoch history, native-BA checkpoint selection with earliest ties, saved
normalization, checkpoint hash, frozen ion membership and every preserved
replay attempt. The original strict exporter still runs once with `1e-6`.
Core qualification records its original strict outcome separately and never
writes a replacement legacy replay receipt. A core failure is preserved for
diagnosis, not retried until a chance pass.

The fixed bound is an engineering agreement convention informed by the earlier
FP32 diagnostic and testing convention. It is not a mathematical error bound,
a claim of deterministic CUDA execution, calibration evidence, or a model
selection advantage.

## Explicit historical integration

The policy pins exactly the nine core fold-0 checkpoints, their original
predictions and the complete known replay-export hash sets. All preserved
attempts and backup copies are checked. Historical acceptance is separately
labeled `retrospective_core_agreement` with `post_observation=true`; the seven
original strict passes keep that status, and both GVP strict failures remain
`original_strict_failed`. The older v2.1 screen report remains unchanged and
retains its original, different nine-arm scope.

GVP5 and GVP6 require exact predictive-input equality and repeatability
diagnostics. The only supported exclusion is the explicitly audited disabled
EC target `y_ec` under the existing v1.1 recovery protocol, including CPU
logits/loss equivalence; metal targets are never excluded. Original-window
cache comparison is retrospective evidence because the original in-memory
training inputs were not contemporaneously fingerprinted.

Each diagnostic preserves two fresh processes per condition, five forwards
per process, under original and strict deterministic settings. Unsupported
strict operations must be explicit preserved failures, not omitted results.
The consumer binds the audited tool, full identity, input manifest, graph
fields, saved snapshot, process reports and pass arrays; recomputes array
agreement/classes and cross-process differences; and verifies unchanged
model/input hashes. Stable original passes are valid observations and need
not be called numerically variable.

The separate campaign-relative index
`runtime/core_replay_v1/historical_diagnostics.json` binds each failure's
diagnostic summary hash to this policy. Absent or inconsistent evidence fails
closed. The readiness action validates all nine historical units and binds
the implementation and integration evidence; it neither completes the 45-fit
grid nor grants spending or held-out access.

## Assessment and final-refit boundary

The core assessor requires all nine configurations on all five folds and
five verified PMM folds on exactly the same validation ions. It verifies
complete OOF coverage, retains each unit's replay qualification, and separates
mean-fold common-four metrics from pooled OOF and native metrics. Six active
target contrasts use the original 10,000-resample/seed-42 paired-fold method,
95% intervals, common-four recall protection of 0.03 and simultaneous
denominator 9, retaining the three paused awareness hypotheses. Five-versus-six
is descriptive; multiplicity is not reduced after the screen.

The distinct `core_cv_*` outputs and `core_validation_decision.json` do not
overwrite or impersonate the old grid's decision. Selection follows the
predeclared common-four control and tie rules. Refit preview revalidates that
complete decision and its source evidence, then writes `core_stage6b_*`
artifacts without starting training. The full-training recipe uses seed 42,
50 epochs, full-cohort normalization/common-four weights and terminal epoch 50.
An existing-refit verifier is implemented, but no completed refit is implied.

Candidate-versus-Only-GVP-four intervals are inherited nominal 95% selection
diagnostics. The simultaneous bound covers the declared target-formulation
contrast family only; it does not establish simultaneous architecture
superiority or familywise error control for the selection procedure.

Only the declared secondary, possibly overlapping Zenodo PMM reference route
is supported by the preview; the primary final-test route remains unresolved.
A separately authorized neural refit, compatible PMM refit and reporting
decision are still required before Stage 7. No action in this root entry point
executes a refit or opens held-out inputs.
