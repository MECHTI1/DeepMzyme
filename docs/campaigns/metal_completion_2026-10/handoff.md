# Independent metal evaluation

Optimization is complete. The retained B direct-four configuration uses the v3 late-fusion refit `a0007846…`. Its [Stage 6B compatibility record](evidence.md#stage-6b-source-and-compatibility-2026-10-16) passed all 25 checks. Keep the designated architecture and feature set fixed. Adding an atom-level encoder or surrounding-atom inputs would change the model. Structural-context, symmetry and trunk-LR investigations are closed. EC remains deferred pending repair and verification of [TECH-030](../../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-030--discovered-ec-cache-files-can-bypass-the-training-path-guard).

## Stage 7 decisions

Only reading and CPU preparation are authorized. GPU use, spending, training, candidate-test collection, test-side reads and held-out scoring require separate approval.

1. Confirm the primary evaluation route. The proposed temporal PDB cohort must be disjoint from training, previously opened tests and their near-copies. The refit has already had one secondary held-out pass. Its use for a primary Stage 7 report requires an explicit exception to the one-shot rule or a Plan amendment.
2. Once approved, complete the [prior-use inventory](evaluation.md#remaining-blockers) using its record allowlist. Obtain separate approval for the network request and record the read in [DATASETS](../../DATASETS.md#test-use-ledger).
3. Resolve the remaining protocol decisions in their listed order. Each requires approval. Campaign closure remains a separate decision.

## Existing records

Under `DeepMzyme_Data/campaigns/metal_completion_2026-10/preparation_20261014T164111Z/evidence/`:

- `execution_closeout.json` records the completed fits.
- `stage6b_source_20261016/` contains the source compatibility checks.
- `evaluation_prep_20261016/` contains training-side release dates, obsolete entries and the blocked evaluation template.

At handover the package was on the secondary drive only. Reuse existing checks while their source, inputs and environment match. The [metal playbook](../../METAL_TRAINING_PIPELINE_PLAYBOOK.md#metal-completion-successor-cpu-preparation-and-execution-gates) records the per-boot kernel selection for the GCP host, where the NVIDIA driver failed to build for `7.0.0-1014-gcp`.
