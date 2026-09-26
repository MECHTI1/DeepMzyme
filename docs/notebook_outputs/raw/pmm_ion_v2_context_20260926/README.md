# PMM ion v2 context campaign — portable evidence

This is the 2026-09-25 21:56 UTC snapshot summarized
[here](../../summaries/summary_pmm_ion_v2_context_20260926.md). Files below are
byte-identical copies from
`/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v2_context/`.
[`SHA256SUMS`](SHA256SUMS) records their copied checksums; verify with
`sha256sum -c SHA256SUMS` from this directory.
The later feature-generation and implementation-check receipts were added at
22:24 UTC; the original snapshot receipts remain unchanged.
Completed input certification, its verified host backup, the final graph-cache
source revision and 137 passing targeted checks were added after resumption at
2026-09-26 00:10 UTC.
The first full fit, its verified host acknowledgment, operational batch probes
and proposed continuation budget were added after 01:00 UTC.

| Files | Evidence |
|---|---|
| [`campaign_manifest.json`](campaign_manifest.json) | Source/cohort/structure identities and preserved v1 parent provenance |
| [`train_audit.json`](train_audit.json), [`train_context_audit.json`](train_context_audit.json) | 503 exclusions, 7,398 retained ions/3,992 groups, certified input contract and explicit source-context parity limitation |
| [`fold_class_weights.json`](fold_class_weights.json) | Frozen fold identity, native/common-four supports and training-fold weights |
| [`esm_generation_plan.json`](esm_generation_plan.json) | Preparation counts, longest sequence, storage estimate and full plan checksum; not a completed feature certificate |
| [`pmm_comparator/`](pmm_comparator/) | Released recipe, resolved parameters, environment, five fold receipts and all 7,398 validation predictions (about 1.2 MB total) |
| [`pmm_comparator_run.log`](pmm_comparator_run.log) | CPU comparator completion and released-recipe warnings |
| [`runtime/esmc_preflight.json`](runtime/esmc_preflight.json) | Successful real L4 ESMC sample inference, software, timing, dtypes and memory |
| [`runtime/esmc_preflight_exact_arch_rejected.json`](runtime/esmc_preflight_exact_arch_rejected.json) | Earlier exact-architecture-string rejection before sample inference |
| [`esm_generation_receipt.json`](esm_generation_receipt.json), [`runtime/generate.resources.txt`](runtime/generate.resources.txt) | All 7,664 payloads generated; inference/write timing and whole-process time/RSS |
| [`runtime/esmc_checkpoint_identity.json`](runtime/esmc_checkpoint_identity.json) | Exact public weight revision and content SHA-256 |
| [`runtime/source_revision.json`](runtime/source_revision.json), [`runtime/local_tests.json`](runtime/local_tests.json) | Admission optimization deployed before any neural fit; 114 passing targeted tests |
| [`runtime/feature_certification_summary.json`](runtime/feature_certification_summary.json), [`runtime/certify.resources.txt`](runtime/certify.resources.txt) | All 7,398 graphs certified; full inventory hash and verified host-backup acknowledgment; 32m28s certification |
| [`runtime/source_revision_graph_cache.json`](runtime/source_revision_graph_cache.json) | Final graph-cache source deployed before any neural smoke or full fit |
| [`runtime/resume_tests.json`](runtime/resume_tests.json), [`runtime/resume_regression_tests.log`](runtime/resume_regression_tests.log) | 114 campaign regressions plus 23 cache tests passed; separate cache-enabled training/replay check |
| [`runtime/smoke_validation_summary.json`](runtime/smoke_validation_summary.json) | All nine one-epoch GPU cases passed completion and independent replay; identities, checkpoint hashes and runtime profiles |
| [`runtime/first_full_fit_summary.json`](runtime/first_full_fit_summary.json) | First 50-epoch Only-ESM/direct-four/fold-0 fit, independent replay, selected metrics and 71-file verified host acknowledgment |
| [`runtime/deepmzyme_cuda_batch_probe.py`](runtime/deepmzyme_cuda_batch_probe.py), [`runtime/gvp_batch_probe.json`](runtime/gvp_batch_probe.json), [`runtime/late_batch_probe.json`](runtime/late_batch_probe.json) | Reproducible operational timing on copied smoke checkpoints; no saved weights or scientific comparison result |
| [`runtime/remaining_budget_forecast.json`](runtime/remaining_budget_forecast.json) | Measured costs, explicit forecast assumptions and proposed unapproved 30-hour / $34 gross continuation cap |
| [`runtime/session_closeout.json`](runtime/session_closeout.json) | Controller-confirmed stopped state and session accounting; no new allocation |
| [`runtime/esm_binding_screen_execution.json`](runtime/esm_binding_screen_execution.json) | Approved one-fit ESMC screen; three authorized starts rejected for L4 capacity, latest alternate-zone hint and local controller limitation recorded; no new fit |

The campaign manifest records the cohort-creation source snapshot. The GPU
receipt records the later frozen execution source
`68f1fa9a59d5db1e0de9dc39092756e8390acb2d90d3d22f9917e11dee05afaa`.
The later final training source is
`adc95c42261448dc9d35572a138a3b8349de79124d9708618f511c27a848dd23`.
These are distinct provenance moments; the original receipts are unchanged.

Full cohort/disposition tables, fold membership, sequence-plan CSV, source data,
structures, embeddings and checkpoints remain in canonical storage. Their
absence here is intentional: this is a compact evidence package, not a runnable
dataset bundle. No credentials, private cloud configuration, held-out files or
model checkpoints are included. Full remote environment setup and training
completion are not established by the ESMC sample receipt.

PMM's completed fivefold comparison has Grade 2 evidence. The individual full
neural fit has Grade 5 evidence; the unfinished neural grid and runtime probes
have Grade 6 evidence. No superiority, promotion, final-refit or held-out result
is present.
