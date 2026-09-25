# PMM ion v2 context campaign — portable evidence

This is the 2026-09-25 21:56 UTC snapshot summarized
[here](../../summaries/summary_pmm_ion_v2_context_20260926.md). Files below are
byte-identical copies from
`/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v2_context/`.
[`SHA256SUMS`](SHA256SUMS) records their copied checksums; verify with
`sha256sum -c SHA256SUMS` from this directory.
The later feature-generation and implementation-check receipts were added at
22:24 UTC; the original snapshot receipts remain unchanged.

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

The campaign manifest records the cohort-creation source snapshot. The GPU
receipt records the later frozen execution source
`68f1fa9a59d5db1e0de9dc39092756e8390acb2d90d3d22f9917e11dee05afaa`.
These are distinct provenance moments; the original receipts are unchanged.

Full cohort/disposition tables, fold membership, sequence-plan CSV, source data,
structures, embeddings and checkpoints remain in canonical storage. Their
absence here is intentional: this is a compact evidence package, not a runnable
dataset bundle. No credentials, private cloud configuration, held-out files or
model checkpoints are included. Full remote environment setup and training
completion are not established by the ESMC sample receipt.

PMM's completed fivefold comparison has Grade 2 evidence. The unfinished neural
grid and runtime preflight have Grade 6 evidence. No neural-fit, superiority,
promotion, final-refit or held-out result is present at this snapshot.
