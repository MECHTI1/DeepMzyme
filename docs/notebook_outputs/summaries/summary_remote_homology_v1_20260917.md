# Validation checkpoint reuse and sequence remoteness — 2026-09-17

**Twelve existing selected checkpoints reproduced their saved validation
metrics on CPU. No training, GPU allocation, new split, promotion or held-out
evaluation occurred.** The ≤20% interaction is **unestimable** in the completed
represented-coordinate-chain diagnostic. This is not evidence for or against
an advantage of GVP on remote homologs.

The [addendum](../../REMOTE_HOMOLOGY_ADDENDUM.md) owns the protocol;
[portable evidence](../raw/remote_homology_v1_20260917/README.md) preserves exact
receipts and reports. [Current status](../../../EXPERIMENT_STATUS.md) owns the
remaining campaign gates.

## Completed work

- Reused four original direct-four metal GVP/ESM checkpoints, four matched
  geometry-shell GVP RING off/on checkpoints, and four EC1 GVP/ESM checkpoints.
  The selected recipes and seeds remain exactly as recorded in the reuse ledger.
- Verified frozen model/training source, exact saved validation identities,
  selected checkpoints, saved normalization and whole-validation metrics.
  The final resume pass verified all twelve exports without repeating inference,
  including all 48 bound checkpoint/configuration/metadata/membership files.
- Preserved the historically absent ESM inputs for Only-GVP. EC protein
  predictions use mean logits, with explicit accession/group mapping.
- Prepared 1,270 metal and 793 EC coordinate sequences, plus exact memberships
  for the twelve runs and five already-prepared metal folds.
- Completed one pinned CPU MMseqs2 search per task. Successful elapsed times
  were 18.721 seconds for EC and 31.462 seconds for metal. The initial EC 1 GiB
  prefilter request failed before producing alignments; its log is preserved.
  The successful requests used the 2 GiB split-memory limit. This is not a
  total-process memory guarantee.
- Froze 3,753 **run/fold annotation rows** before inference. Repeated examples
  across runs remain the same scientific units, not extra independent proteins.
- Produced separate metal/EC reports with primary 80% and sensitivity 50%
  shorter-sequence coverage and the prespecified component bootstrap.

## Support and interpretation

Primary coverage on each fixed validation membership:

| Task | >30% pockets / groups | (20%,30%] pockets / groups | ≤20% qualifying-hit pockets | No qualifying hit pockets / groups |
|---|---:|---:|---:|---:|
| Metal | 159 / 85 | 7 / 3 | 0 | 42 / 22 |
| EC1 | 61 / 17 | 24 / 5 | 0 | 90 / 20 |

None of the five prepared metal folds contains qualifying-hit ≤20% examples
under either coverage rule. No-hit examples remain **unclassified**, not 0%
identity. Missing significance/coverage-qualified hits do not establish
absence of homology or certify an extreme-remoteness bin.

Full-protein readiness failed explicitly: the 1,270 metal chains lack certified
full-protein/site-chain provenance and seven CARE training mappings differ
from coordinate sequences. No full-protein FASTA or primary full-protein result
was substituted. EC7 has only six development proteins, independently limiting
the proposed seven-class support gate.

Even the >30% metal contrasts remain descriptive: GVP−ESM is +4.883 percentage
points, paired 95% component interval [−10.604, 26.308]; matched RING-on−off is
+0.531 points, interval [−4.997, 4.910]. Both fail the prespecified component
support floor. The EC >30% stratum lacks EC5, so its full-seven-class BA and
contrast are undefined. No architecture is promoted from these strata.

## Integrity and implementation findings

MMseqs2's exported `nident` understated the exact aligned-character count by
one in 107 metal and 16 EC rows. The parser validates alignment strings against
the frozen sequences, recounts identities and records the discrepancy. Larger
unexplained discrepancies or identical ambiguous symbols fail closed. This
protects the 20%/15% boundaries from serialization rounding.

One exporter worker overlapped a change that added metadata-file hash checks.
Its original receipt hashed the on-disk exporter at completion rather than the
earlier source it executed. The separate
[source audit](../raw/remote_homology_v1_20260917/exporter_source_audit.json)
preserves both exact source versions, the patch, process/file timing and all
original receipt hashes. The executed version is reconstructed from that
evidence; no process-memory source hash was captured. The change affected only
file verification, with no numerical inference/model changes, and all frozen
scientific sources and bound original files were reverified. Original receipts
remain unchanged. Future workers execute an immutable source snapshot and check
its hash at entry and completion.

The integrated implementation suite passed **261 tests**. After the final
exporter source-isolation change, **14 exporter tests** passed, including six
subtests. These are engineering checks, not additional model experiments.

## Runtime work and remaining gates

The generic Stage 6 importer/command builder now preserves the five audited
scientific settings. Durable pause/lease/accounting controls and a maintained
same-endpoint supervisor are implemented with fake-provider failure tests;
the installed Colab CLI argument shapes were checked without provider calls.
Session 9's 384.788991 seconds were reconciled once, bringing both ledgers to
8,041.997553 seconds. A second apply added nothing. The GPU pause remains active.

A verified offline source/data-identity snapshot is available locally. It is
not a fully prepared operational continuation. Live controller transport and
throughput, remaining discovery/confirmation, EC workflow migration, cross-task
certification, final refit and a scientific final-test-route decision remain
open. No new ≤20% blocked training campaign is justified by this diagnostic.
