# Independent metal evaluation preparation

Status: offline preparation only; the primary test remains unresolved. This is a proposed temporal experimental
PDB source, not an assembled or certified test. [Plan](../../../Plan.md#pipeline-design-trade-offs) owns scientific
policy; [DATASETS](../../DATASETS.md#final-reporting-status) owns test identities and access history. No candidate
query, download, new held-out read, embedding generation, prediction or evaluation is authorized by this document.

The [offline tool](../../../metal_completion_evaluation.py) writes a proposed RCSB query and its preparation receipt.
It has no network client and reads only the explicitly supplied, hash-pinned metadata manifest plus its own source
and this protocol. It never follows an evidence reference into a dataset or test file.

The proposed source is protein-containing experimental RCSB/wwPDB structures initially released strictly after the latest initial release
of every accounted-for previously used source entry, through an explicitly supplied upper date. Previously used or
opened PDB IDs are excluded independently of date. The payload uses the [official Search API](https://search.rcsb.org/)
and its [attribute schema](https://search.rcsb.org/rcsbsearch/v2/metadata/schema); the 2026-10-14 implementation
session reported inspecting that schema but recorded no snapshot hash, so the attribute names must be re-verified
against the live schema before any authorized submission. API submission and request-size/pagination validation are
untested; a large exclusion list may need an equivalent reviewed retrieval strategy. The upper date may not lie in the
future, and the receipt records the preparation date, so the frozen window cannot grow with the execution date.
Query preparation does not establish that a nonempty, adequate cohort exists.

## Input and offline interface

Run `metal_completion_evaluation.py template --output <new-path.json>` with the AGENTS-prescribed interpreter to
write a deliberately blocked template. It contains no invented dates or verified-coverage declarations. After a
separately authorized provenance review, `prepare --provenance <path> --provenance-sha256 <sha256>
--through-date <YYYY-MM-DD> --output-dir <new-directory>` produces `query.json`, a byte-identical `provenance.json`
and `preparation.json` (which records `prepared_on`). A through date later than the preparation day is refused.
Existing output directories are refused. No live query is submitted.

The manifest schema is `metal-completion-evaluation-provenance-v1`:

- `metadata_only: true`; a `verification` object with `status: verified`, a nonempty `review_reference`, its
  `evidence_sha256`, and true declarations for all supervised-development sources, previously opened test IDs and
  custom split variants. These are review attestations, not independently proved by the preparer.
- `sources`: unique source IDs, verified status, an evidence SHA-256, `use_status` and `pdb_ids`. Required accounting
  covers every family in the [DATASETS overview](../../DATASETS.md#dataset-overview): `pinmymetal-source`,
  `pinmymetal-zenodo-exact`, `pinmymetal-exact`, `pinmymetal-nonoverlap`, `pinmymetal-harsh`, `pinmymetal-common70`,
  and the EC-side `clean30-original`, `clean30-conservative`, `care-task1-clusterres30` and `care-task1-legacy30`,
  whose structures the standalone metal model never saw and which are therefore expected to be recorded as
  `not_used` with an evidence-backed reason; add every other used source. `not_used` needs that reason and no IDs;
  missing evidence remains unresolved. A family name alone never proves complete membership coverage.
- `entries`: exactly one record per excluded PDB ID with only `pdb_id`, `initial_release_date` and `evidence_sha256`.
  Missing/unverified dates block preparation; do not replace them with download dates, revisions or guesses.

All entries are metadata about prior use, not candidate-test labels or model scores. SHA-256 pins bind supplied
metadata and review attestations; they do not authenticate a false declaration. A reviewer must establish their
meaning and completeness before recording `verified`. This task has not performed that review or created a real
verified manifest. The example values in tests are synthetic and must never serve as actual provenance.

## Scientific contract and remaining gates

- Keep known-ion classification. The objectives are four, five and six classes, all reporting Mn/Cu/Zn/Class VIII
  (Fe+Co+Ni), with native five/six metrics retained. The query covers declared elemental CCD aliases of all six metals,
  including common oxidation-state aliases; cofactors are outside this proposal. After authorized retrieval, verify
  chemical-component identity against the frozen alias list and atom records; do not claim exhaustive CCD coverage.
- An entry hit is only a candidate. Before opening structures, freeze atom identity, alternate/mixed occupancy,
  missing-context, quality and unscorable-case rules. Preserve the v3 input contract and explicit missing-symmetry
  treatment. Do not infer catalytic or physiological truth from PMM compatibility fields, AlphaFill or MAHOMES.
  Deposit-derived labels need their own provenance; target element and label-derived chemistry cannot enter features.
- Before inclusion, certify exclusions across all prior development, historical opened tests and custom variants,
  using every represented context chain and the exact frozen v3 near-copy rule. Preserve PDB/related-entry groups.
  Temporal separation alone does not prove sequence separation; near-copy separation is not remote-homology proof.
  Report similarity strata and disclose possible pretrained-encoder sequence exposure.
- Before data collection or selection, preregister a precision objective and group-based support requirement for
  every common-four class, plus native rare-class reporting. No sample-size threshold is fabricated here. An
  authorized metadata feasibility pass may establish whether the proposed rule is attainable; no predictions may
  guide that decision. Insufficient support is a blocker to a strong primary claim, not permission to relax rules
  after scores are known.
- Freeze query, eligibility/disposition rules, source snapshots and provenance before candidate materialization.
  Keep labels outside development selection. Freeze the selected configuration, full non-test refit, reporting set,
  calibration/ensemble rule, exclusions and denominator policy before one-shot scoring. Record access in DATASETS.
  Preserve v3's historical secondary reports; do not rerun them to select this successor.

Completion of this preparation means only tested offline tooling and a reviewable protocol. Actual source dates,
verified complete source coverage, precise quality/support rules, candidate retrieval authority, cohort certification,
and the final model/report freezes remain prerequisites. None is implied by a `query.json` file.
