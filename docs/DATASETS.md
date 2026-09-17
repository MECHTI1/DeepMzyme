# DeepMzyme Dataset, Split, Bundle, and Test-Use Reference

This is the authoritative human-readable record of DeepMzyme datasets, split
relationships, preparation status, bundle inclusion, and known test access.
Scientific split and evaluation policy remains in [`Plan.md`](../Plan.md).
Preparation procedures remain in their pipeline directories.

Last broad evidence audit: 2026-09-14. Pilot PROPKA overlay audit: 2026-09-15.

## Final-reporting status

> **Primary final-test route: unresolved scientific decision required before final reporting.**

Relevant facts:

- The legacy non-overlap PinMyMetal test was historically evaluated in seven
  early Only-GVP runs and is not pristine or unopened.
- Exact PinMyMetal contains train/test PDB-ID overlap.
- The non-overlapped PinMyMetal root is present locally but its historical test
  was already accessed; it is included in v11/v12 (absent from historical v10).
  The harsh root is absent locally and from v10/v11/v12.
- CLEAN and CARE datasets have different scientific purposes and cannot be
  silently designated as replacement final tests.
- This documentation cleanup does not select a replacement dataset or change
  evaluation behavior.

The historical test values are preserved as access evidence, not parameter or
model-selection evidence.

### Bounded metal architecture pilot and later PinMyMetal comparison

The bounded architecture pilot uses only the **train** membership of
`train_and_test_sets_structures_non_overlapped_pinmymetal`, with internal
`pdbid`-grouped validation and a shared native-six-eligible cohort across
four-, five-, and six-class targets. Exact parameter values and the runnable
profile are owned by the
[metal playbook](METAL_TRAINING_PIPELINE_PLAYBOOK.md#bounded-metal-architecture-pilot--stage-0-through-stage-2b).
This prevents exact-test PDB IDs from entering pilot training; it does not
establish homology separation or resolve the primary final-test route.

The pilot's repaired PROPKA external features form a training-only overlay with
their own manifest/checksum. They do not change the archived v12 bundle or
scientific split membership. Record both the immutable base-bundle identity
and the overlay identity in the campaign; cache-file counts alone do not prove
that an external channel was successfully measured.

The 2026-09-15 overlay audit passed for **all 1,304 non-overlap training
structures**, verifying file hashes and unchanged geometry/structure identities.
It covers 484,682 residue rows: 148,607 have computed pKa-derived features and
336,075 retain missingness, including non-titratable and structurally incomplete
residues. Forty-seven structures were refreshed after parser/alignment repairs:
46 for compact wide-residue tokens and `3q6v` for insertion-code handling.
Original v12 cache files remain unchanged. This audit does **not** certify or
repair other legacy datasets' caches or any held-out caches. The
[audit](notebook_outputs/raw/metal_architecture_pilot_20260915/preparation/feature_overlay_audit.json),
[overlay manifest](notebook_outputs/raw/metal_architecture_pilot_20260915/preparation/feature_overlay_manifest.json),
and [pilot summary](notebook_outputs/summaries/summary_metal_architecture_pilot_20260915.md)
preserve exact scope, missingness, hashes, and source provenance.

Exact PinMyMetal remains a proposed secondary reference comparison after
validation-based selection and final-refit/reporting choices are frozen.
Reconstruct an auditable mapping from original site identifiers to evaluated
DeepMzyme pockets, recording exact, ambiguous, and unmatched sites. Report on
the traceably matched cohort with the same metric definition and disclose
overlapping PDB IDs. DeepMzyme's supplied metal-centered pocket input must be
distinguished from any benchmark that also predicts candidate sites/coordinates.

Exact and non-overlap retain the same test structures. The absence of an
exact-trained test report does not make that shared test newly untouched:
the non-overlap access ledger still applies. A later comparison of frozen
exact-trained and non-overlap-trained refits requires the final reporting
protocol to be resolved and implemented first; neither test result may tune
the other refit. The pilot does not launch these benchmark evaluations.

## Status vocabulary

### Development sequence provenance for remoteness reporting

The [remote-homology addendum](REMOTE_HOMOLOGY_ADDENDUM.md) uses only existing
external-training memberships and their internal validation/fold partitions.
Preparation on 2026-09-17 found hash-valid coordinate-sequence sidecars for all
1,270 retained metal structures and 793 EC proteins. These are represented
coordinate-chain sequences; they do not certify biological full-length proteins
or all chains of the original metal site. PDB grouping does not establish
sequence separation. No scientific split was changed.

CARE source sequences agree with coordinate sequences for 786 proteins. The
seven training discrepancies are `B7G620`, `A4VCL2`, `Q01433`, `P23109`,
`O88483`, `Q96KN2`, and `Q96MI6`. Full-protein preparation fails until these
mappings and metal full-sequence provenance are resolved. Diagnostic coordinate
reporting retains its separate endpoint label. Source/coordinate differences do
not authorize regenerating completed model features.

Generated manifests and blocker reports are under
`DeepMzyme_Data/notebook_outputs/remote_homology_v1/`. They include exact saved
training/validation membership and the five already-prepared metal folds. No
held-out sequence or prediction evaluation is added. Family/clan/domain
coverage remains uncertified.

### Interpretation of inventory status

| Field | Meaning |
|---|---|
| Materialized | Dataset structures/tables were found in the current local data tree |
| Membership available | Train/test or fold membership can be inspected, even if structures are absent |
| Evaluation found | A completed model evaluation artifact was found during the audit |
| Selection use established | Repository evidence establishes that evaluation values affected a later choice |
| Protected | Current records say the data must not be used for further tuning or selection |

“No evaluation found” means exactly that: the inspected repository and local
artifacts did not show one. It does not prove that an evaluation never occurred
elsewhere.

## Metal-label provenance vocabulary

Keep metal identity and its source explicit in dataset records and experiment
reports:

| Provenance category | Meaning | Required interpretation |
|---|---|---|
| Experimentally observed metal | Metal identity supported by an experimental structure or assay record | Describe the specific experimental source; availability still does not make metal type deterministically equivalent to EC |
| Curated database annotation | Metal identity asserted by a curated database record | Name the database and accession or evidence field when available |
| Computationally transferred assignment | AlphaFill/MAHOMES or another computational procedure transferred or inferred a metal/site assignment | Label the method and thresholds; do not call the assignment perfect ground truth |
| Model-predicted metal | A trained model produced a metal label, probability vector, or embedding | Keep it separate from observed, curated, and transferred labels and record the source model/checkpoint |

Known-metal-to-EC analyses may use an available category as an explicitly
labeled diagnostic or special inference mode. Never merge these provenance
categories silently. The scientific conditioning and auxiliary-learning policy
is owned by [`Plan.md`](../Plan.md).

## Dataset overview

| Dataset ID | Scientific purpose | Materialized locally | In current v12 bundle | Test/fold evaluation record | Current interpretation |
|---|---|---:|---:|---|---|
| `pinmymetal-source` | Original PinMyMetal class-model membership and site provenance | Source files tracked | No, source membership only | Not an executable split by itself | Primary membership evidence |
| `pinmymetal-exact` | Supported-structure projection preserving original train/test side | Yes | Yes | No completed test evaluation found | Possibly overlapped comparison/validation route |
| `pinmymetal-nonoverlap` | Remove exact-test PDB IDs from train; retain the original exact test | Yes | Yes | Seven early test evaluations found | Historically accessed; not pristine |
| `pinmymetal-harsh` | Put all common exact-split PDB IDs on the test side | No | No | No evaluation found | Documented severe comparison variant |
| `pinmymetal-common70` | Custom zero-overlap assignment of common PDB IDs, seed 42 | Yes | Yes | No evaluation found | Custom comparison split, not a selected final test |
| `clean30-original` | CLEAN official split30 fold benchmark with shared multi-donor structures | Yes | Yes | Fold evaluation is the intended benchmark design | Five fold pairs; report aggregate across folds |
| `clean30-conservative` | One deterministic supported-metal AlphaFill donor per CLEAN target/fold | Yes; current `CLEAN_30_main` | Yes | No completed DeepMzyme/CLEAN-predictor result found in inspected evidence | Current preferred CLEAN metallo source |
| `clean10` | Potential CLEAN 10%-identity benchmark | No | No | No evidence found | Not present or documented |
| `care-task1-legacy30` | Older CARE Task 1 30%-identity preparation route | Scripts/docs only | No distinct legacy root in v12 | No evaluation found | Historical/secondary preparation track |
| `care-task1-clusterres30` | Representative CARE Task 1 metallo subset for EC/joint work | Yes | Yes | Test prepared and bundled; no completed evaluation found | Current prepared CARE route |

## Local structure storage

The 2026-09-14 byte-level audit found 10,875 logical references across 20
dataset/fold directories, backed by 3,211 distinct structure contents. The
local data tree now stores those 3,211 objects once under
`DeepMzyme_Data/structure_store/objects/`. Every dataset or CLEAN fold view uses
`structure_manifest.csv` to declare membership. No structure file remains in a
split directory, and no structure symlink remains.

This deduplication does not rewrite scientific membership. In particular, 179
byte-identical structure filenames remain members of both exact PinMyMetal
sides, corresponding to 177 overlapping PDB-ID groups. Conversely, 265 CARE
and CLEAN filenames have different bytes and are retained as separate
hash-addressed objects. Their PDB coordinate/site records are identical; only
their CARE/CLEAN `HEADER` and `COMPND` provenance text differs. See
[`STRUCTURE_STORE.md`](STRUCTURE_STORE.md) for the full audit, manifest schema,
pairwise interpretation, and verification command.

## PinMyMetal

### Original membership

Authoritative source files:

- [`classmodel_train_set`](../prepare_training_and_test_set/pinmymetal_files/classmodel_train_set)  
  SHA256 `4748babd2b6ac0706cd9ed4bcfd4855c8d2c5535f01813fb2e10b68b31a24c0f`
- [`classmodel_test_set`](../prepare_training_and_test_set/pinmymetal_files/classmodel_test_set)  
  SHA256 `ec6427ad0b8a18261dbc1d6822bdcfe7194730c7da4a36c208957bcb00f6a0f2`

| Source side | Rows | Unique PDB IDs |
|---|---:|---:|
| Train | 7,920 | 4,195 |
| Test | 1,488 | 1,179 |

There are 668 PDB IDs on both source sides. These files preserve the original
PDB, `residueid_ion`, `metalid`, and other row-level source fields and must
remain byte-identical.

Preparation scripts and Step 5 notebooks are under
[`prepare_training_and_test_set/`](../prepare_training_and_test_set/).
In Step 5 output, `native=0` can mean an unknown or unsupported chain
annotation; it is not by itself proof that a site is non-native.

### Exact PinMyMetal

Path:
`DeepMzyme_Data/train_and_test_sets_structures_exact_pinmymetal`

Purpose: project the original train/test membership onto available supported
structures without resolving PDB IDs that occur on both source sides.

| Measure | Train | Test | Overlap |
|---|---:|---:|---:|
| Available structure files | 1,483 | 316 | 179 identical filenames |
| Available unique PDB IDs | 1,472 | 313 | 177 |
| Primary site-level rows | 2,144 | 490 | — |

The current DeepMzyme summary tables do not retain the original PinMyMetal
`residueid_ion`/`metalid` identifiers. “Exact” therefore describes available
PDB-ID/structure-side membership, not exact reconstruction of every source site
row.

Status:

- Materialized locally: yes.
- Included in v10: yes.
- Test membership and labels: materialized.
- Completed test evaluation found: no.
- Selection use established: no.
- Protection: must remain labeled exact/possibly-overlapped; it is not silently
  interchangeable with a zero-overlap final split.

"No completed test evaluation found" here means no exact-trained evaluation
artifact was identified. The same test structures occur in the historically
accessed non-overlap test; see the shared-test qualification above.

Tracked generated metadata:
[`prepare_training_and_test_set/provenance/exact/`](../prepare_training_and_test_set/provenance/exact/).

### Non-overlapped PinMyMetal

Path:
`DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal`

Construction intent: remove every exact-test PDB ID from train while retaining
the original exact test side. The intended PDB-ID overlap is zero.

Status:

- Membership construction code: tracked.
- Materialized locally: yes; 1,304 train structures and 316 test structures
  are represented by manifests in the shared structure store.
- Included in v10: no.
- Historical model evaluations found: exactly seven.
- Test pockets per report: 352.
- Selection use established: not established by repository evidence.
- Protection: historically accessed; do not use its metrics for current HPO,
  model ranking, promotion, or rejection.

> The non-overlap PinMyMetal test was historically evaluated in seven early
> runs and is therefore not pristine or unopened. Whether those values
> influenced subsequent selection is not established by repository evidence.
> These test metrics must not be used for current HPO recommendations or model
> selection.

The seven exact configurations, test reports, per-class recalls, commands, and
metrics are preserved under
[`legacy_nonoverlap_test_access/`](notebook_outputs/raw/legacy_nonoverlap_test_access/).
The archived early narrative is
[`experiment_notes_legacy.md`](archive/experiments/experiment_notes_legacy.md).

### Harsh PinMyMetal

Historical path:
`DeepMzyme_Data/train_and_test_sets_structures_harsh_pinmymetal`

Construction intent: retain train-only PDB IDs in train and put test-only plus
every common exact-split PDB ID in test as whole PDB groups.

Status:

- Construction code: tracked.
- Materialized locally: no.
- Included in v10: no.
- Completed evaluation found: no.
- Selection use established: no.

The existing script filename
`step6_create_additional_split_non_overalpped_structures.py` and its argparse
description use inconsistent “non-overlapped”/“harsh” terminology. This cleanup
does not rename or modify the script.

### Common-PDBID 70/30 PinMyMetal

Path:
`DeepMzyme_Data/train_and_test_sets_structures_common_pdbid_70_30_pinmymetal`

Purpose: retain train-only/test-only PDB IDs on their original sides while
assigning only the 177 common exact-split PDB IDs as whole groups, using seed
`42`.

| Measure | Train | Test |
|---|---:|---:|
| Final PDB IDs | 1,419 | 189 |
| Common PDB IDs assigned | 124 | 53 |
| Primary site-level rows | 2,042 | 271 |

Final train/test PDB-ID overlap is zero.

Status:

- Materialized locally: yes.
- Included in v10: yes.
- Test membership and labels: materialized.
- Completed evaluation found: no.
- Selection use established: no.
- Scientific role: custom comparison split, not an automatically selected
  final test.

Tracked generated metadata:
[`prepare_training_and_test_set/provenance/common_pdbid_70_30/`](../prepare_training_and_test_set/provenance/common_pdbid_70_30/).

## CLEAN

CLEAN folds are benchmark train/test fold pairs. They are not five additive
partitions of one train/test split and are not a one-shot sealed final-test
route. Report all-five-fold aggregates when making CLEAN benchmark claims.

The metal/catalytic assignments are computational AlphaFill/MAHOMES-derived
evidence, not experimental validation.

### CLEAN30 original shared

Path: `DeepMzyme_Data/CLEAN_30_shared`

- Identity family: CLEAN `split30`.
- Shared structures: 740.
- Structure storage: 740 manifest references into the global content-addressed
  store; no separate shared hardlink copy.
- Scientific role: multi-donor reference.

| Fold | Train sites/structures | Test sites/structures |
|---:|---:|---:|
| 0 | 1,102 / 622 | 229 / 118 |
| 1 | 1,063 / 578 | 179 / 109 |
| 2 | 1,024 / 562 | 220 / 121 |
| 3 | 1,034 / 586 | 208 / 98 |
| 4 | 1,101 / 612 | 187 / 99 |

Tracked metadata:
[`CLEAN_prepare_training_and_test_set/provenance/original_shared/`](../CLEAN_prepare_training_and_test_set/provenance/original_shared/).

### CLEAN30 conservative/current main

Paths:

- `DeepMzyme_Data/CLEAN_30_shared_single_donor_supported_metal_conservative`
- `DeepMzyme_Data/CLEAN_30_main` — local symlink to the path above at audit time

Purpose: select one AlphaFill donor per UniProt target within each official
CLEAN fold using deterministic quality tie-breakers. The construction retains
supported transition metals and applies `2.0 Å` within-donor site
deduplication.

Exact metal stoichiometry was not supplied. Every retained target is therefore
recorded as `metal_supported_but_count_unknown`; this must not be interpreted as
an exact metal-count label.

| Fold | Train source → retained sites | Test source → retained sites |
|---:|---:|---:|
| 0 | 1,102 → 743 | 229 → 139 |
| 1 | 1,063 → 698 | 179 → 128 |
| 2 | 1,024 → 668 | 220 → 148 |
| 3 | 1,034 → 696 | 208 → 119 |
| 4 | 1,101 → 723 | 187 → 124 |

All source targets were retained; site-count reduction reflects donor
selection rather than target removal.

Status:

- Materialized locally: yes.
- Included in v10 and CLEAN predictor v2: yes.
- Preferred current CLEAN metallo source: yes.
- Completed model-result evidence found during audit: no.

Tracked selection audit and metadata:
[`CLEAN_prepare_training_and_test_set/provenance/conservative_single_donor/`](../CLEAN_prepare_training_and_test_set/provenance/conservative_single_donor/).
Procedure:
[`CLEAN_prepare_training_and_test_set/README.md`](../CLEAN_prepare_training_and_test_set/README.md).

The materialized roots named `CLEAN_30_train_test_split_0` through
`CLEAN_30_train_test_split_4` contain train/test structure manifests. Their
existing marker records point to the original `CLEAN_30_shared` source. A fold
directory name alone therefore does not identify whether original or
conservative metadata was used; the notebook rewrites these views from the
selected shared source when its source/version marker changes.

### CLEAN10

No `CLEAN10`/`CLEAN_10` root, preparation script, notebook option, or active
documentation was found. Status: **not present or documented**.

## CARE

CARE metallo subsets are computationally filtered AlphaFill/MAHOMES
preparations. They are not the full CARE benchmark and are not experimental
validation.

### Legacy Task 1 `30_identity`

The older preparation route and commands remain in
[`CARE_prepare_training_and_test_set/README.md`](../CARE_prepare_training_and_test_set/README.md).
Its expected root is `CARE_task1_30_train_test_metallo`.

Status:

- Preparation scripts/documentation: present.
- Distinct materialized legacy output root: not found.
- Current notebook/bundle route: no; compatibility aliases may resolve the old
  name to clusterRes30 but do not make the scientific datasets identical.
- Completed evaluation found: no.

### Task 1 clusterRes30

Path:
`DeepMzyme_Data/CARE_task1_30_clusterRes30_train_test_metallo`

Purpose: use `clusterRes30` representatives from CARE Task 1 train data and the
30%-identity test source, then apply UniProt supported-metal filtering,
AlphaFill structure preparation, and MAHOMES catalytic-site filtering.

Preparation thresholds:

- AlphaFill identity at least `0.30`;
- alignment length at least `85`;
- site deduplication distance `1.0 Å`;
- UniProt policy `require_supported`;
- supported metals `CO/CU/FE/MN/NI/ZN`.

| Funnel stage | Train | Test |
|---|---:|---:|
| Selected CARE source rows | 10,321 | 432 |
| Unique proteins | 9,466 | 432 |
| UniProt-supported proteins | 1,769 | 115 |
| AlphaFill-fetched proteins | 1,021 | 45 |
| Catalytic/exported structures | 817 | 34 |
| Catalytic sites | 1,520 | 76 |

The full train source contained 184,529 rows before representative filtering.
There were 9,594 unique `clusterRes30` representatives, 9,466 with source
entries and 128 without matching entry rows. Exact missing representative and
test-EC lists remain in the tracked audit JSON.

Status:

- Materialized locally: yes.
- Included in v10 and CLEAN predictor v2: yes.
- Test membership, structures, and labels: prepared and bundled.
- CARE ESM/external/RING caches completed locally on 2026-09-14; three
  validation-only EC1 Colab GPU smoke runs passed. See the
  [repair and execution evidence](notebook_outputs/summaries/summary_colab_care_cache_smoke_20260914.md).
- Completed test evaluation found: no.
- Selection use established: no.
- Upstream CARE repository/source URL or formal citation: not found; provenance
  gap remains open.

Tracked metadata:
[`CARE_prepare_training_and_test_set/provenance/clusterRes30/`](../CARE_prepare_training_and_test_set/provenance/clusterRes30/).

## Test-use ledger

| Dataset | Labels/membership materialized | Evaluation artifacts found | Selection influence established | Current record |
|---|---:|---:|---:|---|
| Exact PinMyMetal | Yes | No | No | Possibly overlapped; label every use |
| Non-overlapped PinMyMetal | Present locally and in v12; absent from historical v10 | Yes — seven early reports | Not established | Historically accessed; metrics excluded from current selection |
| Harsh PinMyMetal | No current root | No | No | Availability must be restored before use |
| Common-PDBID 70/30 | Yes | No | No | Custom comparison only |
| CLEAN30 fold pairs | Yes | No completed result found | No | Evaluate as five-fold benchmark, not sealed one-shot test |
| CARE clusterRes30 | Yes | No | No | Prepared/bundled test; do not equate preparation with evaluation |

**2026-09-15 incidental CARE metadata access:** while locating a development
PDB-to-UniProt mapping for the metal–EC association analysis, a broad source
search printed snippets from
`CARE_prepare_training_and_test_set/provenance/clusterRes30/metadata/test/candidate_site_summary.csv`.
The truncated output included identifiers, metal and EC annotations, catalytic
and selection fields, and AlphaFill/cluster provenance. It also printed the
header of that directory's
`data_summarazing_table_transition_metals_whether_catalytic.csv`. The exact
number of displayed candidate rows was not retained. No test predictions,
aggregate association statistics, training, or model selections used those
snippets. Subsequent association inputs are explicitly allowlisted development
artifacts. This records metadata exposure; it is not a held-out model
evaluation or a claim that no test metadata has ever been viewed.

## Current bundles

### Hugging Face repository inventory

Repository:
[`GMBioinformatics/DeepMzyme`](https://huggingface.co/datasets/GMBioinformatics/DeepMzyme)

The repository tree was rechecked through the Hugging Face dataset API on
2026-09-14. The historical inventory below is retained; the current v12 release
is listed in its own section. V10 and its checksum, plus the CLEAN predictor
checksum, were subsequently removed from `main`; historical download links
below are pinned to the earlier publication commit. Those deletions were
preserved when adding v12. The dataset-card `README.md` is only a 31-byte license
header, so it is not currently a useful file or provenance index. This section is the
human-readable inventory.

| Hugging Face path | Size (bytes) | SHA256 / verification | Purpose |
|---|---:|---|---|
| `.gitattributes` | 2,504 | Not applicable | Git LFS rules |
| `README.md` | 31 | Not recorded | License-only dataset card header |
| `CLEAN_predictor_baselines_v2_clean30x5_single_donor_supported_metal_conservative_care30_sources.tar.zst` | 29,237,125 | `5124b0b514b49affc158df121a87f5389ec1e027d14e0cf0a53cfb13a602c0f0` | CLEAN predictor/baseline bundle described below |
| `CLEAN_predictor_baselines_v2_clean30x5_single_donor_supported_metal_conservative_care30_sources.tar.zst.sha256` | 170 | Contains the archive checksum | Portable checksum sidecar |
| `DeepMzyme_Data_v10_exact_common70_clean30main_clean30x5_care30_esm_ring_external.tar.zst` | 3,822,130,168 | `09525aad00d6c35e32a3601ff3ecf338978c465cec1ccfc18e47b9222b220aba` | Main Colab training/data bundle described below |
| `DeepMzyme_Data_v10_exact_common70_clean30main_clean30x5_care30_esm_ring_external.tar.zst.sha256` | 155 | Contains the archive checksum | Portable checksum sidecar |
| `benchmarks/gvp_esm_hybrid_realistic_subset_v1/realistic_subset.json` | 1,087 | Local tracked copy SHA256 `f4660b80ffeeb4e6e158791943a0dc5ba771461b5cf2a0080fff158ab1e7e6b5` | Historical v1 manifest; incomplete cohort provenance |
| `benchmarks/gvp_esm_hybrid_realistic_subset_v1/realistic_subset.pt` | 51,844,189 | `84e7e039f1df5b3a7b32dc3d4ac1b8fa21bba2827679b4d3f1650d394e2754bf` | Historical v1 `PocketData` pickle; requires DeepMzyme during unsafe loading and is not a plain portable PyG artifact |

Use the artifacts as follows:

- normal Colab DeepMzyme training: the current v12 archive plus its SHA256;
  see its complete CARE cache audit and release verification below;
- CLEAN predictor baselines without graph assets: the CLEAN predictor archive
  plus its SHA256;
- G4/A100 throughput reproduction only: both files under
  `benchmarks/gvp_esm_hybrid_realistic_subset_v1/`;
- scientific dataset membership, bundle contents, and test-use interpretation:
  the sections below, not the minimal Hugging Face dataset card.

Direct benchmark downloads:
[manifest JSON](https://huggingface.co/datasets/GMBioinformatics/DeepMzyme/resolve/4c36d3bfbf5e0c892fb165cbf101184e4151dda2/benchmarks/gvp_esm_hybrid_realistic_subset_v1/realistic_subset.json)
and
[historical v1 subset](https://huggingface.co/datasets/GMBioinformatics/DeepMzyme/resolve/4c36d3bfbf5e0c892fb165cbf101184e4151dda2/benchmarks/gvp_esm_hybrid_realistic_subset_v1/realistic_subset.pt).

The hosted v1 subset contains the project-defined
`graph.construction.PocketData` class and requires
`torch.load(..., weights_only=False)` with the project importable. Preserve it
and its hash as historical evidence; do not describe it as independently
portable. The local v2 builder and runner now define a tensor-only safe-loading
contract at [`bench/README.md`](../bench/README.md), but no v2 subset, G4/A100
result, or Hugging Face upload exists yet because regeneration and external
publication require separate authorization.

The benchmark subset is compute evidence only. V1 selected 240 pockets from 342
feature-complete eligible CARE training pockets; those 342 pockets are not the
full CARE training source, whose tracked provenance contains 1,520 catalytic
sites before the benchmark's strict label/ESM/RING/external-feature/structure
eligibility filters. V1 discarded the load/skip report, so its exact reduction
cannot be reconstructed from the manifest. It does not authorize held-out
evaluation or stand in for a model-quality dataset. Audited legacy G4/A100
results are summarized in [`EXPERIMENT_STATUS.md`](../EXPERIMENT_STATUS.md).

### Main Colab bundle v11 (historical hosted release)

- Filename: `DeepMzyme_Data_v11_manifest_exact_common70_nonoverlap_clean30_care30_esm_ring_external.tar.gz`
- Download: [v11 gzip archive](https://huggingface.co/datasets/GMBioinformatics/DeepMzyme/resolve/4c36d3bfbf5e0c892fb165cbf101184e4151dda2/DeepMzyme_Data_v11_manifest_exact_common70_nonoverlap_clean30_care30_esm_ring_external.tar.gz)
- Size: `3808796819` bytes.
- SHA256: `8f869b8aa78dd2dc2af5efb856d137c01ba289b8fe6327b66a8139b0166975c7`
- Verified Hugging Face publication commit: `4c36d3bfbf5e0c892fb165cbf101184e4151dda2`.
  Remote LFS SHA256 and byte size match the local archive; the downloaded
  checksum sidecar also matches.
- Sidecars: [checksum](https://huggingface.co/datasets/GMBioinformatics/DeepMzyme/resolve/4c36d3bfbf5e0c892fb165cbf101184e4151dda2/DeepMzyme_Data_v11_manifest_exact_common70_nonoverlap_clean30_care30_esm_ring_external.tar.gz.sha256), [file manifest and feature coverage](https://huggingface.co/datasets/GMBioinformatics/DeepMzyme/resolve/4c36d3bfbf5e0c892fb165cbf101184e4151dda2/DeepMzyme_Data_v11_manifest_exact_common70_nonoverlap_clean30_care30_esm_ring_external.tar.gz.manifest.json).
- The release also publishes `DeepMzyme_Data_v11_manifest_exact_common70_nonoverlap_clean30_care30_esm_ring_external.tar.gz.validation.json` with extraction,
  notebook dataset-resolution, CLEAN fold-materialization, and sampled loader checks.

This gzip release packages the current manifest-backed data layout and all
3,211 referenced structure-store objects. It includes exact, Common-PDBID
70/30, and historical non-overlapped PinMyMetal; both CLEAN30 shared sources
with the conservative `CLEAN_30_main` alias; CARE clusterRes30; and available
ESMC, external, RING, and RING-runtime files. CLEAN fold directories are
materialized from the selected shared source by the notebook. Harsh PinMyMetal
is absent. This is a data archive; source code comes from the GitHub checkout.

The publication repair adds the previously missing `structure_store.py`
dependency to GitHub. Use the updated notebook and a fresh/updated checkout.
The notebook supports `.tar.gz`; its current filename, pinned URL, and checksum
select v12 below. A separately saved older notebook requires updating those
three values. Non-overlapped data remains accessible with `DATASET_ROOT_OVERRIDE`;
adding it to the archive does not promote it as a scientific final-test route.

**Cache readiness is dataset-specific.** All included PinMyMetal structures
and all 740 shared CLEAN structures have ESM, external, and RING caches.
Available embedding metadata identifies `esmc_300m`, dimension 960, with no
missing sidecars. The unchanged **hosted v11 archive** has these CARE gaps:

| CARE side | Structures | Missing ESM | Missing external | Missing RING |
|---|---:|---:|---:|---:|
| Train | 817 | 571 | 572 | 572 |
| Test | 34 | 14 | 14 | 14 |

CARE in the unchanged v11 archive is included for preservation and preparation, but is **not ready for
full feature-dependent training without generation**. Repacking does not fill
these gaps. Missing-feature allowances must not silently change comparison
cohorts. Archive checks inspect data integrity and membership; they do not
train a model, evaluate held-out performance, or certify a GPU runtime.

Verification passed for all 15,126 inventoried file hashes, all relocated
structure manifests, every configured notebook dataset choice, and all five
CLEAN folds for each source option. Forty-eight deterministic feature-complete
structure samples loaded with no alignment errors (73 retained pockets).
The isolated release checkout passed 61 regression tests plus nine subtests,
and `src/train.py --help` succeeded. No Colab GPU training was run as part of
that v11 publication check. The later local CARE repair and three actual GPU
smoke runs are recorded separately below.

The archive contains `DeepMzyme_Data/bundle_metadata/v11/bundle_manifest.json`.
It records the exact input roots and per-file sizes/SHA256 values. Build with
`build_colab_bundle.build_bundle(selected_paths, output_bundle=...)`, supplying
the recorded roots plus that metadata directory. The builder now selects real
gzip for `.tar.gz`/`.tgz` outputs and retains `.tar.zst` support.

### Main Colab bundle v12 (current hosted release; CARE complete)

- Filename: `DeepMzyme_Data_v12_manifest_exact_common70_nonoverlap_clean30_care30_complete_esm_ring_external.tar.gz`
- Local directory: `DeepMzyme_Data/DeepMzyme_Colab_Bundles/releases/v12/`.
- Size: `4731180661` bytes.
- SHA256: `90c0899829e0ac5ca94a5ef34484b74ca1014d3e9b6b1fba90bd338ee3440dee`.
- Download: [v12 gzip archive](https://huggingface.co/datasets/GMBioinformatics/DeepMzyme/resolve/0f9d1efb835bdd5597b2762b8608947d7b82cdf2/DeepMzyme_Data_v12_manifest_exact_common70_nonoverlap_clean30_care30_complete_esm_ring_external.tar.gz).
- Verified Hugging Face publication commit: `0f9d1efb835bdd5597b2762b8608947d7b82cdf2`.
  Remote LFS size/SHA256 match the local archive; the public archive URL responds
  successfully and all three downloaded sidecars match their local SHA256 values.
- Published sidecars: [checksum](https://huggingface.co/datasets/GMBioinformatics/DeepMzyme/resolve/0f9d1efb835bdd5597b2762b8608947d7b82cdf2/DeepMzyme_Data_v12_manifest_exact_common70_nonoverlap_clean30_care30_complete_esm_ring_external.tar.gz.sha256),
  [manifest](https://huggingface.co/datasets/GMBioinformatics/DeepMzyme/resolve/0f9d1efb835bdd5597b2762b8608947d7b82cdf2/DeepMzyme_Data_v12_manifest_exact_common70_nonoverlap_clean30_care30_complete_esm_ring_external.tar.gz.manifest.json), [build validation](https://huggingface.co/datasets/GMBioinformatics/DeepMzyme/resolve/0f9d1efb835bdd5597b2762b8608947d7b82cdf2/DeepMzyme_Data_v12_manifest_exact_common70_nonoverlap_clean30_care30_complete_esm_ring_external.tar.gz.validation.json).
- Portable [validation receipt](notebook_outputs/raw/colab_care_cache_smoke_20260914/v12_bundle_validation.json)
  and [checksum](notebook_outputs/raw/colab_care_cache_smoke_20260914/v12_bundle.sha256)
  are tracked with the evidence. The separate [publication receipt](notebook_outputs/raw/colab_care_cache_smoke_20260914/v12_publication.json)
  records the later upload. The immutable build metadata and validation sidecar
  retain their pre-publication wording and `published: false`; they describe
  build-time state, not current hosting.

This successor preserves the v11 selected roots, all structure and membership
bytes, and the conservative CLEAN alias. It adds 2,930 files and repairs 265
existing external-feature files. All **18,058 archived regular files** and the
alias symlink were checked against the rebuilt manifest. All 817 CARE training
structures and 34 test structures have ESM, external, and RING caches with zero
audit failures. See the [repair and GPU smoke summary](notebook_outputs/summaries/summary_colab_care_cache_smoke_20260914.md)
for the additional ESM alias, the older PROPKA failures, and actual Colab
verification. Other dataset caches retain their prior feature-generation
state, except for the 265 repaired files shared with CARE.

Use the exact-cache ESM lookup fix in `src/training/esm_feature_loading.py`
with this bundle: exact and historical EC-annotation filenames coexist, and
the older loader could read both as duplicate residues. Three EC1 notebook
smoke families passed with the corrected loader, full feature coverage, and
matched validation membership. No held-out evaluation was performed.

The archive and cache tensors are ignored by Git; v12 is published separately
on Hugging Face. The local notebook now selects its immutable publication URL
and checksum. Optional ESM generation and auto-install are disabled for these
precomputed caches, avoiding the pinned ESM package installation on Python 3.13.
GitHub changes still require the user's commit/push. No GitHub push was performed.
For other datasets needing new embeddings, follow the optional generation route
in [COLAB_GPU_RUNBOOK.md](COLAB_GPU_RUNBOOK.md#esm-generation-on-a-python-313-colab-runtime).

Rebuild with `src/rebuild_care_complete_bundle.py`, providing the immutable v11
base manifest, the completed local CARE audit, and a new output archive path.
The builder refuses to overwrite an existing archive and performs no upload.

### Main Colab bundle v10 (historical)

Filename:
`DeepMzyme_Data_v10_exact_common70_clean30main_clean30x5_care30_esm_ring_external.tar.zst`

- Download URL:
  `https://huggingface.co/datasets/GMBioinformatics/DeepMzyme/resolve/4c36d3bfbf5e0c892fb165cbf101184e4151dda2/DeepMzyme_Data_v10_exact_common70_clean30main_clean30x5_care30_esm_ring_external.tar.zst`
- SHA256:
  `09525aad00d6c35e32a3601ff3ecf338978c465cec1ccfc18e47b9222b220aba`
- Verified upload commit:
  `88bedfd81f927aa8ad8b0a115ee52e6325cd163a`
- Portable checksum-sidecar normalization commit:
  `ed6ae8acbeedbe4f686891a1a10c1fa215028163`
- Local archive at previously documented
  `/media/Data/deepmzyme_colab_bundles/`: absent at audit time.

Included scientific roots:

- exact PinMyMetal;
- Common-PDBID 70/30 PinMyMetal;
- conservative `CLEAN_30_main`;
- conservative and original CLEAN shared roots/folds;
- CARE Task 1 clusterRes30;
- shared ESM, updated external features, RING features, and RING runtime.

Not included: non-overlapped or harsh PinMyMetal.

This hosted v10 archive predates the content-addressed local migration and
contains the legacy structure-directory layout. The manifest-aware loader keeps
it usable. Do not rebuild different contents under the v10 filename or
checksum. A newly versioned bundle built from the current local tree must also
contain every referenced `DeepMzyme_Data/structure_store/objects/...` file;
`src/build_colab_bundle.py` adds those dependencies automatically for supplied
train/test roots. A direct `tar` assembly that includes manifest-backed shared
CLEAN roots must include `DeepMzyme_Data/structure_store/` explicitly.

Assembly command preserved from the previous README:

```bash
tar --zstd -cf /media/Data/deepmzyme_colab_bundles/DeepMzyme_Data_v10_exact_common70_clean30main_clean30x5_care30_esm_ring_external.tar.zst \
  DeepMzyme_Data/train_and_test_sets_structures_exact_pinmymetal \
  DeepMzyme_Data/train_and_test_sets_structures_common_pdbid_70_30_pinmymetal \
  DeepMzyme_Data/CLEAN_30_main \
  DeepMzyme_Data/CLEAN_30_shared_single_donor_supported_metal_conservative \
  DeepMzyme_Data/CLEAN_30_shared \
  DeepMzyme_Data/CARE_task1_30_clusterRes30_train_test_metallo \
  DeepMzyme_Data/DeepMzyme_Colab_Bundles/train_and_test_sets_structures_exact_pinmymetal \
  DeepMzyme_Data/DeepMzyme_Colab_Bundles/train_and_test_sets_structures_common_pdbid_70_30_pinmymetal \
  DeepMzyme_Data/DeepMzyme_Colab_Bundles/CARE_task1_30_clusterRes30_train_test_metallo \
  DeepMzyme_Data/esm_embeddings \
  DeepMzyme_Data/updated_feature_extraction \
  DeepMzyme_Data/RING_features \
  DeepMzyme_Data/ring-4.0
```

### CLEAN predictor bundle v2

Filename:
`CLEAN_predictor_baselines_v2_clean30x5_single_donor_supported_metal_conservative_care30_sources.tar.zst`

- Download URL:
  `https://huggingface.co/datasets/GMBioinformatics/DeepMzyme/resolve/4c36d3bfbf5e0c892fb165cbf101184e4151dda2/CLEAN_predictor_baselines_v2_clean30x5_single_donor_supported_metal_conservative_care30_sources.tar.zst`
- SHA256:
  `5124b0b514b49affc158df121a87f5389ec1e027d14e0cf0a53cfb13a602c0f0`
- Verified upload commit:
  `88bedfd81f927aa8ad8b0a115ee52e6325cd163a`
- Sidecar normalization commit:
  `ed6ae8acbeedbe4f686891a1a10c1fa215028163`
- Local archive at previously documented
  `/media/Data/clean_predictor_bundles/`: absent at audit time.

This bundle contains CLEAN sequence/split CSVs, original and conservative CLEAN
metallo folds, CARE clusterRes30 metallo CSVs, and manifest metadata. It omits
DeepMzyme structures, ESMC embeddings, RING files, and graph external features.

## Provenance map

| Evidence | Tracked location |
|---|---|
| Original PinMyMetal membership | [`prepare_training_and_test_set/pinmymetal_files/`](../prepare_training_and_test_set/pinmymetal_files/) |
| Exact/common70 generated metadata | [`prepare_training_and_test_set/provenance/`](../prepare_training_and_test_set/provenance/) |
| Legacy non-overlap test access | [`docs/notebook_outputs/raw/legacy_nonoverlap_test_access/`](notebook_outputs/raw/legacy_nonoverlap_test_access/) |
| CLEAN generated metadata/audit | [`CLEAN_prepare_training_and_test_set/provenance/`](../CLEAN_prepare_training_and_test_set/provenance/) |
| CARE clusterRes30 metadata/audit | [`CARE_prepare_training_and_test_set/provenance/`](../CARE_prepare_training_and_test_set/provenance/) |
| Preparation procedures | PinMyMetal, CLEAN, and CARE preparation directories |
| Current scientific policy | [`Plan.md`](../Plan.md) |
| Current project state | [`EXPERIMENT_STATUS.md`](../EXPERIMENT_STATUS.md) |

## Unresolved records

- Primary final-test route requires a separate scientific decision.
- CARE upstream source URL/citation is missing.
- The non-overlapped PinMyMetal root is present locally but absent from the
  historical v10 bundle (included in v11/v12); the harsh root is unavailable
  locally and in all three bundles.
- Exact PinMyMetal retains 177 overlapping PDB IDs.
- Historical non-overlap test access exists, but its influence on subsequent
  selection cannot be established.
- CLEAN materialized fold views must record their original versus conservative
  source explicitly.
