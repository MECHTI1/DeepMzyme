# DeepMzyme Dataset, Split, Bundle, and Test-Use Reference

This is the authoritative human-readable record of DeepMzyme datasets, split
relationships, preparation status, bundle inclusion, and known test access.
Scientific split and evaluation policy remains in [`Plan.md`](../Plan.md).
Preparation procedures remain in their pipeline directories.

Last evidence audit: 2026-08-20.

## Final-reporting status

> **Primary final-test route: unresolved scientific decision required before final reporting.**

Relevant facts:

- The legacy non-overlap PinMyMetal test was historically evaluated in seven
  early Only-GVP runs and is not pristine or unopened.
- Exact PinMyMetal contains train/test PDB-ID overlap.
- Non-overlapped and harsh PinMyMetal roots are absent from the current local
  data tree and current v10 bundle.
- CLEAN and CARE datasets have different scientific purposes and cannot be
  silently designated as replacement final tests.
- This documentation cleanup does not select a replacement dataset or change
  evaluation behavior.

The historical test values are preserved as access evidence, not parameter or
model-selection evidence.

## Status vocabulary

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

## Dataset overview

| Dataset ID | Scientific purpose | Materialized locally | In current v10 bundle | Test/fold evaluation record | Current interpretation |
|---|---|---:|---:|---|---|
| `pinmymetal-source` | Original PinMyMetal class-model membership and site provenance | Source files tracked | No, source membership only | Not an executable split by itself | Primary membership evidence |
| `pinmymetal-exact` | Supported-structure projection preserving original train/test side | Yes | Yes | No completed test evaluation found | Possibly overlapped comparison/validation route |
| `pinmymetal-nonoverlap` | Remove exact-test PDB IDs from train; retain the original exact test | No | No | Seven early test evaluations found | Historically accessed; not pristine |
| `pinmymetal-harsh` | Put all common exact-split PDB IDs on the test side | No | No | No evaluation found | Documented severe comparison variant |
| `pinmymetal-common70` | Custom zero-overlap assignment of common PDB IDs, seed 42 | Yes | Yes | No evaluation found | Custom comparison split, not a selected final test |
| `clean30-original` | CLEAN official split30 fold benchmark with shared multi-donor structures | Yes | Yes | Fold evaluation is the intended benchmark design | Five fold pairs; report aggregate across folds |
| `clean30-conservative` | One deterministic supported-metal AlphaFill donor per CLEAN target/fold | Yes; current `CLEAN_30_main` | Yes | No completed DeepMzyme/CLEAN-predictor result found in inspected evidence | Current preferred CLEAN metallo source |
| `clean10` | Potential CLEAN 10%-identity benchmark | No | No | No evidence found | Not present or documented |
| `care-task1-legacy30` | Older CARE Task 1 30%-identity preparation route | Scripts/docs only | No distinct legacy root in v10 | No evaluation found | Historical/secondary preparation track |
| `care-task1-clusterres30` | Representative CARE Task 1 metallo subset for EC/joint work | Yes | Yes | Test prepared and bundled; no completed evaluation found | Current prepared CARE route |

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

Tracked generated metadata:
[`prepare_training_and_test_set/provenance/exact/`](../prepare_training_and_test_set/provenance/exact/).

### Non-overlapped PinMyMetal

Historical path:
`DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal`

Construction intent: remove every exact-test PDB ID from train while retaining
the original exact test side. The intended PDB-ID overlap is zero.

Status:

- Membership construction code: tracked.
- Materialized locally: no.
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
- Structure storage: one shared hardlinked copy.
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
`CLEAN_30_train_test_split_4` currently contain marker records pointing to the
original `CLEAN_30_shared` source. A fold directory name alone therefore does
not identify whether original or conservative metadata was used.

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
| Non-overlapped PinMyMetal | Historical dataset absent now | Yes — seven early reports | Not established | Historically accessed; metrics excluded from current selection |
| Harsh PinMyMetal | No current root | No | No | Availability must be restored before use |
| Common-PDBID 70/30 | Yes | No | No | Custom comparison only |
| CLEAN30 fold pairs | Yes | No completed result found | No | Evaluate as five-fold benchmark, not sealed one-shot test |
| CARE clusterRes30 | Yes | No | No | Prepared/bundled test; do not equate preparation with evaluation |

## Current bundles

### Hugging Face repository inventory

Repository:
[`GMBioinformatics/DeepMzyme`](https://huggingface.co/datasets/GMBioinformatics/DeepMzyme)

The repository tree was enumerated through the Hugging Face dataset API on
2026-08-22. The dataset-card `README.md` is only a 31-byte license header, so it
is not currently a useful file or provenance index. This section is the
human-readable inventory.

| Hugging Face path | Size (bytes) | SHA256 / verification | Purpose |
|---|---:|---|---|
| `.gitattributes` | 2,504 | Not applicable | Git LFS rules |
| `README.md` | 31 | Not recorded | License-only dataset card header |
| `CLEAN_predictor_baselines_v2_clean30x5_single_donor_supported_metal_conservative_care30_sources.tar.zst` | 29,237,125 | `5124b0b514b49affc158df121a87f5389ec1e027d14e0cf0a53cfb13a602c0f0` | CLEAN predictor/baseline bundle described below |
| `CLEAN_predictor_baselines_v2_clean30x5_single_donor_supported_metal_conservative_care30_sources.tar.zst.sha256` | 170 | Contains the archive checksum | Portable checksum sidecar |
| `DeepMzyme_Data_v10_exact_common70_clean30main_clean30x5_care30_esm_ring_external.tar.zst` | 3,822,130,168 | `09525aad00d6c35e32a3601ff3ecf338978c465cec1ccfc18e47b9222b220aba` | Main Colab training/data bundle described below |
| `DeepMzyme_Data_v10_exact_common70_clean30main_clean30x5_care30_esm_ring_external.tar.zst.sha256` | 155 | Contains the archive checksum | Portable checksum sidecar |
| `benchmarks/gvp_esm_hybrid_realistic_subset_v1/realistic_subset.json` | 1,087 | Not separately recorded | Portable benchmark manifest |
| `benchmarks/gvp_esm_hybrid_realistic_subset_v1/realistic_subset.pt` | 51,844,189 | `84e7e039f1df5b3a7b32dc3d4ac1b8fa21bba2827679b4d3f1650d394e2754bf` | Plain PyG-data benchmark subset for realistic GVP+ESM compute probes |

Use the artifacts as follows:

- normal Colab DeepMzyme training: the main v10 archive plus its SHA256;
- CLEAN predictor baselines without graph assets: the CLEAN predictor archive
  plus its SHA256;
- G4/A100 throughput reproduction only: both files under
  `benchmarks/gvp_esm_hybrid_realistic_subset_v1/`;
- scientific dataset membership, bundle contents, and test-use interpretation:
  the sections below, not the minimal Hugging Face dataset card.

Direct benchmark downloads:
[manifest JSON](https://huggingface.co/datasets/GMBioinformatics/DeepMzyme/resolve/main/benchmarks/gvp_esm_hybrid_realistic_subset_v1/realistic_subset.json)
and
[portable PyG subset](https://huggingface.co/datasets/GMBioinformatics/DeepMzyme/resolve/main/benchmarks/gvp_esm_hybrid_realistic_subset_v1/realistic_subset.pt).

The benchmark subset is compute evidence only. It was derived from CARE
clusterRes30 training pockets and does not authorize held-out evaluation or
stand in for a model-quality dataset. Audited G4/A100 results are summarized in
[`EXPERIMENT_STATUS.md`](../EXPERIMENT_STATUS.md).

### Main Colab bundle v10

Filename:
`DeepMzyme_Data_v10_exact_common70_clean30main_clean30x5_care30_esm_ring_external.tar.zst`

- Download URL:
  `https://huggingface.co/datasets/GMBioinformatics/DeepMzyme/resolve/main/DeepMzyme_Data_v10_exact_common70_clean30main_clean30x5_care30_esm_ring_external.tar.zst`
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
  `https://huggingface.co/datasets/GMBioinformatics/DeepMzyme/resolve/main/CLEAN_predictor_baselines_v2_clean30x5_single_donor_supported_metal_conservative_care30_sources.tar.zst`
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
- Non-overlapped and harsh PinMyMetal roots are unavailable in the current
  local data/bundle.
- Exact PinMyMetal retains 177 overlapping PDB IDs.
- Historical non-overlap test access exists, but its influence on subsequent
  selection cannot be established.
- CLEAN materialized fold views must record their original versus conservative
  source explicitly.
