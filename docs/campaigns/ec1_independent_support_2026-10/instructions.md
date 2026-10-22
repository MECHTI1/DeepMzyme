# EC1 independent-support preparation: instructions

Campaign [README](README.md). CPU evidence only. The reporter has no training, inference, download, similarity-search,
feature-generation or split-writing mode, and it never reads held-out records.

## Tool

[`scripts/report_ec1_component_support.py`](../../../scripts/report_ec1_component_support.py), tested by
[`tests/test_ec1_component_support.py`](../../../tests/test_ec1_component_support.py). It reuses, read-only:

- [`src/audit_sequence_remoteness.py`](../../../src/audit_sequence_remoteness.py): saved-membership checks, the
  protocol, search-receipt and alignment validation, and `Components`. The component rule is the frozen
  `annotate_task` rule: same protein, identical sequence, or an alignment with E-value <= 0.001, >= 50 residue pairs,
  shorter-sequence coverage >= 0.5 and identity > 0.2 (a test proves equal component identities).
- [`CARE_prepare_training_and_test_set/care_prepare.py`](../../../CARE_prepare_training_and_test_set/care_prepare.py):
  EC parsing, `extract_candidate_dicts`, `canonicalize_metal`, the annotation-cache reader, the prediction-label
  parser and the build parser's policy defaults. Its download and build commands are never invoked.

## Inputs (read only)

- `--baseline-split`: `docs/notebook_outputs/raw/ec1_standalone_v12_20260914/expected_split.json` (saved EC1
  development membership; 993 train and 175 validation pockets).
- `--remoteness-root`: `DeepMzyme_Data/notebook_outputs/remote_homology_v1/` (workstation NVMe): `protocol.json`,
  `preparation_freeze.json`, `sequence_manifest.json`, `memberships.json`, `ec/development.fasta`,
  `ec/alignments.tsv`, `ec/search_receipt.json`, `full_protein_readiness/preparation_report.json`.
- `--care-metadata-root`: `CARE_prepare_training_and_test_set/provenance/clusterRes30/metadata/`:
  `care_task1_30_train_proteins.csv`, `build_parameters.json` (the recorded build policy), and under `train/`
  `build_summary.csv`, `candidate_site_summary.csv`, `prediction_results_summary.csv` and
  `data_summarazing_table_transition_metals_whether_catalytic.csv`. `fetch_summary.csv` is filtered as it is read
  to `split == "train"` and the 24-accession allowlist; every other row is discarded unused.
- `--care-work-root`: `<secondary-drive>/care_sets/task1_30_clusterRes30`: for allowlisted accessions only,
  `alphafill/train/json/<ACC>.json`, `alphafill/train/cif/<ACC>.cif` and `uniprot/train/<ACC>.json` are checked, and
  `uniprot/uniprot_annotation_cache.csv` is read for allowlisted training rows only. No directory is scanned.

The allowlist is derived, not supplied: unambiguous EC7 training proteins (every EC number starts with 7) outside the
baseline whose recorded build status is `MISSING_ALPHAFILL_JSON_OR_CIF`.

## Command

```bash
P="${DEEPMZYME_PYTHON:-.venv/bin/python}"
systemd-run --user --scope --quiet -p CPUQuota=200% -p MemoryMax=3G -p MemorySwapMax=0 \
  env OMP_NUM_THREADS=1 taskset -c 0,1 nice -n 10 $P scripts/report_ec1_component_support.py \
  --baseline-split docs/notebook_outputs/raw/ec1_standalone_v12_20260914/expected_split.json \
  --remoteness-root DeepMzyme_Data/notebook_outputs/remote_homology_v1 \
  --care-metadata-root CARE_prepare_training_and_test_set/provenance/clusterRes30/metadata \
  --care-work-root "${SECONDARY_DRIVE:?}/care_sets/task1_30_clusterRes30" \
  --output-dir "${SECONDARY_DRIVE:?}/DeepMzyme_Data/campaigns/ec1_independent_support_2026-10/report_<UTC>"
```

The output directory must be new or empty and outside every input root; a file with different content is never
overwritten. Inputs are hashed before and after the run, and a change refuses the run.

## Outputs

`support_report.json` (results, reproduction check against the handoff values, input and source hashes),
`proteins.csv` (one row per development protein with its component), `ec7_links.csv` (every pair involving an EC7
protein), `ec7_source_inventory.csv` (the 179 EC7 training proteins and their attrition reason),
`ec7_asset_rejections.csv` (the 24 missing-asset accessions), `artifact_manifest.json` (command, inputs, sources,
output hashes), `report.md` and `next_scientific_decision.md`.

## Refusals and checks

- A saved EC run membership that differs from the baseline, a changed frozen remoteness file, an alignment that
  disagrees with the frozen sequences, or contradictory EC labels within one protein.
- A non-training row in a training-side CARE table, or a saved prediction that disagrees with the final site table.
- The candidate table's `whether_catalytic=0` is the build's placeholder; catalytic counts come from the saved
  (predictive) final table.
- Rejection categories for the allowlist: no qualifying donor/transplant; compound IDs outside the exact-element
  mapping (FES, SF4, hemes; a descriptive lookup, never a filter); elemental candidates rejected by annotation;
  recorded missing JSON; unexpected change requiring review.

Tests: `pytest tests/test_ec1_component_support.py` (synthetic fixtures; no real data).

## Training-source inventory (ec1s-002)

[`scripts/inventory_ec_support_sources.py`](../../../scripts/inventory_ec_support_sources.py), tested by
[`tests/test_inventory_ec_support_sources.py`](../../../tests/test_inventory_ec_support_sources.py). Read-only; no
download, alignment search, build, MAHOMES, training or split-writing mode. It reuses the reporter above
(`care_attrition`, `asset_rejections`), `care_prepare.py` (candidate extraction, the annotation rule) and the training
label rule (`src/training/labels.py`).

Sources and what is read (training side only):
- **CARE full training metadata**, `--care-full-train` `DeepMzyme_Data/CARE_dataset/CARE_datasets/splits/task1/protein_train.csv`.
  An accession's EC numbers are the union of its rows. It is a representative when its `clusterRes30` value is itself.
  Non-representatives are the source-A units and are never independent of their representative.
- **The unclustered CARE work root**, `--care-full-work-root` `<secondary-drive>/care_sets/task1_30`:
  - `uniprot/uniprot_annotation_cache.csv`: training rows of allowlisted accessions only, filtered while streaming
    (its first data row is a test row; never print it);
  - listings of `alphafill/train/json` and `alphafill/train/cif`, and the JSON files of supported-metal candidates;
  - `mahomes_inputs/build_parameters.json`, which must equal the recorded clusterRes30 policy;
  - the training build summary `mahomes_inputs/train/build_summary.csv` (a smoke build);
  - the `rcsb/entry` listing (donor metadata cache).
- **CARE clusterRes30 cohort:** `--care-metadata-root`, `--care-clusterres30-work-root` (as for the reporter) and
  `--care-prepared-train-dir` `DeepMzyme_Data/CARE_task1_30_clusterRes30_train_test_metallo/train`.
- **Non-overlap PinMyMetal training side**, `--pmm-train-dir`
  `DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train`: the catalytic site table, the
  structure manifest, `cofactor_analysis_by_structure_chain.csv` (SIFTS accessions) and the `_with_native` table. The
  folder must sit under the non-overlap root; manifest paths are checked for existence and size only.
- **Reused evidence:** `--baseline-split` and `--remoteness-root` (development membership and sequences), and
  `--previous-report` (the ec1s-001 report: its `proteins.csv` must match its manifest and the membership).

```bash
P="${DEEPMZYME_PYTHON:-.venv/bin/python}"
systemd-run --user --scope --quiet -p CPUQuota=200% -p MemoryMax=3G -p MemorySwapMax=0 \
  env OMP_NUM_THREADS=1 taskset -c 0,1 nice -n 10 $P scripts/inventory_ec_support_sources.py \
  --baseline-split docs/notebook_outputs/raw/ec1_standalone_v12_20260914/expected_split.json \
  --remoteness-root DeepMzyme_Data/notebook_outputs/remote_homology_v1 \
  --previous-report "${SECONDARY_DRIVE:?}/DeepMzyme_Data/campaigns/ec1_independent_support_2026-10/report_20261013T191929Z" \
  --care-full-train DeepMzyme_Data/CARE_dataset/CARE_datasets/splits/task1/protein_train.csv \
  --care-full-work-root "${SECONDARY_DRIVE:?}/care_sets/task1_30" \
  --care-metadata-root CARE_prepare_training_and_test_set/provenance/clusterRes30/metadata \
  --care-clusterres30-work-root "${SECONDARY_DRIVE:?}/care_sets/task1_30_clusterRes30" \
  --care-prepared-train-dir DeepMzyme_Data/CARE_task1_30_clusterRes30_train_test_metallo/train \
  --pmm-train-dir DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train \
  --output-dir "${SECONDARY_DRIVE:?}/DeepMzyme_Data/campaigns/ec1_independent_support_2026-10/inventory_<UTC>"
```

Outputs (about 20 s; fresh output folder outside every input root; never overwritten):
- `candidate_inventory.csv`: one row per unit and focal class, with relation, evidence level, eligibility, missing
  evidence, homology and held-out status;
- `attrition.csv`: funnels by source and class, with units, clusters and those without a known relation;
- `download_candidates.csv` and `build_donor_entries.csv`: the exact later-step lists;
- `inventory.json`: counts, the requested category counts (`user_categories`), cohort and PinMyMetal summaries, and the expectation check;
- `alphafill_json_inputs.csv` and `reused_asset_inputs.csv`: per-file hashes;
- `report.md`, `recommendation.md` and `artifact_manifest.json` (command, working directory, git HEAD, inputs,
  listing digests and sources).

Rules and boundaries:
- **Eligibility** is the unchanged v12 definition, and missing evidence is "unknown". A missing or `NOT_FOUND`
  training annotation stays unknown; this is a conservative difference from the scripted build, which would record no
  supported metal. It has no current impact.
- **Relations** are positive evidence only: development member, exact-sequence alias, the same clusterRes30 as a
  development protein, or an exact duplicate. "Without a known relation" is not independence.
- **Absent representatives:** a cluster whose representative is absent from the training entries is a separate,
  unresolved category.
- **Held-out boundary:**
  - Input paths, including resolved symlinks, are refused when they name a held-out directory or file.
  - The two annotation caches and the clusterRes30 fetch summary are mixed-split. They are hashed whole, and their
    rows are parsed and discarded by the split filter; no non-training value is used.
