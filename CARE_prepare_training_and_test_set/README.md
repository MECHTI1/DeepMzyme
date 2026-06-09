# CARE Task 1 30% Preparation Pipeline

This directory prepares a CARE Task 1 `<30%` identity-derived metalloenzyme subset for DeepMzyme without touching the CLEAN or PinMyMetal preparation workflows.

CARE starts from protein/EC rows, UniProt IDs, and sequences. It does not provide DeepMzyme-ready metal-site labels. This pipeline therefore uses AlphaFill/AlphaFold as the structure-plus-transferred-metal source, then runs MAHOMES to keep candidate sites predicted as catalytic.

This is a CARE-derived AlphaFill-MAHOMES catalytic metalloenzyme subset. It is not the full CARE benchmark.

## Difference From CLEAN

The CLEAN pipeline starts from fold-specific CLEAN split files where rows are expected to be unique proteins. CARE Task 1 can contain protein-EC rows, so this pipeline writes two manifest levels:

- Pair-level manifests preserve original CARE rows.
- Unique-protein manifests collapse duplicate UniProt accessions for UniProt, AlphaFill, and MAHOMES work, merging EC labels as semicolon-separated unique values.

The CARE pipeline uses source-neutral metadata columns such as `source_dataset`, `source_task`, `source_split_name`, and `source_split`. It does not write CLEAN-specific fields such as `clean_identity`, `clean_fold`, or `clean_split`.

## Default Paths

Local CARE input root, treated as read-only:

```bash
DeepMzyme_Data/CARE_dataset
```

Default generated work root:

```bash
/media/Data/care_sets/task1_30
```

Default exported DeepMzyme dataset root:

```bash
DeepMzyme_Data/CARE_task1_30_train_test_metallo
```

Shared defaults live in `00_common_care_task1_30.sh`. Override values such as `WORK_ROOT`, `OUTPUT_ROOT`, `CARE_TRAIN_CSV`, `CARE_TEST_CSV`, or `MAHOMES_N_JOBS` by exporting them before running a step. Each step writes a matching log under `$WORK_ROOT/logs/`.

## CARE File Discovery

By default, `care_prepare.py audit-care-task1` recursively searches under `CARE_ROOT` for:

```text
protein_train.csv
30_protein_test.csv
```

It prefers paths containing `splits/task1`, including layouts such as:

```text
DeepMzyme_Data/CARE_dataset/splits/task1/protein_train.csv
DeepMzyme_Data/CARE_dataset/splits/task1/30_protein_test.csv
DeepMzyme_Data/CARE_dataset/CARE/splits/task1/protein_train.csv
DeepMzyme_Data/CARE_dataset/CARE/splits/task1/30_protein_test.csv
DeepMzyme_Data/CARE_dataset/CARE-main/splits/task1/protein_train.csv
DeepMzyme_Data/CARE_dataset/CARE-main/splits/task1/30_protein_test.csv
DeepMzyme_Data/CARE_dataset/CARE_datasets/splits/task1/protein_train.csv
DeepMzyme_Data/CARE_dataset/CARE_datasets/splits/task1/30_protein_test.csv
```

If discovery is ambiguous, set both paths explicitly:

```bash
export CARE_TRAIN_CSV=/absolute/path/to/protein_train.csv
export CARE_TEST_CSV=/absolute/path/to/30_protein_test.csv
bash CARE_prepare_training_and_test_set/01_audit_care_task1_30.sh
```

The audit requires a UniProt/accession-like column. It does not implement sequence-to-UniProt online mapping.

Supported accession/protein columns include `Entry`, `ID`, `UniProt`, `UniProt ID`, `uniprot_id`, `accession`, `protein_id`, and `protein`.

Supported EC columns include `EC number`, `EC`, `ecnumber`, `ec_number`, and `label`.

Supported sequence columns include `Sequence`, `Sequences`, `protein_sequence`, and `amino_acid_sequence`.

## Step Order

Run in filename order:

```bash
bash CARE_prepare_training_and_test_set/01_audit_care_task1_30.sh
bash CARE_prepare_training_and_test_set/02_fetch_alphafill_care_task1_30_smoke.sh
bash CARE_prepare_training_and_test_set/03_build_mahomes_inputs_care_task1_30_smoke.sh
bash CARE_prepare_training_and_test_set/04_run_mahomes_train_care_task1_30.sh
bash CARE_prepare_training_and_test_set/05_run_mahomes_test_care_task1_30.sh
bash CARE_prepare_training_and_test_set/06_summarize_mahomes_care_task1_30.sh
bash CARE_prepare_training_and_test_set/07_export_dataset_care_task1_30.sh
```

Use `08_monitor_mahomes_care_task1_30.sh` during long MAHOMES runs to inspect progress and disk usage without modifying outputs.

## Smoke Commands

Audit and write manifests:

```bash
bash CARE_prepare_training_and_test_set/01_audit_care_task1_30.sh
```

Fetch only 20 unique proteins per split, prefiltered by UniProt-supported transition-metal annotations:

```bash
bash CARE_prepare_training_and_test_set/02_fetch_alphafill_care_task1_30_smoke.sh
```

Build MAHOMES inputs from whichever smoke-limited AlphaFill files are present, using the same serious thresholds as the full pipeline:

```bash
bash CARE_prepare_training_and_test_set/03_build_mahomes_inputs_care_task1_30_smoke.sh
```

Do not run the MAHOMES or export steps until the smoke fetch/build output has been inspected.

## Full Commands

After the smoke path is accepted:

```bash
bash CARE_prepare_training_and_test_set/02_fetch_alphafill_care_task1_30_full.sh
bash CARE_prepare_training_and_test_set/03_build_mahomes_inputs_care_task1_30_full.sh
bash CARE_prepare_training_and_test_set/04_run_mahomes_train_care_task1_30.sh
bash CARE_prepare_training_and_test_set/05_run_mahomes_test_care_task1_30.sh
bash CARE_prepare_training_and_test_set/06_summarize_mahomes_care_task1_30.sh
bash CARE_prepare_training_and_test_set/07_export_dataset_care_task1_30.sh
```

The full fetch step is allowed to use UniProt and AlphaFill only when you run it manually. This repository change does not download CARE, clone CARE, or run external API calls during validation.

The full fetch step first batch-prefetches UniProt cofactor/binding-site annotations into:

```text
/media/Data/care_sets/task1_30/uniprot/uniprot_annotation_cache.csv
```

It then uses that cache to query AlphaFill only for proteins with UniProt-supported transition-metal annotations. AlphaFill fetching is disk-aware: JSON is downloaded first, and the larger CIF is downloaded only when the JSON contains a threshold-passing candidate for the UniProt-supported biological metal.

Default full-fetch network settings live in `00_common_care_task1_30.sh`:

```bash
CARE_UNIPROT_BATCH_SIZE=200
CARE_ALPHAFILL_N_JOBS=3
CARE_ALPHAFILL_TIMEOUT=90
CARE_ALPHAFILL_RETRIES=4
```

If AlphaFill refuses connections, wait and rerun:

```bash
bash CARE_prepare_training_and_test_set/02_fetch_alphafill_care_task1_30_full.sh
```

Already downloaded AlphaFill JSON/CIF files are reused unless `--overwrite` is added manually.

## Outputs

Audit/manifests:

```text
/media/Data/care_sets/task1_30/manifests/care_task1_30_train_pairs.csv
/media/Data/care_sets/task1_30/manifests/care_task1_30_test_pairs.csv
/media/Data/care_sets/task1_30/manifests/care_task1_30_train_proteins.csv
/media/Data/care_sets/task1_30/manifests/care_task1_30_test_proteins.csv
/media/Data/care_sets/task1_30/manifests/care_task1_30_audit.csv
/media/Data/care_sets/task1_30/manifests/care_task1_30_audit.json
```

AlphaFill/UniProt caches:

```text
/media/Data/care_sets/task1_30/alphafill/train/json/
/media/Data/care_sets/task1_30/alphafill/train/cif/
/media/Data/care_sets/task1_30/alphafill/test/json/
/media/Data/care_sets/task1_30/alphafill/test/cif/
/media/Data/care_sets/task1_30/uniprot/train/
/media/Data/care_sets/task1_30/uniprot/test/
```

MAHOMES inputs:

```text
/media/Data/care_sets/task1_30/mahomes_inputs/train/*.pdb
/media/Data/care_sets/task1_30/mahomes_inputs/train/candidate_site_summary.csv
/media/Data/care_sets/task1_30/mahomes_inputs/test/*.pdb
/media/Data/care_sets/task1_30/mahomes_inputs/test/candidate_site_summary.csv
```

MAHOMES summaries:

```text
/media/Data/care_sets/task1_30/mahomes_outputs/train/final_data_summarazing_table_transition_metals_only_catalytic.csv
/media/Data/care_sets/task1_30/mahomes_outputs/test/final_data_summarazing_table_transition_metals_only_catalytic.csv
```

Final exported dataset:

```text
DeepMzyme_Data/CARE_task1_30_train_test_metallo/train/
DeepMzyme_Data/CARE_task1_30_train_test_metallo/test/
DeepMzyme_Data/CARE_task1_30_train_test_metallo/metadata/
DeepMzyme_Data/CARE_task1_30_train_test_metallo/split_metadata.json
DeepMzyme_Data/CARE_task1_30_train_test_metallo/README.md
```

## Interpretation Rules

- This is a CARE-derived computational subset, not a PinMyMetal split and not the full CARE benchmark.
- AlphaFill transferred metals are computational hypotheses from homologous structures.
- MAHOMES catalytic filtering is computational catalytic-site evidence.
- Do not call the result experimentally validated metalloenzymes without independent evidence.
- Do not tune, compare, run HPO, run seed-repeat selection, or select checkpoints on the exported CARE test split.
- Use only the exported CARE train directory for train/validation splitting during model selection.

## Training Notebook Usage

The training notebook supports the exported CARE dataset through either alias:

```python
DATASET_NAME = "care_task1_30_metallo_alphafill_mahomes"
```

or:

```python
DATASET_NAME = "CARE_task1_30_train_test_metallo"
```

Both resolve to:

```text
DeepMzyme_Data/CARE_task1_30_train_test_metallo
```

If the dataset root is outside the normal bundle, repository, or Drive search paths, leave `DATASET_NAME` as the CARE alias for provenance and set:

```python
DATASET_ROOT_OVERRIDE = "/absolute/path/to/CARE_task1_30_train_test_metallo"
```

For EC Task 1 baseline work, keep the held-out CARE test split disabled during training:

```python
TASK = "ec"
DATASET_NAME = "care_task1_30_metallo_alphafill_mahomes"
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"
OPTUNA_SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False

EC_LABEL_DEPTHS_CSV = "1"
EC_CONTRASTIVE_WEIGHTS_CSV = "0.0"
EC_GROUP_WEIGHTING = "structure_id"

RING_EDGE_MODE = "with_ring"
REQUIRE_RING_EDGES = False
PREPARE_MISSING_RING_EDGES = True
```
