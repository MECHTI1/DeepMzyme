# CLEAN Preparation Pipeline

This directory prepares a CLEAN-derived metalloenzyme subset for DeepMzyme without contaminating the existing PinMyMetal preparation workflow.

CLEAN starts from UniProt IDs, EC labels, and sequences. It does not provide PDB structures or metal-site labels. This pipeline therefore uses AlphaFill/AlphaFold as the structure-plus-metal source, then runs MAHOMES to keep candidate sites predicted as catalytic.

Do not place CLEAN jobs or outputs under `prepare_training_and_test_set/` or any `*_pinmymetal` data directory.

## Recommended Pilot

Use CLEAN `split30` fold `0` first. The five `split30` pairs are folds, not additive partitions. Do not choose the best fold after looking at model results.

Default pilot work root:

```bash
/media/Data/clean_sets/split30/fold0
```

Default exported DeepMzyme split root:

```bash
DeepMzyme_Data/CLEAN_30_train_test_split_0
```

## Step 0: Audit And Normalize CLEAN Fold

Repeatable full-run scripts for `split30` fold `0` are provided in filename
order:

```bash
bash CLEAN_prepare_training_and_test_set/01_audit_split30_fold0.sh
bash CLEAN_prepare_training_and_test_set/02_fetch_alphafill_split30_fold0_full.sh
bash CLEAN_prepare_training_and_test_set/03_build_mahomes_inputs_split30_fold0_full.sh
bash CLEAN_prepare_training_and_test_set/04_run_mahomes_train_split30_fold0.sh
bash CLEAN_prepare_training_and_test_set/05_run_mahomes_test_split30_fold0.sh
bash CLEAN_prepare_training_and_test_set/06_summarize_mahomes_split30_fold0.sh
bash CLEAN_prepare_training_and_test_set/07_export_dataset_split30_fold0.sh
bash CLEAN_prepare_training_and_test_set/09_generate_runtime_features_split30_fold0.sh
bash CLEAN_prepare_training_and_test_set/10_validate_colab_inputs_split30_fold0.sh
bash CLEAN_prepare_training_and_test_set/11_build_colab_bundle_split30_fold0.sh
bash CLEAN_prepare_training_and_test_set/12_repartition_exported_split30_all_folds.sh
bash CLEAN_prepare_training_and_test_set/13_validate_colab_inputs_split30_all_folds.sh
bash CLEAN_prepare_training_and_test_set/14_build_colab_bundle_split30_all_folds.sh
bash CLEAN_prepare_training_and_test_set/15_build_shared_clean_split30_layout.sh
bash CLEAN_prepare_training_and_test_set/16_build_shared_clean_split30_bundle.sh
```

Shared defaults live in `00_common_split30_fold0.sh`. Override values such as
`WORK_ROOT` or `MAHOMES_N_JOBS` by exporting them before running a step.
Each step writes a matching log under `$WORK_ROOT/logs/`.
Use `08_monitor_mahomes_split30_fold0.sh` during long MAHOMES runs to inspect
job progress and disk usage without modifying outputs.

```bash
/home/mechti/miniconda3/envs/DeepMzyme/bin/python \
  CLEAN_prepare_training_and_test_set/clean_prepare.py audit-split \
  --identity 30 \
  --fold 0 \
  --work-root /media/Data/clean_sets/split30/fold0
```

Outputs:

```text
/media/Data/clean_sets/split30/fold0/manifests/clean_split30_fold0_train.csv
/media/Data/clean_sets/split30/fold0/manifests/clean_split30_fold0_test.csv
/media/Data/clean_sets/split30/fold0/manifests/clean_split30_audit.csv
```

## Step 1: Fetch AlphaFill Entries

Small pilot smoke:

```bash
/home/mechti/miniconda3/envs/DeepMzyme/bin/python \
  CLEAN_prepare_training_and_test_set/clean_prepare.py fetch-alphafill \
  --identity 30 \
  --fold 0 \
  --work-root /media/Data/clean_sets/split30/fold0 \
  --splits train test \
  --limit-per-split 20
```

Full fold after the smoke succeeds:

```bash
/home/mechti/miniconda3/envs/DeepMzyme/bin/python \
  CLEAN_prepare_training_and_test_set/clean_prepare.py fetch-alphafill \
  --identity 30 \
  --fold 0 \
  --work-root /media/Data/clean_sets/split30/fold0 \
  --splits train test \
  --prefilter-uniprot-supported-metals
```

Outputs are kept under:

```text
/media/Data/clean_sets/split30/fold0/alphafill/train/json/
/media/Data/clean_sets/split30/fold0/alphafill/train/cif/
/media/Data/clean_sets/split30/fold0/alphafill/test/json/
/media/Data/clean_sets/split30/fold0/alphafill/test/cif/
```

For the full serious run, UniProt is fetched first and AlphaFill is fetched
only when UniProt annotates one of the supported transition metals
`MN`, `FE`, `CO`, `NI`, `CU`, or `ZN`.

## Step 2: Build MAHOMES Inputs

Default serious threshold uses AlphaFill donor alignment identity `>= 0.30` and alignment length `>= 85`.
By default, candidate metals must also match a supported transition-metal annotation in the UniProt record.

```bash
/home/mechti/miniconda3/envs/DeepMzyme/bin/python \
  CLEAN_prepare_training_and_test_set/clean_prepare.py build-mahomes-inputs \
  --identity 30 \
  --fold 0 \
  --work-root /media/Data/clean_sets/split30/fold0 \
  --splits train test \
  --min-alphafill-identity 0.30 \
  --min-alignment-length 85 \
  --site-dedup-distance 1.0 \
  --uniprot-metal-policy require_supported
```

For a coverage audit only, compare with `--min-alphafill-identity 0.25`. Keep those results labeled separately.

Outputs:

```text
/media/Data/clean_sets/split30/fold0/mahomes_inputs/train/*.pdb
/media/Data/clean_sets/split30/fold0/mahomes_inputs/train/candidate_site_summary.csv
/media/Data/clean_sets/split30/fold0/mahomes_inputs/test/*.pdb
/media/Data/clean_sets/split30/fold0/mahomes_inputs/test/candidate_site_summary.csv
```

The generated PDBs contain AlphaFill protein chain `A` plus accepted supported transition-metal ions placed on chain `A` with residue numbers starting at `9001`. Supported metal labels are `MN`, `FE`, `CO`, `NI`, `CU`, and `ZN`.

Near-duplicate AlphaFill alternatives closer than `1.0` A are collapsed to one metal site. The selector first keeps candidates whose metal symbol is supported by UniProt cofactor/binding-site annotation, then breaks ties by donor PDB resolution from RCSB, then by AlphaFill alignment and local geometry quality. If no supported UniProt transition-metal annotation exists, `require_supported` excludes that accession from the serious dataset. Use `--uniprot-metal-policy prefer_supported` only for a labeled coverage/debug audit.

## Step 3: Run MAHOMES

Train split:

```bash
WORK_ROOT=/media/Data/clean_sets/split30/fold0 \
SPLIT=train \
N_JOBS=4 \
bash CLEAN_prepare_training_and_test_set/run_mahomes_clean.sh
```

Test split:

```bash
WORK_ROOT=/media/Data/clean_sets/split30/fold0 \
SPLIT=test \
N_JOBS=4 \
bash CLEAN_prepare_training_and_test_set/run_mahomes_clean.sh
```

Outputs:

```text
/media/Data/clean_sets/split30/fold0/mahomes/train/job_*/predictions.csv
/media/Data/clean_sets/split30/fold0/mahomes/test/job_*/predictions.csv
```

## Step 4: Summarize MAHOMES Predictions

```bash
/home/mechti/miniconda3/envs/DeepMzyme/bin/python \
  CLEAN_prepare_training_and_test_set/clean_prepare.py summarize-mahomes \
  --work-root /media/Data/clean_sets/split30/fold0 \
  --splits train test
```

Outputs:

```text
/media/Data/clean_sets/split30/fold0/mahomes_outputs/train/final_data_summarazing_table_transition_metals_only_catalytic.csv
/media/Data/clean_sets/split30/fold0/mahomes_outputs/test/final_data_summarazing_table_transition_metals_only_catalytic.csv
```

## Step 5: Export DeepMzyme Dataset

```bash
/home/mechti/miniconda3/envs/DeepMzyme/bin/python \
  CLEAN_prepare_training_and_test_set/clean_prepare.py export-dataset \
  --work-root /media/Data/clean_sets/split30/fold0 \
  --identity 30 \
  --fold 0 \
  --overwrite
```

Final layout:

```text
DeepMzyme_Data/CLEAN_30_train_test_split_0/train/
DeepMzyme_Data/CLEAN_30_train_test_split_0/test/
DeepMzyme_Data/CLEAN_30_train_test_split_0/metadata/
DeepMzyme_Data/CLEAN_30_train_test_split_0/split_metadata.json
```

## Step 6: Generate Runtime Features For Regular DeepMzyme Training

The regular Colab notebook ESM/fusion pipeline expects complete ESMC residue
embeddings. The current notebook graph defaults also expect updated external
residue features and RING edge files. Generate all three feature families with:

```bash
bash CLEAN_prepare_training_and_test_set/09_generate_runtime_features_split30_fold0.sh
```

Outputs are written under the standard shared DeepMzyme feature roots:

```text
DeepMzyme_Data/esm_embeddings/
DeepMzyme_Data/updated_feature_extraction/
DeepMzyme_Data/RING_features/
```

The script is resumable: existing valid ESM embeddings, external feature JSONs,
and RING edge files are skipped. Override `CLEAN_EXTERNAL_FEATURE_JOBS` or
`RING_EDGE_JOBS` before running if a different local parallelism level is
needed. ESMC generation is intentionally single-process because it loads the
ESMC model.

## Step 7: Validate And Build The Full Colab Bundle

Validate that the CLEAN split can be bundled with complete ESM coverage and
strict feature alignment:

```bash
bash CLEAN_prepare_training_and_test_set/10_validate_colab_inputs_split30_fold0.sh
```

Build the full Colab bundle:

```bash
bash CLEAN_prepare_training_and_test_set/11_build_colab_bundle_split30_fold0.sh
```

By default the final bundle is written outside the nearly-full project
filesystem:

```text
/media/Data/clean_sets/split30/fold0/bundles/DeepMzyme_Data_v6_clean30_split0_full_esm.tar.zst
```

Override `CLEAN_BUNDLE_OUTPUT` to place it elsewhere. The build step also writes
a sibling `.sha256` file.

## Step 8: Repartition The Completed CLEAN 30 Source Into All Five Folds

After fold `0` has completed AlphaFill, MAHOMES, runtime-feature generation,
and single-fold validation, the accepted structures can be repartitioned into
the other CLEAN `split30` folds without rerunning AlphaFill or MAHOMES:

```bash
bash CLEAN_prepare_training_and_test_set/12_repartition_exported_split30_all_folds.sh
```

This creates:

```text
DeepMzyme_Data/CLEAN_30_train_test_split_1/
DeepMzyme_Data/CLEAN_30_train_test_split_2/
DeepMzyme_Data/CLEAN_30_train_test_split_3/
DeepMzyme_Data/CLEAN_30_train_test_split_4/
```

Fold `0` is left as the already completed source export. The repartition step
keeps only accessions that are present in each target fold's official CLEAN
membership files; it does not force fold-0-only accessions into other folds.

Validate all five exported roots against the regular DeepMzyme ESM/RING/external
feature pipeline:

```bash
bash CLEAN_prepare_training_and_test_set/13_validate_colab_inputs_split30_all_folds.sh
```

Build one full all-fold Colab bundle:

```bash
bash CLEAN_prepare_training_and_test_set/14_build_colab_bundle_split30_all_folds.sh
```

Default all-fold bundle output:

```text
/media/Data/clean_sets/split30/all_folds/bundles/DeepMzyme_Data_v7_clean30_all5_full_esm.tar.zst
```

The notebook can then select the fold with `CLEAN_FOLD_INDEX = 0..4`, which
sets `DATASET_NAME` to `CLEAN_30_train_test_split_<fold>`.

## Step 9: Preferred Compact Shared-Structure Bundle

The duplicated all-fold bundle from Step 8 is compatibility-first. The preferred
cleaner layout stores each structure once and stores fold membership as
site-level CSVs only:

```text
DeepMzyme_Data/CLEAN_30_shared/structures/
DeepMzyme_Data/CLEAN_30_shared/folds/CLEAN_30_train_test_split_0_train.csv
DeepMzyme_Data/CLEAN_30_shared/folds/CLEAN_30_train_test_split_0_test.csv
...
DeepMzyme_Data/CLEAN_30_shared/folds/CLEAN_30_train_test_split_4_train.csv
DeepMzyme_Data/CLEAN_30_shared/folds/CLEAN_30_train_test_split_4_test.csv
```

Build and validate that shared layout:

```bash
bash CLEAN_prepare_training_and_test_set/15_build_shared_clean_split30_layout.sh
```

Build the compact Colab bundle:

```bash
bash CLEAN_prepare_training_and_test_set/16_build_shared_clean_split30_bundle.sh
```

Default compact bundle output:

```text
/media/Data/clean_sets/split30/shared/bundles/DeepMzyme_Data_v8_clean30_shared_full_esm.tar.zst
```

The notebook supports this layout directly: after unpacking, it materializes the
selected `CLEAN_FOLD_INDEX` into the normal `CLEAN_30_train_test_split_<fold>/`
`train/` and `test/` view expected by `src/train.py`.

## Interpretation Rules

- This is a CLEAN-derived computational subset, not a PinMyMetal split.
- AlphaFill transferred metals are computational hypotheses from homologous structures.
- MAHOMES catalytic filtering is computational catalytic-site evidence.
- Do not call the result experimentally validated metalloenzymes without separate experimental evidence.
- Keep model selection validation-only; do not tune on the exported CLEAN test split.
- For publication-grade CLEAN benchmarking, process all five `split30` folds and report mean/std across folds.
