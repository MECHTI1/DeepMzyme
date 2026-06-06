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

## Interpretation Rules

- This is a CLEAN-derived computational subset, not a PinMyMetal split.
- AlphaFill transferred metals are computational hypotheses from homologous structures.
- MAHOMES catalytic filtering is computational catalytic-site evidence.
- Do not call the result experimentally validated metalloenzymes without separate experimental evidence.
- Keep model selection validation-only; do not tune on the exported CLEAN test split.
- For publication-grade CLEAN benchmarking, process all five `split30` folds and report mean/std across folds.
