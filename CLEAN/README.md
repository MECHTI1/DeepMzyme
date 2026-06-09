# CLEAN Predictor Baselines For Metalloenzyme EC Tests

This folder is separate from the DeepMzyme training pipeline. It is for running the original CLEAN sequence-based EC predictor on the same metalloenzyme test sets used for DeepMzyme.

## Purpose

The comparison asks whether the CLEAN predictor is better than DeepMzyme for EC prediction on metalloenzymes.

For each benchmark, evaluate all methods on the same metalloenzyme-only test set:

| Dataset | Method/Baseline | Training data | Test data |
| --- | --- | --- | --- |
| CLEAN30 | DeepMzyme | CLEAN30 metalloenzyme train | CLEAN30 metalloenzyme test |
| CLEAN30 | CLEAN-matched | CLEAN30 metalloenzyme train | same CLEAN30 metalloenzyme test |
| CLEAN30 | CLEAN-full | full original CLEAN30 train split | same CLEAN30 metalloenzyme test |
| CARE30 | DeepMzyme | CARE30 metalloenzyme train | CARE30 metalloenzyme test |
| CARE30 | CLEAN-matched | CARE30 metalloenzyme train | same CARE30 metalloenzyme test |
| CARE30 | CLEAN-full | full original CARE Task 1 train set | same CARE30 metalloenzyme test |

This gives both a same-data comparison and a stronger sequence-only baseline where CLEAN can use the full original benchmark training set.

## Notebook

Run:

```text
CLEAN/train_clean_predictor_baselines.ipynb
```

The notebook does four things:

1. Normalizes source splits to the tab-delimited format expected by the official CLEAN code: `Entry`, `EC number`, `Sequence`.
2. Creates four CLEAN predictor train/test jobs:
   - `clean30_fold{fold}_metallo`: CLEAN trained on extracted CLEAN30 metalloenzymes.
   - `clean30_fold{fold}_full`: CLEAN trained on the full original CLEAN30 split.
   - `care30_metallo`: CLEAN trained on extracted CARE30 metalloenzymes.
   - `care30_full`: CLEAN trained on the full original CARE Task 1 train set.
3. Optionally clones and installs the official CLEAN implementation from `https://github.com/tttianhao/CLEAN`.
4. Optionally generates ESM-1b embeddings, trains CLEAN triplet models, runs max-separation inference, and scores EC1/EC2 predictions.

The notebook defaults to preparing tables only. Long steps require explicit flags.

## Source Data

CLEAN30 full split source:

```text
DeepMzyme_Data/CLEAN_all_train_valid_splits/split30/split30_train_split_{fold}.csv
DeepMzyme_Data/CLEAN_all_train_valid_splits/split30/split30_test_split_{fold}_curate.csv
```

CLEAN30 metalloenzyme source:

```text
DeepMzyme_Data/CLEAN_30_train_test_split_{fold}/train/final_data_summarazing_table_transition_metals_only_catalytic.csv
DeepMzyme_Data/CLEAN_30_train_test_split_{fold}/test/final_data_summarazing_table_transition_metals_only_catalytic.csv
```

CARE30 full split source:

```text
DeepMzyme_Data/CARE_dataset/CARE_datasets/splits/task1/protein_train.csv
DeepMzyme_Data/CARE_dataset/CARE_datasets/splits/task1/30_protein_test.csv
```

CARE30 metalloenzyme source, after the CARE AlphaFill + MAHOMES2 export finishes:

```text
/media/Data/care_sets/task1_30_clusterRes30/exported/CARE_task1_30_clusterRes30_train_test_metallo/train/final_data_summarazing_table_transition_metals_only_catalytic.csv
/media/Data/care_sets/task1_30_clusterRes30/exported/CARE_task1_30_clusterRes30_train_test_metallo/test/final_data_summarazing_table_transition_metals_only_catalytic.csv
```

If the CARE export is not finished, the notebook will prepare CLEAN30 jobs and skip/raise for CARE metalloenzyme jobs unless `ALLOW_PROVISIONAL_CARE_INPUTS = True` is set. The default is `False` to avoid accidentally using pre-MAHOMES candidate sites as final labels.

## Output Location

Generated files are written under:

```text
CLEAN/work/
```

This directory is ignored by git. It may become large because official CLEAN uses ESM-1b embeddings and model checkpoints.

## Reporting Rule

Report CLEAN-derived and CARE-derived results separately. Do not average them into one main number.

For the metalloenzyme EC task, the primary metrics should be EC1 and EC2:

```text
EC1 top-1 any-true accuracy
EC1 macro F1 / macro recall
EC2 top-1 any-true accuracy
EC2 macro F1 / macro recall
number of test proteins
number of EC classes
```

EC3/EC4 should be reported only when class support is sufficient.
