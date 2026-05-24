# DeepMzyme

DeepMzyme is a deep-learning framework for predicting metalloenzyme metal type and EC/function labels from protein structural pocket graphs, residue-level features, and optional ESMC embeddings.

## Main Tasks

1. Metal-type classification
2. EC/function classification
3. Joint metal + EC prediction

The live Colab notebook currently defaults to the five-class target scheme for
the active exploratory joint run: `Mn`, `Cu`, `Zn`, `Fe`, and grouped `Co/Ni`
via `METAL_LABEL_SCHEME = "five_class"` / `--metal-label-scheme five_class`.
Six-class metal runs (`Mn`, `Cu`, `Zn`, `Fe`, `Co`, `Ni`) remain supported and
must be labeled separately from five-class evidence.

## Current Notebook Defaults

The current notebook default variables are an exploratory validation-side joint
configuration. Treat them as the current launch surface, not as held-out test
evidence.

| Area | Current value |
| --- | --- |
| Task / target | `TASK = "joint"`, `METAL_LABEL_SCHEME = "five_class"` |
| Run mode | `RUN_MODE = "single"`, `RECOMMENDED_RUN_SET = "custom"` |
| Model preset | `MODEL_PRESET = "GVP + hybrid fusion"` |
| Dataset | `DATASET_NAME = "train_and_test_sets_structures_exact_pinmymetal"` |
| Graph defaults | `RING_EDGE_MODE = "with_ring"`, `METAL_NODE_MODE = "per_metal"`, `STRUCTURAL_READOUT_SCOPE = "auto"` |
| Capacity/search CSVs | `HIDDEN_S_VALUES_CSV = "128"`, `HIDDEN_V_VALUES_CSV = "8,16"`, `EDGE_HIDDEN_VALUES_CSV = "64"`, `GVP_LAYERS_VALUES_CSV = "2,3"` |
| Geometry/fusion CSVs | `EDGE_RADIUS_VALUES_CSV = "6, 8, 10"`, `ESM_FUSION_DIM_VALUES_CSV = "64,128,256"`, `EARLY_ESM_DIM_VALUES_CSV = "48"` |
| Training | `EPOCHS = 50`, `BATCH_SIZES_CSV = "12"`, `LEARNING_RATES_CSV = "3.705631497756492e-05"`, `WEIGHT_DECAYS_CSV = "1e-5,1e-4,1e-3"` |
| Validation | `VAL_FRACTION = 0.18`, `SPLIT_BY = "pdbid"`, `SELECTION_METRIC = "val_metal_balanced_acc"` |
| Schedule/loss | `LR_SCHEDULES_CSV = "cosine"`, `METAL_CLASS_WEIGHT_MODES_CSV = "effective_number"`, `METAL_LOSS_FUNCTION = "cross_entropy"` |
| Task weights | `METAL_LOSS_WEIGHT = 2.0`, `EC_LOSS_WEIGHT = 0.25` |
| Held-out test | `INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False` |

Older validation anchors and copied outputs may use `VAL_FRACTION = 0.15` and
six-class metal labels. Do not mix those results with the current `0.18`
five-class notebook-default runs without labeling the difference.

## Recommended Split

Current experiment evidence uses the legacy **Non-overlapped PinMyMetal** split (`DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal`) for final held-out evaluation. Select checkpoints and model variants with validation metrics only; use the held-out test set for final reporting of the selected checkpoint.

Other named split variants are available for secondary comparisons:

- **Harsh Split PinMyMetal**: `DeepMzyme_Data/train_and_test_sets_structures_harsh_pinmymetal`, where every common exact-split PDB ID is assigned as a whole group to test.
- **Metal Split PinMyMetal**: `DeepMzyme_Data/train_and_test_sets_structures_exact_pinmymetal`, matching the exact PinMyMetal train/test PDB-ID membership for available supported structures and possibly containing train/test overlap; the summary CSVs do not retain the original PinMyMetal site-row identifiers.
- **Common-PDBID 70/30 Split PinMyMetal**: `DeepMzyme_Data/train_and_test_sets_structures_common_pdbid_70_30_pinmymetal`, where train-only PDB IDs stay in train, test-only PDB IDs stay in test, and only common exact-split PDB IDs are assigned as whole PDB-ID groups with 70% to train and 30% to test.

To regenerate the exact and Common-PDBID 70/30 folders from local preparation
outputs, rebuild the exact train/test folders first, rerun the Step 5 train/test
notebooks if biological-metal support CSVs are needed, refresh the exact split
metadata, and then recreate the derived common-PDBID split:

```bash
PYTHONPATH=src /home/mechti/miniconda3/envs/DeepMzyme/bin/python prepare_training_and_test_set/step4_moveto_repo_data_train_structures_and_csv.py
PYTHONPATH=src /home/mechti/miniconda3/envs/DeepMzyme/bin/python prepare_training_and_test_set/step4b_moveto_repo_data_test_structures_and_csv.py
# Run prepare_training_and_test_set/step5_add_verified_only_biologicalmetal_to_datacsvtrain.ipynb
# Run prepare_training_and_test_set/step5b_add_verified_only_biologicalmetal_to_datacsvtest.ipynb
PYTHONPATH=src /home/mechti/miniconda3/envs/DeepMzyme/bin/python prepare_training_and_test_set/step5c_filter_exact_pinmymetal_tables_to_supported_transition_metals.py
/home/mechti/miniconda3/envs/DeepMzyme/bin/python prepare_training_and_test_set/step4c_write_exact_pinmymetal_metadata.py
/home/mechti/miniconda3/envs/DeepMzyme/bin/python prepare_training_and_test_set/step6b_create_pinmymetal_split_variants.py --mode common-pdbid-70-30 --overwrite
```



## Quick Start

Use the project interpreter from the repository root:

```bash
/home/mechti/miniconda3/envs/DeepMzyme/bin/python -c "import sys; print(sys.executable)"
```

Show the training CLI:

```bash
/home/mechti/miniconda3/envs/DeepMzyme/bin/python src/train.py --help
```

Build the current Colab bundle used by the notebook. `DeepMzyme_Data_v2.tar.zst`
contains the exact PinMyMetal split and the Common-PDBID 70/30 split, plus the
shared ESM, external-feature, and RING assets when present:

```bash
PYTHONPATH=src /home/mechti/miniconda3/envs/DeepMzyme/bin/python src/build_colab_bundle.py \
  --dataset-root DeepMzyme_Data/train_and_test_sets_structures_exact_pinmymetal \
  --dataset-root DeepMzyme_Data/train_and_test_sets_structures_common_pdbid_70_30_pinmymetal \
  --include-esm-embeddings \
  --output-bundle DeepMzyme_Data/DeepMzyme_Colab_Bundles/DeepMzyme_Data_v2.tar.zst
```

The bundle includes the site-level MAHOMES summary CSVs used by training. It also includes structure-level CSV artifacts for inspection; structures with multiple catalytic metal sites are represented there with semicolon-joined metal labels such as `Co;Cu`.

Current bundles should also include `esm_embeddings/`,
`updated_feature_extraction/`, `RING_features/`, and `ring-4.0/` when those
directories exist locally, because ESM/fusion presets require precomputed ESMC
embeddings and the metal notebook defaults to strict updated external features
and RING-enabled graph construction.

The Colab notebook supports three dataset input modes through `COLAB_DATA_SOURCE`:

- `huggingface_link`: downloads `https://huggingface.co/datasets/GMBioinformatics/DeepMzyme/resolve/main/DeepMzyme_Data_v2.tar.zst`, verifies SHA256, and unpacks it under `/content`.
- `upload_file`: prompts for a local `.tar.zst` upload in the Colab runtime. The current exact/common-PDBID 70/30 bundle is `DeepMzyme_Data_v2.tar.zst`.
- `drive`: uses the configured Google Drive data path after Drive is mounted.

Current `DeepMzyme_Data_v2.tar.zst` SHA256:
`12181d6bd7cb8e853cc0ea1d69dc50482dffe60392ad97089ccb3a5466059ba3`.

Example trusted-split Only-GVP metal validation run:

```bash
PYTHONPATH=src /home/mechti/miniconda3/envs/DeepMzyme/bin/python src/train.py \
  --task metal \
  --metal-label-scheme six_class \
  --model-architecture only_gvp \
  --structure-dir DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train \
  --summary-csv DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv \
  --test-structure-dir DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/test \
  --test-summary-csv DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/test/final_data_summarazing_table_transition_metals_only_catalytic.csv \
  --runs-dir DeepMzyme_Data/runs_baseline_first \
  --run-name metal_only_gvp_seed42 \
  --seed 42 \
  --val-fraction 0.18 \
  --train-val-split-by pdbid \
  --selection-metric val_metal_balanced_acc \
  --epochs 50 \
  --batch-size 8 \
  --external-feature-source updated \
  --use-ring-edges \
  --ring-features-dir DeepMzyme_Data/RING_features \
  --prepare-missing-ring-edges
```

Do not add `--run-test-eval` to exploratory baselines, HPO, or grouped-fold
confirmation. Held-out test evaluation is reserved for the final
validation-selected configuration under the one-shot Stage 7 policy.

Detailed validation-only command examples are in `list_train_commands.md`. The
documentation index, run-order map, and folder ownership rules are in
`docs/README.md`. The
interactive workflow is in `notebooks/DeepMzyme_training_colab.ipynb`; exact
metal stage blocks live in `docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`, and EC
stage blocks live in `docs/EC_TRAINING_PIPELINE_PLAYBOOK.md`.

For copied notebook-output evidence, start with the cross-family snapshot in
`docs/notebook_outputs/summaries/LEADERBOARD.md`, then the concise run
summaries in `docs/notebook_outputs/summaries/`, and finally the raw captured
outputs in `docs/notebook_outputs/raw/` when exact logs or run commands are
needed. Current status, selected validation anchors, and next-step
recommendations live in `EXPERIMENT_STATUS.md`.

Generated ESM embeddings should normally live outside the Git repository, then
be passed with `--esm-embeddings-dir`. Use validation metrics for model and
hyperparameter choice; reserve the held-out test set for final reporting of the
selected checkpoint. Current mutable anchors and next-step recommendations live
in `EXPERIMENT_STATUS.md`; exact current training values belong in the task
playbooks, not in this README.

Optional reproducibility and joint-loss controls include `--deterministic`, `--metal-loss-weight`, and `--ec-loss-weight`.

## EC Group Weighting

Metal prediction remains a pocket/site-level task, so metal loss and metal metrics are computed per pocket. EC/function prediction is structure/protein/chain-level; by default `--ec-group-weighting structure_id` weights EC cross-entropy so multiple separated EC-supervised pockets from the same structure contribute one total EC unit per split group. True multinuclear pockets are not downweighted by raw metal atom count, because nearby metals are already represented as one pocket by the extraction logic.

Use `--ec-group-weighting none` to recover unweighted pocket-level EC loss. Validation and held-out test reports include both pocket-level EC metrics and EC group-level metrics based on mean logits per group.
