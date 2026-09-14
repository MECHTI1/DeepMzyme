# Active Run Configuration

Source mode: notebook planning
Result stage: single
Task: ec
Run mode: single
Run batch ID: standalone_ec1_care30_only_esm_smoke_v1
Dataset name: CARE_task1_30_clusterRes30_train_test_metallo
Resolved dataset root: /content/DeepMzyme/DeepMzyme_Data/CARE_task1_30_clusterRes30_train_test_metallo
Train dir: /content/DeepMzyme/DeepMzyme_Data/CARE_task1_30_clusterRes30_train_test_metallo/train
Test dir: /content/care_smoke_stock/only_esm/unused_test
Train CSV: /content/DeepMzyme/DeepMzyme_Data/CARE_task1_30_clusterRes30_train_test_metallo/train/final_data_summarazing_table_transition_metals_only_catalytic.csv
Test CSV: /content/care_smoke_stock/only_esm/unused_test.csv
Split policy: split_by=pdbid, val_fraction=0.15, split_seed=42, n_folds=, fold_index=
Summary basename: ec_four_class_only_esm_care_task1_30_clusterres30_metallo_batch-standalone_ec1_care30_only_esm_smoke_v1_split-pdbid_fixedval (auto)
Selection metric: val_ec_group_level_1_balanced_acc
Model preset: Only-ESM
Architecture/fusion: only_esm / none
Run name: standalone_ec1_care30_only_esm_smoke_v1_ec_only_esm_archonly_esm_fusionnone_ringno_esmyes_mwinverse_frequency_fullfeatures_ep1_lr3e_5c1d7b31
Run directory: /content/care_smoke_stock/only_esm/runs/standalone_ec1_care30_only_esm_smoke_v1/standalone_ec1_care30_only_esm_smoke_v1_ec_only_esm_archonly_esm_fusionnone_ringno_esmyes_mwinverse_frequency_fullfeatures_ep1_lr3e_5c1d7b31

## Command
```bash
DEEPGM_METAL_LABEL_SCHEME=merge_fe_class_viii PYTHONPATH=/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task ec --metal-label-scheme merge_fe_class_viii --metal-eligibility-scheme active --structure-dir /content/DeepMzyme/DeepMzyme_Data/CARE_task1_30_clusterRes30_train_test_metallo/train --summary-csv /content/DeepMzyme/DeepMzyme_Data/CARE_task1_30_clusterRes30_train_test_metallo/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/care_smoke_stock/only_esm/runs/standalone_ec1_care30_only_esm_smoke_v1 --run-name standalone_ec1_care30_only_esm_smoke_v1_ec_only_esm_archonly_esm_fusionnone_ringno_esmyes_mwinverse_frequency_fullfeatures_ep1_lr3e_5c1d7b31 --dataset-bundle-id v11+care-complete-20260914:care-audit-manifest --dataset-bundle-sha256 1b02920c13d837e75984761d22110fdcc2aca338220be96421e83586120f07e2 --model-architecture only_esm --epochs 1 --batch-size 4 --learning-rate 3e-05 --weight-decay 0.0001 --seed 42 --split-seed 42 --split-stratify-by active_targets --val-fraction 0.15 --train-val-split-by pdbid --selection-metric val_ec_group_level_1_balanced_acc --device cuda --num-workers 0 --pin-memory --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 128 --head-mlp-layers 2 --head-mlp-dropout 0.2 --esm-graph-encoder-dropout 0.1 --classifier-pool-distance-cutoff 0.0 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-collapsed-loss-weight 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy error --lr-schedule fixed --deterministic --log-per-class-metrics --require-all-task-classes --external-features-root-dir /content/DeepMzyme/DeepMzyme_Data/updated_feature_extraction --ec-label-depth 1 --ec-group-weighting structure_id --ec-class-weight-unit group --ec-contrastive-weight 0.0 --ec-contrastive-temperature 0.1 --esm-embeddings-dir /content/DeepMzyme/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges
```

## Machine-Readable Payload
See `active_run_config.json` in the same directory.
