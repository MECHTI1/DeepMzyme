TASK = "metal"
RUN_MODE = "manual_configurations"
ADVANCED_MODE = False

RECOMMENDED_RUN_SET = "custom"
MODEL_PRESET = "GVP + node-level late fusion"
MAX_CONFIGURATION_RUNS = 5
STOP_ON_FIRST_FAILURE = False

COLAB_DATA_SOURCE = "huggingface_link"
REPO_GIT_URL = "https://github.com/MECHTI1/DeepMzyme.git"
REPO_GIT_BRANCH = "main"
REPO_ROOT = ""
MOUNT_DRIVE = False

DATASET_NAME = "train_and_test_sets_structures_non_overlapped_pinmymetal"
TRAIN_DIR_OVERRIDE = ""
TRAIN_SITE_SUMMARY_CSV_OVERRIDE = ""
TEST_DIR_OVERRIDE = ""
TEST_SITE_SUMMARY_CSV_OVERRIDE = ""

ESM_EMBEDDINGS_DIR = ""
ALLOW_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_ESM_EMBEDDINGS = False

RING_FEATURES_DIR = ""
RING_EDGE_MODE = "without_ring"
REQUIRE_RING_EDGES = False
PREPARE_MISSING_RING_EDGES = False

RUNS_DIR = ""
RUN_BATCH_ID = "metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1"

HIDDEN_S_VALUES_CSV = "256"
HIDDEN_V_VALUES_CSV = "32"
EDGE_HIDDEN_VALUES_CSV = "128"
GVP_LAYERS_VALUES_CSV = "4"
HEAD_MLP_LAYERS_VALUES_CSV = "1"
EDGE_RADIUS_VALUES_CSV = "6.0"
ESM_FUSION_DIM_VALUES_CSV = "64"

EPOCHS = 50
BATCH_SIZES_CSV = "8"
LEARNING_RATES_CSV = "1.6801503587890522e-05"
WEIGHT_DECAYS_CSV = "1e-05"
SEEDS_CSV = "42,123,2026,43,44"
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_metal_balanced_acc"
LR_SCHEDULES_CSV = "fixed"

METAL_CLASS_WEIGHT_MODES_CSV = "inverse_frequency"
BALANCE_METAL_SITE_SYMBOLS = False
METAL_LOSS_FUNCTION = "cross_entropy"
METAL_FOCAL_GAMMA = 2.0
METAL_LABEL_SMOOTHING = 0.0

RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = False
RETRAIN_BEST_CONFIG_AFTER_HPO = False

INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
COPY_OUTPUTS_TO_DRIVE = False
SUMMARY_BASENAME = "deepmzyme_nonoverlap_model_comparison"
RUN_NAME_PREFIX = "deepmzyme_nonoverlap_baseline"


#--------------------------------------------------------------------------





Summary scanning scope: current RUN_BATCH_ID folder
RUN_BATCH_ID: metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1
Runs directory scanned: /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1
/usr/bin/python3 /content/DeepMzyme/src/report_runs.py --runs-dir /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1 --out-csv /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_model_comparison_completed_only.csv --out-figure /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_model_comparison.png
Completed-run summary CSV: /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_model_comparison_completed_only.csv
Summary source mode: planned table from current notebook state
Summary source scope: current planned rows plus completed runs under the scanned directory.
Comparison CSV: /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_model_comparison.csv
rank	source_mode	config_source	result_stage	run_name	status	error_message	model_preset	model_display	model_architecture	...	missing_train_metal_classes	missing_val_metal_classes	missing_train_ec_classes	missing_val_ec_classes	selected_best_validation_metric_value	held_out_test_metric_name	held_out_test_metric_value	run_dir	stdout_log_path	stderr_log_path
0	1	planned table	current notebook config	validation-only	deepmzyme_nonoverlap_baseline_batchmetal_node_...	completed		GVP + node-level late fusion	GVP + ESM node-level late fusion	gvp	...	NaN	NaN	NaN	NaN	0.633163	test_metal_balanced_acc	NaN	/content/deepmzyme_outputs/runs/metal_node_lev...	/content/deepmzyme_outputs/runs/metal_node_lev...	/content/deepmzyme_outputs/runs/metal_node_lev...
1	2	planned table	current notebook config	validation-only	deepmzyme_nonoverlap_baseline_batchmetal_node_...	completed		GVP + node-level late fusion	GVP + ESM node-level late fusion	gvp	...	NaN	NaN	NaN	6	0.619761	test_metal_balanced_acc	NaN	/content/deepmzyme_outputs/runs/metal_node_lev...	/content/deepmzyme_outputs/runs/metal_node_lev...	/content/deepmzyme_outputs/runs/metal_node_lev...
2	3	planned table	current notebook config	validation-only	deepmzyme_nonoverlap_baseline_batchmetal_node_...	completed		GVP + node-level late fusion	GVP + ESM node-level late fusion	gvp	...	NaN	NaN	NaN	4;6	0.614297	test_metal_balanced_acc	NaN	/content/deepmzyme_outputs/runs/metal_node_lev...	/content/deepmzyme_outputs/runs/metal_node_lev...	/content/deepmzyme_outputs/runs/metal_node_lev...
3	4	planned table	current notebook config	validation-only	deepmzyme_nonoverlap_baseline_batchmetal_node_...	completed		GVP + node-level late fusion	GVP + ESM node-level late fusion	gvp	...	NaN	NaN	NaN	6	0.590902	test_metal_balanced_acc	NaN	/content/deepmzyme_outputs/runs/metal_node_lev...	/content/deepmzyme_outputs/runs/metal_node_lev...	/content/deepmzyme_outputs/runs/metal_node_lev...
4	5	planned table	current notebook config	validation-only	deepmzyme_nonoverlap_baseline_batchmetal_node_...	completed		GVP + node-level late fusion	GVP + ESM node-level late fusion	gvp	...	NaN	NaN	NaN	NaN	0.574873	test_metal_balanced_acc	NaN	/content/deepmzyme_outputs/runs/metal_node_lev...	/content/deepmzyme_outputs/runs/metal_node_lev...	/content/deepmzyme_outputs/runs/metal_node_lev...
5 rows × 63 columns




Ranked table sorted by validation selection metric:
#1: deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_da157306 | stage=validation-only | class_weight=inverse_frequency | val_metal_balanced_acc=0.6331631856986827 | status=completed
#2: deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_a7759f1f | stage=validation-only | class_weight=inverse_frequency | val_metal_balanced_acc=0.6197609538487087 | status=completed
#3: deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_539ea463 | stage=validation-only | class_weight=inverse_frequency | val_metal_balanced_acc=0.614296708983616 | status=completed
#4: deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_fcc23cc2 | stage=validation-only | class_weight=inverse_frequency | val_metal_balanced_acc=0.5909019723421346 | status=completed
#5: deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_c8a9d9c9 | stage=validation-only | class_weight=inverse_frequency | val_metal_balanced_acc=0.5748731632352322 | status=completed

Best overall configuration: deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_da157306
{
  "run_name": "deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_da157306",
  "result_stage": "validation-only",
  "model_preset": "GVP + node-level late fusion",
  "model_architecture": "gvp",
  "fusion_mode": "node_level_late_fusion",
  "metal_class_weight_mode": "inverse_frequency",
  "balance_metal_site_symbols": false,
  "selection_metric": "val_metal_balanced_acc",
  "selected_best_validation_metric_value": 0.6331631856986827,
  "run_dir": "/content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_da157306"
}

Best configuration per model preset/mode:
GVP + node-level late fusion: deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_da157306 | class_weight=inverse_frequency | val_metal_balanced_acc=0.6331631856986827

Best Only-GVP configuration: not available
Best ESM-based configuration: deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_da157306
Best RING vs non-RING comparison: not available unless both modes have completed numeric validation metrics.

Automatic interpretation
Best validation config: deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_da157306 with val_metal_balanced_acc = 0.6331631856986827
Top fusion mode: node_level_late_fusion
Best learning-rate region: low around 1e-5 to 3e-5 (lr=1.6801503587890522e-05)
Held-out test results present: False
Recommended next step: select/retrain the final validation-best configuration, then run held-out test evaluation once
Drive copy skipped. Outputs remain under: /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1



































#---------------------------------------------------------

Runnable planned configurations: 5
================================================================================
[#001 | 1/5] deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_539ea463
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1 --run-name deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_539ea463 --model-architecture gvp --epochs 50 --batch-size 8 --learning-rate 1.6801503587890522e-05 --weight-decay 1e-05 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 256 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 4 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode node_level_late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges
stdout log: /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/_execution_logs/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_539ea463.stdout.log
stderr log: /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/_execution_logs/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_539ea463.stderr.log
================================================================================

=== Split diagnostics (passive; training behavior unchanged) ===
task=metal split_by=pdbid val_fraction=0.15 n_folds=None fold_index=None
pockets: train=1031 validation=182
groups by pdbid: train=1001 validation=93
train/validation overlap counts: pdbid=0, pdbid_chain=0, structure_id=0, pocket_id=0
train metal distribution: Mn=475, Cu=60, Zn=156, Fe=223, Co=67, Ni=50
validation metal distribution: Mn=85, Cu=13, Zn=31, Fe=33, Co=13, Ni=7
missing train metal classes: none
missing validation metal classes: none
train EC distribution: 1=450, 2=158, 3=272, 4=53, 5=70, 6=15
validation EC distribution: 1=54, 2=13, 3=53, 4=0, 5=60, 6=0
missing train EC classes: none
missing validation EC classes: 4, 6
===============================================================

epoch=1 train_loss=1.7733 lr=1.68015e-05 train_metal_acc=0.4918 val_loss=1.6817 val_metal_acc=0.5220 val_metal_min_recall=0.0000 val_fe_recall=0.3030 val_joint_bal_acc=0.2172 val_joint_macro_f1=0.1864 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.7029 lr=1.68015e-05 train_metal_acc=0.5228 val_loss=1.6642 val_metal_acc=0.3846 val_metal_min_recall=0.0000 val_fe_recall=0.6667 val_joint_bal_acc=0.3674 val_joint_macro_f1=0.3529 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.5958 lr=1.68015e-05 train_metal_acc=0.5325 val_loss=1.5054 val_metal_acc=0.3626 val_metal_min_recall=0.0000 val_fe_recall=0.3636 val_joint_bal_acc=0.3223 val_joint_macro_f1=0.3337 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.4915 lr=1.68015e-05 train_metal_acc=0.5781 val_loss=1.4251 val_metal_acc=0.4011 val_metal_min_recall=0.0000 val_fe_recall=0.4848 val_joint_bal_acc=0.4194 val_joint_macro_f1=0.4061 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.3244 lr=1.68015e-05 train_metal_acc=0.5839 val_loss=1.3550 val_metal_acc=0.4066 val_metal_min_recall=0.0000 val_fe_recall=0.8485 val_joint_bal_acc=0.4598 val_joint_macro_f1=0.4118 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.2229 lr=1.68015e-05 train_metal_acc=0.6605 val_loss=1.3422 val_metal_acc=0.4176 val_metal_min_recall=0.0000 val_fe_recall=0.5455 val_joint_bal_acc=0.4710 val_joint_macro_f1=0.4550 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=1.1600 lr=1.68015e-05 train_metal_acc=0.6479 val_loss=1.2921 val_metal_acc=0.3846 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.4264 val_joint_macro_f1=0.4375 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=1.0785 lr=1.68015e-05 train_metal_acc=0.6382 val_loss=1.2715 val_metal_acc=0.3956 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.4822 val_joint_macro_f1=0.4836 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=1.0172 lr=1.68015e-05 train_metal_acc=0.7245 val_loss=1.2354 val_metal_acc=0.4615 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4976 val_joint_macro_f1=0.4835 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.9456 lr=1.68015e-05 train_metal_acc=0.7071 val_loss=1.2630 val_metal_acc=0.4451 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5145 val_joint_macro_f1=0.4748 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.9350 lr=1.68015e-05 train_metal_acc=0.7003 val_loss=1.2461 val_metal_acc=0.4231 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.4783 val_joint_macro_f1=0.4787 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.8919 lr=1.68015e-05 train_metal_acc=0.7536 val_loss=1.2090 val_metal_acc=0.4176 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.4327 val_joint_macro_f1=0.4384 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.8612 lr=1.68015e-05 train_metal_acc=0.7759 val_loss=1.1620 val_metal_acc=0.5659 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5587 val_joint_macro_f1=0.5404 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.8128 lr=1.68015e-05 train_metal_acc=0.7963 val_loss=1.2232 val_metal_acc=0.5330 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5442 val_joint_macro_f1=0.5562 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.7657 lr=1.68015e-05 train_metal_acc=0.7983 val_loss=1.1383 val_metal_acc=0.5824 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5476 val_joint_macro_f1=0.5622 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.7309 lr=1.68015e-05 train_metal_acc=0.8080 val_loss=1.1543 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5574 val_joint_macro_f1=0.5913 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.7360 lr=1.68015e-05 train_metal_acc=0.8215 val_loss=1.1319 val_metal_acc=0.6813 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.6057 val_joint_macro_f1=0.6072 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.6896 lr=1.68015e-05 train_metal_acc=0.8138 val_loss=1.1227 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5456 val_joint_macro_f1=0.5785 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.6399 lr=1.68015e-05 train_metal_acc=0.8215 val_loss=1.2191 val_metal_acc=0.5824 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5342 val_joint_macro_f1=0.5526 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.6486 lr=1.68015e-05 train_metal_acc=0.8351 val_loss=1.1112 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5618 val_joint_macro_f1=0.5776 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.6084 lr=1.68015e-05 train_metal_acc=0.8409 val_loss=1.1096 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5622 val_joint_macro_f1=0.5723 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.5962 lr=1.68015e-05 train_metal_acc=0.8400 val_loss=1.2089 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.6143 val_joint_macro_f1=0.6123 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.5790 lr=1.68015e-05 train_metal_acc=0.8196 val_loss=1.1138 val_metal_acc=0.6099 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5344 val_joint_macro_f1=0.5508 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.5303 lr=1.68015e-05 train_metal_acc=0.8506 val_loss=1.1436 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5382 val_joint_macro_f1=0.5540 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.4949 lr=1.68015e-05 train_metal_acc=0.8186 val_loss=1.3213 val_metal_acc=0.5879 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5617 val_joint_macro_f1=0.5655 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.4952 lr=1.68015e-05 train_metal_acc=0.8477 val_loss=1.1730 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5711 val_joint_macro_f1=0.5852 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.4796 lr=1.68015e-05 train_metal_acc=0.8778 val_loss=1.1861 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5382 val_joint_macro_f1=0.5608 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.4754 lr=1.68015e-05 train_metal_acc=0.8438 val_loss=1.2464 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5755 val_joint_macro_f1=0.5458 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.4254 lr=1.68015e-05 train_metal_acc=0.8875 val_loss=1.1872 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5264 val_joint_macro_f1=0.5581 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.4709 lr=1.68015e-05 train_metal_acc=0.8952 val_loss=1.1973 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5501 val_joint_macro_f1=0.5863 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.4173 lr=1.68015e-05 train_metal_acc=0.8991 val_loss=1.2207 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5559 val_joint_macro_f1=0.5685 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.4060 lr=1.68015e-05 train_metal_acc=0.8943 val_loss=1.2737 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5549 val_joint_macro_f1=0.5793 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.3893 lr=1.68015e-05 train_metal_acc=0.8613 val_loss=1.2271 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5603 val_joint_macro_f1=0.5410 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.3763 lr=1.68015e-05 train_metal_acc=0.9059 val_loss=1.2415 val_metal_acc=0.6429 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5346 val_joint_macro_f1=0.5502 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.3797 lr=1.68015e-05 train_metal_acc=0.8991 val_loss=1.2574 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5721 val_joint_macro_f1=0.5637 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.3585 lr=1.68015e-05 train_metal_acc=0.8904 val_loss=1.2721 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.6970 val_joint_bal_acc=0.5597 val_joint_macro_f1=0.5879 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.3214 lr=1.68015e-05 train_metal_acc=0.9049 val_loss=1.3145 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5549 val_joint_macro_f1=0.5780 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.3256 lr=1.68015e-05 train_metal_acc=0.8943 val_loss=1.1941 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.6970 val_joint_bal_acc=0.5597 val_joint_macro_f1=0.5888 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.3193 lr=1.68015e-05 train_metal_acc=0.9040 val_loss=1.2637 val_metal_acc=0.6264 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5287 val_joint_macro_f1=0.5658 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.3248 lr=1.68015e-05 train_metal_acc=0.8923 val_loss=1.3191 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5866 val_joint_macro_f1=0.5786 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=41 train_loss=0.3201 lr=1.68015e-05 train_metal_acc=0.9079 val_loss=1.2417 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5768 val_joint_macro_f1=0.5763 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=42 train_loss=0.3053 lr=1.68015e-05 train_metal_acc=0.9166 val_loss=1.3693 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5454 val_joint_macro_f1=0.5951 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=43 train_loss=0.2922 lr=1.68015e-05 train_metal_acc=0.9117 val_loss=1.2289 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.7273 val_joint_bal_acc=0.5457 val_joint_macro_f1=0.5739 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=44 train_loss=0.2771 lr=1.68015e-05 train_metal_acc=0.9127 val_loss=1.4869 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5366 val_joint_macro_f1=0.5899 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=45 train_loss=0.2679 lr=1.68015e-05 train_metal_acc=0.9282 val_loss=1.3407 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5489 val_joint_macro_f1=0.5664 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=46 train_loss=0.2596 lr=1.68015e-05 train_metal_acc=0.9263 val_loss=1.4120 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5553 val_joint_macro_f1=0.5719 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=47 train_loss=0.2488 lr=1.68015e-05 train_metal_acc=0.9185 val_loss=1.3349 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5745 val_joint_macro_f1=0.5777 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=48 train_loss=0.2416 lr=1.68015e-05 train_metal_acc=0.9340 val_loss=1.3986 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5449 val_joint_macro_f1=0.5702 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=49 train_loss=0.2469 lr=1.68015e-05 train_metal_acc=0.9214 val_loss=1.3969 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5892 val_joint_macro_f1=0.6158 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=50 train_loss=0.2374 lr=1.68015e-05 train_metal_acc=0.9370 val_loss=1.4439 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5499 val_joint_macro_f1=0.5742 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_539ea463/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_539ea463/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_539ea463/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_539ea463/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_539ea463/run_metadata.json
Completed: /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_539ea463
================================================================================
[#002 | 2/5] deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_a7759f1f
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1 --run-name deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_a7759f1f --model-architecture gvp --epochs 50 --batch-size 8 --learning-rate 1.6801503587890522e-05 --weight-decay 1e-05 --seed 123 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 256 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 4 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode node_level_late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges
stdout log: /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/_execution_logs/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_a7759f1f.stdout.log
stderr log: /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/_execution_logs/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_a7759f1f.stderr.log
================================================================================

=== Split diagnostics (passive; training behavior unchanged) ===
task=metal split_by=pdbid val_fraction=0.15 n_folds=None fold_index=None
pockets: train=1031 validation=182
groups by pdbid: train=999 validation=95
train/validation overlap counts: pdbid=0, pdbid_chain=0, structure_id=0, pocket_id=0
train metal distribution: Mn=475, Cu=62, Zn=158, Fe=218, Co=67, Ni=51
validation metal distribution: Mn=85, Cu=11, Zn=29, Fe=38, Co=13, Ni=6
missing train metal classes: none
missing validation metal classes: none
train EC distribution: 1=450, 2=155, 3=280, 4=48, 5=74, 6=15
validation EC distribution: 1=54, 2=16, 3=45, 4=5, 5=56, 6=0
missing train EC classes: none
missing validation EC classes: 6
===============================================================

epoch=1 train_loss=1.7845 lr=1.68015e-05 train_metal_acc=0.1843 val_loss=1.7903 val_metal_acc=0.1703 val_metal_min_recall=0.0000 val_fe_recall=0.3684 val_joint_bal_acc=0.2980 val_joint_macro_f1=0.1619 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.7139 lr=1.68015e-05 train_metal_acc=0.5238 val_loss=1.6302 val_metal_acc=0.3901 val_metal_min_recall=0.0000 val_fe_recall=0.5526 val_joint_bal_acc=0.3330 val_joint_macro_f1=0.2765 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.6116 lr=1.68015e-05 train_metal_acc=0.4985 val_loss=1.5707 val_metal_acc=0.3297 val_metal_min_recall=0.0000 val_fe_recall=0.6842 val_joint_bal_acc=0.3940 val_joint_macro_f1=0.3307 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.4744 lr=1.68015e-05 train_metal_acc=0.5082 val_loss=1.4621 val_metal_acc=0.3187 val_metal_min_recall=0.0000 val_fe_recall=0.6579 val_joint_bal_acc=0.3388 val_joint_macro_f1=0.2317 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.3389 lr=1.68015e-05 train_metal_acc=0.6101 val_loss=1.3522 val_metal_acc=0.3626 val_metal_min_recall=0.0000 val_fe_recall=0.3421 val_joint_bal_acc=0.4328 val_joint_macro_f1=0.3448 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.2326 lr=1.68015e-05 train_metal_acc=0.6945 val_loss=1.2800 val_metal_acc=0.4066 val_metal_min_recall=0.0000 val_fe_recall=0.4737 val_joint_bal_acc=0.3879 val_joint_macro_f1=0.3701 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=1.1462 lr=1.68015e-05 train_metal_acc=0.7265 val_loss=1.2001 val_metal_acc=0.5440 val_metal_min_recall=0.0000 val_fe_recall=0.6053 val_joint_bal_acc=0.4193 val_joint_macro_f1=0.3932 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=1.0648 lr=1.68015e-05 train_metal_acc=0.7158 val_loss=1.2353 val_metal_acc=0.4341 val_metal_min_recall=0.0769 val_fe_recall=0.5789 val_joint_bal_acc=0.4375 val_joint_macro_f1=0.4035 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=1.0041 lr=1.68015e-05 train_metal_acc=0.7119 val_loss=1.2470 val_metal_acc=0.4176 val_metal_min_recall=0.0000 val_fe_recall=0.4737 val_joint_bal_acc=0.4909 val_joint_macro_f1=0.3865 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.9650 lr=1.68015e-05 train_metal_acc=0.7565 val_loss=1.2086 val_metal_acc=0.4835 val_metal_min_recall=0.0000 val_fe_recall=0.4737 val_joint_bal_acc=0.4479 val_joint_macro_f1=0.3826 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.9338 lr=1.68015e-05 train_metal_acc=0.7624 val_loss=1.1208 val_metal_acc=0.6099 val_metal_min_recall=0.0000 val_fe_recall=0.6053 val_joint_bal_acc=0.4836 val_joint_macro_f1=0.4753 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.8623 lr=1.68015e-05 train_metal_acc=0.7565 val_loss=1.1666 val_metal_acc=0.5165 val_metal_min_recall=0.0000 val_fe_recall=0.7632 val_joint_bal_acc=0.4509 val_joint_macro_f1=0.4005 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.8232 lr=1.68015e-05 train_metal_acc=0.7934 val_loss=1.0676 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.6579 val_joint_bal_acc=0.5898 val_joint_macro_f1=0.5505 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.8039 lr=1.68015e-05 train_metal_acc=0.7779 val_loss=1.0855 val_metal_acc=0.6209 val_metal_min_recall=0.1667 val_fe_recall=0.6579 val_joint_bal_acc=0.5251 val_joint_macro_f1=0.5199 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.7613 lr=1.68015e-05 train_metal_acc=0.8099 val_loss=1.0690 val_metal_acc=0.6319 val_metal_min_recall=0.0769 val_fe_recall=0.6842 val_joint_bal_acc=0.5121 val_joint_macro_f1=0.5070 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.7019 lr=1.68015e-05 train_metal_acc=0.8157 val_loss=1.0557 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6579 val_joint_bal_acc=0.5392 val_joint_macro_f1=0.5581 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.6932 lr=1.68015e-05 train_metal_acc=0.8177 val_loss=1.0415 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.5789 val_joint_bal_acc=0.5573 val_joint_macro_f1=0.5369 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.6529 lr=1.68015e-05 train_metal_acc=0.8215 val_loss=1.0548 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.6842 val_joint_bal_acc=0.5664 val_joint_macro_f1=0.5326 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.6386 lr=1.68015e-05 train_metal_acc=0.8050 val_loss=1.1017 val_metal_acc=0.6099 val_metal_min_recall=0.0769 val_fe_recall=0.6053 val_joint_bal_acc=0.5802 val_joint_macro_f1=0.5646 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.6088 lr=1.68015e-05 train_metal_acc=0.8390 val_loss=1.0448 val_metal_acc=0.6429 val_metal_min_recall=0.0769 val_fe_recall=0.6579 val_joint_bal_acc=0.5816 val_joint_macro_f1=0.5534 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.5850 lr=1.68015e-05 train_metal_acc=0.8594 val_loss=1.0248 val_metal_acc=0.6319 val_metal_min_recall=0.0769 val_fe_recall=0.6579 val_joint_bal_acc=0.5241 val_joint_macro_f1=0.5415 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.5358 lr=1.68015e-05 train_metal_acc=0.8438 val_loss=1.0428 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.7368 val_joint_bal_acc=0.5392 val_joint_macro_f1=0.5573 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.5134 lr=1.68015e-05 train_metal_acc=0.8322 val_loss=1.0602 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.6053 val_joint_bal_acc=0.5824 val_joint_macro_f1=0.5665 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.5201 lr=1.68015e-05 train_metal_acc=0.8739 val_loss=1.0054 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.6579 val_joint_bal_acc=0.5357 val_joint_macro_f1=0.5504 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.5014 lr=1.68015e-05 train_metal_acc=0.8720 val_loss=1.0354 val_metal_acc=0.6319 val_metal_min_recall=0.0769 val_fe_recall=0.5789 val_joint_bal_acc=0.5302 val_joint_macro_f1=0.5226 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.4877 lr=1.68015e-05 train_metal_acc=0.8855 val_loss=1.0485 val_metal_acc=0.6264 val_metal_min_recall=0.0769 val_fe_recall=0.6579 val_joint_bal_acc=0.5424 val_joint_macro_f1=0.5407 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.4762 lr=1.68015e-05 train_metal_acc=0.8797 val_loss=1.0567 val_metal_acc=0.6319 val_metal_min_recall=0.0769 val_fe_recall=0.7632 val_joint_bal_acc=0.4872 val_joint_macro_f1=0.5069 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.4761 lr=1.68015e-05 train_metal_acc=0.8555 val_loss=1.0405 val_metal_acc=0.6319 val_metal_min_recall=0.0769 val_fe_recall=0.5263 val_joint_bal_acc=0.5549 val_joint_macro_f1=0.5505 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.4264 lr=1.68015e-05 train_metal_acc=0.8904 val_loss=1.0367 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.7105 val_joint_bal_acc=0.5198 val_joint_macro_f1=0.5425 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.4283 lr=1.68015e-05 train_metal_acc=0.8623 val_loss=1.0853 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.6053 val_joint_bal_acc=0.5738 val_joint_macro_f1=0.5734 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.4109 lr=1.68015e-05 train_metal_acc=0.8933 val_loss=1.0824 val_metal_acc=0.6264 val_metal_min_recall=0.0769 val_fe_recall=0.5526 val_joint_bal_acc=0.5068 val_joint_macro_f1=0.5359 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.3920 lr=1.68015e-05 train_metal_acc=0.8982 val_loss=1.1224 val_metal_acc=0.6923 val_metal_min_recall=0.0769 val_fe_recall=0.5789 val_joint_bal_acc=0.5290 val_joint_macro_f1=0.5572 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.3758 lr=1.68015e-05 train_metal_acc=0.8943 val_loss=1.1736 val_metal_acc=0.6923 val_metal_min_recall=0.0769 val_fe_recall=0.5263 val_joint_bal_acc=0.5242 val_joint_macro_f1=0.5545 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.3598 lr=1.68015e-05 train_metal_acc=0.9146 val_loss=1.1437 val_metal_acc=0.6264 val_metal_min_recall=0.0769 val_fe_recall=0.6316 val_joint_bal_acc=0.5217 val_joint_macro_f1=0.5327 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.3514 lr=1.68015e-05 train_metal_acc=0.9137 val_loss=1.1918 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.5789 val_joint_bal_acc=0.5283 val_joint_macro_f1=0.5438 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.3485 lr=1.68015e-05 train_metal_acc=0.9098 val_loss=1.1481 val_metal_acc=0.6099 val_metal_min_recall=0.0769 val_fe_recall=0.5526 val_joint_bal_acc=0.4934 val_joint_macro_f1=0.5098 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.3549 lr=1.68015e-05 train_metal_acc=0.9088 val_loss=1.1937 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.7632 val_joint_bal_acc=0.5188 val_joint_macro_f1=0.5326 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.3191 lr=1.68015e-05 train_metal_acc=0.9108 val_loss=1.1627 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.6316 val_joint_bal_acc=0.5294 val_joint_macro_f1=0.5373 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.3150 lr=1.68015e-05 train_metal_acc=0.9117 val_loss=1.1915 val_metal_acc=0.6429 val_metal_min_recall=0.0769 val_fe_recall=0.6316 val_joint_bal_acc=0.5534 val_joint_macro_f1=0.5594 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.2873 lr=1.68015e-05 train_metal_acc=0.9195 val_loss=1.2038 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.6842 val_joint_bal_acc=0.5822 val_joint_macro_f1=0.5770 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=41 train_loss=0.2993 lr=1.68015e-05 train_metal_acc=0.9253 val_loss=1.1886 val_metal_acc=0.6154 val_metal_min_recall=0.0769 val_fe_recall=0.5789 val_joint_bal_acc=0.5336 val_joint_macro_f1=0.5268 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=42 train_loss=0.3147 lr=1.68015e-05 train_metal_acc=0.9263 val_loss=1.1811 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.6579 val_joint_bal_acc=0.5968 val_joint_macro_f1=0.5875 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=43 train_loss=0.2980 lr=1.68015e-05 train_metal_acc=0.9214 val_loss=1.2437 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.7368 val_joint_bal_acc=0.6198 val_joint_macro_f1=0.6012 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=44 train_loss=0.2760 lr=1.68015e-05 train_metal_acc=0.9253 val_loss=1.2235 val_metal_acc=0.6429 val_metal_min_recall=0.0769 val_fe_recall=0.6316 val_joint_bal_acc=0.5426 val_joint_macro_f1=0.5489 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=45 train_loss=0.2679 lr=1.68015e-05 train_metal_acc=0.9243 val_loss=1.2589 val_metal_acc=0.6319 val_metal_min_recall=0.0769 val_fe_recall=0.6579 val_joint_bal_acc=0.5373 val_joint_macro_f1=0.5490 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=46 train_loss=0.2669 lr=1.68015e-05 train_metal_acc=0.9224 val_loss=1.3594 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.7368 val_joint_bal_acc=0.5646 val_joint_macro_f1=0.5605 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=47 train_loss=0.2557 lr=1.68015e-05 train_metal_acc=0.9321 val_loss=1.2541 val_metal_acc=0.6264 val_metal_min_recall=0.0769 val_fe_recall=0.6842 val_joint_bal_acc=0.5743 val_joint_macro_f1=0.5667 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=48 train_loss=0.2750 lr=1.68015e-05 train_metal_acc=0.9263 val_loss=1.2745 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.6579 val_joint_bal_acc=0.5631 val_joint_macro_f1=0.5538 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=49 train_loss=0.2493 lr=1.68015e-05 train_metal_acc=0.9311 val_loss=1.3111 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.7105 val_joint_bal_acc=0.5473 val_joint_macro_f1=0.5656 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=50 train_loss=0.2264 lr=1.68015e-05 train_metal_acc=0.9088 val_loss=1.4789 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.4474 val_joint_bal_acc=0.5259 val_joint_macro_f1=0.5407 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_a7759f1f/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_a7759f1f/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_a7759f1f/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_a7759f1f/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_a7759f1f/run_metadata.json
Completed: /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_a7759f1f
================================================================================
[#003 | 3/5] deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_da157306
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1 --run-name deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_da157306 --model-architecture gvp --epochs 50 --batch-size 8 --learning-rate 1.6801503587890522e-05 --weight-decay 1e-05 --seed 2026 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 256 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 4 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode node_level_late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges
stdout log: /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/_execution_logs/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_da157306.stdout.log
stderr log: /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/_execution_logs/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_da157306.stderr.log
================================================================================

=== Split diagnostics (passive; training behavior unchanged) ===
task=metal split_by=pdbid val_fraction=0.15 n_folds=None fold_index=None
pockets: train=1031 validation=182
groups by pdbid: train=1000 validation=94
train/validation overlap counts: pdbid=0, pdbid_chain=0, structure_id=0, pocket_id=0
train metal distribution: Mn=475, Cu=60, Zn=158, Fe=221, Co=67, Ni=50
validation metal distribution: Mn=85, Cu=13, Zn=29, Fe=35, Co=13, Ni=7
missing train metal classes: none
missing validation metal classes: none
train EC distribution: 1=451, 2=159, 3=278, 4=48, 5=70, 6=12
validation EC distribution: 1=53, 2=12, 3=47, 4=5, 5=60, 6=3
missing train EC classes: none
missing validation EC classes: none
===============================================================

epoch=1 train_loss=1.7874 lr=1.68015e-05 train_metal_acc=0.4365 val_loss=1.7438 val_metal_acc=0.2527 val_metal_min_recall=0.0000 val_fe_recall=0.7429 val_joint_bal_acc=0.2065 val_joint_macro_f1=0.1674 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.7102 lr=1.68015e-05 train_metal_acc=0.5112 val_loss=1.6793 val_metal_acc=0.3132 val_metal_min_recall=0.0000 val_fe_recall=0.7143 val_joint_bal_acc=0.2848 val_joint_macro_f1=0.2657 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.6171 lr=1.68015e-05 train_metal_acc=0.5529 val_loss=1.5819 val_metal_acc=0.4066 val_metal_min_recall=0.0000 val_fe_recall=0.3714 val_joint_bal_acc=0.2330 val_joint_macro_f1=0.2511 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.4889 lr=1.68015e-05 train_metal_acc=0.4597 val_loss=1.5988 val_metal_acc=0.3022 val_metal_min_recall=0.0000 val_fe_recall=0.4000 val_joint_bal_acc=0.3395 val_joint_macro_f1=0.2933 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.3721 lr=1.68015e-05 train_metal_acc=0.6033 val_loss=1.4688 val_metal_acc=0.3516 val_metal_min_recall=0.1429 val_fe_recall=0.3429 val_joint_bal_acc=0.3989 val_joint_macro_f1=0.3653 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.2900 lr=1.68015e-05 train_metal_acc=0.6528 val_loss=1.4155 val_metal_acc=0.3626 val_metal_min_recall=0.0769 val_fe_recall=0.4000 val_joint_bal_acc=0.3710 val_joint_macro_f1=0.3822 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=1.1864 lr=1.68015e-05 train_metal_acc=0.6576 val_loss=1.3884 val_metal_acc=0.3791 val_metal_min_recall=0.0769 val_fe_recall=0.4000 val_joint_bal_acc=0.4067 val_joint_macro_f1=0.3776 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=1.1320 lr=1.68015e-05 train_metal_acc=0.6518 val_loss=1.3851 val_metal_acc=0.3297 val_metal_min_recall=0.0769 val_fe_recall=0.5143 val_joint_bal_acc=0.4212 val_joint_macro_f1=0.3549 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=1.0815 lr=1.68015e-05 train_metal_acc=0.7003 val_loss=1.3569 val_metal_acc=0.3956 val_metal_min_recall=0.0769 val_fe_recall=0.3714 val_joint_bal_acc=0.4278 val_joint_macro_f1=0.4068 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=1.0131 lr=1.68015e-05 train_metal_acc=0.7207 val_loss=1.3600 val_metal_acc=0.4066 val_metal_min_recall=0.0000 val_fe_recall=0.6286 val_joint_bal_acc=0.4423 val_joint_macro_f1=0.3973 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.9692 lr=1.68015e-05 train_metal_acc=0.7371 val_loss=1.2680 val_metal_acc=0.5385 val_metal_min_recall=0.0769 val_fe_recall=0.5429 val_joint_bal_acc=0.5821 val_joint_macro_f1=0.5083 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.9283 lr=1.68015e-05 train_metal_acc=0.7779 val_loss=1.2282 val_metal_acc=0.4615 val_metal_min_recall=0.0769 val_fe_recall=0.4286 val_joint_bal_acc=0.4855 val_joint_macro_f1=0.4534 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.8896 lr=1.68015e-05 train_metal_acc=0.7711 val_loss=1.1716 val_metal_acc=0.6099 val_metal_min_recall=0.0769 val_fe_recall=0.4286 val_joint_bal_acc=0.5195 val_joint_macro_f1=0.4981 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.8467 lr=1.68015e-05 train_metal_acc=0.7895 val_loss=1.1505 val_metal_acc=0.6429 val_metal_min_recall=0.0769 val_fe_recall=0.6286 val_joint_bal_acc=0.5261 val_joint_macro_f1=0.5330 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.8083 lr=1.68015e-05 train_metal_acc=0.7924 val_loss=1.1801 val_metal_acc=0.6429 val_metal_min_recall=0.0769 val_fe_recall=0.6571 val_joint_bal_acc=0.5109 val_joint_macro_f1=0.5233 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.7606 lr=1.68015e-05 train_metal_acc=0.7866 val_loss=1.2580 val_metal_acc=0.5275 val_metal_min_recall=0.0769 val_fe_recall=0.4857 val_joint_bal_acc=0.5878 val_joint_macro_f1=0.5164 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.7417 lr=1.68015e-05 train_metal_acc=0.7682 val_loss=1.3239 val_metal_acc=0.4560 val_metal_min_recall=0.1429 val_fe_recall=0.6286 val_joint_bal_acc=0.4636 val_joint_macro_f1=0.4555 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.6932 lr=1.68015e-05 train_metal_acc=0.7730 val_loss=1.3109 val_metal_acc=0.4560 val_metal_min_recall=0.0769 val_fe_recall=0.6571 val_joint_bal_acc=0.4774 val_joint_macro_f1=0.4730 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.7037 lr=1.68015e-05 train_metal_acc=0.8409 val_loss=1.1675 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.5714 val_joint_bal_acc=0.5502 val_joint_macro_f1=0.5612 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.6383 lr=1.68015e-05 train_metal_acc=0.8128 val_loss=1.1530 val_metal_acc=0.6099 val_metal_min_recall=0.0769 val_fe_recall=0.6000 val_joint_bal_acc=0.5658 val_joint_macro_f1=0.5483 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.6206 lr=1.68015e-05 train_metal_acc=0.8487 val_loss=1.1044 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.6000 val_joint_bal_acc=0.5139 val_joint_macro_f1=0.5400 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.6011 lr=1.68015e-05 train_metal_acc=0.8642 val_loss=1.1130 val_metal_acc=0.6868 val_metal_min_recall=0.0769 val_fe_recall=0.5714 val_joint_bal_acc=0.5551 val_joint_macro_f1=0.5646 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.5677 lr=1.68015e-05 train_metal_acc=0.8215 val_loss=1.2048 val_metal_acc=0.6319 val_metal_min_recall=0.0769 val_fe_recall=0.6571 val_joint_bal_acc=0.5620 val_joint_macro_f1=0.5708 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.5598 lr=1.68015e-05 train_metal_acc=0.8293 val_loss=1.2234 val_metal_acc=0.6319 val_metal_min_recall=0.0769 val_fe_recall=0.6286 val_joint_bal_acc=0.5155 val_joint_macro_f1=0.5176 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.5281 lr=1.68015e-05 train_metal_acc=0.8661 val_loss=1.1305 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.5429 val_joint_bal_acc=0.5331 val_joint_macro_f1=0.5572 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.4997 lr=1.68015e-05 train_metal_acc=0.8729 val_loss=1.0934 val_metal_acc=0.6813 val_metal_min_recall=0.0769 val_fe_recall=0.5714 val_joint_bal_acc=0.5675 val_joint_macro_f1=0.5798 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.4880 lr=1.68015e-05 train_metal_acc=0.8477 val_loss=1.0911 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.6571 val_joint_bal_acc=0.5585 val_joint_macro_f1=0.5721 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.4672 lr=1.68015e-05 train_metal_acc=0.8497 val_loss=1.0812 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.6857 val_joint_bal_acc=0.5833 val_joint_macro_f1=0.5971 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.4531 lr=1.68015e-05 train_metal_acc=0.8797 val_loss=1.0849 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.5429 val_joint_bal_acc=0.5902 val_joint_macro_f1=0.5901 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.4354 lr=1.68015e-05 train_metal_acc=0.8943 val_loss=1.1074 val_metal_acc=0.6813 val_metal_min_recall=0.0769 val_fe_recall=0.5714 val_joint_bal_acc=0.5712 val_joint_macro_f1=0.5830 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.4107 lr=1.68015e-05 train_metal_acc=0.8991 val_loss=1.1817 val_metal_acc=0.6868 val_metal_min_recall=0.0769 val_fe_recall=0.5714 val_joint_bal_acc=0.5837 val_joint_macro_f1=0.6010 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.4001 lr=1.68015e-05 train_metal_acc=0.9001 val_loss=1.1557 val_metal_acc=0.6813 val_metal_min_recall=0.0769 val_fe_recall=0.5714 val_joint_bal_acc=0.5637 val_joint_macro_f1=0.5861 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.3858 lr=1.68015e-05 train_metal_acc=0.9011 val_loss=1.1676 val_metal_acc=0.6923 val_metal_min_recall=0.0769 val_fe_recall=0.6000 val_joint_bal_acc=0.5628 val_joint_macro_f1=0.5956 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.3649 lr=1.68015e-05 train_metal_acc=0.8952 val_loss=1.1701 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.6286 val_joint_bal_acc=0.5787 val_joint_macro_f1=0.5994 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.3529 lr=1.68015e-05 train_metal_acc=0.9049 val_loss=1.2193 val_metal_acc=0.6868 val_metal_min_recall=0.0769 val_fe_recall=0.5714 val_joint_bal_acc=0.6169 val_joint_macro_f1=0.6189 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.3417 lr=1.68015e-05 train_metal_acc=0.9011 val_loss=1.1733 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.6000 val_joint_bal_acc=0.6332 val_joint_macro_f1=0.6347 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.3281 lr=1.68015e-05 train_metal_acc=0.9156 val_loss=1.1623 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.5714 val_joint_bal_acc=0.5796 val_joint_macro_f1=0.5921 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.3094 lr=1.68015e-05 train_metal_acc=0.8885 val_loss=1.3457 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.6000 val_joint_bal_acc=0.5797 val_joint_macro_f1=0.5944 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.3110 lr=1.68015e-05 train_metal_acc=0.9049 val_loss=1.2366 val_metal_acc=0.6923 val_metal_min_recall=0.0769 val_fe_recall=0.6000 val_joint_bal_acc=0.5818 val_joint_macro_f1=0.5894 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.3127 lr=1.68015e-05 train_metal_acc=0.9001 val_loss=1.2964 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.6857 val_joint_bal_acc=0.5169 val_joint_macro_f1=0.5450 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=41 train_loss=0.2949 lr=1.68015e-05 train_metal_acc=0.9253 val_loss=1.2740 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.5714 val_joint_bal_acc=0.5617 val_joint_macro_f1=0.5785 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=42 train_loss=0.2719 lr=1.68015e-05 train_metal_acc=0.9243 val_loss=1.2687 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.5714 val_joint_bal_acc=0.5816 val_joint_macro_f1=0.6023 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=43 train_loss=0.2885 lr=1.68015e-05 train_metal_acc=0.8788 val_loss=1.3722 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.5429 val_joint_bal_acc=0.5899 val_joint_macro_f1=0.5923 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=44 train_loss=0.2748 lr=1.68015e-05 train_metal_acc=0.9205 val_loss=1.2981 val_metal_acc=0.6868 val_metal_min_recall=0.0769 val_fe_recall=0.6286 val_joint_bal_acc=0.6149 val_joint_macro_f1=0.6204 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=45 train_loss=0.2661 lr=1.68015e-05 train_metal_acc=0.9321 val_loss=1.2281 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.6000 val_joint_bal_acc=0.5586 val_joint_macro_f1=0.5789 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=46 train_loss=0.2538 lr=1.68015e-05 train_metal_acc=0.9311 val_loss=1.2609 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6286 val_joint_bal_acc=0.5773 val_joint_macro_f1=0.6044 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=47 train_loss=0.2530 lr=1.68015e-05 train_metal_acc=0.9224 val_loss=1.3010 val_metal_acc=0.6923 val_metal_min_recall=0.0769 val_fe_recall=0.6000 val_joint_bal_acc=0.6255 val_joint_macro_f1=0.6226 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=48 train_loss=0.2296 lr=1.68015e-05 train_metal_acc=0.9370 val_loss=1.3459 val_metal_acc=0.6923 val_metal_min_recall=0.0769 val_fe_recall=0.6000 val_joint_bal_acc=0.5704 val_joint_macro_f1=0.5876 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=49 train_loss=0.2310 lr=1.68015e-05 train_metal_acc=0.9340 val_loss=1.2916 val_metal_acc=0.6923 val_metal_min_recall=0.0769 val_fe_recall=0.6000 val_joint_bal_acc=0.5780 val_joint_macro_f1=0.5968 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=50 train_loss=0.2202 lr=1.68015e-05 train_metal_acc=0.9321 val_loss=1.3271 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6286 val_joint_bal_acc=0.6249 val_joint_macro_f1=0.6359 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_da157306/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_da157306/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_da157306/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_da157306/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_da157306/run_metadata.json
Completed: /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_da157306
================================================================================
[#004 | 4/5] deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_fcc23cc2
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1 --run-name deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_fcc23cc2 --model-architecture gvp --epochs 50 --batch-size 8 --learning-rate 1.6801503587890522e-05 --weight-decay 1e-05 --seed 43 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 256 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 4 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode node_level_late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges
stdout log: /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/_execution_logs/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_fcc23cc2.stdout.log
stderr log: /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/_execution_logs/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_fcc23cc2.stderr.log
================================================================================

=== Split diagnostics (passive; training behavior unchanged) ===
task=metal split_by=pdbid val_fraction=0.15 n_folds=None fold_index=None
pockets: train=1031 validation=182
groups by pdbid: train=1000 validation=94
train/validation overlap counts: pdbid=0, pdbid_chain=0, structure_id=0, pocket_id=0
train metal distribution: Mn=475, Cu=60, Zn=158, Fe=220, Co=67, Ni=51
validation metal distribution: Mn=85, Cu=13, Zn=29, Fe=36, Co=13, Ni=6
missing train metal classes: none
missing validation metal classes: none
train EC distribution: 1=455, 2=161, 3=270, 4=50, 5=69, 6=15
validation EC distribution: 1=49, 2=10, 3=55, 4=3, 5=61, 6=0
missing train EC classes: none
missing validation EC classes: 6
===============================================================

epoch=1 train_loss=1.7641 lr=1.68015e-05 train_metal_acc=0.3317 val_loss=1.7175 val_metal_acc=0.3187 val_metal_min_recall=0.0000 val_fe_recall=0.0000 val_joint_bal_acc=0.2743 val_joint_macro_f1=0.1789 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.6921 lr=1.68015e-05 train_metal_acc=0.3919 val_loss=1.6250 val_metal_acc=0.3462 val_metal_min_recall=0.0000 val_fe_recall=0.1111 val_joint_bal_acc=0.3544 val_joint_macro_f1=0.3256 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.5794 lr=1.68015e-05 train_metal_acc=0.4122 val_loss=1.5602 val_metal_acc=0.3242 val_metal_min_recall=0.0000 val_fe_recall=0.5833 val_joint_bal_acc=0.3631 val_joint_macro_f1=0.2611 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.4386 lr=1.68015e-05 train_metal_acc=0.5674 val_loss=1.4013 val_metal_acc=0.5604 val_metal_min_recall=0.0000 val_fe_recall=0.8611 val_joint_bal_acc=0.3951 val_joint_macro_f1=0.3592 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.3477 lr=1.68015e-05 train_metal_acc=0.5742 val_loss=1.3551 val_metal_acc=0.3736 val_metal_min_recall=0.0000 val_fe_recall=0.5833 val_joint_bal_acc=0.4175 val_joint_macro_f1=0.3402 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.2378 lr=1.68015e-05 train_metal_acc=0.6198 val_loss=1.3554 val_metal_acc=0.3901 val_metal_min_recall=0.0000 val_fe_recall=0.5278 val_joint_bal_acc=0.4142 val_joint_macro_f1=0.3490 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=1.1545 lr=1.68015e-05 train_metal_acc=0.5946 val_loss=1.3607 val_metal_acc=0.3901 val_metal_min_recall=0.0769 val_fe_recall=0.5000 val_joint_bal_acc=0.4788 val_joint_macro_f1=0.4142 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=1.0757 lr=1.68015e-05 train_metal_acc=0.7032 val_loss=1.2713 val_metal_acc=0.4231 val_metal_min_recall=0.0000 val_fe_recall=0.6111 val_joint_bal_acc=0.4388 val_joint_macro_f1=0.4020 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=1.0218 lr=1.68015e-05 train_metal_acc=0.7216 val_loss=1.2427 val_metal_acc=0.4341 val_metal_min_recall=0.0000 val_fe_recall=0.5833 val_joint_bal_acc=0.4696 val_joint_macro_f1=0.4238 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=1.0111 lr=1.68015e-05 train_metal_acc=0.7585 val_loss=1.1700 val_metal_acc=0.6154 val_metal_min_recall=0.0769 val_fe_recall=0.5000 val_joint_bal_acc=0.5000 val_joint_macro_f1=0.4919 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.9207 lr=1.68015e-05 train_metal_acc=0.7022 val_loss=1.2566 val_metal_acc=0.4286 val_metal_min_recall=0.0769 val_fe_recall=0.4444 val_joint_bal_acc=0.4948 val_joint_macro_f1=0.4288 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.8734 lr=1.68015e-05 train_metal_acc=0.6799 val_loss=1.2713 val_metal_acc=0.4176 val_metal_min_recall=0.0769 val_fe_recall=0.3889 val_joint_bal_acc=0.4597 val_joint_macro_f1=0.4289 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.8637 lr=1.68015e-05 train_metal_acc=0.7633 val_loss=1.0728 val_metal_acc=0.6484 val_metal_min_recall=0.0000 val_fe_recall=0.6667 val_joint_bal_acc=0.5421 val_joint_macro_f1=0.5410 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.8087 lr=1.68015e-05 train_metal_acc=0.7439 val_loss=1.1268 val_metal_acc=0.5989 val_metal_min_recall=0.0769 val_fe_recall=0.5556 val_joint_bal_acc=0.5662 val_joint_macro_f1=0.5287 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.7595 lr=1.68015e-05 train_metal_acc=0.7895 val_loss=1.1780 val_metal_acc=0.6044 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5493 val_joint_macro_f1=0.5402 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.7378 lr=1.68015e-05 train_metal_acc=0.8147 val_loss=1.0696 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.6111 val_joint_bal_acc=0.5599 val_joint_macro_f1=0.5566 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.6917 lr=1.68015e-05 train_metal_acc=0.8244 val_loss=1.1376 val_metal_acc=0.6648 val_metal_min_recall=0.0000 val_fe_recall=0.5833 val_joint_bal_acc=0.5150 val_joint_macro_f1=0.4972 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.6662 lr=1.68015e-05 train_metal_acc=0.8109 val_loss=1.0754 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.5556 val_joint_bal_acc=0.5391 val_joint_macro_f1=0.5410 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.6583 lr=1.68015e-05 train_metal_acc=0.8206 val_loss=1.0947 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.6111 val_joint_bal_acc=0.5892 val_joint_macro_f1=0.5704 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.6307 lr=1.68015e-05 train_metal_acc=0.7789 val_loss=1.1659 val_metal_acc=0.5659 val_metal_min_recall=0.0769 val_fe_recall=0.5833 val_joint_bal_acc=0.5609 val_joint_macro_f1=0.5344 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.5893 lr=1.68015e-05 train_metal_acc=0.8244 val_loss=1.0761 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.6111 val_joint_bal_acc=0.5654 val_joint_macro_f1=0.5525 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.5637 lr=1.68015e-05 train_metal_acc=0.8555 val_loss=1.1015 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.6111 val_joint_bal_acc=0.5558 val_joint_macro_f1=0.5494 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.5401 lr=1.68015e-05 train_metal_acc=0.8642 val_loss=1.0482 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.6389 val_joint_bal_acc=0.5645 val_joint_macro_f1=0.5658 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.5300 lr=1.68015e-05 train_metal_acc=0.8438 val_loss=1.1391 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.5833 val_joint_bal_acc=0.5570 val_joint_macro_f1=0.5440 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.4984 lr=1.68015e-05 train_metal_acc=0.8477 val_loss=1.0657 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.6111 val_joint_bal_acc=0.5596 val_joint_macro_f1=0.5494 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.4774 lr=1.68015e-05 train_metal_acc=0.8826 val_loss=1.1511 val_metal_acc=0.6868 val_metal_min_recall=0.0000 val_fe_recall=0.6389 val_joint_bal_acc=0.5130 val_joint_macro_f1=0.5060 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.4659 lr=1.68015e-05 train_metal_acc=0.8865 val_loss=1.1442 val_metal_acc=0.7637 val_metal_min_recall=0.0769 val_fe_recall=0.6389 val_joint_bal_acc=0.5587 val_joint_macro_f1=0.5804 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.4516 lr=1.68015e-05 train_metal_acc=0.8623 val_loss=1.1004 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.6389 val_joint_bal_acc=0.5623 val_joint_macro_f1=0.5565 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.4237 lr=1.68015e-05 train_metal_acc=0.8797 val_loss=1.0918 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.6111 val_joint_bal_acc=0.5617 val_joint_macro_f1=0.5519 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.4118 lr=1.68015e-05 train_metal_acc=0.8700 val_loss=1.1156 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.6389 val_joint_bal_acc=0.5644 val_joint_macro_f1=0.5623 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.4004 lr=1.68015e-05 train_metal_acc=0.8943 val_loss=1.1011 val_metal_acc=0.6868 val_metal_min_recall=0.0769 val_fe_recall=0.5833 val_joint_bal_acc=0.5669 val_joint_macro_f1=0.5641 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.3720 lr=1.68015e-05 train_metal_acc=0.8982 val_loss=1.1657 val_metal_acc=0.7692 val_metal_min_recall=0.0769 val_fe_recall=0.6111 val_joint_bal_acc=0.5909 val_joint_macro_f1=0.6080 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.3716 lr=1.68015e-05 train_metal_acc=0.8574 val_loss=1.1597 val_metal_acc=0.6538 val_metal_min_recall=0.1538 val_fe_recall=0.6389 val_joint_bal_acc=0.5751 val_joint_macro_f1=0.5865 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.3674 lr=1.68015e-05 train_metal_acc=0.8768 val_loss=1.1899 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.6389 val_joint_bal_acc=0.5718 val_joint_macro_f1=0.5654 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.3320 lr=1.68015e-05 train_metal_acc=0.9001 val_loss=1.1376 val_metal_acc=0.6813 val_metal_min_recall=0.0769 val_fe_recall=0.5833 val_joint_bal_acc=0.5574 val_joint_macro_f1=0.5538 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.3364 lr=1.68015e-05 train_metal_acc=0.9059 val_loss=1.1339 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.6111 val_joint_bal_acc=0.5619 val_joint_macro_f1=0.5584 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.3215 lr=1.68015e-05 train_metal_acc=0.9137 val_loss=1.2699 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6111 val_joint_bal_acc=0.5794 val_joint_macro_f1=0.5866 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.3010 lr=1.68015e-05 train_metal_acc=0.9166 val_loss=1.2821 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6389 val_joint_bal_acc=0.5747 val_joint_macro_f1=0.5976 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.3216 lr=1.68015e-05 train_metal_acc=0.9243 val_loss=1.2188 val_metal_acc=0.6813 val_metal_min_recall=0.0769 val_fe_recall=0.6389 val_joint_bal_acc=0.5627 val_joint_macro_f1=0.5699 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.2791 lr=1.68015e-05 train_metal_acc=0.9176 val_loss=1.2077 val_metal_acc=0.6923 val_metal_min_recall=0.0769 val_fe_recall=0.6389 val_joint_bal_acc=0.5704 val_joint_macro_f1=0.5757 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=41 train_loss=0.2959 lr=1.68015e-05 train_metal_acc=0.9214 val_loss=1.2284 val_metal_acc=0.6868 val_metal_min_recall=0.0769 val_fe_recall=0.6389 val_joint_bal_acc=0.5722 val_joint_macro_f1=0.5774 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=42 train_loss=0.2913 lr=1.68015e-05 train_metal_acc=0.9176 val_loss=1.2688 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.5833 val_joint_bal_acc=0.5592 val_joint_macro_f1=0.5541 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=43 train_loss=0.2793 lr=1.68015e-05 train_metal_acc=0.9292 val_loss=1.3024 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.6389 val_joint_bal_acc=0.5781 val_joint_macro_f1=0.5909 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=44 train_loss=0.2508 lr=1.68015e-05 train_metal_acc=0.9292 val_loss=1.3442 val_metal_acc=0.6978 val_metal_min_recall=0.0769 val_fe_recall=0.6389 val_joint_bal_acc=0.5762 val_joint_macro_f1=0.5886 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=45 train_loss=0.2583 lr=1.68015e-05 train_metal_acc=0.9302 val_loss=1.2872 val_metal_acc=0.7637 val_metal_min_recall=0.0769 val_fe_recall=0.5833 val_joint_bal_acc=0.5792 val_joint_macro_f1=0.5996 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=46 train_loss=0.2445 lr=1.68015e-05 train_metal_acc=0.9292 val_loss=1.3366 val_metal_acc=0.7637 val_metal_min_recall=0.0769 val_fe_recall=0.5833 val_joint_bal_acc=0.5792 val_joint_macro_f1=0.5948 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=47 train_loss=0.2468 lr=1.68015e-05 train_metal_acc=0.9350 val_loss=1.3270 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6944 val_joint_bal_acc=0.5669 val_joint_macro_f1=0.5893 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=48 train_loss=0.2544 lr=1.68015e-05 train_metal_acc=0.9350 val_loss=1.4410 val_metal_acc=0.7692 val_metal_min_recall=0.0000 val_fe_recall=0.6389 val_joint_bal_acc=0.5386 val_joint_macro_f1=0.5500 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=49 train_loss=0.2385 lr=1.68015e-05 train_metal_acc=0.9408 val_loss=1.3604 val_metal_acc=0.7692 val_metal_min_recall=0.0769 val_fe_recall=0.5833 val_joint_bal_acc=0.5849 val_joint_macro_f1=0.6002 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=50 train_loss=0.2271 lr=1.68015e-05 train_metal_acc=0.9370 val_loss=1.3573 val_metal_acc=0.7747 val_metal_min_recall=0.0769 val_fe_recall=0.6389 val_joint_bal_acc=0.5664 val_joint_macro_f1=0.5865 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_fcc23cc2/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_fcc23cc2/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_fcc23cc2/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_fcc23cc2/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_fcc23cc2/run_metadata.json
Completed: /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_fcc23cc2
================================================================================
[#005 | 5/5] deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_c8a9d9c9
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1 --run-name deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_c8a9d9c9 --model-architecture gvp --epochs 50 --batch-size 8 --learning-rate 1.6801503587890522e-05 --weight-decay 1e-05 --seed 44 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 256 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 4 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode node_level_late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges
stdout log: /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/_execution_logs/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_c8a9d9c9.stdout.log
stderr log: /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/_execution_logs/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_c8a9d9c9.stderr.log
================================================================================

=== Split diagnostics (passive; training behavior unchanged) ===
task=metal split_by=pdbid val_fraction=0.15 n_folds=None fold_index=None
pockets: train=1031 validation=182
groups by pdbid: train=999 validation=95
train/validation overlap counts: pdbid=0, pdbid_chain=0, structure_id=0, pocket_id=0
train metal distribution: Mn=476, Cu=60, Zn=158, Fe=220, Co=68, Ni=49
validation metal distribution: Mn=84, Cu=13, Zn=29, Fe=36, Co=12, Ni=8
missing train metal classes: none
missing validation metal classes: none
train EC distribution: 1=451, 2=154, 3=281, 4=47, 5=72, 6=13
validation EC distribution: 1=53, 2=17, 3=44, 4=6, 5=58, 6=2
missing train EC classes: none
missing validation EC classes: none
===============================================================

epoch=1 train_loss=1.7658 lr=1.68015e-05 train_metal_acc=0.3763 val_loss=1.6608 val_metal_acc=0.2802 val_metal_min_recall=0.0000 val_fe_recall=0.4444 val_joint_bal_acc=0.2414 val_joint_macro_f1=0.1778 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.6850 lr=1.68015e-05 train_metal_acc=0.4646 val_loss=1.6220 val_metal_acc=0.3571 val_metal_min_recall=0.0000 val_fe_recall=0.8611 val_joint_bal_acc=0.2798 val_joint_macro_f1=0.2500 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.5786 lr=1.68015e-05 train_metal_acc=0.5810 val_loss=1.4311 val_metal_acc=0.6319 val_metal_min_recall=0.0000 val_fe_recall=0.5556 val_joint_bal_acc=0.4111 val_joint_macro_f1=0.3581 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.4287 lr=1.68015e-05 train_metal_acc=0.5713 val_loss=1.3798 val_metal_acc=0.3846 val_metal_min_recall=0.0833 val_fe_recall=0.5000 val_joint_bal_acc=0.4390 val_joint_macro_f1=0.4269 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.3027 lr=1.68015e-05 train_metal_acc=0.6654 val_loss=1.2491 val_metal_acc=0.4341 val_metal_min_recall=0.0000 val_fe_recall=0.5833 val_joint_bal_acc=0.4591 val_joint_macro_f1=0.4221 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.1958 lr=1.68015e-05 train_metal_acc=0.6954 val_loss=1.2031 val_metal_acc=0.5385 val_metal_min_recall=0.0000 val_fe_recall=0.5556 val_joint_bal_acc=0.4800 val_joint_macro_f1=0.4753 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=1.1292 lr=1.68015e-05 train_metal_acc=0.7139 val_loss=1.1895 val_metal_acc=0.5275 val_metal_min_recall=0.0833 val_fe_recall=0.5556 val_joint_bal_acc=0.5025 val_joint_macro_f1=0.4778 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=1.0637 lr=1.68015e-05 train_metal_acc=0.7158 val_loss=1.1723 val_metal_acc=0.4725 val_metal_min_recall=0.0833 val_fe_recall=0.5556 val_joint_bal_acc=0.4940 val_joint_macro_f1=0.4656 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=1.0310 lr=1.68015e-05 train_metal_acc=0.7323 val_loss=1.1622 val_metal_acc=0.4945 val_metal_min_recall=0.1667 val_fe_recall=0.6389 val_joint_bal_acc=0.4879 val_joint_macro_f1=0.4939 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.9541 lr=1.68015e-05 train_metal_acc=0.7730 val_loss=1.0925 val_metal_acc=0.6209 val_metal_min_recall=0.0000 val_fe_recall=0.6389 val_joint_bal_acc=0.5205 val_joint_macro_f1=0.5105 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.9222 lr=1.68015e-05 train_metal_acc=0.7750 val_loss=1.1424 val_metal_acc=0.5714 val_metal_min_recall=0.0833 val_fe_recall=0.6389 val_joint_bal_acc=0.5183 val_joint_macro_f1=0.4983 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.8802 lr=1.68015e-05 train_metal_acc=0.7759 val_loss=1.1359 val_metal_acc=0.5495 val_metal_min_recall=0.0833 val_fe_recall=0.5833 val_joint_bal_acc=0.5051 val_joint_macro_f1=0.4769 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.8307 lr=1.68015e-05 train_metal_acc=0.7662 val_loss=1.1494 val_metal_acc=0.5385 val_metal_min_recall=0.0833 val_fe_recall=0.6389 val_joint_bal_acc=0.4922 val_joint_macro_f1=0.4883 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.7783 lr=1.68015e-05 train_metal_acc=0.7953 val_loss=1.0940 val_metal_acc=0.6099 val_metal_min_recall=0.0833 val_fe_recall=0.6667 val_joint_bal_acc=0.5278 val_joint_macro_f1=0.5272 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.7518 lr=1.68015e-05 train_metal_acc=0.7876 val_loss=1.0744 val_metal_acc=0.6154 val_metal_min_recall=0.0833 val_fe_recall=0.6667 val_joint_bal_acc=0.5001 val_joint_macro_f1=0.5157 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.7444 lr=1.68015e-05 train_metal_acc=0.7944 val_loss=1.1039 val_metal_acc=0.5824 val_metal_min_recall=0.0833 val_fe_recall=0.6389 val_joint_bal_acc=0.5562 val_joint_macro_f1=0.5555 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.6982 lr=1.68015e-05 train_metal_acc=0.8186 val_loss=1.0439 val_metal_acc=0.6374 val_metal_min_recall=0.0000 val_fe_recall=0.6667 val_joint_bal_acc=0.4892 val_joint_macro_f1=0.4878 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.6678 lr=1.68015e-05 train_metal_acc=0.8293 val_loss=1.0112 val_metal_acc=0.6429 val_metal_min_recall=0.0833 val_fe_recall=0.6111 val_joint_bal_acc=0.5010 val_joint_macro_f1=0.5159 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.6560 lr=1.68015e-05 train_metal_acc=0.8390 val_loss=1.0501 val_metal_acc=0.6484 val_metal_min_recall=0.0833 val_fe_recall=0.6389 val_joint_bal_acc=0.5160 val_joint_macro_f1=0.5356 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.6289 lr=1.68015e-05 train_metal_acc=0.8390 val_loss=0.9952 val_metal_acc=0.6648 val_metal_min_recall=0.0833 val_fe_recall=0.6111 val_joint_bal_acc=0.5494 val_joint_macro_f1=0.5576 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.5861 lr=1.68015e-05 train_metal_acc=0.8186 val_loss=1.1202 val_metal_acc=0.5989 val_metal_min_recall=0.0833 val_fe_recall=0.6389 val_joint_bal_acc=0.5367 val_joint_macro_f1=0.5398 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.5723 lr=1.68015e-05 train_metal_acc=0.8244 val_loss=1.0375 val_metal_acc=0.6209 val_metal_min_recall=0.0000 val_fe_recall=0.6389 val_joint_bal_acc=0.4985 val_joint_macro_f1=0.4935 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.5545 lr=1.68015e-05 train_metal_acc=0.8565 val_loss=1.0859 val_metal_acc=0.6813 val_metal_min_recall=0.0833 val_fe_recall=0.6111 val_joint_bal_acc=0.5328 val_joint_macro_f1=0.5556 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.5301 lr=1.68015e-05 train_metal_acc=0.8729 val_loss=1.0318 val_metal_acc=0.6593 val_metal_min_recall=0.0000 val_fe_recall=0.6111 val_joint_bal_acc=0.5064 val_joint_macro_f1=0.5065 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.5058 lr=1.68015e-05 train_metal_acc=0.8371 val_loss=1.0925 val_metal_acc=0.6538 val_metal_min_recall=0.0000 val_fe_recall=0.6111 val_joint_bal_acc=0.5087 val_joint_macro_f1=0.5022 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.5033 lr=1.68015e-05 train_metal_acc=0.8797 val_loss=1.0135 val_metal_acc=0.6484 val_metal_min_recall=0.0000 val_fe_recall=0.6111 val_joint_bal_acc=0.4954 val_joint_macro_f1=0.4972 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.4769 lr=1.68015e-05 train_metal_acc=0.8788 val_loss=1.1722 val_metal_acc=0.6593 val_metal_min_recall=0.0000 val_fe_recall=0.6111 val_joint_bal_acc=0.4956 val_joint_macro_f1=0.5015 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.4693 lr=1.68015e-05 train_metal_acc=0.8875 val_loss=1.0310 val_metal_acc=0.6648 val_metal_min_recall=0.0000 val_fe_recall=0.6111 val_joint_bal_acc=0.5192 val_joint_macro_f1=0.5149 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.4357 lr=1.68015e-05 train_metal_acc=0.8807 val_loss=1.1154 val_metal_acc=0.7143 val_metal_min_recall=0.0000 val_fe_recall=0.6389 val_joint_bal_acc=0.5251 val_joint_macro_f1=0.5299 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.4109 lr=1.68015e-05 train_metal_acc=0.8943 val_loss=1.0242 val_metal_acc=0.6484 val_metal_min_recall=0.0000 val_fe_recall=0.6111 val_joint_bal_acc=0.4949 val_joint_macro_f1=0.4981 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.3933 lr=1.68015e-05 train_metal_acc=0.9069 val_loss=1.0862 val_metal_acc=0.6593 val_metal_min_recall=0.0000 val_fe_recall=0.6111 val_joint_bal_acc=0.4956 val_joint_macro_f1=0.5001 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.3753 lr=1.68015e-05 train_metal_acc=0.9088 val_loss=1.0724 val_metal_acc=0.6593 val_metal_min_recall=0.0000 val_fe_recall=0.6111 val_joint_bal_acc=0.4994 val_joint_macro_f1=0.4985 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.3797 lr=1.68015e-05 train_metal_acc=0.9108 val_loss=1.0782 val_metal_acc=0.6538 val_metal_min_recall=0.0000 val_fe_recall=0.6111 val_joint_bal_acc=0.4936 val_joint_macro_f1=0.4952 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.3617 lr=1.68015e-05 train_metal_acc=0.8904 val_loss=1.1024 val_metal_acc=0.6648 val_metal_min_recall=0.0000 val_fe_recall=0.6111 val_joint_bal_acc=0.5164 val_joint_macro_f1=0.5076 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.3488 lr=1.68015e-05 train_metal_acc=0.8962 val_loss=1.0975 val_metal_acc=0.6593 val_metal_min_recall=0.0000 val_fe_recall=0.6389 val_joint_bal_acc=0.5020 val_joint_macro_f1=0.4991 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.3548 lr=1.68015e-05 train_metal_acc=0.9108 val_loss=1.1354 val_metal_acc=0.6703 val_metal_min_recall=0.0000 val_fe_recall=0.7222 val_joint_bal_acc=0.5101 val_joint_macro_f1=0.5088 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.3351 lr=1.68015e-05 train_metal_acc=0.9108 val_loss=1.2178 val_metal_acc=0.6538 val_metal_min_recall=0.0000 val_fe_recall=0.6111 val_joint_bal_acc=0.4974 val_joint_macro_f1=0.4995 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.3247 lr=1.68015e-05 train_metal_acc=0.9234 val_loss=1.1385 val_metal_acc=0.6538 val_metal_min_recall=0.0000 val_fe_recall=0.6111 val_joint_bal_acc=0.4936 val_joint_macro_f1=0.5003 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.3040 lr=1.68015e-05 train_metal_acc=0.8904 val_loss=1.2397 val_metal_acc=0.6648 val_metal_min_recall=0.0833 val_fe_recall=0.6111 val_joint_bal_acc=0.5352 val_joint_macro_f1=0.5411 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.3223 lr=1.68015e-05 train_metal_acc=0.9176 val_loss=1.2394 val_metal_acc=0.6538 val_metal_min_recall=0.0000 val_fe_recall=0.6111 val_joint_bal_acc=0.4974 val_joint_macro_f1=0.5004 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=41 train_loss=0.2947 lr=1.68015e-05 train_metal_acc=0.9321 val_loss=1.2130 val_metal_acc=0.6538 val_metal_min_recall=0.0000 val_fe_recall=0.6111 val_joint_bal_acc=0.4936 val_joint_macro_f1=0.4993 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=42 train_loss=0.2728 lr=1.68015e-05 train_metal_acc=0.9340 val_loss=1.1588 val_metal_acc=0.6593 val_metal_min_recall=0.0000 val_fe_recall=0.6389 val_joint_bal_acc=0.4982 val_joint_macro_f1=0.5035 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=43 train_loss=0.2728 lr=1.68015e-05 train_metal_acc=0.9321 val_loss=1.2257 val_metal_acc=0.6648 val_metal_min_recall=0.0000 val_fe_recall=0.7222 val_joint_bal_acc=0.5082 val_joint_macro_f1=0.5053 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=44 train_loss=0.2616 lr=1.68015e-05 train_metal_acc=0.9350 val_loss=1.2185 val_metal_acc=0.6538 val_metal_min_recall=0.0000 val_fe_recall=0.6111 val_joint_bal_acc=0.4936 val_joint_macro_f1=0.5003 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=45 train_loss=0.2579 lr=1.68015e-05 train_metal_acc=0.9360 val_loss=1.2410 val_metal_acc=0.7473 val_metal_min_recall=0.0000 val_fe_recall=0.6389 val_joint_bal_acc=0.5479 val_joint_macro_f1=0.5544 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=46 train_loss=0.2393 lr=1.68015e-05 train_metal_acc=0.9030 val_loss=1.2788 val_metal_acc=0.6593 val_metal_min_recall=0.0833 val_fe_recall=0.6111 val_joint_bal_acc=0.5257 val_joint_macro_f1=0.5316 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=47 train_loss=0.2332 lr=1.68015e-05 train_metal_acc=0.9370 val_loss=1.2317 val_metal_acc=0.6758 val_metal_min_recall=0.0833 val_fe_recall=0.6389 val_joint_bal_acc=0.5607 val_joint_macro_f1=0.5772 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=48 train_loss=0.2320 lr=1.68015e-05 train_metal_acc=0.9253 val_loss=1.2795 val_metal_acc=0.6593 val_metal_min_recall=0.0833 val_fe_recall=0.6111 val_joint_bal_acc=0.5370 val_joint_macro_f1=0.5489 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=49 train_loss=0.2263 lr=1.68015e-05 train_metal_acc=0.8962 val_loss=1.2163 val_metal_acc=0.6538 val_metal_min_recall=0.1667 val_fe_recall=0.6667 val_joint_bal_acc=0.5749 val_joint_macro_f1=0.5856 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=50 train_loss=0.2345 lr=1.68015e-05 train_metal_acc=0.9350 val_loss=1.3273 val_metal_acc=0.6593 val_metal_min_recall=0.0833 val_fe_recall=0.5833 val_joint_bal_acc=0.5306 val_joint_macro_f1=0.5455 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_c8a9d9c9/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_c8a9d9c9/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_c8a9d9c9/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_c8a9d9c9/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_c8a9d9c9/run_metadata.json
Completed: /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_c8a9d9c9
Completed run directories: ['/content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_539ea463', '/content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_a7759f1f', '/content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_da157306', '/content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_fcc23cc2', '/content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_baseline_batchmetal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1_metal_gvp_+_node_leve_c8a9d9c9']
Failed run directories: []
Execution records JSON: /content/deepmzyme_outputs/runs/metal_node_level_late_fusion_from_latefusion_trial49_seedrepeat_50epoch_v1/deepmzyme_nonoverlap_model_comparison_execution_records.json
