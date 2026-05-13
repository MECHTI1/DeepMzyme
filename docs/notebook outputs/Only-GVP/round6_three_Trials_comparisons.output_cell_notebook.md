Common Settings For All 3
TASK = metal
RUN_MODE = manual_configurations
RECOMMENDED_RUN_SET = custom
MODEL_PRESET = Only-GVP

EPOCHS = 50
BATCH_SIZES_CSV = 8
SEEDS_CSV = 42,123,2026,43,44

VAL_FRACTION = 0.15
SPLIT_BY = pdbid
SELECTION_METRIC = val_metal_balanced_acc
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False

HIDDEN_S_VALUES_CSV = 128
HIDDEN_V_VALUES_CSV = 32
EDGE_HIDDEN_VALUES_CSV = 128
HEAD_MLP_LAYERS_VALUES_CSV = 1
METAL_CLASS_WEIGHT_MODES_CSV = inverse_sqrt_frequency

RING_EDGE_MODE = without_ring
ALLOW_MISSING_EXTERNAL_FEATURES = True
PREPARE_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_RING_EDGES = False

Run 1: Trial12 gvp_layers=3
RUN_BATCH_ID = metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat
LEARNING_RATES_CSV = 4.752317377508605e-05
WEIGHT_DECAYS_CSV = 0.0
EDGE_RADIUS_VALUES_CSV = 6.0
GVP_LAYERS_VALUES_CSV = 3


Run 2: Trial7 gvp_layers=4
RUN_BATCH_ID = metal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat
LEARNING_RATES_CSV = 6.464669746492395e-05
WEIGHT_DECAYS_CSV = 0.001
EDGE_RADIUS_VALUES_CSV = 6.0
GVP_LAYERS_VALUES_CSV = 4


Run 3: Trial12 gvp_layers=2
RUN_BATCH_ID = metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat
LEARNING_RATES_CSV = 4.752317377508605e-05
WEIGHT_DECAYS_CSV = 0.0
EDGE_RADIUS_VALUES_CSV = 6.0
GVP_LAYERS_VALUES_CSV = 2


Each run should produce 5 runs total:
1 config × 5 seeds = 5 runs


#-----------------------------------------------
# Detailed results: Run 1: Trial12 gvp_layers=3
## Summary scanning scope: current RUN_BATCH_ID folder
RUN_BATCH_ID: metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat

Runs directory scanned: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat
/usr/bin/python3 /content/DeepMzyme/src/report_runs.py --runs-dir /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat --out-csv /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_model_comparison_completed_only.csv --out-figure /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_model_comparison.png
Completed-run summary CSV: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_model_comparison_completed_only.csv
Summary source mode: planned table from current notebook state
Summary source scope: current planned rows plus completed runs under the scanned directory.
Comparison CSV: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_model_comparison.csv
rank	source_mode	config_source	result_stage	run_name	status	error_message	model_preset	model_display	model_architecture	...	missing_train_metal_classes	missing_val_metal_classes	missing_train_ec_classes	missing_val_ec_classes	selected_best_validation_metric_value	held_out_test_metric_name	held_out_test_metric_value	run_dir	stdout_log_path	stderr_log_path
0	1	planned table	current notebook config	validation-only	deepmzyme_nonoverlap_baseline_batchmetal_only_...	completed		Only-GVP	Only-GVP (structure only)	only_gvp	...	NaN	NaN	NaN	7	0.618412	test_metal_balanced_acc	NaN	/content/deepmzyme_outputs/runs/metal_only_gvp...	/content/deepmzyme_outputs/runs/metal_only_gvp...	/content/deepmzyme_outputs/runs/metal_only_gvp...
1	2	planned table	current notebook config	validation-only	deepmzyme_nonoverlap_baseline_batchmetal_only_...	completed		Only-GVP	Only-GVP (structure only)	only_gvp	...	NaN	NaN	NaN	7	0.617459	test_metal_balanced_acc	NaN	/content/deepmzyme_outputs/runs/metal_only_gvp...	/content/deepmzyme_outputs/runs/metal_only_gvp...	/content/deepmzyme_outputs/runs/metal_only_gvp...
2	3	planned table	current notebook config	validation-only	deepmzyme_nonoverlap_baseline_batchmetal_only_...	completed		Only-GVP	Only-GVP (structure only)	only_gvp	...	NaN	NaN	NaN	7	0.616986	test_metal_balanced_acc	NaN	/content/deepmzyme_outputs/runs/metal_only_gvp...	/content/deepmzyme_outputs/runs/metal_only_gvp...	/content/deepmzyme_outputs/runs/metal_only_gvp...
3	4	planned table	current notebook config	validation-only	deepmzyme_nonoverlap_baseline_batchmetal_only_...	completed		Only-GVP	Only-GVP (structure only)	only_gvp	...	NaN	NaN	NaN	7	0.615360	test_metal_balanced_acc	NaN	/content/deepmzyme_outputs/runs/metal_only_gvp...	/content/deepmzyme_outputs/runs/metal_only_gvp...	/content/deepmzyme_outputs/runs/metal_only_gvp...
4	5	planned table	current notebook config	validation-only	deepmzyme_nonoverlap_baseline_batchmetal_only_...	completed		Only-GVP	Only-GVP (structure only)	only_gvp	...	NaN	NaN	NaN	7	0.567107	test_metal_balanced_acc	NaN	/content/deepmzyme_outputs/runs/metal_only_gvp...	/content/deepmzyme_outputs/runs/metal_only_gvp...	/content/deepmzyme_outputs/runs/metal_only_gvp...
5 rows × 63 columns




Ranked table sorted by validation selection metric:
#1: deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_0956e7d3 | stage=validation-only | class_weight=inverse_sqrt_frequency | val_metal_balanced_acc=0.6184115476458212 | status=completed
#2: deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_6eab65a2 | stage=validation-only | class_weight=inverse_sqrt_frequency | val_metal_balanced_acc=0.6174587773404129 | status=completed
#3: deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_dcfd4dee | stage=validation-only | class_weight=inverse_sqrt_frequency | val_metal_balanced_acc=0.6169860965797898 | status=completed
#4: deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_c13ce556 | stage=validation-only | class_weight=inverse_sqrt_frequency | val_metal_balanced_acc=0.6153603430889 | status=completed
#5: deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_e9575b8a | stage=validation-only | class_weight=inverse_sqrt_frequency | val_metal_balanced_acc=0.5671069802190937 | status=completed

Best overall configuration: deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_0956e7d3
{
  "run_name": "deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_0956e7d3",
  "result_stage": "validation-only",
  "model_preset": "Only-GVP",
  "model_architecture": "only_gvp",
  "fusion_mode": "none",
  "metal_class_weight_mode": "inverse_sqrt_frequency",
  "balance_metal_site_symbols": false,
  "selection_metric": "val_metal_balanced_acc",
  "selected_best_validation_metric_value": 0.6184115476458212,
  "run_dir": "/content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_0956e7d3"
}

Best configuration per model preset/mode:
Only-GVP: deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_0956e7d3 | class_weight=inverse_sqrt_frequency | val_metal_balanced_acc=0.6184115476458212

Best Only-GVP configuration: deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_0956e7d3
Best ESM-based configuration: not available
Best RING vs non-RING comparison: not available unless both modes have completed numeric validation metrics.

Automatic interpretation
Best validation config: deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_0956e7d3 with val_metal_balanced_acc = 0.6184115476458212
Top fusion mode: none
Best learning-rate region: middle around 1e-4 (lr=4.752317377508605e-05)
Held-out test results present: False
Recommended next step: select/retrain the final validation-best configuration, then run held-out test evaluation once
Drive copy skipped. Outputs remain under: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat
#-------------------------------
## Configured output locations:
  Runs root:       /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat  [exists]
  Summary CSV:     /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_model_comparison.csv  [exists]
  Summary figure:  /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_model_comparison.png  [exists]

Completed run directories found: 5
choice_index	run_name	task	model	fusion	seed	learning_rate	weight_decay	selection_metric	metric_direction	best_validation_value	test_metric	test_metric_value	selected_epoch	test_report_saved	run_dir
0	1	deepmzyme_nonoverlap_baseline_batchmetal_only_...	metal	GVP only	late_fusion	123	0.000048	0.0	val_metal_balanced_acc	higher_is_better	0.618412	test_metal_balanced_acc	None	20	False	/content/deepmzyme_outputs/runs/metal_only_gvp...
1	2	deepmzyme_nonoverlap_baseline_batchmetal_only_...	metal	GVP only	late_fusion	2026	0.000048	0.0	val_metal_balanced_acc	higher_is_better	0.617459	test_metal_balanced_acc	None	36	False	/content/deepmzyme_outputs/runs/metal_only_gvp...
2	3	deepmzyme_nonoverlap_baseline_batchmetal_only_...	metal	GVP only	late_fusion	44	0.000048	0.0	val_metal_balanced_acc	higher_is_better	0.616986	test_metal_balanced_acc	None	25	False	/content/deepmzyme_outputs/runs/metal_only_gvp...
3	4	deepmzyme_nonoverlap_baseline_batchmetal_only_...	metal	GVP only	late_fusion	43	0.000048	0.0	val_metal_balanced_acc	higher_is_better	0.615360	test_metal_balanced_acc	None	37	False	/content/deepmzyme_outputs/runs/metal_only_gvp...
4	5	deepmzyme_nonoverlap_baseline_batchmetal_only_...	metal	GVP only	late_fusion	42	0.000048	0.0	val_metal_balanced_acc	higher_is_better	0.567107	test_metal_balanced_acc	None	47	False	/content/deepmzyme_outputs/runs/metal_only_gvp...



Selected final run: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_0956e7d3
  Selection mode: auto_best_validation
  Task:         metal
  Architecture: only_gvp
  Fusion:       late_fusion
  Seed:         123
  Best val val_metal_balanced_acc: 0.6184  (epoch 20)
  Split:        train_and_test_sets_structures_non_overlapped_pinmymetal
Configured output locations:
  Runs root:       /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat  [exists]
  Summary CSV:     /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_model_comparison.csv  [exists]
  Summary figure:  /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_model_comparison.png  [exists]
  Selected run:    /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_0956e7d3  [exists]
  Run config:      /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_0956e7d3/run_config.json  [exists]
  Run metadata:    /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_0956e7d3/run_metadata.json  [exists]
  Test report:     /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_0956e7d3/test_report.json  [not created yet]
  Selection JSON: /content/deepmzyme_outputs/runs/deepmzyme_final_selected_run.json

No test_report.json found for the selected run.
Use the optional final held-out test evaluation cell next; its default mode evaluates the selected saved checkpoint without retraining.
Keep choosing models by validation metrics; use held-out test metrics only for the selected final run.

#-----------------------------------------------
## epochs
Runnable planned configurations: 5
================================================================================
[#001 | 1/5] deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_e9575b8a
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat --run-name deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_e9575b8a --model-architecture only_gvp --epochs 50 --batch-size 8 --learning-rate 4.752317377508605e-05 --weight-decay 0.0 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 128 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_sqrt_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 3 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges
stdout log: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/_execution_logs/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_e9575b8a.stdout.log
stderr log: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/_execution_logs/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_e9575b8a.stderr.log
================================================================================

=== Split diagnostics (passive; training behavior unchanged) ===
task=metal split_by=pdbid val_fraction=0.15 n_folds=None fold_index=None
pockets: train=1181 validation=208
groups by pdbid: train=1151 validation=110
train/validation overlap counts: pdbid=0, pdbid_chain=0, structure_id=0, pocket_id=0
train metal distribution: Mn=551, Cu=70, Zn=175, Fe=248, Co=73, Ni=64
validation metal distribution: Mn=97, Cu=15, Zn=32, Fe=44, Co=13, Ni=7
missing train metal classes: none
missing validation metal classes: none
train EC distribution: 1=491, 2=197, 3=344, 4=46, 5=69, 6=15, 7=1
validation EC distribution: 1=68, 2=18, 3=48, 4=8, 5=64, 6=2, 7=0
missing train EC classes: none
missing validation EC classes: 7
===============================================================

epoch=1 train_loss=1.7062 lr=4.75232e-05 train_metal_acc=0.4666 val_loss=1.6109 val_metal_acc=0.4663 val_metal_min_recall=0.0000 val_fe_recall=0.0000 val_joint_bal_acc=0.1667 val_joint_macro_f1=0.1060 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.6740 lr=4.75232e-05 train_metal_acc=0.4716 val_loss=1.5783 val_metal_acc=0.4663 val_metal_min_recall=0.0000 val_fe_recall=0.0000 val_joint_bal_acc=0.1667 val_joint_macro_f1=0.1064 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.6455 lr=4.75232e-05 train_metal_acc=0.4877 val_loss=1.5266 val_metal_acc=0.4808 val_metal_min_recall=0.0000 val_fe_recall=0.0682 val_joint_bal_acc=0.1780 val_joint_macro_f1=0.1281 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.5846 lr=4.75232e-05 train_metal_acc=0.5538 val_loss=1.4084 val_metal_acc=0.6538 val_metal_min_recall=0.0000 val_fe_recall=0.6818 val_joint_bal_acc=0.3838 val_joint_macro_f1=0.3582 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.4706 lr=4.75232e-05 train_metal_acc=0.6139 val_loss=1.2217 val_metal_acc=0.7067 val_metal_min_recall=0.0000 val_fe_recall=0.8182 val_joint_bal_acc=0.4420 val_joint_macro_f1=0.4212 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.3598 lr=4.75232e-05 train_metal_acc=0.6257 val_loss=1.2451 val_metal_acc=0.6971 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4700 val_joint_macro_f1=0.4198 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=1.3125 lr=4.75232e-05 train_metal_acc=0.5580 val_loss=1.2044 val_metal_acc=0.7067 val_metal_min_recall=0.0000 val_fe_recall=0.8864 val_joint_bal_acc=0.4822 val_joint_macro_f1=0.4243 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=1.2270 lr=4.75232e-05 train_metal_acc=0.6901 val_loss=1.1432 val_metal_acc=0.7067 val_metal_min_recall=0.0000 val_fe_recall=0.6591 val_joint_bal_acc=0.4790 val_joint_macro_f1=0.4484 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=1.1947 lr=4.75232e-05 train_metal_acc=0.6715 val_loss=1.0735 val_metal_acc=0.7067 val_metal_min_recall=0.0000 val_fe_recall=0.7727 val_joint_bal_acc=0.4649 val_joint_macro_f1=0.4166 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=1.1527 lr=4.75232e-05 train_metal_acc=0.6816 val_loss=1.1969 val_metal_acc=0.6587 val_metal_min_recall=0.0000 val_fe_recall=0.4091 val_joint_bal_acc=0.4483 val_joint_macro_f1=0.4302 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=1.1068 lr=4.75232e-05 train_metal_acc=0.7265 val_loss=1.1018 val_metal_acc=0.6587 val_metal_min_recall=0.0000 val_fe_recall=0.6591 val_joint_bal_acc=0.4944 val_joint_macro_f1=0.4756 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=1.0874 lr=4.75232e-05 train_metal_acc=0.7248 val_loss=1.0843 val_metal_acc=0.6827 val_metal_min_recall=0.0000 val_fe_recall=0.6818 val_joint_bal_acc=0.5015 val_joint_macro_f1=0.4767 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=1.0356 lr=4.75232e-05 train_metal_acc=0.7409 val_loss=1.1302 val_metal_acc=0.6298 val_metal_min_recall=0.0000 val_fe_recall=0.5227 val_joint_bal_acc=0.4891 val_joint_macro_f1=0.4655 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=1.0124 lr=4.75232e-05 train_metal_acc=0.7189 val_loss=1.1078 val_metal_acc=0.6538 val_metal_min_recall=0.0000 val_fe_recall=0.8409 val_joint_bal_acc=0.4845 val_joint_macro_f1=0.4865 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.9890 lr=4.75232e-05 train_metal_acc=0.7511 val_loss=1.0636 val_metal_acc=0.6490 val_metal_min_recall=0.0000 val_fe_recall=0.7045 val_joint_bal_acc=0.5276 val_joint_macro_f1=0.5089 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.9597 lr=4.75232e-05 train_metal_acc=0.7705 val_loss=1.0368 val_metal_acc=0.7356 val_metal_min_recall=0.0769 val_fe_recall=0.7045 val_joint_bal_acc=0.5522 val_joint_macro_f1=0.5268 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.9409 lr=4.75232e-05 train_metal_acc=0.7477 val_loss=1.0376 val_metal_acc=0.6731 val_metal_min_recall=0.0000 val_fe_recall=0.7045 val_joint_bal_acc=0.5176 val_joint_macro_f1=0.5019 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.9244 lr=4.75232e-05 train_metal_acc=0.7773 val_loss=1.0622 val_metal_acc=0.6442 val_metal_min_recall=0.0000 val_fe_recall=0.7045 val_joint_bal_acc=0.5179 val_joint_macro_f1=0.4744 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.8906 lr=4.75232e-05 train_metal_acc=0.7697 val_loss=1.1002 val_metal_acc=0.6154 val_metal_min_recall=0.0000 val_fe_recall=0.6136 val_joint_bal_acc=0.4734 val_joint_macro_f1=0.4547 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.8552 lr=4.75232e-05 train_metal_acc=0.7612 val_loss=1.0999 val_metal_acc=0.6346 val_metal_min_recall=0.0000 val_fe_recall=0.7727 val_joint_bal_acc=0.5392 val_joint_macro_f1=0.4740 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.8570 lr=4.75232e-05 train_metal_acc=0.7925 val_loss=1.0529 val_metal_acc=0.6731 val_metal_min_recall=0.0000 val_fe_recall=0.7727 val_joint_bal_acc=0.5237 val_joint_macro_f1=0.5247 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.8164 lr=4.75232e-05 train_metal_acc=0.7782 val_loss=1.1002 val_metal_acc=0.6106 val_metal_min_recall=0.0000 val_fe_recall=0.5455 val_joint_bal_acc=0.5018 val_joint_macro_f1=0.4742 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.8099 lr=4.75232e-05 train_metal_acc=0.8103 val_loss=1.1104 val_metal_acc=0.5865 val_metal_min_recall=0.0000 val_fe_recall=0.7500 val_joint_bal_acc=0.5094 val_joint_macro_f1=0.4846 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.7831 lr=4.75232e-05 train_metal_acc=0.7824 val_loss=1.1418 val_metal_acc=0.6394 val_metal_min_recall=0.0769 val_fe_recall=0.6818 val_joint_bal_acc=0.5438 val_joint_macro_f1=0.5088 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.7758 lr=4.75232e-05 train_metal_acc=0.8137 val_loss=1.1929 val_metal_acc=0.6010 val_metal_min_recall=0.0000 val_fe_recall=0.8409 val_joint_bal_acc=0.5391 val_joint_macro_f1=0.5102 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.7432 lr=4.75232e-05 train_metal_acc=0.8256 val_loss=1.0913 val_metal_acc=0.6058 val_metal_min_recall=0.0769 val_fe_recall=0.6818 val_joint_bal_acc=0.5374 val_joint_macro_f1=0.5180 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.7122 lr=4.75232e-05 train_metal_acc=0.8163 val_loss=1.2305 val_metal_acc=0.5481 val_metal_min_recall=0.0000 val_fe_recall=0.7500 val_joint_bal_acc=0.5108 val_joint_macro_f1=0.4634 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.7199 lr=4.75232e-05 train_metal_acc=0.8273 val_loss=1.2163 val_metal_acc=0.5673 val_metal_min_recall=0.0769 val_fe_recall=0.7045 val_joint_bal_acc=0.4918 val_joint_macro_f1=0.4831 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.6825 lr=4.75232e-05 train_metal_acc=0.8222 val_loss=1.2083 val_metal_acc=0.5721 val_metal_min_recall=0.0000 val_fe_recall=0.7045 val_joint_bal_acc=0.4859 val_joint_macro_f1=0.4802 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.6829 lr=4.75232e-05 train_metal_acc=0.8425 val_loss=1.1603 val_metal_acc=0.6010 val_metal_min_recall=0.0769 val_fe_recall=0.7500 val_joint_bal_acc=0.5128 val_joint_macro_f1=0.5026 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.6592 lr=4.75232e-05 train_metal_acc=0.8036 val_loss=1.2975 val_metal_acc=0.4856 val_metal_min_recall=0.0000 val_fe_recall=0.7273 val_joint_bal_acc=0.4818 val_joint_macro_f1=0.4411 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.6546 lr=4.75232e-05 train_metal_acc=0.8273 val_loss=1.1928 val_metal_acc=0.6346 val_metal_min_recall=0.0769 val_fe_recall=0.6591 val_joint_bal_acc=0.5142 val_joint_macro_f1=0.5107 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.6173 lr=4.75232e-05 train_metal_acc=0.8290 val_loss=1.2157 val_metal_acc=0.6346 val_metal_min_recall=0.0769 val_fe_recall=0.6591 val_joint_bal_acc=0.5096 val_joint_macro_f1=0.4982 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.5976 lr=4.75232e-05 train_metal_acc=0.8357 val_loss=1.1337 val_metal_acc=0.6731 val_metal_min_recall=0.0769 val_fe_recall=0.7727 val_joint_bal_acc=0.5208 val_joint_macro_f1=0.5169 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.5891 lr=4.75232e-05 train_metal_acc=0.8561 val_loss=1.1331 val_metal_acc=0.6587 val_metal_min_recall=0.1538 val_fe_recall=0.8182 val_joint_bal_acc=0.5250 val_joint_macro_f1=0.5456 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.5674 lr=4.75232e-05 train_metal_acc=0.8332 val_loss=1.3525 val_metal_acc=0.4952 val_metal_min_recall=0.0769 val_fe_recall=0.8864 val_joint_bal_acc=0.4662 val_joint_macro_f1=0.4632 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.5688 lr=4.75232e-05 train_metal_acc=0.8611 val_loss=1.3058 val_metal_acc=0.5337 val_metal_min_recall=0.0769 val_fe_recall=0.7273 val_joint_bal_acc=0.4972 val_joint_macro_f1=0.4852 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.5529 lr=4.75232e-05 train_metal_acc=0.8603 val_loss=1.2317 val_metal_acc=0.6058 val_metal_min_recall=0.1429 val_fe_recall=0.8409 val_joint_bal_acc=0.4777 val_joint_macro_f1=0.4949 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.5364 lr=4.75232e-05 train_metal_acc=0.8476 val_loss=1.2337 val_metal_acc=0.5817 val_metal_min_recall=0.0769 val_fe_recall=0.8636 val_joint_bal_acc=0.4860 val_joint_macro_f1=0.4953 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.5095 lr=4.75232e-05 train_metal_acc=0.8501 val_loss=1.1039 val_metal_acc=0.7260 val_metal_min_recall=0.0000 val_fe_recall=0.7727 val_joint_bal_acc=0.5544 val_joint_macro_f1=0.5425 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=41 train_loss=0.5071 lr=4.75232e-05 train_metal_acc=0.8510 val_loss=1.1734 val_metal_acc=0.6683 val_metal_min_recall=0.0769 val_fe_recall=0.8864 val_joint_bal_acc=0.5491 val_joint_macro_f1=0.5444 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=42 train_loss=0.4881 lr=4.75232e-05 train_metal_acc=0.8823 val_loss=1.2702 val_metal_acc=0.5865 val_metal_min_recall=0.0769 val_fe_recall=0.7955 val_joint_bal_acc=0.4989 val_joint_macro_f1=0.5087 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=43 train_loss=0.4773 lr=4.75232e-05 train_metal_acc=0.8281 val_loss=1.3816 val_metal_acc=0.5337 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.4140 val_joint_macro_f1=0.4250 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=44 train_loss=0.4600 lr=4.75232e-05 train_metal_acc=0.8815 val_loss=1.4750 val_metal_acc=0.5192 val_metal_min_recall=0.0000 val_fe_recall=0.8182 val_joint_bal_acc=0.4610 val_joint_macro_f1=0.4631 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=45 train_loss=0.4537 lr=4.75232e-05 train_metal_acc=0.8925 val_loss=1.3218 val_metal_acc=0.5240 val_metal_min_recall=0.0769 val_fe_recall=0.7727 val_joint_bal_acc=0.4944 val_joint_macro_f1=0.4988 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=46 train_loss=0.4341 lr=4.75232e-05 train_metal_acc=0.8577 val_loss=1.2914 val_metal_acc=0.5433 val_metal_min_recall=0.0000 val_fe_recall=0.7955 val_joint_bal_acc=0.4981 val_joint_macro_f1=0.4982 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=47 train_loss=0.4234 lr=4.75232e-05 train_metal_acc=0.9136 val_loss=1.1323 val_metal_acc=0.7115 val_metal_min_recall=0.0769 val_fe_recall=0.7727 val_joint_bal_acc=0.5671 val_joint_macro_f1=0.5797 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=48 train_loss=0.4056 lr=4.75232e-05 train_metal_acc=0.9035 val_loss=1.3598 val_metal_acc=0.5481 val_metal_min_recall=0.0769 val_fe_recall=0.7727 val_joint_bal_acc=0.4783 val_joint_macro_f1=0.4951 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=49 train_loss=0.3527 lr=4.75232e-05 train_metal_acc=0.8806 val_loss=1.2175 val_metal_acc=0.7260 val_metal_min_recall=0.0769 val_fe_recall=0.8182 val_joint_bal_acc=0.5463 val_joint_macro_f1=0.5424 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=50 train_loss=0.3825 lr=4.75232e-05 train_metal_acc=0.9196 val_loss=1.3098 val_metal_acc=0.5962 val_metal_min_recall=0.0769 val_fe_recall=0.8409 val_joint_bal_acc=0.5334 val_joint_macro_f1=0.5169 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_e9575b8a/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_e9575b8a/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_e9575b8a/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_e9575b8a/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_e9575b8a/run_metadata.json
Completed: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_e9575b8a
================================================================================
[#002 | 2/5] deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_0956e7d3
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat --run-name deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_0956e7d3 --model-architecture only_gvp --epochs 50 --batch-size 8 --learning-rate 4.752317377508605e-05 --weight-decay 0.0 --seed 123 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 128 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_sqrt_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 3 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges
stdout log: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/_execution_logs/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_0956e7d3.stdout.log
stderr log: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/_execution_logs/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_0956e7d3.stderr.log
================================================================================

=== Split diagnostics (passive; training behavior unchanged) ===
task=metal split_by=pdbid val_fraction=0.15 n_folds=None fold_index=None
pockets: train=1181 validation=208
groups by pdbid: train=1152 validation=109
train/validation overlap counts: pdbid=0, pdbid_chain=0, structure_id=0, pocket_id=0
train metal distribution: Mn=551, Cu=70, Zn=173, Fe=251, Co=73, Ni=63
validation metal distribution: Mn=97, Cu=15, Zn=34, Fe=41, Co=13, Ni=8
missing train metal classes: none
missing validation metal classes: none
train EC distribution: 1=495, 2=199, 3=346, 4=46, 5=64, 6=14, 7=1
validation EC distribution: 1=64, 2=16, 3=46, 4=8, 5=69, 6=3, 7=0
missing train EC classes: none
missing validation EC classes: 7
===============================================================

epoch=1 train_loss=1.6968 lr=4.75232e-05 train_metal_acc=0.4666 val_loss=1.6295 val_metal_acc=0.4663 val_metal_min_recall=0.0000 val_fe_recall=0.0000 val_joint_bal_acc=0.1667 val_joint_macro_f1=0.1060 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.6683 lr=4.75232e-05 train_metal_acc=0.4725 val_loss=1.6090 val_metal_acc=0.4808 val_metal_min_recall=0.0000 val_fe_recall=0.0732 val_joint_bal_acc=0.1789 val_joint_macro_f1=0.1296 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.6095 lr=4.75232e-05 train_metal_acc=0.4022 val_loss=1.6088 val_metal_acc=0.4375 val_metal_min_recall=0.0000 val_fe_recall=0.0732 val_joint_bal_acc=0.2774 val_joint_macro_f1=0.2505 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.5115 lr=4.75232e-05 train_metal_acc=0.5927 val_loss=1.3667 val_metal_acc=0.6779 val_metal_min_recall=0.0000 val_fe_recall=0.7561 val_joint_bal_acc=0.4156 val_joint_macro_f1=0.3840 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.4006 lr=4.75232e-05 train_metal_acc=0.5885 val_loss=1.2772 val_metal_acc=0.6298 val_metal_min_recall=0.0000 val_fe_recall=0.4878 val_joint_bal_acc=0.4131 val_joint_macro_f1=0.3407 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.3030 lr=4.75232e-05 train_metal_acc=0.6596 val_loss=1.1867 val_metal_acc=0.7596 val_metal_min_recall=0.0000 val_fe_recall=0.8780 val_joint_bal_acc=0.5545 val_joint_macro_f1=0.5481 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=1.2478 lr=4.75232e-05 train_metal_acc=0.6291 val_loss=1.2567 val_metal_acc=0.5817 val_metal_min_recall=0.0000 val_fe_recall=0.7073 val_joint_bal_acc=0.4522 val_joint_macro_f1=0.4069 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=1.2134 lr=4.75232e-05 train_metal_acc=0.7003 val_loss=1.1277 val_metal_acc=0.7308 val_metal_min_recall=0.0000 val_fe_recall=0.7073 val_joint_bal_acc=0.5310 val_joint_macro_f1=0.5269 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=1.1630 lr=4.75232e-05 train_metal_acc=0.6647 val_loss=1.2015 val_metal_acc=0.6010 val_metal_min_recall=0.0000 val_fe_recall=0.4390 val_joint_bal_acc=0.4427 val_joint_macro_f1=0.4047 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=1.1281 lr=4.75232e-05 train_metal_acc=0.7138 val_loss=1.1056 val_metal_acc=0.7452 val_metal_min_recall=0.0000 val_fe_recall=0.8049 val_joint_bal_acc=0.5519 val_joint_macro_f1=0.5606 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=1.1194 lr=4.75232e-05 train_metal_acc=0.7138 val_loss=1.1545 val_metal_acc=0.6298 val_metal_min_recall=0.0769 val_fe_recall=0.7073 val_joint_bal_acc=0.5281 val_joint_macro_f1=0.5154 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=1.0819 lr=4.75232e-05 train_metal_acc=0.6782 val_loss=1.2072 val_metal_acc=0.5721 val_metal_min_recall=0.1538 val_fe_recall=0.8537 val_joint_bal_acc=0.5167 val_joint_macro_f1=0.4957 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=1.0877 lr=4.75232e-05 train_metal_acc=0.7248 val_loss=1.1190 val_metal_acc=0.6779 val_metal_min_recall=0.0769 val_fe_recall=0.6585 val_joint_bal_acc=0.5344 val_joint_macro_f1=0.5476 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=1.0268 lr=4.75232e-05 train_metal_acc=0.7358 val_loss=1.1599 val_metal_acc=0.6010 val_metal_min_recall=0.0000 val_fe_recall=0.6585 val_joint_bal_acc=0.5022 val_joint_macro_f1=0.4893 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=1.0000 lr=4.75232e-05 train_metal_acc=0.7240 val_loss=1.1884 val_metal_acc=0.6010 val_metal_min_recall=0.0769 val_fe_recall=0.7561 val_joint_bal_acc=0.5543 val_joint_macro_f1=0.5193 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.9862 lr=4.75232e-05 train_metal_acc=0.7494 val_loss=1.0596 val_metal_acc=0.7019 val_metal_min_recall=0.0769 val_fe_recall=0.7805 val_joint_bal_acc=0.5991 val_joint_macro_f1=0.5754 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.9541 lr=4.75232e-05 train_metal_acc=0.6571 val_loss=1.3560 val_metal_acc=0.4471 val_metal_min_recall=0.0000 val_fe_recall=0.5366 val_joint_bal_acc=0.4735 val_joint_macro_f1=0.4250 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.9564 lr=4.75232e-05 train_metal_acc=0.7553 val_loss=1.1559 val_metal_acc=0.6106 val_metal_min_recall=0.0769 val_fe_recall=0.6098 val_joint_bal_acc=0.5341 val_joint_macro_f1=0.5228 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.9240 lr=4.75232e-05 train_metal_acc=0.7646 val_loss=1.1029 val_metal_acc=0.6442 val_metal_min_recall=0.0000 val_fe_recall=0.6829 val_joint_bal_acc=0.5421 val_joint_macro_f1=0.5225 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.9065 lr=4.75232e-05 train_metal_acc=0.7748 val_loss=1.0652 val_metal_acc=0.6779 val_metal_min_recall=0.1538 val_fe_recall=0.7561 val_joint_bal_acc=0.6184 val_joint_macro_f1=0.6078 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.8861 lr=4.75232e-05 train_metal_acc=0.7824 val_loss=1.1575 val_metal_acc=0.6490 val_metal_min_recall=0.1538 val_fe_recall=0.7317 val_joint_bal_acc=0.5612 val_joint_macro_f1=0.5479 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.8339 lr=4.75232e-05 train_metal_acc=0.7197 val_loss=1.1955 val_metal_acc=0.6154 val_metal_min_recall=0.0769 val_fe_recall=0.9756 val_joint_bal_acc=0.5871 val_joint_macro_f1=0.5615 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.8477 lr=4.75232e-05 train_metal_acc=0.7841 val_loss=1.0323 val_metal_acc=0.6875 val_metal_min_recall=0.0000 val_fe_recall=0.8293 val_joint_bal_acc=0.5494 val_joint_macro_f1=0.5309 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.8054 lr=4.75232e-05 train_metal_acc=0.7883 val_loss=1.1464 val_metal_acc=0.6442 val_metal_min_recall=0.1538 val_fe_recall=0.7317 val_joint_bal_acc=0.5628 val_joint_macro_f1=0.5530 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.7877 lr=4.75232e-05 train_metal_acc=0.8052 val_loss=1.0601 val_metal_acc=0.7019 val_metal_min_recall=0.1538 val_fe_recall=0.7805 val_joint_bal_acc=0.5943 val_joint_macro_f1=0.5846 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.7646 lr=4.75232e-05 train_metal_acc=0.8078 val_loss=1.1015 val_metal_acc=0.6827 val_metal_min_recall=0.1538 val_fe_recall=0.7561 val_joint_bal_acc=0.5819 val_joint_macro_f1=0.5774 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.7516 lr=4.75232e-05 train_metal_acc=0.8052 val_loss=1.0706 val_metal_acc=0.6827 val_metal_min_recall=0.0769 val_fe_recall=0.7561 val_joint_bal_acc=0.5643 val_joint_macro_f1=0.5426 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.7462 lr=4.75232e-05 train_metal_acc=0.7968 val_loss=1.0764 val_metal_acc=0.6587 val_metal_min_recall=0.0769 val_fe_recall=0.7317 val_joint_bal_acc=0.5217 val_joint_macro_f1=0.5185 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.7245 lr=4.75232e-05 train_metal_acc=0.8163 val_loss=1.1445 val_metal_acc=0.6587 val_metal_min_recall=0.1538 val_fe_recall=0.6585 val_joint_bal_acc=0.5544 val_joint_macro_f1=0.5402 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.6978 lr=4.75232e-05 train_metal_acc=0.8230 val_loss=1.1066 val_metal_acc=0.6442 val_metal_min_recall=0.0769 val_fe_recall=0.6341 val_joint_bal_acc=0.5644 val_joint_macro_f1=0.5349 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.6798 lr=4.75232e-05 train_metal_acc=0.8340 val_loss=1.1486 val_metal_acc=0.6779 val_metal_min_recall=0.0769 val_fe_recall=0.7317 val_joint_bal_acc=0.5825 val_joint_macro_f1=0.5692 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.6616 lr=4.75232e-05 train_metal_acc=0.8307 val_loss=1.1486 val_metal_acc=0.6683 val_metal_min_recall=0.1538 val_fe_recall=0.6585 val_joint_bal_acc=0.5578 val_joint_macro_f1=0.5531 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.6348 lr=4.75232e-05 train_metal_acc=0.8493 val_loss=1.1223 val_metal_acc=0.6875 val_metal_min_recall=0.0769 val_fe_recall=0.7317 val_joint_bal_acc=0.5829 val_joint_macro_f1=0.5614 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.6418 lr=4.75232e-05 train_metal_acc=0.8450 val_loss=1.0964 val_metal_acc=0.6683 val_metal_min_recall=0.0769 val_fe_recall=0.7805 val_joint_bal_acc=0.5585 val_joint_macro_f1=0.5555 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.6145 lr=4.75232e-05 train_metal_acc=0.8459 val_loss=1.1596 val_metal_acc=0.6587 val_metal_min_recall=0.1538 val_fe_recall=0.9024 val_joint_bal_acc=0.5874 val_joint_macro_f1=0.5875 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.5890 lr=4.75232e-05 train_metal_acc=0.8577 val_loss=1.1112 val_metal_acc=0.6875 val_metal_min_recall=0.1538 val_fe_recall=0.8049 val_joint_bal_acc=0.6074 val_joint_macro_f1=0.5808 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.5887 lr=4.75232e-05 train_metal_acc=0.8442 val_loss=1.1857 val_metal_acc=0.6683 val_metal_min_recall=0.0769 val_fe_recall=0.6341 val_joint_bal_acc=0.5697 val_joint_macro_f1=0.5555 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.5832 lr=4.75232e-05 train_metal_acc=0.8561 val_loss=1.1789 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.7561 val_joint_bal_acc=0.5605 val_joint_macro_f1=0.5402 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.5392 lr=4.75232e-05 train_metal_acc=0.8442 val_loss=1.1479 val_metal_acc=0.6731 val_metal_min_recall=0.1538 val_fe_recall=0.8780 val_joint_bal_acc=0.5904 val_joint_macro_f1=0.5809 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.5301 lr=4.75232e-05 train_metal_acc=0.8603 val_loss=1.2570 val_metal_acc=0.6442 val_metal_min_recall=0.0769 val_fe_recall=0.7805 val_joint_bal_acc=0.5562 val_joint_macro_f1=0.5225 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=41 train_loss=0.5175 lr=4.75232e-05 train_metal_acc=0.8662 val_loss=1.2225 val_metal_acc=0.6587 val_metal_min_recall=0.1538 val_fe_recall=0.6585 val_joint_bal_acc=0.5767 val_joint_macro_f1=0.5563 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=42 train_loss=0.4958 lr=4.75232e-05 train_metal_acc=0.8891 val_loss=1.1311 val_metal_acc=0.6731 val_metal_min_recall=0.0769 val_fe_recall=0.7805 val_joint_bal_acc=0.5729 val_joint_macro_f1=0.5643 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=43 train_loss=0.4734 lr=4.75232e-05 train_metal_acc=0.8400 val_loss=1.3246 val_metal_acc=0.6442 val_metal_min_recall=0.0769 val_fe_recall=0.5366 val_joint_bal_acc=0.5488 val_joint_macro_f1=0.5433 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=44 train_loss=0.4613 lr=4.75232e-05 train_metal_acc=0.8704 val_loss=1.2435 val_metal_acc=0.6587 val_metal_min_recall=0.0769 val_fe_recall=0.8537 val_joint_bal_acc=0.5500 val_joint_macro_f1=0.5529 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=45 train_loss=0.4609 lr=4.75232e-05 train_metal_acc=0.8899 val_loss=1.1673 val_metal_acc=0.6635 val_metal_min_recall=0.0769 val_fe_recall=0.8049 val_joint_bal_acc=0.5591 val_joint_macro_f1=0.5748 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=46 train_loss=0.4724 lr=4.75232e-05 train_metal_acc=0.8865 val_loss=1.3304 val_metal_acc=0.6106 val_metal_min_recall=0.0769 val_fe_recall=0.6829 val_joint_bal_acc=0.5477 val_joint_macro_f1=0.5349 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=47 train_loss=0.4181 lr=4.75232e-05 train_metal_acc=0.8984 val_loss=1.2785 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.6829 val_joint_bal_acc=0.5630 val_joint_macro_f1=0.5425 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=48 train_loss=0.4521 lr=4.75232e-05 train_metal_acc=0.8831 val_loss=1.2610 val_metal_acc=0.6154 val_metal_min_recall=0.1538 val_fe_recall=0.8049 val_joint_bal_acc=0.5371 val_joint_macro_f1=0.5159 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=49 train_loss=0.4021 lr=4.75232e-05 train_metal_acc=0.9009 val_loss=1.2390 val_metal_acc=0.6731 val_metal_min_recall=0.1538 val_fe_recall=0.7317 val_joint_bal_acc=0.5637 val_joint_macro_f1=0.5750 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=50 train_loss=0.4247 lr=4.75232e-05 train_metal_acc=0.8840 val_loss=1.2388 val_metal_acc=0.6635 val_metal_min_recall=0.1538 val_fe_recall=0.8537 val_joint_bal_acc=0.5625 val_joint_macro_f1=0.5846 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_0956e7d3/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_0956e7d3/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_0956e7d3/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_0956e7d3/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_0956e7d3/run_metadata.json
Completed: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_0956e7d3
================================================================================
[#003 | 3/5] deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_6eab65a2
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat --run-name deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_6eab65a2 --model-architecture only_gvp --epochs 50 --batch-size 8 --learning-rate 4.752317377508605e-05 --weight-decay 0.0 --seed 2026 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 128 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_sqrt_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 3 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges
stdout log: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/_execution_logs/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_6eab65a2.stdout.log
stderr log: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/_execution_logs/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_6eab65a2.stderr.log
================================================================================

=== Split diagnostics (passive; training behavior unchanged) ===
task=metal split_by=pdbid val_fraction=0.15 n_folds=None fold_index=None
pockets: train=1181 validation=208
groups by pdbid: train=1152 validation=109
train/validation overlap counts: pdbid=0, pdbid_chain=0, structure_id=0, pocket_id=0
train metal distribution: Mn=549, Cu=72, Zn=173, Fe=251, Co=73, Ni=63
validation metal distribution: Mn=99, Cu=13, Zn=34, Fe=41, Co=13, Ni=8
missing train metal classes: none
missing validation metal classes: none
train EC distribution: 1=495, 2=198, 3=338, 4=47, 5=73, 6=15, 7=1
validation EC distribution: 1=64, 2=17, 3=54, 4=7, 5=60, 6=2, 7=0
missing train EC classes: none
missing validation EC classes: 7
===============================================================

epoch=1 train_loss=1.6795 lr=4.75232e-05 train_metal_acc=0.4615 val_loss=1.6196 val_metal_acc=0.3750 val_metal_min_recall=0.0000 val_fe_recall=0.6585 val_joint_bal_acc=0.1956 val_joint_macro_f1=0.1462 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.6480 lr=4.75232e-05 train_metal_acc=0.4928 val_loss=1.6092 val_metal_acc=0.3173 val_metal_min_recall=0.0000 val_fe_recall=0.1951 val_joint_bal_acc=0.1752 val_joint_macro_f1=0.1557 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.6254 lr=4.75232e-05 train_metal_acc=0.4699 val_loss=1.5233 val_metal_acc=0.4856 val_metal_min_recall=0.0000 val_fe_recall=0.0488 val_joint_bal_acc=0.1748 val_joint_macro_f1=0.1237 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.5723 lr=4.75232e-05 train_metal_acc=0.5605 val_loss=1.4538 val_metal_acc=0.5769 val_metal_min_recall=0.0000 val_fe_recall=0.3902 val_joint_bal_acc=0.3248 val_joint_macro_f1=0.3347 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.4576 lr=4.75232e-05 train_metal_acc=0.5783 val_loss=1.3122 val_metal_acc=0.5913 val_metal_min_recall=0.0000 val_fe_recall=0.4390 val_joint_bal_acc=0.3724 val_joint_macro_f1=0.3077 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.3668 lr=4.75232e-05 train_metal_acc=0.6384 val_loss=1.2623 val_metal_acc=0.6106 val_metal_min_recall=0.0000 val_fe_recall=0.7805 val_joint_bal_acc=0.4526 val_joint_macro_f1=0.4042 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=1.2954 lr=4.75232e-05 train_metal_acc=0.6478 val_loss=1.2202 val_metal_acc=0.6587 val_metal_min_recall=0.0000 val_fe_recall=0.8537 val_joint_bal_acc=0.4797 val_joint_macro_f1=0.4285 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=1.2363 lr=4.75232e-05 train_metal_acc=0.6867 val_loss=1.1614 val_metal_acc=0.6827 val_metal_min_recall=0.0000 val_fe_recall=0.6098 val_joint_bal_acc=0.4548 val_joint_macro_f1=0.4273 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=1.1902 lr=4.75232e-05 train_metal_acc=0.6655 val_loss=1.1827 val_metal_acc=0.6106 val_metal_min_recall=0.0000 val_fe_recall=0.8293 val_joint_bal_acc=0.4510 val_joint_macro_f1=0.3928 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=1.1460 lr=4.75232e-05 train_metal_acc=0.6765 val_loss=1.1239 val_metal_acc=0.6635 val_metal_min_recall=0.0000 val_fe_recall=0.8780 val_joint_bal_acc=0.5093 val_joint_macro_f1=0.4700 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=1.1059 lr=4.75232e-05 train_metal_acc=0.7367 val_loss=1.1087 val_metal_acc=0.6683 val_metal_min_recall=0.0000 val_fe_recall=0.6829 val_joint_bal_acc=0.5465 val_joint_macro_f1=0.5317 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=1.0586 lr=4.75232e-05 train_metal_acc=0.7121 val_loss=1.1610 val_metal_acc=0.6298 val_metal_min_recall=0.0000 val_fe_recall=0.7073 val_joint_bal_acc=0.5418 val_joint_macro_f1=0.5119 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=1.0378 lr=4.75232e-05 train_metal_acc=0.7375 val_loss=1.0446 val_metal_acc=0.6971 val_metal_min_recall=0.0000 val_fe_recall=0.7073 val_joint_bal_acc=0.5109 val_joint_macro_f1=0.4971 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=1.0181 lr=4.75232e-05 train_metal_acc=0.7451 val_loss=1.1089 val_metal_acc=0.6346 val_metal_min_recall=0.0769 val_fe_recall=0.5854 val_joint_bal_acc=0.5267 val_joint_macro_f1=0.5042 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.9901 lr=4.75232e-05 train_metal_acc=0.6969 val_loss=1.1003 val_metal_acc=0.6683 val_metal_min_recall=0.0000 val_fe_recall=0.9512 val_joint_bal_acc=0.5278 val_joint_macro_f1=0.4916 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.9684 lr=4.75232e-05 train_metal_acc=0.7561 val_loss=1.0928 val_metal_acc=0.6202 val_metal_min_recall=0.0769 val_fe_recall=0.6098 val_joint_bal_acc=0.5208 val_joint_macro_f1=0.5009 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.9473 lr=4.75232e-05 train_metal_acc=0.7638 val_loss=1.0310 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.6341 val_joint_bal_acc=0.5575 val_joint_macro_f1=0.5407 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.9345 lr=4.75232e-05 train_metal_acc=0.7638 val_loss=1.0321 val_metal_acc=0.6683 val_metal_min_recall=0.0769 val_fe_recall=0.6341 val_joint_bal_acc=0.5274 val_joint_macro_f1=0.5050 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.9126 lr=4.75232e-05 train_metal_acc=0.7765 val_loss=1.0226 val_metal_acc=0.6731 val_metal_min_recall=0.0769 val_fe_recall=0.6829 val_joint_bal_acc=0.5722 val_joint_macro_f1=0.5672 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.8893 lr=4.75232e-05 train_metal_acc=0.7544 val_loss=1.0484 val_metal_acc=0.6731 val_metal_min_recall=0.0000 val_fe_recall=0.9024 val_joint_bal_acc=0.5361 val_joint_macro_f1=0.5297 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.8591 lr=4.75232e-05 train_metal_acc=0.7434 val_loss=1.0798 val_metal_acc=0.6058 val_metal_min_recall=0.0769 val_fe_recall=0.4634 val_joint_bal_acc=0.5302 val_joint_macro_f1=0.4827 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.8687 lr=4.75232e-05 train_metal_acc=0.7942 val_loss=0.9720 val_metal_acc=0.7260 val_metal_min_recall=0.0769 val_fe_recall=0.7317 val_joint_bal_acc=0.5987 val_joint_macro_f1=0.6066 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.8266 lr=4.75232e-05 train_metal_acc=0.7968 val_loss=0.9822 val_metal_acc=0.7115 val_metal_min_recall=0.0769 val_fe_recall=0.8537 val_joint_bal_acc=0.5669 val_joint_macro_f1=0.5555 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.8038 lr=4.75232e-05 train_metal_acc=0.7858 val_loss=1.0294 val_metal_acc=0.6683 val_metal_min_recall=0.0769 val_fe_recall=0.7805 val_joint_bal_acc=0.5702 val_joint_macro_f1=0.5380 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.7914 lr=4.75232e-05 train_metal_acc=0.8086 val_loss=1.0900 val_metal_acc=0.6154 val_metal_min_recall=0.0769 val_fe_recall=0.7561 val_joint_bal_acc=0.5525 val_joint_macro_f1=0.5316 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.7554 lr=4.75232e-05 train_metal_acc=0.8052 val_loss=1.0455 val_metal_acc=0.6442 val_metal_min_recall=0.0769 val_fe_recall=0.6585 val_joint_bal_acc=0.5373 val_joint_macro_f1=0.5331 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.7621 lr=4.75232e-05 train_metal_acc=0.7443 val_loss=1.1790 val_metal_acc=0.6010 val_metal_min_recall=0.0000 val_fe_recall=0.9512 val_joint_bal_acc=0.5363 val_joint_macro_f1=0.5141 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.7542 lr=4.75232e-05 train_metal_acc=0.8002 val_loss=1.0790 val_metal_acc=0.6346 val_metal_min_recall=0.0000 val_fe_recall=0.7561 val_joint_bal_acc=0.5547 val_joint_macro_f1=0.5180 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.7213 lr=4.75232e-05 train_metal_acc=0.8205 val_loss=1.0530 val_metal_acc=0.6490 val_metal_min_recall=0.1538 val_fe_recall=0.8537 val_joint_bal_acc=0.5516 val_joint_macro_f1=0.5613 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.7015 lr=4.75232e-05 train_metal_acc=0.8349 val_loss=1.0576 val_metal_acc=0.6635 val_metal_min_recall=0.1538 val_fe_recall=0.9024 val_joint_bal_acc=0.6012 val_joint_macro_f1=0.5949 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.6873 lr=4.75232e-05 train_metal_acc=0.7824 val_loss=1.0507 val_metal_acc=0.6154 val_metal_min_recall=0.0769 val_fe_recall=0.5122 val_joint_bal_acc=0.5306 val_joint_macro_f1=0.5279 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.6758 lr=4.75232e-05 train_metal_acc=0.8417 val_loss=1.0214 val_metal_acc=0.6538 val_metal_min_recall=0.0000 val_fe_recall=0.8537 val_joint_bal_acc=0.5740 val_joint_macro_f1=0.5369 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.6463 lr=4.75232e-05 train_metal_acc=0.8146 val_loss=1.0829 val_metal_acc=0.6298 val_metal_min_recall=0.0769 val_fe_recall=0.7805 val_joint_bal_acc=0.6064 val_joint_macro_f1=0.5615 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.6368 lr=4.75232e-05 train_metal_acc=0.8180 val_loss=1.2175 val_metal_acc=0.5913 val_metal_min_recall=0.0000 val_fe_recall=0.9268 val_joint_bal_acc=0.5529 val_joint_macro_f1=0.5339 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.6120 lr=4.75232e-05 train_metal_acc=0.8611 val_loss=1.0527 val_metal_acc=0.6587 val_metal_min_recall=0.0769 val_fe_recall=0.8049 val_joint_bal_acc=0.5871 val_joint_macro_f1=0.5684 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.6088 lr=4.75232e-05 train_metal_acc=0.8095 val_loss=1.2373 val_metal_acc=0.5865 val_metal_min_recall=0.1538 val_fe_recall=0.7561 val_joint_bal_acc=0.6175 val_joint_macro_f1=0.5341 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.5865 lr=4.75232e-05 train_metal_acc=0.8637 val_loss=0.9834 val_metal_acc=0.7548 val_metal_min_recall=0.0000 val_fe_recall=0.8780 val_joint_bal_acc=0.6006 val_joint_macro_f1=0.6077 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.5382 lr=4.75232e-05 train_metal_acc=0.8721 val_loss=1.0782 val_metal_acc=0.6731 val_metal_min_recall=0.0769 val_fe_recall=0.8780 val_joint_bal_acc=0.5688 val_joint_macro_f1=0.5875 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.5470 lr=4.75232e-05 train_metal_acc=0.8552 val_loss=1.0515 val_metal_acc=0.6490 val_metal_min_recall=0.0769 val_fe_recall=0.7073 val_joint_bal_acc=0.5774 val_joint_macro_f1=0.5592 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.5243 lr=4.75232e-05 train_metal_acc=0.8569 val_loss=1.2345 val_metal_acc=0.6394 val_metal_min_recall=0.0769 val_fe_recall=0.9268 val_joint_bal_acc=0.5874 val_joint_macro_f1=0.5857 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=41 train_loss=0.5160 lr=4.75232e-05 train_metal_acc=0.8561 val_loss=1.0526 val_metal_acc=0.7452 val_metal_min_recall=0.0769 val_fe_recall=0.8537 val_joint_bal_acc=0.5981 val_joint_macro_f1=0.6346 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=42 train_loss=0.4931 lr=4.75232e-05 train_metal_acc=0.8620 val_loss=1.2840 val_metal_acc=0.6202 val_metal_min_recall=0.0000 val_fe_recall=0.8780 val_joint_bal_acc=0.5614 val_joint_macro_f1=0.5295 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=43 train_loss=0.5115 lr=4.75232e-05 train_metal_acc=0.8857 val_loss=1.0816 val_metal_acc=0.6971 val_metal_min_recall=0.0000 val_fe_recall=0.8780 val_joint_bal_acc=0.5725 val_joint_macro_f1=0.5702 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=44 train_loss=0.4667 lr=4.75232e-05 train_metal_acc=0.8933 val_loss=1.2212 val_metal_acc=0.6202 val_metal_min_recall=0.0769 val_fe_recall=0.6829 val_joint_bal_acc=0.5841 val_joint_macro_f1=0.5579 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=45 train_loss=0.4622 lr=4.75232e-05 train_metal_acc=0.8848 val_loss=1.1069 val_metal_acc=0.6731 val_metal_min_recall=0.0769 val_fe_recall=0.9024 val_joint_bal_acc=0.5744 val_joint_macro_f1=0.5823 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=46 train_loss=0.4322 lr=4.75232e-05 train_metal_acc=0.8899 val_loss=1.1564 val_metal_acc=0.6683 val_metal_min_recall=0.0769 val_fe_recall=0.6341 val_joint_bal_acc=0.6040 val_joint_macro_f1=0.5630 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=47 train_loss=0.4200 lr=4.75232e-05 train_metal_acc=0.8789 val_loss=1.0909 val_metal_acc=0.7067 val_metal_min_recall=0.0769 val_fe_recall=0.6829 val_joint_bal_acc=0.5841 val_joint_macro_f1=0.5791 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=48 train_loss=0.4100 lr=4.75232e-05 train_metal_acc=0.8857 val_loss=1.3639 val_metal_acc=0.5865 val_metal_min_recall=0.0000 val_fe_recall=0.8049 val_joint_bal_acc=0.5475 val_joint_macro_f1=0.4995 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=49 train_loss=0.4053 lr=4.75232e-05 train_metal_acc=0.8891 val_loss=1.1873 val_metal_acc=0.6490 val_metal_min_recall=0.0769 val_fe_recall=0.5854 val_joint_bal_acc=0.5432 val_joint_macro_f1=0.5400 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=50 train_loss=0.3983 lr=4.75232e-05 train_metal_acc=0.9145 val_loss=1.0999 val_metal_acc=0.7163 val_metal_min_recall=0.0769 val_fe_recall=0.8049 val_joint_bal_acc=0.5786 val_joint_macro_f1=0.5835 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_6eab65a2/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_6eab65a2/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_6eab65a2/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_6eab65a2/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_6eab65a2/run_metadata.json
Completed: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_6eab65a2
================================================================================
[#004 | 4/5] deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_c13ce556
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat --run-name deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_c13ce556 --model-architecture only_gvp --epochs 50 --batch-size 8 --learning-rate 4.752317377508605e-05 --weight-decay 0.0 --seed 43 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 128 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_sqrt_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 3 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges
stdout log: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/_execution_logs/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_c13ce556.stdout.log
stderr log: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/_execution_logs/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_c13ce556.stderr.log
================================================================================

=== Split diagnostics (passive; training behavior unchanged) ===
task=metal split_by=pdbid val_fraction=0.15 n_folds=None fold_index=None
pockets: train=1181 validation=208
groups by pdbid: train=1151 validation=110
train/validation overlap counts: pdbid=0, pdbid_chain=0, structure_id=0, pocket_id=0
train metal distribution: Mn=551, Cu=72, Zn=173, Fe=251, Co=73, Ni=61
validation metal distribution: Mn=97, Cu=13, Zn=34, Fe=41, Co=13, Ni=10
missing train metal classes: none
missing validation metal classes: none
train EC distribution: 1=497, 2=200, 3=332, 4=46, 5=75, 6=14, 7=1
validation EC distribution: 1=62, 2=15, 3=60, 4=8, 5=58, 6=3, 7=0
missing train EC classes: none
missing validation EC classes: 7
===============================================================

epoch=1 train_loss=1.6919 lr=4.75232e-05 train_metal_acc=0.4666 val_loss=1.5953 val_metal_acc=0.4760 val_metal_min_recall=0.0000 val_fe_recall=0.0488 val_joint_bal_acc=0.1748 val_joint_macro_f1=0.1222 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.6471 lr=4.75232e-05 train_metal_acc=0.4869 val_loss=1.5551 val_metal_acc=0.4952 val_metal_min_recall=0.0000 val_fe_recall=0.1463 val_joint_bal_acc=0.1911 val_joint_macro_f1=0.1497 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.5803 lr=4.75232e-05 train_metal_acc=0.5106 val_loss=1.4593 val_metal_acc=0.6202 val_metal_min_recall=0.0000 val_fe_recall=0.9756 val_joint_bal_acc=0.3409 val_joint_macro_f1=0.2911 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.4546 lr=4.75232e-05 train_metal_acc=0.6097 val_loss=1.2774 val_metal_acc=0.7067 val_metal_min_recall=0.0000 val_fe_recall=0.8293 val_joint_bal_acc=0.5159 val_joint_macro_f1=0.4978 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.3780 lr=4.75232e-05 train_metal_acc=0.6283 val_loss=1.2646 val_metal_acc=0.7500 val_metal_min_recall=0.0000 val_fe_recall=0.7805 val_joint_bal_acc=0.6106 val_joint_macro_f1=0.5635 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.3061 lr=4.75232e-05 train_metal_acc=0.6715 val_loss=1.1910 val_metal_acc=0.7548 val_metal_min_recall=0.0000 val_fe_recall=0.8293 val_joint_bal_acc=0.5572 val_joint_macro_f1=0.5316 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=1.2398 lr=4.75232e-05 train_metal_acc=0.6782 val_loss=1.1781 val_metal_acc=0.7644 val_metal_min_recall=0.0000 val_fe_recall=0.8293 val_joint_bal_acc=0.5560 val_joint_macro_f1=0.5593 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=1.1859 lr=4.75232e-05 train_metal_acc=0.6715 val_loss=1.2041 val_metal_acc=0.6442 val_metal_min_recall=0.0000 val_fe_recall=0.8780 val_joint_bal_acc=0.5577 val_joint_macro_f1=0.4778 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=1.1713 lr=4.75232e-05 train_metal_acc=0.7113 val_loss=1.1618 val_metal_acc=0.6587 val_metal_min_recall=0.0000 val_fe_recall=0.6829 val_joint_bal_acc=0.5365 val_joint_macro_f1=0.5105 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=1.1223 lr=4.75232e-05 train_metal_acc=0.6647 val_loss=1.1464 val_metal_acc=0.6779 val_metal_min_recall=0.0000 val_fe_recall=0.7073 val_joint_bal_acc=0.4798 val_joint_macro_f1=0.4419 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=1.0897 lr=4.75232e-05 train_metal_acc=0.7231 val_loss=1.1289 val_metal_acc=0.6635 val_metal_min_recall=0.0000 val_fe_recall=0.8780 val_joint_bal_acc=0.5656 val_joint_macro_f1=0.5404 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=1.0482 lr=4.75232e-05 train_metal_acc=0.7053 val_loss=1.1247 val_metal_acc=0.6683 val_metal_min_recall=0.0000 val_fe_recall=0.8780 val_joint_bal_acc=0.5167 val_joint_macro_f1=0.4930 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=1.0644 lr=4.75232e-05 train_metal_acc=0.7062 val_loss=1.1939 val_metal_acc=0.5962 val_metal_min_recall=0.0000 val_fe_recall=0.8293 val_joint_bal_acc=0.5101 val_joint_macro_f1=0.4681 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.9971 lr=4.75232e-05 train_metal_acc=0.7409 val_loss=1.1635 val_metal_acc=0.6058 val_metal_min_recall=0.0000 val_fe_recall=0.7073 val_joint_bal_acc=0.5199 val_joint_macro_f1=0.4913 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.9804 lr=4.75232e-05 train_metal_acc=0.7113 val_loss=1.1295 val_metal_acc=0.6587 val_metal_min_recall=0.0000 val_fe_recall=0.6585 val_joint_bal_acc=0.4937 val_joint_macro_f1=0.4722 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.9695 lr=4.75232e-05 train_metal_acc=0.7443 val_loss=1.1314 val_metal_acc=0.6346 val_metal_min_recall=0.0000 val_fe_recall=0.6585 val_joint_bal_acc=0.5469 val_joint_macro_f1=0.5218 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.9357 lr=4.75232e-05 train_metal_acc=0.7519 val_loss=1.2578 val_metal_acc=0.5721 val_metal_min_recall=0.2308 val_fe_recall=0.7073 val_joint_bal_acc=0.5498 val_joint_macro_f1=0.5312 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.9221 lr=4.75232e-05 train_metal_acc=0.7375 val_loss=1.1632 val_metal_acc=0.6154 val_metal_min_recall=0.0000 val_fe_recall=0.8537 val_joint_bal_acc=0.5556 val_joint_macro_f1=0.5263 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.8996 lr=4.75232e-05 train_metal_acc=0.7341 val_loss=1.1267 val_metal_acc=0.6587 val_metal_min_recall=0.0000 val_fe_recall=0.7561 val_joint_bal_acc=0.5127 val_joint_macro_f1=0.4810 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.8800 lr=4.75232e-05 train_metal_acc=0.7401 val_loss=1.2797 val_metal_acc=0.6010 val_metal_min_recall=0.0769 val_fe_recall=0.6341 val_joint_bal_acc=0.5554 val_joint_macro_f1=0.5186 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.8753 lr=4.75232e-05 train_metal_acc=0.7722 val_loss=1.2740 val_metal_acc=0.5865 val_metal_min_recall=0.1538 val_fe_recall=0.7805 val_joint_bal_acc=0.5722 val_joint_macro_f1=0.5479 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.8286 lr=4.75232e-05 train_metal_acc=0.7384 val_loss=1.3584 val_metal_acc=0.5385 val_metal_min_recall=0.0000 val_fe_recall=0.8049 val_joint_bal_acc=0.5234 val_joint_macro_f1=0.4858 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.8216 lr=4.75232e-05 train_metal_acc=0.7790 val_loss=1.1205 val_metal_acc=0.6827 val_metal_min_recall=0.0769 val_fe_recall=0.8293 val_joint_bal_acc=0.6056 val_joint_macro_f1=0.5794 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.7836 lr=4.75232e-05 train_metal_acc=0.7790 val_loss=1.1436 val_metal_acc=0.6635 val_metal_min_recall=0.0769 val_fe_recall=0.6341 val_joint_bal_acc=0.5596 val_joint_macro_f1=0.5542 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.7918 lr=4.75232e-05 train_metal_acc=0.7782 val_loss=1.2230 val_metal_acc=0.6490 val_metal_min_recall=0.0769 val_fe_recall=0.7317 val_joint_bal_acc=0.5457 val_joint_macro_f1=0.5518 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.7513 lr=4.75232e-05 train_metal_acc=0.7976 val_loss=1.1629 val_metal_acc=0.6923 val_metal_min_recall=0.0769 val_fe_recall=0.8780 val_joint_bal_acc=0.6083 val_joint_macro_f1=0.5992 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.7512 lr=4.75232e-05 train_metal_acc=0.8027 val_loss=1.1709 val_metal_acc=0.6635 val_metal_min_recall=0.1538 val_fe_recall=0.6585 val_joint_bal_acc=0.5730 val_joint_macro_f1=0.5706 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.7400 lr=4.75232e-05 train_metal_acc=0.8027 val_loss=1.1881 val_metal_acc=0.6346 val_metal_min_recall=0.1538 val_fe_recall=0.8293 val_joint_bal_acc=0.6036 val_joint_macro_f1=0.5969 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.7169 lr=4.75232e-05 train_metal_acc=0.7985 val_loss=1.1974 val_metal_acc=0.6779 val_metal_min_recall=0.1538 val_fe_recall=0.8049 val_joint_bal_acc=0.6008 val_joint_macro_f1=0.5927 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.7046 lr=4.75232e-05 train_metal_acc=0.8036 val_loss=1.1163 val_metal_acc=0.6827 val_metal_min_recall=0.1538 val_fe_recall=0.7805 val_joint_bal_acc=0.5821 val_joint_macro_f1=0.5851 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.6794 lr=4.75232e-05 train_metal_acc=0.7900 val_loss=1.1924 val_metal_acc=0.6635 val_metal_min_recall=0.0769 val_fe_recall=0.9268 val_joint_bal_acc=0.5899 val_joint_macro_f1=0.5701 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.6635 lr=4.75232e-05 train_metal_acc=0.8069 val_loss=1.1788 val_metal_acc=0.6731 val_metal_min_recall=0.1538 val_fe_recall=0.8537 val_joint_bal_acc=0.6124 val_joint_macro_f1=0.5760 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.6545 lr=4.75232e-05 train_metal_acc=0.8298 val_loss=1.1824 val_metal_acc=0.6731 val_metal_min_recall=0.0769 val_fe_recall=0.8049 val_joint_bal_acc=0.5816 val_joint_macro_f1=0.5612 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.6360 lr=4.75232e-05 train_metal_acc=0.8273 val_loss=1.1744 val_metal_acc=0.6875 val_metal_min_recall=0.0000 val_fe_recall=0.8780 val_joint_bal_acc=0.5945 val_joint_macro_f1=0.5564 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.6201 lr=4.75232e-05 train_metal_acc=0.8239 val_loss=1.1600 val_metal_acc=0.6827 val_metal_min_recall=0.0000 val_fe_recall=0.8780 val_joint_bal_acc=0.5842 val_joint_macro_f1=0.5541 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.5974 lr=4.75232e-05 train_metal_acc=0.8383 val_loss=1.2416 val_metal_acc=0.6202 val_metal_min_recall=0.0000 val_fe_recall=0.8780 val_joint_bal_acc=0.5699 val_joint_macro_f1=0.5422 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.5766 lr=4.75232e-05 train_metal_acc=0.8332 val_loss=1.2612 val_metal_acc=0.6298 val_metal_min_recall=0.3077 val_fe_recall=0.8537 val_joint_bal_acc=0.6154 val_joint_macro_f1=0.6046 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.5684 lr=4.75232e-05 train_metal_acc=0.8434 val_loss=1.1737 val_metal_acc=0.6779 val_metal_min_recall=0.0769 val_fe_recall=0.8780 val_joint_bal_acc=0.6016 val_joint_macro_f1=0.5916 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.5721 lr=4.75232e-05 train_metal_acc=0.8654 val_loss=1.1327 val_metal_acc=0.6827 val_metal_min_recall=0.0769 val_fe_recall=0.8780 val_joint_bal_acc=0.5937 val_joint_macro_f1=0.5870 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.5444 lr=4.75232e-05 train_metal_acc=0.8442 val_loss=1.1852 val_metal_acc=0.6635 val_metal_min_recall=0.0769 val_fe_recall=0.8537 val_joint_bal_acc=0.5606 val_joint_macro_f1=0.5324 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=41 train_loss=0.5328 lr=4.75232e-05 train_metal_acc=0.8679 val_loss=1.2777 val_metal_acc=0.6538 val_metal_min_recall=0.1538 val_fe_recall=0.8049 val_joint_bal_acc=0.5971 val_joint_macro_f1=0.5798 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=42 train_loss=0.5076 lr=4.75232e-05 train_metal_acc=0.8323 val_loss=1.3269 val_metal_acc=0.6442 val_metal_min_recall=0.0769 val_fe_recall=0.8780 val_joint_bal_acc=0.5927 val_joint_macro_f1=0.5704 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=43 train_loss=0.4909 lr=4.75232e-05 train_metal_acc=0.8442 val_loss=1.2159 val_metal_acc=0.6779 val_metal_min_recall=0.0769 val_fe_recall=0.9268 val_joint_bal_acc=0.5872 val_joint_macro_f1=0.5886 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=44 train_loss=0.4958 lr=4.75232e-05 train_metal_acc=0.8772 val_loss=1.2558 val_metal_acc=0.6731 val_metal_min_recall=0.0769 val_fe_recall=0.8537 val_joint_bal_acc=0.5991 val_joint_macro_f1=0.5796 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=45 train_loss=0.4640 lr=4.75232e-05 train_metal_acc=0.8831 val_loss=1.1615 val_metal_acc=0.6779 val_metal_min_recall=0.1538 val_fe_recall=0.8780 val_joint_bal_acc=0.6111 val_joint_macro_f1=0.5984 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=46 train_loss=0.4554 lr=4.75232e-05 train_metal_acc=0.8865 val_loss=1.1925 val_metal_acc=0.6587 val_metal_min_recall=0.1538 val_fe_recall=0.8780 val_joint_bal_acc=0.5835 val_joint_macro_f1=0.5769 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=47 train_loss=0.4362 lr=4.75232e-05 train_metal_acc=0.8984 val_loss=1.2484 val_metal_acc=0.6587 val_metal_min_recall=0.1538 val_fe_recall=0.8780 val_joint_bal_acc=0.5899 val_joint_macro_f1=0.5765 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=48 train_loss=0.4404 lr=4.75232e-05 train_metal_acc=0.9018 val_loss=1.1778 val_metal_acc=0.6587 val_metal_min_recall=0.1538 val_fe_recall=0.8780 val_joint_bal_acc=0.5899 val_joint_macro_f1=0.5828 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=49 train_loss=0.3962 lr=4.75232e-05 train_metal_acc=0.9077 val_loss=1.2342 val_metal_acc=0.6635 val_metal_min_recall=0.0769 val_fe_recall=0.8780 val_joint_bal_acc=0.5901 val_joint_macro_f1=0.5783 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=50 train_loss=0.3790 lr=4.75232e-05 train_metal_acc=0.8933 val_loss=1.2173 val_metal_acc=0.6683 val_metal_min_recall=0.1538 val_fe_recall=0.8537 val_joint_bal_acc=0.6085 val_joint_macro_f1=0.5877 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_c13ce556/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_c13ce556/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_c13ce556/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_c13ce556/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_c13ce556/run_metadata.json
Completed: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_c13ce556
================================================================================
[#005 | 5/5] deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_dcfd4dee
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat --run-name deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_dcfd4dee --model-architecture only_gvp --epochs 50 --batch-size 8 --learning-rate 4.752317377508605e-05 --weight-decay 0.0 --seed 44 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 128 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_sqrt_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 3 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges
stdout log: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/_execution_logs/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_dcfd4dee.stdout.log
stderr log: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/_execution_logs/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_dcfd4dee.stderr.log
================================================================================

=== Split diagnostics (passive; training behavior unchanged) ===
task=metal split_by=pdbid val_fraction=0.15 n_folds=None fold_index=None
pockets: train=1181 validation=208
groups by pdbid: train=1152 validation=109
train/validation overlap counts: pdbid=0, pdbid_chain=0, structure_id=0, pocket_id=0
train metal distribution: Mn=551, Cu=70, Zn=173, Fe=250, Co=73, Ni=64
validation metal distribution: Mn=97, Cu=15, Zn=34, Fe=42, Co=13, Ni=7
missing train metal classes: none
missing validation metal classes: none
train EC distribution: 1=489, 2=200, 3=339, 4=48, 5=73, 6=15, 7=1
validation EC distribution: 1=70, 2=15, 3=53, 4=6, 5=60, 6=2, 7=0
missing train EC classes: none
missing validation EC classes: 7
===============================================================

epoch=1 train_loss=1.6956 lr=4.75232e-05 train_metal_acc=0.4428 val_loss=1.6164 val_metal_acc=0.4952 val_metal_min_recall=0.0000 val_fe_recall=0.4048 val_joint_bal_acc=0.2152 val_joint_macro_f1=0.1757 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.6528 lr=4.75232e-05 train_metal_acc=0.4776 val_loss=1.5556 val_metal_acc=0.4904 val_metal_min_recall=0.0000 val_fe_recall=0.1190 val_joint_bal_acc=0.1865 val_joint_macro_f1=0.1429 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.5932 lr=4.75232e-05 train_metal_acc=0.4285 val_loss=1.5542 val_metal_acc=0.5096 val_metal_min_recall=0.0000 val_fe_recall=0.4762 val_joint_bal_acc=0.3117 val_joint_macro_f1=0.2504 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.4345 lr=4.75232e-05 train_metal_acc=0.6401 val_loss=1.2872 val_metal_acc=0.6538 val_metal_min_recall=0.0000 val_fe_recall=0.6905 val_joint_bal_acc=0.3933 val_joint_macro_f1=0.3802 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.3326 lr=4.75232e-05 train_metal_acc=0.6122 val_loss=1.2315 val_metal_acc=0.6635 val_metal_min_recall=0.0000 val_fe_recall=0.9286 val_joint_bal_acc=0.4096 val_joint_macro_f1=0.3696 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.2976 lr=4.75232e-05 train_metal_acc=0.6537 val_loss=1.2189 val_metal_acc=0.6875 val_metal_min_recall=0.0000 val_fe_recall=0.5952 val_joint_bal_acc=0.4435 val_joint_macro_f1=0.4315 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=1.2469 lr=4.75232e-05 train_metal_acc=0.6622 val_loss=1.1772 val_metal_acc=0.6779 val_metal_min_recall=0.0000 val_fe_recall=0.7143 val_joint_bal_acc=0.5298 val_joint_macro_f1=0.4779 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=1.2021 lr=4.75232e-05 train_metal_acc=0.6808 val_loss=1.1438 val_metal_acc=0.6490 val_metal_min_recall=0.0000 val_fe_recall=0.6905 val_joint_bal_acc=0.4607 val_joint_macro_f1=0.4258 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=1.1420 lr=4.75232e-05 train_metal_acc=0.5817 val_loss=1.3726 val_metal_acc=0.4183 val_metal_min_recall=0.0000 val_fe_recall=0.4048 val_joint_bal_acc=0.3590 val_joint_macro_f1=0.3190 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=1.1473 lr=4.75232e-05 train_metal_acc=0.7019 val_loss=1.2051 val_metal_acc=0.6250 val_metal_min_recall=0.0000 val_fe_recall=0.7143 val_joint_bal_acc=0.5427 val_joint_macro_f1=0.5221 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=1.0921 lr=4.75232e-05 train_metal_acc=0.6782 val_loss=1.2754 val_metal_acc=0.5192 val_metal_min_recall=0.0000 val_fe_recall=0.7381 val_joint_bal_acc=0.4379 val_joint_macro_f1=0.3781 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=1.0688 lr=4.75232e-05 train_metal_acc=0.7231 val_loss=1.1859 val_metal_acc=0.5913 val_metal_min_recall=0.0000 val_fe_recall=0.7381 val_joint_bal_acc=0.4950 val_joint_macro_f1=0.4760 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=1.0529 lr=4.75232e-05 train_metal_acc=0.7257 val_loss=1.1728 val_metal_acc=0.6106 val_metal_min_recall=0.0000 val_fe_recall=0.6905 val_joint_bal_acc=0.5447 val_joint_macro_f1=0.5116 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=1.0312 lr=4.75232e-05 train_metal_acc=0.7155 val_loss=1.0987 val_metal_acc=0.6346 val_metal_min_recall=0.0769 val_fe_recall=0.5952 val_joint_bal_acc=0.4891 val_joint_macro_f1=0.4993 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=1.0025 lr=4.75232e-05 train_metal_acc=0.7231 val_loss=1.2232 val_metal_acc=0.5721 val_metal_min_recall=0.0000 val_fe_recall=0.6667 val_joint_bal_acc=0.5383 val_joint_macro_f1=0.4966 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.9809 lr=4.75232e-05 train_metal_acc=0.7257 val_loss=1.1645 val_metal_acc=0.6202 val_metal_min_recall=0.0000 val_fe_recall=0.5476 val_joint_bal_acc=0.5219 val_joint_macro_f1=0.4812 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.9624 lr=4.75232e-05 train_metal_acc=0.7273 val_loss=1.1305 val_metal_acc=0.6010 val_metal_min_recall=0.0769 val_fe_recall=0.4762 val_joint_bal_acc=0.4752 val_joint_macro_f1=0.4625 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.9413 lr=4.75232e-05 train_metal_acc=0.7163 val_loss=1.1103 val_metal_acc=0.6394 val_metal_min_recall=0.0000 val_fe_recall=0.8095 val_joint_bal_acc=0.4556 val_joint_macro_f1=0.4444 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.9196 lr=4.75232e-05 train_metal_acc=0.7544 val_loss=1.0815 val_metal_acc=0.6394 val_metal_min_recall=0.0769 val_fe_recall=0.5952 val_joint_bal_acc=0.5476 val_joint_macro_f1=0.5233 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.8792 lr=4.75232e-05 train_metal_acc=0.7671 val_loss=1.1728 val_metal_acc=0.6058 val_metal_min_recall=0.0769 val_fe_recall=0.4048 val_joint_bal_acc=0.5398 val_joint_macro_f1=0.5294 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.8651 lr=4.75232e-05 train_metal_acc=0.7773 val_loss=1.1730 val_metal_acc=0.5865 val_metal_min_recall=0.0769 val_fe_recall=0.5952 val_joint_bal_acc=0.5510 val_joint_macro_f1=0.5128 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.8492 lr=4.75232e-05 train_metal_acc=0.7722 val_loss=1.1587 val_metal_acc=0.6250 val_metal_min_recall=0.0769 val_fe_recall=0.5476 val_joint_bal_acc=0.5570 val_joint_macro_f1=0.5188 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.8579 lr=4.75232e-05 train_metal_acc=0.7883 val_loss=1.1489 val_metal_acc=0.6058 val_metal_min_recall=0.0769 val_fe_recall=0.6905 val_joint_bal_acc=0.5541 val_joint_macro_f1=0.5322 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.8249 lr=4.75232e-05 train_metal_acc=0.7782 val_loss=1.2372 val_metal_acc=0.5577 val_metal_min_recall=0.0769 val_fe_recall=0.5476 val_joint_bal_acc=0.5202 val_joint_macro_f1=0.4817 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.8104 lr=4.75232e-05 train_metal_acc=0.7714 val_loss=1.2587 val_metal_acc=0.5769 val_metal_min_recall=0.0769 val_fe_recall=0.5952 val_joint_bal_acc=0.6170 val_joint_macro_f1=0.5118 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.7821 lr=4.75232e-05 train_metal_acc=0.7722 val_loss=1.1550 val_metal_acc=0.6202 val_metal_min_recall=0.0000 val_fe_recall=0.7381 val_joint_bal_acc=0.5527 val_joint_macro_f1=0.5279 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.7733 lr=4.75232e-05 train_metal_acc=0.8112 val_loss=1.2029 val_metal_acc=0.5721 val_metal_min_recall=0.0769 val_fe_recall=0.7143 val_joint_bal_acc=0.5475 val_joint_macro_f1=0.5250 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.7484 lr=4.75232e-05 train_metal_acc=0.7934 val_loss=1.1002 val_metal_acc=0.6250 val_metal_min_recall=0.0769 val_fe_recall=0.6429 val_joint_bal_acc=0.5342 val_joint_macro_f1=0.5161 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.7469 lr=4.75232e-05 train_metal_acc=0.7942 val_loss=1.1918 val_metal_acc=0.6010 val_metal_min_recall=0.0769 val_fe_recall=0.7381 val_joint_bal_acc=0.5758 val_joint_macro_f1=0.5284 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.7254 lr=4.75232e-05 train_metal_acc=0.8188 val_loss=1.2280 val_metal_acc=0.5721 val_metal_min_recall=0.0769 val_fe_recall=0.6190 val_joint_bal_acc=0.5353 val_joint_macro_f1=0.4962 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.7114 lr=4.75232e-05 train_metal_acc=0.8239 val_loss=1.2020 val_metal_acc=0.5865 val_metal_min_recall=0.0769 val_fe_recall=0.5714 val_joint_bal_acc=0.5551 val_joint_macro_f1=0.5144 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.6799 lr=4.75232e-05 train_metal_acc=0.8120 val_loss=1.1882 val_metal_acc=0.6202 val_metal_min_recall=0.0769 val_fe_recall=0.7143 val_joint_bal_acc=0.5615 val_joint_macro_f1=0.5268 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.6661 lr=4.75232e-05 train_metal_acc=0.8307 val_loss=1.1941 val_metal_acc=0.6106 val_metal_min_recall=0.0769 val_fe_recall=0.4762 val_joint_bal_acc=0.5260 val_joint_macro_f1=0.4933 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.6457 lr=4.75232e-05 train_metal_acc=0.8577 val_loss=1.2385 val_metal_acc=0.6058 val_metal_min_recall=0.0769 val_fe_recall=0.6429 val_joint_bal_acc=0.5749 val_joint_macro_f1=0.5291 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.6338 lr=4.75232e-05 train_metal_acc=0.8518 val_loss=1.2250 val_metal_acc=0.6106 val_metal_min_recall=0.0769 val_fe_recall=0.6905 val_joint_bal_acc=0.5590 val_joint_macro_f1=0.5618 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.6134 lr=4.75232e-05 train_metal_acc=0.8535 val_loss=1.3744 val_metal_acc=0.5337 val_metal_min_recall=0.0769 val_fe_recall=0.5714 val_joint_bal_acc=0.5234 val_joint_macro_f1=0.4972 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.6114 lr=4.75232e-05 train_metal_acc=0.8196 val_loss=1.1895 val_metal_acc=0.6346 val_metal_min_recall=0.0769 val_fe_recall=0.5238 val_joint_bal_acc=0.5518 val_joint_macro_f1=0.5335 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.5789 lr=4.75232e-05 train_metal_acc=0.8704 val_loss=1.2744 val_metal_acc=0.6058 val_metal_min_recall=0.0769 val_fe_recall=0.6190 val_joint_bal_acc=0.5601 val_joint_macro_f1=0.5350 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.5567 lr=4.75232e-05 train_metal_acc=0.8662 val_loss=1.1895 val_metal_acc=0.6202 val_metal_min_recall=0.0769 val_fe_recall=0.7143 val_joint_bal_acc=0.5679 val_joint_macro_f1=0.5558 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.5779 lr=4.75232e-05 train_metal_acc=0.8620 val_loss=1.2487 val_metal_acc=0.6154 val_metal_min_recall=0.0769 val_fe_recall=0.5714 val_joint_bal_acc=0.5811 val_joint_macro_f1=0.5468 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=41 train_loss=0.5827 lr=4.75232e-05 train_metal_acc=0.8789 val_loss=1.2457 val_metal_acc=0.6058 val_metal_min_recall=0.0769 val_fe_recall=0.5238 val_joint_bal_acc=0.5479 val_joint_macro_f1=0.5168 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=42 train_loss=0.5502 lr=4.75232e-05 train_metal_acc=0.8264 val_loss=1.4526 val_metal_acc=0.5385 val_metal_min_recall=0.0769 val_fe_recall=0.4762 val_joint_bal_acc=0.5289 val_joint_macro_f1=0.4866 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=43 train_loss=0.5157 lr=4.75232e-05 train_metal_acc=0.8535 val_loss=1.3757 val_metal_acc=0.5817 val_metal_min_recall=0.0769 val_fe_recall=0.6429 val_joint_bal_acc=0.5474 val_joint_macro_f1=0.5195 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=44 train_loss=0.4810 lr=4.75232e-05 train_metal_acc=0.8933 val_loss=1.2823 val_metal_acc=0.6250 val_metal_min_recall=0.0769 val_fe_recall=0.6905 val_joint_bal_acc=0.5894 val_joint_macro_f1=0.5637 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=45 train_loss=0.4930 lr=4.75232e-05 train_metal_acc=0.8637 val_loss=1.3387 val_metal_acc=0.6058 val_metal_min_recall=0.0769 val_fe_recall=0.5952 val_joint_bal_acc=0.5799 val_joint_macro_f1=0.5355 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=46 train_loss=0.4699 lr=4.75232e-05 train_metal_acc=0.8874 val_loss=1.3824 val_metal_acc=0.6010 val_metal_min_recall=0.0769 val_fe_recall=0.6429 val_joint_bal_acc=0.5700 val_joint_macro_f1=0.5287 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=47 train_loss=0.4333 lr=4.75232e-05 train_metal_acc=0.8984 val_loss=1.4889 val_metal_acc=0.5673 val_metal_min_recall=0.0769 val_fe_recall=0.5714 val_joint_bal_acc=0.5227 val_joint_macro_f1=0.4894 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=48 train_loss=0.4234 lr=4.75232e-05 train_metal_acc=0.9119 val_loss=1.4099 val_metal_acc=0.5865 val_metal_min_recall=0.0769 val_fe_recall=0.6190 val_joint_bal_acc=0.5562 val_joint_macro_f1=0.4888 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=49 train_loss=0.4104 lr=4.75232e-05 train_metal_acc=0.8764 val_loss=1.3463 val_metal_acc=0.6154 val_metal_min_recall=0.0769 val_fe_recall=0.7619 val_joint_bal_acc=0.5800 val_joint_macro_f1=0.5784 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=50 train_loss=0.3974 lr=4.75232e-05 train_metal_acc=0.9136 val_loss=1.3970 val_metal_acc=0.6202 val_metal_min_recall=0.0769 val_fe_recall=0.7143 val_joint_bal_acc=0.5836 val_joint_macro_f1=0.5238 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_dcfd4dee/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_dcfd4dee/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_dcfd4dee/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_dcfd4dee/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_dcfd4dee/run_metadata.json
Completed: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_dcfd4dee
Completed run directories: ['/content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_e9575b8a', '/content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_0956e7d3', '/content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_6eab65a2', '/content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_c13ce556', '/content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_dcfd4dee']
Failed run directories: []
Execution records JSON: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp3_50epoch_seedrepeat/deepmzyme_nonoverlap_model_comparison_execution_records.json


#-----------------------



# Detailed results: Run 2: Trial7 gvp_layers=4


Configured output locations:
  Runs root:       /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat  [exists]
  Summary CSV:     /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat/deepmzyme_nonoverlap_model_comparison.csv  [exists]
  Summary figure:  /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat/deepmzyme_nonoverlap_model_comparison.png  [exists]

Completed run directories found: 5
choice_index	run_name	task	model	fusion	seed	learning_rate	weight_decay	selection_metric	metric_direction	best_validation_value	test_metric	test_metric_value	selected_epoch	test_report_saved	run_dir
0	1	deepmzyme_nonoverlap_baseline_batchmetal_only_...	metal	GVP only	late_fusion	44	0.000065	0.001	val_metal_balanced_acc	higher_is_better	0.655907	test_metal_balanced_acc	None	33	False	/content/deepmzyme_outputs/runs/metal_only_gvp...
1	2	deepmzyme_nonoverlap_baseline_batchmetal_only_...	metal	GVP only	late_fusion	2026	0.000065	0.001	val_metal_balanced_acc	higher_is_better	0.647724	test_metal_balanced_acc	None	42	False	/content/deepmzyme_outputs/runs/metal_only_gvp...
2	3	deepmzyme_nonoverlap_baseline_batchmetal_only_...	metal	GVP only	late_fusion	43	0.000065	0.001	val_metal_balanced_acc	higher_is_better	0.607691	test_metal_balanced_acc	None	50	False	/content/deepmzyme_outputs/runs/metal_only_gvp...
3	4	deepmzyme_nonoverlap_baseline_batchmetal_only_...	metal	GVP only	late_fusion	42	0.000065	0.001	val_metal_balanced_acc	higher_is_better	0.583875	test_metal_balanced_acc	None	28	False	/content/deepmzyme_outputs/runs/metal_only_gvp...
4	5	deepmzyme_nonoverlap_baseline_batchmetal_only_...	metal	GVP only	late_fusion	123	0.000065	0.001	val_metal_balanced_acc	higher_is_better	0.558363	test_metal_balanced_acc	None	24	False	/content/deepmzyme_outputs/runs/metal_only_gvp...



Selected final run: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_ring_4db78203
  Selection mode: auto_best_validation
  Task:         metal
  Architecture: only_gvp
  Fusion:       late_fusion
  Seed:         44
  Best val val_metal_balanced_acc: 0.6559  (epoch 33)
  Split:        train_and_test_sets_structures_non_overlapped_pinmymetal
Configured output locations:
  Runs root:       /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat  [exists]
  Summary CSV:     /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat/deepmzyme_nonoverlap_model_comparison.csv  [exists]
  Summary figure:  /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat/deepmzyme_nonoverlap_model_comparison.png  [exists]
  Selected run:    /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_ring_4db78203  [exists]
  Run config:      /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_ring_4db78203/run_config.json  [exists]
  Run metadata:    /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_ring_4db78203/run_metadata.json  [exists]
  Test report:     /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_ring_4db78203/test_report.json  [not created yet]
  Selection JSON: /content/deepmzyme_outputs/runs/deepmzyme_final_selected_run.json

No test_report.json found for the selected run.
Use the optional final held-out test evaluation cell next; its default mode evaluates the selected saved checkpoint without retraining.
Keep choosing models by validation metrics; use held-out test metrics only for the selected final run.

-------------
Summary scanning scope: current RUN_BATCH_ID folder
RUN_BATCH_ID: metal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat
Runs directory scanned: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat
/usr/bin/python3 /content/DeepMzyme/src/report_runs.py --runs-dir /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat --out-csv /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat/deepmzyme_nonoverlap_model_comparison_completed_only.csv --out-figure /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat/deepmzyme_nonoverlap_model_comparison.png
Completed-run summary CSV: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat/deepmzyme_nonoverlap_model_comparison_completed_only.csv
Summary source mode: planned table from current notebook state
Summary source scope: current planned rows plus completed runs under the scanned directory.
Comparison CSV: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat/deepmzyme_nonoverlap_model_comparison.csv
rank	source_mode	config_source	result_stage	run_name	status	error_message	model_preset	model_display	model_architecture	...	missing_train_metal_classes	missing_val_metal_classes	missing_train_ec_classes	missing_val_ec_classes	selected_best_validation_metric_value	held_out_test_metric_name	held_out_test_metric_value	run_dir	stdout_log_path	stderr_log_path
0	1	planned table	current notebook config	validation-only	deepmzyme_nonoverlap_baseline_batchmetal_only_...	completed		Only-GVP	Only-GVP (structure only)	only_gvp	...	NaN	NaN	NaN	7	0.655907	test_metal_balanced_acc	NaN	/content/deepmzyme_outputs/runs/metal_only_gvp...	/content/deepmzyme_outputs/runs/metal_only_gvp...	/content/deepmzyme_outputs/runs/metal_only_gvp...
1	2	planned table	current notebook config	validation-only	deepmzyme_nonoverlap_baseline_batchmetal_only_...	completed		Only-GVP	Only-GVP (structure only)	only_gvp	...	NaN	NaN	NaN	7	0.647724	test_metal_balanced_acc	NaN	/content/deepmzyme_outputs/runs/metal_only_gvp...	/content/deepmzyme_outputs/runs/metal_only_gvp...	/content/deepmzyme_outputs/runs/metal_only_gvp...
2	3	planned table	current notebook config	validation-only	deepmzyme_nonoverlap_baseline_batchmetal_only_...	completed		Only-GVP	Only-GVP (structure only)	only_gvp	...	NaN	NaN	NaN	7	0.607691	test_metal_balanced_acc	NaN	/content/deepmzyme_outputs/runs/metal_only_gvp...	/content/deepmzyme_outputs/runs/metal_only_gvp...	/content/deepmzyme_outputs/runs/metal_only_gvp...
3	4	planned table	current notebook config	validation-only	deepmzyme_nonoverlap_baseline_batchmetal_only_...	completed		Only-GVP	Only-GVP (structure only)	only_gvp	...	NaN	NaN	NaN	7	0.583875	test_metal_balanced_acc	NaN	/content/deepmzyme_outputs/runs/metal_only_gvp...	/content/deepmzyme_outputs/runs/metal_only_gvp...	/content/deepmzyme_outputs/runs/metal_only_gvp...
4	5	planned table	current notebook config	validation-only	deepmzyme_nonoverlap_baseline_batchmetal_only_...	completed		Only-GVP	Only-GVP (structure only)	only_gvp	...	NaN	NaN	NaN	7	0.558363	test_metal_balanced_acc	NaN	/content/deepmzyme_outputs/runs/metal_only_gvp...	/content/deepmzyme_outputs/runs/metal_only_gvp...	/content/deepmzyme_outputs/runs/metal_only_gvp...
5 rows × 63 columns




Ranked table sorted by validation selection metric:
#1: deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_ring_4db78203 | stage=validation-only | class_weight=inverse_sqrt_frequency | val_metal_balanced_acc=0.6559072690667597 | status=completed
#2: deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_ring_eaad61f7 | stage=validation-only | class_weight=inverse_sqrt_frequency | val_metal_balanced_acc=0.64772364969639 | status=completed
#3: deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_ring_4ca6f423 | stage=validation-only | class_weight=inverse_sqrt_frequency | val_metal_balanced_acc=0.6076905251955911 | status=completed
#4: deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_ring_0479b0b1 | stage=validation-only | class_weight=inverse_sqrt_frequency | val_metal_balanced_acc=0.583875410099637 | status=completed
#5: deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_ring_e5e50d22 | stage=validation-only | class_weight=inverse_sqrt_frequency | val_metal_balanced_acc=0.558362528037978 | status=completed

Best overall configuration: deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_ring_4db78203
{
  "run_name": "deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_ring_4db78203",
  "result_stage": "validation-only",
  "model_preset": "Only-GVP",
  "model_architecture": "only_gvp",
  "fusion_mode": "none",
  "metal_class_weight_mode": "inverse_sqrt_frequency",
  "balance_metal_site_symbols": false,
  "selection_metric": "val_metal_balanced_acc",
  "selected_best_validation_metric_value": 0.6559072690667597,
  "run_dir": "/content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_ring_4db78203"
}

Best configuration per model preset/mode:
Only-GVP: deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_ring_4db78203 | class_weight=inverse_sqrt_frequency | val_metal_balanced_acc=0.6559072690667597

Best Only-GVP configuration: deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_ring_4db78203
Best ESM-based configuration: not available
Best RING vs non-RING comparison: not available unless both modes have completed numeric validation metrics.

Automatic interpretation
Best validation config: deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_ring_4db78203 with val_metal_balanced_acc = 0.6559072690667597
Top fusion mode: none
Best learning-rate region: middle around 1e-4 (lr=6.464669746492395e-05)
Held-out test results present: False
Recommended next step: select/retrain the final validation-best configuration, then run held-out test evaluation once
Drive copy skipped. Outputs remain under: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial7_gvp4_50epoch_seedrepeat




#-----------------------------------------------
## Detailed results: Run 3: Trial12 gvp_layers=2

Summary scanning scope: current RUN_BATCH_ID folder
RUN_BATCH_ID: metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat
Runs directory scanned: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat
/usr/bin/python3 /content/DeepMzyme/src/report_runs.py --runs-dir /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat --out-csv /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_model_comparison_completed_only.csv --out-figure /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_model_comparison.png
Completed-run summary CSV: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_model_comparison_completed_only.csv
Summary source mode: planned table from current notebook state
Summary source scope: current planned rows plus completed runs under the scanned directory.
Comparison CSV: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_model_comparison.csv
rank	source_mode	config_source	result_stage	run_name	status	error_message	model_preset	model_display	model_architecture	...	missing_train_metal_classes	missing_val_metal_classes	missing_train_ec_classes	missing_val_ec_classes	selected_best_validation_metric_value	held_out_test_metric_name	held_out_test_metric_value	run_dir	stdout_log_path	stderr_log_path
0	1	planned table	current notebook config	validation-only	deepmzyme_nonoverlap_baseline_batchmetal_only_...	completed		Only-GVP	Only-GVP (structure only)	only_gvp	...	NaN	NaN	NaN	7	0.624332	test_metal_balanced_acc	NaN	/content/deepmzyme_outputs/runs/metal_only_gvp...	/content/deepmzyme_outputs/runs/metal_only_gvp...	/content/deepmzyme_outputs/runs/metal_only_gvp...
1	2	planned table	current notebook config	validation-only	deepmzyme_nonoverlap_baseline_batchmetal_only_...	completed		Only-GVP	Only-GVP (structure only)	only_gvp	...	NaN	NaN	NaN	7	0.613586	test_metal_balanced_acc	NaN	/content/deepmzyme_outputs/runs/metal_only_gvp...	/content/deepmzyme_outputs/runs/metal_only_gvp...	/content/deepmzyme_outputs/runs/metal_only_gvp...
2	3	planned table	current notebook config	validation-only	deepmzyme_nonoverlap_baseline_batchmetal_only_...	completed		Only-GVP	Only-GVP (structure only)	only_gvp	...	NaN	NaN	NaN	7	0.596988	test_metal_balanced_acc	NaN	/content/deepmzyme_outputs/runs/metal_only_gvp...	/content/deepmzyme_outputs/runs/metal_only_gvp...	/content/deepmzyme_outputs/runs/metal_only_gvp...
3	4	planned table	current notebook config	validation-only	deepmzyme_nonoverlap_baseline_batchmetal_only_...	completed		Only-GVP	Only-GVP (structure only)	only_gvp	...	NaN	NaN	NaN	7	0.579442	test_metal_balanced_acc	NaN	/content/deepmzyme_outputs/runs/metal_only_gvp...	/content/deepmzyme_outputs/runs/metal_only_gvp...	/content/deepmzyme_outputs/runs/metal_only_gvp...
4	5	planned table	current notebook config	validation-only	deepmzyme_nonoverlap_baseline_batchmetal_only_...	completed		Only-GVP	Only-GVP (structure only)	only_gvp	...	NaN	NaN	NaN	7	0.578510	test_metal_balanced_acc	NaN	/content/deepmzyme_outputs/runs/metal_only_gvp...	/content/deepmzyme_outputs/runs/metal_only_gvp...	/content/deepmzyme_outputs/runs/metal_only_gvp...
5 rows × 63 columns




Ranked table sorted by validation selection metric:
#1: deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_14db2a0a | stage=validation-only | class_weight=inverse_sqrt_frequency | val_metal_balanced_acc=0.6243321546406195 | status=completed
#2: deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_edcdcf3a | stage=validation-only | class_weight=inverse_sqrt_frequency | val_metal_balanced_acc=0.6135864554521834 | status=completed
#3: deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_60d10046 | stage=validation-only | class_weight=inverse_sqrt_frequency | val_metal_balanced_acc=0.5969884067822212 | status=completed
#4: deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_1a1fd84a | stage=validation-only | class_weight=inverse_sqrt_frequency | val_metal_balanced_acc=0.5794419028063957 | status=completed
#5: deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_2c77cc48 | stage=validation-only | class_weight=inverse_sqrt_frequency | val_metal_balanced_acc=0.5785100852755493 | status=completed

Best overall configuration: deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_14db2a0a
{
  "run_name": "deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_14db2a0a",
  "result_stage": "validation-only",
  "model_preset": "Only-GVP",
  "model_architecture": "only_gvp",
  "fusion_mode": "none",
  "metal_class_weight_mode": "inverse_sqrt_frequency",
  "balance_metal_site_symbols": false,
  "selection_metric": "val_metal_balanced_acc",
  "selected_best_validation_metric_value": 0.6243321546406195,
  "run_dir": "/content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_14db2a0a"
}

Best configuration per model preset/mode:
Only-GVP: deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_14db2a0a | class_weight=inverse_sqrt_frequency | val_metal_balanced_acc=0.6243321546406195

Best Only-GVP configuration: deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_14db2a0a
Best ESM-based configuration: not available
Best RING vs non-RING comparison: not available unless both modes have completed numeric validation metrics.

Automatic interpretation
Best validation config: deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_14db2a0a with val_metal_balanced_acc = 0.6243321546406195
Top fusion mode: none
Best learning-rate region: middle around 1e-4 (lr=4.752317377508605e-05)
Held-out test results present: False
Recommended next step: select/retrain the final validation-best configuration, then run held-out test evaluation once
Drive copy skipped. Outputs remain under: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat

-------------------------------------------------------------------------------


Runnable planned configurations: 5
================================================================================
[#001 | 1/5] deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_2c77cc48
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat --run-name deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_2c77cc48 --model-architecture only_gvp --epochs 50 --batch-size 8 --learning-rate 4.752317377508605e-05 --weight-decay 0.0 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 128 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_sqrt_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges
stdout log: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/_execution_logs/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_2c77cc48.stdout.log
stderr log: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/_execution_logs/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_2c77cc48.stderr.log
================================================================================

=== Split diagnostics (passive; training behavior unchanged) ===
task=metal split_by=pdbid val_fraction=0.15 n_folds=None fold_index=None
pockets: train=1181 validation=208
groups by pdbid: train=1151 validation=110
train/validation overlap counts: pdbid=0, pdbid_chain=0, structure_id=0, pocket_id=0
train metal distribution: Mn=551, Cu=70, Zn=175, Fe=248, Co=73, Ni=64
validation metal distribution: Mn=97, Cu=15, Zn=32, Fe=44, Co=13, Ni=7
missing train metal classes: none
missing validation metal classes: none
train EC distribution: 1=491, 2=197, 3=344, 4=46, 5=69, 6=15, 7=1
validation EC distribution: 1=68, 2=18, 3=48, 4=8, 5=64, 6=2, 7=0
missing train EC classes: none
missing validation EC classes: 7
===============================================================

epoch=1 train_loss=1.7002 lr=4.75232e-05 train_metal_acc=0.4666 val_loss=1.6029 val_metal_acc=0.4663 val_metal_min_recall=0.0000 val_fe_recall=0.0000 val_joint_bal_acc=0.1667 val_joint_macro_f1=0.1060 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.6590 lr=4.75232e-05 train_metal_acc=0.4953 val_loss=1.5593 val_metal_acc=0.5000 val_metal_min_recall=0.0000 val_fe_recall=0.1591 val_joint_bal_acc=0.1932 val_joint_macro_f1=0.1503 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.5896 lr=4.75232e-05 train_metal_acc=0.5478 val_loss=1.4072 val_metal_acc=0.6202 val_metal_min_recall=0.0000 val_fe_recall=0.7045 val_joint_bal_acc=0.2893 val_joint_macro_f1=0.2438 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.4719 lr=4.75232e-05 train_metal_acc=0.5995 val_loss=1.3137 val_metal_acc=0.6779 val_metal_min_recall=0.0000 val_fe_recall=0.4773 val_joint_bal_acc=0.4521 val_joint_macro_f1=0.4258 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.3828 lr=4.75232e-05 train_metal_acc=0.6452 val_loss=1.1885 val_metal_acc=0.7019 val_metal_min_recall=0.0000 val_fe_recall=0.6818 val_joint_bal_acc=0.4536 val_joint_macro_f1=0.4447 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.2948 lr=4.75232e-05 train_metal_acc=0.6503 val_loss=1.1283 val_metal_acc=0.7067 val_metal_min_recall=0.0000 val_fe_recall=0.7727 val_joint_bal_acc=0.4649 val_joint_macro_f1=0.3972 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=1.2548 lr=4.75232e-05 train_metal_acc=0.6622 val_loss=1.1123 val_metal_acc=0.7260 val_metal_min_recall=0.0000 val_fe_recall=0.8182 val_joint_bal_acc=0.5015 val_joint_macro_f1=0.4892 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=1.1958 lr=4.75232e-05 train_metal_acc=0.6986 val_loss=1.1035 val_metal_acc=0.7115 val_metal_min_recall=0.0000 val_fe_recall=0.7500 val_joint_bal_acc=0.5216 val_joint_macro_f1=0.5103 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=1.1740 lr=4.75232e-05 train_metal_acc=0.7019 val_loss=1.1649 val_metal_acc=0.6442 val_metal_min_recall=0.0000 val_fe_recall=0.5909 val_joint_bal_acc=0.4900 val_joint_macro_f1=0.4690 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=1.1436 lr=4.75232e-05 train_metal_acc=0.7113 val_loss=1.2162 val_metal_acc=0.6346 val_metal_min_recall=0.0769 val_fe_recall=0.5227 val_joint_bal_acc=0.5111 val_joint_macro_f1=0.5003 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=1.1133 lr=4.75232e-05 train_metal_acc=0.7138 val_loss=1.1282 val_metal_acc=0.6538 val_metal_min_recall=0.0000 val_fe_recall=0.7273 val_joint_bal_acc=0.5058 val_joint_macro_f1=0.4686 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=1.0834 lr=4.75232e-05 train_metal_acc=0.7290 val_loss=1.1018 val_metal_acc=0.6923 val_metal_min_recall=0.0000 val_fe_recall=0.7727 val_joint_bal_acc=0.5342 val_joint_macro_f1=0.5062 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=1.0434 lr=4.75232e-05 train_metal_acc=0.7036 val_loss=1.1929 val_metal_acc=0.6394 val_metal_min_recall=0.0000 val_fe_recall=0.7955 val_joint_bal_acc=0.5255 val_joint_macro_f1=0.4917 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=1.0445 lr=4.75232e-05 train_metal_acc=0.6943 val_loss=1.1964 val_metal_acc=0.6154 val_metal_min_recall=0.0000 val_fe_recall=0.5682 val_joint_bal_acc=0.5033 val_joint_macro_f1=0.4388 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=1.0066 lr=4.75232e-05 train_metal_acc=0.7434 val_loss=1.1399 val_metal_acc=0.6442 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.5046 val_joint_macro_f1=0.4689 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.9777 lr=4.75232e-05 train_metal_acc=0.7214 val_loss=1.2228 val_metal_acc=0.6154 val_metal_min_recall=0.0769 val_fe_recall=0.6136 val_joint_bal_acc=0.5534 val_joint_macro_f1=0.5151 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.9822 lr=4.75232e-05 train_metal_acc=0.7561 val_loss=1.1425 val_metal_acc=0.6635 val_metal_min_recall=0.0769 val_fe_recall=0.7727 val_joint_bal_acc=0.5291 val_joint_macro_f1=0.5265 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.9558 lr=4.75232e-05 train_metal_acc=0.7316 val_loss=1.0604 val_metal_acc=0.6683 val_metal_min_recall=0.0000 val_fe_recall=0.7955 val_joint_bal_acc=0.4998 val_joint_macro_f1=0.4799 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.9251 lr=4.75232e-05 train_metal_acc=0.7375 val_loss=1.1151 val_metal_acc=0.6635 val_metal_min_recall=0.0000 val_fe_recall=0.7727 val_joint_bal_acc=0.5565 val_joint_macro_f1=0.4950 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.8976 lr=4.75232e-05 train_metal_acc=0.7722 val_loss=1.0967 val_metal_acc=0.6779 val_metal_min_recall=0.0769 val_fe_recall=0.6591 val_joint_bal_acc=0.5670 val_joint_macro_f1=0.5509 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.8729 lr=4.75232e-05 train_metal_acc=0.7265 val_loss=1.3125 val_metal_acc=0.5048 val_metal_min_recall=0.0000 val_fe_recall=0.4318 val_joint_bal_acc=0.4722 val_joint_macro_f1=0.4448 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.8601 lr=4.75232e-05 train_metal_acc=0.7663 val_loss=1.1891 val_metal_acc=0.6683 val_metal_min_recall=0.0769 val_fe_recall=0.5227 val_joint_bal_acc=0.5745 val_joint_macro_f1=0.5448 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.8818 lr=4.75232e-05 train_metal_acc=0.7688 val_loss=1.1964 val_metal_acc=0.6298 val_metal_min_recall=0.0769 val_fe_recall=0.6591 val_joint_bal_acc=0.5638 val_joint_macro_f1=0.5340 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.8392 lr=4.75232e-05 train_metal_acc=0.7731 val_loss=1.2719 val_metal_acc=0.6010 val_metal_min_recall=0.0000 val_fe_recall=0.5909 val_joint_bal_acc=0.5235 val_joint_macro_f1=0.4911 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.8130 lr=4.75232e-05 train_metal_acc=0.7976 val_loss=1.1171 val_metal_acc=0.6827 val_metal_min_recall=0.0769 val_fe_recall=0.7500 val_joint_bal_acc=0.5619 val_joint_macro_f1=0.5535 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.7807 lr=4.75232e-05 train_metal_acc=0.7985 val_loss=1.0877 val_metal_acc=0.6827 val_metal_min_recall=0.0769 val_fe_recall=0.7045 val_joint_bal_acc=0.5729 val_joint_macro_f1=0.5546 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.7816 lr=4.75232e-05 train_metal_acc=0.7892 val_loss=1.2632 val_metal_acc=0.5913 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5388 val_joint_macro_f1=0.5123 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.7536 lr=4.75232e-05 train_metal_acc=0.7968 val_loss=1.1122 val_metal_acc=0.6827 val_metal_min_recall=0.0000 val_fe_recall=0.7955 val_joint_bal_acc=0.5526 val_joint_macro_f1=0.5218 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.7549 lr=4.75232e-05 train_metal_acc=0.8163 val_loss=1.1730 val_metal_acc=0.6587 val_metal_min_recall=0.0000 val_fe_recall=0.7955 val_joint_bal_acc=0.5359 val_joint_macro_f1=0.5230 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.7178 lr=4.75232e-05 train_metal_acc=0.8052 val_loss=1.1601 val_metal_acc=0.6971 val_metal_min_recall=0.0769 val_fe_recall=0.6818 val_joint_bal_acc=0.5737 val_joint_macro_f1=0.5628 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.7260 lr=4.75232e-05 train_metal_acc=0.7942 val_loss=1.2682 val_metal_acc=0.6442 val_metal_min_recall=0.0769 val_fe_recall=0.6136 val_joint_bal_acc=0.5368 val_joint_macro_f1=0.5384 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.7084 lr=4.75232e-05 train_metal_acc=0.8095 val_loss=1.2641 val_metal_acc=0.6635 val_metal_min_recall=0.0769 val_fe_recall=0.5682 val_joint_bal_acc=0.5583 val_joint_macro_f1=0.5297 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.6872 lr=4.75232e-05 train_metal_acc=0.8357 val_loss=1.2293 val_metal_acc=0.6490 val_metal_min_recall=0.0769 val_fe_recall=0.6591 val_joint_bal_acc=0.5416 val_joint_macro_f1=0.5229 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.6625 lr=4.75232e-05 train_metal_acc=0.8349 val_loss=1.1905 val_metal_acc=0.6731 val_metal_min_recall=0.0769 val_fe_recall=0.7500 val_joint_bal_acc=0.5515 val_joint_macro_f1=0.5485 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.6399 lr=4.75232e-05 train_metal_acc=0.8467 val_loss=1.1800 val_metal_acc=0.6683 val_metal_min_recall=0.0000 val_fe_recall=0.7727 val_joint_bal_acc=0.5407 val_joint_macro_f1=0.5220 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.6503 lr=4.75232e-05 train_metal_acc=0.8510 val_loss=1.2198 val_metal_acc=0.6635 val_metal_min_recall=0.0769 val_fe_recall=0.7500 val_joint_bal_acc=0.5445 val_joint_macro_f1=0.5359 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.6162 lr=4.75232e-05 train_metal_acc=0.8323 val_loss=1.2304 val_metal_acc=0.6635 val_metal_min_recall=0.0769 val_fe_recall=0.5909 val_joint_bal_acc=0.5604 val_joint_macro_f1=0.5420 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.6137 lr=4.75232e-05 train_metal_acc=0.8484 val_loss=1.2287 val_metal_acc=0.6683 val_metal_min_recall=0.0769 val_fe_recall=0.6818 val_joint_bal_acc=0.5575 val_joint_macro_f1=0.5536 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.6027 lr=4.75232e-05 train_metal_acc=0.8391 val_loss=1.2937 val_metal_acc=0.6490 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5406 val_joint_macro_f1=0.5292 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.6121 lr=4.75232e-05 train_metal_acc=0.8620 val_loss=1.1639 val_metal_acc=0.6779 val_metal_min_recall=0.0769 val_fe_recall=0.7045 val_joint_bal_acc=0.5595 val_joint_macro_f1=0.5495 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=41 train_loss=0.5673 lr=4.75232e-05 train_metal_acc=0.8654 val_loss=1.1400 val_metal_acc=0.6875 val_metal_min_recall=0.0769 val_fe_recall=0.7727 val_joint_bal_acc=0.5646 val_joint_macro_f1=0.5660 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=42 train_loss=0.5564 lr=4.75232e-05 train_metal_acc=0.8425 val_loss=1.2136 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.7727 val_joint_bal_acc=0.5303 val_joint_macro_f1=0.5389 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=43 train_loss=0.5480 lr=4.75232e-05 train_metal_acc=0.8637 val_loss=1.2437 val_metal_acc=0.6731 val_metal_min_recall=0.0769 val_fe_recall=0.7273 val_joint_bal_acc=0.5470 val_joint_macro_f1=0.5477 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=44 train_loss=0.5365 lr=4.75232e-05 train_metal_acc=0.8662 val_loss=1.1641 val_metal_acc=0.6971 val_metal_min_recall=0.0769 val_fe_recall=0.7727 val_joint_bal_acc=0.5785 val_joint_macro_f1=0.5771 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=45 train_loss=0.5130 lr=4.75232e-05 train_metal_acc=0.8764 val_loss=1.2668 val_metal_acc=0.6731 val_metal_min_recall=0.0769 val_fe_recall=0.6591 val_joint_bal_acc=0.5443 val_joint_macro_f1=0.5448 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=46 train_loss=0.5071 lr=4.75232e-05 train_metal_acc=0.8789 val_loss=1.3012 val_metal_acc=0.6635 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5482 val_joint_macro_f1=0.5357 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=47 train_loss=0.5065 lr=4.75232e-05 train_metal_acc=0.8865 val_loss=1.2636 val_metal_acc=0.6923 val_metal_min_recall=0.0769 val_fe_recall=0.7727 val_joint_bal_acc=0.5744 val_joint_macro_f1=0.5674 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=48 train_loss=0.4750 lr=4.75232e-05 train_metal_acc=0.8933 val_loss=1.3175 val_metal_acc=0.6923 val_metal_min_recall=0.0769 val_fe_recall=0.7727 val_joint_bal_acc=0.5674 val_joint_macro_f1=0.5581 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=49 train_loss=0.4969 lr=4.75232e-05 train_metal_acc=0.8908 val_loss=1.2897 val_metal_acc=0.6683 val_metal_min_recall=0.0769 val_fe_recall=0.7727 val_joint_bal_acc=0.5435 val_joint_macro_f1=0.5532 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=50 train_loss=0.4750 lr=4.75232e-05 train_metal_acc=0.8459 val_loss=1.4579 val_metal_acc=0.6587 val_metal_min_recall=0.0769 val_fe_recall=0.7955 val_joint_bal_acc=0.5421 val_joint_macro_f1=0.5512 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_2c77cc48/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_2c77cc48/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_2c77cc48/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_2c77cc48/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_2c77cc48/run_metadata.json
Completed: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_2c77cc48
================================================================================
[#002 | 2/5] deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_1a1fd84a
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat --run-name deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_1a1fd84a --model-architecture only_gvp --epochs 50 --batch-size 8 --learning-rate 4.752317377508605e-05 --weight-decay 0.0 --seed 123 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 128 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_sqrt_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges
stdout log: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/_execution_logs/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_1a1fd84a.stdout.log
stderr log: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/_execution_logs/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_1a1fd84a.stderr.log
================================================================================

=== Split diagnostics (passive; training behavior unchanged) ===
task=metal split_by=pdbid val_fraction=0.15 n_folds=None fold_index=None
pockets: train=1181 validation=208
groups by pdbid: train=1152 validation=109
train/validation overlap counts: pdbid=0, pdbid_chain=0, structure_id=0, pocket_id=0
train metal distribution: Mn=551, Cu=70, Zn=173, Fe=251, Co=73, Ni=63
validation metal distribution: Mn=97, Cu=15, Zn=34, Fe=41, Co=13, Ni=8
missing train metal classes: none
missing validation metal classes: none
train EC distribution: 1=495, 2=199, 3=346, 4=46, 5=64, 6=14, 7=1
validation EC distribution: 1=64, 2=16, 3=46, 4=8, 5=69, 6=3, 7=0
missing train EC classes: none
missing validation EC classes: 7
===============================================================

epoch=1 train_loss=1.7061 lr=4.75232e-05 train_metal_acc=0.4666 val_loss=1.6342 val_metal_acc=0.4663 val_metal_min_recall=0.0000 val_fe_recall=0.0000 val_joint_bal_acc=0.1667 val_joint_macro_f1=0.1060 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.6717 lr=4.75232e-05 train_metal_acc=0.5250 val_loss=1.6023 val_metal_acc=0.5481 val_metal_min_recall=0.0000 val_fe_recall=0.4146 val_joint_bal_acc=0.2358 val_joint_macro_f1=0.1985 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.5773 lr=4.75232e-05 train_metal_acc=0.5715 val_loss=1.3645 val_metal_acc=0.6442 val_metal_min_recall=0.0000 val_fe_recall=0.7073 val_joint_bal_acc=0.3736 val_joint_macro_f1=0.3569 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.4386 lr=4.75232e-05 train_metal_acc=0.5224 val_loss=1.3913 val_metal_acc=0.6298 val_metal_min_recall=0.0000 val_fe_recall=0.9512 val_joint_bal_acc=0.4293 val_joint_macro_f1=0.3456 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.3563 lr=4.75232e-05 train_metal_acc=0.6224 val_loss=1.2344 val_metal_acc=0.6635 val_metal_min_recall=0.0000 val_fe_recall=0.8049 val_joint_bal_acc=0.4117 val_joint_macro_f1=0.3664 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.3125 lr=4.75232e-05 train_metal_acc=0.6452 val_loss=1.1953 val_metal_acc=0.6923 val_metal_min_recall=0.0000 val_fe_recall=0.6098 val_joint_bal_acc=0.4632 val_joint_macro_f1=0.4363 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=1.2509 lr=4.75232e-05 train_metal_acc=0.6732 val_loss=1.1550 val_metal_acc=0.7308 val_metal_min_recall=0.0000 val_fe_recall=0.8537 val_joint_bal_acc=0.4973 val_joint_macro_f1=0.4687 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=1.2110 lr=4.75232e-05 train_metal_acc=0.6672 val_loss=1.1262 val_metal_acc=0.7115 val_metal_min_recall=0.0000 val_fe_recall=0.7561 val_joint_bal_acc=0.4904 val_joint_macro_f1=0.4416 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=1.1674 lr=4.75232e-05 train_metal_acc=0.6969 val_loss=1.1739 val_metal_acc=0.6442 val_metal_min_recall=0.0000 val_fe_recall=0.7561 val_joint_bal_acc=0.4888 val_joint_macro_f1=0.4694 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=1.1619 lr=4.75232e-05 train_metal_acc=0.6283 val_loss=1.3384 val_metal_acc=0.5144 val_metal_min_recall=0.0000 val_fe_recall=0.9024 val_joint_bal_acc=0.4437 val_joint_macro_f1=0.3771 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=1.1177 lr=4.75232e-05 train_metal_acc=0.6977 val_loss=1.1606 val_metal_acc=0.6346 val_metal_min_recall=0.0000 val_fe_recall=0.6829 val_joint_bal_acc=0.4974 val_joint_macro_f1=0.4712 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=1.1039 lr=4.75232e-05 train_metal_acc=0.6977 val_loss=1.1354 val_metal_acc=0.6346 val_metal_min_recall=0.0000 val_fe_recall=0.7805 val_joint_bal_acc=0.4907 val_joint_macro_f1=0.4476 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=1.0864 lr=4.75232e-05 train_metal_acc=0.6198 val_loss=1.3523 val_metal_acc=0.4519 val_metal_min_recall=0.0000 val_fe_recall=0.7317 val_joint_bal_acc=0.4303 val_joint_macro_f1=0.3433 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=1.0759 lr=4.75232e-05 train_metal_acc=0.7062 val_loss=1.1332 val_metal_acc=0.6202 val_metal_min_recall=0.0769 val_fe_recall=0.6098 val_joint_bal_acc=0.4993 val_joint_macro_f1=0.4765 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=1.0514 lr=4.75232e-05 train_metal_acc=0.7189 val_loss=1.2179 val_metal_acc=0.5962 val_metal_min_recall=0.0000 val_fe_recall=0.6098 val_joint_bal_acc=0.5083 val_joint_macro_f1=0.4690 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=1.0243 lr=4.75232e-05 train_metal_acc=0.7079 val_loss=1.2136 val_metal_acc=0.5721 val_metal_min_recall=0.0769 val_fe_recall=0.6341 val_joint_bal_acc=0.5132 val_joint_macro_f1=0.4867 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=1.0180 lr=4.75232e-05 train_metal_acc=0.7384 val_loss=1.1090 val_metal_acc=0.6442 val_metal_min_recall=0.0769 val_fe_recall=0.6341 val_joint_bal_acc=0.5230 val_joint_macro_f1=0.5081 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.9768 lr=4.75232e-05 train_metal_acc=0.7477 val_loss=1.1456 val_metal_acc=0.6058 val_metal_min_recall=0.0769 val_fe_recall=0.6341 val_joint_bal_acc=0.5158 val_joint_macro_f1=0.5111 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.9717 lr=4.75232e-05 train_metal_acc=0.7214 val_loss=1.1465 val_metal_acc=0.6683 val_metal_min_recall=0.0000 val_fe_recall=0.8537 val_joint_bal_acc=0.5639 val_joint_macro_f1=0.5377 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.9432 lr=4.75232e-05 train_metal_acc=0.7206 val_loss=1.1727 val_metal_acc=0.6202 val_metal_min_recall=0.0000 val_fe_recall=0.6829 val_joint_bal_acc=0.4859 val_joint_macro_f1=0.4602 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.9313 lr=4.75232e-05 train_metal_acc=0.7697 val_loss=1.1243 val_metal_acc=0.6490 val_metal_min_recall=0.0000 val_fe_recall=0.7073 val_joint_bal_acc=0.5589 val_joint_macro_f1=0.5276 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.9323 lr=4.75232e-05 train_metal_acc=0.7443 val_loss=1.2462 val_metal_acc=0.6106 val_metal_min_recall=0.0769 val_fe_recall=0.4634 val_joint_bal_acc=0.5360 val_joint_macro_f1=0.5054 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.9003 lr=4.75232e-05 train_metal_acc=0.7756 val_loss=1.1889 val_metal_acc=0.5865 val_metal_min_recall=0.0769 val_fe_recall=0.7073 val_joint_bal_acc=0.5540 val_joint_macro_f1=0.5247 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.8870 lr=4.75232e-05 train_metal_acc=0.7324 val_loss=1.3108 val_metal_acc=0.5625 val_metal_min_recall=0.0769 val_fe_recall=0.8049 val_joint_bal_acc=0.5198 val_joint_macro_f1=0.4974 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.8552 lr=4.75232e-05 train_metal_acc=0.7688 val_loss=1.2440 val_metal_acc=0.5625 val_metal_min_recall=0.0000 val_fe_recall=0.5610 val_joint_bal_acc=0.5457 val_joint_macro_f1=0.4899 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.8306 lr=4.75232e-05 train_metal_acc=0.7934 val_loss=1.2794 val_metal_acc=0.5577 val_metal_min_recall=0.0769 val_fe_recall=0.6585 val_joint_bal_acc=0.5167 val_joint_macro_f1=0.4986 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.8318 lr=4.75232e-05 train_metal_acc=0.7587 val_loss=1.1995 val_metal_acc=0.6442 val_metal_min_recall=0.0000 val_fe_recall=0.7317 val_joint_bal_acc=0.5373 val_joint_macro_f1=0.4883 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.8088 lr=4.75232e-05 train_metal_acc=0.7959 val_loss=1.1987 val_metal_acc=0.6250 val_metal_min_recall=0.0769 val_fe_recall=0.7317 val_joint_bal_acc=0.5447 val_joint_macro_f1=0.5307 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.7879 lr=4.75232e-05 train_metal_acc=0.7968 val_loss=1.2842 val_metal_acc=0.5769 val_metal_min_recall=0.0000 val_fe_recall=0.7317 val_joint_bal_acc=0.5227 val_joint_macro_f1=0.4895 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.7866 lr=4.75232e-05 train_metal_acc=0.7985 val_loss=1.2831 val_metal_acc=0.5673 val_metal_min_recall=0.0769 val_fe_recall=0.7561 val_joint_bal_acc=0.5170 val_joint_macro_f1=0.4899 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.7729 lr=4.75232e-05 train_metal_acc=0.7722 val_loss=1.4699 val_metal_acc=0.5096 val_metal_min_recall=0.0769 val_fe_recall=0.6098 val_joint_bal_acc=0.5172 val_joint_macro_f1=0.4894 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.7405 lr=4.75232e-05 train_metal_acc=0.7976 val_loss=1.2008 val_metal_acc=0.6490 val_metal_min_recall=0.0769 val_fe_recall=0.7317 val_joint_bal_acc=0.5278 val_joint_macro_f1=0.5130 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.7358 lr=4.75232e-05 train_metal_acc=0.8146 val_loss=1.2898 val_metal_acc=0.6154 val_metal_min_recall=0.0769 val_fe_recall=0.7317 val_joint_bal_acc=0.5794 val_joint_macro_f1=0.5350 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.7304 lr=4.75232e-05 train_metal_acc=0.8129 val_loss=1.2555 val_metal_acc=0.6346 val_metal_min_recall=0.0769 val_fe_recall=0.6829 val_joint_bal_acc=0.5561 val_joint_macro_f1=0.5347 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.6950 lr=4.75232e-05 train_metal_acc=0.7993 val_loss=1.2050 val_metal_acc=0.6538 val_metal_min_recall=0.0000 val_fe_recall=0.7317 val_joint_bal_acc=0.5630 val_joint_macro_f1=0.5208 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.6937 lr=4.75232e-05 train_metal_acc=0.8180 val_loss=1.4162 val_metal_acc=0.5721 val_metal_min_recall=0.0769 val_fe_recall=0.6098 val_joint_bal_acc=0.5554 val_joint_macro_f1=0.5224 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.6893 lr=4.75232e-05 train_metal_acc=0.8205 val_loss=1.3414 val_metal_acc=0.5817 val_metal_min_recall=0.0769 val_fe_recall=0.6098 val_joint_bal_acc=0.5589 val_joint_macro_f1=0.5335 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.6634 lr=4.75232e-05 train_metal_acc=0.8222 val_loss=1.3298 val_metal_acc=0.6106 val_metal_min_recall=0.0769 val_fe_recall=0.6098 val_joint_bal_acc=0.5563 val_joint_macro_f1=0.5385 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.6429 lr=4.75232e-05 train_metal_acc=0.7731 val_loss=1.6374 val_metal_acc=0.4856 val_metal_min_recall=0.0000 val_fe_recall=0.7561 val_joint_bal_acc=0.5020 val_joint_macro_f1=0.4364 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.6847 lr=4.75232e-05 train_metal_acc=0.8264 val_loss=1.3904 val_metal_acc=0.5577 val_metal_min_recall=0.1538 val_fe_recall=0.7073 val_joint_bal_acc=0.5294 val_joint_macro_f1=0.5012 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=41 train_loss=0.6414 lr=4.75232e-05 train_metal_acc=0.8408 val_loss=1.5001 val_metal_acc=0.5529 val_metal_min_recall=0.1538 val_fe_recall=0.6829 val_joint_bal_acc=0.5410 val_joint_macro_f1=0.5192 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=42 train_loss=0.6070 lr=4.75232e-05 train_metal_acc=0.8459 val_loss=1.3773 val_metal_acc=0.6202 val_metal_min_recall=0.0769 val_fe_recall=0.7317 val_joint_bal_acc=0.5780 val_joint_macro_f1=0.5449 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=43 train_loss=0.6167 lr=4.75232e-05 train_metal_acc=0.8425 val_loss=1.3161 val_metal_acc=0.5962 val_metal_min_recall=0.0769 val_fe_recall=0.7561 val_joint_bal_acc=0.5654 val_joint_macro_f1=0.5282 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=44 train_loss=0.5958 lr=4.75232e-05 train_metal_acc=0.8518 val_loss=1.3702 val_metal_acc=0.5913 val_metal_min_recall=0.0000 val_fe_recall=0.7561 val_joint_bal_acc=0.5621 val_joint_macro_f1=0.5229 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=45 train_loss=0.5658 lr=4.75232e-05 train_metal_acc=0.8501 val_loss=1.4674 val_metal_acc=0.5385 val_metal_min_recall=0.0000 val_fe_recall=0.7073 val_joint_bal_acc=0.5258 val_joint_macro_f1=0.4852 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=46 train_loss=0.5546 lr=4.75232e-05 train_metal_acc=0.8501 val_loss=1.4611 val_metal_acc=0.5625 val_metal_min_recall=0.0000 val_fe_recall=0.7317 val_joint_bal_acc=0.5367 val_joint_macro_f1=0.4757 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=47 train_loss=0.5628 lr=4.75232e-05 train_metal_acc=0.8671 val_loss=1.6043 val_metal_acc=0.5144 val_metal_min_recall=0.0769 val_fe_recall=0.6585 val_joint_bal_acc=0.5363 val_joint_macro_f1=0.4958 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=48 train_loss=0.5445 lr=4.75232e-05 train_metal_acc=0.8603 val_loss=1.3596 val_metal_acc=0.6010 val_metal_min_recall=0.0000 val_fe_recall=0.7317 val_joint_bal_acc=0.5536 val_joint_macro_f1=0.5081 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=49 train_loss=0.5752 lr=4.75232e-05 train_metal_acc=0.8645 val_loss=1.3712 val_metal_acc=0.5913 val_metal_min_recall=0.1538 val_fe_recall=0.7073 val_joint_bal_acc=0.5573 val_joint_macro_f1=0.5375 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=50 train_loss=0.5307 lr=4.75232e-05 train_metal_acc=0.8730 val_loss=1.4624 val_metal_acc=0.5721 val_metal_min_recall=0.0000 val_fe_recall=0.7073 val_joint_bal_acc=0.5442 val_joint_macro_f1=0.5058 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_1a1fd84a/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_1a1fd84a/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_1a1fd84a/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_1a1fd84a/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_1a1fd84a/run_metadata.json
Completed: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_1a1fd84a
================================================================================
[#003 | 3/5] deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_14db2a0a
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat --run-name deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_14db2a0a --model-architecture only_gvp --epochs 50 --batch-size 8 --learning-rate 4.752317377508605e-05 --weight-decay 0.0 --seed 2026 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 128 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_sqrt_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges
stdout log: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/_execution_logs/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_14db2a0a.stdout.log
stderr log: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/_execution_logs/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_14db2a0a.stderr.log
================================================================================

=== Split diagnostics (passive; training behavior unchanged) ===
task=metal split_by=pdbid val_fraction=0.15 n_folds=None fold_index=None
pockets: train=1181 validation=208
groups by pdbid: train=1152 validation=109
train/validation overlap counts: pdbid=0, pdbid_chain=0, structure_id=0, pocket_id=0
train metal distribution: Mn=549, Cu=72, Zn=173, Fe=251, Co=73, Ni=63
validation metal distribution: Mn=99, Cu=13, Zn=34, Fe=41, Co=13, Ni=8
missing train metal classes: none
missing validation metal classes: none
train EC distribution: 1=495, 2=198, 3=338, 4=47, 5=73, 6=15, 7=1
validation EC distribution: 1=64, 2=17, 3=54, 4=7, 5=60, 6=2, 7=0
missing train EC classes: none
missing validation EC classes: 7
===============================================================

epoch=1 train_loss=1.6984 lr=4.75232e-05 train_metal_acc=0.4623 val_loss=1.6332 val_metal_acc=0.5000 val_metal_min_recall=0.0000 val_fe_recall=0.1220 val_joint_bal_acc=0.1870 val_joint_macro_f1=0.1438 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.6683 lr=4.75232e-05 train_metal_acc=0.4098 val_loss=1.6255 val_metal_acc=0.3990 val_metal_min_recall=0.0000 val_fe_recall=0.9512 val_joint_bal_acc=0.2326 val_joint_macro_f1=0.1599 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.6163 lr=4.75232e-05 train_metal_acc=0.5428 val_loss=1.4756 val_metal_acc=0.5721 val_metal_min_recall=0.0000 val_fe_recall=0.6098 val_joint_bal_acc=0.2821 val_joint_macro_f1=0.2535 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.4808 lr=4.75232e-05 train_metal_acc=0.5868 val_loss=1.3910 val_metal_acc=0.6538 val_metal_min_recall=0.0000 val_fe_recall=0.8293 val_joint_bal_acc=0.3627 val_joint_macro_f1=0.3544 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.3926 lr=4.75232e-05 train_metal_acc=0.6054 val_loss=1.2747 val_metal_acc=0.6538 val_metal_min_recall=0.0000 val_fe_recall=0.7317 val_joint_bal_acc=0.3927 val_joint_macro_f1=0.3624 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.3147 lr=4.75232e-05 train_metal_acc=0.5876 val_loss=1.2235 val_metal_acc=0.6250 val_metal_min_recall=0.0000 val_fe_recall=0.7561 val_joint_bal_acc=0.4617 val_joint_macro_f1=0.3934 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=1.2700 lr=4.75232e-05 train_metal_acc=0.6562 val_loss=1.1665 val_metal_acc=0.6538 val_metal_min_recall=0.0000 val_fe_recall=0.5366 val_joint_bal_acc=0.4311 val_joint_macro_f1=0.3783 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=1.2293 lr=4.75232e-05 train_metal_acc=0.6943 val_loss=1.1533 val_metal_acc=0.6394 val_metal_min_recall=0.0000 val_fe_recall=0.6098 val_joint_bal_acc=0.4718 val_joint_macro_f1=0.4205 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=1.1950 lr=4.75232e-05 train_metal_acc=0.6757 val_loss=1.1514 val_metal_acc=0.6106 val_metal_min_recall=0.0000 val_fe_recall=0.6585 val_joint_bal_acc=0.4246 val_joint_macro_f1=0.3674 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=1.1630 lr=4.75232e-05 train_metal_acc=0.6926 val_loss=1.1194 val_metal_acc=0.6106 val_metal_min_recall=0.0000 val_fe_recall=0.6829 val_joint_bal_acc=0.4574 val_joint_macro_f1=0.4138 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=1.1282 lr=4.75232e-05 train_metal_acc=0.6181 val_loss=1.2071 val_metal_acc=0.5769 val_metal_min_recall=0.0000 val_fe_recall=0.7317 val_joint_bal_acc=0.5000 val_joint_macro_f1=0.4390 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=1.1186 lr=4.75232e-05 train_metal_acc=0.7121 val_loss=1.1246 val_metal_acc=0.5913 val_metal_min_recall=0.0000 val_fe_recall=0.6341 val_joint_bal_acc=0.4109 val_joint_macro_f1=0.3962 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=1.0670 lr=4.75232e-05 train_metal_acc=0.7155 val_loss=1.0736 val_metal_acc=0.6538 val_metal_min_recall=0.0000 val_fe_recall=0.8780 val_joint_bal_acc=0.4727 val_joint_macro_f1=0.4507 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=1.0525 lr=4.75232e-05 train_metal_acc=0.7223 val_loss=1.0911 val_metal_acc=0.6346 val_metal_min_recall=0.0000 val_fe_recall=0.5366 val_joint_bal_acc=0.5204 val_joint_macro_f1=0.4682 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=1.0327 lr=4.75232e-05 train_metal_acc=0.6884 val_loss=1.1462 val_metal_acc=0.6442 val_metal_min_recall=0.0000 val_fe_recall=0.8780 val_joint_bal_acc=0.4932 val_joint_macro_f1=0.4300 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=1.0146 lr=4.75232e-05 train_metal_acc=0.7367 val_loss=1.0910 val_metal_acc=0.6058 val_metal_min_recall=0.0000 val_fe_recall=0.5122 val_joint_bal_acc=0.4617 val_joint_macro_f1=0.4343 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.9962 lr=4.75232e-05 train_metal_acc=0.7384 val_loss=1.0686 val_metal_acc=0.6394 val_metal_min_recall=0.0000 val_fe_recall=0.6829 val_joint_bal_acc=0.4869 val_joint_macro_f1=0.4788 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.9712 lr=4.75232e-05 train_metal_acc=0.7570 val_loss=1.0532 val_metal_acc=0.6683 val_metal_min_recall=0.0000 val_fe_recall=0.7805 val_joint_bal_acc=0.5496 val_joint_macro_f1=0.5127 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.9545 lr=4.75232e-05 train_metal_acc=0.7375 val_loss=1.0857 val_metal_acc=0.6875 val_metal_min_recall=0.0000 val_fe_recall=0.9268 val_joint_bal_acc=0.5612 val_joint_macro_f1=0.5367 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.9342 lr=4.75232e-05 train_metal_acc=0.7638 val_loss=1.0709 val_metal_acc=0.6490 val_metal_min_recall=0.0000 val_fe_recall=0.5854 val_joint_bal_acc=0.5129 val_joint_macro_f1=0.4759 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.9176 lr=4.75232e-05 train_metal_acc=0.7680 val_loss=1.0577 val_metal_acc=0.6587 val_metal_min_recall=0.0000 val_fe_recall=0.6341 val_joint_bal_acc=0.5545 val_joint_macro_f1=0.5218 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.9000 lr=4.75232e-05 train_metal_acc=0.7604 val_loss=0.9881 val_metal_acc=0.6731 val_metal_min_recall=0.0000 val_fe_recall=0.8780 val_joint_bal_acc=0.5321 val_joint_macro_f1=0.5110 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.8810 lr=4.75232e-05 train_metal_acc=0.7587 val_loss=1.0386 val_metal_acc=0.6490 val_metal_min_recall=0.0000 val_fe_recall=0.6829 val_joint_bal_acc=0.4554 val_joint_macro_f1=0.4697 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.8620 lr=4.75232e-05 train_metal_acc=0.7705 val_loss=1.0584 val_metal_acc=0.6490 val_metal_min_recall=0.0769 val_fe_recall=0.6098 val_joint_bal_acc=0.5408 val_joint_macro_f1=0.5354 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.8659 lr=4.75232e-05 train_metal_acc=0.7494 val_loss=0.9872 val_metal_acc=0.6875 val_metal_min_recall=0.0769 val_fe_recall=0.7561 val_joint_bal_acc=0.5268 val_joint_macro_f1=0.5039 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.8555 lr=4.75232e-05 train_metal_acc=0.7697 val_loss=0.9891 val_metal_acc=0.6635 val_metal_min_recall=0.0769 val_fe_recall=0.6098 val_joint_bal_acc=0.5538 val_joint_macro_f1=0.5468 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.8236 lr=4.75232e-05 train_metal_acc=0.7417 val_loss=1.1509 val_metal_acc=0.5721 val_metal_min_recall=0.0000 val_fe_recall=0.8780 val_joint_bal_acc=0.5064 val_joint_macro_f1=0.4711 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.8078 lr=4.75232e-05 train_metal_acc=0.7942 val_loss=0.9855 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.8049 val_joint_bal_acc=0.5358 val_joint_macro_f1=0.5144 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.7934 lr=4.75232e-05 train_metal_acc=0.7815 val_loss=1.0010 val_metal_acc=0.6587 val_metal_min_recall=0.0769 val_fe_recall=0.5854 val_joint_bal_acc=0.5242 val_joint_macro_f1=0.5227 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.7691 lr=4.75232e-05 train_metal_acc=0.8027 val_loss=0.9981 val_metal_acc=0.6731 val_metal_min_recall=0.0769 val_fe_recall=0.8049 val_joint_bal_acc=0.5778 val_joint_macro_f1=0.5579 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.7649 lr=4.75232e-05 train_metal_acc=0.8010 val_loss=1.0713 val_metal_acc=0.6346 val_metal_min_recall=0.0769 val_fe_recall=0.8780 val_joint_bal_acc=0.5204 val_joint_macro_f1=0.5219 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.7679 lr=4.75232e-05 train_metal_acc=0.7925 val_loss=0.9828 val_metal_acc=0.6779 val_metal_min_recall=0.0769 val_fe_recall=0.7073 val_joint_bal_acc=0.5651 val_joint_macro_f1=0.5639 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.7465 lr=4.75232e-05 train_metal_acc=0.8069 val_loss=1.0233 val_metal_acc=0.6442 val_metal_min_recall=0.0000 val_fe_recall=0.7073 val_joint_bal_acc=0.5469 val_joint_macro_f1=0.5128 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.7099 lr=4.75232e-05 train_metal_acc=0.8086 val_loss=1.0232 val_metal_acc=0.6587 val_metal_min_recall=0.0769 val_fe_recall=0.8780 val_joint_bal_acc=0.6243 val_joint_macro_f1=0.5608 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.7195 lr=4.75232e-05 train_metal_acc=0.8137 val_loss=0.9957 val_metal_acc=0.6779 val_metal_min_recall=0.0769 val_fe_recall=0.7073 val_joint_bal_acc=0.5619 val_joint_macro_f1=0.5545 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.6890 lr=4.75232e-05 train_metal_acc=0.8230 val_loss=0.9630 val_metal_acc=0.6587 val_metal_min_recall=0.0769 val_fe_recall=0.6829 val_joint_bal_acc=0.5560 val_joint_macro_f1=0.5367 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.6807 lr=4.75232e-05 train_metal_acc=0.8188 val_loss=0.9611 val_metal_acc=0.7067 val_metal_min_recall=0.0769 val_fe_recall=0.8537 val_joint_bal_acc=0.5655 val_joint_macro_f1=0.5609 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.6635 lr=4.75232e-05 train_metal_acc=0.8366 val_loss=0.9805 val_metal_acc=0.6731 val_metal_min_recall=0.0769 val_fe_recall=0.7561 val_joint_bal_acc=0.6080 val_joint_macro_f1=0.5623 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.6675 lr=4.75232e-05 train_metal_acc=0.8323 val_loss=1.0155 val_metal_acc=0.6635 val_metal_min_recall=0.0769 val_fe_recall=0.7317 val_joint_bal_acc=0.5864 val_joint_macro_f1=0.5518 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.6335 lr=4.75232e-05 train_metal_acc=0.8222 val_loss=1.1275 val_metal_acc=0.6346 val_metal_min_recall=0.0000 val_fe_recall=0.8293 val_joint_bal_acc=0.5716 val_joint_macro_f1=0.5270 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=41 train_loss=0.6397 lr=4.75232e-05 train_metal_acc=0.8357 val_loss=1.0204 val_metal_acc=0.6635 val_metal_min_recall=0.0769 val_fe_recall=0.6829 val_joint_bal_acc=0.5625 val_joint_macro_f1=0.5426 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=42 train_loss=0.6080 lr=4.75232e-05 train_metal_acc=0.8298 val_loss=1.0021 val_metal_acc=0.6731 val_metal_min_recall=0.0769 val_fe_recall=0.8537 val_joint_bal_acc=0.5665 val_joint_macro_f1=0.5419 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=43 train_loss=0.5904 lr=4.75232e-05 train_metal_acc=0.8408 val_loss=1.0963 val_metal_acc=0.6731 val_metal_min_recall=0.0000 val_fe_recall=0.8293 val_joint_bal_acc=0.5658 val_joint_macro_f1=0.5212 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=44 train_loss=0.5864 lr=4.75232e-05 train_metal_acc=0.8281 val_loss=1.0384 val_metal_acc=0.6683 val_metal_min_recall=0.0769 val_fe_recall=0.8049 val_joint_bal_acc=0.5920 val_joint_macro_f1=0.5495 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=45 train_loss=0.5709 lr=4.75232e-05 train_metal_acc=0.7638 val_loss=1.2785 val_metal_acc=0.5913 val_metal_min_recall=0.0000 val_fe_recall=0.8537 val_joint_bal_acc=0.5282 val_joint_macro_f1=0.4823 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=46 train_loss=0.5678 lr=4.75232e-05 train_metal_acc=0.8569 val_loss=1.1228 val_metal_acc=0.6827 val_metal_min_recall=0.0769 val_fe_recall=0.8293 val_joint_bal_acc=0.5994 val_joint_macro_f1=0.5637 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=47 train_loss=0.5418 lr=4.75232e-05 train_metal_acc=0.8544 val_loss=1.0286 val_metal_acc=0.6923 val_metal_min_recall=0.0769 val_fe_recall=0.8293 val_joint_bal_acc=0.5501 val_joint_macro_f1=0.5560 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=48 train_loss=0.5322 lr=4.75232e-05 train_metal_acc=0.8527 val_loss=1.1125 val_metal_acc=0.6779 val_metal_min_recall=0.0769 val_fe_recall=0.7561 val_joint_bal_acc=0.5413 val_joint_macro_f1=0.5419 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=49 train_loss=0.5415 lr=4.75232e-05 train_metal_acc=0.8188 val_loss=1.1715 val_metal_acc=0.6442 val_metal_min_recall=0.0769 val_fe_recall=0.4878 val_joint_bal_acc=0.5448 val_joint_macro_f1=0.5223 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=50 train_loss=0.5090 lr=4.75232e-05 train_metal_acc=0.8603 val_loss=1.1827 val_metal_acc=0.6587 val_metal_min_recall=0.0769 val_fe_recall=0.6829 val_joint_bal_acc=0.6024 val_joint_macro_f1=0.5547 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_14db2a0a/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_14db2a0a/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_14db2a0a/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_14db2a0a/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_14db2a0a/run_metadata.json
Completed: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_14db2a0a
================================================================================
[#004 | 4/5] deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_edcdcf3a
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat --run-name deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_edcdcf3a --model-architecture only_gvp --epochs 50 --batch-size 8 --learning-rate 4.752317377508605e-05 --weight-decay 0.0 --seed 43 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 128 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_sqrt_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges
stdout log: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/_execution_logs/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_edcdcf3a.stdout.log
stderr log: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/_execution_logs/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_edcdcf3a.stderr.log
================================================================================

=== Split diagnostics (passive; training behavior unchanged) ===
task=metal split_by=pdbid val_fraction=0.15 n_folds=None fold_index=None
pockets: train=1181 validation=208
groups by pdbid: train=1151 validation=110
train/validation overlap counts: pdbid=0, pdbid_chain=0, structure_id=0, pocket_id=0
train metal distribution: Mn=551, Cu=72, Zn=173, Fe=251, Co=73, Ni=61
validation metal distribution: Mn=97, Cu=13, Zn=34, Fe=41, Co=13, Ni=10
missing train metal classes: none
missing validation metal classes: none
train EC distribution: 1=497, 2=200, 3=332, 4=46, 5=75, 6=14, 7=1
validation EC distribution: 1=62, 2=15, 3=60, 4=8, 5=58, 6=3, 7=0
missing train EC classes: none
missing validation EC classes: 7
===============================================================

epoch=1 train_loss=1.7005 lr=4.75232e-05 train_metal_acc=0.4666 val_loss=1.6390 val_metal_acc=0.4663 val_metal_min_recall=0.0000 val_fe_recall=0.0000 val_joint_bal_acc=0.1667 val_joint_macro_f1=0.1060 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.6671 lr=4.75232e-05 train_metal_acc=0.4733 val_loss=1.6099 val_metal_acc=0.4808 val_metal_min_recall=0.0000 val_fe_recall=0.0732 val_joint_bal_acc=0.1789 val_joint_macro_f1=0.1296 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.5874 lr=4.75232e-05 train_metal_acc=0.5538 val_loss=1.4262 val_metal_acc=0.6106 val_metal_min_recall=0.0000 val_fe_recall=0.6585 val_joint_bal_acc=0.3133 val_joint_macro_f1=0.2922 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.4480 lr=4.75232e-05 train_metal_acc=0.5961 val_loss=1.2927 val_metal_acc=0.6346 val_metal_min_recall=0.0000 val_fe_recall=0.6341 val_joint_bal_acc=0.3704 val_joint_macro_f1=0.3659 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.3585 lr=4.75232e-05 train_metal_acc=0.6367 val_loss=1.2557 val_metal_acc=0.7019 val_metal_min_recall=0.0000 val_fe_recall=0.8780 val_joint_bal_acc=0.4655 val_joint_macro_f1=0.4315 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.2971 lr=4.75232e-05 train_metal_acc=0.6215 val_loss=1.2580 val_metal_acc=0.6731 val_metal_min_recall=0.0000 val_fe_recall=0.9024 val_joint_bal_acc=0.4559 val_joint_macro_f1=0.3752 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=1.2363 lr=4.75232e-05 train_metal_acc=0.6723 val_loss=1.2257 val_metal_acc=0.6923 val_metal_min_recall=0.0000 val_fe_recall=0.6098 val_joint_bal_acc=0.4903 val_joint_macro_f1=0.4367 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=1.1944 lr=4.75232e-05 train_metal_acc=0.6274 val_loss=1.2547 val_metal_acc=0.6058 val_metal_min_recall=0.0000 val_fe_recall=0.9024 val_joint_bal_acc=0.4681 val_joint_macro_f1=0.4353 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=1.1625 lr=4.75232e-05 train_metal_acc=0.6867 val_loss=1.1771 val_metal_acc=0.6346 val_metal_min_recall=0.0000 val_fe_recall=0.5854 val_joint_bal_acc=0.4876 val_joint_macro_f1=0.4638 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=1.1613 lr=4.75232e-05 train_metal_acc=0.6435 val_loss=1.2109 val_metal_acc=0.6875 val_metal_min_recall=0.0000 val_fe_recall=0.9024 val_joint_bal_acc=0.4846 val_joint_macro_f1=0.4222 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=1.1115 lr=4.75232e-05 train_metal_acc=0.6960 val_loss=1.2061 val_metal_acc=0.6058 val_metal_min_recall=0.0000 val_fe_recall=0.8049 val_joint_bal_acc=0.5433 val_joint_macro_f1=0.4991 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=1.0546 lr=4.75232e-05 train_metal_acc=0.6698 val_loss=1.3437 val_metal_acc=0.4567 val_metal_min_recall=0.0000 val_fe_recall=0.5854 val_joint_bal_acc=0.4731 val_joint_macro_f1=0.4332 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=1.0554 lr=4.75232e-05 train_metal_acc=0.7299 val_loss=1.1485 val_metal_acc=0.6154 val_metal_min_recall=0.0000 val_fe_recall=0.6829 val_joint_bal_acc=0.5232 val_joint_macro_f1=0.4896 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=1.0493 lr=4.75232e-05 train_metal_acc=0.7257 val_loss=1.1705 val_metal_acc=0.6346 val_metal_min_recall=0.0000 val_fe_recall=0.7317 val_joint_bal_acc=0.5466 val_joint_macro_f1=0.5058 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=1.0124 lr=4.75232e-05 train_metal_acc=0.7265 val_loss=1.1665 val_metal_acc=0.6394 val_metal_min_recall=0.0000 val_fe_recall=0.8049 val_joint_bal_acc=0.5222 val_joint_macro_f1=0.4846 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.9905 lr=4.75232e-05 train_metal_acc=0.7299 val_loss=1.1935 val_metal_acc=0.5673 val_metal_min_recall=0.0000 val_fe_recall=0.6829 val_joint_bal_acc=0.5306 val_joint_macro_f1=0.4991 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.9800 lr=4.75232e-05 train_metal_acc=0.7104 val_loss=1.2211 val_metal_acc=0.6106 val_metal_min_recall=0.0769 val_fe_recall=0.8293 val_joint_bal_acc=0.5222 val_joint_macro_f1=0.4906 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.9591 lr=4.75232e-05 train_metal_acc=0.7350 val_loss=1.2266 val_metal_acc=0.6106 val_metal_min_recall=0.0000 val_fe_recall=0.7805 val_joint_bal_acc=0.5395 val_joint_macro_f1=0.4942 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.9460 lr=4.75232e-05 train_metal_acc=0.7451 val_loss=1.2497 val_metal_acc=0.5481 val_metal_min_recall=0.0000 val_fe_recall=0.6341 val_joint_bal_acc=0.5136 val_joint_macro_f1=0.4889 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.9321 lr=4.75232e-05 train_metal_acc=0.7494 val_loss=1.3177 val_metal_acc=0.5000 val_metal_min_recall=0.0000 val_fe_recall=0.6829 val_joint_bal_acc=0.5043 val_joint_macro_f1=0.4697 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.9341 lr=4.75232e-05 train_metal_acc=0.7460 val_loss=1.2808 val_metal_acc=0.5481 val_metal_min_recall=0.0769 val_fe_recall=0.6341 val_joint_bal_acc=0.5120 val_joint_macro_f1=0.4931 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.8942 lr=4.75232e-05 train_metal_acc=0.7578 val_loss=1.2684 val_metal_acc=0.5769 val_metal_min_recall=0.0000 val_fe_recall=0.8049 val_joint_bal_acc=0.5585 val_joint_macro_f1=0.5222 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.8925 lr=4.75232e-05 train_metal_acc=0.7663 val_loss=1.2947 val_metal_acc=0.5529 val_metal_min_recall=0.0769 val_fe_recall=0.8293 val_joint_bal_acc=0.5277 val_joint_macro_f1=0.5149 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.8469 lr=4.75232e-05 train_metal_acc=0.7697 val_loss=1.2370 val_metal_acc=0.6154 val_metal_min_recall=0.0769 val_fe_recall=0.6341 val_joint_bal_acc=0.5541 val_joint_macro_f1=0.5283 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.8546 lr=4.75232e-05 train_metal_acc=0.7621 val_loss=1.2463 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.7073 val_joint_bal_acc=0.5985 val_joint_macro_f1=0.5563 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.8513 lr=4.75232e-05 train_metal_acc=0.7401 val_loss=1.2728 val_metal_acc=0.5817 val_metal_min_recall=0.0769 val_fe_recall=0.8537 val_joint_bal_acc=0.5451 val_joint_macro_f1=0.5182 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.8114 lr=4.75232e-05 train_metal_acc=0.7798 val_loss=1.3047 val_metal_acc=0.5192 val_metal_min_recall=0.0000 val_fe_recall=0.6585 val_joint_bal_acc=0.5024 val_joint_macro_f1=0.4811 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.7960 lr=4.75232e-05 train_metal_acc=0.7900 val_loss=1.2809 val_metal_acc=0.6106 val_metal_min_recall=0.0000 val_fe_recall=0.8049 val_joint_bal_acc=0.5504 val_joint_macro_f1=0.4988 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.7930 lr=4.75232e-05 train_metal_acc=0.7688 val_loss=1.2585 val_metal_acc=0.5817 val_metal_min_recall=0.0769 val_fe_recall=0.6585 val_joint_bal_acc=0.5359 val_joint_macro_f1=0.5162 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.7751 lr=4.75232e-05 train_metal_acc=0.7925 val_loss=1.3009 val_metal_acc=0.5913 val_metal_min_recall=0.0769 val_fe_recall=0.8537 val_joint_bal_acc=0.5588 val_joint_macro_f1=0.5439 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.7628 lr=4.75232e-05 train_metal_acc=0.8078 val_loss=1.2396 val_metal_acc=0.6298 val_metal_min_recall=0.0769 val_fe_recall=0.8049 val_joint_bal_acc=0.5939 val_joint_macro_f1=0.5562 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.7450 lr=4.75232e-05 train_metal_acc=0.7595 val_loss=1.3786 val_metal_acc=0.5721 val_metal_min_recall=0.0769 val_fe_recall=0.8293 val_joint_bal_acc=0.5464 val_joint_macro_f1=0.5219 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.7192 lr=4.75232e-05 train_metal_acc=0.7951 val_loss=1.2341 val_metal_acc=0.6250 val_metal_min_recall=0.0769 val_fe_recall=0.6585 val_joint_bal_acc=0.5599 val_joint_macro_f1=0.5308 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.7095 lr=4.75232e-05 train_metal_acc=0.8146 val_loss=1.1862 val_metal_acc=0.6394 val_metal_min_recall=0.0769 val_fe_recall=0.8049 val_joint_bal_acc=0.5823 val_joint_macro_f1=0.5525 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.6996 lr=4.75232e-05 train_metal_acc=0.7587 val_loss=1.4326 val_metal_acc=0.5433 val_metal_min_recall=0.0769 val_fe_recall=0.6341 val_joint_bal_acc=0.5443 val_joint_macro_f1=0.5191 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.6881 lr=4.75232e-05 train_metal_acc=0.8044 val_loss=1.3813 val_metal_acc=0.5865 val_metal_min_recall=0.1538 val_fe_recall=0.7561 val_joint_bal_acc=0.5762 val_joint_macro_f1=0.5413 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.6776 lr=4.75232e-05 train_metal_acc=0.8036 val_loss=1.3081 val_metal_acc=0.6106 val_metal_min_recall=0.0769 val_fe_recall=0.6585 val_joint_bal_acc=0.5611 val_joint_macro_f1=0.5332 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.6593 lr=4.75232e-05 train_metal_acc=0.8213 val_loss=1.2704 val_metal_acc=0.6058 val_metal_min_recall=0.0769 val_fe_recall=0.7561 val_joint_bal_acc=0.5593 val_joint_macro_f1=0.5403 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.6571 lr=4.75232e-05 train_metal_acc=0.7892 val_loss=1.2463 val_metal_acc=0.5913 val_metal_min_recall=0.0769 val_fe_recall=0.5854 val_joint_bal_acc=0.5504 val_joint_macro_f1=0.5301 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.6314 lr=4.75232e-05 train_metal_acc=0.8112 val_loss=1.2650 val_metal_acc=0.6202 val_metal_min_recall=0.0769 val_fe_recall=0.6829 val_joint_bal_acc=0.5669 val_joint_macro_f1=0.5254 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=41 train_loss=0.6205 lr=4.75232e-05 train_metal_acc=0.7875 val_loss=1.2521 val_metal_acc=0.6346 val_metal_min_recall=0.0000 val_fe_recall=0.8780 val_joint_bal_acc=0.5915 val_joint_macro_f1=0.5466 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=42 train_loss=0.5955 lr=4.75232e-05 train_metal_acc=0.8264 val_loss=1.2139 val_metal_acc=0.6587 val_metal_min_recall=0.0769 val_fe_recall=0.7805 val_joint_bal_acc=0.6136 val_joint_macro_f1=0.5665 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=43 train_loss=0.5923 lr=4.75232e-05 train_metal_acc=0.8417 val_loss=1.2377 val_metal_acc=0.6250 val_metal_min_recall=0.0769 val_fe_recall=0.7317 val_joint_bal_acc=0.5702 val_joint_macro_f1=0.5412 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=44 train_loss=0.5805 lr=4.75232e-05 train_metal_acc=0.8188 val_loss=1.2525 val_metal_acc=0.6442 val_metal_min_recall=0.0769 val_fe_recall=0.9024 val_joint_bal_acc=0.5594 val_joint_macro_f1=0.5555 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=45 train_loss=0.5689 lr=4.75232e-05 train_metal_acc=0.8688 val_loss=1.2162 val_metal_acc=0.6490 val_metal_min_recall=0.0769 val_fe_recall=0.8293 val_joint_bal_acc=0.5786 val_joint_macro_f1=0.5516 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=46 train_loss=0.5748 lr=4.75232e-05 train_metal_acc=0.8561 val_loss=1.2034 val_metal_acc=0.6346 val_metal_min_recall=0.0769 val_fe_recall=0.7805 val_joint_bal_acc=0.5783 val_joint_macro_f1=0.5521 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=47 train_loss=0.5318 lr=4.75232e-05 train_metal_acc=0.8645 val_loss=1.2908 val_metal_acc=0.6250 val_metal_min_recall=0.0769 val_fe_recall=0.9024 val_joint_bal_acc=0.5898 val_joint_macro_f1=0.5782 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=48 train_loss=0.5442 lr=4.75232e-05 train_metal_acc=0.8476 val_loss=1.2557 val_metal_acc=0.6442 val_metal_min_recall=0.0769 val_fe_recall=0.6585 val_joint_bal_acc=0.5604 val_joint_macro_f1=0.5381 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=49 train_loss=0.5135 lr=4.75232e-05 train_metal_acc=0.8561 val_loss=1.2218 val_metal_acc=0.6490 val_metal_min_recall=0.0769 val_fe_recall=0.7561 val_joint_bal_acc=0.5811 val_joint_macro_f1=0.5434 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=50 train_loss=0.5155 lr=4.75232e-05 train_metal_acc=0.8713 val_loss=1.2591 val_metal_acc=0.6490 val_metal_min_recall=0.0769 val_fe_recall=0.7317 val_joint_bal_acc=0.6023 val_joint_macro_f1=0.5504 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_edcdcf3a/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_edcdcf3a/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_edcdcf3a/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_edcdcf3a/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_edcdcf3a/run_metadata.json
Completed: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_edcdcf3a
================================================================================
[#005 | 5/5] deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_60d10046
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat --run-name deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_60d10046 --model-architecture only_gvp --epochs 50 --batch-size 8 --learning-rate 4.752317377508605e-05 --weight-decay 0.0 --seed 44 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 128 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_sqrt_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges
stdout log: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/_execution_logs/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_60d10046.stdout.log
stderr log: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/_execution_logs/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_60d10046.stderr.log
================================================================================

=== Split diagnostics (passive; training behavior unchanged) ===
task=metal split_by=pdbid val_fraction=0.15 n_folds=None fold_index=None
pockets: train=1181 validation=208
groups by pdbid: train=1152 validation=109
train/validation overlap counts: pdbid=0, pdbid_chain=0, structure_id=0, pocket_id=0
train metal distribution: Mn=551, Cu=70, Zn=173, Fe=250, Co=73, Ni=64
validation metal distribution: Mn=97, Cu=15, Zn=34, Fe=42, Co=13, Ni=7
missing train metal classes: none
missing validation metal classes: none
train EC distribution: 1=489, 2=200, 3=339, 4=48, 5=73, 6=15, 7=1
validation EC distribution: 1=70, 2=15, 3=53, 4=6, 5=60, 6=2, 7=0
missing train EC classes: none
missing validation EC classes: 7
===============================================================

epoch=1 train_loss=1.6995 lr=4.75232e-05 train_metal_acc=0.4877 val_loss=1.5997 val_metal_acc=0.5048 val_metal_min_recall=0.0000 val_fe_recall=0.1905 val_joint_bal_acc=0.1984 val_joint_macro_f1=0.1615 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.6550 lr=4.75232e-05 train_metal_acc=0.4962 val_loss=1.5692 val_metal_acc=0.5240 val_metal_min_recall=0.0000 val_fe_recall=0.2857 val_joint_bal_acc=0.2143 val_joint_macro_f1=0.1780 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.5856 lr=4.75232e-05 train_metal_acc=0.5047 val_loss=1.4817 val_metal_acc=0.6298 val_metal_min_recall=0.0000 val_fe_recall=0.8095 val_joint_bal_acc=0.3767 val_joint_macro_f1=0.3128 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.4597 lr=4.75232e-05 train_metal_acc=0.5436 val_loss=1.2972 val_metal_acc=0.6875 val_metal_min_recall=0.0000 val_fe_recall=0.9286 val_joint_bal_acc=0.4371 val_joint_macro_f1=0.4192 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.3747 lr=4.75232e-05 train_metal_acc=0.5910 val_loss=1.2393 val_metal_acc=0.6683 val_metal_min_recall=0.0000 val_fe_recall=0.8333 val_joint_bal_acc=0.4492 val_joint_macro_f1=0.3879 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.3096 lr=4.75232e-05 train_metal_acc=0.6257 val_loss=1.2259 val_metal_acc=0.6154 val_metal_min_recall=0.0000 val_fe_recall=0.7143 val_joint_bal_acc=0.4699 val_joint_macro_f1=0.4072 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=1.2620 lr=4.75232e-05 train_metal_acc=0.6503 val_loss=1.1383 val_metal_acc=0.6779 val_metal_min_recall=0.0000 val_fe_recall=0.8333 val_joint_bal_acc=0.5033 val_joint_macro_f1=0.4753 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=1.2099 lr=4.75232e-05 train_metal_acc=0.6715 val_loss=1.1585 val_metal_acc=0.6394 val_metal_min_recall=0.0000 val_fe_recall=0.7857 val_joint_bal_acc=0.4882 val_joint_macro_f1=0.4459 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=1.1827 lr=4.75232e-05 train_metal_acc=0.6655 val_loss=1.2118 val_metal_acc=0.5529 val_metal_min_recall=0.0000 val_fe_recall=0.5476 val_joint_bal_acc=0.4318 val_joint_macro_f1=0.3738 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=1.1508 lr=4.75232e-05 train_metal_acc=0.6715 val_loss=1.1352 val_metal_acc=0.6154 val_metal_min_recall=0.0000 val_fe_recall=0.7143 val_joint_bal_acc=0.4506 val_joint_macro_f1=0.3954 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=1.1223 lr=4.75232e-05 train_metal_acc=0.7036 val_loss=1.1471 val_metal_acc=0.5962 val_metal_min_recall=0.0000 val_fe_recall=0.6905 val_joint_bal_acc=0.4860 val_joint_macro_f1=0.4499 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=1.0921 lr=4.75232e-05 train_metal_acc=0.7079 val_loss=1.0834 val_metal_acc=0.6394 val_metal_min_recall=0.0000 val_fe_recall=0.7857 val_joint_bal_acc=0.4978 val_joint_macro_f1=0.4721 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=1.0634 lr=4.75232e-05 train_metal_acc=0.7121 val_loss=1.2270 val_metal_acc=0.5721 val_metal_min_recall=0.0769 val_fe_recall=0.3810 val_joint_bal_acc=0.3935 val_joint_macro_f1=0.4145 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=1.0464 lr=4.75232e-05 train_metal_acc=0.7248 val_loss=1.1279 val_metal_acc=0.6298 val_metal_min_recall=0.0000 val_fe_recall=0.7143 val_joint_bal_acc=0.5475 val_joint_macro_f1=0.5274 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=1.0061 lr=4.75232e-05 train_metal_acc=0.7163 val_loss=1.1273 val_metal_acc=0.6442 val_metal_min_recall=0.0000 val_fe_recall=0.7381 val_joint_bal_acc=0.5644 val_joint_macro_f1=0.5183 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.9871 lr=4.75232e-05 train_metal_acc=0.6571 val_loss=1.3842 val_metal_acc=0.5048 val_metal_min_recall=0.0000 val_fe_recall=0.8571 val_joint_bal_acc=0.4755 val_joint_macro_f1=0.4212 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.9682 lr=4.75232e-05 train_metal_acc=0.7130 val_loss=1.1713 val_metal_acc=0.6250 val_metal_min_recall=0.0769 val_fe_recall=0.5952 val_joint_bal_acc=0.5330 val_joint_macro_f1=0.4998 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.9903 lr=4.75232e-05 train_metal_acc=0.7282 val_loss=1.2128 val_metal_acc=0.5240 val_metal_min_recall=0.0769 val_fe_recall=0.4524 val_joint_bal_acc=0.3923 val_joint_macro_f1=0.4197 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.9546 lr=4.75232e-05 train_metal_acc=0.7528 val_loss=1.2002 val_metal_acc=0.5769 val_metal_min_recall=0.0000 val_fe_recall=0.5952 val_joint_bal_acc=0.5396 val_joint_macro_f1=0.5209 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.9320 lr=4.75232e-05 train_metal_acc=0.7587 val_loss=1.2004 val_metal_acc=0.5673 val_metal_min_recall=0.0769 val_fe_recall=0.5476 val_joint_bal_acc=0.5238 val_joint_macro_f1=0.5059 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.8905 lr=4.75232e-05 train_metal_acc=0.7561 val_loss=1.2104 val_metal_acc=0.5721 val_metal_min_recall=0.0769 val_fe_recall=0.5238 val_joint_bal_acc=0.5391 val_joint_macro_f1=0.5104 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.8887 lr=4.75232e-05 train_metal_acc=0.7604 val_loss=1.1420 val_metal_acc=0.6154 val_metal_min_recall=0.0769 val_fe_recall=0.6429 val_joint_bal_acc=0.5562 val_joint_macro_f1=0.5503 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.8531 lr=4.75232e-05 train_metal_acc=0.7409 val_loss=1.3314 val_metal_acc=0.5144 val_metal_min_recall=0.0000 val_fe_recall=0.3333 val_joint_bal_acc=0.4925 val_joint_macro_f1=0.4643 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.8590 lr=4.75232e-05 train_metal_acc=0.7223 val_loss=1.4949 val_metal_acc=0.4663 val_metal_min_recall=0.0000 val_fe_recall=0.6667 val_joint_bal_acc=0.5164 val_joint_macro_f1=0.4814 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.8323 lr=4.75232e-05 train_metal_acc=0.7773 val_loss=1.3403 val_metal_acc=0.5192 val_metal_min_recall=0.0769 val_fe_recall=0.7143 val_joint_bal_acc=0.5445 val_joint_macro_f1=0.5201 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.8200 lr=4.75232e-05 train_metal_acc=0.7756 val_loss=1.2395 val_metal_acc=0.5913 val_metal_min_recall=0.0769 val_fe_recall=0.6190 val_joint_bal_acc=0.5454 val_joint_macro_f1=0.5086 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.8003 lr=4.75232e-05 train_metal_acc=0.7875 val_loss=1.3888 val_metal_acc=0.5192 val_metal_min_recall=0.0769 val_fe_recall=0.7143 val_joint_bal_acc=0.5477 val_joint_macro_f1=0.5383 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.7672 lr=4.75232e-05 train_metal_acc=0.8002 val_loss=1.2385 val_metal_acc=0.6298 val_metal_min_recall=0.0769 val_fe_recall=0.7619 val_joint_bal_acc=0.5854 val_joint_macro_f1=0.5944 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.7703 lr=4.75232e-05 train_metal_acc=0.8129 val_loss=1.2973 val_metal_acc=0.5913 val_metal_min_recall=0.0769 val_fe_recall=0.7619 val_joint_bal_acc=0.5716 val_joint_macro_f1=0.5644 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.7504 lr=4.75232e-05 train_metal_acc=0.7934 val_loss=1.2520 val_metal_acc=0.5817 val_metal_min_recall=0.0769 val_fe_recall=0.5952 val_joint_bal_acc=0.5397 val_joint_macro_f1=0.4960 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.7371 lr=4.75232e-05 train_metal_acc=0.7959 val_loss=1.2897 val_metal_acc=0.5625 val_metal_min_recall=0.0769 val_fe_recall=0.5952 val_joint_bal_acc=0.5296 val_joint_macro_f1=0.5209 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.7172 lr=4.75232e-05 train_metal_acc=0.8027 val_loss=1.3880 val_metal_acc=0.5337 val_metal_min_recall=0.0000 val_fe_recall=0.5476 val_joint_bal_acc=0.5196 val_joint_macro_f1=0.4935 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.7205 lr=4.75232e-05 train_metal_acc=0.8154 val_loss=1.2543 val_metal_acc=0.5721 val_metal_min_recall=0.0769 val_fe_recall=0.6190 val_joint_bal_acc=0.5385 val_joint_macro_f1=0.5072 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.6884 lr=4.75232e-05 train_metal_acc=0.8103 val_loss=1.3384 val_metal_acc=0.5913 val_metal_min_recall=0.0000 val_fe_recall=0.8571 val_joint_bal_acc=0.5631 val_joint_macro_f1=0.5553 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.6831 lr=4.75232e-05 train_metal_acc=0.8112 val_loss=1.2565 val_metal_acc=0.6106 val_metal_min_recall=0.0769 val_fe_recall=0.5476 val_joint_bal_acc=0.5551 val_joint_macro_f1=0.5020 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.6662 lr=4.75232e-05 train_metal_acc=0.8103 val_loss=1.3358 val_metal_acc=0.5625 val_metal_min_recall=0.0769 val_fe_recall=0.5000 val_joint_bal_acc=0.5523 val_joint_macro_f1=0.5077 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.6589 lr=4.75232e-05 train_metal_acc=0.8205 val_loss=1.2518 val_metal_acc=0.6250 val_metal_min_recall=0.0769 val_fe_recall=0.7857 val_joint_bal_acc=0.5668 val_joint_macro_f1=0.5501 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.6427 lr=4.75232e-05 train_metal_acc=0.8239 val_loss=1.3823 val_metal_acc=0.4904 val_metal_min_recall=0.0769 val_fe_recall=0.5952 val_joint_bal_acc=0.5262 val_joint_macro_f1=0.5010 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.6307 lr=4.75232e-05 train_metal_acc=0.8340 val_loss=1.3003 val_metal_acc=0.6010 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5820 val_joint_macro_f1=0.5498 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.6203 lr=4.75232e-05 train_metal_acc=0.8307 val_loss=1.2598 val_metal_acc=0.6587 val_metal_min_recall=0.0769 val_fe_recall=0.8095 val_joint_bal_acc=0.5970 val_joint_macro_f1=0.5852 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=41 train_loss=0.6100 lr=4.75232e-05 train_metal_acc=0.8332 val_loss=1.1992 val_metal_acc=0.6298 val_metal_min_recall=0.0769 val_fe_recall=0.5952 val_joint_bal_acc=0.5569 val_joint_macro_f1=0.5140 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=42 train_loss=0.6019 lr=4.75232e-05 train_metal_acc=0.8273 val_loss=1.2780 val_metal_acc=0.5865 val_metal_min_recall=0.0769 val_fe_recall=0.7381 val_joint_bal_acc=0.5552 val_joint_macro_f1=0.5615 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=43 train_loss=0.5761 lr=4.75232e-05 train_metal_acc=0.7782 val_loss=1.5497 val_metal_acc=0.5433 val_metal_min_recall=0.0000 val_fe_recall=0.9286 val_joint_bal_acc=0.4894 val_joint_macro_f1=0.4768 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=44 train_loss=0.5783 lr=4.75232e-05 train_metal_acc=0.8434 val_loss=1.4366 val_metal_acc=0.5673 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5574 val_joint_macro_f1=0.5550 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=45 train_loss=0.5708 lr=4.75232e-05 train_metal_acc=0.8552 val_loss=1.3067 val_metal_acc=0.5962 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5551 val_joint_macro_f1=0.5445 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=46 train_loss=0.5532 lr=4.75232e-05 train_metal_acc=0.8561 val_loss=1.1932 val_metal_acc=0.6250 val_metal_min_recall=0.0000 val_fe_recall=0.6667 val_joint_bal_acc=0.5699 val_joint_macro_f1=0.5380 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=47 train_loss=0.5299 lr=4.75232e-05 train_metal_acc=0.8577 val_loss=1.2208 val_metal_acc=0.6346 val_metal_min_recall=0.0769 val_fe_recall=0.7857 val_joint_bal_acc=0.5607 val_joint_macro_f1=0.5451 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=48 train_loss=0.5284 lr=4.75232e-05 train_metal_acc=0.8696 val_loss=1.3074 val_metal_acc=0.6202 val_metal_min_recall=0.0000 val_fe_recall=0.6667 val_joint_bal_acc=0.5650 val_joint_macro_f1=0.5314 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=49 train_loss=0.5052 lr=4.75232e-05 train_metal_acc=0.8569 val_loss=1.4091 val_metal_acc=0.5385 val_metal_min_recall=0.0769 val_fe_recall=0.5952 val_joint_bal_acc=0.5273 val_joint_macro_f1=0.4935 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=50 train_loss=0.4840 lr=4.75232e-05 train_metal_acc=0.8730 val_loss=1.2323 val_metal_acc=0.6106 val_metal_min_recall=0.0769 val_fe_recall=0.5952 val_joint_bal_acc=0.5564 val_joint_macro_f1=0.5477 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_60d10046/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_60d10046/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_60d10046/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_60d10046/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_60d10046/run_metadata.json
Completed: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_60d10046
Completed run directories: ['/content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_2c77cc48', '/content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_1a1fd84a', '/content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_14db2a0a', '/content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_edcdcf3a', '/content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_baseline_batchmetal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat_metal_only_gvp_archonly_gvp_fusionnone_rin_60d10046']
Failed run directories: []
Execution records JSON: /content/deepmzyme_outputs/runs/metal_only_gvp_anchor_trial12_gvp2_50epoch_seedrepeat/deepmzyme_nonoverlap_model_comparison_execution_records.json




