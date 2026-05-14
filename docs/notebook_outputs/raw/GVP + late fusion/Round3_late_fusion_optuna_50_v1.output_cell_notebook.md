Configured output locations:
  Runs root:       /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1  [exists]
  Summary CSV:     /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/deepmzyme_nonoverlap_model_comparison.csv  [exists]
  Summary figure:  /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/deepmzyme_nonoverlap_model_comparison.png  [exists]

Completed run directories found: 65
choice_index	run_name	task	model	fusion	seed	learning_rate	weight_decay	selection_metric	metric_direction	best_validation_value	test_metric	test_metric_value	selected_epoch	test_report_saved	run_dir
0	1	optuna_deepmzyme_controlled_hpo_trial0049_deep...	metal	GVP + ESM late fusion	late_fusion	42	0.000017	0.00001	val_metal_balanced_acc	higher_is_better	0.675013	test_metal_balanced_acc	None	37	False	/content/deepmzyme_outputs/runs/metal_late_fus...
1	2	optuna_deepmzyme_controlled_hpo_trial0032_deep...	metal	GVP + ESM late fusion	late_fusion	42	0.000055	0.00100	val_metal_balanced_acc	higher_is_better	0.658512	test_metal_balanced_acc	None	22	False	/content/deepmzyme_outputs/runs/metal_late_fus...
2	3	optuna_deepmzyme_controlled_hpo_trial0015_deep...	metal	GVP + ESM late fusion	late_fusion	42	0.000070	0.00100	val_metal_balanced_acc	higher_is_better	0.655096	test_metal_balanced_acc	None	22	False	/content/deepmzyme_outputs/runs/metal_late_fus...
3	4	optuna_deepmzyme_controlled_hpo_trial0043_deep...	metal	GVP + ESM late fusion	late_fusion	42	0.000066	0.00100	val_metal_balanced_acc	higher_is_better	0.655096	test_metal_balanced_acc	None	22	False	/content/deepmzyme_outputs/runs/metal_late_fus...
4	5	optuna_deepmzyme_controlled_hpo_trial0044_deep...	metal	GVP + ESM late fusion	late_fusion	42	0.000069	0.00100	val_metal_balanced_acc	higher_is_better	0.655096	test_metal_balanced_acc	None	22	False	/content/deepmzyme_outputs/runs/metal_late_fus...
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
60	61	top2_trial32_deepmzyme_controlled_hpo_seed2026...	metal	GVP + ESM late fusion	late_fusion	2026	0.000055	0.00100	val_metal_balanced_acc	higher_is_better	0.309287	test_metal_balanced_acc	None	1	False	/content/deepmzyme_outputs/runs/metal_late_fus...
61	62	top2_trial32_deepmzyme_controlled_hpo_seed44_d...	metal	GVP + ESM late fusion	late_fusion	44	0.000055	0.00100	val_metal_balanced_acc	higher_is_better	0.309064	test_metal_balanced_acc	None	1	False	/content/deepmzyme_outputs/runs/metal_late_fus...
62	63	top1_trial49_deepmzyme_controlled_hpo_seed123_...	metal	GVP + ESM late fusion	late_fusion	123	0.000017	0.00001	val_metal_balanced_acc	higher_is_better	0.301330	test_metal_balanced_acc	None	1	False	/content/deepmzyme_outputs/runs/metal_late_fus...
63	64	top3_trial15_deepmzyme_controlled_hpo_seed123_...	metal	GVP + ESM late fusion	late_fusion	123	0.000070	0.00100	val_metal_balanced_acc	higher_is_better	0.272707	test_metal_balanced_acc	None	1	False	/content/deepmzyme_outputs/runs/metal_late_fus...
64	65	top1_trial49_deepmzyme_controlled_hpo_seed2026...	metal	GVP + ESM late fusion	late_fusion	2026	0.000017	0.00001	val_metal_balanced_acc	higher_is_better	0.247059	test_metal_balanced_acc	None	1	False	/content/deepmzyme_outputs/runs/metal_late_fus...
65 rows × 16 columns




Selected final run: /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0049_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d32997f7
  Selection mode: auto_best_validation
  Task:         metal
  Architecture: gvp
  Fusion:       late_fusion
  Seed:         42
  Best val val_metal_balanced_acc: 0.6750  (epoch 37)
  Split:        train_and_test_sets_structures_non_overlapped_pinmymetal
Configured output locations:
  Runs root:       /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1  [exists]
  Summary CSV:     /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/deepmzyme_nonoverlap_model_comparison.csv  [exists]
  Summary figure:  /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/deepmzyme_nonoverlap_model_comparison.png  [exists]
  Selected run:    /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0049_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d32997f7  [exists]
  Run config:      /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0049_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d32997f7/run_config.json  [exists]
  Run metadata:    /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0049_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d32997f7/run_metadata.json  [exists]
  Test report:     /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0049_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d32997f7/test_report.json  [not created yet]
  Selection JSON: /content/deepmzyme_outputs/runs/deepmzyme_final_selected_run.json

No test_report.json found for the selected run.
Use the optional final held-out test evaluation cell next; its default mode evaluates the selected saved checkpoint without retraining.
Keep choosing models by validation metrics; use held-out test metrics only for the selected final run.





#-------------------------------

Summary scanning scope: current RUN_BATCH_ID folder
RUN_BATCH_ID: metal_late_fusion_optuna_50_v1
Runs directory scanned: /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1
/usr/bin/python3 /content/DeepMzyme/src/report_runs.py --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --out-csv /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/deepmzyme_nonoverlap_model_comparison_completed_only.csv --out-figure /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/deepmzyme_nonoverlap_model_comparison.png
Completed-run summary CSV: /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/deepmzyme_nonoverlap_model_comparison_completed_only.csv
Summary source mode: planned table from current notebook state
Summary source scope: current planned rows plus completed runs under the scanned directory.
STRONG WARNING: This summary may mix old runs from the same RUNS_ROOT. Do not use this mixed table for model selection unless every scanned run is intentionally part of the same comparison.
Comparison CSV: /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/deepmzyme_nonoverlap_model_comparison.csv
rank	source_mode	config_source	result_stage	run_name	status	error_message	model_preset	model_display	model_architecture	...	missing_train_metal_classes	missing_val_metal_classes	missing_train_ec_classes	missing_val_ec_classes	selected_best_validation_metric_value	held_out_test_metric_name	held_out_test_metric_value	run_dir	stdout_log_path	stderr_log_path
0	1.0	completed-only scan	scanned run directory	validation-only	optuna_deepmzyme_controlled_hpo_trial0049_deep...	completed		gvp	GVP + ESM late_fusion	gvp	...	[]	[]	[]	[4, 6]	0.675013	test_metal_balanced_acc	None	/content/deepmzyme_outputs/runs/metal_late_fus...	None	None
1	2.0	completed-only scan	scanned run directory	validation-only	optuna_deepmzyme_controlled_hpo_trial0032_deep...	completed		gvp	GVP + ESM late_fusion	gvp	...	[]	[]	[]	[4, 6]	0.658512	test_metal_balanced_acc	None	/content/deepmzyme_outputs/runs/metal_late_fus...	None	None
2	3.0	completed-only scan	scanned run directory	validation-only	optuna_deepmzyme_controlled_hpo_trial0015_deep...	completed		gvp	GVP + ESM late_fusion	gvp	...	[]	[]	[]	[4, 6]	0.655096	test_metal_balanced_acc	None	/content/deepmzyme_outputs/runs/metal_late_fus...	None	None
3	4.0	completed-only scan	scanned run directory	validation-only	optuna_deepmzyme_controlled_hpo_trial0043_deep...	completed		gvp	GVP + ESM late_fusion	gvp	...	[]	[]	[]	[4, 6]	0.655096	test_metal_balanced_acc	None	/content/deepmzyme_outputs/runs/metal_late_fus...	None	None
4	5.0	completed-only scan	scanned run directory	validation-only	optuna_deepmzyme_controlled_hpo_trial0044_deep...	completed		gvp	GVP + ESM late_fusion	gvp	...	[]	[]	[]	[4, 6]	0.655096	test_metal_balanced_acc	None	/content/deepmzyme_outputs/runs/metal_late_fus...	None	None
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
61	62.0	completed-only scan	scanned run directory	validation-only	top2_trial32_deepmzyme_controlled_hpo_seed44_d...	completed		gvp	GVP + ESM late_fusion	gvp	...	[]	[]	[]	[]	0.309064	test_metal_balanced_acc	None	/content/deepmzyme_outputs/runs/metal_late_fus...	None	None
62	63.0	completed-only scan	scanned run directory	validation-only	top1_trial49_deepmzyme_controlled_hpo_seed123_...	completed		gvp	GVP + ESM late_fusion	gvp	...	[]	[]	[]	[6]	0.301330	test_metal_balanced_acc	None	/content/deepmzyme_outputs/runs/metal_late_fus...	None	None
63	64.0	completed-only scan	scanned run directory	validation-only	top3_trial15_deepmzyme_controlled_hpo_seed123_...	completed		gvp	GVP + ESM late_fusion	gvp	...	[]	[]	[]	[6]	0.272707	test_metal_balanced_acc	None	/content/deepmzyme_outputs/runs/metal_late_fus...	None	None
64	65.0	completed-only scan	scanned run directory	validation-only	top1_trial49_deepmzyme_controlled_hpo_seed2026...	completed		gvp	GVP + ESM late_fusion	gvp	...	[]	[]	[]	[]	0.247059	test_metal_balanced_acc	None	/content/deepmzyme_outputs/runs/metal_late_fus...	None	None
65	NaN	planned table	current notebook config	validation-only	deepmzyme_nonoverlap_baseline_batchmetal_late_...	planned		GVP + late fusion	GVP + ESM late_fusion	gvp	...	NaN	NaN	NaN	NaN	NaN	None	None	/content/deepmzyme_outputs/runs/metal_late_fus...	/content/deepmzyme_outputs/runs/metal_late_fus...	/content/deepmzyme_outputs/runs/metal_late_fus...
66 rows × 63 columns



STRONG WARNING: Mixed or missing RUN_BATCH_ID values were found in the summary table: ['', 'metal_late_fusion_optuna_50_v1']

Ranked table sorted by validation selection metric:
#1: optuna_deepmzyme_controlled_hpo_trial0049_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d32997f7 | stage=validation-only | class_weight=inverse_frequency | val_metal_balanced_acc=0.6750130535709283 | status=completed
#2: optuna_deepmzyme_controlled_hpo_trial0032_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_faf01e7c | stage=validation-only | class_weight=inverse_frequency | val_metal_balanced_acc=0.6585119076580177 | status=completed
#3: optuna_deepmzyme_controlled_hpo_trial0015_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a931e009 | stage=validation-only | class_weight=inverse_frequency | val_metal_balanced_acc=0.6550963478857217 | status=completed
#4: optuna_deepmzyme_controlled_hpo_trial0043_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_8d03d6e9 | stage=validation-only | class_weight=inverse_frequency | val_metal_balanced_acc=0.6550963478857217 | status=completed
#5: optuna_deepmzyme_controlled_hpo_trial0044_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_2fd12175 | stage=validation-only | class_weight=inverse_frequency | val_metal_balanced_acc=0.6550963478857217 | status=completed
#6: optuna_deepmzyme_controlled_hpo_trial0016_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a973315e | stage=validation-only | class_weight=inverse_frequency | val_metal_balanced_acc=0.6531355635719962 | status=completed
#7: optuna_deepmzyme_controlled_hpo_trial0025_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_5305cf54 | stage=validation-only | class_weight=inverse_frequency | val_metal_balanced_acc=0.6530285232372519 | status=completed
#8: optuna_deepmzyme_controlled_hpo_trial0008_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_93d46ebc | stage=validation-only | class_weight=inverse_frequency | val_metal_balanced_acc=0.6529553937523577 | status=completed
#9: optuna_deepmzyme_controlled_hpo_trial0022_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_abf5e755 | stage=validation-only | class_weight=inverse_frequency | val_metal_balanced_acc=0.6529553937523577 | status=completed
#10: optuna_deepmzyme_controlled_hpo_trial0023_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_f32e42f6 | stage=validation-only | class_weight=inverse_frequency | val_metal_balanced_acc=0.6529553937523577 | status=completed
#11: optuna_deepmzyme_controlled_hpo_trial0031_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c145b928 | stage=validation-only | class_weight=inverse_frequency | val_metal_balanced_acc=0.6529553937523577 | status=completed
#12: optuna_deepmzyme_controlled_hpo_trial0042_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_8e2c2aa2 | stage=validation-only | class_weight=inverse_frequency | val_metal_balanced_acc=0.6529553937523577 | status=completed
#13: optuna_deepmzyme_controlled_hpo_trial0037_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_19d4f091 | stage=validation-only | class_weight=inverse_frequency | val_metal_balanced_acc=0.6525003775478159 | status=completed
#14: optuna_deepmzyme_controlled_hpo_trial0021_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_87949ab6 | stage=validation-only | class_weight=inverse_frequency | val_metal_balanced_acc=0.6524493848972027 | status=completed
#15: optuna_deepmzyme_controlled_hpo_trial0041_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_e1dec6d2 | stage=validation-only | class_weight=inverse_frequency | val_metal_balanced_acc=0.6524493848972027 | status=completed
#16: optuna_deepmzyme_controlled_hpo_trial0018_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_98f0fd36 | stage=validation-only | class_weight=inverse_sqrt_frequency | val_metal_balanced_acc=0.6513935779590428 | status=completed
#17: optuna_deepmzyme_controlled_hpo_trial0020_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_04f4992c | stage=validation-only | class_weight=inverse_frequency | val_metal_balanced_acc=0.6511747792582707 | status=completed
#18: optuna_deepmzyme_controlled_hpo_trial0027_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_e110d321 | stage=validation-only | class_weight=inverse_frequency | val_metal_balanced_acc=0.6479780181867468 | status=completed
#19: optuna_deepmzyme_controlled_hpo_trial0046_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_2627ca5a | stage=validation-only | class_weight=inverse_sqrt_frequency | val_metal_balanced_acc=0.6469561220984371 | status=completed
#20: optuna_deepmzyme_controlled_hpo_trial0017_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_0e910c6b | stage=validation-only | class_weight=inverse_frequency | val_metal_balanced_acc=0.6456913948375049 | status=completed

Best overall configuration: optuna_deepmzyme_controlled_hpo_trial0049_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d32997f7
{
  "run_name": "optuna_deepmzyme_controlled_hpo_trial0049_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d32997f7",
  "result_stage": "validation-only",
  "model_preset": "gvp",
  "model_architecture": "gvp",
  "fusion_mode": "late_fusion",
  "metal_class_weight_mode": "inverse_frequency",
  "balance_metal_site_symbols": false,
  "selection_metric": "val_metal_balanced_acc",
  "selected_best_validation_metric_value": 0.6750130535709283,
  "run_dir": "/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0049_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d32997f7"
}

Best configuration per model preset/mode:
gvp: optuna_deepmzyme_controlled_hpo_trial0049_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d32997f7 | class_weight=inverse_frequency | val_metal_balanced_acc=0.6750130535709283

Best Only-GVP configuration: not available
Best ESM-based configuration: optuna_deepmzyme_controlled_hpo_trial0049_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d32997f7
Best RING vs non-RING comparison: not available unless both modes have completed numeric validation metrics.

Automatic interpretation
Best validation config: optuna_deepmzyme_controlled_hpo_trial0049_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d32997f7 with val_metal_balanced_acc = 0.6750130535709283
Top fusion mode: late_fusion
Best learning-rate region: low around 1e-5 to 3e-5 (lr=1.6801503587890522e-05)
Held-out test results present: False
Recommended next step: run top-K seed-repeat validation for the best Optuna configurations
Drive copy skipped. Outputs remain under: /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1



#------------------------------------------------------------



Short-run policy: 1-3 epoch runs are smoke/debug only and are not model-quality comparisons.
[I 2026-05-14 00:31:57,344] A new study created in memory with name: deepmzyme_controlled_hpo
Controlled Optuna HPO mode
Optuna architecture scope: this notebook optimizes hyperparameters inside the selected base MODEL_PRESET only; it does not compare model_architecture or fusion_mode unless multiple base configs are explicitly added outside this HPO path.
Optuna base model preset: GVP + late fusion arch= gvp fusion= late_fusion
Optuna search preset: custom
Optuna search-space fields: {
  "batch_size": "8",
  "cross_attention_dropout": "inactive",
  "cross_attention_heads": "inactive",
  "cross_attention_layers": "inactive",
  "early_esm_dim": "inactive",
  "early_esm_dropout": "inactive",
  "edge_hidden": "64,128",
  "edge_radius": "6.0,8.0",
  "esm_fusion_dim": "64,128,256",
  "gvp_layers": "2,3,4",
  "head_mlp_layers": "1,2,3",
  "hidden_s": "128,256",
  "hidden_v": "16,32",
  "learning_rate": "1e-5,1e-4",
  "metal_class_weight_mode": "inverse_frequency,inverse_sqrt_frequency",
  "metal_focal_gamma": "inactive",
  "search_preset": "custom",
  "weight_decay": "0.0,1e-5,1e-4,1e-3"
}
Study name: deepmzyme_controlled_hpo
Intensity: custom
Selection metric: val_metal_balanced_acc (explicit OPTUNA_SELECTION_METRIC)
Direction: maximize
Trials: 50
Trial epochs: 40 (MAX_EPOCHS_PER_TRIAL)
Normal/final retrain epochs: 1 (EPOCHS)
Timeout minutes: none
Fixed HPO split/model seed: 42
Validation split policy: split_by=pdbid, val_fraction=0.15, n_folds=None, fold_index=None
Study output directory: /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna/deepmzyme_controlled_hpo
Storage: in-memory (temporary/debug only)
Base configuration run name: deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_fusionlate_fusion_ringno_esmyes_m_43bcac41
================================================================================
[Optuna trial 0] optuna_deepmzyme_controlled_hpo_trial0000_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_bf8671dd
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 8.0,
  "esm_fusion_dim": 128,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 2,
  "head_mlp_layers": 1,
  "hidden_s": 128,
  "hidden_v": 16,
  "learning_rate": 2.368863950364079e-05,
  "metal_class_weight_mode": "inverse_sqrt_frequency",
  "metal_weighting_setup": "inverse_sqrt_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.0,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0000_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_bf8671dd --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 2.368863950364079e-05 --weight-decay 0.0 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 128 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_sqrt_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 8.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 16 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.6690 lr=2.36886e-05 train_metal_acc=0.5024 val_loss=1.5904 val_metal_acc=0.5165 val_metal_min_recall=0.0000 val_fe_recall=0.3030 val_joint_bal_acc=0.2152 val_joint_macro_f1=0.1827 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.5865 lr=2.36886e-05 train_metal_acc=0.5432 val_loss=1.5016 val_metal_acc=0.5549 val_metal_min_recall=0.0000 val_fe_recall=0.4545 val_joint_bal_acc=0.2512 val_joint_macro_f1=0.2273 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.4650 lr=2.36886e-05 train_metal_acc=0.6149 val_loss=1.3913 val_metal_acc=0.6374 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4043 val_joint_macro_f1=0.3818 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.3347 lr=2.36886e-05 train_metal_acc=0.6537 val_loss=1.2699 val_metal_acc=0.6703 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4369 val_joint_macro_f1=0.4404 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.2203 lr=2.36886e-05 train_metal_acc=0.6760 val_loss=1.1951 val_metal_acc=0.6593 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4361 val_joint_macro_f1=0.4360 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.1578 lr=2.36886e-05 train_metal_acc=0.6790 val_loss=1.1726 val_metal_acc=0.6593 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4368 val_joint_macro_f1=0.4438 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=1.0873 lr=2.36886e-05 train_metal_acc=0.7042 val_loss=1.1628 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5127 val_joint_macro_f1=0.5409 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=1.0286 lr=2.36886e-05 train_metal_acc=0.7304 val_loss=1.1779 val_metal_acc=0.5110 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4780 val_joint_macro_f1=0.4527 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.9785 lr=2.36886e-05 train_metal_acc=0.7546 val_loss=1.1067 val_metal_acc=0.6374 val_metal_min_recall=0.0000 val_fe_recall=0.5152 val_joint_bal_acc=0.4998 val_joint_macro_f1=0.5002 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.9223 lr=2.36886e-05 train_metal_acc=0.7653 val_loss=1.0809 val_metal_acc=0.6319 val_metal_min_recall=0.0000 val_fe_recall=0.6667 val_joint_bal_acc=0.5071 val_joint_macro_f1=0.4895 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.8844 lr=2.36886e-05 train_metal_acc=0.7876 val_loss=1.0467 val_metal_acc=0.6429 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5315 val_joint_macro_f1=0.5685 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.8562 lr=2.36886e-05 train_metal_acc=0.7905 val_loss=1.0453 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5161 val_joint_macro_f1=0.5431 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.7984 lr=2.36886e-05 train_metal_acc=0.8012 val_loss=1.0198 val_metal_acc=0.6429 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4869 val_joint_macro_f1=0.4975 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.7736 lr=2.36886e-05 train_metal_acc=0.8080 val_loss=1.0077 val_metal_acc=0.6648 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.5084 val_joint_macro_f1=0.5118 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.7484 lr=2.36886e-05 train_metal_acc=0.8147 val_loss=1.0765 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5396 val_joint_macro_f1=0.5552 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.7107 lr=2.36886e-05 train_metal_acc=0.8322 val_loss=0.9893 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5400 val_joint_macro_f1=0.5600 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.6979 lr=2.36886e-05 train_metal_acc=0.8390 val_loss=0.9746 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5542 val_joint_macro_f1=0.5766 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.6617 lr=2.36886e-05 train_metal_acc=0.8380 val_loss=1.0547 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5135 val_joint_macro_f1=0.5164 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.6322 lr=2.36886e-05 train_metal_acc=0.8555 val_loss=0.9742 val_metal_acc=0.6923 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5945 val_joint_macro_f1=0.6009 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.6055 lr=2.36886e-05 train_metal_acc=0.8477 val_loss=1.0264 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5892 val_joint_macro_f1=0.5984 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.5797 lr=2.36886e-05 train_metal_acc=0.8661 val_loss=0.9798 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5602 val_joint_macro_f1=0.5666 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.5702 lr=2.36886e-05 train_metal_acc=0.8749 val_loss=0.9835 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5652 val_joint_macro_f1=0.5829 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.5452 lr=2.36886e-05 train_metal_acc=0.8710 val_loss=0.9641 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5326 val_joint_macro_f1=0.5543 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.5271 lr=2.36886e-05 train_metal_acc=0.8807 val_loss=0.9913 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5714 val_joint_macro_f1=0.5702 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.5184 lr=2.36886e-05 train_metal_acc=0.8797 val_loss=0.9798 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5680 val_joint_macro_f1=0.5889 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.4896 lr=2.36886e-05 train_metal_acc=0.8788 val_loss=1.0114 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5768 val_joint_macro_f1=0.5862 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.4735 lr=2.36886e-05 train_metal_acc=0.8904 val_loss=0.9787 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5564 val_joint_macro_f1=0.5641 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.4589 lr=2.36886e-05 train_metal_acc=0.8855 val_loss=0.9904 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5714 val_joint_macro_f1=0.5888 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.4420 lr=2.36886e-05 train_metal_acc=0.8982 val_loss=1.0003 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5934 val_joint_macro_f1=0.6050 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.4346 lr=2.36886e-05 train_metal_acc=0.8933 val_loss=0.9999 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5680 val_joint_macro_f1=0.5819 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.4161 lr=2.36886e-05 train_metal_acc=0.8991 val_loss=1.0223 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5449 val_joint_macro_f1=0.5551 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.3951 lr=2.36886e-05 train_metal_acc=0.9030 val_loss=1.0096 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5703 val_joint_macro_f1=0.5875 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.3851 lr=2.36886e-05 train_metal_acc=0.9117 val_loss=0.9870 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5595 val_joint_macro_f1=0.5790 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.3686 lr=2.36886e-05 train_metal_acc=0.8952 val_loss=1.0008 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5541 val_joint_macro_f1=0.5931 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.3621 lr=2.36886e-05 train_metal_acc=0.9156 val_loss=0.9820 val_metal_acc=0.6319 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5341 val_joint_macro_f1=0.5586 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.3485 lr=2.36886e-05 train_metal_acc=0.9156 val_loss=1.0594 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5473 val_joint_macro_f1=0.5703 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.3434 lr=2.36886e-05 train_metal_acc=0.9156 val_loss=1.0514 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5862 val_joint_macro_f1=0.6230 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.3175 lr=2.36886e-05 train_metal_acc=0.9137 val_loss=1.0820 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5520 val_joint_macro_f1=0.5953 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.3144 lr=2.36886e-05 train_metal_acc=0.9205 val_loss=1.0354 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5738 val_joint_macro_f1=0.5769 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.3137 lr=2.36886e-05 train_metal_acc=0.9137 val_loss=1.0753 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5714 val_joint_macro_f1=0.5831 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0000_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_bf8671dd/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0000_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_bf8671dd/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0000_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_bf8671dd/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0000_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_bf8671dd/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0000_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_bf8671dd/run_metadata.json
[I 2026-05-14 00:40:44,445] Trial 0 finished with value: 0.594459641138958 and parameters: {'learning_rate': 2.368863950364079e-05, 'weight_decay': 0.0, 'hidden_s': 128, 'head_mlp_layers': 1, 'edge_hidden': 128, 'gvp_layers': 2, 'edge_radius': 8.0, 'hidden_v': 16, 'esm_fusion_dim': 128, 'metal_class_weight_mode': 'inverse_sqrt_frequency'}. Best is trial 0 with value: 0.594459641138958.
Optuna trial 0 completed: val_metal_balanced_acc=0.594459641138958
================================================================================
[Optuna trial 1] optuna_deepmzyme_controlled_hpo_trial0001_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_6ddff8f9
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 64,
  "edge_radius": 6.0,
  "esm_fusion_dim": 64,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 4,
  "head_mlp_layers": 3,
  "hidden_s": 256,
  "hidden_v": 16,
  "learning_rate": 2.858051065806938e-05,
  "metal_class_weight_mode": "inverse_sqrt_frequency",
  "metal_weighting_setup": "inverse_sqrt_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.0,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0001_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_6ddff8f9 --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 2.858051065806938e-05 --weight-decay 0.0 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 256 --esm-fusion-dim 64 --head-mlp-layers 3 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_sqrt_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 64 --gvp-layers 4 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 16 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7131 lr=2.85805e-05 train_metal_acc=0.4607 val_loss=1.6533 val_metal_acc=0.4670 val_metal_min_recall=0.0000 val_fe_recall=0.0000 val_joint_bal_acc=0.1667 val_joint_macro_f1=0.1061 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.6609 lr=2.85805e-05 train_metal_acc=0.4627 val_loss=1.5995 val_metal_acc=0.4670 val_metal_min_recall=0.0000 val_fe_recall=0.0000 val_joint_bal_acc=0.1667 val_joint_macro_f1=0.1061 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.5892 lr=2.85805e-05 train_metal_acc=0.5635 val_loss=1.4780 val_metal_acc=0.5659 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.2607 val_joint_macro_f1=0.2176 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.4727 lr=2.85805e-05 train_metal_acc=0.5800 val_loss=1.4797 val_metal_acc=0.5110 val_metal_min_recall=0.0000 val_fe_recall=0.8182 val_joint_bal_acc=0.2903 val_joint_macro_f1=0.2520 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.3630 lr=2.85805e-05 train_metal_acc=0.6266 val_loss=1.2520 val_metal_acc=0.6374 val_metal_min_recall=0.0000 val_fe_recall=0.6970 val_joint_bal_acc=0.3888 val_joint_macro_f1=0.3688 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.2515 lr=2.85805e-05 train_metal_acc=0.6925 val_loss=1.2531 val_metal_acc=0.6209 val_metal_min_recall=0.0000 val_fe_recall=0.6970 val_joint_bal_acc=0.4846 val_joint_macro_f1=0.4898 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=1.1629 lr=2.85805e-05 train_metal_acc=0.7148 val_loss=1.1082 val_metal_acc=0.6978 val_metal_min_recall=0.0000 val_fe_recall=0.6667 val_joint_bal_acc=0.4666 val_joint_macro_f1=0.4685 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=1.0793 lr=2.85805e-05 train_metal_acc=0.7371 val_loss=1.0882 val_metal_acc=0.7253 val_metal_min_recall=0.0000 val_fe_recall=0.6667 val_joint_bal_acc=0.5256 val_joint_macro_f1=0.5370 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=1.0030 lr=2.85805e-05 train_metal_acc=0.7294 val_loss=1.0237 val_metal_acc=0.7033 val_metal_min_recall=0.0000 val_fe_recall=0.6667 val_joint_bal_acc=0.4904 val_joint_macro_f1=0.5124 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.9397 lr=2.85805e-05 train_metal_acc=0.7517 val_loss=1.1359 val_metal_acc=0.5495 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4836 val_joint_macro_f1=0.4790 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.8731 lr=2.85805e-05 train_metal_acc=0.7682 val_loss=1.0851 val_metal_acc=0.6319 val_metal_min_recall=0.0000 val_fe_recall=0.6667 val_joint_bal_acc=0.5243 val_joint_macro_f1=0.5357 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.8339 lr=2.85805e-05 train_metal_acc=0.7886 val_loss=1.0069 val_metal_acc=0.6538 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.5294 val_joint_macro_f1=0.5450 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.7831 lr=2.85805e-05 train_metal_acc=0.8147 val_loss=1.0010 val_metal_acc=0.6538 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.5294 val_joint_macro_f1=0.5450 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.7855 lr=2.85805e-05 train_metal_acc=0.7944 val_loss=1.0504 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5502 val_joint_macro_f1=0.5955 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.7272 lr=2.85805e-05 train_metal_acc=0.8303 val_loss=1.0919 val_metal_acc=0.6429 val_metal_min_recall=0.0000 val_fe_recall=0.5455 val_joint_bal_acc=0.5295 val_joint_macro_f1=0.5307 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.6767 lr=2.85805e-05 train_metal_acc=0.8322 val_loss=0.9776 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5431 val_joint_macro_f1=0.5768 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.6549 lr=2.85805e-05 train_metal_acc=0.8312 val_loss=1.0799 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.6127 val_joint_macro_f1=0.6129 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.6193 lr=2.85805e-05 train_metal_acc=0.8477 val_loss=1.0917 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5636 val_joint_macro_f1=0.5681 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.5993 lr=2.85805e-05 train_metal_acc=0.8613 val_loss=1.0562 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5582 val_joint_macro_f1=0.5827 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.5759 lr=2.85805e-05 train_metal_acc=0.8497 val_loss=1.0772 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5908 val_joint_macro_f1=0.5873 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.5614 lr=2.85805e-05 train_metal_acc=0.8429 val_loss=1.0703 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5986 val_joint_macro_f1=0.5809 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.5323 lr=2.85805e-05 train_metal_acc=0.8768 val_loss=1.0944 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5652 val_joint_macro_f1=0.5825 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.5056 lr=2.85805e-05 train_metal_acc=0.8661 val_loss=0.9770 val_metal_acc=0.6978 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6065 val_joint_macro_f1=0.6167 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.4845 lr=2.85805e-05 train_metal_acc=0.8468 val_loss=0.9661 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.7273 val_joint_bal_acc=0.6051 val_joint_macro_f1=0.6198 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.4736 lr=2.85805e-05 train_metal_acc=0.8894 val_loss=1.1540 val_metal_acc=0.7637 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5764 val_joint_macro_f1=0.6152 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.4470 lr=2.85805e-05 train_metal_acc=0.8991 val_loss=1.0625 val_metal_acc=0.7582 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5704 val_joint_macro_f1=0.6115 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.4280 lr=2.85805e-05 train_metal_acc=0.8807 val_loss=1.2957 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5652 val_joint_macro_f1=0.5768 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.4314 lr=2.85805e-05 train_metal_acc=0.9040 val_loss=1.2520 val_metal_acc=0.7747 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6087 val_joint_macro_f1=0.6408 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.3929 lr=2.85805e-05 train_metal_acc=0.9040 val_loss=1.2115 val_metal_acc=0.6813 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6009 val_joint_macro_f1=0.5955 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.3787 lr=2.85805e-05 train_metal_acc=0.9059 val_loss=1.1938 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5849 val_joint_macro_f1=0.6132 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.3825 lr=2.85805e-05 train_metal_acc=0.9098 val_loss=1.2435 val_metal_acc=0.6978 val_metal_min_recall=0.0769 val_fe_recall=0.6970 val_joint_bal_acc=0.6013 val_joint_macro_f1=0.6162 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.3816 lr=2.85805e-05 train_metal_acc=0.9195 val_loss=1.3617 val_metal_acc=0.7637 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5826 val_joint_macro_f1=0.6141 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.3579 lr=2.85805e-05 train_metal_acc=0.9224 val_loss=1.3850 val_metal_acc=0.7747 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5834 val_joint_macro_f1=0.6172 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.3422 lr=2.85805e-05 train_metal_acc=0.9185 val_loss=1.5901 val_metal_acc=0.6813 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5753 val_joint_macro_f1=0.5984 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.3351 lr=2.85805e-05 train_metal_acc=0.9185 val_loss=1.4202 val_metal_acc=0.7692 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5886 val_joint_macro_f1=0.6147 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.3077 lr=2.85805e-05 train_metal_acc=0.9282 val_loss=1.4927 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5938 val_joint_macro_f1=0.6226 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.2991 lr=2.85805e-05 train_metal_acc=0.9263 val_loss=1.5509 val_metal_acc=0.7582 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5775 val_joint_macro_f1=0.6073 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.2985 lr=2.85805e-05 train_metal_acc=0.9292 val_loss=1.7094 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5938 val_joint_macro_f1=0.6035 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.2803 lr=2.85805e-05 train_metal_acc=0.9214 val_loss=1.4817 val_metal_acc=0.6978 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.6235 val_joint_macro_f1=0.6033 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.2926 lr=2.85805e-05 train_metal_acc=0.9379 val_loss=1.7152 val_metal_acc=0.7527 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5718 val_joint_macro_f1=0.5994 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0001_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_6ddff8f9/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0001_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_6ddff8f9/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0001_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_6ddff8f9/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0001_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_6ddff8f9/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0001_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_6ddff8f9/run_metadata.json
[I 2026-05-14 00:49:50,684] Trial 1 finished with value: 0.6235097181966253 and parameters: {'learning_rate': 2.858051065806938e-05, 'weight_decay': 0.0, 'hidden_s': 256, 'head_mlp_layers': 3, 'edge_hidden': 64, 'gvp_layers': 4, 'edge_radius': 6.0, 'hidden_v': 16, 'esm_fusion_dim': 64, 'metal_class_weight_mode': 'inverse_sqrt_frequency'}. Best is trial 1 with value: 0.6235097181966253.
Optuna trial 1 completed: val_metal_balanced_acc=0.6235097181966253
================================================================================
[Optuna trial 2] optuna_deepmzyme_controlled_hpo_trial0002_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a273d109
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 6.0,
  "esm_fusion_dim": 256,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 4,
  "head_mlp_layers": 1,
  "hidden_s": 128,
  "hidden_v": 16,
  "learning_rate": 3.521358805467871e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 1e-05,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0002_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a273d109 --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 3.521358805467871e-05 --weight-decay 1e-05 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 256 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 4 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 16 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7394 lr=3.52136e-05 train_metal_acc=0.3307 val_loss=1.6609 val_metal_acc=0.2637 val_metal_min_recall=0.0000 val_fe_recall=0.3939 val_joint_bal_acc=0.3434 val_joint_macro_f1=0.3097 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.5525 lr=3.52136e-05 train_metal_acc=0.5761 val_loss=1.4642 val_metal_acc=0.4176 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4584 val_joint_macro_f1=0.4298 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.3728 lr=3.52136e-05 train_metal_acc=0.6751 val_loss=1.3239 val_metal_acc=0.4615 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4420 val_joint_macro_f1=0.4245 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.2568 lr=3.52136e-05 train_metal_acc=0.6906 val_loss=1.2783 val_metal_acc=0.4615 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5279 val_joint_macro_f1=0.5067 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.1531 lr=3.52136e-05 train_metal_acc=0.7158 val_loss=1.1897 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5114 val_joint_macro_f1=0.5381 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.0770 lr=3.52136e-05 train_metal_acc=0.7274 val_loss=1.2149 val_metal_acc=0.4560 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.4806 val_joint_macro_f1=0.4799 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=0.9706 lr=3.52136e-05 train_metal_acc=0.7759 val_loss=1.1226 val_metal_acc=0.6319 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5853 val_joint_macro_f1=0.5974 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=0.9083 lr=3.52136e-05 train_metal_acc=0.8012 val_loss=1.0719 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5879 val_joint_macro_f1=0.6091 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.8248 lr=3.52136e-05 train_metal_acc=0.7886 val_loss=1.1157 val_metal_acc=0.6429 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5647 val_joint_macro_f1=0.5838 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.7798 lr=3.52136e-05 train_metal_acc=0.8157 val_loss=1.0558 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.6050 val_joint_macro_f1=0.6035 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.7209 lr=3.52136e-05 train_metal_acc=0.8225 val_loss=1.0726 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5812 val_joint_macro_f1=0.5928 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.6791 lr=3.52136e-05 train_metal_acc=0.8526 val_loss=1.0537 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5873 val_joint_macro_f1=0.6106 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.6543 lr=3.52136e-05 train_metal_acc=0.8516 val_loss=0.9883 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5883 val_joint_macro_f1=0.6221 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.5868 lr=3.52136e-05 train_metal_acc=0.8419 val_loss=1.0544 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5792 val_joint_macro_f1=0.5829 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.5652 lr=3.52136e-05 train_metal_acc=0.8807 val_loss=1.0371 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5579 val_joint_macro_f1=0.5755 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.5272 lr=3.52136e-05 train_metal_acc=0.8661 val_loss=1.0666 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5848 val_joint_macro_f1=0.5983 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.5134 lr=3.52136e-05 train_metal_acc=0.8923 val_loss=1.0058 val_metal_acc=0.7637 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6314 val_joint_macro_f1=0.6506 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.4890 lr=3.52136e-05 train_metal_acc=0.8923 val_loss=1.0297 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5828 val_joint_macro_f1=0.5840 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.4472 lr=3.52136e-05 train_metal_acc=0.9020 val_loss=1.0084 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6051 val_joint_macro_f1=0.6198 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.4226 lr=3.52136e-05 train_metal_acc=0.9059 val_loss=1.0179 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6060 val_joint_macro_f1=0.6173 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.3970 lr=3.52136e-05 train_metal_acc=0.9098 val_loss=1.0528 val_metal_acc=0.6923 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5977 val_joint_macro_f1=0.6131 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.3734 lr=3.52136e-05 train_metal_acc=0.9214 val_loss=1.0358 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6036 val_joint_macro_f1=0.6196 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.3551 lr=3.52136e-05 train_metal_acc=0.9098 val_loss=1.1655 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5830 val_joint_macro_f1=0.6130 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.3762 lr=3.52136e-05 train_metal_acc=0.9146 val_loss=1.0043 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5884 val_joint_macro_f1=0.5829 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.3230 lr=3.52136e-05 train_metal_acc=0.9263 val_loss=1.0853 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6070 val_joint_macro_f1=0.6283 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.3236 lr=3.52136e-05 train_metal_acc=0.9273 val_loss=1.1914 val_metal_acc=0.7637 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5979 val_joint_macro_f1=0.6339 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.3144 lr=3.52136e-05 train_metal_acc=0.9137 val_loss=1.1896 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5817 val_joint_macro_f1=0.5918 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.3020 lr=3.52136e-05 train_metal_acc=0.9214 val_loss=1.1602 val_metal_acc=0.6978 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5761 val_joint_macro_f1=0.5808 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.2809 lr=3.52136e-05 train_metal_acc=0.9302 val_loss=1.1685 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5943 val_joint_macro_f1=0.6162 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.2583 lr=3.52136e-05 train_metal_acc=0.9302 val_loss=1.1817 val_metal_acc=0.7033 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.6060 val_joint_macro_f1=0.6294 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.2846 lr=3.52136e-05 train_metal_acc=0.9059 val_loss=1.1902 val_metal_acc=0.6648 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.5991 val_joint_macro_f1=0.5823 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.2530 lr=3.52136e-05 train_metal_acc=0.9146 val_loss=1.3159 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5879 val_joint_macro_f1=0.6112 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.2480 lr=3.52136e-05 train_metal_acc=0.9292 val_loss=1.4074 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5776 val_joint_macro_f1=0.6076 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.2485 lr=3.52136e-05 train_metal_acc=0.9321 val_loss=1.3057 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5907 val_joint_macro_f1=0.6042 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.2516 lr=3.52136e-05 train_metal_acc=0.9370 val_loss=1.3576 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5841 val_joint_macro_f1=0.6172 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.2316 lr=3.52136e-05 train_metal_acc=0.9370 val_loss=1.3137 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5726 val_joint_macro_f1=0.5777 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.2275 lr=3.52136e-05 train_metal_acc=0.9302 val_loss=1.4453 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5824 val_joint_macro_f1=0.6104 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.2187 lr=3.52136e-05 train_metal_acc=0.9408 val_loss=1.3124 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5762 val_joint_macro_f1=0.5826 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.2114 lr=3.52136e-05 train_metal_acc=0.9408 val_loss=1.3834 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5816 val_joint_macro_f1=0.5969 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.2155 lr=3.52136e-05 train_metal_acc=0.9389 val_loss=1.5609 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5821 val_joint_macro_f1=0.6156 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0002_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a273d109/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0002_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a273d109/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0002_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a273d109/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0002_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a273d109/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0002_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a273d109/run_metadata.json
[I 2026-05-14 00:58:52,667] Trial 2 finished with value: 0.6313553408430069 and parameters: {'learning_rate': 3.521358805467871e-05, 'weight_decay': 1e-05, 'hidden_s': 128, 'head_mlp_layers': 1, 'edge_hidden': 128, 'gvp_layers': 4, 'edge_radius': 6.0, 'hidden_v': 16, 'esm_fusion_dim': 256, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 2 with value: 0.6313553408430069.
Optuna trial 2 completed: val_metal_balanced_acc=0.6313553408430069
================================================================================
[Optuna trial 3] optuna_deepmzyme_controlled_hpo_trial0003_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_67900e48
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 64,
  "edge_radius": 8.0,
  "esm_fusion_dim": 128,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 4,
  "head_mlp_layers": 2,
  "hidden_s": 256,
  "hidden_v": 16,
  "learning_rate": 1.012796325733148e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.0,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0003_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_67900e48 --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 1.012796325733148e-05 --weight-decay 0.0 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 256 --esm-fusion-dim 128 --head-mlp-layers 2 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 64 --gvp-layers 4 --edge-radius 8.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 16 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7822 lr=1.0128e-05 train_metal_acc=0.4646 val_loss=1.7536 val_metal_acc=0.4615 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.2265 val_joint_macro_f1=0.1738 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.7622 lr=1.0128e-05 train_metal_acc=0.4617 val_loss=1.7365 val_metal_acc=0.3791 val_metal_min_recall=0.0000 val_fe_recall=0.9697 val_joint_bal_acc=0.2342 val_joint_macro_f1=0.1537 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.7371 lr=1.0128e-05 train_metal_acc=0.5189 val_loss=1.6901 val_metal_acc=0.4505 val_metal_min_recall=0.0000 val_fe_recall=0.4848 val_joint_bal_acc=0.2341 val_joint_macro_f1=0.2204 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.7209 lr=1.0128e-05 train_metal_acc=0.5344 val_loss=1.6588 val_metal_acc=0.4670 val_metal_min_recall=0.0000 val_fe_recall=0.5152 val_joint_bal_acc=0.3374 val_joint_macro_f1=0.3271 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.6689 lr=1.0128e-05 train_metal_acc=0.5044 val_loss=1.6221 val_metal_acc=0.4451 val_metal_min_recall=0.0000 val_fe_recall=1.0000 val_joint_bal_acc=0.3477 val_joint_macro_f1=0.2866 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.5999 lr=1.0128e-05 train_metal_acc=0.5723 val_loss=1.5293 val_metal_acc=0.3901 val_metal_min_recall=0.0000 val_fe_recall=0.7879 val_joint_bal_acc=0.3917 val_joint_macro_f1=0.3879 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=1.5277 lr=1.0128e-05 train_metal_acc=0.5829 val_loss=1.4569 val_metal_acc=0.3791 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.3679 val_joint_macro_f1=0.3406 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=1.4984 lr=1.0128e-05 train_metal_acc=0.5723 val_loss=1.4266 val_metal_acc=0.3846 val_metal_min_recall=0.0000 val_fe_recall=0.5152 val_joint_bal_acc=0.4061 val_joint_macro_f1=0.3532 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=1.4447 lr=1.0128e-05 train_metal_acc=0.6334 val_loss=1.3741 val_metal_acc=0.4011 val_metal_min_recall=0.0000 val_fe_recall=0.5455 val_joint_bal_acc=0.3907 val_joint_macro_f1=0.3847 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=1.3817 lr=1.0128e-05 train_metal_acc=0.6615 val_loss=1.3153 val_metal_acc=0.4121 val_metal_min_recall=0.0000 val_fe_recall=0.5455 val_joint_bal_acc=0.3809 val_joint_macro_f1=0.3786 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=1.3582 lr=1.0128e-05 train_metal_acc=0.6596 val_loss=1.2910 val_metal_acc=0.6044 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.4726 val_joint_macro_f1=0.4911 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=1.2981 lr=1.0128e-05 train_metal_acc=0.6469 val_loss=1.2799 val_metal_acc=0.4286 val_metal_min_recall=0.0769 val_fe_recall=0.6970 val_joint_bal_acc=0.4773 val_joint_macro_f1=0.4688 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=1.2651 lr=1.0128e-05 train_metal_acc=0.7022 val_loss=1.2199 val_metal_acc=0.5385 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5297 val_joint_macro_f1=0.5276 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=1.2278 lr=1.0128e-05 train_metal_acc=0.7013 val_loss=1.2219 val_metal_acc=0.6044 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5062 val_joint_macro_f1=0.5366 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=1.1922 lr=1.0128e-05 train_metal_acc=0.7110 val_loss=1.1772 val_metal_acc=0.5549 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5449 val_joint_macro_f1=0.5437 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=1.1389 lr=1.0128e-05 train_metal_acc=0.6809 val_loss=1.1948 val_metal_acc=0.4560 val_metal_min_recall=0.0000 val_fe_recall=0.6667 val_joint_bal_acc=0.5053 val_joint_macro_f1=0.4886 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=1.1344 lr=1.0128e-05 train_metal_acc=0.7265 val_loss=1.1477 val_metal_acc=0.5879 val_metal_min_recall=0.2308 val_fe_recall=0.6364 val_joint_bal_acc=0.5463 val_joint_macro_f1=0.5806 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=1.0887 lr=1.0128e-05 train_metal_acc=0.7449 val_loss=1.1209 val_metal_acc=0.5659 val_metal_min_recall=0.1538 val_fe_recall=0.6667 val_joint_bal_acc=0.5525 val_joint_macro_f1=0.5734 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=1.0673 lr=1.0128e-05 train_metal_acc=0.7439 val_loss=1.1023 val_metal_acc=0.5879 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5413 val_joint_macro_f1=0.5652 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=1.0347 lr=1.0128e-05 train_metal_acc=0.7381 val_loss=1.1326 val_metal_acc=0.5440 val_metal_min_recall=0.0769 val_fe_recall=0.6970 val_joint_bal_acc=0.5540 val_joint_macro_f1=0.5598 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.9716 lr=1.0128e-05 train_metal_acc=0.7575 val_loss=1.0630 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.7273 val_joint_bal_acc=0.5837 val_joint_macro_f1=0.5956 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.9670 lr=1.0128e-05 train_metal_acc=0.7468 val_loss=1.1306 val_metal_acc=0.5220 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5434 val_joint_macro_f1=0.5465 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.9380 lr=1.0128e-05 train_metal_acc=0.7585 val_loss=1.1260 val_metal_acc=0.5220 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5280 val_joint_macro_f1=0.5306 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.9344 lr=1.0128e-05 train_metal_acc=0.7546 val_loss=1.0750 val_metal_acc=0.6978 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5717 val_joint_macro_f1=0.5972 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.9101 lr=1.0128e-05 train_metal_acc=0.7701 val_loss=1.1183 val_metal_acc=0.5604 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5571 val_joint_macro_f1=0.5487 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.8673 lr=1.0128e-05 train_metal_acc=0.7856 val_loss=1.0751 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5727 val_joint_macro_f1=0.5850 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.8486 lr=1.0128e-05 train_metal_acc=0.7905 val_loss=1.0406 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5465 val_joint_macro_f1=0.5681 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.8318 lr=1.0128e-05 train_metal_acc=0.7381 val_loss=1.1922 val_metal_acc=0.4890 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5320 val_joint_macro_f1=0.5158 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.7882 lr=1.0128e-05 train_metal_acc=0.7837 val_loss=1.0709 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.7273 val_joint_bal_acc=0.5934 val_joint_macro_f1=0.6126 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.7713 lr=1.0128e-05 train_metal_acc=0.7953 val_loss=1.0813 val_metal_acc=0.6209 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5660 val_joint_macro_f1=0.5755 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.7444 lr=1.0128e-05 train_metal_acc=0.8050 val_loss=1.0560 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5817 val_joint_macro_f1=0.5882 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.7461 lr=1.0128e-05 train_metal_acc=0.7624 val_loss=1.1822 val_metal_acc=0.5385 val_metal_min_recall=0.0769 val_fe_recall=0.6970 val_joint_bal_acc=0.5555 val_joint_macro_f1=0.5489 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.7423 lr=1.0128e-05 train_metal_acc=0.7886 val_loss=1.1239 val_metal_acc=0.5824 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5619 val_joint_macro_f1=0.5629 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.6858 lr=1.0128e-05 train_metal_acc=0.8070 val_loss=1.1053 val_metal_acc=0.6319 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5668 val_joint_macro_f1=0.5850 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.6905 lr=1.0128e-05 train_metal_acc=0.8322 val_loss=1.0432 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5810 val_joint_macro_f1=0.5982 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.6471 lr=1.0128e-05 train_metal_acc=0.8147 val_loss=1.0733 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5809 val_joint_macro_f1=0.5905 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.6267 lr=1.0128e-05 train_metal_acc=0.8206 val_loss=1.0817 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5708 val_joint_macro_f1=0.5956 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.6462 lr=1.0128e-05 train_metal_acc=0.8283 val_loss=1.0335 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5791 val_joint_macro_f1=0.5894 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.6000 lr=1.0128e-05 train_metal_acc=0.8361 val_loss=1.0814 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5626 val_joint_macro_f1=0.5797 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.5796 lr=1.0128e-05 train_metal_acc=0.8283 val_loss=1.1201 val_metal_acc=0.6429 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5769 val_joint_macro_f1=0.5866 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0003_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_67900e48/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0003_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_67900e48/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0003_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_67900e48/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0003_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_67900e48/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0003_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_67900e48/run_metadata.json
[I 2026-05-14 01:08:42,694] Trial 3 finished with value: 0.59340248619186 and parameters: {'learning_rate': 1.012796325733148e-05, 'weight_decay': 0.0, 'hidden_s': 256, 'head_mlp_layers': 2, 'edge_hidden': 64, 'gvp_layers': 4, 'edge_radius': 8.0, 'hidden_v': 16, 'esm_fusion_dim': 128, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 2 with value: 0.6313553408430069.
Optuna trial 3 completed: val_metal_balanced_acc=0.59340248619186
================================================================================
[Optuna trial 4] optuna_deepmzyme_controlled_hpo_trial0004_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_6b955af4
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 8.0,
  "esm_fusion_dim": 64,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 4,
  "head_mlp_layers": 2,
  "hidden_s": 128,
  "hidden_v": 16,
  "learning_rate": 3.332213575546236e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.0,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0004_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_6b955af4 --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 3.332213575546236e-05 --weight-decay 0.0 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 64 --head-mlp-layers 2 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 4 --edge-radius 8.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 16 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7805 lr=3.33221e-05 train_metal_acc=0.4433 val_loss=1.7473 val_metal_acc=0.2637 val_metal_min_recall=0.0000 val_fe_recall=0.7879 val_joint_bal_acc=0.1779 val_joint_macro_f1=0.1173 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.7525 lr=3.33221e-05 train_metal_acc=0.5238 val_loss=1.7082 val_metal_acc=0.5659 val_metal_min_recall=0.0000 val_fe_recall=0.0909 val_joint_bal_acc=0.3295 val_joint_macro_f1=0.3231 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.6921 lr=3.33221e-05 train_metal_acc=0.5587 val_loss=1.6254 val_metal_acc=0.4725 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.3559 val_joint_macro_f1=0.3689 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.5984 lr=3.33221e-05 train_metal_acc=0.6091 val_loss=1.5377 val_metal_acc=0.4176 val_metal_min_recall=0.0000 val_fe_recall=0.4848 val_joint_bal_acc=0.4000 val_joint_macro_f1=0.3864 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.5034 lr=3.33221e-05 train_metal_acc=0.6382 val_loss=1.4727 val_metal_acc=0.4286 val_metal_min_recall=0.0000 val_fe_recall=0.5152 val_joint_bal_acc=0.4247 val_joint_macro_f1=0.3879 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.3986 lr=3.33221e-05 train_metal_acc=0.6605 val_loss=1.3914 val_metal_acc=0.4286 val_metal_min_recall=0.0000 val_fe_recall=0.5455 val_joint_bal_acc=0.4244 val_joint_macro_f1=0.4016 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=1.3215 lr=3.33221e-05 train_metal_acc=0.6712 val_loss=1.3384 val_metal_acc=0.4505 val_metal_min_recall=0.0000 val_fe_recall=0.5455 val_joint_bal_acc=0.4643 val_joint_macro_f1=0.4383 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=1.2259 lr=3.33221e-05 train_metal_acc=0.6916 val_loss=1.2700 val_metal_acc=0.4451 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4446 val_joint_macro_f1=0.4224 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=1.1357 lr=3.33221e-05 train_metal_acc=0.7352 val_loss=1.2256 val_metal_acc=0.4725 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.4697 val_joint_macro_f1=0.4587 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=1.0628 lr=3.33221e-05 train_metal_acc=0.7313 val_loss=1.1878 val_metal_acc=0.5165 val_metal_min_recall=0.0000 val_fe_recall=0.5455 val_joint_bal_acc=0.4824 val_joint_macro_f1=0.4844 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=1.0090 lr=3.33221e-05 train_metal_acc=0.7498 val_loss=1.1462 val_metal_acc=0.6484 val_metal_min_recall=0.0000 val_fe_recall=0.5152 val_joint_bal_acc=0.5803 val_joint_macro_f1=0.5519 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.9466 lr=3.33221e-05 train_metal_acc=0.7953 val_loss=1.0780 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5475 val_joint_macro_f1=0.5525 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.8831 lr=3.33221e-05 train_metal_acc=0.7953 val_loss=1.0663 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5885 val_joint_macro_f1=0.5826 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.8295 lr=3.33221e-05 train_metal_acc=0.8021 val_loss=1.0722 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5600 val_joint_macro_f1=0.5523 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.7974 lr=3.33221e-05 train_metal_acc=0.7866 val_loss=1.0788 val_metal_acc=0.6429 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5841 val_joint_macro_f1=0.5737 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.7400 lr=3.33221e-05 train_metal_acc=0.7944 val_loss=1.1222 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5614 val_joint_macro_f1=0.5662 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.7133 lr=3.33221e-05 train_metal_acc=0.8303 val_loss=1.0423 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5755 val_joint_macro_f1=0.5699 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.6847 lr=3.33221e-05 train_metal_acc=0.8274 val_loss=1.0159 val_metal_acc=0.6484 val_metal_min_recall=0.1538 val_fe_recall=0.4848 val_joint_bal_acc=0.5416 val_joint_macro_f1=0.5549 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.6605 lr=3.33221e-05 train_metal_acc=0.8196 val_loss=1.0553 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5580 val_joint_macro_f1=0.5479 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.6190 lr=3.33221e-05 train_metal_acc=0.8177 val_loss=1.1136 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5853 val_joint_macro_f1=0.5788 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.5863 lr=3.33221e-05 train_metal_acc=0.8535 val_loss=1.0543 val_metal_acc=0.6703 val_metal_min_recall=0.1538 val_fe_recall=0.5455 val_joint_bal_acc=0.5795 val_joint_macro_f1=0.5911 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.5504 lr=3.33221e-05 train_metal_acc=0.8555 val_loss=1.0607 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5717 val_joint_macro_f1=0.5739 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.5208 lr=3.33221e-05 train_metal_acc=0.8506 val_loss=1.1563 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5681 val_joint_macro_f1=0.5685 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.5344 lr=3.33221e-05 train_metal_acc=0.8749 val_loss=1.0949 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5796 val_joint_macro_f1=0.5996 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.4815 lr=3.33221e-05 train_metal_acc=0.8671 val_loss=1.1642 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5677 val_joint_macro_f1=0.5845 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.4767 lr=3.33221e-05 train_metal_acc=0.8681 val_loss=1.1572 val_metal_acc=0.6648 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.5772 val_joint_macro_f1=0.5915 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.4528 lr=3.33221e-05 train_metal_acc=0.8855 val_loss=1.1810 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5699 val_joint_macro_f1=0.5825 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.4362 lr=3.33221e-05 train_metal_acc=0.8846 val_loss=1.1915 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5638 val_joint_macro_f1=0.5596 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.4274 lr=3.33221e-05 train_metal_acc=0.9049 val_loss=1.2087 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5855 val_joint_macro_f1=0.5998 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.3892 lr=3.33221e-05 train_metal_acc=0.9079 val_loss=1.2056 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5870 val_joint_macro_f1=0.6049 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.3688 lr=3.33221e-05 train_metal_acc=0.8904 val_loss=1.2416 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5955 val_joint_macro_f1=0.5976 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.3571 lr=3.33221e-05 train_metal_acc=0.8962 val_loss=1.2954 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5894 val_joint_macro_f1=0.6090 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.3488 lr=3.33221e-05 train_metal_acc=0.8991 val_loss=1.3246 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5913 val_joint_macro_f1=0.5851 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.3614 lr=3.33221e-05 train_metal_acc=0.9146 val_loss=1.2757 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5829 val_joint_macro_f1=0.6058 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.3419 lr=3.33221e-05 train_metal_acc=0.9117 val_loss=1.2901 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5814 val_joint_macro_f1=0.6048 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.3310 lr=3.33221e-05 train_metal_acc=0.9176 val_loss=1.4264 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5818 val_joint_macro_f1=0.6042 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.3136 lr=3.33221e-05 train_metal_acc=0.9098 val_loss=1.4523 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5949 val_joint_macro_f1=0.6020 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.3035 lr=3.33221e-05 train_metal_acc=0.9214 val_loss=1.3665 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5761 val_joint_macro_f1=0.5973 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.2974 lr=3.33221e-05 train_metal_acc=0.9059 val_loss=1.5264 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5593 val_joint_macro_f1=0.5696 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.3181 lr=3.33221e-05 train_metal_acc=0.9282 val_loss=1.4573 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5858 val_joint_macro_f1=0.6185 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0004_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_6b955af4/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0004_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_6b955af4/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0004_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_6b955af4/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0004_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_6b955af4/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0004_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_6b955af4/run_metadata.json
[I 2026-05-14 01:18:20,973] Trial 4 finished with value: 0.5955376902245972 and parameters: {'learning_rate': 3.332213575546236e-05, 'weight_decay': 0.0, 'hidden_s': 128, 'head_mlp_layers': 2, 'edge_hidden': 128, 'gvp_layers': 4, 'edge_radius': 8.0, 'hidden_v': 16, 'esm_fusion_dim': 64, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 2 with value: 0.6313553408430069.
Optuna trial 4 completed: val_metal_balanced_acc=0.5955376902245972
================================================================================
[Optuna trial 5] optuna_deepmzyme_controlled_hpo_trial0005_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_f07cdf8c
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 64,
  "edge_radius": 8.0,
  "esm_fusion_dim": 64,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 4,
  "head_mlp_layers": 1,
  "hidden_s": 256,
  "hidden_v": 16,
  "learning_rate": 6.418597685324415e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.0,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0005_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_f07cdf8c --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 6.418597685324415e-05 --weight-decay 0.0 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 256 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 64 --gvp-layers 4 --edge-radius 8.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 16 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7753 lr=6.4186e-05 train_metal_acc=0.2871 val_loss=1.6999 val_metal_acc=0.2692 val_metal_min_recall=0.0000 val_fe_recall=0.8485 val_joint_bal_acc=0.3380 val_joint_macro_f1=0.2704 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.5393 lr=6.4186e-05 train_metal_acc=0.6343 val_loss=1.3936 val_metal_acc=0.4286 val_metal_min_recall=0.0000 val_fe_recall=0.5455 val_joint_bal_acc=0.4292 val_joint_macro_f1=0.4097 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.3129 lr=6.4186e-05 train_metal_acc=0.6857 val_loss=1.2818 val_metal_acc=0.4560 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.4683 val_joint_macro_f1=0.4671 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.1416 lr=6.4186e-05 train_metal_acc=0.6896 val_loss=1.2459 val_metal_acc=0.4670 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5432 val_joint_macro_f1=0.5107 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=0.9950 lr=6.4186e-05 train_metal_acc=0.7391 val_loss=1.1699 val_metal_acc=0.6154 val_metal_min_recall=0.1538 val_fe_recall=0.4848 val_joint_bal_acc=0.5872 val_joint_macro_f1=0.5917 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=0.8784 lr=6.4186e-05 train_metal_acc=0.7769 val_loss=1.1688 val_metal_acc=0.5330 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5661 val_joint_macro_f1=0.5739 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=0.8350 lr=6.4186e-05 train_metal_acc=0.7818 val_loss=1.2008 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5969 val_joint_macro_f1=0.5545 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=0.7442 lr=6.4186e-05 train_metal_acc=0.7953 val_loss=1.1820 val_metal_acc=0.6044 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5655 val_joint_macro_f1=0.5151 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.7049 lr=6.4186e-05 train_metal_acc=0.8147 val_loss=1.1212 val_metal_acc=0.6593 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.5971 val_joint_macro_f1=0.6108 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.6191 lr=6.4186e-05 train_metal_acc=0.8448 val_loss=1.1828 val_metal_acc=0.6593 val_metal_min_recall=0.1538 val_fe_recall=0.5455 val_joint_bal_acc=0.6056 val_joint_macro_f1=0.6143 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.5809 lr=6.4186e-05 train_metal_acc=0.8477 val_loss=1.2547 val_metal_acc=0.6923 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.6106 val_joint_macro_f1=0.5983 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.5334 lr=6.4186e-05 train_metal_acc=0.8477 val_loss=1.1892 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5874 val_joint_macro_f1=0.5611 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.4862 lr=6.4186e-05 train_metal_acc=0.8584 val_loss=1.1121 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5642 val_joint_macro_f1=0.5771 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.4771 lr=6.4186e-05 train_metal_acc=0.8991 val_loss=1.2013 val_metal_acc=0.6868 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5708 val_joint_macro_f1=0.5787 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.4232 lr=6.4186e-05 train_metal_acc=0.8865 val_loss=1.2470 val_metal_acc=0.6923 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5724 val_joint_macro_f1=0.5969 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.4054 lr=6.4186e-05 train_metal_acc=0.9098 val_loss=1.3007 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5606 val_joint_macro_f1=0.5954 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.3575 lr=6.4186e-05 train_metal_acc=0.8972 val_loss=1.3451 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6970 val_joint_bal_acc=0.5929 val_joint_macro_f1=0.6250 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.3229 lr=6.4186e-05 train_metal_acc=0.9079 val_loss=1.4211 val_metal_acc=0.6978 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5840 val_joint_macro_f1=0.5967 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.3506 lr=6.4186e-05 train_metal_acc=0.9049 val_loss=1.3063 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5662 val_joint_macro_f1=0.5510 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.2764 lr=6.4186e-05 train_metal_acc=0.9098 val_loss=1.4076 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5517 val_joint_macro_f1=0.5679 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.2704 lr=6.4186e-05 train_metal_acc=0.9263 val_loss=1.5141 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.6970 val_joint_bal_acc=0.5553 val_joint_macro_f1=0.5858 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.2626 lr=6.4186e-05 train_metal_acc=0.9243 val_loss=1.6240 val_metal_acc=0.6813 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5764 val_joint_macro_f1=0.6102 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.2691 lr=6.4186e-05 train_metal_acc=0.8758 val_loss=1.8210 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5329 val_joint_macro_f1=0.5782 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.2674 lr=6.4186e-05 train_metal_acc=0.9292 val_loss=1.8462 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5436 val_joint_macro_f1=0.5830 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.2491 lr=6.4186e-05 train_metal_acc=0.9360 val_loss=1.7325 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6970 val_joint_bal_acc=0.5472 val_joint_macro_f1=0.5897 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.1936 lr=6.4186e-05 train_metal_acc=0.9418 val_loss=1.6219 val_metal_acc=0.6868 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5817 val_joint_macro_f1=0.5948 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.2015 lr=6.4186e-05 train_metal_acc=0.9360 val_loss=1.7326 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5612 val_joint_macro_f1=0.5950 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.2075 lr=6.4186e-05 train_metal_acc=0.9544 val_loss=1.7596 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5375 val_joint_macro_f1=0.5550 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.2208 lr=6.4186e-05 train_metal_acc=0.9331 val_loss=1.7086 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5683 val_joint_macro_f1=0.5999 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.1895 lr=6.4186e-05 train_metal_acc=0.9534 val_loss=1.6554 val_metal_acc=0.6868 val_metal_min_recall=0.1538 val_fe_recall=0.7273 val_joint_bal_acc=0.5663 val_joint_macro_f1=0.6049 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.1778 lr=6.4186e-05 train_metal_acc=0.9447 val_loss=1.8523 val_metal_acc=0.6538 val_metal_min_recall=0.1538 val_fe_recall=0.6667 val_joint_bal_acc=0.5484 val_joint_macro_f1=0.5859 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.1606 lr=6.4186e-05 train_metal_acc=0.9564 val_loss=2.0766 val_metal_acc=0.6868 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5664 val_joint_macro_f1=0.6006 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.1994 lr=6.4186e-05 train_metal_acc=0.9554 val_loss=2.0266 val_metal_acc=0.6813 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5760 val_joint_macro_f1=0.6083 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.1879 lr=6.4186e-05 train_metal_acc=0.9573 val_loss=2.0905 val_metal_acc=0.6813 val_metal_min_recall=0.0769 val_fe_recall=0.6970 val_joint_bal_acc=0.5757 val_joint_macro_f1=0.6088 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.1636 lr=6.4186e-05 train_metal_acc=0.9554 val_loss=2.1745 val_metal_acc=0.6813 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5760 val_joint_macro_f1=0.6107 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.1852 lr=6.4186e-05 train_metal_acc=0.9554 val_loss=2.1450 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5590 val_joint_macro_f1=0.5607 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.1977 lr=6.4186e-05 train_metal_acc=0.9622 val_loss=2.1372 val_metal_acc=0.6813 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5794 val_joint_macro_f1=0.6036 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.1386 lr=6.4186e-05 train_metal_acc=0.9631 val_loss=2.6118 val_metal_acc=0.6868 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5561 val_joint_macro_f1=0.5998 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.1583 lr=6.4186e-05 train_metal_acc=0.9631 val_loss=2.2318 val_metal_acc=0.6813 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5545 val_joint_macro_f1=0.5922 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.1281 lr=6.4186e-05 train_metal_acc=0.9641 val_loss=2.3593 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.6970 val_joint_bal_acc=0.5353 val_joint_macro_f1=0.5724 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0005_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_f07cdf8c/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0005_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_f07cdf8c/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0005_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_f07cdf8c/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0005_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_f07cdf8c/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0005_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_f07cdf8c/run_metadata.json
[I 2026-05-14 01:28:03,764] Trial 5 finished with value: 0.6105840606789373 and parameters: {'learning_rate': 6.418597685324415e-05, 'weight_decay': 0.0, 'hidden_s': 256, 'head_mlp_layers': 1, 'edge_hidden': 64, 'gvp_layers': 4, 'edge_radius': 8.0, 'hidden_v': 16, 'esm_fusion_dim': 64, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 2 with value: 0.6313553408430069.
Optuna trial 5 completed: val_metal_balanced_acc=0.6105840606789373
================================================================================
[Optuna trial 6] optuna_deepmzyme_controlled_hpo_trial0006_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d0b2747e
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 64,
  "edge_radius": 6.0,
  "esm_fusion_dim": 256,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 3,
  "head_mlp_layers": 3,
  "hidden_s": 256,
  "hidden_v": 32,
  "learning_rate": 1.9268171109476203e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 1e-05,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0006_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d0b2747e --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 1.9268171109476203e-05 --weight-decay 1e-05 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 256 --esm-fusion-dim 256 --head-mlp-layers 3 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 64 --gvp-layers 3 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7855 lr=1.92682e-05 train_metal_acc=0.4607 val_loss=1.7548 val_metal_acc=0.4670 val_metal_min_recall=0.0000 val_fe_recall=0.0000 val_joint_bal_acc=0.1667 val_joint_macro_f1=0.1061 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.7700 lr=1.92682e-05 train_metal_acc=0.5170 val_loss=1.7364 val_metal_acc=0.4945 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.2383 val_joint_macro_f1=0.1846 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.7335 lr=1.92682e-05 train_metal_acc=0.5907 val_loss=1.6795 val_metal_acc=0.4011 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.3499 val_joint_macro_f1=0.3375 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.6544 lr=1.92682e-05 train_metal_acc=0.6091 val_loss=1.5780 val_metal_acc=0.6374 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4078 val_joint_macro_f1=0.3929 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.5627 lr=1.92682e-05 train_metal_acc=0.6469 val_loss=1.4771 val_metal_acc=0.4341 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.3830 val_joint_macro_f1=0.3688 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.4573 lr=1.92682e-05 train_metal_acc=0.6566 val_loss=1.3592 val_metal_acc=0.6538 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4355 val_joint_macro_f1=0.4380 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=1.3583 lr=1.92682e-05 train_metal_acc=0.6945 val_loss=1.2889 val_metal_acc=0.4451 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4593 val_joint_macro_f1=0.4586 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=1.2606 lr=1.92682e-05 train_metal_acc=0.6945 val_loss=1.2411 val_metal_acc=0.4560 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4264 val_joint_macro_f1=0.4251 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=1.1812 lr=1.92682e-05 train_metal_acc=0.7304 val_loss=1.1955 val_metal_acc=0.4725 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4613 val_joint_macro_f1=0.4722 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=1.1217 lr=1.92682e-05 train_metal_acc=0.7468 val_loss=1.1915 val_metal_acc=0.4725 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.5118 val_joint_macro_f1=0.5069 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=1.0692 lr=1.92682e-05 train_metal_acc=0.7401 val_loss=1.1762 val_metal_acc=0.4725 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.5152 val_joint_macro_f1=0.5055 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.9959 lr=1.92682e-05 train_metal_acc=0.7536 val_loss=1.1749 val_metal_acc=0.4725 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4749 val_joint_macro_f1=0.4661 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.9632 lr=1.92682e-05 train_metal_acc=0.7847 val_loss=1.1415 val_metal_acc=0.5549 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5422 val_joint_macro_f1=0.5494 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.9145 lr=1.92682e-05 train_metal_acc=0.7643 val_loss=1.1527 val_metal_acc=0.4670 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.4920 val_joint_macro_f1=0.4990 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.8485 lr=1.92682e-05 train_metal_acc=0.7886 val_loss=1.0829 val_metal_acc=0.6813 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.5476 val_joint_macro_f1=0.5805 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.8045 lr=1.92682e-05 train_metal_acc=0.7924 val_loss=1.1380 val_metal_acc=0.5659 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5495 val_joint_macro_f1=0.5524 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.7721 lr=1.92682e-05 train_metal_acc=0.7798 val_loss=1.1860 val_metal_acc=0.5055 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5348 val_joint_macro_f1=0.5220 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.7406 lr=1.92682e-05 train_metal_acc=0.8254 val_loss=1.0780 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.6088 val_joint_macro_f1=0.6228 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.6782 lr=1.92682e-05 train_metal_acc=0.8274 val_loss=1.0913 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5867 val_joint_macro_f1=0.5934 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.6420 lr=1.92682e-05 train_metal_acc=0.8080 val_loss=1.0982 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5843 val_joint_macro_f1=0.5826 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.6274 lr=1.92682e-05 train_metal_acc=0.8186 val_loss=1.1433 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5936 val_joint_macro_f1=0.5952 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.6014 lr=1.92682e-05 train_metal_acc=0.8535 val_loss=1.0823 val_metal_acc=0.7527 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.6127 val_joint_macro_f1=0.6346 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.5417 lr=1.92682e-05 train_metal_acc=0.8371 val_loss=1.1648 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5846 val_joint_macro_f1=0.5718 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.5199 lr=1.92682e-05 train_metal_acc=0.8749 val_loss=1.1753 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5955 val_joint_macro_f1=0.6045 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.5451 lr=1.92682e-05 train_metal_acc=0.8332 val_loss=1.1381 val_metal_acc=0.6648 val_metal_min_recall=0.2308 val_fe_recall=0.6061 val_joint_bal_acc=0.6096 val_joint_macro_f1=0.6157 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.4985 lr=1.92682e-05 train_metal_acc=0.8739 val_loss=1.1750 val_metal_acc=0.7692 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6320 val_joint_macro_f1=0.6485 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.4568 lr=1.92682e-05 train_metal_acc=0.8952 val_loss=1.2038 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5694 val_joint_macro_f1=0.5822 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.4564 lr=1.92682e-05 train_metal_acc=0.8923 val_loss=1.2127 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6124 val_joint_macro_f1=0.6230 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.4268 lr=1.92682e-05 train_metal_acc=0.9030 val_loss=1.2961 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.6112 val_joint_macro_f1=0.6165 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.4019 lr=1.92682e-05 train_metal_acc=0.9001 val_loss=1.4050 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5866 val_joint_macro_f1=0.5758 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.3742 lr=1.92682e-05 train_metal_acc=0.8991 val_loss=1.3807 val_metal_acc=0.6648 val_metal_min_recall=0.1538 val_fe_recall=0.5455 val_joint_bal_acc=0.5994 val_joint_macro_f1=0.5965 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.3668 lr=1.92682e-05 train_metal_acc=0.9176 val_loss=1.4324 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.6122 val_joint_macro_f1=0.6205 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.3439 lr=1.92682e-05 train_metal_acc=0.9253 val_loss=1.4370 val_metal_acc=0.7582 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6212 val_joint_macro_f1=0.6296 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.3236 lr=1.92682e-05 train_metal_acc=0.9166 val_loss=1.5251 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.6176 val_joint_macro_f1=0.6242 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.3083 lr=1.92682e-05 train_metal_acc=0.9166 val_loss=1.4989 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5963 val_joint_macro_f1=0.6066 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.3144 lr=1.92682e-05 train_metal_acc=0.9253 val_loss=1.5812 val_metal_acc=0.7692 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6316 val_joint_macro_f1=0.6536 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.3169 lr=1.92682e-05 train_metal_acc=0.9273 val_loss=1.6286 val_metal_acc=0.7582 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.6181 val_joint_macro_f1=0.6310 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.2932 lr=1.92682e-05 train_metal_acc=0.9311 val_loss=1.6653 val_metal_acc=0.7637 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6266 val_joint_macro_f1=0.6374 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.2779 lr=1.92682e-05 train_metal_acc=0.9253 val_loss=1.5784 val_metal_acc=0.7582 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6246 val_joint_macro_f1=0.6350 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.3042 lr=1.92682e-05 train_metal_acc=0.9331 val_loss=1.6930 val_metal_acc=0.7582 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6212 val_joint_macro_f1=0.6329 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0006_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d0b2747e/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0006_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d0b2747e/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0006_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d0b2747e/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0006_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d0b2747e/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0006_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d0b2747e/run_metadata.json
[I 2026-05-14 01:36:37,663] Trial 6 finished with value: 0.6319607442947102 and parameters: {'learning_rate': 1.9268171109476203e-05, 'weight_decay': 1e-05, 'hidden_s': 256, 'head_mlp_layers': 3, 'edge_hidden': 64, 'gvp_layers': 3, 'edge_radius': 6.0, 'hidden_v': 32, 'esm_fusion_dim': 256, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 6 with value: 0.6319607442947102.
Optuna trial 6 completed: val_metal_balanced_acc=0.6319607442947102
================================================================================
[Optuna trial 7] optuna_deepmzyme_controlled_hpo_trial0007_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_338d57f2
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 64,
  "edge_radius": 6.0,
  "esm_fusion_dim": 64,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 4,
  "head_mlp_layers": 2,
  "hidden_s": 256,
  "hidden_v": 32,
  "learning_rate": 1.098436970170748e-05,
  "metal_class_weight_mode": "inverse_sqrt_frequency",
  "metal_weighting_setup": "inverse_sqrt_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 1e-05,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0007_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_338d57f2 --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 1.098436970170748e-05 --weight-decay 1e-05 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 256 --esm-fusion-dim 64 --head-mlp-layers 2 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_sqrt_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 64 --gvp-layers 4 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7203 lr=1.09844e-05 train_metal_acc=0.4607 val_loss=1.6378 val_metal_acc=0.4670 val_metal_min_recall=0.0000 val_fe_recall=0.0000 val_joint_bal_acc=0.1667 val_joint_macro_f1=0.1061 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.6542 lr=1.09844e-05 train_metal_acc=0.4898 val_loss=1.6131 val_metal_acc=0.4835 val_metal_min_recall=0.0000 val_fe_recall=0.0909 val_joint_bal_acc=0.1818 val_joint_macro_f1=0.1345 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.6197 lr=1.09844e-05 train_metal_acc=0.4918 val_loss=1.5713 val_metal_acc=0.4835 val_metal_min_recall=0.0000 val_fe_recall=0.0909 val_joint_bal_acc=0.1818 val_joint_macro_f1=0.1345 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.5921 lr=1.09844e-05 train_metal_acc=0.5209 val_loss=1.5607 val_metal_acc=0.4945 val_metal_min_recall=0.0000 val_fe_recall=0.7576 val_joint_bal_acc=0.2537 val_joint_macro_f1=0.1920 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.5497 lr=1.09844e-05 train_metal_acc=0.5179 val_loss=1.5308 val_metal_acc=0.4835 val_metal_min_recall=0.0000 val_fe_recall=0.9394 val_joint_bal_acc=0.2683 val_joint_macro_f1=0.1912 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.5029 lr=1.09844e-05 train_metal_acc=0.5616 val_loss=1.4332 val_metal_acc=0.5934 val_metal_min_recall=0.0000 val_fe_recall=0.7576 val_joint_bal_acc=0.2890 val_joint_macro_f1=0.2332 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=1.4426 lr=1.09844e-05 train_metal_acc=0.5955 val_loss=1.3739 val_metal_acc=0.6703 val_metal_min_recall=0.0000 val_fe_recall=0.8788 val_joint_bal_acc=0.4225 val_joint_macro_f1=0.3959 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=1.3756 lr=1.09844e-05 train_metal_acc=0.6343 val_loss=1.3102 val_metal_acc=0.6264 val_metal_min_recall=0.0000 val_fe_recall=0.6667 val_joint_bal_acc=0.3784 val_joint_macro_f1=0.3611 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=1.3198 lr=1.09844e-05 train_metal_acc=0.6731 val_loss=1.2643 val_metal_acc=0.6758 val_metal_min_recall=0.0000 val_fe_recall=0.6970 val_joint_bal_acc=0.4339 val_joint_macro_f1=0.4318 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=1.2847 lr=1.09844e-05 train_metal_acc=0.6460 val_loss=1.2384 val_metal_acc=0.6484 val_metal_min_recall=0.0000 val_fe_recall=0.6667 val_joint_bal_acc=0.4073 val_joint_macro_f1=0.4021 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=1.2352 lr=1.09844e-05 train_metal_acc=0.6499 val_loss=1.2730 val_metal_acc=0.6319 val_metal_min_recall=0.0000 val_fe_recall=0.9394 val_joint_bal_acc=0.4539 val_joint_macro_f1=0.4520 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=1.1783 lr=1.09844e-05 train_metal_acc=0.6673 val_loss=1.1719 val_metal_acc=0.6484 val_metal_min_recall=0.0000 val_fe_recall=0.6970 val_joint_bal_acc=0.4070 val_joint_macro_f1=0.3974 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=1.1407 lr=1.09844e-05 train_metal_acc=0.6974 val_loss=1.1330 val_metal_acc=0.7033 val_metal_min_recall=0.0000 val_fe_recall=0.7273 val_joint_bal_acc=0.5122 val_joint_macro_f1=0.5210 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=1.0915 lr=1.09844e-05 train_metal_acc=0.7061 val_loss=1.1279 val_metal_acc=0.6813 val_metal_min_recall=0.0000 val_fe_recall=0.7273 val_joint_bal_acc=0.4389 val_joint_macro_f1=0.4373 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=1.0669 lr=1.09844e-05 train_metal_acc=0.7129 val_loss=1.1636 val_metal_acc=0.6154 val_metal_min_recall=0.0000 val_fe_recall=0.5152 val_joint_bal_acc=0.4285 val_joint_macro_f1=0.4279 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=1.0399 lr=1.09844e-05 train_metal_acc=0.7294 val_loss=1.1009 val_metal_acc=0.6813 val_metal_min_recall=0.0000 val_fe_recall=0.6667 val_joint_bal_acc=0.4976 val_joint_macro_f1=0.5215 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=1.0080 lr=1.09844e-05 train_metal_acc=0.7410 val_loss=1.1202 val_metal_acc=0.5659 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4918 val_joint_macro_f1=0.4887 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.9646 lr=1.09844e-05 train_metal_acc=0.7575 val_loss=1.1110 val_metal_acc=0.5879 val_metal_min_recall=0.0000 val_fe_recall=0.5455 val_joint_bal_acc=0.4724 val_joint_macro_f1=0.4795 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.9317 lr=1.09844e-05 train_metal_acc=0.7536 val_loss=1.0927 val_metal_acc=0.6154 val_metal_min_recall=0.0000 val_fe_recall=0.5455 val_joint_bal_acc=0.4890 val_joint_macro_f1=0.4939 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.8975 lr=1.09844e-05 train_metal_acc=0.7692 val_loss=1.1131 val_metal_acc=0.5549 val_metal_min_recall=0.0000 val_fe_recall=0.5455 val_joint_bal_acc=0.4777 val_joint_macro_f1=0.4784 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.8877 lr=1.09844e-05 train_metal_acc=0.7711 val_loss=1.1070 val_metal_acc=0.5495 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4853 val_joint_macro_f1=0.4986 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.8615 lr=1.09844e-05 train_metal_acc=0.7895 val_loss=1.1070 val_metal_acc=0.5549 val_metal_min_recall=0.0000 val_fe_recall=0.5455 val_joint_bal_acc=0.4811 val_joint_macro_f1=0.5009 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.8443 lr=1.09844e-05 train_metal_acc=0.7798 val_loss=1.0964 val_metal_acc=0.5879 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.5243 val_joint_macro_f1=0.5402 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.8138 lr=1.09844e-05 train_metal_acc=0.7944 val_loss=1.0623 val_metal_acc=0.6429 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.5483 val_joint_macro_f1=0.5578 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.7951 lr=1.09844e-05 train_metal_acc=0.7963 val_loss=1.0422 val_metal_acc=0.6484 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.5349 val_joint_macro_f1=0.5424 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.7636 lr=1.09844e-05 train_metal_acc=0.7983 val_loss=1.0190 val_metal_acc=0.6703 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.5760 val_joint_macro_f1=0.5736 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.7527 lr=1.09844e-05 train_metal_acc=0.8089 val_loss=1.0459 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5366 val_joint_macro_f1=0.5695 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.7296 lr=1.09844e-05 train_metal_acc=0.8118 val_loss=1.0484 val_metal_acc=0.6538 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.5478 val_joint_macro_f1=0.5619 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.7065 lr=1.09844e-05 train_metal_acc=0.8264 val_loss=1.0318 val_metal_acc=0.6648 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.5521 val_joint_macro_f1=0.5672 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.6714 lr=1.09844e-05 train_metal_acc=0.8157 val_loss=0.9835 val_metal_acc=0.6648 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.5404 val_joint_macro_f1=0.5477 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.6722 lr=1.09844e-05 train_metal_acc=0.8264 val_loss=1.0040 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5470 val_joint_macro_f1=0.5877 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.6558 lr=1.09844e-05 train_metal_acc=0.8351 val_loss=0.9777 val_metal_acc=0.6484 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.5356 val_joint_macro_f1=0.5580 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.6388 lr=1.09844e-05 train_metal_acc=0.8371 val_loss=1.0144 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5621 val_joint_macro_f1=0.5929 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.6233 lr=1.09844e-05 train_metal_acc=0.8371 val_loss=1.0113 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5771 val_joint_macro_f1=0.6026 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.6021 lr=1.09844e-05 train_metal_acc=0.8283 val_loss=0.9480 val_metal_acc=0.6374 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.5061 val_joint_macro_f1=0.5350 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.5968 lr=1.09844e-05 train_metal_acc=0.8535 val_loss=1.0400 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5828 val_joint_macro_f1=0.5978 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.5805 lr=1.09844e-05 train_metal_acc=0.8506 val_loss=0.9867 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5800 val_joint_macro_f1=0.6065 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.5806 lr=1.09844e-05 train_metal_acc=0.8535 val_loss=0.9982 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5665 val_joint_macro_f1=0.5954 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.5701 lr=1.09844e-05 train_metal_acc=0.8565 val_loss=1.0101 val_metal_acc=0.6868 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5705 val_joint_macro_f1=0.6046 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.5435 lr=1.09844e-05 train_metal_acc=0.8623 val_loss=0.9756 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5873 val_joint_macro_f1=0.6150 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0007_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_338d57f2/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0007_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_338d57f2/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0007_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_338d57f2/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0007_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_338d57f2/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0007_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_338d57f2/run_metadata.json
[I 2026-05-14 01:45:41,437] Trial 7 finished with value: 0.5873150463283291 and parameters: {'learning_rate': 1.098436970170748e-05, 'weight_decay': 1e-05, 'hidden_s': 256, 'head_mlp_layers': 2, 'edge_hidden': 64, 'gvp_layers': 4, 'edge_radius': 6.0, 'hidden_v': 32, 'esm_fusion_dim': 64, 'metal_class_weight_mode': 'inverse_sqrt_frequency'}. Best is trial 6 with value: 0.6319607442947102.
Optuna trial 7 completed: val_metal_balanced_acc=0.5873150463283291
================================================================================
[Optuna trial 8] optuna_deepmzyme_controlled_hpo_trial0008_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_93d46ebc
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 6.0,
  "esm_fusion_dim": 64,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 2,
  "head_mlp_layers": 1,
  "hidden_s": 128,
  "hidden_v": 32,
  "learning_rate": 7.950932329069363e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.001,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0008_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_93d46ebc --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 7.950932329069363e-05 --weight-decay 0.001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7475 lr=7.95093e-05 train_metal_acc=0.5383 val_loss=1.5598 val_metal_acc=0.5769 val_metal_min_recall=0.0000 val_fe_recall=0.4242 val_joint_bal_acc=0.3795 val_joint_macro_f1=0.3622 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.4959 lr=7.95093e-05 train_metal_acc=0.6654 val_loss=1.3493 val_metal_acc=0.6648 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4736 val_joint_macro_f1=0.4937 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.2819 lr=7.95093e-05 train_metal_acc=0.6974 val_loss=1.2542 val_metal_acc=0.5989 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4972 val_joint_macro_f1=0.5021 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.1427 lr=7.95093e-05 train_metal_acc=0.6576 val_loss=1.2889 val_metal_acc=0.4176 val_metal_min_recall=0.0000 val_fe_recall=0.5455 val_joint_bal_acc=0.4710 val_joint_macro_f1=0.4376 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.0149 lr=7.95093e-05 train_metal_acc=0.7682 val_loss=1.1344 val_metal_acc=0.6868 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4855 val_joint_macro_f1=0.4985 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=0.9283 lr=7.95093e-05 train_metal_acc=0.7721 val_loss=1.2126 val_metal_acc=0.4670 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5291 val_joint_macro_f1=0.5175 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=0.8313 lr=7.95093e-05 train_metal_acc=0.7915 val_loss=1.1760 val_metal_acc=0.5385 val_metal_min_recall=0.1538 val_fe_recall=0.4848 val_joint_bal_acc=0.5734 val_joint_macro_f1=0.5558 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=0.7552 lr=7.95093e-05 train_metal_acc=0.8274 val_loss=1.1425 val_metal_acc=0.6154 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5879 val_joint_macro_f1=0.6058 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.7152 lr=7.95093e-05 train_metal_acc=0.8147 val_loss=1.1227 val_metal_acc=0.6484 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5904 val_joint_macro_f1=0.5982 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.6531 lr=7.95093e-05 train_metal_acc=0.8497 val_loss=1.0650 val_metal_acc=0.6538 val_metal_min_recall=0.1538 val_fe_recall=0.4848 val_joint_bal_acc=0.5825 val_joint_macro_f1=0.5849 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.5828 lr=7.95093e-05 train_metal_acc=0.8565 val_loss=1.0838 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5723 val_joint_macro_f1=0.5917 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.5632 lr=7.95093e-05 train_metal_acc=0.8594 val_loss=1.0251 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.6027 val_joint_macro_f1=0.6093 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.5182 lr=7.95093e-05 train_metal_acc=0.8768 val_loss=1.0460 val_metal_acc=0.6978 val_metal_min_recall=0.1538 val_fe_recall=0.5455 val_joint_bal_acc=0.6043 val_joint_macro_f1=0.6015 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.4639 lr=7.95093e-05 train_metal_acc=0.8855 val_loss=1.0930 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.6157 val_joint_macro_f1=0.6321 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.4247 lr=7.95093e-05 train_metal_acc=0.8914 val_loss=1.0464 val_metal_acc=0.6923 val_metal_min_recall=0.1538 val_fe_recall=0.6970 val_joint_bal_acc=0.6178 val_joint_macro_f1=0.6425 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.4148 lr=7.95093e-05 train_metal_acc=0.8904 val_loss=1.2242 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5628 val_joint_macro_f1=0.5833 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.4006 lr=7.95093e-05 train_metal_acc=0.9020 val_loss=1.1179 val_metal_acc=0.6868 val_metal_min_recall=0.1538 val_fe_recall=0.6970 val_joint_bal_acc=0.6124 val_joint_macro_f1=0.6504 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.3828 lr=7.95093e-05 train_metal_acc=0.9030 val_loss=1.0872 val_metal_acc=0.7473 val_metal_min_recall=0.3077 val_fe_recall=0.6364 val_joint_bal_acc=0.6530 val_joint_macro_f1=0.6842 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.3431 lr=7.95093e-05 train_metal_acc=0.9127 val_loss=1.0899 val_metal_acc=0.7143 val_metal_min_recall=0.2308 val_fe_recall=0.6364 val_joint_bal_acc=0.6303 val_joint_macro_f1=0.6433 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.3228 lr=7.95093e-05 train_metal_acc=0.9117 val_loss=1.1845 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.6177 val_joint_macro_f1=0.6457 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.3209 lr=7.95093e-05 train_metal_acc=0.9176 val_loss=1.2367 val_metal_acc=0.7527 val_metal_min_recall=0.1538 val_fe_recall=0.6667 val_joint_bal_acc=0.6158 val_joint_macro_f1=0.6525 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.2864 lr=7.95093e-05 train_metal_acc=0.9059 val_loss=1.1035 val_metal_acc=0.7637 val_metal_min_recall=0.3077 val_fe_recall=0.6970 val_joint_bal_acc=0.6514 val_joint_macro_f1=0.6718 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.2701 lr=7.95093e-05 train_metal_acc=0.9117 val_loss=1.3253 val_metal_acc=0.7198 val_metal_min_recall=0.2308 val_fe_recall=0.6970 val_joint_bal_acc=0.5893 val_joint_macro_f1=0.6380 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.2801 lr=7.95093e-05 train_metal_acc=0.9253 val_loss=1.2260 val_metal_acc=0.7363 val_metal_min_recall=0.3077 val_fe_recall=0.6364 val_joint_bal_acc=0.6524 val_joint_macro_f1=0.6750 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.2558 lr=7.95093e-05 train_metal_acc=0.9399 val_loss=1.4222 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5847 val_joint_macro_f1=0.6142 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.2391 lr=7.95093e-05 train_metal_acc=0.9331 val_loss=1.4767 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6077 val_joint_macro_f1=0.6401 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.2437 lr=7.95093e-05 train_metal_acc=0.9379 val_loss=1.5630 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6142 val_joint_macro_f1=0.6526 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.2183 lr=7.95093e-05 train_metal_acc=0.9263 val_loss=1.4046 val_metal_acc=0.7473 val_metal_min_recall=0.3077 val_fe_recall=0.6061 val_joint_bal_acc=0.6328 val_joint_macro_f1=0.6620 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.2270 lr=7.95093e-05 train_metal_acc=0.9418 val_loss=1.5502 val_metal_acc=0.7198 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5876 val_joint_macro_f1=0.6152 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.2118 lr=7.95093e-05 train_metal_acc=0.9292 val_loss=1.4066 val_metal_acc=0.7088 val_metal_min_recall=0.3077 val_fe_recall=0.6061 val_joint_bal_acc=0.6327 val_joint_macro_f1=0.6529 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.2387 lr=7.95093e-05 train_metal_acc=0.9447 val_loss=1.5461 val_metal_acc=0.7527 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6230 val_joint_macro_f1=0.6658 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.1832 lr=7.95093e-05 train_metal_acc=0.9447 val_loss=1.6132 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.5980 val_joint_macro_f1=0.6435 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.1973 lr=7.95093e-05 train_metal_acc=0.9408 val_loss=1.5888 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6120 val_joint_macro_f1=0.6431 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.2092 lr=7.95093e-05 train_metal_acc=0.9437 val_loss=1.7651 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.5946 val_joint_macro_f1=0.6399 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.2150 lr=7.95093e-05 train_metal_acc=0.9467 val_loss=1.8497 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5881 val_joint_macro_f1=0.6280 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.2283 lr=7.95093e-05 train_metal_acc=0.9467 val_loss=1.7932 val_metal_acc=0.7143 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5638 val_joint_macro_f1=0.6115 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.2018 lr=7.95093e-05 train_metal_acc=0.9486 val_loss=1.8334 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5949 val_joint_macro_f1=0.6342 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.1948 lr=7.95093e-05 train_metal_acc=0.9389 val_loss=1.7824 val_metal_acc=0.7143 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.5887 val_joint_macro_f1=0.6056 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.1727 lr=7.95093e-05 train_metal_acc=0.9486 val_loss=1.9621 val_metal_acc=0.7418 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6057 val_joint_macro_f1=0.6448 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.1678 lr=7.95093e-05 train_metal_acc=0.9525 val_loss=1.8657 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.5991 val_joint_macro_f1=0.6420 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0008_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_93d46ebc/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0008_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_93d46ebc/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0008_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_93d46ebc/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0008_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_93d46ebc/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0008_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_93d46ebc/run_metadata.json
[I 2026-05-14 01:53:43,997] Trial 8 finished with value: 0.6529553937523577 and parameters: {'learning_rate': 7.950932329069363e-05, 'weight_decay': 0.001, 'hidden_s': 128, 'head_mlp_layers': 1, 'edge_hidden': 128, 'gvp_layers': 2, 'edge_radius': 6.0, 'hidden_v': 32, 'esm_fusion_dim': 64, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 8 with value: 0.6529553937523577.
Optuna trial 8 completed: val_metal_balanced_acc=0.6529553937523577
================================================================================
[Optuna trial 9] optuna_deepmzyme_controlled_hpo_trial0009_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_921d5065
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 64,
  "edge_radius": 8.0,
  "esm_fusion_dim": 64,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 2,
  "head_mlp_layers": 1,
  "hidden_s": 128,
  "hidden_v": 16,
  "learning_rate": 2.115435061637711e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.0001,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0009_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_921d5065 --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 2.115435061637711e-05 --weight-decay 0.0001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 64 --gvp-layers 2 --edge-radius 8.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 16 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7947 lr=2.11544e-05 train_metal_acc=0.4462 val_loss=1.7396 val_metal_acc=0.5055 val_metal_min_recall=0.0000 val_fe_recall=0.2424 val_joint_bal_acc=0.2262 val_joint_macro_f1=0.2136 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.7479 lr=2.11544e-05 train_metal_acc=0.5044 val_loss=1.7011 val_metal_acc=0.5220 val_metal_min_recall=0.0000 val_fe_recall=0.4848 val_joint_bal_acc=0.3334 val_joint_macro_f1=0.3268 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.6937 lr=2.11544e-05 train_metal_acc=0.4743 val_loss=1.6453 val_metal_acc=0.3297 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.3567 val_joint_macro_f1=0.3191 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.6240 lr=2.11544e-05 train_metal_acc=0.5703 val_loss=1.5247 val_metal_acc=0.6154 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4116 val_joint_macro_f1=0.4072 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.5444 lr=2.11544e-05 train_metal_acc=0.6334 val_loss=1.4585 val_metal_acc=0.3901 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.4014 val_joint_macro_f1=0.4236 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.4751 lr=2.11544e-05 train_metal_acc=0.6440 val_loss=1.3836 val_metal_acc=0.6099 val_metal_min_recall=0.0000 val_fe_recall=0.5152 val_joint_bal_acc=0.4566 val_joint_macro_f1=0.4462 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=1.4207 lr=2.11544e-05 train_metal_acc=0.6605 val_loss=1.3635 val_metal_acc=0.4066 val_metal_min_recall=0.0000 val_fe_recall=0.5455 val_joint_bal_acc=0.4534 val_joint_macro_f1=0.4154 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=1.3557 lr=2.11544e-05 train_metal_acc=0.6566 val_loss=1.3268 val_metal_acc=0.4066 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4388 val_joint_macro_f1=0.4144 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=1.2965 lr=2.11544e-05 train_metal_acc=0.6702 val_loss=1.3086 val_metal_acc=0.4121 val_metal_min_recall=0.0000 val_fe_recall=0.5152 val_joint_bal_acc=0.4373 val_joint_macro_f1=0.4146 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=1.2512 lr=2.11544e-05 train_metal_acc=0.6935 val_loss=1.2683 val_metal_acc=0.4451 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4549 val_joint_macro_f1=0.4538 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=1.2137 lr=2.11544e-05 train_metal_acc=0.6440 val_loss=1.2877 val_metal_acc=0.4011 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.4654 val_joint_macro_f1=0.4326 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=1.1502 lr=2.11544e-05 train_metal_acc=0.7110 val_loss=1.2441 val_metal_acc=0.4451 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.4729 val_joint_macro_f1=0.4652 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=1.1112 lr=2.11544e-05 train_metal_acc=0.7236 val_loss=1.1817 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5806 val_joint_macro_f1=0.5706 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=1.0686 lr=2.11544e-05 train_metal_acc=0.7420 val_loss=1.2090 val_metal_acc=0.4396 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4717 val_joint_macro_f1=0.4315 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=1.0101 lr=2.11544e-05 train_metal_acc=0.7274 val_loss=1.1778 val_metal_acc=0.5055 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5482 val_joint_macro_f1=0.4992 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.9971 lr=2.11544e-05 train_metal_acc=0.7362 val_loss=1.1747 val_metal_acc=0.4560 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.5366 val_joint_macro_f1=0.4908 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.9693 lr=2.11544e-05 train_metal_acc=0.7643 val_loss=1.1505 val_metal_acc=0.4615 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5276 val_joint_macro_f1=0.4889 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.9305 lr=2.11544e-05 train_metal_acc=0.7585 val_loss=1.2041 val_metal_acc=0.4451 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5251 val_joint_macro_f1=0.4812 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.9155 lr=2.11544e-05 train_metal_acc=0.7721 val_loss=1.1225 val_metal_acc=0.5220 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5457 val_joint_macro_f1=0.5194 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.8622 lr=2.11544e-05 train_metal_acc=0.7818 val_loss=1.1642 val_metal_acc=0.4560 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5297 val_joint_macro_f1=0.4940 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.8541 lr=2.11544e-05 train_metal_acc=0.7818 val_loss=1.1304 val_metal_acc=0.6099 val_metal_min_recall=0.0769 val_fe_recall=0.4545 val_joint_bal_acc=0.5678 val_joint_macro_f1=0.5315 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.8035 lr=2.11544e-05 train_metal_acc=0.7915 val_loss=1.1365 val_metal_acc=0.5275 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5549 val_joint_macro_f1=0.5206 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.7842 lr=2.11544e-05 train_metal_acc=0.7963 val_loss=1.0388 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.6038 val_joint_macro_f1=0.5940 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.7554 lr=2.11544e-05 train_metal_acc=0.8254 val_loss=1.1096 val_metal_acc=0.6154 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5797 val_joint_macro_f1=0.5633 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.7361 lr=2.11544e-05 train_metal_acc=0.8235 val_loss=1.0573 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5657 val_joint_macro_f1=0.5616 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.7215 lr=2.11544e-05 train_metal_acc=0.8225 val_loss=1.0647 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5841 val_joint_macro_f1=0.5602 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.6950 lr=2.11544e-05 train_metal_acc=0.8380 val_loss=1.0391 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5662 val_joint_macro_f1=0.5704 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.6734 lr=2.11544e-05 train_metal_acc=0.8167 val_loss=1.0703 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5846 val_joint_macro_f1=0.5706 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.6479 lr=2.11544e-05 train_metal_acc=0.8361 val_loss=1.0272 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5876 val_joint_macro_f1=0.5718 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.6275 lr=2.11544e-05 train_metal_acc=0.8380 val_loss=1.0568 val_metal_acc=0.6319 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5385 val_joint_macro_f1=0.5320 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.6136 lr=2.11544e-05 train_metal_acc=0.8215 val_loss=1.0846 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5725 val_joint_macro_f1=0.5539 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.5993 lr=2.11544e-05 train_metal_acc=0.8506 val_loss=1.0318 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5688 val_joint_macro_f1=0.5598 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.5754 lr=2.11544e-05 train_metal_acc=0.8545 val_loss=1.0309 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5623 val_joint_macro_f1=0.5568 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.5483 lr=2.11544e-05 train_metal_acc=0.8555 val_loss=1.0488 val_metal_acc=0.6978 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.6122 val_joint_macro_f1=0.5917 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.5600 lr=2.11544e-05 train_metal_acc=0.8729 val_loss=1.0602 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.4545 val_joint_bal_acc=0.5729 val_joint_macro_f1=0.5619 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.5235 lr=2.11544e-05 train_metal_acc=0.8448 val_loss=1.0638 val_metal_acc=0.6209 val_metal_min_recall=0.1538 val_fe_recall=0.4545 val_joint_bal_acc=0.5458 val_joint_macro_f1=0.5392 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.5209 lr=2.11544e-05 train_metal_acc=0.8661 val_loss=1.0677 val_metal_acc=0.6319 val_metal_min_recall=0.0769 val_fe_recall=0.4545 val_joint_bal_acc=0.5422 val_joint_macro_f1=0.5321 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.5041 lr=2.11544e-05 train_metal_acc=0.8768 val_loss=1.0567 val_metal_acc=0.6813 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5654 val_joint_macro_f1=0.5766 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.4872 lr=2.11544e-05 train_metal_acc=0.8720 val_loss=1.1036 val_metal_acc=0.6429 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5554 val_joint_macro_f1=0.5558 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.4882 lr=2.11544e-05 train_metal_acc=0.8846 val_loss=1.0692 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5791 val_joint_macro_f1=0.5942 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0009_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_921d5065/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0009_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_921d5065/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0009_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_921d5065/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0009_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_921d5065/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0009_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_921d5065/run_metadata.json
[I 2026-05-14 02:02:29,761] Trial 9 finished with value: 0.6122190059571464 and parameters: {'learning_rate': 2.115435061637711e-05, 'weight_decay': 0.0001, 'hidden_s': 128, 'head_mlp_layers': 1, 'edge_hidden': 64, 'gvp_layers': 2, 'edge_radius': 8.0, 'hidden_v': 16, 'esm_fusion_dim': 64, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 8 with value: 0.6529553937523577.
Optuna trial 9 completed: val_metal_balanced_acc=0.6122190059571464
================================================================================
[Optuna trial 10] optuna_deepmzyme_controlled_hpo_trial0010_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_0f0f1cd5
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 6.0,
  "esm_fusion_dim": 128,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 2,
  "head_mlp_layers": 1,
  "hidden_s": 128,
  "hidden_v": 32,
  "learning_rate": 9.534059717443556e-05,
  "metal_class_weight_mode": "inverse_sqrt_frequency",
  "metal_weighting_setup": "inverse_sqrt_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.001,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0010_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_0f0f1cd5 --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 9.534059717443556e-05 --weight-decay 0.001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 128 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_sqrt_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.6031 lr=9.53406e-05 train_metal_acc=0.5470 val_loss=1.4183 val_metal_acc=0.5659 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.2607 val_joint_macro_f1=0.2198 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.2908 lr=9.53406e-05 train_metal_acc=0.7042 val_loss=1.2407 val_metal_acc=0.6868 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4599 val_joint_macro_f1=0.4534 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.0653 lr=9.53406e-05 train_metal_acc=0.6799 val_loss=1.1853 val_metal_acc=0.6593 val_metal_min_recall=0.0000 val_fe_recall=0.4848 val_joint_bal_acc=0.4315 val_joint_macro_f1=0.4522 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=0.9216 lr=9.53406e-05 train_metal_acc=0.7953 val_loss=1.0419 val_metal_acc=0.7088 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.5674 val_joint_macro_f1=0.5901 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=0.7846 lr=9.53406e-05 train_metal_acc=0.8303 val_loss=1.0224 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5717 val_joint_macro_f1=0.5917 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=0.6890 lr=9.53406e-05 train_metal_acc=0.8419 val_loss=1.0374 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5513 val_joint_macro_f1=0.5671 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=0.6230 lr=9.53406e-05 train_metal_acc=0.8506 val_loss=0.9742 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5571 val_joint_macro_f1=0.5863 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=0.5582 lr=9.53406e-05 train_metal_acc=0.8788 val_loss=0.9674 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5410 val_joint_macro_f1=0.5786 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.5140 lr=9.53406e-05 train_metal_acc=0.8885 val_loss=1.0314 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5548 val_joint_macro_f1=0.5972 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.4705 lr=9.53406e-05 train_metal_acc=0.8846 val_loss=1.1101 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5355 val_joint_macro_f1=0.5746 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.4258 lr=9.53406e-05 train_metal_acc=0.9059 val_loss=1.0602 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5588 val_joint_macro_f1=0.6005 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.4042 lr=9.53406e-05 train_metal_acc=0.9020 val_loss=1.0918 val_metal_acc=0.7527 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.6174 val_joint_macro_f1=0.6583 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.3677 lr=9.53406e-05 train_metal_acc=0.9108 val_loss=1.1062 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5594 val_joint_macro_f1=0.5839 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.3155 lr=9.53406e-05 train_metal_acc=0.9108 val_loss=1.2272 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5407 val_joint_macro_f1=0.5807 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.3329 lr=9.53406e-05 train_metal_acc=0.9195 val_loss=1.1911 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5816 val_joint_macro_f1=0.5824 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.3111 lr=9.53406e-05 train_metal_acc=0.9331 val_loss=1.2220 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5943 val_joint_macro_f1=0.6089 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.2754 lr=9.53406e-05 train_metal_acc=0.9350 val_loss=1.2565 val_metal_acc=0.7253 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5930 val_joint_macro_f1=0.6288 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.2787 lr=9.53406e-05 train_metal_acc=0.9331 val_loss=1.3570 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5833 val_joint_macro_f1=0.6234 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.2547 lr=9.53406e-05 train_metal_acc=0.9350 val_loss=1.2983 val_metal_acc=0.6648 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5748 val_joint_macro_f1=0.6047 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.2528 lr=9.53406e-05 train_metal_acc=0.9399 val_loss=1.3312 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5816 val_joint_macro_f1=0.5859 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.2382 lr=9.53406e-05 train_metal_acc=0.9389 val_loss=1.4660 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5606 val_joint_macro_f1=0.5968 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.2477 lr=9.53406e-05 train_metal_acc=0.9321 val_loss=1.4945 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5788 val_joint_macro_f1=0.5855 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.2229 lr=9.53406e-05 train_metal_acc=0.9418 val_loss=1.4899 val_metal_acc=0.6813 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5207 val_joint_macro_f1=0.5587 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.2144 lr=9.53406e-05 train_metal_acc=0.9117 val_loss=1.5713 val_metal_acc=0.6429 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5732 val_joint_macro_f1=0.5884 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.2295 lr=9.53406e-05 train_metal_acc=0.9331 val_loss=1.4468 val_metal_acc=0.7088 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5803 val_joint_macro_f1=0.6100 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.2229 lr=9.53406e-05 train_metal_acc=0.9428 val_loss=1.6117 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5742 val_joint_macro_f1=0.5974 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.2173 lr=9.53406e-05 train_metal_acc=0.9437 val_loss=1.7485 val_metal_acc=0.6978 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5334 val_joint_macro_f1=0.5744 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.2294 lr=9.53406e-05 train_metal_acc=0.9476 val_loss=1.5795 val_metal_acc=0.6044 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5246 val_joint_macro_f1=0.5381 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.2073 lr=9.53406e-05 train_metal_acc=0.9496 val_loss=1.6042 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5571 val_joint_macro_f1=0.5532 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.1943 lr=9.53406e-05 train_metal_acc=0.9467 val_loss=1.7347 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5255 val_joint_macro_f1=0.5379 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.1865 lr=9.53406e-05 train_metal_acc=0.9573 val_loss=1.7761 val_metal_acc=0.6813 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5596 val_joint_macro_f1=0.5830 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.2023 lr=9.53406e-05 train_metal_acc=0.9544 val_loss=1.7895 val_metal_acc=0.6923 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5485 val_joint_macro_f1=0.5633 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.1806 lr=9.53406e-05 train_metal_acc=0.9573 val_loss=1.8279 val_metal_acc=0.6923 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5497 val_joint_macro_f1=0.5848 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.1986 lr=9.53406e-05 train_metal_acc=0.9680 val_loss=1.8994 val_metal_acc=0.6978 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5505 val_joint_macro_f1=0.5760 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.1614 lr=9.53406e-05 train_metal_acc=0.9593 val_loss=2.0740 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5550 val_joint_macro_f1=0.5776 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.1793 lr=9.53406e-05 train_metal_acc=0.9583 val_loss=1.9101 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5496 val_joint_macro_f1=0.5595 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.1836 lr=9.53406e-05 train_metal_acc=0.9631 val_loss=2.0413 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5434 val_joint_macro_f1=0.5708 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.1681 lr=9.53406e-05 train_metal_acc=0.9651 val_loss=2.1337 val_metal_acc=0.7143 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5385 val_joint_macro_f1=0.5867 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.1753 lr=9.53406e-05 train_metal_acc=0.9593 val_loss=2.1609 val_metal_acc=0.6923 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5232 val_joint_macro_f1=0.5449 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.1733 lr=9.53406e-05 train_metal_acc=0.9680 val_loss=2.1125 val_metal_acc=0.6978 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5218 val_joint_macro_f1=0.5423 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0010_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_0f0f1cd5/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0010_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_0f0f1cd5/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0010_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_0f0f1cd5/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0010_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_0f0f1cd5/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0010_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_0f0f1cd5/run_metadata.json
[I 2026-05-14 02:10:47,820] Trial 10 finished with value: 0.6174222783330943 and parameters: {'learning_rate': 9.534059717443556e-05, 'weight_decay': 0.001, 'hidden_s': 128, 'head_mlp_layers': 1, 'edge_hidden': 128, 'gvp_layers': 2, 'edge_radius': 6.0, 'hidden_v': 32, 'esm_fusion_dim': 128, 'metal_class_weight_mode': 'inverse_sqrt_frequency'}. Best is trial 8 with value: 0.6529553937523577.
Optuna trial 10 completed: val_metal_balanced_acc=0.6174222783330943
================================================================================
[Optuna trial 11] optuna_deepmzyme_controlled_hpo_trial0011_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_e1ed9295
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 6.0,
  "esm_fusion_dim": 256,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 3,
  "head_mlp_layers": 3,
  "hidden_s": 256,
  "hidden_v": 32,
  "learning_rate": 1.5616422459730675e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.001,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0011_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_e1ed9295 --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 1.5616422459730675e-05 --weight-decay 0.001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 256 --esm-fusion-dim 256 --head-mlp-layers 3 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 3 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7876 lr=1.56164e-05 train_metal_acc=0.4704 val_loss=1.7640 val_metal_acc=0.5000 val_metal_min_recall=0.0000 val_fe_recall=0.1818 val_joint_bal_acc=0.1970 val_joint_macro_f1=0.1590 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.7754 lr=1.56164e-05 train_metal_acc=0.5092 val_loss=1.7570 val_metal_acc=0.4176 val_metal_min_recall=0.0000 val_fe_recall=0.6667 val_joint_bal_acc=0.2170 val_joint_macro_f1=0.1610 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.7596 lr=1.56164e-05 train_metal_acc=0.5218 val_loss=1.7251 val_metal_acc=0.4286 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.2219 val_joint_macro_f1=0.1904 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.7128 lr=1.56164e-05 train_metal_acc=0.5723 val_loss=1.6538 val_metal_acc=0.5440 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.3810 val_joint_macro_f1=0.3788 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.6454 lr=1.56164e-05 train_metal_acc=0.6023 val_loss=1.5688 val_metal_acc=0.5440 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.3984 val_joint_macro_f1=0.3984 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.5732 lr=1.56164e-05 train_metal_acc=0.6072 val_loss=1.4860 val_metal_acc=0.6154 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4071 val_joint_macro_f1=0.4002 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=1.5041 lr=1.56164e-05 train_metal_acc=0.6324 val_loss=1.4093 val_metal_acc=0.6484 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4151 val_joint_macro_f1=0.3991 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=1.4335 lr=1.56164e-05 train_metal_acc=0.6479 val_loss=1.3424 val_metal_acc=0.5549 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4159 val_joint_macro_f1=0.4093 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=1.3434 lr=1.56164e-05 train_metal_acc=0.6634 val_loss=1.3000 val_metal_acc=0.4780 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4342 val_joint_macro_f1=0.4324 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=1.2847 lr=1.56164e-05 train_metal_acc=0.6722 val_loss=1.2506 val_metal_acc=0.5495 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4276 val_joint_macro_f1=0.4178 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=1.2124 lr=1.56164e-05 train_metal_acc=0.6896 val_loss=1.2211 val_metal_acc=0.5275 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4737 val_joint_macro_f1=0.4844 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=1.1636 lr=1.56164e-05 train_metal_acc=0.7245 val_loss=1.1914 val_metal_acc=0.5934 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4836 val_joint_macro_f1=0.5072 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=1.0897 lr=1.56164e-05 train_metal_acc=0.7333 val_loss=1.1919 val_metal_acc=0.6319 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.4958 val_joint_macro_f1=0.5120 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=1.0645 lr=1.56164e-05 train_metal_acc=0.7304 val_loss=1.1858 val_metal_acc=0.4780 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4766 val_joint_macro_f1=0.4763 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=1.0104 lr=1.56164e-05 train_metal_acc=0.7420 val_loss=1.1695 val_metal_acc=0.4560 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4551 val_joint_macro_f1=0.4672 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.9636 lr=1.56164e-05 train_metal_acc=0.7662 val_loss=1.1562 val_metal_acc=0.4890 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4774 val_joint_macro_f1=0.4691 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.9164 lr=1.56164e-05 train_metal_acc=0.7818 val_loss=1.1470 val_metal_acc=0.5769 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5128 val_joint_macro_f1=0.5268 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.8797 lr=1.56164e-05 train_metal_acc=0.7779 val_loss=1.1584 val_metal_acc=0.5110 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5367 val_joint_macro_f1=0.5365 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.8541 lr=1.56164e-05 train_metal_acc=0.7983 val_loss=1.1087 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5592 val_joint_macro_f1=0.5780 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.8090 lr=1.56164e-05 train_metal_acc=0.8012 val_loss=1.1164 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5403 val_joint_macro_f1=0.5428 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.7831 lr=1.56164e-05 train_metal_acc=0.7973 val_loss=1.1398 val_metal_acc=0.6209 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5288 val_joint_macro_f1=0.5379 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.7249 lr=1.56164e-05 train_metal_acc=0.8109 val_loss=1.1449 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5445 val_joint_macro_f1=0.5525 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.7043 lr=1.56164e-05 train_metal_acc=0.8235 val_loss=1.1606 val_metal_acc=0.6923 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5980 val_joint_macro_f1=0.6044 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.6787 lr=1.56164e-05 train_metal_acc=0.8080 val_loss=1.1919 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5600 val_joint_macro_f1=0.5574 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.6220 lr=1.56164e-05 train_metal_acc=0.8206 val_loss=1.2110 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5639 val_joint_macro_f1=0.5664 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.6345 lr=1.56164e-05 train_metal_acc=0.8235 val_loss=1.1883 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5678 val_joint_macro_f1=0.5747 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.5825 lr=1.56164e-05 train_metal_acc=0.8438 val_loss=1.1353 val_metal_acc=0.6813 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5941 val_joint_macro_f1=0.5890 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.5646 lr=1.56164e-05 train_metal_acc=0.8526 val_loss=1.2065 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5985 val_joint_macro_f1=0.6073 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.5450 lr=1.56164e-05 train_metal_acc=0.8516 val_loss=1.1745 val_metal_acc=0.6923 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.6014 val_joint_macro_f1=0.6047 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.4943 lr=1.56164e-05 train_metal_acc=0.8138 val_loss=1.2643 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5891 val_joint_macro_f1=0.5822 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.4859 lr=1.56164e-05 train_metal_acc=0.8661 val_loss=1.2015 val_metal_acc=0.6813 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5722 val_joint_macro_f1=0.5773 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.4727 lr=1.56164e-05 train_metal_acc=0.8758 val_loss=1.2268 val_metal_acc=0.6923 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5946 val_joint_macro_f1=0.5891 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.4645 lr=1.56164e-05 train_metal_acc=0.8826 val_loss=1.3150 val_metal_acc=0.6923 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5946 val_joint_macro_f1=0.6088 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.4666 lr=1.56164e-05 train_metal_acc=0.8894 val_loss=1.3148 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.6191 val_joint_macro_f1=0.6201 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.4253 lr=1.56164e-05 train_metal_acc=0.8952 val_loss=1.3464 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.6103 val_joint_macro_f1=0.6277 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.4179 lr=1.56164e-05 train_metal_acc=0.8962 val_loss=1.3935 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.6122 val_joint_macro_f1=0.6300 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.4138 lr=1.56164e-05 train_metal_acc=0.8982 val_loss=1.3174 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6111 val_joint_macro_f1=0.6041 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.3905 lr=1.56164e-05 train_metal_acc=0.9069 val_loss=1.4160 val_metal_acc=0.7582 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.6150 val_joint_macro_f1=0.6284 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.3920 lr=1.56164e-05 train_metal_acc=0.9020 val_loss=1.4653 val_metal_acc=0.7637 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6163 val_joint_macro_f1=0.6372 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.3548 lr=1.56164e-05 train_metal_acc=0.8962 val_loss=1.5538 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5955 val_joint_macro_f1=0.6048 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0011_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_e1ed9295/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0011_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_e1ed9295/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0011_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_e1ed9295/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0011_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_e1ed9295/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0011_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_e1ed9295/run_metadata.json
[I 2026-05-14 02:19:58,834] Trial 11 finished with value: 0.6190671019893031 and parameters: {'learning_rate': 1.5616422459730675e-05, 'weight_decay': 0.001, 'hidden_s': 256, 'head_mlp_layers': 3, 'edge_hidden': 128, 'gvp_layers': 3, 'edge_radius': 6.0, 'hidden_v': 32, 'esm_fusion_dim': 256, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 8 with value: 0.6529553937523577.
Optuna trial 11 completed: val_metal_balanced_acc=0.6190671019893031
================================================================================
[Optuna trial 12] optuna_deepmzyme_controlled_hpo_trial0012_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_bce5e2a1
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 6.0,
  "esm_fusion_dim": 256,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 3,
  "head_mlp_layers": 3,
  "hidden_s": 256,
  "hidden_v": 32,
  "learning_rate": 5.099642277855573e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 1e-05,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0012_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_bce5e2a1 --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 5.099642277855573e-05 --weight-decay 1e-05 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 256 --esm-fusion-dim 256 --head-mlp-layers 3 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 3 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7827 lr=5.09964e-05 train_metal_acc=0.5034 val_loss=1.7322 val_metal_acc=0.4780 val_metal_min_recall=0.0000 val_fe_recall=0.4242 val_joint_bal_acc=0.2343 val_joint_macro_f1=0.2156 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.6921 lr=5.09964e-05 train_metal_acc=0.5626 val_loss=1.5551 val_metal_acc=0.5989 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.3975 val_joint_macro_f1=0.3713 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.4844 lr=5.09964e-05 train_metal_acc=0.6188 val_loss=1.3307 val_metal_acc=0.4121 val_metal_min_recall=0.0000 val_fe_recall=0.6667 val_joint_bal_acc=0.3919 val_joint_macro_f1=0.3563 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.2734 lr=5.09964e-05 train_metal_acc=0.7032 val_loss=1.2134 val_metal_acc=0.6209 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4838 val_joint_macro_f1=0.5002 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.1232 lr=5.09964e-05 train_metal_acc=0.7061 val_loss=1.2152 val_metal_acc=0.4341 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.4764 val_joint_macro_f1=0.4862 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=0.9973 lr=5.09964e-05 train_metal_acc=0.7439 val_loss=1.1729 val_metal_acc=0.5055 val_metal_min_recall=0.1429 val_fe_recall=0.5455 val_joint_bal_acc=0.4770 val_joint_macro_f1=0.4811 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=0.9047 lr=5.09964e-05 train_metal_acc=0.7944 val_loss=1.1265 val_metal_acc=0.6813 val_metal_min_recall=0.1429 val_fe_recall=0.5758 val_joint_bal_acc=0.5394 val_joint_macro_f1=0.5604 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=0.8054 lr=5.09964e-05 train_metal_acc=0.8109 val_loss=1.0934 val_metal_acc=0.7527 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5956 val_joint_macro_f1=0.6187 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.6916 lr=5.09964e-05 train_metal_acc=0.7808 val_loss=1.1716 val_metal_acc=0.6538 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.6020 val_joint_macro_f1=0.5967 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.6632 lr=5.09964e-05 train_metal_acc=0.8419 val_loss=1.1528 val_metal_acc=0.6813 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5975 val_joint_macro_f1=0.5982 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.5592 lr=5.09964e-05 train_metal_acc=0.8642 val_loss=1.1905 val_metal_acc=0.7637 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.6241 val_joint_macro_f1=0.6424 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.5451 lr=5.09964e-05 train_metal_acc=0.8788 val_loss=1.3182 val_metal_acc=0.6923 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5796 val_joint_macro_f1=0.5993 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.4836 lr=5.09964e-05 train_metal_acc=0.8972 val_loss=1.4040 val_metal_acc=0.6813 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5879 val_joint_macro_f1=0.5842 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.4701 lr=5.09964e-05 train_metal_acc=0.8526 val_loss=1.3241 val_metal_acc=0.6758 val_metal_min_recall=0.2308 val_fe_recall=0.5758 val_joint_bal_acc=0.6138 val_joint_macro_f1=0.6288 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.4460 lr=5.09964e-05 train_metal_acc=0.9020 val_loss=1.3756 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6013 val_joint_macro_f1=0.6167 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.3999 lr=5.09964e-05 train_metal_acc=0.9069 val_loss=1.4819 val_metal_acc=0.6923 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5918 val_joint_macro_f1=0.5726 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.3887 lr=5.09964e-05 train_metal_acc=0.9137 val_loss=1.6995 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5932 val_joint_macro_f1=0.6048 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.3904 lr=5.09964e-05 train_metal_acc=0.9117 val_loss=1.6836 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5838 val_joint_macro_f1=0.5644 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.3564 lr=5.09964e-05 train_metal_acc=0.9205 val_loss=1.6599 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6115 val_joint_macro_f1=0.6295 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.3308 lr=5.09964e-05 train_metal_acc=0.9117 val_loss=2.0607 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5642 val_joint_macro_f1=0.6037 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.3273 lr=5.09964e-05 train_metal_acc=0.9263 val_loss=2.0000 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5549 val_joint_macro_f1=0.5929 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.3168 lr=5.09964e-05 train_metal_acc=0.9302 val_loss=2.1080 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5744 val_joint_macro_f1=0.6103 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.3168 lr=5.09964e-05 train_metal_acc=0.9370 val_loss=2.1573 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5558 val_joint_macro_f1=0.5815 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.2913 lr=5.09964e-05 train_metal_acc=0.9273 val_loss=2.0641 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5757 val_joint_macro_f1=0.5852 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.2679 lr=5.09964e-05 train_metal_acc=0.9156 val_loss=2.5619 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5802 val_joint_macro_f1=0.5987 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.3183 lr=5.09964e-05 train_metal_acc=0.9418 val_loss=2.1714 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5651 val_joint_macro_f1=0.5944 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.2601 lr=5.09964e-05 train_metal_acc=0.9370 val_loss=2.4608 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6051 val_joint_macro_f1=0.6313 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.3007 lr=5.09964e-05 train_metal_acc=0.9379 val_loss=2.2118 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5726 val_joint_macro_f1=0.5831 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.3086 lr=5.09964e-05 train_metal_acc=0.9379 val_loss=2.6856 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5699 val_joint_macro_f1=0.5972 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.2552 lr=5.09964e-05 train_metal_acc=0.9205 val_loss=2.6015 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5967 val_joint_macro_f1=0.6052 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.2563 lr=5.09964e-05 train_metal_acc=0.9447 val_loss=2.4332 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6012 val_joint_macro_f1=0.6241 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.2603 lr=5.09964e-05 train_metal_acc=0.9399 val_loss=2.3056 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5680 val_joint_macro_f1=0.5457 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.2640 lr=5.09964e-05 train_metal_acc=0.9418 val_loss=2.6933 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5595 val_joint_macro_f1=0.5812 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.2787 lr=5.09964e-05 train_metal_acc=0.9457 val_loss=2.6999 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5782 val_joint_macro_f1=0.6026 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.2602 lr=5.09964e-05 train_metal_acc=0.9544 val_loss=2.5324 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5943 val_joint_macro_f1=0.6144 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.2473 lr=5.09964e-05 train_metal_acc=0.9564 val_loss=2.6317 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5870 val_joint_macro_f1=0.6093 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.2613 lr=5.09964e-05 train_metal_acc=0.9564 val_loss=2.4423 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5594 val_joint_macro_f1=0.5836 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.2042 lr=5.09964e-05 train_metal_acc=0.9515 val_loss=2.7822 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5544 val_joint_macro_f1=0.5884 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.2564 lr=5.09964e-05 train_metal_acc=0.9544 val_loss=3.0965 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5379 val_joint_macro_f1=0.5731 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.2522 lr=5.09964e-05 train_metal_acc=0.9573 val_loss=2.8275 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5779 val_joint_macro_f1=0.6100 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0012_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_bce5e2a1/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0012_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_bce5e2a1/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0012_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_bce5e2a1/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0012_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_bce5e2a1/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0012_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_bce5e2a1/run_metadata.json
[I 2026-05-14 02:29:03,962] Trial 12 finished with value: 0.6241077286618083 and parameters: {'learning_rate': 5.099642277855573e-05, 'weight_decay': 1e-05, 'hidden_s': 256, 'head_mlp_layers': 3, 'edge_hidden': 128, 'gvp_layers': 3, 'edge_radius': 6.0, 'hidden_v': 32, 'esm_fusion_dim': 256, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 8 with value: 0.6529553937523577.
Optuna trial 12 completed: val_metal_balanced_acc=0.6241077286618083
================================================================================
[Optuna trial 13] optuna_deepmzyme_controlled_hpo_trial0013_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_6cd4bd68
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 64,
  "edge_radius": 6.0,
  "esm_fusion_dim": 256,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 3,
  "head_mlp_layers": 3,
  "hidden_s": 128,
  "hidden_v": 32,
  "learning_rate": 9.985330377627194e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.001,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0013_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_6cd4bd68 --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 9.985330377627194e-05 --weight-decay 0.001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 256 --head-mlp-layers 3 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 64 --gvp-layers 3 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7701 lr=9.98533e-05 train_metal_acc=0.4588 val_loss=1.7227 val_metal_acc=0.2692 val_metal_min_recall=0.0000 val_fe_recall=0.6667 val_joint_bal_acc=0.1880 val_joint_macro_f1=0.1405 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.6300 lr=9.98533e-05 train_metal_acc=0.6334 val_loss=1.4477 val_metal_acc=0.6429 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4097 val_joint_macro_f1=0.3927 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.3580 lr=9.98533e-05 train_metal_acc=0.6770 val_loss=1.2274 val_metal_acc=0.5934 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4166 val_joint_macro_f1=0.3999 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.1895 lr=9.98533e-05 train_metal_acc=0.7352 val_loss=1.1555 val_metal_acc=0.6758 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.5389 val_joint_macro_f1=0.5558 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.0110 lr=9.98533e-05 train_metal_acc=0.7856 val_loss=1.1170 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5658 val_joint_macro_f1=0.5898 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=0.8947 lr=9.98533e-05 train_metal_acc=0.8012 val_loss=1.1263 val_metal_acc=0.6593 val_metal_min_recall=0.1538 val_fe_recall=0.4848 val_joint_bal_acc=0.5947 val_joint_macro_f1=0.5912 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=0.7334 lr=9.98533e-05 train_metal_acc=0.8361 val_loss=1.1510 val_metal_acc=0.6319 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5845 val_joint_macro_f1=0.5681 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=0.6402 lr=9.98533e-05 train_metal_acc=0.8710 val_loss=1.1705 val_metal_acc=0.6978 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.6108 val_joint_macro_f1=0.6121 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.5960 lr=9.98533e-05 train_metal_acc=0.8361 val_loss=1.1695 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5885 val_joint_macro_f1=0.5976 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.5684 lr=9.98533e-05 train_metal_acc=0.8817 val_loss=1.2738 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5801 val_joint_macro_f1=0.5670 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.4891 lr=9.98533e-05 train_metal_acc=0.8991 val_loss=1.4827 val_metal_acc=0.7527 val_metal_min_recall=0.1538 val_fe_recall=0.5455 val_joint_bal_acc=0.5959 val_joint_macro_f1=0.6303 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.4629 lr=9.98533e-05 train_metal_acc=0.8594 val_loss=1.4576 val_metal_acc=0.6429 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5906 val_joint_macro_f1=0.5803 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.4504 lr=9.98533e-05 train_metal_acc=0.9137 val_loss=1.5697 val_metal_acc=0.7582 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6184 val_joint_macro_f1=0.6298 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.4142 lr=9.98533e-05 train_metal_acc=0.9253 val_loss=1.8099 val_metal_acc=0.7582 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6150 val_joint_macro_f1=0.6560 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.4188 lr=9.98533e-05 train_metal_acc=0.9282 val_loss=1.9152 val_metal_acc=0.7527 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6056 val_joint_macro_f1=0.6384 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.3472 lr=9.98533e-05 train_metal_acc=0.9302 val_loss=1.7968 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6350 val_joint_macro_f1=0.6504 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.3236 lr=9.98533e-05 train_metal_acc=0.9146 val_loss=1.7048 val_metal_acc=0.6429 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5887 val_joint_macro_f1=0.5333 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.3312 lr=9.98533e-05 train_metal_acc=0.9146 val_loss=1.8334 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5777 val_joint_macro_f1=0.5805 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.3187 lr=9.98533e-05 train_metal_acc=0.9399 val_loss=2.2517 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5805 val_joint_macro_f1=0.6205 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.2974 lr=9.98533e-05 train_metal_acc=0.9399 val_loss=2.1097 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5595 val_joint_macro_f1=0.5711 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.3389 lr=9.98533e-05 train_metal_acc=0.9389 val_loss=2.3494 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5680 val_joint_macro_f1=0.5905 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.2686 lr=9.98533e-05 train_metal_acc=0.9340 val_loss=2.3690 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5906 val_joint_macro_f1=0.5885 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.3308 lr=9.98533e-05 train_metal_acc=0.9476 val_loss=2.4499 val_metal_acc=0.6978 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.5218 val_joint_macro_f1=0.5350 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.2560 lr=9.98533e-05 train_metal_acc=0.9496 val_loss=2.5574 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5385 val_joint_macro_f1=0.5652 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.2296 lr=9.98533e-05 train_metal_acc=0.9467 val_loss=2.2942 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5835 val_joint_macro_f1=0.5969 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.2698 lr=9.98533e-05 train_metal_acc=0.9525 val_loss=2.4084 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5871 val_joint_macro_f1=0.5980 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.2961 lr=9.98533e-05 train_metal_acc=0.9447 val_loss=2.2953 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5927 val_joint_macro_f1=0.6008 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.2177 lr=9.98533e-05 train_metal_acc=0.9554 val_loss=2.6213 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5599 val_joint_macro_f1=0.5889 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.2391 lr=9.98533e-05 train_metal_acc=0.9534 val_loss=2.4757 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5375 val_joint_macro_f1=0.5555 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.2099 lr=9.98533e-05 train_metal_acc=0.9476 val_loss=2.9556 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5694 val_joint_macro_f1=0.5847 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.2615 lr=9.98533e-05 train_metal_acc=0.9486 val_loss=2.3564 val_metal_acc=0.6319 val_metal_min_recall=0.3077 val_fe_recall=0.6364 val_joint_bal_acc=0.5666 val_joint_macro_f1=0.5750 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.2379 lr=9.98533e-05 train_metal_acc=0.9554 val_loss=2.7320 val_metal_acc=0.6429 val_metal_min_recall=0.1538 val_fe_recall=0.6667 val_joint_bal_acc=0.5479 val_joint_macro_f1=0.5648 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.2332 lr=9.98533e-05 train_metal_acc=0.9467 val_loss=2.7832 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5547 val_joint_macro_f1=0.5815 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.2462 lr=9.98533e-05 train_metal_acc=0.9534 val_loss=2.8998 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5422 val_joint_macro_f1=0.5596 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.2272 lr=9.98533e-05 train_metal_acc=0.9544 val_loss=2.5347 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5651 val_joint_macro_f1=0.5831 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.2037 lr=9.98533e-05 train_metal_acc=0.9554 val_loss=2.8124 val_metal_acc=0.6923 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5413 val_joint_macro_f1=0.5522 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.1966 lr=9.98533e-05 train_metal_acc=0.9176 val_loss=2.6951 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.7879 val_joint_bal_acc=0.5781 val_joint_macro_f1=0.5500 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.2070 lr=9.98533e-05 train_metal_acc=0.9583 val_loss=2.9048 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5321 val_joint_macro_f1=0.5436 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.2058 lr=9.98533e-05 train_metal_acc=0.9583 val_loss=2.8792 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5385 val_joint_macro_f1=0.5396 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.2031 lr=9.98533e-05 train_metal_acc=0.9583 val_loss=2.8846 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5204 val_joint_macro_f1=0.5252 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0013_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_6cd4bd68/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0013_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_6cd4bd68/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0013_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_6cd4bd68/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0013_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_6cd4bd68/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0013_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_6cd4bd68/run_metadata.json
[I 2026-05-14 02:38:04,825] Trial 13 finished with value: 0.6349550933991162 and parameters: {'learning_rate': 9.985330377627194e-05, 'weight_decay': 0.001, 'hidden_s': 128, 'head_mlp_layers': 3, 'edge_hidden': 64, 'gvp_layers': 3, 'edge_radius': 6.0, 'hidden_v': 32, 'esm_fusion_dim': 256, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 8 with value: 0.6529553937523577.
Optuna trial 13 completed: val_metal_balanced_acc=0.6349550933991162
================================================================================
[Optuna trial 14] optuna_deepmzyme_controlled_hpo_trial0014_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_064df228
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 64,
  "edge_radius": 6.0,
  "esm_fusion_dim": 256,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 3,
  "head_mlp_layers": 3,
  "hidden_s": 128,
  "hidden_v": 32,
  "learning_rate": 9.988061483429802e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.001,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0014_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_064df228 --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 9.988061483429802e-05 --weight-decay 0.001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 256 --head-mlp-layers 3 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 64 --gvp-layers 3 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7700 lr=9.98806e-05 train_metal_acc=0.4588 val_loss=1.7226 val_metal_acc=0.2692 val_metal_min_recall=0.0000 val_fe_recall=0.6667 val_joint_bal_acc=0.1880 val_joint_macro_f1=0.1405 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.6299 lr=9.98806e-05 train_metal_acc=0.6334 val_loss=1.4475 val_metal_acc=0.6429 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4097 val_joint_macro_f1=0.3927 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.3579 lr=9.98806e-05 train_metal_acc=0.6770 val_loss=1.2273 val_metal_acc=0.5934 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4166 val_joint_macro_f1=0.3999 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.1894 lr=9.98806e-05 train_metal_acc=0.7352 val_loss=1.1555 val_metal_acc=0.6758 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.5389 val_joint_macro_f1=0.5558 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.0109 lr=9.98806e-05 train_metal_acc=0.7856 val_loss=1.1170 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5658 val_joint_macro_f1=0.5898 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=0.8946 lr=9.98806e-05 train_metal_acc=0.8012 val_loss=1.1264 val_metal_acc=0.6593 val_metal_min_recall=0.1538 val_fe_recall=0.4848 val_joint_bal_acc=0.5947 val_joint_macro_f1=0.5912 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=0.7333 lr=9.98806e-05 train_metal_acc=0.8361 val_loss=1.1510 val_metal_acc=0.6319 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5845 val_joint_macro_f1=0.5681 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=0.6401 lr=9.98806e-05 train_metal_acc=0.8710 val_loss=1.1706 val_metal_acc=0.6978 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.6108 val_joint_macro_f1=0.6121 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.5960 lr=9.98806e-05 train_metal_acc=0.8361 val_loss=1.1694 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5885 val_joint_macro_f1=0.5976 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.5683 lr=9.98806e-05 train_metal_acc=0.8817 val_loss=1.2740 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5801 val_joint_macro_f1=0.5670 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.4890 lr=9.98806e-05 train_metal_acc=0.8991 val_loss=1.4828 val_metal_acc=0.7527 val_metal_min_recall=0.1538 val_fe_recall=0.5455 val_joint_bal_acc=0.5959 val_joint_macro_f1=0.6303 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.4628 lr=9.98806e-05 train_metal_acc=0.8594 val_loss=1.4579 val_metal_acc=0.6429 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5906 val_joint_macro_f1=0.5803 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.4503 lr=9.98806e-05 train_metal_acc=0.9137 val_loss=1.5701 val_metal_acc=0.7582 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6184 val_joint_macro_f1=0.6298 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.4142 lr=9.98806e-05 train_metal_acc=0.9253 val_loss=1.8102 val_metal_acc=0.7582 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6150 val_joint_macro_f1=0.6560 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.4188 lr=9.98806e-05 train_metal_acc=0.9282 val_loss=1.9156 val_metal_acc=0.7527 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6056 val_joint_macro_f1=0.6384 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.3472 lr=9.98806e-05 train_metal_acc=0.9302 val_loss=1.7971 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6350 val_joint_macro_f1=0.6504 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.3236 lr=9.98806e-05 train_metal_acc=0.9146 val_loss=1.7049 val_metal_acc=0.6429 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5887 val_joint_macro_f1=0.5333 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.3312 lr=9.98806e-05 train_metal_acc=0.9146 val_loss=1.8337 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5777 val_joint_macro_f1=0.5805 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.3187 lr=9.98806e-05 train_metal_acc=0.9399 val_loss=2.2517 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5805 val_joint_macro_f1=0.6205 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.2973 lr=9.98806e-05 train_metal_acc=0.9399 val_loss=2.1104 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5595 val_joint_macro_f1=0.5711 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.3389 lr=9.98806e-05 train_metal_acc=0.9389 val_loss=2.3489 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5680 val_joint_macro_f1=0.5905 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.2686 lr=9.98806e-05 train_metal_acc=0.9340 val_loss=2.3685 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5906 val_joint_macro_f1=0.5885 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.3308 lr=9.98806e-05 train_metal_acc=0.9476 val_loss=2.4498 val_metal_acc=0.6978 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.5218 val_joint_macro_f1=0.5350 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.2557 lr=9.98806e-05 train_metal_acc=0.9505 val_loss=2.5569 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5385 val_joint_macro_f1=0.5652 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.2296 lr=9.98806e-05 train_metal_acc=0.9467 val_loss=2.2936 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5835 val_joint_macro_f1=0.5969 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.2696 lr=9.98806e-05 train_metal_acc=0.9525 val_loss=2.4077 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5851 val_joint_macro_f1=0.5955 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.2955 lr=9.98806e-05 train_metal_acc=0.9447 val_loss=2.2973 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5927 val_joint_macro_f1=0.6008 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.2178 lr=9.98806e-05 train_metal_acc=0.9554 val_loss=2.6218 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5653 val_joint_macro_f1=0.5951 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.2391 lr=9.98806e-05 train_metal_acc=0.9544 val_loss=2.4776 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5375 val_joint_macro_f1=0.5555 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.2102 lr=9.98806e-05 train_metal_acc=0.9467 val_loss=2.9455 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5565 val_joint_macro_f1=0.5756 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.2601 lr=9.98806e-05 train_metal_acc=0.9476 val_loss=2.3489 val_metal_acc=0.6319 val_metal_min_recall=0.3077 val_fe_recall=0.6364 val_joint_bal_acc=0.5666 val_joint_macro_f1=0.5750 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.2383 lr=9.98806e-05 train_metal_acc=0.9534 val_loss=2.7212 val_metal_acc=0.6429 val_metal_min_recall=0.1538 val_fe_recall=0.6667 val_joint_bal_acc=0.5479 val_joint_macro_f1=0.5648 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.2306 lr=9.98806e-05 train_metal_acc=0.9467 val_loss=2.7653 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5544 val_joint_macro_f1=0.5817 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.2416 lr=9.98806e-05 train_metal_acc=0.9544 val_loss=2.9166 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5462 val_joint_macro_f1=0.5635 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.2160 lr=9.98806e-05 train_metal_acc=0.9554 val_loss=2.5100 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5799 val_joint_macro_f1=0.5952 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.2079 lr=9.98806e-05 train_metal_acc=0.9564 val_loss=2.8195 val_metal_acc=0.6923 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5413 val_joint_macro_f1=0.5523 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.2007 lr=9.98806e-05 train_metal_acc=0.9360 val_loss=2.6478 val_metal_acc=0.6264 val_metal_min_recall=0.0769 val_fe_recall=0.6970 val_joint_bal_acc=0.5497 val_joint_macro_f1=0.5107 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.2171 lr=9.98806e-05 train_metal_acc=0.9564 val_loss=2.9434 val_metal_acc=0.6813 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5122 val_joint_macro_f1=0.5293 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.1931 lr=9.98806e-05 train_metal_acc=0.9505 val_loss=2.8048 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5504 val_joint_macro_f1=0.5565 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.2046 lr=9.98806e-05 train_metal_acc=0.9583 val_loss=3.0630 val_metal_acc=0.6319 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5253 val_joint_macro_f1=0.5506 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0014_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_064df228/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0014_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_064df228/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0014_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_064df228/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0014_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_064df228/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0014_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_064df228/run_metadata.json
[I 2026-05-14 02:47:01,110] Trial 14 finished with value: 0.6349550933991162 and parameters: {'learning_rate': 9.988061483429802e-05, 'weight_decay': 0.001, 'hidden_s': 128, 'head_mlp_layers': 3, 'edge_hidden': 64, 'gvp_layers': 3, 'edge_radius': 6.0, 'hidden_v': 32, 'esm_fusion_dim': 256, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 8 with value: 0.6529553937523577.
Optuna trial 14 completed: val_metal_balanced_acc=0.6349550933991162
================================================================================
[Optuna trial 15] optuna_deepmzyme_controlled_hpo_trial0015_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a931e009
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 6.0,
  "esm_fusion_dim": 64,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 2,
  "head_mlp_layers": 1,
  "hidden_s": 128,
  "hidden_v": 32,
  "learning_rate": 7.032630334240692e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.001,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0015_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a931e009 --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 7.032630334240692e-05 --weight-decay 0.001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7554 lr=7.03263e-05 train_metal_acc=0.5344 val_loss=1.5883 val_metal_acc=0.5714 val_metal_min_recall=0.0000 val_fe_recall=0.3939 val_joint_bal_acc=0.3744 val_joint_macro_f1=0.3570 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.5272 lr=7.03263e-05 train_metal_acc=0.6372 val_loss=1.3762 val_metal_acc=0.6593 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4719 val_joint_macro_f1=0.4936 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.3227 lr=7.03263e-05 train_metal_acc=0.6838 val_loss=1.2755 val_metal_acc=0.6429 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.5128 val_joint_macro_f1=0.5208 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.1867 lr=7.03263e-05 train_metal_acc=0.6431 val_loss=1.2874 val_metal_acc=0.4176 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4707 val_joint_macro_f1=0.4468 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.0614 lr=7.03263e-05 train_metal_acc=0.7604 val_loss=1.1465 val_metal_acc=0.6813 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4733 val_joint_macro_f1=0.4841 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=0.9750 lr=7.03263e-05 train_metal_acc=0.7595 val_loss=1.2290 val_metal_acc=0.4560 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5252 val_joint_macro_f1=0.5105 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=0.8790 lr=7.03263e-05 train_metal_acc=0.7779 val_loss=1.1836 val_metal_acc=0.5000 val_metal_min_recall=0.1538 val_fe_recall=0.4848 val_joint_bal_acc=0.5596 val_joint_macro_f1=0.5387 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=0.8027 lr=7.03263e-05 train_metal_acc=0.8109 val_loss=1.1444 val_metal_acc=0.5824 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.5662 val_joint_macro_f1=0.5868 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.7598 lr=7.03263e-05 train_metal_acc=0.8012 val_loss=1.1356 val_metal_acc=0.5934 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5742 val_joint_macro_f1=0.5830 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.6987 lr=7.03263e-05 train_metal_acc=0.8409 val_loss=1.0607 val_metal_acc=0.6538 val_metal_min_recall=0.1538 val_fe_recall=0.4848 val_joint_bal_acc=0.5790 val_joint_macro_f1=0.5836 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.6285 lr=7.03263e-05 train_metal_acc=0.8468 val_loss=1.0715 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5938 val_joint_macro_f1=0.6143 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.6035 lr=7.03263e-05 train_metal_acc=0.8497 val_loss=1.0194 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.6047 val_joint_macro_f1=0.6167 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.5608 lr=7.03263e-05 train_metal_acc=0.8691 val_loss=1.0275 val_metal_acc=0.6868 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5973 val_joint_macro_f1=0.5886 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.5025 lr=7.03263e-05 train_metal_acc=0.8739 val_loss=1.0707 val_metal_acc=0.6868 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5964 val_joint_macro_f1=0.6076 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.4636 lr=7.03263e-05 train_metal_acc=0.8817 val_loss=1.0132 val_metal_acc=0.6868 val_metal_min_recall=0.2308 val_fe_recall=0.6364 val_joint_bal_acc=0.6239 val_joint_macro_f1=0.6408 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.4493 lr=7.03263e-05 train_metal_acc=0.8817 val_loss=1.1996 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5866 val_joint_macro_f1=0.5965 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.4343 lr=7.03263e-05 train_metal_acc=0.8855 val_loss=1.0920 val_metal_acc=0.6978 val_metal_min_recall=0.1538 val_fe_recall=0.6667 val_joint_bal_acc=0.6235 val_joint_macro_f1=0.6583 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.4142 lr=7.03263e-05 train_metal_acc=0.9011 val_loss=1.0569 val_metal_acc=0.7418 val_metal_min_recall=0.2308 val_fe_recall=0.6667 val_joint_bal_acc=0.6398 val_joint_macro_f1=0.6774 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.3730 lr=7.03263e-05 train_metal_acc=0.9069 val_loss=1.0573 val_metal_acc=0.6923 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6116 val_joint_macro_f1=0.6169 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.3501 lr=7.03263e-05 train_metal_acc=0.9059 val_loss=1.1369 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.6226 val_joint_macro_f1=0.6465 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.3472 lr=7.03263e-05 train_metal_acc=0.9127 val_loss=1.1673 val_metal_acc=0.7637 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.6191 val_joint_macro_f1=0.6417 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.3141 lr=7.03263e-05 train_metal_acc=0.9030 val_loss=1.0597 val_metal_acc=0.7637 val_metal_min_recall=0.3077 val_fe_recall=0.6667 val_joint_bal_acc=0.6551 val_joint_macro_f1=0.6699 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.2934 lr=7.03263e-05 train_metal_acc=0.9117 val_loss=1.2460 val_metal_acc=0.7363 val_metal_min_recall=0.2308 val_fe_recall=0.6970 val_joint_bal_acc=0.6123 val_joint_macro_f1=0.6570 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.2984 lr=7.03263e-05 train_metal_acc=0.9224 val_loss=1.1750 val_metal_acc=0.7253 val_metal_min_recall=0.3077 val_fe_recall=0.6364 val_joint_bal_acc=0.6485 val_joint_macro_f1=0.6756 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.2762 lr=7.03263e-05 train_metal_acc=0.9292 val_loss=1.3283 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5897 val_joint_macro_f1=0.6256 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.2569 lr=7.03263e-05 train_metal_acc=0.9311 val_loss=1.3817 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6077 val_joint_macro_f1=0.6411 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.2564 lr=7.03263e-05 train_metal_acc=0.9321 val_loss=1.4422 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6083 val_joint_macro_f1=0.6529 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.2318 lr=7.03263e-05 train_metal_acc=0.9205 val_loss=1.3457 val_metal_acc=0.7308 val_metal_min_recall=0.2308 val_fe_recall=0.6061 val_joint_bal_acc=0.6126 val_joint_macro_f1=0.6389 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.2384 lr=7.03263e-05 train_metal_acc=0.9379 val_loss=1.4595 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.5980 val_joint_macro_f1=0.6332 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.2244 lr=7.03263e-05 train_metal_acc=0.9253 val_loss=1.3300 val_metal_acc=0.7088 val_metal_min_recall=0.3077 val_fe_recall=0.6061 val_joint_bal_acc=0.6361 val_joint_macro_f1=0.6528 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.2505 lr=7.03263e-05 train_metal_acc=0.9437 val_loss=1.4322 val_metal_acc=0.7527 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6264 val_joint_macro_f1=0.6673 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.1954 lr=7.03263e-05 train_metal_acc=0.9379 val_loss=1.5084 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6068 val_joint_macro_f1=0.6515 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.2078 lr=7.03263e-05 train_metal_acc=0.9331 val_loss=1.4845 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6120 val_joint_macro_f1=0.6372 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.2165 lr=7.03263e-05 train_metal_acc=0.9428 val_loss=1.6422 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.5946 val_joint_macro_f1=0.6399 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.2207 lr=7.03263e-05 train_metal_acc=0.9467 val_loss=1.6765 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6042 val_joint_macro_f1=0.6441 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.2288 lr=7.03263e-05 train_metal_acc=0.9437 val_loss=1.6450 val_metal_acc=0.7198 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5692 val_joint_macro_f1=0.6175 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.2099 lr=7.03263e-05 train_metal_acc=0.9486 val_loss=1.6971 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5949 val_joint_macro_f1=0.6342 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.2005 lr=7.03263e-05 train_metal_acc=0.9292 val_loss=1.6207 val_metal_acc=0.7088 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.5868 val_joint_macro_f1=0.6018 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.1802 lr=7.03263e-05 train_metal_acc=0.9447 val_loss=1.8433 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6003 val_joint_macro_f1=0.6403 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.1775 lr=7.03263e-05 train_metal_acc=0.9496 val_loss=1.7892 val_metal_acc=0.7418 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6122 val_joint_macro_f1=0.6565 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0015_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a931e009/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0015_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a931e009/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0015_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a931e009/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0015_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a931e009/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0015_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a931e009/run_metadata.json
[I 2026-05-14 02:55:09,668] Trial 15 finished with value: 0.6550963478857217 and parameters: {'learning_rate': 7.032630334240692e-05, 'weight_decay': 0.001, 'hidden_s': 128, 'head_mlp_layers': 1, 'edge_hidden': 128, 'gvp_layers': 2, 'edge_radius': 6.0, 'hidden_v': 32, 'esm_fusion_dim': 64, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 15 with value: 0.6550963478857217.
Optuna trial 15 completed: val_metal_balanced_acc=0.6550963478857217
================================================================================
[Optuna trial 16] optuna_deepmzyme_controlled_hpo_trial0016_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a973315e
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 6.0,
  "esm_fusion_dim": 64,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 2,
  "head_mlp_layers": 1,
  "hidden_s": 128,
  "hidden_v": 32,
  "learning_rate": 6.292363726994639e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.001,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0016_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a973315e --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 6.292363726994639e-05 --weight-decay 0.001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7615 lr=6.29236e-05 train_metal_acc=0.5228 val_loss=1.6116 val_metal_acc=0.5659 val_metal_min_recall=0.0000 val_fe_recall=0.3939 val_joint_bal_acc=0.3724 val_joint_macro_f1=0.3562 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.5549 lr=6.29236e-05 train_metal_acc=0.6285 val_loss=1.4020 val_metal_acc=0.6538 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4631 val_joint_macro_f1=0.4881 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.3592 lr=6.29236e-05 train_metal_acc=0.6702 val_loss=1.2984 val_metal_acc=0.6429 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4910 val_joint_macro_f1=0.4954 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.2260 lr=6.29236e-05 train_metal_acc=0.6314 val_loss=1.2901 val_metal_acc=0.4176 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4703 val_joint_macro_f1=0.4429 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.1037 lr=6.29236e-05 train_metal_acc=0.7468 val_loss=1.1593 val_metal_acc=0.6758 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4676 val_joint_macro_f1=0.4774 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.0183 lr=6.29236e-05 train_metal_acc=0.7439 val_loss=1.2371 val_metal_acc=0.4505 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5198 val_joint_macro_f1=0.5065 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=0.9236 lr=6.29236e-05 train_metal_acc=0.7682 val_loss=1.1897 val_metal_acc=0.4780 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5341 val_joint_macro_f1=0.5032 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=0.8473 lr=6.29236e-05 train_metal_acc=0.7973 val_loss=1.1513 val_metal_acc=0.5275 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5404 val_joint_macro_f1=0.5489 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.8025 lr=6.29236e-05 train_metal_acc=0.7886 val_loss=1.1510 val_metal_acc=0.5275 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5541 val_joint_macro_f1=0.5577 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.7428 lr=6.29236e-05 train_metal_acc=0.8303 val_loss=1.0650 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5589 val_joint_macro_f1=0.5493 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.6738 lr=6.29236e-05 train_metal_acc=0.8351 val_loss=1.0630 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5938 val_joint_macro_f1=0.6143 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.6438 lr=6.29236e-05 train_metal_acc=0.8438 val_loss=1.0155 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5865 val_joint_macro_f1=0.5877 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.6021 lr=6.29236e-05 train_metal_acc=0.8594 val_loss=1.0158 val_metal_acc=0.6758 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5934 val_joint_macro_f1=0.5839 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.5419 lr=6.29236e-05 train_metal_acc=0.8565 val_loss=1.0567 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5866 val_joint_macro_f1=0.5965 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.5031 lr=6.29236e-05 train_metal_acc=0.8720 val_loss=0.9909 val_metal_acc=0.6758 val_metal_min_recall=0.2308 val_fe_recall=0.6061 val_joint_bal_acc=0.6135 val_joint_macro_f1=0.6294 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.4845 lr=6.29236e-05 train_metal_acc=0.8681 val_loss=1.1684 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5866 val_joint_macro_f1=0.5964 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.4681 lr=6.29236e-05 train_metal_acc=0.8691 val_loss=1.0787 val_metal_acc=0.6923 val_metal_min_recall=0.1538 val_fe_recall=0.6667 val_joint_bal_acc=0.6215 val_joint_macro_f1=0.6540 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.4469 lr=6.29236e-05 train_metal_acc=0.8972 val_loss=1.0289 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6188 val_joint_macro_f1=0.6507 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.4040 lr=6.29236e-05 train_metal_acc=0.9020 val_loss=1.0338 val_metal_acc=0.6703 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.5942 val_joint_macro_f1=0.6030 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.3798 lr=6.29236e-05 train_metal_acc=0.8991 val_loss=1.1016 val_metal_acc=0.7088 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.6147 val_joint_macro_f1=0.6374 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.3756 lr=6.29236e-05 train_metal_acc=0.9098 val_loss=1.1164 val_metal_acc=0.7802 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.6352 val_joint_macro_f1=0.6507 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.3434 lr=6.29236e-05 train_metal_acc=0.8991 val_loss=1.0307 val_metal_acc=0.7582 val_metal_min_recall=0.3077 val_fe_recall=0.6667 val_joint_bal_acc=0.6531 val_joint_macro_f1=0.6673 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.3208 lr=6.29236e-05 train_metal_acc=0.9069 val_loss=1.1808 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6970 val_joint_bal_acc=0.6247 val_joint_macro_f1=0.6648 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.3202 lr=6.29236e-05 train_metal_acc=0.9176 val_loss=1.1392 val_metal_acc=0.6978 val_metal_min_recall=0.3077 val_fe_recall=0.6061 val_joint_bal_acc=0.6356 val_joint_macro_f1=0.6612 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.3001 lr=6.29236e-05 train_metal_acc=0.9205 val_loss=1.2527 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5839 val_joint_macro_f1=0.6105 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.2777 lr=6.29236e-05 train_metal_acc=0.9243 val_loss=1.2858 val_metal_acc=0.7527 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6127 val_joint_macro_f1=0.6455 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.2731 lr=6.29236e-05 train_metal_acc=0.9263 val_loss=1.3421 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6137 val_joint_macro_f1=0.6571 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.2497 lr=6.29236e-05 train_metal_acc=0.9205 val_loss=1.2905 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6018 val_joint_macro_f1=0.6276 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.2539 lr=6.29236e-05 train_metal_acc=0.9350 val_loss=1.3749 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6000 val_joint_macro_f1=0.6361 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.2391 lr=6.29236e-05 train_metal_acc=0.9243 val_loss=1.2656 val_metal_acc=0.7033 val_metal_min_recall=0.3077 val_fe_recall=0.6061 val_joint_bal_acc=0.6342 val_joint_macro_f1=0.6499 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.2609 lr=6.29236e-05 train_metal_acc=0.9418 val_loss=1.3378 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6171 val_joint_macro_f1=0.6503 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.2096 lr=6.29236e-05 train_metal_acc=0.9350 val_loss=1.4274 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6176 val_joint_macro_f1=0.6608 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.2195 lr=6.29236e-05 train_metal_acc=0.9292 val_loss=1.3850 val_metal_acc=0.7527 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6301 val_joint_macro_f1=0.6522 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.2267 lr=6.29236e-05 train_metal_acc=0.9437 val_loss=1.5211 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.5762 val_joint_macro_f1=0.6230 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.2259 lr=6.29236e-05 train_metal_acc=0.9399 val_loss=1.5346 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6077 val_joint_macro_f1=0.6474 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.2337 lr=6.29236e-05 train_metal_acc=0.9389 val_loss=1.5299 val_metal_acc=0.7253 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5711 val_joint_macro_f1=0.6139 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.2211 lr=6.29236e-05 train_metal_acc=0.9457 val_loss=1.5735 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5969 val_joint_macro_f1=0.6367 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.2078 lr=6.29236e-05 train_metal_acc=0.9282 val_loss=1.5294 val_metal_acc=0.7198 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5978 val_joint_macro_f1=0.6140 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.1878 lr=6.29236e-05 train_metal_acc=0.9408 val_loss=1.7295 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5710 val_joint_macro_f1=0.6075 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.1840 lr=6.29236e-05 train_metal_acc=0.9467 val_loss=1.6831 val_metal_acc=0.7527 val_metal_min_recall=0.1538 val_fe_recall=0.6667 val_joint_bal_acc=0.6226 val_joint_macro_f1=0.6745 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0016_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a973315e/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0016_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a973315e/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0016_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a973315e/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0016_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a973315e/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0016_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a973315e/run_metadata.json
[I 2026-05-14 03:03:55,801] Trial 16 finished with value: 0.6531355635719962 and parameters: {'learning_rate': 6.292363726994639e-05, 'weight_decay': 0.001, 'hidden_s': 128, 'head_mlp_layers': 1, 'edge_hidden': 128, 'gvp_layers': 2, 'edge_radius': 6.0, 'hidden_v': 32, 'esm_fusion_dim': 64, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 15 with value: 0.6550963478857217.
Optuna trial 16 completed: val_metal_balanced_acc=0.6531355635719962
================================================================================
[Optuna trial 17] optuna_deepmzyme_controlled_hpo_trial0017_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_0e910c6b
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 6.0,
  "esm_fusion_dim": 64,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 2,
  "head_mlp_layers": 1,
  "hidden_s": 128,
  "hidden_v": 32,
  "learning_rate": 4.647699935657328e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.0001,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0017_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_0e910c6b --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 4.647699935657328e-05 --weight-decay 0.0001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7736 lr=4.6477e-05 train_metal_acc=0.4859 val_loss=1.6588 val_metal_acc=0.5659 val_metal_min_recall=0.0000 val_fe_recall=0.3939 val_joint_bal_acc=0.3833 val_joint_macro_f1=0.3694 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.6246 lr=4.6477e-05 train_metal_acc=0.5723 val_loss=1.4771 val_metal_acc=0.6319 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4385 val_joint_macro_f1=0.4610 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.4557 lr=4.6477e-05 train_metal_acc=0.6314 val_loss=1.3732 val_metal_acc=0.6484 val_metal_min_recall=0.0000 val_fe_recall=0.6970 val_joint_bal_acc=0.4773 val_joint_macro_f1=0.4760 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.3296 lr=4.6477e-05 train_metal_acc=0.6208 val_loss=1.3184 val_metal_acc=0.4066 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4343 val_joint_macro_f1=0.4049 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.2172 lr=4.6477e-05 train_metal_acc=0.7090 val_loss=1.2018 val_metal_acc=0.6813 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4695 val_joint_macro_f1=0.4753 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.1371 lr=4.6477e-05 train_metal_acc=0.7051 val_loss=1.2484 val_metal_acc=0.4396 val_metal_min_recall=0.1538 val_fe_recall=0.5455 val_joint_bal_acc=0.5053 val_joint_macro_f1=0.4893 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=1.0483 lr=4.6477e-05 train_metal_acc=0.7294 val_loss=1.2080 val_metal_acc=0.4615 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5158 val_joint_macro_f1=0.4632 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=0.9737 lr=4.6477e-05 train_metal_acc=0.7517 val_loss=1.1742 val_metal_acc=0.4451 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.5001 val_joint_macro_f1=0.5072 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.9266 lr=4.6477e-05 train_metal_acc=0.7624 val_loss=1.1902 val_metal_acc=0.4560 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5178 val_joint_macro_f1=0.4978 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.8715 lr=4.6477e-05 train_metal_acc=0.7924 val_loss=1.1200 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5841 val_joint_macro_f1=0.5696 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.8104 lr=4.6477e-05 train_metal_acc=0.8060 val_loss=1.0669 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5899 val_joint_macro_f1=0.6081 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.7660 lr=4.6477e-05 train_metal_acc=0.8118 val_loss=1.0352 val_metal_acc=0.6923 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5785 val_joint_macro_f1=0.5776 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.7269 lr=4.6477e-05 train_metal_acc=0.8244 val_loss=1.0297 val_metal_acc=0.6429 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5677 val_joint_macro_f1=0.5539 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.6657 lr=4.6477e-05 train_metal_acc=0.8235 val_loss=1.0564 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5849 val_joint_macro_f1=0.5903 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.6268 lr=4.6477e-05 train_metal_acc=0.8380 val_loss=0.9782 val_metal_acc=0.6593 val_metal_min_recall=0.1538 val_fe_recall=0.5455 val_joint_bal_acc=0.5906 val_joint_macro_f1=0.5972 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.5977 lr=4.6477e-05 train_metal_acc=0.8283 val_loss=1.1234 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5818 val_joint_macro_f1=0.5794 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.5772 lr=4.6477e-05 train_metal_acc=0.8371 val_loss=1.0653 val_metal_acc=0.6758 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6095 val_joint_macro_f1=0.6231 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.5548 lr=4.6477e-05 train_metal_acc=0.8710 val_loss=0.9741 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6001 val_joint_macro_f1=0.6181 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.5064 lr=4.6477e-05 train_metal_acc=0.8826 val_loss=0.9924 val_metal_acc=0.6648 val_metal_min_recall=0.1538 val_fe_recall=0.5455 val_joint_bal_acc=0.5891 val_joint_macro_f1=0.5985 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.4796 lr=4.6477e-05 train_metal_acc=0.8720 val_loss=1.0380 val_metal_acc=0.6703 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.6010 val_joint_macro_f1=0.6196 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.4722 lr=4.6477e-05 train_metal_acc=0.8904 val_loss=1.0330 val_metal_acc=0.7637 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.6293 val_joint_macro_f1=0.6493 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.4378 lr=4.6477e-05 train_metal_acc=0.8817 val_loss=0.9718 val_metal_acc=0.7582 val_metal_min_recall=0.2308 val_fe_recall=0.6667 val_joint_bal_acc=0.6457 val_joint_macro_f1=0.6717 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.4173 lr=4.6477e-05 train_metal_acc=0.8865 val_loss=1.0704 val_metal_acc=0.6868 val_metal_min_recall=0.1538 val_fe_recall=0.6970 val_joint_bal_acc=0.6158 val_joint_macro_f1=0.6503 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.4017 lr=4.6477e-05 train_metal_acc=0.8914 val_loss=1.0746 val_metal_acc=0.6813 val_metal_min_recall=0.3077 val_fe_recall=0.5758 val_joint_bal_acc=0.6267 val_joint_macro_f1=0.6537 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.3884 lr=4.6477e-05 train_metal_acc=0.8991 val_loss=1.1234 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5714 val_joint_macro_f1=0.5872 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.3596 lr=4.6477e-05 train_metal_acc=0.9088 val_loss=1.1034 val_metal_acc=0.7527 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6158 val_joint_macro_f1=0.6248 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.3426 lr=4.6477e-05 train_metal_acc=0.9146 val_loss=1.1389 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6119 val_joint_macro_f1=0.6419 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.3244 lr=4.6477e-05 train_metal_acc=0.9127 val_loss=1.1460 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5927 val_joint_macro_f1=0.6190 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.3196 lr=4.6477e-05 train_metal_acc=0.9185 val_loss=1.1535 val_metal_acc=0.7637 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6235 val_joint_macro_f1=0.6592 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.3070 lr=4.6477e-05 train_metal_acc=0.9079 val_loss=1.1406 val_metal_acc=0.6813 val_metal_min_recall=0.3077 val_fe_recall=0.6364 val_joint_bal_acc=0.6294 val_joint_macro_f1=0.6524 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.3126 lr=4.6477e-05 train_metal_acc=0.9195 val_loss=1.1577 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6254 val_joint_macro_f1=0.6521 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.2675 lr=4.6477e-05 train_metal_acc=0.9224 val_loss=1.2476 val_metal_acc=0.7582 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6209 val_joint_macro_f1=0.6531 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.2771 lr=4.6477e-05 train_metal_acc=0.9185 val_loss=1.1926 val_metal_acc=0.7143 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6198 val_joint_macro_f1=0.6436 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.2714 lr=4.6477e-05 train_metal_acc=0.9263 val_loss=1.2321 val_metal_acc=0.7527 val_metal_min_recall=0.1538 val_fe_recall=0.6667 val_joint_bal_acc=0.6158 val_joint_macro_f1=0.6589 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.2651 lr=4.6477e-05 train_metal_acc=0.9292 val_loss=1.2835 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6244 val_joint_macro_f1=0.6639 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.2656 lr=4.6477e-05 train_metal_acc=0.9321 val_loss=1.2808 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5757 val_joint_macro_f1=0.6171 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.2665 lr=4.6477e-05 train_metal_acc=0.9360 val_loss=1.2529 val_metal_acc=0.7418 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6091 val_joint_macro_f1=0.6381 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.2421 lr=4.6477e-05 train_metal_acc=0.9137 val_loss=1.2773 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.6020 val_joint_macro_f1=0.6092 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.2296 lr=4.6477e-05 train_metal_acc=0.9370 val_loss=1.4183 val_metal_acc=0.7582 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6106 val_joint_macro_f1=0.6473 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.2126 lr=4.6477e-05 train_metal_acc=0.9350 val_loss=1.4764 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5855 val_joint_macro_f1=0.6285 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0017_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_0e910c6b/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0017_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_0e910c6b/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0017_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_0e910c6b/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0017_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_0e910c6b/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0017_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_0e910c6b/run_metadata.json
[I 2026-05-14 03:12:34,986] Trial 17 finished with value: 0.6456913948375049 and parameters: {'learning_rate': 4.647699935657328e-05, 'weight_decay': 0.0001, 'hidden_s': 128, 'head_mlp_layers': 1, 'edge_hidden': 128, 'gvp_layers': 2, 'edge_radius': 6.0, 'hidden_v': 32, 'esm_fusion_dim': 64, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 15 with value: 0.6550963478857217.
Optuna trial 17 completed: val_metal_balanced_acc=0.6456913948375049
================================================================================
[Optuna trial 18] optuna_deepmzyme_controlled_hpo_trial0018_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_98f0fd36
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 6.0,
  "esm_fusion_dim": 64,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 2,
  "head_mlp_layers": 1,
  "hidden_s": 128,
  "hidden_v": 32,
  "learning_rate": 6.377867731540385e-05,
  "metal_class_weight_mode": "inverse_sqrt_frequency",
  "metal_weighting_setup": "inverse_sqrt_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.001,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0018_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_98f0fd36 --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 6.377867731540385e-05 --weight-decay 0.001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_sqrt_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.6470 lr=6.37787e-05 train_metal_acc=0.5228 val_loss=1.4963 val_metal_acc=0.5275 val_metal_min_recall=0.0000 val_fe_recall=0.3939 val_joint_bal_acc=0.2284 val_joint_macro_f1=0.1938 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.4312 lr=6.37787e-05 train_metal_acc=0.6343 val_loss=1.3282 val_metal_acc=0.6593 val_metal_min_recall=0.0000 val_fe_recall=0.6970 val_joint_bal_acc=0.4320 val_joint_macro_f1=0.4226 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.2408 lr=6.37787e-05 train_metal_acc=0.6906 val_loss=1.2288 val_metal_acc=0.6758 val_metal_min_recall=0.0000 val_fe_recall=0.6667 val_joint_bal_acc=0.4382 val_joint_macro_f1=0.4344 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.1141 lr=6.37787e-05 train_metal_acc=0.7119 val_loss=1.2344 val_metal_acc=0.4396 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4212 val_joint_macro_f1=0.4083 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.0055 lr=6.37787e-05 train_metal_acc=0.7236 val_loss=1.1107 val_metal_acc=0.6703 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4335 val_joint_macro_f1=0.4275 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=0.9293 lr=6.37787e-05 train_metal_acc=0.7992 val_loss=1.1170 val_metal_acc=0.6538 val_metal_min_recall=0.0000 val_fe_recall=0.5152 val_joint_bal_acc=0.5099 val_joint_macro_f1=0.5242 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=0.8500 lr=6.37787e-05 train_metal_acc=0.8186 val_loss=1.0570 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5423 val_joint_macro_f1=0.5654 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=0.7819 lr=6.37787e-05 train_metal_acc=0.8128 val_loss=1.0189 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5547 val_joint_macro_f1=0.5907 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.7341 lr=6.37787e-05 train_metal_acc=0.8322 val_loss=1.0003 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5393 val_joint_macro_f1=0.5464 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.6735 lr=6.37787e-05 train_metal_acc=0.8545 val_loss=0.9793 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5754 val_joint_macro_f1=0.6004 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.6272 lr=6.37787e-05 train_metal_acc=0.8535 val_loss=0.9980 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5521 val_joint_macro_f1=0.5863 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.5957 lr=6.37787e-05 train_metal_acc=0.8613 val_loss=0.9642 val_metal_acc=0.7582 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.6079 val_joint_macro_f1=0.6384 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.5579 lr=6.37787e-05 train_metal_acc=0.8778 val_loss=0.9389 val_metal_acc=0.7637 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.6309 val_joint_macro_f1=0.6579 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.5057 lr=6.37787e-05 train_metal_acc=0.8729 val_loss=0.9510 val_metal_acc=0.7582 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6028 val_joint_macro_f1=0.6308 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.4734 lr=6.37787e-05 train_metal_acc=0.8894 val_loss=0.9064 val_metal_acc=0.7802 val_metal_min_recall=0.1538 val_fe_recall=0.6970 val_joint_bal_acc=0.6458 val_joint_macro_f1=0.6848 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.4573 lr=6.37787e-05 train_metal_acc=0.8807 val_loss=1.0267 val_metal_acc=0.6813 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5535 val_joint_macro_f1=0.5824 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.4379 lr=6.37787e-05 train_metal_acc=0.9001 val_loss=0.9566 val_metal_acc=0.7253 val_metal_min_recall=0.1538 val_fe_recall=0.6667 val_joint_bal_acc=0.6265 val_joint_macro_f1=0.6671 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.4176 lr=6.37787e-05 train_metal_acc=0.9098 val_loss=0.9301 val_metal_acc=0.7802 val_metal_min_recall=0.1538 val_fe_recall=0.7273 val_joint_bal_acc=0.6270 val_joint_macro_f1=0.6730 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.3844 lr=6.37787e-05 train_metal_acc=0.9127 val_loss=0.9469 val_metal_acc=0.7747 val_metal_min_recall=0.1538 val_fe_recall=0.6667 val_joint_bal_acc=0.6407 val_joint_macro_f1=0.6738 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.3591 lr=6.37787e-05 train_metal_acc=0.9127 val_loss=1.0072 val_metal_acc=0.7637 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6340 val_joint_macro_f1=0.6808 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.3436 lr=6.37787e-05 train_metal_acc=0.9166 val_loss=1.0234 val_metal_acc=0.7802 val_metal_min_recall=0.0769 val_fe_recall=0.6970 val_joint_bal_acc=0.6315 val_joint_macro_f1=0.6758 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.3277 lr=6.37787e-05 train_metal_acc=0.9146 val_loss=0.9549 val_metal_acc=0.7747 val_metal_min_recall=0.2308 val_fe_recall=0.6970 val_joint_bal_acc=0.6512 val_joint_macro_f1=0.6801 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.3103 lr=6.37787e-05 train_metal_acc=0.9137 val_loss=1.0656 val_metal_acc=0.7527 val_metal_min_recall=0.1538 val_fe_recall=0.6970 val_joint_bal_acc=0.6039 val_joint_macro_f1=0.6475 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.3052 lr=6.37787e-05 train_metal_acc=0.9253 val_loss=1.0191 val_metal_acc=0.7637 val_metal_min_recall=0.2308 val_fe_recall=0.6364 val_joint_bal_acc=0.6514 val_joint_macro_f1=0.6788 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.2821 lr=6.37787e-05 train_metal_acc=0.9340 val_loss=1.1241 val_metal_acc=0.7582 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5956 val_joint_macro_f1=0.6335 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.2738 lr=6.37787e-05 train_metal_acc=0.9331 val_loss=1.1698 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5780 val_joint_macro_f1=0.6090 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.2615 lr=6.37787e-05 train_metal_acc=0.9321 val_loss=1.2306 val_metal_acc=0.7527 val_metal_min_recall=0.1538 val_fe_recall=0.6667 val_joint_bal_acc=0.5974 val_joint_macro_f1=0.6548 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.2414 lr=6.37787e-05 train_metal_acc=0.9331 val_loss=1.1756 val_metal_acc=0.7418 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5873 val_joint_macro_f1=0.6250 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.2485 lr=6.37787e-05 train_metal_acc=0.9389 val_loss=1.2360 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6000 val_joint_macro_f1=0.6361 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.2337 lr=6.37787e-05 train_metal_acc=0.9331 val_loss=1.1482 val_metal_acc=0.7418 val_metal_min_recall=0.3077 val_fe_recall=0.6061 val_joint_bal_acc=0.6411 val_joint_macro_f1=0.6658 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.2522 lr=6.37787e-05 train_metal_acc=0.9418 val_loss=1.1906 val_metal_acc=0.7692 val_metal_min_recall=0.1538 val_fe_recall=0.6667 val_joint_bal_acc=0.6353 val_joint_macro_f1=0.6770 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.2129 lr=6.37787e-05 train_metal_acc=0.9379 val_loss=1.2788 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.6667 val_joint_bal_acc=0.5846 val_joint_macro_f1=0.6329 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.2080 lr=6.37787e-05 train_metal_acc=0.9360 val_loss=1.2958 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6142 val_joint_macro_f1=0.6500 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.2178 lr=6.37787e-05 train_metal_acc=0.9447 val_loss=1.3725 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.5728 val_joint_macro_f1=0.6215 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.2225 lr=6.37787e-05 train_metal_acc=0.9437 val_loss=1.4220 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6042 val_joint_macro_f1=0.6441 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.2237 lr=6.37787e-05 train_metal_acc=0.9447 val_loss=1.4109 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5731 val_joint_macro_f1=0.6224 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.2084 lr=6.37787e-05 train_metal_acc=0.9486 val_loss=1.3961 val_metal_acc=0.7418 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5989 val_joint_macro_f1=0.6393 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.2002 lr=6.37787e-05 train_metal_acc=0.9331 val_loss=1.3715 val_metal_acc=0.7253 val_metal_min_recall=0.1538 val_fe_recall=0.6667 val_joint_bal_acc=0.5957 val_joint_macro_f1=0.6256 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.1862 lr=6.37787e-05 train_metal_acc=0.9428 val_loss=1.5422 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5858 val_joint_macro_f1=0.6315 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.1887 lr=6.37787e-05 train_metal_acc=0.9476 val_loss=1.4975 val_metal_acc=0.7527 val_metal_min_recall=0.1538 val_fe_recall=0.6667 val_joint_bal_acc=0.6192 val_joint_macro_f1=0.6727 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0018_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_98f0fd36/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0018_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_98f0fd36/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0018_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_98f0fd36/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0018_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_98f0fd36/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0018_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_98f0fd36/run_metadata.json
[I 2026-05-14 03:21:21,909] Trial 18 finished with value: 0.6513935779590428 and parameters: {'learning_rate': 6.377867731540385e-05, 'weight_decay': 0.001, 'hidden_s': 128, 'head_mlp_layers': 1, 'edge_hidden': 128, 'gvp_layers': 2, 'edge_radius': 6.0, 'hidden_v': 32, 'esm_fusion_dim': 64, 'metal_class_weight_mode': 'inverse_sqrt_frequency'}. Best is trial 15 with value: 0.6550963478857217.
Optuna trial 18 completed: val_metal_balanced_acc=0.6513935779590428
================================================================================
[Optuna trial 19] optuna_deepmzyme_controlled_hpo_trial0019_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_538dfc3c
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 6.0,
  "esm_fusion_dim": 64,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 2,
  "head_mlp_layers": 1,
  "hidden_s": 128,
  "hidden_v": 32,
  "learning_rate": 4.231725529550071e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.001,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0019_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_538dfc3c --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 4.231725529550071e-05 --weight-decay 0.001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7765 lr=4.23173e-05 train_metal_acc=0.4888 val_loss=1.6693 val_metal_acc=0.5549 val_metal_min_recall=0.0000 val_fe_recall=0.3939 val_joint_bal_acc=0.3577 val_joint_macro_f1=0.3507 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.6441 lr=4.23173e-05 train_metal_acc=0.5587 val_loss=1.5031 val_metal_acc=0.6209 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4128 val_joint_macro_f1=0.4162 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.4845 lr=4.23173e-05 train_metal_acc=0.6111 val_loss=1.3992 val_metal_acc=0.6429 val_metal_min_recall=0.0000 val_fe_recall=0.6970 val_joint_bal_acc=0.4719 val_joint_macro_f1=0.4705 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.3610 lr=4.23173e-05 train_metal_acc=0.6198 val_loss=1.3327 val_metal_acc=0.4176 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4383 val_joint_macro_f1=0.4105 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.2517 lr=4.23173e-05 train_metal_acc=0.6906 val_loss=1.2183 val_metal_acc=0.6813 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4695 val_joint_macro_f1=0.4753 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.1734 lr=4.23173e-05 train_metal_acc=0.6887 val_loss=1.2542 val_metal_acc=0.4176 val_metal_min_recall=0.1538 val_fe_recall=0.5455 val_joint_bal_acc=0.4838 val_joint_macro_f1=0.4684 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=1.0872 lr=4.23173e-05 train_metal_acc=0.7100 val_loss=1.2175 val_metal_acc=0.5604 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5542 val_joint_macro_f1=0.5044 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=1.0136 lr=4.23173e-05 train_metal_acc=0.7430 val_loss=1.1819 val_metal_acc=0.4451 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.4998 val_joint_macro_f1=0.5183 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.9669 lr=4.23173e-05 train_metal_acc=0.7536 val_loss=1.1992 val_metal_acc=0.4560 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5178 val_joint_macro_f1=0.4978 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.9133 lr=4.23173e-05 train_metal_acc=0.7808 val_loss=1.1398 val_metal_acc=0.5769 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5626 val_joint_macro_f1=0.5458 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.8553 lr=4.23173e-05 train_metal_acc=0.7963 val_loss=1.0768 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5899 val_joint_macro_f1=0.6081 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.8070 lr=4.23173e-05 train_metal_acc=0.8041 val_loss=1.0513 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5925 val_joint_macro_f1=0.5841 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.7689 lr=4.23173e-05 train_metal_acc=0.8109 val_loss=1.0472 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5657 val_joint_macro_f1=0.5516 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.7083 lr=4.23173e-05 train_metal_acc=0.8196 val_loss=1.0675 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5903 val_joint_macro_f1=0.5934 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.6691 lr=4.23173e-05 train_metal_acc=0.8235 val_loss=0.9880 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5721 val_joint_macro_f1=0.5763 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.6374 lr=4.23173e-05 train_metal_acc=0.8206 val_loss=1.1243 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5853 val_joint_macro_f1=0.5804 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.6162 lr=4.23173e-05 train_metal_acc=0.8244 val_loss=1.0626 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5916 val_joint_macro_f1=0.5840 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.5937 lr=4.23173e-05 train_metal_acc=0.8565 val_loss=0.9692 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6055 val_joint_macro_f1=0.6227 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.5433 lr=4.23173e-05 train_metal_acc=0.8729 val_loss=0.9857 val_metal_acc=0.6758 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5992 val_joint_macro_f1=0.6160 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.5162 lr=4.23173e-05 train_metal_acc=0.8642 val_loss=1.0273 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5832 val_joint_macro_f1=0.5944 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.5081 lr=4.23173e-05 train_metal_acc=0.8797 val_loss=1.0229 val_metal_acc=0.7637 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.6293 val_joint_macro_f1=0.6493 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.4722 lr=4.23173e-05 train_metal_acc=0.8788 val_loss=0.9608 val_metal_acc=0.7527 val_metal_min_recall=0.2308 val_fe_recall=0.6364 val_joint_bal_acc=0.6406 val_joint_macro_f1=0.6638 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.4528 lr=4.23173e-05 train_metal_acc=0.8739 val_loss=1.0523 val_metal_acc=0.6758 val_metal_min_recall=0.1538 val_fe_recall=0.6667 val_joint_bal_acc=0.6088 val_joint_macro_f1=0.6433 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.4350 lr=4.23173e-05 train_metal_acc=0.8855 val_loss=1.0610 val_metal_acc=0.6648 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.5991 val_joint_macro_f1=0.6093 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.4220 lr=4.23173e-05 train_metal_acc=0.8914 val_loss=1.0978 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5694 val_joint_macro_f1=0.5850 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.3926 lr=4.23173e-05 train_metal_acc=0.9011 val_loss=1.0683 val_metal_acc=0.7637 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6198 val_joint_macro_f1=0.6341 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.3714 lr=4.23173e-05 train_metal_acc=0.9079 val_loss=1.0955 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6134 val_joint_macro_f1=0.6408 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.3554 lr=4.23173e-05 train_metal_acc=0.9059 val_loss=1.1017 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.6122 val_joint_macro_f1=0.6333 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.3469 lr=4.23173e-05 train_metal_acc=0.9117 val_loss=1.0995 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6176 val_joint_macro_f1=0.6515 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.3371 lr=4.23173e-05 train_metal_acc=0.9001 val_loss=1.1069 val_metal_acc=0.6758 val_metal_min_recall=0.3077 val_fe_recall=0.6061 val_joint_bal_acc=0.6244 val_joint_macro_f1=0.6450 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.3368 lr=4.23173e-05 train_metal_acc=0.9176 val_loss=1.1209 val_metal_acc=0.7033 val_metal_min_recall=0.1538 val_fe_recall=0.6667 val_joint_bal_acc=0.6186 val_joint_macro_f1=0.6501 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.2927 lr=4.23173e-05 train_metal_acc=0.9146 val_loss=1.2007 val_metal_acc=0.7637 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6263 val_joint_macro_f1=0.6562 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.3017 lr=4.23173e-05 train_metal_acc=0.9146 val_loss=1.1453 val_metal_acc=0.6868 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.6069 val_joint_macro_f1=0.6292 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.2927 lr=4.23173e-05 train_metal_acc=0.9214 val_loss=1.1665 val_metal_acc=0.7527 val_metal_min_recall=0.1538 val_fe_recall=0.6667 val_joint_bal_acc=0.6192 val_joint_macro_f1=0.6608 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.2851 lr=4.23173e-05 train_metal_acc=0.9253 val_loss=1.2318 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6130 val_joint_macro_f1=0.6408 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.2847 lr=4.23173e-05 train_metal_acc=0.9273 val_loss=1.2189 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5757 val_joint_macro_f1=0.6128 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.2825 lr=4.23173e-05 train_metal_acc=0.9273 val_loss=1.1865 val_metal_acc=0.7473 val_metal_min_recall=0.2308 val_fe_recall=0.6061 val_joint_bal_acc=0.6288 val_joint_macro_f1=0.6594 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.2597 lr=4.23173e-05 train_metal_acc=0.9108 val_loss=1.2198 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.6127 val_joint_macro_f1=0.6171 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.2493 lr=4.23173e-05 train_metal_acc=0.9350 val_loss=1.3528 val_metal_acc=0.7582 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6106 val_joint_macro_f1=0.6473 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.2291 lr=4.23173e-05 train_metal_acc=0.9321 val_loss=1.4118 val_metal_acc=0.6813 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5747 val_joint_macro_f1=0.6103 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0019_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_538dfc3c/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0019_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_538dfc3c/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0019_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_538dfc3c/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0019_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_538dfc3c/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0019_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_538dfc3c/run_metadata.json
[I 2026-05-14 03:30:07,264] Trial 19 finished with value: 0.6406408897869998 and parameters: {'learning_rate': 4.231725529550071e-05, 'weight_decay': 0.001, 'hidden_s': 128, 'head_mlp_layers': 1, 'edge_hidden': 128, 'gvp_layers': 2, 'edge_radius': 6.0, 'hidden_v': 32, 'esm_fusion_dim': 64, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 15 with value: 0.6550963478857217.
Optuna trial 19 completed: val_metal_balanced_acc=0.6406408897869998
================================================================================
[Optuna trial 20] optuna_deepmzyme_controlled_hpo_trial0020_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_04f4992c
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 6.0,
  "esm_fusion_dim": 64,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 2,
  "head_mlp_layers": 1,
  "hidden_s": 128,
  "hidden_v": 32,
  "learning_rate": 6.071674274302573e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.001,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0020_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_04f4992c --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 6.071674274302573e-05 --weight-decay 0.001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7632 lr=6.07167e-05 train_metal_acc=0.5189 val_loss=1.6184 val_metal_acc=0.5659 val_metal_min_recall=0.0000 val_fe_recall=0.3939 val_joint_bal_acc=0.3724 val_joint_macro_f1=0.3562 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.5636 lr=6.07167e-05 train_metal_acc=0.6246 val_loss=1.4104 val_metal_acc=0.6538 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4631 val_joint_macro_f1=0.4881 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.3708 lr=6.07167e-05 train_metal_acc=0.6673 val_loss=1.3063 val_metal_acc=0.6429 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4910 val_joint_macro_f1=0.4917 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.2385 lr=6.07167e-05 train_metal_acc=0.6305 val_loss=1.2920 val_metal_acc=0.4121 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4650 val_joint_macro_f1=0.4392 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.1172 lr=6.07167e-05 train_metal_acc=0.7439 val_loss=1.1637 val_metal_acc=0.6813 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4695 val_joint_macro_f1=0.4796 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.0323 lr=6.07167e-05 train_metal_acc=0.7391 val_loss=1.2387 val_metal_acc=0.4451 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5144 val_joint_macro_f1=0.4994 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=0.9381 lr=6.07167e-05 train_metal_acc=0.7633 val_loss=1.1915 val_metal_acc=0.4615 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5180 val_joint_macro_f1=0.4870 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=0.8618 lr=6.07167e-05 train_metal_acc=0.7924 val_loss=1.1541 val_metal_acc=0.5110 val_metal_min_recall=0.1538 val_fe_recall=0.5455 val_joint_bal_acc=0.5377 val_joint_macro_f1=0.5457 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.8166 lr=6.07167e-05 train_metal_acc=0.7856 val_loss=1.1563 val_metal_acc=0.5165 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5502 val_joint_macro_f1=0.5528 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.7574 lr=6.07167e-05 train_metal_acc=0.8235 val_loss=1.0685 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5589 val_joint_macro_f1=0.5493 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.6889 lr=6.07167e-05 train_metal_acc=0.8312 val_loss=1.0611 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5938 val_joint_macro_f1=0.6143 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.6573 lr=6.07167e-05 train_metal_acc=0.8341 val_loss=1.0150 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5868 val_joint_macro_f1=0.5863 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.6158 lr=6.07167e-05 train_metal_acc=0.8565 val_loss=1.0136 val_metal_acc=0.6758 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5934 val_joint_macro_f1=0.5839 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.5552 lr=6.07167e-05 train_metal_acc=0.8535 val_loss=1.0538 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5815 val_joint_macro_f1=0.5863 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.5165 lr=6.07167e-05 train_metal_acc=0.8671 val_loss=0.9858 val_metal_acc=0.6758 val_metal_min_recall=0.2308 val_fe_recall=0.6061 val_joint_bal_acc=0.6135 val_joint_macro_f1=0.6294 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.4965 lr=6.07167e-05 train_metal_acc=0.8642 val_loss=1.1590 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5866 val_joint_macro_f1=0.5964 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.4796 lr=6.07167e-05 train_metal_acc=0.8623 val_loss=1.0759 val_metal_acc=0.6923 val_metal_min_recall=0.1538 val_fe_recall=0.6667 val_joint_bal_acc=0.6215 val_joint_macro_f1=0.6490 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.4581 lr=6.07167e-05 train_metal_acc=0.8952 val_loss=1.0202 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6188 val_joint_macro_f1=0.6493 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.4147 lr=6.07167e-05 train_metal_acc=0.9001 val_loss=1.0270 val_metal_acc=0.6703 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.5942 val_joint_macro_f1=0.6030 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.3901 lr=6.07167e-05 train_metal_acc=0.8982 val_loss=1.0916 val_metal_acc=0.7088 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.6147 val_joint_macro_f1=0.6374 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.3854 lr=6.07167e-05 train_metal_acc=0.9079 val_loss=1.1022 val_metal_acc=0.7802 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.6352 val_joint_macro_f1=0.6507 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.3533 lr=6.07167e-05 train_metal_acc=0.8962 val_loss=1.0220 val_metal_acc=0.7527 val_metal_min_recall=0.3077 val_fe_recall=0.6667 val_joint_bal_acc=0.6512 val_joint_macro_f1=0.6681 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.3306 lr=6.07167e-05 train_metal_acc=0.9069 val_loss=1.1622 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6970 val_joint_bal_acc=0.6247 val_joint_macro_f1=0.6648 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.3280 lr=6.07167e-05 train_metal_acc=0.9146 val_loss=1.1295 val_metal_acc=0.6868 val_metal_min_recall=0.3077 val_fe_recall=0.6061 val_joint_bal_acc=0.6317 val_joint_macro_f1=0.6565 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.3088 lr=6.07167e-05 train_metal_acc=0.9176 val_loss=1.2313 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6000 val_joint_macro_f1=0.6219 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.2853 lr=6.07167e-05 train_metal_acc=0.9224 val_loss=1.2559 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5999 val_joint_macro_f1=0.6254 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.2794 lr=6.07167e-05 train_metal_acc=0.9243 val_loss=1.3115 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6137 val_joint_macro_f1=0.6571 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.2565 lr=6.07167e-05 train_metal_acc=0.9176 val_loss=1.2767 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6018 val_joint_macro_f1=0.6276 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.2598 lr=6.07167e-05 train_metal_acc=0.9340 val_loss=1.3484 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6000 val_joint_macro_f1=0.6361 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.2449 lr=6.07167e-05 train_metal_acc=0.9253 val_loss=1.2474 val_metal_acc=0.6978 val_metal_min_recall=0.3077 val_fe_recall=0.6061 val_joint_bal_acc=0.6322 val_joint_macro_f1=0.6527 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.2646 lr=6.07167e-05 train_metal_acc=0.9418 val_loss=1.3112 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6171 val_joint_macro_f1=0.6503 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.2149 lr=6.07167e-05 train_metal_acc=0.9340 val_loss=1.4046 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6176 val_joint_macro_f1=0.6608 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.2244 lr=6.07167e-05 train_metal_acc=0.9273 val_loss=1.3571 val_metal_acc=0.7418 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6262 val_joint_macro_f1=0.6471 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.2314 lr=6.07167e-05 train_metal_acc=0.9437 val_loss=1.4873 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.5762 val_joint_macro_f1=0.6230 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.2281 lr=6.07167e-05 train_metal_acc=0.9370 val_loss=1.4960 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6077 val_joint_macro_f1=0.6474 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.2347 lr=6.07167e-05 train_metal_acc=0.9370 val_loss=1.4950 val_metal_acc=0.7253 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5711 val_joint_macro_f1=0.6139 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.2254 lr=6.07167e-05 train_metal_acc=0.9457 val_loss=1.5349 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5969 val_joint_macro_f1=0.6367 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.2099 lr=6.07167e-05 train_metal_acc=0.9263 val_loss=1.4986 val_metal_acc=0.7198 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5978 val_joint_macro_f1=0.6140 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.1908 lr=6.07167e-05 train_metal_acc=0.9418 val_loss=1.6877 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5691 val_joint_macro_f1=0.6049 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.1863 lr=6.07167e-05 train_metal_acc=0.9447 val_loss=1.6551 val_metal_acc=0.7527 val_metal_min_recall=0.1538 val_fe_recall=0.6667 val_joint_bal_acc=0.6226 val_joint_macro_f1=0.6745 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0020_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_04f4992c/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0020_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_04f4992c/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0020_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_04f4992c/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0020_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_04f4992c/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0020_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_04f4992c/run_metadata.json
[I 2026-05-14 03:38:53,428] Trial 20 finished with value: 0.6511747792582707 and parameters: {'learning_rate': 6.071674274302573e-05, 'weight_decay': 0.001, 'hidden_s': 128, 'head_mlp_layers': 1, 'edge_hidden': 128, 'gvp_layers': 2, 'edge_radius': 6.0, 'hidden_v': 32, 'esm_fusion_dim': 64, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 15 with value: 0.6550963478857217.
Optuna trial 20 completed: val_metal_balanced_acc=0.6511747792582707
================================================================================
[Optuna trial 21] optuna_deepmzyme_controlled_hpo_trial0021_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_87949ab6
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 6.0,
  "esm_fusion_dim": 64,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 2,
  "head_mlp_layers": 1,
  "hidden_s": 128,
  "hidden_v": 32,
  "learning_rate": 7.286446905106348e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.001,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0021_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_87949ab6 --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 7.286446905106348e-05 --weight-decay 0.001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7532 lr=7.28645e-05 train_metal_acc=0.5354 val_loss=1.5803 val_metal_acc=0.5714 val_metal_min_recall=0.0000 val_fe_recall=0.3939 val_joint_bal_acc=0.3744 val_joint_macro_f1=0.3569 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.5182 lr=7.28645e-05 train_metal_acc=0.6489 val_loss=1.3683 val_metal_acc=0.6648 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4770 val_joint_macro_f1=0.4979 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.3109 lr=7.28645e-05 train_metal_acc=0.6896 val_loss=1.2689 val_metal_acc=0.6429 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.5128 val_joint_macro_f1=0.5208 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.1741 lr=7.28645e-05 train_metal_acc=0.6460 val_loss=1.2874 val_metal_acc=0.4176 val_metal_min_recall=0.0000 val_fe_recall=0.5455 val_joint_bal_acc=0.4710 val_joint_macro_f1=0.4374 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.0479 lr=7.28645e-05 train_metal_acc=0.7595 val_loss=1.1427 val_metal_acc=0.6813 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4733 val_joint_macro_f1=0.4841 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=0.9613 lr=7.28645e-05 train_metal_acc=0.7653 val_loss=1.2252 val_metal_acc=0.4560 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5218 val_joint_macro_f1=0.5106 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=0.8651 lr=7.28645e-05 train_metal_acc=0.7818 val_loss=1.1816 val_metal_acc=0.5055 val_metal_min_recall=0.1538 val_fe_recall=0.4848 val_joint_bal_acc=0.5582 val_joint_macro_f1=0.5409 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=0.7887 lr=7.28645e-05 train_metal_acc=0.8147 val_loss=1.1431 val_metal_acc=0.5934 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.5702 val_joint_macro_f1=0.5913 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.7467 lr=7.28645e-05 train_metal_acc=0.8021 val_loss=1.1314 val_metal_acc=0.5934 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5708 val_joint_macro_f1=0.5761 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.6852 lr=7.28645e-05 train_metal_acc=0.8429 val_loss=1.0611 val_metal_acc=0.6538 val_metal_min_recall=0.1538 val_fe_recall=0.4848 val_joint_bal_acc=0.5790 val_joint_macro_f1=0.5836 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.6148 lr=7.28645e-05 train_metal_acc=0.8506 val_loss=1.0749 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5958 val_joint_macro_f1=0.6205 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.5913 lr=7.28645e-05 train_metal_acc=0.8516 val_loss=1.0210 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.6047 val_joint_macro_f1=0.6167 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.5481 lr=7.28645e-05 train_metal_acc=0.8700 val_loss=1.0324 val_metal_acc=0.6868 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5973 val_joint_macro_f1=0.5898 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.4908 lr=7.28645e-05 train_metal_acc=0.8788 val_loss=1.0764 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.6019 val_joint_macro_f1=0.6160 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.4518 lr=7.28645e-05 train_metal_acc=0.8846 val_loss=1.0217 val_metal_acc=0.6813 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6111 val_joint_macro_f1=0.6245 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.4388 lr=7.28645e-05 train_metal_acc=0.8846 val_loss=1.2077 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5866 val_joint_macro_f1=0.5965 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.4242 lr=7.28645e-05 train_metal_acc=0.8914 val_loss=1.0978 val_metal_acc=0.7033 val_metal_min_recall=0.1538 val_fe_recall=0.6970 val_joint_bal_acc=0.6286 val_joint_macro_f1=0.6628 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.4046 lr=7.28645e-05 train_metal_acc=0.9011 val_loss=1.0656 val_metal_acc=0.7473 val_metal_min_recall=0.2308 val_fe_recall=0.6667 val_joint_bal_acc=0.6452 val_joint_macro_f1=0.6803 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.3639 lr=7.28645e-05 train_metal_acc=0.9088 val_loss=1.0658 val_metal_acc=0.7088 val_metal_min_recall=0.2308 val_fe_recall=0.6364 val_joint_bal_acc=0.6284 val_joint_macro_f1=0.6408 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.3415 lr=7.28645e-05 train_metal_acc=0.9098 val_loss=1.1499 val_metal_acc=0.7198 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.6118 val_joint_macro_f1=0.6385 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.3391 lr=7.28645e-05 train_metal_acc=0.9127 val_loss=1.1856 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.6030 val_joint_macro_f1=0.6271 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.3055 lr=7.28645e-05 train_metal_acc=0.9030 val_loss=1.0707 val_metal_acc=0.7637 val_metal_min_recall=0.3077 val_fe_recall=0.6667 val_joint_bal_acc=0.6517 val_joint_macro_f1=0.6705 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.2859 lr=7.28645e-05 train_metal_acc=0.9137 val_loss=1.2687 val_metal_acc=0.7143 val_metal_min_recall=0.2308 val_fe_recall=0.6970 val_joint_bal_acc=0.5908 val_joint_macro_f1=0.6388 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.2923 lr=7.28645e-05 train_metal_acc=0.9224 val_loss=1.1883 val_metal_acc=0.7363 val_metal_min_recall=0.3077 val_fe_recall=0.6364 val_joint_bal_acc=0.6524 val_joint_macro_f1=0.6804 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.2696 lr=7.28645e-05 train_metal_acc=0.9321 val_loss=1.3552 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5847 val_joint_macro_f1=0.6142 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.2513 lr=7.28645e-05 train_metal_acc=0.9321 val_loss=1.4113 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6077 val_joint_macro_f1=0.6411 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.2520 lr=7.28645e-05 train_metal_acc=0.9340 val_loss=1.4768 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6102 val_joint_macro_f1=0.6475 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.2274 lr=7.28645e-05 train_metal_acc=0.9205 val_loss=1.3643 val_metal_acc=0.7308 val_metal_min_recall=0.2308 val_fe_recall=0.6061 val_joint_bal_acc=0.6126 val_joint_macro_f1=0.6389 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.2346 lr=7.28645e-05 train_metal_acc=0.9399 val_loss=1.4864 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.5980 val_joint_macro_f1=0.6332 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.2205 lr=7.28645e-05 train_metal_acc=0.9253 val_loss=1.3523 val_metal_acc=0.7143 val_metal_min_recall=0.3077 val_fe_recall=0.6061 val_joint_bal_acc=0.6381 val_joint_macro_f1=0.6553 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.2470 lr=7.28645e-05 train_metal_acc=0.9437 val_loss=1.4649 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6210 val_joint_macro_f1=0.6632 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.1911 lr=7.28645e-05 train_metal_acc=0.9389 val_loss=1.5365 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6068 val_joint_macro_f1=0.6515 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.2045 lr=7.28645e-05 train_metal_acc=0.9350 val_loss=1.5153 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6120 val_joint_macro_f1=0.6372 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.2142 lr=7.28645e-05 train_metal_acc=0.9437 val_loss=1.6781 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.5946 val_joint_macro_f1=0.6399 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.2187 lr=7.28645e-05 train_metal_acc=0.9476 val_loss=1.7250 val_metal_acc=0.7418 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5989 val_joint_macro_f1=0.6393 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.2281 lr=7.28645e-05 train_metal_acc=0.9457 val_loss=1.6869 val_metal_acc=0.7143 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5638 val_joint_macro_f1=0.6115 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.2073 lr=7.28645e-05 train_metal_acc=0.9476 val_loss=1.7383 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5949 val_joint_macro_f1=0.6342 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.1980 lr=7.28645e-05 train_metal_acc=0.9292 val_loss=1.6519 val_metal_acc=0.7088 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.5868 val_joint_macro_f1=0.6018 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.1791 lr=7.28645e-05 train_metal_acc=0.9457 val_loss=1.8819 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6003 val_joint_macro_f1=0.6403 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.1755 lr=7.28645e-05 train_metal_acc=0.9486 val_loss=1.8199 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6176 val_joint_macro_f1=0.6608 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0021_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_87949ab6/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0021_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_87949ab6/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0021_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_87949ab6/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0021_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_87949ab6/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0021_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_87949ab6/run_metadata.json
[I 2026-05-14 03:47:37,238] Trial 21 finished with value: 0.6524493848972027 and parameters: {'learning_rate': 7.286446905106348e-05, 'weight_decay': 0.001, 'hidden_s': 128, 'head_mlp_layers': 1, 'edge_hidden': 128, 'gvp_layers': 2, 'edge_radius': 6.0, 'hidden_v': 32, 'esm_fusion_dim': 64, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 15 with value: 0.6550963478857217.
Optuna trial 21 completed: val_metal_balanced_acc=0.6524493848972027
================================================================================
[Optuna trial 22] optuna_deepmzyme_controlled_hpo_trial0022_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_abf5e755
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 6.0,
  "esm_fusion_dim": 64,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 2,
  "head_mlp_layers": 1,
  "hidden_s": 128,
  "hidden_v": 32,
  "learning_rate": 8.013708084465491e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.001,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0022_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_abf5e755 --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 8.013708084465491e-05 --weight-decay 0.001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7470 lr=8.01371e-05 train_metal_acc=0.5393 val_loss=1.5579 val_metal_acc=0.5769 val_metal_min_recall=0.0000 val_fe_recall=0.4242 val_joint_bal_acc=0.3795 val_joint_macro_f1=0.3622 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.4938 lr=8.01371e-05 train_metal_acc=0.6654 val_loss=1.3476 val_metal_acc=0.6648 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4736 val_joint_macro_f1=0.4937 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.2793 lr=8.01371e-05 train_metal_acc=0.6984 val_loss=1.2530 val_metal_acc=0.6099 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.5079 val_joint_macro_f1=0.5182 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.1399 lr=8.01371e-05 train_metal_acc=0.6576 val_loss=1.2892 val_metal_acc=0.4176 val_metal_min_recall=0.0000 val_fe_recall=0.5455 val_joint_bal_acc=0.4710 val_joint_macro_f1=0.4376 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.0120 lr=8.01371e-05 train_metal_acc=0.7682 val_loss=1.1337 val_metal_acc=0.6868 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4855 val_joint_macro_f1=0.4985 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=0.9254 lr=8.01371e-05 train_metal_acc=0.7721 val_loss=1.2112 val_metal_acc=0.4670 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5291 val_joint_macro_f1=0.5175 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=0.8283 lr=8.01371e-05 train_metal_acc=0.7924 val_loss=1.1754 val_metal_acc=0.5330 val_metal_min_recall=0.1538 val_fe_recall=0.4848 val_joint_bal_acc=0.5496 val_joint_macro_f1=0.5379 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=0.7522 lr=8.01371e-05 train_metal_acc=0.8264 val_loss=1.1427 val_metal_acc=0.6154 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5879 val_joint_macro_f1=0.6058 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.7125 lr=8.01371e-05 train_metal_acc=0.8157 val_loss=1.1221 val_metal_acc=0.6484 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5904 val_joint_macro_f1=0.5982 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.6503 lr=8.01371e-05 train_metal_acc=0.8506 val_loss=1.0654 val_metal_acc=0.6538 val_metal_min_recall=0.1538 val_fe_recall=0.4848 val_joint_bal_acc=0.5825 val_joint_macro_f1=0.5849 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.5800 lr=8.01371e-05 train_metal_acc=0.8565 val_loss=1.0843 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5723 val_joint_macro_f1=0.5917 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.5607 lr=8.01371e-05 train_metal_acc=0.8603 val_loss=1.0253 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.6027 val_joint_macro_f1=0.6093 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.5156 lr=8.01371e-05 train_metal_acc=0.8768 val_loss=1.0469 val_metal_acc=0.6978 val_metal_min_recall=0.1538 val_fe_recall=0.5455 val_joint_bal_acc=0.6043 val_joint_macro_f1=0.6015 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.4616 lr=8.01371e-05 train_metal_acc=0.8875 val_loss=1.0941 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.6157 val_joint_macro_f1=0.6321 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.4224 lr=8.01371e-05 train_metal_acc=0.8914 val_loss=1.0484 val_metal_acc=0.6923 val_metal_min_recall=0.1538 val_fe_recall=0.6970 val_joint_bal_acc=0.6178 val_joint_macro_f1=0.6425 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.4127 lr=8.01371e-05 train_metal_acc=0.8914 val_loss=1.2251 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5628 val_joint_macro_f1=0.5833 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.3985 lr=8.01371e-05 train_metal_acc=0.9011 val_loss=1.1197 val_metal_acc=0.6868 val_metal_min_recall=0.1538 val_fe_recall=0.6970 val_joint_bal_acc=0.6124 val_joint_macro_f1=0.6504 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.3810 lr=8.01371e-05 train_metal_acc=0.9030 val_loss=1.0890 val_metal_acc=0.7473 val_metal_min_recall=0.3077 val_fe_recall=0.6364 val_joint_bal_acc=0.6530 val_joint_macro_f1=0.6842 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.3413 lr=8.01371e-05 train_metal_acc=0.9127 val_loss=1.0919 val_metal_acc=0.7143 val_metal_min_recall=0.2308 val_fe_recall=0.6364 val_joint_bal_acc=0.6303 val_joint_macro_f1=0.6433 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.3213 lr=8.01371e-05 train_metal_acc=0.9137 val_loss=1.1876 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.6177 val_joint_macro_f1=0.6457 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.3194 lr=8.01371e-05 train_metal_acc=0.9176 val_loss=1.2412 val_metal_acc=0.7527 val_metal_min_recall=0.1538 val_fe_recall=0.6667 val_joint_bal_acc=0.6158 val_joint_macro_f1=0.6528 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.2849 lr=8.01371e-05 train_metal_acc=0.9069 val_loss=1.1066 val_metal_acc=0.7637 val_metal_min_recall=0.3077 val_fe_recall=0.6970 val_joint_bal_acc=0.6514 val_joint_macro_f1=0.6718 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.2688 lr=8.01371e-05 train_metal_acc=0.9117 val_loss=1.3302 val_metal_acc=0.7198 val_metal_min_recall=0.2308 val_fe_recall=0.6970 val_joint_bal_acc=0.5893 val_joint_macro_f1=0.6380 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.2792 lr=8.01371e-05 train_metal_acc=0.9243 val_loss=1.2294 val_metal_acc=0.7308 val_metal_min_recall=0.3077 val_fe_recall=0.6364 val_joint_bal_acc=0.6471 val_joint_macro_f1=0.6712 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.2545 lr=8.01371e-05 train_metal_acc=0.9399 val_loss=1.4283 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5847 val_joint_macro_f1=0.6142 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.2380 lr=8.01371e-05 train_metal_acc=0.9331 val_loss=1.4822 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6077 val_joint_macro_f1=0.6401 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.2431 lr=8.01371e-05 train_metal_acc=0.9379 val_loss=1.5703 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6142 val_joint_macro_f1=0.6526 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.2175 lr=8.01371e-05 train_metal_acc=0.9263 val_loss=1.4076 val_metal_acc=0.7418 val_metal_min_recall=0.2308 val_fe_recall=0.6061 val_joint_bal_acc=0.6200 val_joint_macro_f1=0.6460 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.2264 lr=8.01371e-05 train_metal_acc=0.9418 val_loss=1.5557 val_metal_acc=0.7198 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5876 val_joint_macro_f1=0.6152 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.2111 lr=8.01371e-05 train_metal_acc=0.9292 val_loss=1.4112 val_metal_acc=0.7088 val_metal_min_recall=0.3077 val_fe_recall=0.6061 val_joint_bal_acc=0.6327 val_joint_macro_f1=0.6529 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.2381 lr=8.01371e-05 train_metal_acc=0.9437 val_loss=1.5528 val_metal_acc=0.7527 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6230 val_joint_macro_f1=0.6658 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.1826 lr=8.01371e-05 train_metal_acc=0.9447 val_loss=1.6187 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.5980 val_joint_macro_f1=0.6435 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.1968 lr=8.01371e-05 train_metal_acc=0.9437 val_loss=1.5947 val_metal_acc=0.7253 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6101 val_joint_macro_f1=0.6478 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.2084 lr=8.01371e-05 train_metal_acc=0.9437 val_loss=1.7701 val_metal_acc=0.7253 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5896 val_joint_macro_f1=0.6277 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.2149 lr=8.01371e-05 train_metal_acc=0.9467 val_loss=1.8589 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5881 val_joint_macro_f1=0.6280 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.2283 lr=8.01371e-05 train_metal_acc=0.9476 val_loss=1.8000 val_metal_acc=0.7143 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5638 val_joint_macro_f1=0.6115 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.2012 lr=8.01371e-05 train_metal_acc=0.9486 val_loss=1.8397 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5949 val_joint_macro_f1=0.6342 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.1949 lr=8.01371e-05 train_metal_acc=0.9389 val_loss=1.7917 val_metal_acc=0.7143 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.5887 val_joint_macro_f1=0.6056 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.1721 lr=8.01371e-05 train_metal_acc=0.9496 val_loss=1.9638 val_metal_acc=0.7418 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6057 val_joint_macro_f1=0.6448 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.1668 lr=8.01371e-05 train_metal_acc=0.9534 val_loss=1.8714 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.5957 val_joint_macro_f1=0.6391 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0022_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_abf5e755/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0022_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_abf5e755/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0022_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_abf5e755/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0022_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_abf5e755/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0022_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_abf5e755/run_metadata.json
[I 2026-05-14 03:56:21,418] Trial 22 finished with value: 0.6529553937523577 and parameters: {'learning_rate': 8.013708084465491e-05, 'weight_decay': 0.001, 'hidden_s': 128, 'head_mlp_layers': 1, 'edge_hidden': 128, 'gvp_layers': 2, 'edge_radius': 6.0, 'hidden_v': 32, 'esm_fusion_dim': 64, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 15 with value: 0.6550963478857217.
Optuna trial 22 completed: val_metal_balanced_acc=0.6529553937523577
================================================================================
[Optuna trial 23] optuna_deepmzyme_controlled_hpo_trial0023_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_f32e42f6
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 6.0,
  "esm_fusion_dim": 64,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 2,
  "head_mlp_layers": 1,
  "hidden_s": 128,
  "hidden_v": 32,
  "learning_rate": 7.886105601636527e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.001,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0023_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_f32e42f6 --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 7.886105601636527e-05 --weight-decay 0.001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7481 lr=7.88611e-05 train_metal_acc=0.5383 val_loss=1.5617 val_metal_acc=0.5769 val_metal_min_recall=0.0000 val_fe_recall=0.4242 val_joint_bal_acc=0.3795 val_joint_macro_f1=0.3622 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.4980 lr=7.88611e-05 train_metal_acc=0.6654 val_loss=1.3510 val_metal_acc=0.6648 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4736 val_joint_macro_f1=0.4937 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.2847 lr=7.88611e-05 train_metal_acc=0.6984 val_loss=1.2555 val_metal_acc=0.5989 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4972 val_joint_macro_f1=0.5021 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.1457 lr=7.88611e-05 train_metal_acc=0.6566 val_loss=1.2887 val_metal_acc=0.4176 val_metal_min_recall=0.0000 val_fe_recall=0.5455 val_joint_bal_acc=0.4710 val_joint_macro_f1=0.4376 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.0180 lr=7.88611e-05 train_metal_acc=0.7672 val_loss=1.1351 val_metal_acc=0.6868 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4855 val_joint_macro_f1=0.4985 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=0.9314 lr=7.88611e-05 train_metal_acc=0.7711 val_loss=1.2140 val_metal_acc=0.4670 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5291 val_joint_macro_f1=0.5175 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=0.8344 lr=7.88611e-05 train_metal_acc=0.7905 val_loss=1.1765 val_metal_acc=0.5330 val_metal_min_recall=0.1538 val_fe_recall=0.4848 val_joint_bal_acc=0.5680 val_joint_macro_f1=0.5530 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=0.7583 lr=7.88611e-05 train_metal_acc=0.8274 val_loss=1.1423 val_metal_acc=0.6154 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5879 val_joint_macro_f1=0.6058 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.7181 lr=7.88611e-05 train_metal_acc=0.8099 val_loss=1.1234 val_metal_acc=0.6484 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5904 val_joint_macro_f1=0.5982 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.6560 lr=7.88611e-05 train_metal_acc=0.8477 val_loss=1.0645 val_metal_acc=0.6484 val_metal_min_recall=0.1538 val_fe_recall=0.4848 val_joint_bal_acc=0.5771 val_joint_macro_f1=0.5814 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.5857 lr=7.88611e-05 train_metal_acc=0.8565 val_loss=1.0831 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5723 val_joint_macro_f1=0.5917 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.5657 lr=7.88611e-05 train_metal_acc=0.8584 val_loss=1.0247 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.6027 val_joint_macro_f1=0.6093 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.5209 lr=7.88611e-05 train_metal_acc=0.8768 val_loss=1.0449 val_metal_acc=0.6978 val_metal_min_recall=0.1538 val_fe_recall=0.5455 val_joint_bal_acc=0.6043 val_joint_macro_f1=0.6015 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.4663 lr=7.88611e-05 train_metal_acc=0.8836 val_loss=1.0915 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.6137 val_joint_macro_f1=0.6298 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.4272 lr=7.88611e-05 train_metal_acc=0.8914 val_loss=1.0441 val_metal_acc=0.6923 val_metal_min_recall=0.1538 val_fe_recall=0.6970 val_joint_bal_acc=0.6178 val_joint_macro_f1=0.6418 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.4169 lr=7.88611e-05 train_metal_acc=0.8904 val_loss=1.2230 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5628 val_joint_macro_f1=0.5765 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.4027 lr=7.88611e-05 train_metal_acc=0.9011 val_loss=1.1158 val_metal_acc=0.6923 val_metal_min_recall=0.1538 val_fe_recall=0.6970 val_joint_bal_acc=0.6178 val_joint_macro_f1=0.6539 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.3847 lr=7.88611e-05 train_metal_acc=0.9020 val_loss=1.0852 val_metal_acc=0.7473 val_metal_min_recall=0.3077 val_fe_recall=0.6364 val_joint_bal_acc=0.6530 val_joint_macro_f1=0.6842 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.3449 lr=7.88611e-05 train_metal_acc=0.9127 val_loss=1.0877 val_metal_acc=0.7143 val_metal_min_recall=0.2308 val_fe_recall=0.6364 val_joint_bal_acc=0.6303 val_joint_macro_f1=0.6433 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.3245 lr=7.88611e-05 train_metal_acc=0.9117 val_loss=1.1812 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.6177 val_joint_macro_f1=0.6457 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.3225 lr=7.88611e-05 train_metal_acc=0.9185 val_loss=1.2318 val_metal_acc=0.7527 val_metal_min_recall=0.1538 val_fe_recall=0.6667 val_joint_bal_acc=0.6158 val_joint_macro_f1=0.6525 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.2881 lr=7.88611e-05 train_metal_acc=0.9049 val_loss=1.1003 val_metal_acc=0.7637 val_metal_min_recall=0.3077 val_fe_recall=0.6970 val_joint_bal_acc=0.6514 val_joint_macro_f1=0.6670 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.2714 lr=7.88611e-05 train_metal_acc=0.9117 val_loss=1.3202 val_metal_acc=0.7253 val_metal_min_recall=0.2308 val_fe_recall=0.6970 val_joint_bal_acc=0.5947 val_joint_macro_f1=0.6436 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.2811 lr=7.88611e-05 train_metal_acc=0.9243 val_loss=1.2223 val_metal_acc=0.7363 val_metal_min_recall=0.3077 val_fe_recall=0.6364 val_joint_bal_acc=0.6524 val_joint_macro_f1=0.6750 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.2570 lr=7.88611e-05 train_metal_acc=0.9379 val_loss=1.4159 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5847 val_joint_macro_f1=0.6142 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.2401 lr=7.88611e-05 train_metal_acc=0.9340 val_loss=1.4709 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6077 val_joint_macro_f1=0.6401 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.2443 lr=7.88611e-05 train_metal_acc=0.9379 val_loss=1.5553 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6142 val_joint_macro_f1=0.6526 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.2192 lr=7.88611e-05 train_metal_acc=0.9263 val_loss=1.4012 val_metal_acc=0.7473 val_metal_min_recall=0.3077 val_fe_recall=0.6061 val_joint_bal_acc=0.6328 val_joint_macro_f1=0.6620 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.2277 lr=7.88611e-05 train_metal_acc=0.9408 val_loss=1.5445 val_metal_acc=0.7253 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5930 val_joint_macro_f1=0.6211 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.2126 lr=7.88611e-05 train_metal_acc=0.9282 val_loss=1.4019 val_metal_acc=0.7088 val_metal_min_recall=0.3077 val_fe_recall=0.6061 val_joint_bal_acc=0.6327 val_joint_macro_f1=0.6529 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.2394 lr=7.88611e-05 train_metal_acc=0.9437 val_loss=1.5390 val_metal_acc=0.7527 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6230 val_joint_macro_f1=0.6658 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.1839 lr=7.88611e-05 train_metal_acc=0.9428 val_loss=1.6073 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.5980 val_joint_macro_f1=0.6435 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.1979 lr=7.88611e-05 train_metal_acc=0.9389 val_loss=1.5824 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6120 val_joint_macro_f1=0.6431 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.2101 lr=7.88611e-05 train_metal_acc=0.9437 val_loss=1.7591 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.5946 val_joint_macro_f1=0.6399 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.2151 lr=7.88611e-05 train_metal_acc=0.9467 val_loss=1.8398 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5881 val_joint_macro_f1=0.6280 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.2282 lr=7.88611e-05 train_metal_acc=0.9467 val_loss=1.7863 val_metal_acc=0.7143 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5638 val_joint_macro_f1=0.6115 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.2024 lr=7.88611e-05 train_metal_acc=0.9496 val_loss=1.8266 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5949 val_joint_macro_f1=0.6342 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.1946 lr=7.88611e-05 train_metal_acc=0.9389 val_loss=1.7694 val_metal_acc=0.7143 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.5887 val_joint_macro_f1=0.6056 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.1738 lr=7.88611e-05 train_metal_acc=0.9486 val_loss=1.9583 val_metal_acc=0.7418 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6057 val_joint_macro_f1=0.6448 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.1688 lr=7.88611e-05 train_metal_acc=0.9525 val_loss=1.8595 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.5991 val_joint_macro_f1=0.6420 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0023_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_f32e42f6/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0023_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_f32e42f6/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0023_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_f32e42f6/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0023_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_f32e42f6/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0023_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_f32e42f6/run_metadata.json
[I 2026-05-14 04:05:05,027] Trial 23 finished with value: 0.6529553937523577 and parameters: {'learning_rate': 7.886105601636527e-05, 'weight_decay': 0.001, 'hidden_s': 128, 'head_mlp_layers': 1, 'edge_hidden': 128, 'gvp_layers': 2, 'edge_radius': 6.0, 'hidden_v': 32, 'esm_fusion_dim': 64, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 15 with value: 0.6550963478857217.
Optuna trial 23 completed: val_metal_balanced_acc=0.6529553937523577
================================================================================
[Optuna trial 24] optuna_deepmzyme_controlled_hpo_trial0024_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c215ce22
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 6.0,
  "esm_fusion_dim": 64,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 2,
  "head_mlp_layers": 1,
  "hidden_s": 128,
  "hidden_v": 32,
  "learning_rate": 4.955010192592303e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.0001,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0024_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c215ce22 --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 4.955010192592303e-05 --weight-decay 0.0001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7715 lr=4.95501e-05 train_metal_acc=0.5005 val_loss=1.6506 val_metal_acc=0.5659 val_metal_min_recall=0.0000 val_fe_recall=0.3939 val_joint_bal_acc=0.3724 val_joint_macro_f1=0.3613 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.6107 lr=4.95501e-05 train_metal_acc=0.5868 val_loss=1.4603 val_metal_acc=0.6429 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4490 val_joint_macro_f1=0.4716 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.4357 lr=4.95501e-05 train_metal_acc=0.6343 val_loss=1.3561 val_metal_acc=0.6429 val_metal_min_recall=0.0000 val_fe_recall=0.6970 val_joint_bal_acc=0.4753 val_joint_macro_f1=0.4739 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.3081 lr=4.95501e-05 train_metal_acc=0.6246 val_loss=1.3102 val_metal_acc=0.4121 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4397 val_joint_macro_f1=0.4111 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.1934 lr=4.95501e-05 train_metal_acc=0.7139 val_loss=1.1917 val_metal_acc=0.6813 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4695 val_joint_macro_f1=0.4763 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.1120 lr=4.95501e-05 train_metal_acc=0.7090 val_loss=1.2455 val_metal_acc=0.4396 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5057 val_joint_macro_f1=0.4916 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=1.0217 lr=4.95501e-05 train_metal_acc=0.7430 val_loss=1.2031 val_metal_acc=0.4341 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.4877 val_joint_macro_f1=0.4457 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=0.9465 lr=4.95501e-05 train_metal_acc=0.7624 val_loss=1.1693 val_metal_acc=0.4505 val_metal_min_recall=0.1538 val_fe_recall=0.5455 val_joint_bal_acc=0.4990 val_joint_macro_f1=0.5011 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.8995 lr=4.95501e-05 train_metal_acc=0.7692 val_loss=1.1836 val_metal_acc=0.4560 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5178 val_joint_macro_f1=0.5057 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.8433 lr=4.95501e-05 train_metal_acc=0.8031 val_loss=1.1059 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5807 val_joint_macro_f1=0.5684 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.7802 lr=4.95501e-05 train_metal_acc=0.8099 val_loss=1.0624 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5899 val_joint_macro_f1=0.6081 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.7387 lr=4.95501e-05 train_metal_acc=0.8147 val_loss=1.0267 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5844 val_joint_macro_f1=0.5796 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.6990 lr=4.95501e-05 train_metal_acc=0.8322 val_loss=1.0212 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5696 val_joint_macro_f1=0.5561 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.6375 lr=4.95501e-05 train_metal_acc=0.8264 val_loss=1.0520 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5849 val_joint_macro_f1=0.5903 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.5989 lr=4.95501e-05 train_metal_acc=0.8477 val_loss=0.9753 val_metal_acc=0.6593 val_metal_min_recall=0.1538 val_fe_recall=0.5455 val_joint_bal_acc=0.5906 val_joint_macro_f1=0.5972 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.5715 lr=4.95501e-05 train_metal_acc=0.8371 val_loss=1.1275 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5818 val_joint_macro_f1=0.5794 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.5520 lr=4.95501e-05 train_metal_acc=0.8409 val_loss=1.0669 val_metal_acc=0.6868 val_metal_min_recall=0.1538 val_fe_recall=0.6667 val_joint_bal_acc=0.6196 val_joint_macro_f1=0.6460 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.5295 lr=4.95501e-05 train_metal_acc=0.8758 val_loss=0.9818 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6001 val_joint_macro_f1=0.6193 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.4825 lr=4.95501e-05 train_metal_acc=0.8894 val_loss=0.9992 val_metal_acc=0.6648 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.5922 val_joint_macro_f1=0.5995 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.4562 lr=4.95501e-05 train_metal_acc=0.8778 val_loss=1.0486 val_metal_acc=0.6758 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.6030 val_joint_macro_f1=0.6220 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.4493 lr=4.95501e-05 train_metal_acc=0.8972 val_loss=1.0440 val_metal_acc=0.7637 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.6293 val_joint_macro_f1=0.6505 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.4157 lr=4.95501e-05 train_metal_acc=0.8875 val_loss=0.9820 val_metal_acc=0.7582 val_metal_min_recall=0.2308 val_fe_recall=0.6667 val_joint_bal_acc=0.6457 val_joint_macro_f1=0.6717 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.3945 lr=4.95501e-05 train_metal_acc=0.8991 val_loss=1.0870 val_metal_acc=0.6923 val_metal_min_recall=0.1538 val_fe_recall=0.6970 val_joint_bal_acc=0.6178 val_joint_macro_f1=0.6527 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.3813 lr=4.95501e-05 train_metal_acc=0.8962 val_loss=1.0862 val_metal_acc=0.6813 val_metal_min_recall=0.3077 val_fe_recall=0.5758 val_joint_bal_acc=0.6267 val_joint_macro_f1=0.6537 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.3669 lr=4.95501e-05 train_metal_acc=0.9079 val_loss=1.1433 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5734 val_joint_macro_f1=0.5902 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.3390 lr=4.95501e-05 train_metal_acc=0.9127 val_loss=1.1323 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6048 val_joint_macro_f1=0.6187 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.3250 lr=4.95501e-05 train_metal_acc=0.9176 val_loss=1.1734 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6062 val_joint_macro_f1=0.6392 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.3053 lr=4.95501e-05 train_metal_acc=0.9146 val_loss=1.1814 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.5455 val_joint_bal_acc=0.5990 val_joint_macro_f1=0.6285 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.3030 lr=4.95501e-05 train_metal_acc=0.9214 val_loss=1.1969 val_metal_acc=0.7527 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6127 val_joint_macro_f1=0.6512 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.2891 lr=4.95501e-05 train_metal_acc=0.9156 val_loss=1.1649 val_metal_acc=0.6813 val_metal_min_recall=0.3077 val_fe_recall=0.6364 val_joint_bal_acc=0.6294 val_joint_macro_f1=0.6529 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.2979 lr=4.95501e-05 train_metal_acc=0.9273 val_loss=1.1891 val_metal_acc=0.7418 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6293 val_joint_macro_f1=0.6577 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.2523 lr=4.95501e-05 train_metal_acc=0.9273 val_loss=1.2810 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6101 val_joint_macro_f1=0.6450 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.2618 lr=4.95501e-05 train_metal_acc=0.9195 val_loss=1.2283 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6257 val_joint_macro_f1=0.6507 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.2594 lr=4.95501e-05 train_metal_acc=0.9331 val_loss=1.2852 val_metal_acc=0.7418 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6054 val_joint_macro_f1=0.6500 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.2536 lr=4.95501e-05 train_metal_acc=0.9340 val_loss=1.3250 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6102 val_joint_macro_f1=0.6540 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.2548 lr=4.95501e-05 train_metal_acc=0.9350 val_loss=1.3268 val_metal_acc=0.7418 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.5835 val_joint_macro_f1=0.6316 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.2565 lr=4.95501e-05 train_metal_acc=0.9389 val_loss=1.3119 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6111 val_joint_macro_f1=0.6407 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.2318 lr=4.95501e-05 train_metal_acc=0.9176 val_loss=1.3239 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5958 val_joint_macro_f1=0.6041 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.2185 lr=4.95501e-05 train_metal_acc=0.9379 val_loss=1.4709 val_metal_acc=0.7637 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6160 val_joint_macro_f1=0.6529 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.2037 lr=4.95501e-05 train_metal_acc=0.9379 val_loss=1.5201 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5874 val_joint_macro_f1=0.6310 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0024_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c215ce22/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0024_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c215ce22/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0024_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c215ce22/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0024_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c215ce22/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0024_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c215ce22/run_metadata.json
[I 2026-05-14 04:13:45,042] Trial 24 finished with value: 0.6456913948375049 and parameters: {'learning_rate': 4.955010192592303e-05, 'weight_decay': 0.0001, 'hidden_s': 128, 'head_mlp_layers': 1, 'edge_hidden': 128, 'gvp_layers': 2, 'edge_radius': 6.0, 'hidden_v': 32, 'esm_fusion_dim': 64, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 15 with value: 0.6550963478857217.
Optuna trial 24 completed: val_metal_balanced_acc=0.6456913948375049
================================================================================
[Optuna trial 25] optuna_deepmzyme_controlled_hpo_trial0025_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_5305cf54
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 6.0,
  "esm_fusion_dim": 128,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 2,
  "head_mlp_layers": 2,
  "hidden_s": 128,
  "hidden_v": 32,
  "learning_rate": 5.845630767888206e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.001,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0025_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_5305cf54 --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 5.845630767888206e-05 --weight-decay 0.001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 128 --head-mlp-layers 2 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7733 lr=5.84563e-05 train_metal_acc=0.4006 val_loss=1.7292 val_metal_acc=0.2692 val_metal_min_recall=0.0000 val_fe_recall=0.4545 val_joint_bal_acc=0.2346 val_joint_macro_f1=0.1524 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.6426 lr=5.84563e-05 train_metal_acc=0.6460 val_loss=1.5591 val_metal_acc=0.4066 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4094 val_joint_macro_f1=0.3969 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.4551 lr=5.84563e-05 train_metal_acc=0.6295 val_loss=1.4508 val_metal_acc=0.4176 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4570 val_joint_macro_f1=0.4434 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.2933 lr=5.84563e-05 train_metal_acc=0.6925 val_loss=1.3398 val_metal_acc=0.4341 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4492 val_joint_macro_f1=0.4464 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.1433 lr=5.84563e-05 train_metal_acc=0.7304 val_loss=1.2713 val_metal_acc=0.4560 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.4918 val_joint_macro_f1=0.5015 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.0330 lr=5.84563e-05 train_metal_acc=0.7527 val_loss=1.2234 val_metal_acc=0.4725 val_metal_min_recall=0.1538 val_fe_recall=0.5455 val_joint_bal_acc=0.5239 val_joint_macro_f1=0.5287 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=0.9263 lr=5.84563e-05 train_metal_acc=0.7701 val_loss=1.2275 val_metal_acc=0.4560 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.4956 val_joint_macro_f1=0.4910 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=0.8345 lr=5.84563e-05 train_metal_acc=0.7973 val_loss=1.1352 val_metal_acc=0.5824 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5154 val_joint_macro_f1=0.5246 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.7672 lr=5.84563e-05 train_metal_acc=0.8167 val_loss=1.0697 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5801 val_joint_macro_f1=0.5851 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.6969 lr=5.84563e-05 train_metal_acc=0.8089 val_loss=1.0147 val_metal_acc=0.7527 val_metal_min_recall=0.2308 val_fe_recall=0.5758 val_joint_bal_acc=0.6160 val_joint_macro_f1=0.6512 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.6418 lr=5.84563e-05 train_metal_acc=0.8438 val_loss=1.0797 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5797 val_joint_macro_f1=0.5729 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.5759 lr=5.84563e-05 train_metal_acc=0.8584 val_loss=1.0567 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5889 val_joint_macro_f1=0.6149 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.5336 lr=5.84563e-05 train_metal_acc=0.8545 val_loss=1.0829 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.5455 val_joint_bal_acc=0.6249 val_joint_macro_f1=0.6289 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.4964 lr=5.84563e-05 train_metal_acc=0.8885 val_loss=1.0595 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5920 val_joint_macro_f1=0.6260 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.4903 lr=5.84563e-05 train_metal_acc=0.8933 val_loss=1.0972 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.6057 val_joint_macro_f1=0.6192 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.4415 lr=5.84563e-05 train_metal_acc=0.9049 val_loss=1.0407 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6139 val_joint_macro_f1=0.6316 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.3875 lr=5.84563e-05 train_metal_acc=0.8952 val_loss=1.0768 val_metal_acc=0.7692 val_metal_min_recall=0.2308 val_fe_recall=0.6667 val_joint_bal_acc=0.6530 val_joint_macro_f1=0.6804 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.3544 lr=5.84563e-05 train_metal_acc=0.9079 val_loss=1.2852 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5563 val_joint_macro_f1=0.5895 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.3625 lr=5.84563e-05 train_metal_acc=0.9127 val_loss=1.2002 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5977 val_joint_macro_f1=0.6100 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.3268 lr=5.84563e-05 train_metal_acc=0.9059 val_loss=1.1891 val_metal_acc=0.7473 val_metal_min_recall=0.2308 val_fe_recall=0.5455 val_joint_bal_acc=0.6328 val_joint_macro_f1=0.6463 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.3226 lr=5.84563e-05 train_metal_acc=0.9127 val_loss=1.3324 val_metal_acc=0.7527 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.6162 val_joint_macro_f1=0.6318 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.3188 lr=5.84563e-05 train_metal_acc=0.9195 val_loss=1.3022 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6062 val_joint_macro_f1=0.6237 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.2902 lr=5.84563e-05 train_metal_acc=0.9156 val_loss=1.3602 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6667 val_joint_bal_acc=0.6022 val_joint_macro_f1=0.6458 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.2845 lr=5.84563e-05 train_metal_acc=0.9292 val_loss=1.6344 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5710 val_joint_macro_f1=0.6150 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.2766 lr=5.84563e-05 train_metal_acc=0.9205 val_loss=1.5398 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5549 val_joint_macro_f1=0.5984 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.2521 lr=5.84563e-05 train_metal_acc=0.9321 val_loss=1.5010 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5809 val_joint_macro_f1=0.6185 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.2543 lr=5.84563e-05 train_metal_acc=0.9214 val_loss=1.5678 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5759 val_joint_macro_f1=0.5836 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.2631 lr=5.84563e-05 train_metal_acc=0.9263 val_loss=1.6078 val_metal_acc=0.6978 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5436 val_joint_macro_f1=0.5571 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.2560 lr=5.84563e-05 train_metal_acc=0.9340 val_loss=1.6791 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5841 val_joint_macro_f1=0.6165 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.2451 lr=5.84563e-05 train_metal_acc=0.9370 val_loss=1.8036 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5563 val_joint_macro_f1=0.5885 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.2700 lr=5.84563e-05 train_metal_acc=0.9302 val_loss=1.7250 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5790 val_joint_macro_f1=0.6050 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.2907 lr=5.84563e-05 train_metal_acc=0.9370 val_loss=1.8994 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5782 val_joint_macro_f1=0.5907 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.2286 lr=5.84563e-05 train_metal_acc=0.9418 val_loss=1.7872 val_metal_acc=0.7198 val_metal_min_recall=0.2308 val_fe_recall=0.6061 val_joint_bal_acc=0.5766 val_joint_macro_f1=0.6199 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.2217 lr=5.84563e-05 train_metal_acc=0.9418 val_loss=1.8519 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5544 val_joint_macro_f1=0.5800 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.2340 lr=5.84563e-05 train_metal_acc=0.9379 val_loss=1.8843 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5743 val_joint_macro_f1=0.5645 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.2319 lr=5.84563e-05 train_metal_acc=0.9447 val_loss=1.9302 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5802 val_joint_macro_f1=0.6021 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.2558 lr=5.84563e-05 train_metal_acc=0.9428 val_loss=1.9983 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5802 val_joint_macro_f1=0.6021 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.2456 lr=5.84563e-05 train_metal_acc=0.9467 val_loss=2.2657 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5841 val_joint_macro_f1=0.6146 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.2532 lr=5.84563e-05 train_metal_acc=0.9418 val_loss=2.0998 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5691 val_joint_macro_f1=0.6056 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.2817 lr=5.84563e-05 train_metal_acc=0.9447 val_loss=1.9736 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5891 val_joint_macro_f1=0.5843 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0025_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_5305cf54/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0025_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_5305cf54/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0025_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_5305cf54/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0025_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_5305cf54/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0025_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_5305cf54/run_metadata.json
[I 2026-05-14 04:22:28,658] Trial 25 finished with value: 0.6530285232372519 and parameters: {'learning_rate': 5.845630767888206e-05, 'weight_decay': 0.001, 'hidden_s': 128, 'head_mlp_layers': 2, 'edge_hidden': 128, 'gvp_layers': 2, 'edge_radius': 6.0, 'hidden_v': 32, 'esm_fusion_dim': 128, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 15 with value: 0.6550963478857217.
Optuna trial 25 completed: val_metal_balanced_acc=0.6530285232372519
================================================================================
[Optuna trial 26] optuna_deepmzyme_controlled_hpo_trial0026_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_3db29e38
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 8.0,
  "esm_fusion_dim": 128,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 2,
  "head_mlp_layers": 2,
  "hidden_s": 128,
  "hidden_v": 32,
  "learning_rate": 4.186863436908622e-05,
  "metal_class_weight_mode": "inverse_sqrt_frequency",
  "metal_weighting_setup": "inverse_sqrt_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.001,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0026_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_3db29e38 --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 4.186863436908622e-05 --weight-decay 0.001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 128 --head-mlp-layers 2 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_sqrt_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 8.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.6905 lr=4.18686e-05 train_metal_acc=0.4753 val_loss=1.6345 val_metal_acc=0.5000 val_metal_min_recall=0.0000 val_fe_recall=0.1818 val_joint_bal_acc=0.1970 val_joint_macro_f1=0.1582 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.5934 lr=4.18686e-05 train_metal_acc=0.5577 val_loss=1.5232 val_metal_acc=0.5549 val_metal_min_recall=0.0000 val_fe_recall=0.5152 val_joint_bal_acc=0.2506 val_joint_macro_f1=0.2159 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.4477 lr=4.18686e-05 train_metal_acc=0.6169 val_loss=1.4322 val_metal_acc=0.6429 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4097 val_joint_macro_f1=0.3995 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.3133 lr=4.18686e-05 train_metal_acc=0.6625 val_loss=1.3445 val_metal_acc=0.6648 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4316 val_joint_macro_f1=0.4281 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.1925 lr=4.18686e-05 train_metal_acc=0.6984 val_loss=1.2632 val_metal_acc=0.6703 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4578 val_joint_macro_f1=0.4648 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.0880 lr=4.18686e-05 train_metal_acc=0.7478 val_loss=1.2159 val_metal_acc=0.6703 val_metal_min_recall=0.0000 val_fe_recall=0.5152 val_joint_bal_acc=0.4652 val_joint_macro_f1=0.4683 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=1.0011 lr=4.18686e-05 train_metal_acc=0.7643 val_loss=1.1997 val_metal_acc=0.5385 val_metal_min_recall=0.0000 val_fe_recall=0.5455 val_joint_bal_acc=0.4602 val_joint_macro_f1=0.4652 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=0.9202 lr=4.18686e-05 train_metal_acc=0.7750 val_loss=1.1074 val_metal_acc=0.7308 val_metal_min_recall=0.0000 val_fe_recall=0.5152 val_joint_bal_acc=0.5189 val_joint_macro_f1=0.5367 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.8552 lr=4.18686e-05 train_metal_acc=0.8041 val_loss=1.0489 val_metal_acc=0.7253 val_metal_min_recall=0.0000 val_fe_recall=0.5152 val_joint_bal_acc=0.5169 val_joint_macro_f1=0.5343 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.7886 lr=4.18686e-05 train_metal_acc=0.8293 val_loss=1.0095 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5332 val_joint_macro_f1=0.5614 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.7425 lr=4.18686e-05 train_metal_acc=0.8390 val_loss=1.0076 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5619 val_joint_macro_f1=0.5865 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.6838 lr=4.18686e-05 train_metal_acc=0.8526 val_loss=0.9593 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5620 val_joint_macro_f1=0.6002 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.6307 lr=4.18686e-05 train_metal_acc=0.8468 val_loss=0.9736 val_metal_acc=0.7527 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5762 val_joint_macro_f1=0.5998 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.5984 lr=4.18686e-05 train_metal_acc=0.8661 val_loss=0.9424 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5651 val_joint_macro_f1=0.6027 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.5742 lr=4.18686e-05 train_metal_acc=0.8749 val_loss=0.9631 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5620 val_joint_macro_f1=0.5875 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.5422 lr=4.18686e-05 train_metal_acc=0.8807 val_loss=0.9302 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5733 val_joint_macro_f1=0.6130 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.4931 lr=4.18686e-05 train_metal_acc=0.8904 val_loss=0.9493 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5884 val_joint_macro_f1=0.6114 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.4632 lr=4.18686e-05 train_metal_acc=0.8846 val_loss=1.0920 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5492 val_joint_macro_f1=0.5998 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.4354 lr=4.18686e-05 train_metal_acc=0.9030 val_loss=1.0017 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5901 val_joint_macro_f1=0.6164 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.4021 lr=4.18686e-05 train_metal_acc=0.9020 val_loss=0.9773 val_metal_acc=0.7582 val_metal_min_recall=0.2308 val_fe_recall=0.6061 val_joint_bal_acc=0.6245 val_joint_macro_f1=0.6522 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.3869 lr=4.18686e-05 train_metal_acc=0.9059 val_loss=1.0598 val_metal_acc=0.7527 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5974 val_joint_macro_f1=0.6299 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.3803 lr=4.18686e-05 train_metal_acc=0.9166 val_loss=1.0135 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5951 val_joint_macro_f1=0.6274 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.3533 lr=4.18686e-05 train_metal_acc=0.9146 val_loss=1.0307 val_metal_acc=0.7637 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.6075 val_joint_macro_f1=0.6383 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.3357 lr=4.18686e-05 train_metal_acc=0.9224 val_loss=1.1452 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5705 val_joint_macro_f1=0.6042 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.3172 lr=4.18686e-05 train_metal_acc=0.9088 val_loss=1.2082 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5619 val_joint_macro_f1=0.6150 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.3050 lr=4.18686e-05 train_metal_acc=0.9253 val_loss=1.1677 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5790 val_joint_macro_f1=0.6163 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.2966 lr=4.18686e-05 train_metal_acc=0.9273 val_loss=1.1657 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5648 val_joint_macro_f1=0.5733 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.2863 lr=4.18686e-05 train_metal_acc=0.9292 val_loss=1.2105 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5668 val_joint_macro_f1=0.5999 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.2777 lr=4.18686e-05 train_metal_acc=0.9350 val_loss=1.3204 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5656 val_joint_macro_f1=0.6084 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.2642 lr=4.18686e-05 train_metal_acc=0.9360 val_loss=1.3298 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5668 val_joint_macro_f1=0.6068 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.2660 lr=4.18686e-05 train_metal_acc=0.9253 val_loss=1.2474 val_metal_acc=0.7527 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5971 val_joint_macro_f1=0.6291 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.2620 lr=4.18686e-05 train_metal_acc=0.9370 val_loss=1.3878 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5687 val_joint_macro_f1=0.6071 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.2289 lr=4.18686e-05 train_metal_acc=0.9379 val_loss=1.3947 val_metal_acc=0.7198 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5658 val_joint_macro_f1=0.6066 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.2318 lr=4.18686e-05 train_metal_acc=0.9350 val_loss=1.4685 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5648 val_joint_macro_f1=0.5959 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.2393 lr=4.18686e-05 train_metal_acc=0.9331 val_loss=1.4106 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5612 val_joint_macro_f1=0.5684 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.2355 lr=4.18686e-05 train_metal_acc=0.9350 val_loss=1.5889 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5583 val_joint_macro_f1=0.5902 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.2564 lr=4.18686e-05 train_metal_acc=0.9408 val_loss=1.6917 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5855 val_joint_macro_f1=0.6143 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.2379 lr=4.18686e-05 train_metal_acc=0.9428 val_loss=1.6714 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5855 val_joint_macro_f1=0.6202 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.2514 lr=4.18686e-05 train_metal_acc=0.9399 val_loss=1.6566 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5617 val_joint_macro_f1=0.5866 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.2623 lr=4.18686e-05 train_metal_acc=0.9408 val_loss=1.4983 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5598 val_joint_macro_f1=0.5828 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0026_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_3db29e38/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0026_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_3db29e38/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0026_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_3db29e38/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0026_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_3db29e38/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0026_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_3db29e38/run_metadata.json
[I 2026-05-14 04:32:09,057] Trial 26 finished with value: 0.6244943334127395 and parameters: {'learning_rate': 4.186863436908622e-05, 'weight_decay': 0.001, 'hidden_s': 128, 'head_mlp_layers': 2, 'edge_hidden': 128, 'gvp_layers': 2, 'edge_radius': 8.0, 'hidden_v': 32, 'esm_fusion_dim': 128, 'metal_class_weight_mode': 'inverse_sqrt_frequency'}. Best is trial 15 with value: 0.6550963478857217.
Optuna trial 26 completed: val_metal_balanced_acc=0.6244943334127395
================================================================================
[Optuna trial 27] optuna_deepmzyme_controlled_hpo_trial0027_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_e110d321
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 6.0,
  "esm_fusion_dim": 128,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 2,
  "head_mlp_layers": 2,
  "hidden_s": 128,
  "hidden_v": 32,
  "learning_rate": 5.56210918619566e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.001,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0027_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_e110d321 --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 5.56210918619566e-05 --weight-decay 0.001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 128 --head-mlp-layers 2 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7744 lr=5.56211e-05 train_metal_acc=0.3977 val_loss=1.7331 val_metal_acc=0.2692 val_metal_min_recall=0.0000 val_fe_recall=0.4545 val_joint_bal_acc=0.2346 val_joint_macro_f1=0.1524 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.6530 lr=5.56211e-05 train_metal_acc=0.6421 val_loss=1.5700 val_metal_acc=0.4011 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.3856 val_joint_macro_f1=0.3548 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.4722 lr=5.56211e-05 train_metal_acc=0.6295 val_loss=1.4637 val_metal_acc=0.4121 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4550 val_joint_macro_f1=0.4395 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.3147 lr=5.56211e-05 train_metal_acc=0.6867 val_loss=1.3531 val_metal_acc=0.4176 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4181 val_joint_macro_f1=0.4054 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.1670 lr=5.56211e-05 train_metal_acc=0.7294 val_loss=1.2799 val_metal_acc=0.4505 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.4646 val_joint_macro_f1=0.4749 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.0579 lr=5.56211e-05 train_metal_acc=0.7439 val_loss=1.2388 val_metal_acc=0.4670 val_metal_min_recall=0.1538 val_fe_recall=0.5455 val_joint_bal_acc=0.5001 val_joint_macro_f1=0.5215 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=0.9527 lr=5.56211e-05 train_metal_acc=0.7653 val_loss=1.2363 val_metal_acc=0.4560 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.4956 val_joint_macro_f1=0.4909 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=0.8607 lr=5.56211e-05 train_metal_acc=0.7934 val_loss=1.1487 val_metal_acc=0.5385 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.4997 val_joint_macro_f1=0.5066 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.7916 lr=5.56211e-05 train_metal_acc=0.8157 val_loss=1.0836 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5747 val_joint_macro_f1=0.5817 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.7199 lr=5.56211e-05 train_metal_acc=0.8060 val_loss=1.0277 val_metal_acc=0.7143 val_metal_min_recall=0.2308 val_fe_recall=0.5455 val_joint_bal_acc=0.6026 val_joint_macro_f1=0.6326 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.6662 lr=5.56211e-05 train_metal_acc=0.8400 val_loss=1.0813 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5778 val_joint_macro_f1=0.5706 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.5980 lr=5.56211e-05 train_metal_acc=0.8526 val_loss=1.0483 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5889 val_joint_macro_f1=0.6134 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.5529 lr=5.56211e-05 train_metal_acc=0.8419 val_loss=1.0781 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.6023 val_joint_macro_f1=0.5964 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.5139 lr=5.56211e-05 train_metal_acc=0.8826 val_loss=1.0467 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5920 val_joint_macro_f1=0.6260 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.5068 lr=5.56211e-05 train_metal_acc=0.8885 val_loss=1.0871 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.6057 val_joint_macro_f1=0.6192 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.4585 lr=5.56211e-05 train_metal_acc=0.8972 val_loss=1.0321 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.6088 val_joint_macro_f1=0.6253 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.4032 lr=5.56211e-05 train_metal_acc=0.8991 val_loss=1.0627 val_metal_acc=0.7637 val_metal_min_recall=0.2308 val_fe_recall=0.6364 val_joint_bal_acc=0.6480 val_joint_macro_f1=0.6707 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.3695 lr=5.56211e-05 train_metal_acc=0.9040 val_loss=1.2561 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5544 val_joint_macro_f1=0.5913 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.3735 lr=5.56211e-05 train_metal_acc=0.9117 val_loss=1.1660 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5977 val_joint_macro_f1=0.6100 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.3366 lr=5.56211e-05 train_metal_acc=0.9020 val_loss=1.1561 val_metal_acc=0.7527 val_metal_min_recall=0.2308 val_fe_recall=0.5455 val_joint_bal_acc=0.6382 val_joint_macro_f1=0.6502 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.3314 lr=5.56211e-05 train_metal_acc=0.9108 val_loss=1.2811 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.6142 val_joint_macro_f1=0.6245 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.3267 lr=5.56211e-05 train_metal_acc=0.9205 val_loss=1.2613 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6062 val_joint_macro_f1=0.6249 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.2971 lr=5.56211e-05 train_metal_acc=0.9137 val_loss=1.3256 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5894 val_joint_macro_f1=0.6244 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.2889 lr=5.56211e-05 train_metal_acc=0.9282 val_loss=1.5612 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5725 val_joint_macro_f1=0.6082 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.2813 lr=5.56211e-05 train_metal_acc=0.9176 val_loss=1.4811 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5549 val_joint_macro_f1=0.5984 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.2585 lr=5.56211e-05 train_metal_acc=0.9292 val_loss=1.4545 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5809 val_joint_macro_f1=0.6180 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.2569 lr=5.56211e-05 train_metal_acc=0.9224 val_loss=1.5154 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5813 val_joint_macro_f1=0.5886 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.2652 lr=5.56211e-05 train_metal_acc=0.9243 val_loss=1.5555 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5674 val_joint_macro_f1=0.5749 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.2584 lr=5.56211e-05 train_metal_acc=0.9340 val_loss=1.6321 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5841 val_joint_macro_f1=0.6165 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.2480 lr=5.56211e-05 train_metal_acc=0.9370 val_loss=1.7279 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5563 val_joint_macro_f1=0.5896 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.2679 lr=5.56211e-05 train_metal_acc=0.9253 val_loss=1.6483 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5770 val_joint_macro_f1=0.6025 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.2877 lr=5.56211e-05 train_metal_acc=0.9350 val_loss=1.8500 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5782 val_joint_macro_f1=0.5907 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.2295 lr=5.56211e-05 train_metal_acc=0.9408 val_loss=1.7323 val_metal_acc=0.7253 val_metal_min_recall=0.2308 val_fe_recall=0.6061 val_joint_bal_acc=0.5820 val_joint_macro_f1=0.6254 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.2236 lr=5.56211e-05 train_metal_acc=0.9408 val_loss=1.8084 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5524 val_joint_macro_f1=0.5775 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.2356 lr=5.56211e-05 train_metal_acc=0.9360 val_loss=1.8765 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5870 val_joint_macro_f1=0.5804 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.2336 lr=5.56211e-05 train_metal_acc=0.9467 val_loss=1.8967 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5802 val_joint_macro_f1=0.6010 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.2575 lr=5.56211e-05 train_metal_acc=0.9399 val_loss=1.9742 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5802 val_joint_macro_f1=0.6076 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.2445 lr=5.56211e-05 train_metal_acc=0.9457 val_loss=2.2212 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5821 val_joint_macro_f1=0.6116 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.2537 lr=5.56211e-05 train_metal_acc=0.9408 val_loss=2.0794 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5637 val_joint_macro_f1=0.5963 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.2823 lr=5.56211e-05 train_metal_acc=0.9447 val_loss=1.9352 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5891 val_joint_macro_f1=0.5843 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0027_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_e110d321/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0027_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_e110d321/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0027_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_e110d321/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0027_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_e110d321/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0027_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_e110d321/run_metadata.json
[I 2026-05-14 04:40:52,517] Trial 27 finished with value: 0.6479780181867468 and parameters: {'learning_rate': 5.56210918619566e-05, 'weight_decay': 0.001, 'hidden_s': 128, 'head_mlp_layers': 2, 'edge_hidden': 128, 'gvp_layers': 2, 'edge_radius': 6.0, 'hidden_v': 32, 'esm_fusion_dim': 128, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 15 with value: 0.6550963478857217.
Optuna trial 27 completed: val_metal_balanced_acc=0.6479780181867468
================================================================================
[Optuna trial 28] optuna_deepmzyme_controlled_hpo_trial0028_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_409c9d6c
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 6.0,
  "esm_fusion_dim": 128,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 2,
  "head_mlp_layers": 2,
  "hidden_s": 128,
  "hidden_v": 32,
  "learning_rate": 3.901101221031459e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.001,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0028_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_409c9d6c --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 3.901101221031459e-05 --weight-decay 0.001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 128 --head-mlp-layers 2 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7792 lr=3.9011e-05 train_metal_acc=0.4365 val_loss=1.7512 val_metal_acc=0.2967 val_metal_min_recall=0.0000 val_fe_recall=0.5152 val_joint_bal_acc=0.2506 val_joint_macro_f1=0.1704 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.7138 lr=3.9011e-05 train_metal_acc=0.5926 val_loss=1.6419 val_metal_acc=0.3407 val_metal_min_recall=0.0000 val_fe_recall=0.5152 val_joint_bal_acc=0.3026 val_joint_macro_f1=0.3119 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.5795 lr=3.9011e-05 train_metal_acc=0.5965 val_loss=1.5568 val_metal_acc=0.3736 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.3840 val_joint_macro_f1=0.3639 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.4566 lr=3.9011e-05 train_metal_acc=0.6285 val_loss=1.4562 val_metal_acc=0.3791 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.3620 val_joint_macro_f1=0.3549 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.3322 lr=3.9011e-05 train_metal_acc=0.6877 val_loss=1.3679 val_metal_acc=0.4396 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4177 val_joint_macro_f1=0.4153 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.2338 lr=3.9011e-05 train_metal_acc=0.7148 val_loss=1.3355 val_metal_acc=0.4615 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.4904 val_joint_macro_f1=0.5082 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=1.1424 lr=3.9011e-05 train_metal_acc=0.7391 val_loss=1.2956 val_metal_acc=0.4615 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.4720 val_joint_macro_f1=0.4817 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=1.0573 lr=3.9011e-05 train_metal_acc=0.7624 val_loss=1.2349 val_metal_acc=0.4725 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.4725 val_joint_macro_f1=0.4920 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.9827 lr=3.9011e-05 train_metal_acc=0.7769 val_loss=1.1894 val_metal_acc=0.5000 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5010 val_joint_macro_f1=0.5067 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.9061 lr=3.9011e-05 train_metal_acc=0.7653 val_loss=1.1658 val_metal_acc=0.4890 val_metal_min_recall=0.2308 val_fe_recall=0.5455 val_joint_bal_acc=0.5004 val_joint_macro_f1=0.5096 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.8566 lr=3.9011e-05 train_metal_acc=0.7934 val_loss=1.1653 val_metal_acc=0.5440 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5423 val_joint_macro_f1=0.5295 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.7786 lr=3.9011e-05 train_metal_acc=0.8041 val_loss=1.0879 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5602 val_joint_macro_f1=0.5818 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.7175 lr=3.9011e-05 train_metal_acc=0.7818 val_loss=1.1374 val_metal_acc=0.5769 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5683 val_joint_macro_f1=0.5619 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.6744 lr=3.9011e-05 train_metal_acc=0.8244 val_loss=1.0271 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5831 val_joint_macro_f1=0.6123 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.6594 lr=3.9011e-05 train_metal_acc=0.8458 val_loss=1.0717 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5693 val_joint_macro_f1=0.5752 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.6133 lr=3.9011e-05 train_metal_acc=0.8584 val_loss=0.9963 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6029 val_joint_macro_f1=0.6376 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.5560 lr=3.9011e-05 train_metal_acc=0.8497 val_loss=1.0266 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6134 val_joint_macro_f1=0.6188 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.5157 lr=3.9011e-05 train_metal_acc=0.8739 val_loss=1.1209 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5694 val_joint_macro_f1=0.6192 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.4938 lr=3.9011e-05 train_metal_acc=0.8768 val_loss=1.0505 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.6088 val_joint_macro_f1=0.6096 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.4441 lr=3.9011e-05 train_metal_acc=0.8603 val_loss=1.0336 val_metal_acc=0.7473 val_metal_min_recall=0.2308 val_fe_recall=0.5758 val_joint_bal_acc=0.6393 val_joint_macro_f1=0.6495 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.4366 lr=3.9011e-05 train_metal_acc=0.8855 val_loss=1.1061 val_metal_acc=0.7582 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6246 val_joint_macro_f1=0.6172 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.4226 lr=3.9011e-05 train_metal_acc=0.8943 val_loss=1.0653 val_metal_acc=0.7527 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6298 val_joint_macro_f1=0.6418 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.3854 lr=3.9011e-05 train_metal_acc=0.9030 val_loss=1.1084 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5951 val_joint_macro_f1=0.6265 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.3611 lr=3.9011e-05 train_metal_acc=0.9117 val_loss=1.1806 val_metal_acc=0.7527 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6158 val_joint_macro_f1=0.6413 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.3477 lr=3.9011e-05 train_metal_acc=0.9079 val_loss=1.1302 val_metal_acc=0.7473 val_metal_min_recall=0.2308 val_fe_recall=0.6061 val_joint_bal_acc=0.6069 val_joint_macro_f1=0.6556 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.3343 lr=3.9011e-05 train_metal_acc=0.9176 val_loss=1.1649 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5901 val_joint_macro_f1=0.6242 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.3164 lr=3.9011e-05 train_metal_acc=0.9156 val_loss=1.2110 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6116 val_joint_macro_f1=0.6027 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.3126 lr=3.9011e-05 train_metal_acc=0.9166 val_loss=1.2277 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5977 val_joint_macro_f1=0.5986 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.3014 lr=3.9011e-05 train_metal_acc=0.9243 val_loss=1.2835 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5875 val_joint_macro_f1=0.6267 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.2908 lr=3.9011e-05 train_metal_acc=0.9292 val_loss=1.3273 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5997 val_joint_macro_f1=0.6285 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.2905 lr=3.9011e-05 train_metal_acc=0.9166 val_loss=1.2925 val_metal_acc=0.7527 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6039 val_joint_macro_f1=0.6259 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.3002 lr=3.9011e-05 train_metal_acc=0.9176 val_loss=1.4055 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6008 val_joint_macro_f1=0.6138 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.2420 lr=3.9011e-05 train_metal_acc=0.9292 val_loss=1.3649 val_metal_acc=0.7143 val_metal_min_recall=0.2308 val_fe_recall=0.6061 val_joint_bal_acc=0.5747 val_joint_macro_f1=0.6130 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.2465 lr=3.9011e-05 train_metal_acc=0.9292 val_loss=1.4544 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5663 val_joint_macro_f1=0.5916 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.2484 lr=3.9011e-05 train_metal_acc=0.9263 val_loss=1.4560 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6012 val_joint_macro_f1=0.6022 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.2498 lr=3.9011e-05 train_metal_acc=0.9311 val_loss=1.5992 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5728 val_joint_macro_f1=0.5893 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.2700 lr=3.9011e-05 train_metal_acc=0.9370 val_loss=1.6494 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5855 val_joint_macro_f1=0.6159 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.2483 lr=3.9011e-05 train_metal_acc=0.9360 val_loss=1.7599 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5617 val_joint_macro_f1=0.6016 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.2665 lr=3.9011e-05 train_metal_acc=0.9321 val_loss=1.6700 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5705 val_joint_macro_f1=0.6024 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.2677 lr=3.9011e-05 train_metal_acc=0.9302 val_loss=1.5429 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5870 val_joint_macro_f1=0.5970 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0028_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_409c9d6c/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0028_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_409c9d6c/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0028_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_409c9d6c/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0028_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_409c9d6c/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0028_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_409c9d6c/run_metadata.json
[I 2026-05-14 04:49:39,123] Trial 28 finished with value: 0.6393317835443072 and parameters: {'learning_rate': 3.901101221031459e-05, 'weight_decay': 0.001, 'hidden_s': 128, 'head_mlp_layers': 2, 'edge_hidden': 128, 'gvp_layers': 2, 'edge_radius': 6.0, 'hidden_v': 32, 'esm_fusion_dim': 128, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 15 with value: 0.6550963478857217.
Optuna trial 28 completed: val_metal_balanced_acc=0.6393317835443072
================================================================================
[Optuna trial 29] optuna_deepmzyme_controlled_hpo_trial0029_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_7e958f6c
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 8.0,
  "esm_fusion_dim": 128,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 2,
  "head_mlp_layers": 2,
  "hidden_s": 128,
  "hidden_v": 32,
  "learning_rate": 2.758432620478292e-05,
  "metal_class_weight_mode": "inverse_sqrt_frequency",
  "metal_weighting_setup": "inverse_sqrt_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.0001,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0029_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_7e958f6c --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 2.758432620478292e-05 --weight-decay 0.0001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 128 --head-mlp-layers 2 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_sqrt_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 8.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7044 lr=2.75843e-05 train_metal_acc=0.4607 val_loss=1.6483 val_metal_acc=0.4670 val_metal_min_recall=0.0000 val_fe_recall=0.0000 val_joint_bal_acc=0.1667 val_joint_macro_f1=0.1061 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.6418 lr=2.75843e-05 train_metal_acc=0.5073 val_loss=1.5944 val_metal_acc=0.5275 val_metal_min_recall=0.0000 val_fe_recall=0.3333 val_joint_bal_acc=0.2222 val_joint_macro_f1=0.1913 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.5538 lr=2.75843e-05 train_metal_acc=0.5664 val_loss=1.5296 val_metal_acc=0.5989 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.3655 val_joint_macro_f1=0.3454 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.4565 lr=2.75843e-05 train_metal_acc=0.6023 val_loss=1.4438 val_metal_acc=0.6264 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.3862 val_joint_macro_f1=0.3696 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.3630 lr=2.75843e-05 train_metal_acc=0.6431 val_loss=1.3759 val_metal_acc=0.6538 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4205 val_joint_macro_f1=0.4169 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.2741 lr=2.75843e-05 train_metal_acc=0.6634 val_loss=1.3348 val_metal_acc=0.6813 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4446 val_joint_macro_f1=0.4565 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=1.1984 lr=2.75843e-05 train_metal_acc=0.6925 val_loss=1.3016 val_metal_acc=0.5549 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.3961 val_joint_macro_f1=0.4030 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=1.1268 lr=2.75843e-05 train_metal_acc=0.7197 val_loss=1.2396 val_metal_acc=0.6703 val_metal_min_recall=0.0000 val_fe_recall=0.5455 val_joint_bal_acc=0.4444 val_joint_macro_f1=0.4544 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=1.0614 lr=2.75843e-05 train_metal_acc=0.7313 val_loss=1.1876 val_metal_acc=0.6868 val_metal_min_recall=0.0000 val_fe_recall=0.5455 val_joint_bal_acc=0.4605 val_joint_macro_f1=0.4693 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.9950 lr=2.75843e-05 train_metal_acc=0.7507 val_loss=1.1537 val_metal_acc=0.6978 val_metal_min_recall=0.0000 val_fe_recall=0.5455 val_joint_bal_acc=0.4747 val_joint_macro_f1=0.4807 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.9475 lr=2.75843e-05 train_metal_acc=0.7837 val_loss=1.1630 val_metal_acc=0.6484 val_metal_min_recall=0.0000 val_fe_recall=0.5152 val_joint_bal_acc=0.4963 val_joint_macro_f1=0.5054 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.8876 lr=2.75843e-05 train_metal_acc=0.8002 val_loss=1.0838 val_metal_acc=0.7143 val_metal_min_recall=0.0000 val_fe_recall=0.5152 val_joint_bal_acc=0.5164 val_joint_macro_f1=0.5310 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.8302 lr=2.75843e-05 train_metal_acc=0.8031 val_loss=1.1180 val_metal_acc=0.6538 val_metal_min_recall=0.0000 val_fe_recall=0.5152 val_joint_bal_acc=0.5304 val_joint_macro_f1=0.5412 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.7954 lr=2.75843e-05 train_metal_acc=0.8177 val_loss=1.0120 val_metal_acc=0.7253 val_metal_min_recall=0.0000 val_fe_recall=0.5152 val_joint_bal_acc=0.5169 val_joint_macro_f1=0.5357 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.7679 lr=2.75843e-05 train_metal_acc=0.8361 val_loss=1.0196 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5624 val_joint_macro_f1=0.5972 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.7359 lr=2.75843e-05 train_metal_acc=0.8380 val_loss=0.9550 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5570 val_joint_macro_f1=0.5952 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.6892 lr=2.75843e-05 train_metal_acc=0.8458 val_loss=0.9639 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5677 val_joint_macro_f1=0.5930 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.6549 lr=2.75843e-05 train_metal_acc=0.8332 val_loss=1.0185 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5462 val_joint_macro_f1=0.5897 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.6211 lr=2.75843e-05 train_metal_acc=0.8535 val_loss=0.9606 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5842 val_joint_macro_f1=0.6130 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.5829 lr=2.75843e-05 train_metal_acc=0.8526 val_loss=0.9331 val_metal_acc=0.7582 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.6031 val_joint_macro_f1=0.6216 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.5637 lr=2.75843e-05 train_metal_acc=0.8710 val_loss=0.9638 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5858 val_joint_macro_f1=0.6125 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.5493 lr=2.75843e-05 train_metal_acc=0.8749 val_loss=0.9120 val_metal_acc=0.7527 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5971 val_joint_macro_f1=0.6366 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.5170 lr=2.75843e-05 train_metal_acc=0.8768 val_loss=0.9210 val_metal_acc=0.7527 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5971 val_joint_macro_f1=0.6291 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.4853 lr=2.75843e-05 train_metal_acc=0.8904 val_loss=0.9502 val_metal_acc=0.7582 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5990 val_joint_macro_f1=0.6334 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.4652 lr=2.75843e-05 train_metal_acc=0.8875 val_loss=0.9811 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5472 val_joint_macro_f1=0.5963 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.4486 lr=2.75843e-05 train_metal_acc=0.8933 val_loss=0.9879 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5829 val_joint_macro_f1=0.6234 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.4349 lr=2.75843e-05 train_metal_acc=0.9079 val_loss=0.9621 val_metal_acc=0.7527 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5971 val_joint_macro_f1=0.6304 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.4103 lr=2.75843e-05 train_metal_acc=0.9069 val_loss=0.9838 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5951 val_joint_macro_f1=0.6152 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.3948 lr=2.75843e-05 train_metal_acc=0.9030 val_loss=1.0330 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5795 val_joint_macro_f1=0.6293 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.3770 lr=2.75843e-05 train_metal_acc=0.9146 val_loss=1.0237 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5756 val_joint_macro_f1=0.6155 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.3634 lr=2.75843e-05 train_metal_acc=0.9079 val_loss=1.0061 val_metal_acc=0.7582 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6059 val_joint_macro_f1=0.6346 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.3588 lr=2.75843e-05 train_metal_acc=0.9166 val_loss=1.0234 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5948 val_joint_macro_f1=0.6217 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.3187 lr=2.75843e-05 train_metal_acc=0.9137 val_loss=1.0406 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5772 val_joint_macro_f1=0.6188 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.3206 lr=2.75843e-05 train_metal_acc=0.9195 val_loss=1.0956 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5806 val_joint_macro_f1=0.6185 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.3221 lr=2.75843e-05 train_metal_acc=0.9137 val_loss=1.1139 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5932 val_joint_macro_f1=0.6093 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.3076 lr=2.75843e-05 train_metal_acc=0.9224 val_loss=1.1944 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5664 val_joint_macro_f1=0.6057 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.3054 lr=2.75843e-05 train_metal_acc=0.9292 val_loss=1.1890 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5756 val_joint_macro_f1=0.6137 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.2877 lr=2.75843e-05 train_metal_acc=0.9302 val_loss=1.1774 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5756 val_joint_macro_f1=0.6133 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.2933 lr=2.75843e-05 train_metal_acc=0.9253 val_loss=1.2134 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5809 val_joint_macro_f1=0.6189 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.2837 lr=2.75843e-05 train_metal_acc=0.9302 val_loss=1.1407 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5809 val_joint_macro_f1=0.6176 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0029_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_7e958f6c/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0029_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_7e958f6c/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0029_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_7e958f6c/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0029_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_7e958f6c/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0029_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_7e958f6c/run_metadata.json
[I 2026-05-14 04:59:14,485] Trial 29 finished with value: 0.6058645971359443 and parameters: {'learning_rate': 2.758432620478292e-05, 'weight_decay': 0.0001, 'hidden_s': 128, 'head_mlp_layers': 2, 'edge_hidden': 128, 'gvp_layers': 2, 'edge_radius': 8.0, 'hidden_v': 32, 'esm_fusion_dim': 128, 'metal_class_weight_mode': 'inverse_sqrt_frequency'}. Best is trial 15 with value: 0.6550963478857217.
Optuna trial 29 completed: val_metal_balanced_acc=0.6058645971359443
================================================================================
[Optuna trial 30] optuna_deepmzyme_controlled_hpo_trial0030_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a6745f6e
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 6.0,
  "esm_fusion_dim": 128,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 2,
  "head_mlp_layers": 2,
  "hidden_s": 128,
  "hidden_v": 32,
  "learning_rate": 6.954830191634325e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.001,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0030_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a6745f6e --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 6.954830191634325e-05 --weight-decay 0.001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 128 --head-mlp-layers 2 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7686 lr=6.95483e-05 train_metal_acc=0.4491 val_loss=1.7121 val_metal_acc=0.3187 val_metal_min_recall=0.0000 val_fe_recall=0.5152 val_joint_bal_acc=0.3453 val_joint_macro_f1=0.2850 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.6031 lr=6.95483e-05 train_metal_acc=0.6508 val_loss=1.5197 val_metal_acc=0.4121 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4148 val_joint_macro_f1=0.3995 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.3924 lr=6.95483e-05 train_metal_acc=0.6363 val_loss=1.4075 val_metal_acc=0.4341 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4950 val_joint_macro_f1=0.4757 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.2174 lr=6.95483e-05 train_metal_acc=0.7110 val_loss=1.2991 val_metal_acc=0.4505 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.5091 val_joint_macro_f1=0.4965 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.0609 lr=6.95483e-05 train_metal_acc=0.7420 val_loss=1.2429 val_metal_acc=0.4505 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5151 val_joint_macro_f1=0.5108 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=0.9474 lr=6.95483e-05 train_metal_acc=0.7633 val_loss=1.1606 val_metal_acc=0.5385 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5444 val_joint_macro_f1=0.5389 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=0.8375 lr=6.95483e-05 train_metal_acc=0.7837 val_loss=1.1894 val_metal_acc=0.4451 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.4882 val_joint_macro_f1=0.4836 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=0.7485 lr=6.95483e-05 train_metal_acc=0.8186 val_loss=1.0925 val_metal_acc=0.6593 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5506 val_joint_macro_f1=0.5715 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.6886 lr=6.95483e-05 train_metal_acc=0.8293 val_loss=1.0317 val_metal_acc=0.6868 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5926 val_joint_macro_f1=0.6003 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.6247 lr=6.95483e-05 train_metal_acc=0.8487 val_loss=1.0051 val_metal_acc=0.7582 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.6037 val_joint_macro_f1=0.6408 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.5637 lr=6.95483e-05 train_metal_acc=0.8729 val_loss=1.0896 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.6103 val_joint_macro_f1=0.6031 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.5068 lr=6.95483e-05 train_metal_acc=0.8846 val_loss=1.1072 val_metal_acc=0.7527 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5909 val_joint_macro_f1=0.6275 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.4740 lr=6.95483e-05 train_metal_acc=0.8855 val_loss=1.1169 val_metal_acc=0.7692 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.6397 val_joint_macro_f1=0.6458 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.4464 lr=6.95483e-05 train_metal_acc=0.9059 val_loss=1.1141 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5870 val_joint_macro_f1=0.6205 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.4428 lr=6.95483e-05 train_metal_acc=0.9069 val_loss=1.1448 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.6038 val_joint_macro_f1=0.6162 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.3941 lr=6.95483e-05 train_metal_acc=0.9117 val_loss=1.0834 val_metal_acc=0.7582 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6243 val_joint_macro_f1=0.6388 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.3461 lr=6.95483e-05 train_metal_acc=0.9011 val_loss=1.1599 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6667 val_joint_bal_acc=0.6079 val_joint_macro_f1=0.6330 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.3101 lr=6.95483e-05 train_metal_acc=0.9195 val_loss=1.3888 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5544 val_joint_macro_f1=0.5830 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.3354 lr=6.95483e-05 train_metal_acc=0.9176 val_loss=1.3582 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5909 val_joint_macro_f1=0.6131 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.3008 lr=6.95483e-05 train_metal_acc=0.9098 val_loss=1.3214 val_metal_acc=0.7637 val_metal_min_recall=0.2308 val_fe_recall=0.6061 val_joint_bal_acc=0.6449 val_joint_macro_f1=0.6616 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.3015 lr=6.95483e-05 train_metal_acc=0.9146 val_loss=1.5487 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.6009 val_joint_macro_f1=0.6312 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.3037 lr=6.95483e-05 train_metal_acc=0.9292 val_loss=1.4286 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6176 val_joint_macro_f1=0.6498 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.2760 lr=6.95483e-05 train_metal_acc=0.9195 val_loss=1.4946 val_metal_acc=0.7418 val_metal_min_recall=0.1538 val_fe_recall=0.6667 val_joint_bal_acc=0.5969 val_joint_macro_f1=0.6383 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.2722 lr=6.95483e-05 train_metal_acc=0.9321 val_loss=1.8028 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5860 val_joint_macro_f1=0.6216 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.2632 lr=6.95483e-05 train_metal_acc=0.9234 val_loss=1.7276 val_metal_acc=0.7253 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.5640 val_joint_macro_f1=0.6123 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.2321 lr=6.95483e-05 train_metal_acc=0.9360 val_loss=1.6659 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5759 val_joint_macro_f1=0.6063 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.2524 lr=6.95483e-05 train_metal_acc=0.9273 val_loss=1.6948 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5709 val_joint_macro_f1=0.5757 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.2533 lr=6.95483e-05 train_metal_acc=0.9292 val_loss=1.7835 val_metal_acc=0.6978 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5477 val_joint_macro_f1=0.5615 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.2462 lr=6.95483e-05 train_metal_acc=0.9311 val_loss=1.8322 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5737 val_joint_macro_f1=0.6004 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.2363 lr=6.95483e-05 train_metal_acc=0.9408 val_loss=2.0739 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5510 val_joint_macro_f1=0.5826 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.2742 lr=6.95483e-05 train_metal_acc=0.9360 val_loss=1.8570 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5648 val_joint_macro_f1=0.5902 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.2940 lr=6.95483e-05 train_metal_acc=0.9418 val_loss=2.0456 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5782 val_joint_macro_f1=0.5907 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.2253 lr=6.95483e-05 train_metal_acc=0.9418 val_loss=1.8709 val_metal_acc=0.7033 val_metal_min_recall=0.2308 val_fe_recall=0.6061 val_joint_bal_acc=0.5421 val_joint_macro_f1=0.5824 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.2227 lr=6.95483e-05 train_metal_acc=0.9437 val_loss=1.9547 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5544 val_joint_macro_f1=0.5800 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.2346 lr=6.95483e-05 train_metal_acc=0.9437 val_loss=1.9193 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5743 val_joint_macro_f1=0.5626 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.2165 lr=6.95483e-05 train_metal_acc=0.9447 val_loss=2.0636 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5802 val_joint_macro_f1=0.5920 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.2488 lr=6.95483e-05 train_metal_acc=0.9467 val_loss=2.1388 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5855 val_joint_macro_f1=0.6066 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.2501 lr=6.95483e-05 train_metal_acc=0.9505 val_loss=2.3633 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5895 val_joint_macro_f1=0.6127 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.2641 lr=6.95483e-05 train_metal_acc=0.9476 val_loss=2.3139 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5622 val_joint_macro_f1=0.5956 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.2713 lr=6.95483e-05 train_metal_acc=0.9399 val_loss=2.0291 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5743 val_joint_macro_f1=0.5497 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0030_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a6745f6e/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0030_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a6745f6e/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0030_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a6745f6e/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0030_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a6745f6e/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0030_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a6745f6e/run_metadata.json
[I 2026-05-14 05:08:02,199] Trial 30 finished with value: 0.6448882974499673 and parameters: {'learning_rate': 6.954830191634325e-05, 'weight_decay': 0.001, 'hidden_s': 128, 'head_mlp_layers': 2, 'edge_hidden': 128, 'gvp_layers': 2, 'edge_radius': 6.0, 'hidden_v': 32, 'esm_fusion_dim': 128, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 15 with value: 0.6550963478857217.
Optuna trial 30 completed: val_metal_balanced_acc=0.6448882974499673
================================================================================
[Optuna trial 31] optuna_deepmzyme_controlled_hpo_trial0031_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c145b928
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 6.0,
  "esm_fusion_dim": 64,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 2,
  "head_mlp_layers": 1,
  "hidden_s": 128,
  "hidden_v": 32,
  "learning_rate": 8.55126186078418e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.001,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0031_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c145b928 --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 8.55126186078418e-05 --weight-decay 0.001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7422 lr=8.55126e-05 train_metal_acc=0.5441 val_loss=1.5424 val_metal_acc=0.5769 val_metal_min_recall=0.0000 val_fe_recall=0.4242 val_joint_bal_acc=0.3795 val_joint_macro_f1=0.3544 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.4770 lr=8.55126e-05 train_metal_acc=0.6673 val_loss=1.3342 val_metal_acc=0.6648 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4736 val_joint_macro_f1=0.4948 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.2578 lr=8.55126e-05 train_metal_acc=0.7051 val_loss=1.2438 val_metal_acc=0.5659 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4860 val_joint_macro_f1=0.4832 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.1167 lr=8.55126e-05 train_metal_acc=0.6673 val_loss=1.2918 val_metal_acc=0.4176 val_metal_min_recall=0.0000 val_fe_recall=0.5455 val_joint_bal_acc=0.4710 val_joint_macro_f1=0.4376 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=0.9877 lr=8.55126e-05 train_metal_acc=0.7701 val_loss=1.1286 val_metal_acc=0.6923 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4909 val_joint_macro_f1=0.5015 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=0.9014 lr=8.55126e-05 train_metal_acc=0.7798 val_loss=1.1986 val_metal_acc=0.4615 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5019 val_joint_macro_f1=0.5032 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=0.8038 lr=8.55126e-05 train_metal_acc=0.7973 val_loss=1.1710 val_metal_acc=0.5549 val_metal_min_recall=0.1538 val_fe_recall=0.4848 val_joint_bal_acc=0.5574 val_joint_macro_f1=0.5473 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=0.7278 lr=8.55126e-05 train_metal_acc=0.8332 val_loss=1.1460 val_metal_acc=0.6154 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5879 val_joint_macro_f1=0.6075 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.6901 lr=8.55126e-05 train_metal_acc=0.8312 val_loss=1.1179 val_metal_acc=0.6538 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5924 val_joint_macro_f1=0.6004 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.6276 lr=8.55126e-05 train_metal_acc=0.8545 val_loss=1.0717 val_metal_acc=0.6484 val_metal_min_recall=0.1538 val_fe_recall=0.4848 val_joint_bal_acc=0.5771 val_joint_macro_f1=0.5796 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.5577 lr=8.55126e-05 train_metal_acc=0.8661 val_loss=1.0932 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5723 val_joint_macro_f1=0.5917 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.5415 lr=8.55126e-05 train_metal_acc=0.8642 val_loss=1.0294 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.5455 val_joint_bal_acc=0.6078 val_joint_macro_f1=0.6186 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.4944 lr=8.55126e-05 train_metal_acc=0.8826 val_loss=1.0584 val_metal_acc=0.7088 val_metal_min_recall=0.1538 val_fe_recall=0.5455 val_joint_bal_acc=0.6082 val_joint_macro_f1=0.6062 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.4436 lr=8.55126e-05 train_metal_acc=0.8952 val_loss=1.1087 val_metal_acc=0.7527 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.6196 val_joint_macro_f1=0.6389 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.4039 lr=8.55126e-05 train_metal_acc=0.9001 val_loss=1.0693 val_metal_acc=0.6923 val_metal_min_recall=0.1538 val_fe_recall=0.6970 val_joint_bal_acc=0.6178 val_joint_macro_f1=0.6429 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.3965 lr=8.55126e-05 train_metal_acc=0.8933 val_loss=1.2334 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5624 val_joint_macro_f1=0.5850 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.3823 lr=8.55126e-05 train_metal_acc=0.9069 val_loss=1.1385 val_metal_acc=0.7088 val_metal_min_recall=0.1538 val_fe_recall=0.6970 val_joint_bal_acc=0.6203 val_joint_macro_f1=0.6582 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.3667 lr=8.55126e-05 train_metal_acc=0.9049 val_loss=1.1076 val_metal_acc=0.7473 val_metal_min_recall=0.3077 val_fe_recall=0.6364 val_joint_bal_acc=0.6530 val_joint_macro_f1=0.6847 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.3270 lr=8.55126e-05 train_metal_acc=0.9146 val_loss=1.1148 val_metal_acc=0.7198 val_metal_min_recall=0.2308 val_fe_recall=0.6364 val_joint_bal_acc=0.6323 val_joint_macro_f1=0.6412 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.3094 lr=8.55126e-05 train_metal_acc=0.9166 val_loss=1.2173 val_metal_acc=0.7418 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6228 val_joint_macro_f1=0.6517 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.3077 lr=8.55126e-05 train_metal_acc=0.9234 val_loss=1.2830 val_metal_acc=0.7527 val_metal_min_recall=0.1538 val_fe_recall=0.6667 val_joint_bal_acc=0.6158 val_joint_macro_f1=0.6528 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.2728 lr=8.55126e-05 train_metal_acc=0.9098 val_loss=1.1393 val_metal_acc=0.7363 val_metal_min_recall=0.3077 val_fe_recall=0.6970 val_joint_bal_acc=0.6245 val_joint_macro_f1=0.6455 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.2597 lr=8.55126e-05 train_metal_acc=0.9137 val_loss=1.3768 val_metal_acc=0.7143 val_metal_min_recall=0.2308 val_fe_recall=0.6970 val_joint_bal_acc=0.5839 val_joint_macro_f1=0.6320 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.2726 lr=8.55126e-05 train_metal_acc=0.9282 val_loss=1.2648 val_metal_acc=0.7088 val_metal_min_recall=0.3077 val_fe_recall=0.6061 val_joint_bal_acc=0.6293 val_joint_macro_f1=0.6562 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.2460 lr=8.55126e-05 train_metal_acc=0.9457 val_loss=1.4787 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5886 val_joint_macro_f1=0.6195 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.2308 lr=8.55126e-05 train_metal_acc=0.9350 val_loss=1.5262 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6077 val_joint_macro_f1=0.6401 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.2389 lr=8.55126e-05 train_metal_acc=0.9418 val_loss=1.6304 val_metal_acc=0.7418 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6054 val_joint_macro_f1=0.6447 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.2115 lr=8.55126e-05 train_metal_acc=0.9302 val_loss=1.4350 val_metal_acc=0.7363 val_metal_min_recall=0.2308 val_fe_recall=0.6061 val_joint_bal_acc=0.6146 val_joint_macro_f1=0.6390 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.2220 lr=8.55126e-05 train_metal_acc=0.9457 val_loss=1.6058 val_metal_acc=0.7198 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5876 val_joint_macro_f1=0.6172 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.2051 lr=8.55126e-05 train_metal_acc=0.9302 val_loss=1.4584 val_metal_acc=0.7033 val_metal_min_recall=0.3077 val_fe_recall=0.6061 val_joint_bal_acc=0.6273 val_joint_macro_f1=0.6421 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.2346 lr=8.55126e-05 train_metal_acc=0.9467 val_loss=1.6124 val_metal_acc=0.7527 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6230 val_joint_macro_f1=0.6658 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.1776 lr=8.55126e-05 train_metal_acc=0.9467 val_loss=1.6699 val_metal_acc=0.7143 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5856 val_joint_macro_f1=0.6203 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.1939 lr=8.55126e-05 train_metal_acc=0.9467 val_loss=1.6549 val_metal_acc=0.7253 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6066 val_joint_macro_f1=0.6461 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.2035 lr=8.55126e-05 train_metal_acc=0.9418 val_loss=1.8168 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5915 val_joint_macro_f1=0.6302 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.2144 lr=8.55126e-05 train_metal_acc=0.9486 val_loss=1.9478 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5881 val_joint_macro_f1=0.6280 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.2251 lr=8.55126e-05 train_metal_acc=0.9515 val_loss=1.8629 val_metal_acc=0.7088 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5618 val_joint_macro_f1=0.6086 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.1956 lr=8.55126e-05 train_metal_acc=0.9515 val_loss=1.9087 val_metal_acc=0.7253 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5896 val_joint_macro_f1=0.6273 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.1868 lr=8.55126e-05 train_metal_acc=0.9457 val_loss=1.8403 val_metal_acc=0.7033 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5783 val_joint_macro_f1=0.5912 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.1692 lr=8.55126e-05 train_metal_acc=0.9534 val_loss=2.0316 val_metal_acc=0.7418 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6057 val_joint_macro_f1=0.6448 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.1633 lr=8.55126e-05 train_metal_acc=0.9554 val_loss=1.9242 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.5884 val_joint_macro_f1=0.6317 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0031_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c145b928/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0031_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c145b928/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0031_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c145b928/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0031_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c145b928/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0031_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c145b928/run_metadata.json
[I 2026-05-14 05:16:57,415] Trial 31 finished with value: 0.6529553937523577 and parameters: {'learning_rate': 8.55126186078418e-05, 'weight_decay': 0.001, 'hidden_s': 128, 'head_mlp_layers': 1, 'edge_hidden': 128, 'gvp_layers': 2, 'edge_radius': 6.0, 'hidden_v': 32, 'esm_fusion_dim': 64, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 15 with value: 0.6550963478857217.
Optuna trial 31 completed: val_metal_balanced_acc=0.6529553937523577
================================================================================
[Optuna trial 32] optuna_deepmzyme_controlled_hpo_trial0032_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_faf01e7c
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 6.0,
  "esm_fusion_dim": 64,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 2,
  "head_mlp_layers": 1,
  "hidden_s": 128,
  "hidden_v": 32,
  "learning_rate": 5.4715836015281065e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.001,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0032_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_faf01e7c --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 5.4715836015281065e-05 --weight-decay 0.001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7678 lr=5.47158e-05 train_metal_acc=0.5073 val_loss=1.6362 val_metal_acc=0.5659 val_metal_min_recall=0.0000 val_fe_recall=0.3939 val_joint_bal_acc=0.3724 val_joint_macro_f1=0.3616 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.5883 lr=5.47158e-05 train_metal_acc=0.6014 val_loss=1.4355 val_metal_acc=0.6538 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4631 val_joint_macro_f1=0.4889 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.4043 lr=5.47158e-05 train_metal_acc=0.6460 val_loss=1.3309 val_metal_acc=0.6429 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4910 val_joint_macro_f1=0.4918 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.2744 lr=5.47158e-05 train_metal_acc=0.6285 val_loss=1.2997 val_metal_acc=0.4011 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4358 val_joint_macro_f1=0.4053 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.1564 lr=5.47158e-05 train_metal_acc=0.7236 val_loss=1.1774 val_metal_acc=0.6813 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4695 val_joint_macro_f1=0.4761 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.0731 lr=5.47158e-05 train_metal_acc=0.7245 val_loss=1.2423 val_metal_acc=0.4396 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5091 val_joint_macro_f1=0.4954 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=0.9806 lr=5.47158e-05 train_metal_acc=0.7536 val_loss=1.1970 val_metal_acc=0.4451 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5053 val_joint_macro_f1=0.4698 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=0.9048 lr=5.47158e-05 train_metal_acc=0.7769 val_loss=1.1621 val_metal_acc=0.4780 val_metal_min_recall=0.1538 val_fe_recall=0.5455 val_joint_bal_acc=0.5191 val_joint_macro_f1=0.5182 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.8585 lr=5.47158e-05 train_metal_acc=0.7808 val_loss=1.1715 val_metal_acc=0.4670 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5325 val_joint_macro_f1=0.5296 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.8008 lr=5.47158e-05 train_metal_acc=0.8089 val_loss=1.0849 val_metal_acc=0.6319 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5569 val_joint_macro_f1=0.5565 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.7347 lr=5.47158e-05 train_metal_acc=0.8186 val_loss=1.0590 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5899 val_joint_macro_f1=0.6087 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.6980 lr=5.47158e-05 train_metal_acc=0.8244 val_loss=1.0177 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5883 val_joint_macro_f1=0.5851 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.6573 lr=5.47158e-05 train_metal_acc=0.8438 val_loss=1.0133 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5736 val_joint_macro_f1=0.5607 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.5959 lr=5.47158e-05 train_metal_acc=0.8380 val_loss=1.0499 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5765 val_joint_macro_f1=0.5755 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.5575 lr=5.47158e-05 train_metal_acc=0.8565 val_loss=0.9768 val_metal_acc=0.6703 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6007 val_joint_macro_f1=0.6150 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.5335 lr=5.47158e-05 train_metal_acc=0.8497 val_loss=1.1389 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5815 val_joint_macro_f1=0.5860 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.5152 lr=5.47158e-05 train_metal_acc=0.8477 val_loss=1.0705 val_metal_acc=0.6923 val_metal_min_recall=0.1538 val_fe_recall=0.6667 val_joint_bal_acc=0.6215 val_joint_macro_f1=0.6476 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.4931 lr=5.47158e-05 train_metal_acc=0.8826 val_loss=0.9981 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6021 val_joint_macro_f1=0.6265 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.4480 lr=5.47158e-05 train_metal_acc=0.8982 val_loss=1.0112 val_metal_acc=0.6648 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.5922 val_joint_macro_f1=0.6007 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.4225 lr=5.47158e-05 train_metal_acc=0.8885 val_loss=1.0671 val_metal_acc=0.6758 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.6030 val_joint_macro_f1=0.6240 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.4164 lr=5.47158e-05 train_metal_acc=0.9011 val_loss=1.0679 val_metal_acc=0.7637 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.6293 val_joint_macro_f1=0.6460 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.3840 lr=5.47158e-05 train_metal_acc=0.8923 val_loss=0.9999 val_metal_acc=0.7637 val_metal_min_recall=0.3077 val_fe_recall=0.6667 val_joint_bal_acc=0.6585 val_joint_macro_f1=0.6830 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.3617 lr=5.47158e-05 train_metal_acc=0.9020 val_loss=1.1185 val_metal_acc=0.7253 val_metal_min_recall=0.1538 val_fe_recall=0.6970 val_joint_bal_acc=0.6296 val_joint_macro_f1=0.6665 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.3533 lr=5.47158e-05 train_metal_acc=0.9059 val_loss=1.1054 val_metal_acc=0.6868 val_metal_min_recall=0.3077 val_fe_recall=0.5758 val_joint_bal_acc=0.6286 val_joint_macro_f1=0.6560 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.3367 lr=5.47158e-05 train_metal_acc=0.9127 val_loss=1.1795 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5961 val_joint_macro_f1=0.6169 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.3106 lr=5.47158e-05 train_metal_acc=0.9176 val_loss=1.1842 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6013 val_joint_macro_f1=0.6160 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.3008 lr=5.47158e-05 train_metal_acc=0.9214 val_loss=1.2339 val_metal_acc=0.7418 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6190 val_joint_macro_f1=0.6534 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.2793 lr=5.47158e-05 train_metal_acc=0.9156 val_loss=1.2344 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6037 val_joint_macro_f1=0.6294 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.2801 lr=5.47158e-05 train_metal_acc=0.9273 val_loss=1.2695 val_metal_acc=0.7527 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6127 val_joint_macro_f1=0.6512 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.2650 lr=5.47158e-05 train_metal_acc=0.9214 val_loss=1.2009 val_metal_acc=0.6813 val_metal_min_recall=0.3077 val_fe_recall=0.6061 val_joint_bal_acc=0.6263 val_joint_macro_f1=0.6466 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.2790 lr=5.47158e-05 train_metal_acc=0.9340 val_loss=1.2423 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6278 val_joint_macro_f1=0.6589 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.2322 lr=5.47158e-05 train_metal_acc=0.9331 val_loss=1.3360 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6176 val_joint_macro_f1=0.6608 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.2414 lr=5.47158e-05 train_metal_acc=0.9263 val_loss=1.2852 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.6246 val_joint_macro_f1=0.6444 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.2445 lr=5.47158e-05 train_metal_acc=0.9379 val_loss=1.3822 val_metal_acc=0.7418 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6054 val_joint_macro_f1=0.6500 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.2382 lr=5.47158e-05 train_metal_acc=0.9379 val_loss=1.4006 val_metal_acc=0.7418 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6088 val_joint_macro_f1=0.6546 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.2416 lr=5.47158e-05 train_metal_acc=0.9379 val_loss=1.4031 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.5816 val_joint_macro_f1=0.6311 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.2406 lr=5.47158e-05 train_metal_acc=0.9437 val_loss=1.4152 val_metal_acc=0.7418 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6023 val_joint_macro_f1=0.6407 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.2192 lr=5.47158e-05 train_metal_acc=0.9214 val_loss=1.4049 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6052 val_joint_macro_f1=0.6271 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.2039 lr=5.47158e-05 train_metal_acc=0.9408 val_loss=1.5678 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5814 val_joint_macro_f1=0.6234 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.1937 lr=5.47158e-05 train_metal_acc=0.9408 val_loss=1.5853 val_metal_acc=0.7527 val_metal_min_recall=0.1538 val_fe_recall=0.6667 val_joint_bal_acc=0.6226 val_joint_macro_f1=0.6751 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0032_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_faf01e7c/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0032_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_faf01e7c/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0032_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_faf01e7c/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0032_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_faf01e7c/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0032_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_faf01e7c/run_metadata.json
[I 2026-05-14 05:25:43,319] Trial 32 finished with value: 0.6585119076580177 and parameters: {'learning_rate': 5.4715836015281065e-05, 'weight_decay': 0.001, 'hidden_s': 128, 'head_mlp_layers': 1, 'edge_hidden': 128, 'gvp_layers': 2, 'edge_radius': 6.0, 'hidden_v': 32, 'esm_fusion_dim': 64, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 32 with value: 0.6585119076580177.
Optuna trial 32 completed: val_metal_balanced_acc=0.6585119076580177
================================================================================
[Optuna trial 33] optuna_deepmzyme_controlled_hpo_trial0033_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_62019c53
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 6.0,
  "esm_fusion_dim": 128,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 2,
  "head_mlp_layers": 1,
  "hidden_s": 128,
  "hidden_v": 32,
  "learning_rate": 5.737914983627733e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.001,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0033_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_62019c53 --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 5.737914983627733e-05 --weight-decay 0.001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 128 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7583 lr=5.73791e-05 train_metal_acc=0.5005 val_loss=1.6644 val_metal_acc=0.3681 val_metal_min_recall=0.0000 val_fe_recall=0.5455 val_joint_bal_acc=0.1870 val_joint_macro_f1=0.1406 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.5595 lr=5.73791e-05 train_metal_acc=0.6149 val_loss=1.4606 val_metal_acc=0.3956 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4369 val_joint_macro_f1=0.4150 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.3377 lr=5.73791e-05 train_metal_acc=0.6343 val_loss=1.3079 val_metal_acc=0.6319 val_metal_min_recall=0.0769 val_fe_recall=0.4545 val_joint_bal_acc=0.4234 val_joint_macro_f1=0.4375 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.2079 lr=5.73791e-05 train_metal_acc=0.6974 val_loss=1.2594 val_metal_acc=0.4505 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5308 val_joint_macro_f1=0.4971 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.0543 lr=5.73791e-05 train_metal_acc=0.7216 val_loss=1.1992 val_metal_acc=0.4505 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5130 val_joint_macro_f1=0.5069 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=0.9352 lr=5.73791e-05 train_metal_acc=0.7333 val_loss=1.2160 val_metal_acc=0.4396 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5371 val_joint_macro_f1=0.5061 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=0.8579 lr=5.73791e-05 train_metal_acc=0.7750 val_loss=1.1259 val_metal_acc=0.6429 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5943 val_joint_macro_f1=0.5964 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=0.7796 lr=5.73791e-05 train_metal_acc=0.8225 val_loss=1.0294 val_metal_acc=0.7253 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.5885 val_joint_macro_f1=0.6170 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.7355 lr=5.73791e-05 train_metal_acc=0.8390 val_loss=1.0768 val_metal_acc=0.6648 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.5519 val_joint_macro_f1=0.5909 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.6854 lr=5.73791e-05 train_metal_acc=0.8429 val_loss=1.0528 val_metal_acc=0.6703 val_metal_min_recall=0.1538 val_fe_recall=0.5455 val_joint_bal_acc=0.6013 val_joint_macro_f1=0.6102 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.6162 lr=5.73791e-05 train_metal_acc=0.8652 val_loss=1.0733 val_metal_acc=0.6758 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5917 val_joint_macro_f1=0.6022 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.5814 lr=5.73791e-05 train_metal_acc=0.8729 val_loss=1.1053 val_metal_acc=0.6648 val_metal_min_recall=0.1538 val_fe_recall=0.5455 val_joint_bal_acc=0.6144 val_joint_macro_f1=0.6228 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.5300 lr=5.73791e-05 train_metal_acc=0.8661 val_loss=1.0439 val_metal_acc=0.6813 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.6057 val_joint_macro_f1=0.6092 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.4681 lr=5.73791e-05 train_metal_acc=0.8671 val_loss=1.1463 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5894 val_joint_macro_f1=0.6295 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.4669 lr=5.73791e-05 train_metal_acc=0.8943 val_loss=1.1492 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5818 val_joint_macro_f1=0.5622 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.4458 lr=5.73791e-05 train_metal_acc=0.8943 val_loss=1.1517 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5890 val_joint_macro_f1=0.5853 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.3937 lr=5.73791e-05 train_metal_acc=0.9088 val_loss=1.0965 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6179 val_joint_macro_f1=0.6428 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.3955 lr=5.73791e-05 train_metal_acc=0.9049 val_loss=1.1619 val_metal_acc=0.6758 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.5989 val_joint_macro_f1=0.6187 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.3718 lr=5.73791e-05 train_metal_acc=0.8875 val_loss=1.1257 val_metal_acc=0.6758 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.6248 val_joint_macro_f1=0.6234 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.3478 lr=5.73791e-05 train_metal_acc=0.9079 val_loss=1.2097 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5975 val_joint_macro_f1=0.5638 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.3197 lr=5.73791e-05 train_metal_acc=0.9117 val_loss=1.2047 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6077 val_joint_macro_f1=0.6537 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.3317 lr=5.73791e-05 train_metal_acc=0.9176 val_loss=1.2780 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5949 val_joint_macro_f1=0.6196 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.2896 lr=5.73791e-05 train_metal_acc=0.9273 val_loss=1.2092 val_metal_acc=0.7198 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.5873 val_joint_macro_f1=0.6197 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.2972 lr=5.73791e-05 train_metal_acc=0.9156 val_loss=1.3323 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6136 val_joint_macro_f1=0.6095 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.2771 lr=5.73791e-05 train_metal_acc=0.9205 val_loss=1.2729 val_metal_acc=0.6758 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5958 val_joint_macro_f1=0.6097 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.2647 lr=5.73791e-05 train_metal_acc=0.9185 val_loss=1.3453 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5918 val_joint_macro_f1=0.5994 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.2506 lr=5.73791e-05 train_metal_acc=0.9321 val_loss=1.4297 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5818 val_joint_macro_f1=0.6178 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.2594 lr=5.73791e-05 train_metal_acc=0.9370 val_loss=1.3484 val_metal_acc=0.6319 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5631 val_joint_macro_f1=0.5735 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.2303 lr=5.73791e-05 train_metal_acc=0.9195 val_loss=1.3539 val_metal_acc=0.6319 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5587 val_joint_macro_f1=0.5683 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.2311 lr=5.73791e-05 train_metal_acc=0.9311 val_loss=1.4804 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5783 val_joint_macro_f1=0.5710 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.2170 lr=5.73791e-05 train_metal_acc=0.9399 val_loss=1.4649 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5791 val_joint_macro_f1=0.5955 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.2363 lr=5.73791e-05 train_metal_acc=0.9418 val_loss=1.5488 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5450 val_joint_macro_f1=0.5619 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.2075 lr=5.73791e-05 train_metal_acc=0.9447 val_loss=1.4886 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5768 val_joint_macro_f1=0.5949 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.2391 lr=5.73791e-05 train_metal_acc=0.9389 val_loss=1.5636 val_metal_acc=0.6429 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5561 val_joint_macro_f1=0.5611 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.1852 lr=5.73791e-05 train_metal_acc=0.9467 val_loss=1.7029 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5861 val_joint_macro_f1=0.6081 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.2109 lr=5.73791e-05 train_metal_acc=0.9418 val_loss=1.6315 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5856 val_joint_macro_f1=0.5868 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.2227 lr=5.73791e-05 train_metal_acc=0.9428 val_loss=1.6950 val_metal_acc=0.6264 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5577 val_joint_macro_f1=0.5551 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.2054 lr=5.73791e-05 train_metal_acc=0.9486 val_loss=1.7490 val_metal_acc=0.7033 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5667 val_joint_macro_f1=0.6062 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.2084 lr=5.73791e-05 train_metal_acc=0.9408 val_loss=1.7649 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5289 val_joint_macro_f1=0.5441 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.2003 lr=5.73791e-05 train_metal_acc=0.9457 val_loss=1.8122 val_metal_acc=0.6868 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5363 val_joint_macro_f1=0.5474 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0033_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_62019c53/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0033_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_62019c53/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0033_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_62019c53/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0033_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_62019c53/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0033_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_62019c53/run_metadata.json
[I 2026-05-14 05:34:23,878] Trial 33 finished with value: 0.6248305984548869 and parameters: {'learning_rate': 5.737914983627733e-05, 'weight_decay': 0.001, 'hidden_s': 128, 'head_mlp_layers': 1, 'edge_hidden': 128, 'gvp_layers': 2, 'edge_radius': 6.0, 'hidden_v': 32, 'esm_fusion_dim': 128, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 32 with value: 0.6585119076580177.
Optuna trial 33 completed: val_metal_balanced_acc=0.6248305984548869
================================================================================
[Optuna trial 34] optuna_deepmzyme_controlled_hpo_trial0034_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_ed1da180
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 6.0,
  "esm_fusion_dim": 64,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 2,
  "head_mlp_layers": 1,
  "hidden_s": 128,
  "hidden_v": 16,
  "learning_rate": 5.2098355597118633e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.0,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0034_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_ed1da180 --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 5.2098355597118633e-05 --weight-decay 0.0 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 16 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7724 lr=5.20984e-05 train_metal_acc=0.3608 val_loss=1.6621 val_metal_acc=0.2747 val_metal_min_recall=0.0000 val_fe_recall=0.4242 val_joint_bal_acc=0.2233 val_joint_macro_f1=0.1638 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.6194 lr=5.20984e-05 train_metal_acc=0.5364 val_loss=1.5042 val_metal_acc=0.3846 val_metal_min_recall=0.0000 val_fe_recall=0.6970 val_joint_bal_acc=0.3675 val_joint_macro_f1=0.3317 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.4461 lr=5.20984e-05 train_metal_acc=0.6596 val_loss=1.3380 val_metal_acc=0.7198 val_metal_min_recall=0.0000 val_fe_recall=0.6667 val_joint_bal_acc=0.5249 val_joint_macro_f1=0.5379 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.3025 lr=5.20984e-05 train_metal_acc=0.5742 val_loss=1.3154 val_metal_acc=0.3791 val_metal_min_recall=0.0000 val_fe_recall=0.5455 val_joint_bal_acc=0.4689 val_joint_macro_f1=0.4002 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.1773 lr=5.20984e-05 train_metal_acc=0.7081 val_loss=1.1717 val_metal_acc=0.4505 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4500 val_joint_macro_f1=0.4440 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.0749 lr=5.20984e-05 train_metal_acc=0.7294 val_loss=1.1044 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5170 val_joint_macro_f1=0.5151 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=0.9827 lr=5.20984e-05 train_metal_acc=0.7556 val_loss=1.0674 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.6044 val_joint_macro_f1=0.6211 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=0.9137 lr=5.20984e-05 train_metal_acc=0.7963 val_loss=0.9989 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.6305 val_joint_macro_f1=0.6222 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.8477 lr=5.20984e-05 train_metal_acc=0.8060 val_loss=0.9448 val_metal_acc=0.7418 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.6147 val_joint_macro_f1=0.6301 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.7989 lr=5.20984e-05 train_metal_acc=0.7876 val_loss=1.0341 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5839 val_joint_macro_f1=0.6054 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.7404 lr=5.20984e-05 train_metal_acc=0.8264 val_loss=0.9255 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.6188 val_joint_macro_f1=0.6343 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.7063 lr=5.20984e-05 train_metal_acc=0.8438 val_loss=0.9913 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5828 val_joint_macro_f1=0.6154 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.6500 lr=5.20984e-05 train_metal_acc=0.8041 val_loss=1.0984 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.6145 val_joint_macro_f1=0.5461 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.6355 lr=5.20984e-05 train_metal_acc=0.8371 val_loss=0.9615 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.6114 val_joint_macro_f1=0.6076 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.5859 lr=5.20984e-05 train_metal_acc=0.8729 val_loss=0.9875 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5889 val_joint_macro_f1=0.6025 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.5251 lr=5.20984e-05 train_metal_acc=0.8768 val_loss=0.9820 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.6225 val_joint_macro_f1=0.6092 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.5257 lr=5.20984e-05 train_metal_acc=0.8720 val_loss=1.0283 val_metal_acc=0.7582 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6431 val_joint_macro_f1=0.6345 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.4864 lr=5.20984e-05 train_metal_acc=0.8846 val_loss=0.9866 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5884 val_joint_macro_f1=0.5910 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.4531 lr=5.20984e-05 train_metal_acc=0.8632 val_loss=0.9618 val_metal_acc=0.7527 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6223 val_joint_macro_f1=0.6336 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.4603 lr=5.20984e-05 train_metal_acc=0.8788 val_loss=1.0070 val_metal_acc=0.7692 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6316 val_joint_macro_f1=0.6434 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.4219 lr=5.20984e-05 train_metal_acc=0.8749 val_loss=0.9590 val_metal_acc=0.7418 val_metal_min_recall=0.1538 val_fe_recall=0.5455 val_joint_bal_acc=0.6166 val_joint_macro_f1=0.6310 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.4042 lr=5.20984e-05 train_metal_acc=0.8991 val_loss=1.0850 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.6111 val_joint_macro_f1=0.6134 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.3714 lr=5.20984e-05 train_metal_acc=0.9049 val_loss=1.0695 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.6040 val_joint_macro_f1=0.6141 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.3605 lr=5.20984e-05 train_metal_acc=0.9108 val_loss=1.0713 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5804 val_joint_macro_f1=0.6085 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.3540 lr=5.20984e-05 train_metal_acc=0.9049 val_loss=1.0638 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5958 val_joint_macro_f1=0.5986 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.3551 lr=5.20984e-05 train_metal_acc=0.9195 val_loss=1.1474 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5611 val_joint_macro_f1=0.6015 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.3273 lr=5.20984e-05 train_metal_acc=0.9069 val_loss=1.1558 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5954 val_joint_macro_f1=0.6200 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.3236 lr=5.20984e-05 train_metal_acc=0.9185 val_loss=1.2295 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5922 val_joint_macro_f1=0.6199 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.2856 lr=5.20984e-05 train_metal_acc=0.9108 val_loss=1.1248 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.6970 val_joint_bal_acc=0.5945 val_joint_macro_f1=0.6180 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.2837 lr=5.20984e-05 train_metal_acc=0.9185 val_loss=1.2272 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5630 val_joint_macro_f1=0.5981 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.2645 lr=5.20984e-05 train_metal_acc=0.9282 val_loss=1.1408 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5795 val_joint_macro_f1=0.6050 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.2656 lr=5.20984e-05 train_metal_acc=0.9370 val_loss=1.1838 val_metal_acc=0.7527 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5902 val_joint_macro_f1=0.6172 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.2586 lr=5.20984e-05 train_metal_acc=0.9292 val_loss=1.2454 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6135 val_joint_macro_f1=0.6350 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.2585 lr=5.20984e-05 train_metal_acc=0.9370 val_loss=1.2331 val_metal_acc=0.7637 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.6007 val_joint_macro_f1=0.6277 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.2423 lr=5.20984e-05 train_metal_acc=0.9350 val_loss=1.4124 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5637 val_joint_macro_f1=0.5959 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.2378 lr=5.20984e-05 train_metal_acc=0.9282 val_loss=1.3013 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5991 val_joint_macro_f1=0.6137 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.2302 lr=5.20984e-05 train_metal_acc=0.9408 val_loss=1.2992 val_metal_acc=0.7527 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5902 val_joint_macro_f1=0.6169 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.2237 lr=5.20984e-05 train_metal_acc=0.9399 val_loss=1.2394 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5645 val_joint_macro_f1=0.6058 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.2174 lr=5.20984e-05 train_metal_acc=0.9370 val_loss=1.2580 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5897 val_joint_macro_f1=0.6120 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.2325 lr=5.20984e-05 train_metal_acc=0.9292 val_loss=1.3567 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5912 val_joint_macro_f1=0.6103 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0034_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_ed1da180/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0034_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_ed1da180/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0034_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_ed1da180/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0034_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_ed1da180/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0034_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_ed1da180/run_metadata.json
[I 2026-05-14 05:43:17,782] Trial 34 finished with value: 0.6430567956184655 and parameters: {'learning_rate': 5.2098355597118633e-05, 'weight_decay': 0.0, 'hidden_s': 128, 'head_mlp_layers': 1, 'edge_hidden': 128, 'gvp_layers': 2, 'edge_radius': 6.0, 'hidden_v': 16, 'esm_fusion_dim': 64, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 32 with value: 0.6585119076580177.
Optuna trial 34 completed: val_metal_balanced_acc=0.6430567956184655
================================================================================
[Optuna trial 35] optuna_deepmzyme_controlled_hpo_trial0035_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_ea85b3c9
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 6.0,
  "esm_fusion_dim": 64,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 2,
  "head_mlp_layers": 2,
  "hidden_s": 128,
  "hidden_v": 16,
  "learning_rate": 3.735274364088297e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.001,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0035_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_ea85b3c9 --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 3.735274364088297e-05 --weight-decay 0.001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 64 --head-mlp-layers 2 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 16 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7777 lr=3.73527e-05 train_metal_acc=0.4471 val_loss=1.7638 val_metal_acc=0.2637 val_metal_min_recall=0.0000 val_fe_recall=0.6970 val_joint_bal_acc=0.1652 val_joint_macro_f1=0.1072 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.7361 lr=3.73527e-05 train_metal_acc=0.5655 val_loss=1.6945 val_metal_acc=0.5220 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.3349 val_joint_macro_f1=0.3167 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.6563 lr=3.73527e-05 train_metal_acc=0.5849 val_loss=1.6192 val_metal_acc=0.3956 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.3066 val_joint_macro_f1=0.3064 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.5657 lr=3.73527e-05 train_metal_acc=0.6275 val_loss=1.5324 val_metal_acc=0.4066 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.3951 val_joint_macro_f1=0.4012 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.4613 lr=3.73527e-05 train_metal_acc=0.6838 val_loss=1.4021 val_metal_acc=0.4176 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.3819 val_joint_macro_f1=0.4011 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.3603 lr=3.73527e-05 train_metal_acc=0.7236 val_loss=1.3371 val_metal_acc=0.4670 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.4435 val_joint_macro_f1=0.4634 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=1.2649 lr=3.73527e-05 train_metal_acc=0.7168 val_loss=1.2487 val_metal_acc=0.6813 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4988 val_joint_macro_f1=0.5178 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=1.1833 lr=3.73527e-05 train_metal_acc=0.7556 val_loss=1.2253 val_metal_acc=0.4945 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.4731 val_joint_macro_f1=0.4892 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=1.1096 lr=3.73527e-05 train_metal_acc=0.7323 val_loss=1.1964 val_metal_acc=0.5659 val_metal_min_recall=0.0000 val_fe_recall=0.6667 val_joint_bal_acc=0.5028 val_joint_macro_f1=0.5042 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=1.0262 lr=3.73527e-05 train_metal_acc=0.7662 val_loss=1.2154 val_metal_acc=0.4780 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5479 val_joint_macro_f1=0.4940 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.9645 lr=3.73527e-05 train_metal_acc=0.7730 val_loss=1.1971 val_metal_acc=0.4945 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5416 val_joint_macro_f1=0.5163 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.9026 lr=3.73527e-05 train_metal_acc=0.7856 val_loss=1.1788 val_metal_acc=0.5330 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.4947 val_joint_macro_f1=0.4974 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.8550 lr=3.73527e-05 train_metal_acc=0.7953 val_loss=1.1728 val_metal_acc=0.6429 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5613 val_joint_macro_f1=0.5582 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.7976 lr=3.73527e-05 train_metal_acc=0.8041 val_loss=1.1532 val_metal_acc=0.5604 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5192 val_joint_macro_f1=0.5286 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.7419 lr=3.73527e-05 train_metal_acc=0.8215 val_loss=1.1673 val_metal_acc=0.5549 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5103 val_joint_macro_f1=0.5001 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.7317 lr=3.73527e-05 train_metal_acc=0.8244 val_loss=1.1171 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5272 val_joint_macro_f1=0.5496 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.6846 lr=3.73527e-05 train_metal_acc=0.7983 val_loss=1.2028 val_metal_acc=0.5549 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5312 val_joint_macro_f1=0.5252 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.6395 lr=3.73527e-05 train_metal_acc=0.8322 val_loss=1.1132 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5491 val_joint_macro_f1=0.5638 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.5916 lr=3.73527e-05 train_metal_acc=0.8244 val_loss=1.0947 val_metal_acc=0.6429 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5414 val_joint_macro_f1=0.5478 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.5659 lr=3.73527e-05 train_metal_acc=0.8332 val_loss=1.2273 val_metal_acc=0.5934 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5446 val_joint_macro_f1=0.5419 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.5497 lr=3.73527e-05 train_metal_acc=0.8613 val_loss=1.1699 val_metal_acc=0.6429 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5383 val_joint_macro_f1=0.5585 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.5297 lr=3.73527e-05 train_metal_acc=0.8768 val_loss=1.1771 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5447 val_joint_macro_f1=0.5622 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.5000 lr=3.73527e-05 train_metal_acc=0.8691 val_loss=1.2080 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5616 val_joint_macro_f1=0.5648 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.4803 lr=3.73527e-05 train_metal_acc=0.8855 val_loss=1.1756 val_metal_acc=0.6429 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5414 val_joint_macro_f1=0.5536 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.4501 lr=3.73527e-05 train_metal_acc=0.8797 val_loss=1.2439 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5736 val_joint_macro_f1=0.5465 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.4299 lr=3.73527e-05 train_metal_acc=0.8846 val_loss=1.2864 val_metal_acc=0.6429 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5411 val_joint_macro_f1=0.5572 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.4178 lr=3.73527e-05 train_metal_acc=0.8972 val_loss=1.3031 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5656 val_joint_macro_f1=0.5930 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.3806 lr=3.73527e-05 train_metal_acc=0.8972 val_loss=1.3156 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5391 val_joint_macro_f1=0.5418 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.3792 lr=3.73527e-05 train_metal_acc=0.9001 val_loss=1.2952 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5395 val_joint_macro_f1=0.5489 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.3748 lr=3.73527e-05 train_metal_acc=0.8933 val_loss=1.3532 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5502 val_joint_macro_f1=0.5611 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.3568 lr=3.73527e-05 train_metal_acc=0.9030 val_loss=1.3398 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5567 val_joint_macro_f1=0.5521 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.3463 lr=3.73527e-05 train_metal_acc=0.9146 val_loss=1.3598 val_metal_acc=0.6429 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5411 val_joint_macro_f1=0.5620 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.3097 lr=3.73527e-05 train_metal_acc=0.9137 val_loss=1.4006 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5391 val_joint_macro_f1=0.5519 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.3179 lr=3.73527e-05 train_metal_acc=0.9195 val_loss=1.4435 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5397 val_joint_macro_f1=0.5566 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.3064 lr=3.73527e-05 train_metal_acc=0.9234 val_loss=1.4842 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5431 val_joint_macro_f1=0.5547 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.3003 lr=3.73527e-05 train_metal_acc=0.9273 val_loss=1.5369 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5583 val_joint_macro_f1=0.5833 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.2680 lr=3.73527e-05 train_metal_acc=0.9292 val_loss=1.5969 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5455 val_joint_macro_f1=0.5650 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.2842 lr=3.73527e-05 train_metal_acc=0.9253 val_loss=1.5008 val_metal_acc=0.6264 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5318 val_joint_macro_f1=0.5297 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.2900 lr=3.73527e-05 train_metal_acc=0.9243 val_loss=1.5603 val_metal_acc=0.6319 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5338 val_joint_macro_f1=0.5438 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.2470 lr=3.73527e-05 train_metal_acc=0.9263 val_loss=1.7102 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5612 val_joint_macro_f1=0.5846 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0035_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_ea85b3c9/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0035_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_ea85b3c9/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0035_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_ea85b3c9/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0035_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_ea85b3c9/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0035_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_ea85b3c9/run_metadata.json
[I 2026-05-14 05:52:05,957] Trial 35 finished with value: 0.5735771923627712 and parameters: {'learning_rate': 3.735274364088297e-05, 'weight_decay': 0.001, 'hidden_s': 128, 'head_mlp_layers': 2, 'edge_hidden': 128, 'gvp_layers': 2, 'edge_radius': 6.0, 'hidden_v': 16, 'esm_fusion_dim': 64, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 32 with value: 0.6585119076580177.
Optuna trial 35 completed: val_metal_balanced_acc=0.5735771923627712
================================================================================
[Optuna trial 36] optuna_deepmzyme_controlled_hpo_trial0036_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c81c6ebc
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 8.0,
  "esm_fusion_dim": 128,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 4,
  "head_mlp_layers": 1,
  "hidden_s": 128,
  "hidden_v": 32,
  "learning_rate": 2.8957914121445454e-05,
  "metal_class_weight_mode": "inverse_sqrt_frequency",
  "metal_weighting_setup": "inverse_sqrt_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.0,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0036_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c81c6ebc --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 2.8957914121445454e-05 --weight-decay 0.0 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 128 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_sqrt_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 4 --edge-radius 8.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.6630 lr=2.89579e-05 train_metal_acc=0.5218 val_loss=1.5797 val_metal_acc=0.5330 val_metal_min_recall=0.0000 val_fe_recall=0.3636 val_joint_bal_acc=0.2273 val_joint_macro_f1=0.1893 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.5476 lr=2.89579e-05 train_metal_acc=0.5606 val_loss=1.4924 val_metal_acc=0.4945 val_metal_min_recall=0.0000 val_fe_recall=0.5152 val_joint_bal_acc=0.2290 val_joint_macro_f1=0.1975 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.4225 lr=2.89579e-05 train_metal_acc=0.6343 val_loss=1.3582 val_metal_acc=0.5330 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.3535 val_joint_macro_f1=0.3521 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.2947 lr=2.89579e-05 train_metal_acc=0.6634 val_loss=1.2980 val_metal_acc=0.5385 val_metal_min_recall=0.0000 val_fe_recall=0.5455 val_joint_bal_acc=0.3810 val_joint_macro_f1=0.3947 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.2000 lr=2.89579e-05 train_metal_acc=0.7119 val_loss=1.2720 val_metal_acc=0.4066 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.3538 val_joint_macro_f1=0.3669 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.1241 lr=2.89579e-05 train_metal_acc=0.7371 val_loss=1.2197 val_metal_acc=0.4670 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4467 val_joint_macro_f1=0.4558 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=1.0485 lr=2.89579e-05 train_metal_acc=0.7139 val_loss=1.1992 val_metal_acc=0.6593 val_metal_min_recall=0.0000 val_fe_recall=0.4848 val_joint_bal_acc=0.4275 val_joint_macro_f1=0.4194 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=0.9907 lr=2.89579e-05 train_metal_acc=0.7177 val_loss=1.1185 val_metal_acc=0.6758 val_metal_min_recall=0.0000 val_fe_recall=0.7273 val_joint_bal_acc=0.4629 val_joint_macro_f1=0.4659 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.9220 lr=2.89579e-05 train_metal_acc=0.7827 val_loss=1.1361 val_metal_acc=0.5549 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.5122 val_joint_macro_f1=0.5199 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.8732 lr=2.89579e-05 train_metal_acc=0.7837 val_loss=1.1377 val_metal_acc=0.6044 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5731 val_joint_macro_f1=0.5865 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.8214 lr=2.89579e-05 train_metal_acc=0.8138 val_loss=1.0735 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5593 val_joint_macro_f1=0.5822 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.7731 lr=2.89579e-05 train_metal_acc=0.8177 val_loss=1.0448 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5133 val_joint_macro_f1=0.5436 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.7491 lr=2.89579e-05 train_metal_acc=0.8303 val_loss=1.0342 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5444 val_joint_macro_f1=0.5850 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.6972 lr=2.89579e-05 train_metal_acc=0.8477 val_loss=0.9805 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5510 val_joint_macro_f1=0.5863 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.6588 lr=2.89579e-05 train_metal_acc=0.8419 val_loss=0.9865 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5649 val_joint_macro_f1=0.5894 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.6276 lr=2.89579e-05 train_metal_acc=0.8681 val_loss=0.9790 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5510 val_joint_macro_f1=0.5908 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.5970 lr=2.89579e-05 train_metal_acc=0.8700 val_loss=0.9604 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5765 val_joint_macro_f1=0.6037 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.5748 lr=2.89579e-05 train_metal_acc=0.8691 val_loss=0.9588 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5578 val_joint_macro_f1=0.5904 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.5374 lr=2.89579e-05 train_metal_acc=0.8749 val_loss=1.0104 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5405 val_joint_macro_f1=0.5683 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.5113 lr=2.89579e-05 train_metal_acc=0.8797 val_loss=0.9696 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5527 val_joint_macro_f1=0.5841 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.4980 lr=2.89579e-05 train_metal_acc=0.8855 val_loss=0.9722 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5356 val_joint_macro_f1=0.5757 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.4728 lr=2.89579e-05 train_metal_acc=0.8972 val_loss=0.9741 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5682 val_joint_macro_f1=0.5968 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.4619 lr=2.89579e-05 train_metal_acc=0.8904 val_loss=1.0516 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5439 val_joint_macro_f1=0.5806 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.4424 lr=2.89579e-05 train_metal_acc=0.9011 val_loss=0.9800 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5708 val_joint_macro_f1=0.6046 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.4105 lr=2.89579e-05 train_metal_acc=0.8991 val_loss=1.0256 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5950 val_joint_macro_f1=0.6024 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.4047 lr=2.89579e-05 train_metal_acc=0.9108 val_loss=1.0658 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5590 val_joint_macro_f1=0.5895 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.3893 lr=2.89579e-05 train_metal_acc=0.9011 val_loss=1.0929 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5769 val_joint_macro_f1=0.6069 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.3720 lr=2.89579e-05 train_metal_acc=0.9011 val_loss=1.1798 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5334 val_joint_macro_f1=0.5849 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.3570 lr=2.89579e-05 train_metal_acc=0.8972 val_loss=1.2141 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5251 val_joint_macro_f1=0.5728 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.3481 lr=2.89579e-05 train_metal_acc=0.9234 val_loss=1.0729 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5655 val_joint_macro_f1=0.6048 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.3348 lr=2.89579e-05 train_metal_acc=0.9195 val_loss=1.0880 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5410 val_joint_macro_f1=0.5824 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.3074 lr=2.89579e-05 train_metal_acc=0.9253 val_loss=1.0849 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5601 val_joint_macro_f1=0.5824 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.2978 lr=2.89579e-05 train_metal_acc=0.9273 val_loss=1.1551 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5570 val_joint_macro_f1=0.5918 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.2957 lr=2.89579e-05 train_metal_acc=0.9282 val_loss=1.1495 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5570 val_joint_macro_f1=0.5918 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.2862 lr=2.89579e-05 train_metal_acc=0.9302 val_loss=1.1662 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5363 val_joint_macro_f1=0.5822 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.2820 lr=2.89579e-05 train_metal_acc=0.9273 val_loss=1.1391 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5728 val_joint_macro_f1=0.5817 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.2731 lr=2.89579e-05 train_metal_acc=0.9321 val_loss=1.1227 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5594 val_joint_macro_f1=0.5784 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.2502 lr=2.89579e-05 train_metal_acc=0.9389 val_loss=1.2571 val_metal_acc=0.6978 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5440 val_joint_macro_f1=0.5847 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.2513 lr=2.89579e-05 train_metal_acc=0.9370 val_loss=1.2025 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5598 val_joint_macro_f1=0.5924 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.2446 lr=2.89579e-05 train_metal_acc=0.9399 val_loss=1.2988 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5510 val_joint_macro_f1=0.6013 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0036_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c81c6ebc/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0036_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c81c6ebc/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0036_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c81c6ebc/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0036_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c81c6ebc/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0036_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c81c6ebc/run_metadata.json
[I 2026-05-14 06:02:37,927] Trial 36 finished with value: 0.5949780137635925 and parameters: {'learning_rate': 2.8957914121445454e-05, 'weight_decay': 0.0, 'hidden_s': 128, 'head_mlp_layers': 1, 'edge_hidden': 128, 'gvp_layers': 4, 'edge_radius': 8.0, 'hidden_v': 32, 'esm_fusion_dim': 128, 'metal_class_weight_mode': 'inverse_sqrt_frequency'}. Best is trial 32 with value: 0.6585119076580177.
Optuna trial 36 completed: val_metal_balanced_acc=0.5949780137635925
================================================================================
[Optuna trial 37] optuna_deepmzyme_controlled_hpo_trial0037_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_19d4f091
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 6.0,
  "esm_fusion_dim": 64,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 2,
  "head_mlp_layers": 2,
  "hidden_s": 256,
  "hidden_v": 16,
  "learning_rate": 6.662606527595876e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 1e-05,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0037_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_19d4f091 --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 6.662606527595876e-05 --weight-decay 1e-05 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 256 --esm-fusion-dim 64 --head-mlp-layers 2 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 16 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7696 lr=6.66261e-05 train_metal_acc=0.4976 val_loss=1.6892 val_metal_acc=0.5440 val_metal_min_recall=0.0000 val_fe_recall=0.4242 val_joint_bal_acc=0.2659 val_joint_macro_f1=0.2485 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.5849 lr=6.66261e-05 train_metal_acc=0.6431 val_loss=1.4186 val_metal_acc=0.4011 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.3648 val_joint_macro_f1=0.3481 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.3247 lr=6.66261e-05 train_metal_acc=0.6586 val_loss=1.2929 val_metal_acc=0.4176 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4703 val_joint_macro_f1=0.4483 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.1532 lr=6.66261e-05 train_metal_acc=0.6954 val_loss=1.2292 val_metal_acc=0.4286 val_metal_min_recall=0.0000 val_fe_recall=0.5455 val_joint_bal_acc=0.5152 val_joint_macro_f1=0.4975 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.0199 lr=6.66261e-05 train_metal_acc=0.7391 val_loss=1.1522 val_metal_acc=0.4615 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4898 val_joint_macro_f1=0.4725 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=0.9059 lr=6.66261e-05 train_metal_acc=0.7595 val_loss=1.1436 val_metal_acc=0.5989 val_metal_min_recall=0.0000 val_fe_recall=0.5455 val_joint_bal_acc=0.5794 val_joint_macro_f1=0.5592 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=0.8116 lr=6.66261e-05 train_metal_acc=0.7876 val_loss=1.0768 val_metal_acc=0.6923 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.6005 val_joint_macro_f1=0.5996 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=0.7563 lr=6.66261e-05 train_metal_acc=0.8041 val_loss=1.0524 val_metal_acc=0.6758 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.6070 val_joint_macro_f1=0.6217 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.6600 lr=6.66261e-05 train_metal_acc=0.8147 val_loss=1.0825 val_metal_acc=0.7637 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.6525 val_joint_macro_f1=0.6454 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.6143 lr=6.66261e-05 train_metal_acc=0.8157 val_loss=1.0680 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6001 val_joint_macro_f1=0.5822 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.5640 lr=6.66261e-05 train_metal_acc=0.8681 val_loss=1.0898 val_metal_acc=0.7527 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.6196 val_joint_macro_f1=0.6357 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.5021 lr=6.66261e-05 train_metal_acc=0.8332 val_loss=1.1953 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6066 val_joint_macro_f1=0.5783 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.4859 lr=6.66261e-05 train_metal_acc=0.8749 val_loss=1.1784 val_metal_acc=0.7527 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.6036 val_joint_macro_f1=0.6307 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.4704 lr=6.66261e-05 train_metal_acc=0.8729 val_loss=1.2704 val_metal_acc=0.6978 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.6229 val_joint_macro_f1=0.6266 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.4315 lr=6.66261e-05 train_metal_acc=0.8885 val_loss=1.2448 val_metal_acc=0.7637 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6365 val_joint_macro_f1=0.6390 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.3912 lr=6.66261e-05 train_metal_acc=0.8943 val_loss=1.3003 val_metal_acc=0.7692 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6388 val_joint_macro_f1=0.6360 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.3667 lr=6.66261e-05 train_metal_acc=0.9011 val_loss=1.2908 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6153 val_joint_macro_f1=0.6350 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.3667 lr=6.66261e-05 train_metal_acc=0.8914 val_loss=1.4108 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6970 val_joint_bal_acc=0.6236 val_joint_macro_f1=0.6229 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.3275 lr=6.66261e-05 train_metal_acc=0.9195 val_loss=1.4227 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5932 val_joint_macro_f1=0.6159 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.3253 lr=6.66261e-05 train_metal_acc=0.9040 val_loss=1.4282 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6063 val_joint_macro_f1=0.6161 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.3035 lr=6.66261e-05 train_metal_acc=0.9166 val_loss=1.5657 val_metal_acc=0.7582 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.6055 val_joint_macro_f1=0.6272 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.2929 lr=6.66261e-05 train_metal_acc=0.9205 val_loss=1.7845 val_metal_acc=0.7747 val_metal_min_recall=0.0769 val_fe_recall=0.6970 val_joint_bal_acc=0.6213 val_joint_macro_f1=0.6540 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.3076 lr=6.66261e-05 train_metal_acc=0.9292 val_loss=1.6427 val_metal_acc=0.7582 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6093 val_joint_macro_f1=0.6336 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.2756 lr=6.66261e-05 train_metal_acc=0.9214 val_loss=1.7568 val_metal_acc=0.6813 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5794 val_joint_macro_f1=0.5900 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.2765 lr=6.66261e-05 train_metal_acc=0.9040 val_loss=2.0130 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.6027 val_joint_macro_f1=0.6005 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.2691 lr=6.66261e-05 train_metal_acc=0.9127 val_loss=1.6677 val_metal_acc=0.7637 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6368 val_joint_macro_f1=0.6328 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.2820 lr=6.66261e-05 train_metal_acc=0.9311 val_loss=1.8284 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6051 val_joint_macro_f1=0.6364 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.2514 lr=6.66261e-05 train_metal_acc=0.9370 val_loss=1.9249 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5954 val_joint_macro_f1=0.6064 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.2595 lr=6.66261e-05 train_metal_acc=0.9428 val_loss=2.0719 val_metal_acc=0.7582 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5908 val_joint_macro_f1=0.6135 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.2765 lr=6.66261e-05 train_metal_acc=0.9302 val_loss=2.4531 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6970 val_joint_bal_acc=0.5341 val_joint_macro_f1=0.5677 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.2528 lr=6.66261e-05 train_metal_acc=0.9389 val_loss=2.0856 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5881 val_joint_macro_f1=0.6073 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.2235 lr=6.66261e-05 train_metal_acc=0.9360 val_loss=2.4608 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5932 val_joint_macro_f1=0.6317 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.2593 lr=6.66261e-05 train_metal_acc=0.9389 val_loss=2.0645 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.6275 val_joint_macro_f1=0.6340 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.2627 lr=6.66261e-05 train_metal_acc=0.9418 val_loss=2.4752 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5563 val_joint_macro_f1=0.6038 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.2552 lr=6.66261e-05 train_metal_acc=0.9467 val_loss=2.3181 val_metal_acc=0.6868 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5807 val_joint_macro_f1=0.5881 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.2637 lr=6.66261e-05 train_metal_acc=0.9418 val_loss=2.2384 val_metal_acc=0.6813 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5794 val_joint_macro_f1=0.5748 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.2596 lr=6.66261e-05 train_metal_acc=0.9505 val_loss=2.2963 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5564 val_joint_macro_f1=0.5830 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.2378 lr=6.66261e-05 train_metal_acc=0.9408 val_loss=2.3915 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5977 val_joint_macro_f1=0.6187 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.2243 lr=6.66261e-05 train_metal_acc=0.9467 val_loss=2.5991 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5779 val_joint_macro_f1=0.6133 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.2120 lr=6.66261e-05 train_metal_acc=0.9467 val_loss=2.4042 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5669 val_joint_macro_f1=0.6026 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0037_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_19d4f091/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0037_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_19d4f091/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0037_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_19d4f091/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0037_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_19d4f091/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0037_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_19d4f091/run_metadata.json
[I 2026-05-14 06:11:21,665] Trial 37 finished with value: 0.6525003775478159 and parameters: {'learning_rate': 6.662606527595876e-05, 'weight_decay': 1e-05, 'hidden_s': 256, 'head_mlp_layers': 2, 'edge_hidden': 128, 'gvp_layers': 2, 'edge_radius': 6.0, 'hidden_v': 16, 'esm_fusion_dim': 64, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 32 with value: 0.6585119076580177.
Optuna trial 37 completed: val_metal_balanced_acc=0.6525003775478159
================================================================================
[Optuna trial 38] optuna_deepmzyme_controlled_hpo_trial0038_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_803373b1
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 6.0,
  "esm_fusion_dim": 64,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 4,
  "head_mlp_layers": 1,
  "hidden_s": 128,
  "hidden_v": 32,
  "learning_rate": 4.606682080690019e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.001,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0038_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_803373b1 --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 4.606682080690019e-05 --weight-decay 0.001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 4 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7694 lr=4.60668e-05 train_metal_acc=0.4433 val_loss=1.7011 val_metal_acc=0.3626 val_metal_min_recall=0.0000 val_fe_recall=0.0909 val_joint_bal_acc=0.2029 val_joint_macro_f1=0.1685 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.6751 lr=4.60668e-05 train_metal_acc=0.5684 val_loss=1.5761 val_metal_acc=0.4286 val_metal_min_recall=0.0000 val_fe_recall=0.3636 val_joint_bal_acc=0.3861 val_joint_macro_f1=0.3803 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.5304 lr=4.60668e-05 train_metal_acc=0.6052 val_loss=1.4554 val_metal_acc=0.4451 val_metal_min_recall=0.0000 val_fe_recall=0.4545 val_joint_bal_acc=0.4039 val_joint_macro_f1=0.3999 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.3907 lr=4.60668e-05 train_metal_acc=0.6421 val_loss=1.4177 val_metal_acc=0.4066 val_metal_min_recall=0.0000 val_fe_recall=0.5152 val_joint_bal_acc=0.4572 val_joint_macro_f1=0.4259 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.2754 lr=4.60668e-05 train_metal_acc=0.6809 val_loss=1.3835 val_metal_acc=0.4066 val_metal_min_recall=0.1429 val_fe_recall=0.5152 val_joint_bal_acc=0.4352 val_joint_macro_f1=0.4151 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.1828 lr=4.60668e-05 train_metal_acc=0.7042 val_loss=1.3546 val_metal_acc=0.4231 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.4841 val_joint_macro_f1=0.4582 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=1.0944 lr=4.60668e-05 train_metal_acc=0.6712 val_loss=1.4116 val_metal_acc=0.3956 val_metal_min_recall=0.0769 val_fe_recall=0.4545 val_joint_bal_acc=0.5084 val_joint_macro_f1=0.4530 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=1.0157 lr=4.60668e-05 train_metal_acc=0.7808 val_loss=1.2901 val_metal_acc=0.4451 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.4899 val_joint_macro_f1=0.4647 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.9299 lr=4.60668e-05 train_metal_acc=0.7682 val_loss=1.2864 val_metal_acc=0.4835 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5334 val_joint_macro_f1=0.5211 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.8888 lr=4.60668e-05 train_metal_acc=0.7730 val_loss=1.3400 val_metal_acc=0.4396 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5204 val_joint_macro_f1=0.4610 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.8287 lr=4.60668e-05 train_metal_acc=0.7682 val_loss=1.3899 val_metal_acc=0.4615 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5450 val_joint_macro_f1=0.5072 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.7772 lr=4.60668e-05 train_metal_acc=0.8128 val_loss=1.2441 val_metal_acc=0.5110 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5206 val_joint_macro_f1=0.5122 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.7300 lr=4.60668e-05 train_metal_acc=0.8235 val_loss=1.2168 val_metal_acc=0.5934 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5497 val_joint_macro_f1=0.5371 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.6839 lr=4.60668e-05 train_metal_acc=0.8497 val_loss=1.1683 val_metal_acc=0.6154 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5285 val_joint_macro_f1=0.5418 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.6485 lr=4.60668e-05 train_metal_acc=0.8409 val_loss=1.2018 val_metal_acc=0.6264 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5615 val_joint_macro_f1=0.5558 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.6001 lr=4.60668e-05 train_metal_acc=0.8458 val_loss=1.1755 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5616 val_joint_macro_f1=0.5682 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.5885 lr=4.60668e-05 train_metal_acc=0.8206 val_loss=1.3276 val_metal_acc=0.5385 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5294 val_joint_macro_f1=0.4721 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.5499 lr=4.60668e-05 train_metal_acc=0.8681 val_loss=1.1893 val_metal_acc=0.6319 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5413 val_joint_macro_f1=0.5514 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.5188 lr=4.60668e-05 train_metal_acc=0.8710 val_loss=1.2141 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5586 val_joint_macro_f1=0.5625 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.4802 lr=4.60668e-05 train_metal_acc=0.8652 val_loss=1.1977 val_metal_acc=0.6429 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5739 val_joint_macro_f1=0.5646 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.4659 lr=4.60668e-05 train_metal_acc=0.8846 val_loss=1.2033 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5331 val_joint_macro_f1=0.5437 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.4533 lr=4.60668e-05 train_metal_acc=0.8797 val_loss=1.2697 val_metal_acc=0.6209 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5691 val_joint_macro_f1=0.5440 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.4224 lr=4.60668e-05 train_metal_acc=0.8923 val_loss=1.2119 val_metal_acc=0.6264 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5287 val_joint_macro_f1=0.5447 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.4063 lr=4.60668e-05 train_metal_acc=0.9040 val_loss=1.2283 val_metal_acc=0.6429 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5411 val_joint_macro_f1=0.5497 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.3837 lr=4.60668e-05 train_metal_acc=0.8797 val_loss=1.3185 val_metal_acc=0.6319 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5703 val_joint_macro_f1=0.5426 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.3660 lr=4.60668e-05 train_metal_acc=0.9117 val_loss=1.2751 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5470 val_joint_macro_f1=0.5641 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.3549 lr=4.60668e-05 train_metal_acc=0.8875 val_loss=1.3729 val_metal_acc=0.6209 val_metal_min_recall=0.0769 val_fe_recall=0.4545 val_joint_bal_acc=0.5533 val_joint_macro_f1=0.5084 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.3355 lr=4.60668e-05 train_metal_acc=0.8991 val_loss=1.2916 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5665 val_joint_macro_f1=0.5648 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.3125 lr=4.60668e-05 train_metal_acc=0.9127 val_loss=1.2408 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5870 val_joint_macro_f1=0.5959 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.3059 lr=4.60668e-05 train_metal_acc=0.9156 val_loss=1.3135 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5690 val_joint_macro_f1=0.5698 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.3045 lr=4.60668e-05 train_metal_acc=0.9253 val_loss=1.3341 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5723 val_joint_macro_f1=0.5718 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.2894 lr=4.60668e-05 train_metal_acc=0.9214 val_loss=1.3449 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5805 val_joint_macro_f1=0.5853 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.2717 lr=4.60668e-05 train_metal_acc=0.9292 val_loss=1.3692 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5614 val_joint_macro_f1=0.5833 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.2481 lr=4.60668e-05 train_metal_acc=0.9273 val_loss=1.4110 val_metal_acc=0.6429 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5561 val_joint_macro_f1=0.5552 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.2539 lr=4.60668e-05 train_metal_acc=0.9263 val_loss=1.3660 val_metal_acc=0.7033 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.5923 val_joint_macro_f1=0.6001 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.2544 lr=4.60668e-05 train_metal_acc=0.9340 val_loss=1.4068 val_metal_acc=0.6813 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5477 val_joint_macro_f1=0.5748 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.2440 lr=4.60668e-05 train_metal_acc=0.9399 val_loss=1.4073 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5555 val_joint_macro_f1=0.5691 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.2387 lr=4.60668e-05 train_metal_acc=0.9234 val_loss=1.4074 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5649 val_joint_macro_f1=0.5581 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.2273 lr=4.60668e-05 train_metal_acc=0.9360 val_loss=1.5328 val_metal_acc=0.6978 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5505 val_joint_macro_f1=0.5616 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.2266 lr=4.60668e-05 train_metal_acc=0.9340 val_loss=1.5511 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5593 val_joint_macro_f1=0.5763 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0038_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_803373b1/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0038_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_803373b1/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0038_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_803373b1/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0038_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_803373b1/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0038_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_803373b1/run_metadata.json
[I 2026-05-14 06:21:03,242] Trial 38 finished with value: 0.5922924218939399 and parameters: {'learning_rate': 4.606682080690019e-05, 'weight_decay': 0.001, 'hidden_s': 128, 'head_mlp_layers': 1, 'edge_hidden': 128, 'gvp_layers': 4, 'edge_radius': 6.0, 'hidden_v': 32, 'esm_fusion_dim': 64, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 32 with value: 0.6585119076580177.
Optuna trial 38 completed: val_metal_balanced_acc=0.5922924218939399
================================================================================
[Optuna trial 39] optuna_deepmzyme_controlled_hpo_trial0039_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a1d013d0
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 64,
  "edge_radius": 8.0,
  "esm_fusion_dim": 128,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 2,
  "head_mlp_layers": 1,
  "hidden_s": 256,
  "hidden_v": 32,
  "learning_rate": 5.785908967671049e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.0,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0039_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a1d013d0 --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 5.785908967671049e-05 --weight-decay 0.0 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 256 --esm-fusion-dim 128 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 64 --gvp-layers 2 --edge-radius 8.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7788 lr=5.78591e-05 train_metal_acc=0.5325 val_loss=1.6081 val_metal_acc=0.5385 val_metal_min_recall=0.0000 val_fe_recall=0.7273 val_joint_bal_acc=0.3532 val_joint_macro_f1=0.3226 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.4837 lr=5.78591e-05 train_metal_acc=0.6052 val_loss=1.3953 val_metal_acc=0.3297 val_metal_min_recall=0.0000 val_fe_recall=0.5455 val_joint_bal_acc=0.4560 val_joint_macro_f1=0.3688 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.2311 lr=5.78591e-05 train_metal_acc=0.6557 val_loss=1.2837 val_metal_acc=0.4011 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4252 val_joint_macro_f1=0.4075 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.0814 lr=5.78591e-05 train_metal_acc=0.7342 val_loss=1.2125 val_metal_acc=0.4505 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5162 val_joint_macro_f1=0.5077 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=0.9270 lr=5.78591e-05 train_metal_acc=0.7507 val_loss=1.2147 val_metal_acc=0.5055 val_metal_min_recall=0.1538 val_fe_recall=0.4848 val_joint_bal_acc=0.5090 val_joint_macro_f1=0.5128 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=0.8392 lr=5.78591e-05 train_metal_acc=0.7759 val_loss=1.1873 val_metal_acc=0.6099 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5655 val_joint_macro_f1=0.5485 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=0.7553 lr=5.78591e-05 train_metal_acc=0.7643 val_loss=1.2176 val_metal_acc=0.4725 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5369 val_joint_macro_f1=0.5198 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=0.6792 lr=5.78591e-05 train_metal_acc=0.8225 val_loss=1.0830 val_metal_acc=0.6868 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5985 val_joint_macro_f1=0.6017 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.6266 lr=5.78591e-05 train_metal_acc=0.8380 val_loss=1.1770 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5628 val_joint_macro_f1=0.5673 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.5684 lr=5.78591e-05 train_metal_acc=0.8594 val_loss=1.2170 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.6111 val_joint_macro_f1=0.6258 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.5080 lr=5.78591e-05 train_metal_acc=0.8681 val_loss=1.1904 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5837 val_joint_macro_f1=0.5773 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.4794 lr=5.78591e-05 train_metal_acc=0.8584 val_loss=1.1895 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5960 val_joint_macro_f1=0.5951 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.4316 lr=5.78591e-05 train_metal_acc=0.8972 val_loss=1.2374 val_metal_acc=0.6923 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5622 val_joint_macro_f1=0.5815 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.4194 lr=5.78591e-05 train_metal_acc=0.9040 val_loss=1.2466 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5632 val_joint_macro_f1=0.5821 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.3728 lr=5.78591e-05 train_metal_acc=0.8982 val_loss=1.2216 val_metal_acc=0.6978 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5843 val_joint_macro_f1=0.5930 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.3610 lr=5.78591e-05 train_metal_acc=0.8952 val_loss=1.2870 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5734 val_joint_macro_f1=0.5796 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.3524 lr=5.78591e-05 train_metal_acc=0.9001 val_loss=1.3483 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5913 val_joint_macro_f1=0.5810 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.3246 lr=5.78591e-05 train_metal_acc=0.9156 val_loss=1.3074 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6101 val_joint_macro_f1=0.6297 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.3002 lr=5.78591e-05 train_metal_acc=0.9273 val_loss=1.4337 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5795 val_joint_macro_f1=0.6134 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.2713 lr=5.78591e-05 train_metal_acc=0.9108 val_loss=1.2493 val_metal_acc=0.7143 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.5942 val_joint_macro_f1=0.5901 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.2529 lr=5.78591e-05 train_metal_acc=0.9137 val_loss=1.4074 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6086 val_joint_macro_f1=0.6009 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.2581 lr=5.78591e-05 train_metal_acc=0.9273 val_loss=1.5885 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5960 val_joint_macro_f1=0.6258 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.2581 lr=5.78591e-05 train_metal_acc=0.9263 val_loss=1.5975 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5743 val_joint_macro_f1=0.5738 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.2384 lr=5.78591e-05 train_metal_acc=0.9350 val_loss=1.7365 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5691 val_joint_macro_f1=0.6041 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.2133 lr=5.78591e-05 train_metal_acc=0.9389 val_loss=1.7215 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5855 val_joint_macro_f1=0.6089 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.2261 lr=5.78591e-05 train_metal_acc=0.9331 val_loss=1.4600 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5541 val_joint_macro_f1=0.5731 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.2298 lr=5.78591e-05 train_metal_acc=0.9282 val_loss=1.5574 val_metal_acc=0.6923 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5977 val_joint_macro_f1=0.5928 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.2170 lr=5.78591e-05 train_metal_acc=0.9418 val_loss=1.7206 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5681 val_joint_macro_f1=0.5919 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.2131 lr=5.78591e-05 train_metal_acc=0.9311 val_loss=1.6909 val_metal_acc=0.6923 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5567 val_joint_macro_f1=0.5597 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.2060 lr=5.78591e-05 train_metal_acc=0.9117 val_loss=1.7011 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5884 val_joint_macro_f1=0.5821 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.1871 lr=5.78591e-05 train_metal_acc=0.9505 val_loss=1.7125 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5810 val_joint_macro_f1=0.5961 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.1994 lr=5.78591e-05 train_metal_acc=0.9476 val_loss=2.0914 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5664 val_joint_macro_f1=0.6039 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.2057 lr=5.78591e-05 train_metal_acc=0.9525 val_loss=2.0307 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5617 val_joint_macro_f1=0.5837 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.1892 lr=5.78591e-05 train_metal_acc=0.9496 val_loss=1.9725 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5855 val_joint_macro_f1=0.6159 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.1984 lr=5.78591e-05 train_metal_acc=0.9486 val_loss=1.9610 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5802 val_joint_macro_f1=0.5862 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.1705 lr=5.78591e-05 train_metal_acc=0.9505 val_loss=2.2550 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5611 val_joint_macro_f1=0.5880 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.1848 lr=5.78591e-05 train_metal_acc=0.9583 val_loss=2.2648 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5722 val_joint_macro_f1=0.6128 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.1811 lr=5.78591e-05 train_metal_acc=0.9554 val_loss=2.1637 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5430 val_joint_macro_f1=0.5703 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.1838 lr=5.78591e-05 train_metal_acc=0.9515 val_loss=2.3020 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5436 val_joint_macro_f1=0.5785 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.1833 lr=5.78591e-05 train_metal_acc=0.9651 val_loss=2.1958 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5436 val_joint_macro_f1=0.5725 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0039_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a1d013d0/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0039_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a1d013d0/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0039_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a1d013d0/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0039_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a1d013d0/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0039_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a1d013d0/run_metadata.json
[I 2026-05-14 06:30:35,106] Trial 39 finished with value: 0.611107046021657 and parameters: {'learning_rate': 5.785908967671049e-05, 'weight_decay': 0.0, 'hidden_s': 256, 'head_mlp_layers': 1, 'edge_hidden': 64, 'gvp_layers': 2, 'edge_radius': 8.0, 'hidden_v': 32, 'esm_fusion_dim': 128, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 32 with value: 0.6585119076580177.
Optuna trial 39 completed: val_metal_balanced_acc=0.611107046021657
================================================================================
[Optuna trial 40] optuna_deepmzyme_controlled_hpo_trial0040_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_aed7a382
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 6.0,
  "esm_fusion_dim": 64,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 4,
  "head_mlp_layers": 2,
  "hidden_s": 128,
  "hidden_v": 16,
  "learning_rate": 8.631087594909479e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 1e-05,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0040_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_aed7a382 --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 8.631087594909479e-05 --weight-decay 1e-05 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 64 --head-mlp-layers 2 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 4 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 16 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7696 lr=8.63109e-05 train_metal_acc=0.5383 val_loss=1.6982 val_metal_acc=0.3571 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.3100 val_joint_macro_f1=0.3013 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.6286 lr=8.63109e-05 train_metal_acc=0.6314 val_loss=1.5230 val_metal_acc=0.4286 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4473 val_joint_macro_f1=0.4435 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.3906 lr=8.63109e-05 train_metal_acc=0.6809 val_loss=1.3442 val_metal_acc=0.4396 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4177 val_joint_macro_f1=0.3944 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.1821 lr=8.63109e-05 train_metal_acc=0.7304 val_loss=1.2481 val_metal_acc=0.4505 val_metal_min_recall=0.0000 val_fe_recall=0.5152 val_joint_bal_acc=0.4728 val_joint_macro_f1=0.4630 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.0215 lr=8.63109e-05 train_metal_acc=0.7730 val_loss=1.1540 val_metal_acc=0.6319 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5047 val_joint_macro_f1=0.4973 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=0.8849 lr=8.63109e-05 train_metal_acc=0.7924 val_loss=1.1159 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5432 val_joint_macro_f1=0.5494 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=0.8235 lr=8.63109e-05 train_metal_acc=0.7866 val_loss=1.0708 val_metal_acc=0.6374 val_metal_min_recall=0.1538 val_fe_recall=0.4848 val_joint_bal_acc=0.5650 val_joint_macro_f1=0.5717 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=0.7525 lr=8.63109e-05 train_metal_acc=0.8264 val_loss=1.0809 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5678 val_joint_macro_f1=0.5903 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.6437 lr=8.63109e-05 train_metal_acc=0.8303 val_loss=1.1067 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5670 val_joint_macro_f1=0.5710 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.6155 lr=8.63109e-05 train_metal_acc=0.8225 val_loss=1.0883 val_metal_acc=0.6703 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6041 val_joint_macro_f1=0.6076 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.5712 lr=8.63109e-05 train_metal_acc=0.8661 val_loss=1.2068 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5924 val_joint_macro_f1=0.5786 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.5319 lr=8.63109e-05 train_metal_acc=0.8826 val_loss=1.1252 val_metal_acc=0.7582 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.6150 val_joint_macro_f1=0.6300 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.4716 lr=8.63109e-05 train_metal_acc=0.8982 val_loss=1.1582 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5748 val_joint_macro_f1=0.5957 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.4477 lr=8.63109e-05 train_metal_acc=0.8962 val_loss=1.2648 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5736 val_joint_macro_f1=0.5883 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.4131 lr=8.63109e-05 train_metal_acc=0.8885 val_loss=1.2528 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6048 val_joint_macro_f1=0.6188 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.3922 lr=8.63109e-05 train_metal_acc=0.8952 val_loss=1.3832 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5701 val_joint_macro_f1=0.5760 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.3677 lr=8.63109e-05 train_metal_acc=0.9049 val_loss=1.4778 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5754 val_joint_macro_f1=0.5940 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.3629 lr=8.63109e-05 train_metal_acc=0.9098 val_loss=1.4045 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5653 val_joint_macro_f1=0.5865 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.3412 lr=8.63109e-05 train_metal_acc=0.9166 val_loss=1.3882 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5845 val_joint_macro_f1=0.6034 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.3425 lr=8.63109e-05 train_metal_acc=0.9079 val_loss=1.4759 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5681 val_joint_macro_f1=0.5867 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.2998 lr=8.63109e-05 train_metal_acc=0.9234 val_loss=1.4627 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5798 val_joint_macro_f1=0.6044 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.2997 lr=8.63109e-05 train_metal_acc=0.9282 val_loss=1.5251 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6062 val_joint_macro_f1=0.6076 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.3010 lr=8.63109e-05 train_metal_acc=0.9224 val_loss=1.6920 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5921 val_joint_macro_f1=0.5829 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.3160 lr=8.63109e-05 train_metal_acc=0.9360 val_loss=1.9181 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5537 val_joint_macro_f1=0.5860 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.2523 lr=8.63109e-05 train_metal_acc=0.9389 val_loss=1.9688 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5687 val_joint_macro_f1=0.6016 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.3186 lr=8.63109e-05 train_metal_acc=0.9350 val_loss=1.7658 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6091 val_joint_macro_f1=0.6131 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.2757 lr=8.63109e-05 train_metal_acc=0.9331 val_loss=2.0197 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5611 val_joint_macro_f1=0.5979 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.2684 lr=8.63109e-05 train_metal_acc=0.9408 val_loss=2.0674 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5798 val_joint_macro_f1=0.6033 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.2823 lr=8.63109e-05 train_metal_acc=0.9457 val_loss=1.9980 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5886 val_joint_macro_f1=0.6013 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.2142 lr=8.63109e-05 train_metal_acc=0.9350 val_loss=2.1098 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5852 val_joint_macro_f1=0.6156 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.2414 lr=8.63109e-05 train_metal_acc=0.9340 val_loss=2.0221 val_metal_acc=0.6484 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5792 val_joint_macro_f1=0.5747 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.2476 lr=8.63109e-05 train_metal_acc=0.9399 val_loss=2.2830 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5829 val_joint_macro_f1=0.6117 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.2627 lr=8.63109e-05 train_metal_acc=0.9331 val_loss=2.1370 val_metal_acc=0.6923 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5588 val_joint_macro_f1=0.5583 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.2890 lr=8.63109e-05 train_metal_acc=0.9399 val_loss=2.0838 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5467 val_joint_macro_f1=0.5827 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.2542 lr=8.63109e-05 train_metal_acc=0.9486 val_loss=2.2481 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5480 val_joint_macro_f1=0.5817 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.2785 lr=8.63109e-05 train_metal_acc=0.9428 val_loss=2.3862 val_metal_acc=0.7198 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5658 val_joint_macro_f1=0.6024 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.2575 lr=8.63109e-05 train_metal_acc=0.9389 val_loss=2.4158 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5953 val_joint_macro_f1=0.6077 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.2260 lr=8.63109e-05 train_metal_acc=0.9408 val_loss=2.4657 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5764 val_joint_macro_f1=0.6001 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.2758 lr=8.63109e-05 train_metal_acc=0.9505 val_loss=2.3865 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5839 val_joint_macro_f1=0.6049 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.2408 lr=8.63109e-05 train_metal_acc=0.9515 val_loss=2.4818 val_metal_acc=0.7527 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5868 val_joint_macro_f1=0.6259 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0040_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_aed7a382/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0040_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_aed7a382/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0040_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_aed7a382/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0040_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_aed7a382/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0040_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_aed7a382/run_metadata.json
[I 2026-05-14 06:40:21,317] Trial 40 finished with value: 0.615028614649108 and parameters: {'learning_rate': 8.631087594909479e-05, 'weight_decay': 1e-05, 'hidden_s': 128, 'head_mlp_layers': 2, 'edge_hidden': 128, 'gvp_layers': 4, 'edge_radius': 6.0, 'hidden_v': 16, 'esm_fusion_dim': 64, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 32 with value: 0.6585119076580177.
Optuna trial 40 completed: val_metal_balanced_acc=0.615028614649108
================================================================================
[Optuna trial 41] optuna_deepmzyme_controlled_hpo_trial0041_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_e1dec6d2
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 6.0,
  "esm_fusion_dim": 64,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 2,
  "head_mlp_layers": 1,
  "hidden_s": 128,
  "hidden_v": 32,
  "learning_rate": 7.388763169038574e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.001,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0041_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_e1dec6d2 --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 7.388763169038574e-05 --weight-decay 0.001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7524 lr=7.38876e-05 train_metal_acc=0.5354 val_loss=1.5771 val_metal_acc=0.5714 val_metal_min_recall=0.0000 val_fe_recall=0.3939 val_joint_bal_acc=0.3744 val_joint_macro_f1=0.3569 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.5147 lr=7.38876e-05 train_metal_acc=0.6518 val_loss=1.3652 val_metal_acc=0.6648 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4770 val_joint_macro_f1=0.4979 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.3063 lr=7.38876e-05 train_metal_acc=0.6906 val_loss=1.2664 val_metal_acc=0.6429 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.5128 val_joint_macro_f1=0.5209 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.1691 lr=7.38876e-05 train_metal_acc=0.6489 val_loss=1.2876 val_metal_acc=0.4176 val_metal_min_recall=0.0000 val_fe_recall=0.5455 val_joint_bal_acc=0.4710 val_joint_macro_f1=0.4374 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.0426 lr=7.38876e-05 train_metal_acc=0.7624 val_loss=1.1413 val_metal_acc=0.6868 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4787 val_joint_macro_f1=0.4887 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=0.9560 lr=7.38876e-05 train_metal_acc=0.7662 val_loss=1.2236 val_metal_acc=0.4560 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5218 val_joint_macro_f1=0.5106 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=0.8596 lr=7.38876e-05 train_metal_acc=0.7847 val_loss=1.1808 val_metal_acc=0.5165 val_metal_min_recall=0.1538 val_fe_recall=0.4848 val_joint_bal_acc=0.5621 val_joint_macro_f1=0.5458 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=0.7833 lr=7.38876e-05 train_metal_acc=0.8186 val_loss=1.1428 val_metal_acc=0.5934 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.5702 val_joint_macro_f1=0.5913 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.7415 lr=7.38876e-05 train_metal_acc=0.8041 val_loss=1.1298 val_metal_acc=0.6209 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5806 val_joint_macro_f1=0.5872 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.6799 lr=7.38876e-05 train_metal_acc=0.8429 val_loss=1.0615 val_metal_acc=0.6538 val_metal_min_recall=0.1538 val_fe_recall=0.4848 val_joint_bal_acc=0.5790 val_joint_macro_f1=0.5836 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.6095 lr=7.38876e-05 train_metal_acc=0.8535 val_loss=1.0764 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5958 val_joint_macro_f1=0.6205 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.5867 lr=7.38876e-05 train_metal_acc=0.8526 val_loss=1.0216 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.6047 val_joint_macro_f1=0.6167 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.5432 lr=7.38876e-05 train_metal_acc=0.8720 val_loss=1.0344 val_metal_acc=0.6868 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5973 val_joint_macro_f1=0.5898 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.4863 lr=7.38876e-05 train_metal_acc=0.8797 val_loss=1.0788 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.6019 val_joint_macro_f1=0.6160 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.4473 lr=7.38876e-05 train_metal_acc=0.8855 val_loss=1.0253 val_metal_acc=0.6813 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6111 val_joint_macro_f1=0.6245 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.4348 lr=7.38876e-05 train_metal_acc=0.8855 val_loss=1.2107 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5628 val_joint_macro_f1=0.5765 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.4203 lr=7.38876e-05 train_metal_acc=0.8933 val_loss=1.1005 val_metal_acc=0.7033 val_metal_min_recall=0.1538 val_fe_recall=0.6970 val_joint_bal_acc=0.6286 val_joint_macro_f1=0.6628 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.4009 lr=7.38876e-05 train_metal_acc=0.9001 val_loss=1.0691 val_metal_acc=0.7473 val_metal_min_recall=0.2308 val_fe_recall=0.6667 val_joint_bal_acc=0.6452 val_joint_macro_f1=0.6803 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.3604 lr=7.38876e-05 train_metal_acc=0.9088 val_loss=1.0694 val_metal_acc=0.7088 val_metal_min_recall=0.2308 val_fe_recall=0.6364 val_joint_bal_acc=0.6284 val_joint_macro_f1=0.6408 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.3383 lr=7.38876e-05 train_metal_acc=0.9108 val_loss=1.1554 val_metal_acc=0.7198 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.6118 val_joint_macro_f1=0.6385 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.3360 lr=7.38876e-05 train_metal_acc=0.9146 val_loss=1.1933 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.6030 val_joint_macro_f1=0.6271 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.3022 lr=7.38876e-05 train_metal_acc=0.9049 val_loss=1.0754 val_metal_acc=0.7637 val_metal_min_recall=0.3077 val_fe_recall=0.6667 val_joint_bal_acc=0.6517 val_joint_macro_f1=0.6705 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.2832 lr=7.38876e-05 train_metal_acc=0.9127 val_loss=1.2780 val_metal_acc=0.7143 val_metal_min_recall=0.2308 val_fe_recall=0.6970 val_joint_bal_acc=0.5908 val_joint_macro_f1=0.6388 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.2901 lr=7.38876e-05 train_metal_acc=0.9224 val_loss=1.1942 val_metal_acc=0.7363 val_metal_min_recall=0.3077 val_fe_recall=0.6364 val_joint_bal_acc=0.6524 val_joint_macro_f1=0.6757 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.2672 lr=7.38876e-05 train_metal_acc=0.9340 val_loss=1.3658 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5847 val_joint_macro_f1=0.6142 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.2491 lr=7.38876e-05 train_metal_acc=0.9331 val_loss=1.4224 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6077 val_joint_macro_f1=0.6411 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.2504 lr=7.38876e-05 train_metal_acc=0.9340 val_loss=1.4906 val_metal_acc=0.7418 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6122 val_joint_macro_f1=0.6500 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.2258 lr=7.38876e-05 train_metal_acc=0.9214 val_loss=1.3716 val_metal_acc=0.7308 val_metal_min_recall=0.2308 val_fe_recall=0.6061 val_joint_bal_acc=0.6126 val_joint_macro_f1=0.6389 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.2332 lr=7.38876e-05 train_metal_acc=0.9408 val_loss=1.4973 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.5980 val_joint_macro_f1=0.6321 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.2190 lr=7.38876e-05 train_metal_acc=0.9253 val_loss=1.3619 val_metal_acc=0.7143 val_metal_min_recall=0.3077 val_fe_recall=0.6061 val_joint_bal_acc=0.6381 val_joint_macro_f1=0.6553 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.2454 lr=7.38876e-05 train_metal_acc=0.9437 val_loss=1.4787 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6210 val_joint_macro_f1=0.6632 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.1896 lr=7.38876e-05 train_metal_acc=0.9379 val_loss=1.5497 val_metal_acc=0.7418 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6088 val_joint_macro_f1=0.6541 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.2031 lr=7.38876e-05 train_metal_acc=0.9370 val_loss=1.5274 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6120 val_joint_macro_f1=0.6372 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.2133 lr=7.38876e-05 train_metal_acc=0.9447 val_loss=1.6932 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.5946 val_joint_macro_f1=0.6399 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.2177 lr=7.38876e-05 train_metal_acc=0.9476 val_loss=1.7465 val_metal_acc=0.7418 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5989 val_joint_macro_f1=0.6393 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.2275 lr=7.38876e-05 train_metal_acc=0.9457 val_loss=1.7049 val_metal_acc=0.7143 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5638 val_joint_macro_f1=0.6115 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.2063 lr=7.38876e-05 train_metal_acc=0.9476 val_loss=1.7549 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5949 val_joint_macro_f1=0.6342 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.1966 lr=7.38876e-05 train_metal_acc=0.9292 val_loss=1.6645 val_metal_acc=0.7088 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.5868 val_joint_macro_f1=0.6018 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.1789 lr=7.38876e-05 train_metal_acc=0.9486 val_loss=1.8999 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6003 val_joint_macro_f1=0.6403 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.1748 lr=7.38876e-05 train_metal_acc=0.9486 val_loss=1.8312 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6176 val_joint_macro_f1=0.6608 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0041_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_e1dec6d2/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0041_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_e1dec6d2/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0041_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_e1dec6d2/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0041_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_e1dec6d2/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0041_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_e1dec6d2/run_metadata.json
[I 2026-05-14 06:49:03,766] Trial 41 finished with value: 0.6524493848972027 and parameters: {'learning_rate': 7.388763169038574e-05, 'weight_decay': 0.001, 'hidden_s': 128, 'head_mlp_layers': 1, 'edge_hidden': 128, 'gvp_layers': 2, 'edge_radius': 6.0, 'hidden_v': 32, 'esm_fusion_dim': 64, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 32 with value: 0.6585119076580177.
Optuna trial 41 completed: val_metal_balanced_acc=0.6524493848972027
================================================================================
[Optuna trial 42] optuna_deepmzyme_controlled_hpo_trial0042_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_8e2c2aa2
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 6.0,
  "esm_fusion_dim": 64,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 2,
  "head_mlp_layers": 1,
  "hidden_s": 128,
  "hidden_v": 32,
  "learning_rate": 8.662875976318268e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.001,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0042_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_8e2c2aa2 --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 8.662875976318268e-05 --weight-decay 0.001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7413 lr=8.66288e-05 train_metal_acc=0.5441 val_loss=1.5394 val_metal_acc=0.5769 val_metal_min_recall=0.0000 val_fe_recall=0.4242 val_joint_bal_acc=0.3795 val_joint_macro_f1=0.3544 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.4737 lr=8.66288e-05 train_metal_acc=0.6683 val_loss=1.3316 val_metal_acc=0.6648 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4736 val_joint_macro_f1=0.4909 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.2535 lr=8.66288e-05 train_metal_acc=0.7051 val_loss=1.2422 val_metal_acc=0.5659 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4895 val_joint_macro_f1=0.4858 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.1121 lr=8.66288e-05 train_metal_acc=0.6673 val_loss=1.2925 val_metal_acc=0.4231 val_metal_min_recall=0.0000 val_fe_recall=0.5455 val_joint_bal_acc=0.4764 val_joint_macro_f1=0.4412 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=0.9829 lr=8.66288e-05 train_metal_acc=0.7692 val_loss=1.1277 val_metal_acc=0.6868 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4889 val_joint_macro_f1=0.4992 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=0.8967 lr=8.66288e-05 train_metal_acc=0.7789 val_loss=1.1957 val_metal_acc=0.4615 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5019 val_joint_macro_f1=0.5032 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=0.7991 lr=8.66288e-05 train_metal_acc=0.7963 val_loss=1.1701 val_metal_acc=0.5604 val_metal_min_recall=0.1538 val_fe_recall=0.4848 val_joint_bal_acc=0.5594 val_joint_macro_f1=0.5497 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=0.7231 lr=8.66288e-05 train_metal_acc=0.8332 val_loss=1.1470 val_metal_acc=0.6154 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5879 val_joint_macro_f1=0.6075 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.6858 lr=8.66288e-05 train_metal_acc=0.8312 val_loss=1.1171 val_metal_acc=0.6538 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5924 val_joint_macro_f1=0.6004 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.6232 lr=8.66288e-05 train_metal_acc=0.8535 val_loss=1.0734 val_metal_acc=0.6484 val_metal_min_recall=0.1538 val_fe_recall=0.4848 val_joint_bal_acc=0.5771 val_joint_macro_f1=0.5796 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.5535 lr=8.66288e-05 train_metal_acc=0.8681 val_loss=1.0956 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5723 val_joint_macro_f1=0.5917 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.5379 lr=8.66288e-05 train_metal_acc=0.8661 val_loss=1.0305 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.5455 val_joint_bal_acc=0.6078 val_joint_macro_f1=0.6186 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.4903 lr=8.66288e-05 train_metal_acc=0.8826 val_loss=1.0612 val_metal_acc=0.7143 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.6133 val_joint_macro_f1=0.6158 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.4402 lr=8.66288e-05 train_metal_acc=0.8962 val_loss=1.1127 val_metal_acc=0.7582 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.6215 val_joint_macro_f1=0.6369 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.4005 lr=8.66288e-05 train_metal_acc=0.9001 val_loss=1.0747 val_metal_acc=0.6923 val_metal_min_recall=0.1538 val_fe_recall=0.6970 val_joint_bal_acc=0.6178 val_joint_macro_f1=0.6420 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.3935 lr=8.66288e-05 train_metal_acc=0.8933 val_loss=1.2371 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5624 val_joint_macro_f1=0.5850 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.3792 lr=8.66288e-05 train_metal_acc=0.9098 val_loss=1.1438 val_metal_acc=0.7198 val_metal_min_recall=0.1538 val_fe_recall=0.6970 val_joint_bal_acc=0.6242 val_joint_macro_f1=0.6628 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.3640 lr=8.66288e-05 train_metal_acc=0.9059 val_loss=1.1120 val_metal_acc=0.7473 val_metal_min_recall=0.3077 val_fe_recall=0.6364 val_joint_bal_acc=0.6530 val_joint_macro_f1=0.6847 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.3243 lr=8.66288e-05 train_metal_acc=0.9156 val_loss=1.1207 val_metal_acc=0.7198 val_metal_min_recall=0.2308 val_fe_recall=0.6364 val_joint_bal_acc=0.6323 val_joint_macro_f1=0.6412 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.3072 lr=8.66288e-05 train_metal_acc=0.9185 val_loss=1.2247 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6247 val_joint_macro_f1=0.6542 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.3055 lr=8.66288e-05 train_metal_acc=0.9234 val_loss=1.2926 val_metal_acc=0.7527 val_metal_min_recall=0.1538 val_fe_recall=0.6667 val_joint_bal_acc=0.6158 val_joint_macro_f1=0.6528 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.2705 lr=8.66288e-05 train_metal_acc=0.9117 val_loss=1.1468 val_metal_acc=0.7363 val_metal_min_recall=0.3077 val_fe_recall=0.6970 val_joint_bal_acc=0.6245 val_joint_macro_f1=0.6455 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.2580 lr=8.66288e-05 train_metal_acc=0.9137 val_loss=1.3865 val_metal_acc=0.7143 val_metal_min_recall=0.2308 val_fe_recall=0.6970 val_joint_bal_acc=0.5839 val_joint_macro_f1=0.6320 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.2714 lr=8.66288e-05 train_metal_acc=0.9292 val_loss=1.2727 val_metal_acc=0.7088 val_metal_min_recall=0.3077 val_fe_recall=0.6061 val_joint_bal_acc=0.6293 val_joint_macro_f1=0.6562 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.2443 lr=8.66288e-05 train_metal_acc=0.9457 val_loss=1.4903 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5886 val_joint_macro_f1=0.6195 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.2295 lr=8.66288e-05 train_metal_acc=0.9350 val_loss=1.5352 val_metal_acc=0.7418 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6057 val_joint_macro_f1=0.6376 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.2382 lr=8.66288e-05 train_metal_acc=0.9437 val_loss=1.6453 val_metal_acc=0.7418 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6054 val_joint_macro_f1=0.6447 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.2104 lr=8.66288e-05 train_metal_acc=0.9302 val_loss=1.4408 val_metal_acc=0.7363 val_metal_min_recall=0.2308 val_fe_recall=0.6061 val_joint_bal_acc=0.6146 val_joint_macro_f1=0.6390 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.2211 lr=8.66288e-05 train_metal_acc=0.9447 val_loss=1.6175 val_metal_acc=0.7198 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5876 val_joint_macro_f1=0.6172 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.2038 lr=8.66288e-05 train_metal_acc=0.9321 val_loss=1.4699 val_metal_acc=0.7033 val_metal_min_recall=0.3077 val_fe_recall=0.6061 val_joint_bal_acc=0.6273 val_joint_macro_f1=0.6421 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.2339 lr=8.66288e-05 train_metal_acc=0.9476 val_loss=1.6247 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6179 val_joint_macro_f1=0.6536 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.1765 lr=8.66288e-05 train_metal_acc=0.9467 val_loss=1.6810 val_metal_acc=0.7143 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5856 val_joint_macro_f1=0.6193 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.1935 lr=8.66288e-05 train_metal_acc=0.9457 val_loss=1.6692 val_metal_acc=0.7253 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6066 val_joint_macro_f1=0.6461 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.2024 lr=8.66288e-05 train_metal_acc=0.9418 val_loss=1.8259 val_metal_acc=0.7253 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5896 val_joint_macro_f1=0.6274 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.2144 lr=8.66288e-05 train_metal_acc=0.9505 val_loss=1.9657 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5881 val_joint_macro_f1=0.6280 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.2250 lr=8.66288e-05 train_metal_acc=0.9515 val_loss=1.8749 val_metal_acc=0.7088 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5618 val_joint_macro_f1=0.6086 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.1939 lr=8.66288e-05 train_metal_acc=0.9525 val_loss=1.9228 val_metal_acc=0.7253 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5896 val_joint_macro_f1=0.6273 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.1868 lr=8.66288e-05 train_metal_acc=0.9467 val_loss=1.8506 val_metal_acc=0.7088 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5837 val_joint_macro_f1=0.5972 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.1690 lr=8.66288e-05 train_metal_acc=0.9534 val_loss=2.0367 val_metal_acc=0.7418 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6057 val_joint_macro_f1=0.6448 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.1629 lr=8.66288e-05 train_metal_acc=0.9554 val_loss=1.9375 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.5830 val_joint_macro_f1=0.6261 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0042_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_8e2c2aa2/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0042_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_8e2c2aa2/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0042_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_8e2c2aa2/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0042_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_8e2c2aa2/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0042_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_8e2c2aa2/run_metadata.json
[I 2026-05-14 06:57:47,581] Trial 42 finished with value: 0.6529553937523577 and parameters: {'learning_rate': 8.662875976318268e-05, 'weight_decay': 0.001, 'hidden_s': 128, 'head_mlp_layers': 1, 'edge_hidden': 128, 'gvp_layers': 2, 'edge_radius': 6.0, 'hidden_v': 32, 'esm_fusion_dim': 64, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 32 with value: 0.6585119076580177.
Optuna trial 42 completed: val_metal_balanced_acc=0.6529553937523577
================================================================================
[Optuna trial 43] optuna_deepmzyme_controlled_hpo_trial0043_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_8d03d6e9
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 6.0,
  "esm_fusion_dim": 64,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 2,
  "head_mlp_layers": 1,
  "hidden_s": 128,
  "hidden_v": 32,
  "learning_rate": 6.649236220481763e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.001,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0043_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_8d03d6e9 --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 6.649236220481763e-05 --weight-decay 0.001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7586 lr=6.64924e-05 train_metal_acc=0.5286 val_loss=1.6005 val_metal_acc=0.5714 val_metal_min_recall=0.0000 val_fe_recall=0.3939 val_joint_bal_acc=0.3744 val_joint_macro_f1=0.3578 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.5413 lr=6.64924e-05 train_metal_acc=0.6295 val_loss=1.3891 val_metal_acc=0.6538 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4666 val_joint_macro_f1=0.4869 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.3411 lr=6.64924e-05 train_metal_acc=0.6731 val_loss=1.2867 val_metal_acc=0.6429 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4910 val_joint_macro_f1=0.4954 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.2066 lr=6.64924e-05 train_metal_acc=0.6363 val_loss=1.2882 val_metal_acc=0.4176 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4703 val_joint_macro_f1=0.4472 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.0827 lr=6.64924e-05 train_metal_acc=0.7527 val_loss=1.1528 val_metal_acc=0.6758 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4676 val_joint_macro_f1=0.4774 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=0.9968 lr=6.64924e-05 train_metal_acc=0.7565 val_loss=1.2337 val_metal_acc=0.4560 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5252 val_joint_macro_f1=0.5105 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=0.9014 lr=6.64924e-05 train_metal_acc=0.7692 val_loss=1.1867 val_metal_acc=0.4890 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5415 val_joint_macro_f1=0.5084 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=0.8250 lr=6.64924e-05 train_metal_acc=0.8031 val_loss=1.1475 val_metal_acc=0.5440 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.5525 val_joint_macro_f1=0.5710 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.7811 lr=6.64924e-05 train_metal_acc=0.7953 val_loss=1.1430 val_metal_acc=0.5934 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5776 val_joint_macro_f1=0.5858 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.7207 lr=6.64924e-05 train_metal_acc=0.8341 val_loss=1.0617 val_metal_acc=0.6484 val_metal_min_recall=0.1538 val_fe_recall=0.4848 val_joint_bal_acc=0.5737 val_joint_macro_f1=0.5753 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.6509 lr=6.64924e-05 train_metal_acc=0.8400 val_loss=1.0668 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5938 val_joint_macro_f1=0.6143 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.6234 lr=6.64924e-05 train_metal_acc=0.8477 val_loss=1.0172 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5919 val_joint_macro_f1=0.5919 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.5813 lr=6.64924e-05 train_metal_acc=0.8632 val_loss=1.0209 val_metal_acc=0.6813 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5953 val_joint_macro_f1=0.5862 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.5219 lr=6.64924e-05 train_metal_acc=0.8691 val_loss=1.0629 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5866 val_joint_macro_f1=0.5965 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.4830 lr=6.64924e-05 train_metal_acc=0.8758 val_loss=1.0010 val_metal_acc=0.6758 val_metal_min_recall=0.2308 val_fe_recall=0.6061 val_joint_bal_acc=0.6135 val_joint_macro_f1=0.6310 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.4666 lr=6.64924e-05 train_metal_acc=0.8739 val_loss=1.1843 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5866 val_joint_macro_f1=0.5964 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.4509 lr=6.64924e-05 train_metal_acc=0.8788 val_loss=1.0847 val_metal_acc=0.6978 val_metal_min_recall=0.1538 val_fe_recall=0.6667 val_joint_bal_acc=0.6235 val_joint_macro_f1=0.6583 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.4303 lr=6.64924e-05 train_metal_acc=0.8991 val_loss=1.0430 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6219 val_joint_macro_f1=0.6534 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.3881 lr=6.64924e-05 train_metal_acc=0.9049 val_loss=1.0452 val_metal_acc=0.6758 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5992 val_joint_macro_f1=0.6037 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.3645 lr=6.64924e-05 train_metal_acc=0.9030 val_loss=1.1182 val_metal_acc=0.7253 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.6206 val_joint_macro_f1=0.6442 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.3611 lr=6.64924e-05 train_metal_acc=0.9108 val_loss=1.1405 val_metal_acc=0.7802 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.6352 val_joint_macro_f1=0.6507 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.3285 lr=6.64924e-05 train_metal_acc=0.9020 val_loss=1.0444 val_metal_acc=0.7637 val_metal_min_recall=0.3077 val_fe_recall=0.6667 val_joint_bal_acc=0.6551 val_joint_macro_f1=0.6700 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.3065 lr=6.64924e-05 train_metal_acc=0.9098 val_loss=1.2122 val_metal_acc=0.7363 val_metal_min_recall=0.2308 val_fe_recall=0.6970 val_joint_bal_acc=0.6157 val_joint_macro_f1=0.6596 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.3088 lr=6.64924e-05 train_metal_acc=0.9195 val_loss=1.1561 val_metal_acc=0.7143 val_metal_min_recall=0.3077 val_fe_recall=0.6364 val_joint_bal_acc=0.6446 val_joint_macro_f1=0.6708 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.2877 lr=6.64924e-05 train_metal_acc=0.9243 val_loss=1.2885 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5858 val_joint_macro_f1=0.6129 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.2668 lr=6.64924e-05 train_metal_acc=0.9253 val_loss=1.3337 val_metal_acc=0.7527 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6127 val_joint_macro_f1=0.6455 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.2642 lr=6.64924e-05 train_metal_acc=0.9292 val_loss=1.3903 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6137 val_joint_macro_f1=0.6571 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.2401 lr=6.64924e-05 train_metal_acc=0.9185 val_loss=1.3157 val_metal_acc=0.7253 val_metal_min_recall=0.2308 val_fe_recall=0.6061 val_joint_bal_acc=0.6107 val_joint_macro_f1=0.6364 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.2455 lr=6.64924e-05 train_metal_acc=0.9370 val_loss=1.4167 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6000 val_joint_macro_f1=0.6361 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.2313 lr=6.64924e-05 train_metal_acc=0.9273 val_loss=1.2967 val_metal_acc=0.7088 val_metal_min_recall=0.3077 val_fe_recall=0.6061 val_joint_bal_acc=0.6361 val_joint_macro_f1=0.6528 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.2556 lr=6.64924e-05 train_metal_acc=0.9428 val_loss=1.3835 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6244 val_joint_macro_f1=0.6569 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.2023 lr=6.64924e-05 train_metal_acc=0.9350 val_loss=1.4675 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6142 val_joint_macro_f1=0.6590 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.2130 lr=6.64924e-05 train_metal_acc=0.9311 val_loss=1.4325 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6174 val_joint_macro_f1=0.6412 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.2208 lr=6.64924e-05 train_metal_acc=0.9418 val_loss=1.5809 val_metal_acc=0.7253 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.5708 val_joint_macro_f1=0.6190 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.2230 lr=6.64924e-05 train_metal_acc=0.9428 val_loss=1.6014 val_metal_acc=0.7527 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6096 val_joint_macro_f1=0.6501 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.2309 lr=6.64924e-05 train_metal_acc=0.9418 val_loss=1.5843 val_metal_acc=0.7198 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5692 val_joint_macro_f1=0.6113 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.2150 lr=6.64924e-05 train_metal_acc=0.9486 val_loss=1.6337 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5969 val_joint_macro_f1=0.6367 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.2043 lr=6.64924e-05 train_metal_acc=0.9292 val_loss=1.5747 val_metal_acc=0.7088 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5871 val_joint_macro_f1=0.6026 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.1838 lr=6.64924e-05 train_metal_acc=0.9428 val_loss=1.7963 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6077 val_joint_macro_f1=0.6488 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.1811 lr=6.64924e-05 train_metal_acc=0.9476 val_loss=1.7356 val_metal_acc=0.7527 val_metal_min_recall=0.1538 val_fe_recall=0.6667 val_joint_bal_acc=0.6226 val_joint_macro_f1=0.6745 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0043_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_8d03d6e9/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0043_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_8d03d6e9/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0043_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_8d03d6e9/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0043_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_8d03d6e9/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0043_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_8d03d6e9/run_metadata.json
[I 2026-05-14 07:06:37,730] Trial 43 finished with value: 0.6550963478857217 and parameters: {'learning_rate': 6.649236220481763e-05, 'weight_decay': 0.001, 'hidden_s': 128, 'head_mlp_layers': 1, 'edge_hidden': 128, 'gvp_layers': 2, 'edge_radius': 6.0, 'hidden_v': 32, 'esm_fusion_dim': 64, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 32 with value: 0.6585119076580177.
Optuna trial 43 completed: val_metal_balanced_acc=0.6550963478857217
================================================================================
[Optuna trial 44] optuna_deepmzyme_controlled_hpo_trial0044_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_2fd12175
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 6.0,
  "esm_fusion_dim": 64,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 2,
  "head_mlp_layers": 1,
  "hidden_s": 128,
  "hidden_v": 32,
  "learning_rate": 6.853351991201396e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.001,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0044_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_2fd12175 --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 6.853351991201396e-05 --weight-decay 0.001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7569 lr=6.85335e-05 train_metal_acc=0.5315 val_loss=1.5940 val_metal_acc=0.5714 val_metal_min_recall=0.0000 val_fe_recall=0.3939 val_joint_bal_acc=0.3744 val_joint_macro_f1=0.3570 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.5337 lr=6.85335e-05 train_metal_acc=0.6343 val_loss=1.3821 val_metal_acc=0.6593 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4719 val_joint_macro_f1=0.4920 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.3312 lr=6.85335e-05 train_metal_acc=0.6799 val_loss=1.2806 val_metal_acc=0.6429 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4910 val_joint_macro_f1=0.4954 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.1959 lr=6.85335e-05 train_metal_acc=0.6411 val_loss=1.2876 val_metal_acc=0.4121 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4653 val_joint_macro_f1=0.4444 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.0712 lr=6.85335e-05 train_metal_acc=0.7585 val_loss=1.1493 val_metal_acc=0.6758 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4679 val_joint_macro_f1=0.4792 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=0.9850 lr=6.85335e-05 train_metal_acc=0.7565 val_loss=1.2314 val_metal_acc=0.4560 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5252 val_joint_macro_f1=0.5105 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=0.8893 lr=6.85335e-05 train_metal_acc=0.7711 val_loss=1.1851 val_metal_acc=0.4945 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5468 val_joint_macro_f1=0.5111 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=0.8129 lr=6.85335e-05 train_metal_acc=0.8050 val_loss=1.1457 val_metal_acc=0.5549 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.5564 val_joint_macro_f1=0.5756 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.7696 lr=6.85335e-05 train_metal_acc=0.7953 val_loss=1.1389 val_metal_acc=0.5934 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5742 val_joint_macro_f1=0.5830 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.7088 lr=6.85335e-05 train_metal_acc=0.8390 val_loss=1.0609 val_metal_acc=0.6484 val_metal_min_recall=0.1538 val_fe_recall=0.4848 val_joint_bal_acc=0.5737 val_joint_macro_f1=0.5767 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.6387 lr=6.85335e-05 train_metal_acc=0.8448 val_loss=1.0692 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5938 val_joint_macro_f1=0.6143 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.6126 lr=6.85335e-05 train_metal_acc=0.8497 val_loss=1.0184 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.6047 val_joint_macro_f1=0.6167 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.5701 lr=6.85335e-05 train_metal_acc=0.8642 val_loss=1.0243 val_metal_acc=0.6813 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5953 val_joint_macro_f1=0.5862 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.5113 lr=6.85335e-05 train_metal_acc=0.8700 val_loss=1.0669 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5885 val_joint_macro_f1=0.5987 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.4724 lr=6.85335e-05 train_metal_acc=0.8807 val_loss=1.0074 val_metal_acc=0.6813 val_metal_min_recall=0.2308 val_fe_recall=0.6061 val_joint_bal_acc=0.6189 val_joint_macro_f1=0.6344 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.4572 lr=6.85335e-05 train_metal_acc=0.8778 val_loss=1.1929 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5866 val_joint_macro_f1=0.5965 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.4419 lr=6.85335e-05 train_metal_acc=0.8826 val_loss=1.0885 val_metal_acc=0.6978 val_metal_min_recall=0.1538 val_fe_recall=0.6667 val_joint_bal_acc=0.6235 val_joint_macro_f1=0.6583 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.4215 lr=6.85335e-05 train_metal_acc=0.9001 val_loss=1.0506 val_metal_acc=0.7418 val_metal_min_recall=0.2308 val_fe_recall=0.6667 val_joint_bal_acc=0.6398 val_joint_macro_f1=0.6774 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.3798 lr=6.85335e-05 train_metal_acc=0.9069 val_loss=1.0515 val_metal_acc=0.6813 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6046 val_joint_macro_f1=0.6073 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.3566 lr=6.85335e-05 train_metal_acc=0.9049 val_loss=1.1280 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.6226 val_joint_macro_f1=0.6465 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.3535 lr=6.85335e-05 train_metal_acc=0.9127 val_loss=1.1548 val_metal_acc=0.7692 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.6245 val_joint_macro_f1=0.6460 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.3207 lr=6.85335e-05 train_metal_acc=0.9030 val_loss=1.0524 val_metal_acc=0.7637 val_metal_min_recall=0.3077 val_fe_recall=0.6667 val_joint_bal_acc=0.6551 val_joint_macro_f1=0.6699 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.2993 lr=6.85335e-05 train_metal_acc=0.9108 val_loss=1.2301 val_metal_acc=0.7363 val_metal_min_recall=0.2308 val_fe_recall=0.6970 val_joint_bal_acc=0.6157 val_joint_macro_f1=0.6596 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.3031 lr=6.85335e-05 train_metal_acc=0.9214 val_loss=1.1661 val_metal_acc=0.7253 val_metal_min_recall=0.3077 val_fe_recall=0.6364 val_joint_bal_acc=0.6485 val_joint_macro_f1=0.6756 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.2813 lr=6.85335e-05 train_metal_acc=0.9273 val_loss=1.3096 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5878 val_joint_macro_f1=0.6230 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.2613 lr=6.85335e-05 train_metal_acc=0.9311 val_loss=1.3596 val_metal_acc=0.7527 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6127 val_joint_macro_f1=0.6521 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.2598 lr=6.85335e-05 train_metal_acc=0.9302 val_loss=1.4179 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6083 val_joint_macro_f1=0.6529 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.2354 lr=6.85335e-05 train_metal_acc=0.9195 val_loss=1.3317 val_metal_acc=0.7253 val_metal_min_recall=0.2308 val_fe_recall=0.6061 val_joint_bal_acc=0.6107 val_joint_macro_f1=0.6364 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.2415 lr=6.85335e-05 train_metal_acc=0.9370 val_loss=1.4397 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.5980 val_joint_macro_f1=0.6332 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.2274 lr=6.85335e-05 train_metal_acc=0.9263 val_loss=1.3145 val_metal_acc=0.7088 val_metal_min_recall=0.3077 val_fe_recall=0.6061 val_joint_bal_acc=0.6361 val_joint_macro_f1=0.6528 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.2528 lr=6.85335e-05 train_metal_acc=0.9437 val_loss=1.4096 val_metal_acc=0.7527 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6264 val_joint_macro_f1=0.6673 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.1987 lr=6.85335e-05 train_metal_acc=0.9360 val_loss=1.4903 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6142 val_joint_macro_f1=0.6590 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.2102 lr=6.85335e-05 train_metal_acc=0.9321 val_loss=1.4602 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6120 val_joint_macro_f1=0.6372 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.2183 lr=6.85335e-05 train_metal_acc=0.9418 val_loss=1.6145 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.5946 val_joint_macro_f1=0.6399 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.2218 lr=6.85335e-05 train_metal_acc=0.9447 val_loss=1.6410 val_metal_acc=0.7527 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6096 val_joint_macro_f1=0.6501 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.2295 lr=6.85335e-05 train_metal_acc=0.9428 val_loss=1.6166 val_metal_acc=0.7198 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5692 val_joint_macro_f1=0.6175 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.2122 lr=6.85335e-05 train_metal_acc=0.9486 val_loss=1.6677 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5949 val_joint_macro_f1=0.6342 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.2022 lr=6.85335e-05 train_metal_acc=0.9302 val_loss=1.5996 val_metal_acc=0.7088 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5871 val_joint_macro_f1=0.6026 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.1817 lr=6.85335e-05 train_metal_acc=0.9447 val_loss=1.8261 val_metal_acc=0.7473 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6077 val_joint_macro_f1=0.6488 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.1795 lr=6.85335e-05 train_metal_acc=0.9496 val_loss=1.7663 val_metal_acc=0.7418 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6122 val_joint_macro_f1=0.6565 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0044_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_2fd12175/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0044_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_2fd12175/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0044_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_2fd12175/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0044_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_2fd12175/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0044_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_2fd12175/run_metadata.json
[I 2026-05-14 07:15:19,909] Trial 44 finished with value: 0.6550963478857217 and parameters: {'learning_rate': 6.853351991201396e-05, 'weight_decay': 0.001, 'hidden_s': 128, 'head_mlp_layers': 1, 'edge_hidden': 128, 'gvp_layers': 2, 'edge_radius': 6.0, 'hidden_v': 32, 'esm_fusion_dim': 64, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 32 with value: 0.6585119076580177.
Optuna trial 44 completed: val_metal_balanced_acc=0.6550963478857217
================================================================================
[Optuna trial 45] optuna_deepmzyme_controlled_hpo_trial0045_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c23f4d70
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 64,
  "edge_radius": 6.0,
  "esm_fusion_dim": 64,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 3,
  "head_mlp_layers": 1,
  "hidden_s": 128,
  "hidden_v": 32,
  "learning_rate": 6.741792218624217e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.001,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0045_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c23f4d70 --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 6.741792218624217e-05 --weight-decay 0.001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 64 --gvp-layers 3 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7533 lr=6.74179e-05 train_metal_acc=0.2939 val_loss=1.6683 val_metal_acc=0.2637 val_metal_min_recall=0.0000 val_fe_recall=0.3636 val_joint_bal_acc=0.3177 val_joint_macro_f1=0.2469 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.5631 lr=6.74179e-05 train_metal_acc=0.6508 val_loss=1.4662 val_metal_acc=0.4231 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4044 val_joint_macro_f1=0.3873 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.3643 lr=6.74179e-05 train_metal_acc=0.6799 val_loss=1.3221 val_metal_acc=0.5110 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4262 val_joint_macro_f1=0.4085 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.2217 lr=6.74179e-05 train_metal_acc=0.6702 val_loss=1.3277 val_metal_acc=0.4341 val_metal_min_recall=0.0000 val_fe_recall=0.6667 val_joint_bal_acc=0.4537 val_joint_macro_f1=0.4285 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.0954 lr=6.74179e-05 train_metal_acc=0.7468 val_loss=1.1973 val_metal_acc=0.4725 val_metal_min_recall=0.0000 val_fe_recall=0.5152 val_joint_bal_acc=0.5244 val_joint_macro_f1=0.4893 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=0.9864 lr=6.74179e-05 train_metal_acc=0.7478 val_loss=1.2010 val_metal_acc=0.4505 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5308 val_joint_macro_f1=0.4735 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=0.8882 lr=6.74179e-05 train_metal_acc=0.7963 val_loss=1.1105 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5908 val_joint_macro_f1=0.5896 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=0.8197 lr=6.74179e-05 train_metal_acc=0.7973 val_loss=1.0710 val_metal_acc=0.7473 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.5880 val_joint_macro_f1=0.6032 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.7592 lr=6.74179e-05 train_metal_acc=0.8206 val_loss=1.0762 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.6131 val_joint_macro_f1=0.5969 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.7001 lr=6.74179e-05 train_metal_acc=0.8429 val_loss=1.0775 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5716 val_joint_macro_f1=0.5596 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.6499 lr=6.74179e-05 train_metal_acc=0.8555 val_loss=1.0686 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5750 val_joint_macro_f1=0.5649 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.6028 lr=6.74179e-05 train_metal_acc=0.8535 val_loss=1.0352 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5590 val_joint_macro_f1=0.5879 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.5538 lr=6.74179e-05 train_metal_acc=0.8729 val_loss=1.0664 val_metal_acc=0.7582 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5997 val_joint_macro_f1=0.6116 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.5316 lr=6.74179e-05 train_metal_acc=0.8904 val_loss=1.0922 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5828 val_joint_macro_f1=0.6024 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.4858 lr=6.74179e-05 train_metal_acc=0.8758 val_loss=1.0888 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5998 val_joint_macro_f1=0.5872 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.4520 lr=6.74179e-05 train_metal_acc=0.8797 val_loss=1.0506 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5771 val_joint_macro_f1=0.6016 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.4371 lr=6.74179e-05 train_metal_acc=0.8972 val_loss=1.1424 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5586 val_joint_macro_f1=0.5901 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.4194 lr=6.74179e-05 train_metal_acc=0.8681 val_loss=1.1638 val_metal_acc=0.6154 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5709 val_joint_macro_f1=0.5488 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.3763 lr=6.74179e-05 train_metal_acc=0.9011 val_loss=1.1523 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5479 val_joint_macro_f1=0.5835 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.3673 lr=6.74179e-05 train_metal_acc=0.9011 val_loss=1.1485 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6153 val_joint_macro_f1=0.6331 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.3471 lr=6.74179e-05 train_metal_acc=0.9146 val_loss=1.2714 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5796 val_joint_macro_f1=0.5984 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.3192 lr=6.74179e-05 train_metal_acc=0.9098 val_loss=1.3560 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5702 val_joint_macro_f1=0.5986 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.3290 lr=6.74179e-05 train_metal_acc=0.8865 val_loss=1.2347 val_metal_acc=0.6484 val_metal_min_recall=0.1538 val_fe_recall=0.4848 val_joint_bal_acc=0.5655 val_joint_macro_f1=0.5606 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.3258 lr=6.74179e-05 train_metal_acc=0.9214 val_loss=1.2617 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5816 val_joint_macro_f1=0.6009 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.3048 lr=6.74179e-05 train_metal_acc=0.9224 val_loss=1.3421 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5619 val_joint_macro_f1=0.5986 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.2991 lr=6.74179e-05 train_metal_acc=0.9311 val_loss=1.3725 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5687 val_joint_macro_f1=0.6177 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.2732 lr=6.74179e-05 train_metal_acc=0.9166 val_loss=1.2475 val_metal_acc=0.7363 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6140 val_joint_macro_f1=0.6161 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.2556 lr=6.74179e-05 train_metal_acc=0.9340 val_loss=1.2994 val_metal_acc=0.7253 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.5995 val_joint_macro_f1=0.6173 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.2495 lr=6.74179e-05 train_metal_acc=0.9370 val_loss=1.4636 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5476 val_joint_macro_f1=0.5786 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.2579 lr=6.74179e-05 train_metal_acc=0.9243 val_loss=1.4239 val_metal_acc=0.7527 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5971 val_joint_macro_f1=0.6362 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.2508 lr=6.74179e-05 train_metal_acc=0.9321 val_loss=1.4944 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5751 val_joint_macro_f1=0.5997 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.2336 lr=6.74179e-05 train_metal_acc=0.9360 val_loss=1.5046 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5997 val_joint_macro_f1=0.6136 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.2291 lr=6.74179e-05 train_metal_acc=0.9389 val_loss=1.6458 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5569 val_joint_macro_f1=0.5846 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.2333 lr=6.74179e-05 train_metal_acc=0.9399 val_loss=1.6924 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5660 val_joint_macro_f1=0.5740 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.2200 lr=6.74179e-05 train_metal_acc=0.9418 val_loss=1.6643 val_metal_acc=0.7143 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5788 val_joint_macro_f1=0.6033 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.2363 lr=6.74179e-05 train_metal_acc=0.9399 val_loss=1.6554 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5598 val_joint_macro_f1=0.5830 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.2183 lr=6.74179e-05 train_metal_acc=0.9418 val_loss=1.6502 val_metal_acc=0.6923 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5383 val_joint_macro_f1=0.5617 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.2203 lr=6.74179e-05 train_metal_acc=0.9447 val_loss=1.6327 val_metal_acc=0.7033 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5639 val_joint_macro_f1=0.5808 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.2250 lr=6.74179e-05 train_metal_acc=0.9467 val_loss=1.7579 val_metal_acc=0.7253 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5930 val_joint_macro_f1=0.6227 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.2158 lr=6.74179e-05 train_metal_acc=0.9389 val_loss=1.6697 val_metal_acc=0.7198 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5978 val_joint_macro_f1=0.6248 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0045_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c23f4d70/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0045_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c23f4d70/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0045_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c23f4d70/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0045_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c23f4d70/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0045_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c23f4d70/run_metadata.json
[I 2026-05-14 07:24:38,373] Trial 45 finished with value: 0.6153257031814906 and parameters: {'learning_rate': 6.741792218624217e-05, 'weight_decay': 0.001, 'hidden_s': 128, 'head_mlp_layers': 1, 'edge_hidden': 64, 'gvp_layers': 3, 'edge_radius': 6.0, 'hidden_v': 32, 'esm_fusion_dim': 64, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 32 with value: 0.6585119076580177.
Optuna trial 45 completed: val_metal_balanced_acc=0.6153257031814906
================================================================================
[Optuna trial 46] optuna_deepmzyme_controlled_hpo_trial0046_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_2627ca5a
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 6.0,
  "esm_fusion_dim": 64,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 2,
  "head_mlp_layers": 1,
  "hidden_s": 256,
  "hidden_v": 32,
  "learning_rate": 7.495573011938715e-05,
  "metal_class_weight_mode": "inverse_sqrt_frequency",
  "metal_weighting_setup": "inverse_sqrt_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.001,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0046_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_2627ca5a --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 7.495573011938715e-05 --weight-decay 0.001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 256 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_sqrt_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.6297 lr=7.49557e-05 train_metal_acc=0.5296 val_loss=1.4092 val_metal_acc=0.5385 val_metal_min_recall=0.0000 val_fe_recall=0.4242 val_joint_bal_acc=0.2354 val_joint_macro_f1=0.2026 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.3548 lr=7.49557e-05 train_metal_acc=0.6790 val_loss=1.1445 val_metal_acc=0.6868 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4565 val_joint_macro_f1=0.4602 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.0789 lr=7.49557e-05 train_metal_acc=0.6867 val_loss=1.1078 val_metal_acc=0.6703 val_metal_min_recall=0.0000 val_fe_recall=0.4545 val_joint_bal_acc=0.4753 val_joint_macro_f1=0.4796 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=0.9386 lr=7.49557e-05 train_metal_acc=0.7507 val_loss=1.0761 val_metal_acc=0.7363 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.5120 val_joint_macro_f1=0.5096 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=0.8368 lr=7.49557e-05 train_metal_acc=0.7983 val_loss=1.0670 val_metal_acc=0.6538 val_metal_min_recall=0.0000 val_fe_recall=0.5455 val_joint_bal_acc=0.5266 val_joint_macro_f1=0.5316 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=0.7478 lr=7.49557e-05 train_metal_acc=0.8235 val_loss=0.9818 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5902 val_joint_macro_f1=0.6114 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=0.6874 lr=7.49557e-05 train_metal_acc=0.8206 val_loss=0.9779 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5576 val_joint_macro_f1=0.6061 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=0.6313 lr=7.49557e-05 train_metal_acc=0.8526 val_loss=0.9640 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6119 val_joint_macro_f1=0.6427 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.5746 lr=7.49557e-05 train_metal_acc=0.8671 val_loss=0.9603 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.6165 val_joint_macro_f1=0.6139 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.5450 lr=7.49557e-05 train_metal_acc=0.8574 val_loss=0.9158 val_metal_acc=0.7527 val_metal_min_recall=0.0769 val_fe_recall=0.6970 val_joint_bal_acc=0.6297 val_joint_macro_f1=0.6428 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.4963 lr=7.49557e-05 train_metal_acc=0.8836 val_loss=0.9498 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.7273 val_joint_bal_acc=0.5800 val_joint_macro_f1=0.6043 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.4518 lr=7.49557e-05 train_metal_acc=0.8982 val_loss=0.9781 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5697 val_joint_macro_f1=0.5970 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.4518 lr=7.49557e-05 train_metal_acc=0.8855 val_loss=1.0540 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5828 val_joint_macro_f1=0.6010 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.3927 lr=7.49557e-05 train_metal_acc=0.9001 val_loss=1.1006 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5877 val_joint_macro_f1=0.5947 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.3811 lr=7.49557e-05 train_metal_acc=0.9020 val_loss=1.1524 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5629 val_joint_macro_f1=0.6008 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.3527 lr=7.49557e-05 train_metal_acc=0.8991 val_loss=1.1890 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5797 val_joint_macro_f1=0.6066 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.3407 lr=7.49557e-05 train_metal_acc=0.9166 val_loss=1.0927 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5736 val_joint_macro_f1=0.6015 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.3150 lr=7.49557e-05 train_metal_acc=0.9166 val_loss=1.0537 val_metal_acc=0.7582 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6470 val_joint_macro_f1=0.6256 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.3144 lr=7.49557e-05 train_metal_acc=0.9176 val_loss=1.0829 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.6044 val_joint_macro_f1=0.6216 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.2922 lr=7.49557e-05 train_metal_acc=0.9185 val_loss=1.3118 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5456 val_joint_macro_f1=0.5770 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.2847 lr=7.49557e-05 train_metal_acc=0.9340 val_loss=1.1642 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6290 val_joint_macro_f1=0.6239 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.2681 lr=7.49557e-05 train_metal_acc=0.9321 val_loss=1.2581 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5540 val_joint_macro_f1=0.5808 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.2452 lr=7.49557e-05 train_metal_acc=0.9273 val_loss=1.4272 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5927 val_joint_macro_f1=0.5998 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.2491 lr=7.49557e-05 train_metal_acc=0.9340 val_loss=1.4257 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5449 val_joint_macro_f1=0.5818 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.2517 lr=7.49557e-05 train_metal_acc=0.9156 val_loss=1.3594 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5911 val_joint_macro_f1=0.6147 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.2420 lr=7.49557e-05 train_metal_acc=0.9234 val_loss=1.4225 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5714 val_joint_macro_f1=0.5868 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.2328 lr=7.49557e-05 train_metal_acc=0.9379 val_loss=1.3270 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5431 val_joint_macro_f1=0.5638 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.2365 lr=7.49557e-05 train_metal_acc=0.9486 val_loss=1.4302 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5594 val_joint_macro_f1=0.5757 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.2177 lr=7.49557e-05 train_metal_acc=0.9467 val_loss=1.5025 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5886 val_joint_macro_f1=0.6139 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.2103 lr=7.49557e-05 train_metal_acc=0.9331 val_loss=1.5510 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5487 val_joint_macro_f1=0.5702 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.2026 lr=7.49557e-05 train_metal_acc=0.9437 val_loss=1.5383 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5688 val_joint_macro_f1=0.5752 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.2083 lr=7.49557e-05 train_metal_acc=0.9418 val_loss=1.7240 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5634 val_joint_macro_f1=0.5982 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.2079 lr=7.49557e-05 train_metal_acc=0.9370 val_loss=1.7757 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5365 val_joint_macro_f1=0.5734 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.2285 lr=7.49557e-05 train_metal_acc=0.9428 val_loss=1.8477 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5580 val_joint_macro_f1=0.5916 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.2141 lr=7.49557e-05 train_metal_acc=0.9399 val_loss=1.5472 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5759 val_joint_macro_f1=0.5946 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.2022 lr=7.49557e-05 train_metal_acc=0.9467 val_loss=1.6601 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5541 val_joint_macro_f1=0.5671 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.2099 lr=7.49557e-05 train_metal_acc=0.9476 val_loss=1.6724 val_metal_acc=0.7088 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5431 val_joint_macro_f1=0.5508 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.2172 lr=7.49557e-05 train_metal_acc=0.9486 val_loss=1.6976 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5560 val_joint_macro_f1=0.5649 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.2114 lr=7.49557e-05 train_metal_acc=0.9525 val_loss=1.7414 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5762 val_joint_macro_f1=0.6069 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.1987 lr=7.49557e-05 train_metal_acc=0.9525 val_loss=1.7782 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5594 val_joint_macro_f1=0.5881 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0046_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_2627ca5a/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0046_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_2627ca5a/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0046_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_2627ca5a/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0046_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_2627ca5a/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0046_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_2627ca5a/run_metadata.json
[I 2026-05-14 07:33:22,358] Trial 46 finished with value: 0.6469561220984371 and parameters: {'learning_rate': 7.495573011938715e-05, 'weight_decay': 0.001, 'hidden_s': 256, 'head_mlp_layers': 1, 'edge_hidden': 128, 'gvp_layers': 2, 'edge_radius': 6.0, 'hidden_v': 32, 'esm_fusion_dim': 64, 'metal_class_weight_mode': 'inverse_sqrt_frequency'}. Best is trial 32 with value: 0.6585119076580177.
Optuna trial 46 completed: val_metal_balanced_acc=0.6469561220984371
================================================================================
[Optuna trial 47] optuna_deepmzyme_controlled_hpo_trial0047_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_03b5419e
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 8.0,
  "esm_fusion_dim": 64,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 2,
  "head_mlp_layers": 1,
  "hidden_s": 128,
  "hidden_v": 32,
  "learning_rate": 3.317996512442953e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.0001,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0047_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_03b5419e --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 3.317996512442953e-05 --weight-decay 0.0001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 8.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7889 lr=3.318e-05 train_metal_acc=0.4403 val_loss=1.7006 val_metal_acc=0.4890 val_metal_min_recall=0.0000 val_fe_recall=0.3636 val_joint_bal_acc=0.2550 val_joint_macro_f1=0.2236 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.6920 lr=3.318e-05 train_metal_acc=0.5325 val_loss=1.5852 val_metal_acc=0.6154 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4000 val_joint_macro_f1=0.4069 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.5605 lr=3.318e-05 train_metal_acc=0.5403 val_loss=1.4705 val_metal_acc=0.5769 val_metal_min_recall=0.0000 val_fe_recall=0.7273 val_joint_bal_acc=0.4194 val_joint_macro_f1=0.3989 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.4439 lr=3.318e-05 train_metal_acc=0.6149 val_loss=1.3815 val_metal_acc=0.3681 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.3783 val_joint_macro_f1=0.3610 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.3438 lr=3.318e-05 train_metal_acc=0.6673 val_loss=1.2863 val_metal_acc=0.6758 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4744 val_joint_macro_f1=0.4781 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.2669 lr=3.318e-05 train_metal_acc=0.6770 val_loss=1.2865 val_metal_acc=0.4176 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.4238 val_joint_macro_f1=0.4250 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=1.1871 lr=3.318e-05 train_metal_acc=0.6790 val_loss=1.2657 val_metal_acc=0.6264 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5320 val_joint_macro_f1=0.4902 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=1.1130 lr=3.318e-05 train_metal_acc=0.6828 val_loss=1.2542 val_metal_acc=0.4451 val_metal_min_recall=0.0769 val_fe_recall=0.6970 val_joint_bal_acc=0.4682 val_joint_macro_f1=0.4748 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=1.0746 lr=3.318e-05 train_metal_acc=0.6935 val_loss=1.2525 val_metal_acc=0.4451 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5138 val_joint_macro_f1=0.4876 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=1.0303 lr=3.318e-05 train_metal_acc=0.7507 val_loss=1.2050 val_metal_acc=0.4560 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5229 val_joint_macro_f1=0.4872 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.9791 lr=3.318e-05 train_metal_acc=0.7905 val_loss=1.1462 val_metal_acc=0.6044 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5400 val_joint_macro_f1=0.5585 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.9242 lr=3.318e-05 train_metal_acc=0.7808 val_loss=1.1560 val_metal_acc=0.5165 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5478 val_joint_macro_f1=0.5192 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.8846 lr=3.318e-05 train_metal_acc=0.7682 val_loss=1.1561 val_metal_acc=0.5495 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5309 val_joint_macro_f1=0.5050 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.8287 lr=3.318e-05 train_metal_acc=0.7818 val_loss=1.1443 val_metal_acc=0.5440 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5491 val_joint_macro_f1=0.5459 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.7913 lr=3.318e-05 train_metal_acc=0.8128 val_loss=1.0673 val_metal_acc=0.6319 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5563 val_joint_macro_f1=0.5658 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.7514 lr=3.318e-05 train_metal_acc=0.7759 val_loss=1.2037 val_metal_acc=0.4835 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5245 val_joint_macro_f1=0.5100 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.7306 lr=3.318e-05 train_metal_acc=0.8167 val_loss=1.0932 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5812 val_joint_macro_f1=0.5777 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.7036 lr=3.318e-05 train_metal_acc=0.8264 val_loss=1.0302 val_metal_acc=0.6538 val_metal_min_recall=0.1538 val_fe_recall=0.5455 val_joint_bal_acc=0.5852 val_joint_macro_f1=0.5998 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.6502 lr=3.318e-05 train_metal_acc=0.8419 val_loss=1.0205 val_metal_acc=0.6429 val_metal_min_recall=0.1538 val_fe_recall=0.5455 val_joint_bal_acc=0.5710 val_joint_macro_f1=0.5877 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.6296 lr=3.318e-05 train_metal_acc=0.8409 val_loss=1.0529 val_metal_acc=0.6538 val_metal_min_recall=0.1538 val_fe_recall=0.4848 val_joint_bal_acc=0.5859 val_joint_macro_f1=0.5917 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.6145 lr=3.318e-05 train_metal_acc=0.8535 val_loss=1.0473 val_metal_acc=0.7363 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.6059 val_joint_macro_f1=0.6299 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.5773 lr=3.318e-05 train_metal_acc=0.8555 val_loss=0.9922 val_metal_acc=0.6813 val_metal_min_recall=0.1538 val_fe_recall=0.6364 val_joint_bal_acc=0.6009 val_joint_macro_f1=0.6203 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.5584 lr=3.318e-05 train_metal_acc=0.8458 val_loss=1.0782 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5963 val_joint_macro_f1=0.6218 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.5410 lr=3.318e-05 train_metal_acc=0.8545 val_loss=1.0595 val_metal_acc=0.6429 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5711 val_joint_macro_f1=0.5609 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.5216 lr=3.318e-05 train_metal_acc=0.8749 val_loss=1.0613 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5933 val_joint_macro_f1=0.6120 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.4895 lr=3.318e-05 train_metal_acc=0.8788 val_loss=1.0351 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6041 val_joint_macro_f1=0.6219 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.4629 lr=3.318e-05 train_metal_acc=0.8885 val_loss=1.0568 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5879 val_joint_macro_f1=0.6081 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.4437 lr=3.318e-05 train_metal_acc=0.8933 val_loss=1.0485 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5828 val_joint_macro_f1=0.5904 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.4317 lr=3.318e-05 train_metal_acc=0.8904 val_loss=1.0388 val_metal_acc=0.6703 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.5976 val_joint_macro_f1=0.6043 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.4277 lr=3.318e-05 train_metal_acc=0.8933 val_loss=1.0692 val_metal_acc=0.6703 val_metal_min_recall=0.2308 val_fe_recall=0.5758 val_joint_bal_acc=0.6085 val_joint_macro_f1=0.6318 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.4143 lr=3.318e-05 train_metal_acc=0.8933 val_loss=1.0760 val_metal_acc=0.6703 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.6007 val_joint_macro_f1=0.6271 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.3735 lr=3.318e-05 train_metal_acc=0.9011 val_loss=1.1217 val_metal_acc=0.6868 val_metal_min_recall=0.0769 val_fe_recall=0.6970 val_joint_bal_acc=0.5982 val_joint_macro_f1=0.6310 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.3776 lr=3.318e-05 train_metal_acc=0.8972 val_loss=1.0985 val_metal_acc=0.6648 val_metal_min_recall=0.1538 val_fe_recall=0.5758 val_joint_bal_acc=0.5956 val_joint_macro_f1=0.6163 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.3683 lr=3.318e-05 train_metal_acc=0.8982 val_loss=1.0722 val_metal_acc=0.6923 val_metal_min_recall=0.1538 val_fe_recall=0.6970 val_joint_bal_acc=0.6178 val_joint_macro_f1=0.6511 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.3534 lr=3.318e-05 train_metal_acc=0.9049 val_loss=1.1482 val_metal_acc=0.6593 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5791 val_joint_macro_f1=0.5962 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.3508 lr=3.318e-05 train_metal_acc=0.9098 val_loss=1.1339 val_metal_acc=0.7308 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6042 val_joint_macro_f1=0.6201 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.3361 lr=3.318e-05 train_metal_acc=0.9146 val_loss=1.1074 val_metal_acc=0.6978 val_metal_min_recall=0.2308 val_fe_recall=0.6364 val_joint_bal_acc=0.6244 val_joint_macro_f1=0.6506 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.3214 lr=3.318e-05 train_metal_acc=0.9088 val_loss=1.1361 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5895 val_joint_macro_f1=0.5818 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.3111 lr=3.318e-05 train_metal_acc=0.9176 val_loss=1.2205 val_metal_acc=0.7473 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6170 val_joint_macro_f1=0.6409 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.2877 lr=3.318e-05 train_metal_acc=0.9117 val_loss=1.2728 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5915 val_joint_macro_f1=0.6162 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0047_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_03b5419e/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0047_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_03b5419e/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0047_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_03b5419e/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0047_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_03b5419e/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0047_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_03b5419e/run_metadata.json
[I 2026-05-14 07:42:54,665] Trial 47 finished with value: 0.6244486064220409 and parameters: {'learning_rate': 3.317996512442953e-05, 'weight_decay': 0.0001, 'hidden_s': 128, 'head_mlp_layers': 1, 'edge_hidden': 128, 'gvp_layers': 2, 'edge_radius': 8.0, 'hidden_v': 32, 'esm_fusion_dim': 64, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 32 with value: 0.6585119076580177.
Optuna trial 47 completed: val_metal_balanced_acc=0.6244486064220409
================================================================================
[Optuna trial 48] optuna_deepmzyme_controlled_hpo_trial0048_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c00511b0
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 64,
  "edge_radius": 6.0,
  "esm_fusion_dim": 256,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 3,
  "head_mlp_layers": 3,
  "hidden_s": 128,
  "hidden_v": 16,
  "learning_rate": 6.334432151526677e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 0.001,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0048_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c00511b0 --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 6.334432151526677e-05 --weight-decay 0.001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 256 --head-mlp-layers 3 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 64 --gvp-layers 3 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 16 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7790 lr=6.33443e-05 train_metal_acc=0.5121 val_loss=1.7373 val_metal_acc=0.5385 val_metal_min_recall=0.0000 val_fe_recall=0.4242 val_joint_bal_acc=0.2354 val_joint_macro_f1=0.2002 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.7143 lr=6.33443e-05 train_metal_acc=0.6654 val_loss=1.6365 val_metal_acc=0.4341 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.3967 val_joint_macro_f1=0.3780 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.5651 lr=6.33443e-05 train_metal_acc=0.6780 val_loss=1.4788 val_metal_acc=0.4505 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.4135 val_joint_macro_f1=0.3721 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.3641 lr=6.33443e-05 train_metal_acc=0.6935 val_loss=1.3450 val_metal_acc=0.4670 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.4255 val_joint_macro_f1=0.3913 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.1773 lr=6.33443e-05 train_metal_acc=0.7274 val_loss=1.3328 val_metal_acc=0.4231 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.4435 val_joint_macro_f1=0.4176 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.0606 lr=6.33443e-05 train_metal_acc=0.7827 val_loss=1.2466 val_metal_acc=0.4670 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.4521 val_joint_macro_f1=0.4506 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=0.9499 lr=6.33443e-05 train_metal_acc=0.7915 val_loss=1.1928 val_metal_acc=0.6703 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5277 val_joint_macro_f1=0.5391 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=0.8590 lr=6.33443e-05 train_metal_acc=0.8070 val_loss=1.1131 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5592 val_joint_macro_f1=0.5797 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=0.7672 lr=6.33443e-05 train_metal_acc=0.8429 val_loss=1.1267 val_metal_acc=0.6703 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5983 val_joint_macro_f1=0.6086 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=0.6835 lr=6.33443e-05 train_metal_acc=0.8555 val_loss=1.1524 val_metal_acc=0.6868 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5933 val_joint_macro_f1=0.5873 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=0.6205 lr=6.33443e-05 train_metal_acc=0.8477 val_loss=1.2192 val_metal_acc=0.6593 val_metal_min_recall=0.1538 val_fe_recall=0.4848 val_joint_bal_acc=0.5912 val_joint_macro_f1=0.5893 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.5265 lr=6.33443e-05 train_metal_acc=0.8555 val_loss=1.2255 val_metal_acc=0.6648 val_metal_min_recall=0.2308 val_fe_recall=0.6061 val_joint_bal_acc=0.6096 val_joint_macro_f1=0.6072 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.5093 lr=6.33443e-05 train_metal_acc=0.8885 val_loss=1.3037 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5848 val_joint_macro_f1=0.5741 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.4541 lr=6.33443e-05 train_metal_acc=0.8729 val_loss=1.3003 val_metal_acc=0.6648 val_metal_min_recall=0.3077 val_fe_recall=0.6364 val_joint_bal_acc=0.6167 val_joint_macro_f1=0.6124 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.4009 lr=6.33443e-05 train_metal_acc=0.9069 val_loss=1.4706 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5533 val_joint_macro_f1=0.5746 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.3848 lr=6.33443e-05 train_metal_acc=0.9108 val_loss=1.3900 val_metal_acc=0.6978 val_metal_min_recall=0.2308 val_fe_recall=0.6061 val_joint_bal_acc=0.6077 val_joint_macro_f1=0.6084 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.3546 lr=6.33443e-05 train_metal_acc=0.9234 val_loss=1.6394 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5495 val_joint_macro_f1=0.5681 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.3697 lr=6.33443e-05 train_metal_acc=0.9195 val_loss=1.7152 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5845 val_joint_macro_f1=0.5785 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.3461 lr=6.33443e-05 train_metal_acc=0.9253 val_loss=1.6772 val_metal_acc=0.6703 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5904 val_joint_macro_f1=0.5916 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.3523 lr=6.33443e-05 train_metal_acc=0.9195 val_loss=1.9822 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5391 val_joint_macro_f1=0.5456 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.3276 lr=6.33443e-05 train_metal_acc=0.9273 val_loss=2.0631 val_metal_acc=0.6593 val_metal_min_recall=0.2308 val_fe_recall=0.6061 val_joint_bal_acc=0.5653 val_joint_macro_f1=0.5989 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.3130 lr=6.33443e-05 train_metal_acc=0.9234 val_loss=2.1278 val_metal_acc=0.6264 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5352 val_joint_macro_f1=0.5394 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.3278 lr=6.33443e-05 train_metal_acc=0.9273 val_loss=2.2601 val_metal_acc=0.6813 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5412 val_joint_macro_f1=0.5565 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.3168 lr=6.33443e-05 train_metal_acc=0.9331 val_loss=2.0749 val_metal_acc=0.6209 val_metal_min_recall=0.3077 val_fe_recall=0.6061 val_joint_bal_acc=0.5522 val_joint_macro_f1=0.5710 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.2899 lr=6.33443e-05 train_metal_acc=0.9399 val_loss=2.3732 val_metal_acc=0.6319 val_metal_min_recall=0.2308 val_fe_recall=0.6061 val_joint_bal_acc=0.5452 val_joint_macro_f1=0.5730 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.2657 lr=6.33443e-05 train_metal_acc=0.9408 val_loss=2.4027 val_metal_acc=0.6923 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5457 val_joint_macro_f1=0.5749 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.2694 lr=6.33443e-05 train_metal_acc=0.9350 val_loss=2.6222 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5549 val_joint_macro_f1=0.5884 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.2626 lr=6.33443e-05 train_metal_acc=0.9379 val_loss=2.4166 val_metal_acc=0.6868 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5616 val_joint_macro_f1=0.5652 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.2344 lr=6.33443e-05 train_metal_acc=0.9379 val_loss=2.9525 val_metal_acc=0.6923 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5130 val_joint_macro_f1=0.5490 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.2572 lr=6.33443e-05 train_metal_acc=0.9476 val_loss=2.5138 val_metal_acc=0.6538 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5456 val_joint_macro_f1=0.5739 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.2871 lr=6.33443e-05 train_metal_acc=0.9496 val_loss=2.8334 val_metal_acc=0.7088 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5550 val_joint_macro_f1=0.5875 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.2245 lr=6.33443e-05 train_metal_acc=0.9486 val_loss=2.8659 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5439 val_joint_macro_f1=0.5503 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.2516 lr=6.33443e-05 train_metal_acc=0.9476 val_loss=2.8599 val_metal_acc=0.6209 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5227 val_joint_macro_f1=0.5452 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.2703 lr=6.33443e-05 train_metal_acc=0.9496 val_loss=2.8324 val_metal_acc=0.6209 val_metal_min_recall=0.0769 val_fe_recall=0.7576 val_joint_bal_acc=0.5501 val_joint_macro_f1=0.5504 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.2304 lr=6.33443e-05 train_metal_acc=0.9496 val_loss=2.8661 val_metal_acc=0.6374 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5466 val_joint_macro_f1=0.5693 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.2102 lr=6.33443e-05 train_metal_acc=0.9554 val_loss=2.8230 val_metal_acc=0.6374 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5152 val_joint_macro_f1=0.5322 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.2116 lr=6.33443e-05 train_metal_acc=0.9583 val_loss=3.0451 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5416 val_joint_macro_f1=0.5710 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.2558 lr=6.33443e-05 train_metal_acc=0.9583 val_loss=3.0136 val_metal_acc=0.6154 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4999 val_joint_macro_f1=0.5062 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.2067 lr=6.33443e-05 train_metal_acc=0.9544 val_loss=2.9854 val_metal_acc=0.6264 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.5290 val_joint_macro_f1=0.5480 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.2187 lr=6.33443e-05 train_metal_acc=0.9564 val_loss=3.4014 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5294 val_joint_macro_f1=0.5620 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0048_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c00511b0/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0048_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c00511b0/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0048_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c00511b0/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0048_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c00511b0/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0048_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c00511b0/run_metadata.json
[I 2026-05-14 07:52:09,541] Trial 48 finished with value: 0.6167125095018834 and parameters: {'learning_rate': 6.334432151526677e-05, 'weight_decay': 0.001, 'hidden_s': 128, 'head_mlp_layers': 3, 'edge_hidden': 64, 'gvp_layers': 3, 'edge_radius': 6.0, 'hidden_v': 16, 'esm_fusion_dim': 256, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 32 with value: 0.6585119076580177.
Optuna trial 48 completed: val_metal_balanced_acc=0.6167125095018834
================================================================================
[Optuna trial 49] optuna_deepmzyme_controlled_hpo_trial0049_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d32997f7
Sampled parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 6.0,
  "esm_fusion_dim": 64,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 4,
  "head_mlp_layers": 1,
  "hidden_s": 256,
  "hidden_v": 32,
  "learning_rate": 1.6801503587890522e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 1e-05,
  "zn_loss_multiplier": 1.0
}
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name optuna_deepmzyme_controlled_hpo_trial0049_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d32997f7 --model-architecture gvp --epochs 40 --batch-size 8 --learning-rate 1.6801503587890522e-05 --weight-decay 1e-05 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 256 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 4 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7862 lr=1.68015e-05 train_metal_acc=0.3531 val_loss=1.6990 val_metal_acc=0.3242 val_metal_min_recall=0.0000 val_fe_recall=0.5455 val_joint_bal_acc=0.3265 val_joint_macro_f1=0.2501 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=2 train_loss=1.7190 lr=1.68015e-05 train_metal_acc=0.3278 val_loss=1.6486 val_metal_acc=0.3407 val_metal_min_recall=0.0000 val_fe_recall=0.5758 val_joint_bal_acc=0.3525 val_joint_macro_f1=0.3043 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=3 train_loss=1.6517 lr=1.68015e-05 train_metal_acc=0.4210 val_loss=1.5665 val_metal_acc=0.3077 val_metal_min_recall=0.0000 val_fe_recall=0.2121 val_joint_bal_acc=0.3329 val_joint_macro_f1=0.2708 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=4 train_loss=1.5654 lr=1.68015e-05 train_metal_acc=0.5412 val_loss=1.4479 val_metal_acc=0.3846 val_metal_min_recall=0.0000 val_fe_recall=0.7879 val_joint_bal_acc=0.3591 val_joint_macro_f1=0.3072 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=5 train_loss=1.4453 lr=1.68015e-05 train_metal_acc=0.4374 val_loss=1.4692 val_metal_acc=0.3791 val_metal_min_recall=0.0000 val_fe_recall=0.6061 val_joint_bal_acc=0.4267 val_joint_macro_f1=0.4055 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=6 train_loss=1.3740 lr=1.68015e-05 train_metal_acc=0.5917 val_loss=1.2613 val_metal_acc=0.4890 val_metal_min_recall=0.1538 val_fe_recall=0.6970 val_joint_bal_acc=0.4946 val_joint_macro_f1=0.4949 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=7 train_loss=1.2934 lr=1.68015e-05 train_metal_acc=0.6537 val_loss=1.2309 val_metal_acc=0.3956 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.4641 val_joint_macro_f1=0.4706 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=8 train_loss=1.2270 lr=1.68015e-05 train_metal_acc=0.7158 val_loss=1.1822 val_metal_acc=0.4451 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.4803 val_joint_macro_f1=0.5171 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=9 train_loss=1.1652 lr=1.68015e-05 train_metal_acc=0.6945 val_loss=1.1814 val_metal_acc=0.4505 val_metal_min_recall=0.1538 val_fe_recall=0.5152 val_joint_bal_acc=0.5027 val_joint_macro_f1=0.5146 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=10 train_loss=1.1172 lr=1.68015e-05 train_metal_acc=0.6790 val_loss=1.1771 val_metal_acc=0.4451 val_metal_min_recall=0.1538 val_fe_recall=0.7879 val_joint_bal_acc=0.4542 val_joint_macro_f1=0.4676 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=11 train_loss=1.0575 lr=1.68015e-05 train_metal_acc=0.7391 val_loss=1.0866 val_metal_acc=0.5385 val_metal_min_recall=0.1538 val_fe_recall=0.6061 val_joint_bal_acc=0.4982 val_joint_macro_f1=0.5227 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=12 train_loss=0.9873 lr=1.68015e-05 train_metal_acc=0.7294 val_loss=1.1729 val_metal_acc=0.4835 val_metal_min_recall=0.0000 val_fe_recall=0.5455 val_joint_bal_acc=0.4911 val_joint_macro_f1=0.4887 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=13 train_loss=0.9546 lr=1.68015e-05 train_metal_acc=0.7207 val_loss=1.2160 val_metal_acc=0.4670 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5148 val_joint_macro_f1=0.4925 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=14 train_loss=0.9138 lr=1.68015e-05 train_metal_acc=0.7488 val_loss=1.0512 val_metal_acc=0.6593 val_metal_min_recall=0.0000 val_fe_recall=0.6364 val_joint_bal_acc=0.5617 val_joint_macro_f1=0.5510 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=15 train_loss=0.8743 lr=1.68015e-05 train_metal_acc=0.7546 val_loss=1.1470 val_metal_acc=0.4835 val_metal_min_recall=0.0769 val_fe_recall=0.5152 val_joint_bal_acc=0.5213 val_joint_macro_f1=0.5045 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=16 train_loss=0.8470 lr=1.68015e-05 train_metal_acc=0.7983 val_loss=1.0669 val_metal_acc=0.5879 val_metal_min_recall=0.0769 val_fe_recall=0.6970 val_joint_bal_acc=0.5485 val_joint_macro_f1=0.5620 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=17 train_loss=0.8106 lr=1.68015e-05 train_metal_acc=0.7915 val_loss=1.0097 val_metal_acc=0.6484 val_metal_min_recall=0.0769 val_fe_recall=0.6970 val_joint_bal_acc=0.6109 val_joint_macro_f1=0.5975 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=18 train_loss=0.7621 lr=1.68015e-05 train_metal_acc=0.7808 val_loss=1.0674 val_metal_acc=0.6044 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.5901 val_joint_macro_f1=0.5599 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=19 train_loss=0.7523 lr=1.68015e-05 train_metal_acc=0.7905 val_loss=0.9834 val_metal_acc=0.6813 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6209 val_joint_macro_f1=0.5934 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=20 train_loss=0.7344 lr=1.68015e-05 train_metal_acc=0.8186 val_loss=0.9601 val_metal_acc=0.6758 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.6279 val_joint_macro_f1=0.6233 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=21 train_loss=0.7095 lr=1.68015e-05 train_metal_acc=0.8274 val_loss=1.0138 val_metal_acc=0.6319 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5549 val_joint_macro_f1=0.5711 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=22 train_loss=0.6620 lr=1.68015e-05 train_metal_acc=0.7992 val_loss=1.0391 val_metal_acc=0.6319 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.6023 val_joint_macro_f1=0.5761 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=23 train_loss=0.6433 lr=1.68015e-05 train_metal_acc=0.8380 val_loss=1.0238 val_metal_acc=0.6538 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5549 val_joint_macro_f1=0.5675 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=24 train_loss=0.6214 lr=1.68015e-05 train_metal_acc=0.8458 val_loss=0.9740 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.6970 val_joint_bal_acc=0.6338 val_joint_macro_f1=0.6391 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=25 train_loss=0.6142 lr=1.68015e-05 train_metal_acc=0.8458 val_loss=0.9593 val_metal_acc=0.6813 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6077 val_joint_macro_f1=0.6171 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=26 train_loss=0.5648 lr=1.68015e-05 train_metal_acc=0.8293 val_loss=0.9930 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.8485 val_joint_bal_acc=0.6275 val_joint_macro_f1=0.6261 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=27 train_loss=0.5715 lr=1.68015e-05 train_metal_acc=0.8623 val_loss=1.0039 val_metal_acc=0.7143 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.5559 val_joint_macro_f1=0.5700 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=28 train_loss=0.5664 lr=1.68015e-05 train_metal_acc=0.8177 val_loss=1.1291 val_metal_acc=0.6099 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.5813 val_joint_macro_f1=0.5907 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=29 train_loss=0.5447 lr=1.68015e-05 train_metal_acc=0.8206 val_loss=0.9404 val_metal_acc=0.7308 val_metal_min_recall=0.1538 val_fe_recall=0.6667 val_joint_bal_acc=0.6441 val_joint_macro_f1=0.6492 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=30 train_loss=0.5111 lr=1.68015e-05 train_metal_acc=0.8642 val_loss=1.0524 val_metal_acc=0.6978 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.5702 val_joint_macro_f1=0.5894 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=31 train_loss=0.5043 lr=1.68015e-05 train_metal_acc=0.8739 val_loss=0.9737 val_metal_acc=0.7637 val_metal_min_recall=0.0769 val_fe_recall=0.6061 val_joint_bal_acc=0.6667 val_joint_macro_f1=0.6516 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=32 train_loss=0.4815 lr=1.68015e-05 train_metal_acc=0.8458 val_loss=0.9673 val_metal_acc=0.6868 val_metal_min_recall=0.0769 val_fe_recall=0.6667 val_joint_bal_acc=0.6461 val_joint_macro_f1=0.6236 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=33 train_loss=0.4814 lr=1.68015e-05 train_metal_acc=0.8603 val_loss=1.0420 val_metal_acc=0.6648 val_metal_min_recall=0.0769 val_fe_recall=0.4848 val_joint_bal_acc=0.6116 val_joint_macro_f1=0.5891 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=34 train_loss=0.4538 lr=1.68015e-05 train_metal_acc=0.8826 val_loss=1.0000 val_metal_acc=0.6813 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.5818 val_joint_macro_f1=0.6004 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=35 train_loss=0.4334 lr=1.68015e-05 train_metal_acc=0.8720 val_loss=1.0221 val_metal_acc=0.7033 val_metal_min_recall=0.0769 val_fe_recall=0.6364 val_joint_bal_acc=0.6190 val_joint_macro_f1=0.6215 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=36 train_loss=0.4133 lr=1.68015e-05 train_metal_acc=0.8652 val_loss=1.0387 val_metal_acc=0.7253 val_metal_min_recall=0.0769 val_fe_recall=0.4545 val_joint_bal_acc=0.6382 val_joint_macro_f1=0.5860 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=37 train_loss=0.4035 lr=1.68015e-05 train_metal_acc=0.9069 val_loss=0.9751 val_metal_acc=0.7747 val_metal_min_recall=0.0769 val_fe_recall=0.5758 val_joint_bal_acc=0.6750 val_joint_macro_f1=0.6577 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=38 train_loss=0.3866 lr=1.68015e-05 train_metal_acc=0.8855 val_loss=1.0249 val_metal_acc=0.7198 val_metal_min_recall=0.0769 val_fe_recall=0.7879 val_joint_bal_acc=0.6040 val_joint_macro_f1=0.6099 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=39 train_loss=0.3817 lr=1.68015e-05 train_metal_acc=0.8652 val_loss=1.0306 val_metal_acc=0.7582 val_metal_min_recall=0.0769 val_fe_recall=0.9091 val_joint_bal_acc=0.6035 val_joint_macro_f1=0.6141 metal_loss_scale=1.0000 ec_loss_scale=1.0000
epoch=40 train_loss=0.3789 lr=1.68015e-05 train_metal_acc=0.9040 val_loss=1.0809 val_metal_acc=0.7418 val_metal_min_recall=0.0769 val_fe_recall=0.5455 val_joint_bal_acc=0.5838 val_joint_macro_f1=0.5906 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0049_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d32997f7/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0049_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d32997f7/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0049_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d32997f7/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0049_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d32997f7/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0049_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d32997f7/run_metadata.json
[I 2026-05-14 08:01:52,721] Trial 49 finished with value: 0.6750130535709283 and parameters: {'learning_rate': 1.6801503587890522e-05, 'weight_decay': 1e-05, 'hidden_s': 256, 'head_mlp_layers': 1, 'edge_hidden': 128, 'gvp_layers': 4, 'edge_radius': 6.0, 'hidden_v': 32, 'esm_fusion_dim': 64, 'metal_class_weight_mode': 'inverse_frequency'}. Best is trial 49 with value: 0.6750130535709283.
Optuna trial 49 completed: val_metal_balanced_acc=0.6750130535709283
Seed-repeat source study: deepmzyme_controlled_hpo
Seed-repeat source table: /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna/deepmzyme_controlled_hpo/top_trials.csv
Seed-repeat source config JSON: /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna/deepmzyme_controlled_hpo/top_trial_configs.json
Seed-repeat source note: using the just-completed Optuna study in this notebook run; not scanning old mixed run directories.
Seed-repeat source configurations below are printed before commands are generated.
Selected seed-repeat source config: {"batch_size": 8, "edge_hidden": 128, "edge_radius": 6.0, "fusion_mode": "late_fusion", "gvp_layers": 4, "head_mlp_layers": 1, "hidden_s": 256, "hidden_v": 32, "learning_rate": 1.6801503587890522e-05, "metal_class_weight_mode": "inverse_frequency", "model_architecture": "gvp", "rank": 1, "source_run_dir": "/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0049_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d32997f7", "trial_number": 49, "use_early_esm": false, "use_esm": true, "validation_metric": 0.6750130535709283, "weight_decay": 1e-05}
Selected seed-repeat rerun: {"batch_size": 8, "edge_hidden": 128, "edge_radius": 6.0, "fusion_mode": "late_fusion", "gvp_layers": 4, "head_mlp_layers": 1, "hidden_s": 256, "hidden_v": 32, "learning_rate": 1.6801503587890522e-05, "metal_class_weight_mode": "inverse_frequency", "model_architecture": "gvp", "requires_esm": true, "seed": 42, "source": "seed-repeat source", "source_rank": 1, "source_run_dir": "/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0049_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d32997f7", "source_trial_number": 49, "use_early_esm": false, "use_esm": true, "weight_decay": 1e-05}
Selected seed-repeat rerun: {"batch_size": 8, "edge_hidden": 128, "edge_radius": 6.0, "fusion_mode": "late_fusion", "gvp_layers": 4, "head_mlp_layers": 1, "hidden_s": 256, "hidden_v": 32, "learning_rate": 1.6801503587890522e-05, "metal_class_weight_mode": "inverse_frequency", "model_architecture": "gvp", "requires_esm": true, "seed": 123, "source": "seed-repeat source", "source_rank": 1, "source_run_dir": "/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0049_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d32997f7", "source_trial_number": 49, "use_early_esm": false, "use_esm": true, "weight_decay": 1e-05}
Selected seed-repeat rerun: {"batch_size": 8, "edge_hidden": 128, "edge_radius": 6.0, "fusion_mode": "late_fusion", "gvp_layers": 4, "head_mlp_layers": 1, "hidden_s": 256, "hidden_v": 32, "learning_rate": 1.6801503587890522e-05, "metal_class_weight_mode": "inverse_frequency", "model_architecture": "gvp", "requires_esm": true, "seed": 2026, "source": "seed-repeat source", "source_rank": 1, "source_run_dir": "/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0049_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d32997f7", "source_trial_number": 49, "use_early_esm": false, "use_esm": true, "weight_decay": 1e-05}
Selected seed-repeat rerun: {"batch_size": 8, "edge_hidden": 128, "edge_radius": 6.0, "fusion_mode": "late_fusion", "gvp_layers": 4, "head_mlp_layers": 1, "hidden_s": 256, "hidden_v": 32, "learning_rate": 1.6801503587890522e-05, "metal_class_weight_mode": "inverse_frequency", "model_architecture": "gvp", "requires_esm": true, "seed": 43, "source": "seed-repeat source", "source_rank": 1, "source_run_dir": "/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0049_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d32997f7", "source_trial_number": 49, "use_early_esm": false, "use_esm": true, "weight_decay": 1e-05}
Selected seed-repeat rerun: {"batch_size": 8, "edge_hidden": 128, "edge_radius": 6.0, "fusion_mode": "late_fusion", "gvp_layers": 4, "head_mlp_layers": 1, "hidden_s": 256, "hidden_v": 32, "learning_rate": 1.6801503587890522e-05, "metal_class_weight_mode": "inverse_frequency", "model_architecture": "gvp", "requires_esm": true, "seed": 44, "source": "seed-repeat source", "source_rank": 1, "source_run_dir": "/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0049_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d32997f7", "source_trial_number": 49, "use_early_esm": false, "use_esm": true, "weight_decay": 1e-05}
Selected seed-repeat source config: {"batch_size": 8, "edge_hidden": 128, "edge_radius": 6.0, "fusion_mode": "late_fusion", "gvp_layers": 2, "head_mlp_layers": 1, "hidden_s": 128, "hidden_v": 32, "learning_rate": 5.4715836015281065e-05, "metal_class_weight_mode": "inverse_frequency", "model_architecture": "gvp", "rank": 2, "source_run_dir": "/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0032_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_faf01e7c", "trial_number": 32, "use_early_esm": false, "use_esm": true, "validation_metric": 0.6585119076580177, "weight_decay": 0.001}
Selected seed-repeat rerun: {"batch_size": 8, "edge_hidden": 128, "edge_radius": 6.0, "fusion_mode": "late_fusion", "gvp_layers": 2, "head_mlp_layers": 1, "hidden_s": 128, "hidden_v": 32, "learning_rate": 5.4715836015281065e-05, "metal_class_weight_mode": "inverse_frequency", "model_architecture": "gvp", "requires_esm": true, "seed": 42, "source": "seed-repeat source", "source_rank": 2, "source_run_dir": "/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0032_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_faf01e7c", "source_trial_number": 32, "use_early_esm": false, "use_esm": true, "weight_decay": 0.001}
Selected seed-repeat rerun: {"batch_size": 8, "edge_hidden": 128, "edge_radius": 6.0, "fusion_mode": "late_fusion", "gvp_layers": 2, "head_mlp_layers": 1, "hidden_s": 128, "hidden_v": 32, "learning_rate": 5.4715836015281065e-05, "metal_class_weight_mode": "inverse_frequency", "model_architecture": "gvp", "requires_esm": true, "seed": 123, "source": "seed-repeat source", "source_rank": 2, "source_run_dir": "/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0032_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_faf01e7c", "source_trial_number": 32, "use_early_esm": false, "use_esm": true, "weight_decay": 0.001}
Selected seed-repeat rerun: {"batch_size": 8, "edge_hidden": 128, "edge_radius": 6.0, "fusion_mode": "late_fusion", "gvp_layers": 2, "head_mlp_layers": 1, "hidden_s": 128, "hidden_v": 32, "learning_rate": 5.4715836015281065e-05, "metal_class_weight_mode": "inverse_frequency", "model_architecture": "gvp", "requires_esm": true, "seed": 2026, "source": "seed-repeat source", "source_rank": 2, "source_run_dir": "/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0032_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_faf01e7c", "source_trial_number": 32, "use_early_esm": false, "use_esm": true, "weight_decay": 0.001}
Selected seed-repeat rerun: {"batch_size": 8, "edge_hidden": 128, "edge_radius": 6.0, "fusion_mode": "late_fusion", "gvp_layers": 2, "head_mlp_layers": 1, "hidden_s": 128, "hidden_v": 32, "learning_rate": 5.4715836015281065e-05, "metal_class_weight_mode": "inverse_frequency", "model_architecture": "gvp", "requires_esm": true, "seed": 43, "source": "seed-repeat source", "source_rank": 2, "source_run_dir": "/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0032_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_faf01e7c", "source_trial_number": 32, "use_early_esm": false, "use_esm": true, "weight_decay": 0.001}
Selected seed-repeat rerun: {"batch_size": 8, "edge_hidden": 128, "edge_radius": 6.0, "fusion_mode": "late_fusion", "gvp_layers": 2, "head_mlp_layers": 1, "hidden_s": 128, "hidden_v": 32, "learning_rate": 5.4715836015281065e-05, "metal_class_weight_mode": "inverse_frequency", "model_architecture": "gvp", "requires_esm": true, "seed": 44, "source": "seed-repeat source", "source_rank": 2, "source_run_dir": "/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0032_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_faf01e7c", "source_trial_number": 32, "use_early_esm": false, "use_esm": true, "weight_decay": 0.001}
Selected seed-repeat source config: {"batch_size": 8, "edge_hidden": 128, "edge_radius": 6.0, "fusion_mode": "late_fusion", "gvp_layers": 2, "head_mlp_layers": 1, "hidden_s": 128, "hidden_v": 32, "learning_rate": 7.032630334240692e-05, "metal_class_weight_mode": "inverse_frequency", "model_architecture": "gvp", "rank": 3, "source_run_dir": "/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0015_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a931e009", "trial_number": 15, "use_early_esm": false, "use_esm": true, "validation_metric": 0.6550963478857217, "weight_decay": 0.001}
Selected seed-repeat rerun: {"batch_size": 8, "edge_hidden": 128, "edge_radius": 6.0, "fusion_mode": "late_fusion", "gvp_layers": 2, "head_mlp_layers": 1, "hidden_s": 128, "hidden_v": 32, "learning_rate": 7.032630334240692e-05, "metal_class_weight_mode": "inverse_frequency", "model_architecture": "gvp", "requires_esm": true, "seed": 42, "source": "seed-repeat source", "source_rank": 3, "source_run_dir": "/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0015_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a931e009", "source_trial_number": 15, "use_early_esm": false, "use_esm": true, "weight_decay": 0.001}
Selected seed-repeat rerun: {"batch_size": 8, "edge_hidden": 128, "edge_radius": 6.0, "fusion_mode": "late_fusion", "gvp_layers": 2, "head_mlp_layers": 1, "hidden_s": 128, "hidden_v": 32, "learning_rate": 7.032630334240692e-05, "metal_class_weight_mode": "inverse_frequency", "model_architecture": "gvp", "requires_esm": true, "seed": 123, "source": "seed-repeat source", "source_rank": 3, "source_run_dir": "/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0015_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a931e009", "source_trial_number": 15, "use_early_esm": false, "use_esm": true, "weight_decay": 0.001}
Selected seed-repeat rerun: {"batch_size": 8, "edge_hidden": 128, "edge_radius": 6.0, "fusion_mode": "late_fusion", "gvp_layers": 2, "head_mlp_layers": 1, "hidden_s": 128, "hidden_v": 32, "learning_rate": 7.032630334240692e-05, "metal_class_weight_mode": "inverse_frequency", "model_architecture": "gvp", "requires_esm": true, "seed": 2026, "source": "seed-repeat source", "source_rank": 3, "source_run_dir": "/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0015_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a931e009", "source_trial_number": 15, "use_early_esm": false, "use_esm": true, "weight_decay": 0.001}
Selected seed-repeat rerun: {"batch_size": 8, "edge_hidden": 128, "edge_radius": 6.0, "fusion_mode": "late_fusion", "gvp_layers": 2, "head_mlp_layers": 1, "hidden_s": 128, "hidden_v": 32, "learning_rate": 7.032630334240692e-05, "metal_class_weight_mode": "inverse_frequency", "model_architecture": "gvp", "requires_esm": true, "seed": 43, "source": "seed-repeat source", "source_rank": 3, "source_run_dir": "/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0015_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a931e009", "source_trial_number": 15, "use_early_esm": false, "use_esm": true, "weight_decay": 0.001}
Selected seed-repeat rerun: {"batch_size": 8, "edge_hidden": 128, "edge_radius": 6.0, "fusion_mode": "late_fusion", "gvp_layers": 2, "head_mlp_layers": 1, "hidden_s": 128, "hidden_v": 32, "learning_rate": 7.032630334240692e-05, "metal_class_weight_mode": "inverse_frequency", "model_architecture": "gvp", "requires_esm": true, "seed": 44, "source": "seed-repeat source", "source_rank": 3, "source_run_dir": "/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0015_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a931e009", "source_trial_number": 15, "use_early_esm": false, "use_esm": true, "weight_decay": 0.001}

Running validation-only seed-repeat evaluation
Top configurations: 3
Repeat seeds: [42, 123, 2026, 43, 44]
================================================================================
[Seed repeat top1 trial49 seed42]
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name top1_trial49_deepmzyme_controlled_hpo_seed42_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_a198eee2 --model-architecture gvp --epochs 1 --batch-size 8 --learning-rate 1.6801503587890522e-05 --weight-decay 1e-05 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 256 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 4 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7862 lr=1.68015e-05 train_metal_acc=0.3531 val_loss=1.6990 val_metal_acc=0.3242 val_metal_min_recall=0.0000 val_fe_recall=0.5455 val_joint_bal_acc=0.3265 val_joint_macro_f1=0.2501 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top1_trial49_deepmzyme_controlled_hpo_seed42_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_a198eee2/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top1_trial49_deepmzyme_controlled_hpo_seed42_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_a198eee2/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top1_trial49_deepmzyme_controlled_hpo_seed42_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_a198eee2/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top1_trial49_deepmzyme_controlled_hpo_seed42_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_a198eee2/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top1_trial49_deepmzyme_controlled_hpo_seed42_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_a198eee2/run_metadata.json
================================================================================
[Seed repeat top1 trial49 seed123]
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name top1_trial49_deepmzyme_controlled_hpo_seed123_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archg_631e5ea0 --model-architecture gvp --epochs 1 --batch-size 8 --learning-rate 1.6801503587890522e-05 --weight-decay 1e-05 --seed 123 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 256 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 4 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7727 lr=1.68015e-05 train_metal_acc=0.2386 val_loss=1.7335 val_metal_acc=0.2033 val_metal_min_recall=0.0000 val_fe_recall=0.3947 val_joint_bal_acc=0.3013 val_joint_macro_f1=0.1914 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top1_trial49_deepmzyme_controlled_hpo_seed123_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archg_631e5ea0/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top1_trial49_deepmzyme_controlled_hpo_seed123_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archg_631e5ea0/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top1_trial49_deepmzyme_controlled_hpo_seed123_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archg_631e5ea0/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top1_trial49_deepmzyme_controlled_hpo_seed123_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archg_631e5ea0/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top1_trial49_deepmzyme_controlled_hpo_seed123_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archg_631e5ea0/run_metadata.json
================================================================================
[Seed repeat top1 trial49 seed2026]
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name top1_trial49_deepmzyme_controlled_hpo_seed2026_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_arch_bcda140d --model-architecture gvp --epochs 1 --batch-size 8 --learning-rate 1.6801503587890522e-05 --weight-decay 1e-05 --seed 2026 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 256 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 4 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7854 lr=1.68015e-05 train_metal_acc=0.1833 val_loss=1.8205 val_metal_acc=0.1538 val_metal_min_recall=0.0000 val_fe_recall=0.4000 val_joint_bal_acc=0.2471 val_joint_macro_f1=0.1146 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top1_trial49_deepmzyme_controlled_hpo_seed2026_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_arch_bcda140d/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top1_trial49_deepmzyme_controlled_hpo_seed2026_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_arch_bcda140d/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top1_trial49_deepmzyme_controlled_hpo_seed2026_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_arch_bcda140d/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top1_trial49_deepmzyme_controlled_hpo_seed2026_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_arch_bcda140d/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top1_trial49_deepmzyme_controlled_hpo_seed2026_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_arch_bcda140d/run_metadata.json
================================================================================
[Seed repeat top1 trial49 seed43]
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name top1_trial49_deepmzyme_controlled_hpo_seed43_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_56eb2f7c --model-architecture gvp --epochs 1 --batch-size 8 --learning-rate 1.6801503587890522e-05 --weight-decay 1e-05 --seed 43 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 256 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 4 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7854 lr=1.68015e-05 train_metal_acc=0.2124 val_loss=1.7213 val_metal_acc=0.2802 val_metal_min_recall=0.0000 val_fe_recall=0.4167 val_joint_bal_acc=0.3206 val_joint_macro_f1=0.2133 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top1_trial49_deepmzyme_controlled_hpo_seed43_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_56eb2f7c/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top1_trial49_deepmzyme_controlled_hpo_seed43_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_56eb2f7c/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top1_trial49_deepmzyme_controlled_hpo_seed43_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_56eb2f7c/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top1_trial49_deepmzyme_controlled_hpo_seed43_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_56eb2f7c/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top1_trial49_deepmzyme_controlled_hpo_seed43_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_56eb2f7c/run_metadata.json
================================================================================
[Seed repeat top1 trial49 seed44]
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name top1_trial49_deepmzyme_controlled_hpo_seed44_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_a1ad5381 --model-architecture gvp --epochs 1 --batch-size 8 --learning-rate 1.6801503587890522e-05 --weight-decay 1e-05 --seed 44 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 256 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 4 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7736 lr=1.68015e-05 train_metal_acc=0.3734 val_loss=1.7173 val_metal_acc=0.3352 val_metal_min_recall=0.0000 val_fe_recall=0.3889 val_joint_bal_acc=0.3539 val_joint_macro_f1=0.3257 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top1_trial49_deepmzyme_controlled_hpo_seed44_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_a1ad5381/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top1_trial49_deepmzyme_controlled_hpo_seed44_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_a1ad5381/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top1_trial49_deepmzyme_controlled_hpo_seed44_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_a1ad5381/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top1_trial49_deepmzyme_controlled_hpo_seed44_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_a1ad5381/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top1_trial49_deepmzyme_controlled_hpo_seed44_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_a1ad5381/run_metadata.json
================================================================================
[Seed repeat top2 trial32 seed42]
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name top2_trial32_deepmzyme_controlled_hpo_seed42_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_91620ace --model-architecture gvp --epochs 1 --batch-size 8 --learning-rate 5.4715836015281065e-05 --weight-decay 0.001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7678 lr=5.47158e-05 train_metal_acc=0.5073 val_loss=1.6362 val_metal_acc=0.5659 val_metal_min_recall=0.0000 val_fe_recall=0.3939 val_joint_bal_acc=0.3724 val_joint_macro_f1=0.3616 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top2_trial32_deepmzyme_controlled_hpo_seed42_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_91620ace/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top2_trial32_deepmzyme_controlled_hpo_seed42_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_91620ace/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top2_trial32_deepmzyme_controlled_hpo_seed42_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_91620ace/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top2_trial32_deepmzyme_controlled_hpo_seed42_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_91620ace/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top2_trial32_deepmzyme_controlled_hpo_seed42_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_91620ace/run_metadata.json
================================================================================
[Seed repeat top2 trial32 seed123]
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name top2_trial32_deepmzyme_controlled_hpo_seed123_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archg_c3065b84 --model-architecture gvp --epochs 1 --batch-size 8 --learning-rate 5.4715836015281065e-05 --weight-decay 0.001 --seed 123 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7881 lr=5.47158e-05 train_metal_acc=0.2687 val_loss=1.7284 val_metal_acc=0.2802 val_metal_min_recall=0.0000 val_fe_recall=0.0263 val_joint_bal_acc=0.3655 val_joint_macro_f1=0.2304 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top2_trial32_deepmzyme_controlled_hpo_seed123_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archg_c3065b84/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top2_trial32_deepmzyme_controlled_hpo_seed123_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archg_c3065b84/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top2_trial32_deepmzyme_controlled_hpo_seed123_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archg_c3065b84/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top2_trial32_deepmzyme_controlled_hpo_seed123_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archg_c3065b84/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top2_trial32_deepmzyme_controlled_hpo_seed123_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archg_c3065b84/run_metadata.json
================================================================================
[Seed repeat top2 trial32 seed2026]
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name top2_trial32_deepmzyme_controlled_hpo_seed2026_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_arch_8c788af7 --model-architecture gvp --epochs 1 --batch-size 8 --learning-rate 5.4715836015281065e-05 --weight-decay 0.001 --seed 2026 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7841 lr=5.47158e-05 train_metal_acc=0.2396 val_loss=1.7700 val_metal_acc=0.1648 val_metal_min_recall=0.0000 val_fe_recall=0.2286 val_joint_bal_acc=0.3093 val_joint_macro_f1=0.2159 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top2_trial32_deepmzyme_controlled_hpo_seed2026_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_arch_8c788af7/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top2_trial32_deepmzyme_controlled_hpo_seed2026_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_arch_8c788af7/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top2_trial32_deepmzyme_controlled_hpo_seed2026_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_arch_8c788af7/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top2_trial32_deepmzyme_controlled_hpo_seed2026_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_arch_8c788af7/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top2_trial32_deepmzyme_controlled_hpo_seed2026_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_arch_8c788af7/run_metadata.json
================================================================================
[Seed repeat top2 trial32 seed43]
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name top2_trial32_deepmzyme_controlled_hpo_seed43_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_105a1203 --model-architecture gvp --epochs 1 --batch-size 8 --learning-rate 5.4715836015281065e-05 --weight-decay 0.001 --seed 43 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7687 lr=5.47158e-05 train_metal_acc=0.3637 val_loss=1.7231 val_metal_acc=0.2692 val_metal_min_recall=0.0000 val_fe_recall=0.4722 val_joint_bal_acc=0.3182 val_joint_macro_f1=0.2035 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top2_trial32_deepmzyme_controlled_hpo_seed43_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_105a1203/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top2_trial32_deepmzyme_controlled_hpo_seed43_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_105a1203/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top2_trial32_deepmzyme_controlled_hpo_seed43_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_105a1203/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top2_trial32_deepmzyme_controlled_hpo_seed43_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_105a1203/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top2_trial32_deepmzyme_controlled_hpo_seed43_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_105a1203/run_metadata.json
================================================================================
[Seed repeat top2 trial32 seed44]
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name top2_trial32_deepmzyme_controlled_hpo_seed44_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_fde44e63 --model-architecture gvp --epochs 1 --batch-size 8 --learning-rate 5.4715836015281065e-05 --weight-decay 0.001 --seed 44 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7834 lr=5.47158e-05 train_metal_acc=0.3957 val_loss=1.7132 val_metal_acc=0.2418 val_metal_min_recall=0.0000 val_fe_recall=0.5278 val_joint_bal_acc=0.3091 val_joint_macro_f1=0.2236 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top2_trial32_deepmzyme_controlled_hpo_seed44_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_fde44e63/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top2_trial32_deepmzyme_controlled_hpo_seed44_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_fde44e63/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top2_trial32_deepmzyme_controlled_hpo_seed44_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_fde44e63/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top2_trial32_deepmzyme_controlled_hpo_seed44_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_fde44e63/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top2_trial32_deepmzyme_controlled_hpo_seed44_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_fde44e63/run_metadata.json
================================================================================
[Seed repeat top3 trial15 seed42]
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name top3_trial15_deepmzyme_controlled_hpo_seed42_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_d290fa70 --model-architecture gvp --epochs 1 --batch-size 8 --learning-rate 7.032630334240692e-05 --weight-decay 0.001 --seed 42 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7554 lr=7.03263e-05 train_metal_acc=0.5344 val_loss=1.5883 val_metal_acc=0.5714 val_metal_min_recall=0.0000 val_fe_recall=0.3939 val_joint_bal_acc=0.3744 val_joint_macro_f1=0.3570 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top3_trial15_deepmzyme_controlled_hpo_seed42_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_d290fa70/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top3_trial15_deepmzyme_controlled_hpo_seed42_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_d290fa70/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top3_trial15_deepmzyme_controlled_hpo_seed42_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_d290fa70/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top3_trial15_deepmzyme_controlled_hpo_seed42_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_d290fa70/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top3_trial15_deepmzyme_controlled_hpo_seed42_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_d290fa70/run_metadata.json
================================================================================
[Seed repeat top3 trial15 seed123]
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name top3_trial15_deepmzyme_controlled_hpo_seed123_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archg_e379bb89 --model-architecture gvp --epochs 1 --batch-size 8 --learning-rate 7.032630334240692e-05 --weight-decay 0.001 --seed 123 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7828 lr=7.03263e-05 train_metal_acc=0.2890 val_loss=1.7145 val_metal_acc=0.1648 val_metal_min_recall=0.0000 val_fe_recall=0.0263 val_joint_bal_acc=0.2727 val_joint_macro_f1=0.1994 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top3_trial15_deepmzyme_controlled_hpo_seed123_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archg_e379bb89/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top3_trial15_deepmzyme_controlled_hpo_seed123_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archg_e379bb89/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top3_trial15_deepmzyme_controlled_hpo_seed123_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archg_e379bb89/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top3_trial15_deepmzyme_controlled_hpo_seed123_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archg_e379bb89/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top3_trial15_deepmzyme_controlled_hpo_seed123_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archg_e379bb89/run_metadata.json
================================================================================
[Seed repeat top3 trial15 seed2026]
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name top3_trial15_deepmzyme_controlled_hpo_seed2026_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_arch_b5449d7b --model-architecture gvp --epochs 1 --batch-size 8 --learning-rate 7.032630334240692e-05 --weight-decay 0.001 --seed 2026 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7771 lr=7.03263e-05 train_metal_acc=0.2997 val_loss=1.7517 val_metal_acc=0.1868 val_metal_min_recall=0.0000 val_fe_recall=0.2857 val_joint_bal_acc=0.3265 val_joint_macro_f1=0.2386 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top3_trial15_deepmzyme_controlled_hpo_seed2026_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_arch_b5449d7b/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top3_trial15_deepmzyme_controlled_hpo_seed2026_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_arch_b5449d7b/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top3_trial15_deepmzyme_controlled_hpo_seed2026_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_arch_b5449d7b/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top3_trial15_deepmzyme_controlled_hpo_seed2026_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_arch_b5449d7b/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top3_trial15_deepmzyme_controlled_hpo_seed2026_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_arch_b5449d7b/run_metadata.json
================================================================================
[Seed repeat top3 trial15 seed43]
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name top3_trial15_deepmzyme_controlled_hpo_seed43_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_7bc4a615 --model-architecture gvp --epochs 1 --batch-size 8 --learning-rate 7.032630334240692e-05 --weight-decay 0.001 --seed 43 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7598 lr=7.03263e-05 train_metal_acc=0.4646 val_loss=1.6997 val_metal_acc=0.3077 val_metal_min_recall=0.0000 val_fe_recall=0.5278 val_joint_bal_acc=0.3562 val_joint_macro_f1=0.2426 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top3_trial15_deepmzyme_controlled_hpo_seed43_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_7bc4a615/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top3_trial15_deepmzyme_controlled_hpo_seed43_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_7bc4a615/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top3_trial15_deepmzyme_controlled_hpo_seed43_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_7bc4a615/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top3_trial15_deepmzyme_controlled_hpo_seed43_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_7bc4a615/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top3_trial15_deepmzyme_controlled_hpo_seed43_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_7bc4a615/run_metadata.json
================================================================================
[Seed repeat top3 trial15 seed44]
PYTHONPATH=/content/DeepMzyme/src:/content/DeepMzyme/src:/content/DeepMzyme/src:/env/python /usr/bin/python3 /content/DeepMzyme/src/train.py --task metal --structure-dir /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train --summary-csv /content/deepmzyme_bundle/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --runs-dir /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1 --run-name top3_trial15_deepmzyme_controlled_hpo_seed44_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_fdfac180 --model-architecture gvp --epochs 1 --batch-size 8 --learning-rate 7.032630334240692e-05 --weight-decay 0.001 --seed 44 --val-fraction 0.15 --split-by pdbid --selection-metric val_metal_balanced_acc --device cuda --node-feature-set conservative --hidden-s 128 --esm-fusion-dim 64 --head-mlp-layers 1 --external-feature-source updated --metal-loss-function cross_entropy --metal-focal-gamma 2.0 --metal-label-smoothing 0.0 --metal-loss-weight 1.0 --ec-loss-weight 1.0 --metal-class-weight-mode inverse_frequency --mn-loss-multiplier 1.0 --cu-loss-multiplier 1.0 --zn-loss-multiplier 1.0 --fe-loss-multiplier 1.0 --co-loss-multiplier 1.0 --ni-loss-multiplier 1.0 --class-viii-loss-multiplier 1.0 --unsupported-metal-policy error --invalid-structure-policy skip --lr-schedule fixed --edge-hidden 128 --gvp-layers 2 --edge-radius 6.0 --node-rbf-sigma 0.75 --edge-rbf-sigma 0.75 --hidden-v 32 --allow-missing-external-features --fusion-mode late_fusion --esm-embeddings-dir /content/deepmzyme_bundle/DeepMzyme_Data/esm_embeddings --no-prepare-missing-esm-embeddings --no-prepare-missing-ring-edges

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

epoch=1 train_loss=1.7741 lr=7.03263e-05 train_metal_acc=0.4210 val_loss=1.6984 val_metal_acc=0.2582 val_metal_min_recall=0.0000 val_fe_recall=0.5278 val_joint_bal_acc=0.3150 val_joint_macro_f1=0.2258 metal_loss_scale=1.0000 ec_loss_scale=1.0000
Saved checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top3_trial15_deepmzyme_controlled_hpo_seed44_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_fdfac180/last_model_checkpoint.pt
Saved best checkpoint to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top3_trial15_deepmzyme_controlled_hpo_seed44_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_fdfac180/best_model_checkpoint.pt
Saved dataset summary to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top3_trial15_deepmzyme_controlled_hpo_seed44_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_fdfac180/dataset_summary.json
Saved run config to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top3_trial15_deepmzyme_controlled_hpo_seed44_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_fdfac180/run_config.json
Saved run metadata to /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top3_trial15_deepmzyme_controlled_hpo_seed44_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_fdfac180/run_metadata.json
Seed-repeat validation summary:
[
  {
    "source_top_rank": 2,
    "trial_number": 32,
    "selection_metric": "val_metal_balanced_acc",
    "n_seeds_completed": 5,
    "mean_validation_balanced_acc": 0.33490255121673934,
    "mean_validation_metric": 0.33490255121673934,
    "std_validation_metric": 0.031422094392314186,
    "min_validation_metric": 0.30906417975383493,
    "max_validation_metric": 0.3724438959733078,
    "result_stage": "seed-repeat validation",
    "seed_repeat_rank_by_mean_validation_metric": 1
  },
  {
    "source_top_rank": 3,
    "trial_number": 15,
    "selection_metric": "val_metal_balanced_acc",
    "n_seeds_completed": 5,
    "mean_validation_balanced_acc": 0.3289684151101894,
    "mean_validation_metric": 0.3289684151101894,
    "std_validation_metric": 0.03928018169981929,
    "min_validation_metric": 0.27270718617467843,
    "max_validation_metric": 0.3744046802870333,
    "result_stage": "seed-repeat validation",
    "seed_repeat_rank_by_mean_validation_metric": 2
  },
  {
    "source_top_rank": 1,
    "trial_number": 49,
    "selection_metric": "val_metal_balanced_acc",
    "n_seeds_completed": 5,
    "mean_validation_balanced_acc": 0.3098742052523086,
    "mean_validation_metric": 0.3098742052523086,
    "std_validation_metric": 0.039841555042331014,
    "min_validation_metric": 0.24705882352941178,
    "max_validation_metric": 0.35392106297278714,
    "result_stage": "seed-repeat validation",
    "seed_repeat_rank_by_mean_validation_metric": 3
  }
]
Seed-repeat results CSV: /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna/deepmzyme_controlled_hpo/seed_repeat_results.csv
Seed-repeat summary CSV: /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna/deepmzyme_controlled_hpo/seed_repeat_summary.csv

Best Optuna trial summary
Best trial number: 49
Best validation metric: 0.6750130535709283
Best parameters: {
  "balance_metal_site_symbols": false,
  "batch_size": 8,
  "class_viii_loss_multiplier": 1.0,
  "co_loss_multiplier": 1.0,
  "cu_loss_multiplier": 1.0,
  "edge_hidden": 128,
  "edge_radius": 6.0,
  "esm_fusion_dim": 64,
  "fe_loss_multiplier": 1.0,
  "gvp_layers": 4,
  "head_mlp_layers": 1,
  "hidden_s": 256,
  "hidden_v": 32,
  "learning_rate": 1.6801503587890522e-05,
  "metal_class_weight_mode": "inverse_frequency",
  "metal_weighting_setup": "inverse_frequency",
  "mn_loss_multiplier": 1.0,
  "ni_loss_multiplier": 1.0,
  "weight_decay": 1e-05,
  "zn_loss_multiplier": 1.0
}
Best run directory: /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0049_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d32997f7
All trials CSV: /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna/deepmzyme_controlled_hpo/all_trials.csv
Top trials CSV: /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna/deepmzyme_controlled_hpo/top_trials.csv
Best trial JSON: /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna/deepmzyme_controlled_hpo/best_trial.json
Best config command: /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna/deepmzyme_controlled_hpo/best_config_command.txt
Top-k reevaluation commands: /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna/deepmzyme_controlled_hpo/top_reevaluation_commands.txt
Study summary: /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna/deepmzyme_controlled_hpo/optuna_study_summary.md
Completed run directories: ['/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0000_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_bf8671dd', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0001_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_6ddff8f9', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0002_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a273d109', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0003_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_67900e48', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0004_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_6b955af4', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0005_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_f07cdf8c', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0006_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d0b2747e', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0007_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_338d57f2', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0008_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_93d46ebc', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0009_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_921d5065', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0010_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_0f0f1cd5', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0011_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_e1ed9295', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0012_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_bce5e2a1', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0013_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_6cd4bd68', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0014_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_064df228', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0015_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a931e009', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0016_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a973315e', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0017_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_0e910c6b', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0018_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_98f0fd36', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0019_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_538dfc3c', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0020_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_04f4992c', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0021_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_87949ab6', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0022_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_abf5e755', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0023_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_f32e42f6', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0024_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c215ce22', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0025_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_5305cf54', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0026_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_3db29e38', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0027_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_e110d321', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0028_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_409c9d6c', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0029_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_7e958f6c', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0030_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a6745f6e', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0031_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c145b928', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0032_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_faf01e7c', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0033_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_62019c53', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0034_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_ed1da180', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0035_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_ea85b3c9', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0036_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c81c6ebc', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0037_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_19d4f091', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0038_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_803373b1', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0039_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_a1d013d0', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0040_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_aed7a382', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0041_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_e1dec6d2', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0042_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_8e2c2aa2', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0043_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_8d03d6e9', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0044_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_2fd12175', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0045_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c23f4d70', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0046_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_2627ca5a', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0047_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_03b5419e', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0048_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_c00511b0', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/optuna_deepmzyme_controlled_hpo_trial0049_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgvp_f_d32997f7', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top1_trial49_deepmzyme_controlled_hpo_seed42_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_a198eee2', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top1_trial49_deepmzyme_controlled_hpo_seed123_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archg_631e5ea0', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top1_trial49_deepmzyme_controlled_hpo_seed2026_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_arch_bcda140d', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top1_trial49_deepmzyme_controlled_hpo_seed43_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_56eb2f7c', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top1_trial49_deepmzyme_controlled_hpo_seed44_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_a1ad5381', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top2_trial32_deepmzyme_controlled_hpo_seed42_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_91620ace', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top2_trial32_deepmzyme_controlled_hpo_seed123_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archg_c3065b84', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top2_trial32_deepmzyme_controlled_hpo_seed2026_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_arch_8c788af7', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top2_trial32_deepmzyme_controlled_hpo_seed43_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_105a1203', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top2_trial32_deepmzyme_controlled_hpo_seed44_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_fde44e63', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top3_trial15_deepmzyme_controlled_hpo_seed42_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_d290fa70', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top3_trial15_deepmzyme_controlled_hpo_seed123_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archg_e379bb89', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top3_trial15_deepmzyme_controlled_hpo_seed2026_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_arch_b5449d7b', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top3_trial15_deepmzyme_controlled_hpo_seed43_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_7bc4a615', '/content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/top3_trial15_deepmzyme_controlled_hpo_seed44_deepmzyme_nonoverlap_baseline_batchmetal_late_fusion_optuna_50_v1_metal_gvp_+_late_fusion_archgv_fdfac180']
Failed run directories: []
Execution records JSON: /content/deepmzyme_outputs/runs/metal_late_fusion_optuna_50_v1/deepmzyme_nonoverlap_model_comparison_execution_records.json
