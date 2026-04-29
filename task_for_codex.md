# DeepMzyme audit against `Plan.md`

## 2026-04-29 update after the new `test/` directories were added

- Verified that `.data/train_and_test_sets_structures_exact_pinmymetal/test` and `.data/train_and_test_sets_structures_non_overlapped_pinmymetal/test` both exist and contain `334` `.pdb` files; the two test directories are identical at the filename level.
- Verified that the paired train directories exist and currently contain `1738` `.pdb` files for the exact split and `1552` `.pdb` files for the non-overlapped split.
- Updated the code so the real MAHOMES catalytic summary CSV format now loads correctly: `src/training/site_filter.py` accepts the on-disk headers (`structure`, `chain_resi`, `ecnumber`, `metaltype`) and filters catalytic rows when `whether_catalytic` is present.
- Updated `src/label_schemes.py` so summary-site labels such as `FE2` normalize into the existing `Fe` class rather than failing the real-data load path.
- Updated `src/build_dataset_csv.py` so it validates structure-directory coverage and can emit semicolon-joined structure labels for genuinely multi-metal structures when `--allow-multi-metal-structures` is enabled.
- Updated `src/build_colab_bundle.py` into the missing end-to-end CLI that can derive train/test CSVs from a dataset root and then bundle the structures plus generated CSVs in one pass.
- Checks run:
  - `/home/mechti/miniconda3/envs/DeepMzyme/bin/python -m py_compile src/label_schemes.py src/training/site_filter.py src/build_dataset_csv.py src/build_colab_bundle.py`
  - `load_labeled_pockets_with_report_from_dir(...)` against `.data/train_and_test_sets_structures_exact_pinmymetal/test` with the real catalytic summary CSV; this succeeded and returned real pockets.
  - A real-data metal smoke run with validation plus held-out evaluation wrote `.data/real_eval_smoke/runs/metal_test_eval_exact_smoke/test_report.json`, including both `test_metal_balanced_acc` and `test_metal_collapsed4_balanced_acc`.
  - A real-data EC smoke run with `--ec-label-depth 2` plus held-out evaluation wrote `.data/real_eval_smoke/runs/ec_test_eval_exact_smoke/test_report.json`, including both `test_ec_level_1_*` and `test_ec_level_2_*` metrics.
  - `src/build_colab_bundle.py` completed end to end on `.data/real_eval_smoke/exact_supported`, producing `.data/real_eval_smoke/exact_supported_bundle.tar.zst`.
- Remaining data blocker for Plan item 1:
  - Full CSV staging for the complete exact/non-overlapped train roots is still not cleanly completable under the current 6-class transition-metal policy because the train splits still contain Mg-bearing structures from the MAHOMES catalytic summary.
  - Summary-based counts show `29` exact-train and `26` non-overlapped-train structures are Mg-only, and another `68` exact-train and `62` non-overlapped-train structures mix Mg with one of the supported transition-metal labels. Those cases need a project decision: exclude them from the release CSVs, represent multiple metal types explicitly including Mg, or expand the label policy beyond the current 6-class scheme.

## Already correctly implemented

- Validation-aware checkpoint selection already exists when validation is enabled. `src/training/config.py::default_selection_metric_for_task`, `src/training/trask_entrypoint.py::validate_separate_task_config`, and `src/training/run.py::train_and_select_checkpoint` all point toward selecting checkpoints from validation metrics rather than the test set.
- Reproducibility metadata is already being saved. `src/training/run.py::checkpoint_payload` and `src/training/run.py::persist_run_outputs` store config, label maps, normalization stats, dataset summary, and training history.
- Split handling is already stronger than a minimal baseline. `src/training/splits.py` supports grouped validation splits and k-fold splits, and `src/training/preflight.py` checks leakage and missing-class problems.
- Runtime preparation for ESM embeddings, ring edges, and external residue features already exists in `src/training/runtime_preparation.py`.
- Summary-CSV site filtering is already implemented. `src/training/site_filter.py` and `src/training/structure_loading.py::load_structure_pockets` use summary rows to restrict allowed catalytic metal sites and to map site metals conservatively, and this path now matches the real MAHOMES header schema on disk.

## Missing

- There is still no true Colab notebook/UI layer in active `src/` code; the repository now has CLI utilities for bundling and summarizing runs, but not a notebook-facing interactive interface.
- There is still no automated publication-style figure generator for comparing screened models and parameters; the current reporting foundation is CSV-based.
- The project still lacks an agreed release policy for Mg-bearing structures that appear in the exact and non-overlapped train splits, so Plan item 1 cannot yet be marked fully complete for the full split assets.

## Partially implemented or inconsistent

- `--node-feature-set` is now wired through graph construction and validated, but there is still only one concrete feature set (`conservative`) in `src/data_structures.py`.
- The EC-depth workflow is configurable, but structures whose multiple EC annotations disagree at the requested depth are intentionally left without an EC target rather than being forced into a guessed class.
- The end-to-end dataset-release CLI now exists, but it still stops on the full exact/non-overlapped train roots because the current split contents and the current 6-class metal policy disagree on how Mg-bearing structures should be represented.

## Likely files needing edits

- `src/train_metal.py`
- `src/train_ec.py`
- `src/training/config.py`
- `src/training/run.py`
- `src/training/trask_entrypoint.py`
- `src/model.py`
- `src/label_schemes.py`
- `src/training/labels.py`
- `src/training/structure_loading.py`
- `src/training/graph_dataset.py`
- `src/graph/construction.py`
- `src/build_colab_bundle.py`
- `src/build_dataset_csv.py`
- `src/training/site_filter.py`
- Likely new modules under `src/model_variants/`
- Likely new evaluation/reporting scripts for held-out test execution and comparison outputs
- Likely new non-ignored dataset-preparation scripts for Plan item 1

## Implementation checklist

- [x] Restore the broken model-selection and task-entrypoint surface before adding new behavior
  - Reason: Plan item 4 depends on configurable model types and dedicated task entrypoints, but the current active CLI/import surface is internally broken.
  - Evidence in current code: `src/training/config.py` imports `model_variants` and `model_variants.factory`, but no such package exists in the repository; `src/train_metal.py` and `src/train_ec.py` import `training.task_entrypoint`, but the actual file is `src/training/trask_entrypoint.py`.
  - Likely files: `src/train_metal.py`, `src/train_ec.py`, `src/training/config.py`, `src/training/trask_entrypoint.py`, likely new `src/model_variants/__init__.py`, `src/model_variants/factory.py`
  - DONE: the compatibility shim and model factory package are present, import checks passed for `training.task_entrypoint` and `model_variants.factory`, and `--help` works for `src/train.py`, `src/train_metal.py`, and `src/train_ec.py`.
  - Risk level: was medium; now low
  - Risk change reason: this surface was verified by syntax/import checks and CLI entrypoint checks rather than code inspection alone.

- [x] Make the metal training label policy match the plan’s 6-class train/validate setup
  - Reason: Plan item 2 says train/validate on 6 classes: Mn, Fe, Zn, Cu, Co, and Ni separately.
  - Evidence in current code: `src/label_schemes.py::active_metal_label_scheme_name` defaults to `split_fe`, which maps Co and Ni into `Class VIII` and therefore does not train the requested 6-way setup by default.
  - Likely files: `src/label_schemes.py`, `src/training/config.py`, `src/training/labels.py`, `src/training/structure_loading.py`
  - DONE: the active default label policy now targets the 6-class metal setup, and the current training/model path is internally consistent with `n_metal=6`.
  - Risk level: was high; now medium
  - Risk change reason: class-count wiring and output shapes were lightly verified, but real labeled data loading was not exercised.

- [x] Add metal held-out test evaluation that reports both 6-class performance and the collapsed 4-class view after checkpoint selection
  - Reason: Plan item 2 requires testing on both the original 6 classes and the 4-class collapsed view where Fe, Co, and Ni are merged into Class VIII, while AGENTS.md requires keeping the test set for final reporting only.
  - Evidence in current code: there is no test loader/evaluator in `src/training/run.py`; `src/training/config.py` advertises collapsed-4 validation metrics, but `src/training/run.py::evaluate_split_metrics` does not compute collapsed-4 metrics anywhere.
  - Likely files: `src/training/run.py`, `src/training/config.py`, `src/label_schemes.py`, likely a new dedicated evaluation script
  - DONE: a real-data smoke run against `.data/real_eval_smoke/exact_supported/test` wrote `.data/real_eval_smoke/runs/metal_test_eval_exact_smoke/test_report.json`, and that report contains both the 6-class metrics and the collapsed-4 metrics after validation-based checkpoint selection.
  - Risk level: was high; now low
  - Risk change reason: the held-out path was exercised end to end on real split-derived test files, not only by inspection.

- [x] Wire the existing parsed hyperparameters into the actual model/optimizer/scheduler construction
  - Reason: Plan item 4 explicitly calls for many parameters to be configurable, but that only helps if the parsed values reach the real training objects.
  - Evidence in current code: `src/training/run.py::prepare_run` only passes a narrow subset of config into `GVPPocketClassifier`; `hidden_s`, `hidden_v`, `edge_hidden`, `esm_fusion_dim`, `node_rbf_sigma`, `edge_rbf_sigma`, `fusion_mode`, `cross_attention_*`, `use_early_esm`, `early_esm_*`, `metal_loss_function`, and scheduler fields are not fully wired through; `mn_loss_multiplier` and `zn_loss_multiplier` are parsed in `src/training/config.py` but never applied in `src/training/run.py`.
  - Likely files: `src/training/run.py`, `src/training/config.py`, `src/model.py`
  - DONE: the parsed config is wired into model construction and training setup, and the previously parsed-but-unused `save_epoch_checkpoints` path is now implemented.
  - Risk level: medium
  - Risk change reason: unchanged because the wiring was inspected carefully and partially exercised by the synthetic smoke test, but not every scheduler and training option was runtime-tested.

- [x] Implement a real model-architecture factory that supports the plan’s model families
  - Reason: Plan item 4 requires GVP+ESM, Only-ESM, Only-GVP, and SimpleGNN+ESM as selectable model types.
  - Evidence in current code: the active codebase only contains `src/model.py::GVPPocketClassifier`; the intended `model_variants` package is missing; `src/training/smoke_test.py` already expects a `build_pocket_classifier(...)` factory that does not exist.
  - Likely files: `src/model.py`, `src/training/config.py`, `src/training/run.py`, `src/training/smoke_test.py`, likely new files under `src/model_variants/`
  - DONE: `build_pocket_classifier(...)` exists and all configured architectures (`gvp`, `only_esm`, `only_gvp`, `simple_gnn_esm`) passed a tiny synthetic CPU train/predict smoke test.
  - Risk level: was high; now low
  - Risk change reason: this task was verified by actual model construction plus one-step runtime execution for every configured architecture.

- [x] Normalize the fusion-mode API to the plan’s named fusion strategies
  - Reason: Plan item 4 asks for explicit fusion-mode experiments: late fusion, early fusion, node-level late fusion, hybrid, and cross-modal attention.
  - Evidence in current code: `src/model.py` already contains useful pieces (`use_early_esm`, gated late fusion, localized cross-attention), but `src/training/config.py` only exposes `fusion_mode` choices `gated` and `cross_attention`, which do not cleanly match the plan’s experiment taxonomy.
  - Likely files: `src/model.py`, `src/training/config.py`, likely `src/model_variants/factory.py`
  - DONE: the exposed fusion-mode API now matches the plan taxonomy and the classifier default was corrected to `late_fusion`.
  - Risk level: was high; now medium
  - Risk change reason: CLI/config exposure and default handling were verified, but each fusion behavior was not exercised separately at runtime.

- [x] Replace top-level-only EC labeling with an explicit, configurable EC-depth policy
  - Reason: Plan item 3 requires learning on the first EC digit and following digits, but it explicitly leaves the exact depth undecided, so the code needs a configurable EC label-depth mechanism rather than a hardcoded assumption.
  - Evidence in current code: `src/training/labels.py::parse_ec_top_level_from_structure_path` returns only the first EC digit from the structure filename; `src/training/structure_loading.py` uses that single value as the EC target.
  - Likely files: `src/training/labels.py`, `src/training/structure_loading.py`, `src/training/config.py`, likely a new EC label utility module
  - DONE: the real-data EC path was exercised with `--ec-label-depth 2` on split-derived files, and the resulting held-out report includes both level-1 and level-2 EC metrics.
  - Risk level: was high; now low
  - Risk change reason: the runtime check reached real parsed structures and real held-out reporting rather than stopping at code inspection.

- [ ] Add contrastive learning for the EC task in a configurable way
  - Reason: Plan item 3 explicitly says the EC-number model should use contrastive learning.
  - Evidence in current code: there are no contrastive objectives, projections, pair samplers, or contrastive metrics anywhere in active `src/`; EC loss in `src/model.py::_compute_supervised_loss` is plain cross-entropy only.
  - Likely files: `src/model.py`, `src/training/run.py`, `src/training/loop.py`, `src/training/config.py`
  - REVIEW_NEEDED: contrastive-loss configuration and loss wiring are present, but that branch was not enabled or exercised in the synthetic smoke test.
  - Risk level: was high; now medium
  - Risk change reason: the code exists and compiles, but behavior under nonzero contrastive settings remains unverified.

- [x] Add held-out EC test evaluation across the configured EC levels
  - Reason: Plan item 3 requires testing on all levels of digits to get a broad performance view, and AGENTS.md requires test reporting to remain separate from model selection.
  - Evidence in current code: there is no test-evaluation path in `src/training/run.py`, and the current label path only supports one EC target per pocket.
  - Likely files: `src/training/run.py`, `src/training/config.py`, `src/training/labels.py`, likely a new evaluation/reporting script
  - DONE: a real-data smoke run against `.data/real_eval_smoke/exact_supported/test` wrote `.data/real_eval_smoke/runs/ec_test_eval_exact_smoke/test_report.json`, and that report contains `test_ec_level_1_*` and `test_ec_level_2_*` metrics after validation-based checkpoint selection.
  - Risk level: was high; now low
  - Risk change reason: the held-out evaluation path was exercised end to end on real split-derived files.

- [x] Make `--node-feature-set` a real graph-construction switch, or remove the dead option until more feature sets exist
  - Reason: Plan item 4 explicitly lists `node_feature_set` as a configurable parameter.
  - Evidence in current code: `src/data_structures.py` defines only one feature set (`conservative`); `src/training/graph_dataset.py` stores no `node_feature_set`; `src/graph/construction.py::pocket_to_pyg_data` does not accept or branch on a feature-set argument.
  - Likely files: `src/data_structures.py`, `src/featurization.py`, `src/graph/construction.py`, `src/training/graph_dataset.py`, `src/training/config.py`
  - DONE: `node_feature_set` is passed through featurization, graph construction, graph dataset preparation, preflight validation, and runtime configuration for the active `conservative` feature set.
  - Risk level: was medium; now low
  - Risk change reason: the active path was lightly verified through compile/import checks and the synthetic GVP-based smoke test, even though only one feature-set choice exists.

- [ ] Implement active, non-ignored dataset-preparation code for the train/test CSVs and structure-to-CSV consistency checks
  - Reason: Plan item 1 requires creating training/test CSVs with structure name, EC number(s), and metal type, and it explicitly requires consistency between structure files and CSV labels.
  - Evidence in current code: active `src/` code only consumes a pre-existing summary CSV via `src/training/defaults.py` and `src/training/site_filter.py`; there is no active, non-ignored CSV-generation path in the inspected code surface.
  - Likely files: `src/build_dataset_csv.py`, `src/build_colab_bundle.py`, `src/training/site_filter.py`, `src/label_schemes.py`
  - PARTIAL: the missing end-to-end CLI now exists and was verified on `.data/real_eval_smoke/exact_supported`, but the full exact/non-overlapped train roots still fail strict structure-to-CSV validation because Mg-only and mixed Mg+transition-metal structures do not fit the current 6-class release assumption.
  - Risk level: was high; now medium
  - Risk change reason: the code path is now real and verified, but the full dataset assets still need a project-level labeling/exclusion decision before this item can be closed.

- [ ] Refactor the Colab bundle/reporting workflow into a configurable experiment interface
  - Reason: Plan item 5 requires a configurable Google Colab training/testing setup plus comparison tables or professional figures summarizing screened parameters and models.
  - Evidence in current code: `src/build_colab_bundle.py` is a fixed bundling script with hardcoded train/test paths and no parameter-screening interface, no test execution flow, and no reporting/figure generation.
  - Likely files: `src/build_colab_bundle.py`, likely new reporting or notebook-facing scripts under `src/`
  - PARTIAL: `src/build_colab_bundle.py` is now a configurable CLI, it can derive train/test CSVs from a split root, and it was runtime-verified on `.data/real_eval_smoke/exact_supported`, but there is still no true notebook-facing experiment UI or figure-generation layer.
  - Risk level: medium
  - Risk change reason: lowered only for the CLI release path because the bundle workflow now ran end to end on real split-derived files; the missing notebook/reporting layer is unchanged.
