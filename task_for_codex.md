# DeepMzyme audit against `Plan.md`

## 2026-04-29 update after the Mg fix re-check

- Re-checked the current split contents instead of relying on the previous audit text. The live counts are now:
  - `.data/train_and_test_sets_structures_non_overlapped_pinmymetal/train`: `1304` `.pdb` files
  - `.data/train_and_test_sets_structures_non_overlapped_pinmymetal/test`: `316` `.pdb` files
  - `.data/train_and_test_sets_structures_exact_pinmymetal/train`: `1483` `.pdb` files
  - `.data/train_and_test_sets_structures_exact_pinmymetal/test`: `316` `.pdb` files
- Verified the Mg blocker is resolved in the current split-local catalytic summary CSVs for the main non-overlapped split: the generated release CSVs contain no unsupported `Mg` label and no other unsupported metal labels.
- Found and fixed a real workflow mismatch in `src/build_colab_bundle.py`: the live split uses separate train/test summary CSVs under each split directory, but the bundle CLI previously assumed one shared summary path. The CLI now resolves split-local summary CSVs by default and also accepts separate `--train-summary-csv` / `--test-summary-csv` overrides.
- Re-ran the full non-overlapped CSV-generation workflow on the live split and validated the current outputs:
  - `.data/Colab_Bundles/train_and_test_sets_structures_non_overlapped_pinmymetal/train_and_test_sets_structures_non_overlapped_pinmymetal_train.csv`
  - `.data/Colab_Bundles/train_and_test_sets_structures_non_overlapped_pinmymetal/train_and_test_sets_structures_non_overlapped_pinmymetal_test.csv`
- Verified those non-overlapped CSVs against the live structure directories:
  - required columns are exactly `structure_name`, `ec_numbers`, `metal_type`
  - CSV row counts match structure-file counts exactly: `1304` train rows and `316` test rows
  - there are no missing rows, no unexpected rows, and no empty `ec_numbers`
  - train/test filename overlap is `0`; train/test PDB-ID overlap is also `0`
  - all emitted metal labels stay within the current 6-class policy (`Mn`, `Fe`, `Zn`, `Cu`, `Co`, `Ni`)
  - the non-overlapped release CSVs still contain real multi-metal structures, but they are now represented only with semicolon-joined supported labels; counts are `48` multi-label train rows and `8` multi-label test rows
- Also checked the exact PinMyMetal split as a secondary reference point. It is not suitable as the final held-out split because the current exact train/test directories overlap on `179` full structure filenames and `177` PDB IDs.
- Re-ran lightweight but real held-out evaluation smokes on current post-fix non-overlapped data using a temporary subset under `/tmp/deepmzyme_non_overlap_smoke_20260429`:
  - metal task report: `/tmp/deepmzyme_smoke_runs/metal_non_overlap_post_mg_smoke/test_report.json`
  - EC task report with `--ec-label-depth 2`: `/tmp/deepmzyme_smoke_runs/ec_non_overlap_post_mg_smoke/test_report.json`
  - both runs completed and wrote held-out `test_report.json` files; the metal report includes `test_metal_balanced_acc` and `test_metal_collapsed4_balanced_acc`, and the EC report includes `test_ec_level_1_*` and `test_ec_level_2_*`
- Bundle status:
  - the old `.data/Colab_Bundles/train_and_test_sets_structures_non_overlapped_pinmymetal_colab_bundle_structures.tar.zst` file should be treated as stale/untrusted because listing it during this audit returned `Unexpected EOF`
  - a validated replacement bundle was created at `.data/Colab_Bundles/train_and_test_sets_structures_non_overlapped_pinmymetal_colab_bundle_structures_validated_fast.tar.zst`
  - that validated bundle contains both split directories plus the generated train/test CSV files (`1626` total archive members)
- Checks run:
  - `git status --short`
  - `/home/mechti/miniconda3/envs/DeepMzyme/bin/python -c "import sys; print(sys.executable)"`
  - `/home/mechti/miniconda3/envs/DeepMzyme/bin/python -m py_compile src/build_colab_bundle.py src/build_dataset_csv.py`
  - `PYTHONPATH=src /home/mechti/miniconda3/envs/DeepMzyme/bin/python src/build_colab_bundle.py --dataset-root .data/train_and_test_sets_structures_non_overlapped_pinmymetal --allow-multi-metal-structures --skip-bundle`
  - `PYTHONPATH=src /home/mechti/miniconda3/envs/DeepMzyme/bin/python src/train.py --task metal --structure-dir /tmp/deepmzyme_non_overlap_smoke_20260429/train --summary-csv /tmp/deepmzyme_non_overlap_smoke_20260429/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --test-structure-dir /tmp/deepmzyme_non_overlap_smoke_20260429/test --test-summary-csv /tmp/deepmzyme_non_overlap_smoke_20260429/test/final_data_summarazing_table_transition_metals_only_catalytic.csv --run-test-eval --model-architecture only_gvp --allow-missing-esm-embeddings --no-prepare-missing-esm-embeddings --allow-missing-external-features --epochs 1 --batch-size 4 --val-fraction 0.25 --device cpu --runs-dir /tmp/deepmzyme_smoke_runs --run-name metal_non_overlap_post_mg_smoke`
  - `PYTHONPATH=src /home/mechti/miniconda3/envs/DeepMzyme/bin/python src/train.py --task ec --structure-dir /tmp/deepmzyme_non_overlap_smoke_20260429/train --summary-csv /tmp/deepmzyme_non_overlap_smoke_20260429/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --test-structure-dir /tmp/deepmzyme_non_overlap_smoke_20260429/test --test-summary-csv /tmp/deepmzyme_non_overlap_smoke_20260429/test/final_data_summarazing_table_transition_metals_only_catalytic.csv --run-test-eval --model-architecture only_gvp --allow-missing-esm-embeddings --no-prepare-missing-esm-embeddings --allow-missing-external-features --epochs 1 --batch-size 4 --val-fraction 0.25 --device cpu --ec-label-depth 2 --runs-dir /tmp/deepmzyme_smoke_runs --run-name ec_non_overlap_post_mg_smoke`
  - `tar --use-compress-program='zstd -T0 -3' -cf .data/Colab_Bundles/train_and_test_sets_structures_non_overlapped_pinmymetal_colab_bundle_structures_validated_fast.tar.zst -C /home/mechti/PycharmProjects/DeepMzyme .data/train_and_test_sets_structures_non_overlapped_pinmymetal/train .data/train_and_test_sets_structures_non_overlapped_pinmymetal/test .data/Colab_Bundles/train_and_test_sets_structures_non_overlapped_pinmymetal/train_and_test_sets_structures_non_overlapped_pinmymetal_train.csv .data/Colab_Bundles/train_and_test_sets_structures_non_overlapped_pinmymetal/train_and_test_sets_structures_non_overlapped_pinmymetal_test.csv`

## Already correctly implemented

- Validation-aware checkpoint selection already exists when validation is enabled. `src/training/config.py::default_selection_metric_for_task`, `src/training/trask_entrypoint.py::validate_separate_task_config`, and `src/training/run.py::train_and_select_checkpoint` all point toward selecting checkpoints from validation metrics rather than the test set.
- Reproducibility metadata is already being saved. `src/training/run.py::checkpoint_payload` and `src/training/run.py::persist_run_outputs` store config, label maps, normalization stats, dataset summary, and training history.
- Split handling is already stronger than a minimal baseline. `src/training/splits.py` supports grouped validation splits and k-fold splits, and `src/training/preflight.py` checks leakage and missing-class problems.
- Runtime preparation for ESM embeddings, ring edges, and external residue features already exists in `src/training/runtime_preparation.py`.
- Summary-CSV site filtering is already implemented. `src/training/site_filter.py` and `src/training/structure_loading.py::load_structure_pockets` use summary rows to restrict allowed catalytic metal sites and to map site metals conservatively, and this path now matches the real MAHOMES header schema on disk.

## Missing

- There is still no true Colab notebook/UI layer in active `src/` code; the repository now has CLI utilities for bundling and summarizing runs, but not a notebook-facing interactive interface.
- There is still no automated publication-style figure generator for comparing screened models and parameters; the current reporting foundation is CSV-based.

## Partially implemented or inconsistent

- `--node-feature-set` is now wired through graph construction and validated, but there is still only one concrete feature set (`conservative`) in `src/data_structures.py`.
- The EC-depth workflow is configurable, but structures whose multiple EC annotations disagree at the requested depth are intentionally left without an EC target rather than being forced into a guessed class.
- The end-to-end dataset-release CLI now works on the full non-overlapped split, but the exact PinMyMetal split is still a poor final-evaluation candidate because its current train/test directories overlap.

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

- [x] Implement active, non-ignored dataset-preparation code for the train/test CSVs and structure-to-CSV consistency checks
  - Reason: Plan item 1 requires creating training/test CSVs with structure name, EC number(s), and metal type, and it explicitly requires consistency between structure files and CSV labels.
  - Evidence in current code: active `src/` code only consumes a pre-existing summary CSV via `src/training/defaults.py` and `src/training/site_filter.py`; there is no active, non-ignored CSV-generation path in the inspected code surface.
  - Likely files: `src/build_dataset_csv.py`, `src/build_colab_bundle.py`, `src/training/site_filter.py`, `src/label_schemes.py`
  - DONE: the live non-overlapped split now passes full structure-to-CSV validation on the current data, the generated train/test CSVs match the on-disk `.pdb` files exactly, no unsupported `Mg` label remains, and a validated release bundle was created at `.data/Colab_Bundles/train_and_test_sets_structures_non_overlapped_pinmymetal_colab_bundle_structures_validated_fast.tar.zst`.
  - Risk level: was high; now low
  - Risk change reason: this path was re-run and validated on the full current non-overlapped split rather than being inferred from smaller earlier smoke assets.

- [ ] Refactor the Colab bundle/reporting workflow into a configurable experiment interface
  - Reason: Plan item 5 requires a configurable Google Colab training/testing setup plus comparison tables or professional figures summarizing screened parameters and models.
  - Evidence in current code: `src/build_colab_bundle.py` is a fixed bundling script with hardcoded train/test paths and no parameter-screening interface, no test execution flow, and no reporting/figure generation.
  - Likely files: `src/build_colab_bundle.py`, likely new reporting or notebook-facing scripts under `src/`
  - PARTIAL: `src/build_colab_bundle.py` is now a configurable CLI, it can derive train/test CSVs from a split root using split-local train/test summary CSVs, and the non-overlapped release bundle path was validated on the live dataset, but there is still no true notebook-facing experiment UI or figure-generation layer.
  - Risk level: medium
  - Risk change reason: lowered only for the CLI release path because the bundle workflow now ran end to end on real split-derived files; the missing notebook/reporting layer is unchanged.
