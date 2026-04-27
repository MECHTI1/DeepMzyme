# DeepMzyme audit against `Plan.md`

## Already correctly implemented

- Validation-aware checkpoint selection already exists when validation is enabled. `src/training/config.py::default_selection_metric_for_task`, `src/training/trask_entrypoint.py::validate_separate_task_config`, and `src/training/run.py::train_and_select_checkpoint` all point toward selecting checkpoints from validation metrics rather than the test set.
- Reproducibility metadata is already being saved. `src/training/run.py::checkpoint_payload` and `src/training/run.py::persist_run_outputs` store config, label maps, normalization stats, dataset summary, and training history.
- Split handling is already stronger than a minimal baseline. `src/training/splits.py` supports grouped validation splits and k-fold splits, and `src/training/preflight.py` checks leakage and missing-class problems.
- Runtime preparation for ESM embeddings, ring edges, and external residue features already exists in `src/training/runtime_preparation.py`.
- Summary-CSV site filtering is already implemented. `src/training/site_filter.py` and `src/training/structure_loading.py::load_structure_pockets` use summary rows to restrict allowed catalytic metal sites and to map site metals conservatively.

## Missing

- There is still no single end-to-end non-ignored script that assembles both the train and test CSV artifacts, validates them against prebuilt split assets, and stages them for release in one pass.
- There is still no true Colab notebook/UI layer in active `src/` code; the repository now has CLI utilities for bundling and summarizing runs, but not a notebook-facing interactive interface.
- There is still no automated publication-style figure generator for comparing screened models and parameters; the current reporting foundation is CSV-based.

## Partially implemented or inconsistent

- `--node-feature-set` is now wired through graph construction and validated, but there is still only one concrete feature set (`conservative`) in `src/data_structures.py`.
- The EC-depth workflow is configurable, but structures whose multiple EC annotations disagree at the requested depth are intentionally left without an EC target rather than being forced into a guessed class.
- Held-out evaluation is now implemented as an explicit post-selection step via `--run-test-eval`, but it depends on the runtime training stack and its installed dependencies rather than a lighter standalone evaluator.
- The current environment used for verification does not have `torch_geometric` installed, so full import/runtime checks for the training stack were not possible here even though the code compiles.

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
- Likely new modules under `src/model_variants/`
- Likely new evaluation/reporting scripts for held-out test execution and comparison outputs
- Likely new non-ignored dataset-preparation scripts for Plan item 1

## Implementation checklist

- [x] Restore the broken model-selection and task-entrypoint surface before adding new behavior
  - Reason: Plan item 4 depends on configurable model types and dedicated task entrypoints, but the current active CLI/import surface is internally broken.
  - Evidence in current code: `src/training/config.py` imports `model_variants` and `model_variants.factory`, but no such package exists in the repository; `src/train_metal.py` and `src/train_ec.py` import `training.task_entrypoint`, but the actual file is `src/training/trask_entrypoint.py`.
  - Likely files: `src/train_metal.py`, `src/train_ec.py`, `src/training/config.py`, `src/training/trask_entrypoint.py`, likely new `src/model_variants/__init__.py`, `src/model_variants/factory.py`
  - Risk level: medium

- [x] Make the metal training label policy match the plan’s 6-class train/validate setup
  - Reason: Plan item 2 says train/validate on 6 classes: Mn, Fe, Zn, Cu, Co, and Ni separately.
  - Evidence in current code: `src/label_schemes.py::active_metal_label_scheme_name` defaults to `split_fe`, which maps Co and Ni into `Class VIII` and therefore does not train the requested 6-way setup by default.
  - Likely files: `src/label_schemes.py`, `src/training/config.py`, `src/training/labels.py`, `src/training/structure_loading.py`
  - Risk level: high

- [x] Add metal held-out test evaluation that reports both 6-class performance and the collapsed 4-class view after checkpoint selection
  - Reason: Plan item 2 requires testing on both the original 6 classes and the 4-class collapsed view where Fe, Co, and Ni are merged into Class VIII, while AGENTS.md requires keeping the test set for final reporting only.
  - Evidence in current code: there is no test loader/evaluator in `src/training/run.py`; `src/training/config.py` advertises collapsed-4 validation metrics, but `src/training/run.py::evaluate_split_metrics` does not compute collapsed-4 metrics anywhere.
  - Likely files: `src/training/run.py`, `src/training/config.py`, `src/label_schemes.py`, likely a new dedicated evaluation script
  - Risk level: high

- [x] Wire the existing parsed hyperparameters into the actual model/optimizer/scheduler construction
  - Reason: Plan item 4 explicitly calls for many parameters to be configurable, but that only helps if the parsed values reach the real training objects.
  - Evidence in current code: `src/training/run.py::prepare_run` only passes a narrow subset of config into `GVPPocketClassifier`; `hidden_s`, `hidden_v`, `edge_hidden`, `esm_fusion_dim`, `node_rbf_sigma`, `edge_rbf_sigma`, `fusion_mode`, `cross_attention_*`, `use_early_esm`, `early_esm_*`, `metal_loss_function`, and scheduler fields are not fully wired through; `mn_loss_multiplier` and `zn_loss_multiplier` are parsed in `src/training/config.py` but never applied in `src/training/run.py`.
  - Likely files: `src/training/run.py`, `src/training/config.py`, `src/model.py`
  - Risk level: medium

- [x] Implement a real model-architecture factory that supports the plan’s model families
  - Reason: Plan item 4 requires GVP+ESM, Only-ESM, Only-GVP, and SimpleGNN+ESM as selectable model types.
  - Evidence in current code: the active codebase only contains `src/model.py::GVPPocketClassifier`; the intended `model_variants` package is missing; `src/training/smoke_test.py` already expects a `build_pocket_classifier(...)` factory that does not exist.
  - Likely files: `src/model.py`, `src/training/config.py`, `src/training/run.py`, `src/training/smoke_test.py`, likely new files under `src/model_variants/`
  - Risk level: high

- [x] Normalize the fusion-mode API to the plan’s named fusion strategies
  - Reason: Plan item 4 asks for explicit fusion-mode experiments: late fusion, early fusion, node-level late fusion, hybrid, and cross-modal attention.
  - Evidence in current code: `src/model.py` already contains useful pieces (`use_early_esm`, gated late fusion, localized cross-attention), but `src/training/config.py` only exposes `fusion_mode` choices `gated` and `cross_attention`, which do not cleanly match the plan’s experiment taxonomy.
  - Likely files: `src/model.py`, `src/training/config.py`, likely `src/model_variants/factory.py`
  - Risk level: high

- [x] Replace top-level-only EC labeling with an explicit, configurable EC-depth policy
  - Reason: Plan item 3 requires learning on the first EC digit and following digits, but it explicitly leaves the exact depth undecided, so the code needs a configurable EC label-depth mechanism rather than a hardcoded assumption.
  - Evidence in current code: `src/training/labels.py::parse_ec_top_level_from_structure_path` returns only the first EC digit from the structure filename; `src/training/structure_loading.py` uses that single value as the EC target.
  - Likely files: `src/training/labels.py`, `src/training/structure_loading.py`, `src/training/config.py`, likely a new EC label utility module
  - Risk level: high

- [x] Add contrastive learning for the EC task in a configurable way
  - Reason: Plan item 3 explicitly says the EC-number model should use contrastive learning.
  - Evidence in current code: there are no contrastive objectives, projections, pair samplers, or contrastive metrics anywhere in active `src/`; EC loss in `src/model.py::_compute_supervised_loss` is plain cross-entropy only.
  - Likely files: `src/model.py`, `src/training/run.py`, `src/training/loop.py`, `src/training/config.py`
  - Risk level: high

- [x] Add held-out EC test evaluation across the configured EC levels
  - Reason: Plan item 3 requires testing on all levels of digits to get a broad performance view, and AGENTS.md requires test reporting to remain separate from model selection.
  - Evidence in current code: there is no test-evaluation path in `src/training/run.py`, and the current label path only supports one EC target per pocket.
  - Likely files: `src/training/run.py`, `src/training/config.py`, `src/training/labels.py`, likely a new evaluation/reporting script
  - Risk level: high

- [x] Make `--node-feature-set` a real graph-construction switch, or remove the dead option until more feature sets exist
  - Reason: Plan item 4 explicitly lists `node_feature_set` as a configurable parameter.
  - Evidence in current code: `src/data_structures.py` defines only one feature set (`conservative`); `src/training/graph_dataset.py` stores no `node_feature_set`; `src/graph/construction.py::pocket_to_pyg_data` does not accept or branch on a feature-set argument.
  - Likely files: `src/data_structures.py`, `src/featurization.py`, `src/graph/construction.py`, `src/training/graph_dataset.py`, `src/training/config.py`
  - Risk level: medium

- [ ] Implement active, non-ignored dataset-preparation code for the train/test CSVs and structure-to-CSV consistency checks
  - Reason: Plan item 1 requires creating training/test CSVs with structure name, EC number(s), and metal type, and it explicitly requires consistency between structure files and CSV labels.
  - Evidence in current code: active `src/` code only consumes a pre-existing summary CSV via `src/training/defaults.py` and `src/training/site_filter.py`; there is no active, non-ignored CSV-generation path in the inspected code surface.
  - Likely files: likely new dataset-preparation scripts under `src/`, possibly `src/training/site_filter.py` for shared validation helpers, possibly `src/project_paths.py`
  - Risk level: high
  - PARTIAL: added `src/build_dataset_csv.py`, which generates a non-ignored structure-level CSV (`structure_name`, `ec_numbers`, `metal_type`) from the active labeling pipeline and validates duplicate/missing rows. Remaining work is a fuller train/test dataset-preparation workflow with explicit consistency validation against existing prebuilt CSVs and split assets.

- [ ] Refactor the Colab bundle/reporting workflow into a configurable experiment interface
  - Reason: Plan item 5 requires a configurable Google Colab training/testing setup plus comparison tables or professional figures summarizing screened parameters and models.
  - Evidence in current code: `src/build_colab_bundle.py` is a fixed bundling script with hardcoded train/test paths and no parameter-screening interface, no test execution flow, and no reporting/figure generation.
  - Likely files: `src/build_colab_bundle.py`, likely new reporting or notebook-facing scripts under `src/`
  - Risk level: medium
  - PARTIAL: `src/build_colab_bundle.py` is now a configurable CLI that can bundle train/test structures plus CSV files, and `src/summarize_runs.py` produces a comparison CSV from run directories. Remaining work is a true Colab-facing input UI and automated publication-style figure generation.
