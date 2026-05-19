# DeepMzyme — Research Design and Technical Authority

DeepMzyme is a deep-learning framework for predicting metal type and enzyme class
(EC number) from metalloenzyme structural pocket graphs and optional ESMC residue
embeddings.

This document is the primary design and policy authority. Source code and run
outputs are evidence of implemented behavior; this document states the intended
architecture, training policy, and experiment governance. When any other file
conflicts with this document, prefer this document unless the source code
clearly contains newer working logic that should be preserved.

**Where to find related information:**

| Need | Go to |
| --- | --- |
| Documentation index, validation/testing order, and output folder map | `docs/README.md` |
| Current experiment progress and next planned action | `EXPERIMENT_STATUS.md` |
| Notebook workflow and option reference | `docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md` |
| Copy-paste-ready metal training stages | `docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md` |
| G4-class Optuna policy and exact stage budgets | `docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md` ("G4-Class Optuna Policy") |
| Copy-paste-ready EC training stages | `docs/EC_TRAINING_PIPELINE_PLAYBOOK.md` |
| Raw experiment results | `docs/notebook_outputs/` |
| Current best-configuration snapshot | `docs/notebook_outputs/summaries/LEADERBOARD.md` |
| CLI command examples | `list_train_commands.md` |
| Public-facing overview and quick-start | `README.md` |

---

## 1) Training/Test Sets

The primary data source is the PinMyMetal train/test split, converted to
site-level summary CSVs compatible with the MAHOMES format. Each row represents
one catalytic metal site.

CSV format: structure name, EC number(s), metal type.

Site-level MAHOMES summary CSVs are the training source of truth. Structure-level
CSV artifacts may contain semicolon-joined metal labels for structures with
multiple catalytic metal sites; these are for inspection only and must not
replace site-level labels for single-label metal training.

Data integrity rule: the only metal types present in structure files must match
those in the CSV exactly, and vice versa.

Preparation scripts live under `prepare_training_and_test_set/`. These scripts
download structures, create non-redundant chain-level files, and run MAHOMES
activation to produce the site-level summary CSVs used for training.

Colab data bundles are built with `src/build_colab_bundle.py`. The bundle packs
the site-level summary CSVs and structure files into a compressed archive for
upload to HuggingFace or for local use. The notebook consumes this bundle via
`COLAB_DATA_SOURCE`.

---

## 2) Train the Metal-Classification Model

Train and validate on six metal classes: Mn, Fe, Zn, Cu, Co, Ni.

Best checkpoint selection: highest validation balanced accuracy
(`val_metal_balanced_acc`).

The default and main reportable target scheme is six-class. An explicit
five-class scheme is available for validation-only comparison runs:
`--metal-label-scheme five_class` in the CLI or
`METAL_LABEL_SCHEME = "five_class"` in the notebook. It keeps Mn, Cu, Zn, and
Fe separate while grouping Co and Ni into the fifth class. Use a separate run
name and separate Optuna study when changing the metal label scheme, and do not
compare five-class validation numbers directly against six-class validation
numbers without labeling the scheme.

Final test reporting: six-class metrics and collapsed-4 metrics, where Fe, Co,
and Ni are merged into Class VIII.

Six-class metal classification remains the main task. Collapsed-4 is an
auxiliary/supplemental view only: it may be used as an optional validation-only
auxiliary loss experiment, but it must not replace six-class metrics, six-class
confusion matrices, or six-class rare-class recall checks.

For the staged training pipeline (smoke, baseline, HPO, grouped-fold
confirmation, final test) with copy-paste notebook configuration blocks, use
`docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`.

### Canonical Colab metal-training pipeline

The canonical metal-training workflow is
`notebooks/DeepMzyme_training_colab.ipynb` driven by the staged blocks in
`docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`. The pipeline has eight stages with
explicit decision gates: Stage 0 (environment/data readiness), Stage 1
(1-epoch smoke), Stage 2A (Only-GVP validation anchor), Stage 2B (baseline
family comparison), Stage 3 (Optuna plumbing debug), Stage 4 (optional medium
per-family Optuna), Stage 5A-5F (serious per-family HPO), Stage 5G
(RING/radius-only ablation), Stage 6 (top-K seed/split confirmation), and
Stage 7 (one-shot held-out test).

Authoritative rules for the pipeline:

- One `MODEL_PRESET` per Optuna study. Optuna never compares model families.
- Hardware target is a G4-class GPU; the playbook defines exact budgets,
  storage, search spaces, seed lists, and decision gates.
- No held-out test evaluation before Stage 7 and no Stage 7 launch without
  Stage 6 grouped-fold confirmation evidence.
- Stage 7 remains a one-shot held-out test event for a fixed
  validation-selected configuration. Optional ensemble, calibration,
  temperature-scaling, plot, or confidence-interval outputs are reporting
  additions only and must not feed back into model/configuration/checkpoint
  selection.
- Stage 6 model promotion uses paired comparisons over shared validation
  folds/splits, paired bootstrap 95% confidence intervals, and rare-class
  recall protection. Raw validation deltas alone are not sufficient promotion
  evidence.
- Optional multi-objective HPO may be used as validation-only rare-class
  protection tooling. Its primary objectives are `val_metal_balanced_acc` and
  active metal-scheme `val_metal_min_recall`; for default reportable runs that
  is six-class minimum recall. Collapsed-4 recall is supplemental and must not
  hide Fe/Co/Ni failures.
- Serious validation-only metal Optuna searches should keep the current
  validated batch size in scope and compare the next larger practical batch
  size; reserve very small batches for smoke/debug or memory fallback, and
  reserve much larger batches for explicitly labeled ablations.
- The advanced fusion order is Stage 5C -> Stage 5D -> Stage 5E -> Stage 5F,
  gated by validation evidence and thresholds defined in the playbook.

### Metal Colab Parameter Ownership Rule

Exact notebook parameter values for the metal-training pipeline must live only
in `docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`.

`Plan.md` defines the research policy and stage ordering, including
validation-only selection, held-out-test protection, `val_metal_balanced_acc`
as the metal-selection metric, grouped validation splitting by `pdbid`, one
`MODEL_PRESET` per Optuna study, baseline-first family promotion, and the
advanced-fusion gate. It must not duplicate full stage configuration blocks or
exact stage values.

When a stage budget, Optuna search space, Stage 6 confirmation policy, or
final-test configuration changes, update the files in this order:

1. `docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md` - exact executable values.
2. `docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md` - option explanation or
   crosswalk, only if the meaning or stage mapping changed.
3. `AGENTS.md` - agent instructions, only if the expected agent behavior
   changed.
4. `Plan.md` - only if the research policy or stage ordering changed.

Do not add full Stage 0-7 blocks to `Plan.md`.

---

## 3) Train the EC-Number Classification Model

Train and validate on all EC first-digit classes, and progressively on deeper
EC digits. The decision on maximum training depth is open; start with depth 1
(`--ec-label-depth 1`) and expand to deeper digits after the depth-1 baseline
is stable.

Use supervised contrastive learning as a secondary loss. Start with
`--ec-contrastive-weight 0.0` for the clean baseline; explore non-zero
contrastive weight only after the supervised baseline is validated.

EC supervision is structure/protein-level even when extraction creates multiple
separated metal-pocket samples for the same structure. Use group weighting at
`structure_id` to avoid over-counting such structures.

Final test reporting: level-1 balanced accuracy, macro F1, and per-class recall
at each trained depth. Report deeper-level metrics when deeper depths are trained.

For the staged training pipeline with copy-paste notebook configuration blocks,
use `docs/EC_TRAINING_PIPELINE_PLAYBOOK.md`.

---

## 4) Make Important Parameters and Model Types Configurable

The main training entry point is `src/train.py`, which delegates to
`src/training/config.py` and `src/training/task_entrypoint.py`. Task-specific
thin wrappers `src/train_metal.py` and `src/train_ec.py` are available for
single-task invocations that bypass the joint-task dispatch.

The Colab notebook (`notebooks/DeepMzyme_training_colab.ipynb`) exposes the
commonly used controls and documents the rest clearly enough that advanced users
can reproduce a command-line run.

#### Supported configurable training parameters

| Area | Parameter / CLI flag | Supported values or default | Plain-language meaning | Colab status |
| --- | --- | --- | --- | --- |
| Data paths | `--structure-dir` | train structure directory | Directory containing training `.pdb`, `.cif`, or `.mmcif` structures. | Expose |
| Data paths | `--summary-csv` | train site-level summary CSV | MAHOMES-style site-level labels used for training. | Expose |
| Data paths | `--test-structure-dir` | optional test structure directory | Held-out test structures, used only when `--run-test-eval` is enabled. | Expose |
| Data paths | `--test-summary-csv` | optional test summary CSV | Held-out site-level labels paired with `--test-structure-dir`. | Expose |
| Output/reporting | `--runs-dir` | output root | Parent directory for all run folders and reports. | Expose |
| Output/reporting | `--run-name` | optional | Human-readable run folder name; auto-generated if blank. | Expose |
| Output/reporting | `--run-test-eval` | off by default in CLI | Runs held-out test reporting for the validation-selected checkpoint. | Expose with warnings |
| Output/reporting | `--selection-metric` | task-dependent default | Metric used to select the best checkpoint. Use validation metrics for real comparisons. | Expose |
| Output/reporting | `--save-epoch-checkpoints` | false | Save every epoch checkpoint, not only the selected/best checkpoint. | Advanced |
| Output/reporting | `--allow-train-loss-test-eval-debug` | false | Debug-only override allowing held-out test evaluation without validation selection. | Advanced warning |
| Runtime | `--device` | `cpu` | PyTorch device such as `cuda` or `cpu`. | Expose |
| Runtime | `--deterministic` | false | Enables stricter deterministic PyTorch behavior for reproducibility, possibly slower. | Expose |
| Runtime | `--num-workers` | `0` | Number of DataLoader worker processes. Default preserves single-process loading. | Advanced |
| Runtime | `--pin-memory` | false | Enables pinned DataLoader host memory only for CUDA runs. CPU runs ignore it. | Advanced |
| Task | `--task` | `joint`; choices `joint`, `metal`, `ec` | Selects metal-only, EC-only, or joint prediction heads and losses. | Expose |
| Target labels | `--metal-label-scheme` | `split_all_metals`; aliases `six_class`, `five_class`, `four_class` | Selects metal target classes. `five_class` means Mn/Cu/Zn/Fe plus grouped Co/Ni. Changing this creates a different prediction problem. | Expose |
| Training | `--epochs` | `10` | Maximum number of training epochs. | Expose |
| Training | `--batch-size` | `8` | Number of pocket graphs per mini-batch. | Expose / sweep |
| Training | `--learning-rate` | `3e-4` | Optimizer step size. Previous serious baselines often start at `3e-5`. | Expose / sweep |
| Training | `--grad-clip-norm` | `1.0` | Gradient clipping max norm. Values `<= 0` disable clipping. | Advanced |
| Training | `--amp` | false | Optional CUDA automatic mixed precision for training only; evaluation stays FP32. | Advanced |
| Training | `--grad-accum-steps` | `1` | Number of mini-batches accumulated before each optimizer step. | Advanced |
| Training | `--weight-decay` | `1e-4` | L2-style optimizer regularization. | Expose / sweep |
| Training | `--seed` | `42` | Random seed for split/sampling/model initialization. | Expose / sweep |
| Training | `--lr-schedule` | `fixed`; choices `fixed`, `cosine`, `step` | Learning-rate schedule. | Expose / sweep |
| Training | `--lr-step-size` | `0`; required positive for `step` | Epoch interval for step LR decay. | Expose |
| Training | `--lr-decay-gamma` | `0.5` | Multiplicative LR decay for step schedule. | Expose |
| Split/validation | `--val-fraction` | `0.0` in CLI | Fraction of training data reserved for validation when not using folds. Real model selection should use validation. | Expose |
| Split/validation | `--split-by` | `pdbid`; choices `pdbid`, `pdbid_chain`, `structure_id`, `pocket_id` | Group identity used to avoid leakage when splitting train/validation. | Expose |
| Split/validation | `--n-folds`, `--fold-index` | optional pair | Enables one fold of grouped cross-validation instead of a simple validation fraction. | Advanced |
| Data policy | `--unsupported-metal-policy` | `error`; choices `error`, `skip` | Whether unsupported metal labels should fail or be skipped during loading. | Advanced |
| Data policy | `--invalid-structure-policy` | `skip`; choices `error`, `skip` | Whether unreadable/invalid structures should fail or be skipped. | Advanced |
| Data policy | `--require-all-task-classes` | false | Fail if the training split lacks a class needed by the selected task. | Advanced |
| Model family | `--model-architecture` | `gvp`; choices `gvp`, `only_esm`, `only_gvp`, `simple_gnn_esm` | Selects the graph/ESM architecture family. | Expose |
| Model size | `--hidden-s` | `128` | Scalar hidden channel width used by GVP/GNN and classifier projections. | Expose / sweep |
| Model size | `--hidden-v` | `16` | Vector hidden channel width for GVP models. Ignored by non-GVP variants. | Expose / sweep |
| Model size | `--edge-hidden` | `64` | Hidden width for encoded edge features. | Expose / sweep |
| Model size | `--gvp-layers` | `4` | Number of graph message-passing layers. Default/recommended search spaces should cap this at 4 unless a deeper-depth ablation is explicitly labeled. | Expose / sweep |
| Model size | `--head-mlp-layers` | `2` | Number of linear layers in metal/EC classifier heads. | Expose / sweep |
| Graph construction | `--edge-radius` | project default currently `8.0` in code | Residue-neighbor radius for graph edges before optional RING edges. | Expose / sweep |
| Node/edge encoders | `--node-feature-set` | `conservative` only | Named set of residue/node features. Only `conservative` is currently implemented. | Expose |
| Node/edge encoders | `--node-rbf-sigma` | `0.75` | Width of distance radial-basis features for node distance features. | Advanced |
| Node/edge encoders | `--edge-rbf-sigma` | `0.75` | Width of distance radial-basis features for edge distance features. | Advanced |
| Node/edge encoders | `--node-rbf-use-raw-distances` | false | Uses raw, unnormalized node distances for node RBF expansion when available. | Advanced |
| Training augmentation | `--position-noise-std` | `0.0` | Training-only Gaussian coordinate noise. Validation and held-out test graphs stay unaugmented. | Advanced / optional sweep |
| Training augmentation | `--second-shell-dropout` | `0.0` | Training-only dropout probability for second-shell residues. Labels and cached source structures are unchanged. | Advanced / optional sweep |
| ESM inputs | `--esm-embeddings-dir` | optional path | Directory containing precomputed ESMC residue embeddings. Needed by ESM-using models unless generation/missing behavior is enabled. | Expose |
| ESM inputs | `--esm-dim` | code default ESMC dimension | Expected dimension of residue ESM embeddings. | Advanced |
| ESM inputs | `--allow-missing-esm-embeddings` | false | Allows ESM-using runs to continue when embeddings are missing; use only for explicit debugging/ablation. | Expose with warning |
| ESM inputs | `--no-prepare-missing-esm-embeddings` | false | Disables automatic generation of missing ESM embeddings. | Expose as prepare-missing toggle |
| ESM inputs | `--disable-esm-branch` | false | Disables late ESM branch for compatible graph models. Usually prefer `only_gvp` for graph-only baseline. | Advanced |
| External features | `--external-features-root-dir` | optional path | Root directory for residue-level external features such as updated SASA/electrostatics. | Advanced |
| External features | `--external-feature-source` | `auto`; choices `auto`, `bluues_rosetta`, `updated` | Selects which external feature layout/source to read. | Advanced |
| External features | `--allow-missing-external-features` | false | Allows training if external feature files are missing, filling defaults where possible. | Expose |
| ESM fusion | `--fusion-mode` | `late_fusion`; choices `late_fusion`, `early_fusion`, `node_level_late_fusion`, `hybrid`, `cross_modal_attention` | Controls where ESM information is combined with graph states. | Expose via presets |
| ESM fusion | `--esm-fusion-dim` | `128` | Projection width for graph-level ESM pooling/fusion. | Expose / sweep |
| Early ESM | `--use-early-esm` | false | Adds residue-level ESM features before graph message passing. Automatically implied by early/hybrid fusion presets. | Preset/advanced |
| Early ESM | `--early-esm-dim` | `32` | Bottleneck dimension for early residue-level ESM projection. | Expose |
| Early ESM | `--early-esm-dropout` | `0.2` | Dropout in the early ESM projection. | Expose |
| Early ESM | `--early-esm-raw` | false | Uses raw full-size ESM vectors as early node features; high-dimensional ablation. | Advanced warning |
| Early ESM | `--early-esm-scope` | `all`; choices `all`, `first_shell`, `first_second_shell` | Limits early ESM injection to all residues or selected shell residues. | Advanced |
| Cross-attention | `--cross-attention-layers` | `1` | Number of cross-modal attention blocks. Only active for cross-modal attention fusion. | Expose / sweep |
| Cross-attention | `--cross-attention-heads` | `4` | Number of attention heads per cross-modal block. | Expose / sweep |
| Cross-attention | `--cross-attention-dropout` | `0.1` | Dropout inside cross-modal attention blocks. | Expose |
| Cross-attention | `--cross-attention-neighborhood` | `all`; choices `all`, `first_shell`, `first_second_shell` | Which residues participate in localized cross-attention. | Expose |
| Cross-attention | `--cross-attention-bidirectional` | false | Allows ESM states to also attend back to structure states. | Expose |
| RING edges | `--ring-features-dir` | optional path | Directory containing RING edge files, or output directory for generated RING files. | Expose |
| RING edges | `--use-ring-edges` | false in raw CLI; notebook default is RING-enabled | Adds RING interaction edges in addition to radius edges when files are available. | Expose via mode |
| RING edges | `--require-ring-edges` | false | Fails if RING edge files are missing for requested structures. | Expose with warning |
| RING edges | `--prepare-missing-ring-edges` | false flag, but current config prepares by default unless disabled | Generate missing RING edge files during preflight when RING is active. Notebook default is `with_ring`, with `REQUIRE_RING_EDGES=False` and missing-edge preparation enabled. | Expose |
| RING edges | `--no-prepare-missing-ring-edges` | false | Prevents RING generation during preflight. | Expose as prepare-missing toggle |
| Metal loss | `--balance-metal-site-symbols` | false | Uses a weighted sampler to balance metal classes and, when Co/Ni are grouped, Co/Ni site symbols inside the grouped class. | Expose |
| Metal loss | `--metal-loss-function` | `cross_entropy`; choices `cross_entropy`, `focal` | Loss function for metal classification. | Expose |
| Metal loss | `--metal-focal-gamma` | `2.0` | Focal-loss gamma when focal loss is selected. | Expose |
| Metal loss | `--metal-label-smoothing` | `0.0` | Label smoothing for metal cross-entropy. | Expose |
| Metal loss | `--metal-collapsed-loss-weight` | `0.0` | Optional validation-only collapsed-4 auxiliary metal loss weight. `0.0` preserves the standard six-class objective. | Advanced |
| Metal loss | `--metal-class-weight-mode` | `inverse_frequency`; choices `none`, `manual`, `inverse_frequency`, `inverse_sqrt_frequency`, `effective_number` | Controls class weights for the metal loss. `manual` starts from `1.0` for every class and uses the per-class loss multipliers as exact class weights. | Expose |
| Metal loss | `--mn-loss-multiplier`, `--cu-loss-multiplier`, `--zn-loss-multiplier`, `--fe-loss-multiplier`, `--co-loss-multiplier`, `--ni-loss-multiplier`, `--class-viii-loss-multiplier` | `1.0` each | Per-class multipliers applied to computed metal class weights; with `--metal-class-weight-mode manual`, they are the exact manual class weights. | Advanced |
| Joint loss | `--joint-loss-weighting` | `auto`; choices `auto`, `fixed`, `uncertainty` | Controls task-level metal/EC loss balancing. `auto` uses learned uncertainty weighting for joint runs and fixed weighting for single-task runs. | Expose |
| Joint loss | `--metal-loss-weight` | `1.0` | Base task-level multiplier for the metal loss; mainly useful with `--joint-loss-weighting fixed` or deliberate ablations. | Expose |
| Joint loss | `--ec-loss-weight` | `1.0` | Base task-level multiplier for the EC loss; mainly useful with `--joint-loss-weighting fixed` or deliberate ablations. | Expose |
| EC labels/loss | `--ec-label-depth` | `1` | EC hierarchy depth used to build EC labels. | Expose / sweep |
| EC labels/loss | `--ec-group-weighting` | `structure_id`; choices `none`, `structure_id`, `pdbid_chain`, `pdbid` | Weights EC loss so multiple pockets from the same structure/group do not over-count one protein. | Expose |
| EC labels/loss | `--ec-contrastive-weight` | `0.0` | Optional supervised contrastive loss weight for EC representations. Keep `0.0` for the clean baseline. | Expose / sweep |
| EC labels/loss | `--ec-contrastive-temperature` | `0.1` | Temperature used by EC contrastive loss. | Expose |

#### Supported model families and fusion modes

- `only_gvp`: graph-only GVP baseline. It should not require ESM embeddings.
- `only_esm`: ESM-only baseline. It requires ESM embeddings unless missing embeddings are explicitly allowed or generated.
- `gvp`: GVP structure model with optional ESM branch/fusion.
- `simple_gnn_esm`: non-GVP graph + ESM comparison model.

For `gvp` and `simple_gnn_esm`, supported fusion modes are:

- `late_fusion`: pool graph states and ESM states separately, then fuse near the classifier head.
- `early_fusion`: inject residue-level ESM features before graph message passing and disable the late ESM branch.
- `node_level_late_fusion`: inject ESM into node states after graph message passing and before pooling.
- `hybrid`: use both early residue-level ESM and late graph-level ESM.
- `cross_modal_attention`: advanced graph/ESM attention fusion; use only after simpler baselines are stable.

#### Desired future work not currently supported

- Additional `node_feature_set` values beyond `conservative`.
- A general EC loss-function selector equivalent to `--metal-loss-function`; EC currently uses cross-entropy plus optional contrastive loss.
- Generic class-loss multiplier flags for EC classes. Current per-class multipliers are metal-specific.
- Additional LR schedules beyond `fixed`, `cosine`, and `step`.

#### Practical notebook-ready training pipelines

`Plan.md` is the high-level research and design authority. The concrete,
copy-paste-ready notebook pipelines live in task-specific playbooks:

- Metal classification: `docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`
- EC-number classification: `docs/EC_TRAINING_PIPELINE_PLAYBOOK.md`

Each playbook covers staged notebook configuration blocks for:

- smoke/readiness checks
- baseline model comparison
- controlled medium Optuna searches
- large controlled Optuna searches
- top-K grouped-fold confirmation
- final held-out test evaluation after validation-based selection only

For the metal notebook pipeline, the playbook must keep exact values for:

- `TASK`, `RUN_MODE`, `RECOMMENDED_RUN_SET`, `MODEL_PRESET`
- split and selection controls: `DATASET_NAME`, `VAL_FRACTION`, `SPLIT_BY`,
  `SELECTION_METRIC`, `OPTUNA_SELECTION_METRIC`
- feature controls: `RING_EDGE_MODE`, RING preparation/requirement flags,
  ESM embedding flags, and external-feature strictness
- baseline budgets: epochs, batch sizes, learning rates, weight decay, seeds,
  and maximum planned rows
- Optuna budgets and sampler controls: `N_OPTUNA_TRIALS`,
  `MAX_EPOCHS_PER_TRIAL`, `OPTUNA_N_STARTUP_TRIALS`,
  `OPTUNA_TPE_MULTIVARIATE`, `OPTUNA_TPE_GROUP`,
  `OPTUNA_AUTO_CONFIGURE_BUDGET`, storage, search preset, and search ranges
- Optional validation-only objective controls:
  `METAL_COLLAPSED_LOSS_WEIGHT`,
  `OPTUNA_METAL_COLLAPSED_LOSS_WEIGHTS_CSV`, and
  `OPTUNA_MULTIOBJECTIVE`
- Stage 6 confirmation controls: top-K, grouped-fold count, split seed, model
  seed list for fallback repeats, mismatch guard, paired-bootstrap comparison,
  and final-test exclusion
- final-test controls: preview mode first, explicit one-shot confirmation,
  source-run selection, predeclared primary report, optional five-checkpoint
  softmax-mean ensemble reporting, calibration settings, bootstrap confidence
  intervals, and repeat/mixed-batch guards

Keep current best-result notes and mutable next-step status in
`EXPERIMENT_STATUS.md`. Keep raw and summarized run evidence in
`docs/notebook_outputs/`.

Pipeline governance:

- The playbook holds the exact parameter values for every stage. Plan.md does
  not duplicate them.
- Changes to stage budgets or stage ordering must update the playbook first,
  then this section, then `METAL_NOTEBOOK_CONFIGURATION_GUIDE.md`'s crosswalk
  table.
- Changes to the held-out test policy must update Plan.md first and propagate
  to the playbook's Stage 7.

### Pipeline design trade-offs

The sequential baseline-first architecture search is publication-safe because
each added modeling component has validation evidence against a stable simpler
anchor. The trade-off is that it may miss global optima that would appear only
from a joint architecture, capacity, feature, and loss search.

Separating Optuna discovery from Stage 6 grouped-fold stability checks costs
more compute than selecting the single best trial directly, but it makes the
analysis cleaner: HPO finds candidates, while shared-fold validation estimates
whether a candidate is stable enough to promote.

The non-overlapped PinMyMetal held-out test is useful as the current final
reporting split. Stronger future claims may need additional splits, such as a
temporal split, a sequence-identity-clustered split, or an EC-stratified split,
so that generalization is not tied to one historical benchmark construction.

---

## 5) Colab Notebook and Data Bundle

The interactive training workflow is in `notebooks/DeepMzyme_training_colab.ipynb`.
The notebook supports run planning, training execution, result summarization, and
final held-out test evaluation in a staged, guarded workflow.

Colab data input modes (controlled by `COLAB_DATA_SOURCE`):

- `huggingface_link`: downloads and verifies the bundle from the project
  HuggingFace repository. Recommended default for cloud use.
- `upload_file`: prompts for a local `.tar.zst` upload in the Colab runtime.
- `drive`: uses the configured Google Drive path after Drive is mounted.

The bundle is built from the trusted non-overlapped PinMyMetal split using
`src/build_colab_bundle.py`. It includes:

- site-level MAHOMES train and test summary CSVs (training source of truth)
- training and test structure files (`.pdb`, `.cif`)
- structure-level CSV artifacts (for inspection; not used for training labels)

Comparison table and professional figure output are generated by the summarize
cell at the end of each run batch. Detailed notebook option reference is in
`docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md`.

---

## 6) Experiment Tracking and Reproducible Run Summaries

Every training run should save enough information to reproduce and compare the
result.

Each run should save:

- full config / hyperparameters
- metal label scheme
- random seed
- dataset paths and split identity
- dataset bundle identifier and checksum for serious validation or final-test
  runs, when the run uses a bundle
- key library versions for serious validation or final-test runs, including
  PyTorch, torch-geometric, ESM/ESMC, Optuna, NumPy, and scikit-learn when
  available
- model architecture
- fusion mode
- node feature set
- EC label depth, if relevant
- contrastive-learning settings, if relevant
- validation metric used for checkpoint selection
- selected checkpoint path
- final held-out test metrics, if test evaluation was requested
- git commit hash, if available

`src/report_runs.py` summarizes multiple run directories into one CSV table.

Run tiers:

| Tier | Purpose | Evidence status | Required record |
| --- | --- | --- | --- |
| Debug | Path, syntax, smoke, and plumbing checks | Not model-selection evidence | Enough config to reproduce the failure/smoke behavior |
| Serious validation | Baseline comparison, HPO, grouped-fold confirmation, or validation ablation | Eligible for model-selection discussion if gates pass | Full config, split/group policy, seeds/folds, dataset bundle checksum, git commit, key library versions, and validation artifacts |
| Final test | One-shot held-out reporting for a fixed validation-selected configuration | Final report only; never feeds back into selection | All serious-validation records plus source-run identity, checkpoint, primary-report declaration, calibration/CI settings, and no-test-selection statement |

There is currently no checked-in environment specification file. Until one is
added, serious validation and final-test records must capture enough key
library versions to make environment drift visible. A future `environment.yml`,
`requirements.txt`, or equivalent should complement this per-run record; it must
not replace run-specific metadata.

The summary table should include, when available:

- run name
- task
- metal label scheme
- model architecture
- fusion mode
- seed
- node feature set
- EC label depth
- selection metric
- best validation metrics
- final held-out test metrics
- final held-out calibration metrics and bootstrap confidence intervals
- metal active-scheme metrics, including six-class metrics for default runs and
  five-class metrics for explicitly labeled five-class runs
- metal collapsed-4 metrics
- EC level-1 / level-2 metrics
- split name/type used for the run
- whether train/test overlap was detected

Important rules:

- Validation metrics are used for checkpoint selection and hyperparameter choice.
- Held-out test metrics are used only for final reporting.
- Do not choose models by repeatedly checking the held-out test set.
- Stage 7 may report a predeclared five-checkpoint softmax-mean ensemble, but
  the ensemble source runs, averaging rule, and primary result label must be
  fixed before opening the held-out test.
- Stage 7 may include calibration metrics, validation-fitted temperature
  scaling, reliability/confidence plots, and bootstrap confidence intervals.
  Temperature fitting must use validation logits only.
- Primary final reports and secondary/diagnostic reports must be labeled
  clearly, and held-out test metrics must never be used to switch the primary
  report after evaluation.
- For a new check or fresh experiment request, previous raw notebook outputs are
  context and guardrails by default, not the main source for narrowing the new
  search. Use previous raw outputs heavily only when the request explicitly asks
  to rely on prior runs/results/raws.
- If the request is a fresh Optuna check and no narrower continuation is
  requested, use the largest sensible validation-only search space for the
  selected task/model family. Keep common-sense limits: one named model family
  or fusion mode per study, no held-out test use, plausible runtime, fixed split
  policy, fixed EC depth per EC study, and only features that are available or
  deliberately prepared.

Statistical methodology:

- Model and hyperparameter selection must be validation-only.
- Single-split validation is useful for screening but should not be the final
  promotion criterion when Stage 6 grouped-fold confirmation is available.
- Stage 6 comparisons should use shared validation units and paired confidence
  intervals, plus rare-class recall protection, before promoting a candidate.
- Calibration, temperature scaling, ensemble membership, thresholds, and primary
  report choice must be fixed from validation evidence before Stage 7.
- Stage 7 reports uncertainty, calibration, and diagnostic views after the
  held-out test is opened, but those reports must not change the selected
  configuration.

Limited-compute fallback:

- Debug and medium validation runs may be used to identify promising directions,
  but they are provisional unless they satisfy the relevant playbook decision
  gate.
- If compute is insufficient for the full serious route, stop at a labeled
  validation-only result instead of launching Stage 7 from incomplete evidence.
- A final held-out report still requires one fixed validation-selected
  configuration and the one-shot Stage 7 policy.


---

## 7) Baseline-First Model Comparison Policy

Before testing complex fusion models, establish clean baselines.

Recommended order:

1. Only-GVP
2. Only-ESM
3. GVP + simple late ESM fusion
4. GVP + early residue-level ESM fusion
5. More complex fusion modes only if simpler baselines justify them

`GVP + early fusion` is a supported preset and may be used in ESM-ready manual
comparisons. It is not a required stage in the canonical metal HPO route unless
`docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md` defines an exact executable block for
it.

Complex fusion modes include:

- node-level late fusion
- hybrid fusion
- cross-modal attention

The comparison should be sequential across model families, not a free search
over every architecture at every stage:

1. Tune and validate the simplest relevant model first.
2. Select a stable validation-best anchor from multiple seeds where possible.
3. When deliberately continuing from an anchor, carry forward shared settings
   from the simpler anchor when adding one new source of complexity. Shared
   settings include the split policy, epoch budget, Stage 6 fold plan, graph
   radius, GVP capacity, class-weighting policy, and validation selection
   metric.
4. When the user instead asks for a new check or fresh Optuna sweep, do not
   over-constrain the search to prior raw outputs. Search broadly within the
   selected model family/fusion mode, while keeping the validation/test and
   runtime safeguards above.
5. Move to the next more complex model only when validation evidence justifies the added parameters.

For metal GVP/ESM fusion, the advanced-fusion order should be:

1. Node-level late fusion after the late-fusion baseline is stable.
2. Hybrid fusion only after early or late fusion shows useful validation signal.
3. Cross-modal attention last, starting with a narrow one-layer configuration, because it has the most tuning degrees of freedom and the greatest overfitting risk.

`simple_gnn_esm` should be treated as an auxiliary architecture ablation, not the main next step in the best-pipeline search. Use it after the GVP and ESM baselines are stable when the question is whether GVP vector geometry is actually helping compared with a simpler scalar graph model.
It is supported by the notebook/model stack, but it is not a required metal HPO
stage unless the metal playbook adds a canonical executable block for it.

For each task:

- compare models using validation metrics first
- select checkpoints using validation metrics
- evaluate the selected model once on the held-out test set for final reporting

The goal is to avoid adding complex architecture before proving that it improves over simple baselines.

---

## 8) Data Leakage and Split Policy

The non-overlapped PinMyMetal split remains the historically trusted split for
final held-out evaluation unless a new experiment explicitly switches to a
newer split variant.

Named split variants:

- **Harsh Split PinMyMetal**:
  `DeepMzyme_Data/train_and_test_sets_structures_harsh_pinmymetal`.
  Every PDB ID shared by the exact PinMyMetal train and test inputs is assigned
  as a whole PDB-ID group to test, including exact-train structures/rows for
  that shared PDB ID.
- **Non-overlapped PinMyMetal**:
  `DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal`.
  This is the legacy trusted final held-out split used by current experiment
  evidence. It removes shared PDB IDs from train and keeps the original exact
  test structures in test.
- **Metal Split PinMyMetal**:
  `DeepMzyme_Data/train_and_test_sets_structures_exact_pinmymetal`.
  This follows the exact `prepare_training_and_test_set/pinmymetal_files`
  train/test membership for available supported structures and may contain
  train/test PDB-ID overlap.
- **Common-PDBID 70/30 Split PinMyMetal**:
  `DeepMzyme_Data/train_and_test_sets_structures_common_pdbid_70_30_pinmymetal`.
  Train-only PDB IDs remain train-only, test-only PDB IDs remain test-only, and
  PDB IDs common to the exact train/test inputs are assigned as whole PDB-ID
  groups with 70% to train and 30% to test. This is a custom comparison split,
  not the main final held-out split.

For the EC-number classification task:

- The non-overlapped PinMyMetal split should be treated as mandatory for final held-out testing.
- The exact PinMyMetal split should not be used as the final EC held-out test split if train/test structures overlap.
- EC supervision is structure/protein/chain-level even when extraction creates multiple separated metal-pocket samples for the same structure. EC cross-entropy should use group weighting, by default at `structure_id`, so such structures are not over-counted; this does not divide by raw metal atom count and does not downweight true multinuclear pockets.

For the metal-type classification task:

- The non-overlapped PinMyMetal split should be the preferred final held-out test split.
- The exact PinMyMetal split may be kept as an optional secondary metal-testing mode.
- If the exact PinMyMetal split is used for metal testing, the result must be clearly labeled as using the exact/possibly-overlapped split.
- Metal results from the exact/possibly-overlapped split should not be presented as the main final held-out result if train/test overlap exists.

The code and/or result summary files should clearly record which split was used:

- non-overlapped PinMyMetal split
- exact PinMyMetal split
- any other custom split

If the exact PinMyMetal split is used as an optional metal-testing mode, the output summary should explicitly warn that this split may contain train/test overlap and should be interpreted only as a secondary/reference result.

Before final training/evaluation, validate train/test overlap by:



- full structure filename
- PDB ID
- preferably PDB-chain or pocket ID when available

The held-out test set must remain separate from model selection.

Use only validation or cross-validation for:

- checkpoint selection
- hyperparameter choices
- model architecture choices
- fusion-mode choices
- temperature or calibration-method choices
- ensemble membership, ensemble weighting, or threshold choices

Use the held-out test set only for final reporting of selected models.
