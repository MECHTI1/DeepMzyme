# EC Training Pipeline Playbook

## Standalone validation campaign — Stage 0 through Stage 2B


The EC standalone block below is reconciled against current notebook controls.
It uses CARE clusterRes30 training membership and EC depth 1 only. In this
dataset, the parser's `pdbid` grouping token is the **full UniProt accession**
before `__chain_`, not a truncated four-character PDB ID. All pockets from
that protein stay together. Ambiguous EC1 prefixes are excluded by the
single-label loader; record their exclusion and the retained class vocabulary.
CARE metal/site assignments are computational AlphaFill/MAHOMES transfers.
Three families × four runs = 12 baselines, after three one-epoch smoke runs.
No new EC HPO or Stage 6/6B/7 recipe is certified by this baseline-only change.

EC group recall/support dictionaries and the group confusion matrix are saved
in the epoch records as `val_ec_group_per_class_recall`,
`val_ec_group_per_class_support`, and `val_ec_group_confusion_matrix`;
`val_ec_group_min_recall` supports the rare-class gate. Use the record at the
EC-selected checkpoint, not the best recall from another epoch.

This is the standalone-only preparation route. It supersedes the retained
baseline examples below for this campaign; later HPO and final-reporting recipes
are outside this certification. Keep all launch controls false while preparing.
No comparison result or promotion is implied by a passing configuration check.

Run the block once per family and (for metal) target arm. Stage 0 is the
planning/data/feature pass with the smoke configuration and the launch switch
off. Stage 1 is one epoch for each variant after readiness passes. Stage 2A
establishes Only-GVP; Stage 2B completes Only-ESM and graph-level late fusion.
For baselines change only `STANDALONE_PHASE` to `"baseline"`.
Run Only-GVP first, Only-ESM second, and late fusion third. Do not run HPO yet.

These are modest controlled baseline screens, not historical reproduction:
two LRs × two model seeds per variant, fixed split seed 42, 15% internal
validation from the named external **train** directory. The radius-6 choice
uses validation context from PARAMETER_FINDINGS; it avoids repeating the weaker
tested radius-10 family without claiming universal superiority. All families
receive the same LR opportunities, seeds and epoch budget within a task.
Only-ESM uses the current **pocket-residue ESMC pooling** implementation; do not
describe it as an independently tested full-protein pooling model.

The graph arms use the same conservative residue features, radius edges,
residue-only readout, graph/head capacity, and no RING or augmentation.
ESMC-300m embeddings must be 960-dimensional with matching residue alignment
and source metadata. Require complete ESM and external-feature coverage for
the whole campaign cohort before launching any family; do not let missing
features silently produce different cohorts. The loader runs with invalid
structures treated as errors. Preparation switches remain off: resolve missing
features in an explicitly reviewed preparation step, then rerun readiness.

EC uses structure-weighted cross-entropy and inverse-frequency class weights
computed from **training groups**, with group-level logit averaging for
validation. Metal uses training-only inverse-frequency class weights in each
arm's native target space, unit manual multipliers, ordinary cross-entropy,
and no collapsed auxiliary loss or site sampler. Native class balancing is a
declared part of the target-formulation comparison.

### Exact standalone notebook block

```python
# Paste at the END of Main configuration, before Build central CONFIG.
# Use a fresh kernel when switching task/target, then rebuild CONFIG and the plan.
STANDALONE_PHASE = "smoke"  # "smoke" first; "baseline" only after its gate passes.
MODEL_PRESET = "Only-GVP"  # then "Only-ESM", then "GVP + late fusion"
assert STANDALONE_PHASE in {"smoke", "baseline"}
assert MODEL_PRESET in {"Only-GVP", "Only-ESM", "GVP + late fusion"}
RUN_MODE = "single" if STANDALONE_PHASE == "smoke" else "manual_configurations"
RECOMMENDED_RUN_SET = "custom"  # Named run sets can override the declared axes.
SPLIT_BY = "pdbid"
SPLIT_SEED = "42"
VAL_FRACTION = 0.15
N_FOLDS = ""
FOLD_INDEX = ""
SEEDS_CSV = "42" if STANDALONE_PHASE == "smoke" else "42,43"
LEARNING_RATES_CSV = "3e-5" if STANDALONE_PHASE == "smoke" else "3e-5,1e-4"
WEIGHT_DECAYS_CSV = "1e-4"
MAX_CONFIGURATION_RUNS = 1 if STANDALONE_PHASE == "smoke" else 4
LR_SCHEDULES_CSV = "fixed"
HIDDEN_S_VALUES_CSV = "128"
HIDDEN_V_VALUES_CSV = "16"
EDGE_HIDDEN_VALUES_CSV = "64"
GVP_LAYERS_VALUES_CSV = "4"
HEAD_MLP_LAYERS_VALUES_CSV = "2"
HEAD_MLP_DROPOUT_VALUES_CSV = "0.2"
EDGE_RADIUS_VALUES_CSV = "6.0"
ESM_FUSION_DIM_VALUES_CSV = "128"
ESM_GRAPH_ENCODER_DROPOUT_VALUES_CSV = "0.1"
EARLY_ESM_DIM_VALUES_CSV = "32"
EARLY_ESM_DROPOUT_VALUES_CSV = "0.0"
CROSS_ATTENTION_DROPOUT_VALUES_CSV = "0.0"
CLASSIFIER_POOL_DISTANCE_CUTOFF_VALUES_CSV = "0.0"
POSITION_NOISE_STDS_CSV = "0.0"
SECOND_SHELL_DROPOUTS_CSV = "0.0"
OUTER_RESIDUE_DROPOUTS_CSV = "0.0"
OMIT_NODE_FEATURE_SETS = ""
METAL_NODE_MODE = "none"
STRUCTURAL_READOUT_SCOPE = "residue_only"
RING_EDGE_MODE = "without_ring"
REQUIRE_RING_EDGES = False
PREPARE_MISSING_RING_EDGES = False
ALLOW_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_ESM_EMBEDDINGS = False
ALLOW_MISSING_EXTERNAL_FEATURES = False
PREPARE_MISSING_EXTERNAL_FEATURES = False
ESM_EMBEDDINGS_DIR = ""  # Auto-resolve the selected bundle's esm_embeddings.
EXTERNAL_FEATURES_ROOT_DIR = ""  # Auto-resolve updated_feature_extraction.
METAL_CLASS_WEIGHT_MODES_CSV = "inverse_frequency"
METAL_LOSS_FUNCTIONS_CSV = "cross_entropy"
METAL_FOCAL_GAMMA_VALUES_CSV = "2.0"
METAL_LABEL_SMOOTHING_VALUES_CSV = "0.0"
METAL_COLLAPSED_LOSS_WEIGHTS_CSV = "0.0"
BALANCE_METAL_SITE_SYMBOLS_CSV = "False"
MN_LOSS_MULTIPLIER = CU_LOSS_MULTIPLIER = ZN_LOSS_MULTIPLIER = 1.0
FE_LOSS_MULTIPLIER = CO_LOSS_MULTIPLIER = NI_LOSS_MULTIPLIER = 1.0
CLASS_VIII_LOSS_MULTIPLIER = 1.0
EC_LABEL_DEPTHS_CSV = "1"
EC_CONTRASTIVE_WEIGHTS_CSV = "0.0"
EC_GROUP_WEIGHTING = "structure_id"
EC_CLASS_WEIGHT_UNIT = "group"
METAL_LOSS_WEIGHT_VALUES_CSV = "1.0"
EC_LOSS_WEIGHT_VALUES_CSV = "1.0"
REQUIRE_ALL_TASK_CLASSES = True
INVALID_STRUCTURE_POLICY = "error"
UNSUPPORTED_METAL_POLICY = "error"
DATALOADER_NUM_WORKERS = 0
DATALOADER_PIN_MEMORY = True
DEVICE = "cuda"
DETERMINISTIC = True
LOG_PER_CLASS_EPOCH_METRICS = True
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_TRAIN_LOSS_TEST_EVAL_DEBUG = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
ALLOW_MODEL_PRESET_MISMATCH = False
ALLOW_SINGLE_MODE_TO_TRUNCATE_COMPARISON = False
OPTUNA_ALLOW_INCOMPATIBLE_STUDY_REUSE = False
RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = False
LAUNCH_PLANNED_MAIN_TRAINING_RUNS = False
LAUNCH_PLANNED_TRAINING_RUNS = False
LAUNCH_STAGE6_TOP_K_CONFIRMATION = False
LAUNCH_STAGE6B_FINAL_REFIT = False
LAUNCH_FINAL_HELD_OUT_TEST_EVAL = False
SKIP_EXISTING_RUNS = True
STOP_ON_FIRST_FAILURE = True
OUTPUT_LAYOUT = "classic_runs"
WORKFLOW_OUTPUT_NAME = ""
RUNS_DIR = ""
COPY_OUTPUTS_TO_DRIVE = True
SUMMARY_BASENAME = ""
DATASET_ROOT_OVERRIDE = ""
TRAIN_DIR_OVERRIDE = TRAIN_SITE_SUMMARY_CSV_OVERRIDE = ""
TEST_DIR_OVERRIDE = TEST_SITE_SUMMARY_CSV_OVERRIDE = ""
USE_CLEAN_FOLD_SELECTOR = False
TASK = "ec"
DATASET_NAME = "CARE_task1_30_clusterRes30_train_test_metallo"
METAL_LABEL_SCHEME = "four_class"  # Inert for EC supervision; avoids stale joint resume state.
SPLIT_STRATIFY_BY = "active_targets"
METAL_ELIGIBILITY_SCHEME = "active"  # EC eligibility depends only on EC supervision.
SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"
OPTUNA_SELECTION_METRIC = SELECTION_METRIC
SINGLE_AND_MANUAL_CONFIG_EPOCHS = 1 if STANDALONE_PHASE == "smoke" else 30
EPOCHS = SINGLE_AND_MANUAL_CONFIG_EPOCHS
BATCH_SIZES_CSV = "4"
_family = {"Only-GVP": "only_gvp", "Only-ESM": "only_esm", "GVP + late fusion": "late_fusion"}[MODEL_PRESET]
RUN_BATCH_ID = f"standalone_ec1_care30_{_family}_{STANDALONE_PHASE}_v1"
RUN_NAME_PREFIX = RUN_BATCH_ID

```

### Readiness, outputs, and explicit launch gate

Before enabling training, inspect the printed shell-safe command and verify
task, dataset path, scheme/depth, split seed, grouping, features, run count,
epoch budget and selection metric. Generated commands must contain no
`--run-test-eval`, `--test-structure-dir`, or `--test-summary-csv`.
Inspect train/test **membership only** for overlap; no held-out model inference
or metrics are authorized. All planned training inputs come from the train
directory. Do not reinterpret the named dataset as the resolved primary final
test.

The notebook writes under `<DRIVE_ROOT>/notebook_outputs/runs/<RUN_BATCH_ID>/`:

- `active_run_config.json`, `active_run_config.md`;
- `<SUMMARY_BASENAME>_planned_runs.csv`;
- `<SUMMARY_BASENAME>_planned_run_dictionary.json`;
- metal weight diagnostics CSV when the metal head is enabled.

The auto-generated summary basename is printed by the planner. After an
authorized launch, each run must preserve `run_config.json`,
`run_metadata.json`, `split_diagnostics.json`, `dataset_summary.json`,
`prepare_status.json`, `epoch_metrics.csv`, `train_metrics.csv`,
`val_metrics.csv`, and `best_model_checkpoint.pt`.
The summary cell writes `<SUMMARY_BASENAME>.csv`,
`<SUMMARY_BASENAME>_completed_only.csv`, and a PNG when plotting succeeds.
No `test_report.json` may be created. Preserve actual code commit/dirty state,
runtime versions, CSV checksums and the verified bundle identity; do not
attribute the local materialization to a hosted bundle without verifying the actual input
archive.

The ordinary launch gate is the dedicated **Main planned training launch
switch** cell: `LAUNCH_PLANNED_MAIN_TRAINING_RUNS`. Its legacy alias alone
does not control the current execution cell. Leave it false for review; after
explicit launch approval, enable that switch and run **Optional training
execution**. Never enable Stage 6, Stage 6B or Stage 7 during this campaign.

Stage 0 passes only when inputs, feature coverage and class support are valid.
Stage 1 requires a completed one-epoch run of each variant with valid outputs,
finite losses and no held-out report; ignore smoke accuracy for selection.
Stage 2 requires all four baseline runs per variant and identical retained
train/validation **example identities and groups** across the compared runs.
Compare the identity fields in `dataset_summary.json`; the existing hash also
contains targets, so its raw value is expected to differ between four- and
six-class arms. Check identity tuples separately and verify the deterministic
target mapping.

Require every active class in both internal splits and inspect per-class
recalls. A missing validation class blocks a reportable baseline even if the
aggregate metric exists. No family is promoted with a zero seed-mean recall
for an active class. Two model seeds on one fixed split are preliminary
Grade-3 evidence, not fold confirmation. Do not claim formulation/modality
superiority from this screen: shared grouped folds × seeds with paired
bootstrap intervals and rare-class recall protection are still required.
Stage 6B final refit must occur after eventual selection and before any
one-shot Stage 7 reporting; neither transition is authorized here.

Keep metal and EC summaries/rankings separate. Update EXPERIMENT_STATUS and
the evidence index only when measured training/validation results exist.

Within each task, summarize seed mean, sample SD, minimum and per-class recall
for each family/target/LR configuration. Compare the metal arms at matched LR
and seed first, using the common four-class metric. Select a provisional LR
by seed-mean task metric with the recall gate; do not select a lucky seed or
pool different learning rates into one family mean. Equal-budget per-family
selection remains preliminary fixed-split evidence.


This playbook is the practical, notebook-ready pipeline for DeepMzyme EC-number
classification, one of the project's two independent primary missions. It
follows the same stage structure as
`docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`. Read that playbook for the full
rationale behind each stage; this document covers only the EC-specific
differences.

`Plan.md` remains the high-level research and design authority. Current best
validation evidence belongs in `EXPERIMENT_STATUS.md` and
`docs/notebook_outputs/`, not in this stable playbook.

For the cross-document run order and output-folder map, see `docs/README.md`.
For environment/data orientation use `docs/GETTING_STARTED.md`; for Colab GPU
connection and dependency installation use `docs/COLAB_GPU_RUNBOOK.md`.

## Compatibility Warning — Later Stages Still Require Reconciliation

The opening standalone Stage 0–2B block replaces the legacy baseline blocks
below for executable use. Its command expansion is tested against the notebook
and generic `src/train.py`; the dedicated `train_ec.py` wrapper has narrower
batch-size/metric constraints and is not this recipe's entry point.

This playbook preserves important EC budgets, ranges, label-depth progression,
group weighting, and contrastive-loss intent, but affected blocks are **not
currently certified executable** against
`notebooks/DeepMzyme_training_colab.ipynb`.

Static audit found these playbook assignments absent from the current notebook
assignment surface:

- `CONFIRM_ONE_SHOT_POLICY`
- `OPTUNA_BATCH_SIZES_CSV`
- `OPTUNA_EDGE_HIDDEN_VALUES_CSV`
- `OPTUNA_EDGE_RADIUS_VALUES_CSV`
- `OPTUNA_ESM_FUSION_DIM_VALUES_CSV`
- `OPTUNA_GVP_LAYERS_VALUES_CSV`
- `OPTUNA_HEAD_MLP_LAYERS_VALUES_CSV`
- `OPTUNA_HIDDEN_S_VALUES_CSV`
- `OPTUNA_HIDDEN_V_VALUES_CSV`
- `OPTUNA_WEIGHT_DECAYS_CSV`

Stage 7 examples also use `FINAL_TEST_WORKFLOW = "preview_only"` and
`"evaluate_selected_checkpoint"`, while the current notebook accepts only
`"evaluate_stage6_selected_candidate"`. The EC Stage 6/7
sequence also predates the current metal-style named Stage 6B workflow.

The existing blocks below are retained as historical/intended recipes so their
scientific search spaces do not disappear. Do not rename controls or modernize
the EC Stage 6/6B/7 flow piecemeal. Reconcile the complete pipeline in a
separate task; see
[`TECH-002`](FOLLOW_UP_TECHNICAL_ISSUES.md#tech-002--ec-playbook-assignments-do-not-match-the-notebook-surface).

> **Primary final-test route: unresolved scientific decision required before final reporting.**

These legacy HPO/final blocks remain quarantined; do not execute them as part
of the standalone baseline campaign. Their complete migration remains open.


## EC-Specific Rules Before You Start

**Label depth.** Start with `EC_LABEL_DEPTHS_CSV = "1"` (first EC digit, seven
classes). Add depth-2 and deeper runs only after depth-1 behavior is stable.
Each depth level is a separate classification problem with its own class count
and difficulty.

**Target scope.** The current implementation assigns one categorical EC target
at the selected depth. Multiple source EC annotations that share one prefix at
that depth can map to the same class; annotations with multiple distinct
prefixes receive no target. This is not full multi-label EC prediction. Keep
depth-1 classification, deeper hierarchical classification, multiple source
annotations, and a future full multi-label objective distinct.

**Group weighting.** Always keep `EC_GROUP_WEIGHTING = "structure_id"`. EC
supervision is at the protein/structure level. Multiple metal-pocket samples from
the same structure share the same EC annotation; group weighting prevents those
structures from dominating the loss.

**Contrastive loss.** Start with `EC_CONTRASTIVE_WEIGHTS_CSV = "0.0"` for all
first baselines. Contrastive loss is a secondary feature; enabling it before the
plain supervised baseline is stable adds a confound.

**Split policy.** The primary final-test route is unresolved and must be chosen
in a separate scientific decision before Stage 7. The exact PinMyMetal split
must not be used as the main final EC held-out split if train/test structures
overlap (see Plan.md section 8 and `docs/DATASETS.md`).

**Held-out test policy.** Identical to metal: never use the held-out test set
for model comparison, Optuna HPO, seed-repeat validation, or cross-validation.
Use only validation metrics for all selection decisions. After Stage 6 selects
one EC configuration, train/refit the final EC model with that frozen
configuration, then use the held-out test once for final reporting.

**Fresh-check default.** If the user asks for a new check, new run, or fresh
Optuna sweep without explicitly asking to rely on previous raws/results, use
previous notebook outputs only as context and safety checks. Prefer the broadest
sensible validation-only Optuna search within the selected `MODEL_PRESET`, with
fixed EC depth, fixed selection metric, and common-sense runtime and
feature-availability limits. If the user explicitly asks to rely on previous
running/results/raws, inspect that evidence and use it to narrow, continue, or
repeat the prior configuration.

**Selection metric.** For depth-1 EC training use
`val_ec_group_level_1_balanced_acc`. This uses group weighting and balanced
accuracy, which is the right choice for potentially imbalanced EC first-digit
classes. Update the metric name when the label depth changes (e.g., use
`val_ec_group_level_2_balanced_acc` for depth-2 runs).

**ESM and model order.** For EC classification, sequence-level information
(ESM) may matter more than for metal. Still follow the baseline-first order:
1. Only-GVP first (structure-only baseline).
2. Only-ESM (sequence-only baseline, important reference for EC).
3. GVP + graph-level late fusion after both simple baselines are measured.
4. Advanced fusion only if simpler models justify the added complexity.

## Phase 3 — training-only metal × EC1 association

The executable profile `metal_ec1_development_association_v1` is a local CPU
analysis of **separate** PinMyMetal and CARE source panels. It consumes the exact
retained training membership from the metal pilot and EC standalone readiness
artifacts. It does not train a model or certify a shared-encoder split.

Preparation fixes the source hashes and analysis rules before calculating
statistics. It reconstructs canonical metal clusters only for retained training
pockets, matches the catalytic-site training summary, and requires equality
with every saved source-scheme metal and EC1 target (native-six in metal
readiness, common-four in EC readiness). Conflicting duplicate site labels,
changed sources, unexpected targets, or training/internal-validation group
overlap stop the analysis. Only group identifiers from saved internal validation
are used for the exclusion check. No external test file is an input; external
training membership is inherited from the audited standalone artifacts rather
than independently re-certified against external test identities.

Exact inputs are the non-overlap PinMyMetal `train` summary/structure manifest
and CARE clusterRes30 `train` summary/structure manifest under `DATA_ROOT`, plus
`docs/notebook_outputs/raw/metal_architecture_pilot_20260915/readiness/expected_split.json`
and `docs/notebook_outputs/raw/ec1_standalone_v12_20260914/expected_split.json`.
The latter artifacts own the retained pocket and split identities. PinMyMetal
labels on experimentally resolved structures and computational AlphaFill/MAHOMES
transfers in CARE remain separate provenance categories. Neither panel is an
unselected sample of all metalloenzymes.

```bash
# Run from the repository root, using the configured project interpreter.
PYTHON=/home/mechti/miniconda3/envs/DeepMzyme/bin/python
DATA_ROOT=DeepMzyme_Data
ASSOCIATION_OUTPUT=DeepMzyme_Data/notebook_outputs/analyses/metal_ec1_development_association_v1_20260915
"$PYTHON" src/analyze_metal_ec_association.py prepare \
  --data-root "$DATA_ROOT" --output-dir "$ASSOCIATION_OUTPUT" \
  --permutations 9999 --seed 42
"$PYTHON" src/analyze_metal_ec_association.py execute \
  --output-dir "$ASSOCIATION_OUTPUT"
```

Use a fresh output directory. `prepare` writes `analysis_manifest.json` and its
SHA-256, `retained_training_pairs.csv`, and `eligibility_audit.json` before
`execute` computes any association statistics. Execution verifies their hashes
and the source hashes, then writes `association_results.json`,
`contingencies_and_conditionals.csv`, and `execution_receipt.json`.

Each panel has native-six and inclusive common-four views. An additional
common-four view restricted to native-six-eligible pockets is a descriptive
cohort-composition sensitivity check. Missing EC1 and ambiguous/missing metal
targets are excluded explicitly per view; exclusions can overlap. Multiple
native metals within VIII can have a valid common-four target while lacking a
single native-six target. No validation statistics are part of this recipe.

For every view, report raw pocket counts and both conditional distributions,
then a protein-group-weighted table: each eligible PDB group or full CARE
UniProt accession contributes total weight one, divided among its eligible
pockets. Groups must have a single retained EC1 target. This protects against
repeated pocket annotations; PDB grouping does not establish sequence-homology
independence or a cross-source PDB-to-UniProt identity union.

Report Pearson's chi-square statistic and expected-cell diagnostics, Cramér's
V, mutual information in natural-log units, and arithmetic-normalized mutual
information `2 MI / (H(metal) + H(EC1))`. Empty or constant marginals produce
undefined association measures, reported as null. The conventional expected
cell heuristic is no expected cell below one and at most 20% below five; even
passing it does not make repeated pockets independent. **No asymptotic
site-level chi-square p-value is reported.**

The exploratory inference uses 9,999 seed-42 permutations of whole-group EC1
labels, retaining each group's metal profile and weight. Its statistic is
group-weighted MI, with `(exceedances + 1) / (permutations + 1)` and Holm
adjustment across the four primary panel × native-six/common-four tests. The
matched-cohort sensitivity gets no additional permutation test. Exchangeability
between groups is an assumption; homology and source selection remain possible
violations. These p-values are not evidence that auxiliary supervision helps.

**Completion gate:** all frozen-input checks pass; exclusion counts, both
weighting views, conditionals, statistics, assumption diagnostics, provenance,
and limitations are saved. Review these descriptive results before designing
the auxiliary challenger. Joint training still requires its own certified
cross-source identity and held-out exclusion protocol below.

## Initial Auxiliary Metal-Supervision Boundary

The first experiment connecting the two primary tasks asks whether metal
supervision improves EC prediction. Compare a matched EC-only model with EC
plus an auxiliary metal loss. Both variants must make independent predictions
from a shared encoder; neither head's output is an input to the other head.
Treat the auxiliary variant as a challenger and report negative transfer when
it occurs.

Limit that first comparison to GVP + graph-level late fusion. Keep Only-GVP
and Only-ESM as standalone baselines. Do not add hybrid,
node-level late fusion, cross-attention, soft predicted-metal conditioning, or
a hard predicted-metal-to-EC cascade. Known-metal conditioning is a later
diagnostic only when the observed, curated, computationally transferred, or
model-predicted provenance is explicit.

Metal labels are pocket/site-level and EC labels are protein/structure-level.
Keep EC group weighting and build split membership across every task label
source before training. A protein held out for EC cannot enter shared-encoder
training through its metal label, and the reverse rule also applies. Held-out
data cannot choose auxiliary loss weights, joint versus single-task training,
features, thresholds, architecture, or HPO settings.

No certified executable block for this controlled comparison exists here yet.
Add it only after the EC notebook compatibility work and cross-task exclusion
checks are complete; see
[`TECH-011`](FOLLOW_UP_TECHNICAL_ISSUES.md#tech-011--controlled-ec-primary-auxiliary-learning-protocol-is-not-certified).
The reverse metal-primary auxiliary question is optional and does not block
completion of the EC mission.

### Minimum CLI controls for a matched intersection comparison

The opt-in `--controlled-ec-auxiliary` guard is implemented for `src/train.py`.
It does not certify the notebook recipes or authorize an experiment. Both arms
use `--task joint`, retaining the identical fully labelled intersection and
both prediction heads, including when the metal loss weight is zero. The
full-data standalone `--task ec` baseline is a separate comparison.

Required common controls are `--metal-label-scheme four_class`,
`--model-architecture gvp`, `--fusion-mode late_fusion`,
`--joint-loss-weighting fixed`, `--ec-label-depth 1`, `--ec-loss-weight 1`,
`--ec-contrastive-weight 0`, `--ec-group-weighting structure_id`,
`--train-val-split-by pdbid`, and
`--selection-metric val_ec_group_balanced_acc`. Validation must be enabled;
held-out evaluation and test overrides are rejected. Early ESM is disabled.

Construct arm B from the same resolved configuration as arm A, changing only
`metal_loss_weight` from zero to a predeclared positive value, plus its output
run name. Keep dataset, retained examples, features, architecture, optimizer,
budget, folds and seeds identical. No auxiliary weight or budget is selected
by this prerequisite change. Do not combine separate task datasets or holdouts:
the single input root must already exclude every reserved protein for either
task. That provenance still requires certification before execution.

Compare `dataset_summary.json`'s `retained_split_identity` for both partitions:
it records ordered example/target/group identities, SHA-256 fingerprints and
exact example, structure and group counts. The summary is also embedded in
`run_config.json` and `run_metadata.json`. Compare the saved config, source
artifact hashes/bundle identity, runtime feature metadata and Git state too;
matching example IDs alone does not establish matching structure/feature bytes.
Report EC validation differences on paired folds/seeds, including negative
transfer. These controls are prerequisites, not experimental evidence.

## EC Run Tiers And Reproducibility Records

Use the same run-tier policy as the metal playbook:

| Tier | EC stages | Selection/reporting status | Required record |
| --- | --- | --- | --- |
| Debug | Stage 1, Stage 3 | Not model-selection evidence | Resolved notebook config, planned commands, run logs, and failure context |
| Serious validation | Stage 2, Stage 4, Stage 5, Stage 6 | Validation-only evidence if the stage gate passes | Full config, EC label depth, contrastive settings, split/seed identity, Optuna metadata, dataset bundle ID/checksum, git commit, and key library versions |
| Final test | Stage 7 | One-shot held-out reporting only | Stage 6 selection evidence, final training/refit source run/checkpoint, EC depth, primary report declaration, dataset bundle ID/checksum, git commit, key library versions, and no-test-selection statement |

Serious validation and final-test records should capture key library versions
when available: PyTorch, torch-geometric, ESM/ESMC, Optuna, NumPy, and
scikit-learn. This repository currently has no checked-in environment spec, so
per-run version records are required until an environment file is added.

Limited-compute fallback: stop at a clearly labeled validation-only result if
Stage 6 cannot be completed. Do not launch Stage 7 from provisional EC evidence,
and do not launch it before the selected EC configuration has been trained/refit
as a frozen final source run. Depth-2 or deeper EC cycles must restart the
staged validation workflow instead of inheriting a depth-1 final-test decision.

Pruning is disabled unless an EC stage explicitly opts into it. If pruning is
enabled manually for a serious 50-epoch EC HPO run, use the notebook's serious
minimum-epoch rule (`OPTUNA_PRUNING_MIN_EPOCH >= 8`) and record that override in
the run summary.

## Common Defaults

Use these shared defaults unless a stage overrides them.

```python
TASK = "ec"
DATASET_NAME = "train_and_test_sets_structures_non_overlapped_pinmymetal"
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"
OPTUNA_SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False

EC_LABEL_DEPTHS_CSV = "1"
EC_CONTRASTIVE_WEIGHTS_CSV = "0.0"
EC_GROUP_WEIGHTING = "structure_id"

RING_EDGE_MODE = "with_ring"
REQUIRE_RING_EDGES = False
PREPARE_MISSING_RING_EDGES = True

ALLOW_MISSING_EXTERNAL_FEATURES = False
PREPARE_MISSING_EXTERNAL_FEATURES = False
EXTERNAL_FEATURES_ROOT_DIR = ""

COPY_OUTPUTS_TO_DRIVE = True
```

If using `Only-ESM`, `GVP + late fusion`, or any ESM-requiring preset, provide
`ESM_EMBEDDINGS_DIR` or set `PREPARE_MISSING_ESM_EMBEDDINGS = True`
deliberately. Do not use `ALLOW_MISSING_ESM_EMBEDDINGS = True` for reportable
runs.

Update `SELECTION_METRIC` and `OPTUNA_SELECTION_METRIC` when running depth-2 or
deeper EC experiments — change `level_1` to the matching level number.

## CARE Task 1 30% Metallo Subset Configuration

Use this block only after
`CARE_prepare_training_and_test_set/07_export_dataset_care_task1_30_clusterRes30.sh`
has created `DeepMzyme_Data/CARE_task1_30_clusterRes30_train_test_metallo` or
after that root is available through the selected Colab data source. Legacy CARE
dataset names remain notebook aliases for this root. This is a CARE-derived
AlphaFill-MAHOMES catalytic metalloenzyme subset, not the full CARE benchmark.

```python
TASK = "ec"
DATASET_NAME = "CARE_task1_30_clusterRes30_train_test_metallo"
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"
OPTUNA_SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False

EC_LABEL_DEPTHS_CSV = "1"
EC_CONTRASTIVE_WEIGHTS_CSV = "0.0"
EC_GROUP_WEIGHTING = "structure_id"

RING_EDGE_MODE = "with_ring"
REQUIRE_RING_EDGES = False
PREPARE_MISSING_RING_EDGES = True
```

If the exported CARE root is not in the normal bundle, Drive, or repository
search path, keep `DATASET_NAME` as above for provenance and set:

```python
DATASET_ROOT_OVERRIDE = "/absolute/path/to/CARE_task1_30_clusterRes30_train_test_metallo"
```

Do not use the exported CARE test split during training, validation, HPO,
seed-repeat selection, cross-validation, or model selection. The notebook should
make the internal validation split only from the exported CARE `train/`
directory and reserve exported CARE `test/` for final held-out reporting.

## Stage 1 — Smoke And Readiness Check

Purpose: verify Colab setup, data paths, EC CSV detection, graph construction,
and the EC training command path.

When to use it: first run in a fresh Colab/runtime, after changing the notebook,
or after changing data bundle paths.

Expected scale/runtime: smoke/debug, minutes.

Notebook configuration block:

```python
TASK = "ec"
RUN_MODE = "single"
RECOMMENDED_RUN_SET = "only_gvp_smoke"
MODEL_PRESET = "Only-GVP"
RUN_BATCH_ID = "ec_smoke_readiness"
SUMMARY_BASENAME = "ec_smoke_readiness"
RUN_NAME_PREFIX = "ec_smoke"

EPOCHS = 1
BATCH_SIZES_CSV = "4"
LEARNING_RATES_CSV = "3e-5"
WEIGHT_DECAYS_CSV = "1e-4"
SEEDS_CSV = "42"
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"

EC_LABEL_DEPTHS_CSV = "1"
EC_CONTRASTIVE_WEIGHTS_CSV = "0.0"
EC_GROUP_WEIGHTING = "structure_id"

RING_EDGE_MODE = "with_ring"
ESM_EMBEDDINGS_DIR = ""
ALLOW_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_ESM_EMBEDDINGS = False
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
MAX_CONFIGURATION_RUNS = 1
```

In the **Optional training execution** cell:

```python
LAUNCH_PLANNED_TRAINING_RUNS = True
```

Expected outputs/files:

- `<RUNS_DIR>/<SUMMARY_BASENAME>_planned_runs.csv`
- `<RUNS_DIR>/<SUMMARY_BASENAME>_planned_run_dictionary.json`
- One run directory under `<RUNS_DIR>/`
- `run_metadata.json`, `run_config.json`, `split_diagnostics.json`
- No `test_report.json`

Success criteria:

- Planning prints one runnable Only-GVP EC command.
- Training completes without missing-path, split, EC CSV detection, or CLI
  errors.
- Train and validation EC diagnostics print (class counts at EC level 1).
- No held-out test report is produced.

Decision after this stage:

- If it fails, fix data paths, EC CSV format, bundle setup, or structure
  parsing before running real comparisons.
- If it succeeds, move to baseline model comparison. Ignore the 1-epoch metric.

## Stage 2 — Baseline Model Comparison

Purpose: establish clean EC validation baselines before adding fusion or HPO.

When to use it: after smoke passes and before Optuna or advanced fusion. Run
2A (Only-GVP) first. Run 2B (multi-model) once ESM embeddings are ready.

Expected scale/runtime: medium validation run, hours.

### 2A — Structure-Only Only-GVP Baseline

Notebook configuration block:

```python
TASK = "ec"
RUN_MODE = "manual_configurations"
RECOMMENDED_RUN_SET = "only_gvp_lr_seed"
MODEL_PRESET = "Only-GVP"
RUN_BATCH_ID = "ec_only_gvp_baseline_lr_seed"
SUMMARY_BASENAME = "ec_only_gvp_baseline_lr_seed"
RUN_NAME_PREFIX = "ec_only_gvp_baseline"

EPOCHS = 30
BATCH_SIZES_CSV = "4,8"
LEARNING_RATES_CSV = "3e-5,1e-4"
WEIGHT_DECAYS_CSV = "1e-4"
SEEDS_CSV = "42,43"
MAX_CONFIGURATION_RUNS = 8

HIDDEN_S_VALUES_CSV = "128"
HIDDEN_V_VALUES_CSV = "16"
EDGE_HIDDEN_VALUES_CSV = "64"
GVP_LAYERS_VALUES_CSV = "4"
HEAD_MLP_LAYERS_VALUES_CSV = "2"
EDGE_RADIUS_VALUES_CSV = "8.0"

EC_LABEL_DEPTHS_CSV = "1"
EC_CONTRASTIVE_WEIGHTS_CSV = "0.0"
EC_GROUP_WEIGHTING = "structure_id"
SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"

RING_EDGE_MODE = "with_ring"
ESM_EMBEDDINGS_DIR = ""
ALLOW_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_ESM_EMBEDDINGS = False
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

### 2B — ESM-Ready Baseline Comparison

Run this only after ESM embeddings are available or after you intentionally
allow the notebook to prepare missing embeddings. The
`baseline_model_comparison` preset overrides `MODEL_PRESET` and runs
`Only-GVP`, `Only-ESM`, and `GVP + late fusion`.

Note: for EC classification, the `Only-ESM` baseline is especially important
because EC function often correlates strongly with sequence. Do not skip it.

Notebook configuration block:

```python
TASK = "ec"
RUN_MODE = "manual_configurations"
RECOMMENDED_RUN_SET = "baseline_model_comparison"
# MODEL_PRESET is overridden by baseline_model_comparison (runs Only-GVP, Only-ESM, GVP + late fusion)
RUN_BATCH_ID = "ec_baseline_model_comparison"
SUMMARY_BASENAME = "ec_baseline_model_comparison"
RUN_NAME_PREFIX = "ec_baseline"

EPOCHS = 30
BATCH_SIZES_CSV = "4"
LEARNING_RATES_CSV = "3e-5,1e-4"
WEIGHT_DECAYS_CSV = "1e-4"
SEEDS_CSV = "42,43"
MAX_CONFIGURATION_RUNS = 12

HIDDEN_S_VALUES_CSV = "128"
HIDDEN_V_VALUES_CSV = "16"
EDGE_HIDDEN_VALUES_CSV = "64"
GVP_LAYERS_VALUES_CSV = "4"
HEAD_MLP_LAYERS_VALUES_CSV = "2"
EDGE_RADIUS_VALUES_CSV = "8.0"
ESM_FUSION_DIM_VALUES_CSV = "128"
EARLY_ESM_DIM_VALUES_CSV = "32"

EC_LABEL_DEPTHS_CSV = "1"
EC_CONTRASTIVE_WEIGHTS_CSV = "0.0"
EC_GROUP_WEIGHTING = "structure_id"
SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"

ESM_EMBEDDINGS_DIR = ""  # set to your embeddings folder when available
ALLOW_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_ESM_EMBEDDINGS = True
RING_EDGE_MODE = "with_ring"
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

In the **Optional training execution** cell:

```python
LAUNCH_PLANNED_TRAINING_RUNS = True
```

Expected outputs/files:

- Completed run directories for each planned validation-only run
- `<SUMMARY_BASENAME>.csv` and `<SUMMARY_BASENAME>_completed_only.csv`
- No `test_report.json`

Success criteria:

- Each run uses `selection_metric = val_ec_group_level_1_balanced_acc`.
- `split_diagnostics.json` shows usable train/validation EC class coverage at
  depth 1.
- Comparison tables rank only validation rows.

Decision after this stage:

- Choose a baseline anchor by validation evidence, not by held-out test.
- Note whether `Only-ESM` outperforms `Only-GVP` — this is a key signal for EC.
- If Only-ESM is clearly stronger, the ESM-fusion models become the priority.
- If explicitly continuing from this baseline, use the selected simpler anchor
  to constrain the next HPO stage.
- If launching a fresh Optuna check, do not over-constrain it to prior raw
  outputs; search broadly within the selected model family/fusion mode while
  keeping EC depth fixed per study.

## Stage 3 — Small Debug Optuna

Purpose: verify the controlled Optuna path, storage, and search-space parsing
for EC without treating the result as model-selection evidence.

When to use it: first Optuna run in a new runtime or after editing Optuna
configuration fields for EC.

Expected scale/runtime: smoke/debug, minutes to under an hour.

Notebook configuration block:

```python
TASK = "ec"
RUN_MODE = "controlled_hpo_optuna"
RECOMMENDED_RUN_SET = "custom"
MODEL_PRESET = "Only-GVP"
RUN_BATCH_ID = "ec_only_gvp_optuna_debug"
SUMMARY_BASENAME = "ec_only_gvp_optuna_debug"
RUN_NAME_PREFIX = "ec_only_gvp_optuna_debug"

EPOCHS = 10
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"

EC_LABEL_DEPTHS_CSV = "1"
EC_CONTRASTIVE_WEIGHTS_CSV = "0.0"
EC_GROUP_WEIGHTING = "structure_id"

HIDDEN_S_VALUES_CSV = "128"
HIDDEN_V_VALUES_CSV = "16"
EDGE_HIDDEN_VALUES_CSV = "64"
GVP_LAYERS_VALUES_CSV = "4"
HEAD_MLP_LAYERS_VALUES_CSV = "2"
EDGE_RADIUS_VALUES_CSV = "8.0"
RING_EDGE_MODE = "with_ring"

OPTUNA_INTENSITY = "debug"
OPTUNA_TARGET_COMPLETE_TRIALS = 4
MAX_EPOCHS_PER_TRIAL = 3
OPTUNA_SEARCH_PRESET = "first_useful_only_gvp_narrow"
OPTUNA_STUDY_NAME = "ec_only_gvp_optuna_debug"
OPTUNA_STORAGE = ""
OPTUNA_SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"
OPTUNA_LEARNING_RATE_RANGE = "1e-5,3e-4"
OPTUNA_WEIGHT_DECAYS_CSV = "0.0,1e-5,1e-4"
OPTUNA_BATCH_SIZES_CSV = "4,8"
RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = False

INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

In the **Optional training execution** cell:

```python
LAUNCH_PLANNED_TRAINING_RUNS = True
```

Expected outputs/files:

- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/all_trials.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/top_trials.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/best_trial.json`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/optuna_study_summary.md`
- `top_reevaluation_commands.txt`

Success criteria:

- Optuna launches and completes the debug trials against the EC task.
- Trial commands omit held-out test evaluation.
- Best-trial summary uses `val_ec_group_level_1_balanced_acc`.

Decision after this stage:

- If debug Optuna works, move to medium controlled Optuna.
- Do not choose hyperparameters from this debug run.

## Stage 4 — Controlled Medium Optuna Search

Purpose: run a useful but bounded HPO pass for EC inside one selected model
family, with EC label depth and contrastive weight fixed.

When to use it: after baseline behavior is understood and a model family is
selected for EC HPO, usually Only-GVP first.

Expected scale/runtime: medium validation run to serious run, hours.

Notebook configuration block for first useful Only-GVP EC HPO:

```python
TASK = "ec"
RUN_MODE = "controlled_hpo_optuna"
RECOMMENDED_RUN_SET = "custom"
MODEL_PRESET = "Only-GVP"
RUN_BATCH_ID = "ec_only_gvp_optuna_medium"
SUMMARY_BASENAME = "ec_only_gvp_optuna_medium"
RUN_NAME_PREFIX = "ec_only_gvp_optuna_medium"

EPOCHS = 50
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"

EC_LABEL_DEPTHS_CSV = "1"
EC_CONTRASTIVE_WEIGHTS_CSV = "0.0"
EC_GROUP_WEIGHTING = "structure_id"

HIDDEN_S_VALUES_CSV = "128"
HIDDEN_V_VALUES_CSV = "16"
EDGE_HIDDEN_VALUES_CSV = "64"
GVP_LAYERS_VALUES_CSV = "4"
HEAD_MLP_LAYERS_VALUES_CSV = "2"
EDGE_RADIUS_VALUES_CSV = "8.0"
RING_EDGE_MODE = "with_ring"

OPTUNA_INTENSITY = "first_useful"
OPTUNA_SEARCH_PRESET = "first_useful_only_gvp_narrow"
OPTUNA_STUDY_NAME = "ec_only_gvp_optuna_medium"
OPTUNA_STORAGE = "sqlite:////content/drive/MyDrive/DeepMzyme/optuna/ec_only_gvp_optuna_medium.db"
OPTUNA_SPLIT_SEED = 42
OPTUNA_SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"
OPTUNA_LEARNING_RATE_RANGE = "1e-5,3e-4"
OPTUNA_WEIGHT_DECAYS_CSV = "0.0,1e-5,1e-4"
OPTUNA_BATCH_SIZES_CSV = "4,8"
RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = False
TOP_K_CONFIGS_FOR_SEED_REPEAT = 3
REPEAT_SEEDS = "42,123,2026"

INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

Expected outputs/files:

- Optuna study directory under `<RUNS_DIR>/optuna/`
- `all_trials.csv`, `top_trials.csv`, `best_trial.json`
- `optuna_best_config.json`, `best_config_command.txt`
- `top_reevaluation_commands.txt`

Success criteria:

- Best-trial summary uses `val_ec_group_level_1_balanced_acc`.
- Trial logs show validation-only runs.
- Top candidates are plausible and not dominated by missing-class diagnostics.

Decision after this stage:

- Do not pick the final model from one Optuna trial alone.
- Run top-K seed-repeat validation before considering a configuration stable.

## Stage 5 — Large Extensive Optuna Search

Purpose: perform a longer, controlled EC search after the simpler baseline and
medium HPO justify the model family and search axes.

When to use it: after at least one medium HPO or seed-repeat batch identifies
the model family and axes worth expanding, or when the user asks for a fresh
broad Optuna check and does not explicitly ask to rely on previous raw outputs.

Expected scale/runtime: large Optuna search, potentially very long or
overnight. A 200-trial run can be substantially longer than one night.

Important scope rule: fix `EC_LABEL_DEPTHS_CSV` to one depth for the full
study. Mixing depths inside a single study makes the metric comparison
meaningless. Fix the depth, complete the study, then start a new study for a
different depth.

### 5A — 200-Trial Only-GVP EC Capacity Search

Notebook configuration block:

```python
TASK = "ec"
RUN_MODE = "controlled_hpo_optuna"
RECOMMENDED_RUN_SET = "custom"
MODEL_PRESET = "Only-GVP"
RUN_BATCH_ID = "ec_only_gvp_optuna_200_capacity"
SUMMARY_BASENAME = "ec_only_gvp_optuna_200_capacity"
RUN_NAME_PREFIX = "ec_only_gvp_optuna_200_capacity"

EPOCHS = 50
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"
RING_EDGE_MODE = "with_ring"

EC_LABEL_DEPTHS_CSV = "1"
EC_CONTRASTIVE_WEIGHTS_CSV = "0.0"
EC_GROUP_WEIGHTING = "structure_id"

OPTUNA_INTENSITY = "custom"
OPTUNA_TARGET_COMPLETE_TRIALS = 200
MAX_EPOCHS_PER_TRIAL = 50
OPTUNA_SEARCH_PRESET = "later_capacity"
OPTUNA_STUDY_NAME = "ec_only_gvp_optuna_200_capacity"
OPTUNA_STORAGE = "sqlite:////content/drive/MyDrive/DeepMzyme/optuna/ec_only_gvp_optuna_200_capacity.db"
OPTUNA_SPLIT_SEED = 42
OPTUNA_TIMEOUT_MINUTES = 0
OPTUNA_SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"

OPTUNA_LEARNING_RATE_RANGE = "5e-6,3e-4"
OPTUNA_WEIGHT_DECAYS_CSV = "0.0,1e-6,1e-5,1e-4"
OPTUNA_BATCH_SIZES_CSV = "4,8"

OPTUNA_HIDDEN_S_VALUES_CSV = "128,256"
OPTUNA_HIDDEN_V_VALUES_CSV = "16,32"
OPTUNA_EDGE_HIDDEN_VALUES_CSV = "64,128"
OPTUNA_GVP_LAYERS_VALUES_CSV = "2,3,4"
OPTUNA_HEAD_MLP_LAYERS_VALUES_CSV = "1,2"
OPTUNA_EDGE_RADIUS_VALUES_CSV = "6.0,8.0,10.0"

RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = False
TOP_K_CONFIGS_FOR_SEED_REPEAT = 3
REPEAT_SEEDS = "42,123,2026,43,44"
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

### 5B — 200-Trial GVP + Late-Fusion EC Search

Run this only after ESM coverage is valid and simpler baselines justify ESM
fusion for EC.

Notebook configuration block:

```python
TASK = "ec"
RUN_MODE = "controlled_hpo_optuna"
RECOMMENDED_RUN_SET = "custom"
MODEL_PRESET = "GVP + late fusion"
RUN_BATCH_ID = "ec_late_fusion_optuna_200_controlled"
SUMMARY_BASENAME = "ec_late_fusion_optuna_200_controlled"
RUN_NAME_PREFIX = "ec_late_fusion_optuna_200"

EPOCHS = 50
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"
RING_EDGE_MODE = "with_ring"

EC_LABEL_DEPTHS_CSV = "1"
EC_CONTRASTIVE_WEIGHTS_CSV = "0.0"
EC_GROUP_WEIGHTING = "structure_id"

ESM_EMBEDDINGS_DIR = ""  # set to your embeddings folder when available
ALLOW_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_ESM_EMBEDDINGS = True

OPTUNA_INTENSITY = "custom"
OPTUNA_TARGET_COMPLETE_TRIALS = 200
MAX_EPOCHS_PER_TRIAL = 50
OPTUNA_SEARCH_PRESET = "custom"
OPTUNA_STUDY_NAME = "ec_late_fusion_optuna_200_controlled"
OPTUNA_STORAGE = "sqlite:////content/drive/MyDrive/DeepMzyme/optuna/ec_late_fusion_optuna_200_controlled.db"
OPTUNA_SPLIT_SEED = 42
OPTUNA_TIMEOUT_MINUTES = 0
OPTUNA_SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"

OPTUNA_LEARNING_RATE_RANGE = "5e-6,2e-4"
OPTUNA_WEIGHT_DECAYS_CSV = "0.0,1e-6,1e-5,1e-4"
OPTUNA_BATCH_SIZES_CSV = "4,8"

OPTUNA_HIDDEN_S_VALUES_CSV = "128,256"
OPTUNA_HIDDEN_V_VALUES_CSV = "16,32"
OPTUNA_EDGE_HIDDEN_VALUES_CSV = "64,128"
OPTUNA_GVP_LAYERS_VALUES_CSV = "2,3,4"
OPTUNA_HEAD_MLP_LAYERS_VALUES_CSV = "1,2"
OPTUNA_EDGE_RADIUS_VALUES_CSV = "6.0,8.0,10.0"
OPTUNA_ESM_FUSION_DIM_VALUES_CSV = "64,128,256"

RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = False
TOP_K_CONFIGS_FOR_SEED_REPEAT = 3
REPEAT_SEEDS = "42,123,2026,43,44"
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

Expected outputs/files:

- Large persistent SQLite Optuna study in Drive
- Complete Optuna CSV/JSON/Markdown outputs under `<RUNS_DIR>/optuna/`
- Per-trial run directories and logs
- No held-out test report

Success criteria:

- Search space preview shows fixed EC label depth and group weighting.
- No test-report files are created by the HPO runs.
- Top trials improve or clarify EC validation behavior without relying on one
  lucky seed.

Decision after this stage:

- Choose the top 2-3 candidates for seed-repeat validation.
- Do not finalize the model from the raw 200-trial ranking alone.

## Stage 6 — Top-K Seed-Repeat Validation

Purpose: confirm whether top HPO candidates are stable across random seeds.

When to use it: after a medium or large Optuna search has produced top
candidates.

Expected scale/runtime: serious run, long or overnight.

```python
RUN_MODE = "controlled_hpo_optuna"
RECOMMENDED_RUN_SET = "custom"
RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = True
TOP_K_CONFIGS_FOR_SEED_REPEAT = 3
REPEAT_SEEDS = "42,123,2026,43,44"
ALLOW_SEED_REPEAT_MODEL_PRESET_MISMATCH = False
RETRAIN_BEST_CONFIG_AFTER_HPO = False

EPOCHS = 50
SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"
OPTUNA_SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

If the Optuna study is already complete and `RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION`
was not set, inspect:

```text
<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/top_reevaluation_commands.txt
```

Those commands omit held-out test evaluation. Run only the top-K commands you
predeclare, with the seed list you predeclare.

Expected outputs/files:

- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/seed_repeat_results.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/seed_repeat_summary.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/seed_repeat_summary.json`
- One validation-only run directory per top-K/seed pair
- No `test_report.json`

Success criteria:

- All top-K/seed runs complete or failures are documented.
- The selected candidate has the best validation mean or an acceptable
  mean/variance tradeoff on `val_ec_group_level_1_balanced_acc`.
- Diagnostics do not show missing validation EC classes or invalid split
  coverage.

Decision after this stage:

- Select one final configuration using validation evidence only.
- Record the mean validation score, variability, per-class diagnostics, split,
  seed list, depth, and epoch budget.
- Train/refit one final EC model from that frozen configuration. The final
  training/refit run must not include held-out test evaluation and must record
  `run_config.json` and `run_metadata.json`.
- Only after that final training/refit run is frozen, move to Stage 7.

## Stage 7 — Final Held-Out Test Evaluation

Purpose: report held-out test performance for the frozen final training/refit
run produced from the final validation-selected EC configuration.

When to use it: only after model family, hyperparameters, EC label depth,
contrastive weight, checkpoint-selection metric, seed-repeat interpretation,
final training/refit run, and final source checkpoint are fixed.

**EC split policy reminder**: the primary final-test route remains unresolved.
Do not launch this stage until a scientifically approved route is documented,
and do not report the exact/possibly-overlapped split as the main final EC
result.

First run the **Select final run and show saved outputs** cell:

```python
FINAL_RUN_SELECTION_MODE = "auto_best_validation"
FINAL_RUN_TABLE_INDEX = 1
FINAL_RUN_DIR = ""
FINAL_REPORT_BASENAME = "deepmzyme_ec_final_selected_run"
```

Then run the **Optional final held-out test evaluation** cell in preview mode:

```python
FINAL_TEST_WORKFLOW = "preview_only"
LAUNCH_FINAL_HELD_OUT_TEST_EVAL = False
CONFIRM_ONE_SHOT_POLICY = False
FINAL_TEST_SOURCE_RUN_DIR = ""
FINAL_TEST_SOURCE_CHOICE_INDEX = 0
FINAL_TEST_BATCH_PARENT_DIR = ""
FINAL_TEST_BATCH_RUN_GLOB = "*"
FINAL_TEST_RUN_NAME_PREFIX = "ec_final_test"
FINAL_TEST_BATCH_SUMMARY_BASENAME = "ec_final_test_batch_summary"
FINAL_TEST_METAL_REPORT_VIEW = "use_METAL_REPORT_VIEW"
ALLOW_REPEAT_FINAL_TEST_EVAL = False
ALLOW_MIXED_FINAL_TEST_BATCH = False
```

Inspect the pre-flight checklist. If the selected source run is the final
training/refit run derived from the validation-selected EC configuration and
this is final reporting, switch to launch:

```python
FINAL_TEST_WORKFLOW = "evaluate_selected_checkpoint"
LAUNCH_FINAL_HELD_OUT_TEST_EVAL = True
CONFIRM_ONE_SHOT_POLICY = True
FINAL_TEST_SOURCE_RUN_DIR = ""
FINAL_TEST_SOURCE_CHOICE_INDEX = 0
FINAL_TEST_RUN_NAME_PREFIX = "ec_final_test"
FINAL_TEST_METAL_REPORT_VIEW = "use_METAL_REPORT_VIEW"
ALLOW_REPEAT_FINAL_TEST_EVAL = False
ALLOW_MIXED_FINAL_TEST_BATCH = False
```

Expected outputs/files:

- A new final-test run folder under the resolved `RUNS_DIR`
- `test_report.json` in the final-test output folder
- Updated final-test summary CSV/PNG
- The source final training/refit run remains unchanged

Success criteria:

- The source run has validation-selected checkpoint metadata, or a documented
  Stage-6-fixed epoch/checkpoint rule if a future full-train refit path is used.
- The final-test run uses `best_model_checkpoint.pt` or the explicitly selected
  fixed checkpoint.
- The output folder is separate from the source final training/refit run.
- The source run is the final training/refit run derived from Stage 6 evidence,
  not a raw Optuna trial, arbitrary seed-repeat run, or provisional validation
  checkpoint.
- The test report includes EC level-1 metrics (and deeper levels when trained).
- The split in `dataset_summary.json` confirms the scientifically approved
  final route and its group-overlap checks.

Decision after this stage:

- Report final held-out EC metrics: level-1 balanced accuracy, macro F1, and
  per-class recall across all seven EC first-digit classes.
- If deeper EC levels were trained, also report the depth-matched metrics.
- Do not choose a different configuration because another candidate has a
  better held-out test score. If more development is needed, return to
  validation-only experiments and treat this test result as already spent.

## EC Label Depth Progression

After the depth-1 standalone validation baseline is stable and the depth-1
scientific cycle has a predeclared reason to expand, a separate depth-2
experiment cycle starts from Stage 1 using the same structure but:

```python
EC_LABEL_DEPTHS_CSV = "2"
SELECTION_METRIC = "val_ec_group_level_2_balanced_acc"
OPTUNA_SELECTION_METRIC = "val_ec_group_level_2_balanced_acc"
RUN_BATCH_ID = "ec_only_gvp_depth2_baseline_lr_seed"
```

Keep depth-1 and depth-2 run batches separate. Do not compare depth-1 and
depth-2 validation metrics directly. Do not open a held-out test merely to
authorize deeper development; each depth keeps its own validation-only
selection and final-reporting policy.

## Contrastive Loss Exploration

After a clean depth-1 baseline is established with `EC_CONTRASTIVE_WEIGHTS_CSV = "0.0"`,
run a narrow controlled comparison with contrastive loss enabled. This is a
Stage 2-style manual comparison, not an Optuna stage:

```python
EC_CONTRASTIVE_WEIGHTS_CSV = "0.0,0.1,0.5"
EC_CONTRASTIVE_TEMPERATURE = 0.1
RUN_BATCH_ID = "ec_contrastive_comparison"
SUMMARY_BASENAME = "ec_contrastive_comparison"
```

Keep the architecture fixed to the best depth-1 Only-GVP or fusion anchor.
Compare by `val_ec_group_level_1_balanced_acc` only.

## Safety Guards To Check

Before any reportable comparison or HPO launch, confirm:

- `INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False`
- `FINAL_TEST_WORKFLOW = "preview_only"` or the final-test cell has not been run
- `VAL_FRACTION > 0` or a fold split is explicitly configured
- `SELECTION_METRIC = "val_ec_group_level_1_balanced_acc"` (or matching depth level)
- `EC_GROUP_WEIGHTING = "structure_id"`
- `EC_LABEL_DEPTHS_CSV` is fixed to one depth value per study
- `ALLOW_SHORT_TRAINING_FOR_DEBUG = False` for reportable runs
- `ALLOW_SEED_REPEAT_MODEL_PRESET_MISMATCH = False`
- `ALLOW_MIXED_FINAL_TEST_BATCH = False`
- `ALLOW_REPEAT_FINAL_TEST_EVAL = False`
- The primary final-test route has been resolved scientifically, and the saved
  dataset identity and group-overlap checks match that approved route
