### 1) Create training/test sets CSVs with summary of metal-type and EC-number
- Used for the classification tasks of metal-type and EC-number, as start, the PinMyMetal training/test sets
- The CSV will be written in the following format: structure name, EC number/s, Metal-Type
- Important: Be sure that the only Metal-Type found in structures files, are those found in the CSV (and opposite).
- Create a colab-bundle which contains all the training/test structure files and CSV files. Should be compressed.

### 2) Train the Metal classification model
- Train/validate on 6 classes: Mn,Fe,Zn,Cu,Co and Ni separately.
  - best_checkpoint = highest validation balanced accuracy 
- Test the test-set on prediction performance on thr 6 Classes and on 4 classes, whereas Fe+Ni+Co are concatenated to VIII class.

### 3) Train the EC-number classification model
- Train/validate on all EC classes first digit **AND** following digits (Need yet to think how many digit to train)
- Use contrastive learning
- Test on the test-set on all levels of digits so will have broad view on the prediction performance.
- 
### 4) Make important parameters and model types configurable

The current training entry point is `src/train.py`, which delegates to `src/training/config.py`. The Colab notebook should expose the commonly used controls below and document the rest clearly enough that advanced users can reproduce a command-line run.

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
| Task | `--task` | `joint`; choices `joint`, `metal`, `ec` | Selects metal-only, EC-only, or joint prediction heads and losses. | Expose |
| Training | `--epochs` | `10` | Maximum number of training epochs. | Expose |
| Training | `--batch-size` | `8` | Number of pocket graphs per mini-batch. | Expose / sweep |
| Training | `--learning-rate` | `3e-4` | Optimizer step size. Previous serious baselines often start at `3e-5`. | Expose / sweep |
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
| Model size | `--gvp-layers` | `4` | Number of graph message-passing layers. | Expose / sweep |
| Model size | `--head-mlp-layers` | `2` | Number of linear layers in metal/EC classifier heads. | Expose / sweep |
| Graph construction | `--edge-radius` | project default currently `8.0` in code | Residue-neighbor radius for graph edges before optional RING edges. | Expose / sweep |
| Node/edge encoders | `--node-feature-set` | `conservative` only | Named set of residue/node features. Only `conservative` is currently implemented. | Expose |
| Node/edge encoders | `--node-rbf-sigma` | `0.75` | Width of distance radial-basis features for node distance features. | Advanced |
| Node/edge encoders | `--edge-rbf-sigma` | `0.75` | Width of distance radial-basis features for edge distance features. | Advanced |
| Node/edge encoders | `--node-rbf-use-raw-distances` | false | Uses raw, unnormalized node distances for node RBF expansion when available. | Advanced |
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
| RING edges | `--use-ring-edges` | false | Adds RING interaction edges in addition to radius edges when files are available. | Expose via mode |
| RING edges | `--require-ring-edges` | false | Fails if RING edge files are missing for requested structures. | Expose with warning |
| RING edges | `--prepare-missing-ring-edges` | false flag, but current config prepares by default unless disabled | Generate missing RING edge files during preflight when RING is active. Notebook should keep default radius-only and no required RING. | Expose |
| RING edges | `--no-prepare-missing-ring-edges` | false | Prevents RING generation during preflight. | Expose as prepare-missing toggle |
| Metal loss | `--balance-metal-site-symbols` | false | Uses a weighted sampler to balance metal classes and Co/Ni symbols inside Class VIII. | Expose |
| Metal loss | `--metal-loss-function` | `cross_entropy`; choices `cross_entropy`, `focal` | Loss function for metal classification. | Expose |
| Metal loss | `--metal-focal-gamma` | `2.0` | Focal-loss gamma when focal loss is selected. | Expose |
| Metal loss | `--metal-label-smoothing` | `0.0` | Label smoothing for metal cross-entropy. | Expose |
| Metal loss | `--mn-loss-multiplier`, `--cu-loss-multiplier`, `--zn-loss-multiplier`, `--fe-loss-multiplier`, `--co-loss-multiplier`, `--ni-loss-multiplier`, `--class-viii-loss-multiplier` | `1.0` each | Per-class multipliers applied to metal class weights. | Advanced |
| Joint loss | `--metal-loss-weight` | `1.0` | Task-level multiplier for the metal loss in joint or metal runs. | Expose |
| Joint loss | `--ec-loss-weight` | `1.0` | Task-level multiplier for the EC loss in joint or EC runs. | Expose |
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

### 5) Create a Google Colab configurable training/test set
    - Do the training model flexibile by configurable options to input screening different parameters/models, make convinent nice interface for inputs.
    - In the end make a comparison table/proffesional figure for analyse prediction results which including all selected screened variety of parametrs/differrent models of choice.
    - Colab bundles should include the site-level MAHOMES train/test summary CSVs as the training source of truth. Structure-level CSV artifacts may contain semicolon-joined metal labels for structures with multiple catalytic metal sites; these artifacts are for inspection and should not replace site-level labels for single-label metal training.

---

### 6) Experiment tracking and reproducible run summaries

Every training run should save enough information to reproduce and compare the result.

Each run should save:

- full config / hyperparameters
- random seed
- dataset paths and split identity
- model architecture
- fusion mode
- node feature set
- EC label depth, if relevant
- contrastive-learning settings, if relevant
- validation metric used for checkpoint selection
- selected checkpoint path
- final held-out test metrics, if test evaluation was requested
- git commit hash, if available

Create a reporting script that summarizes multiple run directories into one CSV table.

The summary table should include, when available:

- run name
- task
- model architecture
- fusion mode
- seed
- node feature set
- EC label depth
- selection metric
- best validation metrics
- final held-out test metrics
- metal 6-class metrics
- metal collapsed-4 metrics
- EC level-1 / level-2 metrics
- split name/type used for the run
- whether train/test overlap was detected

Important rules:

- Validation metrics are used for checkpoint selection and hyperparameter choice.
- Held-out test metrics are used only for final reporting.
- Do not choose models by repeatedly checking the held-out test set.

---

### 7) Baseline-first model comparison policy

Before testing complex fusion models, establish clean baselines.

Recommended order:

1. Only-GVP
2. Only-ESM
3. GVP + simple late ESM fusion
4. GVP + early residue-level ESM fusion
5. More complex fusion modes only if simpler baselines justify them

Complex fusion modes include:

- hybrid fusion
- node-level late fusion
- cross-modal attention

For each task:

- compare models using validation metrics first
- select checkpoints using validation metrics
- evaluate the selected model once on the held-out test set for final reporting

The goal is to avoid adding complex architecture before proving that it improves over simple baselines.

---

### 8) Data leakage and split policy

The non-overlapped PinMyMetal split is the main trusted split for final held-out evaluation.

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

Use the held-out test set only for final reporting of selected models.
