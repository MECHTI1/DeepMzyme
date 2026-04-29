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
### 4) Do many parameters and model-types configurable
- The parameters include:
  - `epochs`
  - `batch_size`
  - `learning_rate`
  - `weight_decay`
  - `seed`
  - `val_fraction`
  - `n_folds`
  - `fold_index`
  - `edge_radius`
  - `hidden_s`
  - `hidden_v`
  - `edge_hidden`
  - `gvp_layers`
  - `esm_fusion_dim`
  - `head_mlp_layers`
  - `node_rbf_sigma`
  - `edge_rbf_sigma`
  - `node_feature_set`
  - `cross_attention_layers`
  - `cross_attention_heads`
  - `cross_attention_dropout`
  - `cross_attention_bidirectional` — bool
  - `early_esm_dropout`
  - `{class}_loss_multiplier`
  - `class_loss_function` — default: `cross_entropy`
  - `metal_focal_gamma`
  - `lr_schedule` — fixed, cosine, step, or more if needed
  - `lr_step_size`
  - `require_ring_edges` — bool
  - `allow_missing_esm_embeddings` — off by default        
- The models include: 
    - GVP+ESM, Only-ESM, Only-GVP, SimpleGNN+ESM
    - fusion mode-For those which include The ESM+graph-model, Try: Late-fusion, Early-fusion, Node-Level Late Fusion, Hybrid and Cross-Modal Attention

### 5) Create a Google Colab configurable training/test set
    - Do the training model flexibile by configurable options to input screening different parameters/models, make convinent nice interface for inputs.
    - In the end make a comparison table/proffesional figure for analyse prediction results which including all selected screened variety of parametrs/differrent models of choice.

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