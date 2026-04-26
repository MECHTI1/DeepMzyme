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