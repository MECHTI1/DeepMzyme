# DeepMzyme Training Notebook — Fix Prompt

You are editing `DeepMzyme_training_colab.ipynb`, a Jupyter/Colab training notebook for the DeepMzyme project. Below is an ordered list of concrete bugs and design problems to fix. For each issue the location, the problem, and the required fix are specified. Make only the changes described — do not refactor unrelated code.

---

## Issue 1 — NODE_FEATURE_SET: Replace useless single-option dropdown with sub-feature checkboxes

**Location:** `#@title Default run values` cell (Section 6) and `#@title Interactive configuration panel` cell.

**Problem:** `NODE_FEATURE_SET = 'conservative'  #@param ['conservative']` is a dropdown with exactly one option — nothing to select. In the widget panel, `_multi_select_widgets['NODE_FEATURE_SETS']` also lists only `['conservative']`. The paired `_multi_custom_widgets['NODE_FEATURE_SETS']` text box is labeled `'Custom node features CSV'` with placeholder `'currently only conservative'`, making both controls useless.

**Fix:**

1. Define a list of the individual node features that make up the conservative set (e.g. `amino_acid_onehot`, `backbone_dihedrals`, `sidechain_angles`, `sasa`, `secondary_structure`, `bfactor`, `charge`, `hydrophobicity` — adjust to match what `src/train.py --node-feature-set conservative` actually loads). Call this `CONSERVATIVE_NODE_FEATURES`.

2. Replace the `NODE_FEATURE_SET = 'conservative'  #@param ['conservative']` line with a comment explaining that sub-feature selection is controlled below, and set `NODE_FEATURE_SET = 'conservative'` as a fixed default.

3. In the widget panel, replace the `_multi_select_widgets['NODE_FEATURE_SETS']` entry with a new `_multi_select` over `CONSERVATIVE_NODE_FEATURES` — all items selected by default. Name the widget `NODE_FEATURE_SUBSET`. Remove the `_multi_custom_widgets['NODE_FEATURE_SETS']` text box entirely.

4. In the "Read widget values" cell, read `NODE_FEATURE_SUBSET` from the new widget and derive `NODE_FEATURE_SET` by:
   - If all features are selected → pass `'conservative'` (existing CLI value, no change needed)
   - If a subset is selected → build a comma-separated string of selected features and pass it as `--node-feature-set` (assuming `train.py` accepts this; if not, add a `--omit-node-features` flag approach instead — see Option B below)

   **Option B (simpler if `train.py` can't accept a subset string):** Keep `NODE_FEATURE_SET = 'conservative'` always, but add a separate `NODE_FEATURES_OMIT` multi-select (default: nothing selected) showing the full conservative feature list. Pass the selected omissions as `--omit-node-features feature1,feature2` to `train.py`. In `build_train_command`, add:
   ```python
   if config.get('omit_node_features'):
       cmd.extend(['--omit-node-features', ','.join(config['omit_node_features'])])
   ```
   Choose Option A or Option B based on what `src/train.py` supports.

5. Remove `NODE_FEATURE_SET_OPTIONS = ['conservative']` from the validation list and remove the `_validate_choices('NODE_FEATURE_SETS', ...)` call, since the new control no longer sweeps over named presets.

---

## Issue 2 — Remove redundant custom CSV text boxes for LR schedules and node features

**Location:** `#@title Interactive configuration panel` cell — `_multi_custom_widgets` dict.

**Problem:** `_multi_custom_widgets['LR_SCHEDULES']` is a text box labeled `'Custom LR schedules CSV'` with placeholder `'fixed,cosine,step'`. All three valid LR schedule values are already fully selectable in `_multi_select_widgets['LR_SCHEDULES']`. The custom box adds nothing and is confusing.

Similarly, `_multi_custom_widgets['NODE_FEATURE_SETS']` (placeholder `'currently only conservative'`) is being removed as part of Issue 1 above.

**Fix:**

1. Remove the `'LR_SCHEDULES': _text(...)` entry from `_multi_custom_widgets`.
2. In `_selected_values('LR_SCHEDULES')`, the function merges `_multi_select_widgets[name].value` with `_csv_tokens(_multi_custom_widgets[name].value)`. After removing the custom widget, update `_selected_values` to skip the custom lookup for `LR_SCHEDULES` (or handle the missing key gracefully with `.get()`).
3. Remove the custom LR schedules text box from whatever section/VBox it is placed in within the panel layout.

---

## Issue 3 — LR_STEP_SIZE and LR_DECAY_GAMMA: Add conditional visibility based on LR schedule selection

**Location:** `#@title Interactive configuration panel` cell — `_run_widgets` and the panel layout section that contains `LR_STEP_SIZE` / `LR_DECAY_GAMMA`.

**Problem:** `LR_STEP_SIZE` and `LR_DECAY_GAMMA` are only meaningful when `lr_schedule == 'step'`, but they are always visible. The panel already conditionally hides sections using `_set_visible` and `observe` (e.g. `_metal_section`, `_ec_section`, `_cross_attention_section`). These two fields should follow the same pattern.

**Fix:**

1. Wrap the `LR_STEP_SIZE` and `LR_DECAY_GAMMA` widgets in a `VBox` or a named section (e.g. `_lr_step_section`).
2. Add an observer on `_multi_select_widgets['LR_SCHEDULES']`:
   ```python
   def _refresh_lr_step_section(_change=None):
       _set_visible(_lr_step_section, 'step' in _multi_select_widgets['LR_SCHEDULES'].value)
   if hasattr(_multi_select_widgets['LR_SCHEDULES'], 'observe'):
       _multi_select_widgets['LR_SCHEDULES'].observe(_refresh_lr_step_section, names='value')
   _refresh_lr_step_section()
   ```
3. Call `_refresh_contextual_sections` or `_refresh_lr_step_section` at initial render time.

---

## Issue 4 — Sweep config cell hardcodes `LR_SCHEDULES = ['fixed']` instead of reading from `LR_SCHEDULE`

**Location:** `#@title Sweep configuration` cell (Section 7).

**Problem:** Every other sweep list variable reads from its corresponding single-value default — e.g. `LEARNING_RATES = [LEARNING_RATE]`, `SEEDS = [SEED]` — but LR schedules breaks the pattern with a hardcoded `LR_SCHEDULES = ['fixed']`. If a user sets `LR_SCHEDULE = 'cosine'` in the defaults cell and skips the widget panel, their choice is silently ignored.

**Fix:** Change:
```python
LR_SCHEDULES = ['fixed']
```
to:
```python
LR_SCHEDULES = [LR_SCHEDULE]
```

---

## Issue 5 — Remove `ALLOW_SINGLE_AND_SWEEP = True` dead code

**Location:** `#@title Default run values` cell (Section 6), near line `ALLOW_SINGLE_AND_SWEEP = True`.

**Problem:** This variable is assigned but never referenced anywhere else in the notebook. It is dead code that misleads readers into thinking it controls some behavior.

**Fix:** Delete the line `ALLOW_SINGLE_AND_SWEEP = True`.

---

## Issue 6 — Cross-attention params in Run tab lack conditional visibility

**Location:** `#@title Interactive configuration panel` cell — the Run tab section and `_run_widgets` entries for `CROSS_ATTENTION_DROPOUT`, `CROSS_ATTENTION_NEIGHBORHOOD`, `CROSS_ATTENTION_BIDIRECTIONAL`.

**Problem:** The Grid tab already conditionally hides `_cross_attention_section` when `GVP + cross-modal attention` is not selected. But the Run tab shows `CROSS_ATTENTION_DROPOUT`, `CROSS_ATTENTION_NEIGHBORHOOD`, and `CROSS_ATTENTION_BIDIRECTIONAL` unconditionally, even when the selected model preset makes them irrelevant.

**Fix:**

1. Wrap the three cross-attention run widgets in a `VBox` (e.g. `_run_cross_attention_section`).
2. Add an observer on `_multi_select_widgets['MODEL_PRESETS']` — reuse or extend `_refresh_contextual_sections`:
   ```python
   def _refresh_contextual_sections(_change=None):
       _set_visible(_metal_section, _has_metal_task())
       _set_visible(_ec_section, _has_ec_task())
       _set_visible(_cross_attention_section, _has_cross_attention_model())
       _set_visible(_run_cross_attention_section, _has_cross_attention_model())
   ```
3. Ensure `_refresh_contextual_sections()` is called at initial render.

---

## Issue 7 — `preset_map` contains dead data entries that mislead readers

**Location:** `#@title Default run values` cell (Section 6) — the `preset_map` dict.

**Problem:** The `GVP + early fusion` and `GVP + hybrid fusion` entries in `preset_map` include `'early_esm_dim': 32` and `'early_esm_dropout': 0.2`. However, `build_run_config` only reads `model_architecture`, `fusion_mode`, and `uses_esm` from the preset dict. The `early_esm_dim` and `early_esm_dropout` preset values are completely ignored — the widget globals `EARLY_ESM_DIM` and `EARLY_ESM_DROPOUT` always win. This implies false defaults that don't actually apply.

**Fix:** Remove the `'early_esm_dim'` and `'early_esm_dropout'` keys from all entries in `preset_map`. Add a comment near the `EARLY_ESM_DIM` and `EARLY_ESM_DROPOUT` widget/default definitions noting that these are the actual defaults for early fusion presets.

---

## Issue 8 — `drive_runs_dir` NameError in Section 11 report cell

**Location:** `#@title Generate run summary` cell (Section 11).

**Problem:** The cell does:
```python
summary_runs_dir = local_runs_dir if SUMMARY_SOURCE == 'local_runs' else drive_runs_dir
```
But `drive_runs_dir` is only assigned inside the copy cell (Section 10), which may not have run (it is skipped in local mode or when Drive is not mounted). If `SUMMARY_SOURCE == 'drive_outputs'` and Section 10 was not run, this raises `NameError: name 'drive_runs_dir' is not defined`.

**Fix:** At the top of Section 11, add a safe fallback:
```python
if 'drive_runs_dir' not in globals():
    drive_runs_dir = drive_output_dir / 'runs'
    print('Warning: drive_runs_dir was not set by the copy cell; defaulting to', drive_runs_dir)
```

---

## Issue 9 — `CROSS_ATTENTION_LAYERS` / `CROSS_ATTENTION_HEADS` change type between cells

**Location:** `#@title Default run values` cell (Section 6) and `#@title Sweep configuration` cell (Section 7).

**Problem:** In Section 6, `CROSS_ATTENTION_LAYERS` is an int (from `#@param`). In Section 7, the sweep cell renames it via `CROSS_ATTENTION_LAYERS = [SINGLE_CROSS_ATTENTION_LAYERS]`, changing the type to list. The `SINGLE_*` aliases exist only to bridge this type change. If a user runs Section 8 (`Build Commands`) after Section 6 but before Section 7, `CROSS_ATTENTION_LAYERS` is still an int and `build_sweep_runs()` crashes.

**Fix:**

1. In Section 6, initialize `CROSS_ATTENTION_LAYERS` and `CROSS_ATTENTION_HEADS` directly as single-element lists:
   ```python
   CROSS_ATTENTION_LAYERS = [1]  #@param {type:"integer"} — will be wrapped in list
   CROSS_ATTENTION_HEADS = [4]   #@param {type:"integer"}
   ```
   Or, if `#@param` must emit an int, add immediately after the `#@param` lines:
   ```python
   CROSS_ATTENTION_LAYERS = [CROSS_ATTENTION_LAYERS]
   CROSS_ATTENTION_HEADS = [CROSS_ATTENTION_HEADS]
   ```
2. Remove the `SINGLE_CROSS_ATTENTION_LAYERS` and `SINGLE_CROSS_ATTENTION_HEADS` intermediate aliases from Section 7 — they are no longer needed.
3. Update Section 7 to simply preserve the existing lists: `CROSS_ATTENTION_LAYERS = CROSS_ATTENTION_LAYERS` (no-op, just for documentation clarity, or remove entirely since the lists are already set).

---

## Issue 10 — EC sweep values silently dropped for non-EC tasks with no warning

**Location:** `build_sweep_runs()` function in Section 8.

**Problem:** When `task_mode` is `metal_6_class` or `metal_collapsed4_metric`, `build_sweep_runs()` silently replaces `EC_LABEL_DEPTHS` with `[EC_LABEL_DEPTH]` and `EC_CONTRASTIVE_WEIGHTS` with `[EC_CONTRASTIVE_WEIGHT]`. If a user has configured multiple EC depths for a sweep but selected a metal-only task, they get fewer runs than expected with no indication why.

**Fix:** At the start of `build_sweep_runs()`, before the main product loop, add:
```python
non_ec_tasks = [t for t in TASK_MODES if t not in {'ec_prediction', 'joint_metal_ec'}]
if non_ec_tasks and (len(EC_LABEL_DEPTHS) > 1 or len(EC_CONTRASTIVE_WEIGHTS) > 1):
    print(
        f'Note: EC_LABEL_DEPTHS={EC_LABEL_DEPTHS} and EC_CONTRASTIVE_WEIGHTS={EC_CONTRASTIVE_WEIGHTS} '
        f'are ignored for non-EC task modes {non_ec_tasks}. '
        'They only expand the sweep for ec_prediction and joint_metal_ec tasks.'
    )
```

---

## Issue 11 — No GPU memory guidance for Colab users

**Location:** Markdown cell in Section 6 (the training config documentation block) or as an `_info(...)` widget in the Grid tab.

**Problem:** Colab T4 GPUs have ~15 GB VRAM. Users varying `hidden_s` (128→192), `hidden_v` (16→32), or `gvp_layers` (4→6+) can silently OOM mid-sweep with no prior warning. There is no guidance on which config sizes are safe for a T4.

**Fix:** Add a brief note to the Section 6 markdown block (and/or as an `_info` widget in the Grid → GVP capacity section):

```
GPU memory guidance (Colab T4, ~15 GB):
- Safe baseline: gvp_layers=4, hidden_s=128, hidden_v=16, edge_hidden=64, batch_size=8
- Approaching limit: hidden_s=192 or hidden_v=32 — reduce batch_size to 4 if OOM
- gvp_layers >= 6 with hidden_s=192: likely OOM on T4 at batch_size=8; use batch_size=4 or smaller
- Only-ESM preset uses much less VRAM than GVP variants
```

---

## Summary of changes by file location

| Cell title | Changes |
|---|---|
| `#@title Default run values` | Remove `ALLOW_SINGLE_AND_SWEEP`; fix `CROSS_ATTENTION_LAYERS`/`HEADS` to list; remove dead `early_esm_dim`/`dropout` from `preset_map`; add GPU note to markdown above |
| `#@title Sweep configuration` | Fix `LR_SCHEDULES = [LR_SCHEDULE]`; remove `SINGLE_*` aliases |
| `#@title Interactive configuration panel` | Replace node feature multi-select; remove custom LR CSV box; add `_lr_step_section` conditional; add `_run_cross_attention_section` conditional |
| `#@title Read widget values` | Update `_selected_values` for removed LR custom widget; read new `NODE_FEATURE_SUBSET` widget |
| Section 8 (`build_sweep_runs`) | Add EC sweep warning for non-EC tasks |
| Section 11 (`Generate run summary`) | Add `drive_runs_dir` fallback guard |
