## Stopped PDB-grouped L4 draft

Archived executable plan, preserved to explain the stopped allocation.
**Do not run these commands.** The corrected protocol is in the
[metal playbook](../../METAL_TRAINING_PIPELINE_PLAYBOOK.md#exact-pocket-antigravity-matched-l4-diagnostic).

**Status: stopped at user request; do not resume this frozen study.** It used
PDB-ID-grouped folds, whereas the requested comparison must match the recent
Antigravity pocket-level protocol. Only two one-epoch smokes completed; no
full fit or result comparison exists. The main training files remain
unchanged. L4 readiness/fit and stop evidence is tracked in
[`GVP_FUSION_L4_EXECUTION_20260924.md`](../../agents_report/GVP_FUSION_L4_EXECUTION_20260924.md). This
separately named `gvp_fusion_diagnostic_l4_v1` is a focused test of two
findings from the 2026-09-23 GVP/fusion review. It does not reset the budget
or reuse the results of `metal_single_gpu_20h_v2`, the frozen GVP capacity
study, or the already opened exact PinMyMetal test. It is not the required
four-versus-six target matrix or a Stage 6/6B/7 promotion route.

Use the local CPU for source and dataset checks, command generation, tests,
artifact verification, and analysis. Use one **verified Colab L4** for model
training. The recent Antigravity/PyCharm benchmark recorded completed L4
50-epoch folds of about **719–918 seconds**, including held-out evaluation;
those timings are only an admission estimate. Colab hardware, compute-unit
cost, and runtime length can vary. Follow
[`COLAB_GPU_RUNBOOK.md`](../../COLAB_GPU_RUNBOOK.md) and verify the actual GPU with
`colab status -s <owned-session>` before training.

### Frozen comparison

| Field | Exact value |
| --- | --- |
| Dataset | v12 hosted bundle, SHA256 `90c0899829e0ac5ca94a5ef34484b74ca1014d3e9b6b1fba90bd338ee3440dee`; use only `train_and_test_sets_structures_non_overlapped_pinmymetal/train` |
| Task | Metal; direct `four_class` / `merge_fe_class_viii`; `metal_eligibility_scheme=six_class` for one shared eligible cohort |
| Validation | Five `pdbid`-grouped folds, `split_stratify_by=metal_site`, fixed split seed `42`; model seeds `42,43` crossed with every fold and arm |
| Common training | 50 epochs, batch 16, 8 Å edge radius, conservative nodes, raw-distance RBF, no RING/metal nodes/augmentation, legacy site geometry, fixed schedule, weight decay `1e-4`, class-weighted CE |
| Only-GVP | Architecture `only_gvp`, LR `3e-4` |
| Joint | Architecture `gvp`, `late_fusion`, cached ESMC-300m/960, base LR `3e-5`, GVP LR `3e-4` |
| Selection | Highest native `val_metal_balanced_acc`; all other metrics from that same selected checkpoint; no test arguments or test-file access |

Predeclare **four** arms: `G0` = Only-GVP with current summed messages;
`G1` = G0 plus existing `--gvp-normalize-message-aggregation`; `F0` =
current joint optimizer grouping; `F1` = F0 with only `init_vec_proj`,
`gvp_attn_pool`, and `gvp_fusion_proj` moved to the `3e-4` GVP group.
`head_metal`, `fusion_gate`, and ESM modules remain at `3e-5` in F1. This
isolates the readout-rate question. Do not combine G1 and F1 in this grid.
The exact five-class/pocket-level Antigravity benchmark is context, **not** a
control for this new dataset, target, and grouping policy. Its saved
multimodal folds had 58, 56, 43, 49, and 45 train/validation PDB-ID overlaps
under `pocket_id` grouping; the exact train/test membership also shares 177
PDB IDs. Within that historical benchmark, the three model arms did share
the same pocket folds. The new study uses PDB-ID-disjoint folds and a
PDB-ID-disjoint non-overlap train/test source; compare only its own arms on
their shared validation identities. The measured split audit is in the
[execution ledger](../../agents_report/GVP_FUSION_L4_EXECUTION_20260924.md).

### Required updated scripts before any allocation

Implement `scripts/run_gvp_fusion_l4_diagnostic.py` as a new validation-only
runner; do not change the historical benchmark runner's behavior. It must
support `plan`, `run`, `status`, and `summarize`. `plan` writes a frozen
`study_manifest.json` and all **40** full commands (`4 arms × 5 folds × 2
seeds`), binding the worktree source snapshot, bundle SHA256, split seed,
feature policy and per-arm options. Each run name is
`gvp_l4_v1_<arm>_seed<seed>_fold<fold>`. The runner rejects any command
containing a test path, `--run-test-eval`, a held-out overlap override, or a
final-test report identifier. It executes one fit at a time, retains incomplete
attempts, verifies completed runs by config hash, 50 validation rows,
checkpoint and prediction files, and never deletes a prior run to retry it.

Add `--gvp-lr-scope {trunk,branch}` to the trainer, with `trunk` preserving
current behavior. `branch` adds exactly the three F1 module prefixes to the
GVP-rate optimizer group. Save each optimizer group's parameter names,
counts, LR and weight decay in `run_metadata.json`. Add
`--export-selected-val-predictions`: reload the selected checkpoint and save
aligned pocket IDs, PDB IDs, targets, logits, selected epoch, fold, seed and
label scheme to `selected_val_predictions.pt` and a JSON manifest **without
loading a held-out dataset**. Currently validation prediction export is
coupled to the final-test path in `src/training/run.py`; the updated path must
work when `run_test_eval=False`. CPU tests must prove flag defaults reproduce
the current model, optimizer groups are exact, source/split identity checks
reject mismatches, a partial run is not skipped as complete, and no generated
command contains test options.

The generated trainer command uses the following fixed core; the runner fills
the arm, fold, seed, paths and architecture before saving the expanded command.
`--gvp-lr-scope` and validation export are implemented CLI flags in the
isolated study snapshot, not in the shared project's live trainer.

```bash
python -u /content/DeepMzyme/src/train.py \
  --task metal --metal-label-scheme four_class \
  --metal-eligibility-scheme six_class \
  --structure-dir /content/DeepMzyme_Data/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train \
  --summary-csv /content/DeepMzyme_Data/DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv \
  --external-feature-source updated \
  --external-features-root-dir /content/DeepMzyme_Data/DeepMzyme_Data/updated_feature_extraction \
  --runs-dir /content/runs/gvp_fusion_diagnostic_l4_v1 \
  --run-name gvp_l4_v1_<arm>_seed<seed>_fold<fold> \
  --model-architecture <only_gvp_or_gvp> \
  --epochs 50 --batch-size 16 --edge-radius 8 \
  --node-feature-set conservative --metal-node-mode none \
  --site-geometry-features legacy --position-noise-std 0 \
  --second-shell-dropout 0 --outer-residue-dropout 0 \
  --metal-class-weight-mode inverse_frequency \
  --metal-loss-function cross_entropy \
  --learning-rate <3e-4_or_3e-5> --weight-decay 1e-4 \
  --rbf-use-raw-distances --lr-schedule fixed \
  --n-folds 5 --fold-index <fold> --train-val-split-by pdbid \
  --split-stratify-by metal_site --split-seed 42 --seed <seed> \
  --selection-metric val_metal_balanced_acc \
  --gvp-lr-scope <trunk_or_branch> \
  --dataset-bundle-id DeepMzyme_Data_v12_manifest_exact_common70_nonoverlap_clean30_care30_complete_esm_ring_external \
  --dataset-bundle-sha256 90c0899829e0ac5ca94a5ef34484b74ca1014d3e9b6b1fba90bd338ee3440dee \
  --export-selected-val-predictions --device cuda
```

For `G1`, append `--gvp-normalize-message-aggregation`. For `F0/F1`,
append `--fusion-mode late_fusion --gvp-learning-rate 3e-4` and the ESM
embeddings directory; only F1 sets `--gvp-lr-scope branch`. No test or final
report flags are generated. Archive the **current working tree**, including
staged/uncommitted source, and verify its SHA256; a GitHub clone alone is
insufficient. Verify the v12 bundle SHA256 before extraction. Keep stock
Colab PyTorch, install only `requirements/colab-overlay.txt`, check CUDA
architecture support and cached ESM/external coverage, and generate no new
embeddings.

### L4 schedule, budget admission, and result gate

The **proposed**, separately accounted ceiling is **16 cumulative L4
allocation hours**, including setup, failed attempts, transfer, idle time and
verified teardown. It is not an authorization to launch or a completion
guarantee. Use at most four hours per owned session, reserving the last 15
minutes for verified transfer and `colab stop -s <owned-session>`. Measure
smoke, full-fit, memory and transfer costs on the assigned L4. Before
admitting a complete comparison block, forecast remaining fits at **1.25 ×
the worst measured full-fit time**, plus measured operations and shutdown
reserve. Never reduce epochs, seeds or folds to fit a ceiling.

1. CPU stage: implement the flags and runner, test, freeze hashes, save the
   manifest and all commands. Inspect that no held-out path appears.
2. L4 readiness: verify assignment and environment, then run four **one-epoch
   fold-0 smokes** (G0/G1/F0/F1, seed 42). Smoke scores cannot select an arm.
3. Full fits: complete `F0/F1` for all five folds and both seeds (**20 fits**),
   then `G0/G1` on the same fold/seed grid (**20 fits**). Transfer each
   completed run to local durable storage with a checked archive SHA256
   before the next launch. An interrupted fit may restart once from the same
   frozen arm/fold/seed after the original allocation is verified stopped.
   No watchdog may allocate a new VM or fall back to T4 automatically.
4. Analysis: compare `F1−F0` and `G1−G0` on paired fold/seed units.
   Average seeds within each fold, use a **10,000-resample paired fold
   bootstrap 95% CI**, and report native four-class BA, Mn/Cu/Zn/Class VIII
   recalls, all ten paired differences, selected epochs, loss curves and
   late-epoch mean BA. A substantial supported gain requires mean BA
   **≥+0.015**, a CI lower bound **>0**, and no mean class-recall loss
   **>0.03**. Incomplete grids cannot declare a winner.
5. Close out: verify the named session is stopped, then write
   `comparison.csv`, `paired_bootstrap.json`, `class_recall.csv`,
   `artifact_manifest.json`, and `campaign_report.md`. Each fit retains
   `run_config.json`, `run_metadata.json`, split diagnostics, epoch metrics,
   selected checkpoint and validation predictions. No Stage 6B refit or
   Stage 7 test is implied.

If the measured ceiling cannot fit both complete comparisons, finish F0/F1
first and mark G0/G1 incomplete or unrun. The next distinct question is
additive first-shell/global pooling (`F2` against a matched F0). Its model
dimensions, empty-shell fallback, CPU tests, and measured L4 budget must be
frozen in a **separate** block before execution; the existing CA-distance
pooling cutoff is not an equivalent test.
