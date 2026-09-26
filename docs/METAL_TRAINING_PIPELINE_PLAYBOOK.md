# Metal Training Pipeline Playbook

For the new bounded, one-GPU discovery and confirmation campaign use
[metal_single_gpu_20h_v2](#single-gpu-metal-campaign).
The [earlier architecture pilot](#bounded-metal-architecture-pilot--stage-0-through-stage-2b)
and Common70 standalone block remain separate historical recipes. Current
execution and evidence state belongs to [`EXPERIMENT_STATUS.md`](../EXPERIMENT_STATUS.md).
The user-requested exact-pocket continuation of the recent Antigravity
benchmark is [below](#exact-pocket-antigravity-matched-l4-diagnostic). The
earlier PDB-grouped L4 draft is [stopped](#stopped-pdb-grouped-l4-draft) and
cannot supply comparison runs.

### Metal-ion examples within pocket-grouped folds

For standalone metal training, `--metal-example-unit ion` creates one labeled
example per parsed metal ion. Keep `--train-val-split-by pocket_id` when matching
the historical *grouping method*: all ions from a clustered pocket share its
parent pocket fold. The graph for each ion uses that ion's coordinate and its
own 10 Å residue neighborhood. With a catalytic summary CSV, each ion must
match its own summary site; missing or conflicting sites are logged as skips.
The `pocket` default preserves the existing one-label-per-cluster behavior.

This **does not preserve the historical ordered fold memberships** or the
1,597-pocket cohort: mixed pockets may now contribute individually labeled
ions, and a parent with multiple ions has multiple examples. Before any new
validation campaign, freeze its measured ion cohort, five pocket-grouped fold
identities, eligibility counts and matched ion-unit control commands in a
separate recipe/study. The earlier pocket-unit F0/F1/G0/G1 archives are
historical context only. Do not compare their scores as paired controls or use
held-out test data to choose the unit. The ion option is currently
implemented and unit-tested, not experimentally evaluated.

## PMM ion-level metal comparison campaign (`pmm_ion_metal_v2_context`)

Plan: [`docs/plans/metal_level_metal_task_compared_PMM_final_plan.md`](plans/metal_level_metal_task_compared_PMM_final_plan.md).
Stage: Stage 2B-style matched baseline comparison on grouped folds
(**one-seed grouped-fold comparison**, not Stage 6 seed-repeat confirmation).
Development commands below never discover or read the held-out side. Their
read guard denies `test/`, the mixed `site_crosswalk.csv` and the repository's
`classmodel_test_set`. The separately gated reference command appears after
the validation/refit decision gates.

The v2 profile consistently excludes target ions with explicit protein LINK
symmetry context outside the supplied coordinates. It preserves first-model,
all-local-protein-chain inputs; absent LINK records are not proof of historical
PMM assembly parity. Keep v1 outputs as historical evidence. Counts and current
certification state belong in `DATASETS.md` and `EXPERIMENT_STATUS.md`.

Frozen profile (owner: `src/benchmarking/pmm_ion_campaign.py`, `PROFILE`):
`task=metal`, `metal_example_unit=ion`, frozen source-row cohort
(`--source-cohort-csv`), five PDB-grouped folds (`split_seed=42`,
`split_stratify_by=metal_site`, frozen `fold_membership.csv`), model seed 42,
50 epochs, batch 16, AdamW, weight decay `1e-4`, fixed LR, selection on native
`val_metal_balanced_acc` (earliest-epoch tie), 10 Å pocket, 8 Å edges,
`metal_node_mode=none`, `structural_readout_scope=residue_only`,
`shell_role_source=geometry`, RING off, GVP 128/16/64/4 layers, ESM projection
128, two head layers (dropout 0.2), ESM encoder dropout 0.1, conservative
features with `biotite_residue_sasa`, `custom_charge_distance_proxy` and
`dpka_titr` omitted, ESMC-600M (`esmc_600m`, `--esm-dim 1152`).
Class weights: per training fold, `w_c = N_train / (4 n_c)` on the common four
classes via `--metal-class-weight-mode manual`; the six-class arm assigns
`w_VIII` to Fe, Co and Ni separately.

| Family | Architecture | LR | Targets | Readout variant |
|---|---|---|---|---|
| Only-ESM | `only_esm` | `3e-5` | `four_class`, `six_class` | `first_shell_bias` (four only) |
| Only-GVP | `only_gvp`, raw RBF | `3e-4` | `four_class`, `six_class` | `first_shell_bias` (four only) |
| GVP + ESMC | `gvp`, `late_fusion`, early ESM off, raw RBF | `3e-5`, GVP group `3e-4` | `four_class`, `six_class` | `first_shell_bias` (four only) |

`first_shell_bias` adds zero-initialized learned logit biases for the target's
geometric first-shell residues in each readout pooling branch (mean and
attention; GVP and, where active, late ESM). At zero it reproduces the ordinary
readout. With it, Only-ESM is labeled **ESMC with target-shell-conditioned
readout**, not sequence-only.

Deferred full grid: 3 families x 2 targets x 5 folds + 3 aware direct-four arms x 5 folds =
**45 fits**, plus the PinMyMetal released-recipe comparator refit on the same
five training partitions (CPU).

### Immediate exploratory screen: Only-ESM binding-awareness pair

The user-approved next execution reuses the completed ordinary direct-four
Only-ESM fold-0 baseline and adds only its `first_shell_bias` counterpart.
This is one-seed, single-fold exploration, not grouped-fold confirmation.
All frozen profile values above remain unchanged. Do not rerun preparation,
smokes, the ordinary baseline or PMM folds when their existing receipts verify.

Use one requested VM hour (the controller reserves three minutes, leaving at
most 57 minutes), a conservative 1,800-second complete-unit forecast, the
existing 25% admission margin and 900-second closeout reserve. The following
block runs on the existing VM after obtaining its actual session identity and
timestamps; expired allocation values must not be reused:

```bash
PY=/home/mechti/venvs/deepmzyme/bin/python
T=/home/mechti/deepmzyme_data/pmm/train_and_test_sets_structures_zenodo_pmm_exact/train
C=/home/mechti/deepmzyme_runs/pmm_ion_metal_v2_context
$PY scripts/run_metal_5fold_cv.py --campaign-dir "$C" --train-dir "$T" \
  --campaign-action run --device cuda --load-workers 4 \
  --families only_esm --targets four_class --readouts first_shell_bias --folds 0 \
  --session-id "${PMM_ALLOCATION_ID:?}" \
  --allocation-started "${PMM_ALLOCATION_STARTED:?}" \
  --execution-deadline "${PMM_HARD_STOP:?}" \
  --execution-max-seconds "${PMM_ALLOWED_SECONDS:?}" \
  --estimated-fit-seconds 1800 \
  --durable-root /media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v2_context \
  --persistence-mode host_pull
```

Complete the existing workstation host-pull/hash-acknowledgment step before
closing the allocation. Expected run directory:
`runs/only_esm__four_class__first_shell_bias__fold0__seed42/`, including
`run_config.json`, `run_metadata.json`, `epoch_metrics.csv`,
`selected_checkpoint.json`, `best_model_checkpoint.pt`, `val_predictions.csv`,
`runtime_profile.json`, and `independent_validation_replay/replay_receipt.json`.
Save the paired report as `runtime/esm_binding_screen.json` and
`runtime/esm_binding_screen.md`, with validation identities, checkpoint hashes,
BA/macro-F1/class-recall differences, confusion matrices, selected epochs,
learning curves, learned biases and empty-shell coverage. No held-out access,
full-grid `assess`, promotion or final-refit action belongs to this stage.

Acceptance requires both completed 50-epoch receipts, matching frozen validation
units and independent replay, verified host backup and controller-confirmed
`TERMINATED`. A positive or negative single-fold trend completes this screen;
it is not a superiority claim. Later confirmation of the ESMC intervention
requires both arms on folds 1–4, with fold 0's screening role disclosed.

Commands (run from the repository root; `T` is the dataset's `train/`
directory, `C` the campaign root; outputs never go under `DeepMzyme_Data/`
runs of other campaigns):

```bash
PY=/home/mechti/miniconda3/envs/DeepMzyme/bin/python
T=/media/mechti/Data1/DeepMzyme_PMM_Zenodo_Exact_Dataset/dataset/train
C=/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v2_context
# Step 1 (CPU, training-only): freeze cohort, dispositions, audit, campaign manifest
$PY src/benchmarking/pmm_ion_cohort.py --train-dir $T --out-dir $C --workers 2
$PY src/benchmarking/pmm_ion_cohort.py --train-dir $T --out-dir $C --verify-only
$PY src/benchmarking/pmm_ion_cohort.py --train-dir $T --out-dir $C --context-audit
# Step 3 folds and class weights (CPU)
$PY scripts/run_metal_5fold_cv.py --campaign-dir $C --train-dir $T --campaign-action folds
# Step 2 inputs: plan context chains (CPU), generate ESMC-600M (GPU), certify (CPU/GPU host)
$PY src/benchmarking/pmm_ion_features.py plan --campaign-dir $C --train-dir $T
$PY src/benchmarking/pmm_ion_features.py generate --campaign-dir $C --train-dir $T --device cuda
$PY src/benchmarking/pmm_ion_features.py certify --campaign-dir $C --train-dir $T
# Set these from the actual owned allocation and measured timing receipt.
# GCP can use host_pull with verified workstation acknowledgment between units.
RUNTIME=(--session-id "${PMM_ALLOCATION_ID:?}" \
  --allocation-started "${PMM_ALLOCATION_STARTED:?}" \
  --execution-deadline "${PMM_HARD_STOP:?}" \
  --execution-max-seconds "${PMM_ALLOWED_SECONDS:?}" \
  --estimated-fit-seconds "${PMM_UNIT_SECONDS:?}" \
  --durable-root "${PMM_DURABLE_ROOT:?}" \
  --persistence-mode "${PMM_PERSISTENCE_MODE:?mounted or host_pull}")
# Step 5 smoke: all nine configurations, 1 epoch, first class-complete smoke fold
$PY scripts/run_metal_5fold_cv.py --campaign-dir $C --train-dir $T \
  --campaign-action smoke --device cuda "${RUNTIME[@]}"
# Deferred full Step 6: use only under separately approved full-grid execution.
# The immediate single-fit screen is the bounded block above.
$PY scripts/run_metal_5fold_cv.py --campaign-dir $C --train-dir $T --campaign-action plan
$PY scripts/run_metal_5fold_cv.py --campaign-dir $C --train-dir $T \
  --campaign-action run --device cuda "${RUNTIME[@]}"
# PinMyMetal comparator (needs imbalanced-learn; isolated venv) and assessment
$PY scripts/run_metal_5fold_cv.py --campaign-dir $C --train-dir $T --campaign-action pmm \
  --pmm-python /media/mechti/Data1/DeepMzyme_Data/campaigns/_envs/pmm_released_py311/bin/python
$PY scripts/run_metal_5fold_cv.py --campaign-dir $C --train-dir $T --campaign-action assess
```

`--families`, `--targets`, `--readouts` and `--folds` restrict `run`/`plan` to
a subset without changing the grid. A unit is reused only when its
`selected_checkpoint.json`, `run_metadata.json` fit status, campaign identity
(cohort, fold membership, feature inventory, profile hash, family, target,
readout, fold, seed, epochs) and checkpoint/prediction hashes all match; a
partial directory is moved to `runs/_incomplete_attempts/` and the fit restarts
from scratch (no optimizer-state resume). Full development admission requires
the complete schema-v2 feature inventory. Certification validates every embedding
payload, sidecar, sequence and residue order; fit admission rechecks the frozen
structure, plan, coverage and payload/sidecar hashes without repeating that
semantic parsing. An explicit full semantic re-audit remains available through
`verify_frozen_feature_inventory(..., verify_payloads=True)`. `certify --structure-only`
is a diagnostic and an authorized Only-GVP reference option; it does not admit
the development grid. Each fitted checkpoint is independently reloaded and
replayed on its frozen validation membership before completion is certified.

The campaign sets `DEEPMZYME_GRAPH_CACHE_DIR=<campaign>/raw_graph_cache` to
reuse raw radius-only graphs across matching fits. Each entry binds complete
pocket contents, graph options, active target scheme, source and library versions;
its payload checksum is verified on read. Fold normalization is fitted afterward
on training graphs only. This cache is optional outside the campaign, and RING
graphs bypass it. A corrupt entry stops the unit with its exact path for recovery.

Use actual allocation start, not runner start, with the current four-hour
session ceiling and provider/daily limits. Forecast the next full unit with
1.25 margin and 900 seconds for closeout, including checkpoint replay and
transfer. `host_pull` exits after a terminal unit until its transfer manifest
has been independently verified on the workstation and its acknowledgment
returned. Consult the [GPU routing guide](GPU_EXECUTION_CASCADE_PLAYBOOK.md)
for ownership, measured efficiency and shutdown.

Expected outputs under `C`: `campaign_manifest.json`, `train_cohort.csv`,
`train_row_dispositions.csv`, `train_audit.json`, `train_context_audit.json`, `fold_membership.csv`,
`fold_class_weights.json`, `esm_generation_plan.{csv,json}`,
`feature_inventory.json`, `commands/<run>.json`, per fit `run_config.json`,
`run_metadata.json`, `split_diagnostics.json`, `selected_checkpoint.json`,
`runtime_profile.json` (phase wall times, process RSS and CUDA allocator peaks),
`independent_validation_replay/replay_receipt.json`,
`val_predictions.csv` (UID-keyed native and common-four probabilities of the
selected checkpoint), `pmm_comparator/`, `cv_fold_metrics.csv`,
`cv_oof_predictions.csv`, `cv_paired_deltas.csv`, `validation_decision.json`.

Decision gate (validation only): the primary estimand is mean common-four
balanced accuracy over the five selected fold checkpoints. The six predeclared
contrasts (six vs four per family; aware vs ordinary per direct-four family)
use paired fold-difference bootstrap (10,000 resamples, seed 42, 95 %
percentile intervals). Promotion requires a positive mean difference, a
positive lower 95 % bound, no missing or zero mean class recall, and no
common-four class recall drop above 0.03; a simultaneous claim also needs a
positive Bonferroni (`1 - 0.05/6`) lower bound. Otherwise retain direct four
and the ordinary readout. The PinMyMetal comparator is descriptive (different
inputs), and paper figures remain contextual. A complete development decision
requires all 45 verified fits and all five compatible PMM comparator outputs;
a neural-only grid is incomplete.

Stage 6B and Stage 7 below implement the separately authorized secondary PMM
reference route. They do not designate the project's primary final test.
The fixed fallback is ordinary direct-four Only-GVP. A challenger must pass
its applicable matched contrast and paired-CI/rare-recall gates against that
control. Rank eligible configurations by mean common-four balanced accuracy;
within 0.002, prefer mean minimum recall, worst-fold accuracy, lower fold SD,
then the declared complexity proxy and configuration ID. Select only a
configuration actually evaluated in the grid.

```bash
# Preview first: writes the frozen selection and exact full-train command.
$PY scripts/run_metal_5fold_cv.py --campaign-dir $C --train-dir $T \
  --campaign-action refit-preview --device cuda
# Once the complete validation decision passes: 50 full-train epochs, seed42,
# terminal epoch50 checkpoint; full-training normalization/common-four weights.
$PY scripts/run_metal_5fold_cv.py --campaign-dir $C --train-dir $T \
  --campaign-action refit --device cuda "${RUNTIME[@]}" \
  --pmm-python "${PMM_RELEASED_PYTHON:?}"
# Both final refits must be complete before even reference input certification.
$PY scripts/run_metal_5fold_cv.py --campaign-dir $C --train-dir $T \
  --campaign-action reference-test --device cuda "${RUNTIME[@]}" \
  --pmm-python "${PMM_RELEASED_PYTHON:?}" \
  --reference-dir "${PMM_REFERENCE_DIR:?original dataset/test}" \
  --reference-source-csv "${PMM_REFERENCE_SOURCE:?pinned classmodel_test_set}"
```

Preview outputs: `stage6b_decision.json`, `stage6b_ranked_candidates.csv`, and
`stage6b_final_refit_command.txt`. Only a completed terminal refit creates
`stage6b_selected_final_refit_candidate.json`; PMM writes
`final_refit/pmm/pmm_refit_receipt.json`. Reference outputs live exclusively
under `reference/`: route/access receipts, reconstruction dispositions,
`coverage_and_overlap.json`, both prediction files and `reference_report.json`.
The report uses one frozen model per system, no ensemble/calibration, and
10,000 paired PDB-group bootstrap resamples with seed42. It discloses exclusions
and train/reference overlap and is labeled a secondary matched known-ion
comparison. A failed reference attempt requires artifact reconciliation before
recovery; it does not authorize a new selection or a second exploratory test.

## Exact-pocket Antigravity-matched L4 diagnostic

**Pocket-unit historical protocol stopped after the user's ion-unit correction.**
Its frozen commands and verified partial outputs remain in the
[v2 execution ledger](agents_report/GVP_FUSION_EXACT_POCKET_L4_V2_EXECUTION.md).
Use the identity `gvp_fusion_exact_pocket_l4_v2`. This is a *secondary exact
PinMyMetal benchmark continuation* requested to compare focused GVP and
late-fusion changes with the 2026-09-23 Antigravity run. It does not replace
the primary `pdbid`-grouped development/Stage 6 policy in `Plan.md`, and it
does not promote an exact-split held-out test to the primary final test. The
exact cohort has 177 train/test PDB-ID overlaps; its internal pocket folds
also mix PDB IDs. Label all results accordingly.

### Locked historical protocol

The controlling evidence is the **actual**
`runs/benchmark_exact_pinmymetal_5fold/{benchmark_enhanced_gvp_esmc,benchmark_enhanced_only_gvp}_fold{0..4}/run_config.json`,
`dataset_summary.json`, and `split_diagnostics.json`, together with
`scripts/run_exact_pinmymetal_5fold_cv.py` (and the generalized multi-dataset, dual-granularity runner `scripts/run_metal_5fold_cv.py`; see [`GENERALIZED_METAL_5FOLD_CV_GUIDE.md`](GENERALIZED_METAL_5FOLD_CV_GUIDE.md)). The copied summary is explanatory,
not the source of executable defaults. Reproduce the old internal CV, not the
stopped non-overlap/PDB-grouped draft:

Here a *pocket* is one spatial cluster of metal ions and nearby residues, not
necessarily one ion. Ions connected within the default 4.5 Å merge distance
form one pocket, so a binuclear cluster stays together. A PDB structure can
still contribute several separate pockets, potentially with different metals.
The historical `pocket_id` folds assign those pockets separately, so pockets
from one PDB may appear in both training and validation. A cluster with metals
mapping to different target classes has no single metal target and is excluded
from this metal-supervised cohort: Co+Ni both map to Class VIII in the
historical five-class scheme, whereas Fe+Mn does not map to one class. This is
a single-label experiment, not mixed-metal multi-label prediction. This split
intentionally matches Antigravity for a paired comparison, but its PDB overlap
limits generalization claims.

| Field | Frozen value for the corrected study |
| --- | --- |
| Data | Verified v12 archive SHA256 `90c0899829e0ac5ca94a5ef34484b74ca1014d3e9b6b1fba90bd338ee3440dee`; **exact** PinMyMetal `train` directory and its catalytic transition-metal summary, 1,597 retained training pockets; cached ESM and updated external features |
| Target/cohort | Native `five_class`, `metal_eligibility_scheme=active`; no four-class relabeling or six-class eligibility filter during fitting. Report collapsed-four metrics from the same selected checkpoint. |
| Internal split | `--train-val-split-by pocket_id --n-folds 5 --fold-index 0..4 --seed 42`; `split_stratify_by=active_targets` by default; **omit** `--split-seed` so the recorded value remains `null` and the effective seed is 42. No PDB-ID regrouping. |
| Shared training | 50 epochs, batch 16, edge radius 8 Å, conservative node features, legacy geometry, raw-distance RBF, fixed LR schedule, weight decay `1e-4`, inverse-frequency class-weighted CE, no RING/metal nodes/augmentation |
| G0/G1 | `only_gvp`, LR `3e-4`; G1 alone enables `--gvp-normalize-message-aggregation` |
| F0/F1 | `gvp` with graph-level `late_fusion`, cached ESMC-300m/960, base LR `3e-5`, GVP LR `3e-4`; F1 alone sets `--gvp-lr-scope branch` to move `init_vec_proj`, `gvp_attn_pool`, `gvp_fusion_proj` to the fast group |
| Selection | Native `val_metal_balanced_acc` selects the checkpoint, as in the historical run; native-five and collapsed-four metrics and recalls are read at **that same epoch** |
| Test | No test path, `--run-test-eval`, overlap override, or final-test report ID in this diagnostic. The historical test was already opened; it must not guide this comparison. |

The five historical **ordered** train/validation membership hashes are hard
acceptance gates, not optional context. For every arm, the corresponding
`dataset_summary.json` hashes must match the row below exactly. The original
three model arms shared these hashes within each fold. The historical PDB-ID
train/validation overlap count is expected to be nonzero and must also match;
zero PDB overlap here would reveal the wrong split protocol.

| Fold | Train pockets | Validation pockets | Train SHA256 | Validation SHA256 | Shared train/validation PDB IDs |
| --- | ---: | ---: | --- | --- | ---: |
| 0 | 1272 | 325 | `4b7c20d5bd235e1431b3c747dcf30914eea2ee6b21e3632ea34cd8b5d2230bec` | `5e45b29050dfec6334f314928bfade081af55d8be97a85e6902d5ce35387c80f` | 58 |
| 1 | 1272 | 325 | `228bec7c597aa1bc2181ec6ab0329bd2bf892699b81465ec2dcf3c3c9ca99705` | `dd0a6baa3bdc9db713984f8a7e80fef66aa13051f9d3adb52df2083168205e61` | 56 |
| 2 | 1272 | 325 | `7057d455fa23287e268dfc4facb247280bcac5280922553bb8e7cb83793efc71` | `ea91f0ad2426726b29a6b6f77c9d33c7f21e1afeb28d35c54cb00d1a99274a4c` | 43 |
| 3 | 1272 | 325 | `767783ed70875a2e9e8f2b6224c06f4f333f8c02e493fc0ccf41ff8663704e8e` | `3b6c313894293b5914bd089f932ba56ecb833b176d7ca6f572eca1c220ff381f` | 49 |
| 4 | 1300 | 297 | `db252dd6edaae04a45032f6a900acf4634affd8f33d476907b1be26906a9725b` | `a1ad850864a88913a2ef5d5cdf9b2b3d533e0ebde820fbb66cd448ca35074ca5` | 45 |

### Isolated implementation and exact command

Work only under
`DeepMzyme_Data/notebook_outputs/plans/gvp_fusion_exact_pocket_l4_v2/`
and on the named Colab VM. Copy the tested experimental trainer changes from
the stopped study's `source_snapshot.tar.gz` into a **new isolated source
checkout**; freeze and hash that copy before use. Leave the shared checkout's
`src/`, `scripts/`, `tests/`, historical runner, and staged patch untouched.
The stopped `gvp_fusion_diagnostic_l4_v1` commands, manifest, source hash,
non-overlap data, and two one-epoch smokes are **not reusable** in v2. Keep
their ledger and archives intact as evidence of the stopped attempt.

Implement a fresh validation-only runner with `plan`, `smoke`, `run`, `status`,
and `summarize`. Generate exactly **20** full commands (four arms × five
historical folds × model seed 42) in a new immutable manifest. Fail closed if
the source/data hash, native target, `pocket_id` grouping, effective seed,
`active_targets` stratification, historical ordered membership hashes, or
expected per-fold overlap counts differ. Verify those properties before
launching any 50-epoch fit and again from every completed run. A partial run
is never skipped as complete or silently deleted. Do not reuse a persistent
study or output directory from v1.

The common trainer command is below. The runner fills the arm, fold, paths,
and learning-rate options while preserving these exact split and target
settings. Source/bundle SHA256 belong in the new manifest and run receipt;
they do not alter the historical split. The two new CLI flags exist only in
the isolated experimental source until reviewed for promotion.

```bash
/usr/local/bin/python -u /content/DeepMzyme/src/train.py \
  --task metal --metal-label-scheme five_class \
  --structure-dir /content/DeepMzyme_Data/DeepMzyme_Data/train_and_test_sets_structures_exact_pinmymetal/train \
  --summary-csv /content/DeepMzyme_Data/DeepMzyme_Data/train_and_test_sets_structures_exact_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv \
  --external-feature-source updated \
  --external-features-root-dir /content/DeepMzyme_Data/DeepMzyme_Data/updated_feature_extraction \
  --runs-dir /content/runs/gvp_fusion_exact_pocket_l4_v2 \
  --run-name gvp_exact_l4_v2_<arm>_fold<fold> \
  --model-architecture <only_gvp_or_gvp> \
  --epochs 50 --batch-size 16 --edge-radius 8 \
  --node-feature-set conservative --metal-node-mode none \
  --site-geometry-features legacy \
  --position-noise-std 0 --second-shell-dropout 0 --outer-residue-dropout 0 \
  --metal-class-weight-mode inverse_frequency --metal-loss-function cross_entropy \
  --learning-rate <3e-4_or_3e-5> --weight-decay 1e-4 \
  --rbf-use-raw-distances --lr-schedule fixed \
  --n-folds 5 --fold-index <fold> --train-val-split-by pocket_id \
  --seed 42 --selection-metric val_metal_balanced_acc \
  --gvp-lr-scope <trunk_or_branch> \
  --export-selected-val-predictions --device cuda
```

For F0/F1 append `--fusion-mode late_fusion --gvp-learning-rate 3e-4
--esm-embeddings-dir /content/DeepMzyme_Data/DeepMzyme_Data/esm_embeddings`;
for G1 append `--gvp-normalize-message-aggregation`; F1 alone uses
`--gvp-lr-scope branch`. Omit `--split-seed` and
`--split-stratify-by` to match the saved historical config exactly. Omit
all test and final-report options. Baseline F0/G0 must be rerun under the
same isolated source as their challengers; the historical runs are
reproduction references, not substitutes for matched controls.

### Execution gates and interpretation

1. CPU: confirm the shared source matches the pre-study Git state and the
   staged patch fingerprint in the [stop ledger](agents_report/GVP_FUSION_L4_EXECUTION_20260924.md).
   Test unchanged trunk defaults, exact optimizer-group movement, command
   exclusion of all test options, and refusal of wrong cohort/fold hashes.
   Freeze source, manifest, 20 commands, and an artifact inventory.
2. L4: verify assignment with `colab status`, stock PyTorch/CUDA, v12 bundle
   SHA256, complete train-side ESM/external coverage, and exact-train-only
   extraction. Run four one-epoch fold-0 smokes. Each smoke must reproduce
   the historical fold-0 membership hashes and 58 shared PDB IDs. Smoke
   scores are not selection evidence.
3. Cost gate: charge the stopped v1 allocation (approximately 0.5 hour) to
   the original 16-hour L4 ceiling; do not reset that ceiling by renaming
   the study. Use one verified L4, at most four hours per owned session with
   15 minutes reserved for transfer and teardown. Admit a complete pair only
   when 1.25 × the worst measured 50-epoch fit time × remaining fits, plus
   setup/transfer/teardown reserve, fits the remaining ceiling. Do not cut
   epochs/folds or launch half a pair as a result claim.
4. Full fits: run and verify F0/F1 on all five shared historical folds, then
   G0/G1 on those same five folds. Complete one fit at a time; copy each
   archive locally and match remote/local SHA256 before the next fit. Keep
   incomplete attempts and record their allocated time. No automatic GPU
   reprovisioning or fallback.
5. Analysis: compare F1−F0 and G1−G0 on five paired folds using the native
   selected-checkpoint five-class BA and the same-checkpoint collapsed-four
   view. Report every fold's difference, selected epoch, losses, native
   Mn/Cu/Zn/Fe/CoNi recalls and collapsed-four recalls. A 10,000-resample
   paired fold bootstrap 95% CI is descriptive with only five folds. Apply
   the predeclared ≥1.5 percentage-point mean gain, CI lower bound >0, and
   no >3-point mean recall loss in any non-Mn native class (Cu, Zn, Fe, or
   Co+Ni/Class VIII); this conservative definition of rare-class protection
   is fixed before seeing full-fit results. No winner from incomplete
   pairs. Reconcile F0/G0 baseline metrics with the historical *selected*
   checkpoints before claiming a gain relative to Antigravity; do not use
   max-over-epoch collapsed-four summary numbers as selected-checkpoint
   results.
6. Close out: verify the named session is stopped. Save `comparison.csv`,
   `paired_bootstrap.json`, `class_recall.csv`, `artifact_manifest.json`,
   `split_audit.json`, and `campaign_report.md`, plus each fit's config,
   metadata, 50 validation rows, selected checkpoint, and aligned validation
   predictions. State the exact-split PDB-overlap limitation beside every
   benchmark comparison. No held-out test, Stage 6B refit, or Stage 7
   promotion is authorized by this diagnostic.

## Stopped PDB-grouped L4 draft

`gvp_fusion_diagnostic_l4_v1` was stopped on 2026-09-24 because its
non-overlap cohort, direct four-class target, and PDB-ID-grouped folds do not
match the user's intended Antigravity comparison. Its two completed
one-epoch smokes are preparation evidence only; no 50-epoch fit ran. The
[full stopped plan](archive/experiments/gvp_fusion_l4_v1_stopped_plan.md)
and [execution ledger](agents_report/GVP_FUSION_L4_EXECUTION_20260924.md)
remain available for audit. Do not run its commands or mix its artifacts with
`gvp_fusion_exact_pocket_l4_v2`.

## Recovery of a frozen GVP capacity diagnostic

`scripts/resume_gvp_capacity_colab.py` continues an existing, reduced
`gvp_capacity_diagnostic_v1` study. It requires the original study directory,
including `manifest.json`, frozen splits/matrices, input archive and transfer
inventory, allocation ledger, and all completed runs with their retrieval
archives and verification receipts. It does not create a replacement study or
reconstruct missing evidence from summary scores. Current availability belongs
to [`EXPERIMENT_STATUS.md`](../EXPERIMENT_STATUS.md).

The preserved schedule contains fourteen discovery fits (reference, widths
64/96, depths 2/3, weight decay and dropout; seeds 42/43), then twenty
confirmation fits (reference and the selected pure-width challenger, five
shared folds and both seeds). Every fit retains 50 epochs, direct
`four_class`/`merge_fe_class_viii`, and `val_metal_balanced_acc` selection.
All other settings, ordered memberships, normalization rules and source hashes
come from the original frozen study. Optional refinement and a third
confirmation candidate remain omitted. This is bounded Stage 2B and
exploratory Stage 6; held-out evaluation, Stage 6B and Stage 7 stay disabled.

From the repository root:

```bash
CAPACITY_PY=/home/mechti/miniconda3/envs/DeepMzyme/bin/python
"$CAPACITY_PY" scripts/resume_gvp_capacity_colab.py status
"$CAPACITY_PY" scripts/resume_gvp_capacity_colab.py resume
```

Both commands accept `--root` and `--output`; the default output is
`DeepMzyme_Data/notebook_outputs/plans/gvp_capacity_diagnostic_v1/`. `status`
verifies the original archive and completed results without changing study
state. `resume` acquires the existing exclusive host lock and extracts the
verified original source separately from working-tree edits. It preserves
prior charges and results, confirms the lost session's absence, and permits
one replacement G4. A missing original absence receipt is captured from the
provider; its observation time is a conservative accounting upper bound, not
an exact termination time. An unverified or missing artifact blocks execution.

The original six-hour cumulative ceiling includes every allocation interval,
setup, failed/interrupted work, transfer and teardown. The replacement retains
a four-hour session ceiling, a fifteen-minute shutdown reserve and an independent
watchdog. Admission uses the worst measured complete-fit time multiplied by
1.25 and the original operations allowance. Completed fits are never repeated;
a lost fit can restart once from its original seed. Transport ambiguity never
causes another launch, and partial downloads are preserved while retrying only
the transfer. If the protected full comparison no longer fits, stop incomplete.

Recovery receipts are under `execution/recovery/`; allocation/hardware/launch/
terminal/stop evidence is under `execution/session_2/`. Each new fit retains
`run_config.json`, `run_metadata.json`, checkpoints, predictions and a hash-bound
`capacity_result.json`, independently verified locally before the next launch.
After all 34 results and provider teardown are verified, the frozen report
code writes `confirmation_report.json` and `comparison.csv`. The primary
width-minus-reference gain must exceed 0.002, have a positive paired-fold
bootstrap 95% lower bound, positive mean class recalls and no mean class-recall
drop above 0.03 to support an improvement. No notebook `active_run_config`
files or Optuna studies are involved.

## Validation-only sequence-remoteness attachment

This is inference/reporting for completed checkpoints and existing Stage 6 fold
definitions, not a new training stage. The
[remote addendum](REMOTE_HOMOLOGY_ADDENDUM.md) owns the protocol and support
limits. Preserve all completed fits and the campaign pause. Preparation uses
the reviewed twelve-run ledger (eight metal, four EC), frozen source snapshots,
and task-specific sequence provenance. The ledger and protocol must exist and
pass review before these commands; a fresh directory needs its own identities.

The local diagnostic recipe is:

```bash
RH_PY=/home/mechti/miniconda3/envs/DeepMzyme/bin/python
RH_ROOT=DeepMzyme_Data/notebook_outputs/remote_homology_v1
RH_PLAN=DeepMzyme_Data/notebook_outputs/plans/metal_single_gpu_33h_v2_authorized_runtime_g4
RH_MMSEQS=DeepMzyme_Data/tools/mmseqs2-18-8cc5c/mmseqs/bin/mmseqs

"$RH_PY" src/audit_sequence_remoteness.py prepare \
  --reuse-ledger "$RH_ROOT/reuse_ledger.json" \
  --fold-plan "$RH_PLAN/fold_plan.json" \
  --protocol "$RH_ROOT/protocol.json" \
  --endpoint represented_coordinate_chain --output-dir "$RH_ROOT"

"$RH_PY" scripts/run_remote_homology_search.py \
  --output-dir "$RH_ROOT" --protocol "$RH_ROOT/protocol.json" \
  --binary "$RH_MMSEQS" \
  --binary-sha256 b7ef6e0e33df5dd4fa9cf988cbd8b4988c11a3a1255d2c377f13e1bb40c157fb \
  --threads 2 --split-memory-limit 2G

"$RH_PY" src/audit_sequence_remoteness.py annotate \
  --reuse-ledger "$RH_ROOT/reuse_ledger.json" \
  --protocol "$RH_ROOT/protocol.json" --output-dir "$RH_ROOT" --counts-only
```

Before inference, inspect `counts_freeze.json`. Create the separate
`reuse_ledger_replay.json` from the original ledger with a `counts_freeze`
object containing its absolute/ledger-relative `path` and `sha256`. Do not edit
the original prepared ledger. The replay ledger binds the eligibility decision
before any prediction export. The following reuses completed exports:

```bash
"$RH_PY" src/export_validation_predictions.py \
  --reuse-ledger "$RH_ROOT/reuse_ledger_replay.json" \
  --validation-only --device cpu --output-dir "$RH_ROOT/predictions" --resume

"$RH_PY" src/report_remote_homology.py --task metal \
  --protocol "$RH_ROOT/protocol.json" \
  --prediction-manifest "$RH_ROOT/predictions/manifest.json" \
  --remoteness-manifest "$RH_ROOT/remoteness_manifest.json" \
  --output-dir "$RH_ROOT/reports/metal"
```

Expected outputs: sequence and membership manifests, per-task search receipts
and alignments, primary/audit coverage support counts, pocket/protein prediction
CSV files, per-run reproduction receipts, prediction manifest and separate JSON
and Markdown reports. Exact saved training configurations remain in the
original `run_config.json`/`run_metadata.json`; this local CLI attachment does
not create notebook `active_run_config` files or modify original runs.

Decision gate: missing provenance cannot certify full-protein remoteness;
no-hit cases remain unclassified. Missing classes or inadequate component
support prohibit a full-task interaction claim. Whole-validation reproduction
and artifact hashes must pass before using predictions. This attachment
retains `four_class` and `val_metal_balanced_acc` for metal and the saved
`pdbid` split/seed/fraction (or existing grouped folds); it creates no Optuna
study. Held-out evaluation remains disabled. Any later model promotion still
requires the unchanged Stage 6 gates, completed Stage 6B refit and resolved
Stage 7 test route.

## Single-GPU metal campaign

Profile: **`metal_single_gpu_20h_v2`**. This is a separately identified,
validation-only discovery and exploratory grouped-fold campaign. It gives
hybrid an unconditional initial tuning opportunity and reserves time for
confirmation instead of automatically opening every serious Optuna search.
Existing pilot results and 120–200-trial recipes retain their own identities;
this profile creates no Optuna study. Implementation, runtime readiness and
experimental evaluation are separate states.

### Allocation and admission policy

| Allowance | Maximum | Charged work |
|---|---:|---|
| Operations | 4 GPU-allocation hours | Setup, verification, failed/interrupted attempts, recovery, transfer, idle time, shutdown |
| Discovery | 6 GPU-allocation hours | Completed initial screens, selected seed repeats and diagnosed refinement |
| Confirmation | 10 GPU-allocation hours | Frozen shared-fold/seed comparisons; protected from further tuning |
| Total | **20 GPU-allocation hours** | Cumulative across every session; no reconnect resets |

The table is the original planned ceiling. The user explicitly authorized the
already accepted campaign to continue beyond it. The 2026-09-17 continuation
records initially allowed 33 cumulative hours, then increased the operations
allowance to account for measured artifact-transfer/readback overhead. The
current continuation ceilings are **9 operations hours, 9 discovery hours, 18
confirmation hours, and 36 cumulative hours**. The four-hour per-session limit
and 15-minute closeout reserve do not change. The worker and host controller
must bind the authorization through `budget_authorizations.json`; a larger
ceiling must not be inferred from prose alone.

These are ceilings, not spending targets. Unused discovery allowance may fund
confirmation; confirmation allowance cannot fund tuning. Count the entire
allocated interval, beginning with actual provisioning time and ending with
verified shutdown. Never count overlapping training and allocation time twice.
Charge a successful fit's complete attempt interval, including its in-process
preparation and result verification, to its scientific stage. Charge a failed,
interrupted, timed-out or verification-rejected attempt to operations once.
Operations also receives allocated time outside completed scientific attempts;
do not add a failed-attempt duration to that residual a second time. Keep an
active attempt's elapsed time separate until its outcome is known.
Use **one allocated GPU session and one training process**, with loader workers
zero. Each session lasts at most **4 hours**, with the final **15 minutes**
reserved for verified persistence and shutdown. Shorter sessions are valid.
An external owner must stop the actual VM; closing a ledger is not VM teardown.

Prefer G4 when available, but record the actual device name, memory, CUDA
capability, PyTorch support, full-fit duration and peak GPU memory. Colab Pro
does not promise a GPU type or uninterrupted session. Do not assume G4 means
16 GB, or call the runtime persistent. Copy input data to VM-local disk and
use cached ESMC/features; prepare or analyze on the local CPU when practical.
Historical minutes-per-fit observations are planning context only. Obtain
measurements for the active hardware and update them after each allocation.

Forecast complete comparison blocks with a **1.25 cost multiplier**. Protect
the forecast confirmation grid before admitting discretionary refinement.
Never reduce an individual arm's epochs, seeds or cohort to make it fit. If
costs exceed the currently authorized allowance, persist and present the user
with the measured stop-versus-increase choice unless the user already approved
the increase. Apply an approved larger ceiling without changing the scientific
plan. If no increase is authorized, stop with an explicit incomplete outcome;
spending the cap is not a success criterion.

### Frozen input and training contract

| Area | Exact campaign setting |
|---|---|
| Input | Existing non-overlap PinMyMetal **training** membership only; fixed native-six eligibility |
| Discovery split | `pdbid` grouping; original `metal_site` stratification; split seed 42; validation fraction 0.15 |
| Targets | Separate `four_class` / `merge_fe_class_viii` and `six_class` search identities; one fixed `five_class` late-fusion confirmation challenger |
| Checkpoint/recipe selection | Native `val_metal_balanced_acc` in every arm; six-class common-four results from that same checkpoint |
| Full fit | 50 epochs; batch 8; no pruning or early stopping; fixed LR schedule; weight decay `1e-4` |
| Loss | Cross-entropy, training-fitted inverse-frequency native class weights, unit manual multipliers; no auxiliary collapsed loss or site sampler |
| ESM | Frozen ESMC-300m residue embeddings, dimension 960; graph-ESM dropout 0.1 |
| Features | Conservative; certified training-only repaired PROPKA overlay; complete shared ESM/external coverage; no on-demand generation |
| Graph | Extraction 10 Å; residue edge radius 6 Å; pooling cutoff 0.0; residue-only readout; no explicit metal nodes |
| Defaults held fixed | Edge hidden width 64; head dropout 0.2; early bottleneck 32; early dropout 0.0; all augmentation off |
| Edges | Radius-only throughout discovery/base confirmation; original shell-role semantics retained until the separately matched RING block |
| Runtime | CUDA; one process; loader workers 0; deterministic; per-class epoch metrics; invalid/unsupported structures are errors |
| Test controls | `INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False`; no test paths/inference; no automatic HPO, Stage 6B or Stage 7 launch |

Only-ESM uses pocket-residue ESMC pooling. Classifier pooling does not select
the full protein, and disabling explicit metal nodes does not remove metal
coordinates from pocket construction or geometric features. Frozen source,
input membership, bundle/overlay/embedding hashes and fitted preprocessing
must be bound to results. For grouped validation, fit normalization and class
weights on each fold's training partition only.

Reuse a completed fit only after proving identical effective configuration,
scientific source behavior, input/feature contents, eligible examples, split,
seed, epoch budget, normalization and selected checkpoint. Otherwise keep it
as historical evidence. A matching label or filename is insufficient; prior
observed seed repeats do not become new confirmation evidence.

### Initial screen, top-two repeats and larger late fusion

The eight arms are Only-GVP, Only-ESM and graph-level late fusion, each with
four-/six-class training, plus direct-four early and hybrid fusion. Every arm
receives `1e-5`, `3e-5`, `1e-4` crossed with these two capacity profiles:

| Capacity | Compact | Reference |
|---|---:|---:|
| Scalar hidden width | 128 | 128 |
| GVP vector width | 8 | 16 |
| GVP layers | 2 | 4 |
| Classifier layers | 1 | 2 |
| ESM projection width, where applicable | 64 | 128 |

Graph-only fields do not apply to Only-ESM. These bundled capacity changes do
not isolate individual width/depth effects. The initial grid contains **48
full seed-42 fits before strict compatible reuse**. Rank by seed-42 native BA,
native minimum recall, parameter count, then stable configuration ID. Repeat
each arm's **top two distinct screened recipes** with seed **43**, adding
**16 fits**, before choosing by two-seed evidence. Do not gate hybrid on
early-fusion results.

Add exactly one larger graph-level late-fusion recipe: scalar/vector/edge
hidden widths **256/32/128**, four GVP layers, two classifier layers, ESM
projection 128, LR **`3e-5`**, and all other reference settings unchanged.
Run both four-/six-class targets at seeds **42 and 43**, adding **four fits**.
These remain separate from the baseline top-two repeats. Other families do
not receive an implicit large-capacity grid. Required discovery therefore has
**68 full fits before compatible reuse**, not 56.

Before admitting those full fits, run one one-epoch reference-capacity smoke
per base arm at LR `3e-5`, seed 42, charged to operations. Add two large-late
smokes, one fixed late-five smoke, and two matched GVP RING smokes: **13 smoke
templates** in total. These operational measurements price future comparison
blocks; running a smoke does not admit its full comparison. Require finite metrics,
all native classes, shared retained membership, complete features and valid
checkpoints. Smokes measure hardware/setup cost; their scores do
not rank architectures. Conservative smoke-based projections are replaced
with measured full-fit durations as they become available.

### Mixed diagnostic refinement

One mixed diagnostic block per family may follow. Collect applicable
diagnostics from both target arms and allocate at most **two extra settings
per arm**. First take the first variant from each distinct triggered category
in the table's priority order; if only one category triggers, fill the second
slot from that category's next variant. Both targets receive the same diagnostic
opportunity relative to their own selected recipe. For the first LR setting,
each boundary-selected target receives its own outward extension; an interior
target follows the triggering partner. With opposite target boundaries, the
second LR setting offers the counterpart only if no second diagnostic category
needs that slot. This preserves a slot for a visible class-recall problem.

| Priority | Diagnostic | Candidate menu |
|---:|---|---|
| 1 | Selected LR is on the initial range boundary | Below: `5e-6`; above: `3e-4` for GVP, `2e-4` for ESM/early/late, `1.5e-4` for hybrid |
| 2 | Any native class has zero recall in a selected repeated fit | `inverse_sqrt_frequency` and `effective_number` class weighting |
| 3 | Mean selected-checkpoint train-minus-validation BA exceeds 0.10 | `(weight decay, head dropout)` = `(1e-3, 0.3)` and `(1e-4, 0.4)` |
| 4 | Early/hybrid remains weak without another diagnostic | Early bottlenecks 16 and 64, each with early dropout 0.1 |

The early/hybrid weakness comparison uses its repeated mean against
direct-four graph-level late fusion. Each candidate runs seeds 42 and 43, at most
**four additional fits per arm**, at most **32** across the eight arms.
Hold other settings fixed. Process eligible
families in this order: **GVP, ESM, late, early, hybrid**. Admit complete matched
blocks only after protecting confirmation. A diagnostic result earns further
numeric exploration only through the bounded continuation gate below.

### Bounded numeric continuation

After mixed diagnostics, a family can receive at most **six further fits per
arm**, including missing seed repeats and isolation controls. A complete new
setting uses seeds 42 and 43; at most three such steps fit when no control
completion is needed. Visit families in the frozen GVP, ESM, late, early,
hybrid order, one round at a time. Maintain one active axis and direction per
family. Resolve competing trends by the frozen parameter order below and
stable configuration identity, never by the largest observed gain.

A continuation requires parent and candidate results on both seeds and the
same certified discovery cohort. Both native-BA differences must be positive,
their mean must exceed **0.002**, no common-four mean class recall may decline
by more than **0.03**, and no previously nonzero native-class recall may become
zero in either seed. If either core target passes, run the next matched setting
for **both** four-/six-class targets using their own comparable parents.
If their observed directions disagree, choose the qualifying family direction
by the fixed parameter order, lower direction first, then stable configuration
identity. Each target moves from its own current candidate in that direction;
only a matching observed direction can supply the improvement gate. Reuse
already verified next-setting seeds within the campaign and count only missing
fits against the allowance. Preserve native-six class failures in every decision.

| Parameter priority | Frozen step rule and boundary |
|---:|---|
| 1. Learning rate | First boundary extension uses the family-specific diagnostic rate above; subsequent steps multiply/divide by 2, staying in `[1e-6, 1e-3]` |
| 2. Weight decay | Adjacent value in `1e-6, 1e-5, 1e-4, 1e-3, 1e-2` |
| 3. Head dropout | Adjacent value in `0.0, 0.1, 0.2, 0.3, 0.4, 0.5` |
| 4. Early bottleneck | Adjacent dimension in `8, 16, 32, 64, 128`, holding early dropout fixed |
| 5. Late capacity | Width bundle `128/16/64 → 256/32/128 → 384/48/192 → 512/64/256`, holding GVP/head layers, ESM projection and every other setting fixed |

Categorical choices do not create continuation chains. A coupled diagnostic
such as weight decay plus dropout cannot establish a one-parameter trend;
complete a matched isolation control before extending either axis. A missing
reference seed for the mandatory larger-late comparison likewise counts
against the same six-fit allowance. Recheck the gate after controls finish;
do not schedule the next candidate alongside unverified controls. Stop a chain
at its first failed gate, boundary, budget refusal or exhausted fit cap. Do
not restart from an older favorable comparison after a newer step fails.

Freeze one recipe per arm using the two-seed mean native BA among completely
repeated recipes, then mean native minimum recall, smaller parameter count,
and stable configuration ID. Eligible paired recipes include the mandatory
larger-late pair. Do not rank native-four against native-six BA.
Batch/LR interaction, schedules/duration, graph geometry/pooling, focal loss,
smoothing, augmentation, node-level fusion and cross-attention remain later
separately costed comparisons, not an implicit expansion of this grid.

Record a full fit as **potentially training-budget-limited** if its selected
native-validation checkpoint is in epochs 46–50, or mean native validation BA
over epochs 46–50 exceeds the mean over epochs 41–45 by more than **0.002**.
Save the selected-epoch-50 flag and the late-window gain alongside that label.
These are advisory diagnostics: retain the same selected checkpoint, keep the
50-epoch limit, and do not interpret the flag as architectural ineffectiveness
or extend training after confirmation results are viewed.

### Exploratory Stage 6 confirmation

Use **Stage 6: top-K seed/split confirmation** semantics with an explicit
frozen candidate list, `group_kfold_seed_repeat`, five shared `pdbid` folds,
fold seed **42**, active model seeds **42, 43**, 50 epochs and one process.
Verify adequate native-six support in each training and validation fold before
launch. The eight base candidates require **80 full fits**. Automatic top-K
expansion is disabled.

The separately labeled **late-five** challenger uses the fixed historical
reference-capacity recipe at LR `3e-5`, native-five checkpoint selection,
and the same common-four probability-collapse reporting. It adds ten shared
fold/seed fits when admitted; it is not an expanded five-class HPO campaign.
Predeclare its paired common-four contrasts against both selected late-four
and late-six recipes. If its confirmation is deferred or incomplete, retain
it as an unresolved contender; the selected four/six recipe cannot be said
to beat all tested target formulations.

Add a direct-four Only-GVP RING comparison using its selected recipe. Both
RING-on and RING-off use `shell_role_source=geometry`; every other scientific
setting and fitted-input policy is matched. Reserve ten on fits and **ten
fresh off fits**. This v2 profile budgets and declares both controls explicitly;
it does not silently substitute a historical off arm. Existing pilot RING results
do not replace this shared-fold comparison. Run node/shell/edge-control audits
before the corresponding RING fits and charge their allocated time to
operations. Combined models remain RING-off.

Before refinement, forecast the full confirmation grid with the 1.25 margin.
Reduce discretionary refinement first. If full coverage still cannot fit,
freeze this cost-only fallback order **before any fold-result inspection**:

1. All six core target/formulation arms (60 fits).
2. Fixed late-five challenger (10 fits).
3. Early and hybrid together (20 fits).
4. The matched GVP RING block (20 fresh off/on fits).

Never choose which blocks complete using favorable fold results. Incomplete
blocks receive explicit missing-coverage labels; no opportunistic seed/fold
subset becomes a completed comparison.

Average model seeds within each fold and use paired **10,000-resample**
fold-level bootstrap intervals. Compare target formulations on common-four
probabilities, summing Fe/Co/Ni before argmax, while retaining native-six BA and
Fe/Co/Ni recalls. An improvement claim requires a positive paired interval and
no common-four mean class-recall loss exceeding **0.03** against its declared
comparator. Use a **0.002 BA** practical tie band, then recall, stability and
model simplicity. The comparisons are six-versus-four within each core family,
modality comparisons among direct-four core models, early/hybrid versus
graph-level late fusion, and matched RING-on versus off.

These intervals describe stability conditional on the preceding search. They
do not remove selection bias or substitute for a final held-out evaluation.
Do not tune after viewing confirmation. The unresolved final-test route makes
this exploratory validation: no Stage 6B refit or Stage 7 artifacts are emitted.
A final reporting cycle must first resolve/freeze its data route, then pass
Stage 6, Stage 6B and Stage 7 in order. EC confirmation and EC-primary auxiliary
learning follow in separately costed, certified campaigns.

### Count and measured-budget audit

| Work | Full-fit count before reuse |
|---|---:|
| Eight-arm baseline LR/capacity screen | 48 |
| Top-two seed-43 repeats | 16 |
| Fixed large-late four/six × two seeds | 4 |
| Mixed diagnostics | 0–32 |
| Protected numeric chains, including needed controls | 0–48 |
| Eight base candidates plus fresh RING controls | 100 |
| Fixed late-five confirmation, if admitted | 10 |

The required discovery floor is 68 fits. Full required confirmation without
late-five gives 168 fits before optional tuning; adding late-five gives 178.
The absolute full-fit ceiling is **258** with all diagnostic and chain
allowances used and all confirmation blocks admitted. The 13 initial smokes,
new-hardware re-profiling, audits, failed attempts and transfers are additional
operational work. These counts describe coverage and caps; they are not an
authorization to spend beyond the currently recorded allocation ceiling.

`preview` writes a historical timing estimate before any allocation.
It credits **zero certified reuse** by default. Up to 14 old reference-screen
cells and seven old seed repeats are theoretical overlaps only: selected
recipes may differ, source/configuration/content identity must pass, and old
fixed-split observations do not replace new grouped-fold fits. Do not count
unverified reuse as a budget saving.

For perspective, using historical reference-capacity G4 fit means, a
one-epoch hybrid extrapolation, a 25% training margin, and four operational
hours gives roughly **21.54 hours** for 68 discovery plus 110 confirmation
fits, with no optional refinement. That calculation optimistically prices
large late like reference late and does not include the live forecast's
conservative discovery-to-fold adjustment. It is a lower planning proxy,
not a measured completion prediction. Larger late recipes may also win
selection, increasing subsequent confirmation costs. Earlier closed
allocations spent about **2.66 of 7.31 hours** outside completed full fits;
four future operational hours are not guaranteed sufficient.
The timing basis is the archived
[architecture attempt ledger](notebook_outputs/raw/metal_architecture_pilot_20260915/continuation/finalization/campaign_attempt_ledger.json),
[RING attempt ledger](notebook_outputs/raw/metal_ring_pilot_20260915/finalization/campaign_attempt_ledger.json)
and [closed allocation ledger](notebook_outputs/raw/metal_ring_pilot_20260915/host_closeout_allocation3/allocation_ledger.json),
with the [geometry attempt ledger](notebook_outputs/raw/metal_coordination_geometry_pilot_20260915/finalization/campaign_attempt_ledger.json)
included in the operational-overhead calculation.

Before full training, use current-hardware measurements and an explicit
remaining-operations plan covering setup, smoke work, audits, persistent
copies/readbacks, recovery and each shutdown. Apply the 1.25 margin and show
all three phase caps plus the total. A larger model requires its own timing
measurement; a smaller model's runtime cannot price it. Completed exact
full-fit timings replace smoke projections. A changed GPU invalidates old
runtime estimates even when scientific results remain reusable.

The revised floor exceeded the original six-hour discovery ceiling. If the
measured required discovery plus protected confirmation does not fit, the
runner must refuse full-fit admission and report the deficit. Preserve the
prepared plan and evidence; do not silently change the allocation split,
drop a required initial arm, shorten training, or borrow confirmation time.
Use the accepted-work escalation policy to record an authorized increase;
the failed original forecast does not revoke a later user authorization.
Optional blocks consume only measured remaining capacity and follow the
frozen fallback order before confirmation results are opened.

### Serial campaign commands

The worker runner is `src/run_metal_single_gpu_campaign.py`; the separate
ownership/watchdog controller is `scripts/colab_serial_metal_host.py`.
Planning, preparation, queue transitions and reports never allocate a GPU.
Only the host's explicit `allocate` action provisions one. Keep the ordinary
notebook's training, HPO, Stage 6B and final-test launch switches false.

Set `CAMPAIGN_PYTHON` to the verified interpreter, `CAMPAIGN_DATA_ROOT` to the
verified inputs, and the overlay variables to the certified repaired features.
Initially `CAMPAIGN_OUTPUT_DIR` is a fresh local planning directory. The
repository's prescribed local interpreter is
`/home/mechti/miniconda3/envs/DeepMzyme/bin/python`.

```bash
"${CAMPAIGN_PYTHON:?verified interpreter required}" -c 'import sys; print(sys.executable)'
"${CAMPAIGN_PYTHON:?}" src/run_metal_single_gpu_campaign.py plan \
  --repo-root "$PWD" \
  --data-root "${CAMPAIGN_DATA_ROOT:?verified data root required}" \
  --output-dir "${CAMPAIGN_OUTPUT_DIR:?fresh local planning directory required}" \
  --external-features-root-dir "${CAMPAIGN_EXTERNAL_FEATURES_ROOT_DIR:?certified overlay required}" \
  --feature-overlay-manifest "${CAMPAIGN_FEATURE_OVERLAY_MANIFEST:?verified overlay manifest required}"
"${CAMPAIGN_PYTHON:?}" src/run_metal_single_gpu_campaign.py prepare \
  --output-dir "${CAMPAIGN_OUTPUT_DIR:?}"
```

Inspect `commands.txt`, `run_matrix.csv`, `fold_plan.json`, feature identities
and `budget_forecast.md`. The historical preview is explicitly not admission.
Export an untouched initial plan to the exact future worker paths **before**
allocation. This path-only export preserves scientific recipes and hashes; it
cannot relocate a started campaign. Choose a compatible worker interpreter
path using the Colab runbook and recreate these paths on subsequent sessions.

```bash
"${CAMPAIGN_PYTHON:?}" src/run_metal_single_gpu_campaign.py export-runtime-plan \
  --output-dir "${CAMPAIGN_OUTPUT_DIR:?}" \
  --destination-dir "${CAMPAIGN_RUNTIME_EXPORT:?local export directory required}" \
  --worker-repo-root /content/DeepMzyme \
  --worker-data-root /content/DeepMzyme/DeepMzyme_Data \
  --worker-output-dir "${CAMPAIGN_WORKER_OUTPUT:?persistent or transferred worker path required}" \
  --worker-external-features-root "${CAMPAIGN_WORKER_OVERLAY:?worker overlay path required}" \
  --worker-python "${CAMPAIGN_WORKER_PYTHON:?compatible worker interpreter path required}"
"${CAMPAIGN_PYTHON:?}" scripts/colab_serial_metal_host.py prepare \
  --output "${CAMPAIGN_RUNTIME_EXPORT:?}" \
  --session-id "${CAMPAIGN_SESSION_ID:?fresh deepmzyme-prefixed name required}" \
  --gpu G4
```

The final command prepares ownership/watchdog configuration only. An eventual
explicit `allocate` with the same host output and session name arms the
watchdog before requesting the GPU. Exact available names are T4, L4, G4,
H100 and A100; choose one explicitly and verify the assigned hardware.
Copy the exported plan, frozen source and verified data to their declared
worker locations. Run worker `prepare` again; allocated setup is operations.

On the surviving host, obtain a fresh launch receipt with:

```bash
"${CAMPAIGN_PYTHON:?}" scripts/colab_serial_metal_host.py status \
  --output "${CAMPAIGN_RUNTIME_EXPORT:?}" --session-id "${CAMPAIGN_SESSION_ID:?}" \
  --worker-receipt "${CAMPAIGN_HOST_RECEIPT:?local receipt file required}"
```

Transfer that receipt promptly to the worker; it expires within 120 seconds.
For the following worker commands, set `CAMPAIGN_PYTHON` and
`CAMPAIGN_OUTPUT_DIR` to the frozen **worker** interpreter and output path.
Use the exact allocation request timestamp from the host receipt, including
all setup time. Register the session and verify hardware/inputs:

```bash
"${CAMPAIGN_PYTHON:?}" src/run_metal_single_gpu_campaign.py session-open \
  --output-dir "${CAMPAIGN_OUTPUT_DIR:?}" \
  --session-id "${CAMPAIGN_SESSION_ID:?owned session required}" \
  --allocation-started-epoch "${CAMPAIGN_ALLOCATION_STARTED_EPOCH:?actual allocation start required}" \
  --host-receipt-json "${CAMPAIGN_WORKER_HOST_RECEIPT:?fresh transferred receipt required}"
"${CAMPAIGN_PYTHON:?}" src/run_metal_single_gpu_campaign.py readiness \
  --output-dir "${CAMPAIGN_OUTPUT_DIR:?}" \
  --operations-plan-json "${CAMPAIGN_OPERATIONS_PLAN:?future operations forecast required}"
```

The operations JSON records future raw `setup_seconds`, `smoke_seconds`,
`audit_seconds`, `persistence_seconds`, `recovery_seconds`, `shutdown_seconds`,
`sessions_remaining` and a nonempty evidence `basis`. Shutdown must reserve
at least 900 seconds per remaining session. The runner adds the 25% margin.
Refresh this future-cost estimate through `readiness` when circumstances
change; a stale overestimate may safely halt admission.

First execute and persist the one-epoch operations probes, one per call.
They price all prospective confirmation blocks and are not selection evidence.
Each probe has a bounded bootstrap forecast of 600 seconds; it does not give
a full-fit timing claim. Before each execution refresh the host receipt.

```bash
"${CAMPAIGN_PYTHON:?}" src/run_metal_single_gpu_campaign.py execute \
  --output-dir "${CAMPAIGN_OUTPUT_DIR:?}" \
  --host-receipt-json "${CAMPAIGN_WORKER_HOST_RECEIPT:?fresh transferred receipt required}"
"${CAMPAIGN_PYTHON:?}" src/run_metal_single_gpu_campaign.py report \
  --output-dir "${CAMPAIGN_OUTPUT_DIR:?}"
```

After profiling, `forecast` prints measured admission and missing coverage.
Import strictly compatible cells with `reuse --run-id ... --source-campaign ...
--source-run-id ...` **before** reserving or attempting them. Then call `admit`
to reserve complete scientific blocks across sessions. `advance` freezes the
top-two shortlist, diagnostics, chains or final confirmation queue as their
gates pass; it never trains. After each new scientific block, call `admit`
before `execute`. Neither command silently changes folds, seeds or epochs.

Verify every terminal attempt, including failed logs, before another fit.
An actual mounted Drive output is read back automatically. Otherwise transfer
the attempt directory, independently read it back, and submit a receipt with
`method="verified_transfer"`, `destination_uri`, `readback_root` and the
exact `artifacts` inventory from `attempts.json`:

```bash
"${CAMPAIGN_PYTHON:?}" src/run_metal_single_gpu_campaign.py verify-transfer \
  --output-dir "${CAMPAIGN_OUTPUT_DIR:?}" \
  --attempt-id "${CAMPAIGN_ATTEMPT_ID:?terminal attempt required}" \
  --receipt-json "${CAMPAIGN_TRANSFER_RECEIPT:?verified readback receipt required}"
```

For non-Drive output, also call `export-state`, transfer the returned immutable
snapshot, and call `verify-state-transfer --receipt-json ...`. That receipt
contains `state_sha256`, `destination_uri` and independent `readback_root`.
The current queue, allocation, comparison and attempt ledgers must match it
before execution; queue changes require a new snapshot. A new VM needs the
verified complete state and artifacts restored at the same paths.
For the transfer route, the first execution request also writes an immutable
launch intent before starting a process. Persist that new state, refresh the
host receipt, and repeat execution. This preserves the attempt identity even
if the VM disappears before its final ledger is copied. Reconciliation of a
lost attempt requires verified death of its original allocation, records any
artifact loss explicitly, and consumes the original attempt's retry allowance.

Before a RING fold, use `ring-audit --fold-index 0..4 --host-receipt-json ...`.
It registers and runs one bounded CPU audit charged to operations; for a
transfer route, persist its newly registered queue state and repeat the call.
Persist audit results before the matched fits. Interrupted fits restart once
in a separate directory; `reconcile` requires proof the old process or VM died.

After verified persistence, use the host's explicit `stop` action with its
owned session name. The watchdog is the backstop, not permission to leave a
VM idle. Close the worker ledger from the surviving controller only after
provider absence has been verified:

```bash
"${CAMPAIGN_PYTHON:?}" src/run_metal_single_gpu_campaign.py session-close \
  --output-dir "${CAMPAIGN_OUTPUT_DIR:?}" \
  --session-id "${CAMPAIGN_SESSION_ID:?}" \
  --stopped-epoch "${CAMPAIGN_STOPPED_EPOCH:?actual verified stop required}" \
  --stop-evidence-json "${CAMPAIGN_STOP_EVIDENCE:?actual stop receipt required}"
```

Retain the host and worker ledgers together. A new allocation requires a fresh
owned name, a verified previous stop and hardware readiness. The cumulative
allocation history and currently authorized ceiling persist. Do not use a blind unbounded shell loop: follow each
reported admission, persistence, recovery and closeout result.

For a user-authorized budget increase, first finish/persist any safe active fit
and verify the owned allocation stopped. Save a JSON authorization object with
`sequence`, `status="authorized"`, `authorized_by="user"`, `authorized_at`,
`decision="continue_beyond_planned_ceiling"`, the user's authorization basis
in `reason`, and `held_out_evaluation=false`. Include
`previous_limits_seconds` and `authorized_limits_seconds`, each containing
exactly `total_seconds`, `discovery_seconds`, `confirmation_seconds`,
`operations_seconds`, `session_seconds`, and `closeout_seconds`. The previous
limits must equal the latest ledger entry (or the original table for sequence
1). The current continuation values in seconds are respectively **129600,
32400, 64800, 32400, 14400, 900**.

```bash
"${CAMPAIGN_PYTHON:?}" src/run_metal_single_gpu_campaign.py authorize-budget \
  --output-dir "${CAMPAIGN_OUTPUT_DIR:?}" \
  --authorization-json "${CAMPAIGN_BUDGET_AUTHORIZATION:?recorded user decision required}"
```

The command validates and appends the object to `budget_authorizations.json`.
It cannot lower a previously authorized cumulative ceiling or change the
session/closeout limits. Export and independently verify the updated state,
then prepare a fresh host session so its configuration and worker receipts
bind the same authorization digest. Do not edit an active host configuration,
reset spent time, or modify a frozen scientific comparison to apply an increase.

### Expected artifacts

All campaign-level artifacts live below the explicit `--output-dir`:

- `campaign_manifest.json`, `commands.txt`, `run_matrix.csv`: frozen source,
  configuration and initial queue identity.
- `notebook_planning/`: notebook adapter's planning-only metadata, kept separate
  from actual fit artifacts in `runs/`. These helper files do not replace the
  campaign manifest or count as attempted training.
- `budget_forecast.json`, `budget_forecast.md`: historical preview or current
  measured admission forecast, including unmeasured costs and phase deficits.
- `budget_authorizations.json`, when present: ordered user-authorized ceiling
  increases, their reason and unchanged session/closeout safeguards. Host
  configurations and launch receipts bind its digest.
- `queue.json`, `input_identity.json`, `preparation.json`, `fold_plan.json`,
  `training_cache_audit.json`:
  current queue plus certified shared inputs and declared folds.
- `sessions.json`, `attempts.json`, `readiness/<session>.json`,
  `persistence/<attempt>.json`: cumulative allocation, linked attempts,
  hardware readiness and verified artifact persistence.
- `comparisons.json`: reservations for complete comparisons across sessions.
  `state_snapshots/` and `state_persistence.json` record controller-state transfer
  and independent read-back when the output is not directly persistent.
- `launch_intents.json` and `active_process.json`, when present, preserve
  prelaunch identities and process ownership for interruption reconciliation.
  State snapshots also include terminal-attempt persistence receipts.
- `runtime_export.json` in an exported plan records path/interpreter remapping;
  it does not certify worker inputs. The separate host output's `host_control/`
  directory contains allocation ownership, watchdog and teardown records.
- `refinement_decisions.json`, `discovery_closed.json`,
  `chain_decisions.json`, `confirmation_manifest.json`: diagnostic/continuation choices, discovery freeze and
  predeclared candidate/fold/seed admission.
- `campaign_report.json`, `campaign_report.md`, `campaign_results.csv`,
  `confirmation_summary.csv`, `confirmation_pairwise.csv`: completed-result
  evidence, uncertainty/recall gates and explicit missing coverage.

Each completed fit also keeps `run_config.json`, `run_metadata.json`,
`split_diagnostics.json`, `dataset_summary.json`, `prepare_status.json`,
`epoch_metrics.csv`, `train_metrics.csv`, `val_metrics.csv`, fitted
normalization and `best_model_checkpoint.pt` in its recorded attempt directory.
`performance_profile.json` records setup/training time and peak CUDA memory,
including profiling status when training fails.
The standalone manifest owns this recipe; notebook-generated
`active_run_config.json` / `active_run_config.md` describe ordinary notebook
plans and must not override it. No `test_report.json` or final-refit selection
artifact is expected or permitted.

### Recovery and decision gate

Preserve completed fits and restart an interrupted fit once from its original
seed, linking both attempts. The trainer has no exact epoch/state resume.
Retries remain charged; a second interruption is an explicit unresolved unit.
Every completed fit must retain its checkpoint, effective config, metadata,
normalization and metrics, with verified persistent copies before proceeding.

The campaign is complete only when its frozen admitted comparison grid is
complete, the artifacts and transfers verify, native/common-four selection
semantics match the recipe, paired class-recall diagnostics are present, and
the allocation closes within budget. Report the full required matrix's missing
blocks even when the admitted subset finishes. A budget stop, negative or
inconclusive comparison is valid evidence; it is not model promotion. Keep
strict study/recipe identity and test exclusion throughout.

## Retained Common70 standalone validation campaign — Stage 0 through Stage 2B


For the direct arm, selection is `val_metal_balanced_acc`. For the required
six-class arm, selection is explicitly `val_metal_collapsed4_balanced_acc`
so both arms optimize the common endpoint. This is a scoped exception to the
retained six-class native-selection recipes below. The implementation sums
Fe/Co/Ni probabilities before argmax; do not just collapse the six-way argmax.
Retain native six-class BA, macro-F1 and Fe/Co/Ni recalls from the **same
selected checkpoint**. Never compare native-six BA numerically against
direct-four BA. Six variants × four runs = 24 baseline runs; six separate
one-epoch smoke runs precede them. No Optuna study is created.

This is the earlier standalone-only preparation route. Its six-class
collapsed-four-selection exception is historical and is not the pilot's
native-selection policy. It supersedes the older baseline examples below only
for this retained campaign; later HPO and final-reporting recipes
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
TASK = "metal"
DATASET_NAME = "train_and_test_sets_structures_common_pdbid_70_30_pinmymetal"
METAL_LABEL_SCHEME = "four_class"  # Repeat separately with "six_class"; BOTH are required.
assert METAL_LABEL_SCHEME in {"four_class", "six_class"}
SPLIT_STRATIFY_BY = "metal_site"
METAL_ELIGIBILITY_SCHEME = "six_class"
SELECTION_METRIC = ("val_metal_balanced_acc" if METAL_LABEL_SCHEME == "four_class"
                    else "val_metal_collapsed4_balanced_acc")
OPTUNA_SELECTION_METRIC = SELECTION_METRIC
METAL_REPORT_VIEW = "both"
SINGLE_AND_MANUAL_CONFIG_EPOCHS = 1 if STANDALONE_PHASE == "smoke" else 50
EPOCHS = SINGLE_AND_MANUAL_CONFIG_EPOCHS
BATCH_SIZES_CSV = "8"
_family = {"Only-GVP": "only_gvp", "Only-ESM": "only_esm", "GVP + late fusion": "late_fusion"}[MODEL_PRESET]
RUN_BATCH_ID = f"standalone_metal_common70_{METAL_LABEL_SCHEME}_{_family}_{STANDALONE_PHASE}_v1"
RUN_NAME_PREFIX = RUN_BATCH_ID

```

### Readiness, outputs, and explicit launch gate

For the current-code snapshot upload and exact browser cell order, follow the
[Chat 4 Colab handoff](COLAB_GPU_RUNBOOK.md#chat-4-standalone-smoke-with-a-working-code-snapshot).
Its optional notebook cell applies the standalone block above and preserves
the data-bundle selection from Main configuration.

`METAL_ELIGIBILITY_SCHEME="six_class"` fixes the shared eligible cohort:
both arms require a uniquely defined native-six target. Mixed Fe/Co/Ni pockets
that become single-label only after four-class merging are excluded from both
arms and recorded as missing required supervision. This explicit intersection
policy isolates training formulation; it is not a claim about a larger
four-class-only cohort. The split-symbol control alone cannot enforce this.

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

## Bounded metal architecture pilot — Stage 0 through Stage 2B

Profile: **`metal_architecture_pilot_10h_v1`**. This is a fresh, bounded
validation-only metal screen using existing architectures. It provides an
early-fusion comparison immediately while protecting required target-formulation
coverage before optional hybrid work. It is neither a serious Optuna stage nor
Stage 6 promotion. Read `EXPERIMENT_STATUS.md` for actual execution state.

### Exact fixed recipe

The runner materializes these values in every planned command and records its
source identity. Other notebook grid/HPO settings do not override this profile.

| Area | Pilot value |
|---|---|
| Input | Non-overlap PinMyMetal **train** structures and train site summary only |
| Task and selection | `metal`; native `val_metal_balanced_acc` in every target arm |
| Targets | `four_class`, `five_class`, `six_class`; separate variant/run identities |
| Eligibility and split | Native-six eligibility; `pdbid`; `metal_site` stratification; split seed 42; validation fraction 0.15 |
| Training | 50 epochs; batch 8; LRs `3e-5,1e-4`; fixed schedule; weight decay `1e-4`; model seed 42, then selected-LR seed 43 |
| Capacity | Hidden scalar/vector 128/16; edge hidden 64; 4 GVP layers; 2 head layers; head dropout 0.2 |
| ESM | Frozen ESMC-300m residue embeddings, dimension 960; fusion dimension 128; graph-ESM dropout 0.1; early bottleneck 32; early dropout 0.0 |
| Features | Conservative; complete ESM/external coverage over the shared cohort; certified PROPKA feature overlay; no automatic feature generation during training |
| Loss | Cross-entropy; training-only inverse-frequency native class weights; unit manual multipliers; no site sampler, label smoothing, or collapsed auxiliary loss |
| Geometry | Extraction 10 Å by any-atom-to-site-metal distance; edge radius 6 Å by closest residue atom distances; pooling cutoff 0.0; residue-only readout; no explicit metal nodes |
| Edges and augmentation | Radius-only, no RING; position noise, second-shell dropout, outer-residue dropout all 0 |
| Runtime | One CUDA worker; loader workers 0; deterministic; class-level metrics; invalid/unsupported structures are errors |
| Test and HPO | Held-out inputs/evaluation disabled; no Optuna study, Stage 6, Stage 6B, or Stage 7 launch |

Pooling cutoff zero includes **all residue nodes already extracted for the
pocket**, using mean plus learned attention pooling. It does not make the edge
radius the pocket radius, and it does not select the whole protein. Positive
pooling cutoffs use Cα-to-metal distances after message passing. Disabling
explicit metal nodes still leaves metal coordinates in pocket extraction and
geometric features. Record extracted and pooled residue counts; they must match
for this zero-cutoff residue-only profile. Only-ESM pools pocket-residue ESM
features, not a separately established full-protein representation.

Feature readiness must inspect availability/provenance masks as well as file
existence: a present cache with unavailable PROPKA/pKa values is not complete
measured feature coverage. This pilot requires a certified training-only
external-feature overlay with `tooling.pka="propka"` and a hashed provenance
manifest. Generate and verify it on CPU before allocating the GPU. Freeze and
hash the overlay separately from the original bundle and use the same repaired
inputs for every arm; never relabel modified caches as the unchanged published
bundle. This requirement does not assert that every non-titratable residue has
a measured pKa. Record missingness and tool failures explicitly.

### Exact schedule and budget

The overall allocation cap is **600 minutes**, including setup, interruption
losses, verification, and shutdown. Planning allocations are profile/setup 60,
main training 450, linked retry recovery 60, and closeout 30 minutes. Stop
training no later than minute 570. Preserve the original allocation-start time
across repeated calls and reconnects; a new call cannot reset the ledger.
These original-runner commands use the timestamp of their live allocation.
A later GPU allocation requires the verified recovery workflow described in
the geometry execution prerequisites below, retaining prior closed intervals.

Run seven one-epoch smoke configurations first: Only-GVP four/five/six; then
Only-ESM, early, late, and hybrid with direct-four targets. Require finite
training/validation outputs, every native class in both splits, matching retained
example/group identities, complete required features, and no test report.
Measure setup plus training/validation/save costs and peak GPU memory. Smoke
scores are not architecture evidence.

| Order | Block | Variants | LR | Model seed | Full runs |
|---:|---|---|---|---:|---:|
| 1 | A1 | Direct-four Only-GVP, Only-ESM, early, late | `3e-5` | 42 | 4 |
| 2 | A2 | Same four variants | `1e-4` | 42 | 4 |
| 3 | T1 | Five/six targets × Only-GVP, Only-ESM, late | `3e-5` | 42 | 6 |
| 4 | T2 | Same six variants | `1e-4` | 42 | 6 |
| 5 | H, conditional | Direct-four hybrid | Both LRs | 42 | 2 |
| 6 | R | Four/five/six × three core families, each chosen LR | Chosen per variant | 43 | 9 |
| 7 | RE | Direct-four early, chosen LR | Chosen | 43 | 1 |
| 8 | RH, conditional | Direct-four hybrid, chosen LR | Chosen | 43 | 1 |

The authorized coordination-geometry extension runs as a **separate profile
after A1/A2 and before T1/T2**; see the [geometry recipe](#bounded-coordination-geometry-pilot--stage-2b).
It shares this campaign's cumulative allocation cap. The table above preserves
the original profile's block identities and internal order; the orchestration
pause does not change completed runs or rewrite their frozen manifest.

H is prioritized only if early's best native balanced accuracy is at least
0.01 above Only-GVP's best and its minimum native recall is no more than 0.03
below Only-GVP's. Choose each model's LR by native balanced accuracy, then
native minimum recall; break a remaining tie deterministically by the lower
LR. Apply the gate after A1/A2, but complete T1/T2 before spending optional
hybrid time. If the gate fails, label hybrid **deferred under this budget**.
This single-seed gate is a scheduling heuristic, not statistical evidence for
promotion or rejection; hybrid's extra ESM pathway could still help.

Admit a new block only when its full remaining cost forecast, multiplied by
1.25, fits the remaining allowance. Update estimates from actual completed
runs and changed hardware. Never shorten selected models' epochs to make a
block appear complete. Preserve unfinished blocks and exclude them from
paired comparison summaries. The maximum planned work is 33 full runs plus
seven smokes when hybrid is included and all budget gates pass; completion
within the allocation is not guaranteed.

### Exact notebook preview block

After normal repository and data setup, use the optional **Bounded metal
architecture pilot** notebook cell. Its controls are:

```python
METAL_PILOT_PROFILE = "metal_architecture_pilot_10h_v1"
METAL_PILOT_OUTPUT_DIR = "/content/drive/MyDrive/DeepMzyme/notebook_outputs/campaigns/metal_architecture_pilot_10h_v1"
METAL_PILOT_EXTERNAL_FEATURES_ROOT_DIR = "/content/metal_architecture_pilot_features/external_overlay"
METAL_PILOT_FEATURE_OVERLAY_MANIFEST = "/content/metal_architecture_pilot_features/external_overlay/feature_overlay_manifest.json"
```

These paths are explicit examples: use the actual mounted Drive root and the
verified, extracted overlay/manifest locations. All three paths are required
by the opt-in cell. It calls
`plan(Path(REPO_DIR), Path(DATA_ROOT), output, "notebook_working_snapshot", external_features_root_dir=overlay_root, feature_overlay_manifest=overlay_manifest)`
and previews the queue. It does not launch training. Keep ordinary notebook
main/Optuna/Stage 6/6B/7 launch switches off. Do not replan a started campaign
under changed source/data; preserve its original identity.

### Execution and persistence contract

Prepare the feature overlay and source snapshot on CPU before allocating the
GPU. Run from the repository root, setting `PILOT_CPU_PYTHON` to the verified
DeepMzyme development interpreter. The task-specific path below keeps generated
files outside tracked source and preserves the original caches:

```bash
PILOT_PREP_DIR="DeepMzyme_Data/notebook_outputs/plans/metal_architecture_pilot_10h_v1"
"${PILOT_CPU_PYTHON:?verified DeepMzyme interpreter required}" scripts/repair_metal_pka_cache.py \
  --data-root DeepMzyme_Data \
  --output-root "${PILOT_PREP_DIR}/external_overlay" \
  --jobs 4
```

Require `external_overlay/feature_overlay_manifest.json` to report
`status="complete"`, no failures, full training-structure coverage, and valid
per-file hashes. `--limit` is a preparation probe only and does not certify a
full campaign. Existing compatible repaired files are reusable. For an overlay
generated before the compact PROPKA residue-token parser fix, refresh affected
wide residue numbers using:

```bash
"${PILOT_CPU_PYTHON:?}" scripts/repair_metal_pka_cache.py \
  --data-root DeepMzyme_Data \
  --output-root "${PILOT_PREP_DIR:?}/external_overlay" \
  --jobs 4 \
  --refresh-wide-residue-numbers
```

The refresh targets structures with standard-residue numbers at least 1000 or
at most -100, and structures with insertion codes. For insertions, temporary
unique numbering and an explicit map preserve each original residue identity;
unmeasurable sidechains retain their missing-value flags. The refresh reuses
other compatible overlay files. Recheck the final manifest and transfer the
overlay with that manifest before planning GPU runs.

After code/documentation checks pass, package the working tree and the exact
non-overlap training-membership receipt:

```bash
"${PILOT_CPU_PYTHON:?}" scripts/prepare_colab_smoke_snapshot.py \
  --dataset train_and_test_sets_structures_non_overlapped_pinmymetal \
  --output-dir "${PILOT_PREP_DIR:?}/colab" \
  --archive-name deepmzyme-metal-pilot-code.tar.gz
```

This produces `deepmzyme-metal-pilot-code.tar.gz`, its `.sha256` sidecar, and
`code_snapshot_manifest.json`; the archive includes
`colab_expected_metal_inputs.json`. It captures working files, including
uncommitted source, and does not create a clean Git commit. It excludes data
and the feature overlay, which must be transferred and verified separately.
Freeze this source snapshot before creating the campaign manifest. The
snapshot script's historical defaults remain unchanged; the explicit dataset
and archive arguments above select this pilot.

Use `src/run_metal_architecture_pilot.py` through its importable `plan`,
`preflight`, `execute`, and `summarize` entry points. `plan(root, data, output,
source_commit)` takes `Path` arguments and writes the preview;
`preflight(root, output, allocation_started_epoch, persistence_receipt=None)`
checks the actual CUDA runtime and complete shared cohort.
`execute(root, output, allocation_started_epoch, persistence_receipt=None,
max_runs=1)` advances a bounded step in the persistent queue. `summarize(output)`
refreshes the coverage/decision reports. Use the same paths, frozen source,
original allocation timestamp, and verified persistence receipt on every step.

For terminal execution from the repository root, set `PILOT_PYTHON` to the
verified runtime interpreter, `PILOT_OUTPUT_DIR` to the planned output, and
`PILOT_ALLOCATION_STARTED_EPOCH` to the recorded UTC epoch of GPU allocation
(before setup, not the current time). The mounted-Drive path is:

```bash
"${PILOT_PYTHON:?verified interpreter required}" src/run_metal_architecture_pilot.py preflight \
  --output-dir "${PILOT_OUTPUT_DIR:?planned output required}" \
  --allocation-started-epoch "${PILOT_ALLOCATION_STARTED_EPOCH:?original allocation time required}"
"${PILOT_PYTHON:?}" src/run_metal_architecture_pilot.py execute \
  --output-dir "${PILOT_OUTPUT_DIR:?}" \
  --allocation-started-epoch "${PILOT_ALLOCATION_STARTED_EPOCH:?}"
"${PILOT_PYTHON:?}" src/run_metal_architecture_pilot.py summarize \
  --output-dir "${PILOT_OUTPUT_DIR:?}"
```

For verified archive transfer, append `--persistence-receipt` with the path to
the verified storage receipt to preflight and execute. Each execute call runs
at most one attempt. Archive and verify it before the next call; inspect
`campaign_state.json` and continue only while it requests the next step. Stop
on a completed campaign, budget stop, or unresolved failure. For a separately
prepared feature overlay, pass its root and manifest when initially calling
`plan` through `external_features_root_dir` and `feature_overlay_manifest`;
do not replace feature paths after planning.

Execution requires durable artifacts, either a verified mounted Drive output
directory or a verified transfer workflow. For the transfer workflow, copy and
verify each completed attempt locally and in Drive before permitting the next
attempt, and retain the receipt. A preview is not proof that persistence works.
Charge every failed/interrupted attempt; allow one linked retry per
configuration within the global 60-minute retry allowance.
Completed-run reuse validates artifacts and the source/configuration/data
identities. This supports campaign continuation, not mid-epoch training resume.
Follow `COLAB_GPU_RUNBOOK.md` for runtime teardown after verified closeout.

### Expected outputs and decision gate

Expected campaign files are:

- `campaign_manifest.json`, `commands.txt`, and `run_matrix.csv`;
- `campaign_attempt_ledger.json`, `campaign_attempt_ledger.csv`, and
  `allocation_ledger.json`;
- `expected_split.json`, `pooling_diagnostics.json`, and `readiness.json`;
- `architecture_screen.csv`, `target_formulation_screen.csv`,
  `campaign_coverage.json`, `validation_results.json`, and `decision_record.md`;
- `campaign_state.json` for the current queue/stop decision, plus
  `transfer_receipts/attempt_NNN.json` when verified transfer is used.

Each completed training run preserves `run_config.json`,
`run_metadata.json`, `dataset_summary.json`, `split_diagnostics.json`,
`prepare_status.json`, `epoch_metrics.csv`, `train_metrics.csv`,
`val_metrics.csv`, and `best_model_checkpoint.pt`. The pilot manifest records
each command's resolved configuration; the ordinary notebook planner's
`active_run_config.json`/`.md` remain separate from this pilot manifest.
No `test_report.json` may be present.

Report native metrics and native per-class recalls from the checkpoint selected
by `val_metal_balanced_acc`. Form common-four predictions by summing Fe/Co/Ni
probabilities (or Fe with grouped Co+Ni for five classes) before argmax; report
that view from the same checkpoint. Compare target formulations on the common
four-class endpoint. Compare combined models with both unimodal baselines;
keep runtime, incomplete coverage, zero-recall classes, and LR opportunities
visible. Report chosen-LR seed repeats separately from unmatched discovery
rows; never average different LRs into one family score.

The screen ends with a **continue, rescue once, park, or confirm** decision
based only on completed validation evidence. No model is promoted by this
screen. A later cycle must be separately costed; freeze one configuration per
required comparison arm before shared five-fold grouped confirmation crossed
with model seeds 42/43. Paired fold-level confidence intervals, seeds averaged
within folds, and rare-class recall protection are required for promotion.
Final reporting still requires resolution of the final-test route, Stage 6
confirmation, a completed Stage 6B full-train refit, and one-shot Stage 7.
Exact PinMyMetal is a later separately labeled reference benchmark; its
overlapping PDB IDs and the historical use of its shared test remain disclosed.

## Bounded coordination-geometry pilot — Stage 2B

Profile: **`metal_coordination_geometry_pilot_v1`**. This is a separate
validation-only, direct-four Only-GVP comparison of added candidate-ligand
counts, angular summaries, and generic metal nodes. It follows complete A1
and A2 blocks of `metal_architecture_pilot_10h_v1`, then returns control to
that profile's remaining blocks. Read `EXPERIMENT_STATUS.md` for measured
completion and recovery status; this recipe defines authorized work, not
execution evidence.

### Exact arms and fixed controls

| Arm | `--site-geometry-features` | `--metal-node-mode` | Added count slots | Added angular slots |
|---|---|---|---|---|
| A | `none` | `none` | Masked | Masked |
| B | `counts` | `none` | `log1p` counts | Masked |
| C | `counts_angles` | `none` | `log1p` counts | Six summaries / 180 |
| D | `counts` | `per_metal` | `log1p` counts | Masked |
| E | `counts_angles` | `per_metal` | `log1p` counts | Six summaries / 180 |

All five arms use four existing site inputs plus the same eight geometry
slots, the same generic residue/metal node-type embeddings, and identical
model parameter shapes. `none` masks the added geometry slots; it does not
remove the existing site inputs or other geometric features. Historical
`legacy` Only-GVP results are context: run a fresh A control with the matched
explicit machinery. No arm encodes the true metal symbol, atomic number, or
target class as an input.

| Control | Fixed geometry-pilot value |
|---|---|
| Task, architecture, target | `metal`; `only_gvp`; direct `four_class` (`merge_fe_class_viii`); no ESM branch |
| Checkpoint selection | Native `val_metal_balanced_acc`; same selected checkpoint for all reported metrics |
| Data and split | Original pilot's non-overlap training-only cohort and certified external overlay; native-six eligibility; `pdbid` groups; `metal_site` stratification; split seed 42; validation fraction 0.15 |
| Training | 50 epochs; batch 8; LRs `3e-5,1e-4`; fixed schedule; weight decay `1e-4`; initial model seed 42, selected-LR repeat seed 43 |
| Capacity | Hidden scalar/vector 128/16; edge hidden 64; 4 GVP layers; 2 head layers; head dropout 0.2 |
| Features and loss | Conservative; required certified external features; cross-entropy; training-only inverse-frequency class weights; manual multipliers all 1; no label smoothing, site sampler, or collapsed auxiliary loss |
| Extraction and edges | Pocket extraction 10 Å by any atom; residue edges 6 Å by closest atom; radius-only; RING disabled |
| Readout | Explicit `--structural-readout-scope residue_only`; `--classifier-pool-distance-cutoff 0.0`; mean plus learned attention pooling over every extracted residue |
| Runtime and augmentation | One CUDA worker; loader workers 0; deterministic; no position noise, second-shell dropout, or outer-residue dropout |
| Excluded stages | No held-out inputs or evaluation, Optuna, Stage 6 confirmation, final refit, or promotion |

With metal nodes enabled, their messages can change residue representations,
but the nodes are excluded from final pooling. Do not use readout `auto` for
this comparison: it would also change the pooled node types.
Residue-node normalization excludes metal nodes. Edge-distance and
sequence-distance normalization still uses all training edges, so adding
metal edges changes the fitted edge statistics. Preserve this existing
pipeline; the metal-node arms test representation, connectivity, and edge
normalization together. They do not isolate a topology-only effect.

The first two geometry slots are the number of residue–metal candidate
geometries and the number of within-metal ligand-vector pairs. The other six
are angle minimum, mean, maximum, population standard deviation, mean absolute
deviation from 109.47°, and mean deviation from the nearer of 90°/180°. Apply
`log1p` only to counts and divide each angular summary by 180. For counts-only
arms, keep the angular slots zero rather than changing input width. These
explicit modes consume separately preserved raw summaries; they do not
transform the legacy pipeline's standardized summary tensor. Preserve that
legacy normalization for historical configurations.

The existing helper is unchanged. It chooses one nearest candidate per residue
per metal from up to two listed donor atoms, with a functional-group-centroid
fallback if no listed donor is available; this may ultimately use a sidechain
centroid or Cα. It omits waters, cofactors, and noncanonical residues. A
first-shell assignment can retain the nearest metal beyond the usual ligand
cutoff. Angular pairs are formed within individual centers and then pooled
across centers, so a bridging residue can count once per center. Metal-edge
construction can force a nearest connection for an otherwise disconnected
center; the summary helper does not. Record summary counts and graph edges
separately. These are candidate-geometry features, not certified coordination
numbers or full coordination-shape labels.

### Matched schedule and shared budget

Run five fresh one-epoch smokes, one per arm at LR `3e-5`, model seed 42.
They must verify finite losses/metrics, all classes present, the exact shared
cohort, expected masked/scaled site inputs, identical parameter shapes,
generic node types, and residue-only pooling. Inspect empty/degenerate angle
cases and metal-edge versus summary-count differences. Measure complete
prepare/train/validation/save time on the current GPU; smoke scores do not
rank arms.
Require identical fitted-normalization hashes within A/B/C and within D/E,
and preserve each arm's hash across its later runs. A cross-group difference
is expected to be possible because metal edges participate in normalization.

Then run all five arms for 50 epochs at LR `3e-5`, followed by all five at
`1e-4`, with model seed 42:

| Block | Arms | LR | Model seed | Epochs | Runs |
|---|---|---|---:|---:|---:|
| S | A–E | `3e-5` | 42 | 1 | 5 |
| G1 | A–E | `3e-5` | 42 | 50 | 5 |
| G2 | A–E | `1e-4` | 42 | 50 | 5 |
| GR | A–E | Selected independently per arm | 43 | 50 | 5 |

Choose one LR independently for each arm by native
validation balanced accuracy, then minimum class recall, then lower LR for a
remaining tie. Repeat that selected configuration with model seed 43. The
maximum geometry work is **five smokes plus 15 full runs**: ten LR-screen
runs and five repeats. Split seed and validation membership remain fixed.

The geometry pilot shares the original **600-minute cumulative allocation
cap**; it does not receive another ten hours. Preserve the original planning
allowances: setup/profile 60, main training 450, retries 60, closeout 30
minutes, with training ending by cumulative minute 570. Count prior closed
allocations, new setup, transfers, failed attempts, recovery, and teardown.
Read current usage from the original campaign's allocation ledger and verified
closeout evidence rather than recalculating it from successful training time.
Additional geometry readiness and smokes consume the shared main allowance;
linked retries consume the shared retry allowance.

Before the first full geometry fit, forecast **all 15 full runs**, including
the selected-LR repeats, from fresh per-arm smoke costs on the allocated GPU.
Update the estimate with full-run measurements and multiply the cost of all
remaining full runs by **1.25** before each admission. If the remaining
comparison does not fit, stop and report incomplete coverage. This protects
the five-arm comparison and its seed repeats together. Do not selectively
shorten epochs or silently drop arms to manufacture a complete comparison.
Run one worker and archive each bounded attempt before advancing. The original
remaining architecture blocks retain their identities and may also remain
unfinished under the shared cap.

### Exact standalone execution

Use `src/run_metal_coordination_geometry_pilot.py`; no notebook mutation is
required. Before the first geometry smoke or full fit, verify the original
A1/A2 artifacts, their local and Drive copies, and the validated recovery of
the original campaign ledger if a new GPU allocation is needed. Preserve the
closed allocation usage.
Freeze the new source snapshot separately, including the geometry runner and
this recipe. Keep the original source and run receipts intact.

The operational recovery route is `scripts/colab_metal_pilot_resume.py`,
documented with the runtime workflow and current execution evidence. It checks
the original manifest, cohort, and content hashes and writes
`cross_session_recovery/readiness.json`; it does not add a new resume command
to the frozen original runner. Require successful recovery evidence before
advancing that original queue on a new allocation.

CPU planning and bounded readiness may precede A1/A2 completion, but the
execute gate requires both blocks complete and no active original training
process. Keep all GPU work serial.

From the verified checkout root, set `GEOMETRY_PYTHON` to its verified runtime
interpreter, `GEOMETRY_DATA_ROOT` to the data root, `GEOMETRY_OUTPUT_DIR` to a
new geometry campaign directory, and `GEOMETRY_PARENT_CAMPAIGN_DIR` to the
restored original architecture campaign directory. The overlay variables
identify the same certified training-only features used by the parent.
`GEOMETRY_SOURCE_ID` identifies the new frozen source snapshot.

```bash
"${GEOMETRY_PYTHON:?verified interpreter required}" src/run_metal_coordination_geometry_pilot.py plan \
  --data-root "${GEOMETRY_DATA_ROOT:?verified data root required}" \
  --output-dir "${GEOMETRY_OUTPUT_DIR:?new geometry output required}" \
  --source-commit "${GEOMETRY_SOURCE_ID:?frozen source identity required}" \
  --parent-campaign-dir "${GEOMETRY_PARENT_CAMPAIGN_DIR:?original campaign required}" \
  --external-features-root-dir "${GEOMETRY_EXTERNAL_FEATURES_ROOT_DIR:?certified overlay required}" \
  --feature-overlay-manifest "${GEOMETRY_FEATURE_OVERLAY_MANIFEST:?verified overlay manifest required}"
```

Planning writes a reviewable queue; it is not permission to bypass the parent
completion, budget, source, or storage checks. Set
`GEOMETRY_ALLOCATION_STARTED_EPOCH` to the recorded start of the actual live
allocation, including setup. Preserve it across calls within that allocation;
record a new interval only through the validated recovery/allocation workflow.
`--budget-root` always points to the original campaign so that switching
profiles does not reset usage. For the verified archive-transfer route:

```bash
"${GEOMETRY_PYTHON:?}" src/run_metal_coordination_geometry_pilot.py preflight \
  --output-dir "${GEOMETRY_OUTPUT_DIR:?}" \
  --allocation-started-epoch "${GEOMETRY_ALLOCATION_STARTED_EPOCH:?recorded allocation start required}" \
  --budget-root "${GEOMETRY_PARENT_CAMPAIGN_DIR:?}" \
  --persistence-receipt "${GEOMETRY_PERSISTENCE_RECEIPT:?verified persistence receipt required}"
"${GEOMETRY_PYTHON:?}" src/run_metal_coordination_geometry_pilot.py execute \
  --output-dir "${GEOMETRY_OUTPUT_DIR:?}" \
  --allocation-started-epoch "${GEOMETRY_ALLOCATION_STARTED_EPOCH:?}" \
  --budget-root "${GEOMETRY_PARENT_CAMPAIGN_DIR:?}" \
  --persistence-receipt "${GEOMETRY_PERSISTENCE_RECEIPT:?}" \
  --max-runs 1
"${GEOMETRY_PYTHON:?}" src/run_metal_coordination_geometry_pilot.py summarize \
  --output-dir "${GEOMETRY_OUTPUT_DIR:?}"
```

Each execute call is bounded to one attempt. Verify its local and Drive
archives and receipt before the next call. Inspect the persistent next-step
decision after every attempt; do not use an unchecked shell loop. Mounted
Drive output follows the runner's verified mounted-storage contract instead.
Follow `COLAB_GPU_RUNBOOK.md` for connection handling and verified teardown.

### Expected geometry outputs

The separate geometry directory preserves:

- `campaign_manifest.json`, `commands.txt`, `run_matrix.csv`, and
  `expected_split.json`, with the parent manifest/cohort and new source hashes;
- `readiness.json`, `geometry_diagnostics.json`, and
  `training_cache_audit.json`;
- `geometry_normalization_controls.json`, recording within-graph-group and
  within-arm normalization checks before full-run admission;
- `campaign_attempt_ledger.json`, `campaign_attempt_ledger.csv`,
  `campaign_state.json`, and `transfer_receipts/attempt_NNN.json` when using
  verified transfer;
- `geometry_campaign_admission.json` after full-campaign admission and
  `geometry_selected_learning_rates.json` before the seed-repeat block;
- `geometry_screen.csv`, `geometry_validation_results.json`,
  `geometry_coverage.json`, and `geometry_decision_record.md`.

The manifest includes both LR alternatives as seed-43 templates; only the
selected LR is active for each arm. Count completed runs from coverage and
verified attempt artifacts, not the number of manifest templates. The original
budget directory owns `allocation_ledger.json` and receives
`coordination_geometry_budget_usage.json` so later original-profile block
admission includes geometry costs. This handoff is enforced by the host
orchestration; it does not modify the frozen original runner.

Every completed fit preserves `run_config.json`, `run_metadata.json`,
`dataset_summary.json`, `split_diagnostics.json`, `prepare_status.json`,
`epoch_metrics.csv`, `train_metrics.csv`, `val_metrics.csv`, and
`best_model_checkpoint.pt`. Keep checkpoint archives locally and in Drive;
portable documentation copies contain lightweight evidence with the shared
cohort recorded once. No `test_report.json` may be present. Notebook
`active_run_config.json`/`.md` are not this standalone runner's resolved
configuration; the campaign manifest and per-run files own that record.

### Interpretation and decision gate

Inspect B−A for counts without metal nodes; C−B for angular summaries without
metal nodes; D−B for metal nodes with counts fixed; E−C for metal nodes with
counts and angles fixed; and E−D for angular summaries with metal nodes.
The node contrasts include the edge-normalization changes described above;
C−B and E−D keep the graph and its fitted normalization fixed.
A−E changes several components and cannot isolate one mechanism. The design
does not include a nodes-only arm, so it does not estimate a node effect with
all added geometry inputs masked.

Report each completed run's selected native balanced accuracy, minimum and
per-class recall, epoch, LR, seed, duration, and measured memory with its
sampling limitations. Compare arms at matched LR/seed first. Keep discovery
rows distinct from selected-LR repeats; do not average different LRs into one
arm score. With two seeds, report mean and sample SD for the fixed selected
configuration, including recall deterioration and disagreement between seeds.

End with an exploratory continue, park, or separately costed confirmation
decision. A single fixed split and two seeds cannot provide the grouped-fold
evidence required for promotion. Do not select from held-out results or launch
new fusion, RING, loss, capacity, or EC combinations from this queue. Continue
the original remaining blocks only after the geometry queue reaches its
completed or explicitly budget-stopped state and the shared budget permits
the next complete block. Stage 6/6B/7 requirements remain unchanged.

This playbook is the practical, notebook-ready pipeline for DeepMzyme metal
classification. It complements `Plan.md`, which remains the high-level research
and design authority. Current best validation evidence belongs in
`EXPERIMENT_STATUS.md` and `docs/notebook_outputs/`, not in this stable
playbook.

For the cross-document run order and output-folder map, see `docs/README.md`.

## Bounded matched RING continuation — Stage 2B

This is a separately identified continuation after the architecture and
coordination-geometry pilots have closed. It addresses one missing controlled
comparison in `Plan.md`; it does not replace Stage 5G, serious HPO, or Stage 6.
Use `src/run_metal_ring_pilot.py` with profile `metal_ring_pilot_v1`.

### Fixed comparison and schedule

Both arms use direct `four_class` training, canonical `merge_fe_class_viii`,
with `metal_eligibility_scheme=six_class` to retain the original certified
development cohort. Use only the external training partition of the
non-overlapped PinMyMetal dataset, split internally by `pdbid`, validation
fraction 0.15, split seed 42, and metal-site stratification. Freeze the parent
manifest and ordered retained split by SHA-256 before preparing inputs.

| Block | Family | RING | Learning rates | Model seeds | Epochs | Fits |
|---|---|---|---|---|---|---|
| S | Only-GVP and graph-level late fusion | Off, on | `3e-5` | 42 | 1 | 4 |
| GVP | Only-GVP | Off, on | `3e-5`, `1e-4` | 42, 43 | 50 | 8 |
| LATE, optional | GVP + graph-level late fusion | Off, on | `3e-5`, `1e-4` | 42, 43 | 50 | 8 |

Train all controls freshly under the same frozen source. Do not reuse an old
RING-off result as a cell in this matrix. Run each family in increasing LR,
then seed, with RING off followed by on. No LR is selected to prune this matrix.

Fixed controls: conservative node features, radius 6 Å, 10 Å pocket extraction,
classifier pooling cutoff 0, residue-only readout, no explicit metal nodes,
`site_geometry_features=legacy`, no augmentation, batch 8, weight decay
`1e-4`, inverse-frequency weighted cross entropy, zero label smoothing, no
collapsed auxiliary loss, deterministic fixed LR schedule. Select checkpoints
by `val_metal_balanced_acc`. Preserve the baseline encoder dimensions and
head settings resolved from the standalone notebook recipe; save every
expanded CLI command and parsed configuration in the manifest.

Set `--shell-role-source geometry` in **both** arms. The default `edge_mode`
otherwise changes second-shell node annotations when RING is enabled, which
would confound an edge comparison. This new option preserves old behavior by
default and is explicitly set by this standalone runner; no notebook default
is changed. The off arm omits `--use-ring-edges` and `--require-ring-edges`;
the on arm requires both. Both disable missing-RING preparation. Missing,
malformed, or inconsistent input features fail readiness.

The comparison estimates RING edges and interaction features **with their
training-fitted edge normalization**. Node inputs must match exactly in the
full retained-cohort audit. Edge normalization may differ between arms and
must be reported. RING can annotate existing radius edges and add pairs;
record these separately. This recipe does not consume the raw RING `Angle`
column and does not add a metal node or explicit coordination-angle inputs.

### Original budget and persistence

Use one named G4 Colab allocation at a time. Carry both closed earlier
allocation intervals and exact category costs into a new immutable
`budget_handoff.json` under profile `metal_ring_continuation_budget_v1`.
Do not alter the original campaign receipts or restart its clock.

- Original total allocation cap: 36,000 seconds; training cutoff: 34,200
  cumulative seconds, retaining 1,800 seconds for closeout.
- Original normal-work cap: 27,000 seconds; retry cap: 3,600 seconds. Deduct
  earlier recorded usage before admitting any new work.
- New provisioning, uploads and bootstrap: at most 1,800 seconds, charged
  once to normal work through `bootstrap_budget_usage.json` and continuously
  to the allocation clock.
- New full-cohort readiness: at most 1,200 seconds per attempt, charged to
  normal work, or retry allowance for its one linked retry. This is not a
  renewed original first-hour profiling allowance. The ceiling accommodates
  four complete model preparations plus full-cohort graph/cache checks;
  ordinary prior single-fit setup timings are insufficient to budget this
  combined audit. It does not increase the original cumulative caps.
  Readiness uses one CPU thread for Torch/OMP/MKL/OpenBLAS to avoid excessive
  threading overhead on many small graph operations. Full training keeps the
  same runtime settings for both edge arms.
- Admit a complete remaining eight-fit family block only when the sum of
  measured per-family/per-RING setup plus epoch forecasts, multiplied by
  1.25, fits both the remaining normal-work allowance and training deadline.
  Recheck before each fit. A partial family block is incomplete evidence.
- Persist each completed or failed attempt to local storage and Drive,
  verify its archive hash locally and Drive ID/size/parent, and return the
  receipt before the next attempt. Preserve failure logs and linked retries.
- Stop the owned allocation on completion or failure and verify its absence.

### Exact standalone execution

Bootstrap prepares these explicit paths and verified persistence receipt.
`SOURCE_COMMIT` identifies the Git base; the source archive and manifest
separately hash the complete working snapshot, including uncommitted files.

```bash
python /content/DeepMzyme_ring_v1/src/run_metal_ring_pilot.py plan \
  --data-root /content/deepmzyme_bundle/DeepMzyme_Data \
  --output-dir /content/metal_ring_pilot_v1 \
  --source-commit "$SOURCE_COMMIT" \
  --parent-reference-dir /content/metal_ring_parent_reference \
  --external-features-root-dir /content/metal_architecture_pilot_features/external_overlay \
  --feature-overlay-manifest /content/metal_architecture_pilot_features/external_overlay/feature_overlay_manifest.json \
  --budget-root /content/metal_ring_continuation_budget_v1

python /content/DeepMzyme_ring_v1/src/run_metal_ring_pilot.py preflight \
  --output-dir /content/metal_ring_pilot_v1 \
  --budget-root /content/metal_ring_continuation_budget_v1 \
  --allocation-started-epoch "$ALLOCATION_STARTED_EPOCH" \
  --persistence-receipt /content/metal_ring_persistence_receipt.json

# Invoke once per attempt, after archiving and verifying the previous attempt.
python /content/DeepMzyme_ring_v1/src/run_metal_ring_pilot.py execute \
  --output-dir /content/metal_ring_pilot_v1 \
  --budget-root /content/metal_ring_continuation_budget_v1 \
  --allocation-started-epoch "$ALLOCATION_STARTED_EPOCH" \
  --persistence-receipt /content/metal_ring_persistence_receipt.json \
  --max-runs 1

python /content/DeepMzyme_ring_v1/src/run_metal_ring_pilot.py summarize \
  --output-dir /content/metal_ring_pilot_v1
```

### Outputs and decision gate

Before training require `campaign_manifest.json`, `commands.txt`,
`run_matrix.csv`, `expected_split.json`, `ring_input_audit.json`,
`training_cache_audit.json`, `readiness.json` and the cumulative budget
handoff/ledger. Readiness certifies cohort equality, fixed node/site inputs,
valid RING edges, feature availability, and actual-data CUDA forward/backward.
All four one-epoch smoke runs must complete before full fits are admitted.

Each fit saves `run_config.json`, `run_metadata.json`, `dataset_summary.json`,
`epoch_metrics.csv`, `best_model_checkpoint.pt`, and
`last_model_checkpoint.pt`. The runner also writes
`campaign_attempt_ledger.json`, `campaign_attempt_ledger.csv`,
`campaign_state.json`, `ring_validation_results.json`, `ring_coverage.json`,
`ring_screen.csv`, and `ring_decision_record.md`. Family admission receipts
are `gvp_admission.json` and, if admitted, `late_admission.json`.
This standalone execution does not generate notebook `active_run_config.*`.

Report on-minus-off validation balanced-accuracy differences separately at
each matched LR and seed, selected epochs, class recalls, duration, memory
measurement limits, and edge-normalization differences. Retain seed/LR
disagreement and rare-class deterioration. Do not select a winning LR from
the test set or average distinct LRs into one architecture score.

The completed eight-fit block supports exploratory prioritization only.
Two model seeds on one split do not supply independent validation folds or
the paired bootstrap CIs required for promotion. Keep all held-out evaluation
disabled (`INCLUDE_HELD_OUT_TEST_DURING_TRAINING=False` in notebook terms).
Any later superiority claim requires matched Stage 6 grouped folds/seeds,
paired-CI and rare-recall gates, followed by Stage 6B final refit before Stage
7. Missing the optional LATE block leaves that comparison planned.

## Current Dataset/Final-Reporting Warning

> **Primary final-test route: unresolved scientific decision required before final reporting.**

The exact stage blocks below are preserved unchanged. Their common dataset
default, current bundle availability, `Plan.md` final-split intent, and the
historical access record for non-overlapped PinMyMetal do not currently define
one unambiguous primary final-test route. Do not silently substitute exact
PinMyMetal or describe the historical non-overlap test as pristine.

Before any final reporting, read [`DATASETS.md`](DATASETS.md) and
[`TECH-006`](FOLLOW_UP_TECHNICAL_ISSUES.md#tech-006--final-dataset-implementation-and-availability-conflict).
This warning changes no stage value, budget, range, seed, gate, output name, or
notebook behavior.

## Primary Four-Class Policy / Recipe Reconciliation Warning

`Plan.md` now designates Mn, Cu, Zn, and Class VIII = Fe+Co+Ni as the intended
primary PinMyMetal-compatible reporting endpoint. It requires a controlled
comparison of direct `four_class` / `merge_fe_class_viii` training against
matched `six_class` training followed by collapsed-four evaluation.

The new single-GPU campaign supplies both required target arms through bounded
discovery and exploratory grouped-fold confirmation. The historical bounded
pilot supplies both target arms and a labeled five-class
challenger, with native-selection and same-checkpoint common-four reporting.
The retained opening standalone Stage 0–2B recipe supplies its historical pair.
The older blocks below retain their six-class common recipe as historical
workflow references; they are not a direct-four campaign or a matched pair.

For the new bounded metal campaign use the
[single-GPU recipe](#single-gpu-metal-campaign), including native selection and
target-independent split control. Before extending it
to HPO or final reporting, reconcile paired later-stage blocks. Coordinate study/run identities, common-view
metrics, active-class metrics, rare-class gates, and Stage 6/6B/7 provenance
with the notebook launch surface. Do not patch one label value in isolation or
reuse an incompatible persistent study. This open work is tracked in
[`TECH-010`](FOLLOW_UP_TECHNICAL_ISSUES.md#tech-010--four-class-endpoint-and-paired-metal-target-recipes-are-not-reconciled).


## Quick-Paste Stage Selector

This playbook is the operational pipeline for
`notebooks/DeepMzyme_training_colab.ipynb`. Each stage is a self-contained
notebook configuration block; you paste one block at a time into the Main
configuration cell. Stages 1, 3 are smoke checks. Stages 2, 4, 5, 6 are
validation-only. Stage 6B is the validation-to-final-refit bridge: it applies
promotion gates to Stage 6 evidence and trains/refits one final model on the
full non-test training set. Stage 7 is the only stage that touches the held-out
test set, and it is run exactly once for that frozen Stage 6B final-refit run.

Use these stage names exactly when planning, documenting, or asking an agent
what to run next:

- Stage 0: environment/data readiness
- Stage 1: 1-epoch smoke
- Stage 2A: Only-GVP validation anchor
- Stage 2B: baseline family comparison
- Stage 3: Optuna plumbing debug
- Stage 4: medium per-family Optuna, optional on G4
- Stage 5A: serious Only-GVP HPO
- Stage 5B: Only-ESM HPO
- Stage 5C: GVP + late fusion HPO
- Stage 5D: GVP + node-level late fusion HPO
- Stage 5E: GVP + hybrid fusion HPO
- Stage 5F: GVP + cross-attention HPO
- Stage 5G: RING/radius-only ablation
- Stage 6: top-K seed/split confirmation
- Stage 6B: promotion gates and final full-train refit
- Stage 7: one-shot held-out test

## Pipeline Overview At A Glance

| Stage | Purpose | Owns exact budget? | G4 wall-time (approx.) | Pass/fail decision gate | Required outputs |
| --- | --- | --- | --- | --- | --- |
| Stage 0 | Environment, Drive, data bundle, RING/ESM/external-feature readiness | Yes, planning-only | 10-20 min | Planned config resolves under Drive, coverage diagnostics pass, no test artifacts | Planned-run CSV/dictionary, optional metal-weight diagnostics |
| Stage 1 | 1-epoch smoke to prove notebook and training path | Yes, smoke budget | 5-15 min | One validation-only run completes; no missing paths/classes; no test artifacts | Planned files, one run dir, `run_config.json`, `run_metadata.json`, `split_diagnostics.json` |
| Stage 2A | Only-GVP validation anchor | Yes | 10-16 h | All planned validation runs complete and rare-class diagnostics are usable | Planned files, run dirs, summary CSV/PNG, no `test_report.json` |
| Stage 2B | Baseline family comparison after ESM is ready | Yes | 8-14 h | All planned validation runs complete and ESM coverage is valid | Planned files, run dirs, summary CSV/PNG, no `test_report.json` |
| Stage 3 | Optuna plumbing debug | Yes, debug only | 20-40 min | Four complete validation-only trials and valid persistent-storage plumbing | Optuna `all_trials.csv`, `top_trials.csv`, `best_trial.json`, study summary |
| Stage 4 | Medium per-family Optuna, optional on G4 | Yes | 8-16 h | Sixty-four complete validation-only trials in one `MODEL_PRESET` | Optuna CSV/JSON/Markdown outputs and per-trial run dirs |
| Stage 5A | Serious Only-GVP HPO | Yes | 36-60 h | Two hundred complete validation-only trials in the Only-GVP study | Optuna CSV/JSON/Markdown outputs and per-trial run dirs |
| Stage 5B | Only-ESM HPO | Yes | 24-48 h | One hundred twenty complete validation-only trials with valid ESM coverage | Optuna CSV/JSON/Markdown outputs and per-trial run dirs |
| Stage 5C | GVP + late fusion HPO | Yes | 36-60 h | Two hundred complete validation-only trials with valid ESM coverage | Optuna CSV/JSON/Markdown outputs and per-trial run dirs |
| Stage 5D | GVP + node-level late fusion HPO | Yes | 36-60 h | Stage 5C gate passed, then two hundred complete validation-only trials | Optuna CSV/JSON/Markdown outputs and per-trial run dirs |
| Stage 5E | GVP + hybrid fusion HPO | Yes | 36-60 h | Stage 5C gate passed, then two hundred complete validation-only trials | Optuna CSV/JSON/Markdown outputs and per-trial run dirs |
| Stage 5F | GVP + cross-attention HPO | Yes | 30-55 h | Stage 5C gate passed, then one hundred twenty complete validation-only trials | Optuna CSV/JSON/Markdown outputs and per-trial run dirs |
| Stage 5G | RING/radius-only ablation | Yes, ablation budget | 6-10 h | Matching radius-only validation runs complete and are labeled as ablation | Planned files, run dirs, summary CSV/PNG, no `test_report.json` |
| Stage 6 | Top-K seed/split confirmation | Yes | 15-25 h for one seed; more with extra seeds | All predeclared top-K x fold x active-seed validation runs complete; one candidate selected by paired validation evidence | `seed_repeat_results.csv`, `seed_repeat_summary.csv`, `seed_repeat_summary.json`, `seed_repeat_pairwise_bootstrap.csv`, `seed_repeat_pairwise_bootstrap.json`, `stage6_ranked_candidates.csv`, `stage6_selected_final_candidate.json`, run dirs |
| Stage 6B | Promotion gates and final full-train refit | Yes | One final training run | Stage 6 candidate passes configured paired-CI, rare-recall, and tie-breaker gates; final refit completes with no test report | `stage6b_ranked_candidates.csv`, `stage6b_decision.json`, `stage6b_final_refit_command.txt`, `stage6b_selected_final_refit_candidate.json`, final-refit run dir |
| Stage 7 | One-shot held-out test | Yes, final only | 20-60 min | Source is the frozen Stage 6B final-refit run and one-shot policy is confirmed | Separate final-test run dir, `test_report.json`, final-test summary |

All configuration blocks below use variables that exist in
`notebooks/DeepMzyme_training_colab.ipynb` as of this repository state. For
ordinary planned training and HPO stages, edit the notebook's **Main
configuration** cell directly or paste the block at the end of that cell before
running **Build central CONFIG dictionary**. Stage 6 grouped-fold confirmation
uses the dedicated **Stage 6 controls and existing Optuna/HPO reuse** panel; for
an already completed HPO directory, Stage 6 can run in standalone existing-HPO
mode independently of the Main configuration cell.

## How To Use This Playbook

Use exactly one stage block at a time. For ordinary planned training or HPO,
after editing the notebook's **Main configuration** cell, run the CONFIG/planning
cells and inspect the resolved commands before setting
`LAUNCH_PLANNED_MAIN_TRAINING_RUNS = True` in the dedicated launch-switch cell.
For Stage 6 from an already completed HPO directory, keep that switch `False`,
fill the Stage 6 controls, and use the dedicated Stage 6 launch cell. It is now
safe to use Colab **Run all** for this path: setup/clone/data cells run, the
ordinary main training/HPO cell no-ops, and the Stage 6 launch cell switches to
standalone existing-HPO mode when an old HPO source is configured. If no Stage 6
source is configured, the Stage 6 launch cell no-ops with a message instead of
trying to import from an empty current run folder.

Notebook execution order:

1. Setup/install and data-source cells.
2. Main planned-training launch switch. Keep it `False` while planning or
   loading helpers; set it `True` only before ordinary planned runs or HPO.
3. For ordinary planned runs/HPO: Main configuration cell with one block from
   this playbook.
4. For ordinary planned runs/HPO: Build central `CONFIG`.
5. For ordinary planned runs/HPO: Planning/preflight cells.
6. For ordinary planned runs/HPO: Optional training execution cell.
7. Summarize/report cell for the current `RUN_BATCH_ID` when relevant.
8. For Stage 6 only: Stage 6 controls/checklist, then
   **Launch Stage 6 top-K grouped-fold confirmation**.
9. For Stage 6B: apply promotion gates, preview the final-refit command, then
   launch one final full-train refit from the selected configuration without
   changing model-selection choices.
10. For final testing only: select that Stage 6B final-refit run, preview final
    held-out test, then launch once.

For all comparison, HPO, and Stage 6 confirmation stages:

- Keep `INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False`.
- Keep `VAL_FRACTION = 0.15` and `SPLIT_BY = "pdbid"` unless a new split
  experiment is explicitly being labeled. `pdbid` grouping is stricter than
  `pdbid_chain` grouping: it keeps all chains and pockets from one PDB entry on
  one side, so binuclear or repeated same-chain metal sites cannot leak across
  train/validation. Stage 6 grouped-fold confirmation is the planned exception:
  it sets `VAL_FRACTION = 0.0`, `SPLIT_BY = "pdbid"`,
  `SEED_REPEAT_N_FOLDS = 5`, a fixed `SEED_REPEAT_SPLIT_SEED`, and the
  predeclared `REPEAT_SEEDS` model-seed list. Stage 6B is the final-refit
  exception: it uses `VAL_FRACTION = 0.0` because the selected configuration is
  retrained on the full non-test training set after validation/CV selection.
- Use validation metrics, usually `val_metal_balanced_acc`, for checkpoint,
  hyperparameter, architecture, and fusion decisions.
- Do not run the optional final held-out test cell until the final
  validation-selected configuration is fixed and its Stage 6B final-refit run
  has been completed and frozen.
- If the user asks for a new check, new run, or fresh Optuna sweep without
  explicitly asking to rely on previous raws/results, use previous notebook
  outputs only as context and safety checks. Prefer the broadest sensible
  validation-only Optuna search within the selected `MODEL_PRESET`, with
  common-sense runtime and feature-availability limits.
- If the user explicitly asks to rely on previous running/results/raws, inspect
  the relevant copied evidence and use it to narrow, continue, or repeat that
  prior configuration.

## Run Tiers And Reproducibility Records

| Tier | Playbook stages | Selection/reporting status | Required record |
| --- | --- | --- | --- |
| Debug | Stage 0, Stage 1, Stage 3 | Not model-selection evidence | Resolved notebook config, planned commands, run logs, and any failure context |
| Serious validation | Stage 2, Stage 4, Stage 5, Stage 6 | Validation-only model-selection evidence if the stage gate passes | Full run config, split/fold identity, seeds, Optuna study metadata, dataset bundle ID/checksum, git commit, and key library versions |
| Final refit | Stage 6B | Final training from validation-selected configuration; no held-out test | Stage 6 evidence, Stage 6B decision JSON, final-refit command, final-refit run config/metadata/checkpoint |
| Final test | Stage 7 | One-shot held-out reporting only | Stage 6/6B selection evidence, Stage 6B final-refit source run/checkpoint, primary report declaration, calibration/CI settings, dataset bundle ID/checksum, git commit, key library versions, and no-test-selection statement |

Serious validation and final-test records should capture key library versions
when available: PyTorch, torch-geometric, ESM/ESMC, Optuna, NumPy, and
scikit-learn. This repository currently has no checked-in environment spec, so
per-run version records are required until an environment file is added.

Limited-compute fallback: use Stage 4 instead of Stage 5 for candidate
discovery, or stop after Stage 2 with a clearly labeled provisional
validation-only result. Do not launch Stage 7 from a provisional result. A final
held-out report still requires one fixed validation-selected configuration, one
frozen Stage 6B final-refit run derived from it, and the one-shot Stage 7
policy.

`EPOCHS` and `MAX_EPOCHS_PER_TRIAL` have different roles:

- `EPOCHS` is the normal training budget for manual comparison runs, Stage 6
  grouped-fold confirmation runs, and final retraining/evaluation workflows.
- `MAX_EPOCHS_PER_TRIAL` is the per-trial cap only inside
  `RUN_MODE = "controlled_hpo_optuna"`.
- If `MAX_EPOCHS_PER_TRIAL < EPOCHS`, Optuna ranks early-training behavior.
  Stage 6 must then confirm candidates at the full validation budget before any
  final selection.

Bootstrap counts intentionally differ by stage. Stage 6 uses 10,000 paired
bootstrap resamples over shared fold-level differences for candidate-promotion
decisions. Stage 7 uses 1,000 stratified bootstrap resamples by default for
held-out-test reporting uncertainty. Do not use Stage 7 CIs to change the
selected model.

Pruning is now enabled by default in the canonical reportable metal Stage 4,
5A, 5C, 5D, 5E, and 5F blocks using `MedianPruner` with
`OPTUNA_PRUNING_MIN_EPOCH = 25`. The notebook monitors real per-epoch metric
CSVs, reports intermediate values to Optuna, and terminates pruned subprocess
process groups. Pruning can bias the TPE trajectory toward early-learning
behavior, so keep the pruner type and minimum epoch fixed within a study and
record pruned-attempt counts separately from completed trials. Consequence:
`OPTUNA_TARGET_COMPLETE_TRIALS` counts only non-pruned completions, so total
trial attempts will be larger than the target -- plan compute accordingly.
Stage 3 may lower the minimum epoch only for plumbing/debug.

Supported presets without canonical serious HPO blocks:

- `GVP + early fusion` is implemented in the notebook/model preset map and may
  be used in ESM-ready manual comparisons. The bounded architecture pilot owns
  its manual screen recipe. This playbook does not currently own a standalone
  serious HPO block for it; the limited screen does not replace the required
  matched fusion-position confirmation.
- `SimpleGNN + ESM` is implemented as an auxiliary scalar-graph ablation. This
  playbook does not currently own a standalone serious HPO block for it.

Do not present either preset as a required metal HPO stage unless an exact
executable block is added here.

## Retained Six-Class Common Defaults — Reconciliation Required

These are retained six-class values for the older stage examples below.
Use the separate single-GPU recipe for the new bounded campaign. Later-stage
paired HPO and final-reporting recipes still require reconciliation.

```python
TASK = "metal"
METAL_LABEL_SCHEME = "six_class"
DATASET_NAME = "train_and_test_sets_structures_exact_pinmymetal"
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_metal_balanced_acc"
OPTUNA_SELECTION_METRIC = "val_metal_balanced_acc"
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False

RING_EDGE_MODE = "with_ring"
REQUIRE_RING_EDGES = False
PREPARE_MISSING_RING_EDGES = True
RING_FEATURES_DIR = ""
RING_EXE_PATH = "DeepMzyme_Data/ring-4.0/out/bin/ring"

ALLOW_MISSING_EXTERNAL_FEATURES = False
PREPARE_MISSING_EXTERNAL_FEATURES = True
EXTERNAL_FEATURES_ROOT_DIR = ""

CLASSIFIER_POOL_DISTANCE_CUTOFF_VALUES_CSV = "0.0"
POSITION_NOISE_STDS_CSV = "0.0"
SECOND_SHELL_DROPOUTS_CSV = "0.0"  # Fixed off for canonical HPO; use outer-residue dropout instead.
OUTER_RESIDUE_DROPOUTS_CSV = "0.0"

METAL_CLASS_WEIGHT_MODES_CSV = "inverse_frequency"
METAL_LOSS_FUNCTIONS_CSV = "cross_entropy"
METAL_LABEL_SMOOTHING_VALUES_CSV = "0.0"
METAL_COLLAPSED_LOSS_WEIGHTS_CSV = "0.0"
BALANCE_METAL_SITE_SYMBOLS_CSV = "False"

COPY_OUTPUTS_TO_DRIVE = True
METAL_REPORT_VIEW = "both"

DEVICE = "cuda"
SKIP_EXISTING_RUNS = True
STOP_ON_FIRST_FAILURE = False
ALLOW_MODEL_PRESET_MISMATCH = False
ALLOW_SINGLE_MODE_TO_TRUNCATE_COMPARISON = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False

OPTUNA_DIRECTION = "maximize"
OPTUNA_TPE_MULTIVARIATE = True
OPTUNA_TPE_GROUP = True
OPTUNA_TPE_CONSTANT_LIAR = True
OPTUNA_PARALLEL_WORKERS = 1
OPTUNA_PARALLEL_STARTUP_STAGGER_SECONDS = 10.0
OPTUNA_STOP_ON_PARALLEL_CUDA_OOM = True
OPTUNA_SAMPLER_SEED = None
OPTUNA_AUTO_CONFIGURE_BUDGET = False
OPTUNA_USE_PRUNING = False
OPTUNA_PRUNER_TYPE = "none"
OPTUNA_PRUNING_MIN_EPOCH = 20
OPTUNA_TIMEOUT_MINUTES = 0
OPTUNA_MULTIOBJECTIVE = False
```

`DATASET_NAME` chooses the external train/test dataset split. The current
bundle and its verified checksum are listed in [DATASETS.md](DATASETS.md#main-colab-bundle-v12-current-hosted-release-care-complete).
It contains exact, Common-PDBID 70/30, and historical non-overlapped PinMyMetal;
CLEAN30 folds 0-4 through conservative and original shared sources; and CARE
clusterRes30 with complete audited ESM, external, and RING caches. Non-overlapped
data requires an explicit dataset-root override because it is not a dropdown choice. Harsh
PinMyMetal remains unavailable. See the dataset inventory for readiness and
scientific-use limits.
`SPLIT_BY` controls
only the internal train/validation grouping inside the selected external train
split and is emitted to the CLI as `--train-val-split-by`; it never changes the
external test directory or test CSV. When the exact split is requested, the
notebook must stop if that dataset root is missing; it must not silently fall
back to the non-overlapped split.

Summary CSV/PNG basenames are generated from the live provenance by default:
task, metal label scheme, model preset or run set, `DATASET_NAME`,
`RUN_BATCH_ID`, `SPLIT_BY`, and validation mode. A manual `SUMMARY_BASENAME`
override is still allowed, but the notebook warns and records metadata when
the manual name appears inconsistent with the resolved dataset, batch, or split
policy.

ESM/fusion stages currently assume canonical ESMC `esmc_300m` residue
embeddings with `embedding_dim=960`. Newly generated embeddings write a
`*.pt.json` sidecar with model name, embedding dimension, generation time, code
version, and source structure/sequence metadata. Older embeddings without
sidecars must be labeled as `unknown_in_older_embeddings` in run metadata and
status notes rather than guessed.

## G4-Class Optuna Policy

These retained serious-HPO budgets target a verified G4-class GPU. The actual
GPU model, VRAM, throughput and CUDA compatibility must be inspected at each
allocation; G4 is not a 16-GB guarantee and a Colab runtime is not persistent.
Durability comes from verified external artifacts and storage. The separately
budgeted [single-GPU campaign](#single-gpu-metal-campaign) does not launch these
studies. All serious Optuna stages must use:

- `OPTUNA_INTENSITY = "custom"` - never rely on `first_useful`/`serious`
  notebook presets for reportable HPO.
- `OPTUNA_TPE_MULTIVARIATE = True`, `OPTUNA_TPE_GROUP = True`,
  `OPTUNA_TPE_CONSTANT_LIAR = True` so shared-storage studies support multiple
  parallel workers without duplicate/in-flight TPE suggestions.
- `OPTUNA_PARALLEL_WORKERS = 1` is the canonical default and preserves
  historical serial trial execution. For a separately scoped campaign, `2`
  remains an optional validation-only acceleration override after Stage 3 or another short debug
  study confirms there is CUDA memory headroom for the active model family,
  batch-size range, and feature set. Keep `OPTUNA_TPE_CONSTANT_LIAR = True`,
  keep persistent storage enabled, and keep
  `OPTUNA_STOP_ON_PARALLEL_CUDA_OOM = True` when using more than one worker.
  The single-GPU campaign requires one process even if memory would fit two.
- `OPTUNA_SAMPLER_SEED = None` unless deliberately re-exploring the same split
  with a different Optuna trajectory. With `None`, the sampler seed follows
  `OPTUNA_SPLIT_SEED`.
- `OPTUNA_AUTO_CONFIGURE_BUDGET = False` (explicit budgets only).
- `OPTUNA_USE_PRUNING = True` by default in the canonical reportable metal
  Stage 4/5A/5C/5D/5E/5F blocks, with `OPTUNA_PRUNER_TYPE = "median"` and
  `OPTUNA_PRUNING_MIN_EPOCH = 25`. The notebook monitors real per-epoch metric
  CSVs from each trial run directory, reports intermediate values to Optuna, and
  terminates pruned subprocess process groups. Keep the pruner fixed within a
  study because pruner decisions can bias the TPE trajectory. Consequence:
  `OPTUNA_TARGET_COMPLETE_TRIALS` counts only non-pruned completions, so total
  trial attempts will be larger than the target -- plan compute accordingly.
- Persistent SQLite storage in Drive:
  `sqlite:////content/drive/MyDrive/DeepMzyme/optuna/<study_name>.db`.
- Startup trials: use the stage table below. The default rule is at least
  `max(20, 0.2 x OPTUNA_TARGET_COMPLETE_TRIALS)`; the 120-trial Only-ESM study
  uses 30 startup trials to cover its conditional space.
- `OPTUNA_SPLIT_SEED = 42` for every study. Stage 6 uses a separate fixed
  `SEED_REPEAT_SPLIT_SEED` for grouped-fold definitions, so every compared
  candidate sees the same validation folds.
- `OPTUNA_SELECTION_METRIC = "val_metal_balanced_acc"`,
  `OPTUNA_DIRECTION = "maximize"`.
- When `RUN_MODE = "controlled_hpo_optuna"`, active Optuna categorical choices
  reuse the normal CSV fields where those fields exist. This includes
  `LR_SCHEDULES_CSV`, `BATCH_SIZES_CSV`, `METAL_CLASS_WEIGHT_MODES_CSV`, loss
  mode fields, booleans, and `CROSS_ATTENTION_HEADS_CSV`.
- Numeric Optuna fields can optionally use explicit `OPTUNA_*_RANGE`
  overrides. Blank range fields keep the CSV behavior. Nonblank float ranges
  use `low,high`; nonblank integer ranges use `low,high,step`; weight decay is
  log-sampled. Use a fresh Optuna study name for any range-enabled experiment.
  `OPTUNA_SEARCH_PRESET` still decides whether each model-capacity field is
  sampled or fixed.
- `OPTUNA_MULTIOBJECTIVE = False` by default. Optional multi-objective studies
  are validation-only Stage 5A experiments and use NSGA-II over
  `val_metal_balanced_acc` and active metal-scheme `val_metal_min_recall`; they
  do not replace the normal single-objective path.
- Record both the split seed and the sampler seed in the notebook output, study
  summary, and per-run artifacts. If the sampler seed is `None`, record the
  effective sampler seed as the split seed.
- Record `OPTUNA_PARALLEL_WORKERS`, startup stagger seconds, and CUDA-OOM stop
  behavior in the study metadata. Parallel trial order is inherently
  nondeterministic, so Stage 6 grouped-fold confirmation remains mandatory
  before promotion.
- Record `DATALOADER_NUM_WORKERS` and `DATALOADER_PIN_MEMORY` in the generated
  run configuration. These are runtime-throughput controls, not model-selection
  knobs; keep them fixed within a comparable study unless a run is explicitly
  labeled as a DataLoader throughput/debug check.

Forbidden in serious stages:

- Mixing `MODEL_PRESET` values inside one study (Optuna optimizes one family at
  a time).
- Reusing a persistent study DB for a different model preset, architecture,
  fusion mode, split, task, selection metric, or search-space hash. The notebook
  hard-stops incompatible persistent-study reuse unless
  `OPTUNA_ALLOW_INCOMPATIBLE_STUDY_REUSE = True`; leave that override false for
  reportable HPO.
- Held-out test evaluation inside trials
  (`INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False`).
- Using collapsed-4 metrics or multi-objective Pareto review to select,
  inspect, or repeat the held-out test.
- Letting `EPOCHS <= 3` reach Stage 4/5 (the short-training guard will block
  this; do not override).
- Reportable HPO with `OPTUNA_INTENSITY != "custom"` or blank/nonpersistent
  `OPTUNA_STORAGE`.
- Reportable HPO with `OPTUNA_PARALLEL_WORKERS > 1` and blank/nonpersistent
  `OPTUNA_STORAGE`, or with `OPTUNA_TPE_CONSTANT_LIAR = False`.
- `ALLOW_MISSING_ESM_EMBEDDINGS = True` for ESM or fusion stages.

Batch-size policy for serious stages:

- Use `4` only for smoke/debug runs or when a memory failure forces it.
- Use `8,16` as the default serious validation-only Optuna batch-size search
  space. This keeps the current validated `batch_size=8` anchor in scope while
  testing whether `16` improves minority-class stability and GPU utilization.
- Stage 5A may add `32` as an exploratory Only-GVP value because it does not
  carry the ESM/fusion memory footprint. Watch every `batch_size=32` trial for
  CUDA OOM and for degraded minority-class recall from fewer optimizer updates
  per epoch.
- Do not include `32` in fusion stages unless a separate memory/quality ablation
  explicitly justifies it.

Recommended G4 budgets (canonical):

| Stage | `OPTUNA_TARGET_COMPLETE_TRIALS` | `MAX_EPOCHS_PER_TRIAL` | `OPTUNA_N_STARTUP_TRIALS` |
| --- | --- | --- | --- |
| Stage 3 (debug) | 4 | 3 | 4 |
| Stage 4 (medium per family) | 64 | 35 | 20 |
| Stage 5A (Only-GVP) | 200 | 50 | 40 |
| Stage 5B (Only-ESM) | 120 | 50 | 30 |
| Stage 5C (GVP+late) | 200 | 50 | 40 |
| Stage 5D (GVP+node-late) | 200 | 50 | 40 |
| Stage 5E (GVP+hybrid) | 200 | 50 | 40 |
| Stage 5F (GVP+cross-attn) | 120 | 50 | 30 |

Serious learning-rate ranges:

| Stage | `OPTUNA_LEARNING_RATE_RANGE` |
| --- | --- |
| Stage 3 | `1e-5,3e-4` |
| Stage 4 | `1e-5,3e-4` |
| Stage 5A | `5e-6,3e-4` |
| Stage 5B | `5e-6,2e-4` |
| Stage 5C | `5e-6,2e-4` |
| Stage 5D | `5e-6,2e-4` |
| Stage 5E | `5e-6,1.5e-4` |
| Stage 5F | `5e-6,1e-4` |

Serious LR schedule choices:

| Stage | `LR_SCHEDULES_CSV` |
| --- | --- |
| Stage 5A | `fixed,cosine` |
| Stage 5C | `fixed,cosine` |
| Stage 5D | `fixed,cosine` |
| Stage 5E | `fixed,cosine` |

Do not add `step` to Optuna LR-schedule search until `lr_step_size` and
`lr_decay_gamma` are also part of the Optuna search space. The notebook exposes
manual step-decay controls, but Optuna currently searches only `fixed` and
`cosine`. TODO: warmup is not currently a training CLI/config option, so do not
add warmup choices until a real warmup implementation exists.

Serious class-weight and loss search ranges:

| Stage | Class weighting | Losses | Label smoothing | Sampling balance |
| --- | --- | --- | --- | --- |
| Stage 5A | `none,inverse_frequency,inverse_sqrt_frequency,effective_number` | `cross_entropy,focal` | `0.0,0.03,0.05,0.1` | `False,True` |
| Stage 5B | `none,inverse_frequency,inverse_sqrt_frequency,effective_number` | `cross_entropy` | `0.0,0.03,0.05,0.1` | `False,True` |
| Stage 5C | `inverse_frequency,inverse_sqrt_frequency,effective_number` | `cross_entropy` | `0.0,0.03,0.05,0.1` | `False,True` |
| Stage 5D | `inverse_frequency,inverse_sqrt_frequency,effective_number` | `cross_entropy` | `0.0,0.03,0.05,0.1` | `False,True` |
| Stage 5E | `inverse_frequency,inverse_sqrt_frequency,effective_number` | `cross_entropy` | `0.0,0.03,0.05,0.1` | `False,True` |
| Stage 5F | `inverse_frequency,inverse_sqrt_frequency,effective_number` | `cross_entropy` | `0.0,0.03,0.05,0.1` | `False` |

Serious capacity/search-space policy:

- Stage 5A searches narrowed-from-the-top GVP capacity, edge radius, head
  dropout, training-only regularization/augmentation, class weighting, focal
  loss, and batch size inside `MODEL_PRESET = "Only-GVP"`.
- Stage 5B searches ESM-only classifier capacity and metal imbalance settings
  inside `MODEL_PRESET = "Only-ESM"`.
- Stage 5C/5D search narrowed-from-the-top GVP capacity, edge radius, ESM fusion
  dimension, head dropout, ESM graph encoder dropout, training-only
  regularization/augmentation, class weighting, and batch size inside their
  single fusion preset.
- Stage 5E additionally searches early-ESM bottleneck and dropout.
- Stage 5F keeps attention narrow: one layer, limited heads/dropout, no
  bidirectionality in the first serious search.
- Common training-only graph augmentation defaults remain off. Canonical Stage
  4, 5A, 5C, 5D, 5E, and 5F blocks explicitly sample position noise and
  outer-residue dropout inside one model-family study. Augmentation never runs
  for validation or held-out test inference.

## Conservative First-Pass Anti-Overfitting GVP Profile

Use this profile as a recommended conservative starting point for GVP-based
metal-focused runs when the goal is to reduce overfitting risk before a wider
second-stage expansion. It is not a universal optimum, not held-out-test
selected evidence, and not a replacement for Stage 6 confirmation.

The current GVP input is already information-rich. Node scalar inputs include
amino-acid chemistry, hydrophobicity, donor/acceptor/aromatic/acidic/basic
flags, shell role, distance/RBF-derived terms, and burial/SASA/electrostatics/
PROPKA-like features where available. The graph also has explicit residue
vector channels plus edge scalar, RING, and radius features. Because the
dataset is modest, first-stage capacity should stay conservative.

Main capacity knobs:

- `HIDDEN_S_VALUES_CSV`
- `HIDDEN_V_VALUES_CSV`
- `EDGE_HIDDEN_VALUES_CSV`
- `GVP_LAYERS_VALUES_CSV`
- `EDGE_RADIUS_VALUES_CSV`
- `ESM_FUSION_DIM_VALUES_CSV`
- `EARLY_ESM_DIM_VALUES_CSV`
- `HEAD_MLP_LAYERS_VALUES_CSV`

Optional range overrides for a deliberately labeled range-search experiment:

- `OPTUNA_WEIGHT_DECAY_RANGE`
- `OPTUNA_HIDDEN_S_RANGE`, `OPTUNA_HIDDEN_V_RANGE`, `OPTUNA_EDGE_HIDDEN_RANGE`
- `OPTUNA_GVP_LAYERS_RANGE`, `OPTUNA_HEAD_MLP_LAYERS_RANGE`
- `OPTUNA_EDGE_RADIUS_RANGE`, `OPTUNA_CLASSIFIER_POOL_DISTANCE_CUTOFF_RANGE`
- `OPTUNA_HEAD_MLP_DROPOUT_RANGE`, `OPTUNA_ESM_GRAPH_ENCODER_DROPOUT_RANGE`
- `OPTUNA_POSITION_NOISE_STD_RANGE`, `OPTUNA_SECOND_SHELL_DROPOUT_RANGE`,
  `OPTUNA_OUTER_RESIDUE_DROPOUT_RANGE`
- `OPTUNA_ESM_FUSION_DIM_RANGE`, `OPTUNA_EARLY_ESM_DIM_RANGE`
- `OPTUNA_CROSS_ATTENTION_LAYERS_RANGE`, `OPTUNA_EARLY_ESM_DROPOUT_RANGE`,
  `OPTUNA_CROSS_ATTENTION_DROPOUT_RANGE`
- `OPTUNA_METAL_LABEL_SMOOTHING_RANGE`, `OPTUNA_METAL_FOCAL_GAMMA_RANGE`
- `OPTUNA_METAL_COLLAPSED_LOSS_WEIGHT_RANGE`,
  `OPTUNA_METAL_LOSS_WEIGHT_RANGE`, `OPTUNA_EC_LOSS_WEIGHT_RANGE`

Leave these blank for the canonical CSV-based stage blocks below unless the run
is explicitly named and documented as a validation-only range-search variant.

Notebook profile:

```python
RING_EDGE_MODE = "with_ring"
METAL_NODE_MODE = "per_metal"
STRUCTURAL_READOUT_SCOPE = "auto"

CLASSIFIER_POOL_DISTANCE_CUTOFF_VALUES_CSV = "0.0"
HIDDEN_S_VALUES_CSV = "128"
HIDDEN_V_VALUES_CSV = "8,16"
EDGE_HIDDEN_VALUES_CSV = "64"
GVP_LAYERS_VALUES_CSV = "2,3"
HEAD_MLP_LAYERS_VALUES_CSV = "1,2"
EDGE_RADIUS_VALUES_CSV = "6,8"
ESM_FUSION_DIM_VALUES_CSV = "64,128"
EARLY_ESM_DIM_VALUES_CSV = "32,48"

HEAD_MLP_DROPOUT_VALUES_CSV = "0.2"
ESM_GRAPH_ENCODER_DROPOUT_VALUES_CSV = "0.1"
EARLY_ESM_DROPOUT_VALUES_CSV = "0.05"  # 0.1 is also acceptable for the first pass.
CROSS_ATTENTION_DROPOUT_VALUES_CSV = "0.1"

POSITION_NOISE_STDS_CSV = "0.0,0.03,0.05"
SECOND_SHELL_DROPOUTS_CSV = "0.0"
OUTER_RESIDUE_DROPOUTS_CSV = "0.0,0.1"
```

Rationale:

- `hidden_s=128`, `hidden_v=8/16`, `edge_hidden=64`, and 2-3 GVP layers are
  appropriate low-capacity starting values for roughly one thousand samples.
- `edge_radius=6/8` keeps the radius graph local; radius `10` or higher is a
  second-stage option.
- `esm_fusion_dim=256`, `hidden_s>=192`, `hidden_v>=24`,
  `edge_hidden>=128`, and `gvp_layers>=4` are higher-capacity options and
  should not be first-stage anti-overfitting defaults.
- Position noise and residue dropout are training-only robustness tools. Keep
  coordinate noise mild for metal-site geometry. If using AlphaFold structures,
  mild training-only coordinate noise can be considered, but validation and
  held-out test graphs must remain unchanged.
- Do not claim that coordinate noise or residue dropout improves performance
  without validation evidence.

Budget tiers:

| Profile | `OPTUNA_TARGET_COMPLETE_TRIALS` | `MAX_EPOCHS_PER_TRIAL` / `OPTUNA_SEARCH_HPO_TRIAL_EPOCHS` | `OPTUNA_N_STARTUP_TRIALS` |
| --- | --- | --- | --- |
| Conservative first pass | 64 or 80 | 35-40 | 15-20 |
| Strong controlled | 100 | 50 | 20 |
| Extended serious | Use the canonical Stage 5 table above | Use the canonical Stage 5 table above | Use the canonical Stage 5 table above |

Two hundred complete trials is an extended serious search, not a simple
first-pass anti-overfitting search. Two-hundred-trial studies are acceptable
only when followed by predeclared Stage 6 top-K grouped-fold/seed
confirmation. Do not interpret one validation split or the best single Optuna
trial as conclusive.

This profile applies broadly to GVP-based metal-focused DeepMzyme runs. It is
not specific to `TASK = "joint"`, `METAL_LABEL_SCHEME = "five_class"`,
`MODEL_PRESET = "GVP + hybrid fusion"`, or
`SELECTION_METRIC = "val_metal_balanced_acc"`. If `TASK = "joint"` and
`SELECTION_METRIC = "val_metal_balanced_acc"`, model selection is primarily
metal-optimized and the EC branch is auxiliary. If the goal is EC prediction,
use an EC validation metric instead.

Feature-omission ablations use notebook `OMIT_NODE_FEATURE_SETS`; the CLI flag
is `--omit-node-features`. Do not invent additional notebook omission
variables.

## Optional Objective Experiments

These objective variants are experimental validation-only tools. They are not
defaults, must not be used in Stage 2 baselines, and must not change the
one-shot held-out test policy.

### Optional collapsed-4 auxiliary metal loss

`METAL_COLLAPSED_LOSS_WEIGHTS_CSV` maps to the CLI flag
`--metal-collapsed-loss-weight`; in single mode the first CSV value is used.
For the retained six-class recipe, `0.0` preserves the
six-class objective exactly. When enabled with cross-entropy metal
loss, the training objective is:

```text
L_total = (1 - alpha) * CE_6class + alpha * CE_4class
```

The collapsed view is deterministic: `Mn`, `Cu`, `Zn`, and `Class VIII`, where
`Class VIII = Fe + Co + Ni`. The collapsed logits are computed by log-sum-exp
marginalization from the six-class logits. This is a separate six-class
objective experiment. It is not direct four-class training and cannot replace
the intended primary four-class baseline. Within a six-class experiment,
collapsed-four metrics are supplemental and must not hide Fe/Co/Ni failures.

First-use rule: test this only against a same-scheme six-class validation
baseline before using it broadly. For a separately labeled Stage 5A-only
six-class validation experiment, use:

```python
METAL_COLLAPSED_LOSS_WEIGHTS_CSV = "0.0,0.3,0.5"
METAL_LOSS_FUNCTIONS_CSV = "cross_entropy"
```

The required six-class-trained/collapsed-four challenger uses the standard
six-class objective with this weight at `0.0`; deterministic collapsed-four
evaluation does not require an auxiliary collapsed loss. Treat a nonzero weight
as a third target-objective experiment. Do not add that search axis to direct
four-class training or automatically to Stage 5B-5F. Add it only after a
same-scheme Stage 5A validation comparison and Stage 6 confirmation justify it.

### Optional five-class metal target scheme

The primary reporting endpoint is four-class. Its direct arm uses `four_class`,
and its required standard six-class challenger uses the retained label scheme
above pending TECH-010 reconciliation. `five_class` is a separate alternative
target.

For an explicitly labeled validation-only comparison, use:

```python
METAL_LABEL_SCHEME = "five_class"
```

This changes the training target to five classes: `Mn`, `Cu`, `Zn`, `Fe`, and a
grouped Co/Ni class. It is not a display toggle and is not the same as
`METAL_REPORT_VIEW = "collapsed4"`. When using it, create a new
`RUN_BATCH_ID`, `SUMMARY_BASENAME`, `OPTUNA_STUDY_NAME`, and persistent storage
file so five-class evidence cannot mix with six-class evidence. The notebook
auto-derives the run-name prefix from `RUN_BATCH_ID`. Keep
`SELECTION_METRIC = "val_metal_balanced_acc"`; that metric is then balanced
accuracy over the active five-class target.

Do not use five-class validation numbers to replace or rank six-class anchors
without a separately documented comparison goal. Stage 6, Stage 6B, and Stage 7
source runs must all use the same `METAL_LABEL_SCHEME`.

#### Five-class joint hybrid metal-target overlay

Use this overlay for an explicitly labeled pocket-level validation diagnostic
of the joint-task GVP + hybrid fusion family while selecting checkpoints by
metal balanced accuracy. It keeps the five-class target separate from six-class
evidence and applies additional metal-loss multipliers to Fe and Mn on top of
the selected training-split metal class-weight mode.

This retained advanced diagnostic is not the first metal-EC relationship
experiment. The first EC-primary auxiliary comparison is limited to Only-GVP,
Only-ESM, or GVP + graph-level late fusion with a matched EC-only control.

Notebook configuration block:

```python
TASK = "joint"
METAL_LABEL_SCHEME = "five_class"
RUN_MODE = "single"
RECOMMENDED_RUN_SET = "custom"
MODEL_PRESET = "GVP + hybrid fusion"
RUN_BATCH_ID = "joint_fiveclass_hybrid_metal_target_fe1p7_mn1p5_splitpocket_single"
SUMMARY_BASENAME = ""  # auto from provenance

EPOCHS = 50
VAL_FRACTION = 0.15
SPLIT_BY = "pocket_id"
SELECTION_METRIC = "val_metal_balanced_acc"
RING_EDGE_MODE = "with_ring"

ESM_EMBEDDINGS_DIR = ""  # set to your embeddings folder when available
ALLOW_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_ESM_EMBEDDINGS = True

LEARNING_RATES_CSV = "3.705631497756492e-05"
WEIGHT_DECAYS_CSV = "3e-07"
BATCH_SIZES_CSV = "12"
SEEDS_CSV = "42"
LR_SCHEDULES_CSV = "fixed"

HIDDEN_S_VALUES_CSV = "320"
HIDDEN_V_VALUES_CSV = "16"
EDGE_HIDDEN_VALUES_CSV = "192"
GVP_LAYERS_VALUES_CSV = "4"
HEAD_MLP_LAYERS_VALUES_CSV = "2"
EDGE_RADIUS_VALUES_CSV = "7.0"
ESM_FUSION_DIM_VALUES_CSV = "256"
EARLY_ESM_DIM_VALUES_CSV = "48"
EARLY_ESM_DROPOUT_VALUES_CSV = "0.05"

METAL_CLASS_WEIGHT_MODES_CSV = "effective_number"
BALANCE_METAL_SITE_SYMBOLS_CSV = "False"
METAL_LOSS_FUNCTIONS_CSV = "cross_entropy"
METAL_LABEL_SMOOTHING_VALUES_CSV = "0.0"
METAL_COLLAPSED_LOSS_WEIGHTS_CSV = "0.0"
MN_LOSS_MULTIPLIER = 1.5
FE_LOSS_MULTIPLIER = 1.7
CU_LOSS_MULTIPLIER = 1.0
ZN_LOSS_MULTIPLIER = 1.0
CO_LOSS_MULTIPLIER = 1.0
NI_LOSS_MULTIPLIER = 1.0
CLASS_VIII_LOSS_MULTIPLIER = 1.0

METAL_LOSS_WEIGHT_VALUES_CSV = "2.0"
EC_LOSS_WEIGHT_VALUES_CSV = "0.25"
EC_LABEL_DEPTHS_CSV = "1"
EC_CONTRASTIVE_WEIGHTS_CSV = "0.0"
EC_GROUP_WEIGHTING = "structure_id"

INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

Expected outputs/files:

- One validation-only run directory under `<RUNS_DIR>/<RUN_BATCH_ID>/...`.
- `best_model_checkpoint.pt`, `metrics_history.csv`, `run_config.json`,
  `run_metadata.json`, and split/validation artifacts in the run directory.
- Notebook-generated `active_run_config.json` and `active_run_config.md` before
  launch.
- Summary artifacts using `SUMMARY_BASENAME` after the notebook summary cells.

Decision gate:

- Treat this as a one-off validation-only pocket-level diagnostic, not a
  replacement anchor.
- Proceed to Stage 6 only if the run completes without held-out-test output,
  uses `METAL_LABEL_SCHEME = "five_class"`, `TASK = "joint"`,
  `MODEL_PRESET = "GVP + hybrid fusion"`,
  `SELECTION_METRIC = "val_metal_balanced_acc"`, `VAL_FRACTION = 0.15`,
  `SPLIT_BY = "pocket_id"`, and the saved configs show
  `MN_LOSS_MULTIPLIER = 1.5` and `FE_LOSS_MULTIPLIER = 1.7`.
- Do not compare this pocket-level validation value directly against grouped
  `pdbid` validation anchors, because `SPLIT_BY = "pocket_id"` is a different
  and less conservative split policy.
- Do not compare the five-class validation value directly against six-class
  anchors. Any promotion requires same-scheme Stage 6 grouped-fold
  confirmation with shared folds, paired bootstrap confidence intervals, and
  rare-class recall protection.

### Optional multi-objective Optuna

`OPTUNA_MULTIOBJECTIVE = True` creates a validation-only multi-objective Optuna
study with objectives:

- maximize `val_metal_balanced_acc`
- maximize `val_metal_min_recall`

The second objective is minimum recall across active metal-scheme validation
classes with support > 0. For the direct four-class arm, this means Mn, Cu, Zn,
and Class VIII. For an explicitly labeled `five_class` run it means Mn, Cu, Zn,
Fe, and grouped Co/Ni; for the required standard `six_class` challenger it means
all six classes. In six-class runs, do not substitute
`val_metal_collapsed4_min_recall`, because it can hide separate Fe/Co/Ni
failures. Collapsed-four minimum recall is supplemental context for that
scheme.

Multi-objective HPO writes Pareto review files:

- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/pareto_front.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/pareto_candidates.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/pareto_candidates_ranked_for_review.csv`

The ranked file is a review convenience only. It does not replace Stage 6
grouped-fold confirmation. If pruning is incompatible with the active Optuna
multi-objective study, the notebook disables pruning and prints a warning.

Recommended Stage 5A overlay:

```python
OPTUNA_MULTIOBJECTIVE = True
OPTUNA_SELECTION_METRIC = "val_metal_balanced_acc"
OPTUNA_USE_PRUNING = False
OPTUNA_PRUNER_TYPE = "none"
```

## Canonical G4 Metal Training Route

Use this route when starting a clean, serious metal-classification campaign in
`notebooks/DeepMzyme_training_colab.ipynb`.

### Required order

Recommended linear G4 route:

Stage 0 -> Stage 1 -> Stage 2A -> Stage 2B if ESM is ready -> Stage 3 ->
Stage 5A -> Stage 6 -> Stage 5B/5C/5D/5E/5F only if their gates pass ->
Stage 6B -> Stage 7.

Interpretation:

1. Stage 0: environment/data readiness.
2. Stage 1: 1-epoch smoke.
3. Stage 2A: Only-GVP validation anchor.
4. Stage 2B: baseline family comparison, only after ESM coverage is valid.
5. Stage 3: Optuna plumbing debug.
6. Stage 5A: serious Only-GVP HPO.
7. Stage 6: top-K grouped-fold confirmation for the Only-GVP HPO candidates.
8. Stage 5B and Stage 5C: run only if ESM coverage and baseline gates pass.
9. Stage 5D, Stage 5E, and Stage 5F: run only after the advanced-fusion gate
   below passes.
10. Stage 6 again for every HPO family that may become the final selected
    configuration.
11. Stage 6B: apply promotion gates and train/refit the single selected
    configuration on the full non-test training set.
12. Stage 7: one-shot held-out test for the frozen Stage 6B final-refit run.

### Required metal controlled-comparison matrix

The full primary metal research mission is broader than selecting one best
pipeline. After TECH-010 supplies reconciled paired target recipes and direct
four-class architecture recipes, complete these matched comparisons before
making the corresponding publication claims:

| Question | Playbook coverage | Completion requirement |
|---|---|---|
| Direct four-class training vs six-class training with collapsed-four evaluation | The bounded pilot supplies native-selected matched baseline arms and an additional five-class challenger; later-stage paired blocks remain open | Run both required arms for Only-GVP, Only-ESM, and graph-level late fusion with separate studies, matched data/folds/seeds/features/budgets, common four-class validation metrics, paired CIs, and four-class recall protection; retain native six-class metrics for the six-class arm |
| Only-ESM (using ESMC) vs Only-GVP vs combined GVP+ESMC | Stages 2A/2B establish the simple baselines; Stages 5A/5B/5C tune the serious candidates | Confirm every eligible family on the same Stage 6 folds/seeds and compare with paired CIs and rare-class recall protection |
| Early vs late vs hybrid ESMC fusion | The bounded pilot screens early/late and conditionally hybrid; Stage 5C covers serious late HPO and Stage 5E serious hybrid HPO | A dedicated serious early-HPO recipe remains open; confirm early, late, and hybrid candidates on the shared Stage 6 grid before claiming a fusion-position advantage |
| GVP with vs without RING | Stage 2A supplies the RING-enabled Only-GVP anchor and Stage 5G supplies its radius-only counterpart | Keep all non-edge settings matched and confirm any claimed RING benefit on shared validation units; if the final combined model uses RING, also ablate RING in that same combined family |

Do not fill the remaining serious early-HPO gap by copying another stage's
budget. Add its reviewable executable block here first. Do not infer a RING effect by comparing
the historical Hybrid+RING maximum with a separately tuned no-RING model.
Historical six-class candidates may motivate the search but cannot complete the
paired target-formulation comparison without matched direct-four arms. The
other three rows remain direct-four architecture comparisons. Advanced
candidates in this matrix also stay outside the first EC-primary
auxiliary-learning experiment.

Stage 4 is optional on a G4 GPU and mainly for sanity HPO, search-space
debugging at useful scale, or limited-compute campaigns. For a serious fresh
G4 search, Stage 5 is preferred after Stage 3 passes.

### Advanced fusion gate

Do not launch Stage 5D, Stage 5E, or Stage 5F until Stage 5C has produced a
Stage 6 grouped-fold candidate that clears the paired validation-improvement
threshold defined in this playbook's Stage 5C decision gate.


If this gate is not passed, stop advanced fusion escalation and continue with
the best validated simpler family.

### Final-selection rule


The final selected model must come from Stage 6 grouped-fold validation plus
Stage 6B promotion gates, not from a single Optuna trial. Stage 6B ranks by mean
`val_metal_balanced_acc`, promotes only when the paired bootstrap CI and
rare-recall gates pass, then uses configured tie-breakers such as standard
deviation, worst fold, and model simplicity.

Stage 6 selects the configuration; Stage 6B creates the final test source. Keep
`MODEL_PRESET`, model hyperparameters, feature policy, `METAL_LABEL_SCHEME`,
training budget, fixed final-refit seed, checkpoint rule, calibration rule, and
any ensemble rule fixed from validation evidence before the Stage 6B final
refit starts. Do not choose any of these from held-out test results.

The held-out test is used only once, in Stage 7, after this validation-based
selection and the Stage 6B final-refit run are frozen.

## Optuna Study Naming And Storage

Study naming: `metal_<preset_slug>_<size>_<purpose>`, for example
`metal_only_gvp_200_capacity` or `metal_late_fusion_200_controlled`. Always use
lowercase, underscore-separated names.

Storage path template:
`sqlite:////content/drive/MyDrive/DeepMzyme/optuna/<study_name>.db`. Use one
file per study. Never share a study DB across different `MODEL_PRESET` values.

Resumption rule: re-running the notebook with the same `OPTUNA_STUDY_NAME` and
storage URL resumes the persistent study and launches only the remaining trials
needed to reach `OPTUNA_TARGET_COMPLETE_TRIALS` completed trials. `N_OPTUNA_TRIALS`
is still accepted as a backward-compatible alias in older snippets. To start
fresh, change the study name; do not delete the `.db` unless you mean to discard
history.

Resume policy for reportable HPO:

- Resume only into the same `MODEL_PRESET`, task, split policy, selection
  metric, metal label scheme, search space, and storage URL.
- If any of those values changed, use a new `OPTUNA_STUDY_NAME` and new SQLite
  file.
- If a study was interrupted, resume until the requested number of `COMPLETE`
  trials in the stage gate is reached. Failed/pruned/incomplete trials do not
  count toward the required completed-trial count, so they may make the stored
  total trial count exceed `OPTUNA_TARGET_COMPLETE_TRIALS` while the completed-trial count
  reaches the target.
- The notebook records compatibility metadata and a search-space hash in each
  study. Reusing a persistent study with incompatible metadata stops with a
  clear error unless `OPTUNA_ALLOW_INCOMPATIBLE_STUDY_REUSE = True`.

If a run uses `Only-ESM` or any GVP + ESM fusion preset, set
`ESM_EMBEDDINGS_DIR` to the embeddings folder or set
`PREPARE_MISSING_ESM_EMBEDDINGS = True` deliberately. Do not use
`ALLOW_MISSING_ESM_EMBEDDINGS = True` for reportable runs.

## Exact Run-Configuration Artifacts

Every launched training run should produce a machine-readable record of the
configuration that actually ran. The current training code writes:

- `<run_dir>/run_config.json`
- `<run_dir>/run_metadata.json`
- `<run_dir>/active_run_config.json`
- `<run_dir>/active_run_config.md`

These artifacts are the authoritative record for completed training and
validation runs. They include the resolved config, selection metric, selected
checkpoint metadata, history, metal label scheme, split identity, and embedded
test information when test evaluation was requested. `active_run_config.json` and
`active_run_config.md` are written by the notebook before launch from the live
notebook variables and command configuration; they are especially useful for
failed or pruned subprocess trials that may not reach `run_config.json`.

For serious validation and Stage 7 final-test runs, verify that the run record
also captures the dataset bundle filename/checksum when a bundle is used, the
git commit, and key library versions. If a current artifact lacks one of these
fields, note the gap in the run summary or `EXPERIMENT_STATUS.md` rather than
inferring it later.

For Optuna studies, the notebook also writes
`<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/optuna_study_metadata.json` plus
study-level `active_run_config.json` / `active_run_config.md`.
When `OPTUNA_MULTIOBJECTIVE = True`, it also writes `pareto_front.csv`,
`pareto_candidates.csv`, and `pareto_candidates_ranked_for_review.csv`. These
Pareto files are validation-review artifacts only and do not authorize held-out
test evaluation without Stage 6 confirmation.

## Stage 0 - Environment/Data Readiness

Runtime prerequisite: complete the browser or CLI procedure in
[`COLAB_GPU_RUNBOOK.md`](COLAB_GPU_RUNBOOK.md), including the PyTorch/CUDA
architecture preflight. Do not install `src/requirements.txt` unchanged in
Colab; install only `requirements/colab-overlay.txt`. For a CLI-created VM, use `colab url` when interactive Drive
authorization is required so the browser and CLI remain attached to the same
kernel. The exact Stage 0 configuration below is unchanged by that operational
setup.

Purpose: confirm Drive is mounted, the bundle is present, RING/ESM/external
features coverage is acceptable, and `RUNS_DIR` resolves under Drive.

When to use it: first cell pass in a fresh Colab runtime, or after switching
data bundles.

Configuration block (paste into Main configuration cell):

```python
TASK = "metal"
RUN_MODE = "single"
RECOMMENDED_RUN_SET = "only_gvp_smoke"
MODEL_PRESET = "Only-GVP"
RUN_BATCH_ID = "stage0_environment_check"
SUMMARY_BASENAME = ""  # auto from provenance

EPOCHS = 1
BATCH_SIZES_CSV = "4"
LEARNING_RATES_CSV = "3e-5"
WEIGHT_DECAYS_CSV = "1e-4"
SEEDS_CSV = "42"

DATASET_NAME = "train_and_test_sets_structures_exact_pinmymetal"
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_metal_balanced_acc"

RING_EDGE_MODE = "with_ring"
REQUIRE_RING_EDGES = False
PREPARE_MISSING_RING_EDGES = True
RING_FEATURES_DIR = ""
RING_EXE_PATH = "DeepMzyme_Data/ring-4.0/out/bin/ring"
ESM_EMBEDDINGS_DIR = ""
ALLOW_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_ESM_EMBEDDINGS = False
ALLOW_MISSING_EXTERNAL_FEATURES = False
PREPARE_MISSING_EXTERNAL_FEATURES = True

INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
```

In the dedicated **Main planned training launch switch** cell:

```python
LAUNCH_PLANNED_MAIN_TRAINING_RUNS = False   # planning/helper loading only; do NOT train at Stage 0
```

Success criteria for Stage 0:

- `RUNS_DIR` resolves under `<DRIVE_ROOT>/notebook_outputs/runs`.
- Planning cell prints RING coverage >= 95% (or generation will run).
- Planning cell prints external-features coverage = 100%.
- No model preset mismatch warnings.

Expected outputs/files:

- `<RUNS_DIR>/<SUMMARY_BASENAME>_planned_runs.csv`
- `<RUNS_DIR>/<SUMMARY_BASENAME>_planned_run_dictionary.json`
- `<RUNS_DIR>/<SUMMARY_BASENAME>_metal_weight_diagnostics.csv`, when metal
  diagnostics are generated
- Printed resolved configuration, feature coverage, split diagnostics, and
  shell-safe command preview
- No training run directory and no `test_report.json`

Exact configuration record:

- Planning-only stage: use the planned-run CSV/dictionary and printed
  configuration summary.
- `active_run_config.json` / `active_run_config.md` are generated from the live
  notebook configuration before launch.

### Decision gate after Stage 0

Proceed to Stage 1 only if:

- All four Stage 0 success criteria are met.
- The planned-run table contains exactly one planned Stage 0 row and zero
  launched training runs.
- The expected planned-run CSV and dictionary exist.
- No held-out test files were created.
- `val_metal_balanced_acc` is the selection metric on the planned run.
- Diagnostics report every active metal-scheme class present in both train and validation splits.
- Rare-class protection passes at the split level: no active metal-scheme class is missing
  from train or validation diagnostics.

If gate fails: fix paths, Drive mounting, bundle selection, RING coverage, or
external-feature coverage before any training.

## Stage 1 - 1-Epoch Smoke

Purpose: verify Colab setup, data paths, CSV detection, graph construction, and
the training command path.

When to use it: first run in a fresh Colab/runtime, after changing the notebook,
or after changing data bundle paths.

Expected scale/runtime: smoke/debug, minutes.

Notebook configuration block:

```python
TASK = "metal"
RUN_MODE = "single"
RECOMMENDED_RUN_SET = "only_gvp_smoke"
MODEL_PRESET = "Only-GVP"
RUN_BATCH_ID = "metal_smoke_readiness"
SUMMARY_BASENAME = ""  # auto from provenance

EPOCHS = 1
BATCH_SIZES_CSV = "4"
LEARNING_RATES_CSV = "3e-5"
WEIGHT_DECAYS_CSV = "1e-4"
SEEDS_CSV = "42"
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_metal_balanced_acc"

RING_EDGE_MODE = "with_ring"
ESM_EMBEDDINGS_DIR = ""
ALLOW_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_ESM_EMBEDDINGS = False
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
MAX_CONFIGURATION_RUNS = 1
```

In the dedicated **Main planned training launch switch** cell, then run
**Optional training execution**:

```python
LAUNCH_PLANNED_MAIN_TRAINING_RUNS = True
```

Expected outputs/files:

- `<RUNS_DIR>/<SUMMARY_BASENAME>_planned_runs.csv`
- `<RUNS_DIR>/<SUMMARY_BASENAME>_planned_run_dictionary.json`
- One run directory under `<RUNS_DIR>/`
- `run_metadata.json`, `run_config.json`, `split_diagnostics.json`
- `dataset_summary.json`, `prepare_status.json`
- No `test_report.json`

Exact configuration record:

- `<run_dir>/run_config.json` and `<run_dir>/run_metadata.json`.
- `active_run_config.json` / `active_run_config.md` are generated from the live
  notebook configuration before launch.

Success criteria:

- Planning prints one runnable Only-GVP command.
- Training completes without missing-path, split, feature, or CLI errors.
- Train and validation metal diagnostics are printed.
- No held-out test report is produced.

### Decision gate after Stage 1

Proceed to Stage 2A only if:

- The Stage 1 success criteria are met.
- Exactly one Stage 1 validation-only run directory was launched and completed.
- The expected planned files and run-level JSON files exist.
- No held-out test files were created.
- `val_metal_balanced_acc` is the selection metric on all completed runs.
- Diagnostics report every active metal-scheme class present in both train and validation splits.
- Rare-class protection passes at the split level: no active metal-scheme class is missing
  from train or validation diagnostics. Do not use the 1-epoch recall values as
  model-quality evidence.

If gate fails: return to Stage 0 and fix paths, bundle setup, RING executable
configuration, structure parsing, ESM coverage, or feature availability before
running real comparisons. Ignore the 1-epoch metric as model-quality evidence.

## Stage 2 - Baseline Validation

Purpose: establish clean validation baselines before adding complex fusion or
large HPO.

When to use it: after smoke passes and before Optuna or advanced fusion. If ESM
embeddings are not ready, run the Only-GVP block first. Once ESM embeddings are
ready, run the ESM-ready baseline block.

Expected scale/runtime: medium validation run, hours. Runtime depends on GPU,
ESM coverage, and whether embeddings must be prepared.

### Stage 2A - Only-GVP Validation Anchor

Notebook configuration block:

```python
TASK = "metal"
RUN_MODE = "manual_configurations"
RECOMMENDED_RUN_SET = "only_gvp_broad_comparison"
MODEL_PRESET = "Only-GVP"
RUN_BATCH_ID = "metal_only_gvp_baseline_lr_seed"
SUMMARY_BASENAME = ""  # auto from provenance

EPOCHS = 50
BATCH_SIZES_CSV = "8"
LEARNING_RATES_CSV = "3e-5,1e-4,3e-4"
WEIGHT_DECAYS_CSV = "1e-4"
SEEDS_CSV = "42,123,2026,7,2718"
MAX_CONFIGURATION_RUNS = 15

HIDDEN_S_VALUES_CSV = "128"
HIDDEN_V_VALUES_CSV = "16"
EDGE_HIDDEN_VALUES_CSV = "64"
GVP_LAYERS_VALUES_CSV = "4"
HEAD_MLP_LAYERS_VALUES_CSV = "2"
EDGE_RADIUS_VALUES_CSV = "8.0"

RING_EDGE_MODE = "with_ring"
ESM_EMBEDDINGS_DIR = ""
ALLOW_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_ESM_EMBEDDINGS = False
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

In the dedicated **Main planned training launch switch** cell, then run
**Optional training execution**:

```python
LAUNCH_PLANNED_MAIN_TRAINING_RUNS = True
```

Expected outputs/files:

- `<RUNS_DIR>/<SUMMARY_BASENAME>_planned_runs.csv`
- `<RUNS_DIR>/<SUMMARY_BASENAME>_planned_run_dictionary.json`
- Fifteen completed validation-only run directories under `<RUNS_DIR>/`
- Each completed run directory contains `run_config.json`,
  `run_metadata.json`, `split_diagnostics.json`, `dataset_summary.json`, and
  `prepare_status.json`
- `<RUNS_DIR>/<SUMMARY_BASENAME>.csv`
- `<RUNS_DIR>/<SUMMARY_BASENAME>_completed_only.csv`
- `<RUNS_DIR>/<SUMMARY_BASENAME>.png`, when plotting succeeds
- No `test_report.json`

Exact configuration record:

- Per run: `<run_dir>/run_config.json` and `<run_dir>/run_metadata.json`.
- Batch plan: planned-run CSV/dictionary.
- `active_run_config.json` / `active_run_config.md` are generated from the live
  notebook configuration before launch.

### Decision gate after Stage 2A

Proceed to Stage 2B or Stage 4 only if:

- The Only-GVP validation baseline completes all 15 planned validation-only
  runs.
- The expected planned files, summary files, and run-level JSON files exist.
- No held-out test files were created.
- `val_metal_balanced_acc` is the selection metric on all completed runs.
- Diagnostics report every active metal-scheme class present in both train and validation splits.
- Rare-class recall protection passes: `val_metal_min_recall` and per-class
  recall are available in the run artifacts, and no candidate is promoted if a
  metal class has zero recall across the completed validation runs.
- Stage 2A anchor reliability is sufficient: the standard G4 block uses five
  seeds, `42,123,2026,7,2718`. If a compute-constrained run uses fewer seeds,
  mark the Stage 2A anchor as provisional and record the reason in
  `EXPERIMENT_STATUS.md`.
- Seed variance is acceptable: if seed standard deviation or high-low spread
  suggests `val_metal_balanced_acc` variance above 0.04, rerun with the
  recommended five-seed list before Stage 2B or Stage 4.

If gate fails: rerun Stage 2A with the recommended five-seed list before any
Stage 2B/4 decision, or return to Stage 0/1 if the failure is path, feature, or
split related.

### Stage 2B - Baseline Family Comparison

Run this only after ESM embeddings are available or after you intentionally allow
the notebook to prepare missing ESM embeddings.

Notebook configuration block:

```python
TASK = "metal"
RUN_MODE = "manual_configurations"
RECOMMENDED_RUN_SET = "baseline_model_comparison"
# MODEL_PRESET is overridden by baseline_model_comparison (runs Only-GVP, Only-ESM, GVP + late fusion)
RUN_BATCH_ID = "metal_baseline_model_comparison"
SUMMARY_BASENAME = ""  # auto from provenance

EPOCHS = 50
BATCH_SIZES_CSV = "8"
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

ESM_EMBEDDINGS_DIR = ""  # set to your embeddings folder when available
ALLOW_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_ESM_EMBEDDINGS = True
RING_EDGE_MODE = "with_ring"
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

In the dedicated **Main planned training launch switch** cell, then run
**Optional training execution**:

```python
LAUNCH_PLANNED_MAIN_TRAINING_RUNS = True
```

Expected outputs/files:

- Planned-run CSV and dictionary under `<RUNS_DIR>/`
- Twelve completed validation-only run directories for the exact block above
- `<SUMMARY_BASENAME>.csv` and `<SUMMARY_BASENAME>_completed_only.csv`
- `<SUMMARY_BASENAME>.png` when plotting succeeds
- Each completed run directory contains `run_config.json`,
  `run_metadata.json`, `split_diagnostics.json`, `dataset_summary.json`, and
  `prepare_status.json`
- No `test_report.json`

Exact configuration record:

- Per run: `<run_dir>/run_config.json` and `<run_dir>/run_metadata.json`.
- Batch plan: planned-run CSV/dictionary.
- `active_run_config.json` / `active_run_config.md` are generated from the live
  notebook configuration before launch.

Success criteria:

- All planned runs complete.
- Each run uses `selection_metric = val_metal_balanced_acc`.
- `split_diagnostics.json` shows usable train/validation class coverage.
- Comparison tables rank only validation, group-kfold validation, or
  explicitly labeled seed-repeat validation rows.

### Decision gate after Stage 2B

Proceed to Stage 3 or Stage 4 only if:

- The Stage 2B success criteria are met.
- The expected planned files, summary files, and run-level JSON files exist.
- No held-out test files were created.
- `val_metal_balanced_acc` is the selection metric on all completed runs.
- Diagnostics report every active metal-scheme class present in both train and validation splits.
- Rare-class recall protection passes: per-class recall is available for each
  completed run, and no family is promoted if its seed mean has zero recall for
  any metal class.

If gate fails: fix ESM coverage, rerun the affected baseline family, or fall
back to the Stage 2A Only-GVP anchor until ESM-ready runs are trustworthy. Choose
baseline anchors by validation evidence, not by held-out test, and prefer
stability across seeds over one high run.

## Stage 3 - Optuna Plumbing Debug

Purpose: verify the controlled Optuna path, storage, command generation, and
search-space parsing without treating the result as model-selection evidence.

When to use it: first Optuna run in a new runtime or after editing Optuna
configuration fields.

Expected scale/runtime: smoke/debug, minutes to under an hour.

Stage 3 caveat: this is a plumbing/debug Optuna run. With the canonical
`OPTUNA_TARGET_COMPLETE_TRIALS = 4` and `OPTUNA_N_STARTUP_TRIALS = 4`, every trial is a TPE
startup trial, so Stage 3 is effectively random search. Stage 3 results are not
model-selection evidence; serious model-selection evidence comes from serious
HPO, validation-only comparison, and Stage 6 grouped-fold confirmation.

Notebook configuration block:

```python
TASK = "metal"
RUN_MODE = "controlled_hpo_optuna"
RECOMMENDED_RUN_SET = "custom"
MODEL_PRESET = "Only-GVP"
RUN_BATCH_ID = "metal_only_gvp_optuna_debug"
SUMMARY_BASENAME = ""  # auto from provenance

EPOCHS = 10
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_metal_balanced_acc"

HIDDEN_S_VALUES_CSV = "128"
HIDDEN_V_VALUES_CSV = "16"
EDGE_HIDDEN_VALUES_CSV = "64"
GVP_LAYERS_VALUES_CSV = "4"
HEAD_MLP_LAYERS_VALUES_CSV = "2"
EDGE_RADIUS_VALUES_CSV = "8.0"
RING_EDGE_MODE = "with_ring"

OPTUNA_INTENSITY = "custom"
OPTUNA_TARGET_COMPLETE_TRIALS = 4
MAX_EPOCHS_PER_TRIAL = 3
OPTUNA_N_STARTUP_TRIALS = 4
OPTUNA_TPE_MULTIVARIATE = True
OPTUNA_TPE_GROUP = True
OPTUNA_TPE_CONSTANT_LIAR = True
OPTUNA_PARALLEL_WORKERS = 1
OPTUNA_PARALLEL_STARTUP_STAGGER_SECONDS = 10.0
OPTUNA_STOP_ON_PARALLEL_CUDA_OOM = True
OPTUNA_AUTO_CONFIGURE_BUDGET = False
OPTUNA_USE_PRUNING = False
OPTUNA_PRUNER_TYPE = "none"
OPTUNA_PRUNING_MIN_EPOCH = 2
OPTUNA_SEARCH_PRESET = "first_useful_only_gvp_narrow"
OPTUNA_STUDY_NAME = "metal_only_gvp_optuna_debug"
OPTUNA_STORAGE = "sqlite:////content/drive/MyDrive/DeepMzyme/optuna/metal_only_gvp_optuna_debug.db"
OPTUNA_SPLIT_SEED = 42
OPTUNA_SAMPLER_SEED = None
OPTUNA_LEARNING_RATE_RANGE = "1e-5,3e-4"
WEIGHT_DECAYS_CSV = "0.0,1e-5,1e-4"
BATCH_SIZES_CSV = "4,8"
METAL_CLASS_WEIGHT_MODES_CSV = "none,inverse_frequency,inverse_sqrt_frequency,effective_number"
METAL_LOSS_FUNCTIONS_CSV = "cross_entropy"
METAL_LABEL_SMOOTHING_VALUES_CSV = "0.0"
BALANCE_METAL_SITE_SYMBOLS_CSV = "False"
RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = False

INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

In the dedicated **Main planned training launch switch** cell, then run
**Optional training execution**:

```python
LAUNCH_PLANNED_MAIN_TRAINING_RUNS = True
```

Expected outputs/files:

- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/all_trials.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/top_trials.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/best_trial.json`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/optuna_study_metadata.json`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/active_run_config.json`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/active_run_config.md`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/optuna_study_summary.md`
- `top_reevaluation_commands.txt`
- Four per-trial validation-only run directories under `<RUNS_DIR>/`
- Per-trial `active_run_config.json`, `active_run_config.md`,
  `run_config.json`, `run_metadata.json`, and `split_diagnostics.json`
- No `test_report.json`

Exact configuration record:

- Study level: Optuna CSV/JSON/Markdown outputs listed above.
- Per trial: `<run_dir>/active_run_config.json`,
  `<run_dir>/active_run_config.md`, `<run_dir>/run_config.json`, and
  `<run_dir>/run_metadata.json`.

Success criteria:

- Optuna launches and completes the debug trials.
- Search-space preview shows architecture fixed to Only-GVP.
- Trial commands omit held-out test evaluation.

### Decision gate after Stage 3

Proceed to Stage 4 or Stage 5A only if:

- The Stage 3 success criteria are met.
- `all_trials.csv` has exactly 4 trials and all 4 are `COMPLETE`.
- The expected Optuna files and per-trial run-level JSON files exist.
- No held-out test files were created.
- `val_metal_balanced_acc` is the selection metric on all completed trials.
- One `MODEL_PRESET` is used in the study: `Only-GVP`.
- Diagnostics report every active metal-scheme class present in both train and validation splits.
- Rare-class protection passes at the split/diagnostic level. Do not promote
  any Stage 3 hyperparameter result as model-selection evidence.

If gate fails: fix Optuna storage, search-space parsing, command generation, or
feature paths before launching Stage 4. Do not choose hyperparameters from this
debug run.

## Stage 4 - Medium Per-Family Optuna, Optional On G4

Purpose: run a useful but bounded HPO pass inside one selected model family.

When to use it: after baseline behavior is understood and you have selected a
model family to tune, usually Only-GVP first.

Expected scale/runtime: useful serious run on a G4-class GPU, usually hours.

Notebook configuration block for first useful Only-GVP HPO:

```python
TASK = "metal"
RUN_MODE = "controlled_hpo_optuna"
RECOMMENDED_RUN_SET = "custom"
MODEL_PRESET = "Only-GVP"
RUN_BATCH_ID = "metal_only_gvp_optuna_medium"
SUMMARY_BASENAME = ""  # auto from provenance

EPOCHS = 50
BATCH_SIZES_CSV = "8"
LEARNING_RATES_CSV = "3e-5"
WEIGHT_DECAYS_CSV = "1e-5,1e-4,1e-3"
SEEDS_CSV = "42"

HIDDEN_S_VALUES_CSV = "128"
HIDDEN_V_VALUES_CSV = "8,16"
EDGE_HIDDEN_VALUES_CSV = "64"
GVP_LAYERS_VALUES_CSV = "2,3"
HEAD_MLP_LAYERS_VALUES_CSV = "2"
EDGE_RADIUS_VALUES_CSV = "8.0"
RING_EDGE_MODE = "with_ring"
REQUIRE_RING_EDGES = False
PREPARE_MISSING_RING_EDGES = True
RING_FEATURES_DIR = ""
RING_EXE_PATH = "DeepMzyme_Data/ring-4.0/out/bin/ring"
ALLOW_MISSING_EXTERNAL_FEATURES = False
PREPARE_MISSING_EXTERNAL_FEATURES = True
EXTERNAL_FEATURES_ROOT_DIR = ""

OPTUNA_INTENSITY = "custom"
OPTUNA_TARGET_COMPLETE_TRIALS = 64
MAX_EPOCHS_PER_TRIAL = 35
OPTUNA_N_STARTUP_TRIALS = 20
OPTUNA_TPE_MULTIVARIATE = True
OPTUNA_TPE_GROUP = True
OPTUNA_TPE_CONSTANT_LIAR = True
OPTUNA_PARALLEL_WORKERS = 1
OPTUNA_PARALLEL_STARTUP_STAGGER_SECONDS = 10.0
OPTUNA_STOP_ON_PARALLEL_CUDA_OOM = True
OPTUNA_AUTO_CONFIGURE_BUDGET = False
OPTUNA_USE_PRUNING = True
OPTUNA_PRUNER_TYPE = "median"
OPTUNA_PRUNING_MIN_EPOCH = 25
OPTUNA_SEARCH_PRESET = "first_useful_only_gvp_narrow"
OPTUNA_STUDY_NAME = "metal_only_gvp_optuna_medium"
OPTUNA_STORAGE = "sqlite:////content/drive/MyDrive/DeepMzyme/optuna/metal_only_gvp_optuna_medium.db"
OPTUNA_SPLIT_SEED = 42
OPTUNA_SAMPLER_SEED = None
OPTUNA_LEARNING_RATE_RANGE = "1e-5,3e-4"
WEIGHT_DECAYS_CSV = "1e-5,1e-4,1e-3"
BATCH_SIZES_CSV = "8,16"
METAL_CLASS_WEIGHT_MODES_CSV = "none,inverse_frequency,inverse_sqrt_frequency,effective_number"
METAL_LOSS_FUNCTIONS_CSV = "cross_entropy"
METAL_LABEL_SMOOTHING_VALUES_CSV = "0.0,0.03,0.05,0.1"
BALANCE_METAL_SITE_SYMBOLS_CSV = "False,True"
HEAD_MLP_DROPOUT_VALUES_CSV = "0.1,0.2,0.3"
POSITION_NOISE_STDS_CSV = "0.0,0.05,0.1"
OUTER_RESIDUE_DROPOUTS_CSV = "0.0,0.1,0.2"
RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = False
# Run Stage 6 later from the dedicated Stage 6 launch cell after HPO completes.
TOP_K_CONFIGS_FOR_SEED_REPEAT = "auto"
REPEAT_SEEDS = "42,123,2026,43,44"

INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

Expected outputs/files:

- Optuna study directory under `<RUNS_DIR>/optuna/`
- `all_trials.csv`, `top_trials.csv`, `best_trial.json`
- `optuna_best_config.json`, `best_config_command.txt`
- `top_reevaluation_commands.txt`
- `optuna_study_summary.md`
- Sixty-four per-trial validation-only run directories under `<RUNS_DIR>/`
- Per-trial `active_run_config.json`, `active_run_config.md`,
  `run_config.json`, `run_metadata.json`, and `split_diagnostics.json`
- No `test_report.json`

Exact configuration record:

- Study level: Optuna CSV/JSON/Markdown outputs listed above.
- Per trial: `<run_dir>/active_run_config.json`,
  `<run_dir>/active_run_config.md`, `<run_dir>/run_config.json`, and
  `<run_dir>/run_metadata.json`.
- `active_run_config.json` / `active_run_config.md` are generated from the live
  notebook configuration before launch.

Success criteria:

- The study completes the requested 64-trial count.
- The best-trial summary is based on `val_metal_balanced_acc`.
- Trial logs show validation-only runs, not final-test runs.
- Top candidates have finite selected validation metrics and no missing-class
  diagnostics.

### Decision gate after Stage 4

Proceed to Stage 5 or Stage 6 only if:

- The Stage 4 success criteria are met.
- `all_trials.csv` contains at least 64 `COMPLETE` trials for this
  `MODEL_PRESET`; resume the same study until that count is reached.
- The expected Optuna files and per-trial run-level JSON files exist.
- No held-out test files were created.
- `val_metal_balanced_acc` is the selection metric on all completed runs.
- One `MODEL_PRESET` is used in the study: `Only-GVP`.
- Diagnostics report every active metal-scheme class present in both train and validation splits.
- Rare-class recall protection passes: top candidates have available
  per-class recall, and no candidate is promoted if any metal class has zero
  recall in its validation artifact.
- At least one top candidate exceeds the Stage 2A seed mean on
  `val_metal_balanced_acc`; otherwise Stage 4 may be used only as
  search-space diagnosis, not as a promotion gate.

If gate fails: check the search space; widen `OPTUNA_LEARNING_RATE_RANGE` or
open `HIDDEN_S_VALUES_CSV`. Do not pick the final model from one Optuna
trial alone; run top-K grouped-fold confirmation before considering a
configuration stable.

## Stage 5 - Serious Per-Family HPO

Purpose: perform a longer, controlled search after the simpler baseline and
medium HPO justify the model family and search axes.

When to use it: after at least one medium HPO or Stage 6 confirmation batch
identifies the model family and search axes worth expanding, or when the user
asks for a fresh broad Optuna check and does not explicitly ask to rely on
previous raw outputs.

Expected scale/runtime: large Optuna search, potentially very long or
overnight. A 200-trial run can be substantially longer than one night depending
on GPU and model.

Important scope rule: the notebook's Optuna mode optimizes within the selected
`MODEL_PRESET`. It does not freely search architectures or fusion modes. Choose
the model family explicitly, then search a controlled set of hyperparameters.

Advanced-fusion ordering rule: Stages 5D, 5E, 5F are only valid after Stage 5C
(GVP + late fusion) has produced a Stage 6 grouped-fold candidate that exceeds
the Stage 2A Only-GVP anchor by at least `0.01` mean
`val_metal_balanced_acc`, and the paired bootstrap 95% CI for that improvement
excludes zero. If Stage 5C does not clear that bar, do not launch 5D/5E/5F.

### Shared Stage 5 Output, Config-Record, And Gate Template

This template applies to Stage 5A-5F unless a substage states an addition or
stricter rule.

Expected outputs/files:

- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/all_trials.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/top_trials.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/best_trial.json`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/optuna_study_metadata.json`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/active_run_config.json`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/active_run_config.md`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/optuna_best_config.json`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/best_config_command.txt`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/top_reevaluation_commands.txt`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/optuna_study_summary.md`
- One complete validation-only per-trial run directory for every required
  completed trial in the substage.
- Per-trial `active_run_config.json`, `active_run_config.md`,
  `run_config.json`, `run_metadata.json`, and `split_diagnostics.json`.
- No `test_report.json`.

Exact configuration record:

- Study level: Optuna CSV/JSON/Markdown outputs listed above.
- Per trial: `<run_dir>/active_run_config.json`,
  `<run_dir>/active_run_config.md`, `<run_dir>/run_config.json`, and
  `<run_dir>/run_metadata.json`.

Common decision-gate requirements:

- The expected Optuna files and per-trial run-level JSON files exist.
- `all_trials.csv` contains the required number of `COMPLETE` trials for the
  substage and one `MODEL_PRESET`; resume the same compatible study until that
  count is reached.
- No held-out test files were created.
- `val_metal_balanced_acc` is the selection metric on all completed runs.
- Diagnostics report every active metal-scheme class present in both train and validation splits.
- Rare-class recall protection passes: top candidates have available per-class
  recall, and no candidate is promoted if any metal class has zero recall in
  its validation artifact.
- Top candidates remain review-only until Stage 6 grouped-fold confirmation.

### Stage 5A - Serious Only-GVP HPO

Notebook configuration block:

```python
TASK = "metal"
RUN_MODE = "controlled_hpo_optuna"
RECOMMENDED_RUN_SET = "custom"
MODEL_PRESET = "Only-GVP"
RUN_BATCH_ID = "metal_only_gvp_optuna_200_capacity"
SUMMARY_BASENAME = ""  # auto from provenance

EPOCHS = 50
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_metal_balanced_acc"
RING_EDGE_MODE = "with_ring"
REQUIRE_RING_EDGES = False
PREPARE_MISSING_RING_EDGES = True
RING_FEATURES_DIR = ""
RING_EXE_PATH = "DeepMzyme_Data/ring-4.0/out/bin/ring"
ALLOW_MISSING_EXTERNAL_FEATURES = False
PREPARE_MISSING_EXTERNAL_FEATURES = True
EXTERNAL_FEATURES_ROOT_DIR = ""

OPTUNA_INTENSITY = "custom"
OPTUNA_TARGET_COMPLETE_TRIALS = 200
MAX_EPOCHS_PER_TRIAL = 50
OPTUNA_N_STARTUP_TRIALS = 40
OPTUNA_TPE_MULTIVARIATE = True
OPTUNA_TPE_GROUP = True
OPTUNA_TPE_CONSTANT_LIAR = True
OPTUNA_PARALLEL_WORKERS = 1
OPTUNA_PARALLEL_STARTUP_STAGGER_SECONDS = 10.0
OPTUNA_STOP_ON_PARALLEL_CUDA_OOM = True
OPTUNA_AUTO_CONFIGURE_BUDGET = False
OPTUNA_USE_PRUNING = True
OPTUNA_PRUNER_TYPE = "median"
OPTUNA_PRUNING_MIN_EPOCH = 25
OPTUNA_SEARCH_PRESET = "later_capacity"
OPTUNA_STUDY_NAME = "metal_only_gvp_optuna_200_capacity"
OPTUNA_STORAGE = "sqlite:////content/drive/MyDrive/DeepMzyme/optuna/metal_only_gvp_optuna_200_capacity.db"
OPTUNA_SPLIT_SEED = 42
OPTUNA_SAMPLER_SEED = None
OPTUNA_TIMEOUT_MINUTES = 0
OPTUNA_MULTIOBJECTIVE = False

OPTUNA_LEARNING_RATE_RANGE = "5e-6,3e-4"
LR_SCHEDULES_CSV = "fixed,cosine"
WEIGHT_DECAYS_CSV = "1e-5,1e-4,1e-3"
BATCH_SIZES_CSV = "8,16,32"
METAL_CLASS_WEIGHT_MODES_CSV = "none,inverse_frequency,inverse_sqrt_frequency,effective_number"
METAL_LOSS_FUNCTIONS_CSV = "cross_entropy,focal"
METAL_LABEL_SMOOTHING_VALUES_CSV = "0.0,0.03,0.05,0.1"
METAL_COLLAPSED_LOSS_WEIGHTS_CSV = "0.0"
BALANCE_METAL_SITE_SYMBOLS_CSV = "False,True"
METAL_FOCAL_GAMMA_VALUES_CSV = "1.5,2.0,2.5"
HEAD_MLP_DROPOUT_VALUES_CSV = "0.1,0.2,0.3"
POSITION_NOISE_STDS_CSV = "0.0,0.05,0.1"
OUTER_RESIDUE_DROPOUTS_CSV = "0.0,0.1,0.2"

HIDDEN_S_VALUES_CSV = "128"
HIDDEN_V_VALUES_CSV = "8,16"
EDGE_HIDDEN_VALUES_CSV = "64"
GVP_LAYERS_VALUES_CSV = "2,3"
HEAD_MLP_LAYERS_VALUES_CSV = "1,2"
EDGE_RADIUS_VALUES_CSV = "6.0,8.0,10.0"
CLASSIFIER_POOL_DISTANCE_CUTOFF_VALUES_CSV = "0.0"

RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = False
# Run Stage 6 later from the dedicated Stage 6 launch cell after HPO completes.
TOP_K_CONFIGS_FOR_SEED_REPEAT = "auto"
REPEAT_SEEDS = "42,123,2026,43,44"
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

Optional Stage 5A validation-only objective overlay:

```python
# Collapsed-4 auxiliary-loss probe; keep this out of initial baselines.
METAL_COLLAPSED_LOSS_WEIGHTS_CSV = "0.0,0.3,0.5"
METAL_LOSS_FUNCTIONS_CSV = "cross_entropy"

# Optional rare-class-protection Pareto search.
OPTUNA_MULTIOBJECTIVE = True
OPTUNA_SELECTION_METRIC = "val_metal_balanced_acc"
OPTUNA_USE_PRUNING = False
OPTUNA_PRUNER_TYPE = "none"
```

Use either part of this overlay only for an explicitly labeled validation-only
Stage 5A experiment. Do not enable it for Stage 2 baselines or Stage 7 final
held-out testing.

`batch_size=32` in this Stage 5A block is exploratory for Only-GVP only. Treat
CUDA OOM as a failed trial, not a prune, and inspect whether the larger batch
hurts rare-class recall before promoting any candidate.

Expected outputs/files:

- All shared Stage 5 Optuna and per-trial outputs.
- If `OPTUNA_MULTIOBJECTIVE = True`: `pareto_front.csv`,
  `pareto_candidates.csv`, and `pareto_candidates_ranked_for_review.csv`
- Two hundred complete per-trial validation-only run directories.

Exact configuration record:

- Use the shared Stage 5 exact-configuration record template.

### Decision gate after Stage 5A

Proceed to Stage 6 for Only-GVP candidates, or to Stage 5B/5C for family
comparison, only if:

- The shared Stage 5 decision-gate requirements pass for
  `MODEL_PRESET = "Only-GVP"` and at least 200 `COMPLETE` trials.
- For `OPTUNA_MULTIOBJECTIVE = True`, the study also writes complete
  `pareto_front.csv`, `pareto_candidates.csv`, and
  `pareto_candidates_ranked_for_review.csv`, and any convenience-ranked
  candidate is treated as review-only until Stage 6.
- Select top candidates for Stage 6 only if they do not degrade
  `val_metal_min_recall` by more than 0.05 versus the Stage 2A anchor, unless
  explicitly marked as exploratory.

If gate fails: do not advance to a more complex fusion family; revisit Stage 2A
and the Stage 5A search space.

### Stage 5B - Only-ESM HPO

Run this after ESM coverage is valid. It is the ESM-only baseline HPO; it does
not use graph/RING capacity fields even if those fields remain present in the
notebook.

Notebook configuration block:

```python
TASK = "metal"
RUN_MODE = "controlled_hpo_optuna"
RECOMMENDED_RUN_SET = "custom"
MODEL_PRESET = "Only-ESM"
RUN_BATCH_ID = "metal_only_esm_optuna_120_controlled"
SUMMARY_BASENAME = ""  # auto from provenance

EPOCHS = 50
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_metal_balanced_acc"
RING_EDGE_MODE = "with_ring"

ESM_EMBEDDINGS_DIR = ""  # set to your embeddings folder when available
ALLOW_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_ESM_EMBEDDINGS = True

OPTUNA_INTENSITY = "custom"
OPTUNA_TARGET_COMPLETE_TRIALS = 120
MAX_EPOCHS_PER_TRIAL = 50
OPTUNA_N_STARTUP_TRIALS = 30
OPTUNA_TPE_MULTIVARIATE = True
OPTUNA_TPE_GROUP = True
OPTUNA_TPE_CONSTANT_LIAR = True
OPTUNA_PARALLEL_WORKERS = 1
OPTUNA_PARALLEL_STARTUP_STAGGER_SECONDS = 10.0
OPTUNA_STOP_ON_PARALLEL_CUDA_OOM = True
OPTUNA_AUTO_CONFIGURE_BUDGET = False
OPTUNA_USE_PRUNING = False
OPTUNA_PRUNER_TYPE = "none"
OPTUNA_PRUNING_MIN_EPOCH = 20
OPTUNA_SEARCH_PRESET = "custom"
OPTUNA_STUDY_NAME = "metal_only_esm_optuna_120_controlled"
OPTUNA_STORAGE = "sqlite:////content/drive/MyDrive/DeepMzyme/optuna/metal_only_esm_optuna_120_controlled.db"
OPTUNA_SPLIT_SEED = 42
OPTUNA_SAMPLER_SEED = None
OPTUNA_TIMEOUT_MINUTES = 0

OPTUNA_LEARNING_RATE_RANGE = "5e-6,2e-4"
WEIGHT_DECAYS_CSV = "0.0,1e-6,1e-5,1e-4"
BATCH_SIZES_CSV = "8,16"
METAL_CLASS_WEIGHT_MODES_CSV = "none,inverse_frequency,inverse_sqrt_frequency,effective_number"
METAL_LOSS_FUNCTIONS_CSV = "cross_entropy"
METAL_LABEL_SMOOTHING_VALUES_CSV = "0.0,0.03,0.05,0.1"
BALANCE_METAL_SITE_SYMBOLS_CSV = "False,True"

HIDDEN_S_VALUES_CSV = "128,256"
HEAD_MLP_LAYERS_VALUES_CSV = "1,2,3"

RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = False
# Run Stage 6 later from the dedicated Stage 6 launch cell after HPO completes.
TOP_K_CONFIGS_FOR_SEED_REPEAT = "auto"
REPEAT_SEEDS = "42,123,2026,43,44"
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

Expected outputs/files:

- All shared Stage 5 Optuna and per-trial outputs.
- One hundred twenty complete per-trial validation-only run directories.

Exact configuration record:

- Use the shared Stage 5 exact-configuration record template.

### Decision gate after Stage 5B

Proceed to Stage 6 for Only-ESM candidates, or to Stage 5C, only if:

- The shared Stage 5 decision-gate requirements pass for
  `MODEL_PRESET = "Only-ESM"` and at least 120 `COMPLETE` trials.
- ESM coverage is valid and no run used missing ESM embeddings as a reportable
  fallback.

If gate fails: fix ESM coverage or narrow the Only-ESM search before comparing
ESM-informed model families.

### Stage 5C - GVP + Late Fusion HPO

Run this only after ESM coverage is valid and simpler baselines justify ESM
fusion.

Notebook configuration block:

```python
TASK = "metal"
RUN_MODE = "controlled_hpo_optuna"
RECOMMENDED_RUN_SET = "custom"
MODEL_PRESET = "GVP + late fusion"
RUN_BATCH_ID = "metal_late_fusion_optuna_200_controlled"
SUMMARY_BASENAME = ""  # auto from provenance

EPOCHS = 50
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_metal_balanced_acc"
RING_EDGE_MODE = "with_ring"

ESM_EMBEDDINGS_DIR = ""  # set to your embeddings folder when available
ALLOW_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_ESM_EMBEDDINGS = True

OPTUNA_INTENSITY = "custom"
OPTUNA_TARGET_COMPLETE_TRIALS = 200
MAX_EPOCHS_PER_TRIAL = 50
OPTUNA_N_STARTUP_TRIALS = 40
OPTUNA_TPE_MULTIVARIATE = True
OPTUNA_TPE_GROUP = True
OPTUNA_TPE_CONSTANT_LIAR = True
OPTUNA_PARALLEL_WORKERS = 1
OPTUNA_PARALLEL_STARTUP_STAGGER_SECONDS = 10.0
OPTUNA_STOP_ON_PARALLEL_CUDA_OOM = True
OPTUNA_AUTO_CONFIGURE_BUDGET = False
OPTUNA_USE_PRUNING = True
OPTUNA_PRUNER_TYPE = "median"
OPTUNA_PRUNING_MIN_EPOCH = 25
OPTUNA_SEARCH_PRESET = "custom"
OPTUNA_STUDY_NAME = "metal_late_fusion_optuna_200_controlled"
OPTUNA_STORAGE = "sqlite:////content/drive/MyDrive/DeepMzyme/optuna/metal_late_fusion_optuna_200_controlled.db"
OPTUNA_SPLIT_SEED = 42
OPTUNA_SAMPLER_SEED = None
OPTUNA_TIMEOUT_MINUTES = 0

OPTUNA_LEARNING_RATE_RANGE = "5e-6,2e-4"
LR_SCHEDULES_CSV = "fixed,cosine"
WEIGHT_DECAYS_CSV = "1e-5,1e-4,1e-3"
BATCH_SIZES_CSV = "8,16"
METAL_CLASS_WEIGHT_MODES_CSV = "inverse_frequency,inverse_sqrt_frequency,effective_number"
METAL_LOSS_FUNCTIONS_CSV = "cross_entropy"
METAL_LABEL_SMOOTHING_VALUES_CSV = "0.0,0.03,0.05,0.1"
BALANCE_METAL_SITE_SYMBOLS_CSV = "False,True"
HEAD_MLP_DROPOUT_VALUES_CSV = "0.1,0.2,0.3"
ESM_GRAPH_ENCODER_DROPOUT_VALUES_CSV = "0.0,0.1,0.2"
POSITION_NOISE_STDS_CSV = "0.0,0.05,0.1"
OUTER_RESIDUE_DROPOUTS_CSV = "0.0,0.1,0.2"

HIDDEN_S_VALUES_CSV = "128"
HIDDEN_V_VALUES_CSV = "8,16"
EDGE_HIDDEN_VALUES_CSV = "64"
GVP_LAYERS_VALUES_CSV = "2,3"
HEAD_MLP_LAYERS_VALUES_CSV = "1,2"
EDGE_RADIUS_VALUES_CSV = "6.0,8.0,10.0"
CLASSIFIER_POOL_DISTANCE_CUTOFF_VALUES_CSV = "0.0"
ESM_FUSION_DIM_VALUES_CSV = "64,128,256"

RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = False
# Run Stage 6 later from the dedicated Stage 6 launch cell after HPO completes.
TOP_K_CONFIGS_FOR_SEED_REPEAT = "auto"
REPEAT_SEEDS = "42,123,2026,43,44"
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

Expected outputs/files:

- All shared Stage 5 Optuna and per-trial outputs.
- Two hundred complete per-trial validation-only run directories.

Exact configuration record:

- Use the shared Stage 5 exact-configuration record template.

### Decision gate after Stage 5C

Proceed to Stage 6 only if:

- The shared Stage 5 decision-gate requirements pass for
  `MODEL_PRESET = "GVP + late fusion"` and at least 200 `COMPLETE` trials.
- Top candidates have finite selected validation metrics.

Proceed to Stage 5D/5E/5F only after Stage 6 confirms a late-fusion candidate
that beats the Stage 2A Only-GVP anchor by at least 0.01 mean
`val_metal_balanced_acc`, and the paired bootstrap 95% CI for the improvement
excludes zero.

If gate fails: no candidate from Stage 5C should trigger advanced fusion. Return
to Stage 2A/5A or revise the late-fusion search space.

### Stage 5D - GVP + Node-Level Late Fusion HPO

Run this after the late-fusion baseline has a stable validation anchor.

Notebook configuration block:

```python
TASK = "metal"
RUN_MODE = "controlled_hpo_optuna"
RECOMMENDED_RUN_SET = "custom"
MODEL_PRESET = "GVP + node-level late fusion"
RUN_BATCH_ID = "metal_node_late_fusion_optuna_200_controlled"
SUMMARY_BASENAME = ""  # auto from provenance

EPOCHS = 50
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_metal_balanced_acc"
RING_EDGE_MODE = "with_ring"

ESM_EMBEDDINGS_DIR = ""  # set to your embeddings folder when available
ALLOW_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_ESM_EMBEDDINGS = True

OPTUNA_INTENSITY = "custom"
OPTUNA_TARGET_COMPLETE_TRIALS = 200
MAX_EPOCHS_PER_TRIAL = 50
OPTUNA_N_STARTUP_TRIALS = 40
OPTUNA_TPE_MULTIVARIATE = True
OPTUNA_TPE_GROUP = True
OPTUNA_TPE_CONSTANT_LIAR = True
OPTUNA_PARALLEL_WORKERS = 1
OPTUNA_PARALLEL_STARTUP_STAGGER_SECONDS = 10.0
OPTUNA_STOP_ON_PARALLEL_CUDA_OOM = True
OPTUNA_AUTO_CONFIGURE_BUDGET = False
OPTUNA_USE_PRUNING = True
OPTUNA_PRUNER_TYPE = "median"
OPTUNA_PRUNING_MIN_EPOCH = 25
OPTUNA_SEARCH_PRESET = "custom"
OPTUNA_STUDY_NAME = "metal_node_late_fusion_optuna_200_controlled"
OPTUNA_STORAGE = "sqlite:////content/drive/MyDrive/DeepMzyme/optuna/metal_node_late_fusion_optuna_200_controlled.db"
OPTUNA_SPLIT_SEED = 42
OPTUNA_SAMPLER_SEED = None
OPTUNA_TIMEOUT_MINUTES = 0

OPTUNA_LEARNING_RATE_RANGE = "5e-6,2e-4"
LR_SCHEDULES_CSV = "fixed,cosine"
WEIGHT_DECAYS_CSV = "1e-5,1e-4,1e-3"
BATCH_SIZES_CSV = "8,16"
METAL_CLASS_WEIGHT_MODES_CSV = "inverse_frequency,inverse_sqrt_frequency,effective_number"
METAL_LOSS_FUNCTIONS_CSV = "cross_entropy"
METAL_LABEL_SMOOTHING_VALUES_CSV = "0.0,0.03,0.05,0.1"
BALANCE_METAL_SITE_SYMBOLS_CSV = "False,True"
HEAD_MLP_DROPOUT_VALUES_CSV = "0.1,0.2,0.3"
ESM_GRAPH_ENCODER_DROPOUT_VALUES_CSV = "0.0,0.1,0.2"
POSITION_NOISE_STDS_CSV = "0.0,0.05,0.1"
OUTER_RESIDUE_DROPOUTS_CSV = "0.0,0.1,0.2"

HIDDEN_S_VALUES_CSV = "128"
HIDDEN_V_VALUES_CSV = "8,16"
EDGE_HIDDEN_VALUES_CSV = "64"
GVP_LAYERS_VALUES_CSV = "2,3"
HEAD_MLP_LAYERS_VALUES_CSV = "1,2"
EDGE_RADIUS_VALUES_CSV = "6.0,8.0,10.0"
CLASSIFIER_POOL_DISTANCE_CUTOFF_VALUES_CSV = "0.0"
ESM_FUSION_DIM_VALUES_CSV = "64,128,256"

RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = False
# Run Stage 6 later from the dedicated Stage 6 launch cell after HPO completes.
TOP_K_CONFIGS_FOR_SEED_REPEAT = "auto"
REPEAT_SEEDS = "42,123,2026,43,44"
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

Expected outputs/files:

- All shared Stage 5 Optuna and per-trial outputs.
- Two hundred complete per-trial validation-only run directories.

Exact configuration record:

- Use the shared Stage 5 exact-configuration record template.

### Decision gate after Stage 5D

Proceed to Stage 6 only if:

- Stage 5C previously cleared the advanced-fusion ordering gate.
- The shared Stage 5 decision-gate requirements pass for
  `MODEL_PRESET = "GVP + node-level late fusion"` and at least 200 `COMPLETE`
  trials.

After Stage 6, promote a node-level late-fusion candidate only if it beats the
current best confirmed comparator by at least 0.005 mean
`val_metal_balanced_acc`, and the paired bootstrap 95% CI for the improvement
excludes zero.

If the Stage 5D launch gate or the later Stage 6 promotion gate fails, do not
advance to Stage 5E/5F; revisit Stage 2A or Stage 5C.

### Stage 5E - GVP + Hybrid Fusion HPO

Run this only after early/late ESM evidence justifies injecting ESM before graph
message passing and also using late fusion.

Notebook configuration block:

```python
TASK = "metal"
RUN_MODE = "controlled_hpo_optuna"
RECOMMENDED_RUN_SET = "custom"
MODEL_PRESET = "GVP + hybrid fusion"
RUN_BATCH_ID = "metal_hybrid_fusion_optuna_200_controlled"
SUMMARY_BASENAME = ""  # auto from provenance

EPOCHS = 50
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_metal_balanced_acc"
RING_EDGE_MODE = "with_ring"

ESM_EMBEDDINGS_DIR = ""  # set to your embeddings folder when available
ALLOW_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_ESM_EMBEDDINGS = True

OPTUNA_INTENSITY = "custom"
OPTUNA_TARGET_COMPLETE_TRIALS = 200
MAX_EPOCHS_PER_TRIAL = 50
OPTUNA_N_STARTUP_TRIALS = 40
OPTUNA_TPE_MULTIVARIATE = True
OPTUNA_TPE_GROUP = True
OPTUNA_TPE_CONSTANT_LIAR = True
OPTUNA_PARALLEL_WORKERS = 1
OPTUNA_PARALLEL_STARTUP_STAGGER_SECONDS = 10.0
OPTUNA_STOP_ON_PARALLEL_CUDA_OOM = True
OPTUNA_AUTO_CONFIGURE_BUDGET = False
OPTUNA_USE_PRUNING = True
OPTUNA_PRUNER_TYPE = "median"
OPTUNA_PRUNING_MIN_EPOCH = 25
OPTUNA_SEARCH_PRESET = "custom"
OPTUNA_STUDY_NAME = "metal_hybrid_fusion_optuna_200_controlled"
OPTUNA_STORAGE = "sqlite:////content/drive/MyDrive/DeepMzyme/optuna/metal_hybrid_fusion_optuna_200_controlled.db"
OPTUNA_SPLIT_SEED = 42
OPTUNA_SAMPLER_SEED = None
OPTUNA_TIMEOUT_MINUTES = 0

OPTUNA_LEARNING_RATE_RANGE = "5e-6,1.5e-4"
LR_SCHEDULES_CSV = "fixed,cosine"
WEIGHT_DECAYS_CSV = "1e-5,1e-4,1e-3"
BATCH_SIZES_CSV = "8,16"
METAL_CLASS_WEIGHT_MODES_CSV = "inverse_frequency,inverse_sqrt_frequency,effective_number"
METAL_LOSS_FUNCTIONS_CSV = "cross_entropy"
METAL_LABEL_SMOOTHING_VALUES_CSV = "0.0,0.03,0.05,0.1"
BALANCE_METAL_SITE_SYMBOLS_CSV = "False,True"
HEAD_MLP_DROPOUT_VALUES_CSV = "0.1,0.2,0.3"
ESM_GRAPH_ENCODER_DROPOUT_VALUES_CSV = "0.0,0.1,0.2"
POSITION_NOISE_STDS_CSV = "0.0,0.05,0.1"
OUTER_RESIDUE_DROPOUTS_CSV = "0.0,0.1,0.2"

HIDDEN_S_VALUES_CSV = "128"
HIDDEN_V_VALUES_CSV = "8,16"
EDGE_HIDDEN_VALUES_CSV = "64"
GVP_LAYERS_VALUES_CSV = "2,3"
HEAD_MLP_LAYERS_VALUES_CSV = "1,2"
EDGE_RADIUS_VALUES_CSV = "6.0,8.0,10.0"
CLASSIFIER_POOL_DISTANCE_CUTOFF_VALUES_CSV = "0.0"
ESM_FUSION_DIM_VALUES_CSV = "64,128,256"
EARLY_ESM_DIM_VALUES_CSV = "16,32,64"
EARLY_ESM_DROPOUT_VALUES_CSV = "0.0,0.1,0.2"

RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = False
# Run Stage 6 later from the dedicated Stage 6 launch cell after HPO completes.
TOP_K_CONFIGS_FOR_SEED_REPEAT = "auto"
REPEAT_SEEDS = "42,123,2026,43,44"
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

Expected outputs/files:

- All shared Stage 5 Optuna and per-trial outputs.
- Two hundred complete per-trial validation-only run directories.

Exact configuration record:

- Use the shared Stage 5 exact-configuration record template.

### Decision gate after Stage 5E

Proceed to Stage 6 only if:

- Stage 5C previously cleared the advanced-fusion ordering gate.
- The shared Stage 5 decision-gate requirements pass for
  `MODEL_PRESET = "GVP + hybrid fusion"` and at least 200 `COMPLETE` trials.

After Stage 6, promote a hybrid-fusion candidate only if it beats the current
best confirmed comparator by at least 0.005 mean `val_metal_balanced_acc`, and
the paired bootstrap 95% CI for the improvement excludes zero.

This Stage 5E result does not by itself complete the required fusion-position
comparison. A final early-versus-late-versus-hybrid conclusion also requires
the playbook-defined early-fusion candidate and a matched shared-fold Stage 6
comparison across all three fusion modes.

If the Stage 5E launch gate or the later Stage 6 promotion gate fails, stop
advanced fusion escalation and revisit the simpler late-fusion or Only-GVP
anchors before cross-attention.

### Stage 5F - GVP + Cross-Attention HPO

Run this last among fusion models. Keep attention narrow at first because it has
more overfitting degrees of freedom.

Notebook configuration block:

```python
TASK = "metal"
RUN_MODE = "controlled_hpo_optuna"
RECOMMENDED_RUN_SET = "custom"
MODEL_PRESET = "GVP + cross-modal attention"
RUN_BATCH_ID = "metal_cross_attention_optuna_120_controlled"
SUMMARY_BASENAME = ""  # auto from provenance

EPOCHS = 50
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_metal_balanced_acc"
RING_EDGE_MODE = "with_ring"
CROSS_ATTENTION_NEIGHBORHOOD = "first_second_shell"
CROSS_ATTENTION_BIDIRECTIONAL = False

ESM_EMBEDDINGS_DIR = ""  # set to your embeddings folder when available
ALLOW_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_ESM_EMBEDDINGS = True

OPTUNA_INTENSITY = "custom"
OPTUNA_TARGET_COMPLETE_TRIALS = 120
MAX_EPOCHS_PER_TRIAL = 50
OPTUNA_N_STARTUP_TRIALS = 30
OPTUNA_TPE_MULTIVARIATE = True
OPTUNA_TPE_GROUP = True
OPTUNA_TPE_CONSTANT_LIAR = True
OPTUNA_PARALLEL_WORKERS = 1
OPTUNA_PARALLEL_STARTUP_STAGGER_SECONDS = 10.0
OPTUNA_STOP_ON_PARALLEL_CUDA_OOM = True
OPTUNA_AUTO_CONFIGURE_BUDGET = False
OPTUNA_USE_PRUNING = True
OPTUNA_PRUNER_TYPE = "median"
OPTUNA_PRUNING_MIN_EPOCH = 25
OPTUNA_SEARCH_PRESET = "custom"
OPTUNA_STUDY_NAME = "metal_cross_attention_optuna_120_controlled"
OPTUNA_STORAGE = "sqlite:////content/drive/MyDrive/DeepMzyme/optuna/metal_cross_attention_optuna_120_controlled.db"
OPTUNA_SPLIT_SEED = 42
OPTUNA_SAMPLER_SEED = None
OPTUNA_TIMEOUT_MINUTES = 0

OPTUNA_LEARNING_RATE_RANGE = "5e-6,1e-4"
WEIGHT_DECAYS_CSV = "1e-5,1e-4,1e-3"
BATCH_SIZES_CSV = "8,16"
METAL_CLASS_WEIGHT_MODES_CSV = "inverse_frequency,inverse_sqrt_frequency,effective_number"
METAL_LOSS_FUNCTIONS_CSV = "cross_entropy"
METAL_LABEL_SMOOTHING_VALUES_CSV = "0.0,0.03,0.05,0.1"
BALANCE_METAL_SITE_SYMBOLS_CSV = "False"
HEAD_MLP_DROPOUT_VALUES_CSV = "0.1,0.2,0.3"
POSITION_NOISE_STDS_CSV = "0.0,0.05,0.1"
OUTER_RESIDUE_DROPOUTS_CSV = "0.0,0.1,0.2"

HIDDEN_S_VALUES_CSV = "128"
HIDDEN_V_VALUES_CSV = "8,16"
EDGE_HIDDEN_VALUES_CSV = "64"
GVP_LAYERS_VALUES_CSV = "2,3"
HEAD_MLP_LAYERS_VALUES_CSV = "1,2"
EDGE_RADIUS_VALUES_CSV = "6.0,8.0,10.0"
CLASSIFIER_POOL_DISTANCE_CUTOFF_VALUES_CSV = "0.0"
CROSS_ATTENTION_LAYERS_CSV = "1"
CROSS_ATTENTION_HEADS_CSV = "2,4"
CROSS_ATTENTION_DROPOUT_VALUES_CSV = "0.0,0.1,0.2"

RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = False
# Run Stage 6 later from the dedicated Stage 6 launch cell after HPO completes.
TOP_K_CONFIGS_FOR_SEED_REPEAT = "auto"
REPEAT_SEEDS = "42,123,2026,43,44"
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

Expected outputs/files:

- All shared Stage 5 Optuna and per-trial outputs.
- One hundred twenty complete per-trial validation-only run directories.

Exact configuration record:

- Use the shared Stage 5 exact-configuration record template.

### Decision gate after Stage 5F

Proceed to Stage 6 only if:

- Stage 5C previously cleared the advanced-fusion ordering gate.
- The shared Stage 5 decision-gate requirements pass for
  `MODEL_PRESET = "GVP + cross-modal attention"` and at least 120 `COMPLETE`
  trials.
- Attention candidates justify their extra complexity against the Stage 6
  late-fusion candidate.

After Stage 6, promote a cross-attention candidate only if it beats the current
best confirmed comparator by at least 0.005 mean `val_metal_balanced_acc`, and
the paired bootstrap 95% CI for the improvement excludes zero.

If the Stage 5F launch gate or the later Stage 6 promotion gate fails, do not
broaden cross-attention. Return to the best validated simpler fusion family.

### Stage 5G - RING/Radius-Only Ablation

Use only when you deliberately want to compare against the older radius-only
graph setting. This does not make Optuna sample RING on/off; it fixes the base
run to radius-only graph construction. This standalone block mirrors Stage 2A's
Only-GVP validation anchor while changing only the graph-edge mode and labels
the output as a radius-only ablation.

This is the minimum required RING comparison: matched Only-GVP with radius-only
edges versus radius + RING edges. If the proposed final GVP + ESMC model uses
RING, add a second matched on/off ablation for that same combined architecture
before attributing an improvement to RING.

```python
TASK = "metal"
RUN_MODE = "manual_configurations"
RECOMMENDED_RUN_SET = "only_gvp_broad_comparison"
MODEL_PRESET = "Only-GVP"
RUN_BATCH_ID = "metal_only_gvp_radius_only_ablation"
SUMMARY_BASENAME = ""  # auto from provenance

EPOCHS = 50
BATCH_SIZES_CSV = "8"
LEARNING_RATES_CSV = "3e-5,1e-4,3e-4"
WEIGHT_DECAYS_CSV = "1e-4"
SEEDS_CSV = "42,43,44"
MAX_CONFIGURATION_RUNS = 9

HIDDEN_S_VALUES_CSV = "128"
HIDDEN_V_VALUES_CSV = "16"
EDGE_HIDDEN_VALUES_CSV = "64"
GVP_LAYERS_VALUES_CSV = "4"
HEAD_MLP_LAYERS_VALUES_CSV = "2"
EDGE_RADIUS_VALUES_CSV = "8.0"

RING_EDGE_MODE = "without_ring"
REQUIRE_RING_EDGES = False
PREPARE_MISSING_RING_EDGES = True
RING_FEATURES_DIR = ""

ESM_EMBEDDINGS_DIR = ""
ALLOW_MISSING_ESM_EMBEDDINGS = False
PREPARE_MISSING_ESM_EMBEDDINGS = False
VAL_FRACTION = 0.15
SPLIT_BY = "pdbid"
SELECTION_METRIC = "val_metal_balanced_acc"
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
ALLOW_SHORT_TRAINING_FOR_DEBUG = False
```

In the dedicated **Main planned training launch switch** cell, then run
**Optional training execution**:

```python
LAUNCH_PLANNED_MAIN_TRAINING_RUNS = True
```

Expected outputs/files:

- `<RUNS_DIR>/<SUMMARY_BASENAME>_planned_runs.csv`
- `<RUNS_DIR>/<SUMMARY_BASENAME>_planned_run_dictionary.json`
- Nine completed validation-only radius-only run directories under
  `<RUNS_DIR>/`
- Each completed run directory contains `run_config.json`,
  `run_metadata.json`, `split_diagnostics.json`, `dataset_summary.json`, and
  `prepare_status.json`
- `<RUNS_DIR>/<SUMMARY_BASENAME>.csv`
- `<RUNS_DIR>/<SUMMARY_BASENAME>_completed_only.csv`
- `<RUNS_DIR>/<SUMMARY_BASENAME>.png`, when plotting succeeds
- No `test_report.json`

Exact configuration record:

- Per run: `<run_dir>/run_config.json` and `<run_dir>/run_metadata.json`.
- Batch plan: planned-run CSV/dictionary.
- `active_run_config.json` / `active_run_config.md` are generated from the live
  notebook configuration before launch.

Success criteria:

- Planning table shows `RING_EDGE_MODE = "without_ring"` / radius-only graph
  mode for every planned run.
- No test-report files are created by the ablation runs.
- The ablation clarifies validation behavior without relying on one lucky seed.

### Decision gate after Stage 5G

Proceed to Stage 6 only if:

- The ablation was explicitly labeled radius-only and compared against the
  matching RING-enabled family.
- All 9 planned validation-only ablation runs complete.
- The expected planned files, summary files, and run-level JSON files exist.
- No held-out test files were created.
- `val_metal_balanced_acc` is the selection metric on all completed runs.
- Diagnostics report every active metal-scheme class present in both train and validation splits.
- Rare-class recall protection passes: per-class recall is available for each
  completed run, and the ablation is not promoted if any metal class has zero
  seed-mean recall.

If gate fails: do not use the ablation as model-selection evidence. Choose the
top 2-3 valid candidates for grouped-fold confirmation and do not finalize from
a raw single-batch ranking alone.

## Stage 6 - Top-K Seed/Split Confirmation

Purpose: confirm whether top HPO candidates are stable across validation data
partitions and model-initialization seeds. The standard fold-plus-seed Stage 6
design uses `TOP_CONFIG_REEVALUATION_MODE = "group_kfold_seed_repeat"`:
grouped 5-fold validation by `pdbid` crossed with the configured
`REPEAT_SEEDS`; every compared candidate uses the same fold definitions and
seed list. Candidate ranking uses one primary score, the mean selected
validation metric over all completed fold x seed runs. Paired comparisons
remain conservative by averaging common seeds within each fold and bootstrapping
fold-level differences.

When to use it: after a medium or large Optuna search has produced top
candidates, and before any candidate is treated as final-selection evidence.
For a reportable final-selection cycle, resolve and freeze the primary
final-test dataset route before Stage 6 starts. The route determines which
structures belong to the full non-test training set used in Stage 6/6B, so it
cannot be chosen only at Stage 7. With the route currently unresolved, Stage 6
may be exploratory validation evidence but is not yet the final reportable
confirmation cycle.

Expected scale/runtime: serious run, long or overnight. Training count is
roughly `TOP_K_CONFIGS_FOR_SEED_AND_CROSS_FOLD_REPEAT x SEED_REPEAT_N_FOLDS x
len(REPEAT_SEEDS)` in `group_kfold_seed_repeat` mode, or top-K x folds in
plain `group_kfold` mode; runtime then scales with that count and `EPOCHS`.
Optional Stage 6 parallelism is candidate-scoped: for each ranked top-K
candidate, the notebook can run up to
`STAGE6_PARALLEL_CROSS_VALIDATION_PROCESSES` fold/seed validation subprocesses
at once, waits for that candidate's units to finish, and then moves to the
next ranked candidate. The effective worker count is capped by
`SEED_REPEAT_N_FOLDS` and by the number of remaining units for that candidate.
Keep the default `1` for serial/reproducible behavior; set it to
`SEED_REPEAT_N_FOLDS` only after confirming GPU memory headroom for the active
model.

Preferred same-runtime configuration: run the exact Stage 5 HPO block first
with `RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = False`. After HPO finishes, use
the dedicated **Stage 6 controls and existing Optuna/HPO reuse** panel plus the
**Launch Stage 6 top-K grouped-fold confirmation** cell to import the previous
validation-only HPO candidates, write the top-K commands, and optionally launch
grouped-fold confirmation. Keep the Stage 5 block's `MODEL_PRESET`,
`OPTUNA_STUDY_NAME`, `OPTUNA_STORAGE`, search space, and persistent-storage
settings unchanged.

Existing-HPO standalone configuration: when the HPO directory already exists
from a previous Colab/runtime, keep `LAUNCH_PLANNED_MAIN_TRAINING_RUNS = False`,
set the old HPO source in the Stage 6 controls, and use the Stage 6 launch cell.
You can press **Run all** so the required setup/clone cells execute before Stage
6; ordinary main training/HPO remains skipped. This imports saved run metadata
from the old directory and launches only the new Stage 6 fold/seed runs. When
resuming an interrupted Stage 6, keep the values identical and keep
`SKIP_EXISTING_RUNS = True` so completed folds are reused while missing folds
continue. During an incomplete run, Stage 6 may write a provisional
`stage6_partial/` report. That report is safe to inspect and safe for Stage 6B
preview mode, but it is not promotion evidence and does not replace the
canonical Stage 6 files.

Set these values in the dedicated Stage 6 controls panel:

```python
RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = True
TOP_CONFIG_REEVALUATION_MODE = "group_kfold_seed_repeat"
TOP_K_CONFIGS_FOR_SEED_AND_CROSS_FOLD_REPEAT = "auto"
REPEAT_SEEDS = "42"
SEED_REPEAT_N_FOLDS = 5
STAGE6_PARALLEL_CROSS_VALIDATION_PROCESSES = 1
SEED_REPEAT_SPLIT_SEED = 42
STAGE6_RAW_IMPROVEMENT_THRESHOLD = 0.0
ALLOW_SEED_REPEAT_MODEL_PRESET_MISMATCH = False
SKIP_EXISTING_RUNS = True
WRITE_STAGE6_PARTIAL_PROGRESS_REPORTS = False
USE_EXISTING_OPTUNA_TRIALS_FOR_STAGE6 = False
EXISTING_OPTUNA_TRIALS_BASE_RUNS_DIR = "/content/drive/MyDrive/DeepMzyme/notebook_outputs/runs"
EXISTING_OPTUNA_TRIALS_RUN_BATCH_ID = ""
STAGE6_OUTPUT_RUNS_DIR = ""      # blank = sibling <OLD_HPO_RUN_BATCH_ID>_stage6 directory
STAGE6_OVERWRITE_OUTPUT = False  # False = continue compatible output or create; True = replace incompatible output
STAGE6_EPOCHS = 50
STAGE6_DEVICE = "auto"
STAGE6_SELECTION_METRIC = "val_metal_balanced_acc"
```

For same-runtime HPO, leave `USE_EXISTING_OPTUNA_TRIALS_FOR_STAGE6 = False` so
the Stage 6 launch cell imports from the current notebook run directory. For a
previous HPO directory, set `USE_EXISTING_OPTUNA_TRIALS_FOR_STAGE6 = True` plus
`EXISTING_OPTUNA_TRIALS_BASE_RUNS_DIR / EXISTING_OPTUNA_TRIALS_RUN_BATCH_ID`.
If reuse is enabled but the old-HPO source is blank, the controls cell should
warn without crashing; the Stage 6 launch still refuses to import candidates
until a concrete validation HPO source path is provided.

For standalone existing-HPO Stage 6, leave `STAGE6_OUTPUT_RUNS_DIR` blank to
write a sibling output directory named from the old HPO directory. If that
output already contains a matching `stage6_manifest.json`, the notebook can
continue it and reuse completed fold/seed units when `SKIP_EXISTING_RUNS=True`.
If the output exists but the manifest does not match the requested source,
top-K, folds, seeds, metric, epochs, or selected candidates, the launch stops
unless `STAGE6_OVERWRITE_OUTPUT=True`.

In the **Launch Stage 6 top-K grouped-fold confirmation** cell, first preview
the imported candidates and generated commands:

```python
LAUNCH_STAGE6_TOP_K_CONFIRMATION = False
```

Preview mode does not launch missing folds. If compatible Stage 6 fold/seed
run directories already exist and `WRITE_STAGE6_PARTIAL_PROGRESS_REPORTS=True`,
it scans those completed runs and refreshes the `stage6_partial/` progress
report. The current default is `False`: Stage 6 writes canonical files only
when the declared Stage 6 grid is complete, and Stage 6B can reconstruct a
preview-only partial table from existing CV run folders when complete Stage 6
files are absent. After confirming the import report and generated top-K
commands are correct, launch or resume the missing folds with:

```python
LAUNCH_STAGE6_TOP_K_CONFIRMATION = True
```

Notes:

- `TOP_CONFIG_REEVALUATION_MODE = "group_kfold_seed_repeat"` is the explicit
  reportable fold-plus-seed Stage 6 mode.
- `TOP_CONFIG_REEVALUATION_MODE = "group_kfold"` is grouped-fold confirmation
  with only the first `REPEAT_SEEDS` value. Use it for one-seed confirmation or
  backward-compatible reruns where seed crossing was not intended.
- `TOP_K_CONFIGS_FOR_SEED_AND_CROSS_FOLD_REPEAT = "auto"` resolves from
  completed Optuna trials: fewer than 50 completed trials repeats up to 5
  candidates, fewer than 150 repeats up to 10, and 150 or more repeats up to
  20. A predeclared integer is allowed, including 20, but it should be chosen
  before Stage 6 launches. The older notebook variable
  `TOP_K_CONFIGS_FOR_SEED_REPEAT` remains a backward-compatible alias.
- `REPEAT_SEEDS = "42"` is the comma-separated model-initialization seed list.
  In `group_kfold_seed_repeat`, every seed is crossed with every grouped fold.
  In `group_kfold`, only the first listed seed is used. Add more seeds only
  when the resulting `top_k x folds x seeds` training count is practical and
  predeclared.
- `SEED_REPEAT_SPLIT_SEED = 42` fixes the grouped fold definitions. Keep it
  identical for every candidate in the same comparison.
- `STAGE6_PARALLEL_CROSS_VALIDATION_PROCESSES = 1` preserves the original
  serial Stage 6 launch. Values above `1` run fold/seed units in parallel
  within the current ranked candidate only; all units for top-1 finish before
  top-2 starts. On a single G4/T4-class GPU, set this no higher than
  `SEED_REPEAT_N_FOLDS`; the launcher caps the effective worker count at the
  fold count and only a short launch should confirm CUDA memory headroom.
- The legacy `TOP_CONFIG_REEVALUATION_MODE = "seed_repeat"` mode remains
  available for exploratory checks only. It measures combined initialization
  and split variance, not isolated initialization variance.

If the Optuna study is already complete, use the Stage 6 launch cell in preview
mode first and inspect:

```text
<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/top_reevaluation_commands.txt
```

Those commands intentionally omit held-out test evaluation. Launch only the
top-K commands you predeclare, with the same fold definitions for every
compared candidate, and keep the results as validation-only evidence.

Expected outputs/files:

- `<RUNS_DIR>/stage6_manifest.json` for standalone existing-HPO Stage 6
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/stage6_manifest.json` for standalone existing-HPO Stage 6
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/stage6_existing_trials_import_report.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/stage6_existing_trials_import_report.json`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/top_reevaluation_commands.txt`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/seed_repeat_results.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/seed_repeat_summary.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/seed_repeat_summary.json`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/seed_repeat_pairwise_bootstrap.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/seed_repeat_pairwise_bootstrap.json`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/stage6_ranked_candidates.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/stage6_selected_final_candidate.json`
- One validation-only run directory per top-K/fold/seed unit
- Per-fold `run_config.json`, `run_metadata.json`, and
  `split_diagnostics.json`
- No `test_report.json`

When `WRITE_STAGE6_PARTIAL_PROGRESS_REPORTS=True` and Stage 6 is still
incomplete, provisional progress files are written under:

- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/stage6_partial/stage6_partial_manifest.json`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/stage6_partial/stage6_partial_results.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/stage6_partial/stage6_partial_ranked_candidates.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/stage6_partial/stage6_partial_pairwise_bootstrap.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/stage6_partial/stage6_partial_report.md`

These files are explicitly provisional. They include completion counts and are
ignored by Stage 6 resume planning. With the default
`WRITE_STAGE6_PARTIAL_PROGRESS_REPORTS=False`, these provisional files are not
created by Stage 6; use Stage 6B's reconstruction option if a preview table is
needed before all declared seeds complete. Canonical
`stage6_ranked_candidates.csv` and `stage6_selected_final_candidate.json` are
written only after the planned candidate/fold/seed units are complete.

Stage 6 result rows include:

- candidate identifier
- source Optuna study, top rank, trial number, and source run directory
- model seed
- split seed
- fold index / validation unit
- fold unit and model seed when grouped-fold confirmation uses multiple seeds
- validation balanced accuracy
- validation minimum per-class recall
- per-class recall when available
- collapsed-4 balanced accuracy when available
- run directory
- selected checkpoint path

Exact configuration record:

- Grouped-fold summary: CSV/JSON files listed above.
- Stage 6 ranking and frozen final-candidate selection:
  `stage6_ranked_candidates.csv` and
  `stage6_selected_final_candidate.json`.
- Per repeated run: `<run_dir>/run_config.json` and
  `<run_dir>/run_metadata.json`.
- `active_run_config.json` / `active_run_config.md` are generated from the live
  notebook configuration before launch.

Success criteria:

- All predeclared top-K/fold/active-seed runs complete.
- The number of completed validation-only runs equals
  the resolved `TOP_K_CONFIGS_FOR_SEED_AND_CROSS_FOLD_REPEAT` value times
  `SEED_REPEAT_N_FOLDS` times `len(REPEAT_SEEDS)` for
  `group_kfold_seed_repeat`, or times one active model seed for `group_kfold`.
- Every candidate has the same fold definitions and the same active model-seed
  list.
- No held-out test files were created.
- Candidate ranking uses validation/CV metrics only: highest mean
  `val_metal_balanced_acc` over all completed fold x seed runs, then higher
  mean `val_metal_min_recall`, then lower fold-to-fold standard deviation of
  seed-averaged `val_metal_balanced_acc`, then a simpler/smaller model if still
  tied.
- `stage6_selected_final_candidate.json` records the selected configuration
  ID, selected source run directories, model preset, selected hyperparameters,
  and ranking metrics. It is the selection evidence for the required final
  training/refit run; a raw Stage 6 fold checkpoint is not the preferred primary
  final-test source.
- Pairwise comparisons use paired bootstrap over fold-level differences with
  10,000 resamples. When multiple seeds are configured, the notebook averages
  common seeds within each fold before bootstrapping. Candidate A beats
  candidate B only if mean A-B is positive, the 95% CI excludes zero on the
  positive side, and the raw improvement meets the applicable threshold.
- Diagnostics do not show leakage, missing active metal-scheme validation classes, or invalid
  feature coverage.

### Decision gate after Stage 6

Proceed to Stage 6B, and only then Stage 7, if:

- The Stage 6 success criteria are met.
- `stage6_existing_trials_import_report.csv`,
  `stage6_existing_trials_import_report.json`, and
  `top_reevaluation_commands.txt` exist and point to validation-only HPO
  candidates.
- `seed_repeat_results.csv`, `seed_repeat_summary.csv`,
  `seed_repeat_summary.json`, `seed_repeat_pairwise_bootstrap.csv`,
  `seed_repeat_pairwise_bootstrap.json`, `stage6_ranked_candidates.csv`, and
  `stage6_selected_final_candidate.json` exist.
- The selected candidate has all planned fold/active-seed units completed.
- No held-out test files were created anywhere in the validation chain.
- `val_metal_balanced_acc` is the selection metric on all completed runs.
- Diagnostics report every active metal-scheme class present in both train and validation splits.
- Rare-class recall protection passes: the selected candidate has available
  mean per-class recall across folds and acceptable `val_metal_min_recall`; no
  metal class has zero mean recall.
- Any claimed improvement over a comparator is supported by the paired
  bootstrap 95% CI and the relevant raw-improvement threshold.
- One final configuration is selected using validation/CV evidence only.
- The exact Stage 6 source run directories/checkpoints and selected
  configuration are recorded before Stage 6B launch.
- Stage 6B promotion/refit policy is declared before running it: paired-CI
  thresholds, rare-recall thresholds, tie-breakers, fixed final-refit seed,
  epoch/checkpoint rule, and output folder. None of these may be changed after
  held-out test metrics are seen.

If gate fails: report the top candidates, paired bootstrap rows,
`val_metal_min_recall`, per-class recall, split seed, fold indices, and epoch
budget. Do not launch held-out test evaluation.

## Stage 6B - Promotion Gates And Final Full-Train Refit

Purpose: convert Stage 6 validation/CV evidence into one frozen final model.
Stage 6B ranks candidates by mean `val_metal_balanced_acc`, applies the
predeclared paired-CI, rare-class recall, and tie-breaker policy, then
optionally trains the selected configuration once on the full non-test training
set. Stage 6B does not open the held-out test set.

When to use it: only after Stage 6 has completed and the Stage 6 decision gate
above passes.

Set these values in the dedicated **Stage 6B - promotion gates and final
full-train refit** cell:

```python
RUN_STAGE6B_FINAL_SELECTION = True
LAUNCH_STAGE6B_FINAL_REFIT = False

# Stage 6B uses STAGE6_OUTPUT_RUNS_DIR from the Stage 6 controls.
# Blank = current RUNS_DIR in same-runtime mode, or auto sibling
# <OLD_HPO_RUN_BATCH_ID>_stage6 in standalone existing-HPO mode.
# Backward-compatible explicit optuna/<study> override:
STAGE6B_STAGE6_OPTUNA_DIR = ""

STAGE6B_RECONSTRUCT_PARTIAL_FROM_COMPLETE_SEED_CV = False
STAGE6B_RECONSTRUCT_PARTIAL_SOURCE_RUNS_DIR = ""  # blank = infer Stage 6 output root from STAGE6_OUTPUT_RUNS_DIR/current source
STAGE6B_ALLOW_COMPLETE_SEED_BLOCK_REFIT = False

STAGE6B_RANK_BY_METRIC = "mean_val_metal_balanced_acc"
STAGE6B_TIE_EPSILON = 0.002
STAGE6B_TIE_BREAKERS = "mean_val_metal_min_recall_desc,min_validation_metric_desc,std_val_metal_balanced_acc_asc,model_complexity_proxy_asc"

STAGE6B_REQUIRE_PAIRED_CI_IMPROVEMENT = True
STAGE6B_MIN_RAW_IMPROVEMENT = 0.0
STAGE6B_MIN_CI_LOWER_BOUND = 0.0
STAGE6B_ALLOW_SINGLE_CANDIDATE_WITHOUT_CI = False
STAGE6B_ALLOW_TIE_BREAK_WITHOUT_POSITIVE_CI = True

STAGE6B_BLOCK_ON_MISSING_RARE_RECALL = True
STAGE6B_MIN_MEAN_MIN_RECALL = 0.0
STAGE6B_MIN_WORST_MIN_RECALL = 0.0
STAGE6B_MIN_PER_CLASS_MEAN_RECALL = 0.0
STAGE6B_MAX_MEAN_MIN_RECALL_DROP_VS_COMPARATOR = 0.03

STAGE6B_FINAL_REFIT_EPOCHS = 50
STAGE6B_FINAL_REFIT_SEED = 42  # fixed protocol seed; not a Colab UI input
STAGE6B_FINAL_REFIT_DEVICE = "auto"
STAGE6B_FINAL_REFIT_RUN_NAME_PREFIX = "stage6b_final_refit"
STAGE6B_REUSE_EXISTING_REFIT_RUN = True
```

First run Stage 6B in preview mode with `LAUNCH_STAGE6B_FINAL_REFIT = False`.
Inspect the ranked table, `stage6b_decision.json`, and
`stage6b_final_refit_command.txt`. If the candidate is approved and the command
matches the selected configuration, rerun the same cell with:

```python
LAUNCH_STAGE6B_FINAL_REFIT = True
```

If an old Stage 6 CV run directory predates canonical/partial table writing,
Stage 6B can reconstruct the tables from completed Stage 6 CV run folders. Set
`STAGE6B_RECONSTRUCT_PARTIAL_FROM_COMPLETE_SEED_CV = True` only when you need to
rebuild missing canonical Stage 6 files from completed CV run folders. If
canonical completed Stage 6 files are missing, Stage 6B scans
`STAGE6B_RECONSTRUCT_PARTIAL_SOURCE_RUNS_DIR`, or the Stage 6 output root from
`STAGE6_OUTPUT_RUNS_DIR` when that source is blank. For a standalone
existing-HPO Stage 6 run with blank `STAGE6_OUTPUT_RUNS_DIR`, Stage 6B infers
the auto sibling `<OLD_HPO_RUN_BATCH_ID>_stage6` output root.

Reconstruction supports both current canonical run folders with
`run_config.json` / `run_metadata.json` and older Study6-style active-config run
folders with `active_run_config.json` plus validation metric CSVs. Discovery is
validation-only: folders with `test_report.json` are skipped, `train_metrics.csv`
is not used for candidate ranking, and broad parent directories are ignored.

When a Stage 6 manifest exists and reconstruction proves every manifest-declared
candidate x fold x active-seed unit is present, Stage 6B writes the canonical
Stage 6 files (`seed_repeat_results.csv`, `stage6_ranked_candidates.csv`,
`stage6_selected_final_candidate.json`, and related JSON/CSV summaries), then
normal Stage 6B promotion/refit may proceed. If the manifest is missing or any
declared unit is missing, the default behavior is still to block reportable
Stage 6B/refit and write `stage6_partial/` diagnostics only.

If you explicitly accept a complete-seed-block subset, set
`STAGE6B_ALLOW_COMPLETE_SEED_BLOCK_REFIT = True` with reconstruction enabled.
In that mode Stage 6B drops any candidate/seed block missing one or more
required folds, ranks the remaining candidate/seed blocks that contain the full
fold set, and may launch the final refit from the winning configuration. The
decision JSON records `stage6_evidence_mode = "complete_seed_block_subset"` and
lists the dropped incomplete blocks. This mode is validation-only and still does
not open the held-out test set.

Promotion policy:

- Primary ranking metric is `mean_val_metal_balanced_acc`.
- Paired-CI promotion is checked first. When the selected candidate beats the
  comparator with paired mean improvement at least
  `STAGE6B_MIN_RAW_IMPROVEMENT` and paired 95% CI lower bound greater than
  `STAGE6B_MIN_CI_LOWER_BOUND`, it is promoted as CI-supported.
- If the top candidates are inside `STAGE6B_TIE_EPSILON`, the default
  `STAGE6B_ALLOW_TIE_BREAK_WITHOUT_POSITIVE_CI = True` allows Stage 6B to
  resolve the practical tie with the predeclared tie-breakers. This does not
  claim a CI-supported improvement.
- Rare-class recall protection blocks promotion when required recall values are
  missing, below configured absolute minima, or when mean
  `val_metal_min_recall` drops by more than
  `STAGE6B_MAX_MEAN_MIN_RECALL_DROP_VS_COMPARATOR` versus the comparator.
- Tie-breakers are predeclared by `STAGE6B_TIE_BREAKERS`; the default order is
  higher mean minimum recall, higher worst-fold validation metric, lower
  fold-to-fold standard deviation, then simpler model.
- `STAGE6B_ALLOW_TIE_BREAK_WITHOUT_POSITIVE_CI` should stay `True` unless the
  specific analysis requires strict CI-only promotion. With the default, Stage
  6B still records whether the selected candidate was CI-supported or selected
  by tie-break fallback.

Final refit policy:

- The refit uses `VAL_FRACTION = 0.0`, no k-fold split, and
  `selection_metric = "train_loss"` because it trains on the full non-test
  training set.
- The checkpoint rule is fixed before launch: use the best train-loss
  checkpoint from the final refit. The held-out test is not evaluated during
  Stage 6B.
- For reportable runs, the final refit seed is not a policy choice and is not
  exposed as a Colab input. The notebook uses the fixed, predeclared integer
  `STAGE6B_FINAL_REFIT_SEED = 42`, so the final model seed is known before
  refit and cannot be chosen after looking at held-out metrics.

Expected outputs/files:

- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/stage6b_ranked_candidates.csv`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/stage6b_decision.json`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/stage6b_final_refit_command.txt`
- `<RUNS_DIR>/optuna/<OPTUNA_STUDY_NAME>/stage6b_selected_final_refit_candidate.json`
  after a completed or reused final refit
- `<stage6b_final_refit_run_dir>/active_run_config.json`
- `<stage6b_final_refit_run_dir>/active_run_config.md`
- `<stage6b_final_refit_run_dir>/run_config.json`
- `<stage6b_final_refit_run_dir>/run_metadata.json`
- `<stage6b_final_refit_run_dir>/best_model_checkpoint.pt`
- No `test_report.json`

Decision gate after Stage 6B:

- `stage6b_decision.json` status is `selected_for_final_refit`.
- Stage 6B used canonical completed Stage 6 artifacts, or explicitly recorded
  `stage6_evidence_mode = "complete_seed_block_subset"` after dropping
  incomplete candidate/seed blocks and keeping only complete fold sets.
- `stage6b_selected_final_refit_candidate.json` exists and records
  `protocol_stage = "Stage 6B"`, `selected_before_held_out_test_evaluation =
  True`, `held_out_test_metrics_used = False`, and
  `final_training_refit.status` as `completed` or `existing`.
- The selected final-refit run directory exists and contains
  `run_config.json`, `run_metadata.json`, and a checkpoint.
- The final-refit run used the selected Stage 6 configuration, full non-test
  training set, fixed final-refit seed, fixed epoch budget, and no held-out test
  evaluation.
- Stage 7 points to `stage6b_selected_final_refit_candidate.json` for the
  primary report; if that file is absent, Stage 7 must not be launched as the
  reportable primary route.

If gate fails: stop at validation evidence. Do not open the held-out test set.

### Recommended Stage 6 candidate policy

For each completed Optuna study, repeat the auto-selected top candidates across:

```python
TOP_CONFIG_REEVALUATION_MODE = "group_kfold_seed_repeat"
TOP_K_CONFIGS_FOR_SEED_AND_CROSS_FOLD_REPEAT = "auto"
REPEAT_SEEDS = "42"
SEED_REPEAT_N_FOLDS = 5
STAGE6_PARALLEL_CROSS_VALIDATION_PROCESSES = 1
SEED_REPEAT_SPLIT_SEED = 42
RUN_TOP_CONFIG_SEED_REPEAT_VALIDATION = True
RETRAIN_BEST_CONFIG_AFTER_HPO = False
INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False
```

The default auto rule repeats up to 5 candidates below 50 completed trials, up
to 10 below 150 completed trials, and up to 20 for 150 or more completed
trials. Use a fixed integer only when the top-K count is predeclared before
launch; integer 20 is allowed for serious large HPO and with the default single
seed implies `20 x 5 x 1 = 100` extra validation-only runs.

A candidate is considered stable enough for final selection only if:

- all planned grouped-fold/active-seed runs complete, or failures are explained
  and not biased;
- no held-out test report was created;
- all source runs use the same `METAL_LABEL_SCHEME`;
- all runs use `selection_metric = val_metal_balanced_acc`;
- no active metal-scheme validation class is missing;
- rare-class recall is acceptable;
- paired bootstrap comparisons support the improvement over the relevant
  comparator.

## Stage 7 - One-Shot Held-Out Test

Purpose: report held-out test performance for the frozen Stage 6B final
full-train refit produced from the Stage-6-selected configuration. Stage 7 is
the only stage that may open the held-out test set. The primary final report
must be declared before test evaluation starts.

When to use it: only after model family, hyperparameters, Stage 6
interpretation, Stage 6B promotion decision, final-refit run, final source
checkpoint, and the scientifically approved primary final-test dataset route
are fixed. The dataset route is currently unresolved; do not launch this stage
until `docs/DATASETS.md` and `EXPERIMENT_STATUS.md` record the decision and
frozen provenance. The canonical cell is fail-closed with internal
`FINAL_TEST_PRIMARY_DATASET_ROUTE_STATUS = "unresolved"`; it must be changed
only as part of that coordinated decision, not as an ad-hoc run setting.

Expected scale/runtime: final reporting run, usually minutes to hours depending
on checkpoint loading and evaluation mode.

The cleaned notebook resolves the final source inside the **Optional final
held-out test evaluation** cell; it does not have a separate live “Select final
run” cell. The selected source must be the Stage 6B final full-train refit
derived from the Stage 6 selected configuration, not a raw Optuna trial or an
arbitrary Stage 6 fold checkpoint. The primary Stage 7 cell requires
`stage6b_selected_final_refit_candidate.json`; there is no Stage 6 fallback.
Its semantic pre-execution gate verifies the Stage 6/6B decision links,
selected configuration identity, completed/reused full non-test refit status,
source config/metadata, checkpoint, and absence of prior test evidence before
it inspects held-out input paths.

Then run the **Optional final held-out test evaluation** cell with launch still
disabled:

```python
FINAL_TEST_WORKFLOW = "evaluate_stage6_selected_candidate"
LAUNCH_FINAL_HELD_OUT_TEST_EVAL = False
```

Inspect the printed pre-flight checklist. The final-test cell supports exactly
one workflow value:

- `evaluate_stage6_selected_candidate`: primary serious final-test mode. It
  requires `stage6b_selected_final_refit_candidate.json` as final-source
  evidence for the primary route and must resolve to the frozen Stage 6B
  final-refit run for the Stage-6-selected rank #1 configuration. The output
  role is `primary_preselected`.

The former `exploratory_evaluate_all_stage6_ranked_candidates` option is
disabled in the canonical primary-test cell. Evaluating every ranked candidate
would consume the primary held-out set repeatedly and create a test-informed
comparison even if labeled post-hoc. Any future diagnostic comparison requires
a separately approved secondary dataset/protocol and must remain outside the
primary Stage 7 workflow.

For the single-checkpoint primary report, if the Stage 6 selection evidence,
Stage 6B decision, and Stage 6B final-refit run are all frozen and point to the
same selected configuration, switch to launch:

```python
FINAL_TEST_WORKFLOW = "evaluate_stage6_selected_candidate"
LAUNCH_FINAL_HELD_OUT_TEST_EVAL = True
```

Expected outputs/files:

- One new primary final-test run folder under the resolved `RUNS_DIR`
- `run_config.json` and `run_metadata.json` in the final-test output folder
- `test_report.json` in the final-test output folder
- `<final_run_dir>/test_predictions.pt`
- `<final_run_dir>/test_temperature_validation_predictions.pt`, when
  validation logits are available for temperature fitting
- `<final_run_dir>/test_reliability_diagram.png`
- `<final_run_dir>/test_confidence_histogram.png`
- `<final_run_dir>/test_temperature_scaled_reliability_diagram.png`, when
  temperature scaling is available
- `<final_run_dir>/test_temperature_scaled_confidence_histogram.png`, when
  temperature scaling is available
- Updated final-test summary CSV/PNG when plotting succeeds
- The source Stage 6B final-refit run remains unchanged

Exact configuration record:

- Stage 6/6B selection evidence: `stage6_selected_final_candidate.json`,
  `stage6_ranked_candidates.csv`, paired-bootstrap outputs,
  `stage6b_decision.json`, `stage6b_ranked_candidates.csv`, and
  `stage6b_selected_final_refit_candidate.json`.
- Source Stage 6B final-refit run: `<source_run_dir>/run_config.json` and
  `<source_run_dir>/run_metadata.json`.
- Final-test output: `<final_run_dir>/run_config.json`,
  `<final_run_dir>/run_metadata.json`, and `<final_run_dir>/test_report.json`.
- `active_run_config.json` / `active_run_config.md` are generated from the live
  notebook configuration before launch.

`test_report.json` schema additions:

- `task`
- `final_test_primary_report`
- `final_test_ensemble_mode`
- `final_test_result_role`
- `selected_config_id` / `selected_run_id`
- `checkpoint_paths`
- `seed_values`
- `run_directories`
- `test_structure_dir` / `test_summary_csv`
- `metrics`
- `calibrated_metrics`
- `fitted_temperatures`
- `temperature_scaling`
- `bootstrap_settings`
- CI fields such as `test_metal_balanced_acc_ci95`,
  `test_metal_collapsed4_balanced_acc_ci95`, and per-class recall CI fields
- `calibration_settings`
- `calibration_plot_paths`
- `reliability_diagram_path`
- `confidence_histogram_path`
- `prediction_artifact_path`
- `timestamp`
- `git_commit` / `code_version`
- explicit `selection_policy_statement` that no test metric was used for
  selection

Calibration and temperature scaling:

- Overall ECE uses predicted-class confidence with 15 equal-mass bins.
- Class-wise ECE uses one-vs-rest class probabilities with equal-mass bins.
- NLL is reported when metal probabilities are available.
- Temperature scaling fits one scalar only on validation logits from the
  validation-selected configuration or Stage 6B final-refit source run, then
  applies that temperature to held-out test logits.
- Ensemble temperature scaling uses the fixed rule: fit one scalar temperature
  per fixed checkpoint on that checkpoint's validation logits, apply it to that
  checkpoint's test logits, then average the calibrated softmax probabilities.
- No temperature, calibration method, seed subset, ensemble weight, threshold,
  checkpoint, or primary report can be selected using held-out test metrics.

Bootstrap confidence intervals:

- Default: 1000 stratified bootstrap resamples, 95% confidence intervals,
  `FINAL_TEST_BOOTSTRAP_SEED = 20260518`.
- Stratification is by true class, preserving every class present in the
  original held-out test labels.
- Report CIs for active-scheme balanced accuracy, per-class recall, collapsed-4
  balanced accuracy and collapsed per-class recall, plus ECE when available.

Success criteria:

- The source run is recorded in
  `stage6b_selected_final_refit_candidate.json` with Stage 6B validation/CV
  promotion evidence.
- The final-test run uses `best_model_checkpoint.pt` from the Stage 6B final
  full-train refit, or an explicitly selected fixed checkpoint recorded before
  Stage 7.
- The source run is the Stage 6B final-refit run derived from the Stage 6
  selected configuration, not a raw Optuna trial or arbitrary Stage 6 fold.
- The output folder is separate from the source Stage 6B final-refit run.
- Primary mode loads the selected candidate from
  `stage6b_selected_final_refit_candidate.json` and does not inspect or
  evaluate other candidates.
- Raw filename/structure/PDB/PDB-chain overlap is blocked before model
  preparation or inference, and loaded-pocket/group overlap is blocked again
  before graph construction.
- The test report includes active metal-scheme metrics and collapsed-4 metrics.
- The result is labeled as the preselected primary report.

### Decision gate after Stage 7

Final reporting is complete only if:

- The Stage 7 success criteria are met.
- The source run was the frozen Stage 6B final-refit run derived from the Stage
  6 validation-selected configuration.
- `val_metal_balanced_acc` was the Stage 6/6B selection metric; the Stage 6B
  full-train refit may use `train_loss` only as the predeclared checkpoint rule
  for the full non-test training run.
- The final-test output is a separate folder from the source final
  training/refit run.
- `test_report.json`, `run_config.json`, and `run_metadata.json` exist in the
  final-test output folder.
- `test_report.json` records `final_test_result_role` / `role`, selected
  configuration identity, checkpoint path(s), seed values, run directories,
  calibration settings, bootstrap settings, plot paths, and the
  no-test-selection policy statement.
- Temperature scaling, if present, was fitted only on validation logits from
  the fixed validation-selected configuration.
- Bootstrap CI fields are present for the requested final-test metrics, or the
  report explicitly records why CIs were disabled.
- This is the first and only Stage 7 launch for this validation-selected
  configuration and Stage 6B final-refit run; keep
  `ALLOW_REPEAT_FINAL_TEST_EVAL = False` unless explicitly documenting a
  non-reportable rerun.
- The Stage 7 result is not used to pick a different checkpoint, seed,
  hyperparameter set, model family, fusion mode, ensemble subset, ensemble
  weight, calibration method, temperature, threshold, or primary report.

If gate fails: do not report the run as final. If the test completed, treat the
one-shot final-test result as already spent for that selection cycle. Do not
choose a different configuration because another tested candidate has a better
held-out test score; return to validation-only experiments for new development.

## Safety Guards To Check

Before any reportable comparison or HPO launch, confirm:

- `INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False`
- A new target-formulation campaign uses fully reconciled paired stage blocks:
  a direct `METAL_LABEL_SCHEME = "four_class"` arm and a separately named
  `six_class` arm with collapsed-four evaluation. The retained six-class common
  recipe alone cannot satisfy this requirement until TECH-010 is resolved.
- `FINAL_TEST_WORKFLOW = "evaluate_stage6_selected_candidate"` for primary
  final reporting, or the final-test cell has not been run
- `LAUNCH_FINAL_HELD_OUT_TEST_EVAL = False` until the separate Stage 7 cell is
  intentionally launched
- `stage6_selected_final_candidate.json` exists as Stage 6 evidence, and
  `stage6b_selected_final_refit_candidate.json` exists as the completed Stage
  6B final-refit source before primary Stage 7 launch
- the primary final-test dataset route is scientifically approved and frozen;
  the current unresolved route is a hard stop before Stage 7
- no all-ranked-candidate evaluation is available against the primary held-out
  set; future diagnostics require a separately approved secondary protocol
- `VAL_FRACTION > 0` or a fold split is explicitly configured for validation
  stages; Stage 6B is the only reportable full-train refit path with
  `VAL_FRACTION = 0.0`
- `SELECTION_METRIC = "val_metal_balanced_acc"` for metal model selection
- `RUN_BATCH_ID` identifies the experiment batch clearly, and the default
  `SUMMARY_BASENAME` is derived from live provenance rather than stale manual
  labels
- `ALLOW_SHORT_TRAINING_FOR_DEBUG = False` for reportable runs
- `ALLOW_SEED_REPEAT_MODEL_PRESET_MISMATCH = False`
- `ALLOW_MIXED_FINAL_TEST_BATCH = False`
- `ALLOW_REPEAT_FINAL_TEST_EVAL = False`

The training code requires the explicit final-refit test-evaluation flag and a
predeclared selected-configuration ID for every reportable held-out run. The
canonical notebook additionally verifies the Stage 6B provenance artifact.
Raw structure overlap is checked before preparation/inference and loaded-pocket
overlap is checked before graph construction. The separate explicit debug flag
is non-reportable and must never be pointed at the primary held-out set.
