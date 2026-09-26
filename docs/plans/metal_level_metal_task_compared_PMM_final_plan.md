# Final execution plan: metal-level prediction and PMM comparison

**Status:** original design reviewed 2026-09-25; execution order amended by the user on 2026-09-26. Current completion and allocation records belong in `EXPERIMENT_STATUS.md`.
**Repository originally reviewed:** `63d4cd52bf98b2448d42f4498364828431c962eb`.

## Immediate execution amendment — ESMC pair first

The user approved a small exploratory screen before further grouped-fold work.
This amendment supersedes the immediate execution order and scope of Step 6;
it does not claim the full comparison matrix is complete.

- Reuse the completed ordinary Only-ESM/direct-four fold-0 fit. Run exactly one
  matching `first_shell_bias` fit on the same frozen fold and model seed, with
  the existing 50-epoch recipe unchanged. Reuse certified inputs, embeddings,
  caches and source; no ESMC fine-tuning, HPO or new data preparation.
- Require independent checkpoint replay, exact validation-ion identity matching
  and verified host backup. Report paired balanced accuracy, macro-F1, every
  class recall, confusion matrices, selected epochs, learning curves, learned
  biases and empty-shell coverage. Label all findings exploratory single-fold
  evidence; do not promote a model or invoke the full-grid assessment/refit gates.
- The approved immediate allocation is one bounded one-hour VM session using
  the existing controller and admission safeguards. The larger proposed
  30-hour/$34 continuation is unnecessary for this screen and remains unapproved.
- Defer the remaining 45-fit grid. A later approved broader screen can evaluate
  all nine configurations on the same fold, reusing completed fits. For a
  binding-awareness claim, confirm both the shortlisted aware model and its
  ordinary control on the remaining four frozen folds, then compare on matched
  folds with PMM. Disclose that fold 0 informed screening; do not equate this
  controlled comparison with exact reproduction of PMM's published protocol.
- Keep reference/test inputs closed. The final-refit and reporting gates remain
  intact. One coordinator executes this single-fit stage; additional agents are
  unnecessary. Exact commands and outputs belong in the metal playbook.

The source proposals were read completely during the review and are preserved
in Git at the reviewed revision:
[Antigravity proposal](https://github.com/MECHTI1/DeepMzyme/blob/63d4cd52bf98b2448d42f4498364828431c962eb/docs/plans/metal_level_metal_task_compared_PMM_antigravity_plan.md)
and
[Opus proposal](https://github.com/MECHTI1/DeepMzyme/blob/63d4cd52bf98b2448d42f4498364828431c962eb/docs/plans/metal_level_metal_task_compared_PMM_opus_generated_plan.md).
This final plan supersedes both proposals; their working-tree copies were removed
after consolidation. The historical links retain their original content.
Their assertions about held-out results and held-out subgroups were not independently investigated or used to choose this plan. No held-out dataset files, historical run-output artifacts, model checkpoints, or mixed train/test crosswalk were opened for this review. Checks used source code, policy documentation, the training-side manifest, and PMM's released training code; quoted benchmark figures in these texts were not selection evidence.

## 1. Objective, authority, and scope

Produce a reproducible comparison of **metal identity at a known individual ion coordinate** on the reconstructed Zenodo PMM cohort. Establish matched Only-ESM, Only-GVP, and GVP plus graph-level late-fusion baselines; compare direct four-class training with six-class training collapsed to the same four-class endpoint; then test one narrowly defined binding-residue readout modification in each family.

The latest user instructions and applicable repository instructions govern this work. [`Plan.md`, sections 2, 7, and 8](../../Plan.md) explicitly require the paired target-formulation comparison, validation-only selection, and a final full-train refit before held-out reporting. Existing five-class runner defaults do not supersede that policy.

The original experiment request and an independently attributable record of the alleged earlier five-class clarification were **not available in the inspected repository material**. Consequently:

- **Confirmed requirements:** metal-only scope, correct ion/example identity, PMM mapping, matched comparisons, leakage protection, and the repository's direct-four versus collapsed-six comparison.
- **Inferred objective, adopted here:** the shared proposal to test binding-residue awareness on the Zenodo source-row cohort. Improvement is a hypothesis, not a required outcome.
- **Chosen execution proposals, not recovered user requirements:** the single readout mechanism below, explicit ESMC-600M embeddings, fixed recipes, and a bounded fivefold campaign. Neither source plan establishes authorization to run every arm on test, use a particular VM, or override scientific policy.

**Exclusions:** pocket-level EC prediction, joint learning, EC filtering or supervision, metal-site discovery/localization, new datasets, a homology-search campaign, early/hybrid/cross-attention architecture searches, RING ablations, nuclearity/site-descriptor models, calibration, ensembles, and broad HPO. The wider metal architecture matrix remains a separate research obligation; this task cannot establish its fusion-position or RING conclusions. Historical pocket-unit runs are not matched controls.

## 2. Independent verdicts

**Antigravity:** useful emphasis on ion anchoring, a small family comparison, and durable execution. Its scientific argument is overstated: multiple ions anywhere in one structure do not establish a multinuclear center, identical inputs, conflicting labels, or a performance ceiling. It substitutes five-class supervision, treats reconstructed membership as sufficient for paper-score parity, proposes test evaluation during model development, and omits the final-refit bridge. Its proposed `first_shell` structural-readout value is unsupported, its ESMC-600M dimension is wrong, and a parse cache already exists. Its approval label is not execution authority.

**Opus:** stronger treatment of provenance, input coverage, label-free conditioning, checkpoint replay, and correlated examples. However, its measured coverage/subgroup claims remain assertions unless reproduced under the permitted data boundary. It also retains five-class training, makes PDB grouping optional, substitutes true early-plus-late hybrid for the required late-fusion baseline, and omits the full-train refit. Its mechanism screen, descriptors, combined arms, calibration, and infrastructure experiments add substantial work without resolving those prerequisites. The ESMC-600M dimension is correct, but the repository actually defaults to ESMC-300M. Neither plan is executable unchanged.

## 3. Material decisions and evidence

| Issue/source positions | Final decision and reason | Verified evidence |
|---|---|---|
| Unit/ambiguity: Antigravity treats structure-wide multiplicity as interference; Opus distinguishes nearby ions. | One resolved physical target ion, normally corresponding to one PMM source row. Shared residues do not merge examples; a pocket is context and a PDB is a split group. | `metal_ion_examples()` in [`src/training/metal_examples.py`](../../src/training/metal_examples.py) constructs `metal_coords=[coord]` and retains `parent_pocket_id`. |
| Source identity: both assume reconstructed counts certify the usable cohort. | Add a narrow, opt-in manifest binding; preserve source UIDs through loading, splits, and predictions. Audit duplicates and ambiguous coordinates before fitting. | [`training/site_filter.py`](../../src/training/site_filter.py), `load_allowed_site_metal_labels()` overwrites duplicate dictionary keys; `matched_site_keys_for_pocket()` ignores insertion codes. The ordinary loader consumes the summary CSV, not the source manifest. |
| Target: both propose five-class; Opus records the policy conflict. | Required direct `four_class` and separate `six_class` arms. Select each checkpoint on native validation BA, then compare the common-four view from that checkpoint. | [`Plan.md`, section 2](../../Plan.md); [`src/label_schemes.py`](../../src/label_schemes.py). |
| Combined model: Antigravity means late fusion; Opus means true hybrid. | Use `gvp` with `fusion_mode=late_fusion`, early ESM disabled. Do not call it the implemented `hybrid` mode. | [`model_variants/factory.py`](../../src/model_variants/factory.py), `_apply_fusion_defaults()`; [`scripts/run_metal_5fold_cv.py`](../../scripts/run_metal_5fold_cv.py), `MODEL_CONFIGS`. |
| Folds: Antigravity claims paper parity; Opus makes PDB grouping optional. | PDB-grouped fivefold validation is mandatory. Freeze splits independently of target scheme and model seed. Parent-pocket grouping alone permits other chains/sites of the same PDB to cross folds. | [`training/splits.py`](../../src/training/splits.py), `pocket_split_key()`, `task_label_keys_for_pocket()`; existing `--split-seed` and `--split-stratify-by metal_site`. |
| Awareness: Antigravity favors proximity weighting; Opus screens several mechanisms. | One optional first-shell logit bias in readout, with a neutral initialization. No new descriptors, message-passing topology, or cutoff search. | [`model.py`](../../src/model.py), `AttentionPool`, `ESMGraphEncoder`, `pool_graph_states`; [`graph/shell_roles.py`](../../src/graph/shell_roles.py). |
| Preparation: Antigravity proposes a new cache; Opus highlights missing inputs. | Verify actual residue-aligned ESM coverage; intentionally omit external feature channels. Reuse the existing parse cache with immutable feature inputs. | [`esm_feature_loading.py`](../../src/training/esm_feature_loading.py); [`parallel_loading.py`](../../src/training/parallel_loading.py), `resolve_parse_cache_dir()`. Feature contents are currently keyed by path, not their own checksum. |
| Reporting: Opus identifies replay/bootstrap gaps; neither fixes independently maximized metrics. | Fix checkpoint-bound reporting and export real out-of-fold predictions. Do not compare independent metric maxima or assume prediction order identifies examples. | Runner `summarize_single_fold()` selects native and collapsed maxima independently; `compute_oof_cv_metrics()` averages scalar summaries. [`export_validation_predictions.py`](../../src/export_validation_predictions.py) omits `metal_example_unit` when reloading. |
| Test: Antigravity opens it during development; Opus postpones it but uses fold models without refitting. | Rerun PMM on matched development examples. Paper numbers remain contextual absent parity. Any later test comparison follows selection and a full-train refit. | [`docs/VERY_EXACT_PMM_SETS_PLAN.md`](../VERY_EXACT_PMM_SETS_PLAN.md), comparison protocol; [`Plan.md`, section 8](../../Plan.md). |

The endpoint is **Mn, Cu, Zn, Class VIII = Fe + Co + Ni**. Use `four_class` / `merge_fe_class_viii` with indices `[Mn=0, Cu=1, Zn=2, Class VIII=3]`; use `six_class` / `split_all_metals` with `[Mn=0, Cu=1, Zn=2, Fe=3, Co=4, Ni=5]`. Historical five-class “Class VIII” means Co+Ni in this implementation and must not be confused with the four-class endpoint.

### Additional verified facts and limits

- The training-side `site_manifest.csv` contains **7,911 rows and 4,191 unique PDB IDs**, with no repeated `source_uid` or `(pdbid, chain, resseq)` key in this limited manifest check. This conflicts with the 4,496 train-PDB count in [`docs/DATASETS.md`](../DATASETS.md). Counts do not establish coordinate completeness or the number of examples surviving the loader.
- Training manifest mapping is `1 → Mn`, `6 → Cu`, `7 → Zn`, `2 → Fe/Co/Ni`. Native counts are Mn 2,586; Cu 400; Zn 2,300; Fe 1,759; Co 311; Ni 555. These are reconstruction-manifest observations; bind them to the pinned original **training** source before certification.
- The manifest header lacks explicit model, insertion-code, atom/altloc, and coordinate fields. [`graph/structure_parsing.py`](../../src/graph/structure_parsing.py), `collect_structure_residues_and_metals()`, traverses all models, while ESM residue alignment uses the first model. Multi-model and insertion-code cases therefore need explicit resolution, not an assumption of uniqueness.
- Ion graphs already use a 10 Å target-centered residue neighborhood. Shell roles are two flags, with first shell defined by the existing donor-distance proxy at 2.7 Å. That proxy is not an experimentally complete ligand annotation. `structural_readout_scope` accepts residue/metal-node scopes, not `first_shell`.
- Only-ESM pools `x_esm` and a site-statistics branch; the ion-mode base site statistics are constant. Its pocket membership already uses structure. The awareness variant adds geometric conditioning and must be labeled accordingly, rather than advertised as purely sequence-only.
- [`embed_helpers/esmc.py`](../../src/embed_helpers/esmc.py) defaults to `esmc_300m`; [`esm_feature_loading.py`](../../src/training/esm_feature_loading.py) defaults to 960 dimensions. Inspection of the installed SDK's `esm.pretrained.ESMC_600M_202412` confirms 1,152 dimensions for 600M. Specify both model and dimension explicitly; Opus's example embedding command omits the model override.
- [`scripts/verify_zenodo_pmm_ion_dataset.py`](../../scripts/verify_zenodo_pmm_ion_dataset.py) inspects both data sides, checks a single example, and asserts unequal ion sets rather than disjoint sets. Do not execute it as a development-only or exhaustive certification gate.

## 4. Ordered implementation and execution

All steps below describe **future work**. Steps 1–5 must pass before full training. Source proposals remain immutable historical references in Git; source datasets remain unchanged. Module paths beginning `training/`, `graph/`, `model_variants/`, or `embed_helpers/` are relative to `src/`.

### Step 1 — Freeze a training-only provenance and example contract

**Targets:** `scripts/verify_zenodo_pmm_ion_dataset.py`; an opt-in source-manifest path through `training/config.py`, `training/data.py`, `training/structure_loading.py`, `training/site_filter.py`, and `training/metal_examples.py`. Preserve the default pocket loader and unrelated datasets.

1. Create a campaign identity for `train_and_test_sets_structures_zenodo_pmm_exact`. Bind it to source release/hash, train manifest and summary hashes, structure content hashes, code revision, and schema version. Verify identity by content and metadata, allowing relocation; a folder basename or expected row count is insufficient. Reject accidental substitution of the legacy `exact_pinmymetal` cohort.
2. Introduce an explicit training-only audit mode. It must not discover, load, prepare embeddings for, or summarize test files. Do not run existing reproduction wrappers, verifier defaults, or runner dry-run paths that inspect both sides. Resolve needed training coordinates from training structures; do not scan the mixed crosswalk as a shortcut.
3. Bind each source UID to one ion using structure version/hash, model, chain, residue number, insertion code, atom/altloc, and coordinate. Recover missing identifiers from pinned provenance or require a unique structural match. Keep the source's `residueid_ion` separate from PDB residue numbering. Never choose an ambiguous ion by nearest distance or overwrite a disagreement.
4. Write a disposition for every source row: retained, duplicate alias, unsupported, missing, conflicting label, or unresolved/incomplete context. Preserve raw labels and resolved element separately. Verify the source-code mapping above against the pinned training source; a PMM Class VIII label alone cannot provide separate Fe/Co/Ni supervision.
5. Enforce the following example rules:
   - Distinct ions remain distinct even when pockets or coordinating residues overlap. Identical representations of different valid ions are a modeling limitation, not grounds for exclusion.
   - Repeated ingestion of the same UID is an error. Distinct source UIDs resolving to the same physical ion and label are duplicate aliases: retain all provenance, give the ion one loss contribution and one metric vote, and use the identical deduplicated cohort for PMM. Label any resulting departure from source multiplicity as a matched subset.
   - Conflicting native labels for the same physical ion are quarantined pending provenance resolution, even if they collapse to Class VIII. Do not majority-vote them away.
   - Shared-PDB chains, alternative representations, and overlapping pockets remain in the same validation group. Known exact aliases across PDB IDs must also remain together; do not call PDB grouping homology-disjoint.
6. Keep noncatalytic PMM examples when otherwise eligible. In this profile, `whether_catalytic=1` and `EC_0.0.0.0` are compatibility placeholders, not biological annotations. Add a narrowly scoped source-row policy exception to the legacy catalytic-CSV rule during implementation; no EC head or EC loss is involved.

**Gate:** every retained ion has a unique, reversible provenance mapping and supported native element; every excluded/aliased row is accounted for. Save `train_cohort.csv`, `train_row_dispositions.csv`, `train_audit.json`, and `campaign_manifest.json`. Freeze the cohort before model comparisons; do not permit arm-specific silent skips.

### Step 2 — Certify common inputs and reuse existing preparation

**Targets:** `graph/structure_parsing.py`, `graph/shell_roles.py`, `training/esm_feature_loading.py`, `training/feature_sources.py`, `training/parallel_loading.py`, and `embed_helpers/esmc.py`.

- Verify the selected structural model, protein-chain context, residue identifiers, and target anchor agree with the source resolution. Missing interface or symmetry context must be reconstructed faithfully or declared/excluded consistently. Do not impose an arbitrary descriptor-correlation threshold as proof of correctness.
- Freeze **ESMC-600M**, using `--model-name esmc_600m` during generation and `--esm-dim 1152` during training. This is an explicit common model choice, not a claim about historical caches. Reuse only assets with matching model, sequence/residue alignment, structure provenance, and metadata; matching filenames or tensor dimensions alone are insufficient. Generate missing training assets only, into a new versioned directory.
- Require complete embeddings for every retained residue used by ESM-capable models. Remove `--allow-missing-esm-embeddings` for those arms, disable automatic feature preparation inside fits, and make loader failures fatal. Only-GVP may omit ESM loading, but uses the same eligible ions.
- Use the conservative feature set, geometry-derived shell roles, radius-only edges, and no explicit metal nodes. Intentionally omit `biotite_residue_sasa`, `custom_charge_distance_proxy`, and `dpka_titr` through the existing `omit_node_features` mechanism. External features are optional **only because all their input channels are explicitly masked**; verify those channels are zero even if local files happen to exist. No new PROPKA/external-feature campaign is needed.
- Keep labels, PDB identifiers, source class codes, metal element/residue names, and neighboring metal identities out of predictive tensors. Known target coordinates and protein geometry are permitted inputs. Validate this distinction after the anchor is fixed.
- Reuse `DEEPMZYME_PARSE_CACHE_DIR`. Its namespace must be bound to an immutable feature inventory; generate features before populating it, and use a fresh namespace if feature contents change in place. Cache raw examples/features, never normalization fitted across validation folds. Do not add a monolithic graph cache unless measured preparation remains a material bottleneck after reuse.

**Gate:** `feature_inventory.json` records hashes, model/dimension, residue coverage, intentional omissions, and cache identity; all retained examples load and construct valid graphs under the common contract.

### Step 3 — Make the existing runner enforce the comparison

**Primary target:** `scripts/run_metal_5fold_cv.py`, which already accepts dataset, example unit, label scheme, and grouping. Extend it with an explicit PMM campaign profile/manifest; do not maintain a second new orchestration system. Keep the hardcoded five-class Zenodo wrapper outside the new execution path unless it is made to delegate safely.

Freeze these campaign settings:

| Setting | Value/rule |
|---|---|
| Task and unit | `task=metal`, `metal_example_unit=ion` |
| Targets | Separate `four_class` and `six_class` identities; canonical names retained |
| Folds | Five PDB-grouped folds; `split_seed=42`, `split_stratify_by=metal_site` |
| Active model seeds | `[42]`; explicitly **one-seed grouped-fold comparison**, not seed-repeat confirmation |
| Training budget | 50 epochs, batch size 16, AdamW, fixed learning-rate schedule, weight decay `1e-4`; no HPO |
| Checkpoint selection | Highest native `val_metal_balanced_acc`; deterministic earliest-epoch tie rule |
| Geometry | 10 Å pocket, 8 Å residue-edge radius, `metal_node_mode=none`, `structural_readout_scope=residue_only`, `shell_role_source=geometry`, RING off |
| Capacity | Shared GVP trunk: scalar 128, vector 16, edge hidden 64, four layers; ESM projection 128; two head layers, head dropout 0.2; ESM encoder dropout 0.1 |
| Test handling | Runner `evaluate_test=False` / `--no-evaluate-test`; child `run_test_eval=False`, no test paths, no final-refit/test debug override |

| Family | Existing architecture/fusion | Learning rates | Required targets | Readout variant |
|---|---|---|---|---|
| Only-ESM | `only_esm` | `3e-5` | Four and six | Four only |
| Only-GVP | `only_gvp` | `3e-4`, raw-distance RBF | Four and six | Four only |
| GVP + ESMC | `gvp`, `late_fusion`, early ESM off | `3e-5`; existing GVP optimizer group `3e-4`; raw-distance RBF | Four and six | Four only |

These are fixed runner-derived recipes, not proven optimum learning rates. Preserve optimizer grouping in `training/run.py`; it currently assigns only `layers`, `node_scalar_encoder`, and `edge_scalar_encoder` to the GVP rate. Changing that grouping would introduce another intervention.

**Class imbalance:** use identical per-ion weights across the two target formulations. Within each training fold, calculate common-four weights `w_c = N_train / (4 * n_c)`. Use existing `metal_class_weight_mode=manual` and class multipliers: assign `w_VIII` to Fe, Co, and Ni separately in the six-class arm, and to Class VIII in the direct-four arm. This balances the endpoint without changing Class VIII's aggregate training weight solely because its output was split. Use ordinary shuffled sampling, cross-entropy, no label smoothing, no collapsed auxiliary loss, and no additional element boosts. Save actual weights. Validation metrics are unweighted. Stop if any native class is absent from training or validation.

Generate one `fold_membership.csv` from the frozen cohort using original element symbols, then verify every run reproduces it. A fixed random seed with active-target stratification alone is insufficient. Compare hashes of **membership and ion identity**, not whole diagnostics containing different native labels or model seeds. Reuse `split_pockets_k_fold()` and fail on divergence; do not repurpose the specialized `explicit_membership` diagnostic route, whose current constraints exclude six-class and late-fusion models. If exact aliases require composite grouping, add only that narrowly scoped grouping hook.

Run identity must include campaign/cohort hash, family, target scheme, readout, fold, and model seed. Current runner names omit scheme and seed. `--skip-existing` must require matching identity, configuration, completed fit status, and the actual selected checkpoint; metric-row count or a fallback last checkpoint is insufficient. Save exact commands and resolved configurations before fitting. A validation-only summary must not read existing test reports in a reused directory.

### Step 4 — Implement one binding-residue readout intervention

**Targets:** `AttentionPool`, `ESMGraphEncoder`, `pool_graph_states`, `GVPPocketClassifier` in `src/model.py`; `OnlyESMPocketClassifier` in `src/model_variants/models.py`; factory, configuration, training construction, and replay wiring.

Add one proposed option, `binding_residue_pooling=none|first_shell_bias`, defaulting to `none`. Do not add architecture names or change output sizes.

For residue states `z_i`, existing attention score `a_i`, and target-specific first-shell flag `m_i = (x_role[i,0] > 0.5)`, learn one scalar `b` per active readout branch:

```text
mean branch:      sum_i softmax_i(b * m_i) * z_i
attention branch: sum_i softmax_i(a_i + b * m_i) * z_i
```

Softmax is within each graph's existing residue mask. Verify the shell flags retain their Boolean meaning through normalization; otherwise carry the original mask explicitly. Initialize `b=0`, preserving the baseline mean/attention function and dimensions with identical shared weights. ESM states are projected once. Apply the same mechanism to the GVP branch and, where present, the late ESM branch. The combined intervention tests the combined readout strategy; it does not separately attribute gains to either branch.

Keep the existing 2.7 Å donor proxy and all graph/feature settings fixed. An empty first shell has `m_i=0` everywhere and naturally uses the baseline readout; do not substitute the first arbitrary residue. Log empty-shell support and learned biases. A learned bias is a relative weighting parameter, not evidence that a residue is a true ligand.

The variant is called **ESMC with target-shell-conditioned readout** where applicable. Do not assume adjacent ions have distinguishable first-shell masks or that shared ligands must favor only one ion.

### Step 5 — Repair replay/reporting and pass targeted checks

**Targets:** `src/export_validation_predictions.py`; runner checkpoint resolution, summarization, and aggregation; reusable functions in `training/final_test_reporting.py` where appropriate. Its current ion-wise bootstrap is not suitable for correlated ion examples.

1. Reload `metal_example_unit`, source-manifest identity, all graph/feature options, readout option, native class map, and saved training normalization. Construct models from the complete saved configuration with strict state loading. Carry `source_uid`, physical-ion identity, PDB/group, parent pocket, fold, seed, and checkpoint hash into prediction exports.
2. Compute native and collapsed metrics from **the same selected checkpoint**. For six-class probabilities, form `[p_Mn, p_Cu, p_Zn, p_Fe+p_Co+p_Ni]` before argmax. Do not collapse an already selected native argmax or sum confusion-matrix predictions as a substitute.
3. Assemble actual out-of-fold rows, joined by stable identity. Every eligible ion appears exactly once per arm/seed across the five validation folds. Keep mean fold BA and pooled OOF BA separate.
4. Add only targeted regressions around the changed contracts: mixed-metal sibling targets and shared grouping; duplicate/conflicting IDs and insertion/model ambiguity; four/six fold equality; probability-sum collapse; complete embedding requirements; zero-bias equivalence and gradients in all three families; empty-shell behavior; metadata/label invariance of logits after anchoring; checkpoint/replay round trip; and test-file access denied in development paths. Extend `tests/test_metal_ion_examples.py` and `tests/test_generalized_metal_5fold_cv.py` where suitable.
5. Verify normalization and class weights use training partitions only. Perform a tiny training-only smoke for all nine configurations before the full grid. Use synthetic fixtures or certified training examples, never a held-out regression example.

Use the required DeepMzyme interpreter and verify it before future Python checks. No Python command, smoke, or experiment is part of this review.

### Step 6 — Run and assess the bounded matched comparison

**Deferred full neural grid:** three baseline families × two target formulations × five folds = **30 fits**, plus three direct-four readout variants × five folds = **15 fits**. Total **45 full neural fits**, one model seed. The immediate amendment above runs only the exploratory ESMC pair. Completing the full matrix later requires all three awareness variants, regardless of the Only-ESM result; screening only that family cannot decide usefulness in another family. Additional seeds are a separately budgeted extension using a shared active seed list for every compared candidate.

**PMM comparator:** fit the released four-class classifier recipe anew on each of the same five training partitions and predict its matching validation ions. Bind source version, class-code order, permitted feature columns, preprocessing, and resampling to the run; pin dependency versions and fully resolved estimator defaults. Never use a released full-training fitted model to score its own training rows as CV. PMM's released script calls `dropna()` before selecting features and fits a soft-voting classifier; it is not itself a fivefold runner. Adapt its data boundary, with any fitting/resampling confined to the training fold. [Released PMM training source](https://github.com/hhz-lab/PinMyMetal/blob/main/data_model/train_chedhclassmodel.py).

Resolve PMM feature completeness and filtering in Step 1, so its retained evaluation ions match DeepMzyme's. The comparator uses its published feature recipe; DeepMzyme uses the declared ESM/geometry recipe. Record this input difference and common known-site information. This compares classification systems at known sites, not equal-capacity models or end-to-end localization. If PMM's features or mapping cannot be reproduced, report DeepMzyme validation with PMM comparison **blocked**; do not replace it with unmatched literature subtraction or an arbitrary random forest.

**Metrics and decisions:**

- Primary comparison: mean common-four balanced accuracy over the five selected fold checkpoints. Co-report macro-F1, each class's recall/support, minimum recall, confusion matrices, and pooled OOF metrics. Six-class arms additionally retain native metrics and separate Fe/Co/Ni recalls. Never rank native four-class BA against native six-class BA.
- Predeclare six neural contrasts: six versus direct four within each baseline family, and aware versus ordinary readout within each direct-four family. Report all results, including adverse or inconclusive effects.
- Follow Stage 6's paired fold-difference bootstrap for the mean-fold estimand; freeze 10,000 resamples, 95% intervals, and bootstrap seed 42. If seeds are later added, average the same active seeds within each fold before resampling folds. A separate pooled-OOF interval, if reported, resamples whole PDB/alias groups and is labeled conditional on fitted models. Never resample sibling ions independently or use a pooled-OOF interval as though it estimated mean fold BA.
- Validation promotion requires positive mean difference, a positive lower paired-95%-CI bound, no missing class recall, no zero mean class recall, and no common-four class's mean recall dropping more than **0.03** versus its matched control. The last rule applies the playbook's rare-recall tolerance per class. Report these six contrasts as exploratory; a simultaneous improvement claim additionally requires a positive lower Bonferroni interval bound at `1 - 0.05/6` confidence, computed from the same resamples. These approximate intervals are conditional on the fixed fold/seed design, not proof of a universal architecture advantage.
- Without clear improvement, retain direct-four over six-class and ordinary readout over the new mechanism. Promote only a configuration actually evaluated: do not attach a four-class readout winner to an untested six-class model. Do not demand an invented +1 or +3 percentage-point gain, or success specifically in a small heterometal subset. Completion does not require outperforming PMM.
- Limited train-only diagnostics may count nearby ions, target-shell overlap, and same/different native or collapsed labels. Structure-wide multiplicity must not be called local multinuclearity. Define any distance-based subgroup before examining its errors, show support, and keep sparse subgroups descriptive. Do not infer causal “interference resolution” from aggregate accuracy alone.

These are validation-selected, one-seed CV estimates; using each fold for checkpoint selection is not an unbiased nested estimate of the whole selection process. Do not label them PMM's original folds, seed-stability evidence, or proof of remote-protein generalization.

### Step 7 — Preserve the validation-to-refit-to-test boundary

This step is a **later reporting gate**, not permission to access test during implementation or this review.

1. Produce a validation decision using the common-four comparisons above. Across target schemes, retain the distinction between native checkpoint selection and common-endpoint promotion. Do not send mixed native four/six BA values into an undifferentiated Stage 6B ranking.
2. Before reportable final confirmation/refit, declare the test route and permitted training side. The exact Zenodo PMM source split can support a **secondary, possibly overlapping reference comparison**; it does not resolve the project's outstanding primary final-test route. Do not quietly move overlapping PDBs between original PMM sides or call this a pristine generalization test.
3. Apply **Stage 6B: promotion gates and final full-train refit**. Preview the decision and command with `LAUNCH_STAGE6B_FINAL_REFIT=False`. Freeze one selected DeepMzyme configuration, seed 42, 50 full-training epochs, and the terminal checkpoint rule; refit once on its complete eligible non-test training side, recomputing normalization and weights there. Refit the frozen PMM recipe on the same eligible training cohort. Save the decision, exact command, and completed-refit identity before test access. If promotion is inconclusive, report that and retain the declared simple control; do not manufacture an improvement claim.
4. Only after the route and reporting set are fixed, a separately authorized execution may certify test reconstruction/overlap using the already frozen rules, prepare label-free test inputs, and conduct **Stage 7: one-shot held-out test** for those fixed refits. Save exclusions and coverage; a mismatch yields a clearly named matched-subset report, not a silent protocol change. No per-fold development test evaluation, fold-ensemble replacement for refitting, test-driven calibration, or testing every exploratory arm.
5. Update the test-use ledger after that future access. If paper cohort, input, label, and metric parity remain unverified, published PMM/Metal3D values are contextual references only; no “beats Fig. 2” conclusion is warranted.

### Step 8 — Admit compute only after correctness and timing evidence

The original review authorized only this plan document. Subsequent execution requires concrete compute authorization and storage/runtime admission; the immediate amendment above records the user-approved one-session screen. Use the existing VM controller and [`docs/GCP_GPU_RUNBOOK.md`](../GCP_GPU_RUNBOOK.md) if GCP is chosen; its start authorization and actual caps apply. Neither source proposal's VM status, hourly price, free-space measurement, or elapsed-time forecast establishes current readiness.

Measure preparation and training costs during the authorized smoke/first complete fits. Forecast the 45-fit grid as `15 × (time_ESM + time_GVP + time_late)` plus embedding preparation, PMM CPU fits, validation export, and any separately admitted final refit/report. Do not promise the grid fits one session or copy the old cohort's timings. Preserve the scientific grid when addressing a budget shortfall; obtain a recorded budget decision instead of silently dropping an arm or shrinking its epochs.

Use persistent run storage, epoch checkpoints, and artifact synchronization with verified receipts. Distinguish restart from genuine optimizer-state resume; do not count an incomplete fit as complete. Keep GPU fits serial initially. Stop owned compute at the completion boundary under the runbook.

## 5. Deliverables and measurable completion criteria

Under one new campaign output root, retain:

- `campaign_manifest.json`, `train_cohort.csv`, `train_row_dispositions.csv`, `train_audit.json`, `feature_inventory.json`, and `fold_membership.csv`, all with hashes and schema versions.
- Exact expanded commands; each fit's `run_config.json`, `run_metadata.json`, split diagnostics, fitted normalization/class weights, terminal status, selected checkpoint and checkpoint hash, and UID-keyed validation predictions.
- `cv_fold_metrics.csv`, `cv_oof_predictions.csv`, `cv_paired_deltas.csv`, and `validation_decision.json`; PMM configuration and aligned predictions; an input/cohort/protocol comparison table; a report of all planned arms and any incomplete units.
- If the later reporting gate is executed: `stage6b_decision.json`, `stage6b_ranked_candidates.csv`, `stage6b_final_refit_command.txt`, and `stage6b_selected_final_refit_candidate.json` only after completed/reused compatible refitting; frozen final prediction/report artifacts and the test-access record.

**Development completion:** targeted tests and replay pass; all retained ions have certified provenance and required inputs; all nine neural configurations complete five matching folds with identical evaluation identities; PMM supplies predictions on those same units; native/common-four metrics reconcile to the selected checkpoints; class support and paired uncertainty are reported; no test access occurred in the validation chain. A negative or inconclusive awareness result is a valid completed result. Missing PMM parity or incomplete folds must remain explicitly incomplete, not be disguised by an average over available runs.

**Reference-report completion:** the separately gated refit/report artifacts and coverage/overlap disclosures exist, with no selection changes after test access. This does not complete or designate the project's primary final held-out route.

During implementation, coordinate only the affected documentation owners: dataset/provenance corrections in `docs/DATASETS.md`, exact runnable recipes in the metal playbook or its linked benchmark recipe, unresolved defects in `docs/FOLLOW_UP_TECHNICAL_ISSUES.md`, and mutable progress/evidence links in `EXPERIMENT_STATUS.md` and `docs/notebook_outputs/`. No such edits are part of the present review.

## 6. Remaining uncertainties and smallest resolving checks

| Uncertainty | Smallest check and decision rule |
|---|---|
| Alleged earlier five-class or true-hybrid instruction | Recover an attributable original user instruction if available. In its absence, execute the four/six, late-fusion scope above; neither proposal alone overrides repository policy. |
| Effective PMM cohort and fine-label provenance | Compare the pinned training source, its filtering, manifest UIDs, native elements, and feature availability. Freeze their certified common cohort; unresolved rows are excluded with reasons, and incomplete coverage is labeled a matched subset. |
| Multi-model, insertion-code, altloc, or interface ambiguity | Inspect only affected training structures and their source resolution. A unique faithful anchor/context permits inclusion; unresolved ambiguity blocks that example, not automatic nearest-ion substitution. |
| Embedding availability and runtime cost | Run the training-only input inventory, then an authorized preparation/smoke. Reuse certified 600M assets or generate them in a new namespace; do not change the ESM model or silently accept zeros to fit a budget. |
| Exact paper-comparison validity | Recover released fold/effective-cohort and metric definitions without using benchmark performance to make modeling choices. Unless equivalence is demonstrated, use the matched rerun and contextual literature labeling. |
| Final test route and overlap | Resolve the reference-report designation before reportable confirmation/refit; inspect test identity only under the later authorized gate. Any exact-split overlap prevents a claim of an independent primary held-out result. |

There is **no blocker to completing this plan**. Execution is gated by source-row/input certification, runner/reporting corrections, and compute authorization; primary held-out reporting remains a separate unresolved scientific decision.
