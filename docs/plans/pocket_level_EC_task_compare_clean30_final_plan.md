# Final execution plan: pocket-input EC1 prediction on CLEAN30

**Status: planned.** Review date: 2026-09-25. Implementation, training and benchmark evaluation are future work; this document does not authorize their execution.

## 1. Objective, scope and authority

Compare EC1 prediction from metalloenzyme pocket inputs using **Only-ESMC, Only-GVP and GVP + graph-level late fusion**, then compare their protein-level predictions with the official CLEAN predictor trained under two regimes: the full CLEAN30 training cohort and its metalloenzyme subset. Preserve the five supplied CLEAN benchmark train/test pairs.

The scientific endpoint is **protein-associated EC1 prediction from pocket representations**. Individual pocket predictions are intermediate outputs with inherited, weak supervision. The available annotations do not establish which reaction an individual pocket catalyses. EC numbers classify catalytic reactions; this dataset associates those annotations with proteins. A protein can have several activities and several pockets, without a known correspondence between them. See [IUBMB classification principles](https://iubmb.qmul.ac.uk/enzyme/rules.html) and [Plan.md, EC classification](../../Plan.md#3-train-the-ec-number-classification-model).

Required comparisons under the task interpretation recoverable here:

1. The three DeepMzyme families on identical eligible proteins, pockets, inner memberships and model seeds.
2. CLEAN-full on all benchmark test proteins and on the metalloenzyme subset, using subsets of the same predictions.
3. CLEAN-metallo versus CLEAN-full on the same metalloenzyme test proteins.
4. All five systems on one declared, unambiguous-EC1 metalloenzyme evaluation cohort.

The source plans also preserve a **conditional** request to try extra binding-residue information if prior experiments demonstrated benefit. It is not an unconditional six- or nine-arm architecture campaign. Apply the evidence gate in step 5 below.

The complete original task transcript and any later clarifications are not independently available in the repository materials inspected. The quoted condition and common comparison matrix are therefore a **scope reconstruction**, not proof of the original wording. No instruction available in this review changes the primary depth from **EC1**. A subsequently supplied original instruction overrides this reconstruction before execution.

Excluded: metal-label prediction, ion-level examples, the separate PMM campaign, joint metal–EC learning, CARE experiments, EC2–EC4 training, full multi-label DeepMzyme prediction, advanced fusion, new pretraining and broad HPO. CLEAN may retain its native full-EC training objective; its predictions are projected to EC1 for the comparison. CLEAN30 is a benchmark evaluation, not a newly designated pristine final-test route for the entire project.

## 2. Source plans and independent verdicts

The superseded source proposals are preserved in Git at the reviewed revision:

- [Opus proposal](https://github.com/MECHTI1/DeepMzyme/blob/63d4cd52bf98b2448d42f4498364828431c962eb/docs/plans/pocket_level_EC_task_comapre_clean30_created_by_opus.md) — the spelling **`comapre`** is the historical filename.
- [Antigravity proposal](https://github.com/MECHTI1/DeepMzyme/blob/63d4cd52bf98b2448d42f4498364828431c962eb/docs/plans/pocket_level_EC_task_compare_clean30_antigravity_plan.md).

Both were read completely. These were the only EC/CLEAN30 candidate matches. Commit `63d4cd52bf98b2448d42f4498364828431c962eb`, dated 2026-09-25, introduced both alongside two separately named metal/PMM plans. The Opus title explicitly attributes its author, dates its inspection and names the Antigravity document in its appendix. Content and shared introduction provenance establish the pair; filename similarity alone was not used. This final plan supersedes both proposals; their working-tree copies were removed after consolidation, with the original content retained through the historical links above.

**Opus:** the stronger starting point for cohort provenance, protein weighting, ambiguity handling and the need for homology-aware validation. Its implementation inventory nevertheless contains consequential errors: aggregation is mean logits, protein-level test metrics already exist, summary rows are not necessarily extracted pockets, and conservative CSV membership does not guarantee conservative pocket geometry. Its strict 30% separation claim is not established by local code. Global selection across the five overlapping development sets, a cutoff derived in fold 0 for every other fold, treating fold-by-seed results as independent bootstrap units, unequal CLEAN replication, and skipping final refits are unsuitable. Two-view pooling, secondary hybrid fusion, additional CLEAN variants and the estimated GPU/cost ceiling add unneeded scope or unsupported assumptions.

**Antigravity:** useful for the core comparison matrix and the two CLEAN training regimes. Its baseline is incorrectly described as lacking binding cues; shell flags and metal distances are already present. Its flag `--ec-group-weights` is wrong, `src/graph/features.py` is not the implementation, and `--node-rbf-use-raw-distances false` is not a valid way to add information: the real option is a Boolean switch affecting representation of existing distances. Ordinary accession grouping does not control homologous proteins. Early injection/cross-attention options do not define an ablation of graph-level late fusion. A calendar schedule, 40-epoch budget, optional contrastive/RING additions and unqualified per-fold macro scores are not justified by feasibility or class support.

Neither plan is executable as written. Agreement between them is not treated as verification.

## 3. Verified facts that change the execution design

The inspected source revision was `63d4cd52bf98b2448d42f4498364828431c962eb`. These are implementation facts or explicitly bounded read-only observations, not results from new model runs.

| Finding | Evidence and consequence |
|---|---|
| EC depth defaults to 1; targets are single-label prefixes parsed from `__EC_...` structure names. Distinct prefixes yield no target. The parser does not comprehensively validate EC syntax or the 1–7 range. | [src/training/labels.py](../../src/training/labels.py): `ec_label_token_from_numbers`, `parse_structure_ec_numbers`, `assign_ec_targets`. Validate annotations before loading; do not select the first annotation. |
| CLEAN30 preparation imports membership; it does not implement a 30% sequence filter. | [CLEAN_prepare_training_and_test_set/clean_prepare.py](../../CLEAN_prepare_training_and_test_set/clean_prepare.py): `read_clean_split_file`, `clean_fold_membership`, repartitioning. The identity argument selects upstream `split30` files; accession intersections are checked. |
| The conservative source is a derivative of the upstream benchmark, not the meaning of “30”. | [docs/DATASETS.md, CLEAN](../DATASETS.md#clean); [src/build_clean_single_donor_subset.py](../../src/build_clean_single_donor_subset.py): `choose_candidate`, `deduplicate_rows`, `process_fold_csv`. One donor is selected by metadata/quality, with 2.0 Å within-donor deduplication. These operations do not certify protein homology separation or exact stoichiometry. |
| The main symlink resolves to `CLEAN_30_shared_single_donor_supported_metal_conservative`; the existing fold-0 materialization marker names `CLEAN_30_shared`. | `DeepMzyme_Data/CLEAN_30_main`; `DeepMzyme_Data/CLEAN_30_train_test_split_0/.clean_shared_materialized`. A root name is insufficient to select the intended cohort. |
| Conservative selection retains the original structure bytes. Pocket extraction precedes summary filtering and retains an entire cluster when any site matches. | `populate_structures` in the subset builder; [src/training/structure_loading.py](../../src/training/structure_loading.py): `load_structure_pockets`; [src/training/site_filter.py](../../src/training/site_filter.py): `pocket_matches_allowed_sites`. Unselected metals can still affect extraction, site statistics and binding distances. |
| Default extraction uses 10 Å residue proximity and 4.5 Å connected metal clusters; summary deduplication uses a different 2 Å rule. | [src/graph/structure_parsing.py](../../src/graph/structure_parsing.py): `cluster_metal_records`, `find_pocket_residues_near_metal_cluster`; [src/data_structures.py](../../src/data_structures.py). CSV site rows must not be asserted equal to model pocket counts. |
| Conservative GVP features already include shell flags, distances and metal-relative vectors. First-shell geometry defaults to donor distance ≤2.7 Å. | `NODE_FEATURES_CONSERVATIVE` and `DEFAULT_FIRST_SHELL_CUTOFF`; [src/featurization.py](../../src/featurization.py): `residue_to_stage1_node_features`; [src/graph/shell_roles.py](../../src/graph/shell_roles.py). These are geometric candidate binders, not experimentally assigned catalytic residues. |
| Repository Only-ESMC pools pocket embeddings and also uses four metal-site statistics. Embeddings retain full-chain sequence context. | [src/model_variants/models.py](../../src/model_variants/models.py): `OnlyESMPocketClassifier.forward`; [src/model.py](../../src/model.py): `ESMGraphEncoder`. Describe it as the repository pocket-ESMC baseline, not a structure-free or purely local-sequence model. |
| Both validation and test already have protein/group metrics using **mean pocket logits**. | [src/training/run.py](../../src/training/run.py): `ec_group_metrics_from_logits`, `metrics_from_predictions`, `evaluate_held_out_test_split`. Keep this rule; do not add a duplicate aggregator or silently switch to mean probabilities. |
| Protein weighting and validation prediction export already exist. | [src/training/splits.py](../../src/training/splits.py): `assign_ec_group_metadata`; [src/training/loop.py](../../src/training/loop.py): `balanced_class_weights_from_pockets`; [src/export_validation_predictions.py](../../src/export_validation_predictions.py): `aggregate_ec_logits`, `prediction_rows`. Reuse their semantics and saved-normalization protections. |
| Existing explicit membership is a narrowly constrained metal diagnostic, not a generic EC interface. | [src/training/explicit_membership.py](../../src/training/explicit_membership.py): `validate_mode`. Add a small EC membership path without weakening the diagnostic's safeguards. |
| The generic CLI is the appropriate entry point. The dedicated EC wrapper and historical campaign runner impose incompatible restrictions. | [src/train.py](../../src/train.py); [src/training/task_entrypoint.py](../../src/training/task_entrypoint.py): batch 8/16, a restricted metric list and positive validation fraction; [src/run_ec_baselines.py](../../src/run_ec_baselines.py): CARE-specific constants. Do not repurpose a metal runner or alter the old CARE campaign. |
| The CLEAN notebook has useful preparation/training code but does not implement the full requested evaluation matrix safely. | [CLEAN/train_clean_predictor_baselines.ipynb](../../CLEAN/train_clean_predictor_baselines.ipynb), source cells 3/5/11/13/15/17, zero-based: training sequence lookup reads train and test sources; conflicting sequences keep the first; both regimes currently evaluate metallo only; inference requests immediate metrics. These need focused separation and validation. Notebook outputs were not read. |

### Development-only counts and spot checks

Read-only CSV checks used **only each named fold's training tables**. Columns are proteins, not independent homology groups or verified loaded pockets. “Sites” means conservative summary rows before pocket construction and EC exclusion.

| Fold | Full CLEAN train | Metallo proteins / sites | Ambiguous EC1 proteins | Eligible EC1 support: 1 / 2 / 3 / 4 / 5 / 6 / 7 |
|---:|---:|---:|---:|---|
| 0 | 8,115 | 622 / 743 | 19 | 161 / 86 / 273 / 42 / 17 / 21 / 3 |
| 1 | 8,138 | 578 / 698 | 17 | 144 / 82 / 264 / 37 / 16 / 16 / 2 |
| 2 | 8,170 | 562 / 668 | 17 | 138 / 78 / 258 / 39 / 11 / 17 / 4 |
| 3 | 8,173 | 586 / 696 | 18 | 139 / 81 / 272 / 39 / 15 / 19 / 3 |
| 4 | 8,212 | 612 / 723 | 17 | 154 / 93 / 269 / 39 / 13 / 23 / 4 |

Sources: `DeepMzyme_Data/CLEAN_all_train_valid_splits/split30/split30_train_split_{k}.csv` and `DeepMzyme_Data/CLEAN_30_main/folds/CLEAN_30_train_test_split_{k}_train.csv`. No missing strings, `-` placeholders, conflicting annotations between a protein's summary rows, source-label disagreements or exact-sequence duplicate groups were found among these metallo training proteins. One source annotation, **Q89GR3: `6.2.1.n2`**, has a nonnumeric terminal component in folds 0/1/3/4; the table includes its unambiguous EC1 prefix 6, not a certified complete EC4 label. This does not certify absence of homologues, duplicates across partitions, or losses during graph loading. The Opus fold-0 support figures are not the post-ambiguity-filter counts above.

Additional bounded observations:

- Fold 0 has 511 proteins with one summary row, 101 with two and 10 with three. This establishes repeated supervision, not distinct catalytic activities.
- Five training structures with donor-row reductions were inspected using coordinate records. `A0R3R7` and `B0XTJ7` each have one selected site but a retained 4.5 Å component containing two metal atoms, one unselected. `A3A8Q4` has two selected rows forming one component. This confirms the input-contract problem without running the loader or writing caches.
- Fold-0 training has 622 embedding files and 622 sidecars declaring `esmc_300m`, dimension 960. Filename EC strings agree with the training summaries. **Q96KN2** has one amino-acid substitution between the CLEAN sequence and sidecar `source_sequence`, both length 507. Embedding tensors and coverage in other folds remain unchecked. “740/740 ready” is not established by this review.

### What CLEAN30 means, and what remains unverified

Use the exact upstream **CLEAN `split30` cross-validation release** as dataset identity. The [official CLEAN README](https://github.com/tttianhao/CLEAN#12-quickstart) identifies five released train/test pairs for each identity-clustering setting and says no additional validation sets were created. It also distinguishes those CV files from the similarly named full training datasets. Local preparation preserves the supplied membership and then selects the computational metalloenzyme cohort.

Here “metalloenzyme” is operational: supported transition metals Mn/Fe/Co/Ni/Cu/Zn, UniProt cofactor support, AlphaFill transfer into predicted structures and MAHOMES catalytic-site predictions, followed by conservative donor selection. It is not an exhaustive sample of all metalloenzymes, experimental validation of every site, or a measured metal-stoichiometry dataset. Record attrition from the original sequence cohort through structure/site availability to EC1 eligibility; this ascertainment affects generalization claims. The implementation/procedure sources are `CLEAN_prepare_training_and_test_set/clean_prepare.py` and its [preparation README](../../CLEAN_prepare_training_and_test_set/README.md).

The precise upstream clustering algorithm, identity denominator, alignment coverage, curation rules and any guarantee on **maximum train–test pairwise identity** were not recovered from the inspected implementation. Therefore “strict ≤30% train–test identity” and “homology-free” are unverified claims, not design premises. The smallest remaining provenance check is the upstream Supplementary Text 1 / split-generation recipe and the release identity of the local archive. If that does not certify the stronger claim, retain the supplied benchmark under its source name and use measured, explicitly defined remoteness reporting; do not silently re-split it to make the name true.

## 4. Final scientific decisions

**Labels and supervision.** Keep a seven-category EC1 vocabulary. An incomplete annotation such as `3.4.-.-` is usable at EC1 when its first component is valid and unambiguous; it is not a complete deeper EC label. Likewise, preserve the observed source token `6.2.1.n2` as EC1=6 with an explicit noncanonical-deeper-label flag; a full-numeric EC4 regex must not silently exclude it from an EC1 task. Normalize separators/whitespace and duplicates, retain original strings and provenance, and classify exclusions explicitly: missing, malformed, unknown first digit, multiple EC1 prefixes, or conflicting sources. Multiple complete annotations sharing EC1 are eligible but still do not identify a pocket's reaction. Mixed known/unknown or contradictory annotations require resolution or quarantine, not silently ignoring the problematic token. Never impute labels from metal identity or copy one chosen activity to every pocket as ground truth.

Use inherited protein EC1 labels with the existing `1 / number_of_pockets_in_group` loss weighting and group-count class weights. Assert one biological accession per `structure_id` in this cohort; group aliases/repeated structures at accession level before splitting. If that assertion fails, consolidate the mapping or use the verified `pdbid` accession grouping for loss/metrics. Keep all pockets from a protein together. Mean logits across its eligible pockets produce its primary prediction. Report single-pocket versus multi-pocket protein results descriptively; do not introduce multiple-instance learning or activity-specific annotation work into this baseline campaign.

**Splits and selection.** Preserve every official outer pair. For outer fold `k`, all labels, statistics, clustering, LR choice, epoch choice, feature decisions and any learned thresholds must use only that fold's training membership. Do not pool inner-validation scores across outer folds to choose one shared LR/arm: another fold's training set can contain fold `k`'s outer cases. The same restriction rules out transferring a data-derived cutoff from fold 0 to all folds. Freeze the algorithm and candidate grid globally; fit its choices independently inside each outer fold.

**Model comparisons.** Use the existing three families and graph-level `late_fusion`, with RING, augmentation and EC contrastive loss disabled. Keep the current EC baseline's radius-6 recipe as a declared common choice, not a claim that the CLI default is 6 Å or that 8 Å is inferior. Shared site statistics remain present and must be disclosed. No two-view head, early/cross-attention arm, optional hybrid, whole-protein ESMC control or SupCon-CLEAN variant is needed to answer this task. A null ablation does not prove that a model “does not use” information; redundant features and low power remain explanations.

**CLEAN regimes and fairness.** Preserve CLEAN's native annotation handling and full-EC objective in both regimes. R1 uses all valid proteins in the official training file; R2 uses the conservative metallo subset of that same training file. Do not disadvantage CLEAN by silently deleting multi-EC proteins merely because DeepMzyme is single-label. Log the resulting training-membership and supervision differences. The cross-system table is a comparison of complete systems on a shared endpoint, not an isolated causal test of architecture, pocket localization, or training-scope benefit. The three DeepMzyme families are the controlled architecture comparison; R1 versus R2 is the controlled CLEAN training-scope comparison. Full-sequence versus pocket-selected context, ESM-1b versus ESMC, predicted structures/site priors, full-EC versus EC1 supervision and feature availability must accompany interpretation.

R1 versus R2 deliberately changes training size and diversity together, so it does not isolate a domain-specialization effect at equal sample size. Shared AlphaFill donor templates and possible overlap with pretrained language/structure/site-prediction models are additional provenance limitations. Record donor reuse during the eventual integrity audit; neither accession separation nor an identity threshold proves independence of these upstream information sources. No extra size-matched or pretraining-control campaign is required here.

**Final training and reporting.** Require a full eligible outer-training refit after each fold's validation selection, using a frozen epoch rule and no outer-test selection. This resolves Opus's no-refit proposal against [Plan.md, experiment policy](../../Plan.md#6-experiment-tracking-and-reproducible-run-summaries) and `validate_training_configuration` in `src/training/run.py`. Do not import the separate metal Stage-6 candidate matrix. The CLEAN-specific bridge must be implemented because [TECH-002](../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-002--ec-playbook-assignments-do-not-match-the-notebook-surface) does not certify existing EC final-stage recipes.

## 5. Ordered implementation and execution steps

### Step 1 — Freeze provenance, labels and actual pocket membership

Create a dedicated campaign directory and manifests under `DeepMzyme_Data/notebook_outputs/runs/pocket_level_EC_clean30/` when implementation is authorized. Bind repository revision, source hashes, full CLEAN release, conservative source, training CSVs and structure-object hashes. Never overwrite the existing shared or materialized dataset roots.

Build train-only per-fold structure manifests and summary views. Passing the entire shared structure directory with a train summary is insufficient: `load_structure_pockets` reads a structure and its features before applying the summary filter. Ensure discovery is restricted to the training allowlist before file access, using the existing `allowed_structure_ids` support or a fold-specific manifest.

Add an opt-in **selected-site extraction** path in `src/training/structure_loading.py` / `src/graph/structure_parsing.py`, with keys resolved through `src/training/site_filter.py`. For this campaign, filter metal records to the conservative summary's retained site IDs **before** connected-component clustering, residue extraction, site statistics and shell assignment. Preserve multiple selected ions in a multinuclear pocket; do not switch to ion-level examples. Default behavior for other campaigns stays unchanged. Record the row-to-pocket mapping and the effective extraction settings.

Add preflight label validation, filename/CSV reconciliation and conflict reporting around `src/training/labels.py` and `src/training/preflight.py`. Reject unresolved duplicate-accession sequence conflicts; do not use the CLEAN notebook's keep-first behavior. For Q96KN2, compare the training sequence, coordinate-chain sequence, sidecar and source version. Correct/regenerate the affected training asset if provenance establishes an error; otherwise record a sequence-version discrepancy and apply one predeclared eligibility policy across the matched DeepMzyme cohort. Do not silently overwrite CLEAN's reference sequence.

**Gate:** conservative selected sites are the only metal centers affecting a pocket; every retained protein has a valid EC1 label and at least one usable pocket; all exclusions and mapping changes are counted. The table above is a raw-input reference, not a hardcoded assertion of loaded counts.

### Step 2 — Construct and certify inner validation independently per outer fold

Using only fold `k`'s eligible training proteins, map full UniProt IDs, normalized sequence hashes, structure-object hashes and all associated pockets. Join accession aliases and exact duplicate structures/sequences into indivisible groups. Search sequence similarity and form connected components of detected qualifying relationships. Reuse the search/audit patterns in [scripts/run_remote_homology_search.py](../../scripts/run_remote_homology_search.py) and [src/audit_sequence_remoteness.py](../../src/audit_sequence_remoteness.py), without reusing their old memberships or silently inheriting their coordinate-chain/20% diagnostic protocol.

Proposed inner protocol: full training sequences, exact identity ≥0.30 over aligned columns, coverage ≥0.80 of **both** sequences, pinned MMseqs2 version/settings, plus exact-duplicate/alias links. Freeze this as an operational inner-split definition, independent of the unresolved upstream definition. Record a lower-coverage audit for substantial domain matches; bidirectional full-length coverage alone does not establish domain independence. Use components from qualifying all-versus-all hits, not an assumption that representative-based `easy-cluster` output prohibits every cross-cluster hit. Verify no qualifying detected edge crosses the final boundary. Absence of a detected hit is not proof of no homology.

Assign whole components to approximately 15% validation **by protein count**, split seed 42, stratifying by EC1 where feasible. No score-driven split-seed retries. Persist exact train/validation roles and component IDs, not merely a map that lets each run choose a new partition.

Add a small frozen EC train/validation membership input, with checksum, to `src/training/config.py`, `src/training/splits.py`, `src/training/run.py` and preflight. The new option name/schema must be documented when implemented; it does not currently exist. Enforce exact coverage of the eligible allowlist, unique ownership, no unmapped sample fallback and no simultaneous automatic re-splitting. Preserve the existing metal diagnostic interface.

**Support gate:** recount proteins, pockets and independent components per EC1 in each partition after every filter. Seven-class validation requires every class in training and validation. With only 2–4 EC7 training proteins, this may be impossible or statistically very weak. If a class has one component, keep the component intact: do not split it or replace the split with random accession validation. Deliver a feasibility report and resolve whether to obtain more development data or explicitly accept a limited exploratory benchmark before launching the full comparison. A six-class-present score must never be labeled seven-class BA. Even when all classes are present, tiny rare-class support cannot establish reliable rare-class superiority or safety.

### Step 3 — Certify features and reuse the existing model interfaces

Audit all eligible training structures for residue-aligned ESMC-300m/960 embeddings, source-sequence identity, required external features, finite values and consistent residue numbering. Only-GVP does not consume ESMC, but the three-family comparison must have the same retained cohort; feature failures cannot silently change it. Check whether external features depend on the original, unfiltered metal coordinates and recompute only affected assets if required by selected-site extraction.

Reuse immutable residue assets when valid. Namespace parse/graph caches by source, selected sites, extraction settings and feature manifest. `src/training/parallel_loading.py::resolve_parse_cache_dir` hashes source and load settings but external feature contents are keyed by path; in-place regeneration requires invalidation. Fit normalization and class weights on inner training only, then recompute them on the complete eligible outer training set for the final refit. Do not reuse fitted preprocessing across folds.

Use `src/train.py --task ec`, not `src/train_ec.py`, for this recipe. Coordinate the implemented recipe with the owning [EC playbook](../EC_TRAINING_PIPELINE_PLAYBOOK.md) in the later implementation change. The chosen baseline values come from its current standalone block:

| Shared setting | Planned value |
|---|---|
| Target and selection | `--ec-label-depth 1`; `val_ec_group_level_1_balanced_acc` |
| Loss | EC cross-entropy; `--ec-group-weighting structure_id`; `--ec-class-weight-unit group`; `--ec-contrastive-weight 0.0` |
| Families | `only_esm`; `only_gvp`; `gvp` with `--fusion-mode late_fusion` |
| Optimization | AdamW; LR candidates `3e-5, 1e-4`; fixed schedule; weight decay `1e-4`; 30 epochs; batch 4 |
| Model seeds | 42, 43, both for every family and LR; split seed remains 42 |
| Shared architecture | hidden scalar/vector 128/16; edge hidden 64; four GVP layers; two head layers, dropout 0.2; ESM fusion dimension 128, encoder dropout 0.1 |
| Graph/features | conservative features; residue edges 6 Å; selected-site extraction radius 10 Å; residue-only readout; no explicit metal nodes, RING, augmentation or additional site-angle features |
| Readout | baseline `--classifier-pool-distance-cutoff 0.0`; existing four site statistics; legacy site-feature layout |
| Protections | no test paths or test-evaluation flag in preparation/training commands; strict invalid-structure/feature handling; feature generation disabled during training |

Use the existing legacy site layout rather than passing `--site-geometry-features none` across all families: explicit geometry modes are restricted to GVP architectures and change the site-input layout. Resolve every remaining default into `run_config.json` before launch.

### Step 4 — Implement thin orchestration and common prediction exports

Add a dedicated `scripts/run_clean30_ec.py` with separate prepare/dry-run, smoke, validation, refit and evaluation phases. These are **new planned interfaces**. Reuse `TrainConfig`, `run_training`, the existing model factory and checkpoint format. Do not generalize `scripts/run_metal_5fold_cv.py` or rewrite model classes.

Persist per-pocket logits with `(outer_fold, seed, accession, structure_id, pocket_id)` and class-vocabulary metadata. Reuse mean-logit aggregation from `ec_group_metrics_from_logits` / `aggregate_ec_logits`. Add only the missing bound EC prediction export needed for final comparison; test metrics themselves already exist. Conflicting labels within one protein must fail this campaign instead of disappearing from a denominator.

Use `evaluate_saved_checkpoint` for the later authorized final evaluation only after checking its prepared inputs, restored normalization and supplied checkpoint identity. Keep prediction export paths separate from fitting. `export_validation_predictions.py` is validation-only and must retain its prohibition on held-out inputs; reuse its helpers/patterns without turning off that guard.

### Step 5 — Resolve the conditional binding-residue study without expanding the architecture grid

First document what the existing baselines already receive. The partial original quotation requires prior benefit; no such transferable benefit was independently certified in this review. A small positive historical metal delta, or agreement between the plans, is insufficient. The smallest admissible evidence check is the exact named **validation-only** comparison for the same mechanism, including its paired runs and class tradeoffs; do not inspect historical held-out results or restart the separate metal task.

If that condition is not met, complete the core comparison and report the condition as unmet. Do not launch extra binding arms automatically. If it is met, freeze one matched EC ablation of the evidenced mechanism before outer access, with the same folds/seeds and a separately counted budget. The change must demonstrably alter the intended inputs:

- Adding an existing first-shell flag is a no-op.
- Omitting `is_first_shell,is_second_shell,ca_to_metal,fg_to_metal,min_donor_to_metal` removes those particular channels only; vectors, site statistics and pocket selection still contain metal information.
- Existing distance-restricted pooling is a Cα-proximity readout test, not exact first-shell selection. It affects both branches in late fusion. `metal_distance_pool_mask` falls back to the first available node for an empty mask; count and resolve such cases before interpreting the experiment.
- A donor-shell mask and two-view pooling are different, currently unimplemented changes. Do not substitute them for an evidenced mechanism or add them speculatively.

Do not promote an added arm from a bootstrap of nominally independent fold-by-seed scores. Until support and paired uncertainty justify a stronger conclusion, report it as an exploratory EC ablation and retain all three baseline results.

### Step 6 — Train, select and refit DeepMzyme without outer-fold contamination

After successful development-only smoke checks, run the matched two-LR/two-seed grid in each outer fold. Within each fold and family, choose LR by mean inner-validation group EC1 BA across the two seeds; exact ties use the lower LR. Choose each checkpoint by that same validation metric, retaining the existing earliest-best tie behavior. Never choose a favorable seed or change the candidate grid from another fold's results.

Freeze one configuration per family per outer fold, plus a final duration equal to the rounded median selected epoch across its two seed runs, bounded to 1–30 epochs. Refit independently from initialization on that fold's full eligible training membership for each declared seed. Use the **terminal** `last_model_checkpoint.pt`, not a training-loss-selected “best” checkpoint. Save a selection/refit manifest binding memberships, hyperparameters, epoch rule, normalization, seeds and checkpoint hashes before evaluation. No ensemble is required; report both predefined replicate results and their mean.

For the generic CLI refit, `--val-fraction 0` and `--selection-metric train_loss` satisfy the current configuration interface; the runner must explicitly bind the terminal checkpoint rather than the separate best-training-loss artifact. Keep test evaluation disabled during refitting. Only the later verified evaluation may set `--allow-final-refit-test-eval` with `--final-test-selected-config-id`; never use the debug test bypass. Define median rounding deterministically as nearest integer with half values rounded upward.

The core DeepMzyme grid is 60 validation fits plus 30 refits, conditional on the support gate. This is a transparent count, not a runtime estimate or launch authorization.

### Step 7 — Run the two CLEAN regimes from a pinned source

Use a separate CLEAN environment and record the exact upstream commit and dependency/model versions. Reuse the existing notebook's useful normalization and preprocessing code through a small importable helper/runner; limit notebook edits to that integration, phase separation and the required matrix. Set its benchmark selector to `clean30` only.

Training preparation must read only the current fold's train CSV. Build R2 by an exact accession join against its conservative training summary; never fall back to test sequences. Reject sequence conflicts and keep full native EC annotations. Mutated single-EC examples, distance maps, hard negatives and class centers must be generated only from the current regime's training members. Keep generated assets and model/result names scoped by fold, regime and seed.

Retain the planned upstream triplet recipe: ESM-1b inputs, LR `5e-4`, 2,500 epochs, max-separation inference. Use two actual seeds, 42 and 43, in **both** regimes. The inspected [upstream trainer](https://github.com/tttianhao/CLEAN/blob/main/app/train-triplet.py) does not expose a seed argument and calls `seed_everything()` internally; [upstream utilities](https://github.com/tttianhao/CLEAN/blob/main/app/src/CLEAN/utils.py) default to 1234. A small recorded seed adapter must demonstrably control training and augmentation; merely naming jobs with different seeds is insufficient. Recheck these details at the pinned commit.

Run a short development-only smoke that exercises preprocessing, training, checkpoint reload and inference. Verify the upstream trainer's checkpoint cleanup with the shortened epoch/adaptive-rate settings; its current cleanup assumes certain checkpoint files exist. This is a plumbing check, not a 200-epoch experiment.

R1's eventual all-test inference supplies both its all-protein and metallo scores. R2's required endpoint is metallo only. R2-to-all and the complement subgroup are optional descriptive slices, not prerequisites for completion: only Opus asserts the former as required, and the complete original instruction is unavailable. The complement is “outside the constructed metallo cohort,” not established non-metalloenzymes. Do not use resemblance to published test metrics as a gate for retraining or tuning. Verify recipe, inputs and synthetic scoring instead.

This adds 20 CLEAN training fits. The core campaign therefore contains 110 substantive fits before any separately justified binding ablation. Measure preparation, embeddings, memory, training and transfer costs in smokes; freeze a measured execution ceiling and durable storage plan before long runs. Do not inherit either source plan's dates, GPU availability or unmeasured dollar/GPU-hour estimates.

### Step 8 — Freeze the evaluation contract, then perform authorized benchmark reporting

This step is future work. No held-out files or metrics are needed for steps that choose models. Before opening them, freeze all five folds' configuration/refit manifests, CLEAN models, label rules, cohort rules, prediction rules, contrasts and uncertainty method. A separate non-learning integrity check may then inspect benchmark membership/sequences for duplicate, overlap and remoteness auditing. If it finds a violated benchmark contract, stop reporting and document the issue; do not silently repair membership and retain the original benchmark name.

Define the common evaluation cohort by the conservative metallo source, the frozen unambiguous-EC1 rule and the predeclared input-validity policy. Reconcile `(fold, accession)` predictions against the full intended list. Feature/inference failures must not create a quiet intersection of whichever models succeeded: stop for repair without metric-guided changes, or explicitly retain failures in coverage and failure-inclusive reporting. Apply the same denominator to every compared system. A valid EC1 absent from a training vocabulary must not disappear via `require_full_labels`; record it as unsupported and retain its error in common scoring.

Add `scripts/score_clean30_ec_comparison.py`, reusing repository metric conventions where valid. For the unambiguous cohort, compute protein EC1 BA, macro-F1, accuracy, class supports/recalls and a fixed 7×7 confusion matrix. For CLEAN, use the first/closest returned EC prediction projected to EC1, with a defined missing-prediction outcome. Full-EC distance scores are not calibrated EC1 probabilities. For all-protein CLEAN reporting, separately provide top-1 any-true EC1 accuracy and explicitly labeled set-based metrics on ambiguous proteins; do not label multi-label recall as the single-label head-to-head BA. No EC2–EC4 comparison is required.

Publish all five outer pairs. Primary common-cohort reporting is pooled protein-level EC1 scoring per seed, followed by the mean of the two seed metrics; this is not a prediction ensemble. First verify whether an accession occurs in multiple outer sets. If it does, weight its occurrences to total one protein and retain them together in uncertainty resampling. Report fold results and fold mean/SD where the fixed endpoint is estimable; do not average different “present-class” macro denominators as though they were the same seven-class score. Full-task BA is undefined when a required true class is absent; report the present-class diagnostic separately.

Use paired resampling of the same independent protein/homology groups for model differences, preserving repeated occurrences and the seed replicates. Report 95% intervals with a fixed seed and 10,000 resamples, plus per-class denominators. These are uncertainty estimates conditional on the trained models; shared training sets and two seeds do not provide many independent experiments. Sparse/absent classes can make full-task intervals unestimable; report that outcome instead of forcing a favorable interval. Do not infer equivalence from a nonsignificant difference or declare a winner from several unadjusted comparisons. Remove McNemar as a redundant default.

Record authorized access and benchmark scope in `docs/DATASETS.md`. Outer results never revise the primary report, seed list, epoch rule, training cohort, feature arm or model choice. Existing project uncertainty about a pristine final EC route remains separate from this named CLEAN benchmark.

## 6. Deliverables, proportionate checks and completion

Future implementation deliverables under the campaign directory:

- `protocol.json` and `source_manifest.json`: frozen scope, candidate grid, provenance, annotation rules, search definition, input differences and evaluation contract.
- Per-fold `eligible_proteins.csv`, `label_audit.csv`, `site_to_pocket.csv`, `inner_membership.csv`, `support.json`, `feature_audit.json` and `split_audit.json`, with checksums.
- `commands.txt`, `run_matrix.csv` and a dry-run report proving that preparation/training cannot open outer paths.
- Per-run existing `run_config.json`, `run_metadata.json`, `dataset_summary.json`, epoch metrics and checkpoints; bound pocket/protein prediction tables.
- Per-fold/family `selection.json` and `final_refit_manifest.json`; equivalent CLEAN manifests including actual seed and mutation provenance.
- `comparison.csv`, `comparison.md`, `per_class_metrics.csv`, confusion matrices, paired-difference intervals, coverage/exclusion counts and an access receipt. No guessed test counts from either proposal are hardcoded.

Validate changes first with syntax checks and focused synthetic tests: incomplete/multiple/conflicting EC labels; selected and unselected metals sharing a cluster; two selected ions remaining one pocket; duplicate/protein/component split crossings; frozen roles surviving seed/family changes; group weighting and mean-logit aggregation; empty pooling masks if used; missing class/prediction denominators; fixed final-refit checkpoint identity; CLEAN sequence-conflict rejection and seed propagation. Use temporary directories outside the data tree. Then run the required repository smoke suite and one-epoch training-only model smokes for the three families, plus the short CLEAN smoke. Confirm cache reuse versus fresh preparation agrees and prediction export reproduces selected validation metrics. Do not launch the old CARE runner or a real outer-test smoke.

Implementation targets remain limited to the EC campaign runner/scorer, selected-site and membership support, preflight/export integration, and CLEAN preparation. Reuse existing training, losses, heads and feature encoders. Once implemented, coordinate the EC playbook, dataset provenance, status note and evidence index with their owning facts; this review changes none of them.

Scientific completion requires all five benchmark pairs, all required systems and both declared seeds; reconciled labels/input cohorts; admissible inner splits; validation-only selection; completed frozen refits; a common evaluation denominator; reproducible prediction artifacts; and claims restricted to supported classes and the actual weak-supervision endpoint. A feature or model need not improve to complete the study. A feasibility stop is a valid documented finding but is **not** a completed seven-class comparative benchmark. Missing runs or unsupported rare-class claims must remain visibly incomplete.

## 7. Remaining issues and smallest decisive checks

| Issue | Smallest check | Decision it controls |
|---|---|---|
| Original task/clarifications are only partially preserved. | Obtain any missing original instruction; otherwise retain the explicitly stated scope reconstruction. | Whether optional R2-to-all or an unconditional binding study is actually required. No silent scope expansion. |
| Exact meaning/guarantee of upstream split30. | Read the upstream split-generation method and identify the local release/checksum; later perform the frozen, authorized identity audit. | Whether “30% remote” is a defensible claim; official membership stays unchanged unless a separately named dataset is authorized. |
| Pocket-to-activity truth is absent. | Check schema/source annotations for an explicit catalytic-site-to-EC mapping; none appears in the inspected path. | Continue the stated weakly supervised protein-EC1 endpoint; a genuine pocket-function benchmark requires new curated labels and separate scope. |
| Conservative geometry is currently not enforced. | Synthetic extraction test plus the named training examples `A0R3R7`, `B0XTJ7`, `A3A8Q4`. | Correct selected-site extraction before any fit; then recount pockets and dependent features. |
| Rare-class independent support. | Per-fold train-only similarity components and post-filter support tables. | Admit seven-class validation or stop for a scientific feasibility decision; never break groups to manufacture support. |
| Q96KN2 sequence mismatch and unverified feature coverage beyond fold 0. | Compare sequence/source versions and audit eligible training features per fold. | Repair an erroneous cache or apply a recorded common eligibility/version policy before freezing inputs. |
| Prior benefit needed for the conditional binding request. | One exact validation-only paired comparison of the proposed mechanism. | Admit one matched ablation or record the condition as unmet. |
| Upstream CLEAN version/runtime/seed behavior. | Pin code, inspect options, then synthetic/development smoke with checkpoint reload. | Certify the recipe and measured budget; do not infer readiness from the notebook's presence. |
| Duplicate or overlapping outer evaluation units. | After freeze and authorization, compare accession/sequence/object identities and the declared homology criterion. | Stop on contract violations; otherwise bind pooling weights and grouped uncertainty to the actual units. |

**Review boundary:** source plans, relevant code/configuration, documentation, training tables, five training structures and fold-0 training embedding sidecars were inspected. No training, cache generation, held-out dataset loading or source-plan edits occurred. The required status-document read incidentally exposed unrelated historical test summaries; they were excluded from reasoning and are not reproduced here. Published example outputs on the upstream README were likewise not used as evidence or tuning targets. Only this final plan is created by the review.
