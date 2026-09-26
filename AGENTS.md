# AGENTS.md — DeepMzyme

## Project purpose

This repository develops **DeepMzyme**, a deep-learning framework for metalloenzyme metal-type prediction and EC/function prediction.

The main goals are:

1. Predict the primary four-class transition-metal target (Mn, Cu, Zn, and
   Class VIII = Fe+Co+Ni) from protein structural pocket graphs and ESMC
   embeddings/models.
2. Predict enzyme class / EC-level labels as a separate primary task, beginning
   scientifically at EC depth 1.
3. Evaluate shared metal-EC learning as a controlled challenger while keeping
   both tasks independently trainable, selectable, reportable, and publishable.
4. Compare model variants fairly using validation metrics, with held-out test
   metrics reserved for final reporting after validation-based selection.
5. Complete the separate controlled metal-model comparison matrix: direct
   four-class training vs six-class training with collapsed-four evaluation;
   early vs late vs hybrid ESMC fusion; Only-ESM (ESMC) vs Only-GVP vs combined
   GVP+ESMC; and matched GVP with vs without RING edges.
6. Keep the code reproducible, simple to run, and suitable for publication-quality experiments.


---

## Environment

### Required GPU workflow

Before planning or performing DeepMzyme GPU execution, capacity search,
connection, monitoring or recovery, read and apply
[`gpu-use-skill`](.agents/skills/gpu-use-skill/SKILL.md). This includes training,
HPO, embedding generation, inference and GPU benchmarks. It is the canonical
project GPU entry point; agents without automatic skill discovery must open
the file directly. Pure CPU/code/documentation work does not require a GPU.

The skill owns bounded routing and optional audit/monitor delegation. Provider
runbooks and the installed controller own actual commands and safeguards;
scientific recipes remain in the task playbooks. Reuse existing authorization
and verified artifacts within their scope. A skill invocation alone does not
authorize spending, migration, another provider or a larger budget. One
coordinator owns allocation/submission/shutdown; an optional monitor is read-only
and never substitutes for stop controls. Do not use retired provisioning or
automatic replacement scripts as shortcuts.

The GCP limit is one **active** GPU. A positively confirmed stopped source VM
may coexist temporarily with its replacement through the controller's bounded
`vm-fallback` recovery. Preserve source data until fit/replay/independent-backup
verification permits approved cleanup. Follow the skill and provider runbook;
manual zone changes do not migrate disks or bypass resource-state checks.

### Python interpreter

Use this Python interpreter unless explicitly instructed otherwise:

/home/mechti/miniconda3/envs/DeepMzyme/bin/python

Before running scripts, verify the interpreter with:

/home/mechti/miniconda3/envs/DeepMzyme/bin/python -c "import sys; print(sys.executable)"

Do not assume that the shell's default python or python3 points to the correct environment.

---

## Main development guidance

### 1. Follow Plan.md as the main design authority

Plan.md is the main source of truth for the intended architecture, training logic, experiments, and project direction.

When there is a conflict between Plan.md and other scripts, prefer Plan.md, unless the existing code clearly contains newer working logic that should be preserved.

Before making major changes, inspect Plan.md and any script directly related to the requested task.

Do not make large architectural changes that contradict Plan.md unless explicitly requested.

---

### 1b. Read EXPERIMENT_STATUS.md for current progress

If `EXPERIMENT_STATUS.md` is present at the repository root, read it before suggesting
the next experiment, the next baseline, or the next hyperparameter sweep. It is a
lightweight, frequently updated summary of where the project currently stands
(stage, current best validation result, trusted split, next planned experiment,
known caveats, and test-set rules).

How to weigh these sources:

- For design intent and planned project direction, `Plan.md` remains the main
  design authority.
- `EXPERIMENT_STATUS.md` is a mutable current-progress note and is lower
  authority than `Plan.md`.
- Current source code under `src/` and run outputs under
  `DeepMzyme_Data/notebook_outputs/runs/` are evidence of actual implemented
  behavior and actual results, not statements of design intent.
- If code, run outputs, `Plan.md`, or `EXPERIMENT_STATUS.md` conflict, report
  the conflict clearly instead of silently choosing one.

`EXPERIMENT_STATUS.md` is the right place for short-lived experiment-status notes.
Do not move that kind of frequently changing state into `AGENTS.md` or `Plan.md`.

Default experiment-planning posture:

- When the user asks for a new check, new run, next Optuna sweep, or fresh
  experiment without explicitly saying to rely on previous runs, treat previous
  raw outputs as context and guardrails only. Do not over-anchor the new plan to
  old raw notebook outputs.
- In that default case, prefer the largest sensible validation-only Optuna
  search space the current stage can support, with common-sense constraints on
  runtime, model family, available features, split policy, and final-test
  protection.
- When the user explicitly asks to rely on previous running/results/raws,
  inspect the named evidence and use it directly to narrow, continue, or repeat
  the prior configuration.

---

### 1c. Preserve scientific task and target boundaries

- Metal classification and EC/function classification are separate primary
  missions. Joint learning is an experiment, not a scientific requirement or
  an assumed improvement.
- The intended primary PinMyMetal-compatible reporting endpoint is Mn, Cu, Zn,
  and Class VIII = Fe+Co+Ni. Direct training uses `four_class`, canonical name
  `merge_fe_class_viii`; do not invent another scheme. Keep historical
  six-class and five-class evidence labeled exactly as run. Direct four-class
  training is not equivalent to six-class training followed by collapsed-four
  reporting.
- Both metal training formulations are required in the controlled baseline
  campaign: direct `four_class` training and matched `six_class` training with
  deterministic collapsed-four evaluation. Run both for Only-GVP, Only-ESM,
  and GVP + graph-level late fusion. Compare them on shared validation units
  and the common four-class view; keep native six-class metrics and Fe/Co/Ni
  recalls for the six-class arm. Use separate run/study identities and never
  use held-out results to choose between them.
- EC starts at depth 1. Keep depth-1 classification, deeper hierarchical
  classification, multiple source EC annotations, and full multi-label EC
  prediction distinct. The current target path is single-label at the selected
  depth and is not a full multi-label solution.
- Before cross-task claims, establish standalone Only-GVP, Only-ESM, and GVP +
  graph-level late-fusion baselines for both tasks. Do not add hybrid,
  node-level late fusion, cross-attention, or other complex fusion to the first
  metal-EC relationship experiment.
- The first cross-task question is whether auxiliary metal supervision improves
  EC. Use independent metal and EC predictions from a shared encoder; neither
  head's output feeds the other. Treat negative transfer as possible and retain
  standalone models when they validate better. The reverse direction is
  secondary and optional.
- Do not make predicted metal a mandatory EC input. Soft predicted-metal
  conditioning and known-metal EC diagnostics are later optional ablations.
  Distinguish observed, curated, computationally transferred AlphaFill/MAHOMES,
  and model-predicted metal; never call transferred assignments perfect ground
  truth.
- Metal labels are per clustered pocket or per ion according to
  `metal_example_unit`; EC labels are protein/structure-level.
  Preserve protein/structure grouping and EC group weighting. If any protein is
  held out for either task, exclude it from every shared-encoder training loss,
  including labels from the other task.
- Before describing sample counts, predictions, or multinuclear handling,
  verify `metal_example_unit` in the actual run configuration and follow
  [Plan.md's metal example terminology](Plan.md#metal-example-terminology).
  Never infer the prediction unit from `PocketRecord`, `pocket_id`, or the
  word "pocket." Distinguish the record/example ID, `parent_pocket_id`, and
  validation group; `split_by="pocket_id"` uses the parent when present.
  Report ion examples, parent pockets, and PDB groups separately. Preserve
  historical runs' recorded units; an ion campaign does not change the
  repository-wide default or make EC supervision per ion.
- Any metal x EC1 association analysis is development-data-only and descriptive.
  Plan counts, conditional probabilities, Cramer's V, appropriate chi-square,
  mutual information, and normalized mutual information without using held-out
  data or treating association as proof of learning benefit.
- Use status labels literally: planned, implemented, smoke-tested,
  experimentally evaluated, and promoted are different states. A code path is
  not validation evidence.

### 1d. Key project files and directories

Use this as a navigation map when a task touches the relevant area. Do not read
every file for every small request; inspect the applicable files before making a
claim or change.

#### Primary authority and status

- `Plan.md`: design authority for architecture, experiment policy, validation
  selection, and held-out test rules. Contains the document map for all
  related files.
- `docs/README.md`: top-level documentation index for validation/testing,
  notebook, playbook, copied-output documentation, Drive/local output
  handling, and copied-evidence placement rules.
- `EXPERIMENT_STATUS.md`: current experiment status, selected validation
  anchors, trusted evidence files, caveats, and next planned action.
- `docs/DATASETS.md`: authoritative dataset/split/bundle inventory, current
  availability, provenance links, and historical test-use ledger.
- `docs/PARAMETER_FINDINGS.md`: validation/HPO findings with evidence grades;
  historical test metrics are excluded from parameter conclusions.
- `docs/FOLLOW_UP_TECHNICAL_ISSUES.md`: verified implementation/documentation
  problems that remain deliberately unfixed pending separate authorization.
- `README.md`: public-facing overview and minimal quick start. Good entry point
  for understanding what the project does; it does not own live defaults.
- `docs/GETTING_STARTED.md`: executable orientation, current local-environment
  limits, first checks, and the shortest navigation route from checkout to the
  correct stage/evidence owner.
- `docs/COLAB_GPU_RUNBOOK.md`: browser/CLI same-VM connection, Colab
  PyTorch-preserving dependency installation, CUDA architecture preflight,
  Drive authorization boundary, artifact transfer, and teardown.
  Read it before Colab GPU work: it records verified hurdle fixes and an
  optional Python 3.12 ESMC preparation route for Python 3.13 runtimes.
  That route is a tested convenience, not a requirement; skip the extra
  environment when the runtime already uses compatible Python 3.12.

#### Notebook workflow and training recipes

- `notebooks/DeepMzyme_training_colab.ipynb`: actual Colab planning, command
  expansion, run execution, skipping, capping, and reporting behavior. The
  single notebook supports all tasks (metal, EC, joint) via `TASK` selection.
- `docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md`: stable notebook workflow and
  option meaning for metal classification, including notebook execution order,
  Optuna behavior, and safety policy. It is not a live results table.
- `docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`: staged, copy-paste-ready notebook
  configuration blocks for metal classification. Use this as the practical
  execution recipe and exact parameter source for each training stage (smoke,
  baseline, Optuna, Stage 6 grouped-fold confirmation, final test). For
  G4-class GPU planning, this is where serious/custom Optuna budgets and search
  spaces should be recorded.
- `docs/EC_TRAINING_PIPELINE_PLAYBOOK.md`: same staged structure as the metal
  playbook, covering EC-number classification. Covers EC label depth, group
  weighting, contrastive loss progression, and 200-trial Optuna examples.
  Read its compatibility warning before using affected HPO or final-test
  blocks; reconciliation with the current notebook remains an open technical
  task.
- `docs/archive/workflows/list_train_commands_legacy.md`: historical direct
  `src/train.py` command examples. Use `src/train.py --help` plus current
  playbooks for live execution.

#### Experiment evidence

- `docs/notebook_outputs/README.md`: authoritative experiment-batch index and
  copied-evidence contract; read this before browsing raw files.
- `docs/notebook_outputs/summaries/`: short human-readable run summaries and
  historical planning notes; read these before raw outputs when tracking
  experiment status.
- `docs/notebook_outputs/raw/`: copied notebook outputs used as portable
  evidence.
- `docs/archive/experiments/experiment_notes_legacy.md`: intact early
  learning-rate/epoch notes and historical test-access evidence; not a design
  or current parameter-selection document.
- `docs/archive/`: historical command references, audits, session notes,
  experiment snapshots, and changelogs. They remain recoverable but are not
  current authority.
- `DeepMzyme_Data/notebook_outputs/runs/`: local run outputs when present;
  treat these as measured evidence, not design intent.

#### Training source code

- `src/train.py`: main training entry point for all tasks (`--task metal`,
  `--task ec`, `--task joint`). Delegates to `src/training/config.py` and
  `src/training/task_entrypoint.py`.
- `src/train_metal.py`: thin task-specific wrapper that invokes the metal
  training path directly, bypassing joint-task dispatch.
- `src/train_ec.py`: thin task-specific wrapper that invokes the EC training
  path directly.
- `src/training/config.py`: CLI and training configuration parsing. The
  authoritative list of all CLI flags and their defaults.
- `src/training/task_entrypoint.py`: dispatches training to the correct task
  head after configuration is parsed.
- `src/training/run.py`: main training loop.
- `src/training/loop.py`: epoch-level training and validation logic.
- `src/training/data.py`: dataset loading and preparation.
- `src/training/splits.py`: train/validation split logic.
- `src/training/labels.py`: label extraction and EC depth handling.
- `src/training/preflight.py`: pre-training validation checks (paths, splits,
  feature availability, ESM/RING coverage).
- `src/training/defaults.py`: default values for training configuration.

#### Model source code

- `src/model.py`: model definitions; may contain experimental or non-final
  code. See the caution below before editing.
- `src/model_variants/factory.py`: model instantiation factory.
- `src/model_variants/models.py`: concrete model variant definitions.

#### Graph and feature extraction

- `src/graph/construction.py`: pocket graph construction from structure files.
- `src/graph/ring_edges.py`: RING interaction edge loading and generation.
- `src/graph/shell_roles.py`: first/second shell residue role assignment.
- `src/featurization.py`: residue-level featurization pipeline.
- `src/feature_extraction/`: PROPKA, physicochemical, and external feature
  extraction modules.
- `src/embed_helpers/esmc.py`: ESMC embedding generation and loading.
- `src/embed_helpers/Interaction_edge.py`: interaction edge helpers.
- `src/label_schemes.py`: metal and EC label scheme definitions.
- `src/data_structures.py`: shared data container types.

#### Data preparation utilities

- `src/build_dataset_csv.py`: builds site-level MAHOMES-format summary CSVs
  from PDB structures and PinMyMetal labels. Run this before training when
  creating a new split.
- `src/build_colab_bundle.py`: packs a Colab-ready `.tar.zst` data bundle from
  a specified split directory. Run this to produce the bundle uploaded to
  HuggingFace or used via Drive.
- `src/structure_store.py`: manifest schema and resolver for the shared
  content-addressed structure store.
- `src/manage_structure_store.py`: audits, migrates, and verifies structure
  storage without changing scientific split membership.
- `prepare_training_and_test_set/`: original split preparation scripts.
  Downloads PDB structures, creates non-redundant chain files, and runs MAHOMES
  activation to produce site-level summary CSVs. Scripts are named
  `step1a_...`, `step1b_...`, etc. for sequential execution.
  `prepare_training_and_test_set/pinmymetal_files` contains the original
  PinMyMetal train/test membership files.

#### Reporting

- `src/report_runs.py`: run-summary and comparison-table generation. Summarizes
  multiple run directories into a single CSV. Used by the notebook summary cell
  and can be run standalone.

#### Data directories

- `DeepMzyme_Data/structure_store/`: one object per distinct structure-file
  byte content, keyed by SHA-256. Split and CLEAN shared-structure directories
  use `structure_manifest.csv` to reference these objects; see
  `docs/STRUCTURE_STORE.md`.

- `DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/`:
  historical non-overlap split path. It is present locally in manifest-backed
  form as of 2026-09-14, and its test was evaluated in seven early runs, so it
  is not pristine.
  Do not recommend final reporting until the primary final-test route is
  resolved scientifically; see `docs/DATASETS.md`.
- `DeepMzyme_Data/train_and_test_sets_structures_exact_pinmymetal/`:
  exact PinMyMetal train/test PDB-ID membership for available supported
  structures; may contain train/test PDB-ID overlap.
- `DeepMzyme_Data/train_and_test_sets_structures_harsh_pinmymetal/`:
  harsh split where all common PDB IDs are assigned to test.
- `DeepMzyme_Data/train_and_test_sets_structures_common_pdbid_70_30_pinmymetal/`:
  custom comparison split where train-only PDB IDs stay train, test-only PDB
  IDs stay test, and only common exact-split PDB IDs are assigned as whole
  groups at a 70/30 train/test ratio; not the main final held-out split.
- `DeepMzyme_Data/esm_embeddings/`: precomputed ESMC residue embeddings.
  Pass this path via `--esm-embeddings-dir` or the notebook `ESM_EMBEDDINGS_DIR`
  variable. Do not commit embeddings to git.
- `DeepMzyme_Data/RING_features/`: precomputed RING interaction edge files.
  Pass this path via `--ring-features-dir` or `RING_FEATURES_DIR`.
- `DeepMzyme_Data/notebook_outputs/runs/`: local run output directories.
  Treat as measured evidence, not design intent.
- `DeepMzyme_Data/DeepMzyme_Colab_Bundles/`: built `.tar.zst` data bundles.

Current materialization and bundle inclusion can change. Verify them in
`docs/DATASETS.md` and the filesystem instead of assuming every named path
above exists.

#### Internal and staging

- No `internal/` workflow is part of the active pipeline. If an
  `internal/codex_suggested/` directory is reintroduced later, treat it as
  unreviewed staging material only, not production code.

**Read by default** (small, high-leverage): `Plan.md`, `EXPERIMENT_STATUS.md`,
`docs/README.md`, `docs/DATASETS.md`, and
`docs/notebook_outputs/README.md`.

**On-demand only** (large; do not bulk-load):
- `docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md` — only when editing or running
  the Colab metal workflow.
- `docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md` — only when setting up or
  executing a specific metal training stage.
- `docs/EC_TRAINING_PIPELINE_PLAYBOOK.md` — only when setting up or executing
  a specific EC training stage.
- `docs/archive/workflows/list_train_commands_legacy.md` — historical context
  only when recovering an old direct CLI invocation.
- Files under `docs/notebook_outputs/raw/` — only when a summary cites the
  file or when exact logs / run commands are needed.
- `notebooks/DeepMzyme_training_colab.ipynb` — only for notebook-behavior
  questions.
- `src/training/config.py` — only when checking the exact CLI flag name or
  default value.
- `DeepMzyme_Data/notebook_outputs/runs/*` — only when `EXPERIMENT_STATUS.md`
  names a specific run.

Read individual files under `docs/notebook_outputs/summaries/` by name; do not
bulk-load all summaries.

For experiment-status questions, first read `EXPERIMENT_STATUS.md`, then inspect
the specific evidence files it names. For notebook behavior questions, inspect
the notebook itself rather than relying only on the workflow guide.

To answer "what is the next metal-training step", the agent must:

1. Read `EXPERIMENT_STATUS.md` to find the current stage anchor.
2. Read the corresponding stage block in
   `docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`.
3. Confirm the decision gate of the previous stage was passed.
4. Output: (a) the exact notebook configuration block to paste, (b) the
   expected outputs/files, (c) the decision gate that determines whether to
   proceed to the next stage. Do not invent budgets; cite the playbook stage.
5. Never silently recommend held-out test evaluation before Stage 7, and never
   recommend Stage 7 unless Stage 6 has selected one configuration from
   grouped-fold confirmation or an explicitly labeled fallback, Stage 6B has
   produced a completed/reused final full-train refit for that configuration,
   and that refit is fixed as the Stage 7 source.

Use the playbook stage names exactly:



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

The controlled comparison matrix in `Plan.md` is a required research mission,
not an optional interpretation of unmatched historical maxima. Before claiming
that a target formulation, fusion position, combined modality, or RING edge
source is better, verify that the applicable candidates were compared on shared
validation folds/seeds with paired confidence intervals and rare-class recall
protection. The target-formulation comparison uses the common four-class view;
the other architecture comparisons keep direct four-class training fixed.
This separate metal-architecture matrix does not expand the initial
metal-EC auxiliary experiment beyond Only-GVP, Only-ESM, and graph-level late
fusion.
Use `EXPERIMENT_STATUS.md` for the current completion state and the metal
playbook for exact runnable blocks. If a required block is missing, document it
in the playbook before recommending execution.

#### Metal Colab pipeline documentation policy

When the task touches the metal-training notebook, configuring a stage, or
planning the next Optuna sweep:

- Treat `notebooks/DeepMzyme_training_colab.ipynb` as implemented behavior and
  the live notebook-default surface.
- Treat `docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md` as the exact stage-parameter
  recipe; it owns canonical stage budgets, search spaces, and decision gates.
- Treat `docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md` as the option-meaning
  reference and stage-to-option crosswalk.
- Treat `Plan.md` as design authority and policy (selection metric, split
  policy, held-out test rules, advanced-fusion ordering).
- Keep mutable "current best result" status in `EXPERIMENT_STATUS.md`.

Operational assumptions for this project:

- Hardware: G4-class GPU. All serious Optuna runs use the budgets in the
  playbook's "G4-Class Optuna Policy" subsection.
- Persistent Optuna storage in Drive is mandatory for Stage 4 and Stage 5.
- Persistent Optuna studies must not silently mix incompatible `MODEL_PRESET`
  values or incompatible search spaces. Keep
  `OPTUNA_ALLOW_INCOMPATIBLE_STUDY_REUSE = False` for reportable HPO unless the
  user explicitly asks for a labeled recovery/debug override.
- Stage 7 (held-out test) is one-shot per final training/refit run derived from
  a final validation-selected configuration.
- After Stage 6/cross-validation, the selected configuration must be promoted
  and trained/refit once before held-out testing. Stage 6 selects the
  configuration, Stage 6B applies promotion gates and creates the final
  full-train refit, and Stage 7 reports the frozen final-training
  checkpoint/run.
- Stage 7 reporting improvements (ensemble, calibration, temperature scaling,
  plots, bootstrap CIs) must not weaken the one-shot policy. The primary final
  report, final training/refit source run, ensemble source list, averaging rule,
  and calibration rule must be fixed before opening the held-out test, and test
  metrics must never be used to switch the primary report or choose a different
  checkpoint/configuration.
- Stage 6 defaults to
  `TOP_CONFIG_REEVALUATION_MODE = "group_kfold_seed_repeat"` for top-K 5-fold
  grouped validation by `pdbid` crossed with the configured `REPEAT_SEEDS`
  model-seed list, with the same fold definitions and active seed list for
  every compared candidate. Plain `group_kfold` is a one-seed grouped-fold
  option, and `seed_repeat` is exploratory. Candidate promotion uses paired
  bootstrap confidence intervals and rare-class recall protection, not raw
  validation deltas alone.
- Stage 6B is the named validation-to-final-refit bridge. It ranks Stage 6
  candidates by mean `val_metal_balanced_acc`, applies configurable paired-CI,
  rare-class recall, and tie-breaker gates, then optionally trains one final
  full non-test training-set refit for the selected configuration. Stage 6B
  writes `stage6b_decision.json`, `stage6b_ranked_candidates.csv`,
  `stage6b_final_refit_command.txt`, and, only after a completed/reused refit,
  `stage6b_selected_final_refit_candidate.json`.
- Exact executable stage values, Optuna budgets, search spaces, seed lists,
  expected outputs, and decision gates are owned by
  `docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`; notebook defaults should be kept
  coordinated when they are intended to represent the current canonical workflow.

#### Required answer format for metal notebook stage requests

When the user asks what to run next, how to configure the metal Colab notebook,
or how to update the metal training pipeline documentation, answer using this
format:

1. Current stage assumption:
   - State which stage is being configured.
   - State whether the answer relies on `EXPERIMENT_STATUS.md` or is a fresh
     validation-only plan.

2. Exact notebook block:
   - Copy the block from `docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`.
   - Do not invent budgets.
   - Do not silently change `OPTUNA_TARGET_COMPLETE_TRIALS`,
     `MAX_EPOCHS_PER_TRIAL`, `OPTUNA_N_STARTUP_TRIALS`, search ranges, seed
     lists, or final-test flags.

3. Safety checks:
   - Confirm `INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False` for all non-final
     stages.
   - Confirm `METAL_LABEL_SCHEME = "four_class"` for the direct arm of a new
     primary metal campaign. The required paired target-formulation challenger
     uses a separately named `six_class` run/study and reports
     `val_metal_collapsed4_balanced_acc` on the common endpoint while retaining
     native six-class metrics. Treat `five_class` only as an explicitly labeled
     alternative or historical scheme. If the paired playbook blocks have not
     yet been reconciled, update the playbook before recommending execution.
   - Confirm `SELECTION_METRIC = "val_metal_balanced_acc"`.
   - Confirm `VAL_FRACTION = 0.15` and `SPLIT_BY = "pdbid"` unless the stage is
     Stage 6 grouped-fold confirmation or an explicitly labeled new split
     experiment.
   - For Stage 6 fold-plus-seed confirmation, confirm
     `TOP_CONFIG_REEVALUATION_MODE = "group_kfold_seed_repeat"`,
     `SEED_REPEAT_N_FOLDS = 5`, a fixed `SEED_REPEAT_SPLIT_SEED`, and shared
     fold definitions and active seed list for every compared candidate. If
     `TOP_CONFIG_REEVALUATION_MODE = "group_kfold"` is used, label it as
     one-seed grouped-fold confirmation because only the first `REPEAT_SEEDS`
     value is active.
   - For Stage 6B, confirm the Stage 6 artifacts exist, promotion ranks by
     mean `val_metal_balanced_acc`, paired-CI and rare-recall gates are
     configured, `LAUNCH_STAGE6B_FINAL_REFIT` is still `False` during preview,
     and the final refit uses the full non-test training set with no held-out
     test evaluation.
   - Confirm one `MODEL_PRESET` per Optuna study.
   - Confirm persistent Drive SQLite storage for serious Optuna stages.
   - Confirm incompatible persistent-study reuse remains blocked unless an
     explicit recovery/debug override is requested.

4. Expected outputs:
   - List the exact expected CSV/JSON/Markdown files for that stage.
   - Identify where the exact run configuration is saved (`run_config.json` /
     `run_metadata.json`; notebook-generated `active_run_config.json` and
     `active_run_config.md`).

5. Decision gate:
   - State what must be true before moving to the next stage.
   - Include held-out-test leakage, expected files, selection metric,
     `MODEL_PRESET`/Optuna-study compatibility where applicable, completed
     trial/run counts, paired bootstrap CI requirements where applicable, and
     rare-class recall protection.
   - For Stage 6, Stage 6B, and Stage 7 transitions, include the required
     Stage 6B final training/refit run between validation/CV selection and
     held-out testing.

If the requested stage block is missing or incomplete, update
`docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md` first instead of patching the answer
with undocumented values.

When editing the metal-pipeline documents together, never copy full
configuration blocks into `Plan.md`, `AGENTS.md`, or
`docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md`. Full executable stage blocks
belong in `docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`; notebook defaults may be
updated separately to match the canonical workflow.

#### Documentation Coordination Protocol

For future documentation edits:

- Identify the full coupled document set before editing, including stable docs,
  mutable status notes, run-evidence indexes, notebook context, and CLI examples
  that restate the same facts.
- Assign each fact to one owning document before changing it. Keep exact metal
  executable stage values in `docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`, EC
  equivalents in `docs/EC_TRAINING_PIPELINE_PLAYBOOK.md`, live notebook defaults
  in `notebooks/DeepMzyme_training_colab.ipynb`, notebook option meanings in
  `docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md`, research policy in `Plan.md`,
  mutable status in `EXPERIMENT_STATUS.md`, datasets/test-use in
  `docs/DATASETS.md`, validation/HPO findings in
  `docs/PARAMETER_FINDINGS.md`, and unresolved behavior problems in
  `docs/FOLLOW_UP_TECHNICAL_ISSUES.md`.
- Keep durable metal-EC auxiliary policy and phase order in `Plan.md`. Put an
  exact EC-primary auxiliary recipe in the EC playbook only after its split and
  leakage protocol is certified; put the optional reverse-direction recipe in
  the applicable playbook if that experiment is later approved.
- Update or re-point every cross-reference in the same change set instead of
  leaving duplicated stale text behind.
- Avoid copying current anchors, run IDs, transient trial numbers, local disk
  state, or mutable best-result notes into stable docs.
- Run a consistency sweep after edits for stage names, split policy, selection
  metrics, held-out-test rules, Stage 6/7 safeguards, seed/bootstrap/pruning
  wording, and notebook option names.
- Report unresolved conflicts explicitly instead of silently choosing one source
  when the repository evidence is insufficient.

---

### 2. Be careful with src/model.py

The current src/model.py may contain experimental, non-final, partially inconsistent, or not fully validated code.

Do not assume every implementation detail in src/model.py is final.

When editing src/model.py:

- Prefer additive, configurable changes over hard replacement.
- Keep backward compatibility with existing training scripts when possible.
- If Plan.md and src/model.py disagree, treat this as a design issue and resolve it conservatively. Plan.md should be higher in your hierarchy of decision.

---

### 3. Keep experiments fair and reproducible

When adding or modifying training/evaluation code:

- Use validation metrics for model selection.
- Keep the held-out test set for final reporting only.
- Avoid tuning directly on the test set.
- Save enough metadata to reproduce results, including:
  - model configuration
  - feature set
  - random seed
  - train/validation/test split
  - loss function
  - class weights or sampling strategy
  - learning rate and scheduler
  - checkpoint selection rule
  - dataset bundle identifier/checksum for serious or final runs
  - key library versions for serious or final runs, especially PyTorch,
    torch-geometric, ESM/ESMC, Optuna, NumPy, and scikit-learn when available

Prefer clear experiment names and structured output directories.

---

### 4. Prefer clean, simple, publication-quality code

Code should be understandable and maintainable.

Prefer:

- explicit configuration over hidden constants
- clear function names
- small helper functions
- readable error messages
- comments only where they clarify non-obvious logic
- minimal duplication

Avoid:

- hardcoded absolute paths unless already part of the project convention
- silent failures
- changing unrelated files
- large rewrites when a focused patch is enough
- adding unnecessary dependencies

---

## Project-specific modeling notes

DeepMzyme may use several information sources, including:

- protein structural pocket graphs
- residue-level geometric features
- ESMC residue embeddings or sequence-derived representations
- optional early, late, or gated ESM fusion
- metal-type classification heads
- EC/function classification heads

When adding model options, make them configurable where reasonable.

For example, prefer command-line/config options such as:

- --use_esm
- --esm_fusion_mode
- --early_esm_dim
- --node_feature_set
- --loss_type
- --metal_loss_weight
- --ec_loss_weight

rather than hardcoding one experimental choice.

---

## Testing and validation

After code changes, run the smallest reasonable checks first.

Syntax checks:

```
/home/mechti/miniconda3/envs/DeepMzyme/bin/python -m py_compile src/model.py
/home/mechti/miniconda3/envs/DeepMzyme/bin/python -m py_compile src/train.py
/home/mechti/miniconda3/envs/DeepMzyme/bin/python -m py_compile src/training/config.py
/home/mechti/miniconda3/envs/DeepMzyme/bin/python -m py_compile src/training/run.py
```

Smoke test (fast, no GPU required):

```
/home/mechti/miniconda3/envs/DeepMzyme/bin/python tests/smoke_checks.py
```

If relevant, run a small smoke test before long training jobs.

Do not launch expensive full training runs unless explicitly requested.

Do not write temporary smoke-test files into `DeepMzyme_Data/` unless this is explicitly needed for the test. Prefer using a temporary directory outside the project data tree, and clean up any temporary files immediately after the test.

---

## Data and paths

Be careful with project-relative paths.

Prefer paths based on the repository root rather than paths relative to the currently running script.

For example, avoid assuming that DeepMzyme_Data/... is relative to src/.

Use robust path construction with pathlib.Path.

---

## Expected behavior

When working on this repository:

1. First inspect the relevant files.
2. Compare the requested change against Plan.md.
3. Make the smallest safe change that satisfies the request.
4. Preserve existing useful options.
5. Run syntax or smoke checks when possible.
6. Clearly summarize what changed and what was not changed.
7. Whenever a completed task adds, edits, or deletes files that belong in the
   repository, stage the task's changes with `git add` using the Git workflow
   below. Leave committing and pushing to the user.
8. In the final response, summarize what was staged and provide a suggested
   `git commit -m "<concise commit title>"` command and the appropriate
   `git push` command for the user to run.

When editing AGENTS.md itself, briefly summarize the changed sections in the
response so the user can review the policy delta without re-reading the whole
file.

---

## Accepted-work budget escalation

- A planned GPU-hour, trial-count, runtime, or other operational budget is an
  auditable ceiling, not permission to abandon an accepted task unfinished.
- Before stopping accepted work solely because the remaining plan no longer
  fits its ceiling, ask the user to choose between exactly these outcomes:
  **stop and close out at the current evidence boundary**, or **continue beyond
  the planned ceiling and record a larger ceiling**. Include the measured use,
  forecast, requested increase, and practical consequence of each choice.
- Ask at the first safe decision boundary, before launching work that requires
  the increase. Do not terminate an active safe fit solely because a forecast
  changed. Provider/session shutdown reserves and ownership safeguards still
  apply.
- If the user already authorized the overrun in the current conversation or an
  existing campaign authorization record, continue without asking again. Do
  not reinterpret a higher compute ceiling as permission to alter the
  scientific grid, folds, seeds, target definitions, held-out policy, or other
  accepted plan details.
- Record each approved increase in the campaign evidence and executable budget
  controls. If the new ceiling later becomes insufficient, ask again with the
  same stop/continue choices instead of silently stopping or silently spending
  beyond the recorded authorization.

---

## Git workflow

- This workflow applies to every agent session in DeepMzyme.
- After completing a task that adds, edits, or deletes files intended for
  version control, automatically stage the task's changes before the final
  response. Use `git add -- <explicit paths>` for files containing only task
  changes, including every new file meant to be part of the repository and
  intended deletions. Do not use blanket `git add .` or `git add -A` commands.
- Inspect the working tree and index before editing and before staging.
  Preserve pre-existing or user-owned staged and unstaged changes. When a file
  also contains unrelated edits, stage only the task's hunks; do not stage the
  entire file or reset the user's index. If the changes cannot be separated
  safely, leave the affected task changes unstaged and explain why.
- Leave `git commit` and `git push` to the user unless they explicitly request
  either operation. Authorization to stage does not authorize committing or
  pushing.
- Verify the staged diff, summarize the task changes staged, and provide a
  concise suggested commit command and the appropriate push command in the
  final response. Mention any pre-existing staged changes that the user's
  commit would also include.
- Do not add generated outputs, credentials, ignored local data, or temporary
  files merely to make the worktree clean. Only add files that belong in the
  repository, respect `.gitignore`, and do not force-add ignored files.
- Read-only tasks and tasks that do not change repository files require no
  staging or commit/push instructions. If the user explicitly asks not to stage
  a task, follow that request and report the remaining unstaged task changes.

---

## Review-only and prompt-safety guidance

When the user asks for a comprehensive project review, architecture audit,
MLOps/testing recommendations, or whether a proposed prompt is safe:

- Treat the task as read-only unless the user explicitly asks for edits.
- Inspect `Plan.md`, `EXPERIMENT_STATUS.md`,
  `docs/notebook_outputs/README.md`, and the directly relevant source files
  before making concrete claims.
- Do not modify files, run training, launch Optuna, evaluate the held-out test
  set, install dependencies, create commits, or reorganize experiment evidence
  during a review-only task.
- If `Plan.md`, `EXPERIMENT_STATUS.md`, source code, and run outputs conflict,
  report the conflict clearly instead of resolving it silently.
- Verify whether a feature already exists before recommending it as missing.
  For example, check the implemented CLI/config before suggesting AMP,
  gradient accumulation, schedulers, node feature sets, or tracking hooks.
- When a proposed prompt lists project-improvement ideas, do not treat the
  list as established fact until each claim has been checked against
  `Plan.md`, `EXPERIMENT_STATUS.md`, `docs/notebook_outputs/README.md`, and
  the directly relevant source files. Label each item as already implemented,
  planned, stale, risky, or genuinely missing.
- For each prompt-safety or project-improvement item, report the evidence,
  whether the recommendation is safe, whether it is documentation-only,
  validation-only, or implementation work, and any conflict between docs,
  source code, and run outputs.
- Rewrite unsafe or overconfident prompts as verification-first prompts. Safe
  prompts should ask the agent to inspect evidence before making claims, avoid
  broad implementation instructions unless explicitly requested, and preserve
  the staged validation workflow.
- Frame external experiment-tracking systems such as WandB, MLflow, or Neptune
  as optional adapters unless the user explicitly asks to adopt one. Do not
  make them mandatory project dependencies by default.
- Treat automated tests and CI as complements to the staged validation
  workflow. Do not suggest replacing Stage 1 smoke checks, Stage 6
  confirmation, or Stage 7 held-out-test policy with CI-only checks.
- For schema-safety concerns, prefer targeted dataset/preflight validation and
  explicit model-mode requirements before proposing broad PyG data-class or
  architecture rewrites.
- Recommendations about model families, Optuna budgets, split policy, or
  held-out testing must preserve the validation-only selection rules and the
  required final training/refit step and one-shot Stage 7 policy unless the
  user explicitly requests a policy change.
- A prompt is not safe for metal-stage recommendations if it silently permits
  held-out test use before Stage 7, recommends Stage 7 before a Stage
  6-selected configuration and Stage 6B frozen final training/refit run exist,
  or weakens the current Stage 6 grouped-fold, paired-bootstrap, rare-class
  recall, and Stage 6B promotion-gate requirements.
