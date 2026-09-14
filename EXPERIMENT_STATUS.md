# DeepMzyme Current Experiment Status

This is the sole concise answer to: **Where am I now, and what should I do
next?** It is mutable. Scientific policy is in [`Plan.md`](Plan.md); exact
experiment history is in the [experiment index](docs/notebook_outputs/README.md).

Last experiment-evidence audit: 2026-08-20. Last execution audit: 2026-08-22.
Last scientific-policy documentation update: 2026-09-14.

## Current objective

Advance two independent primary missions: direct four-class transition-metal
classification and EC/function classification, beginning at EC depth 1. The
immediate cross-task roadmap starts with reconciling strong standalone models;
auxiliary shared learning remains a later challenger.

The current Phase 1 action is to reconcile a paired metal-only target-formulation
campaign for the four-class endpoint: (a) direct four-class training and (b)
matched six-class training followed by deterministic collapsed-four evaluation.
Both arms are required across Only-GVP, Only-ESM, and GVP + graph-level late
fusion. Existing historical anchors use the six-class metal target and one fixed
`pdbid`-grouped validation split. They remain historical evidence and do not
constitute the required matched target-formulation comparison.

The separate metal-architecture mission still includes: (a) early versus late
versus hybrid ESMC fusion, (b) Only-ESM using ESMC versus Only-GVP versus
combined GVP+ESMC, and (c) matched GVP runs with versus without RING edges.
Those advanced comparisons are outside the first metal-EC auxiliary experiment.

No copied candidate has yet passed the current grouped-fold Stage 6 promotion
standard.

## Scientific mission and target status

| Item | Implemented state | Evidence state | Current interpretation |
|---|---|---|---|
| Primary metal reporting endpoint | `four_class` aliases `merge_fe_class_viii` and maps Fe/Co/Ni to Class VIII | No trusted completed direct four-class baseline batch is indexed | Four-class endpoint selected; baseline evidence and paired executable recipes still require reconciliation |
| Required target-formulation challenger | `six_class` outputs and collapsed-four validation metrics are implemented | Historical six-class runs exist, but no matched direct-four versus six-trained/collapsed-four comparison is indexed | Required across the three initial metal baseline families; exact paired recipes still require reconciliation |
| Historical metal targets | Six-class and five-class schemes remain implemented | Indexed metal anchors are principally historical six-class results; one five-class local smoke was prepare-only | Preserve under their original scheme; never relabel as direct four-class training |
| Primary EC task | Depth-controlled single-label EC classification and structure-level group weighting exist | No trusted completed EC model result is indexed | Start with EC depth 1; deeper hierarchical and full multi-label claims are unestablished |
| Current EC multi-annotation handling | Multiple annotations with one shared prefix can map to one class; conflicting prefixes at the selected depth receive no target | No full multi-label experiment is indexed | Current code does not solve full multi-label EC prediction |
| Shared-learning topology | A shared pocket representation feeds independent metal and EC heads, and both losses can update shared parameters | Exploratory joint runs exist under unmatched contexts | Implemented and explored; no controlled EC-only versus EC+auxiliary-metal result and no promotion |
| Predicted-metal conditioning | Not part of the inspected shared-head path or a certified primary recipe | No promoted conditioning result | Deferred optional ablation after the simpler auxiliary-loss question |

Status words are literal: a path can be implemented without being smoke-tested,
experimentally evaluated, or promoted. The existence of the joint code path
does not establish that shared learning improves either primary task.

## Where the project is now

- Historical metal anchor: GVP + graph-level late fusion, namespaced candidate
  `metal/late-fusion/round4/trial49/fixed-split-5seed`.
- Evidence for that anchor: five seeds and 50 epochs on one validation split;
  mean `val_metal_balanced_acc = 0.635468206972`.
- Evidence grade: **3 — fixed validation split across seeds**, not grouped-fold
  confirmation.
- Reproduction status: **historical/non-rerunnable from the current repository**.
  The exact source bundle/checksum and checkpoint binaries are absent; do not
  present the anchor as an executable reproduction package.
- Stable comparison anchor: Only-ESM, LR `3e-5` with inverse-frequency
  weighting; five-seed mean `0.625325230595`.
- The tested node-level late-fusion trial-49-derived configuration did not
  replace graph-level trial 49.
- Hybrid and Hybrid+RING results are exploratory joint/single-split evidence,
  not promoted metal anchors.
- No current Stage 6 grouped-fold result, paired-CI promotion artifact, or
  completed Stage 6B final-refit artifact was found in the inspected evidence.

## Required controlled-comparison coverage

| Required question | Current evidence | What remains |
|---|---|---|
| Target formulation: direct four-class vs six-class training with collapsed-four evaluation | Historical six-class metrics and collapsed-four reporting exist; no trusted direct four-class baseline batch is indexed | No matched comparison across Only-GVP, Only-ESM, and graph-level late fusion on shared folds/seeds and the common four-class validation view |
| ESMC fusion position: early vs late vs hybrid | Late fusion has a Grade-3 historical six-class fixed-split anchor; hybrid has exploratory joint-task evidence | No indexed completed early-fusion result and no direct four-class matched shared-fold comparison across all three modes |
| Modality: Only-ESM (ESMC) vs Only-GVP vs combined GVP+ESMC | All three have historical six-class validation evidence, with late fusion representing the strongest established combined anchor | No direct four-class common Stage 6 fold/seed comparison with paired confidence intervals and rare-class recall protection |
| GVP edge source: with RING vs without RING | Radius and RING controls are implemented; Hybrid+RING has a high exploratory six-class single-seed result | No direct four-class matched RING on/off comparison that isolates the RING contribution; Hybrid+RING cannot answer this alone |

Do not claim a significant advantage for target formulation, ESMC fusion
position, combined modalities, or RING until the applicable matched comparison
is complete. Compare the target-formulation arms on the common four-class view;
keep the other architecture comparisons direct-four. This matrix is separate
from the first EC-primary auxiliary experiment.

Detailed parameters and confidence limits:
[`docs/PARAMETER_FINDINGS.md`](docs/PARAMETER_FINDINGS.md).

## Whole-project workstream map

| Workstream | What is implemented or prepared | Result/evidence state | What is not complete |
|---|---|---|---|
| Unified training core | `metal`, `ec`, and `joint` dispatch; configuration, preflight, grouped splitting, training, reporting, and guarded final-test code | CLI help imports and parses successfully | A passing end-to-end suite on all materialized datasets is not established |
| Graph and feature pipeline | Pocket graphs, conservative residue features, optional metal nodes, ESMC embeddings, external features, radius edges, and RING edges | Current v10 bundle contains ESM, external, and RING assets | Additional node feature sets beyond `conservative` remain future work |
| Metal modeling | Direct four-class, historical six-class/five-class targets; Only-GVP, Only-ESM, graph-level late fusion, node-level late fusion, hybrid, cross-attention, and RING/radius controls exist | Preserved anchors are historical six-class fixed-split evidence | Direct four-class baseline recipes/evidence must be reconciled; the advanced comparison matrix and Grade-1/2 promotion remain incomplete |
| EC modeling | Single-label EC-depth handling, independent EC heads, group weighting, and optional contrastive loss exist | No trusted completed EC model result is indexed in the audited evidence | EC depth-1 standalone baselines, EC playbook reconciliation, and a certified staged run are not complete; full multi-label prediction is not implemented |
| Auxiliary metal+EC modeling | Shared representation, independent heads, configurable task losses, and a joint task path exist | Hybrid and Hybrid+RING exploratory validation evidence exists | No matched EC-only versus EC+auxiliary-metal experiment, no promoted auxiliary configuration, and no certified cross-task protocol |
| Dataset preparation | Exact/non-overlap/Common70 PinMyMetal, CLEAN30 variants, and CARE clusterRes30 are prepared; provenance is tracked; local structures are content-addressed with manifest-backed split membership | Materialization, storage audit, and bundle status are in `docs/DATASETS.md` and `docs/STRUCTURE_STORE.md` | CLEAN10, the harsh root, CARE upstream citation, and final-test route remain unresolved |
| Metal confirmation | Stage 6 grouped folds × seeds, paired CI, and rare-class protection are implemented/documented | No completed current Stage 6 artifact found | Candidate set must be frozen and Stage 6 run |
| Final refit/reporting | Stage 6B and fail-closed Stage 7 workflow exist | Safety behavior is documented and smoke-covered before the current suite failure point | No completed Stage 6B refit; Stage 7 scientifically blocked |
| Colab execution | Unified notebook, main HF bundle, CLI/browser same-VM procedure, and G4/A100 compute probes exist | Stock Colab PyTorch worked on audited G4 and A100 runtimes; a separate PyTorch-free overlay is implemented | Unattended Drive mount remains open; legacy benchmark provenance requires authorized v2 regeneration/reruns |

For the shortest path through these owners, use
[`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md).

## Anchor and challengers

| Role | Configuration | Main validation evidence | Grade | Status |
|---|---|---|---:|---|
| Historical six-class metal anchor | Late-fusion trial 49 | mean `0.635468206972`, SD `0.043023727308`, five fixed-split seeds | 3 | Preserve as a six-class reference; it is not the direct four-class anchor |
| Historical six-class stable baseline | Only-ESM `3e-5` + inverse-frequency | mean `0.625325230595`, SD `0.031449451169`, five fixed-split seeds | 3 | Preserve within six-class evidence only |
| Rejected tested variant | Node-level late fusion derived from trial 49 | mean `0.606599196822`, five fixed-split seeds | 3 | Did not replace graph-level trial 49 |
| Exploratory challenger | Joint Hybrid trial 17 | three-seed joint mean `0.697376`; different selection metric | 6 | Incomplete provenance; not directly rankable |
| Exploratory challenger | Joint Hybrid+RING trial 114 | single-seed metal BA `0.7303469775006777` | 5 | Not confirmed; RING contribution inconclusive |

Trial 15—not trial 49—has the largest single late-fusion Round-4 result
(`0.6992755722978847`). Trial 49 remains the historical anchor because its
five-seed mean, variance, and worst-seed result led that batch. Neither fact is
grouped-fold promotion evidence.

## Model-family mission coverage

| Model/task path | Implemented | Audited outcome | Standing |
|---|---:|---|---|
| Metal Only-GVP | Yes | Historical six-class validation batches exist | Historical baseline evidence; no direct four-class grouped-fold promotion |
| Metal Only-ESM | Yes | Historical six-class five-seed mean `0.625325230595` | Historical Grade-3 baseline; direct four-class result absent |
| Metal GVP + graph-level late fusion | Yes | Historical six-class trial-49 five-seed mean `0.635468206972` | Historical Grade-3 anchor; direct four-class result absent |
| Metal GVP + node-level late fusion | Yes | Five-seed mean `0.606599196822` | Tested negative result; rejected as anchor replacement |
| Hybrid fusion | Yes | Joint trial-17 exploratory mean `0.697376` under a different selection metric | Incomplete provenance; not directly rankable |
| Hybrid + RING | Yes | Trial-114 single-seed metal BA `0.7303469775006777` | Exploratory; RING effect not isolated |
| Cross-attention | Yes | No indexed completed comparison found | Implemented but experimentally unestablished |
| RING/radius-only causal ablation | Yes | No reportable causal comparison found | Not completed |
| EC depth-1 standalone campaign | Partly | No trusted completed EC result indexed | Not certified end-to-end; playbook mismatch open |
| EC-primary auxiliary metal comparison | Joint topology exists | No matched controlled result indexed | Planned; recipe and cross-task safeguards not certified |
| Stage 6 grouped-fold confirmation | Yes | No completed current artifact found | Not done |
| Stage 6B full-train refit | Yes | No completed/reused artifact found | Not done |
| Stage 7 final reporting | Guarded implementation exists | No approved current final report | Blocked by Stage 6/6B and dataset-route decisions |

“No indexed result found” is an evidence statement, not proof that no run ever
occurred outside the repository and audited local artifacts.

## Evidence-grade reminder

1. Grouped folds × seeds with paired CI.
2. Grouped folds.
3. Fixed validation split across seeds.
4. HPO discovery on one validation split.
5. Single-seed validation.
6. Exploratory/smoke/partial/incomplete.
7. Superseded historical evidence.

Do not promote a Grade-4/5 Optuna result or a Grade-3 fixed-split anchor as if it
were Grade 1 or 2.

## Dataset readiness

| Dataset | Current readiness |
|---|---|
| Exact PinMyMetal | Present locally and in v10; contains 177 overlapping PDB IDs |
| Non-overlapped PinMyMetal | Present locally, absent from v10; historically evaluated seven times and not pristine |
| Harsh PinMyMetal | Absent locally and from v10 |
| Common-PDBID 70/30 | Present locally and in v10; custom comparison split |
| CLEAN30 original/conservative | Present; `CLEAN_30_main` points to conservative source |
| CLEAN10 | Not present or documented |
| CARE Task 1 clusterRes30 | Present locally and in current bundles |
| CARE legacy base | Scripts/docs remain; distinct legacy output root not found |

Bundle names, hashes, commits, split counts, preparation rules, and provenance:
[`docs/DATASETS.md`](docs/DATASETS.md).

## Colab compute-readiness evidence

The following results are throughput checks for one realistic
GVP+ESM-hybrid training step, not validation accuracy, model selection, or an
end-to-end training result. Both use a 240-pocket sample from CARE
clusterRes30 training data, batch size 12, 3 warm-up steps, 20 measured steps,
FP32, and no held-out test data.

| Evidence | Assigned GPU | PyTorch/CUDA | Median step | Throughput | OOM |
|---|---|---|---:|---:|---:|
| [`bench/g4_realistic.json`](bench/g4_realistic.json) | NVIDIA RTX PRO 6000 Blackwell Server Edition, compute 12.0 | `2.11.0+cu128` / CUDA 12.8, includes `sm_120` | `0.0120814975 s` | `993.254 samples/s` | No |
| [`bench/a100_realistic.json`](bench/a100_realistic.json) | NVIDIA A100-SXM4-40GB, compute 8.0 | `2.11.0+cu128` / CUDA 12.8, includes `sm_80` | `0.0473745975 s` | `253.300 samples/s` | No |

The corresponding historical v1 subset is inventoried in
[`docs/DATASETS.md`](docs/DATASETS.md). These are immutable legacy result-schema
files: they lack the current runner's required input-artifact, command, commit,
runner-hash, and full environment fields, and the v1 subset contains a pickled
project class. Re-run the architecture preflight in
[`docs/COLAB_GPU_RUNBOOK.md`](docs/COLAB_GPU_RUNBOOK.md) because stock Colab
versions can change.

## Test-use status

> The non-overlap PinMyMetal test was historically evaluated in seven early
> runs and is therefore not pristine or unopened. Whether those values
> influenced subsequent selection is not established by repository evidence.
> These test metrics must not be used for current HPO recommendations or model
> selection.

The seven exact reports are now tracked under
[`legacy_nonoverlap_test_access/`](docs/notebook_outputs/raw/legacy_nonoverlap_test_access/).
Later copied model-family anchor batches inspected during cleanup are
validation-only and contain no held-out reports.

> **Primary final-test route: unresolved scientific decision required before final reporting.**

This status does not designate a replacement test, substitute exact PinMyMetal,
or change evaluation behavior.

## Safety corrections implemented 2026-08-20

- Primary Stage 7 now hard-requires semantically valid, completed/reused Stage
  6B final-refit provenance; Stage 6 artifacts are not an executable fallback.
- Raw structure-group overlap is blocked before held-out preparation/inference,
  with a second loaded-pocket/group check before graph construction.
- The canonical all-ranked-candidate primary-test workflow is disabled.
- Stage 7 is additionally fail-closed while the primary dataset-route status
  remains scientifically unresolved.
- Strict RING planning now permits configured/available preparation to run and
  enforces required completeness after preparation.

These corrections do not resolve or select the primary final-test dataset.

## Current blockers

1. The primary final-test route requires a separate scientific decision.
2. The intended primary metal reporting endpoint is four-class, and both direct
   four-class training and a matched six-class-trained/collapsed-four challenger
   are required. The metal playbook common recipe remains six-class, the
   notebook live default is a separate five-class resume value, and no exact
   paired recipe or trusted matched comparison is indexed.
3. Current promotion policy calls for grouped-fold Stage 6 evidence, but no such
   completed evidence was found for the historical six-class
   anchor/challengers.
4. The required controlled metal-model comparison matrix is incomplete: early
   fusion lacks an indexed completed run, the modality comparison lacks shared
   Stage 6 evidence, and RING lacks a matched on/off causal ablation.
5. A reportable Stage 6B final-refit artifact was not found.
6. Hybrid Round-1 full configuration/search-space provenance is missing.
7. The EC playbook has documented incompatibilities with the current notebook
   and is not certified executable in affected sections.
8. No exact, certified recipe or matched evidence exists for the first
   EC-primary auxiliary comparison, and a future multi-source implementation
   must enforce cross-task group exclusion before training.

Execution-readiness issues that do not change the scientific next action:

- the smoke path repair and pytest/CPU-CI conversion are implemented locally;
  CI confirmation depends on running the new workflow (`TECH-007`);
- the notebook's live `MOUNT_DRIVE = True` can block unattended CLI notebook
  execution (`TECH-008`);
- a Python 3.12 Linux CPU lock and PyTorch-free Colab overlay now exist; Colab
  hardware validation remains runtime-specific (`TECH-009`).

Open implementation issues:
[`docs/FOLLOW_UP_TECHNICAL_ISSUES.md`](docs/FOLLOW_UP_TECHNICAL_ISSUES.md).

## Immediate next action

In a separate implementation/recipe task, reconcile the metal playbook and
notebook launch surface with both required target-formulation arms, then define
the matched Phase 1 campaign for Only-GVP, Only-ESM, and GVP + graph-level late
fusion. Each family needs a direct-four arm and a separately named six-class
arm with collapsed-four evaluation. Preserve all historical six-class and
five-class records under their original schemes. Do not use historical test
metrics or open held-out data during that reconciliation.

Do not launch the existing six-class playbook blocks alone and call the required
paired comparison complete. Exact matched direct-four and six-to-collapsed-four
stage blocks must be reviewed in the metal playbook first. If a later campaign
reaches candidate confirmation, the next promotion stage remains grouped-fold
**Stage 6**, then Stage 6B, never direct held-out evaluation.

## Next few actions

1. **Phase 1:** reconcile and establish both metal target formulations for
   Only-GVP, Only-ESM, and GVP + graph-level late fusion: direct four-class
   training and matched six-class training with collapsed-four evaluation.
2. **Phase 2:** reconcile the EC playbook and establish EC depth-1 standalone
   baselines for the same three initial families.
3. **Phase 3:** run the predeclared metal x EC1 descriptive association analysis
   on permitted training/development data only.
4. **Phase 4:** add and run one controlled EC-primary comparison: EC-only versus
   EC plus auxiliary metal, using independent heads with shared learning and
   cross-task group exclusion.
5. **Phase 5:** test metal-only versus metal plus auxiliary EC only if useful or
   scientifically worthwhile.
6. **Phase 6:** only then consider soft metal conditioning or more complicated
   cross-task interactions. Continue the separate direct four-class metal
   architecture matrix when its standalone gates justify it.

Any candidate intended for final reporting still requires shared-fold/seed
validation, the applicable promotion gates, a frozen full non-test refit, and a
scientifically resolved final-test route before one-shot held-out evaluation.

Exact metal stage blocks remain in
[`docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`](docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md).
Do not infer exact budgets from this status file.

## Evidence shortcuts

- [Getting started and navigation](docs/GETTING_STARTED.md)
- [Colab browser/CLI GPU runbook](docs/COLAB_GPU_RUNBOOK.md)
- [Experiment index](docs/notebook_outputs/README.md)
- [Parameter/HPO findings](docs/PARAMETER_FINDINGS.md)
- [Dataset and test-use authority](docs/DATASETS.md)
- [Late-fusion Round-4 summary](docs/notebook_outputs/summaries/summary_run_gvp_late_fusion_round4_top3_seedrepeat_50epoch.md)
- [Recovered late-fusion Round-4 JSON evidence](<docs/notebook_outputs/raw/GVP + late fusion/metal_late_fusion_optuna_top3_seedrepeat_50epoch_v1/>)
- [Node-level negative-result summary](docs/notebook_outputs/summaries/summary_run_gvp_node_level_late_fusion_round1_from_latefusion_trial49_seedrepeat_50epoch.md)
- [Hybrid+RING exploratory summary](docs/notebook_outputs/summaries/summary_run_hybrid_ring_round2_optuna_50epoch_wide_v1_trials105_176.md)

## Update rule

After a meaningful batch, update only:

- current objective/stage;
- anchor/challenger state and evidence grade;
- dataset/test readiness;
- blockers;
- immediate and next few actions;
- links to newly indexed evidence.

Put exact batch history in the experiment index, empirical parameter knowledge
in `PARAMETER_FINDINGS.md`, and stable policy in `Plan.md`.
