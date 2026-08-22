# DeepMzyme Current Experiment Status

This is the sole concise answer to: **Where am I now, and what should I do
next?** It is mutable. Scientific policy is in [`Plan.md`](Plan.md); exact
experiment history is in the [experiment index](docs/notebook_outputs/README.md).

Last experiment-evidence audit: 2026-08-20. Last execution/documentation audit:
2026-08-22.

## Current objective

Select and confirm a reliable metal-classification configuration using
validation evidence, while keeping exploratory joint metal+EC candidates
separate. Existing historical anchors use the six-class metal target and a
fixed `pdbid`-grouped validation split.

No copied candidate has yet passed the current grouped-fold Stage 6 promotion
standard.

## Where the project is now

- Historical metal anchor: GVP + graph-level late fusion, namespaced candidate
  `metal/late-fusion/round4/trial49/fixed-split-5seed`.
- Evidence for that anchor: five seeds and 50 epochs on one validation split;
  mean `val_metal_balanced_acc = 0.635468206972`.
- Evidence grade: **3 — fixed validation split across seeds**, not grouped-fold
  confirmation.
- Stable comparison anchor: Only-ESM, LR `3e-5` with inverse-frequency
  weighting; five-seed mean `0.625325230595`.
- The tested node-level late-fusion trial-49-derived configuration did not
  replace graph-level trial 49.
- Hybrid and Hybrid+RING results are exploratory joint/single-split evidence,
  not promoted metal anchors.
- No current Stage 6 grouped-fold result, paired-CI promotion artifact, or
  completed Stage 6B final-refit artifact was found in the inspected evidence.

Detailed parameters and confidence limits:
[`docs/PARAMETER_FINDINGS.md`](docs/PARAMETER_FINDINGS.md).

## Whole-project workstream map

| Workstream | What is implemented or prepared | Result/evidence state | What is not complete |
|---|---|---|---|
| Unified training core | `metal`, `ec`, and `joint` dispatch; configuration, preflight, grouped splitting, training, reporting, and guarded final-test code | CLI help imports and parses successfully | A passing end-to-end suite on all materialized datasets is not established |
| Graph and feature pipeline | Pocket graphs, conservative residue features, optional metal nodes, ESMC embeddings, external features, radius edges, and RING edges | Current v10 bundle contains ESM, external, and RING assets | Additional node feature sets beyond `conservative` remain future work |
| Metal modeling | Only-GVP, Only-ESM, graph-level late fusion, node-level late fusion, hybrid, cross-attention, and RING/radius controls exist | Fixed-split anchors and exploratory/negative results are indexed below | No Grade-1/2 grouped-fold promotion result |
| EC modeling | EC heads, EC-depth handling, group weighting, and optional contrastive loss exist | No trusted completed EC model result is indexed in the audited evidence | EC playbook reconciliation and a certified staged EC run are not complete |
| Joint modeling | Joint metal+EC task and configurable loss weighting exist | Hybrid and Hybrid+RING exploratory validation evidence exists | No promoted joint configuration or reportable confirmation |
| Dataset preparation | Exact/Common70 PinMyMetal, CLEAN30 variants, and CARE clusterRes30 are prepared; provenance is tracked | Materialization and bundle status are in `docs/DATASETS.md` | CLEAN10, current non-overlap/harsh roots, CARE upstream citation, and final-test route remain unresolved |
| Metal confirmation | Stage 6 grouped folds × seeds, paired CI, and rare-class protection are implemented/documented | No completed current Stage 6 artifact found | Candidate set must be frozen and Stage 6 run |
| Final refit/reporting | Stage 6B and fail-closed Stage 7 workflow exist | Safety behavior is documented and smoke-covered before the current suite failure point | No completed Stage 6B refit; Stage 7 scientifically blocked |
| Colab execution | Unified notebook, main HF bundle, CLI/browser same-VM procedure, and G4/A100 compute probes exist | Stock Colab PyTorch worked on audited G4 and A100 runtimes | Fresh local environment is not locked; unattended Drive mount and smoke-path issues remain open |

For the shortest path through these owners, use
[`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md).

## Anchor and challengers

| Role | Configuration | Main validation evidence | Grade | Status |
|---|---|---|---:|---|
| Historical metal anchor | Late-fusion trial 49 | mean `0.635468206972`, SD `0.043023727308`, five fixed-split seeds | 3 | Keep as anchor until stronger comparable evidence exists |
| Stable baseline | Only-ESM `3e-5` + inverse-frequency | mean `0.625325230595`, SD `0.031449451169`, five fixed-split seeds | 3 | Retain |
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
| Metal Only-GVP | Yes | Historical validation batches exist | Baseline evidence; not a current grouped-fold promotion |
| Metal Only-ESM | Yes | Five-seed mean `0.625325230595` | Stable Grade-3 baseline |
| Metal GVP + graph-level late fusion | Yes | Trial-49 five-seed mean `0.635468206972` | Historical Grade-3 anchor |
| Metal GVP + node-level late fusion | Yes | Five-seed mean `0.606599196822` | Tested negative result; rejected as anchor replacement |
| Hybrid fusion | Yes | Joint trial-17 exploratory mean `0.697376` under a different selection metric | Incomplete provenance; not directly rankable |
| Hybrid + RING | Yes | Trial-114 single-seed metal BA `0.7303469775006777` | Exploratory; RING effect not isolated |
| Cross-attention | Yes | No indexed completed comparison found | Implemented but experimentally unestablished |
| RING/radius-only causal ablation | Yes | No reportable causal comparison found | Not completed |
| EC staged campaign | Partly | No trusted completed EC result indexed | Not certified end-to-end; playbook mismatch open |
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
| Non-overlapped PinMyMetal | Absent locally and from v10; historically evaluated seven times |
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

The corresponding portable subset is inventoried in
[`docs/DATASETS.md`](docs/DATASETS.md). Re-run the architecture preflight in
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
2. Current promotion policy calls for grouped-fold Stage 6 evidence, but no such
   completed evidence was found for the historical anchor/challengers.
3. A reportable Stage 6B final-refit artifact was not found.
4. Hybrid Round-1 full configuration/search-space provenance is missing.
5. The EC playbook has documented incompatibilities with the current notebook
   and is not certified executable in affected sections.

Execution-readiness issues that do not change the scientific next action:

- the smoke suite reaches 37 passes and then stops on a stale removed-document
  path (`TECH-007`);
- the notebook's live `MOUNT_DRIVE = True` can block unattended CLI notebook
  execution (`TECH-008`);
- `src/requirements.txt` is not a complete environment lock and its PyTorch pin
  must be filtered in Colab (`TECH-009`).

Open implementation issues:
[`docs/FOLLOW_UP_TECHNICAL_ISSUES.md`](docs/FOLLOW_UP_TECHNICAL_ISSUES.md).

## Immediate next action

In a separate scientific-planning task, freeze the set of configurations that
should enter reportable comparison and resolve which dataset route may support
final reporting. Do not use historical test metrics in that decision.

If architecture selection is declared complete, the next experiment stage is
the current grouped-fold **Stage 6** comparison—not direct Stage 6B and not
held-out evaluation.

## Next few actions

1. Run the playbook-defined grouped-fold Stage 6 on the frozen candidate set,
   using shared folds/seeds, validation metrics, paired comparisons, and
   rare-class-recall protection.
2. If Stage 6 selects a configuration under policy, run the separate Stage 6B
   full non-test training/refit step without held-out evaluation.
3. Resolve the primary final-test route scientifically before any final report.
4. Only after those gates, consider Stage 7 under the then-approved policy and
   implementation.

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
