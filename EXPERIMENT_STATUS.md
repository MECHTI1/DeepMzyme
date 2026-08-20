# DeepMzyme Current Experiment Status

This is the sole concise answer to: **Where am I now, and what should I do
next?** It is mutable. Scientific policy is in [`Plan.md`](Plan.md); exact
experiment history is in the [experiment index](docs/notebook_outputs/README.md).

Last evidence audit: 2026-08-20.

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
