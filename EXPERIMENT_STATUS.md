# DeepMzyme Current Experiment Status

- Status: active (2026-10-08 v3 step E in progress: 10 of 44 confirmation fits done, four-class baselines on folds 1–4; final D recipes late fusion `headdrop03`, Only-GVP `meanagg`)
- Last execution evidence: 2026-10-08 (v3 step E session 6, 2 h 15 min, $1.98 gross). Docs reconciliation: 2026-10-08.

## Current objective and stage

- Current campaign: pmm_ion_metal_v3, step D running (objectives four/five/six for Only-ESMC, Only-GVP and late fusion) — [README](docs/campaigns/pmm_ion_metal_v3/README.md)
- Other open campaigns: [metal_single_gpu](docs/campaigns/metal_single_gpu/README.md) (paused); [zenodo_pmm_exact_2026-09-24](docs/campaigns/zenodo_pmm_exact_2026-09-24/README.md) (paused; test possibly opened)
- Stage: v3 steps A–D done; step E 10 of 44 fits ([log v3-023](docs/campaigns/pmm_ion_metal_v3/log.md#v3-023)); no Stage 6 confirmation, Stage 6B refit or Stage 7.

**The PMM core v2 campaign was closed at fold 0 on 2026-10-03 (user decision).**
Its nine fold-0 fits are final validation-only evidence; 36 fits were not run and
its neutral four-versus-five/six test is unanswered. No fivefold confirmation,
model/target promotion, final refit or held-out evaluation is claimed ([archived README](docs/archive/campaigns/pmm_ion_metal/README.md)).

## Anchor and evidence state

- Best validation result: none promoted; the closed PMM core fold-0 fits are Grade 5 exploratory evidence (incomplete grid Grade 6) per the [archived PMM README](docs/archive/campaigns/pmm_ion_metal/README.md); results in the [core summary](docs/notebook_outputs/summaries/summary_pmm_core_continuation_20260928.md).

Fold 0 of that campaign holds only multi-ion PDB groups and six Cu groups, so it is
not representative ([findings](docs/PARAMETER_FINDINGS.md#pmm-ion-campaign-single-fold-target-and-readout-comparisons)).
The [EC1 reference](docs/archive/campaigns/ec1_standalone_v12_2026-09-14/README.md)
retains twelve completed fixed-split runs (Grade 3), not promotion. EC workflow
reconciliation and cross-task holdout certification precede auxiliary learning
([issues](docs/FOLLOW_UP_TECHNICAL_ISSUES.md)).

### Known caveats and open mismatches

- Cross-scheme ranking: the notebook Stage 6/6B route is single-scheme; comparing
  target schemes needs a campaign assessor on collapsed-four balanced accuracy
  ([TECH-010](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-010--four-class-endpoint-and-paired-metal-target-recipes-are-not-reconciled)).
- v3 replaces the size-strata folds
  ([TECH-020](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-020--greedy-k-fold-assignment-concentrates-large-groups-in-fold-0))
  and uses one terminal checkpoint rule in CV and refit
  ([TECH-027](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-027--cross-validation-and-final-refit-use-different-checkpoint-rules));
  PMM comparator citations keep the label-leak caveat
  ([TECH-028](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-028--pmm-comparator-inherits-a-true-metal-label-leak)).

## Dataset and test readiness

The primary final-test route remains unresolved. [DATASETS](docs/DATASETS.md#test-use-ledger)
owns the ledger; this reminder preserves the default-read safety boundary:

- Non-overlap and exact PMM tests (same 316-structure set): opened 13 and 21
  times; not eligible as an unopened test. Harsh and Common-PDBID 70/30 test IDs
  come from that opened set.
- Exact Zenodo PMM test: unknown, possibly evaluated; its source test file's
  label/feature metadata was read in aggregate on 2026-10-03 (no evaluation).
- No evaluation artifacts found for CLEAN30 or CARE clusterRes30; CARE had
  incidental test-metadata exposure on 2026-09-15. Absence of found artifacts is
  not proof of no outside run.

## Blockers and immediate next action

- Authorized now: the user's 2026-10-07/08 working window has ended; the advance GPU authorization through step F ([log v3-014](docs/campaigns/pmm_ion_metal_v3/log.md#v3-014)) stands within the [$40 gross ceiling](docs/campaigns/pmm_ion_metal_v3/log.md#v3-003) including storage (about $16.8 left; forecast $33.3–37.1), but confirm the next GPU start with the user. Step F only after its plan and gates; no Round C.
- GPU/VM: `deepmzyme-l4` TERMINATED (verified 2026-10-08 02:23 UTC); disk and snapshot kept; failed-fallback leftovers (two disks, one snapshot, about $1.06/day) await the user's cleanup decision ([v3-019](docs/campaigns/pmm_ion_metal_v3/log.md#v3-019)).

Next: a new chat continues from the [handoff](docs/campaigns/pmm_ion_metal_v3/handoff.md): the remaining step E fits ($16.8_E: four-class baselines first, then the 8 improvement fits, then five/six-class, cache sets cold once), then the step E assessment; open user decisions: Only-ESMC fairness arm, fallback leftovers, cost-gated augmentations, step F plan.
Still open before step E: the final metal test label, after the user's check of
the 2026-09-24 Zenodo run. Stage 6 grouped-fold selection (or a labeled
fallback), a completed Stage 6B full non-test refit, frozen report/checkpoint
rules and a resolved final-test route must precede one-shot Stage 7. No test-based tuning, ranking, promotion, rejection or
checkpoint choice; see [Plan](Plan.md#canonical-staged-metal-training-pipeline).

## History and update rule

Overwrite this file after preserving dated changes in the campaign's `log.md`.
Keep the objective/stage and campaign lines, anchor, best validation
result and grade, caveats and mismatches, dataset/test readiness,
authorization and GPU/VM state, blockers, next action and evidence links. Exact budgets stay in the playbooks; parameter
findings stay in [PARAMETER_FINDINGS](docs/PARAMETER_FINDINGS.md), scientific
policy in [Plan](Plan.md), and batch evidence in the [index](docs/notebook_outputs/README.md).
[History map](docs/archive/consolidation_2026-09/MAP.md) includes the recovered
09-16/17 pause and earlier removed policy; historical next actions do not resume work.
