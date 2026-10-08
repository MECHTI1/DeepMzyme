# DeepMzyme Current Experiment Status

- Status: active (2026-10-08 v3 step E in progress: 24 of 44 confirmation fits done; four-class baselines and improvement fits done on folds 1–4; five/six-class started)
- Last execution evidence: 2026-10-08 (v3 step E session 7, 3 h 29 min, $3.07 gross).

## Current objective and stage

- Current campaign: pmm_ion_metal_v3, step E running (objectives four/five/six for Only-ESMC, Only-GVP and late fusion) — [README](docs/campaigns/pmm_ion_metal_v3/README.md)
- Other open campaigns: [metal_single_gpu](docs/campaigns/metal_single_gpu/README.md) (paused); [zenodo_pmm_exact_2026-09-24](docs/campaigns/zenodo_pmm_exact_2026-09-24/README.md) (paused; test possibly opened)
- Stage: v3 steps A–D done; step E 24 of 44 fits ([log v3-024](docs/campaigns/pmm_ion_metal_v3/log.md#v3-024)); no Stage 6 confirmation, Stage 6B refit or Stage 7.

**The PMM core v2 campaign was closed at fold 0 on 2026-10-03 (user decision).**
Its nine fold-0 fits are final validation-only evidence; 36 fits were not run and
its neutral four-versus-five/six test is unanswered. No fivefold confirmation,
model/target promotion, final refit or held-out evaluation is claimed ([archived README](docs/archive/campaigns/pmm_ion_metal/README.md)).

## Anchor and evidence state

- Best validation result: none promoted; the closed PMM core fold-0 fits are Grade 5 exploratory evidence (incomplete grid Grade 6) per the [archived PMM README](docs/archive/campaigns/pmm_ion_metal/README.md); results in the [core summary](docs/notebook_outputs/summaries/summary_pmm_core_continuation_20260928.md).
- v3 step E (descriptive): the two selected recipes show no clear gain on folds 1–4; the late-fusion baseline is above Only-ESMC on every fold so far ([log v3-024](docs/campaigns/pmm_ion_metal_v3/log.md#v3-024)).

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

- Authorized now: today's single-session arrangement has ended; the advance GPU authorization through step F ([log v3-014](docs/campaigns/pmm_ion_metal_v3/log.md#v3-014)) stands within the [$40 gross ceiling](docs/campaigns/pmm_ion_metal_v3/log.md#v3-003) including storage (about $12.9 left; forecast $33.5–36.4), but confirm the next GPU start with the user. Step F only after its plan and gates; no Round C.
- GPU/VM: `deepmzyme-l4` TERMINATED (verified 2026-10-08 12:16:58 UTC); disk and snapshot kept; failed-fallback leftovers (two disks, one snapshot, about $1.06/day) await the user's cleanup decision ([v3-019](docs/campaigns/pmm_ion_metal_v3/log.md#v3-019)).

Next: a new chat continues from the [handoff](docs/campaigns/pmm_ion_metal_v3/handoff.md): the remaining 20 five/six-class fits (about 4.3 h), then the E assessment and the deferred 10-epoch-schedule review; user decisions open: step F plan (draft in the campaign audits folder), fairness arm, fallback leftovers, cost-gated augmentations, log v3-024 corrections.
Final-test label `both_results_secondary` is recorded ([log v3-022](docs/campaigns/pmm_ion_metal_v3/log.md#v3-022)). Stage 6 selection, a completed Stage 6B refit and frozen report/checkpoint rules precede one-shot Stage 7; no test-based selection ([Plan](Plan.md#canonical-staged-metal-training-pipeline)).

## History and update rule

Overwrite this file after preserving dated changes in the campaign's `log.md`.
Keep the objective/stage and campaign lines, anchor, best validation
result and grade, caveats and mismatches, dataset/test readiness,
authorization and GPU/VM state, blockers, next action and evidence links. Exact budgets stay in the playbooks; parameter
findings stay in [PARAMETER_FINDINGS](docs/PARAMETER_FINDINGS.md), scientific
policy in [Plan](Plan.md), and batch evidence in the [index](docs/notebook_outputs/README.md).
[History map](docs/archive/consolidation_2026-09/MAP.md) includes the recovered
09-16/17 pause and earlier removed policy; historical next actions do not resume work.
