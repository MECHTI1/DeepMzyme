# DeepMzyme Current Experiment Status

- Status: active (2026-10-07 v3 step D: Round A done, Round R 17 of 26; `meanagg`, `wd10` passed)
- Last execution evidence: 2026-10-07 (v3 step D sessions 1–3, 8 h 06 min, $7.13 gross). Documentation reconciliation: 2026-10-07.

## Current objective and stage

- Current campaign: pmm_ion_metal_v3, step D running (objectives four/five/six for Only-ESMC, Only-GVP and late fusion) — [README](docs/campaigns/pmm_ion_metal_v3/README.md)
- Other open campaigns: [metal_single_gpu](docs/campaigns/metal_single_gpu/README.md) (paused); [zenodo_pmm_exact_2026-09-24](docs/campaigns/zenodo_pmm_exact_2026-09-24/README.md) (paused; test possibly opened)
- Stage: v3 steps A–C done; step D Round A complete, Round R 17 of 26 ([log v3-020](docs/campaigns/pmm_ion_metal_v3/log.md#v3-020)); Round B follows; no Stage 6 confirmation, Stage 6B refit or Stage 7.

**The PMM core v2 campaign was closed at fold 0 on 2026-10-03 (user decision).**
Its nine fold-0 fits are final validation-only evidence; 36 fits were not run and
its neutral four-versus-five/six test is unanswered. No fivefold neural
confirmation, model/target promotion, final refit or held-out evaluation is
claimed ([archived README](docs/archive/campaigns/pmm_ion_metal/README.md)).

## Anchor and evidence state

- Best validation result: none promoted; the closed PMM core fold-0 fits are Grade 5 exploratory evidence (incomplete grid Grade 6) per the [archived PMM README](docs/archive/campaigns/pmm_ion_metal/README.md); results in the [core summary](docs/notebook_outputs/summaries/summary_pmm_core_continuation_20260928.md).

Fold 0 of that campaign holds only multi-ion PDB groups and six Cu groups, so its
values are not representative ([findings](docs/PARAMETER_FINDINGS.md#pmm-ion-campaign-single-fold-target-and-readout-comparisons)).
The [EC1 reference](docs/archive/campaigns/ec1_standalone_v12_2026-09-14/README.md)
retains twelve completed fixed-split runs (Grade 3), not promotion. Later EC
workflow reconciliation and cross-task holdout certification remain required
before auxiliary learning; [issues](docs/FOLLOW_UP_TECHNICAL_ISSUES.md) own details.

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
owns the ledger; this short access reminder preserves the default-read safety boundary:

- Non-overlap and exact PMM tests (same 316-structure set): opened 13 and 21
  times; not eligible as an unopened test. Harsh and Common-PDBID 70/30 test IDs
  come from that opened set.
- Exact Zenodo PMM test: unknown, possibly evaluated; its source test file's
  label/feature metadata was read in aggregate on 2026-10-03 (no evaluation).
- No evaluation artifacts found for CLEAN30 or CARE clusterRes30; CARE had
  incidental test-metadata exposure on 2026-09-15. Absence of found artifacts is
  not proof of no outside run.

## Blockers and immediate next action

- Authorized now: advance GPU authorization for the remaining approved v3 plan through step F ([log v3-014](docs/campaigns/pmm_ion_metal_v3/log.md#v3-014)), amended step D included; no new start request needed. Within the [$40 gross ceiling](docs/campaigns/pmm_ion_metal_v3/log.md#v3-003) including storage (about $26.3 left; forecast $34.6–37.7). Step F only after its gates and the final-test label decision; no Round C.
- GPU/VM: `deepmzyme-l4` TERMINATED (verified 2026-10-07 03:37 UTC); disk and snapshot kept; failed-fallback leftovers (two disks, one snapshot, about $1.06/day) await the user's cleanup decision ([v3-019](docs/campaigns/pmm_ion_metal_v3/log.md#v3-019)).

Next: the 9 remaining Round R fits, then Round B (14) and the combinations, per the [playbook](docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md#v3-step-d-regularization-amendment); the user's work window ended 2026-10-07 04:10 UTC. Reforecast before Round B; cost-gated augmentations need the user's OK.
Still open before step E: the final metal test label, after the user's check of
the 2026-09-24 Zenodo run. Stage 6 grouped-fold selection (or an explicitly
labeled fallback), a completed Stage 6B full non-test refit, frozen
report/checkpoint rules and a scientifically resolved final-test route must
precede one-shot Stage 7. No test-based tuning, ranking, promotion, rejection or
checkpoint choice; see [Plan](Plan.md#canonical-staged-metal-training-pipeline).

## History and update rule

Overwrite this file after preserving dated changes in the campaign's `log.md`.
Keep only objective/stage with the current and other open campaign lines,
anchor, best validation result and evidence grade, known caveats and open
mismatches, dataset/test readiness, authorization and GPU/VM state, blockers,
next action and evidence links. Exact budgets stay in the playbooks; parameter
findings stay in [PARAMETER_FINDINGS](docs/PARAMETER_FINDINGS.md), scientific
policy in [Plan](Plan.md), and batch evidence in the [index](docs/notebook_outputs/README.md).
[History map](docs/archive/consolidation_2026-09/MAP.md) includes the recovered
09-16/17 pause and earlier removed policy; historical next actions do not resume work.
