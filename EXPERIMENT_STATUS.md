# DeepMzyme Current Experiment Status

Status: paused (2026-09-28 user decision)
Last execution evidence: 2026-09-28. Documentation reconciliation: 2026-09-28.

## Current objective and stage

Retain the exploratory PMM core v2 fold-0 comparison. **Nine ordinary-readout
fits are retained; the remaining 36 fits are deferred.** No completed fivefold
neural confirmation, model/target promotion, final refit or held-out evaluation
is claimed. See the [campaign README](docs/campaigns/pmm_ion_metal/README.md) and
[v2 scope](docs/plans/pmm_core_scope_v2.json); exact scientific identities,
sub-batches, results and evidence grades live there.

## Anchor and evidence state

Seven original strict replay passes and the GVP5/GVP6 strict failures remain
recorded. All nine core fits have separate retrospective `pmm-core-replay-v1`
qualification; this does not rewrite the original `1e-6` failures as passes.
The [core summary](docs/notebook_outputs/summaries/summary_pmm_core_continuation_20260928.md)
owns the result table and diagnostic limits. The three earlier binding-aware
fits remain separate exploratory evidence; further awareness work is paused.

The [EC1 reference](docs/archive/campaigns/ec1_standalone_v12_2026-09-14/README.md)
retains twelve completed fixed-split runs (Grade 3), not promotion. Later EC
workflow reconciliation and cross-task holdout certification remain required
before auxiliary learning; [issues](docs/FOLLOW_UP_TECHNICAL_ISSUES.md) own details.

## Dataset and test readiness

The primary final-test route remains unresolved. [DATASETS](docs/DATASETS.md#test-use-ledger)
owns the ledger; this short access reminder preserves the default-read safety boundary:

- Non-overlap PMM test: 352 pockets; seven early reports plus six on 2026-09-18
  (`benchmark_50epochs` ×3 and `benchmark_replicated_72pct` ×3).
- Exact PMM test: the same structure set, 352 pockets / 316 structure files /
  313 PDB IDs. Opened 2026-09-22 (three single-split reports) and 2026-09-23
  (15 fold and three ensemble reports). Exploratory selection influence: yes;
  no model promoted. See the [methodological qualification](docs/EXACT_PINMYMETAL_5FOLD_CV_REPRODUCIBILITY.md).
- Exact Zenodo PMM test: unknown, possibly evaluated; the relaunch outcome is
  unrecorded. See its [history](docs/campaigns/zenodo_pmm_exact_2026-09-24/README.md).
- No evaluation artifacts found for Harsh, Common-PDBID 70/30, CLEAN30 or CARE
  clusterRes30. CARE had incidental test-metadata exposure on 2026-09-15,
  not evaluation. Absence of found artifacts is not proof of no outside run.

## Blockers and immediate next action

**No experiment or GPU work is authorized now.** Resume requires an explicit
user request, refreshed remaining-budget authority and new readiness verification
against the execution environment. The proposed additional 34 hours/$34 is
unapproved and no longer awaiting an immediate execution decision. No extra
fold-0 fits or final refits are authorized ([scope authorization](docs/plans/pmm_core_scope_v2.json)).

Last verified VM state: **TERMINATED**, independently checked 2026-09-28
06:00:18 UTC. The retained 150-GB disk still costs approximately $15/month;
this is historical provider evidence, not a new live resource check
([closeout](docs/notebook_outputs/summaries/summary_pmm_core_continuation_20260928.md)).

Stage 6 grouped-fold selection (or explicitly labeled fallback), completed/reused
Stage 6B full non-test refit, frozen report/checkpoint rules and a scientifically
resolved final-test route must precede one-shot Stage 7. No test-based tuning,
ranking, promotion, rejection or checkpoint choice; see [Plan](Plan.md#canonical-staged-metal-training-pipeline).

## History and update rule

Overwrite this file after preserving dated changes in the campaign's `log.md`.
Keep only objective/stage, anchor/evidence grade, dataset/test readiness,
blockers, next action and evidence links. Exact budgets stay in the playbooks;
parameter findings stay in [PARAMETER_FINDINGS](docs/PARAMETER_FINDINGS.md),
scientific policy in [Plan](Plan.md), and batch evidence in the [index](docs/notebook_outputs/README.md).
[History map](docs/archive/consolidation_2026-09/MAP.md) includes the recovered
09-16/17 pause and earlier removed policy; historical next actions do not resume work.
