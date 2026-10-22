# DeepMzyme Current Experiment Status

- Status: active (2026-10-23 EC-class-vs-CLEAN30 campaign opened; metal deployment default still not chosen)
- Last execution evidence: 2026-10-20 CPU-only [matched PinMyMetal comparison](docs/campaigns/phase2_unified_2026-10/pmm_matched/README.md): pre-registered; opened test scored.

## Current objective and stage

- Current campaign: [ec_class_vs_clean30_2026-10](docs/campaigns/ec_class_vs_clean30_2026-10/README.md) (Phase 1, CPU infrastructure).
- Objective: an EC-class (first EC digit) model that outperforms a retrained CLEAN on the official split30 metallo benchmark and on its test proteins without a ≥30% identity hit to training.
- Stage: Phase 1 (CPU code and tests; the private CLEAN and ESM-1b setup is approved); no GPU use or test read authorized yet.
- Other open campaigns: [phase2_unified_2026-10](docs/campaigns/phase2_unified_2026-10/README.md) (metal deployment default not yet chosen: the 55-model best-CV reference against the 20-, 25- and 30-model late-fusion ensembles); [EC support](docs/campaigns/ec1_independent_support_2026-10/README.md) deferred; [metal_single_gpu](docs/campaigns/metal_single_gpu/README.md) and [exact Zenodo](docs/campaigns/zenodo_pmm_exact_2026-09-24/README.md) paused.

## Anchor and evidence state

- Best validation result: 55-model ensemble, `77.17%` pooled OOF, with no CI-supported gain over the 20-, 25- or 30-model late-fusion ensembles (`76.34–76.81%`), so [Plan §7](Plan.md#7-baseline-first-model-comparison-policy) does not justify its extra members; nested selection gives `76.56%` (late fusion) and `76.54%` (55-model blocks). Pre-declared single model: `B 4-class`, `74.34%` OOF; refit `67.55%` Full test ([v3 summary](docs/notebook_outputs/summaries/summary_pmm_ion_metal_v3_20261013.md)). Test scores never select.

### Known caveats and open mismatches

- The 20-, 25- and 30-model recipes follow the full-test order (possibly test-informed; OOF-only nested selection reaches a similar level). Larger ensembles average more checkpoints per OOF site, which favours them; OOF-selected scores are optimistic (nested: −0.25 / −0.63 pp). Test disclosures: [DATASETS](docs/DATASETS.md#test-use-ledger), [TECH-028](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-028--pmm-comparator-inherits-a-true-metal-label-leak).
- EC remains fixed-split Grade 3 evidence; TECH-030/031/032 stay open ([issues](docs/FOLLOW_UP_TECHNICAL_ISSUES.md)).

## Dataset and test readiness

[DATASETS](docs/DATASETS.md#test-use-ledger) records both test evaluations (Step F, 2026-10-13; 180 checkpoints, 2026-10-19). Choose and freeze the default on OOF; an independent cohort can then evaluate it once (temporal PDB [protocol](docs/campaigns/metal_completion_2026-10/evaluation.md), optional).

## Blockers and immediate next action

- Authorized now: Phase 1 CPU work of ec_class_vs_clean30_2026-10 (code and tests only).
- GPU/VM: none in use; the Phase 3 smoke (at most $3) needs its own approval (G2).
- Next: Phase 1, then Phase 2 inputs and Freeze 1; the metal deployment default remains the maintainer's open choice.

## History and update rule

The [phase2_unified log](docs/campaigns/phase2_unified_2026-10/log.md) (p2u-006) preserves the previous STATUS. Policy: [Plan](Plan.md), [DATASETS](docs/DATASETS.md), [PARAMETER_FINDINGS](docs/PARAMETER_FINDINGS.md).
