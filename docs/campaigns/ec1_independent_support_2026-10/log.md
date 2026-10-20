# EC1 independent-support preparation: log
Public-release note (2026-10-20): host paths, device names and cloud resource names below are generalized; facts, hashes and timings are unchanged.

Dated entries, newest last. Campaign [README](README.md).

## ec1s-001

2026-10-13 — **campaign opened as CPU evidence/data preparation; component-support and EC7 attrition report run
once** (user handoff 2026-10-13: stronger EC evidence as the next scientific priority). No training, inference,
download, GPU use, split writing, eligibility change, promotion or held-out access.

Tooling (new, additive): `scripts/report_ec1_component_support.py` (SHA-256 `ca708a04…`),
`tests/test_ec1_component_support.py` (`cb6d291c…`). Reused read-only: `src/audit_sequence_remoteness.py`
(`9137ba06…`), `CARE_prepare_training_and_test_set/care_prepare.py` (`493330bd…`). Commands and inputs:
[instructions](instructions.md).

Run: output `<secondary-drive>/DeepMzyme_Data/campaigns/ec1_independent_support_2026-10/report_20261013T191929Z/`, `artifact_manifest.json`
SHA-256 `6674d052dcf6b31ca5c3dd364d4a0c36281ca9980a2b19154d5cb6adde6cbf69`; `support_report.json` `ee82536e…`;
`next_scientific_decision.md` `47d9e065…`. 39 input files were hashed before and after: unchanged (key inputs:
baseline split `428aa0be…`, memberships `853cd192…`, EC alignments `a4aec7a6…`, train build summary `ea3df04a…`,
fetch summary `04e918f9…`, annotation cache `55e0bd3e…`). Every handoff value reproduced; no discrepancy.

Development support (saved EC1 membership; 4 saved EC run memberships, 1 unique, equal to the baseline):
- 793 proteins, 1,168 pockets, 319 dependence components; largest components 53, 25, 16, 15 and 12 proteins.
- EC1–EC7 proteins 175, 111, 382, 64, 26, 29, 6; components 83, 51, 131, 32, 22, 14, 1 (overlapping across classes).
- EC7: all six (D9IA43, P00402, P0ABI8, P98005, Q37370, Q5SJ79) in one component with one EC1 member (Q59647);
  11 qualifying unordered EC7–EC7 pairs keep them connected without it; no recorded source/coordinate sequence
  discrepancy.
- Interpretation: adopted as indivisible split units, these components would prevent seven-class train/validation
  coverage. They were uncertainty units, not a mandatory split rule; accession-grouped validation remains possible
  with disclosed related-sequence limits. The remoteness minimum-support requirement is not imported.

EC7 attrition in CARE clusterRes30 training data:
- 179 unambiguous EC7 proteins: 6 in the baseline, 173 outside.
- Recorded build outcomes: 141 `NO_UNIPROT_SUPPORTED_TRANSITION_METAL`, 24 `MISSING_ALPHAFILL_JSON_OR_CIF`, 14 `OK`.
- The 14 prepared structures supplied 17 candidate sites; saved final (predictive, MAHOMES) assignments: 8 catalytic,
  9 noncatalytic. The 8 prepared proteins outside the baseline carry only noncatalytic-predicted sites; the baseline
  cohort table is the only-catalytic table. The candidate table's `whether_catalytic=0` is a placeholder.
- The 24 missing-asset accessions: 24 sequences, 22 AlphaFill JSON, 0 CIF, 0 complete pairs; 22 deliberate CIF skips
  (`SKIPPED_NO_JSON_SUPPORTED_CANDIDATE`), 2 JSON HTTP 404 (P08748, Q00227).
- Recorded policy (identity >= 0.30, alignment length >= 85, optional RMSD/PAE/atom-count filters disabled,
  `require_supported`): 0 annotation-compatible candidates from the 22 JSONs (10 elemental candidates, all rejected).
- Rejection categories: 7 elemental candidates rejected by annotation, 12 compound IDs outside the exact-element
  mapping, 3 no qualifying donor/transplant, 2 recorded missing JSON, 0 unexpected changes. 19 accessions have
  qualifying donors with metal-containing compounds.

Tests: `tests/test_ec1_component_support.py` 13 passed; with `tests/test_sequence_remoteness.py` and
`tests/test_care_cache_audit.py`, 35 passed.

Proposal: [proposal](proposal.md). Recommended: Route 1 (preserve eligibility) with a CPU-only inventory of
already-local permitted training-side sources; Route 2 (cofactor-inclusive cohort) only as a separately versioned
exploratory cohort after a scientific decision. Next (the user's decision): which sources are permitted for that
inventory, and whether Route 2 is scoped at all.

## ec1s-002

2026-10-13/14 — **EC7/EC6/EC5 training-source inventory under the current eligibility** (the user's bounded CPU task,
2026-10-13: Route 1 inventory). No GPU/VM start, training, inference, download, new held-out access, alignment search,
cohort merging, eligibility change or cofactor expansion. The VM stayed TERMINATED; the metal reference is unchanged.

Tooling (new, additive): `scripts/inventory_ec_support_sources.py` (SHA-256 `f394ee42…`) and
`tests/test_inventory_ec_support_sources.py` (`8c8e0d43…`). They reuse, read-only, `scripts/report_ec1_component_support.py`
(cohort attrition and asset checks), `care_prepare.py` (candidate extraction, the annotation rule) and the training
label rule `src/training/labels.py`. Inputs, command and boundaries: [instructions](instructions.md#training-source-inventory-ec1s-002).

Run: `<secondary-drive>/DeepMzyme_Data/campaigns/ec1_independent_support_2026-10/inventory_20261013T211519Z/`, `artifact_manifest.json`
`3513f315…` (binds 22 inputs, the 3,369 AlphaFill JSON files read, 72 reused clusterRes30 assets, file listings, git HEAD
`ad99c5f` and the tool hashes); `candidate_inventory.csv` `95cae1a0…`, `recommendation.md` `c5f76fc9…`. The 34 input files
hashed before the task are unchanged. All 27 values computed independently during the source mapping match.

Results (units: CARE accessions or PinMyMetal chains; clusters: CARE clusterRes30; catalytic labels are MAHOMES
predictions):
- **No EC5, EC6 or EC7 protein outside the development membership has documented eligibility** in any source.
- **CARE clusterRes30 cohort** (representatives): EC7 179 (6 in development), EC6 388 (29) plus 14 ambiguous, EC5 459
  (26) plus 39 ambiguous. Build outcomes outside development: EC7 141 no supported metal / 24 missing assets / 8
  noncatalytic-only; EC6 313 / 37 / 9; EC5 410 / 15 / 8. Unknown: the four recorded HTTP-404 JSONs (EC7 P08748, Q00227;
  EC6 P0DX91, P32277). Every built protein with a predicted catalytic site (EC7 6/14, EC6 29/38, EC5 26/34) is a
  development member. The prepared cohort's 24 absent proteins all have EC numbers spanning several first digits (no
  target); three contain EC5 (P55792, Q12362, Q5WL39).
- **CARE full-train non-representatives** (unclustered work root `task1_30`, same recorded policy, checked):
  - EC7: 6,311 → 1,156 supported metal → 392 local JSON → 97 with an annotation-compatible candidate and a local CIF
    (5 clusters; 58 units in 3 clusters without a known development relation: Q00227 47, O30085 9, A4VXE1 2); 764
    supported-metal accessions have no local AlphaFill files (629 without a known relation, 26 clusters).
  - EC6: 21,485 (106 ambiguous) → 4,824 → 2,437 → 1,922 (34 clusters; 625 units in 18 clusters without a known
    relation); 2,387 supported, no local files (1,162 without a known relation, 45 clusters).
  - EC5: 11,854 (503 ambiguous) → 1,193 → 539 → 532 (28 clusters; 111 units in 10 clusters without a known relation);
    654 supported, no local files (214 without a known relation, 21 clusters).
  - None is built or has a MAHOMES prediction (the unclustered run has only a smoke build; no focal accession in it).
- **Non-overlap PinMyMetal training chains** are outside the current definition (PDB `_entity.pdbx_ec` labels,
  experimental structures). EC7: 4 chains, all in or beside the development membership (P98005; P00396 and P07657 in
  development clusters; three carry the obsolete PDB EC 1.9.3.1 that CARE labels 7.1.1.9). EC6: 18 chains, P00968
  (annotation unknown outside CARE training data), Q8RM03 (incompatible annotation), 9 unmapped. EC5: 92 chains, C0SSE7,
  Q818Z9, Q8X5Q7 (annotation unknown), 24 unmapped.
- **Relationships:** exact sequences and clusterRes30 co-membership only. Separate clusters do not prove remoteness:
  development EC7 is 6 proteins in 6 clusters but 1 dependence component (EC6 29 in 29 but 14; EC5 26 in 26 but 22).
  No candidate is an exact alias of a development protein.
- **Held-out exclusion:** not certified for any candidate. 128 clusterRes30 representatives are absent from the training
  entries; 507 focal non-representatives sit in their clusters (7 of them EC6 supported-metal, none partially prepared).
- **Later builds are not download-free:** the partially prepared candidates use 185 donor PDB entries, 80 cached only in
  the clusterRes30 root and 105 not cached (RCSB metadata), and the build has no accession allowlist.

Recommendation ([proposal](proposal.md#route-1-inventory-result-ec1s-002)): relationship search, then a label-blind
held-out check, then a training-only build, then targeted AlphaFill downloads; each needs its own authorization.

Verification: a four-reader read-only source mapping, then four adversarial verifiers. They recomputed every funnel and
cohort/PinMyMetal number independently (all matched) and found no blocker. Corrections made before the recorded run:
- "related" now means positive evidence only (absent-representative clusters are a separate unresolved category);
- missing or FAILED builds are "unknown";
- the held-out path guard covers resolved symlinks and more held-out names;
- the boundary record states that three mixed-split tables are hashed whole and their rows parsed and discarded;
- the build step is described as needing RCSB metadata;
- the later steps are reordered and the download counts scoped to units without a known relation.

Tests: `tests/test_inventory_ec_support_sources.py` 24 passed (duplicates, ambiguous labels, training-only rows, missing
evidence, no held-out path opened or listed, symlink guard, determinism of all outputs under input reordering, unchanged
inputs, real cohort mapping on the reporter's synthetic world).

Incidental test-metadata exposure during the source mapping (recorded in the
[DATASETS ledger](../../DATASETS.md#local-test-access-artifacts-git-ignored)): one agent printed the header and first
(test-split) row of the unclustered annotation cache and two log lines with combined train-plus-test totals. Nothing was
retained or used; the tool never prints rows.

STATUS text replaced by this entry, preserved verbatim:

```text
# DeepMzyme Current Experiment Status

- Status: active (2026-10-13 EC1 independent-support CPU evidence package complete; route decision pending the user)
- Last execution evidence: 2026-10-13 (EC1 component-support report on the workstation CPU, [log ec1s-001](docs/campaigns/ec1_independent_support_2026-10/log.md#ec1s-001); trunk-LR retrospective verification, [log tlr-005](docs/archive/campaigns/pmm_trunk_lr_2026-10/log.md#tlr-005)).

## Current objective and stage

- Current campaign: [ec1_independent_support_2026-10](docs/campaigns/ec1_independent_support_2026-10/README.md): CPU evidence and data preparation toward stronger EC1 evidence; no training, inference, download, promotion or GPU use.
- Other open campaigns: [metal_single_gpu](docs/campaigns/metal_single_gpu/README.md) (paused); [zenodo_pmm_exact_2026-09-24](docs/campaigns/zenodo_pmm_exact_2026-09-24/README.md) (paused; test possibly opened)
- Stage: evidence boundary = `<secondary-drive>/DeepMzyme_Data/campaigns/ec1_independent_support_2026-10/report_20261013T191929Z/` (all expected values reproduced). Development support 793 proteins, 1,168 pockets, 319 dependence components; the six EC7 proteins form one component. Under current eligibility the CARE clusterRes30 training source adds no EC7 example (141 lack a supported metal, 24 lack assets by deliberate filtering, 8 carry only noncatalytic-predicted sites). Recommended: Route 1, preserve eligibility ([proposal](docs/campaigns/ec1_independent_support_2026-10/proposal.md)).

The trunk-LR comparison closed on 2026-10-13: reference retained (+1.16 points, rule (b) not met); the lower trunk rate stays an unpromoted, undisproven candidate ([archived README](docs/archive/campaigns/pmm_trunk_lr_2026-10/README.md)). PMM core v2 closed at fold 0 on 2026-10-03 ([archived README](docs/archive/campaigns/pmm_ion_metal/README.md)).

## Anchor and evidence state

- Best validation result: none promoted. The closed v3 campaign's validation line (Grade 2; five grouped folds, seed 42) is in [findings](docs/PARAMETER_FINDINGS.md#pmm-ion-metal-v3-objective-test-and-stage-6-selection-closed-2026-10-13): late fusion four-class, mean common-four BA 74.28%, better than Only-ESMC on all five folds; no supported gain from five/six-class training. Unchanged v3 result: refit `a0007846…`, one-shot test common-four BA 67.55% [62.93, 71.65], `both_results_secondary` ([summary](docs/notebook_outputs/summaries/summary_pmm_ion_metal_v3_20261013.md)).
- Deferred and never run (untested, not disproven): the cost-gated augmentations `posnoise01` and `outerdrop01`. The Only-ESMC head-dropout fairness arm no longer repairs a mismatch in the selected recipe ([findings](docs/PARAMETER_FINDINGS.md#pmm-ion-metal-v3-objective-test-and-stage-6-selection-closed-2026-10-13)).
- EC: the [EC1 reference](docs/archive/campaigns/ec1_standalone_v12_2026-09-14/README.md) has twelve fixed-split runs (Grade 3), not promotion; cross-task holdout certification precedes auxiliary learning ([issues](docs/FOLLOW_UP_TECHNICAL_ISSUES.md)).

### Known caveats and open mismatches

- Cross-scheme ranking needs a campaign assessor on collapsed-four BA; the notebook Stage 6/6B route is single-scheme ([TECH-010](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-010--four-class-endpoint-and-paired-metal-target-recipes-are-not-reconciled)).
- v3 replaces the size-strata folds ([TECH-020](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-020--greedy-k-fold-assignment-concentrates-large-groups-in-fold-0)) and uses one terminal checkpoint rule in CV and refit ([TECH-027](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-027--cross-validation-and-final-refit-use-different-checkpoint-rules)); PMM comparator citations keep the label-leak caveat ([TECH-028](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-028--pmm-comparator-inherits-a-true-metal-label-leak)).

## Dataset and test readiness

The primary final-test route is unresolved; [DATASETS](docs/DATASETS.md#test-use-ledger) owns the ledger. Safety boundary:

- Non-overlap and exact PMM tests (same 316-structure set): opened 13 and 21 times; not eligible as an unopened test. Harsh and Common-PDBID 70/30 test IDs come from that opened set.
- Exact Zenodo PMM test: evaluated once by v3 step F on 2026-10-13 (confirmed); the 2026-09-24 possible evaluation stays unresolved; metadata read and byte scan disclosed in the ledger.
- No evaluation artifacts found for CLEAN30 or CARE clusterRes30; CARE had incidental test-metadata exposure on 2026-09-15. The EC report read only CARE training-side rows.

## Blockers and immediate next action

- Authorized now: nothing to run; no GPU start, training, inference, download, new held-out evaluation or eligibility change.
- GPU/VM: `<gpu-vm>` TERMINATED (verified 2026-10-13 18:50 UTC); disk kept; snapshots `<paused-snapshot>` and `<v3-final-snapshot>` kept. Local backups and the off-site gap: the private backup record.

Next: the user's route decision ([proposal](docs/campaigns/ec1_independent_support_2026-10/proposal.md#recommendation-and-remaining-decision): which training-side sources the Route 1 inventory may use; whether Route 2 is scoped) and the commit/push of the staged records. Later: cross-task exclusion certification. No test-based selection ([Plan](Plan.md#canonical-staged-metal-training-pipeline)).

## History and update rule

Overwrite this file after preserving dated changes in the campaign's `log.md`; keep the objective/stage and campaign lines, anchor, best result and grade, caveats, dataset/test readiness, authorization and GPU/VM state, blockers, next action and evidence links. Budgets stay in the playbooks, findings in [PARAMETER_FINDINGS](docs/PARAMETER_FINDINGS.md), policy in [Plan](Plan.md), batch evidence in the [index](docs/notebook_outputs/README.md); the [history map](docs/archive/consolidation_2026-09/MAP.md) holds the recovered 09-16/17 pause and removed policy.
```

## ec1s-003

2026-10-14 — deferred by the user’s metal-first choice. The inventory remains complete and its pending relationship/exclusion work remains unexecuted. A synthetic audit found a discovered-child symlink gap in the cache readers; [TECH-030](../../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-030--discovered-ec-cache-files-can-bypass-the-training-path-guard) must be addressed before reuse. This is not evidence that historical input files exploited the gap.
