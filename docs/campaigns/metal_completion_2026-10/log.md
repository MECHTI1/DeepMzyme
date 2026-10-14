# Metal completion campaign log

## 2026-10-14 — approved CPU implementation

The user selected metal first, all three training objectives, a bounded worthwhile optimization pass, and an independent final evaluation. “Implement the approved plan” authorized the CPU preparation described in the campaign README. Subsequent questions about Claude clarified ownership: Codex completes this implementation; Claude receives a current independent-review handoff. No new execution or test access was authorized.

The original review covered the named Claude transcript and more recent Codex evidence. The old Claude handoff predates the completed v3/trunk-LR closeout and EC inventory. Completed evidence remains intact; no blanket rerun is indicated.

## Superseded STATUS snapshot

The following is the exact previous root STATUS, preserved before replacement. Its next-action and authority lines are historical; relative links retain their original repository-root context.

```markdown

# DeepMzyme Current Experiment Status

- Status: active (2026-10-13 EC1 Route 1 training-source inventory complete; next-step decision pending the user)
- Last execution evidence: 2026-10-13 (EC7/EC6/EC5 training-source inventory on the workstation CPU, [log ec1s-002](docs/campaigns/ec1_independent_support_2026-10/log.md#ec1s-002)).

## Current objective and stage

- Current campaign: [ec1_independent_support_2026-10](docs/campaigns/ec1_independent_support_2026-10/README.md): CPU evidence and data preparation toward stronger EC1 evidence; no training, inference, download, promotion or GPU use.
- Other open campaigns: [metal_single_gpu](docs/campaigns/metal_single_gpu/README.md) (paused); [zenodo_pmm_exact_2026-09-24](docs/campaigns/zenodo_pmm_exact_2026-09-24/README.md) (paused; test possibly opened)
- Stage: evidence boundary = Data1 `campaigns/ec1_independent_support_2026-10/inventory_20261013T211519Z/`. No EC5/EC6/EC7 protein outside the development membership (793 proteins; EC7 6 proteins, one component) has documented eligibility in the permitted training sources. Closest: CARE non-representatives with local AlphaFill assets but no build or catalytic prediction (EC7 97 in 5 clusters, 58 of them in 3 clusters without a known development relation); homology and held-out exclusion unresolved ([proposal](docs/campaigns/ec1_independent_support_2026-10/proposal.md#route-1-inventory-result-ec1s-002)).

The trunk-LR comparison closed on 2026-10-13: reference retained; the lower trunk rate stays an unpromoted, undisproven candidate ([archived README](docs/archive/campaigns/pmm_trunk_lr_2026-10/README.md)). PMM core v2 closed at fold 0 on 2026-10-03 ([archived README](docs/archive/campaigns/pmm_ion_metal/README.md)).

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
- Exact Zenodo PMM test: evaluated once by v3 step F on 2026-10-13 (confirmed); the 2026-09-24 possible evaluation stays unresolved.
- No evaluation artifacts found for CLEAN30 or CARE clusterRes30; CARE had incidental test-metadata exposure on 2026-09-15 and 2026-10-13 (one cache row, no use).

## Blockers and immediate next action

- Authorized now: nothing to run; no GPU start, training, inference, download, alignment search, new held-out access or eligibility change.
- GPU/VM: `deepmzyme-l4` TERMINATED (verified 2026-10-13); disk and snapshots `deepmzyme-paused-20261003`, `deepmzyme-v3-final-20261013` kept. Backups: [inventory 5A](docs/BACKUP_AND_DATA_INVENTORY.md#5a-local-independent-backups-and-gcp-state-verified-2026-10-13).

Next: the user's decision on step 1 of the [proposal](docs/campaigns/ec1_independent_support_2026-10/proposal.md#route-1-inventory-result-ec1s-002) (a training-side MMseqs2 relationship search) and the commit/push of the staged records. Later: held-out check, builds, downloads, cohort decision. No test-based selection ([Plan](Plan.md#canonical-staged-metal-training-pipeline)).

## History and update rule

Overwrite this file after preserving dated changes in the campaign's `log.md`; keep the objective/stage and campaign lines, anchor, best result and grade, caveats, dataset/test readiness, authorization and GPU/VM state, blockers, next action and evidence links. Budgets stay in the playbooks, findings in [PARAMETER_FINDINGS](docs/PARAMETER_FINDINGS.md), policy in [Plan](Plan.md), batch evidence in the [index](docs/notebook_outputs/README.md); the [history map](docs/archive/consolidation_2026-09/MAP.md) holds the recovered 09-16/17 pause and removed policy.
```

## 2026-10-14 — CPU preparation completed

Implemented the additive successor runner, immutable phase manifests, actual-input verification, evidence collector,
discovery/confirmation assessor, execution authorization/session/budget guards and offline evaluation preparer.
The new host accepted all 16 historical controls after independent byte/metric reconciliation. Original v3 source,
data, evidence and refit remain unchanged. No new accuracy result or promotion was produced.

Actual preparation completed against 6,443 frozen training structures and all 7,664 ESM payloads plus sidecars.
Discovery manifest SHA-256 `e5905d8ef3855c0dd0814b09c8777520293b63d9e7ead8b817f65907cfc9a5f2`;
45 command previews built; the assessor correctly refused selection with only the 15 discovery controls.
The remaining 30 discovery fits and 29/39 confirmation fits are unexecuted. The
[evidence page](evidence.md) owns local receipt locations, measured timings and the costed proposal.

Final focused tests: 132 passed. CPU smoke: 42 passed, one missing-local-fixture skip (the runner prints
“PASSED 43 smoke checks”). Docs contract: zero strict failures, 11 pre-existing warnings. Synthetic tests additionally
caught and closed early-provider-stop admission, actual training-file binding, held-out child-symlink and unchanged-retry
gaps before packaging. Library versions and code/test/log hashes are recorded in `evidence/cpu_verification.json`.

The original 26 staged entries were preserved; task changes are a staged review draft. No commit/push, cloud
lifecycle command, model training/inference, download or new held-out evaluation occurred. The existing proposal
requires scoped execution/budget approval; future deployment/persistence checks and independent-cohort provenance
remain explicit gates. The [Claude handoff](handoff.md) now requests independent review and concrete fixes instead
of duplicating completed implementation.
