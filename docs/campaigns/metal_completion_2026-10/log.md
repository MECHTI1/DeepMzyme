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

## 2026-10-14 — independent review, focused repairs and re-sealed preparation

Claude reviewed the completed CPU preparation as the handoff requested: six read-only reviewer agents on distinct
questions (recipes/loss mathematics, control reuse, assessor rules, worker/authorization/persistence, offline evaluation
plus documentation, provenance/path binding), every finding adversarially checked by two further agents, one
coordinator applying changes. Thirty-five findings, no blocker. Confirmed majors: the executing `pmm_v3_assessment.py`
was bound by no manifest hash; the worker's durable copy and the required `<unit>.rc` exit file had no documented
path that the collector accepts; the identical-absolute-path deployment prerequisite was unstated. The remaining
findings were minor or informational; the packet-path trust concern was refuted (paths are hash-bound). The discovery
tie-break (all-target mean includes non-qualifying targets) is the frozen rule and is now disclosed, not changed.

Smallest safe code changes, all CPU-tested: `pmm_v3_assessment.py` joins the manifest-bound modules and must equal the
assessor hash frozen by the parent extension; `collect` requires absolute run-map paths, binds the reused control's
directory to the lane the frozen verifier recomputes, and checks the trainer's recorded seed, loss keys and manual
weight mode; confirmation preparation requires the discovery-era specification hash and records the discovery
manifest hash; snapshots captured before the allocation start are refused; symlinked lane `runs/`/`commands/` are
refused before any write; assessment receipts name the ranking metric and unsupported claims; the evaluation
preparer refuses future upper dates, records `prepared_on`, and requires accounting rows for the EC-side dataset
families. Focused tests: 141 passed (nine new cases). CPU smoke: 42 passed, one absent-fixture skip. TECH-031 records
the trainer-level focal/smoothing observation. Documentation: playbook block rewritten (recipe-table wording, full
smoothing disclosure, `-B`, deployment prerequisite, wrapper/exit-file contract, persistence layout, absolute run
maps); DATASETS gains the `pinmymetal-zenodo-exact` row and the step F fact; evaluation.md names all ten families
and the date rule.

Because any change to a bound module invalidates the frozen manifest, the preparation package was re-sealed at
Data1 `campaigns/metal_completion_2026-10/preparation_20261014T164111Z/` (clean code copy, byte-identical parent,
the 2026-10-14 `preparation_20261014/` kept as superseded history) and discovery preparation, control
re-verification and command previews were rerun; [evidence.md](evidence.md) owns the new hashes. The provider was
checked read-only through the installed controller (`vm-status --no-reconcile`, `vm-report --no-ssh`,
2026-10-14T15:51Z): `deepmzyme-l4` TERMINATED; gross running rate $0.87917/h; retained disk plus two snapshots about
$1.00/day, higher than the disk-only figure the proposal used, so the envelope arithmetic in evidence.md is corrected.
No GPU start, training, inference, download, new held-out read, commit or push occurred.

### Superseded STATUS snapshot (before this review)

```markdown
# DeepMzyme Current Experiment Status

- Status: active (2026-10-14 metal completion CPU preparation complete; execution not authorized)
- Last execution evidence: 2026-10-13 EC1 inventory; no new model fits or test evaluations.

## Current objective and stage

- Current campaign: [metal_completion_2026-10](docs/campaigns/metal_completion_2026-10/README.md)
- Objective: finish a bounded metal optimization comparison across four/five/six objectives, then independently evaluate the frozen final model; EC work is deferred by the user's metal-first choice.
- Other open campaigns: [EC independent support](docs/campaigns/ec1_independent_support_2026-10/README.md) (deferred); [metal_single_gpu](docs/campaigns/metal_single_gpu/README.md) (paused); [zenodo_pmm_exact_2026-09-24](docs/campaigns/zenodo_pmm_exact_2026-09-24/README.md) (paused).
- Stage: successor CPU preparation verified; discovery manifest frozen. Next is validation-only discovery followed by Stage 6 grouped-fold/seed confirmation; neither has run. [Campaign evidence](docs/campaigns/metal_completion_2026-10/evidence.md) records checks and control reuse.

## Anchor and evidence state

- Best validation result: retained v3 late-fusion direct-four, terminal 50-epoch cosine training. Grade 2 validation evidence: five grouped folds, seed 42, common-four BA 74.28%; no supported benefit from five/six training. [Findings](docs/PARAMETER_FINDINGS.md#pmm-ion-metal-v3-objective-test-and-stage-6-selection-closed-2026-10-13).
- Lower trunk LR remains unpromoted; the completed comparison retained the reference. [Closed comparison](docs/archive/campaigns/pmm_trunk_lr_2026-10/README.md).
- Historical refit `a0007846…` and secondary test reports remain unchanged. [Preserved summary](docs/notebook_outputs/summaries/summary_pmm_ion_metal_v3_20261013.md). New code and synthetic checks are not evidence of improved accuracy.
- EC remains at fixed-split Grade 3 evidence; the completed support inventory and eligibility restrictions are preserved in its deferred campaign.

### Known caveats and open mismatches

- Additional seeds reuse development folds. Cross-target ranking requires the campaign assessor ([TECH-010](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-010--four-class-endpoint-and-paired-metal-target-recipes-are-not-reconciled)); no new model has been evaluated.
- EC child-file guard repair remains open before reuse ([TECH-030](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-030--discovered-ec-cache-files-can-bypass-the-training-path-guard)).

## Dataset and test readiness

The primary independent final-test route remains unresolved. [DATASETS](docs/DATASETS.md#test-use-ledger) owns historical exposure; opened PMM tests are not pristine. The new [offline evaluation protocol](docs/campaigns/metal_completion_2026-10/evaluation.md) does not authorize candidate retrieval or establish adequate support. Source dates, complete exclusions, near-copy checks and final report/refit freezes precede scoring. Additional optimizer seeds on reused development folds are not independent data.

## Blockers and immediate next action

- Authorized now: CPU implementation, synthetic checks, allowlisted historical development-evidence verification and preparation of a concrete costed proposal. No GPU start, real-data training/inference, download, new held-out access, commit or push.
- GPU/VM: last verified `deepmzyme-l4` TERMINATED on 2026-10-13; no fresh provider check in this task. Retained disks/snapshots continue to cost money. [Backup inventory](docs/BACKUP_AND_DATA_INVENTORY.md#5a-local-independent-backups-and-gcp-state-verified-2026-10-13).
- Next: review the staged draft and costed proposal; obtain scoped execution/budget authority before launch. Independent evaluation has separate provenance and data-access prerequisites. The [updated Claude handoff](docs/campaigns/metal_completion_2026-10/handoff.md) requests independent review of completed work.

## History and update rule

The exact previous STATUS is preserved in the [campaign log](docs/campaigns/metal_completion_2026-10/log.md). Overwrite this page only after preserving dated history. Scientific policy remains in Plan, recipes in the metal playbook, dataset/test access in DATASETS, and validation conclusions in PARAMETER_FINDINGS.
```
