# Metal completion campaign log
Public-release note (2026-10-20): host paths, device names and cloud resource names below are generalized; facts, hashes and timings are unchanged.

## 2026-10-14 — approved CPU implementation

The user selected metal first, all three training objectives, a bounded worthwhile optimization pass, and an independent final evaluation. “Implement the approved plan” authorized the CPU preparation described in the campaign README. Subsequent questions about the independent review agent clarified ownership: Codex completes this implementation; the review agent receives a current independent-review handoff. No new execution or test access was authorized.

The original review covered the named agent transcript and more recent Codex evidence. The old agent handoff predates the completed v3/trunk-LR closeout and EC inventory. Completed evidence remains intact; no blanket rerun is indicated.

## Superseded STATUS snapshot

The following is the exact previous root STATUS, preserved before replacement. Its next-action and authority lines are historical; relative links retain their original repository-root context.

```markdown

# DeepMzyme Current Experiment Status

- Status: active (2026-10-13 EC1 Route 1 training-source inventory complete; next-step decision pending the user)
- Last execution evidence: 2026-10-13 (EC7/EC6/EC5 training-source inventory on the workstation CPU, [log ec1s-002](docs/campaigns/ec1_independent_support_2026-10/log.md#ec1s-002)).

## Current objective and stage

- Current campaign: [ec1_independent_support_2026-10](docs/campaigns/ec1_independent_support_2026-10/README.md): CPU evidence and data preparation toward stronger EC1 evidence; no training, inference, download, promotion or GPU use.
- Other open campaigns: [metal_single_gpu](docs/campaigns/metal_single_gpu/README.md) (paused); [zenodo_pmm_exact_2026-09-24](docs/campaigns/zenodo_pmm_exact_2026-09-24/README.md) (paused; test possibly opened)
- Stage: evidence boundary = `<secondary-drive>/DeepMzyme_Data/campaigns/ec1_independent_support_2026-10/inventory_20261013T211519Z/`. No EC5/EC6/EC7 protein outside the development membership (793 proteins; EC7 6 proteins, one component) has documented eligibility in the permitted training sources. Closest: CARE non-representatives with local AlphaFill assets but no build or catalytic prediction (EC7 97 in 5 clusters, 58 of them in 3 clusters without a known development relation); homology and held-out exclusion unresolved ([proposal](docs/campaigns/ec1_independent_support_2026-10/proposal.md#route-1-inventory-result-ec1s-002)).

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
- GPU/VM: `<gpu-vm>` TERMINATED (verified 2026-10-13); disk and snapshots `<paused-snapshot>`, `<v3-final-snapshot>` kept. Backups: the private backup record.

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
remain explicit gates. The [agent handoff](handoff.md) now requests independent review and concrete fixes instead
of duplicating completed implementation.

## 2026-10-14 — independent review, focused repairs and re-sealed preparation

A review agent reviewed the completed CPU preparation as the handoff requested: six read-only reviewer agents on distinct
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
`<secondary-drive>/DeepMzyme_Data/campaigns/metal_completion_2026-10/preparation_20261014T164111Z/` (clean code copy, byte-identical parent,
the 2026-10-14 `preparation_20261014/` kept as superseded history) and discovery preparation, control
re-verification and command previews were rerun; [evidence.md](evidence.md) owns the new hashes. The provider was
checked read-only through the installed controller (`vm-status --no-reconcile`, `vm-report --no-ssh`,
2026-10-14T15:51Z): `<gpu-vm>` TERMINATED; gross running rate $0.87917/h; retained disk plus two snapshots about
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
- GPU/VM: last verified `<gpu-vm>` TERMINATED on 2026-10-13; no fresh provider check in this task. Retained disks/snapshots continue to cost money. Backups: the private backup record.
- Next: review the staged draft and costed proposal; obtain scoped execution/budget authority before launch. Independent evaluation has separate provenance and data-access prerequisites. The [updated agent handoff](docs/campaigns/metal_completion_2026-10/handoff.md) requests independent review of completed work.

## History and update rule

The exact previous STATUS is preserved in the [campaign log](docs/campaigns/metal_completion_2026-10/log.md). Overwrite this page only after preserving dated history. Scientific policy remains in Plan, recipes in the metal playbook, dataset/test access in DATASETS, and validation conclusions in PARAMETER_FINDINGS.
```

## 2026-10-15 — execution authorized; session 1 (discovery) in progress

The user authorized completing the frozen validation-only discovery and confirmation campaign on the existing GCP
L4 (ceilings: 24 aggregate GPU hours, $32 gross including storage, at most eight sessions of at most four hours;
no migration, larger hardware, extra experiments, refit or held-out access). The decision text is recorded
verbatim in the package (`execution/authorization/user_decision_2026-10-15.txt`) and bound by the discovery
policy (SHA-256 `319890bc…`) to manifest `d405974e…` and the 30 permitted units. The six interrupted Q6
verifier agents left no verdicts; the coordinator completed those three narrow checks (recorded in
`evidence/independent_review_findings.json`). Price re-verified 2026-10-15T12:24Z ($0.879172/h gross).

New coordinator tool `metal_completion_session.py` (tested, 13 CPU tests) binds approvals, refreshes
controller/provider snapshots, deploys the package at the identical absolute paths (bind mounts recorded in
`/etc/fstab`), launches units through the exit-file wrapper, pulls/acknowledges/places lane copies, recovers
interrupted lanes and builds the run map. Deployment and the canonical `plan` passed on the host
(`evidence/vm_plan_…P__four_class__fold0__seed42.json`).

Session 1 started 2026-10-15T12:35:39Z (`session-20261015T123539Z-5930fdc9`, hard stop 16:32:31Z). Incident:
the host had booted kernel 7.0.0-1014-gcp (unattended upgrade of 2026-10-13) for which the NVIDIA driver does not
build; CUDA was unavailable. Three launched units were stopped before any GPU work and archived (one unchanged rerun
each remains); the kernel packages were held and the host rebooted into 7.0.0-1013-gcp with the driver loaded
(`evidence/host_incident_20261015.json`). The loop now refuses launches without CUDA. Discovery relaunched at
13:05Z on three lanes.

Session 1 closed 2026-10-15T16:10Z: 3 h 34 min running, $3.14 gross (controller ledger), VM TERMINATED. Eight
discovery units completed, independently replayed, pulled, acknowledged and placed (P four-class folds 0–4, P
five-class folds 0–2); a ninth launch was refused by the worker's admission clock without creating artifacts. The
coordinator loop had died on that refusal and was repaired (refusals now close the launch window; 300 s launch
margin); `evidence/session_1_accounting.json`.
Session 2 start at 16:11Z was refused by the provider: L4 capacity stocked out in `us-central1-a`
(`ZONE_RESOURCE_POOL_EXHAUSTED`). The same-region `vm-fallback` replacement route is a zone migration with extra
storage charges, which the authorization excludes, so the existing VM is retried every ten minutes.
Session 2 (`session-20261015T162420Z-7b08fa9c`, started 16:24Z after the stockout cleared) closed 19:30Z: 3 h 06 min,
$2.73 gross, VM TERMINATED; twelve units completed and verified (P five-class folds 3–4, P six-class folds 0–4, S
four-class folds 0–4), 20 of 30 discovery units done; `evidence/session_2_accounting.json`.
Session 3 (`session-20261015T194331Z-51cb995f`, 3.5-hour limit, started 19:43Z after a second stockout) closed
22:15Z: 2 h 33 min, $2.24 gross, VM TERMINATED; the last ten units (S five-class and six-class folds 0–4) completed
and verified. **Discovery grid complete: 30 new fits plus 15 reused controls**, after 9 h 13 min running and
$8.11 gross running cost over three sessions; `evidence/session_3_accounting.json`.

## 2026-10-15 — discovery decision (frozen rule)

Collection `execution/collection_discovery.json` (SHA-256 `c9465c9b…`, 45 verified cells) and the assessment
receipt `execution/selection_discovery.json` (SHA-256 `0e048226…`) select the shared recipe **P**
(population-normalized weighted cross-entropy): P qualified on five-class only (mean common-four gain +0.44
points over the B five-class control, folds 1–4 +0.53, fold 0 +0.07; all recall guards passed) and failed on
four-class (−0.19) and six-class (−0.87); S qualified nowhere because every S target failed the three-point
mean-recall-drop guard. Mean common-four balanced accuracies (seed 42, five folds): B four 74.28, B five 73.99,
B six 74.91, P four 74.09, P five 74.43, P six 74.04, S four 74.56, S five 74.10, S six 74.85. This is a
screening decision on reused development folds; it supports no improvement claim. Confirmation: P four/five/six
plus B four on seeds 43/44 (40 cells, 39 new fits; the B four fold-0 seed-43 control is reused).

## 2026-10-15/16 — confirmation sessions

Confirmation manifest `execution/manifest_confirmation.json` (SHA-256 `8342e2ef…`; 40 cells, shared recipe P,
bound to the discovery manifest and receipt) and policy `policy_confirmation.json` (SHA-256 `8aae188a…`, 39
permitted units, superseding the discovery policy so the session count continues). Session 4
(`session-20261015T235207Z-452944f4`) started 23:52Z after six stockout refusals; the controller limited it to
162 minutes by the remaining daily allowance (hard stop 02:34Z). Kernel selection, CUDA check, deploy and the
host `plan` for a confirmation unit passed; the confirmation loop started at 23:59Z.
Session 4 closed 2026-10-16T02:03Z: 2 h 11 min, $1.93 gross, VM TERMINATED; nine confirmation units completed and
verified (P four-class: folds 0–2 both seeds, fold 3 seed 44, fold 4 seed 43, plus fold 3 seed 43); 30 remain.
Running total after four sessions: 11 h 25 min, $10.05 gross running; allocated 50,580 of 86,400 GPU-seconds.
Session 5 start: L4 capacity in `us-central1-a` refused twelve consecutive starts between 02:04Z and 04:01Z
(`ZONE_RESOURCE_POOL_EXHAUSTED`); retries continue every ten minutes. No fallback or migration is attempted.
Session 5 (`session-20261016T040337Z-2f76ac47`, full 237 minutes, hard stop 08:00Z) started 04:04Z on the first
attempt of a second retry cycle; kernel selection, CUDA, mounts, deploy and the host `plan` passed; the
confirmation loop started 04:07Z with 30 units queued.
Session 5 closed 06:57Z: 2 h 53 min, $2.55 gross, VM TERMINATED; twelve confirmation units completed and verified
(P four-class fold 4 seed 44; P five-class all five folds on both seeds; P six-class fold 0 seed 43); 21 of 39
done, 18 remain (nine P six-class, nine B four-class). Running total after five sessions: 14 h 19 min, $12.60 gross
running; allocated 64,800 of 86,400 GPU-seconds (planned durations), leaving one full session and one of about two
hours; `evidence/session_5_accounting.json`.
Session 6 (`session-20261016T073033Z-6466aa0c`, 237 minutes, hard stop 11:27Z) started 07:31Z on the fourth attempt;
kernel selection, CUDA, mounts, deploy and the host `plan` passed; the confirmation loop started 07:35Z with 18
units queued (nine P six-class, nine B four-class).
Session 6 closed 10:57Z: 3 h 27 min, $3.04 gross, VM TERMINATED; fifteen units completed and verified (P six-class
folds 0–4 on both seeds; B four-class fold 0 seed 44, folds 1–2 both seeds, fold 3 seed 43); 36 of 39 done. Running
total after six sessions: 17 h 46 min, $15.64 gross running; allocated 79,020 of 86,400 GPU-seconds, so the last
three controls (B four-class fold 3 seed 44, fold 4 both seeds) run in a two-hour session 7;
`evidence/session_6_accounting.json`.
Session 7 (`session-20261016T111033Z-4645eeb9`, 117 minutes by request, hard stop 13:07Z) started 11:11Z on the
second attempt for the last three controls.
Session 7 closed 11:59Z: 49 minutes, $0.73 gross, VM TERMINATED (confirmed); the last three controls completed and
verified. **Confirmation grid complete: 39 new fits plus the reused B four-class fold-0 seed-43 control.**
Campaign execution totals (controller ledger): seven sessions, 18 h 37 min running, $16.37 gross running cost,
86,040 of 86,400 authorized GPU-seconds allocated (planned durations), seven of eight sessions used; retained
storage is additional (`evidence/campaign_spending_summary.json`).

## 2026-10-16 — confirmation decision (frozen rule) and configuration selection

Collection `execution/collection_confirmation.json` (SHA-256 `37d8c344…`, 40 verified cells) and receipt
`execution/selection_confirmation.json` (SHA-256 `9e1fce84…`). Seed-43/44 differences were averaged within each of
the five folds before inference (n=5, df=4; 10,000 paired fold bootstrap resamples, seed 42; Bonferroni over the
five fixed contrasts). **No challenger passed every gate; the unchanged B four-class configuration is retained.**
Mean common-four balanced accuracy (seed-averaged, five folds): B four 73.87, P four 74.24, P five 73.97, P six
73.31. Contrasts: P4−B4 +0.36 points (above the 0.2 threshold; adjusted bootstrap [−0.39, +1.42], adjusted t
[−1.65, +2.38], lower bounds not positive); P5−B4 +0.10 (not supported); P5−P4 −0.26; P6−B4 −0.56 and P6−P4 −0.93
with the Class VIII mean recall dropping more than three points (guard failed). This is validation-only evidence on
reused development folds with two additional optimizer seeds; it supports no improvement claim, no five-versus-six
claim, and no refit or test readiness. The campaign's selected configuration is therefore the v3 late-fusion
direct-four recipe already recorded in PARAMETER_FINDINGS; whether the historical refit `a0007846…` may be reused
for Stage 7 remains a separate compatibility decision that this run does not make.

## 2026-10-16 — Stage 6B source verified; evaluation preparation reconciled

The user asked to finish the selected B-four model, reusing a compatible full non-test refit rather than retraining
for a campaign name, and to prepare credible independent evaluation. The user authorized CPU verification and
documentation/evidence updates only: no GPU, training, candidate-test collection, test predictions, held-out
evaluation, commit or push; no atom-level encoder or extra inputs; architecture and features unchanged; EC deferred.

- **Stage 6B source.** Three independent CPU verifications (checkpoint identity and source; configuration;
  membership, features and normalization) and three adversarial reviews found no mismatch (persisted as
  `evidence/stage6b_source_20261016/session_reviews_20261016.json`, `a359a969…`). The record
  `stage6b_source_compatibility.json` (SHA-256 `bd8154cf…`, generator `be3af887…`) passes 25 of 25 checks; it was
  regenerated after a documentation review corrected its condition and disclosure wording (first version
  `ae7d678f…`, same checks). Under the user's instruction and the specification's
  `future_refit.baseline_retained` rule, refit `a00078466cf6…` is designated the Stage 7 source without retraining
  ([evidence](evidence.md#stage-6b-source-and-compatibility-2026-10-16)). One policy conflict stays open for
  the user: Plan (sections 2 and 6) and the playbook make Stage 7 one-shot per final training/refit run, and this
  refit already had its pre-declared secondary exact-Zenodo pass, so a primary Stage 7 report from it needs the
  user's explicit exception or a Plan amendment before any scoring.
- **Evaluation preparation.** Training-side prior-use IDs and release dates, obsolete entries and the current blocked
  template are in `evidence/evaluation_prep_20261016/` (CSV `788a7f0a…`, summary `3b992424…`, template `c896d9d3…`);
  the inventory and ordered blockers are in [evaluation.md](evaluation.md#preparation-inventory-2026-10-16). Editing
  evaluation.md supersedes the reviewed protocol copy `4e11014d…`, which stays unchanged in the sealed package; a
  future authorized `prepare` records the copy beside the tool it runs and must name it.
- **Access.** No test data file was deliberately opened. A verifier's repository-wide text search read a CLEAN
  provenance audit file and printed one CLEAN fold-0 test row, and agents read the private local backup's `SHA256SUMS` (names
  and hashes of the evaluated step F test outputs) without opening those files
  ([DATASETS](../../DATASETS.md#test-use-ledger)); nothing was used.
- **Documentation.** STATUS, README, handoff, evidence, PARAMETER_FINDINGS, the metal playbook prose, DATASETS,
  GETTING_STARTED, the backup record and the batch index were reconciled. The STATUS committed at `cb3776d` had
  been overwritten without a snapshot; it is preserved below with the superseded 2026-10-16 STATUS.

### Superseded STATUS snapshot (2026-10-14, committed at cb3776d)

```markdown
# DeepMzyme Current Experiment Status

- Status: active (2026-10-14 metal completion CPU preparation reviewed, repaired and re-sealed; execution not authorized)
- Last execution evidence: 2026-10-13 EC1 inventory; no new model fits or test evaluations.

## Current objective and stage

- Current campaign: [metal_completion_2026-10](docs/campaigns/metal_completion_2026-10/README.md)
- Objective: finish a bounded metal optimization comparison across four/five/six objectives, then independently evaluate the frozen final model; EC work is deferred by the user's metal-first choice.
- Other open campaigns: [EC independent support](docs/campaigns/ec1_independent_support_2026-10/README.md) (deferred); [metal_single_gpu](docs/campaigns/metal_single_gpu/README.md) (paused); [zenodo_pmm_exact_2026-09-24](docs/campaigns/zenodo_pmm_exact_2026-09-24/README.md) (paused).
- Stage: successor CPU preparation independently reviewed (no blocker; majors repaired or pinned) and re-sealed; the discovery manifest is frozen. Next is validation-only discovery, then Stage 6 grouped-fold/seed confirmation; neither has run. [Evidence](docs/campaigns/metal_completion_2026-10/evidence.md) owns hashes, checks and control reuse; the [log](docs/campaigns/metal_completion_2026-10/log.md) records the review.

## Anchor and evidence state

- Best validation result: retained v3 late-fusion direct-four, terminal 50-epoch cosine training. Grade 2 validation evidence: five grouped folds, seed 42, common-four BA 74.28%; no supported benefit from five/six training. [Findings](docs/PARAMETER_FINDINGS.md#pmm-ion-metal-v3-objective-test-and-stage-6-selection-closed-2026-10-13).
- Lower trunk LR remains unpromoted. [Closed comparison](docs/archive/campaigns/pmm_trunk_lr_2026-10/README.md).
- Historical refit `a0007846…` and secondary test reports remain unchanged. [Preserved summary](docs/notebook_outputs/summaries/summary_pmm_ion_metal_v3_20261013.md). New code and synthetic checks are not accuracy evidence; the unexecuted successor is not a better model.
- EC remains at fixed-split Grade 3 evidence (deferred campaign).

### Known caveats and open mismatches

- Additional seeds reuse development folds. Cross-target ranking requires the campaign assessor ([TECH-010](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-010--four-class-endpoint-and-paired-metal-target-recipes-are-not-reconciled)); no new model has been evaluated.
- EC child-file guard repair remains open ([TECH-030](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-030--discovered-ec-cache-files-can-bypass-the-training-path-guard)); the trainer's focal branch ignores label smoothing, unreachable here ([TECH-031](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-031--the-focal-metal-loss-silently-ignores-label-smoothing-and-uses-weighted-ce-confidence)).

## Dataset and test readiness

The primary independent final-test route remains unresolved. [DATASETS](docs/DATASETS.md#test-use-ledger) owns historical exposure; opened PMM tests are not pristine. The [offline evaluation protocol](docs/campaigns/metal_completion_2026-10/evaluation.md) authorizes no candidate retrieval and establishes no support. Source dates, complete exclusions, near-copy checks and report/refit freezes precede scoring.

## Blockers and immediate next action

- Authorized now: nothing to run. No GPU start, real-data training/inference, download, new held-out access, commit or push.
- GPU/VM: `<gpu-vm>` TERMINATED (verified 2026-10-14T15:51Z); gross rate $0.87917/h; retained disk and two snapshots about $1.00/day. Backups: the private backup record.
- Next: the user's scoped decision on the [costed proposal](docs/campaigns/metal_completion_2026-10/evidence.md#proposed-execution-ceiling--approval-pending); then the GPU-host [deployment prerequisite](docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md#cpu-preparation-block) (identical absolute paths) and the authorization/session/spending receipts before the first discovery unit. The [handoff](docs/campaigns/metal_completion_2026-10/handoff.md) is the next-session prompt.

## History and update rule

The exact previous STATUS is preserved in the [campaign log](docs/campaigns/metal_completion_2026-10/log.md). Overwrite this page only after preserving dated history. Scientific policy remains in Plan, recipes in the metal playbook, dataset/test access in DATASETS, and validation conclusions in PARAMETER_FINDINGS.
```

### Superseded STATUS snapshot (2026-10-16 execution closeout, staged)

```markdown
# DeepMzyme Current Experiment Status

- Status: active (2026-10-16 metal completion discovery and confirmation complete; B four-class retained; no execution authorized)
- Last execution evidence: 2026-10-16 confirmation fits on the GCP L4 (69 new fits, seven sessions); no test evaluations.

## Current objective and stage

- Current campaign: [metal_completion_2026-10](docs/campaigns/metal_completion_2026-10/README.md)
- Objective: finish a bounded metal optimization comparison across four/five/six objectives, then independently evaluate the frozen final model; EC work is deferred by the user's metal-first choice.
- Other open campaigns: [EC independent support](docs/campaigns/ec1_independent_support_2026-10/README.md) (deferred); [metal_single_gpu](docs/campaigns/metal_single_gpu/README.md) (paused); [zenodo_pmm_exact_2026-09-24](docs/campaigns/zenodo_pmm_exact_2026-09-24/README.md) (paused).
- Stage: discovery (45 cells) and Stage 6 confirmation (40 cells) complete and assessed by the frozen rules: shared recipe P, then no challenger passed the adjusted paired gates, so the unchanged B four-class configuration (the v3 recipe) is retained. Stage 6B/7 decisions are pending. [Evidence](docs/campaigns/metal_completion_2026-10/evidence.md) owns hashes, checks and control reuse; the [log](docs/campaigns/metal_completion_2026-10/log.md) records the review.

## Anchor and evidence state

- Best validation result: retained v3 late-fusion direct-four, terminal 50-epoch cosine training. Grade 2 validation evidence: five grouped folds, seed 42, common-four BA 74.28%; no supported benefit from five/six training. [Findings](docs/PARAMETER_FINDINGS.md#pmm-ion-metal-v3-objective-test-and-stage-6-selection-closed-2026-10-13).
- Trunk LR unpromoted ([closed](docs/archive/campaigns/pmm_trunk_lr_2026-10/README.md)).
- Historical refit `a0007846…` and secondary test reports remain unchanged. [Preserved summary](docs/notebook_outputs/summaries/summary_pmm_ion_metal_v3_20261013.md). New code and synthetic checks are not accuracy evidence; the unexecuted successor is not a better model.
- EC stays at fixed-split Grade 3 evidence (deferred).

### Known caveats and open mismatches

- Additional seeds reuse development folds. Cross-target ranking requires the campaign assessor ([TECH-010](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-010--four-class-endpoint-and-paired-metal-target-recipes-are-not-reconciled)); no new model has been evaluated.
- EC child-file guard repair remains open ([TECH-030](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-030--discovered-ec-cache-files-can-bypass-the-training-path-guard)); the trainer's focal branch ignores label smoothing, unreachable here ([TECH-031](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-031--the-focal-metal-loss-silently-ignores-label-smoothing-and-uses-weighted-ce-confidence)).

## Dataset and test readiness

The primary independent final-test route remains unresolved. [DATASETS](docs/DATASETS.md#test-use-ledger) owns historical exposure; opened PMM tests are not pristine. The [offline evaluation protocol](docs/campaigns/metal_completion_2026-10/evaluation.md) authorizes no candidate retrieval and establishes no support. Source dates, complete exclusions, near-copy checks and report/refit freezes precede scoring.

## Blockers and immediate next action

- Authorized now: nothing to run; the 2026-10-15 authorization is consumed (86,040 of 86,400 GPU-seconds). No GPU start, refit, test access, download, commit or push.
- GPU/VM: `<gpu-vm>` TERMINATED (confirmed 2026-10-16T11:59Z) after seven sessions: 18 h 37 min, $16.37 gross running; $0.87917/h; storage ~$1/day. Backups: the private backup record.
- Next: the user's decisions on the Stage 6B source (reuse of refit `a0007846…` after compatibility verification, or a newly authorized refit) and the independent evaluation route; see the [handoff](docs/campaigns/metal_completion_2026-10/handoff.md) and [findings](docs/PARAMETER_FINDINGS.md#metal-completion-bps-loss-recipes-on-fourfivesix-objectives-executed-2026-10-1516).

## History and update rule

The exact previous STATUS is preserved in the [campaign log](docs/campaigns/metal_completion_2026-10/log.md). Overwrite this page only after preserving dated history. Scientific policy remains in Plan, recipes in the metal playbook, dataset/test access in DATASETS, and validation conclusions in PARAMETER_FINDINGS.
```
