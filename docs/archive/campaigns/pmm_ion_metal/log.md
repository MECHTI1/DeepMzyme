# Preserved STATUS history

Historical text, copied verbatim before the Job B rewrite. Old next actions,
permissions, counts and claims are dated evidence, not current instructions.
Current authority: [EXPERIMENT_STATUS.md](../../../../EXPERIMENT_STATUS.md).
Source ranges and hashes: [preservation manifest](../../../../docs/archive/consolidation_2026-09/inventory/job_b_status_preservation.json).
Literal blocks preserve original relative link spelling; use this folder’s
README for navigable evidence links. No historical error has been silently fixed.

## pmm-013

2026-10-03 — the user closed this campaign at fold 0 (explicit decision after a
read-only re-audit). Nine of 45 fits are its final, validation-only evidence
(Grade 5 per fit; incomplete grid Grade 6). The 36 fold 1–4 fits were not run;
they did not fail. Timeline: fold-0 fits 2026-09-27/28; deferral 2026-09-28 for
budget; post-hoc checkpoint analysis 2026-09-29; closure 2026-10-03.

Under the pre-registered native-BA selection rule, direct-four was at or above
six-class in all three families and five-class exceeded direct-four only for
Only-ESMC; the post-hoc symmetric and selection-free rules reverse several of
these comparisons and stay labelled post-hoc. The neutral four-versus-five/six
test is therefore unanswered: not "no difference" and not "four wins". The
closure reasons do not depend on which target leads: size-strata folds
([TECH-020](../../../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-020--greedy-k-fold-assignment-concentrates-large-groups-in-fold-0)),
different checkpoint rules for CV and refit
([TECH-027](../../../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-027--cross-validation-and-final-refit-use-different-checkpoint-rules)),
a five-value percentile bootstrap that under-covers, and fold-0 Cu selection luck.
The released PMM comparator's five-fold result stays recorded with the label-leak
caveat of [TECH-028](../../../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-028--pmm-comparator-inherits-a-true-metal-label-leak).
The three binding-aware fold-0 fits stay separate exploratory evidence; their
remaining fits will not run here.

No promotion, Stage 6 confirmation, Stage 6B refit or held-out evaluation
occurred. Fold-0 results of this campaign are never confirmatory evidence for a
later study. All run artifacts, caches, embeddings and the frozen code checkout
`/media/mechti/Data1/DeepMzyme_Data/campaigns/_code/pmm_core_scope_v2` are
preserved. `docs/plans/pmm_core_scope_v2.json` stays unchanged: its
`deferred_by_user` status keeps the runner refusing execution. Successor planning:
[pmm_ion_metal_v3](../../../campaigns/pmm_ion_metal_v3/README.md).

## pmm-012

Preserved verbatim from STATUS before the 2026-10-03 closure update.
Source: `7221189:EXPERIMENT_STATUS.md:1-87`.

```text
# DeepMzyme Current Experiment Status

- Status: paused (2026-09-28 user decision)
- Last execution evidence: 2026-09-28. Documentation reconciliation: 2026-10-03 (storage retirement only).

## Current objective and stage

- Current campaign: pmm_ion_metal, PMM core scope v2 (paused; scope `pmm-core-v2`, campaign `pmm_ion_metal_v2_context`) — [README](docs/campaigns/pmm_ion_metal/README.md)
- Other open campaigns: [metal_single_gpu](docs/campaigns/metal_single_gpu/README.md) (paused); [zenodo_pmm_exact_2026-09-24](docs/campaigns/zenodo_pmm_exact_2026-09-24/README.md) (paused; test possibly opened)
- Stage: exploratory fold-0 comparison retained; no Stage 6 confirmation, Stage 6B refit or Stage 7.

**Nine ordinary-readout fits are retained; the remaining 36 fits are deferred.**
No completed fivefold neural confirmation, model/target promotion, final refit
or held-out evaluation is claimed. Exact scientific identities, sub-batches,
results and evidence grades live in the campaign README and the
[v2 scope](docs/plans/pmm_core_scope_v2.json).

## Anchor and evidence state

- Best validation result: none promoted; the fold-0 core fits are Grade 5 exploratory evidence (incomplete grid Grade 6) per the [PMM README](docs/campaigns/pmm_ion_metal/README.md); results in the [core summary](docs/notebook_outputs/summaries/summary_pmm_core_continuation_20260928.md).

Seven original strict replay passes and the GVP5/GVP6 strict failures remain
recorded. All nine core fits have separate retrospective `pmm-core-replay-v1`
qualification; this does not rewrite the original `1e-6` failures as passes.
The core summary owns the result table and diagnostic limits. The three earlier
binding-aware fits remain separate exploratory evidence; further awareness work
is paused.

The [EC1 reference](docs/archive/campaigns/ec1_standalone_v12_2026-09-14/README.md)
retains twelve completed fixed-split runs (Grade 3), not promotion. Later EC
workflow reconciliation and cross-task holdout certification remain required
before auxiliary learning; [issues](docs/FOLLOW_UP_TECHNICAL_ISSUES.md) own details.

### Known caveats and open mismatches

- Cross-scheme ranking: the notebook Stage 6/6B route is single-scheme; the
  paired-CI gate uses the Stage 6 selection metric (native by default), the
  rare-recall gate is always native, and the default tie-breakers are native.
  Comparing target schemes needs a campaign assessor on collapsed-four balanced
  accuracy. Latent, not active now
  ([TECH-010](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-010--four-class-endpoint-and-paired-metal-target-recipes-are-not-reconciled)).

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

- Authorized now: nothing (no experiment, GPU work, final refit or held-out evaluation).
- GPU/VM: retired; provider inventory verified no VMs/disks at 2026-10-03 13:48:51 UTC. One restore-checked standard snapshot remains (34.77 GiB, approximately $1.74/month gross). [Receipt and recovery boundary](docs/campaigns/pmm_ion_metal/storage_retirement_20261003.json); [storage decision](docs/campaigns/pmm_ion_metal/log.md#pmm-011).

Resume requires an explicit user request, refreshed remaining-budget authority
and new readiness verification against the execution environment. The proposed
additional 34 hours/$34 is unapproved and no longer awaiting an immediate
execution decision. No extra fold-0 fits or final refits are authorized
([scope authorization](docs/plans/pmm_core_scope_v2.json);
[pmm-001](docs/campaigns/pmm_ion_metal/log.md#pmm-001)).

Stage 6 grouped-fold selection (or explicitly labeled fallback), completed/reused
Stage 6B full non-test refit, frozen report/checkpoint rules and a scientifically
resolved final-test route must precede one-shot Stage 7. No test-based tuning,
ranking, promotion, rejection or checkpoint choice; see [Plan](Plan.md#canonical-staged-metal-training-pipeline).

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
```

## pmm-010

Preserved verbatim from STATUS before the user-authorized 2026-10-03 storage
retirement update. Scientific execution remains paused.

```text
- Last execution evidence: 2026-09-28. Documentation reconciliation: 2026-09-28.
- GPU/VM: TERMINATED, independently checked 2026-09-28 06:00:18 UTC; the retained 150-GB disk costs approximately $15/month. This is historical provider evidence, not a new live resource check ([closeout](docs/notebook_outputs/summaries/summary_pmm_core_continuation_20260928.md)).
```

## pmm-011

2026-10-03 — user approved the recommendation to preserve a standard snapshot
and local backups, then retire the stopped VM and its disk to reduce recurring
storage cost. This does not resume scientific work or authorize compute starts.

The [retirement receipt](storage_retirement_20261003.json) owns the resource IDs,
provider inventory, snapshot size, verified price and local evidence path. The
snapshot reached READY and was restored to a temporary disk; its source identity
matched, and the temporary disk was deleted. This verifies provider restoration,
not a filesystem mount or boot. The local 72-file diagnostic backup passed a
fresh SHA-256 readback. Full instance/disk metadata, controller configuration,
controller source and the archive receipt were independently copied and hashed
on the secondary drive before source deletion.

The original VM and 150-GiB disk were deleted through the controller. Independent
inventory found no VMs, disks, reserved addresses or recoverable snapshots; one
standard snapshot remains. Its 34.77 GiB at $0.05/GiB-month gives approximately
$1.74/month gross, versus the previous $15/month disk. Billing stays enabled to
retain the snapshot. No GPU was started and no training/evaluation was performed.

The controller gained a storage-only `vm-archive` preparation command, and its
delete path now explicitly removes the requested boot disk even when recovery
set `autoDelete=false`. All 85 controller tests passed. Current status/report
daily reserves can still show conservative configured-disk estimates; these are
not invoices. The receipt's snapshot-size calculation describes retained storage.

Resume requires a separately authorized controller restore route, using the
saved snapshot and existing boot-disk/automatic-STOP safeguards. The public
archive-to-VM restore command is not yet implemented; default `vm-create` would
create a fresh OS. See the [archive procedure](../../../GCP_GPU_RUNBOOK.md#archive-a-paused-environment).

## pmm-001

Source: `3c0f80c:EXPERIMENT_STATUS.md:10-38`.

```text
**2026-09-28 — user decision: retain fold 0; defer further fits:**
- Keep the nine existing ordinary-readout fold-0 results as the current
  exploratory comparison of four/five/six-class training. Their original
  strict outcomes and separate retrospective qualifications remain attached.
- **The 36 remaining fold-1–4 fits are deferred**, with the full 45-fit design,
  code, caches, checkpoints, PMM folds and resume instructions preserved for
  future work. This is not completed fivefold confirmation or final selection.
- The additional 34-hour/$34 proposal is **not approved and no longer awaiting
  a decision for immediate execution**. Resume only after an explicit user
  request, refreshed remaining-budget authority and new readiness verification.
  No further fits, extra fold-0 experiments or final refits are authorized by
  this decision. The core scope and runner record/refuse the active deferral.
- Deferral verification: all 44 core-runner CPU tests passed. A read-only plan
  against the frozen campaign reports 36 deferred selectors and nine preserved
  fold-0 units; a training attempt is refused before source/artifact access.
  No child process or training output was created by these checks.
- No GPU was started for this update. Last verified VM state is TERMINATED;
  retained disk storage continues. The historical readiness/forecast evidence
  below is preserved; its old scope/implementation hash cannot authorize a
  future run after this update.
- Subsequent user-requested cleanup removed the incomplete Antigravity
  GVP-improvement draft after review confirmed checkpoint/default regressions,
  incompatible experiment commands and unreliable result handling. Five source
  files were restored to their committed versions; three untracked scripts and
  one untracked test file were deleted. The
  [evidence-ranked improvement plan](docs/plans/gvp_and_esmc_evidence_ranked_improvement_plan.md)
  remains saved verbatim for future separately scoped implementation. This
  cleanup does not resume the frozen PMM campaign or authorize GPU execution.

```

## pmm-002

Source: `3c0f80c:EXPERIMENT_STATUS.md:39-73`.

```text
**2026-09-28 — core readiness verified; concurrency measured; GPU stopped:**
- All **nine core fold-0 fits** qualify under explicit retrospective
  `pmm-core-replay-v1` integration. Seven retain strict passes; GVP5/GVP6 retain
  their original strict failures. No fit was repeated or prediction replaced.
  Active grid: **9/45 trained, 36 absent fits**. Full fivefold confirmation,
  promotion, final refit and held-out evaluation remain incomplete.
- GVP5 input diagnosis and all 20 fixed evaluations completed. All classes stayed
  unchanged; original-setting probabilities varied by up to `2.622604e-6`, while
  strict passes were bitwise identical. Maximum difference against preserved
  exports was `3.515757e-6`. Original strict `1e-6` failure stays recorded; the
  separate engineering agreement uses absolute `1e-5` with exact classes/metrics.
- Root continuation/replay/assessment/refit-preview implementation passed
  **195 CPU tests**, plus **42 final runner tests**. Actual host readiness binds
  all nine units and their diagnostic evidence. The frozen training source
  stays unchanged; unrelated shared-checkout edits are preserved. Recompute
  readiness after staging into the execution environment.
- Two disposable GVP processes on one L4 achieved **1.518× step throughput**
  (34.1% less elapsed step time). Both weight comparisons exceeded the probe's
  numerical comparison tolerance; no serial-repeat control was included.
  This is a short microbenchmark, not proven end-to-end speedup or accuracy
  equivalence. **Production remains one training worker per GPU.**
- All 72 remote files were independently hash-verified. Provider confirmed
  **TERMINATED at 06:00:08 UTC**, independently checked at **06:00:18 UTC**.
  Session use: **1,292 seconds / $0.3156 estimated running gross**. Cumulative
  running use: approximately **11.8358 hours / $10.4057**, plus additional
  storage. One 150-GB disk remains, approximately **$15/month**.
- Next boundary: the serial forecast is **21–33 additional VM hours** for the
  36 absent fits. User decision is pending on **34 additional hours / $34
  additional gross**, retaining four-hour/$6 session and six-hour/$10 UTC-day
  limits. The earlier 30-hour/$34 total proposal remains unapproved. Neither
  forecast nor readiness grants compute authority. No allocation remains active.
- [Full results, four/five/six table and limitations](docs/notebook_outputs/summaries/summary_pmm_core_continuation_20260928.md);
  [portable evidence](docs/notebook_outputs/raw/pmm_core_continuation_20260928/README.md);
  [execution contract](docs/plans/pmm_core_execution_v1.md).

```

## pmm-003

Source: `3c0f80c:EXPERIMENT_STATUS.md:74-106`.

```text
**2026-09-28 — three five-class fits trained; two verified, one replay failure; GPU closed:**
- All three new fold-0/seed-42 fits completed 50 epochs. Only-ESMC and graph-level
  late fusion passed original strict replay; Only-GVP did not. All terminal
  artifacts are independently backed up and acknowledged (71/70/71 files).
- Common-four BA from native-selected checkpoints:

  | Family | Five-class training | Six-class training | Higher on this fold |
  |---|---:|---:|---|
  | Only-ESMC | 89.4354% | 85.8393% | Five |
  | Only-GVP, provisional comparison | 78.2397% | 82.8399% | Six |
  | Graph-level late fusion | 87.6608% | 88.8124% | Six |

- GVP5's class predictions and BA match across exports, but maximum probability
  difference `2.38e-6` fails the unchanged `1e-6` limit. No retry or threshold
  change occurred. GVP6 retains historical supplemental-only agreement. See the
  [complete summary and qualifications](docs/notebook_outputs/summaries/summary_pmm_five_class_screen_20260928.md).
  This is exploratory single-fold evidence, not target promotion or fivefold confirmation.
- Recovery `5279a430fc79` succeeded in `us-central1-a`. The VM stopped at
  **04:55:14 UTC**, independently confirmed **TERMINATED at 04:57:26 UTC**.
  Verified cleanup removed the superseded VM/disk and temporary snapshot,
  including its recycle-bin copy. The working 150-GB disk remains, approximately
  **$15/month**. [Closeout evidence](docs/notebook_outputs/raw/pmm_five_class_screen_20260928/execution/verified_closeout.json).
- This session used **2 h 10 m 15 s / $1.9086 estimated running gross**, plus
  additional storage. Campaign cumulative running use is **11.4769 hours /
  $10.0901 estimated gross**, plus storage. Four-hour/$6 session and six-hour/$10
  UTC-day caps stayed unchanged; the larger full-grid proposal is not activated.
- Active core v2: **9/45 trained, 36 untrained**; seven original strict passes,
  one historical supplemental-only GVP6, one uncertified GVP5. The three earlier
  aware fits remain separate; further awareness work is paused. Next resolve
  TECH-023/025 and remaining-budget authority before full-fold execution and
  assessment. No held-out access, HPO, promotion or final refit occurred. Reuse
  all completed fits; do not retry GVP5 until a chance pass.

```

## pmm-004

Source: `3c0f80c:EXPERIMENT_STATUS.md:107-139`.

```text
**2026-09-28 — initial five-class preparation and stockout, superseded above:**
- Add Mn/Cu/Zn/Fe/Co+Ni training to Only-ESMC, Only-GVP and graph-level late
  fusion, ordinary readout. The active [v2 scope](docs/plans/pmm_core_scope_v2.json)
  is **45 ordinary-readout fits**, distinct from the old 45-fit awareness grid.
  At adoption, **6 core fits are trained and 39 remain** (24 four/six, 15 five).
  All previous fits, failures, PMM folds, features and caches are preserved.
- First execute the [three-fit five-class fold-0 screen](docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md#five-class-exploratory-screen-on-the-frozen-pmm-fold-0).
  Native-five BA selects the checkpoint; compare its common-four probabilities
  on the same validation ions. Keep Fe and Co+Ni recalls and distinguish native
  Co+Ni `Class VIII` from common-four Fe+Co+Ni. No five-class result is yet claimed.
- Implemented and pushed in `b80d164`; **72 targeted tests passed**, including
  synthetic CPU training and independent replay. The actual frozen-cohort CPU
  preview produces the three expected commands. See the
  [preparation/capacity evidence](docs/notebook_outputs/raw/pmm_five_class_screen_20260928/README.md).
- The root adapter preserves the frozen scientific source and uses original
  strict `1e-6` independent replay. The retrospective nine-fit v2.1 policy stays
  limited to those old fits. Full-grid TECH-023/025, Stage 6B and held-out gates
  remain open; three exploratory fits do not resolve them.
- The full-grid preview now lists 39 fixed candidate units, without treating
  them as a verified missing-work queue. Five-class folds 1–4 intentionally have
  no execution command yet. Use isolated frozen checkout
  `/media/mechti/Data1/DeepMzyme_Data/campaigns/_code/pmm_core_scope_v2`;
  unrelated shared-checkout model/configuration edits are preserved.
- Binding-aware work remains paused. The previous 30-hour/$34 proposal remains
  unapproved; session/day caps are unchanged. The requested four-hour same-VM
  start failed with `ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS` in `us-central1-c`;
  no alternative zone was suggested. Provider verified **TERMINATED at 01:59:48
  UTC**. No new fit or GPU compute use occurred. The controller refused recovery
  preview because the previous pass is complete; its receipt is preserved.
  **Next:** user decision on a new bounded same-region recovery pass with
  evidence-gated cleanup, or keep stopped and retry later. Latest running use remains
  **9.3061 hours / $8.1815 estimated gross**, plus retained storage.

```

## pmm-005

Source: `3c0f80c:EXPERIMENT_STATUS.md:140-176`.

```text
**2026-09-28 — historical v1 core scope, superseded by the five-class addition above:**
- Prioritize ordinary-readout Only-ESMC, Only-GVP and graph-level late fusion,
  each with direct four-class and six-class training on the same five folds.
  The active matrix is **30 fits**, with **6 trained core fits and 24 remaining**.
  Five core fits pass legacy replay; all six have supplemental fold-0 v2.1
  agreement. TECH-023 remains an explicit certification-integration issue.
- Keep the three trained binding-aware fits as exploratory evidence. Their
  twelve remaining folds, new awareness variants and awareness HPO are paused.
  This decision follows inspection of fold 0; it is not evidence of general
  lack of benefit. Preserve every run and the original 45-fit scope in history.
- [Scope manifest](docs/plans/pmm_core_scope_v1.json),
  [amended plan](docs/plans/metal_level_metal_task_compared_PMM_final_plan.md#active-amendment--prioritize-core-models-pause-binding-awareness),
  and [core recipe](docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md#active-core-only-continuation)
  now govern continuation. The root entry point previews exactly 24 core
  candidates and refuses training/assessment/refit/test pending TECH-023 and
  [TECH-025](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-025--core-scope-needs-a-separate-assessment-and-promotion-bridge).
  Its 32 targeted tests passed. This amendment edits no scientific source.
  Concurrent model/configuration edits in the shared working tree subsequently
  changed its source identity; the guard correctly refused that checkout.
  The isolated checkout at
  `/media/mechti/Data1/DeepMzyme_Data/campaigns/_code/pmm_core_scope_v1`
  passed the same 32 tests and actual CPU preview. It preserves frozen source
  without reverting unrelated edits or bypassing source/configuration checks.
  Canonical `runtime/core_scope_verification.json` binds the successful preview.
- No GPU was started, training repeated or test data accessed for this update.
  Last verified provider state remains TERMINATED. Historical campaign use is
  **9.3061 VM hours / $8.1815 estimated running gross**, plus retained storage.
  Remaining core forecast: **11.91 additional VM hours / $10.47 running gross**,
  excluding additional storage and final refits/reporting. The old 36-fit /
  17.37-hour forecast is superseded; the calculation is saved in canonical
  `runtime/core_scope_amendment_20260928.json`. Actual cost and source records
  are preserved. The old 30-hour/$34 proposal remains unapproved.
- **Next:** finish the prospective replay and core assessment/refit bridge,
  refresh/authorize the reduced-core compute budget, then run only missing core
  units. Do not activate awareness or count a filtered legacy assessment as
  core completion. Final full-train refit and one-shot held-out gates remain.

```

## pmm-006

Source: `3c0f80c:EXPERIMENT_STATUS.md:177-214`.

```text
**2026-09-28 — numerical diagnosis complete; nine-model screen agreement verified:**
- **9/9 trained and qualified under the explicit v2.1 agreement policy;
  8/9 pass the original replay contract.** No fit, checkpoint, prediction export
  or legacy receipt was replaced. The
  [supplemental report](docs/notebook_outputs/raw/pmm_replay_diagnostic_20260928/screen_agreement_v2_1.md)
  uniformly checks every preserved replay with absolute probability difference
  at most `1e-5`, unchanged class predictions and checkpoint metrics. It is
  retrospective single-fold evidence, not a legacy-gate pass or promotion.
- GVP6's predictive tensors match the preserved training cache on all 1,492
  ions / 94 batches. The sole raw difference is an inactive EC label, audited
  separately; CPU equivalence was checked on the first 16-example batch.
  Ten evaluations under original settings varied by up to **2.324581e-6**;
  ten strict deterministic evaluations were bitwise identical across two
  processes. All native/common-four class predictions remained unchanged.
  This establishes numerical variation on fixed inputs; see the
  [diagnostic evidence](docs/notebook_outputs/raw/pmm_replay_diagnostic_20260928/README.md).
- Six-class GVP selected epoch **31**: common-four BA **82.8399%**, macro-F1
  **75.4985%**, native-six BA **63.6275%**; Fe/Co/Ni recalls
  **75.6545% / 34.2857% / 18.8034%**. Its common-four BA is below direct-four
  GVP's **86.0232%**. All three six-class arms have lower common-four BA than
  their matched direct-four arm on this fold; no target formulation is promoted.
  Binding awareness still shows no primary-metric gain in the single-fold screen.
- **VM TERMINATED**, stopped **00:04:22 UTC**, independently checked at
  **00:05:21 UTC**. Session `session-20260927T234132Z-8191aea8` used
  **1,371 seconds / $0.3347 estimated running gross**. All 42 remote artifacts
  verified on the host and acknowledgment was uploaded before shutdown.
  One 150-GB disk remains, approximately **$15/month**. No GPU is running.
  [Closeout](docs/notebook_outputs/raw/pmm_replay_diagnostic_20260928/verified_closeout.json)
  records **9.3061 cumulative VM hours / $8.1815 running gross**, excluding
  additional retained-storage charges.
- **Next boundary:** 36 remaining fold fits forecast **17.37 more VM hours**.
  The **30-hour/$34 total-ceiling question is pending**; existing session/day
  caps are unchanged. Before broader execution, explicitly integrate the
  versioned replay policy without rewriting historical identities. The frozen
  runner still rejects GVP6 under its old gate; do not retry it until a chance
  pass or present the supplemental report as a completed 45-fit campaign.
  No final refit, held-out access or full-grid assessment occurred.

```

## pmm-007

Source: `3c0f80c:EXPERIMENT_STATUS.md:215-258`.

```text
**2026-09-27 19:41 UTC — allocation closed; 9/9 trained, 8/9 certified:**
- The two remaining six-class fold-0 configurations completed 50 epochs under
  the bounded continuation of the same
  [final plan](docs/plans/metal_level_metal_task_compared_PMM_final_plan.md).
  Seven previous fits and all preparation were reused. Source, fold 0, seed 42,
  native checkpoint selection and the validation-only protocol remain fixed.
  The report separates training on four or six classes from common-four
  evaluation (Mn, Cu, Zn, Class VIII = Fe+Co+Ni); no five-class arm was run.
- Six-class late fusion selected epoch **32** and passed independent replay:
  common-four BA **88.8124%**, macro-F1 **85.3215%**, native six-class BA
  **72.0131%**. Compared with direct-four late fusion, common-four BA decreased
  **0.6529 points**, while macro-F1 increased **4.5932 points**. Native Fe/Co/Ni
  recalls are **85.8639% / 22.8571% / 51.2821%**. These are Grade-5 exploratory
  tradeoffs, with no promotion. Its fit/replay took **1,528 seconds**; all 71
  backup files verified and the acknowledgment was uploaded before shutdown.
- Six-class Only-GVP completed all 50
  epochs, but independent replay failed its absolute `1e-6` probability check
  twice, including one replay-only retry. All 1,492 class predictions and
  metadata fields match; maximum probability differences were `3.51e-6` and
  `2.20e-6`. Both failed terminal backups were verified and acknowledged.
  No retraining, source edit or tolerance relaxation occurred. This result is
  **trained but uncertified**, excluded from comparisons; see
  [TECH-023](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-023--full-fit-gvp-independent-replay-exceeds-the-frozen-probability-tolerance).
- **VM confirmed TERMINATED at 19:41:10 UTC / 22:41:10 Israel.**
  `deepmzyme-l4@us-central1-c`, immutable instance `3144565200755786222`,
  stopped at **19:40:52 UTC**, before its 20:38:52 deadline. Session
  `session-20260927T181200Z-259157df` used **5,333 seconds / $1.3023 estimated
  running gross**. No GPU worker remains. One 150-GB disk is retained at
  approximately **$15/month**; ordinary closeout deleted nothing. Cumulative
  campaign running use is **8.9253 hours / $7.8468 estimated gross**, excluding
  additional retained-storage charges.
- **Next:** resolve TECH-023 with a versioned, validated replay/determinism
  policy before certifying GVP6 or expanding the campaign. Preserve the fit;
  do not retrain it or repeat replay until a chance pass. Full grid: **9/45
  trained, 8/45 certified**, leaving **36 untrained fits and one certification
  issue**. The 30-hour/$34 full-grid proposal remains unactivated; controller
  4h/$6 session and 6h/$10 daily caps are unchanged. No full-grid assessment,
  promotion, final refit or test access occurred.
- Evidence: [screen report](docs/notebook_outputs/raw/pmm_ion_v2_context_fold0_20260927/runtime/screen_report.md),
  [completion execution](docs/notebook_outputs/raw/pmm_ion_v2_context_fold0_20260927/runtime/completion_execution.json),
  [verified closeout](docs/notebook_outputs/raw/pmm_ion_v2_context_fold0_20260927/runtime/completion_verified_closeout.json).
  Canonical handoff: campaign `runtime/RESUME_STATE.md`. One coordinator owned
  lifecycle; a focused read-only reviewer audited the replay failure.

```

## pmm-008

Source: `3c0f80c:EXPERIMENT_STATUS.md:259-309`.

```text
**2026-09-27 PMM fold-0 continuation — allocation closed; 7/9 configurations verified:**
- Continues the same [final plan](docs/plans/metal_level_metal_task_compared_PMM_final_plan.md)
  and [playbook queue](docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md#nine-configuration-fold-0-screen-continuation).
  Five additional 50-epoch fits completed; the earlier ESMC pair, embeddings,
  feature certification, smokes and PMM folds were reused. The full grid is
  **7/45 complete**, with no promotion, final refit or held-out access.
- All seven selected checkpoints and independent replays validate on the same
  **1,492 ions / 1,346 parent pockets / 195 PDB groups**, frozen fold 0, seed 42.
  Each new fit has a verified 71-file host backup and uploaded acknowledgment.
  Source remains `adc95c42261448dc9d35572a138a3b8349de79124d9708618f511c27a848dd23`.
  See the [screen report](docs/notebook_outputs/raw/pmm_ion_v2_context_fold0_20260927/runtime/screen_report.md)
  for checkpoint-bound scores and the CSV/JSON for native metrics and recalls.
- Direct-four balanced accuracy, ordinary → first-shell-aware: **ESMC
  88.3945% → 88.3945%; GVP 86.0232% → 84.9441%; late fusion
  89.4653% → 88.7277%**. None improves the primary metric on this fold.
  GVP macro-F1 nevertheless rises **77.2357% → 81.9963%**, with Class VIII
  recall **64.6067% → 76.4045%**. These are Grade-5 single-fold tradeoffs,
  not equivalence, superiority, or evidence against binding information generally.
- Six-class ESMC selected epoch **33**, with common-four BA **85.8393%**,
  macro-F1 **79.8831%**, and native six-class BA **72.2507%**. Native Fe/Co/Ni
  recalls are **79.8429% / 34.2857% / 58.9744%**. The common-four BA is
  2.5552 points below direct-four ESMC. Fit/replay took **2,487 seconds**;
  its 71-file backup verified at **17:39:59 UTC**.
- **VM confirmed TERMINATED at 17:41:59 UTC / 20:41:59 Israel.**
  Session `session-20260927T143239Z-d9c4d944` ran from **14:32:39 to 17:41:40 UTC**
  on `deepmzyme-l4@us-central1-c`, immutable instance `3144565200755786222`:
  **3h 09m 01s**, estimated running gross **$2.7697**. No compute is running.
  One 150-GB persistent disk remains at approximately **$15/month**. Normal
  closeout deleted nothing. The provider deadline had been 18:29:32 UTC.
- **Next missing unit:** `only_gvp__six_class__none__fold0__seed42`, then
  `gvp_late_fusion__six_class__none__fold0__seed42`. The next cold GVP fit's
  3,600-second forecast needs **5,400 seconds** with 1.25 margin and 900-second
  reserve; only **2,973 seconds** remained at the last host acknowledgment.
  It was not submitted. Resume exact missing selectors after the budget/start
  decision; do not rerun completed units or broaden to full-grid assessment yet.
- The larger-budget question remains **unanswered**; the proposed cumulative
  **30 VM hours/$34** ceiling has not been activated. Measured campaign running
  use is **7.4439 hours / $6.5445 estimated gross**, excluding additional
  retained-storage charges. Updated forecast for the remaining 38 grid fits is
  **18.38 VM hours**, based on measured family timings and stated allowances;
  it excludes separately gated final refits/test work. Existing 4h/$6 session
  and 6h/$10 daily caps remain unchanged. See the
  [verified closeout and forecast](docs/notebook_outputs/raw/pmm_ion_v2_context_fold0_20260927/runtime/verified_closeout.json).
- One coordinator owned lifecycle/submission/shutdown; a focused read-only
  reviewer audited the reporting contracts. Exact unit selectors and unique
  status tags preserve old results and mitigate TECH-022. The
  [GPU review](docs/notebook_outputs/raw/pmm_ion_v2_context_fold0_20260927/runtime/gpu_execution_review.md)
  records cold versus warm preparation costs. Canonical artifacts remain at
  `/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v2_context`;
  current resume notes are in its `runtime/RESUME_STATE.md`.

```

## pmm-009

Source: `3c0f80c:EXPERIMENT_STATUS.md:310-352`.

```text
**2026-09-27 PMM ion-level campaign (`pmm_ion_metal_v2_context`) — exploratory ESMC pair and GPU recovery closeout complete:**
- Executes the [metal-level PMM plan](docs/plans/metal_level_metal_task_compared_PMM_final_plan.md)
  and its [ESMC-pair amendment](docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md#immediate-exploratory-screen-only-esm-binding-awareness-pair).
  The broader 45-fit matrix remains deferred; no final refit, promotion or test access occurred.
- Frozen training-only cohort: **7,398 ions / 3,992 PDB groups**, five class-complete
  PDB-grouped folds. Excluded source rows: 9 missing, 10 non-single-metal residues,
  503 missing explicit protein-symmetry context. Canonical artifacts:
  `/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v2_context`.
  Historical v1 evidence is preserved separately.
- Completed preparation is reusable: five PMM CPU folds (mean common-four BA
  **70.2885%**, pooled OOF **70.0766%**), all **7,664 ESMC-600M payloads**,
  full feature certification/input backup and nine GPU smoke/replay cases.
  Engineering evidence: 114 campaign regressions and 23 cache tests passed.
- **Both 50-epoch ESMC fits are complete:** ordinary and `first_shell_bias`,
  direct-four, frozen fold 0, model seed 42. Both select epoch **36**, with
  validation BA **88.3945%**, macro-F1 **84.2951%**, accuracy **84.7185%** and
  identical class recalls. All **1,492 ion class predictions agree**; probabilities
  differ for 1,484 ions. Validation contains **1,346 parent pockets / 195 PDB groups**.
  This is **Grade 5 single-fold evidence**, with no selected-metric gain and no
  equivalence or superiority conclusion. See the [paired report and figure](docs/notebook_outputs/raw/pmm_ion_v2_context_20260926/runtime/esm_binding_screen.md).
- The later `us-central1-c` session on **2026-09-26 04:08:43–04:44:30 UTC**
  completed the missing fit and replay in **1,025 seconds** and verified its
  71-file host backup at **04:43:01 UTC**. This supersedes the earlier admission
  refusal/status note. The local disk was remounted and results reverified on
  2026-09-27 without starting a GPU or repeating training. All immutable baseline
  artifacts also verify; its old manifest's only mismatch is the subsequently
  replaced aggregate `run_status.json` ([TECH-022](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-022--historical-host-manifests-include-mutable-campaign-status)).
- **GPU recovery finalized on 2026-09-27:** exact superseded source VM/disk and
  temporary snapshot, including its recycle-bin copy, were removed after the
  fit/replay/backup gate passed. The restored replacement remains **TERMINATED**.
  One 150-GB disk remains, estimated **$15/month**; no GPU compute is running.
  Recovery plus screen consumed **42m59s / $0.6298 estimated running gross**,
  excluding separately accounted persistent storage. No new allocation was made
  in this resumption. The [verified closeout receipt](docs/notebook_outputs/raw/pmm_ion_v2_context_20260926/runtime/esm_binding_screen_verified_closeout.json)
  binds the later session, results, backup and cleanup.
- **2/45 neural grid fits are complete; 43 remain**, plus later assessment/refits.
  The 30-hour/$34 full-grid proposal remains unapproved. A later binding-awareness
  confirmation must compare both arms on remaining frozen folds 1–4 and disclose
  fold 0's screening role. The primary final-test route remains unresolved;
  PMM's possibly overlapping reference route is secondary, not a pristine test.
  The [evidence summary](docs/notebook_outputs/summaries/summary_pmm_ion_v2_context_20260926.md)
  preserves the historical capacity failures and source/input provenance.

```
