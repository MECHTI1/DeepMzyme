# DeepMzyme Current Experiment Status

This is the sole concise answer to: **Where am I now, and what should I do
next?** It is mutable. Scientific policy is in [`Plan.md`](Plan.md); exact
experiment history is in the [experiment index](docs/notebook_outputs/README.md).

Last historical experiment-evidence audit: 2026-08-20. Last execution audit: 2026-09-15.
Last scientific-policy documentation update: 2026-09-28. Last status entry: 2026-09-28.

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

**2026-09-24 ion-unit correction & Zenodo exact benchmark (RELAUNCHED after total run loss):**
- **Contract & Architecture:** `--metal-example-unit ion` is fully implemented for standalone metal training in CLI, runner scripts, and Colab notebook, ensuring multi-nuclear sites (e.g. `1a0e` 3-Zn center) receive independent coordinates and residue microenvironments while grouping sibling ions under parent pocket IDs.
- **Dataset Fidelity:** Reconstructed from published Zenodo source rows (`classmodel_train_set.csv` and `classmodel_test_set.csv`) at **99.89% exact fidelity** (9,398 / 9,408 source rows: 7,911 train sites across 6,443 PDBs, 1,487 test sites across 1,281 PDBs).
- **Hugging Face Hosting:** Published permanently to [`GMBioinformatics/DeepMzyme`](https://huggingface.co/datasets/GMBioinformatics/DeepMzyme) (`train_and_test_sets_structures_zenodo_pmm_exact.tar.gz`, SHA-256: `24903f120462aacb90b43c4af97f4f08d61f1847d12636606fcc66e6351db296`).
- **Automated Verification:** Verified via `scripts/verify_zenodo_pmm_ion_dataset.py` (passes 100% with 0 errors).
- **1-Command Reproducibility:** Automated via `scripts/reproduce_zenodo_pmm_benchmark.sh` (downloads bundle if missing, verifies SHA-256, runs pre-flight tests, launches cross-validation, and compiles comparison table).
- **Colab GPU Execution (LOST - must be relaunched):**
  - Session `pmm-zenodo` (endpoint `gpu-l4-s-kkb-ass1b1-c4x86vhbzxfb`, NVIDIA L4) was
    **reaped by the Colab backend and no longer exists**. `colab sessions` reports no
    active sessions; local session state was pruned at 2026-09-24 11:48 UTC.
  - **Cause:** the workstation rebooted at 2026-09-24 11:42 UTC (14:42 local). The
    Colab CLI keep-alive daemon runs *locally*, so it died with the machine and the
    backend reclaimed the VM. The same pruning had already happened once earlier to
    the first `pmm-zenodo` VM (created 09:20 UTC, pruned 10:21 UTC).
  - **Result: zero training artifacts survive.** Fold 0 of
    `benchmark_enhanced_only_gvp` (started 10:40:27 UTC) had produced only
    `prepare_status.json` in its run directory. At the last successful telemetry poll
    (10:51:48 UTC, 11m12s of CPU time) `nvidia-smi` reported **0% GPU utilisation and
    3 MiB of 23,034 MiB VRAM in use**, i.e. the job was still in the single-threaded
    CPU graph-construction phase and **had not begun epoch 1**. No checkpoint, no
    `val_metrics.csv`, no `test_report.json` was ever written, and nothing was
    downloaded off the VM before it was reclaimed.
  - **No published benchmark numbers exist for this dataset yet.** Any earlier note
    giving an expected completion time (~11:24 UTC / 14:24 local) is superseded.
  - **Before relaunching**, see the durability gaps recorded in
    [`docs/agents_report/HANDOFF_ZENODO_PMM_EXACT_BENCHMARK.md`](docs/agents_report/HANDOFF_ZENODO_PMM_EXACT_BENCHMARK.md)
    (section "Post-mortem"): artifacts are written only to ephemeral VM disk, the best
    checkpoint is held in memory until the run ends, and the ~11-minute graph
    construction phase emits no progress output and is repeated for every fold.
- **Relaunch in flight (session `pmm-zenodo-v2`, started 2026-09-24 12:05 UTC / 15:05 local):**
  - Fold 0 of `benchmark_enhanced_only_gvp`, 50 epochs, lr=3e-4, raw RBF, seed 42,
    now launched with **`--save-epoch-checkpoints`** so an interrupted run keeps its weights.
  - Artifacts are mirrored off the VM every 5 minutes by
    `scripts/colab_artifact_streamer.py` into `~/zenodo_pmm_artifacts`, so a VM reclaim
    costs at most one poll interval instead of the whole fold.
  - **Measured** structure-parsing throughput: **3.0 structures/s**, i.e. **~36 minutes**
    to parse the 6,443 training structures. The previously documented "~9-11 minutes"
    for this phase was an estimate and is wrong by roughly 3x; budget fold timings
    accordingly (~36 min parse + ~32 min train + test parse/eval).
  - Two runner defects found and fixed before relaunching (see below), either of which
    would have produced an empty comparison table even from a fully successful run.

  - Master Guide: [`docs/ZENODO_PINMYMETAL_EXACT_ION_LEVEL_REPRODUCIBILITY.md`](docs/ZENODO_PINMYMETAL_EXACT_ION_LEVEL_REPRODUCIBILITY.md)

## Current objective

**Exact PinMyMetal 5-Fold Cross-Validation Benchmark COMPLETED (2026-09-23):** Full replication of the Nature Communications (2025) PinMyMetal 5-fold cross-validation protocol (`--train-val-split-by pocket_id --n-folds 5`) across three comparative architectures: Sequence-Only ESM-C (`benchmark_only_esm`), Structure-Only Enhanced GVP (`benchmark_enhanced_only_gvp`), and Multimodal Late Fusion (`benchmark_enhanced_gvp_esmc`).
- **Table 1 (5-Fold CV Val Bal Acc vs PinMyMetal Fig 2a: 75.08%)**:
  - `benchmark_only_esm`: **80.29% ± 3.26%** (+5.21 pp)
  - `benchmark_enhanced_only_gvp`: **74.42% ± 2.39%** (-0.66 pp)
  - `benchmark_enhanced_gvp_esmc`: **80.22% ± 3.31%** (+5.14 pp)
- **Table 2 (Held-Out Test Set 352 Pockets vs PinMyMetal Fig 2b: 67.85% & Metal3D Fig 2c: 61.70%)**:
  - `benchmark_only_esm` 5-Fold Ensemble: **77.37%** (+9.52 pp vs PMM, +15.67 pp vs Metal3D)
  - `benchmark_enhanced_only_gvp` 5-Fold Ensemble: **78.03%** (+10.18 pp vs PMM, +16.33 pp vs Metal3D)
  - `benchmark_enhanced_gvp_esmc` 5-Fold Ensemble: **79.92%** (+12.07 pp vs PMM, +18.22 pp vs Metal3D)
  - Multimodal per-class recalls: Cu: **89.7%** (+30.3 pp vs PMM 59.4%), Zn: **88.0%** (+22.1 pp vs PMM 65.9%), Group VIII: **74.6%** (+17.1 pp vs PMM 57.5%).
Complete documentation is available in [`docs/notebook_outputs/summaries/summary_pinmymetal_5fold_three_models_20260923.md`](docs/notebook_outputs/summaries/summary_pinmymetal_5fold_three_models_20260923.md) and [`docs/EXACT_PINMYMETAL_5FOLD_CV_REPRODUCIBILITY.md`](docs/EXACT_PINMYMETAL_5FOLD_CV_REPRODUCIBILITY.md).

**Next campaign implementation (2026-09-16):** the separately named
[`metal_single_gpu_20h_v2`](docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md#single-gpu-metal-campaign)
adds bounded learning-rate/capacity screening, an initial hybrid arm, one
mixed diagnostic block per family, bounded paired numeric continuation, and
protected grouped-fold confirmation. Both top screened recipes receive a
second seed; the fixed larger late-fusion pair and historical-recipe late-five
challenger retain separate identities. The revised grid is a ceiling, not a
promise that every fit fits the original allocation caps.
Its dedicated serial CLI preserves prior pilot evidence and studies. This is
an implementation/preparation change: **no new GPU allocation, training fit,
hardware readiness result or model promotion is implied**. Exact budgets,
commands and parameter values belong to the playbook. The next runtime action
is measured readiness and cost admission on the actually assigned single GPU,
after freezing inputs and durable storage. Confirmation remains exploratory
until the primary final-test route is resolved.

**Local implementation checks (2026-09-16):** 229 campaign and retained-pilot
regression tests passed. The v2 local plan is at
`DeepMzyme_Data/notebook_outputs/plans/metal_single_gpu_20h_v2_prepared/`, with 65 initial
commands and the full budget preview. CPU preparation passed for **1,389**
eligible pockets and **5,216** cached input files; all six native classes are
present in every declared training/validation fold. Its historical proxy is **21.540 hours**
for complete coverage, including the training margin and four operations hours,
before additional large-capacity cost or optional tuning. Required discovery
alone projects to **6.712 hours** against its six-hour cap. Full training is
therefore **not admitted**; zero historical reuse cells are certified. The
cost-only confirmation order is core, fixed late-five, early/hybrid, then RING.
No new GPU session or training fit was launched; a fresh server check found no
active Colab sessions. The old closed allocation identities remain unchanged.
Earlier unexecuted local previews are retained. The prepared directory binds
the final source and documentation and is the execution-plan source. No local
preview directory contains a training attempt.

**Authorized RING continuation completed (2026-09-15):** all four smokes
and sixteen full 50-epoch fits in the
[bounded matched comparison](docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md#bounded-matched-ring-continuation--stage-2b)
are verified locally and in Drive. Both complete family blocks passed their
budget gates. Terminal checkpoint/configuration/normalization binding passed;
the owned G4 session `deepmzyme-metal-ring-20260915` stopped at
**16:20:52.944619 UTC**, and Colab reported no active sessions.
The original cumulative allocation is **7.312911 hours**, leaving
**2.687089 hours of the original ten-hour cap**. The two prior closed
allocations remain unchanged; the budget was not reset.

Only-GVP's mean RING-on-minus-off BA differences are **+1.426 pp** at
`3e-5` and **+0.387 pp** at `1e-4`; all four LR/seed differences are positive,
but mean Class VIII recall falls by **3.125 / 4.688 pp** respectively.
All four graph-level late-fusion pairs have identical selected BA and class
recalls, not proven identical models or predictions. This is **Grade 3** on
one shared validation split and two training seeds, with no promotion.
The audit found annotations on existing edges, no topology expansion and
matching normalization within all trained pairs. The existing local CPU
audit was not restarted; frozen source, previous pilots and Phase-3 analysis
were preserved. No held-out evaluation or auxiliary training was added.
See the [completed RING summary](docs/notebook_outputs/summaries/summary_metal_ring_pilot_20260915.md),
[portable evidence](docs/notebook_outputs/raw/metal_ring_pilot_20260915/README.md)
and [actual stop receipt](docs/notebook_outputs/raw/metal_ring_pilot_20260915/host_closeout_allocation3/session_stopped.json).

**Phase 3 completed locally (2026-09-15):** separate retained-training
PinMyMetal and CARE panels now have native-six/common-four counts, conditional
probabilities, protein-group-weighted measures and 9,999 whole-group
permutations. PinMyMetal contributes 1,163 usable pairs from 1,136 groups;
CARE contributes 983 common-four pairs from 743 groups, or 982 native-six
pairs from 742 groups. Common-four group-weighted Cramér's V is 0.32667 and
0.50600 respectively, with no finite-sample bias correction. These are
descriptive within-source associations, not evidence of auxiliary-learning
benefit. No validation statistics, external test inputs or model training
entered this analysis. Cross-source identity/homology and shared-training
holdout certification remain required. See the
[association summary](docs/notebook_outputs/summaries/summary_metal_ec1_association_20260915.md)
and [verified portable evidence](docs/notebook_outputs/raw/metal_ec1_association_20260915/README.md).

**Authorized metal pilots completed (2026-09-15):** all 30 original architecture
fits and 15 geometry fits, each 50 epochs, plus 12 model smokes are verified
locally and in Drive. Both queues passed genuine terminal-state verification;
all 15 geometry prediction exports are verified. The owned G4 session
`deepmzyme-metal-geometry-20260915` stopped at **11:37:34.517659 UTC**, and
the server reported no active sessions. Total allocation across both sessions
was **5.350957 hours of the original ten-hour cap**, including recovery,
setup, transfer and analysis.

In the original architecture pilot, selected-LR direct-four means ± sample SD
across training seeds 42/43 are
Only-ESM **74.342 ± 3.138%**, late fusion **72.437 ± 0.177%**, Only-GVP
**72.073 ± 1.663%**, and early fusion **65.926 ± 4.866%**. These are Grade-3
results on the same validation data, with no promotion. Late-five's
common-four mean is **74.718 ± 2.486%**, an exploratory target challenger
with class tradeoffs; native and common-four results come from each
native-selected checkpoint. See the [completed continuation report](docs/notebook_outputs/summaries/summary_metal_architecture_pilot_continuation_20260915.md)
and [verified stop receipt](docs/notebook_outputs/raw/metal_architecture_pilot_20260915/continuation/closeout_allocation2/post_stop/session_stopped.json).

All geometry arms selected LR `1e-4`; increasing LR improved seed-42 BA by
7.6–10.4 percentage points. Selected-LR means across training seeds 42/43 are
A 69.641%, B 70.186%, C 70.007%, D 68.865%, and E 68.030%. B−A and C−B
advantages reverse between seeds; D−B and E−D are negative in both high-LR
seeds. This is Grade-3 repetition on the **same validation data**, with no
architecture promotion or universal rejection. E's worst Zn recall is
15.625% (5/32), illustrating the class tradeoffs hidden by aggregate BA.
The fresh A control masks only the added coordination-count/angle slots: it
retains the original GVP geometry and four base metal-site statistics
(multinuclear flag, metal count, minimum/mean intermetal distances). Its
matched explicit machinery differs from the original legacy GVP baseline.
See the [completed geometry summary](docs/notebook_outputs/summaries/summary_metal_coordination_geometry_pilot_20260915.md)
and its verified prediction/paired-error evidence.

Fitted-normalization hashes match within A/B/C and within D/E, as required.
Adding metal nodes changes representation, connectivity and fitted edge
normalization together; those contrasts do not isolate topology alone.

Recovery verified the original frozen source and manifest, all 1,389 retained
pockets (1,181 train / 208 validation), unchanged feature contents and fresh
cache timestamps. Seven original smokes and all eight A1/A2 results are now
verified, including completed late-fusion linked retry `attempt_013`.
The original interrupted `attempt_012` is reconciled as incomplete; its exact
attempt duration is unknown and was not invented. The full first allocation
of 3,827.969 seconds remains charged to the shared 600-minute cap, together
with the second allocation and recovery. The prior two-allocation closed
total was 19,263.446 seconds, leaving 16,736.554 seconds before the RING
continuation; the current three-allocation total is reported above.
The geometry implementation and recovery controls passed 134 focused tests
in 30.37 seconds; 11 separate owned-teardown tests also passed. Legacy model
outputs remain bitwise unchanged in the checked compatibility comparison.
These are implementation/readiness checks, not geometry-model results.
The largest A1/A2 balanced accuracies are late fusion **0.723121** and
Only-ESM **0.721228**, a gap of only **0.001893**. These are experimentally
evaluated single-seed results, with no established architecture winner or
promotion. Hybrid is **deferred under the budget scheduling gate**, not
rejected: best early fusion did not exceed best Only-GVP by the required
margin. The bounded target/seed matrix is complete; full hybrid comparison
and grouped-fold/paired-CI confirmation, including RING, remain absent.
The separately completed bounded RING pilot does not fill those gates.
No held-out evaluation occurred;
historical multi-seed anchors remain separately labeled.
See the [first-allocation summary](docs/notebook_outputs/summaries/summary_metal_architecture_pilot_20260915.md)
and [geometry recipe](docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md#bounded-coordination-geometry-pilot--stage-2b).

**Completed EC reference (2026-09-14):** the v12 EC1 standalone campaign
