# Exact-pocket GVP/fusion L4 diagnostic: execution ledger

Study: `gvp_fusion_exact_pocket_l4_v2`. This is the validation-only continuation of the 2026-09-23 Antigravity exact-PinMyMetal five-fold benchmark. The executable recipe and decision gates are in the [metal playbook](../METAL_TRAINING_PIPELINE_PLAYBOOK.md#exact-pocket-antigravity-matched-l4-diagnostic). The [stopped v1 ledger](GVP_FUSION_L4_EXECUTION_20260924.md) is historical evidence only. Do not combine v1 smokes with this study.

## 2026-09-23 22:33 UTC — pre-allocation state

- Shared project HEAD: `9a069d226caacbc43c12d0d53fa49ed714c8b263`. The checkout was already dirty when this work began. The pre-existing staged patch fingerprint remains `accc647da5d687f088591407ec734981b61b0948542bfb4173a268c0abb3f8b0`. Shared `src/training/config.py` and `src/training/run.py` match HEAD. Existing staged files and PC background processes were untouched.
- All new experimental source is confined to ignored `DeepMzyme_Data/notebook_outputs/plans/gvp_fusion_exact_pocket_l4_v2/source_checkout/`. It adds selected validation-prediction export and a configurable GVP optimizer scope there; these changes are not promoted into the shared checkout.
- Corrected protocol: v12 bundle SHA256 `90c0899829e0ac5ca94a5ef34484b74ca1014d3e9b6b1fba90bd338ee3440dee`, exact PinMyMetal **train** cohort, native `five_class`, `metal_eligibility_scheme=active`, five `pocket_id` folds, default `active_targets` stratification, model/effective split seed 42, omitted split-seed flag. Historical train/validation ordered membership hashes, counts, and PDB overlaps for all five folds are frozen in the runner and playbook. PDB IDs may legitimately occur on both sides, exactly as in the prior benchmark.
- The runner has a `verify-folds` gate that reloads the exact training cohort and checks all five ordered membership hashes, counts, and PDB overlap counts before any fit. `smoke` and `run` refuse to proceed without its source/bundle-bound receipt. The 20 frozen commands contain no test evaluation or test path. A run's own split artifacts are checked against the same historical fold references before it is counted complete.
- Isolated source snapshot: `DeepMzyme_Data/notebook_outputs/plans/gvp_fusion_exact_pocket_l4_v2/source_snapshot.tar.gz`, SHA256 `577cd6a4e43afc66811e0f691e037737d4a8e104c5903961957706141a4c9d1d`. Manifest SHA256 `7bac42a8a63a829694c1bf89af5a2b658dc1e4b9a8a116c4f5c002c6c1fda9cd`; command count 20. Lightweight syntax check passed. Local full Torch test was deferred because host free RAM and swap are tight; use the isolated Colab VM for it.
- At this checkpoint, no v2 Colab session or fit is active. No held-out test was accessed by v2.

## Recovery rule

Check the latest ledger entry, `colab sessions`, the named session status if one is recorded, `study_manifest.json`, `fold_preflight.json`, and locally verified artifacts before acting. Never relaunch an incomplete run silently, reuse a v1 command, switch folds, or touch an unowned session. A named v2 session must have its own stop guard and be stopped at closeout. Keep partial work and record the reason for any gap.

## 2026-09-23 22:34–22:39 UTC — first owned L4 session and readiness

- `colab sessions` showed no active sessions. Created only `gvp-exact-l4-v2-20260924a`; `colab status` verified **Hardware: L4**, endpoint `gpu-l4-s-kkb-ass1a1-1bqd6xw3n9q07`. A study-owned `deepmzyme-gvp-exact-l4-v2-stop-20260924.timer` is scheduled to stop only that session after 3h45m, leaving 15m of the four-hour ceiling for transfer and teardown. No pre-existing PC process or other Colab session was touched.
- Uploaded and SHA-checked the frozen source archive. `/content/DeepMzyme` contains the isolated source and `/content/extracted_source_sha256.txt` binds it to the archive. Stock PyTorch `2.11.0+cu130` sees CUDA on NVIDIA L4 capability 8.9. Installed only `requirements/colab-overlay.txt` through the Colab CLI; five isolated runner tests passed there.
- Downloaded the published v12 bundle to `/content/bundle_v12.tar.gz` (4,731,180,661 bytes); remote SHA256 equals the frozen `90c089...44de` checksum. Selective extraction produced only exact-train (6 archive members), structure store (6,679), ESM cache (5,893), and updated external features (5,894). The exact-split test directory is absent. Remote receipt: `/content/gvp_fusion_exact_pocket_l4_v2/extraction_receipt.json`.
- Detached fold preflight is running as remote PID `23843`, log `/content/gvp_fusion_exact_pocket_l4_v2/fold_preflight.log`. It must produce `fold_preflight.json` with all five historical ordered hashes before any fit. No v2 fit has begun.

## 2026-09-23 22:43 UTC — exact fold gate passed

- Fold preflight finished and produced the remote receipt, copied to `DeepMzyme_Data/notebook_outputs/plans/gvp_fusion_exact_pocket_l4_v2/fold_preflight.json` (SHA256 `f476557b3e2330448ff8ffce9b0f0008b4c0e7087ae26cdf5a062c8e4005239c`). It loaded 1,597 pockets and matched all five historical ordered train/validation SHA256 identities, train/validation counts, and PDB overlap counts: 58, 56, 43, 49, 45. This is deliberately a **pocket-level** split; regrouping by PDB ID would fail the gate. The exact cohort still has its known train/test PDB-ID overlap and remains a secondary benchmark.
- Only after that gate passed, F0 fold-0 one-epoch smoke began on the L4 as remote control PID `25013`; control log `/content/gvp_fusion_exact_pocket_l4_v2/smoke_F0_control.log`. No full 50-epoch fit has started. Smoke metrics will not determine selection.

## 2026-09-23 22:53 UTC — first smoke preserved

- F0 fold-0 one-epoch smoke completed under the frozen command. Its trainer reported 1,272/325 train/validation pockets and 58 shared PDB IDs; the runner accepted its own split artifacts, selected checkpoint, and aligned validation predictions. This is a plumbing check, not comparison evidence.
- Complete run archive copied to `DeepMzyme_Data/notebook_outputs/plans/gvp_fusion_exact_pocket_l4_v2/artifacts/gvp_exact_l4_v2_F0_fold0_smoke.tar.gz`; remote and local SHA256 both `994d774d03e318be26fda20940a48469313f071d6bd29303b7f3d8d8484f9245` (17,825,070 bytes).
- F1 fold-0 one-epoch smoke then started as remote control PID `27306`, log `/content/gvp_fusion_exact_pocket_l4_v2/smoke_F1_control.log`. No full fit has started.

## 2026-09-23 — second smoke preserved

- F1 fold-0 one-epoch smoke completed with the same frozen cohort/fold and its runner accepted the required artifacts. Complete archive copied locally to `DeepMzyme_Data/notebook_outputs/plans/gvp_fusion_exact_pocket_l4_v2/artifacts/gvp_exact_l4_v2_F1_fold0_smoke.tar.gz`; remote/local SHA256 `bdfda82b99d5763cb014357228d9b4e04c92d15a3e9bcbfde41a777b6063b5a4` (17,824,563 bytes).
- The Colab CLI twice reported `RuntimeError: Connection was lost` during read-only monitoring while the named L4 still appeared in `colab status` and the original F1 process/files progressed. No duplicate run or recovery session was launched. `colab ls` and then the completed control log confirmed F1 success.
- G0 fold-0 one-epoch smoke started only after F1 archive verification, remote control PID `29269`, log `/content/gvp_fusion_exact_pocket_l4_v2/smoke_G0_control.log`. No full fit has started.

## 2026-09-23 — third smoke preserved

- G0 fold-0 one-epoch smoke completed; the runner accepted its historical split and required artifacts. Complete archive copied to `DeepMzyme_Data/notebook_outputs/plans/gvp_fusion_exact_pocket_l4_v2/artifacts/gvp_exact_l4_v2_G0_fold0_smoke.tar.gz`, remote/local SHA256 `f207cefaddbc836e7092af6ccf03bf60089243c1f864419fad4fc57322d81301` (15,045,787 bytes).
- G1 fold-0 one-epoch smoke started only after that transfer, remote control PID `31039`, log `/content/gvp_fusion_exact_pocket_l4_v2/smoke_G1_control.log`. No full fit has started.

## 2026-09-23 23:15 UTC — all smoke gates passed

- G1 fold-0 one-epoch smoke completed and passed the runner's required artifact check. Complete archive copied to `DeepMzyme_Data/notebook_outputs/plans/gvp_fusion_exact_pocket_l4_v2/artifacts/gvp_exact_l4_v2_G1_fold0_smoke.tar.gz`, remote/local SHA256 `c32fafd81ff94675eb1f26ab62e83725d5f590d4cd022357dd201d3d8346c08b` (14,977,898 bytes).
- All four frozen-arm fold-0 smokes now exist locally. They are preparation checks only, not model-selection evidence. All used the historical pocket-level fold-0 membership and validation-only commands; the test directory remained unextracted.
- F0 artifact timestamps show data/graph preparation consumed most of the smoke time, while epoch 1 finished seconds after preparation. Therefore no full-grid runtime estimate is being extrapolated by multiplying the entire smoke duration by 50. The next 50-epoch F0 fold-0 baseline will be a timing measurement; no paired result will be claimed from a single completed arm.

## 2026-09-23 23:15 UTC — first full-fit timing measurement

- Started the frozen 50-epoch F0 fold-0 run only after all four smokes and transfers passed. Remote control PID `32934`, log `/content/gvp_fusion_exact_pocket_l4_v2/full_F0_fold0_control.log`, run directory `/content/runs/gvp_fusion_exact_pocket_l4_v2/gvp_exact_l4_v2_F0_fold0`. It is a matched-baseline candidate and a timing measurement; no winner inference is allowed from it alone.

## 2026-09-23 23:29 UTC — first full fit verified; cost gate

- F0 fold 0 completed all 50 epochs. Remote run timestamps `2026-09-23T23:16:10+00:00` to `23:28:31+00:00` give 741 seconds (12m21s), including preparation. Validation selected epoch 44 by native five-class balanced accuracy; the score is not a paired result yet.
- Complete archive copied to `DeepMzyme_Data/notebook_outputs/plans/gvp_fusion_exact_pocket_l4_v2/artifacts/gvp_exact_l4_v2_F0_fold0.tar.gz`; remote/local SHA256 `b3477cd8a83cb5ab2c52bdd04703a196c997c14d7076de8c742cada859b38eb0` (17,902,719 bytes). The local standard-library audit verified the frozen command, 50 rows, selected epoch/metric, fold hashes/PDB overlap, validation prediction identity, and no test evaluation. `comparison.csv` currently has one full-run row; no pair inference.
- Cost admission: 19 remaining fits × 741s × 1.25 = 17,598.75s (4h53m19s), plus setup/transfer/teardown reserve. This is below the original 16-hour cumulative ceiling even after charging the stopped v1 allocation (~0.5h) and this v2 session so far (~0.9h). The present session has its independent 3h45m stop guard; stop before any fit whose measured worst-case duration would overrun its transfer reserve. The full grid may require a second owned L4 session, but it does not require changing epoch/fold settings.

## 2026-09-23 23:31–23:38 UTC — provider/kernel loss and safe closeout

- F1 fold-0 full fit started under frozen command as remote control PID `36641`. Its last verified state had **zero logged epochs** and no completed archive. At `23:35:58 UTC` the Colab CLI history recorded `session_terminated` with reason `pruned` after a kernel/proxy 404; the monitoring command reported a 404/401 session loss. This is a provider/CLI interruption, not evidence that F1 training completed or failed scientifically. Treat the attempt as **incomplete** and never use it in a score comparison. The VM was inaccessible before its run directory could be copied.
- `colab whoami` verified OAuth and the Colaboratory scope; no authentication command or credential change was made. `colab sessions` showed a `[?]` assignment with endpoint `gpu-l4-s-kkb-ass1a1-1bqd6xw3n9q07`, exactly the endpoint in this study's `session_created` history. The local CLI name had been pruned, so the study-owned stop timer's named command would no longer target it. No other assignment was present.
- To release **only the endpoint this study created**, the empty CLI sessions file was copied to `DeepMzyme_Data/notebook_outputs/plans/gvp_fusion_exact_pocket_l4_v2/cli_sessions_before_recovery.json`. A temporary named CLI record with that exact endpoint and an inert loopback URL was inserted; `colab stop -s gvp-exact-l4-v2-20260924a` then returned `Session terminated`. A fresh `colab sessions` returned **No active sessions found on server**. The study-owned `deepmzyme-gvp-exact-l4-v2-stop-20260924.timer` was stopped and verified `inactive`. No pre-existing PC process, other Colab assignment, or user file was touched.
- Conservatively charge the whole first v2 allocation from `22:34:32` to this closeout at approximately `23:38 UTC` (~1h04m), including setup, smokes, full F0, interrupted F1, and teardown. The stopped v1 allocation (~0.5h) is additional. The original cumulative 16-hour ceiling is **not reset**. No automatic replacement VM was provisioned after the 404/401. The 19 remaining full fits still require fresh admission against the updated budget and a new owned-session stop guard.
- Durable local artifacts: frozen source and 20-command manifest, five-fold preflight, selective-extraction receipt, four SHA-verified smoke archives, one SHA-verified 50-epoch F0 archive, `artifact_manifest.json`, `split_audit.json`, `comparison.csv`, `class_recall.csv`, `paired_bootstrap.json`, and `campaign_report.md`. The report explicitly has no winner inference while pairs are incomplete. F1 fold-0 may only be rerun as a new, clearly labeled attempt after verifying the old session is absent and charging its lost time; do not represent it as a resume or reuse nonexistent artifacts.

## 2026-09-24 03:08–03:14 UTC — user-directed continuation, second owned L4

- User explicitly directed continuation and requested an independent subagent check Colab connectivity at intervals no longer than five minutes. The read-only monitor checks `colab status` and `colab sessions` every four minutes, reports to the primary agent, and does not allocate, stop, or modify any session. Its first heartbeat confirmed no active session before allocation; subsequent heartbeats confirmed the new named L4.
- Rechecked shared Git HEAD `9a069d226caacbc43c12d0d53fa49ed714c8b263`, unchanged staged patch fingerprint `accc647da5d687f088591407ec734981b61b0948542bfb4173a268c0abb3f8b0`, and clean shared `src/training/config.py`/`src/training/run.py`. One full F0 archive and four smokes remained intact locally. Previous L4 was absent.
- Created only `gvp-exact-l4-v2-20260924b`; `colab status` verified **L4**, endpoint `gpu-l4-s-kkb-ass1b0-1wd6pcb3b61r5`. Armed a new study-owned `deepmzyme-gvp-exact-l4-v2-stop-20260924b.timer` for 3h45m after allocation; it targets only this session and reserves 15 minutes for transfer/teardown.
- Uploaded the unchanged source archive and 20-command manifest. Remote source SHA256 matched `577cd6a4...b38eb0`; stock PyTorch `2.11.0+cu130` saw NVIDIA L4 capability 8.9. Installed only `requirements/colab-overlay.txt`. Downloaded v12 bundle, verified its SHA256 `90c089...44de`, and selectively extracted exact-train (6), structure store (6,679), ESM cache (5,893), and updated external features (5,894) archive members. Test directory is absent; receipt `/content/gvp_fusion_exact_pocket_l4_v2/extraction_receipt_b.json`.
- Fresh five-fold preflight started as remote PID `22091`, log `/content/gvp_fusion_exact_pocket_l4_v2/fold_preflight_b.log`. It must match the historical ordered pocket hashes before F1 is retried. Old F0 full fit remains local; it is not rerun. The interrupted first-session F1 has no reusable artifact and is charged as lost time.

## 2026-09-24 03:17 UTC — session-B fold gate and F1 retry

- Fresh `verify-folds` completed on session B: all 1,597 pockets and five historical ordered memberships matched. Receipt copied locally as `fold_preflight_b.json`; `jq -S '.folds'` SHA256 is `cef2f222062a72682f768820ccf20356f758bff67b397ea8c7eed90386ed2abe` for both session A and B receipts. Session B's selective-extraction receipt is also copied locally.
- F1 fold-0 frozen 50-epoch command was launched on the new VM as **attempt B**, remote control PID `23239`, log `/content/gvp_fusion_exact_pocket_l4_v2/full_F1_fold0_control_b.log`. The lost attempt A had no local archive, and F0 was not repeated. No score comparison will be made until the complete five-fold pair is present.

## 2026-09-24 03:31 UTC — F1 fold 0 completed and preserved

- A read-only `colab exec` monitor briefly lost its console connection near the final epochs, but the named session stayed present and the detached run finished. `colab ls` showed the complete required artifact set. No duplicate attempt was launched.
- F1 fold 0 ran from `03:18:19` to `03:31:05 UTC` (766s, 12m46s), completed 50 epochs, and selected epoch 44 by native five-class balanced accuracy. Complete archive copied locally to `artifacts/gvp_exact_l4_v2_F1_fold0.tar.gz`; remote/local SHA256 `19604cea773e118342a774b9cb84f315d81f2c261981f8dc647a092c4ac72713` (17,896,875 bytes). Remote receipt copied alongside it.
- Local `analyze_archives.py` audited 2/20 full runs, including both fold-0 F0/F1; no five-fold paired inference is emitted. The independent subagent continues ≤4-minute L4 reachability checks. Session-B stop guard remains armed.

## 2026-09-24 03:33 UTC — fold 1 baseline started

- With the L4 named session still healthy, launched only the frozen F0 fold-1 50-epoch command as remote control PID `27057`, log `/content/gvp_fusion_exact_pocket_l4_v2/full_F0_fold1_control_b.log`. F1 fold 1 will not start until F0 fold 1 is complete, copied locally, SHA-verified, and audited. The session timer and independent connection monitor remain active.

## 2026-09-24 03:40 UTC — binuclear interpretation clarified during active fit

- The parser clusters metal ions connected within 4.5 Å into one pocket graph and records all member `metal_site_ids`; that binuclear cluster has one `pocket_id` and cannot be divided by a pocket-level fold. Separate clusters from the same PDB remain separate pockets and **can** cross historical train/validation folds. Under the frozen five-class labels, Co+Ni maps to one Class VIII target, while Fe+Mn maps to two target classes; the latter has no single metal label and is excluded from the metal-supervised cohort. This is not multi-label prediction. Corrected the playbook definition accordingly; the frozen data, split, command manifest, and active fit were not changed.
- Read-only Colab inspection at `03:40 UTC` confirmed F0 fold 1 was still preparing/training under remote PID `27057` with its child trainer active on the L4; the split diagnostics match historical fold 1 (1,272/325 pockets and 56 shared PDB IDs). No duplicate was launched.

## 2026-09-24 03:57–04:00 UTC — continuation recovered from the prior Codex chat

- The user clarified that the continuation source is the latest PyCharm ChatGPT/Codex chat, `01a0cfa8-e208-7803-aed9-87cd61ad0ee2` (PyCharm task `1d91a108-d8d4-4f21-9007-78590a64fa2b`), including its authorized L4 execution and independent connection monitor. Recovered its final messages and frozen study state before resuming. The earlier Claude review was background to that chat, not the execution stopping point.
- The existing named session B remained reachable on its recorded L4 endpoint. Its original stop guard remains armed for `2026-09-24 06:54:17 UTC`; no new allocation or budget reset occurred. A replacement read-only monitor checks at intervals no longer than four minutes and appends `monitor_resumed_20260924.jsonl` in the isolated study directory.
- F0 fold 1 had already completed after the prior chat stopped: `03:33:46`–`03:46:42 UTC`, 776 seconds, all 50 epochs. Recovered it without rerunning. Its archive and receipt are now local under `artifacts/`; remote/local SHA256 is `9afff725ace8a827779ca848d47c06713dd24c0da1f3e1d9d3878b9da396ef8b` (17,899,681 bytes). The existing local analyzer accepted its command, historical fold, selected epoch, and required prediction artifacts, bringing the verified count to **3/20**.
- Shared Git HEAD and the pre-existing staged patch fingerprint still match the recorded pre-study state. No shared model/training source or existing PC process was modified. The next frozen command is F1 fold 1; five-fold inference remains withheld until both complete family arms are present.

## 2026-09-24 04:01 UTC — F1 fold 1 resumed in the existing session

- Started the frozen F1 fold-1 command as remote control PID `33422`, with control log `/content/gvp_fusion_exact_pocket_l4_v2/full_F1_fold1_control_resumed.log`. Before launch, verified the source archive hash, absence of another trainer, and absence of an existing F1 fold-1 run directory. Its initial diagnostics match historical fold 1: 1,272/325 pockets and 56 shared PDB IDs.
- Completed runs remain archived and audited before each next launch. The original session-B stop guard and the cumulative 16-hour allocation ceiling still apply.

## 2026-09-24 04:09–04:17 UTC — CLI loss and ion-unit correction

- A local serial controller, kept outside the frozen source, adopted F1 fold 1 at epoch 13 and was designed to archive, checksum and audit each run before the next. At `04:09 UTC` its read-only remote call lost the WebSocket, followed by the Colab CLI message that the named session appeared lost `(404/401)`. The controller stopped. No automatic replacement session, reauthentication, or duplicate training command was issued. `colab whoami` still reported OAuth with the Colaboratory scope and a valid token, so this is not evidence that the user needs to log in again.
- `colab status -s gvp-exact-l4-v2-20260924b` now says the name is not found. `colab sessions` still lists the *same previously recorded, study-owned endpoint* `gpu-l4-s-kkb-ass1b0-1wd6pcb3b61r5` as `[?]`, L4. Its training state and F1 fold-1 artifacts are inaccessible through the named CLI; that attempt is **unverified and excluded**. The independent monitor observed this unchanged through `04:17 UTC`. The original stop guard remains scheduled for `06:54:17 UTC`, but its named stop may be unable to target an assignment whose local name was pruned. Do not claim this session is stopped until confirmed. The three locally archived 50-epoch runs remain verified; the expanded local analyzer has not yet rerun after the correction.
- The user corrected the scientific unit: keep `pocket_id` fold grouping but add a separate supervised example for every metal ion, including different metals in one clustered pocket, and add that as a configurable project option. The pocket-unit v2 study is therefore **stopped scientifically at 3/20 verified runs**; its partial fits are historical evidence, not matched controls for the new ion-unit cohort. No held-out test was accessed.
- Shared project implementation now has `--metal-example-unit ion` for standalone metal training. It assigns per-ion site labels, centers each graph on that ion, filters residues within the same 10 Å radius, and groups all sibling examples by the original parent pocket. The default remains `pocket`. Tests and documentation are tracked separately in the current checkout. A fresh ion-unit cohort/fold audit and matched controls would be required before any new GPU inference.

## 2026-09-24 04:20 UTC — local implementation closeout

- The project CLI and Colab notebook now expose the optional ion example unit for standalone metal training; the default remains the historical pocket unit. The notebook records the unit in run commands/configuration and persistent Optuna compatibility metadata. Its prior-study migration treats missing metadata as `pocket` only; an ion run requires a new incompatible study identity.
- Focused loading/splitting tests plus standalone, EC, auxiliary and explicit-membership regression suites passed: **154 tests total**. Seven touched Python source files and the changed notebook cells compiled; the notebook JSON parsed; `git diff --check` passed. No full ion-unit training, cohort count, fold recheck or held-out evaluation has been run.
- Re-ran the local archive analyzer after strengthening selected-checkpoint/prediction-order checks: the three available full pocket-unit runs still audit cleanly. The partial F1 fold-1 run remains inaccessible and excluded. The independent monitor still sees the original study-owned L4 endpoint as `[?]`; named CLI access has not returned.

## 2026-09-24 04:29 UTC — original L4 assignment cleared

- The independent read-only monitor found that the original endpoint was no longer assigned at 04:29:38 UTC; `colab sessions` reported no active sessions on the server. A fresh independent `colab sessions` check confirmed the same state. No manual disconnect is needed.
- This changes only the resource status. The scientific stop remains at 3/20 locally verified full pocket-unit runs; the partial F1 fold-1 run was not recovered and no ion-unit GPU run has been performed.
