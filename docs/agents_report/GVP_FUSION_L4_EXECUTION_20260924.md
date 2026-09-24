# GVP/fusion L4 diagnostic execution ledger

Study: `gvp_fusion_diagnostic_l4_v1`, **stopped; do not resume**. Its obsolete
design is preserved in
[`gvp_fusion_l4_v1_stopped_plan.md`](../archive/experiments/gvp_fusion_l4_v1_stopped_plan.md).
The user-requested corrected pocket-level protocol has a distinct plan in
[`METAL_TRAINING_PIPELINE_PLAYBOOK.md`](../METAL_TRAINING_PIPELINE_PLAYBOOK.md#exact-pocket-antigravity-matched-l4-diagnostic).
This file is the factual stop and artifact ledger for v1 only.

## 2026-09-23 21:43 UTC — authorized start

- User authorized execution on Colab and required non-interference with all
  background processes on the PC. No existing PC process will be stopped,
  signalled, or reprioritized.
- `colab sessions` returned `No active sessions found on server.` No Colab
  session was stopped or reallocated.
- Host memory was tight (`free -h`: 15 GiB total, 3.5 GiB available and
  3.5/4.0 GiB swap used); local checks will be lightweight and serial.
- Existing staged/untracked work is present. This study will preserve it,
  and the remote source snapshot must include the exact worktree used.
- Current state: implementation pending; no L4 allocated; no fits started;
  held-out test untouched by this study.

## Audit instructions for this stopped study

Read this ledger and the archived v1 plan to understand the stopped attempt.
Do **not** run the v1 manifest, source snapshot, or partial fit. Any new
comparison must use the separately named v2 exact-pocket protocol, with a
new source hash and artifact directory. Do not run the historical
exact-PinMyMetal benchmark runner or watchdog for that diagnostic: they
include held-out testing and session recovery behavior that the
validation-only comparison forbids.

## 2026-09-23 21:50 UTC — local implementation

- Added `--gvp-lr-scope` (default `trunk`, optional `branch`) and
  `--export-selected-val-predictions` to `src/training/config.py`. The branch
  rate moves only `init_vec_proj`, `gvp_attn_pool`, and `gvp_fusion_proj` into
  the GVP optimizer group. The exact parameter names/counts and rates are
  saved in run metadata.
- Added selected-checkpoint validation prediction export in
  `src/training/run.py`, including target/order alignment checks. It does not
  load held-out data when test evaluation is disabled.
- Added a separate validation-only runner at
  `scripts/run_gvp_fusion_l4_diagnostic.py` with frozen command generation,
  one-epoch smoke mode, serial fit execution, completion verification, status,
  and summary. It never allocates, stops, or changes a Colab session.
- Added four focused CPU unit tests. They passed with the user-specified
  `DeepMzyme` conda interpreter. Syntax checks and `git diff --check` passed.
- Next: freeze source snapshot and command manifest, then verify the requested
  L4 assignment and prepare the v12 bundle on that VM.

## 2026-09-23 21:51 UTC — frozen planning artifacts

- Local study directory:
  `DeepMzyme_Data/notebook_outputs/plans/gvp_fusion_diagnostic_l4_v1/`.
- `source_snapshot.tar.gz` includes current staged/uncommitted `src/`,
  `scripts/`, `tests/`, requirements, Plan, and applicable documentation.
  SHA256: `d6d9cd24ee6ac1ebbb74f68c6634770a1d7e03cfbc2e4b7e902384b97500389b`.
- Base Git commit: `9a069d226caacbc43c12d0d53fa49ed714c8b263`.
  The snapshot, not the commit alone, is the executable source identity.
- `study_manifest.json` and `commands.jsonl` freeze 40 commands. Manifest
  SHA256: `f52830126a32df353ffd64c06ad05f35b3af6c89575e820d3698813221e193cd`.
  A literal search found no test-path or test-evaluation options in the
  generated commands.
- No L4 had been allocated at this checkpoint.

## 2026-09-23 21:51:44 UTC — first owned Colab session

- `colab new -s gvp-l4-v1-20260924a --gpu L4` succeeded.
- `colab status -s gvp-l4-v1-20260924a` verified **Hardware: L4**,
  **Status: IDLE**, endpoint `gpu-l4-s-kkb-ass1a0-3ue3qgx9khsvq`.
- This is the only Colab session this study owns. Target mandatory stop by
  2026-09-24 01:51 UTC, with transfer/teardown starting by 01:36 UTC.
- At this checkpoint, no data were transferred and no training began.

## 2026-09-23 21:55 UTC — shared-checkout isolation correction

- User clarified that the previously good project version must remain intact,
  including if this session stops mid-work. The experiment source has been
  unpacked into the isolated local
  `DeepMzyme_Data/notebook_outputs/plans/gvp_fusion_diagnostic_l4_v1/source_checkout/`.
  Byte comparisons confirmed that its two modified trainer files, new runner,
  and tests match the frozen archive before changes to the shared checkout.
- Restored **only** `src/training/config.py` and `src/training/run.py` in the
  shared checkout to their pre-study Git state, and removed this study's two
  newly added root `scripts/` and `tests/` files. `git diff --exit-code` for
  those trainer files is clean. All other pre-existing staged/untracked user
  work remains as it was; no pre-existing process was touched.
- The experiment runs from the isolated snapshot already uploaded to Colab.
  Its source archive SHA256 still matches the frozen manifest. Code changes
  in that snapshot are candidates for review, not promoted into the main
  checkout.
- The Colab download of v12 completed at 4,731,180,661 bytes with exit 0.
  Checksum/extraction have not yet been verified.

### Baseline Git verification requested by user

- The pre-study `git status --short` showed **no changes** in either
  `src/training/config.py` or `src/training/run.py`. Therefore restoring only
  those files to Git HEAD restored their observed pre-study state.
- Git HEAD is `9a069d226caacbc43c12d0d53fa49ed714c8b263`, committed
  2026-09-22 18:35:17 +0300. `git diff HEAD --` for both trainer files is
  empty after restoration.
- The checkout as a whole was **not clean before this study**: eight staged
  entries existed in status, covering experiment docs and two historical
  scripts. They remain staged and were not changed by this study. The only
  study modification in the shared tracked tree is the metal playbook note;
  this execution ledger is a new documentation file.
- Post-isolation fingerprint of the pre-existing staged patch
  (`git diff --cached --binary | sha256sum`):
  `accc647da5d687f088591407ec734981b61b0948542bfb4173a268c0abb3f8b0`.
  Recheck this fingerprint before any future cleanup to avoid touching the
  user's staged work.

## 2026-09-23 21:57 UTC — remote data gate

- On the verified L4, `/content/source_snapshot.tar.gz` matched source SHA256
  `d6d9cd24ee6ac1ebbb74f68c6634770a1d7e03cfbc2e4b7e902384b97500389b`
  and was extracted to `/content/DeepMzyme`. Remote Python is 3.13.15;
  stock PyTorch is 2.11.0+cu130, CUDA is available, device is NVIDIA L4
  with capability (8,9).
- The hosted v12 archive was downloaded to `/content/bundle_v12.tar.gz`,
  exact size 4,731,180,661 bytes. The **remote calculated SHA256** matched
  `90c0899829e0ac5ca94a5ef34484b74ca1014d3e9b6b1fba90bd338ee3440dee`.
- Started remote extraction of **only** the non-overlap train directory,
  shared structure store, cached ESM embeddings, and updated external
  features. No test directory is selected for extraction. Remote receipt/log:
  `/content/extract_selected_receipt.json` and `/content/extract_selected.log`;
  completion marker `/content/extract_selected.exit`. Remote extraction PID
  `12588` belongs to this study. No model fit has begun.

## 2026-09-23 21:59 UTC — readiness checks passed

- Selective extraction exited 0 in about 41 seconds. Receipt counts: train
  6 archive members, ESM cache 5,893, external features 5,894, structure
  store 6,679. The exact train summary and structure manifest exist; the
  non-overlap `test` directory **does not exist** on this VM.
- `colab install -r requirements/colab-overlay.txt` reported a transport
  timeout, but a separate remote verification established that all required
  packages are installed. The training Python imports PyTorch 2.11.0+cu130,
  torch-geometric 2.7.0, NumPy 2.4.4, scikit-learn 1.8.0, pandas 3.0.1,
  Optuna 4.8.0. Stock PyTorch was preserved and the L4 remains CUDA-ready.
- Bundle SHA, source SHA, hardware, dependencies and train-only materialization
  gates have passed. Four one-epoch smokes are next; no full fit has begun.

## 2026-09-23 22:03 UTC — interruption safety guard

- Scheduled a **new, study-owned** user-systemd timer
  `deepmzyme-gvp-l4-stop-20260924.timer` for 2026-09-24 01:36 UTC
  (04:36 IDT). It runs only
  `DeepMzyme_Data/notebook_outputs/plans/gvp_fusion_diagnostic_l4_v1/stop_own_colab_session.sh`,
  which targets only `gvp-l4-v1-20260924a`. This protects against an
  interrupted assistant turn leaving a billable VM running past the
  session's closeout reserve. It does not touch other sessions or processes.
- `systemctl --user list-timers` verified the timer was scheduled. The guard
  writes `session_stop_guard.log` when it fires. If this session is stopped
  earlier, the timer can be removed after verifying the stop, or it will
  harmlessly attempt to stop the already-absent named session at deadline.
- F0 fold-0 seed-42 one-epoch smoke was started on the L4 through the frozen
  runner. It is still in progress at this checkpoint; its run path is
  `/content/runs/gvp_fusion_diagnostic_l4_v1/gvp_l4_v1_F0_seed42_fold0_smoke`.

## 2026-09-23 22:07 UTC — F0 smoke completed and copied

- F0 fold 0, seed 42 smoke ran from 22:00:16 to 22:06:14 UTC, exit 0.
  The runner recheck identified it as `Already complete`, requiring its
  selected checkpoint, 1 validation row, aligned prediction artifact, and
  matching frozen command/config identity. One-epoch BA was 0.2601; **do not
  use this score for arm selection**.
- Split: 1,109 train and 280 validation pockets, 0 PDB-ID overlap. Native
  four-class validation distribution Mn 144, Cu 42, Zn 48, Class VIII 46.
- Run metadata records fast GVP group LR `3e-4` / 506,144 parameters and
  slow group LR `3e-5` / 276,870 parameters.
- Remote archive `/content/gvp_l4_v1_F0_seed42_fold0_smoke.tar.gz` was
  downloaded to local `artifacts/` within this study. Remote and local
  SHA256 both equal
  `c3e222070b92f59988227cc80b032c29852afc88a32e46d39bdead5ad975df5e`.
  No next smoke was launched before transfer verification.

## 2026-09-23 22:15 UTC — F1 smoke completed and copied

- F1 fold 0, seed 42 smoke ran from 22:07:53 to 22:13:38 UTC, exit 0.
  The frozen runner verified it as complete. One-epoch BA was 0.2566; this
  score is **not** a model-selection result.
- Actual model optimizer membership changed exactly as planned: the fast
  group gained only `init_vec_proj`, `gvp_attn_pool`, and `gvp_fusion_proj`.
  Fast parameter count rose from F0's 506,144 to 547,649; the slow group
  retained ESM, fusion gate, and metal head modules.
- Remote archive `/content/gvp_l4_v1_F1_seed42_fold0_smoke.tar.gz` was
  downloaded locally. Both SHA256 values equal
  `2857d50bac995da5fcd5a654f1300d89e987ab3b56880b249208057124307c69`.
  The machine-readable local `artifact_manifest.json` now records both
  transferred smokes. No full fit has begun.

## 2026-09-23 22:19 UTC — PDB split audit requested by user

- New study commands explicitly set `--train-val-split-by pdbid`, five folds,
  `--split-stratify-by metal_site`, and fixed `--split-seed 42`. The splitter
  groups all pockets under `pocket_split_key(..., "pdbid")` before assigning
  any group to a fold (`src/training/splits.py`). Its code therefore does not
  divide a PDB ID between train and validation in any fold. Actual F0 and F1
  fold-0 smoke `split_diagnostics.json` each report overlap counts of **zero**
  for PDB ID, PDB-chain, structure ID, and pocket ID. Their saved ordered
  train and validation identity SHA256 values are identical across F0/F1:
  train `afab791882ea64162f8985501c99a86fbd217985bf30501afa6a7aefbb91d570`,
  validation `b334e6b0036e0486a282da4c8a27b82e5e36f5600319fe85c879073b634008ae`.
- The non-overlap PinMyMetal train/test structure manifests have **zero**
  shared PDB IDs (computed directly from `structure_name` prefixes). The
  Colab VM extracted the train side only; its `test` directory is absent.
  This is PDB-ID isolation, not a claim of sequence-homology independence.
- The recent 2026-09-23 Antigravity exact-PinMyMetal benchmark is a
  **different protocol**: `--train-val-split-by pocket_id`, five-class native
  training, exact-PinMyMetal train membership, and held-out test evaluation.
  Its saved multimodal split diagnostics show **58, 56, 43, 49, 45** shared
  train/validation PDB IDs over folds 0–4 (pocket-ID overlap zero). The
  corresponding GVP and ESM arms used the **same pocket folds within that
  historical benchmark**, as all three arms have one identical saved
  train/validation membership-hash pair per fold. They are not the same
  folds, cohort, or target as the new study.
- The exact historical train/test manifests also share **177** PDB IDs;
  its test pockets are held out by pocket membership but not by PDB ID.
  Thus historical benchmark validation/test numbers must not be treated as
  a matched control or direct improvement claim for this new PDB-grouped
  study. Only comparisons among new-study arms on common folds/seeds are
  eligible for the planned paired analysis.
- Added an isolated, standard-library-only local auditor at
  `DeepMzyme_Data/notebook_outputs/plans/gvp_fusion_diagnostic_l4_v1/audit_artifacts.py`.
  It checks each copied archive's SHA, train-only configuration, absence of
  test artifacts, epoch count, zero actual PDB-ID intersection, and common
  saved membership hashes within each fold. The first run verified both
  F0/F1 smoke archives, writing `split_audit.json` with **0 failures**.

## 2026-09-23 22:21:43 UTC — user-directed stop and closeout

- User explicitly stopped the study after clarifying that the intended
  comparison should reproduce the recent Antigravity **pocket-level** split
  and method. The new study's PDB-ID-grouped design therefore does **not**
  match the user's intended benchmark. It must not be resumed or presented
  as a matched comparison to the recent run.
- G0 fold-0 seed-42 one-epoch smoke had been started, but was interrupted
  during preparation. It is incomplete and was not copied or used. G1 was
  never launched. The only completed/copied runs are the F0/F1 one-epoch
  smokes; no 50-epoch full fit occurred and no study comparison exists.
- `colab stop -s gvp-l4-v1-20260924a` returned `Session terminated`.
  A following `colab sessions` returned `No active sessions found on server.`
  The study-owned stop timer
  `deepmzyme-gvp-l4-stop-20260924.timer` was stopped and verified `inactive`.
  No other Colab session or pre-existing PC background process was touched.
- Main project training source remains at its pre-study Git state. The
  experimental code is isolated in this study's `source_snapshot.tar.gz` and
  `source_checkout/`. The two completed smoke archives, verified checksums,
  manifest, and audit receipt remain in the local study directory.
- To design a future **matched** validation-only comparison, start a **new
  study identity** and explicitly use the historical exact PinMyMetal train
  cohort, `--train-val-split-by pocket_id`, five folds, model seed 42,
  historical target and stratification, and the same saved fold membership
  hashes. The historical trainer command also evaluated a held-out test;
  do not re-open it for optimization without a separate final-report policy.
