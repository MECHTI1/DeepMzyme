# EC1 standalone campaign — interrupted after three verified runs

The authorized matrix is 12 runs: Only-GVP, Only-ESM and GVP + graph-level
late fusion, each at learning rates `3e-5` and `1e-4`, seeds 42 and 43,
30 epochs. Three Only-GVP runs completed and were verified locally and in
Google Drive. Run 4 was launched, but its completion is unverified after the
Colab CLI lost its session record. Runs 5–12 have not launched. This is a
partial campaign, not a completed standalone comparison or model promotion.

## Protocol and provenance

- Source base: `09892f1f8a94337451274cdec29a97bc84516f9a`, plus the captured
  uncommitted campaign runner and notebook output-path fix. Exact source
  hashes are in `campaign_plan.json`; the full snapshot is in the verified
  preparation archive. Cloning the base commit alone is insufficient.
- Executable settings: the existing EC playbook standalone block with
  `STANDALONE_PHASE = "baseline"`, expanded by the current Colab notebook.
- Dataset: v12 `CARE_task1_30_clusterRes30_train_test_metallo/train`, using
  `final_data_summarazing_table_transition_metals_only_catalytic.csv`.
  Bundle SHA256: `90c0899829e0ac5ca94a5ef34484b74ca1014d3e9b6b1fba90bd338ee3440dee`.
- EC depth 1, single-label eligibility; ambiguous EC1 targets excluded.
  Full UniProt accession grouping via `pdbid`, fixed split seed 42,
  configured validation fraction 0.15, active-target stratification.
  Actual retained membership: 993 training pockets / 751 proteins and
  175 validation pockets / 42 proteins. Ordered identities exactly match
  the successful GPU smoke reference.
- Complete existing ESMC-300m and external caches; no regeneration.
  Conservative residue features, radius 6, no RING, no explicit metal nodes,
  residue-only readout, no augmentation or contrastive loss.
- Structure-weighted EC loss, inverse-frequency class weights from training
  groups, group-level validation logit averaging. Batch 4, weight decay
  `1e-4`, fixed learning rate, deterministic execution.
- Checkpoint selection: `val_ec_group_level_1_balanced_acc` only.
  No held-out inference, held-out metrics, HPO, auxiliary or metal training.
- Actual GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition; PyTorch
  `2.11.0+cu128`, CUDA 12.8, `sm_120`. Readiness verified CUDA forward/backward
  and CUDA model parameters. Successful smoke runs were not repeated.

## Measured validation results

Scores and recalls are from each run's EC-selected checkpoint.

| Family | LR | Seed | Selected epoch | Group EC1 balanced accuracy |
|---|---:|---:|---:|---:|
| Only-GVP | 3e-5 | 42 | 29 | 0.419048 |
| Only-GVP | 3e-5 | 43 | 22 | 0.367857 |
| Only-GVP | 1e-4 | 42 | 23 | 0.555952 |

The completed `3e-5` pair has mean **0.393452**, sample SD **0.036197**,
and minimum **0.367857**. The `1e-4` pair is incomplete; do not compare a
single-seed maximum with the two-seed mean. Only-ESM and late fusion have
no measured baseline result in this batch yet.

| Only-GVP configuration | EC1 | EC2 | EC3 | EC4 | EC5 | EC6 | EC7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3e-5, seed 42 | 0.2500 | 0.8333 | 0.6000 | 0.7500 | 0 | 0.5000 | 0 |
| 3e-5, seed 43 | 0.8750 | 1.0000 | 0.4500 | 0.2500 | 0 | 0 | 0 |
| 3e-5, two-seed mean | 0.5625 | 0.9167 | 0.5250 | 0.5000 | 0 | 0.2500 | 0 |
| 1e-4, seed 42 only | 0.8750 | 0.6667 | 0.6000 | 0.7500 | 0 | 0 | 1.0000 |

Validation protein support for EC1–EC7 is **8, 6, 20, 4, 1, 2, 1**.
The one-protein EC5 and EC7 classes particularly limit interpretation.
The completed pair supplies initial fixed-split two-seed evidence (Grade 3);
the overall unfinished campaign is Grade 6. No grouped-fold, paired-CI,
promotion, or final-test claim is supported.

## Persistence and interruption

- [Portable configs, metrics, logs, split identities and receipts](../raw/ec1_standalone_v12_20260914/).
- [Drive campaign folder](https://drive.google.com/drive/folders/1hhfLcjlTSA4i8VBY3LxCP6yCpycvHy9y).
- Local full artifacts: `DeepMzyme_Data/notebook_outputs/runs/ec1_standalone_v12_20260914/`.
- Each of runs 1–3 has best and last checkpoints, complete metrics and
  configurations in an archive whose SHA256 was verified locally. Drive
  uploads were verified by file ID, parent and byte size; this is not a
  claim of a separate Drive-side SHA256 calculation.
- Final source/preparation archive: `ec1_preparation_verified.tar.gz`, SHA256
  `31a8cdeb883144c8ad98693708815a0a818d140b2c75b24984efab68e117e1c8`.
  It supersedes the earlier preparation archive created before the wrapper's
  path-type correction.
- Session: `deepmzyme-ec1-baselines-v12`; created endpoint
  `gpu-g4-s-kkb-use5c0-2sqteooltirnp`. At 2026-09-14 14:28:02 UTC the CLI
  pruned its local record following a `404/401` connection failure. OAuth
  remained valid and the endpoint was still listed as `[?]`. The runtime
  was **not confirmed stopped**. Reconnection authorization was requested
  because the installed Colab skill prohibits targeting `[?]` sessions.

## Continuation and Chat 5

Recover and inspect run 4 before restarting it. Reuse any completed valid
artifacts, then finish the remaining authorized matrix with the same protocol.
Keep the metal campaign pending. Chat 5's baseline prerequisite is unfinished.
After this matrix completes, Chat 5 still needs a separately certified matched
zero-auxiliary control, joint-eligibility and union-of-holdouts checks, and
development-only association analysis. The standalone EC cohort must not be
silently substituted for the joint intersection cohort. No auxiliary work was
started in Chat 4.
