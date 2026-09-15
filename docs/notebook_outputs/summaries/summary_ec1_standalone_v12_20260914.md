# EC1 standalone v12 baselines — completed

All **12 authorized runs completed 30 epochs**: Only-GVP, Only-ESM and GVP +
graph-level late fusion, each at learning rates `3e-5` and `1e-4`, seeds 42
and 43. All checkpoints, configurations, split identities, metrics and logs
are saved locally and in Drive. This is initial fixed-split two-seed evidence
(Grade 3), not final model promotion. Held-out evaluation stayed disabled.

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

| Family | LR | Seed 42 | Seed 43 | Mean ± sample SD | Minimum |
|---|---:|---:|---:|---:|---:|
| Only-GVP | 3e-05 | 0.4190 | 0.3679 | 0.3935 ± 0.0362 | 0.3679 |
| Only-GVP | 0.0001 | 0.5560 | 0.5393 | 0.5476 ± 0.0118 | 0.5393 |
| Only-ESM | 3e-05 | 0.8393 | 0.9036 | 0.8714 ± 0.0455 | 0.8393 |
| Only-ESM | 0.0001 | 0.9821 | 0.9571 | 0.9696 ± 0.0177 | 0.9571 |
| GVP + late fusion | 3e-05 | 0.8571 | 0.8571 | 0.8571 ± 0.0000 | 0.8571 |
| GVP + late fusion | 0.0001 | 0.8571 | 0.9929 | 0.9250 ± 0.0960 | 0.8571 |

Mean per-class recall at each seed’s selected checkpoint:

| Family | LR | EC1 | EC2 | EC3 | EC4 | EC5 | EC6 | EC7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Only-GVP | 3e-05 | 0.5625 | 0.9167 | 0.5250 | 0.5000 | 0.0000 | 0.2500 | 0.0000 |
| Only-GVP | 0.0001 | 0.7500 | 0.8333 | 0.6250 | 0.6250 | 0.0000 | 0.0000 | 1.0000 |
| Only-ESM | 3e-05 | 0.8750 | 1.0000 | 0.9750 | 0.7500 | 0.5000 | 1.0000 | 1.0000 |
| Only-ESM | 0.0001 | 0.9375 | 1.0000 | 0.9750 | 0.8750 | 1.0000 | 1.0000 | 1.0000 |
| GVP + late fusion | 3e-05 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| GVP + late fusion | 0.0001 | 1.0000 | 1.0000 | 0.9750 | 1.0000 | 0.5000 | 1.0000 | 1.0000 |

## Interpretation and limits

The largest observed two-seed mean in this screen is **Only-ESM at `1e-4`**:
**0.969643**, sample SD **0.017678**.
This identifies an initial standalone reference candidate. It does not
establish architecture superiority, grouped-fold robustness or promotion.
All families received identical LR opportunities, model seeds and epoch
budgets; all scores are selected by EC validation alone.

Validation protein support for EC1–EC7 is **8, 6, 20, 4, 1, 2, 1**.
One correct or incorrect EC5 or EC7 protein therefore changes that class's
recall from zero to one. SD across two model seeds measures seed variation
on this split, not uncertainty over new validation folds. A zero seed-mean
class recall fails the playbook's rare-class promotion gate; nonzero recalls
alone do not establish promotion. No grouped-fold or paired-bootstrap
confirmation was run. Only-ESM uses pocket-residue ESMC pooling, not an
independently tested full-protein pooling model.

## Selected checkpoints

| Family | LR | Seed | Selected epoch | Group EC1 balanced accuracy |
|---|---:|---:|---:|---:|
| Only-GVP | 3e-05 | 42 | 29 | 0.419048 |
| Only-GVP | 3e-05 | 43 | 22 | 0.367857 |
| Only-GVP | 0.0001 | 42 | 23 | 0.555952 |
| Only-GVP | 0.0001 | 43 | 23 | 0.539286 |
| Only-ESM | 3e-05 | 42 | 22 | 0.839286 |
| Only-ESM | 3e-05 | 43 | 28 | 0.903571 |
| Only-ESM | 0.0001 | 42 | 10 | 0.982143 |
| Only-ESM | 0.0001 | 43 | 24 | 0.957143 |
| GVP + late fusion | 3e-05 | 42 | 22 | 0.857143 |
| GVP + late fusion | 3e-05 | 43 | 24 | 0.857143 |
| GVP + late fusion | 0.0001 | 42 | 12 | 0.857143 |
| GVP + late fusion | 0.0001 | 43 | 27 | 0.992857 |

The per-seed recalls are preserved in
[`per_run_validation.csv`](../raw/ec1_standalone_v12_20260914/per_run_validation.csv)
and each run's epoch records. `verification.json` independently checks that
each selected epoch maximizes the declared EC metric, that per-class recalls
agree with its group confusion matrix, and that every run retains the exact
same ordered training and validation examples. Every archive SHA256 was
checked locally; Drive uploads were checked by ID, parent folder and byte
size. No separate Drive-side SHA256 calculation is claimed.

## Persistence, source snapshot and recovery

- [Portable evidence](../raw/ec1_standalone_v12_20260914/).
- [Drive campaign folder](https://drive.google.com/drive/folders/1hhfLcjlTSA4i8VBY3LxCP6yCpycvHy9y).
- Full local artifacts: `DeepMzyme_Data/notebook_outputs/runs/ec1_standalone_v12_20260914/`.
- `ec1_run_01.tar.gz` through `ec1_run_12.tar.gz` each include the completed
  run's best and last checkpoints, configs, identities, metrics and log.
  Checkpoint binaries remain outside git; hashes are in `artifact_manifest.json`.
- Exact source/preparation snapshot: `ec1_preparation_verified.tar.gz`, SHA256
  `31a8cdeb883144c8ad98693708815a0a818d140b2c75b24984efab68e117e1c8`.
  It supersedes the first preparation archive created before the runner's
  path-type correction. Current source/notebook/playbook hashes were checked
  against this campaign's frozen plan after recovery.
- The user committed the prepared runner and partial evidence as
  `7b2f2d5bcade238ff30591ebacd15a8c80a2f101` before continuation. No source or
  scientific configuration changed during continuation; no commit or push
  was performed by the agent.
- At 2026-09-14 14:28:02 UTC the CLI pruned its local record after a `404/401`
  connection error. The campaign runtime and original kernel survived. After
  explicit continuation authorization, restoring the known campaign session
  with a fresh runtime-proxy credential recovered completed run 4. Neither
  that run nor the successful smokes were repeated. Runs 5–12 then completed
  on the same GPU. `resume_receipt.json` records the recovery checks.
- Session `deepmzyme-ec1-baselines-v12`, endpoint
  `gpu-g4-s-kkb-use5c0-2sqteooltirnp`, was stopped after all artifacts were
  verified. `shutdown_receipt.json` records the stop and absence from the
  subsequent session listing. The old interruption record is superseded by
  the completion and shutdown records.

## Chat 5 readiness

The EC standalone prerequisite is complete. A new Chat 5 can plan and certify
the controlled EC-primary auxiliary comparison; auxiliary training is not
yet launch-ready merely because these baselines completed.

1. Build a matched zero-auxiliary control on the exact same jointly eligible
   cohort as its positive-metal-loss arm, with identical heads/encoder,
   features, grouping, weighting, splits, seeds, budget and EC1 group-level
   selection. The full-cohort `--task ec` scores here do not replace that
   control. The implemented controlled guard currently uses the joint
   graph-level late-fusion path with zero versus positive metal loss.
2. Certify protein exclusions across all task-label sources and reserved
   validation/test memberships before any shared-encoder loss is enabled.
3. Perform the planned development-only metal–EC1 association analysis before
   expensive auxiliary work, then predeclare the matched recipe and gates.
4. Keep the metal standalone campaign pending as requested. Both metal target
   formulations and their three-family comparisons remain outstanding; do
   not claim that both primary missions are fully validated.

No auxiliary training, broader HPO, dataset regeneration, held-out inference,
metal campaign or final refit was launched here. `Plan.md` is unchanged.
