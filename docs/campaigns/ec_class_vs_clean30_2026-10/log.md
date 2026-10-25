# EC class versus CLEAN30: log

Dated entries, newest last. Campaign [README](README.md).

## evc-001

2026-10-22 — **campaign planned; decisions recorded; nothing executed.** A read-only planning review (2026-10-21) and
the maintainer's decisions (2026-10-21 and 2026-10-22) define the campaign. Nothing was installed, trained, embedded or
evaluated, and no GPU was used. Decision IDs follow the maintainer's campaign plan, which is kept with the private
campaign records.

| ID | Decision | Reason |
|---|---|---|
| U1 | The official CLEAN split30 folds, unchanged; no newly engineered outer split | The maintainer's choice: official datasets make the comparison more credible than a self-made split |
| U2 | Add a whole-protein ESMC branch | The maintainer's choice |
| U3 | Outside the core: non-metalloenzyme training data, a homology-retrieval add-on, a metal-element input | Not selected by the maintainer |
| U4 | Identity rule: ≥30% identity over ≥50 aligned residue pairs, E ≤ 1e-3, any coverage (pinned MMseqs2 18-8cc5c), plus shared selected AlphaFill donors. Used only for the no-hit co-primary subset, identity bins and inner selection splits | Coverage-gated clustering does not count most partial-length ≥30% partners: within fold-0 training, 5.5% of proteins have one at ≥80% coverage of both sequences, 42% at ≥50% coverage of the shorter sequence and 56% at any coverage ([design](design.md#benchmark-systems-and-claims)). Confirmed on 2026-10-22 as an addition to the official folds, not a re-split |
| U5 | No new training data or re-processing | Follows from U1. The cohort's fold 1–4 screening gap ([DATASETS](../../DATASETS.md#clean30-conservativecurrent-main)) is disclosed, not repaired |
| U6 | The CARE clusterRes30 metallo test ([DATASETS](../../DATASETS.md#task-1-clusterres30)) is not used and stays unevaluated | 34 metallo structures are too few for claims, and use would spend a never-evaluated test |
| U7 | Other campaigns keep their status; none is closed | The maintainer's choice |
| U8 | Primary metric: balanced accuracy over EC classes 1–6, from seven-way predictions; seven-class BA, accuracy and macro-F1 secondary | Class 7 (translocases) has at most about 4 test proteins, all heme-copper oxidases; a seven-class bootstrap is undefined in about 37% of resamples |
| U9 | Cosine to zero, terminal checkpoint, the same rule in selection and refit; each fold chooses 12, 20 or 30 epochs; no weight averaging, no warmup | [TECH-027](../../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-027--cross-validation-and-final-refit-use-different-checkpoint-rules): best-epoch selection added 1–7 points on metal (1.5–6.6 under cosine; nine fold-0 runs, one seed, [v3-018](../../archive/campaigns/pmm_ion_metal_v3/log.md#v3-018)), and a refit has no validation set. 30 epochs was inherited, not tuned; a 10-epoch cosine schedule gave +1.99 BA over the 50-epoch metal schedule on folds 1–4 (interval [−0.39, +5.90]; not adopted). Under cosine to zero, weight averaging converges to the final weights. The 20-epoch value was added on 2026-10-22 after the maintainer asked for more options |
| U10 | Candidate levers L1–L9: GVP-trunk weight decay set to a target shrink; raw-distance RBF with 8 Å edges; coordinate noise and residue dropout; auxiliary heads for the second and third EC levels; a supervised contrastive loss on the third EC level with a cross-batch memory; ESM-head dropout; epoch count; an intermediate ESMC layer for the whole-protein branch; post-hoc logit adjustment | No new data or inputs. Each lever must pass a pre-declared promotion rule on inner validation |
| U11 | CLEAN at upstream `f2bf2a4`, in a separate `fair-esm 2.0.0` environment with ESM-1b. CLEAN code, patches and weights are never distributed and are privately backed up; the repository holds the maintainer's runner, setup steps, plain-language patch descriptions and predictions | CLEAN's University of Illinois research-use licence counts distribution as commercial use and allows own copies for non-profit research; the maintainer confirmed non-profit use on 2026-10-22. Every report carries the licence's acknowledgement (§2.7.1) and is notified to the University of Illinois (§2.7.2). ESM-1b code and weights are MIT |
| U12 | CLEAN runs its published recipe without tuning; no extra epoch variants | Disclosed both ways: CLEAN gets no local re-tuning against DeepMzyme's per-fold selection, and its published epoch count was chosen upstream on these same folds, which favours CLEAN |
| U13 | Keep the ESMC-input CLEAN control | Separates the language model from the model design |
| U14 | 5-seed ensembles for DeepMzyme and for CLEAN R1 | Refits are under 3% of fits; metal ensembles gained about +1.2–1.5 BA from 1 to 3 checkpoints per site and about +1 more at 6 (not interval-supported) |
| U17 | Campaign data root on the secondary drive with an NVMe mirror | Durability |

Pending: U15a, a GPU smoke session of at most $3 (gate G2); U15b, the campaign GPU ceiling (G3); U18, the test read
(G5); U16, an optional deployable model trained on all 740 proteins.

Records in this change:
- the [test-use ledger](../../DATASETS.md#test-use-ledger) entry for five incidental exposures on 2026-10-21, with the
  affected ledger rows pointing to it (CLEAN30 also naming this campaign; CARE clusterRes30; non-overlapped and exact
  Zenodo PinMyMetal), and the 2026-10-22 entry for two directory-level reads near test files;
- the fold 1–4 screening-gap note in the [CLEAN30 cohort section](../../DATASETS.md#clean30-conservativecurrent-main);
- the [documentation index](../../README.md#task-plans-and-reviews) row marking the 2026-09-25 CLEAN30 plan superseded;
- [TECH-032](../../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-032--a-gvp-specific-weight-decay-is-silently-ignored-without-a-distinct-gvp-learning-rate),
  the GVP weight-decay defect that lever L1 needs repaired.

Earlier CLEAN-notebook test outputs (Phase 0 check, file names only, no values): a search for the notebook's result
names (`*_maxsep.csv`, `ec_level_metrics.csv`, `all_clean_predictor_metrics.csv`, `scored_results/`) found none in the
working tree, the NVMe backups, the DeepMzyme folders of the secondary drive or its pre-project archive folder
(test-named directories skipped). Google Drive, which holds the
notebook's Colab data root, was not checked in this working session. Until it is, reports state that CLEAN30 test
outputs may have been produced earlier by that notebook.

Next: the maintainer's approval at G0 sets the README to active; Phase 1 (CPU infrastructure) follows.

## evc-002

2026-10-23 — **gate G0 approved; campaign active.** The maintainer approved the plan and the Phase 0 records, so this
campaign is now the current campaign in `EXPERIMENT_STATUS.md`; the previous STATUS is preserved in the
[phase2_unified log](../phase2_unified_2026-10/log.md) (p2u-006). No run, install, GPU use or test read.

Benchmarks: the maintainer decided not to add benchmarks now. CLEAN's own published split30 benchmark stays the only
one in this campaign. The metalloenzyme subset of the DeepFRI/GearNet PDB EC benchmark (experimentally observed metals,
published structure-based comparators) is noted as a possible later campaign; it is outside this campaign (U5).

Open from evc-001: the names-only Google Drive check of earlier CLEAN-notebook outputs.

Next: Phase 1, CPU code and tests only: the fixed seven-class EC-class vocabulary and metrics, frozen memberships,
selected-site extraction, the whole-protein ESMC branch, the [TECH-032](../../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-032--a-gvp-specific-weight-decay-is-silently-ignored-without-a-distinct-gvp-learning-rate)
repair, the optional auxiliary, contrastive and logit-adjustment terms, prediction export, the campaign runner and
scorer, identity tools, the private CLEAN harness and the ESMC layer-cache job.

## evc-003

2026-10-23 — **Phase 1 infrastructure built; gate G1 checks run; G1 awaits the maintainer's approval.** CPU only: no
GPU, no test read, nothing evaluated. The only fits were one-epoch engineering smokes on 32 fold-0 training proteins
(checkpoints deleted after verification) and CLEAN timing smokes on synthetic tensors. The code was uncommitted when
these checks ran.

Built:
- in the trainer:
  - the fixed seven-class EC-class vocabulary and metrics, and frozen EC fold memberships;
  - selected-site pocket extraction;
  - the whole-protein ESMC branch (F4, F5, C-G);
  - the [TECH-032](../../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-032--a-gvp-specific-weight-decay-is-silently-ignored-without-a-distinct-gvp-learning-rate) repair with target shrink;
  - auxiliary EC-level heads, the SupCon memory and per-fit augmentation;
  - the label-free prediction export with replay;
- in `src/ec_clean30/` with scripts:
  - the identity tool and the campaign runner;
  - selection statistics and logit adjustment;
  - the nearest-hit baseline and the scorer;
- for CLEAN:
  - the maintainer's comparator runner, which contains no CLEAN code;
  - a private CLEAN harness outside the repository (licence: never distributed).

| Gate item | Result |
|---|---|
| G1.0 clean tree | Pending: needs the commit of this work |
| G1.1 CLI | All new trainer flags present; `--help` exits 0 for the six campaign CLIs |
| G1.2 new tests | 644 passed in 21 files. The planned label-conflict, auxiliary-head and SupCon files are `test_ec_class_vocabulary.py` and `test_ec_auxiliary_objectives.py` |
| G1.3 regression set | 327 passed, 0 skipped, as at the WP-0 baseline |
| G1.4 torch-free closure | `False` (also covers the baseline, identity, campaign and provenance modules) |
| G1.5 full suite | 2,497 passed, 17 skipped (the baseline set), 1 failed: `test_tree_hash_follows_the_campaign_rule` hashes tracked files and compares them with all files on disk. With the 23 untracked campaign files added the hashes match (checked), so rerun it after the commit |
| G1.6 smoke checks | 45 passed (two added: selected-site extraction, whole-protein ESMC forward) |
| G1.7 docs contract | 0 strict failures |
| G1.8 metal no-regression | Not run (optional) |
| G1.9 identity, fold 0 | Pinned MMseqs2 `b7ef6e0e…`; 622 queries, 8,115 targets; 603 eligible (19 with several EC classes excluded); 4 inner folds of 150–151; certified, 0 cross-fold links. EC class 7: 3 proteins, none in inner folds 2 and 3 |
| G1.10 engineering smokes | F1–F5, C-G and an F4 refit: all 21 checks passed for all 7 units, 5–11 s and about 0.9 GB each; the dry run found no read outside the view |
| G1.11 CLEAN harness | Self-check 22/22, parity 29/29, distance export 19/19, ESM-1b farm 15/15. Timing smoke: the fold-0 R1 distance-map step needs more than 3 GB (killed at the cap, alone and in pairs); full-size training not yet measured |

Receipts (SHA-256). Under `DeepMzyme_Data/campaigns/ec_class_vs_clean30_2026-10/g1_smoke/`:

| File | SHA-256 |
|---|---|
| `identity/fold0/fasta_receipt_metallo_train.json` | `dcc2872c9b06f113367aa60edfd32622d8f4be28330d09ca33d4eea78f510047` |
| `identity/fold0/search_metallo_train/search_receipt.json` | `62e60111d0eda8c3ec918681c294ad27986752508eb46f8850eb3de9d4fe4bd5` |
| `identity/fold0/edges_receipt_metallo_train.json` | `406b61bba97211dd4c3a8b94768845873445d8412948979d6fd5debb2657b0d0` |
| `identity/fold0/components_receipt_inner.json` | `9b4888ca4c6029199b39d984720f0c3023760de58e25d417129308783ae1adbb` |
| `identity/fold0/inner_assignment_receipt.json` | `912275973b8aaa4b8ab46b43a97090a543851721c6759c9e3c4c36bf81957788` |
| `identity/fold0/inner_certification_receipt.json` | `efdc17f90b78daf7b0860b408e4faef0026d8fa1c0f229940b67d2d34c369b78` |
| `view/view_manifest.json` | `1cd2a80606eeccdacd0e9352371ebd7749da0c54ba17f31e2c5108a9d9e997c1` |
| `view/chain_means_final/manifest.json` | `0da84e7136b83df53d6f66296e653ef5bd77aaae0f9b6fb42905bad1e32c83e7` |
| `view/receipts/dry_run_receipt.json` | `131b12bc1e801d2f52405f9d9bd2d70e0a951900389fa12d70c69292142d2285` |
| `runs/verify_unit_receipt.json` | `6180a624e6dc4c0f83e3ea5b7ae8391c83d480c94b1202e0b7db18f4e7e14898` |
| `runs/smoke_checkpoint_deletion_receipt.json` | `64d40dfa7de0f0a81a3cdc342728ce7cc72f13ef13e556ba2da445e8d3e38c0b` |
| `runs/smoke_extra_checkpoint_deletion_receipt.json` | `451b55c2a0fe5922ef6e2f8f915456f6c716c45784e34244bd30b89705eb573f` |

The extra deletion receipt covers the last checkpoints that the first deletion step left behind; that step now deletes
them too. The CLEAN sequence map receipt (`clean_sequence_map/sequence_map_receipt.json`; 10,202 accessions, each in
4 training files, no conflicts) is `28765a796f689eccffda3dcc2c5a0616199c4c38ebc6060b9e467dc5b23b09f1`.

The private harness receipts, in the private record and its NVMe mirror:

| Receipt | SHA-256 |
|---|---|
| Self-check | `13b302165c0cd50a876a0716f667a88b4ba68b4f61a325a5e6b979f8838efd21` |
| Parity | `1d49be0cd585dd3cee51c9e8b309ac25c0ea0d5f3d8f864b52d815d895b01fa9` |
| Distance export | `50a355276b331d6ffdde23c8f2e8d8e23fd6bf736ac4296acb7e19bfb850f157` |
| ESM-1b farm | `c7e646cedd2f598509114bbe1a3df5ecbf12bbb2865142bbb07d8e1733f39341` |
| Patch manifest | `f331019de076ec42f05df057055052c76bab19774b03ed9fd23c6da460aaf53d` |
| Timing-smoke summary | `c6db8ac17008392cbae1904721ee25973e1a6de48dea0ec454082d10a38a6489` |
| Review note on that summary | `fe4cbe182890d983ca7291fd60247281279def650ebec031c22c5342ab774310` |

Found and fixed during G1:
- the identity tool now reads structure stems from each fold's training structure manifest, because the prepared
  training CSVs hold accessions;
- smoke checkpoint deletion now also removes the last and best checkpoints;
- Freeze 2 now collects the CLEAN checkpoints, pins the scorer's imports and the baseline, and refuses inner-fold CLEAN
  fits;
- the scorer requires each CLEAN test export to carry its Freeze-2 binding;
- the list of later EC-class nomenclature transfers is keyed by EC number, so building it never needs test labels.

Records: the [test-use ledger](../../DATASETS.md#test-use-ledger) entry of 2026-10-23 for tracked file names that a
review pass listed near test folders (names only, nothing opened). In the
[technical issues](../../FOLLOW_UP_TECHNICAL_ISSUES.md), TECH-032 is repaired for new runs and TECH-035 is new: a
latent prefix fallback in the per-residue ESM loader. The
[playbook recipe](../../EC_TRAINING_PIPELINE_PLAYBOOK.md#ec-class-versus-clean30-campaign-recipe) lists the commands.

Open for the maintainer:
- G1 approval and the commit (G1.0 and the G1.5 rerun);
- CLEAN distance-map memory: a one-off higher local cap for the five maps per regime and embedding, or VM CPUs. The
  timing smoke is then rerun under 3 GB with the maps linked;
- the names-only Google Drive check from evc-001;
- the Freeze-1 decisions collected during the build (kept with the private campaign plan).

## evc-004

2026-10-25 — **gate G1 approved.** The maintainer approved gate G1 on the results of evc-003, including its open items:
- the clean-tree check (G1.0) and the rerun of the tree-hash test follow the commit of the Phase 1 work;
- the optional metal no-regression check (G1.8) was not run;
- the CLEAN timing smoke is repeated as decided below.

CLEAN memory: CLEAN's one-off distance-map step may run once under a 6 GB local memory cap (2 cores, no swap), with
nothing else running. CLEAN training, DeepMzyme fits and every other local run keep the 2-core, 3 GB cap. The timing
smoke is then repeated under 3 GB with the map linked, alone and as two concurrent fits.

No GPU use, test read or Freeze-1 decision is part of this approval. Next: Phase 2, CPU only (training-side inputs for
all five folds, then the Freeze-1 decisions).
