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

## evc-005

2026-10-25 — **Phase 2 start: CLEAN timing smoke, round-1a decisions, identity chain for all five folds, layer-26
check and the code for the training-side inputs (commit point CP-A).** CPU only, training files only: no GPU, no test
read, nothing evaluated.

**CLEAN timing smoke, repeated.** Fold-0 R1 fits on the real `split30_train_split_0.csv` with synthetic 1280-d tensors
(no ESM-1b), run under the forbidden-read guard. Results were discarded; receipts and loss logs were kept privately and
mirrored to the NVMe drive.

| Step | Setting | Result |
|---|---|---|
| Distance map | Run alone under a 4.5 GB scope. The approved ceiling was 6 GB, but only about 5.9 GB was free | Peak 3.45 GB RSS (scope 3.61 GB), verified |
| One fit, 200 epochs | 3 GB, map linked, 2 threads | Training peak 2.45 GB RSS (scope 2.86 of 3.22 GB); final model present |
| Two fits together | One 3 GB scope, 1 thread each | Both OOM-killed about 4 minutes into training, before the first epoch |
| One fit, 20 epochs, internal ext4 disk | 3 GB, map linked | Median 4.0 s/epoch, refresh about 10 s, peak 2.40 GB RSS |

On the secondary drive the fit ran at a median 9.9 s/epoch and the map took 195 s; on the ext4 disk the map took 72 s.
After a restart the drive was mounted through ntfs-3g (FUSE), which slows the per-file reads of every epoch. At the
published 2,500 epochs a fit takes about 2.9 h on fast storage, so the CLEAN fits planned for five folds need about
160 CPU-hours run serially, above the plan's estimate of 95–135. Two concurrent fits would need about 6 GB.

Deviation: evc-004 allowed the distance-map step to run once above the 3 GB cap. It ran twice, both times alone and
under a 4.5 GB scope (below the approved 6 GB): first for the secondary-drive smoke, then for the ext4 timing check.
The second run was confirmed in round 1a (below).

| Run | Receipt | SHA-256 |
|---|---|---|
| Secondary drive | `phase_distance.json` | `eb9994b855e9da2ab16147fe16dfeaa05b89a99cb23cf438f0e7acbce75e1aee` |
| Secondary drive | `phase_alone.json` | `94aa527fd502be167133142564cb3387843eb030d996ca0fbb36b0890e5ff749` |
| Secondary drive | `phase_pair.json` | `b83fc3af22ac92a2942b783bdbd47debf2bae57f02a297ec7a99a28991fcc4f9` |
| ext4 disk | `phase_distance.json` | `ba94ef0930fb63e12f1bda6c96a7a6b1082417fbd3d8b98ff91b1dd952b73bfe` |
| ext4 disk | `phase_alone.json` | `c66bcfd140b943d8163536e8c38c192c02ebea141ff98744431230ca53fa4ff7` |

**Round-1a decisions: all as recommended.** The maintainer accepted every recommendation of the round-1a sheet, kept
with the private campaign plan:
- CLEAN fits run one at a time on this workstation; the final choice is made in Phase 4. Each one-off distance map may
  use a 4.5 GB limit with nothing else running.
- The second distance-map run (above) is confirmed.
- CLEAN embedding files and fit folders go on the internal ext4 disk, in a folder that DeepMzyme fits may not read.
- Only the fold-0 view, timing sequences from fold 0's training file and the pushed code go to the GPU machine in
  Phase 3.
- The layer-26 vectors are built on this workstation if a 5-protein parity check passes; otherwise in Phase 3.
- The lever settings stay as coded (L1 GVP-trunk target shrink; L3 0.1 Å noise and 0.1 dropout; L5 weight 0.1,
  temperature 0.1, width 128; L8 block 26).
- The confirm-default items stand as listed, including that per-fold EC-class counts are never printed or published
  for more than one fold.

The maintainer will restore the faster kernel NTFS driver for the secondary drive when convenient, and will do the
names-only Google Drive check before round 2.

**Identity chain (P2-1).** The identity tool's fasta, search, edges and inner-components steps ran for folds 0–4 at
the clean commit `697427e`, in `DeepMzyme_Data/campaigns/ec_class_vs_clean30_2026-10/identity/fold<k>/`, in about
2 minutes. Console output was written to private files. The largest inner component holds 4.3–5.5% of a fold's
metallo training proteins, so no fold needs the too-large-component fallback. Fold 0 equals the gate G1.9 run: same
code SHA-256 and the same node, sorted-search, edge, pair and component tables. Only the unsorted raw search output and
the search log differ.

| Fold | `fasta_receipt_metallo_train.json` | `components_receipt_inner.json` |
|---|---|---|
| 0 | `dcc2872c9b06f113367aa60edfd32622d8f4be28330d09ca33d4eea78f510047` | `9b4888ca4c6029199b39d984720f0c3023760de58e25d417129308783ae1adbb` |
| 1 | `ff81f71fbd21b599dc780478eda0f7d2a32503ac537534374fc66be75739a299` | `365abb449b42b7b3e9b363a0dcac31d1893b2a43bf169b7cb32a315a82e7f0a2` |
| 2 | `96e1bac04943926c18aef241278dab87cb8b22dbe3aba3b4d2fe80ee525671f2` | `81982f3a3e4e66e77d36d7241a996f21882bcceac1bec1fb58d42324b71f80cb` |
| 3 | `e7a214af294679f3ae02764a1190044b13fc66db930ff962cf46ac734d273e09` | `d9a2df7041c2ca2a00fc4a8aacca3ee6b86ed79b92a89681c4aeeb471aebcbe5` |
| 4 | `007257cef1281d0c99962369f0039f69b5cc30ba5cd9e381f88bd28f1bd80b87` | `34f8347f59da0c6f51965bb65596888d7ba744734223808d230148988e0afd68` |

**Layer-26 check (P2-0b).** `esmc_chain_means.py from-model --layer-index 26 --device cpu` built block-26 and
final-layer means for 5 fold-0 training chains of the G1 smoke view (ESM C 300M, float32 parameters, no autocast).
Its final-layer means matched the G1 means (manifest `0da84e71…`): minimum cosine 0.99999999999991, largest absolute
difference 1.4e-7 (limits 0.9999 and 1e-4). Parity receipt
`6b93a5d11a640a35703b46c01bbe79f36a30dd1ba8a3e462ef997e4d3c6d699f`; new manifest
`bf29febd38ebdf7d425931024e78fb2d616a7a2bdf9d48c4d92a0059a819b382`. The layer-26 vectors can therefore be built on
this workstation. The run took 21 s including the model load, so the first forecast for 2,960 chains is at most about
3.4 h; P2-8 sets the final forecast from the time per residue. `time -v` reported a peak RSS of 3,202,036 kB, and the
process group reached the 3 GiB cap, held there by file-cache reclaim. This is above the plan's 2.5 GB stop line, so
the full CPU build waits for the maintainer's memory decision.

**Code for the training-side inputs (CP-A).** New actions, each reviewed and tested on synthetic data:
- `label-table` (comparator script): the native EC label table from the five official training files, checked
  against the readers of the scorer, the nearest-hit baseline and the EC-class parser;
- `cohort-audit`: structure, site, pocket, ESM and label checks over the union of cohort proteins. On the 32 G1 smoke
  structures it reproduces the trainer's site-extraction totals (118 metal records, 42 selected, 76 dropped; 16
  structures with unselected metals; none skipped);
- `r2-lists`: CLEAN's R2 accession lists and the fold-0 R2-inner lists, from the identity tables and the cohort audit;
- `overlay build` and `overlay verify`: the data overlay `ecc30-overlay-v1` (a tar with its SHA-256 file), built from
  training files only;
- `simulate`: the null-adoption simulation and promotion-rule calibration;
- before the dry runs, in the view and dry-run code: the read guard as rules, file-mode handling with an
  `apply-view-modes` action, and three Phase 3 hooks (a Freeze-1b field in the unit identity, a per-unit memory limit
  and a cap prefix for `dry-run`).

Per-fold class counts are printed for fold 0 only. Refusals withhold reader messages that could name an accession,
and clean-tree refusals print no file names. Checks with the new files staged: full suite 2,832 passed and 18
skipped; 45 smoke checks passed; docs contract without strict failures;
the campaign modules import without torch; the six identity files under rule R-ID are unchanged since `697427e`. The
[playbook recipe](../../EC_TRAINING_PIPELINE_PLAYBOOK.md#ec-class-versus-clean30-campaign-recipe) lists the new
actions in order.

Records: the [test-use ledger](../../DATASETS.md#test-use-ledger) entries of 2026-10-25: process notes from the
Phase 2 review passes, a note on which test-side counts are derivable from training-side files, and a name search
that printed four lines of a CARE cache-smoke audit for one training protein.

Open for the maintainer:
- the memory limit for the full layer-26 CPU build, or the GPU in Phase 3;
- the names-only Google Drive check before round 2.

Next: P2-2 (sequence map regenerated, native label table) and P2-3 (cohort audit) on the committed code, then
round 1b.

## evc-006

2026-10-26 — **training-side inputs built: label table, regenerated sequence map and cohort audit; round-1b
decisions.** CPU only, training files only, at the clean commit `36498cd`. Console output was written to private files;
no test file was read.

The maintainer chose the layer-26 build route: the full CPU build of block-26 means runs on this workstation once,
alone, under a 4 GB limit (evc-005 recorded the 3 GiB peak of the 5-chain check).

P2-2: `sequence-map` regenerated `sequence_map.tsv` with the same SHA-256 as on 2026-10-23
(`e9d6ddbb9c772f893f7e9194c94ecac139c3b5893a7d40384368b41b3e487fe0`). `label-table` wrote the native label table of the
10,202 proteins in the five official training files: 0 conflicts, 0 fatal statuses, and agreement of the scorer's,
the EC-class parser's and the normaliser's readers on all 10,202 rows (the baseline reader on all 40,808 file rows).
Both list the same accessions.

P2-3: `cohort-audit` over the 740 cohort proteins passed every check in 3.6 min (peak 0.7 GB):
- every structure loads and matches its manifest; ESMC and external files are present and valid for all 740;
- all 882 selected sites resolve to one metal atom of the same element (largest distance 1.8e-6 Å; tolerance 0.01 Å);
- the EC set in each file name equals the native EC set for all 740 (A4);
- the per-fold label-only eligible counts equal the expected values (compared from a private file, not printed);
- the structure sequence equals the ESMC sidecar's for all 740, and equals the CLEAN sequence map for 739; the one
  difference is the known Q96KN2 V113I (A5);
- the shortest cohort sequence has 126 residues (A6: nothing to report);
- 370 string values in ESMC and external files name a test or mixed folder (A13; counted, not shown).

D2: 1,396 of the 2,278 metal records are unselected AlphaFill metals, in 383 proteins. Over the 29,944 pocket residues,
13.7% get a non-zero charge-proxy contribution from unselected metals, with a median share of 57% among them, so the
criteria for keeping the features as they are (A1 option b) fail in all five folds. Recomputed with every metal present, the
charge proxy matches the stored value for all 29,944 pocket residues within the audit's tolerance. 13 of the unselected
metals are copies of a selected site from another donor. PROPKA did not run or failed for 475 of 740 proteins
("unavailable"); for 134 of the 265 PROPKA proteins its input held the unselected metals.

| Output | SHA-256 |
|---|---|
| `cohort_audit_receipt.json` | `476c396dea2ba770faba2af1570953249de4cd9da1ed307c9161cf959f7b1884` |
| `cohort_proteins.tsv` | `a1ef06a4eaa8165fcabadf2b93eb161f20572bd76ce175696a537cae4cd023ec` |
| `cohort_sites_label_free.csv` | `1a9b8797c3bd6d00f4ee7404f9281de719667fdd2667ff1c46ab992e35e78c29` |
| `native_label_table.tsv` | `afc316a961bde53f618b440595bf6f4dbcba63f2755a10b926ad0c1efe01abb3` |
| `label_table_receipt.json` | `a83f1d4197e5b4a857ea305b135c32e9bbff5ec2794008372c6339a71162c2e2` |

Outputs and receipts are mirrored to the NVMe drive and checked by SHA-256.

Round 1b (2026-10-26):
- A1: option (d), at the maintainer's direction that only metals MAHOMES classified as catalytic count. The five
  training site tables list exactly the 882 selected sites, all marked catalytic; none of the 1,396 other metals in the
  structure files is in them. The charge proxy is recomputed in the campaign's data copy with those catalytic metals
  only, so other-site metals and the 13 same-site copies from other donors no longer count; every other feature value
  and every structure file stays unchanged. About 115 of the other metals lie 3–4.5 Å from a catalytic metal (likely
  the second metal of a two-metal centre) and are not counted either. PROPKA's term keeps the other metals for 134
  proteins (disclosed).
- One pocket per protein (added at the maintainer's direction: one pocket, not one per metal, no duplicates): every
  fit builds a single pocket per protein from all of its catalytic metals. With the trainer's default, catalytic
  metals within 4.5 Å share a pocket, which left 57 of the 740 proteins with two or three pockets; 45 of their 59 site
  pairs lie 4.5–12 Å apart, so those 10 Å pockets largely overlap. The new trainer option is used only by this campaign;
  the default and the metal task are unchanged.
- A2: no load failure, so WP-7 is not built. A5: Q96KN2 V113I kept and disclosed. A6: no short sequence (shortest
  126 residues). A13: the count of 370 is disclosed; file bytes unchanged. PROPKA "unavailable" for 475 proteins:
  kept and disclosed.
- Seed-spread stand-in for the simulation: the metal v3 seed-42/43 spreads, 1.5 points (17 late-fusion pairs) as the
  central point and 2.2 points (31 fold-0 pairs) as the upper point and pessimistic corner, each added in quadrature to
  the simulator's per-protein noise floor on the fold's population. The CARE two-seed value (1.8 points) is recorded
  for reference only.

**Code for the round-1b decisions (CP-B).** Reviewed and tested on synthetic data:
- `overlay build --d2-option d`: the overlay's external feature files carry the charge-distance proxy recomputed with
  the selected (catalytic) metals only, as the stored value minus the terms of every other metal. The build first
  checks that each selected site resolves to one trainer metal record and one feature-module metal atom of the same
  element within 0.001 Å, and that the stored value equals a recomputation with every metal present for every residue;
  any mismatch stops it. Every other value and every structure file is unchanged. The manifest marks each such file as
  derived, with its source and derived SHA-256, and `overlay verify` recomputes every derived value. A real overlay is
  refused unless its option equals the cohort audit's recommendation (d). On a synthetic structure with two selected
  metals, a same-site copy and two other metals, the derived values equal a full feature run without the other metals
  within 1e-12. Option b output is byte-identical to CP-A's.
- Trainer option `--selected-site-pocket-grouping {cluster,structure}` (default `cluster`; only with
  `--site-extraction selected_sites`): under `structure` each structure gives one pocket, the residues within 10 Å of
  any selected metal, with all selected metals as its metals; other metals never enter. The option is recorded in the
  run config, the cache keys, the site-extraction summary and the prediction receipts, and the loader refuses a second
  pocket of one structure. Campaign fits carry `structure`; `verify-unit` and the dry run check one pocket per protein.
  Under `cluster`, extraction and loading match `36498cd` byte for byte; every run config gains the one key, as
  `site_extraction` did in Phase 1.
- The cohort audit uses the structure grouping and writes schema 2; the overlay refuses the schema-1 audit above, so
  P2-3 runs again at the CP-B commit. The D2 numbers above do not depend on the grouping.
- Consequence of one pocket per protein: catalytic sites far apart form one multinuclear site (metal–metal edges,
  centroid), and a protein whose sites hold different metals has no metal label; the EC task does not use that label.

Next: P2-3 again at the CP-B commit, then P2-5 (inner folds), P2-6 (R2 lists), P2-7 (overlay, option d) and P2-8
(views, the layer-26 build, dry runs).

## evc-007

2026-10-29 — **cohort audit at the CP-B commit (one pocket per protein).** CPU only, training files only, at the clean
commit `e69a272`. The schema-1 audit of 2026-10-26 was moved to `cohort_schema1_20261026/` (unchanged); the new audit
wrote `cohort/` (schema 2, structure grouping). Console output went to private files.

`cohort-audit`, same inputs and expected values as evc-006, passed every check in 8.0 min (peak 0.7 GB; the secondary
drive is still on the FUSE driver). All 740 proteins give one pocket each (740 pockets; the default grouping would
give 798); 57 proteins had catalytic sites in more than one group and now have one pocket; no selected site lacks a
residue within 10 Å. Every other total equals the schema-1 audit's. The site table and both D2 tables are
byte-identical to evc-006's, so the D2 numbers and the A1 recommendation (d) stand; only `cohort_proteins.tsv` and the
receipt changed.

| Output | SHA-256 |
|---|---|
| `cohort_audit_receipt.json` | `3c10d49afeac52531fa8bfd47f8cd575c20edb0dab4ab3bc613b94f14915e2c5` |
| `cohort_proteins.tsv` | `0e7c180fe9ff06070b93649ba0b5e8edd4946fdce43d524a0f6baec09f9a010c` |

Outputs are mirrored to the NVMe drive and checked by SHA-256.

P2-5, same commit: `assign-inner` (seed 42, largest component share at most 1/4) and `certify-inner` for folds 0–4, and
the fold-0 R1-inner exclusion lists for inner folds 0–3. Console output went to private files. All five memberships are
certified with 0 cross-fold links and no other problem count; the largest inner component holds 4.5–5.7% of a fold's
eligible proteins. The fold-0 membership equals the gate G1 run (`862bbdcc…`). Every fold has at least 10 identity
components in each EC class 1–6, so the inner bootstrap is not stratified (rule D2 of the plan). Some inner folds hold
no class-7 protein; class-7 recall is reported only where it exists (B11; which folds is kept privately). Fold 0's R1-inner lists exclude 189–437 of 8,115 training proteins per inner fold, and no kept protein
is identical to a validation sequence.

P2-6: `r2-lists` wrote CLEAN's R2 lists for the five folds and the fold-0 R2-inner lists. Fold 0: 622 accessions in R2
against 603 DeepMzyme proteins; the 19 not in DeepMzyme all carry several EC classes. The other folds' sizes equal the
plan's and stay in the receipt. Every list loads with the comparator's reader, and no R2-inner list shares an accession
with the R1-inner exclusions of the same inner fold.

| Output | SHA-256 |
|---|---|
| `r2_lists_receipt.json` | `a86ef8e79ab8e9cc61b8c406af02e516f7da9b1a57c87f1d1b18fc6fe9f926b7` |
| `inner_membership_fold0.csv` | `862bbdcc4dcc49b79e5a4285cba21f981dd8dd589c4f174bf4fd009bed33b440` |
| `inner_membership_fold1.csv` | `d5f631327c59fa5e6625392230fb8ef63ad25047743813397ac8e5a59dfab0ea` |
| `inner_membership_fold2.csv` | `47a972f1872b4eef5520cfa5f79873b8e662f67d4ea296b1a9389bede593d72b` |
| `inner_membership_fold3.csv` | `cb52619ba7a6878357367ddcc7954b545d72cc17ad60f305462c38d7e752ebbd` |
| `inner_membership_fold4.csv` | `a4c78e9b0f33746d437f296cd846e8d46f6188c2a2563c5e2a3a6a272d3d699a` |

Outputs are mirrored to the NVMe drive and checked by SHA-256.

P2-7: `overlay build --d2-option d` built `ecc30-overlay-v1` from the cohort audit above, the identity folders and the
R2 lists in 15 min (peak 0.7 GB): 3,094 files for the 740 cohort proteins. The cohort audit recommends option d and
the overlay follows it. All 740 external feature files passed the parity check with every metal present (360,881
residues); 383 of them carry other metal atoms (1,396 in all, against 882 selected), and 49,767 residues have a new
charge-proxy value. The other 357 files are byte-identical to their sources. The string scan found 370 values, equal
to the cohort audit's (A13). `overlay verify` (19 min) rehashed every file, recomputed every derived charge proxy from
the overlay's own structures and site tables, re-created the tar and passed every check. The secondary drive ignores
file modes (FUSE driver), as recorded in the manifest. The tar is mirrored to the NVMe drive and checked by SHA-256.

| Output | SHA-256 |
|---|---|
| `ecc30-overlay-v1.tar` | `60d3a6ecfc50f68f3908fae6407ab9fb6a2f8604b2cad62fc1578ed9f479528a` |
| overlay manifest | `6f259f952aeb48692f741debed3c880c24b0ad25622f3536829346331cf5562f` |
| `ecc30-overlay-v1.build_receipt.json` | `c07e4a29b39217c3dc5721f284d2cfeb1900a705a72d78603502b299d7b407e3` |

P2-8, same commit, one fold at a time: `plan`, `build-view` from the overlay with the certified inner membership,
final-layer chain means from the residue files, block-26 chain means from the model on the CPU (`--layer-index 26
--device cpu`, two threads, 4 GB limit, nothing else running), the final-layer parity check between the two, `verify-view
--require-chain-means final,block26 --check-sources`, and `dry-run`. Console output went to private files.

All five folds passed every step. Each view's eligible proteins equal its certified inner membership: 603, 561, 545,
568 and 595 proteins for folds 0–4 (the views hold 622, 578, 562, 586 and 612 structures). The block-26 builds took
32–92 min per fold, about 5.0 h in all against a forecast of at most 4.2 h (peak 3.3 GB in every fold). The fold-3
build was stopped once at the maintainer's request after 106 of its 586 structures and continued with `--resume`,
which reused those 106 after matching their identity and source records. In every fold the final-layer means from
the model match the residue-file means on every structure (lowest cosine 0.9999999999997, largest absolute difference
2.5e-7; limits 0.9999 and 1e-4), and `verify-view` passed. Each dry run checked 356 commands at the clean commit
`e69a272` and ran the inner and refit probes; all ten probes passed with one pocket per protein. The secondary drive
ignores file modes (FUSE driver), as each view records. The view records, both chain-mean folders and the step
outputs are mirrored to the NVMe drive and checked by SHA-256.

| Fold | View manifest | Final chain means | Block-26 chain means | Dry-run receipt |
|---|---|---|---|---|
| 0 | `2d36e3bc…` | `32bc960f…` | `1b5af792…` | `91f4afbc…` |
| 1 | `c704be50…` | `55bcbaa0…` | `162622fb…` | `ccad7054…` |
| 2 | `50fd0517…` | `29c63f4b…` | `ed40fd71…` | `22b18db5…` |
| 3 | `bec69c7b…` | `4760211a…` | `63eca3ed…` | `c8887ff6…` |
| 4 | `b1b0fe59…` | `7da675f6…` | `1b6b6687…` | `2b5fe66a…` |

P2-9a, part 1, same commit: `simulate` on fold 2's population (545 eligible proteins, the smallest fold), read from
the overlay's copy of the certified membership. The sweep file is kept with the outputs (`simulation/sweep_v1.json`).
Per-protein noise alone gives fold 2 a seed-to-seed spread of 3.61 BA points (floor F). Round 1b's stand-in is added
on top of each noise point's own floor, so that it stays in BA points: target √(F² + s²), with s = 1.5 (central and
one-factor points) and s = 2.2 (upper point and the pessimistic corner). Rules: the current rule (gain 1.0 point,
α 0.10), α 0.05, gain 1.5, and both. One factor at a time around the central point: fit noise 0.4 and 1.2, class
recall −0.10 and +0.10, component and protein difficulty 0.5 and 1.5, and the upper fit shift. Also the L7 and L9
looks (2 and 4) at the central and upper points, and power at true gains of 1, 2 and 3 points.

Before anything ran in the campaign folder, a review pass checked the sweep file against the plan, the round-1b
decisions and the simulator code. Six of its findings were upheld and are built into this file. First, the stand-in
is applied on each point's own floor, as round 1b describes, not as a fixed logit shift. Second, the pessimistic
corner (part 2) is fixed in full in the sweep file: rules, looks, replicates, the rule for picking each factor's end
with a margin of two standard errors (a factor below the margin keeps both ends, and D1 reads the highest one-look
adoption across the resulting corners), and where D1's 15% clause for L7 and L9 is judged. Third, the replicates per
point were raised over the plan's 200, to 1,000 for one-look null points and 400 for the others. The first two timed
points took 0.2–0.4 CPU seconds per replicate, so the whole sweep stays far under the 8 CPU-hour limit. Fourth, every
record's floor must equal the value precomputed for the file, or the sweep is void. Those two timing points ran in a
scratch folder from the first draft; their adoption rates were seen before this file was final.

Part 1 ran 68 points in 55 min (1.66 CPU hours, peak 0.12 GB). Every record's floor equals the value precomputed for
the file. One-look null adoption, current rule: 9.4% at the central point and 12.7% at the upper fit shift; with
α 0.05: 6.4% and 8.0%. The one-factor ends move the four-rule mean by at most 1.8 points. Raising the gain floor to
1.5 points changes no adoption anywhere: every replicate whose lower bound passed already had a gain above 1.5, so α
is the only tightening that acts. The recall guard vetoes about 35% of null replicates. With two looks (L7) and four
(L9), the current rule adopts 14.2% and 21.0% at the central point; with α 0.05, 9.0% and 14.2%; at the upper point,
about 5–8 points more. Power at the central point, one look: a true gain of 1, 2 and 3 points is adopted 14.8%, 23.2%
and 26.0% of the time under the current rule, and 9.8%, 17.8% and 18.8% under α 0.05. By the fixed rule, only protein
difficulty has a clear worse end (1.5); the other three factors keep both ends, so part 2 has eight corners.

| Output | SHA-256 |
|---|---|
| `sweep_v1.json` | `0b5279d775a18af04c57330725d3370e970126bbe1b1331aa1c348146449c91e` |
| `sweep_v1/sweep_receipt.json` | `9f1d8ff47e209f2958cdd4375da34debae174ef2b5c97ff53a8a756d9f258e1d` |
| `sweep_v1/summary.tsv` | `e726d982c36a7439a92cd5e034125c9fd1c483617ae82a9fe54a8993d02b5731` |

Part 2 ran the eight corners (protein difficulty 1.5; both ends of fit noise, class recall and component difficulty;
s = 2.2 on each corner's own floor) in 75 min (2.33 CPU hours). Every record's floor equals the precomputed value. The
highest one-look null adoption across the corners: current rule 14.3% (143 of 1,000; 95% interval 12.3–16.6%), α
0.05 10.1% (101 of 1,000; 8.4–12.1%); the gain-1.5 rules again equal their gain-1.0 counterparts. No rule reaches
10% or less at the corners with one look, so the plan's stop condition for P2-9 applies and the choice goes to the
maintainer. Under α 0.05 the corners give at most 14.5% with two looks (L7) and 21.8% with four (L9); under the
current rule 22.8% and 31.5%. A true gain of 2 points is adopted 16–20% of the time under α 0.05 and 24–27% under
the current rule. The maximum over eight corners, each estimated with 1,000 replicates, reads high by about one
standard error on average; that is part of the rule as fixed.

| Output | SHA-256 |
|---|---|
| `sweep_v1b.json` | `514ac13b25ddee3785bdbe6cbc934f4b96074237b9d1d9b65f267de8a4d0c4da` |
| `sweep_v1b/sweep_receipt.json` | `3a148fc1188ec65f8646f747a55b057f9858324ac3a71c8af7baa5cebbfa92ba` |
| `sweep_v1b/summary.tsv` | `2b2faa7cac7ff02723af3a5a1d6bca87461f50688a7e7a0cb23d37e85bf85779` |

**Plan amendment P2-9-A1 (2026-10-30, maintainer's choice after the stop).** The maintainer chose a ladder of smaller
α, fixed before it runs, with a confirmation on fresh simulation draws, and the plan's seed-variance term as the
fallback. The gain floor stays 1.0, since 1.5 changed nothing. Selection (`sweep_v1c`, the draws of parts 1 and 2):
one look at α 0.04, 0.03 and 0.025; L7 under each α and α/2; L9 under each α and α/4; power at the central point and
the corners. An α passes when its highest one-look null adoption across the eight corners is at most 10%, and L7 and
L9 are at most 15% under α or, failing that, under the divided α. The largest passing α is selected. Confirmation
(`sweep_v1d`, simulation seed 20261030, every corner's floor and target recomputed on those draws): 2,000 replicates
per one-look point and 800 per multi-look point. The selected α must meet the same criteria there; if it fails, the
next smaller α that passed selection is judged on the same draws. If none holds, the seed-variance term is built,
reviewed and committed (CP-B2) before the corners are rerun. The plan also records what D1 sets in code at CP-C.

Selection (`sweep_v1c`) ran 153 points in 80 min (2.53 CPU hours, peak 0.12 GB). Every record's floor equals the
precomputed value. All three α pass, so α 0.04 is selected. Highest one-look null adoption across the corners: α 0.04
9.0% (90 of 1,000; 95% interval 7.4–10.9%), α 0.03 7.4% (5.9–9.2%), α 0.025 6.9% (5.5–8.6%). L7 passes undivided at
every α: 13.5%, 11.2% and 9.8%. L9 fails undivided at α 0.04 (18.8%) and 0.03 (16.0%) and passes divided, at 0.01
(9.5%) and 0.0075 (8.2%); at α 0.025 it passes undivided at 14.8% (11.6–18.6%). Selected levels: α 0.04 one look and
L7, α 0.01 for L9. The recall guard vetoes 35% of null replicates at one look, 26.5% at two and 21.0% at four. Power,
one look: at the central point, a true gain of 1, 2 and 3 points is adopted 9.0%, 14.0% and 15.8% of the time under
α 0.04 (6.8%, 10.5% and 13.2% under α 0.025). A gain of 2 at the corners is adopted 14.2–17.5% of the time per
corner under α 0.04, 12.8–15.8% under 0.03 and 10.8–15.0% under 0.025; each corner's 95% interval spans about ±3.6
points. The corners share simulation draws (one seed), so they are reported one by one and no pooled interval is
formed.

| Output | SHA-256 |
|---|---|
| `sweep_v1c.json` | `f1b0f1bbd8efc921381ba6ccaa1176d3dbca0d6779cd034e9f80b0e6c1b664de` |
| `sweep_v1c/sweep_receipt.json` | `b444cb28c3c22e02d8e6ce3f9aab85a58bed1bbefce4e2f2d8e00d2257dfd930` |
| `sweep_v1c/summary.tsv` | `c871fe5978119a35cddcd06590e1e8cb6f3ca2a3d19398bc3d3d77726d978476` |

Confirmation floors on seed 20261030 (F_c, then target): fnL-rcL-csL 3.4306/4.0755, fnL-rcL-csH 3.1424/3.8360,
fnL-rcH-csL 3.3706/4.0251, fnL-rcH-csH 3.0496/3.7604, fnH-rcL-csL 3.5814/4.2032, fnH-rcL-csH 3.3514/4.0090,
fnH-rcH-csL 3.5669/4.1909, fnH-rcH-csH 3.2540/3.9280 (protein difficulty 1.5 at all). `sweep_v1d.json`:
`cc613f49d367b8d9f5c2160f063e3141129420fa62d6f7dc6852d45b3ff111f7`; 96 points, 96,000 replicates.

Confirmation (`sweep_v1d`) ran 96 points in 71 min (2.24 CPU hours, peak 0.12 GB), and every record's floor equals
the precomputed value. **α 0.04 is confirmed and is D1's rule: one look and L7 at α 0.04, L9 at α 0.01.** The smaller
α also hold on these draws.

| α (L7, L9 level) | One look, highest across corners | L7 | L9 | Gain of 2 points, one look, per corner |
|---|---|---|---|---|
| 0.04 (0.04, 0.01) | 6.8% (135/2,000; 5.7–7.9) | 13.9% (111/800; 11.7–16.4) | 8.9% (71/800; 7.1–11.0) | 13.5–15.8% |
| 0.03 (0.03, 0.0075) | 6.0% (5.1–7.2) | 11.6% (9.6–14.0) | 8.0% (6.3–10.1) | 11.0–14.2% |
| 0.025 (0.025, 0.025) | 5.2% (4.3–6.3) | 10.2% (8.3–12.5) | 13.4% (11.2–15.9) | 9.0–13.2% |

The recall guard vetoes 33.8% of null replicates at one look, 26.3% at two and 20.8% at four. Each per-corner power
value carries a 95% interval of about ±3.6 points; the corners share draws, so no pooled interval is given.

**Low power, disclosed.** Under the confirmed rule, a true gain of 2 BA points is adopted about 14–16% of the time
(selection draws: 14.0% at the central point, 14.2–17.5% at the corners; confirmation draws: 13.5–15.8% at the
corners). A gain of 3 points is adopted 15.8% of the time at the central point. The noise stand-in (about 4 points of
seed-to-seed spread on fold 2) limits power more than α does. The rule mainly keeps null or harmful levers out.
The simulation calibrates one promotion decision at 1, 2 or 4 looks. It does not validate the learning-rate choice,
the family escalation, the screening of up to nine levers, the S3 combination, or real model performance. The inner
promotion α is separate from the claim levels.

| Output | SHA-256 |
|---|---|
| `sweep_v1d/sweep_receipt.json` | `df8ac4bcee89e45f790c68693640f062d2f53f28bae06346c68fdd2da903de3b` |
| `sweep_v1d/summary.tsv` | `84c5f3f3268950fdf7f05ff65c9bf55dd433f4b6382a133d2f24a93369ccab97` |

**Independent review and two amendments before Freeze 1 (2026-10-30).** A review of the scorer, the selection code
and the trainer raised four points. Each was checked against the code on synthetic tables only.
1. **Bootstrap support.** With every EC class 5 protein in one homology family, the stratified bootstrap gave p = 0
   and a lower bound equal to the estimate (+16.67 points). Thirty independent families gave the same output.
2. **Class scope.** When classes were absent, the record said "2-class BA" while the decision sentences and one
   heading read as the six-class endpoint.
3. **L5 coverage.** The trainer kept only the loss, so nothing recorded the share of minibatches with SupCon
   positives that L5 requires.
4. **Power.** The promotion rule's power is low, and the pooled power intervals ignored the corners' shared draws.

The maintainer's decisions:
- **Analysis amendment.** A population whose bootstrap has a stratum of EC classes 1–6 with a single homology cluster
  is reported descriptively, with its superiority inference marked unavailable (class-7-only strata are exempt for
  the primary BA).
  - Clarification (2026-11-01): the gate is applied per metric. A single-cluster class-7-only stratum also makes the
    secondary accuracy comparison descriptive, and the macro-F1 comparison when a compared system assigns one of its
    proteins to an EC class 1–6 present in the population. Those hypotheses stay in the secondary Holm family,
    marked unavailable, with their point estimates.
  - No class or cluster is removed to recover eligibility.
  - Class and cluster support and the stratum sizes are reported for every population.
  - Both P1 and P2 name the classes they evaluate and say when the six-class endpoint was not evaluated.
  - An unavailable population stays in the two-hypothesis Holm family, so the other is still tested at 0.025.
- **Promotion.** Use the confirmed rule and disclose its low power. Add no screening seeds and do not expand the
  simulation. Inner promotion levels stay separate from the claim levels. Power is reported per corner.
- **L5.** Count every training minibatch of a SupCon fit and those with a positive, using the loss's own candidates
  (the other batch items and the memory, other proteins only).
  - The counts are kept in the fit history (checkpoint and `run_config.json`) and summed into the prediction receipt.
  - L5 counts as tested only if each required inner fit reaches 90%. Missing counters leave it uncertified.
  - This package records the counts only; end-to-end L5 certification is not in place yet. The selector built at
    CP-C must derive the full grid of required fits (four inner folds × two seeds) from the frozen plan and reject
    missing, duplicate or foreign fits. It must check complete, valid per-epoch counters against the hash-bound
    checkpoint and run records, recompute the totals, and reject a prediction receipt whose coverage disagrees.
    It then applies the minimum coverage across all eight fits.
  - The change touches three trainer files that the source-freeze rule protects, so the rule is amended for them
    and the five dry runs are repeated at the new commit. Views, chain means, overlay, identity chain and simulation
    records are reused unchanged.

Regression evidence for the change, before the commit. A tiny synthetic CPU parity fit was run at `e69a272` and
with the change: plain, L5, and L5 with the L4 heads, four epochs each. The per-epoch training and validation losses
and every shared state-dict tensor are bitwise identical. The state-dict keys are unchanged, and the history gains
only the two counters. Checkpoints written by either version load strictly with the other. The targeted tests
(193), the campaign tests, the smoke checks and the full suite were run on the change; their counts are in the
commit message. The five dry runs are repeated after the commit.

Smoke-check count, corrected. One smoke check is skipped on this machine: the multi-metal site-granularity check
needs a local Colab-bundle copy of a PinMyMetal training CSV, which is absent. Its two inputs in the training folder
are present. The bundles folder has not changed since 2026-09-14, so the check was skipped at CP-B as well. The CP-B
entry's "45 smoke checks passed" came from the script runner, which prints its total even when a check is skipped;
the correct count was 44 passed and 1 skipped. Smoke counts are now taken from pytest, which reports skips. Nothing
was regenerated and the check is unchanged. The cause was found from the check's source and its skip message. Only
existence tests on the exact paths it names (and on that bundle's folder) and the bundles folder's modification time
were used; no folder was listed and no file was opened.
