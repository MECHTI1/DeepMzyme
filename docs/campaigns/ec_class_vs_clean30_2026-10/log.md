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

## evc-008

2026-11-01 — **dry runs repeated at the SupCon-coverage commit, environment exports, and the Freeze-1 code (CP-C).**
CPU only, training files only.

The maintainer committed the evc-007 package as `451c190`. Because it changed three trainer files under the
source-freeze rule, the five dry runs were repeated at that commit on a clean tree (the secondary drive on the same
FUSE driver as when the views were built), one fold at a time under the 1 GB parent limit, each probe in its own
3 GB scope. Every fold checked its 356 planned commands with no static failure, and both probes passed (one pocket
per protein). Each run took 10–12 min (parent peak 1.3 GB). Each new receipt records the SHA-256 of the P2-8 receipt
it supersedes. Before each run the P2-8 receipt was checked against its NVMe copy and checksum list and kept as a
dated copy; the new receipts and console files are mirrored and checked with `sha256sum -c`. Rule R-CMD restarts at
`451c190`.

| Fold | Dry-run receipt SHA-256 |
|---|---|
| 0 | `f71214bd98cfd5b042549d4a0bd0e2922aa5a573c65288e8548219488dff495a` |
| 1 | `e19350c09c7cc01f251ab46428c7ecaa08abeb4b1495c4dbde7e0c29c1ce0b5d` |
| 2 | `422128fd0e3c3872b9c011fa02528ee740f1cf1e044df456851de255727ff5a0` |
| 3 | `657ee6a60ef268df8d3ce9416085b1bc86b667fb964c3cf8931e8e5cbb658283` |
| 4 | `e27ce825fd1c313d5f624282d01b5ddb1c7b02bed6c5c2dd678fce4c21c3d805` |

P2-9c environment exports (G5) of the campaign environment, mirrored and checked:

| Export | SHA-256 |
|---|---|
| `freeze/env/env_conda_explicit.txt` (`conda list --explicit --md5`) | `c809b698ea3434f66946db2ffaaedb8e6b42b0b7988a3c6050309cdfdac6ecee` |
| `freeze/env/env_pip_freeze.txt` (`pip list --format=freeze`) | `8c64efba609359e896046aed4d1b6357b7b479515a56b815dad59ce4615b1c4f` |

**Freeze-1 code (CP-C), staged for the maintainer's review.**
- D1 in code: `PROMOTION_ALPHA` 0.04 for every promotion including L7's two looks; `L9_PROMOTION_ALPHA` 0.01 through
  L9's own `PromotionConfig` (`promotion_config_for`). No plan SHA-256 changes. A counter given as a boolean no longer
  counts as an integer in the coverage rule.
- WP-12, `transfer-list`: reads the published ENZYME nomenclature file into `ec_transfer_list.csv` in the scorer's
  comma-separated format (the plan's `.tsv` would fail the scorer's header check) and a receipt. Transfers are
  followed to their current entries; a missing target, a cycle or text outside the grammar stops the run; the
  scorer's own reader must read the list the same way. Keyed by EC number; no label is read. The download (A8) waits
  for the maintainer's approval.
- WP-9 and WP-10 (`ec_clean30.freeze`): the real Freeze-1 inputs manifest (exactly folds 0–4, inner membership and
  certificate checked against the overlay's identity copies, one overlay, a smoke overlay refused, per-file SHA-256
  of every tracked `src/` and `scripts/` file) with the R-CMD check (changed files against the allowlist from each
  view's dry-run HEAD, the twelve pinned runner functions with their decorators, a fresh-interpreter import check
  for added modules, and each fold's live plan against `$C/plans`); `freeze1-document` (decisions answered, no
  placeholder or absolute path, live constants including both promotion levels, records bound by SHA-256 only, no
  per-fold class count); `verify-freeze1`; and `freeze1b`.
- WP-11 (`ec_clean30.selector`): `select` for S1, S2 and S3 and the fold-0 `gap-gate`, each appending to or reading
  the hash-chained step records anchored on the Freeze-1 SHA-256. L5 is certified as the maintainer required:
  1. The plan's eight fits (four inner folds × two seeds) come from the frozen plan, which must equal the live plan.
     A missing, duplicate or foreign fit refuses the step.
  2. Each fit's checkpoint, run configuration, predictions and identity are bound by SHA-256 to its prediction
     receipt and the runs folder's verification receipt. The checkpoint's per-epoch counters must equal the run
     configuration's.
  3. The totals are recomputed from that history, and a receipt whose coverage disagrees refuses the step.
  4. The lever's fraction is the minimum over the eight fits, with the 90% rule. Counters that are absent or invalid
     on an intact fit leave L5 "not tested".
  These checks are tested on synthetic fits. L5 is certified for a fold only when its eight real Phase 4 fits pass
  them.

Choices the code makes where the plan is silent are listed for round 2 (S3 with L5, L3 without fits, gap-gate
details, root-level files under R-CMD, required pins and decision identifiers). The runner, the playbook recipe and
the DATASETS overlay entry are updated. Checks with the package staged: full suite 3,103 passed and 18 skipped (the
same skips as before); campaign tests 979 passed and 1 skipped; smoke 44 passed and 1 skipped (the absent local
training CSV of evc-007); docs contract without strict failures.

## evc-009

2026-11-02 — **CP-C completion and round-2 decisions before Freeze 1.** CPU only. The maintainer approved
completion through Freeze 1b, the ENZYME source download, publication of the reviewed campaign commits and the
maintainer-owned CLEAN wrapper, and acknowledgement name `mechti`. Private CLEAN source and weights remain private.
The optional metal-release parity check is omitted. The bounded Drive filename search is recorded in
[DATASETS](../../DATASETS.md#test-use-ledger); no matching accessible output was found and no contents were opened.

The five dry-run receipts at `451c190` match their mirrored bytes and the environment exports match evc-008.
They are reused subject to the real Freeze-1 source and plan checks. The root STATUS still describes CP-B: changing
that protected root file before Freeze 1 would invalidate the dry-run source rule, so current progress is maintained
here and in the campaign README while that discrepancy remains explicit. No source-freeze rule is weakened.

The remaining selection changes are limited to authenticated L3 cost dispositions, reuse of S2 fits when L9 combines
with one fitted lever, descriptive S3 SupCon coverage, and certified gap-gate membership. The one-point S3 discount
applies only to the primary R1 EC1–6 gap. The existing eight-fit S2 SupCon gate, promotion levels, recall guards,
folds, seeds, model families and terminal-checkpoint policy are unchanged. A completed L3 grid is always screened;
missing fits alone never imply a cost skip. The later execution coordinator must record the ceiling, actual use,
forecast, attempted-unit evidence and authority before closing a partial L3 grid without screening it.

**Final simulation specification (resolved before results).** P2-9b had not specified its noise-point set. It retains
the eight declared pessimistic corners on each of five overlay training populations, with 400 replicates at looks
1, 2 and 4: 120 null points, 48,000 replicates. Simulation seed 20261024, bootstrap seed 20261023, 10,000 draws,
ordinary/L7 alpha 0.04, L9 alpha 0.01, gain floor 1 point and the existing guards. Fit noise is 0.4 or 1.2, recall
offset −0.1 or +0.1, component difficulty 0.5 or 1.5, and protein difficulty 1.5. Each fold/corner recomputes its
noise floor F over 1,000 populations and uses `ceil(sqrt(F*F + 2.2*2.2)*10000)/10000` for the seed-spread target.
No other alpha, seed, noise range or power point is searched; the prior confirmation and fold-2 power evidence stay
separate. Per-point intervals are reported without pooling correlated draws. The policy thresholds are 10% for
one look and 15% for L7/L9, not the simulator's generic 10% flag; material contradictions require review without
automatic retuning. The existing eight-CPU-hour forecast safeguard remains in force.

**Verification at final CP-C source.** Campaign and comparator tests: 1,041 passed, 1 skipped. CPU smoke: 44 passed,
1 skipped (the existing missing training fixture). Full suite: 3,155 passed, 18 skipped and 15 subtests passed,
in 15 min 42 s; no new skip. Two inconsistent synthetic L9 fixtures were corrected before this final run.
Documentation contract: no strict failures, the same 11 baseline warnings. The current conda and pip exports are
byte-identical to the environment records of evc-008. SupCon certification is still synthetic-test evidence, not
certification of any real Phase 4 fit. The ENZYME source (release 02-Sep-2026) was downloaded on 2026-11-02, SHA-256
`6e1f553fcaa15ba7018344d4181012002547cf15febeb9d3a03dacdd75963001`; parser preflight found 8,472 records and
1,308 resolved transfer rows. The source and download receipt are mirrored and byte-verified.

## evc-010

2026-11-02 — **final fixed-rule simulation, transfer list and Freeze 1 (CP-D).** CPU only, at the clean
CP-C commit `d96527f`. P2-9b completed all 120 null points and 48,000 replicates in 58 min (1.80 CPU hours,
peak 0.13 GB), including the initial timed point reused by the full run. The eight-CPU-hour ceiling was respected.
A separate verification matched every record, result, source and calibration hash, recomputed the 40
fold/corner floors and targets, and regenerated the summary byte for byte. The outputs and verification record are
mirrored to NVMe and checked by SHA-256. No new power point, alpha, seed or noise range was searched.

**Operating characteristics are not a uniform target pass.** The table gives ranges across the 40 points at each
look count, the number exceeding its point-estimate target, and the worst point's individual 95% Wilson interval.

| Comparison | Target | Observed range | Above target | Worst-point interval |
|---|---|---|---|---|
| Ordinary, one look | 10% | 4.75–12.0% | 6/40 | 9.2–15.6% (48/400) |
| L7, two looks | 15% | 9.5–18.0% | 11/40 | 14.5–22.1% (72/400) |
| L9, four looks | 15% | 5.75–14.0% | 0/40 | 10.9–17.7% (56/400) |

The original selection and independent-confirmation population, fold 2, remains within the targets at every final
corner: maxima 8.75%, 13.5% and 9.5% respectively. The excesses occur on the other folds. With 400 replicates per
point and shared draws, uniform target control across folds is not established; some underlying rates may exceed
their targets. The modest excesses at maxima across 40 points do not establish a substantial reversal of the earlier
fold-2 confirmation. The already confirmed D1 levels are retained without retuning, as fixed before this grid.
No individual interval lies wholly above its target, but that is not an acceptance rule or proof of target control.
The verification record retains `all_points_meet_policy=false`; evidence integrity passed separately. No pooled
interval is formed. These simulations concern individual promotion decisions, not family-wise control of the
complete selection pipeline or real-model superiority; the low-power disclosure in evc-007 remains applicable.

P2-9d at the same commit resolved the approved ENZYME source into 1,308 transfer rows: 214 cross-class and
1,094 within-class transfers. Six source entries resolve entirely to deleted entries and are omitted; 16 chains
were resolved. The scorer reader agrees with the emitted CSV. Source, download receipt, list and parser receipt
are mirrored and verified.

| Evidence | SHA-256 |
|---|---|
| Final sweep | `51f1d48af4c0295f0e962ffe6b012b4d1dad9a91b21f6a39715a488f2d43c7d4` |
| Final sweep receipt | `d3685c31084cb603c3bdfb495c2e3f36ffd6294a9acbafd07b819a023e6c11aa` |
| Final summary | `cf3b80437fd3d3e0063390bcd7d9b3d328ddacc3f2e9b7009ac6804169acbc84` |
| Verification record | `93298ff914f70e739dc38a39e56a42c99fd9bcf3f49b41a936a9e9f1c117aa31` |
| Transfer CSV | `84ec98b484fd8b048ec73a5344b20efc29924ee6726d9ecef448ee33a31de2d7` |
| Transfer receipt | `adcf7e707754b54091c44e3df3a349eba2ff7d0f2a7ed159f274d322609e911f` |

The real Freeze-1 inputs manifest verified all five views, their certified overlay memberships, unchanged plans and
dry-run receipts, and the R-CMD source restrictions at CP-C. The document binds 129 answered decisions, those inputs,
the final simulation, label/cohort/R2 records, environment exports, CLEAN harness pins, ESMC weights and MMseqs2.
Identity source anchor (P2-1, R-ID): `697427e40e18aec018b66cb86cbd3a095fda981b`.
[Freeze 1](freeze1.md) has SHA-256 `28077f4c3701820c9aa20f53b570d12f8b70a46514211fc63e62976e393a3c0c`
for [the canonical JSON](freeze1.json). No campaign complement audit or Phase 7 scoring preceded this freeze.
The publication check must pass before the label-blind identity audit starts; GPU work remains outside this continuation.

## evc-011

2026-11-02 — **published Freeze 1, label-blind identity audit and Freeze 1b (CP-E).** The reviewed campaign commits
through CP-D `ea8417a` were pushed to `origin/main`. After refreshing the remote reference,
`verify-freeze1 --require-pushed` passed for the unchanged Freeze-1 bytes; its receipt SHA-256 is
`7ea5babe8204f076a0400811ea0a62b3e942093629239217086a962884b1fa17`.

P2-11 then ran at that clean commit under the two-core, 3 GB, zero-swap cap: complement FASTA, search and edges for
each fold, cohort components, then all five strata tables. All 21 steps passed in 70.4 s, with no refusal.
No fold-work remapping was needed. Only counts were displayed; no protein identifier, EC label, prediction or
score was displayed. The audit derived test-side cohort complements from the approved training-source columns.
It made no model prediction, score or selection decision.

| Fold | Complement proteins | No hit | ≤20% | (20,25]% | (25,30)% | [30,50]% | >50% |
|---|---|---|---|---|---|---|---|
| 0 | 118 | 14 | 0 | 6 | 30 | 57 | 11 |
| 1 | 162 | 23 | 0 | 7 | 34 | 83 | 15 |
| 2 | 178 | 34 | 1 | 11 | 40 | 78 | 14 |
| 3 | 154 | 22 | 0 | 8 | 44 | 68 | 12 |
| 4 | 128 | 17 | 0 | 6 | 32 | 60 | 13 |

These are cohort-complement identity counts, not single-label evaluation counts. The cohort has 740 proteins in
307 components; the largest component has 39 proteins (5.27%). No class-support or performance claim follows
from these label-blind counts.

The identity folder is now test-bearing. It was mirrored as one deterministic tar without listing its members;
only the archive checksum was read back and verified. `identity_after_p2-11.tar` has SHA-256
`fd9cd6b54254138bf344f69364db8245751554f292aecf39b1722a1c841fb66e`.
Later development uses the frozen inner identity copies in the overlay, not this folder or its archive.
[DATASETS L3/L4](../../DATASETS.md#test-use-ledger) records the access and restrictions.

P2-12 `manifest --kind freeze1b` passed at `ea8417a`: 55 receipts and 75 outputs across five folds, the published
Freeze-1 binding, matching identity code, unchanged R-ID files from `697427e`, and the cohort/strata links.
The private `freeze/freeze1b.json` has SHA-256
`fc6f032805f084d7aef4f704a42149e92fb7894af3699b2ad5c12e86c87bcbdb`. It and the publication receipt, count summary and console
records are mirrored and verified. Freeze 1 remains `28077f4c3701820c9aa20f53b570d12f8b70a46514211fc63e62976e393a3c0c`.

The approved CPU continuation through Freeze 1b is complete. The CP-C test evidence remains applicable because no
source changed afterward: full suite 3,155 passed, 18 skipped, 15 subtests passed; campaign/comparator 1,041 passed,
1 skipped; smoke 44 passed, 1 skipped. Documentation checks have no strict failure and the same 11 baseline
warnings. The final simulation limitations in evc-010 remain explicit. No GPU, Phase 4 fit or Phase 7 scoring ran.
WP-13 and G2 remain later work outside this continuation. Root STATUS retains the disclosed stale state of evc-009.

## evc-012

2026-11-02 — **approved S1 fixed-anchor amendment, implementation and bounded characterization.** Before any
campaign model comparison, the maintainer approved the rule in [the design](design.md): three direct comparisons
against F2, fixed Holm family alpha 0.04, existing gain/recall guards, and the 0.5-point F5/F3/F4 tie preference.
Historical escalation, S2/S3, R2 scoring and its seven-hypothesis secondary family remain unchanged.

The characterization specification was fixed before results: five existing fold populations, the recorded
`fnL-rcL-csL-psH` and `fnH-rcH-csH-psH` noise settings, and five effect patterns (null, F4-only +2/+3 BA points,
mixed F3 benefit and mixed F5/F4 benefit). Both LRs and both screening seeds for all six families are simulated;
the actual LR chooser and amended selector run in every replicate. Fifty scenarios, 200 replicates each, 10,000
bootstrap draws, seed 20261102; no alpha search or outcome-driven scenario expansion. The existing noise records
are authenticated against the original Freeze-1 simulation receipt. No new identity or test read is needed.

The first scenario used 77.94 CPU seconds; its 25%-margin forecast was 1.35 CPU hours. The existing eight-CPU-hour
ceiling and two-core/3 GB/zero-swap scope remain enforced. The first scenario is reused. Source snapshots and
per-scenario hashes make the development characterization reproducible before the amendment is committed;
this does not grant training authority.

All 50 scenarios completed (10,000 selections) in 1.115 CPU hours, 33.59 minutes of summed scenario wall time,
with peak RSS 478.19 MiB. The [fixed specification](s1_amendment/simulation_spec.json) and
[complete summary](s1_amendment/simulation_summary.json) retain every outcome. Summary SHA-256:
`d765f615d61d7d51b2c2b27855a3fba52c710e640c49b6c195be58695cee6624`.

| Scenario pattern | Selection rate across the ten fold/noise settings |
|---|---|
| All challengers null | Any challenger: 3.0–7.0% |
| Only F4 benefits by +2 BA points | F4: 3.0–9.5% |
| Only F4 benefits by +3 BA points | F4: 7.5–16.5% |
| Mixed F3 benefit | Non-beneficial challenger: 1.0–3.5%; harmful: 0–0.5% |
| Mixed F5/F4 benefit | Non-beneficial challenger: 0–2.5%; harmful: 0–0.5% |

The worst observed null setting selected a challenger in 14/200 replicates (7.0%; individual Wilson 95% interval
4.22–11.41%). These are final-selection rates, not the probability that any false hypothesis passes. They do not
establish uniform family-wise error control. The two noise corners and 200 replicates per setting provide bounded
characterization; correlated scenarios must not be pooled for a confidence interval. No alpha or scenario was
changed after results. F4 is reachable under this rule, but the measured selection power remains low.

There were 825 bootstrap aborts among the 10,000 selections (8.25%; 3–38 of 200 per scenario). The same population
draws repeat across the five effect patterns. These arise when too many resamples lack a supported EC class.
The simulation counts them separately and includes them in the F2 count as no challenger selected; production
selection raises an error instead of treating an abort as a successful F2 selection. No guard was relaxed.

All 50 scenario record hashes and 13 simulation source pins verified. The 67 output/snapshot files were mirrored
to the existing NVMe backup and verified byte for byte.

Implementation verification passed: 443 focused tests; CPU smoke 44 passed and 1 skipped. The full suite reported
3,227 passed, 18 skipped and 15 subtests passed, plus one source-tree hash failure because the new simulation
module was not yet indexed. Staging that file resolved the mismatch; the exact failing test then passed.
Source bytes were unchanged throughout these checks, so the remaining full-suite results were reused.
The existing fixture, closed-campaign and opt-in skips remain explicit in the preserved test logs.

The amendment authenticates the exact five changed/new source files, the fixed policy, completed characterization
and verification record. Real selection and gap records reject missing or mixed amendment chains and unpublished
source. Historical selection remains available for replay. The original 217 unamended source pins, twelve
preparation command functions and all five plan/dry-run bindings verified; Freeze 1 and Freeze 1b retain their
evc-011 hashes. No view, overlay, embedding or historical simulation was regenerated. R2 scoring is unchanged.

The [canonical amendment](s1_amendment/amendment.json), [rendering](s1_amendment/amendment.md) and
[verification record](s1_amendment/verification.json) are complete. Amendment SHA-256:
`d07dbb6a68de7f441a0030661f110f2de9dabd4786a702a1e23d92a3a7a242ea`.
The documentation contract passes with zero strict failures and the same eleven baseline warnings.
This reviewed package still requires commit/publication before real execution. The next deliverable is the
CPU-tested WP-13 package in [the continuation](continuation.md), followed by the separately approved G2 smoke.
No GPU work, campaign model training or test scoring occurred in this amendment.

## evc-013

2026-11-02 — **S1 publication and WP-13 CPU execution preparation.** The reviewed S1 package was committed and
pushed as `3154a7f`; its clean published-source verification passed before WP-13 development. No new simulation,
selection rule, scoring rule or secondary multiplicity change was introduced.

WP-13 adds `src/ec_clean30/phase3.py` and its CLI, reusing the frozen command/view code and existing execution
helper. It implements exact Fold-0 transfer, deterministic timing inputs, source authentication, relocated
probing, timed training/replay, the fixed lane diagnostic, both embedding timings, host acknowledgement and
Freeze 1c assembly. The [execution manifest](wp13/execution_manifest.json) pins only the two added source files;
all source already published at `3154a7f` remains unchanged. Original Freeze 1/1b retain the evc-011 hashes.

The prepared archive contains 4,366 files, 1,485,957,120 bytes, SHA-256
`a036a782e3f0de041e896b7e2ff3a79dd13eb6d827629b0eca28dfcbc9b90334`.
Its original dry-run receipt is preserved. Exactly 1,000 distinct official Fold-0 training sequences were selected
by the declared hash/accession ordering; maximum length 1,021, total residues 437,329. Their receipt SHA-256 is
`ee368d99a5e88a6b5f84c3cd131b8d043432d41cea59c5266beb1d9fac48ff2f`.
Existing views, overlay, embeddings and simulations were reused, not regenerated.

Focused checks passed 98 tests; CPU smoke passed 44 with one existing missing-fixture skip. A synthetic five-epoch
F4 fit passed training, terminal verification, independent replay, 26-file host persistence, acknowledgement and
verified reuse in 10.86 seconds for the execution cycle. This is engineering evidence only. Draft authentication
passes; real execution correctly refuses unpublished WP-13. The full suite passed 3,311 tests, with 18 existing
skips and 15 subtest passes, in 713.93 seconds. The [verification record](wp13/verification.json) binds these logs
to unchanged source/test bytes. Documentation checks passed with zero strict failures and eleven baseline
warnings; all 26 pre-existing playbook blocks are preserved. Initial checks corrected a read-only extraction-directory
rename failure and verified the existing MKL GNU threading requirement; the final focused run is clean.

CPU logs, deterministic timing inputs and the synthetic roundtrip archive are mirrored and verified on the
independent NVMe store. The large transfer archive is a reproducible derivative of the already mirrored original
view. Completed old synthetic pytest directories were removed after preserving their evidence; no scientific
artifact or backup was deleted. [G2 readiness](g2_readiness.md) owns current storage sizing, the proposed existing-L4
session and its explicit prerequisites. Weights and the isolated CUDA embedding environment still need preparation
before billing. No GPU start, scientific fit, private weight/environment download or test scoring occurred.
Next: publish WP-13, complete approved G2 preparation/measurements, then set numerical G3 ceilings and finish the
minimal Phase-4 dispatcher for the unchanged 96 S1 plus eight CLEAN inner fits.

## evc-014

2026-11-02 — **WP-13 publication, pre-billing preparation and Fold-0 automation.** WP-13 was committed and pushed
as `10e0e1144f8d73cfbf6bdc96b862038d33479e9d`; strict execution verification passed without draft mode. Original
Freeze 1/1b, the S1 amendment, views, overlay, embeddings and simulations retain their previous bytes.

The workstation driver prepares an isolated published checkout, transfers pinned inputs, checks both CUDA
interpreters, probes the relocated view, and runs the fixed G2 queue. For every terminal unit it pulls the pending
manifest and files to Data1, verifies hashes, acknowledges the transfer and finalizes before admitting another
unit. Lost launch responses reconcile the same immutable job; transfer retries are charged conservatively.
Actual controller identity, timestamps and automatic STOP are checked. One requested two-hour allocation has
a 117-minute provider deadline, a $3 gross ceiling, 900-second reserve and a 3 GiB host-pull payload limit.
The driver verifies shutdown of its own allocation and never automatically starts a replacement session.

Phase 4 reuses the same execution/replay/persistence hooks for exactly 96 terminal 30-epoch S1 fits, the two
required CLEAN embedding farms and eight serial 2,500-epoch CLEAN inner fits. Real work requires published source,
all ten Freeze-1c measurements and explicit numerical G3 CPU/GPU/cost ceilings. S1 verification receipts feed the
unchanged published selector and G4 logic. CLEAN active vectors, links, maps and outputs use ext4; inactive inputs
and acknowledged outputs remain on Data1. The 5 GiB internal reserve is checked before work. CPU metadata
preparation produced the input counts in [G2 readiness](g2_readiness.md), without embedding or fitting a model.

The [extended execution manifest](execution_v2/manifest.json) binds these six execution files to the immutable
published WP-13 manifest. Its SHA-256 is `0ea909ac40523c0253969f9e161733b43951733012437169b14c32e5da2d3b9f`.
The prepared CLEAN receipt binds the same execution source, SHA-256
`682d2df6487b2345b7eef56bc5895cbaacc06fe77ba8f187576bca04f71c7c60`; previous unpublished preparation receipts are
retained, and all thirty scientific payload files and eight unit definitions remain byte-identical.
Selection, scoring, hyperparameters and scientific access rules are unchanged.

ESM-1b and its contact companion were downloaded and fully read back directly on Data1. The isolated CUDA
environment replaces only Torch with `1.11.0+cu113`; all 31 conda records and the remaining package versions match
the existing CPU environment. CPU imports, tokenization and a small synthetic ESM/contact forward pass succeeded.
The original CPU environment remains intact. ESMC-300m offline transfer reuses the frozen weights. Local storage
preparation does not establish L4 kernel compatibility or measured throughput; those checks remain in G2.

Focused authentication tests initially passed 93 cases, Phase-4 tests passed 31 and workstation tests passed 34. The
Phase-4 checks include an actual synthetic 30-epoch CPU fit, terminal verification, independent replay and
acknowledged persistence. CPU smoke passed 44 tests with one existing missing-fixture skip. The final regression
suite passed 3,386 tests, with 18 existing skips and 15 subtest passes, in 710.97 seconds. The earlier partial
regression was interrupted to correct a staging-capacity check. Final request preparation then exposed the need
to separate CLEAN's map memory exception from its training cap. The narrow execution correction keeps the map
and coordinator below 4.5 GB, independently caps train/export children at 3 GiB, and authenticates the map before
training. Resume also verifies its retained cap launcher, map record and fit manifest. All 42 Phase-4 tests passed
after that correction, including actual launcher argument/environment preservation; 240 execution/authentication/
freeze tests then passed against the final source. Unaffected full-suite results are reused, with both source
snapshots retained. The [verification record](execution_v2/verification.json) binds final source, tests and logs.

Independent environment extraction, `conda-unpack`, relocated CPU checks and original CPU-environment preservation
all passed. The initial stock packer exceeded its 3 GiB cap while buffering a large CUDA library; the recovery
streamed three unchanged, prefix-free libraries after checking their official wheel hashes. Every extracted
library hash verified, and the memory cap was preserved. Only the failed temporary archive was removed.
The final archive is 1,924,660,777 bytes, SHA-256
`3a7706486611280cf3c17ce3b1b7be3f4a6ff60252e38d13a484e30f56df7b9d`.
The private preparation receipt SHA-256 is `94ec2bf85a50c549f1f76371956f8f4c030f5206bf4d749907713a5f17ac83c4`.

The local-only launch preview verified all ten staged inputs and twelve queue events, a 19,269,249,867-byte staging
peak and $2.6641 conservative gross envelope. Its `draft_unpublished` status correctly requires publication before
allocation. The concrete private launch configuration and pending Phase-4 requests are prepared; numerical G3
authority is absent. Read-only controller status confirmed `TERMINATED`. Logs, source snapshots, preparation
receipts and synthetic execution evidence are retained on Data1 and the independent NVMe mirror. No paid VM start,
actual campaign fit, new selection simulation or test scoring occurred. Next: publish the execution extension,
then deliberately launch the bounded G2 queue; use its measurements for Freeze 1c and G3 before the first 96+8 fits.

## evc-015

2026-11-02 — **Direct public-weight staging and shorter requested G2 bounds.** The two ESM-1b weight entries now
carry their exact official HTTPS URLs alongside the unchanged SHA-256 pins. Local preflight still reads and hashes
both Data1 copies. The VM download has its own wall-clock bound, no longer than 600 seconds and bounded by remaining
allocation time. Its writer closes before reporting a terminal result. A verified partial receives one independent
size/hash readback before rename; terminal HTTP/download/checksum failure permits the existing verified upload
fallback. An ambiguous SSH failure stops staging. Already verified remote files are reused.

The normal workstation upload falls to 4,743,701,970 bytes; the two directly downloaded files total 7,828,579,640
bytes. This removes the largest uplink transfer without relaxing source, storage, host-acknowledgement or provider
deadline checks. Actual datacenter download speed remains unmeasured until the deliberately started session.

Requested forecasts and per-unit timeouts are now 360 seconds for each five-epoch family/L3 probe and ESMC,
480 seconds for ESM-1b, and 900 seconds for each lane event. These are operational bounds, not measured throughput.
The ESM-1b admission threshold becomes 1,500 seconds (25 minutes), retaining the 1.25 multiplier and 900-second
closeout reserve. The $3 ceiling, 117-minute provider STOP, fixed queue, scientific scope and selection are unchanged.

Only the requested focused checks ran: 50 workstation tests passed in 3.48 seconds and ten execution-extension
authentication tests passed in 3.66 seconds. They cover direct success, terminal-failure fallback, altered local
bytes, incorrect URLs/pins, verified-file reuse, ambiguous SSH, timeout admission and checksum rejection before
promotion. The previous full-suite and unchanged Phase-4 evidence remain retained; the full suite was not rerun.
The [execution manifest](execution_v2/manifest.json) and [verification record](execution_v2/verification.json)
bind the corrected source. Manifest SHA-256:
`5cbad8a934250036160d7c6007947b8092e662fd0649a9ce727b30a33f2ee685`.
The refreshed CLEAN input receipt is `ece4eab3d7da05e4ceb7b785e9865dd20eb3b1eaf0f2a8dbfa8259e2f46842b5`;
all thirty scientific files and eight unit definitions match the preceding preparation. Pending Phase-4 requests
retain their unapproved numerical fields and now reference the updated source/input pins. Post-publication local
readiness must report `ready` before any deliberate G2 start; its receipt is retained with the private launch inputs.

## evc-016

2026-11-02 — **Authorized G2 start blocked by provider capacity.** The execution extension was published at
`589af0a96a44fbe36dc1c2773d7a0221e4e61867`; strict source authentication and local readiness passed. The maintainer
authorized one start of the existing `g2-standard-8` L4 in `us-central1-a`, with a $3 gross ceiling, a two-hour
request and 117-minute automatic STOP. The exact prepared workstation queue was invoked. Local checksum checks
finished before billing, and the refreshed controller preview retained the $2.6641 conservative envelope.

The single start attempt at approximately 11:11 UTC failed with `ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS`.
Google returned no suggested available zone. No allocation was established, no remote staging or measurement
ran, and all twelve queue events remain incomplete. The driver closed the attempt and verified `TERMINATED`;
an independent status check at 11:12 UTC confirmed the same state and zero recorded GPU running time that day.
The recorded running-compute estimate is $0; retained disk/snapshot charges continue and this is not an invoice.

Private evidence: `phase3/g2_closeout_20261102/closeout.json`, SHA-256
`3752dd11d106c4aee85615127d05999039f8287996fd7d584546323438f004cc`, under the campaign output root. It binds
the original launch configuration, authorization, driver error/state, provider status and preceding Phase-4
requests. All thirteen closeout files were independently copied to the NVMe campaign backup and read back by hash.
The launch configuration SHA-256 remains
`abfef1cfde9eda9149671dc82d4fe7ce4a4b91847dd711ea579102f7a8ac0205`.

No Freeze 1c was created: there are zero of ten required verified measurements. The five private Phase-4 request
files now record the capacity blocker while preserving their pending status and unset numerical fields. No G3
ceiling, throughput, lane result, model comparison or test result can be inferred from this attempt. Prepared
views, weights, environments, source pins and scientific evidence remain reusable; no tests or simulations were
rerun. Root STATUS remains the disclosed frozen-source exception of evc-009.

Next: obtain a fresh start decision, then use a new attempt namespace while retaining this closed attempt.
There was no automatic retry, replacement, zone change or budget increase. Complete G2 before assembling
Freeze 1c and proposing numerical G3 authority for the first 96 S1 and eight CLEAN inner fits.

## evc-017

2026-11-02 — **Bounded same-region recovery authorization.** The maintainer authorized one further existing-VM
start in `us-central1-a`, followed only after a confirmed capacity failure by one disk-preserving controller pass
restricted to `us-central1-b` and `us-central1-c`. The shared ceiling remains $3 gross, a two-hour request and
117-minute provider STOP. Cleanup is authorized through the controller's finalization or abandonment evidence
gate. The closed evc-016 attempt is retained; this authorization uses a fresh execution namespace.

The local controller now accepts an explicit destination allowlist in both preview and execution; a resumed
pass cannot omit or change it. Seventeen isolated offline checks cover filtering, CLI propagation, persisted
candidate restrictions, a non-mutating preview and fresh storage-rate reporting. The private controller's prior
files and verification evidence are preserved under `phase3/fallback_update_20261102/`. The pre-change live preview
listed only the two approved destination offerings; this is not evidence of available capacity.

The recovery storage reserve is $0.30, added to the original $2.6641 envelope. At the verified prices, a four-hour
storage window covers the source and two possible destination disks less the disk charge already included in
the base envelope, the temporary snapshot, its one-hour recycle-bin minimum and additional retained-snapshot
time. The resulting conservative total is $2.9641. This accounting window includes setup and cleanup; it does
not enlarge the GPU runtime. Price checks, identity matching, independent persistence and the evidence gate
continue to apply. Scientific source, selection, scoring, freezes and timing inputs are unchanged.

The workstation extension binds recovery to the original source, the controller's selected replacement and its
actual session before further runtime checks. It persists the selected receipt and effective configuration,
allows recovery only of that same active session, and retains closed attempts. Ambiguous replies, foreign
sessions, unapproved candidates and excessive costs refuse further allocation. Local input hashing occurs
before either allocation route. The Phase-4 dispatcher reuses the same endpoint checks; its source is unchanged.

Only the requested two repository test files ran: **98 passed in 6.68 seconds**. The previous full regression
and unchanged scientific checks remain historical evidence. The refreshed execution-manifest SHA-256 is
`5773dbda4a79cf88b423f0fe7d5136621130fdafa807c7caaeff8221289ddc7e`.
Publication and the launch-time preflight precede the authorized retry. Measurements and G3 remain pending.

## evc-018

2026-11-02 — **All authorized zones exhausted; recovery cleaned up.** The combined evc-016 record and recovery
extension were committed and pushed as `4fb4bdd6136b3a8f09cc78f226c082141a62172f`. Strict execution authentication
passed without draft mode. The refreshed CLEAN input receipt is
`f01248bbbe975b0b6a7001af6836643e3383850f2913e69ffa8df532b3785934`; all thirty scientific payloads and eight unit
definitions match their preserved predecessors. The launch configuration is
`59b23f1b58b13dd44b7c3db2f9a6b14fc00f434549d81306c0eee0802a0e388a`.

The queue passed full local preflight before the single source retry. `us-central1-a` returned
`ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS` at 12:09 UTC. Recovery pass `98e162169164` then created its source
snapshot and tried the explicit destination list. Provider insert operations in `us-central1-b` and
`us-central1-c` finished with the same capacity error at approximately 12:20 and 12:23 UTC. No replacement was
selected, no GPU session opened, no remote G2 staging occurred, and zero of ten measurements completed.
The driver closed the attempt and confirmed the source VM stopped. There was no extra retry or zone expansion.

The authorized abandonment preview passed after the attempt window closed and independent backup hashes were
verified. Cleanup completed at 12:27:08 UTC: both restored destination disks, the temporary snapshot and its
recycle-bin copy were removed; all four temporary-storage ledger events closed. The source VM and its original
boot disk were preserved. Independent provider status at 12:28:26 UTC confirmed `TERMINATED` and zero recorded
GPU runtime. The final inventory contains only the original disk and two pre-existing snapshots.

Estimated temporary-storage charge: **$0.0269563**, including the snapshot and recycle-bin one-hour minima.
Original retained storage continues separately; these controller estimates are not a billing invoice. The
private closeout receipt is `phase3/g2_fallback_closeout_20261102/closeout.json`, SHA-256
`d38067b386938399de918978c890a2bf07b4ea70d6bea925d1494a92166dfbd5`. Its evidence and request snapshots were copied
to the independent NVMe backup and read back by hash (28 files).

The five Phase-4 request files retain pending status and unset numerical budgets, now naming the exhausted
capacity pass. Freeze 1c cannot be built and no measured G3 ceiling or lane outcome can be claimed. The published
code and prepared inputs remain reusable. A future authorized allocation must use a fresh attempt namespace;
the closed driver state and archived recovery pass remain immutable.

## evc-019

2026-11-02 — **Cross-region recovery preparation; allocation blocked by its gross ceiling.** The evc-017/018
closeout was committed and pushed as `8d84e95`. The maintainer authorized one source retry followed by the
ordered destinations `us-west1-a`, `us-west1-b`, `us-west1-c`, `us-east4-a`, `us-east4-c`, with two requested
runtime hours and a $3.50 gross ceiling including recovery. This authority does not change scientific scope.

Read-only provider checks found both destination subnets on the expected network and each region's L4 quota
at limit 1, usage 0. The three proposed west compute SKU identifiers returned HTTP 404. The runbook's existing
Americas compute SKUs cover west; east's specified SKUs are valid. The controller uses `PRICE_COMPUTE_SKUS`
and `PRICE_DISK_CAPACITY_SKU_ID`, rather than the proposed `EXPECTED_*` names.

[Official snapshot pricing](https://cloud.google.com/compute/disks-image-pricing#network_charges_for_snapshot_creation_and_restoration)
charges $0.02/GiB for North America cross-region restoration, including disks whose subsequent VM creation
fails. The previous deleted recovery snapshot recorded 69,027,758,175 `downloadBytes` (64.287 GiB), compared
with 20,025,475,200 incremental `storageBytes`. The [API](https://docs.cloud.google.com/compute/docs/reference/rest/v1/snapshots)
defines these separately. Even that historical download size implies $6.429 for five restores, before G2;
it is not a bound for a new snapshot. The safe pre-allocation bound is 150 GiB per candidate, or **$15** for
five restores. G2's existing compute/margin/egress/retained-snapshot envelope adds $2.6641, before temporary
recovery storage. The $3.50 ceiling therefore cannot admit this pass.

The private evidence is `phase3/cross_region_update_20261102/pricing_audit.json`, SHA-256
`6ede236635f70d0b3f680f611dfc69d25e792f78220d93fbb333778b9740830b`. Provider status at 14:04 UTC confirmed
the original VM `TERMINATED` and zero GPU runtime. No billable action was performed for this preparation.

The execution update preserves the default same-region route and adds the exact ordered cross-region scope,
full regional profile binding, priced restoration reserves and a one-hour/five-candidate initiation bound.
The private controller records transfer charges before each restore and retains them for failed candidates.
`--adopt-recovered` uses the existing fit/replay/independent-backup finalization gate; selected-but-unverified
replacements cannot authorize source deletion. No controller caps or selected configuration were changed.

The two focused repository files passed **114 tests in 10.20 seconds**. The full regression suite and simulations
were not repeated. The refreshed execution manifest is
`49c140b0b35b400c1dacd8b6c37af5d9083f6a254ecbb1373a890d681038e7da`. The CLEAN input receipt is
`12b8a88716071d1e719ab4ed648159b6d61f3d0fe510b25797d19594a3ed470b`; all thirty payload files and eight unit
definitions match their preserved predecessors. Freeze 1/1b, the S1 amendment, views and embeddings remain intact.

The actual workstation preflight refused **$18.264096 required versus $3.50 authorized** before large-file
hashing or controller calls. The controller's independent live-price preview refused **$18.254731** before
creating any storage or compute; the workstation reserve rounds storage upward. The source-only leg remains
within its $2.6641 envelope and uses a fresh namespace for the single authorized retry after publication.

Seven new private-controller tests passed, including ordinary central start, regional adoption, five failed
candidates, transfer accounting and cleanup. The legacy run passed 38 of 39 checks; one setup failed because
the simulated provider timestamp had finer precision than the ledger timestamp. That check passed unchanged
on rerun. The stopped-source fixture was then made deterministic and three affected checks passed; the
production capacity guard was unchanged. Detailed counts, logs and hashes are bound in the execution
verification record. Documentation checks passed with zero strict failures and eleven baseline warnings.

## evc-020

2026-11-02 — **Published recovery support; source retry stocked out.** The execution update was committed and
pushed as `1749cdaa52e897c0fa53fb26341bf730325e97c7`; strict execution authentication passed without draft mode.
The source-only launch configuration is `7dc3038d780230a0e48fcb73dd9c6eba873749e4f9f3b78c75ba4f8bf72d2415`.
Its pre-billing input checks passed. An initial read-only start preview could not verify billing because its
provider lookup failed with quota-related error output; no start was requested. Direct provider rechecks
confirmed billing enabled. The same authorized leg then passed its preview and issued exactly one start.

The start attempt began at 14:50:15 UTC and returned `ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS` in
`us-central1-a`. The driver closed the attempt, preserved the error and confirmed `TERMINATED` at 14:51:03.
Independent status at **14:53:21 UTC** reconfirmed shutdown and zero GPU runtime. No G2 unit, remote staging,
cross-region candidate or temporary resource was created. There is no active recovery receipt to finalize
or abandon. The original disk and two retained snapshots remain; their existing storage charges continue.

The cross-region configuration stays blocked at **$18.2641 required versus $3.50 authorized**. A private,
unapproved proposal prices a $19 campaign ceiling and a $25 controller daily guard; the latter accommodates
full-day temporary-storage reservations while retaining the two-hour request and 117-minute STOP. The
proposal changes no live caps or authority. The alternative is to close at this evidence boundary.

The closeout is `phase3/g2_cross_region_closeout_20261102/closeout.json`, SHA-256
`7dd0478d066f26ea3b5b1dd276bd118b409bc297c36384c2041d79e2a520a8bf`. Its source configuration, driver state,
provider checks and logs are preserved on Data1 and independently mirrored on NVMe (eleven closeout files;
twenty-nine preparation files). All five Phase-4 requests
remain pending with the explicit source-stockout/recovery-budget blocker. Zero of ten measurements are
available; no Freeze 1c, measured G2 timing or numerical G3 authority can be produced.

## evc-021

2026-11-02 — **Recovery budget approved.** The maintainer approved a **$19 campaign gross ceiling** and
`MAX_DAILY_ESTIMATED_COST_USD=25.00`, retaining the two-hour request and 117-minute provider STOP.
Scope is one source launch in `us-central1-a`, then one bounded pass through `us-west1-a`, `us-west1-b`,
`us-west1-c`, `us-east4-a`, `us-east4-c` on confirmed stockout. All twelve G2 events retain HostPull;
successful recovery uses evidence-gated adoption, and an exhausted failed pass uses abandonment.

The original controller configuration was backed up; only the approved daily guard changed. The existing
recovery budget now records $19 and is bound to a fresh driver namespace. The $18.2641 conservative envelope
retains the full $15 transfer reserve and $0.60 recovery-storage reserve. No scientific recipe, selector,
scorer, simulation, view, embedding or original freeze is changed. Publication and focused checks precede billing.

The two focused repository files passed **116 tests in 4.42 seconds**, including $19 acceptance, refusal above
$19 and continued refusal of the underfunded $3.50 route. Unchanged controller verification is reused.
The execution manifest is `8ba64dc725ebbe694718dcaf5bc3ab0f0ea0f3695da37b246f0ee18f16d8ce1a`.
The CLEAN receipt was rebound to it; all thirty payload files and eight unit definitions are identical.
The controller status check confirmed `TERMINATED`, zero running time and the $25 daily guard at 16:10 UTC.

The first queue invocation passed local input checks but the controller's fresh price request returned HTTP
429 at 16:30 UTC. This occurred before the provider-start step: the usage ledger contains no new
`START_REQUESTED`, running or transfer event. Shutdown was independently confirmed at 16:33:50 UTC.
The closed driver record is preserved; the authorized allocation remains unused.

The private price client now retries HTTP 429 only, at most three attempts, honoring `Retry-After` or using
30/60-second backoff within a shared 120-second verification deadline. Fresh pricing remains mandatory;
unknown prices and other errors still refuse execution. **18 focused offline checks passed** (nine new,
nine existing) in 6.39 seconds. The source SHA-256 is
`43ecda3a57910dc20494433f55c4cdda8c081073af012d2ccb89ceb2e310e6b3`.
This changes no public executable source or scientific input binding. A fresh driver namespace resumes the
same unconsumed source launch and approved recovery scope.

## evc-022

2026-11-02 — **Approved cross-region pass exhausted; cleanup verified.** The $19 update was published as
`a647b87`, followed by the bounded pricing-retry verification in `243dcab`. Strict execution authentication
passed without draft mode. The fresh launch configuration SHA-256 is
`26d79b23d6ebd474f5dff3b03b2f07736403170383388f963946f717788fcb1f`.

The single source start was requested at 16:55:32 UTC and failed with
`ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS`. Recovery pass `ae739a6f6910` created its source snapshot, then
requested candidates in the approved order:

| Zone | Create request (UTC) | Provider result |
|---|---|---|
| us-west1-a | 17:10:29 | Capacity failure |
| us-west1-b | 17:15:20 | Capacity failure |
| us-west1-c | 17:19:09 | Capacity failure |
| us-east4-a | 17:22:14 | Capacity failure |
| us-east4-c | 17:29:42 | Capacity failure |

All five provider insert operations reached `DONE` with the same stockout code. No replacement was selected.
The driver closed with zero completed events and verified the source stopped. No remote staging, HostPull,
embedding or scientific fit began; this is a capacity result, not G2 runtime evidence.

After independent backup and a successful cleanup preview, authorized abandonment deleted the five exact
restored disks, the temporary snapshot and its recycle-bin copy. It completed at **17:34:56 UTC** and archived
the pass. Independent provider status at **17:35:35 UTC** and the subsequent report confirmed `TERMINATED`,
no active session and zero recorded GPU runtime. The original VM, its 150-GB disk and two pre-existing snapshots
remain. The controller configuration retains the approved $25 daily guard and original central endpoint.

Conservative incremental pass accounting is **$15.0548006054**: $15 for five full-150-GiB transfer allowances,
$0.0342486590 for temporary disks, $0.0102766438 for the snapshot and $0.0102753025 for its recycle minimum.
This is not provider-confirmed billed spend. The full-day controller estimate is $16.56, including earlier and
ongoing storage reservations; the retained disk and snapshots continue to cost approximately $30/month.
All seven temporary storage records are closed; future minimum-charge timestamps do not denote live resources.

The closeout is `phase3/g2_budget19_closeout_20261102/closeout.json`, SHA-256
`b520dee31fb305cb07f73733e146f2996cb6dfb41ecfe3a77b284c976e1ee9f4`. Thirty-three evidence/request files are
preserved on Data1 and independently mirrored with hash readback on NVMe. The abandonment receipt SHA-256 is
`80e338cdc24bdbcafa3d3d9516ffb060a63a680b431b305787720f403a6beb26`.

All five Phase-4 requests remain pending with the explicit all-six-zones-stockout/abandoned blocker. There are
zero of ten required measurements, so no Freeze 1c or numerical G3 proposal is possible. The published source,
views, embeddings, CLEAN input closure and earlier freezes remain reusable. Another allocation requires a
fresh decision; the closed driver and exhausted recovery pass must not restart automatically.

## evc-023

2026-11-02 — **Same-region pass exhausted; abandonment and shutdown verified.** The maintainer authorized
one source start in `us-central1-a`, followed on stockout by `us-central1-b` and `us-central1-c` only.
Cross-region recovery was excluded. The two-hour request, 117-minute provider STOP and exact twelve-event
HostPull queue retained a **$3 gross ceiling**, including the **$0.30 same-region recovery-storage reserve**.
The existing $25 controller daily guard was retained; `config.env` is byte-identical to its pre-pass backup.

The private launch configuration was rebound to `same-region-v1`, without a cross-region budget reference,
and a fresh driver namespace. Its SHA-256 is
`54833fce33fe0399c57982654d64b09aaba3951ef0ca8c9b02471d0a99de927c`.
Published execution authentication and all ten staging-input checks passed. The unchanged execution manifest
is `8ba64dc725ebbe694718dcaf5bc3ab0f0ea0f3695da37b246f0ee18f16d8ce1a`; source HEAD was `f919182`.
Fresh controller pricing admitted the **$2.9640956** conservative envelope, with zero recovery-transfer reserve.
No scientific source, freeze, view, embedding or CLEAN input closure changed; existing code verification was reused.

| Zone | Recorded request (UTC) | Result |
|---|---|---|
| us-central1-a | 18:49:27 source start | Capacity failure |
| us-central1-b | 19:00:45 recovery request | Capacity failure |
| us-central1-c | 19:03:59 recovery request | Capacity failure |

Recovery pass `b3502feb877b` used one central-region snapshot. Both destination insert operations reached
`DONE` with `ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS`. No replacement was selected. The driver closed
with **zero of twelve events completed** and verified the source stopped. No remote staging, HostPull transfer,
embedding or fit began. This is capacity evidence, not a measured G2 result.

After the ten-minute recovery window expired, independent backup verification and the cleanup preview passed.
Authorized `--abandon` removed both restored disks, the temporary snapshot and its recycle-bin copy, then
archived the pass at **19:08:08 UTC**. Independent provider status at **19:08:23 UTC** and the report at
19:08:29 confirmed `TERMINATED`, no active session and zero GPU runtime. The original VM, its 150-GB disk
and two pre-existing snapshots remain. All four temporary-storage records are closed.

Conservative incremental storage accounting is **$0.0267665548**, including snapshot and recycle minimums;
no cross-region transfer charge was recorded. This is not provider-confirmed billed spend. The controller's
full-day estimate is $16.59, including prior passes and ongoing storage reservations, within its unchanged $25
guard. The retained original disk and snapshots continue to cost approximately $30/month.

The closeout is `phase3/g2_same_region_20261102/closeout.json`, SHA-256
`595a5bf2d2ffce4073e5939f7da3f3ac691d2ff99349bfb21fc9c69e9ab281d0`.
All 33 evidence files are preserved on Data1 and independently mirrored to NVMe with hash readback.
The abandonment receipt SHA-256 is
`72a198cd3157fc2e7dbe14dc56fe0a9a871ea56310ae7e0974204ce220603b58`.

Zero of ten required measurements are available; Freeze 1c and numerical G3 remain blocked. Phase-4 requests
remain pending. The closed driver and exhausted recovery pass cannot restart automatically; another allocation
requires a fresh decision. Root STATUS remains unchanged under C-10; current progress stays in this campaign.

## evc-024

2026-11-02 — **Runpod setup and execution-readiness audit.** After the closed GCP capacity pass
([evc-023](#evc-023)), the maintainer authorized Runpod setup. Pod `mbg2lc6vg5oiki` (`deepmzyme-4090`)
started at **20:13:34 UTC** in `EUR-IS-2`. The provider record showed one RTX 4090, 12 assigned vCPUs,
100 GB RAM and an active rate of **$0.89/hour**. Device queries recorded 24,564 MiB VRAM, driver 595.91.07,
CUDA 13.2 driver support and PyTorch CUDA 12.4. The host CPU was AMD EPYC 7452; the memory probe reported zero swap.

The pod was still `RUNNING` at the **20:42:14 UTC** provider read. No automatic-stop deadline or numerical
Runpod spending ceiling had been recorded or verified. The $20 account credit is neither a campaign budget
nor a stop control. The GCP two-hour request, 117-minute STOP and $25 daily controller guard did not transfer
to this allocation. No additional resource or cross-region GCP recovery was launched during this audit.

Creation returned `volumeInGb: 0`; provider inspection showed `mounts: {}` and filesystem inspection showed
container overlay storage. Neither `/workspace` nor the staged inputs under `/home/mechti` have verified
persistence across lifecycle changes. Recovery relies on the retained Data1 input archives and a reconstructed
qualified runtime, not on preserving the container. At **20:39:14 UTC**, ten environment records had been
copied off-pod to `phase3/runpod_setup_20261102/env_records/` under the private campaign output root.
`env_records_sha256.txt` initially recorded hashes computed only from the downloaded files; the later
source/destination comparison and closeout are recorded below.

Five remote staged files were hashed and matched against the existing preparation pins:

| File | SHA-256 |
|---|---|
| `esm1b_t33_650M_UR50S.pt` | `0569754efaff7dcb7e068c27367bc73f10afb4b450ea30aac30d9bc60783a8b1` |
| `esm1b_t33_650M_UR50S-contact-regression.pt` | `77193a8814f0db0b36a03aebb1a311adc6b4745f463c04839defc15407bbb28a` |
| `fold0_view.tar` | `a036a782e3f0de041e896b7e2ff3a79dd13eb6d827629b0eca28dfcbc9b90334` |
| `esmc300_hf_cache.tar` | `47182be2bbc30d1d2b3bd772b8660f7dd36beba71d2803339fe2cb06cfcf9cf9` |
| `clean_esm1b_cuda113_py310.tar.gz` | `3a7706486611280cf3c17ce3b1b7be3f4a6ff60252e38d13a484e30f56df7b9d` |

The ESM-1b weights were downloaded from their official HTTPS source. The private
`phase3/runpod_setup_20261102/staged_archives_sha256.txt` records these five files; its SHA-256 is
`3d65bd7fcd14ded6498cf8b342482d0177d123206acd5faf17d6a50003b08a42`.
This initial check covered five staged files, not all ten required inputs, the extracted view or its read guards.
The later preservation check verified the remaining timing and helper files against their existing pins.

The unpinned Python 3.11 container installation reported `torch 2.4.1+cu124`, `torch-geometric 2.8.1`,
`scikit-learn 1.9.1`, `biopython 1.88`, `optuna 5.0.0` and `pandas 3.1.0`. Imports and device queries passed;
no actual-model, independent replay or HostPull round trip was measured in this environment. Code inspection
identified L4-only CUDA checks, required `systemd-run` resource caps and GCP-bound session/transport checks.
The uploaded repository excluded `.git`, leaving published-source authentication unsatisfied. This is an
inspection finding, not a recorded `verify-execution` failure. No execution guard was bypassed.

The unchanged [WP-13 manifest](wp13/execution_manifest.json) has SHA-256
`6be7525464aff0d2714d5d4fa523bfd9c64483f699c76ccca7d2b6590c35c431`; the unchanged
[automation manifest](execution_v2/manifest.json) has SHA-256
`8ba64dc725ebbe694718dcaf5bc3ab0f0ea0f3695da37b246f0ee18f16d8ce1a`.
[G2 readiness](g2_readiness.md#runpod-execution-qualification) owns the outstanding migration checks.
Any adaptation must preserve source authentication, input/read guards, admission limits, replay and HostPull.

**Preservation and shutdown.** At **20:50:21 UTC**, independent remote/local SHA-256 comparisons matched all
ten copied environment records. The remaining five timing/helper inputs matched their prepared pins at
20:51:17 UTC; together with the earlier five checks, all ten staged-file hashes match. All ten original local
sources remain available. The filesystem inventory contained 6,874 files and found no trained outputs or active
GPU process. A separate 1,936-file comparison, excluding derived views and caches, found no repository path
missing locally. Four older campaign documents matched published commit `cf850ab97577e78e15db6f31df3e00f4bd996051`;
the one differing local configuration file was copied off-pod and hash-verified. Derived views and caches can
be reconstructed from retained inputs. This establishes preservation, not a qualified execution environment.

The private `phase3/runpod_closeout_20261102/preservation_receipt.json` records this check; SHA-256:
`82d8ea3558f93549f0e564d0abeba3876d9fa0fd41ebf5a5821f146b83344006`.
The stop request at **20:52:15 UTC** succeeded; a fresh provider read at **20:52:25 UTC** confirmed
`desiredStatus: EXITED`, `runtimeStatus: stopped` and `volumeInGb: 0`. The pod was not deleted or replaced.
`stop_action.json`, `provider_after.json`, `environment_copy_verification.json` and
`file_identity_comparison.json` retain the corresponding evidence in that closeout directory.

The billing read at 20:52:25 UTC returned no rows, so final billed spend is unknown, not zero. Allocation start
through verified stop spans 2,331.096 seconds (about 38 minutes 51 seconds); multiplying that interval by the
observed $0.89/hour rate gives approximately **$0.5763**, an elapsed-time estimate rather than a provider-confirmed charge.
The final `phase3/runpod_closeout_20261102/closeout.json` has SHA-256
`15c1909d691d2250324cff2041f0b3b4f6c00d13b293abd0266eb0a391684d42`.
All 25 closeout files, including `SHA256SUMS`, are retained on Data1 and independently mirrored to NVMe with
hash readback. A separate GCP status read at **20:52:55 UTC** reconfirmed `TERMINATED`; the 20:53:01 UTC
report confirmed no active session and zero recorded GPU runtime that day. No GCP restart occurred.
Zero of twelve G2 events completed and no scientific fit began. Freeze 1c and numerical G3 remain pending before
the 96 S1 plus eight CLEAN inner fits. Root STATUS and all frozen scientific evidence remain unchanged under C-10.

## evc-025

2026-11-03 — **Runpod qualification extension and replacement preparation.** The evc-024 closeout was
committed and published at `e2c5912`; the untracked launch helper was removed. The new approved scope is
one Secure Cloud L4 or RTX 4090 G2 pass in an available datacenter, a two-hour requested window,
117-minute maximum allocation, and **$5.00 gross** including storage and reserves. The original GCP VM
remains stopped; its controller keeps the $25 daily guard. Phase 4 still requires measured Freeze 1c and
the numerical G3 proposal to be reviewed before its fits begin.

The additive [execution-v3 manifest](execution_v3/manifest.json) preserves the original WP-13 and v2
manifests and authenticates their published source ancestry. It pins the provider controller, direct
cgroup-v2 cap helper, and affected execution code. Remote execution requires a clean published `.git`
checkout, the pinned inputs, an exact interpreter package fingerprint, and the verified relocated view.
Both approved GPU profiles require one nominal 24-GB device and working CUDA in both interpreters.

The new qualification checks exercise a synthetic production F4 checkpoint reconstructed on CPU and CUDA
at the existing `1e-6` absolute logit tolerance. The separate CLEAN interpreter loads all 33 ESM-1b layers,
checks both pinned weight files, transfers the complete model to CUDA, and compares CPU/CUDA representations
for the synthetic sequence `MKT` at predeclared feature tolerances `atol=rtol=1e-4`. These are runtime checks,
not campaign fits. `NVIDIA_TF32_OVERRIDE=0` applies to qualification and the actual pod workloads.
Host, boot, interpreter, source and allocation bindings prevent reuse of another session's proof.

The container helper enforces the original per-child CPU/RAM/zero-swap limits directly through delegated
kernel cgroups. Its local CPU-only live verification passed **21 tests**, including measured throttling,
contained OOM termination, membership and descendant cleanup. Probe SHA-256:
`e03955dbd413874045530833711e38bf82c074a600781d956562c5e90a0a48f7`.
This establishes the helper's behavior on Linux; actual Runpod delegation remains a prerequisite.
Read-only or non-delegated cgroups refuse work before the input closure is staged.

The workstation retains one exact twelve-event queue, the 1.25 admission multiplier, 900-second closeout
reserve, F9 guards, independent replay and HostPull acknowledgments. Each acknowledgment follows verified
Data1 readback and a second NVMe copy. The independent stop process is armed before creation, admission
closes 120 seconds before the allocation ceiling, and provider STOP is read back. Source/config drift,
lost creation replies, partial ownership records and failed bookkeeping are covered by focused tests.
Stopped persistent storage has a priced finite reserve; deletion requires the complete output inventory
and two verified local copies. A stopped failure with unacknowledged data is retained for recovery.

The [CPU verification record](execution_v3/verification.json) records **342 passed, one skipped** across
the six focused files. The skipped check requires a supplied delegated cgroup; the separate local live
check above covers that mechanism. Source stayed unchanged during the focused run. The original frozen
scientific inputs and selection rules were preserved.

Both relocated runtime archives passed CPU imports and tensor checks. The complete
[runtime fingerprints](execution_v3/runtime_locks.json) pin the main Python 3.12/CUDA 13.0 environment
and CLEAN Python 3.10/CUDA 11.3 environment, including archive SHA-256 values and exact package versions.
The main archive adds the pinned missing `gemmi==0.7.3` wheel without modifying the workstation environment.
Only absent bytecode files were omitted; uncached conda metadata warnings and relocation evidence remain
in the private runtime record. Actual CUDA/model/replay qualification on the replacement remains mandatory.

The old stopped, zero-volume pod `mbg2lc6vg5oiki` had lost its assigned host GPU. After checking the retained
preservation receipt, it was terminated under the approved cleanup scope; a fresh provider list confirmed
its absence. Private `phase3/runpod_qualification_20261102/old_pod_deleted.json` records the action.
Replacement qualification and measured G2 results are pending. No Freeze 1c or numerical G3 result is claimed.

## evc-026

2026-11-03 — **Container compatibility and named-pod handoff.** The qualification package was published
at `e30cc4f`. A fresh Secure Cloud L4 diagnostic allocation in EU-RO-1 exposed writable cgroup v1,
which the original v2-only helper correctly refused. No payload or G2 fit ran. Its empty output inventory
was copied to two local devices before provider-confirmed STOP and deletion. Billing returned no rows;
the approximately $0.022 elapsed-time estimate is not a provider-confirmed charge.

The additive [compatibility manifest](execution_v3/manifest_cgroup_v1.json) preserves the original v3, v2
and WP-13 records. Its helper supports both delegated kernel layouts. The v1 route additionally requires
genuine host-wide zero-swap evidence, unchanged swap counters and verified OOM notifications; missing
memsw accounting is reported explicitly. Local live cap verification passed 49 tests. This does not
establish actual Runpod v1 qualification.

An explicitly transferred running RTX 4090 pod, `uahs9wg1dmohv1`, was pinned to the approved image digest
and a 100-GB temporary disk. Its actual unified v2 namespace had finite outer CPU/RAM limits and zero swap.
Only verified container processes were moved into a manager child to enable delegated CPU and memory
controllers. All outer limits stayed unchanged. The actual helper passed throttling, contained OOM, both
resource profiles, membership, zero-swap and descendant-cleanup checks. Probe SHA-256:
`a9afa37bda2da1d23bad01c17b3840beb90faa440c09bf4e5c8a99bffcc57cb2`. Fresh SSH and provider telemetry remained healthy.

The pod has no persistent volume. Original inputs remain local, and each completed unit still requires
Data1 readback plus independent NVMe mirroring before its HostPull acknowledgment. Named adoption
closeout is STOP-only; no deletion is authorized by the handoff. The locked runtime upload and exact
input staging retain the original pins. Public weight downloads are accepted only after full checksum
verification; the ESMC tar was reconstructed byte-for-byte with its existing archive hash.

The independent stop process retains the original absolute ceiling: STOP request at **01:10:02 UTC**,
hard boundary **01:12:02 UTC on November 3**. Adoption retains the pod's original 23:22 allocation clock;
its container restart cannot grant another window. The observed compute rate is $0.89/hour. The private
quote reserves $0.63 for prior use and retains the **$5 gross** ceiling, running storage and margin.
Transport-enforced TF32 settings apply to every remote command and child.

The [compatibility verification record](execution_v3/verification_cgroup_v1.json) records **390 passed,
three skipped** with unchanged source. Separate targeted checks cover the real detached ephemeral
watchdog and dual-device preservation; actual pod probing covers kernel enforcement. Full model/runtime
qualification and the twelve-event timing queue remain pending. No G2 result, Freeze 1c or numerical G3
is claimed. Phase 4 and held-out reads remain gated.

## evc-027

2026-11-03 — **Runtime isolation repair and replacement boundary.** The compatibility package was
published at `cb8d5de`. The first actual interpreter check found 185 distributions against the intended
187-package main lock: `pandas==3.0.1` and `lmdb==2.0.0` came from the workstation user site and were
not inside conda-pack. The earlier CPU import check had inherited that site. The strict comparison
refused execution before CUDA/model qualification or G2 admission. STOP was provider-confirmed at
00:02:12 UTC; an independent read confirmed `EXITED`. The transferred pod was not deleted.

The provider's OOM warnings began at 23:50:16 UTC, during the intentionally contained 64-MiB child
OOM enforcement test. Its child termination and cleanup passed, and the controller continued afterward.
Provider logs record the explicit container stop at 00:02:11 UTC following the package-lock refusal.
No G2 fit had started; the console banner is not evidence that a 31-GB campaign fit failed.

The [runtime repair record](execution_v3/runtime_repair_v1.json) binds the original archive, official
PyPI wheel hashes and the corrected self-contained archive. Original archive bytes remain unchanged.
An isolated audit also found stale setuptools 81 metadata alongside verified conda 82 code. All 319
source files and 120 data files checked against conda 82 matched. Only the stale metadata is removed;
existing compatibility code remains bound to the original archive. Seven corresponding relocation
records are omitted so `conda-unpack` does not visit absent files. The scientific source is unchanged.

The corrected archive has SHA-256 `ec43f3537038522b460175fb90d1c3ca47393b810d85520a54b98aeb130d8863`
and 3,491,750,629 bytes. Its raw metadata has exactly 187 unique distributions. The corrected private
prefix passed isolated full-fingerprint, import, pandas, LMDB and Torch CPU checks with user site
disabled. CLEAN separately passed its unchanged 26-package lock, CPU tensor and ESM-1b tokenization
checks. These preparation checks do not replace fresh unpack, full-model CUDA and replay qualification
on the actual allocation. The [new locks](execution_v3/runtime_locks_v2.json) preserve package versions.

An additional guarded 117-minute attempt was authorized within the unchanged **$5 gross** ceiling.
The supplied migration created `vzi08j1ntuijvn` at 00:24:11 UTC with the approved RTX 4090. Its
workspace was empty. That host remaps container root to UID 100000; the v1 controllers are owned by
unmapped UID 65534. The actual helper returned permission denied before creating a child memory
cgroup. No scientific payload or G2 work was staged. The empty pod was stopped and independently
read back as `EXITED`; no deletion occurred. This was a host-delegation refusal, not a GPU-model mismatch.

The replacement pass retains **02:21:11 UTC** as its absolute hard boundary, with STOP requested
120 seconds earlier. New creation, delayed replies and recovery must not reset it. The $5 envelope
includes a conservative $1.40 prior-use reserve; provider billing still lacks final rows. Both original
transferred pods remain stopped. The [runtime-v2 manifest](execution_v3/manifest_runtime_v2.json) and
[verification record](execution_v3/verification_runtime_v2.json) preserve all original source records.
The final six-file focused suite passed **400 tests, one skipped**, with unchanged source; the
independent deadline review found no blocking issue. Documentation checks had zero strict failures.
Zero G2 events are complete; Freeze 1c and numerical G3 remain unmeasured. Phase 4 and held-out reads stay gated.
