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
