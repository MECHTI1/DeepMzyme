# EC class versus CLEAN30: design

Campaign [README](README.md); decisions and reasons: [log evc-001](log.md#evc-001).

## Benchmark, systems and claims

- **Benchmark.** The five official split30 train/test folds, unchanged, restricted to the prepared conservative metallo
  cohort: 740 proteins, 545 test-structure occurrences, about 530 evaluable after excluding proteins annotated with
  more than one EC class ([cohort and its fold 1–4 screening gap](../../DATASETS.md#clean30-conservativecurrent-main)).
  Structures are AlphaFold models whose metals were transplanted by AlphaFill from one selected donor PDB entry per
  protein; only sites that MAHOMES predicts as catalytic are kept, so metal sites are computational.
- **Populations (co-primary).** B1-all: every evaluable official metallo test protein, pooled over folds. B1-NH30: those
  with no hit of ≥30% identity over ≥50 aligned residue pairs (E ≤ 1e-3, any coverage, pinned MMseqs2 18-8cc5c) to any
  protein of their fold's full training file, and no shared selected AlphaFill donor. split30's nominal 30% is an
  MMseqs2 clustering threshold whose coverage setting was not published (the MMseqs2 default, also used by CARE, is ≥80%
  of both sequences). Within fold-0 training, 5.5% of proteins have a ≥30% partner at that coverage, 42% at ≥50%
  coverage of the shorter sequence and 56% under the rule above. Secondary: a sequence-only no-hit set, identity bins
  and a ≥50%-coverage subset. All are analysis subsets of the official folds, not a re-split.
- **DeepMzyme.** Only-GVP and whole-protein-only controls; candidates Only-ESM (pocket), late fusion, and both with an
  added whole-protein ESMC branch. In every fold one pre-registered algorithm chooses the learning rate, family and
  levers from that fold's training proteins only (4 inner folds under the same identity rule, 2 screening seeds). The
  final model is a 5-seed refit ensemble.
- **Inner selection** (declared deviation from the EC playbook's class-support rule,
  [launch gate](../../EC_TRAINING_PIPELINE_PLAYBOOK.md#readiness-outputs-and-explicit-launch-gate)): pooled inner
  out-of-fold BA over EC classes 1–6, 2 seeds. Class 7 (2–4 training proteins in at most 3 components per fold) may be absent from an
  inner split and is exempt from the zero-recall promotion guard; its recall is reported descriptively.
- **Training rule** (declared deviation from [Plan §7](../../../Plan.md#7-baseline-first-model-comparison-policy)
  checkpoint selection by validation metrics, whose fixed-schedule allowance is written for metal, and from the EC
  playbook default of a fixed LR and the best validation epoch): cosine to zero, terminal checkpoint, the same rule in
  selection and refit, batch 4; each fold chooses 12, 20 or 30 epochs
  ([TECH-027](../../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-027--cross-validation-and-final-refit-use-different-checkpoint-rules)).
- **CLEAN.** Upstream commit `f2bf2a4`, published recipe, not re-tuned here (ESM-1b, triplet loss, LR 5e-4, 2,500
  epochs; that epoch count was chosen upstream on these same folds). R1, the primary comparator: the fold's full
  training file (about 8.1K enzymes), 5 seeds, scored as the seed mean and as a 5-seed ensemble. R2: the fold's prepared
  metallo training subset, including the proteins with more than one EC class that DeepMzyme excludes, 3 seeds. Control: ESMC input instead of
  ESM-1b, 3 seeds. Baseline: MMseqs2 nearest-hit transfer.
- **Primary metric** (declared deviation from the seven-class EC BA default): protein-level balanced accuracy over EC classes 1–6.
  All seven-way predictions count; class 7 (translocases: at most about 4 test proteins, one family) is left out of the
  average.
- **Claim rule.** "Outperforms CLEAN" needs one-sided paired homology-cluster bootstrap lower bounds above 0 against both
  the R1 seed mean and the R1 ensemble, Holm-corrected over B1-all and B1-NH30. Expected detectable margins: at least
  about 8 BA points (B1-all) and 11 (B1-NH30), before homology-cluster inflation. Without a pass the wording is "not
  shown to outperform"; every outcome is reported.

## Test access, disclosures and licence

- One scoring pass after every model and analysis script is frozen and hashed. Before it, no curated test file or
  curated test membership is read and no label is used in a test role. The identity audit after Freeze 1 queries each
  fold's complement (proteins outside its training file, all present in other folds' training files); for fold 0 these
  include its metallo test proteins. Earlier incidental exposures: [test-use ledger](../../DATASETS.md#test-use-ledger).
- Design-time label exposure: each fold's test proteins are labelled training proteins of the other four folds; the
  historical CARE development set behind the LR evidence and the family priors contains at least 48 of the 118 fold-0
  metallo test proteins; the planning pools (1,188 and 1,291 proteins) mix fold-test proteins. ESM-1b and ESMC were
  pretrained on UniRef.
- Disclosed with every result: the official tests are closed to full EC numbers seen in each fold's training (upstream
  removed the others), which favours CLEAN's nearest-centre design; metal sites are computational (AlphaFill/MAHOMES);
  DeepMzyme trains only on metalloenzymes; the cohort's fold 1–4 screening gap; CLEAN is not re-tuned here.
- CLEAN is used under its University of Illinois research-use licence (non-profit research). CLEAN code, patches and
  CLEAN weights are never distributed, only privately backed up. The repository holds the maintainer's own runner,
  setup steps, a plain-language description of each local change, and predictions. Every report carries the licence's
  acknowledgement (§2.7.1) and is notified to the University of Illinois (§2.7.2).
