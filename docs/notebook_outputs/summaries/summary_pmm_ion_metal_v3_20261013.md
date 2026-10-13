# PMM ion metal v3 — completion summary (2026-10-13)

Campaign `pmm_ion_metal_v3` ran all planned steps A–F. Its current state and remaining decisions are in
[`EXPERIMENT_STATUS.md`](../../../EXPERIMENT_STATUS.md). The dated history is in the
[campaign log](../../campaigns/pmm_ion_metal_v3/log.md), and test use is recorded in the
[DATASETS ledger](../../DATASETS.md#test-use-ledger). This summary separates three kinds of evidence, which must not
be mixed: validation findings, the secondary test result, and experiments that were deferred and never run.

**Design:**
- PMM ion-level metal data: 7,398 training ions.
- Strict grouped folds `v3-seqid90-s42-b2`.
- `four_class`, `five_class` and `six_class` arms for Only-ESMC, Only-GVP and graph-level late fusion.
- All arms scored on common-four (Mn, Cu, Zn, Class VIII = Fe+Co+Ni).
- Schedule: the fits behind the findings below and the step F refit used a 50-epoch cosine schedule and its terminal
  checkpoint. The exceptions are the separate exploratory 10-epoch comparison (4 fits on folds 1–4; not adopted) and
  the fold-0 runtime probes of step B.

## 1. Validation findings (development validation; no test data)

| Finding | Design | Grade | Log |
|---|---|---:|---|
| **Neutral test:** no supported gain from five- or six-class training on common-four BA for any family (this does not show that they cannot help), so the tie rule keeps `four_class`. Native rare classes stay weak (six-class Co 2.8–5.3%, Ni 25.3–31.2%) | Five grouped folds, seed 42; paired t intervals, Bonferroni over six | 2 | [v3-031](../../campaigns/pmm_ion_metal_v3/log.md#v3-031) |
| **Stage 6 selection:** `gvp_late_fusion__four_class__baseline`, mean common-four BA **74.28%** (Mn 75.6, Cu 73.4, Zn 81.3, Class VIII 66.8). It beat the Only-ESMC four-class control (72.33%) on all five folds: mean +1.95, bootstrap [+1.09, +2.92], t [+0.50, +3.39] | Five grouped folds, seed 42 | 2 | [v3-031](../../campaigns/pmm_ion_metal_v3/log.md#v3-031) |
| **Step D recipes:** late fusion `headdrop03` showed no positive mean gain on folds 1–4; Only-GVP `meanagg` showed a positive mean gain that was not interval-supported. Neither demonstrated an improvement under the specified gates | Fold-0 screen (seeds 42 and 43), then folds 1–4 | 3 (screen); 2 (confirmation) | [v3-031](../../campaigns/pmm_ion_metal_v3/log.md#v3-031) |
| **E2:** no candidate was eligible to replace the selection. Only-GVP `wd10` gained +3.59 within Only-GVP (interval-supported, but not better than the selection); late fusion `esmdrop02` gained +0.53, not interval-supported | Folds 1–4, seed 42, 36 fits; exploratory, post hoc | 2 (exploratory) | [v3-037](../../campaigns/pmm_ion_metal_v3/log.md#v3-037) |
| **10-epoch schedule:** mean +1.99, bootstrap [−0.39, +5.90], t [−4.28, +8.26]; **not adopted** | Folds 1–4, seed 42, 4 fits; post hoc | 2 (exploratory) | [v3-041](../../campaigns/pmm_ion_metal_v3/log.md#v3-041) |

**Limitations:**
- One training seed (42).
- Folds 1–4 were reused across A4, E2 and the schedule comparison.
- The terminal-checkpoint rule was chosen after a v2 post hoc analysis.
- E2 and the schedule comparison are not independently confirmed.

## 2. Secondary test result (label `both_results_secondary`)

The selection was refit on the full non-test cohort: 50 epochs, seed 42, no validation, calibration or ensemble;
checkpoint `a0007846…`; the reload check passed. The rules and code were frozen in two dated freezes before any label
was read. The test-side preparation and ESMC embeddings ran label-blind on the workstation CPU, followed by one
prediction pass and the frozen report ([v3-045 to v3-048](../../campaigns/pmm_ion_metal_v3/log.md#v3-048)). PMM's
exact Zenodo test had 1,488 rows: 1,485 were scored, and the 3 unscorable rows count as errors.

| Line | Common-four BA (95% PDB-entry bootstrap) | Mn | Cu | Zn | Class VIII |
|---|---|---:|---:|---:|---:|
| Full test | **67.55%** [62.93, 71.65] | 55.69 | 70.31 | 85.07 | 59.13 |
| LINK-free sensitivity | 68.99% [64.40, 73.13] | 56.88 | 72.13 | 85.63 | 61.30 |
| Training-eligible sensitivity | 69.33% [64.85, 73.33] | 56.88 | 73.33 | 85.83 | 61.30 |
| Clean subset (secondary; not pristine; 423 rows) | 48.85% [39.39, 57.34] | 32.43 | 43.75 | 82.61 | 36.62 |

- **Beside PMM's published Figure 2b values** (full test only; not a matched comparison, with no interval, p-value or
  superiority claim): the mean of 67.55 is lower than PMM's 67.85. Mn is lower (55.69 against 88.60); Cu, Zn and
  Class VIII are higher. PMM's features carry a true-metal label leak (TECH-028).
- **Disclosures:**
  - the full test is an "exact, possibly overlapped PMM test" that includes rows from previously opened tests;
  - the exact Zenodo test may have been evaluated on 2026-09-24;
  - its metadata was read in aggregate on 2026-10-03, and its bytes were scanned without display on 2026-10-11;
  - the selection reused development folds;
  - the clean subset cannot exclude pockets whose context chains are all shorter than 30 residues;
  - the test embeddings were generated on the CPU in bfloat16.
- **Observations, not causal claims:**
  - the full-test BA is below the cross-validation mean (74.28), and Mn drops most (55.7 against 75.6);
  - the clean subset, which has no training PDB ID, near-copy chain or previously opened ID, scores much lower on few
    rows (Mn 37, Cu 16).
- **Use:** nothing was selected or tuned after the label read. The exact Zenodo test is now evaluated once by v3.

## 3. Deferred, never run (untested, not disproven)

- The Only-ESMC fairness arm (proposal of log v3-021): kept separate, never approved.
- The cost-gated augmentations `posnoise01` and `outerdrop01` (Only-GVP): never given the cost OK.
- Round C.
- The hypotheses deferred in log v3-039: trunk learning rate 1e-4, three GVP layers, ESM dropout after pooling, a
  LayerNorm decay exemption, population-normalized weighting and label smoothing.
- A matched PMM rerun on the same rows, which is the only fair head-to-head comparison.
- More training seeds.

## Evidence and cost

- **Evidence** (Data1, not copied under `raw/`): `/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v3/`. The
  refit is at `durable/lane0/runs/stage6b_refit__…`, the records at `step_f_evidence/`, and the preparation,
  embeddings, predictions and report at `step_f_test/`.
- **Hashes:** in the log and the DATASETS ledger.
- **Backup coverage:** see the [handoff, section 13](../../campaigns/pmm_ion_metal_v3/handoff.md#13-closeout-proposal-2026-10-13-pending-the-users-approval).
- **GPU spend:** about $41.8 estimated gross of the $55 ceiling. The test side used no GPU time. Storage bills until
  the disk is retired.
