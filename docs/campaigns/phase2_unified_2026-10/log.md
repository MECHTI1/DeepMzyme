# Phase 2 Unified & 180-Checkpoint OOF-CV / Held-Out Evaluation Log
Public-release note (2026-10-20, extended 2026-10-22): host paths, device names and cloud resource names below are generalized; inside the preserved STATUS blocks, role names are also neutralized. Facts, numbers, dates, IDs, decisions, hashes and timings are unchanged; the exact earlier wording of those blocks remains in this file's Git history.

Dated entries, newest first. `EXPERIMENT_STATUS.md` is overwritten only after preserving its previous text here.

## p2u-006 (2026-10-23 — STATUS handed to the EC class vs CLEAN30 campaign)

The maintainer approved gate G0 of [ec_class_vs_clean30_2026-10](../ec_class_vs_clean30_2026-10/README.md) on
2026-10-23, so `EXPERIMENT_STATUS.md` now names it as the current campaign. This campaign stays open: the metal
deployment default is not yet chosen. No run, refit or test access.

Preserved `EXPERIMENT_STATUS.md` text before update:

```markdown
# DeepMzyme Current Experiment Status

- Status: active (2026-10-19 metal ensembles compared on OOF; deployment default not yet chosen)
- Last execution evidence: 2026-10-20 CPU-only [matched PinMyMetal comparison](docs/campaigns/phase2_unified_2026-10/pmm_matched/README.md): pre-registered; opened test scored.

## Current objective and stage

- Current campaign: [phase2_unified_2026-10](docs/campaigns/phase2_unified_2026-10/README.md) (evaluation and release audit complete).
- Objective: choose a deployment ensemble on OOF evidence. The 55-model is the best-CV reference and the predictor's code default; the 20-, 25- and 30-model late-fusion ensembles are the compact candidates.
- Stage: evaluation complete; no rerun, refit, calibration or promotion.
- Other open campaigns: [EC support](docs/campaigns/ec1_independent_support_2026-10/README.md) deferred; [metal_single_gpu](docs/campaigns/metal_single_gpu/README.md) and [exact Zenodo](docs/campaigns/zenodo_pmm_exact_2026-09-24/README.md) paused.

## Anchor and evidence state

- Best validation result: 55-model ensemble, `77.17%` pooled OOF, with no CI-supported gain over the 20-, 25- or 30-model late-fusion ensembles (`76.34–76.81%`), so [Plan §7](Plan.md#7-baseline-first-model-comparison-policy) does not justify its extra members; nested selection gives `76.56%` (late fusion) and `76.54%` (55-model blocks). Pre-declared single model: `B 4-class`, `74.34%` OOF; refit `67.55%` Full test ([v3 summary](docs/notebook_outputs/summaries/summary_pmm_ion_metal_v3_20261013.md)). Test scores never select.

### Known caveats and open mismatches

- The 20-, 25- and 30-model recipes follow the full-test order (possibly test-informed; OOF-only nested selection reaches a similar level). Larger ensembles average more checkpoints per OOF site, which favours them; OOF-selected scores are optimistic (nested: −0.25 / −0.63 pp). Test disclosures: [DATASETS](docs/DATASETS.md#test-use-ledger), [TECH-028](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-028--pmm-comparator-inherits-a-true-metal-label-leak).
- EC remains fixed-split Grade 3 evidence; TECH-030/031 stay open ([issues](docs/FOLLOW_UP_TECHNICAL_ISSUES.md)).

## Dataset and test readiness

[DATASETS](docs/DATASETS.md#test-use-ledger) records both test evaluations (Step F, 2026-10-13; 180 checkpoints, 2026-10-19). Choose and freeze the default on OOF; an independent cohort can then evaluate it once (temporal PDB [protocol](docs/campaigns/metal_completion_2026-10/evaluation.md), optional).

## Blockers and immediate next action

- Authorized now: nothing is running.
- GPU/VM: none in use.
- Next: choose the deployment default, write the metal paper, or open standalone EC work.

## History and update rule

The [campaign log](docs/campaigns/phase2_unified_2026-10/log.md) preserves the previous STATUS. Policy: [Plan](Plan.md), [DATASETS](docs/DATASETS.md), [PARAMETER_FINDINGS](docs/PARAMETER_FINDINGS.md).
```

## p2u-005 (2026-10-20 — Matched Modality × Target × Ensemble Fairness Audit)

CPU-only fairness audit of the existing saved OOF (`nested_selection.json`) and 180-checkpoint held-out test probability archive (`test_checkpoint_probs.npz`) to separate **strictly matched apples-to-apples comparisons** across `Only-GVP`, `Only-ESM-C 600M`, post-hoc unimodal probability averaging (`0.5 Only-ESM + 0.5 Only-GVP`), and learned `Late Fusion` (all trained with identical Seed 42 Recipe `B` hyperparameters across `four_class`, `five_class`, and `six_class`; 45 checkpoints total) from **downstream within-`Late Fusion` optimizations** (`trunklr1e4`, Loss Recipes `P`/`S`, and 3-seed `s42/43/44` confirmation). No new training or test pass was run; all held-out test metrics below carry the `both_results_secondary` disclosure in [`DATASETS.md`](../../DATASETS.md#test-use-ledger).

### 1. Strictly Matched Modality, Target & 15-Checkpoint Ensemble Table (`Seed 42`, Recipe `B`)

| Tier / Model Variant | Ckpts | Mean / Pooled OOF CV (`7,398`) | Full Test (`1,488`) `[95% CI]` | In-Scope Eligible (`1,346`) `[95% CI]` | Clean (`423`) `[95% CI]` | Paired Test Contrast (`Late Fusion − Control` or `Variant − Baseline`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Matched 4-Class Baseline (`s42`)** | | | | | | |
| `Only-GVP` 4-Class (`wd=1e-4`) | 5 | 69.66% / 69.71% | 64.35% `[59.74, 68.57]` | 66.37% `[61.66, 70.72]` | 53.37% `[45.22, 61.20]` | *3D-only 4c control* |
| `Only-ESM-C` 4-Class (`wd=1e-4`) | 5 | 72.33% / 72.38% | 67.72% `[63.13, 71.93]` | 69.56% `[65.03, 73.65]` | 52.74% `[43.13, 61.36]` | *Seq-only 4c control* |
| **`Late Fusion` 4-Class (`Baseline B4`)** | **5** | **74.28% / 74.34%** | **69.02%** `[64.47, 73.05]` | **70.75%** `[66.19, 74.78]` | 52.32% `[42.79, 60.87]` | vs. `ESM 4c`: **+1.30** `[-1.17, +3.76]` Full / **+1.18** `[-1.35, +3.82]` Elig; vs. `GVP 4c`: **+4.67** `[+0.93, +8.39]` Full / **+4.38** `[+0.57, +8.36]` Elig |
| **Matched 5-Class Baseline (`s42`)** | | | | | | |
| `Only-GVP` 5-Class (`5c` $\to$ `4c`) | 5 | — / 68.51% | 63.84% `[58.99, 68.21]` | 65.97% `[61.32, 70.38]` | 53.51% `[44.42, 62.37]` | *3D-only 5c control* |
| `Only-ESM-C` 5-Class (`5c` $\to$ `4c`) | 5 | — / 72.80% | 68.19% `[63.63, 72.42]` | 69.86% `[65.35, 73.96]` | 54.62% `[44.95, 63.06]` | *Seq-only 5c control* |
| **`Late Fusion` 5-Class (`Baseline B5`)** | **5** | — / **74.03%** | **69.48%** `[64.84, 73.62]` | **70.97%** `[66.45, 75.13]` | 53.85% `[44.35, 62.45]` | vs. `ESM 5c`: **+1.29** `[-1.26, +3.70]` Full / **+1.11** `[-1.55, +3.67]` Elig; vs. `GVP 5c`: **+5.63** `[+2.34, +9.06]` Full / **+5.01** `[+1.71, +8.40]` Elig |
| **Matched 6-Class Baseline (`s42`)** | | | | | | |
| `Only-GVP` 6-Class (`6c` $\to$ `4c`) | 5 | — / 67.67% | 64.97% `[60.30, 69.14]` | 66.95% `[62.26, 71.24]` | 53.16% `[45.02, 60.58]` | *3D-only 6c control* |
| `Only-ESM-C` 6-Class (`6c` $\to$ `4c`) | 5 | — / 72.32% | 68.54% `[64.01, 72.77]` | 70.15% `[65.66, 74.27]` | 57.00% `[47.39, 65.52]` | *Seq-only 6c control* |
| **`Late Fusion` 6-Class (`Baseline B6`)** | **5** | — / **74.98%** | **70.65%** `[66.02, 74.85]` | **72.07%** `[67.46, 76.24]` | **57.99%** `[48.37, 66.56]` | vs. `ESM 6c`: **+2.11** `[-0.51, +4.60]` Full / **+1.92** `[-0.73, +4.52]` Elig; vs. `GVP 6c`: **+5.68** `[+2.27, +9.18]` Full / **+5.12** `[+1.55, +8.74]` Elig |
| **Matched 15-Ckpt Multi-Target Ensembles (`4c + 5c + 6c`, `s42`)** | | | | | | |
| `Only-GVP` (`4c + 5c + 6c` Ens) | 15 | — / 71.52% | 64.70% `[60.11, 68.93]` | 66.82% `[62.21, 71.19]` | 54.37% `[46.35, 61.83]` | *3D-only 15-ckpt control* (`+1.81 pp` CV, `+0.35 pp` Full over `GVP 4c`) |
| `Only-ESM-C` (`4c + 5c + 6c` Ens) | 15 | — / 72.74% | 68.34% `[63.81, 72.55]` | 69.99% `[65.45, 74.07]` | 55.18% `[45.63, 63.65]` | *Seq-only 15-ckpt control* (`+0.36 pp` CV, `+0.62 pp` Full over `ESM 4c`) |
| **`Late Fusion` (`4c + 5c + 6c` Ens)** | **15** | — / **75.73%** | **69.77%** `[65.17, 73.91]` | **71.37%** `[66.84, 75.50]` | 54.30% `[44.87, 62.88]` | vs. `ESM 15-ckpt`: **+2.99 pp** CV, **+1.43** `[-1.03, +3.82]` Full / **+1.38** `[-1.16, +3.88]` Elig; vs. `GVP 15-ckpt`: **+4.21 pp** CV, **+5.07** `[+1.73, +8.46]` Full / **+4.55** `[+1.12, +8.12]` Elig |
| **Post-Hoc Unimodal Averaging vs. Learned `Late Fusion` (`s42`)** | | | | | | |
| `Post-Hoc 0.5 Only-ESM + 0.5 Only-GVP` (`4c`) | 10 | 73.79%\* / 73.37% | 68.93% `[64.33, 73.15]` | 70.68% `[66.05, 74.78]` | 56.75% `[47.05, 65.46]` | `LF 4c (5 ckpts) − Post-Hoc (10 ckpts)`: **+0.97 pp** pooled CV (**+0.69 pp** f1–4); **+0.09** `[-2.67, +2.85]` Full / **+0.07** `[-2.83, +3.07]` Elig |
| `Post-Hoc 0.5 Only-ESM + 0.5 Only-GVP` (`4c+5c+6c`) | 30 | — / 75.31% | 69.33% `[64.67, 73.58]` | 70.87% `[66.24, 75.08]` | 55.42% `[45.69, 64.15]` | `LF 15-ckpt − Post-Hoc (30 ckpts)`: **+0.42 pp** pooled CV; **+0.45** `[-1.96, +2.90]` Full / **+0.50** `[-2.04, +3.08]` Elig |
| **Matched Regularization Screen (`Only-GVP` & `Late Fusion`, `s42`)** | | | | | | |
| `Only-GVP` Mean Aggregation (`meanagg`, `4c`) | 5 | 71.10%\* / 70.56% | 64.86% `[60.18, 69.19]` | 66.86% `[62.12, 71.23]` | 51.05% `[41.95, 59.81]` | vs. `GVP 4c`: **+0.85 pp** pooled CV; **+0.51** `[-2.77, +3.68]` Full / **+0.49** `[-2.85, +3.89]` Elig |
| `Only-GVP` Heavy $L_2$ Decay (`wd=1.0`, `4c`) | 5 | 73.25% / 72.94% | 63.35% `[58.52, 67.74]` | 65.02% `[60.22, 69.48]` | 51.58% `[42.06, 60.32]` | vs. `GVP 4c`: **+3.23 pp** pooled CV, but **-1.00** `[-4.04, +1.94]` Full / **-1.35** `[-4.53, +1.84]` Elig Test regression |
| `Late Fusion` Heavy $L_2$ Decay (`Global wd=1.0`, `4c`) | 5 | — / 74.08% | 67.98% `[63.30, 72.22]` | 69.57% `[64.90, 73.80]` | 52.04% `[42.48, 60.71]` | vs. `LF 4c`: **-0.26 pp** pooled CV; **-1.04** `[-2.92, +0.80]` Full / **-1.18** `[-3.07, +0.62]` Elig |
| **Within-`Late Fusion` Trunk-LR Optimization *(Not Run on Unimodal Arms)*** | | | | | | |
| **`Late Fusion` Differential Trunk LR (`trunklr1e4`, `4c`)** | **5** | **75.45% / 75.51%** | **70.75%** `[66.29, 74.77]` | **72.25%** `[67.87, 76.30]` | 55.87% `[46.26, 64.42]` | vs. `LF 4c`: **+1.17 pp** pooled CV; **+1.73** `[+0.01, +3.58]` Full / **+1.50** `[-0.30, +3.43]` Elig / **+3.56** `[+0.46, +7.93]` Clean |

*\*Mean-fold CV on confirmation folds 1–4 (`v3-028` / `v3-032`).*

### 2. Key Fairness & Generalization Conclusions
1. **Strictly Matched Advantage of `Late Fusion`:** Across all three target schemes (`4c`, `5c`, `6c`) and in the matched 15-checkpoint (`4c + 5c + 6c`) ensemble at Seed 42 (Recipe `B`), `Late Fusion` consistently exceeds both `Only-ESM-C` (`+1.23 to +2.66 pp` CV; `+1.29 to +2.11 pp` Full Test; `+1.11 to +1.92 pp` Eligible Test) and `Only-GVP` (`+4.63 to +7.31 pp` CV; `+4.67 to +5.68 pp` Full Test, with all paired 95% CIs vs. `Only-GVP` strictly above zero).
2. **`Only-GVP wd=1.0` Did Not Generalize to Held-Out Test:** Although heavy $L_2$ weight decay (`wd = 1.0`) increased `Only-GVP` 5-fold CV BA by `+3.59 pp` mean-fold (`+3.23 pp` pooled), it regressed on the held-out test set by `-1.00 pp` Full / `-1.35 pp` Eligible relative to `Only-GVP baseline (wd=1e-4)`. Even pooling all 30 `Only-GVP` checkpoints across all 6 `Only-GVP` families reaches `65.36%` Full / `67.39%` Eligible—still `> 3.3 pp` below 5-checkpoint `Late Fusion B4`.
3. **Asymmetric Downstream Funnel Disclosure:** `Only-ESM-C` received no Step D/E2 regularization screen (`v3-022`), `Only-GVP` was not evaluated in a 5-fold `lr=1e-4` campaign, and `metal_completion_2026-10` (Recipes `B/P/S` and seeds `43/44`) was executed only on `Late Fusion`. Accordingly, unimodal-vs-bimodal claims are anchored strictly to the matched Seed-42 Recipe `B` comparisons above.

## p2u-004 (2026-10-20 — Matched PinMyMetal comparison)

Maintainer-authorized, pre-registered matched comparison on the opened exact Zenodo PinMyMetal test: PinMyMetal's released
classifier refit on the same 7,398 training sites and folds (CPU, pinned environment), scored with DeepMzyme's frozen
single model and CV-selected ensembles on the same 1,488 test rows. Results, amendments A1/A2 and disclosures:
[matched comparison report](pmm_matched/README.md); test access: [ledger](../../DATASETS.md#test-use-ledger).
Documentation repaired with it: README and DATASETS comparison claims; PinMyMetal's published CV value relabelled as
macro-precision and its 61.70% as its own score on a Metal3D-derived set (also in the two PinMyMetal reproducibility
docs). Previous `EXPERIMENT_STATUS.md` line, replaced: "- Last execution evidence: 2026-10-19 CPU-only [release
audit](docs/campaigns/phase2_unified_2026-10/release_audit/README.md): OOF scores, nested selection, parity, timing; no
training or test access."

## p2u-003 (2026-10-19 — Release audit and ensemble comparison)

CPU-only audit of the existing checkpoints from saved development artifacts: no training, refit, calibration, GPU use
or test-set access. Results and commands are in the [release audit](release_audit/README.md).

- **OOF reconstruction:** every shortlisted ensemble reproduces its reported pooled OOF score (`74.34%` to `77.17%`),
  with each checkpoint scoring only its excluded fold. The catalog now also covers the 15-model P 5-class and the
  25-model late-fusion ensembles.
- **Ensemble size:** walking the [Plan §7](../../../Plan.md#7-baseline-first-model-comparison-policy) sequential rule
  from the two five-checkpoint models stops at the 20-, 25- or 30-model, depending on the interval and the start, and
  never reaches the 55-model: its OOF gain over those ensembles is not CI-supported by the near-copy-group or the
  fold-level interval. Larger ensembles also average more checkpoints per OOF site, which favours them. The 55-model
  remains the best-CV reference and the predictor's code default; the deployment default is not yet chosen. A
  practical-similarity bar, set after the results were seen, is met by the 25- and 30-model only.
- **Nested selection:** choosing the ensemble on four folds and scoring the fifth gives `76.56%` for late fusion (fixed
  30-model `76.81%`) and `76.54%` for the 55-model's three-block procedure (fixed `77.17%`); the audit lists its limits.
- **Corrections to p2u-001:** `resdrop01`, `meanagg`, `invsqrtw`, `esmdrop02` and `wd10`, listed there without OOF,
  have complete five-fold seed-42 OOF on the same folds (`74.63%`, `74.29%`, `74.63%`, `75.26%`, `74.08%`). The
  recomputed pooled OOF in `families` of [`nested_selection.json`](release_audit/nested_selection.json) supersedes
  seven other entries: S 4-class `74.63%` (listed `74.12%`), P 4-class three seeds `75.62%` (`75.55%`), P 6-class
  three seeds `75.22%` (`75.16%`), `headdrop03` `74.85%` (`74.76%`), five-class `74.03%` (`73.99%`), Only-ESM 6-class
  `72.32%` (`72.42%`) and Only-GVP 6-class `67.67%` (`69.63%`); the 15-model B 4-class, listed by fold mean, pools to
  `75.58%`. The 20-, 25- and 30-model recipes follow the full-test order in that table, not the OOF order.
- **Prediction parity and cost:** on six development sites the predictor reproduced the saved validation predictions
  of all 55 reference members (66 comparisons, maximum probability difference `6.85e-8`). A first replay failed on
  tuple/list differences in saved configuration; the comparison was made canonical and the failed attempt kept.
  One-site CPU time ranges from `4.46 s` (1 checkpoint) to `16.72 s` (55). Parity and timing predate the two added
  catalog entries; `predict.py` is unchanged since.
- **Machine-bound files:** the release manifest, its inventory and receipt, and the parity and benchmark fixtures
  record local paths and are kept outside the repository.

### Previous STATUS (before this update; lightly redacted)

```markdown
# DeepMzyme Current Experiment Status

- Status: active (2026-10-19 metal OOF-CV and held-out evaluation complete; GPU VM TERMINATED)
- Last execution evidence: 2026-10-19 Phase 2 Unified 5-fold L4 run (`1h 09m`, $1.02) and 180-checkpoint OOF-CV / test evaluation ([report](docs/campaigns/phase2_unified_2026-10/README.md); [log](docs/campaigns/phase2_unified_2026-10/log.md)).

## Current objective and stage

- Current campaign: [phase2_unified_2026-10](docs/campaigns/phase2_unified_2026-10/README.md) (complete; [metal_completion_2026-10](docs/campaigns/metal_completion_2026-10/README.md) complete)
- Objective: metal ion classification benchmark complete; EC classification is next when authorized.
- Other open campaigns: [EC independent support](docs/campaigns/ec1_independent_support_2026-10/README.md) (deferred); [metal_single_gpu](docs/campaigns/metal_single_gpu/README.md) (paused); [zenodo_pmm_exact_2026-09-24](docs/campaigns/zenodo_pmm_exact_2026-09-24/README.md) (paused).
- Stage: one-shot Step F refit (`67.55%` Full Test) and 180-checkpoint 5-Fold OOF-CV / held-out progression complete ([summary JSON](docs/campaigns/phase2_unified_2026-10/phase2_unified_summary.json)).

## Anchor and evidence state

- Best validation result: pre-declared single-model `B 4-class` baseline (`74.28%` mean / `74.34%` OOF CV; Step F refit `67.55%` Full, `69.33%` Eligible, `48.85%` Clean; [v3 summary](docs/notebook_outputs/summaries/summary_pmm_ion_metal_v3_20261013.md)). By 7,398-ion OOF CV ([report](docs/campaigns/phase2_unified_2026-10/README.md)): 5-Fold `trunklr1e4` (`75.51%` OOF $\rightarrow$ `70.75%` Full / `72.25%` Eligible); 15-Model `P 5-class` (`75.82%` OOF $\rightarrow$ `71.10%` Full / `72.68%` Eligible); 5-Fold Unified `P+5c+trunklr1e4` (`74.91%` OOF $\rightarrow$ `71.72%` Full / `73.08%` Eligible); `Top-4 LF` (`76.81%` OOF $\rightarrow$ **`72.20%` Full** / `73.94%` Eligible); `CV-Locked Tri-Modal` (**`77.17%` OOF** $\rightarrow$ **`61.33%` Clean**).
- EC stays at fixed-split Grade 3 evidence (deferred).

### Known caveats and open mismatches

- Cross-target ranking used the campaign assessor ([TECH-010](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-010--four-class-endpoint-and-paired-metal-target-recipes-are-not-reconciled)); post-Step-F multi-checkpoint evaluations carry `both_results_secondary` disclosure and PMM label-leak caveat ([TECH-028](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-028--pmm-comparator-inherits-a-true-metal-label-leak)).
- EC child-file guard ([TECH-030](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-030--discovered-ec-cache-files-can-bypass-the-training-path-guard)) and focal-loss label smoothing ([TECH-031](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-031--the-focal-metal-loss-silently-ignores-label-smoothing-and-uses-weighted-ce-confidence)) remain open.

## Dataset and test readiness

[DATASETS](docs/DATASETS.md#test-use-ledger) records both test evaluations (2026-10-13 Step F pass and 2026-10-19 180-checkpoint evaluation). External temporal PDB evaluation remains optional ([protocol](docs/campaigns/metal_completion_2026-10/evaluation.md)).

## Blockers and immediate next action

- Authorized now: nothing to run; no active GPU session.
- GPU/VM: TERMINATED (2026-10-19).
- Next: metal paper writing, or a decision to open standalone EC classification.

## History and update rule

Previous STATUS is preserved in [`docs/campaigns/phase2_unified_2026-10/log.md`](docs/campaigns/phase2_unified_2026-10/log.md). Policy: [Plan](Plan.md), [DATASETS](docs/DATASETS.md), [PARAMETER_FINDINGS](docs/PARAMETER_FINDINGS.md).
```

---

## p2u-002 (2026-10-19 — Verification Audit, OOF-CV Reconciliation & Documentation Repair)

- **Audit of OOF-CV vs. Held-Out Test Selection:**
  - Verified exact forward-pass parity of `stage6b_refit__gvp_late_fusion__four_class__baseline__e50__seed42` against official Step F `test_predictions.csv` ($\max |\Delta p| = 1.61 \times 10^{-6}$, reproducing `67.55%` Full, `68.99%` LINK-Free, `69.33%` Training-Eligible, `48.85%` Clean Subset).
  - Fixed a wildcard glob bug in the earlier ad-hoc OOF script that had matched `sched10` 10-epoch runs for `5-Fold Baseline 4-class (s42)` (`75.30%`) instead of the verified 50-epoch baseline controls (`probe-full-fp32` fold 0 + `gvp_late_fusion__four_class__baseline` folds 1–4). From the verified `val_predictions.csv` files, the true `5-Fold Baseline 4-class (s42)` Cross-Validation score is **`74.34%` Pooled OOF BA / `74.28%` 5-Fold Mean BA** (`[71.62, 73.34, 72.96, 75.17, 78.32]`).
  - With the true baseline OOF CV (`74.34%`) restored, every promoted single-family and multi-family ensemble shows a strictly positive gain on **both** 5-Fold Cross-Validation (`7,398` sites) and the Held-Out Test Set (`1,488` sites):
    - `5-Fold Baseline 4-class (s42)`: `74.34%` OOF (`74.28%` mean) $\rightarrow$ `69.02%` Full / `70.75%` Eligible / `52.32%` Clean
    - `5-Fold Recipe P 5-class (s42)`: `74.48%` OOF (`74.42%` mean) $\rightarrow$ `69.90%` Full / `71.65%` Eligible / `59.05%` Clean
    - `5-Fold Unified P + 5-class + trunklr1e4 (s42)`: `74.91%` OOF (`74.84%` mean) $\rightarrow$ `71.72%` Full / `73.08%` Eligible / `57.58%` Clean
    - `5-Fold 6-class collapsed (s42)`: `74.98%` OOF $\rightarrow$ `70.65%` Full / `72.07%` Eligible / `57.99%` Clean
    - `5-Fold trunklr1e4 4-class (s42)`: `75.51%` OOF (`75.45%` mean) $\rightarrow$ `70.75%` Full / `72.25%` Eligible / `55.87%` Clean
    - `15-Model Multi-Seed Recipe P 5-class (s42,43,44)`: `75.82%` OOF $\rightarrow$ `71.10%` Full / `72.68%` Eligible / `58.21%` Clean
    - `Top-3 LF Group (trunklr1e4 + P5_3s + 6c)`: `76.62%` OOF $\rightarrow$ `71.70%` Full / `73.46%` Eligible / `58.39%` Clean
    - `Top-4 LF Group (trunklr1e4 + P5_3s + S5 + 6c)`: **`76.81%` OOF** *(Highest Late-Fusion OOF CV)* $\rightarrow$ **`72.20%` Full / `73.58%` LINK-Free / `73.94%` Eligible / `59.18%` Clean**
    - `CV-Locked Tri-Modal + Unified (1.00 Top-5 LF + 0.25 ESMC + 0.95 GVP)`: **`77.17%` OOF** *(Highest Overall OOF CV)* $\rightarrow$ `71.00%` Full / `72.76%` Eligible / **`61.33%` Clean Subset** (`60.18%` on the 563-row training-disjoint `C1+C2` subset).
  - Repaired subset denominators across all tables to match `preparation_summary.json`: `Full Test (1,488)`, `LINK-Free (1,349)`, `Training-Eligible (1,346)`, `PDB-Dedup (1,098 pairs across 864 PDB IDs)`, and `Clean Subset (423)` (plus `563` training-disjoint `C1+C2` rows).
  - Updated [`eval_phase2_unified.py`](eval_phase2_unified.py) and [`phase2_unified_summary.json`](phase2_unified_summary.json) so that all 12 primary benchmark rows and bootstrap CIs are reproduced by one command using either the secondary drive or the private local backups.

Preserved `EXPERIMENT_STATUS.md` text before update:

```text
# DeepMzyme Current Experiment Status

- Status: active (2026-10-16 optimization complete; B four-class retained; Stage 6B source verified; independent evaluation not started)
- Last execution evidence: 2026-10-16 confirmation fits on the GCP L4 (69 campaign fits, seven sessions); no test evaluations.

## Current objective and stage

- Current campaign: [metal_completion_2026-10](docs/campaigns/metal_completion_2026-10/README.md)
- Objective: independently evaluate the selected metal model; EC is deferred (the maintainer's metal-first choice).
- Other open campaigns: [EC independent support](docs/campaigns/ec1_independent_support_2026-10/README.md) (deferred); [metal_single_gpu](docs/campaigns/metal_single_gpu/README.md) (paused); [zenodo_pmm_exact_2026-09-24](docs/campaigns/zenodo_pmm_exact_2026-09-24/README.md) (paused).
- Stage: optimization complete (validation only): discovery (45 cells) froze recipe P; Stage 6 confirmation (40 cells) passed no challenger, so the unchanged B four-class v3 recipe is retained. Stage 6B source: v3 full non-test refit `a0007846…`, compatible (25/25 CPU checks), designated without retraining under the [evidence](docs/campaigns/metal_completion_2026-10/evidence.md#stage-6b-source-and-compatibility-2026-10-16) conditions. Stage 7 has not started; the [log](docs/campaigns/metal_completion_2026-10/log.md) records decisions.

## Anchor and evidence state

- Best validation result: retained v3 late-fusion direct-four, terminal 50-epoch cosine. Grade 2: five grouped folds, seed 42, common-four BA 74.28%; no supported benefit from five/six training ([v3](docs/PARAMETER_FINDINGS.md#pmm-ion-metal-v3-objective-test-and-stage-6-selection-closed-2026-10-13)) or from recipes P/S ([completion](docs/PARAMETER_FINDINGS.md#metal-completion-bps-loss-recipes-on-fourfivesix-objectives-executed-2026-10-1516)).
- Trunk LR unpromoted ([closed](docs/archive/campaigns/pmm_trunk_lr_2026-10/README.md)).
- EC stays at fixed-split Grade 3 evidence (deferred).

### Known caveats and open mismatches

- Seeds 43/44 reuse development folds; cross-target ranking used the campaign assessor ([TECH-010](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-010--four-class-endpoint-and-paired-metal-target-recipes-are-not-reconciled)). The refit already had one secondary held-out pass ([summary](docs/notebook_outputs/summaries/summary_pmm_ion_metal_v3_20261013.md)); a primary Stage 7 report from it conflicts with Plan's one-shot rule until the maintainer approves an exception or amends Plan.
- EC child-file guard repair remains open ([TECH-030](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-030--discovered-ec-cache-files-can-bypass-the-training-path-guard)); the trainer's focal branch ignores label smoothing, unreachable here ([TECH-031](docs/FOLLOW_UP_TECHNICAL_ISSUES.md#tech-031--the-focal-metal-loss-silently-ignores-label-smoothing-and-uses-weighted-ce-confidence)).

## Dataset and test readiness

The primary independent final-test route remains unresolved; the reused refit is valid only for an external cohort disjoint from its training cohort, every opened test and their near-copies. [DATASETS](docs/DATASETS.md#test-use-ledger) owns historical exposure; opened PMM tests are not pristine. The proposed temporal PDB cohort is neither assembled nor certified; the [protocol](docs/campaigns/metal_completion_2026-10/evaluation.md#preparation-inventory-2026-10-16) owns its inventory and blockers.

## Blockers and immediate next action

- Authorized now: nothing to run; the 2026-10-15 GPU authorization is consumed. No GPU start, refit, test access, download, commit or push.
- GPU/VM: `<gpu-vm>` TERMINATED (confirmed 2026-10-16T11:59Z) after seven sessions: 18 h 37 min, $16.37 gross running; $0.87917/h; storage ~$1/day. Backups: the private backup record.
- Next: the maintainer's route and one-shot decisions, then approval of the label-free prior-use read ([blockers](docs/campaigns/metal_completion_2026-10/evaluation.md#remaining-blockers); [handoff](docs/campaigns/metal_completion_2026-10/handoff.md)).

## History and update rule

The exact previous STATUS is preserved in the [campaign log](docs/campaigns/metal_completion_2026-10/log.md). Overwrite this page only after preserving dated history. Scientific policy remains in Plan, recipes in the metal playbook, dataset/test access in DATASETS, and validation conclusions in PARAMETER_FINDINGS.
```

---

## p2u-001 (2026-10-19 — Phase 1 175-Checkpoint Evaluation & Phase 2 Unified 5-Fold Training)

### 1. Comprehensive Comparison Table (`180` Checkpoints Across All 5 Test Slices)

| # | Model / Ensemble | Models | 5-Fold OOF CV (`7,398`) | Full Test (`1,488`) [95% CI] | LINK-Free (`1,349`) [95% CI] | Training-Eligible (`1,346`) [95% CI] | PDB-Dedup (`1,098`) | Clean Subset (`423`) [95% CI] | Eligible Per-Class Recalls (`Mn / Cu / Zn / VIII`) |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| — | *Metal3D Published (Fig. 2c)* | — | — | `61.70%` | — | — | — | — | — |
| — | *PinMyMetal Published (Fig. 2a/2b)* | — | `75.08%` | `67.85%` | — | — | — | — | `88.6 / 59.4 / 65.9 / 57.5` *(Full)* |
| **0** | **Official Step F Single Refit (`baseline s42`)** | **1** | — | **`67.55%`** `[62.93, 71.65]` | **`68.99%`** `[64.40, 73.13]` | **`69.33%`** `[64.85, 73.33]` | **`68.45%`** | **`48.85%`** `[39.39, 57.34]` | `56.9 / 73.3 / 85.8 / 61.3` |
| 1 | 5-Fold Baseline 4-class (`s42`) | 5 | `74.34%` *(OOF)* / `74.28%` *(mean)* | `69.02%` `[64.47, 73.05]` | `70.40%` `[65.72, 74.56]` | `70.75%` `[66.19, 74.78]` | `69.76%` | `52.32%` `[42.79, 60.87]` | `60.0 / 73.3 / 87.0 / 62.6` |
| 2 | 15-Model Multi-Seed Baseline B 4-class (`s42,43,44`) | 15 | `73.87%` *(mean)* | `69.94%` `[65.46, 74.03]` | `71.48%` `[66.77, 75.59]` | `71.83%` `[67.45, 75.81]` | `71.13%` | `53.49%` `[43.93, 62.16]` | `61.2 / 73.3 / 88.8 / 63.9` |
| 3 | 15-Model Multi-Seed Recipe P 4-class (`s42,43,44`) | 15 | `75.55%` *(OOF)* / `74.24%` *(mean)* | `70.36%` `[65.76, 74.36]` | `71.78%` `[67.15, 75.84]` | `72.13%` `[67.68, 76.06]` | `71.45%` | `55.46%` `[45.77, 64.08]` | `61.3 / 73.3 / 88.7 / 65.2` |
| 4 | 5-Fold Recipe P 5-class (`s42`, `lr=3e-4`) | 5 | `74.48%` *(OOF)* / `74.42%` *(mean)* | `69.90%` `[65.27, 74.03]` | `71.31%` `[66.65, 75.47]` | `71.65%` `[67.21, 75.75]` | `70.65%` | `59.05%` `[49.44, 67.63]` | `65.0 / 71.7 / 89.1 / 60.9` |
| **5** | **15-Model Multi-Seed Recipe P 5-class (`s42,43,44`, `lr=3e-4`)** | **15** | **`75.82%`** *(OOF)* / `73.97%` *(mean)* | **`71.10%`** `[66.65, 75.14]` | **`72.33%`** `[67.79, 76.33]` | **`72.68%`** `[68.37, 76.56]` | **`71.83%`** | **`58.21%`** `[48.51, 66.95]` | **`64.4 / 75.0 / 89.6 / 61.7`** |
| 6 | 15-Model Multi-Seed Recipe P 6-class (`s42,43,44`) | 15 | `75.16%` *(OOF)* / `73.31%` *(mean)* | `69.45%` `[64.79, 73.54]` | `70.68%` `[65.94, 74.81]` | `71.02%` `[66.44, 75.02]` | `70.30%` | `55.53%` `[45.73, 64.26]` | `62.5 / 71.7 / 89.1 / 60.9` |
| **7** | **5-Fold `trunklr1e4` 4-class (`s42`, `lr=1e-4`)** | **5** | **`75.51%`** *(OOF)* / **`75.45%`** *(mean)* | **`70.75%`** `[66.29, 74.77]` | **`71.89%`** `[67.29, 75.95]` | **`72.25%`** `[67.87, 76.30]` | **`71.43%`** | **`55.87%`** `[46.26, 64.42]` | **`62.5 / 75.0 / 86.3 / 65.2`** |
| **8** | **5-Fold Unified `P + 5-class + trunklr1e4` (`s42`, `lr=1e-4`)** | **5** | **`74.91%`** *(OOF)* / **`74.84%`** *(mean)* | **`71.72%`** `[67.34, 75.64]` | **`72.71%`** `[68.31, 76.66]` | **`73.08%`** `[68.87, 76.96]` | **`72.24%`** | **`57.58%`** `[48.09, 66.17]` | **`65.6 / 76.7 / 88.3 / 61.7`** |
| **9** | **5-Fold 6-class collapsed (`s42`)** | **5** | **`74.98%`** *(OOF)* | **`70.65%`** `[66.04, 74.74]` | **`71.73%`** `[67.07, 75.87]` | **`72.07%`** `[67.57, 76.11]` | **`71.12%`** | **`57.99%`** `[48.32, 66.58]` | **`63.7 / 71.7 / 89.4 / 63.5`** |
| 10 | 5-Fold Recipe S 5-class (`s42`, label smooth `0.1`) | 5 | `74.16%` | `70.52%` | `71.59%` | `71.96%` | `71.09%` | `57.96%` | `63.8 / 76.7 / 88.3 / 59.1` |
| 11 | 5-Fold Recipe S 4-class (`s42`, label smooth `0.1`) | 5 | `74.12%` | `69.89%` | `71.08%` | `71.45%` | `70.99%` | `56.76%` | `60.6 / 78.3 / 86.8 / 60.0` |
| 12 | 5-Fold `resdrop01` 4-class (`s42`) | 5 | — | `69.22%` | `70.85%` | `71.20%` | `70.28%` | `52.05%` | `63.1 / 73.3 / 86.6 / 61.7` |
| 13 | 5-Fold `five_class` baseline (`s42`) | 5 | `73.99%` | `69.48%` | `70.63%` | `70.97%` | `70.09%` | `53.85%` | `61.9 / 71.7 / 88.6 / 61.7` |
| 14 | 5-Fold `meanagg` 4-class (`s42`) | 5 | — | `69.08%` | `69.96%` | `70.30%` | `69.62%` | `57.90%` | `61.9 / 70.0 / 86.3 / 63.0` |
| 15 | 5-Fold `invsqrtw` 4-class (`s42`) | 5 | — | `68.82%` | `70.26%` | `70.61%` | `69.97%` | `52.22%` | `60.0 / 73.3 / 86.5 / 62.6` |
| 16 | 5-Fold `esmdrop02` 4-class (`s42`) | 5 | — | `68.40%` | `70.32%` | `70.66%` | `69.43%` | `54.48%` | `62.5 / 70.0 / 87.9 / 62.2` |
| 17 | 5-Fold `headdrop03` 4-class (`s42`) | 5 | `74.76%` | `68.20%` | `69.82%` | `70.17%` | `69.18%` | `53.33%` | `58.1 / 73.3 / 87.1 / 62.2` |
| 18 | 5-Fold `wd10` 4-class (`s42`) | 5 | — | `67.98%` | `69.23%` | `69.57%` | `69.03%` | `52.04%` | `57.5 / 71.7 / 86.9 / 62.2` |
| 19 | 5-Fold Only-ESMC 6-class (`s42`) | 5 | `72.42%` | `68.54%` | `69.79%` | `70.15%` | `69.73%` | `57.00%` | `58.1 / 75.0 / 86.6 / 60.9` |
| 20 | 5-Fold Only-GVP 6-class (`s42`) | 5 | `69.63%` | `64.97%` | `66.66%` | `66.95%` | `65.91%` | `53.16%` | `67.5 / 60.0 / 86.4 / 53.9` |
| **21** | **Top-3 LF Group (`trunklr1e4 + P5_3s + 6c`)** | **25** | **`76.62%`** | **`71.70%`** `[67.19, 75.67]` | **`73.10%`** `[68.56, 77.10]` | **`73.46%`** `[69.13, 77.41]` | **`72.57%`** | **`58.39%`** `[48.72, 66.96]` | **`65.6 / 75.0 / 89.7 / 63.5`** |
| **22** | **Top-4 LF Group (`trunklr1e4 + P5_s42 + S5_s42 + 6c_s42`, Seed 42 Only)** | **20** | **`76.34%`** | **`71.80%`** `[67.34, 75.79]` | **`73.18%`** `[68.63, 77.17]` | **`73.54%`** `[69.25, 77.43]` | **`72.61%`** | **`58.48%`** `[48.86, 67.02]` | **`65.6 / 75.0 / 89.6 / 63.9`** |
| **23** | **Top-4 LF Group (`trunklr1e4 + P5_3s + S5 + 6c`)** | **30** | **`76.81%`** | **`72.20%`** `[67.73, 76.17]` | **`73.58%`** `[69.04, 77.59]` | **`73.94%`** `[69.63, 77.90]` | **`73.11%`** | **`59.18%`** `[49.46, 67.75]` | **`66.2 / 75.0 / 89.7 / 64.8`** |
| **24** | **Top-5 LF Group (`Unified + trunklr1e4 + P5_3s + S5 + 6c`)** | **35** | **`76.37%`** | **`71.48%`** `[66.96, 75.44]` | **`72.71%`** `[68.18, 76.72]` | **`73.06%`** `[68.79, 76.98]` | **`72.12%`** | **`58.13%`** `[48.56, 66.73]` | **`66.9 / 75.0 / 89.5 / 60.9`** |
| **25** | **CV-Locked Tri-Modal (`1.00 Top-3 LF + 0.25 ESMC + 0.95 GVP`)** | **55** | **`77.05%`** | **`71.17%`** `[66.52, 75.36]` | **`72.64%`** `[67.92, 76.83]` | **`72.98%`** `[68.37, 77.11]` | **`71.89%`** | **`60.56%`** `[50.85, 69.11]` | **`65.6 / 70.0 / 90.6 / 65.7`** |
| **26** | **CV-Locked Tri-Modal + Unified (`1.00 Top-5 LF + 0.25 ESMC + 0.95 GVP`)** | **55** | **`77.17%`** | **`71.00%`** `[66.39, 75.15]` | **`72.42%`** `[67.75, 76.62]` | **`72.76%`** `[68.21, 76.88]` | **`71.64%`** | **`61.33%`** `[51.67, 69.79]` | **`68.1 / 70.0 / 90.3 / 62.6`** |

### 2. Frozen Dataset & Checkpoint Provenance (Bit-Verified SHA-256)

- **Training Cohort (`7,398` ions across `6,443` PDB structures):** `745b19c1644ee17ec24e760b81916ff97d11d3c6017e791d610bca0b6bedc3bd`
- **5-Fold Sequence-Clustered Split (`v3-seqid90-s42-b2`):** `e6b21364159da479dc3fb325a01ede36c62489184a77d858f71fefe6c401da7a`
- **Feature Inventory (13D scalar + 1D vector node, 16D RBF + 1D vector edge, ESMC-600M `1152D`):** `942d1d0e04b4dde177f7ea42cb77f0920bdd80800962624911acc4748d64793c`
- **Trainer Source Tree:** `212db0e3865968701475e7172b2e4cad772e492b35855da293238437fd97c15c`
- **Held-Out Test Set (`1,488` source rows; `1,485` scorable primary ions):** `ec6427ad0b8a18261dbc1d6822bdcfe7194730c7da4a36c208957bcb00f6a0f2`
- **Phase 2 Unified 5-Fold Terminal Checkpoints (`unified__P_five_class_trunklr1e4__fold{0..4}__seed42`):**
  - Fold 0: `e68f504c3ba6ba02bb7aa321074cfd80892c53ab2daa15cffaec9c8ea4f8aede` (`val_predictions.csv` `4ea965bc245b36c0…`)
  - Fold 1: `31f74ee5bb713a0dbd167892370f6518e1b8f2e13c9f96be4c12d3dc4b40bf0a` (`val_predictions.csv` `c8ca6bcdbb191292…`)
  - Fold 2: `44d5cec3ae15b5aa0f1f3722fd6c023519c759545b0a810d1f96c291dfc4a43e` (`val_predictions.csv` `c7f9ab65fcaa5a26…`)
  - Fold 3: `c2bc733a1fed04eaaadae6051ba81e7d98d13c5d2567f8998d4c3bff7eece561` (`val_predictions.csv` `087703b57ef7934c…`)
  - Fold 4: `0b0b818e8d1af0b8183f273d19f80c30b7cb7a30855f85f797cfe47063ec7df1` (`val_predictions.csv` `023449572e4bfd6a…`)
- **GCP Session Closeout:** Session `session-20261019T001352Z-c22f6086` on `<gpu-vm>` (`us-central1-a`) ran `1h 09m` (`\$1.02` gross cost) and is verified `TERMINATED`.

### 3. Five-Tier Backup Inventory

| Storage Tier | Location / URL | SHA-256 / Commit |
|:---|:---|:---|
| **1. GitHub (`MECHTI1/DeepMzyme`)** | [`docs/campaigns/phase2_unified_2026-10/`](README.md) | Branch `main` |
| **2. Hugging Face (`GMBioinformatics/DeepMzyme`)** | [`benchmarks/DeepMzyme_phase2_unified_and_180ckpt_eval_20261019.tar.gz`](https://huggingface.co/datasets/GMBioinformatics/DeepMzyme/resolve/main/benchmarks/DeepMzyme_phase2_unified_and_180ckpt_eval_20261019.tar.gz) (`99 MB`) | `d4d045b5dd9de56a11504639961c3e30e6a4fda5e7e6e60f5c63a563af8951fe` |
| **3–5. Maintainer local disks and the stopped GPU VM disk** | Recorded in the maintainer's private backup record (outside the repository) | Tier 3 verified via `SHA256SUMS`; tier 4 holds the five fold folders and cached graphs; tier 5 is the stopped L4 VM disk |

_Rows 3–5 condensed 2026-10-20: device names, local paths and VM details are kept out of the public repository._
