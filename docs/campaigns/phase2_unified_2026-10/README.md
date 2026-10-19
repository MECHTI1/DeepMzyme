# Phase 2: 180-Checkpoint CV and Reference-Test Evaluation

Status: active (2026-10-19 evaluation and release audit complete; deployment default not yet chosen)

Evaluation of `180` metal checkpoints (`175` prior + `5` Phase 2 Unified `P + five_class + trunklr1e4`) on 5-fold OOF CV (`7,398` sites) and the PinMyMetal reference test set (`1,488` sites). Details: [`log.md`](log.md), [`phase2_unified_summary.json`](phase2_unified_summary.json).

## 1. Benchmark Progression (5-Fold OOF CV vs. Reference Test)

| Method / Selection Tier | Models | 5-Fold CV (`7,398`) | Full (`1,488`) | LINK-Free (`1,349`) | Eligible (`1,346`) | Dedup (`1,098`) | Clean (`423`) | Eligible Recalls (`Mn/Cu/Zn/VIII`) |
|:---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Metal3D *(Published)* | 1 | — | `61.70%` | — | — | — | — | — |
| PinMyMetal *(Published)* | 1 | `75.08%` | `67.85%` | — | — | — | — | `88.6/59.4/65.9/57.5` |
| Single Model, Full-Data Refit | 1 | `74.34%` *(recipe OOF)* | `67.55%` | `68.99%` | `69.33%` | `68.45%` | `48.85%` | `56.9/73.3/85.8/61.3` |
| 5-Fold Baseline 4-class (`s42`) | 5 | `74.34%` *(OOF)* | `69.02%` | `70.40%` | `70.75%` | `69.76%` | `52.32%` | `60.0/73.3/87.0/62.6` |
| 15-Model Baseline B 4-class | 15 | `75.58%` *(OOF)* | `69.94%` | `71.48%` | `71.83%` | `71.13%` | `53.49%` | `61.2/73.3/88.8/63.9` |
| 5-Fold `trunklr1e4` 4-class (`s42`) | 5 | `75.51%` *(OOF)* | `70.75%` | `71.89%` | `72.25%` | `71.43%` | `55.87%` | `62.5/75.0/86.3/65.2` |
| 15-Model Recipe P 5-class | 15 | `75.82%` *(OOF)* | `71.10%` | `72.33%` | `72.68%` | `71.83%` | `58.21%` | `64.4/75.0/89.6/61.7` |
| 5-Fold Unified `P+5c+trunklr1e4` | 5 | `74.91%` *(OOF)* | `71.72%` | `72.71%` | `73.08%` | `72.24%` | `57.58%` | `65.6/76.7/88.3/61.7` |
| 25-Model LF (`trunklr1e4+P5_3s+6c`) | 25 | `76.62%` *(OOF)* | `71.70%` | `73.10%` | `73.46%` | `72.57%` | `58.39%` | `65.6/75.0/89.7/63.5` |
| 30-Model LF (`trunklr1e4+P5_3s+S5+6c`) | 30 | `76.81%` *(OOF)* | `72.20%` | `73.58%` | `73.94%` | `73.11%` | `59.18%` | `66.2/75.0/89.7/64.8` |
| Combined Ensemble (LF + ESM + GVP) | 55 | `77.17%` *(OOF)* | `71.00%` | `72.42%` | `72.76%` | `71.64%` | `61.33%` | `68.1/70.0/90.3/62.6` |

## 2. Key Findings & Reproduction

- **Step F Parity:** `stage6b_refit` reproduces frozen `test_predictions.csv` to $\max |\Delta p| = 1.61 \times 10^{-6}$ (`67.55%` Full, `69.33%` Eligible).
- **Endpoint distinction:** OOF gains are not monotonic on the test set (30-model Full `72.20%`, 55-model `71.00%`); the `61.33%` Clean result is a different population. Test scores never select models.
- **Ensemble choice:** The [release audit](release_audit/README.md) reproduces these OOF scores. Under [Plan §7](../../../Plan.md#7-baseline-first-model-comparison-policy) the 55-model has no CI-supported gain over the 20- to 30-model late-fusion ensembles, which average fewer checkpoints per OOF site: it is the best-CV reference, and no default is chosen. The 20-, 25- and 30-model recipes follow the full-test order (possibly test-informed); OOF-only nested selection reaches a similar level.
- **Disclosures:** [`DATASETS.md`](../../DATASETS.md#test-use-ledger) (test results are secondary). The historical [`eval_phase2_unified.py`](eval_phase2_unified.py) reads test artifacts; use the release audit for OOF.
