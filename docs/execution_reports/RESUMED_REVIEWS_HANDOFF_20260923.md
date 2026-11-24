# GVP and GVP+ESMC benchmark review, 2026-09-23

This partial review examined 13 proposed explanations for the fusion model's performance on the exact PinMyMetal five-fold benchmark. It used existing code and fold outputs from `runs/benchmark_exact_pinmymetal_5fold/`. No models were trained for the review.

The initial attempt produced no results. At the September 23 interruption, all 13 proposals had a first check, and only C2 had a second check. Four of seven independent checks had finished, covering GVP layers, pooling, fusion and optimization. The saved records `resume/cache_audit.json` and `resume/cache_fresh.json` contain those 14 and four completed checks, respectively. Use the [follow-up plan](../plans/gvp_and_esmc_evidence_ranked_improvement_plan.md) to see which proposals received later experimental checks.

<a id="6-substantive-findings-already-established-cached-re-checkable"></a>

## Findings

- **Learning rates (C1).** In the joint model, the 3e-4 learning rate applied to parameters starting with `layers.`, `node_scalar_encoder.` or `edge_scalar_encoder.`. The input projection (`init_vec_proj`) and readout (`gvp_attn_pool`, `gvp_fusion_proj`) trained at 3e-5. The same components trained at 3e-4 in Only-GVP. See `src/training/run.py:1486–1507` and `src/model.py:745–763` in the benchmark revision. Three of four completed independent checks identified this discrepancy. The estimated chance of a gain from changing it was 0.08–0.20. The GVP branch added about one percentage point on validation.
- **Validation loss (C2).** Loss reached a minimum around epochs 21–30 and then rose in all three models, including ESM-only. Balanced accuracy continued to rise through epochs 40–50. These curves suggest increasing overconfidence across models.
- **Equal-weight blend (C3).** On out-of-fold validation, blending the two standalone models scored below joint training on both four-class balanced accuracy (77.70% vs 79.84%) and five-class balanced accuracy (71.68% vs 74.51%). A retrospective test check gave 81.17% vs 79.92% on four classes, with a 95% bootstrap interval of −1.9 to +4.5 percentage points for the difference. Validation did not support adopting the blend.
- **Checkpoint reporting.** The reported four-class validation scores appeared to be maxima across epochs. For ESM-only, GVP-only and late fusion, the reported maxima were 80.29%, 74.42% and 80.22%. The selected-checkpoint out-of-fold means were 78.98%, 73.87% and 79.84%, respectively. The review flagged the distinction for reconciliation before publication.
- **Fold overlap.** With `--train-val-split-by pocket_id`, 58 validation pockets in fold 0 shared a PDB ID with training. Of those, 54 shared a chain. Source: `split_diagnostics.json`.
- **Implementation details.** The PROPKA `dpka_titr` node feature was zero in every fold. The unused focal-loss path derived `p_t` from class-weighted cross-entropy before taking an unweighted mean. The joint model logged the GVP trunk learning rate in `lr` (`src/training/run.py:1584`).

All 13 proposals were graded partly or mostly correct. The review estimated a 4–15% chance of a gain of at least 1.5 percentage points in mean out-of-fold balanced accuracy, down from preliminary estimates of 45–95%. These were judgments. Performance gains require paired validation experiments. The [follow-up plan](../plans/gvp_and_esmc_evidence_ranked_improvement_plan.md) records the rankings and proposed comparisons.
