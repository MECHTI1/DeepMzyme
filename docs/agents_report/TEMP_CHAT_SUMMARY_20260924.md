# Temporary chat summary — DeepMzyme metal-ion examples and Only-ESM

**Date:** 2026-09-24  
**Status:** Temporary conversation handoff. This is not experiment evidence or a replacement for `Plan.md` or `EXPERIMENT_STATUS.md`.

## Resumed work and split decision

- The user asked to continue the older PyCharm ChatGPT work from `docs/agents_report/RESUMED_REVIEWS_HANDOFF_20260923.md` and finish examining the metal split.
- A pocket can contain more than one metal type. The user clarified that **the pocket split must remain**, while **each metal ion needs its own supervised example and target**. Sibling ions from one parent pocket stay in the same train/validation fold.
- Standalone metal training and the Colab notebook now expose `--metal-example-unit ion`; `pocket` remains the default. Each ion example is centered on that ion and selects residues within its own 10 Å neighborhood. The previous pocket-unit cohort and results are not matched ion-unit controls.
- Focused and regression checks passed (**154 tests**). No full ion-unit cohort/fold audit, GPU training, or held-out evaluation has run as of this summary.

## What the current Only-ESM model can distinguish

- The user asked whether Only-ESM would assign the same predicted metal to two differently labeled ions in one pocket. At the current zero distance-pooling cutoff, Only-ESM pools ESMC residue embeddings from each ion's selected neighborhood; its single-ion site statistics are the same for both examples.
- If the two examples contain the same residue embeddings, deterministic evaluation gives the same prediction. If their 10 Å residue sets differ, their predictions can differ, but the frequency and accuracy of that distinction have not been measured on ion-level data.
- In the primary four-class target, Fe, Co, and Ni intentionally share Class VIII. Identical predictions for two ions from *different target classes* cannot make both correct.

## Proposed binding-residue ESMC view

- The user proposed retaining the 10 Å ESMC pocket view and adding a second view pooled from **the binding residues of the target ion**. The user clarified that two ions may share some binding residues while their complete binding-residue sets differ. Those distinct per-ion sets can provide different model inputs even when the wider neighborhoods overlap heavily.
- The recommendation was to reuse the existing full-sequence residue-level ESMC embeddings, apply a separate first-shell mask for each ion, pool the binding-residue and 10 Å views, and combine them with a small learned fusion layer. Metal labels must not be used to choose the binding residues.
- This is a **proposed optional model variant, not an implemented or validated improvement**. Compare it with plain ion-level Only-ESM on the same eligible cohort, folds, and seeds, including mixed-metal pockets and per-class recall. Distinct inputs make different predictions possible but do not guarantee correct predictions.

## Related execution status

- The earlier pocket-unit L4 diagnostic stopped at **3/20 locally verified full runs**; its partial F1 fold-1 run was not recovered. It cannot supply matched ion-unit results.
- The original Colab L4 assignment cleared by 2026-09-24 04:29 UTC. A later session check found no active Colab sessions, so no manual disconnect is needed.

For current scientific policy and implementation status, consult `Plan.md`, `EXPERIMENT_STATUS.md`, and `docs/agents_report/GVP_FUSION_EXACT_POCKET_L4_V2_EXECUTION.md` rather than treating this temporary summary as authority.
