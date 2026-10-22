# Handoff: resumed GVP / GVP+ESMC improvement reviews (2026-09-23)

**Status: work in progress.** This is a review-provenance note, not
a scientific evidence document and not authority for any claim. Nothing in the repo
source, run outputs, Colab sessions or datasets was modified by this work. No
training run was launched. `Plan.md`, `EXPERIMENT_STATUS.md` and
`docs/notebook_outputs/` retain their existing authority and content.

Redaction note (2026-10-22): sections 1–5, 7 and 8 are a condensed provenance
summary. Session identifiers, review-run identifiers, local paths and recovery
mechanics were removed; the complete original record (session logs, review
scripts, review journals and cached review results) is kept in the maintainer's
records (not published). The findings and numbers in section 6 are unchanged.

## 1. Review requests (the "unfinished 2 works")

Two review requests from 2026-09-23 were interrupted before an answer was delivered.

### Work 1 (17:31 UTC)

The maintainer supplied the completed exact-PinMyMetal 5-fold benchmark results
(`EXPERIMENT_STATUS.md` "Current objective" block) and asked:

> Please read the project's gvp and fusion model and try to figure out if there are
> clear/substantial reasons/causes that might have interfere/harm the performance of
> the gvp+esmc?

**This request never ran.** Every step of the working session failed before any
action executed, so it produced no output. Nothing could be recovered from it except
the question itself.

### Work 2 (18:12–18:24 UTC)

Two requests:

1. (18:12) Read the transcript of an earlier PyCharm IDE session, which proposed
   13 issues (claims C1–C13) whose fixes "might have high potential improving the model".
   The maintainer asked for an opinion on each, a ranking by estimated likelihood of
   substantial improvement, and any additional suggestions of the reviewer's own.
2. (18:22) *"Please, do independent new from scratch review for the gvp and gvp+esmc
   (with focus on the fusion) for find what are the issues that fix/change them in
   the code might bring substantial improvements to the performance."*

The working session decoded the earlier IDE session transcript, scouted the code and
run artifacts, and launched two background reviews. **Both were cut off mid-run when
the session ended.** The maintainer received only two preliminary findings and no
final answer.

## 2. Source material

| Item | Location |
| --- | --- |
| Earlier IDE session transcript, decoded (13 claims C1–C13), with its raw IDE logs | Maintainer's records (local, not published) |
| Benchmark run outputs | `runs/benchmark_exact_pinmymetal_5fold/{benchmark_only_esm,benchmark_enhanced_only_gvp,benchmark_enhanced_gvp_esmc}_fold{0..4}/` |

## 3. State of the two interrupted reviews

| Review | Completed before the cut |
| --- | --- |
| `deepmzyme-improvement-audit` (claims C1–C13) | 13/13 `verify:C*`, 1/13 `challenge:C*` (C2). Discovery lenses started, none returned. |
| `gvp-fusion-fresh-review` (independent review) | 4/7 finders: `gvp-core-math`, `readout-pooling`, `fusion-design`, `optimizer-training-loop`. |

The interrupted review runs could not be resumed directly in a new working session,
so the completed results were recovered as described in section 4.

## 4. Recovery of the completed review results

1. Read both session logs and both review journals; confirmed Work 1 produced
   nothing and Work 2 stopped mid-run.
2. **Extracted every completed review result** from the two review journals:
   14 results from the audit review and 4 from the fresh review, saved as
   `cache_audit.json` and `cache_fresh.json`.
3. **Built one combined continuation script** that reproduces both original review
   scripts verbatim, except:
   - each review check returns its cached result when one exists, so only the
     missing checks run;
   - both reviews run concurrently under one shared concurrency cap;
   - a final **synthesis** stage merges both reviews into one ranked list and
     explicitly answers Work 1's question;
   - the per-check resource rules were corrected (see section 7, item 5).
4. Launched it as a new review run and armed a memory monitor.
5. Independently spot-checked the most-cited premise (see section 6, C1).

The continuation script, the cached results and the new run's journal are kept in
the maintainer's records (not published).

## 5. What still has to happen

1. **Finish the combined review run.** Remaining: 12 adversarial challenges (C1, C3–C13),
   5 discovery lenses + merge + per-proposal adversarial checks + critic and gap
   follow-ups (audit review); 3 finders + loop-until-dry sweeps + merge + per-issue
   premise/impact checks + critic (fresh review); then synthesis.
2. **Deliver to the maintainer**, in one report:
   - a direct answer to Work 1 (are there clear/substantial code-level causes that
     harmed GVP+ESMC?);
   - the ranked list with reconciled probabilities, cost, and the cheapest decisive
     experiment per item;
   - the per-claim table for C1–C13 with the reviewer's opinion vs the earlier IDE
     session's ranking, which is what request 2a asked for;
   - which items the independent review found that the earlier IDE session missed,
     and where the two disagree.
3. **Decide nothing scientific from it without an experiment.** Every item is a
   hypothesis with a proposed validation-only paired test; none is evidence.

The 13 claims and the recovered reviews were later consolidated in the
[evidence-ranked improvement plan](../plans/gvp_and_esmc_evidence_ranked_improvement_plan.md).

## 6. Substantive findings already established (cached, re-checkable)

These come from the completed review checks and from direct checks. They are code/artifact
observations and validation-based reasoning — not experimental results.

**Headline: the earlier IDE session's confidence estimates are badly calibrated.** All 13 claims were graded
`partly_correct` or `mostly_correct`; none was fully correct. Every reconciled
probability of a ≥1.5 pp gain in mean out-of-fold validation balanced accuracy came
out at **0.04–0.15**, against the earlier IDE session's stated 45–95%.

Verified facts worth keeping:

- **C1 param-group split is real** (`src/training/run.py:1486-1507`). The fast
  `gvp_learning_rate` (3e-4) is given only to parameters whose names start with
  `layers.`, `node_scalar_encoder.` or `edge_scalar_encoder.` (prefix tuple at
  `src/training/run.py:1492`). In the joint model the GVP input projection
  (`init_vec_proj`, `src/model.py:746`) and the GVP readout (`gvp_attn_pool`
  `src/model.py:745`, `gvp_fusion_proj` `src/model.py:763`) therefore train at 3e-5,
  10× slower, while in Only-GVP they train at 3e-4. This was found independently by
  three of the four fresh-review finders, making it the most corroborated single
  issue. Reconciled probability is still modest (≈0.08–0.20): the head and gate do
  train, and a fully trained GVP branch adds only ~1 pp on validation.
- **The "GVP overfitting" story does not hold** (C2). Validation loss bottoms at
  epoch ~21–30 and rises in *all three* models, including ESM-only, which has no GVP
  trunk. Validation balanced accuracy keeps rising to epochs 40–50. The rising loss is
  overconfidence, not accuracy collapse.
- **The "50/50 blend beats joint training" claim is a test-set artifact** (C3). It
  reproduces on test (81.17 vs 79.92 BA4) but the bootstrap 95% CI of the difference
  is [−1.9, +4.5] pp, and on out-of-fold **validation** the blend is *worse* than the
  joint model (BA4 77.70 vs 79.84; BA5 71.68 vs 74.51). It must not be adopted.
- **Reporting caveat:** the "Table 1" CV numbers in `EXPERIMENT_STATUS.md` and the
  2026-09-23 summary (80.29 / 74.42 / 80.22) appear to be the **max over epochs** of
  validation collapsed-4 BA, not the value at the selected checkpoint. At the selected
  checkpoints the out-of-fold means are only_esm BA4 78.98, only_gvp 73.87,
  joint 79.84. **This needs reconciling before publication.**
- **Split caveat:** `--train-val-split-by pocket_id` puts pockets of the same PDB on
  both sides. Fold 0 has 58 validation pockets whose PDB ID also appears in training,
  54 of them the same chain (`split_diagnostics.json`).
- **Minor latent defects noted in passing:** the PROPKA `dpka_titr` node feature is
  identically zero in every fold (dead input); the focal-loss path computes `p_t` from
  class-weighted CE then takes an unweighted mean (unused in this benchmark); the
  joint model's logged `lr` is the GVP-trunk LR, not the head/ESM LR
  (`src/training/run.py:1584`).

The full structured verdicts (factual findings, mechanism, computed numbers, proposed
experiment per claim) are in the cached result files `resume/cache_audit.json` and
`resume/cache_fresh.json`, kept in the maintainer's records (not published).

## 7. Constraints for any continuation

1. **Held-out test data must not drive design choices** (`Plan.md` and the
   maintainer's local development guidelines, not published).
   Test numbers may be reproduced only to fact-check a claim, labelled as such.
   All improvement judgements rest on out-of-fold validation, code reasoning and
   prior project evidence.
2. **Read-only on the repo** for the review checks; scratch output goes to a local
   scratch directory outside the repo.
3. **No training runs, no dataset builds, no forward passes over data.** The local
   machine has no GPU. Every proposal is delivered as an experiment to run later.
4. Use the project's prescribed interpreter (`<python>`), never bare `python`.
5. **Memory is the real constraint.** ~15.7 GB total, with PyCharm, Chrome and one
   more desktop application resident; free memory was ~1.6 GB when this work started.
   Importing torch costs ~400 MB per process. The earlier run's review instructions
   advertised "~1–2 GB free" and probably contributed to a desktop crash; the resumed
   script tells each review check to keep **one** torch process alive at a time and
   exit promptly. Keep the concurrency cap and that rule.

## 8. Status label

Per §1c of the local development guidelines: this work is **implemented and in progress** as an analysis; its
outputs will be **hypotheses with proposed experiments**, not smoke-tested, not
experimentally evaluated, and not promoted. No model, parameter or architecture
conclusion may cite this document as evidence.
