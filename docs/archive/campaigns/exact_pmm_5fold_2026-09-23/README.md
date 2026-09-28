# Exact PMM pocket benchmark history

Status: closed (2026-09-28 documentation reconciliation)

Historical five-class training with common-four reporting on the catalytic-pocket subset of exact PMM. Pocket-stratified fivefold evaluation can share PDB IDs across folds; Grade 6 exploratory evidence, not grouped-fold confirmation. Selected-checkpoint OOF BA: late fusion 79.84%, ESM-C 78.98%, GVP 73.87%; preserve epoch maxima separately. Held-out test opened 2026-09-22/23 and influenced exploratory ranking; no model promoted. Historical Colab execution is documented in the existing guide. Not like-for-like with PinMyMetal; the qualification remains in that guide.

Sources and evidence:

- [Qualified guide](../../../../docs/EXACT_PINMYMETAL_5FOLD_CV_REPRODUCIBILITY.md)
- [Immutable summary](../../../../docs/notebook_outputs/summaries/summary_pinmymetal_5fold_three_models_20260923.md)
- [Test-use ledger](../../../../docs/DATASETS.md#test-use-ledger)

Files here: this README and [verbatim historical STATUS entries](log.md).
No existing report, summary or raw artifact moved. This is the history-preservation
step of [Job B](../../../../docs/archive/consolidation_2026-09/PLAN_v2.md#11-jobs-and-sequencing-amended).

How to continue: consult [current status](../../../../EXPERIMENT_STATUS.md) and the
linked owner before proposing separately authorized work. Historical next actions
do not authorize a new allocation, training, final refit or held-out evaluation.

Provenance: `git show 3c0f80c:EXPERIMENT_STATUS.md`,
`git show 9a25f36^:EXPERIMENT_STATUS.md`, `git show 9a069d2^:EXPERIMENT_STATUS.md`
and `git show b45b893:EXPERIMENT_STATUS.md`; exact source ranges are recorded
in each [log](log.md). Linked evidence owners qualify the historical text.
Target check: `scripts/run_exact_pinmymetal_5fold_cv.py:208` specifies `five_class`; this is historical runner identity, not the PMM core-v2 target list.
