# Zenodo exact PMM ion benchmark

Status: paused (2026-09-28 documentation reconciliation)

Historical relaunch outcome unknown; this label grants no resume authority. Goal: source-row ion-level PMM comparison; historical runner uses five-class training and common-four reporting, parent-pocket-stratified folds, seed 42, on the Zenodo PMM reconstruction. Colab L4 execution lost its first session; later `pmm-zenodo-v2` outcome is unrecorded. Held-out access is unknown, possibly evaluated; no completed result or promotion is claimed.

Sources and evidence:

- [Guide](../../../docs/ZENODO_PINMYMETAL_EXACT_ION_LEVEL_REPRODUCIBILITY.md)
- [Test-use ledger](../../../docs/DATASETS.md#test-use-ledger)
- [Execution handoff](../../../docs/agents_report/HANDOFF_ZENODO_PMM_EXACT_BENCHMARK.md)

Files here: this README and [verbatim historical STATUS entries](log.md).
No existing report, summary or raw artifact moved. This is the history-preservation
step of [Job B](../../../docs/archive/consolidation_2026-09/PLAN_v2.md#11-jobs-and-sequencing-amended).

How to continue: consult [current status](../../../EXPERIMENT_STATUS.md) and the
linked owner before proposing separately authorized work. Historical next actions
do not authorize a new allocation, training, final refit or held-out evaluation.

Provenance: `git show 3c0f80c:EXPERIMENT_STATUS.md`,
`git show 9a25f36^:EXPERIMENT_STATUS.md`, `git show 9a069d2^:EXPERIMENT_STATUS.md`
and `git show b45b893:EXPERIMENT_STATUS.md`; exact source ranges are recorded
in each [log](log.md). Linked evidence owners qualify the historical text.
Target check: `scripts/run_zenodo_pmm_exact_5fold_cv.py:225` specifies `five_class`; this is historical runner identity, not the PMM core-v2 target list.
