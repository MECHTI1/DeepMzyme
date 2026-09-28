# CARE EC1 standalone reference

Status: closed (2026-09-28 documentation reconciliation)

EC depth-1 single-label classification on CARE clusterRes30 v12; no metal target was trained. Twelve 30-epoch runs, three families × two LRs × seeds 42/43, one fixed protein-grouped split with protein/structure-level EC supervision. Historical Colab G4 execution; Grade 3 fixed-split evidence, no promotion, auxiliary training or held-out evaluation. Only-ESM at 0.0001 had the largest two-seed mean, 0.969643 (sample SD 0.017678); this is not grouped-fold confirmation. The earlier statement that association analysis was still pending is historical.

Sources and evidence:

- [Completed reference](../../../../docs/notebook_outputs/summaries/summary_ec1_standalone_v12_20260914.md)
- [Recipe compatibility](../../../../docs/EC_TRAINING_PIPELINE_PLAYBOOK.md)

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
