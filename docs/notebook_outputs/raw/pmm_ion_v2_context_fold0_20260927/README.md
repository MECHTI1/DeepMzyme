# PMM ion campaign: fold-0 continuation evidence

This batch continues the same frozen `pmm_ion_metal_v2_context` campaign and
[final plan](../../../plans/metal_level_metal_task_compared_PMM_final_plan.md).
The completed [ordinary/aware ESMC pair](../pmm_ion_v2_context_20260926/README.md),
features, smokes and PMM folds are reused. This batch records the continuation
toward all nine configurations on fold 0; it does not complete the 45-fit grid.

- [Execution receipt](runtime/execution.json): user instruction, exact queue,
  selected immutable VM, allocation identity/deadline, unchanged controller
  caps and operational fixes. Historical receipt; current state is owned by
  [EXPERIMENT_STATUS.md](../../../../EXPERIMENT_STATUS.md).
- [Exploratory screen report](runtime/screen_report.md),
  [full metrics/identities](runtime/screen_report.json),
  [metrics CSV](runtime/screen_metrics.csv).
- [Report generator](runtime/report_screen.py): uses existing source validators
  and metric helpers; reads development artifacts only. Run against the canonical
  campaign with `--campaign-dir`, `--train-dir`, `--source-root`, `--out-dir`.
- [Operational launcher](runtime/run_one_fit.sh): binds a current session to the
  selected VM, selects one unit and uses an independent per-unit status file.
  A failed SSH exit requires process reconciliation and explicit terminal
  artifact persistence. It does not authorize allocation or training by itself.

Canonical checkpoint/data/backup location:
`/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v2_context`.
Scientific source hash remains
`adc95c42261448dc9d35572a138a3b8349de79124d9708618f511c27a848dd23`.
The reporting helper lives outside the frozen scientific source tree.

Results remain single-fold, single-seed exploratory validation, with native
checkpoint selection. The common-four view is used for cross-target comparison;
six-class native metrics and Fe/Co/Ni recalls are retained. No model promotion,
final refit, test access, or claim of reproducing PMM's published protocol.
