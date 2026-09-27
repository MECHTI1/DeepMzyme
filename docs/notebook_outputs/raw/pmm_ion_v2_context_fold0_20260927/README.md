# PMM ion campaign: fold-0 continuation evidence

This batch continues the same frozen `pmm_ion_metal_v2_context` campaign and
[final plan](../../../plans/metal_level_metal_task_compared_PMM_final_plan.md).
The completed [ordinary/aware ESMC pair](../pmm_ion_v2_context_20260926/README.md),
features, smokes and PMM folds are reused. This batch records the continuation
toward all nine configurations on fold 0. All nine trained for 50 epochs; eight
are certified and six-class GVP remains blocked by independent replay. The full
grid is nine of 45 trained and eight certified. No GPU allocation is active.

- [Execution receipt](runtime/execution.json): user instruction, exact queue,
  selected immutable VM, allocation identity/deadline, unchanged controller
  caps and operational fixes. Historical receipt; current state is owned by
  [EXPERIMENT_STATUS.md](../../../../EXPERIMENT_STATUS.md).
- [Screen-completion execution](runtime/completion_execution.json): the later
  user-authorized session for the two remaining six-class fold-0 configurations.
  It preserves the earlier allocation and closeout records separately. Six-class
  GVP trained but failed independent probability replay twice; late fusion
  passed and its 71-file backup verified. Current state is owned by
  `EXPERIMENT_STATUS.md`.
- [Screen-completion closeout](runtime/completion_verified_closeout.json):
  provider-confirmed TERMINATED state at 19:41:10 UTC, 5,333-second session,
  $1.3023 estimated running gross, verified late-fusion backup, and the preserved
  GVP replay blocker. [Status](runtime/completion_vm_status.log),
  [stop](runtime/completion_vm_stop.log), and
  [resource/cost report](runtime/completion_vm_report.log) retain the raw evidence.
  One 150-GB disk remains at approximately $15/month. No deletion, promotion,
  refit, test access, or activation of the separate full-grid budget occurred.
- [Verified closeout](runtime/verified_closeout.json): seven completed screen
  configurations, five new verified backups, provider-confirmed TERMINATED
  state, actual session accounting, and the pending budget/start decision.
  [Provider status](runtime/vm_status_closeout.log) and
  [controller report](runtime/vm_report_closeout.log) preserve the closeout evidence.
  At that historical closeout two fold-0 configurations remained; seven of 45
  grid fits were certified.
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
- [Budget forecast](runtime/continuation_budget_forecast.json): measured GVP and
  ESM times, explicit warm-cache/late-fusion proxies, preserved prior use, and
  the pending larger-envelope question. A proposal is not an approved budget.
  The closeout receipt contains the updated forecast after all five new fits.
- [GPU execution review](runtime/gpu_execution_review.md): measured preparation,
  training and persistence costs, cache reuse, and interpretation limits.
- `units/`: portable completed-fit metadata excerpts, selected receipts, learning curves
  and validation predictions. This directory corresponds to canonical `runs/`;
  binary checkpoints stay in the verified local/VM storage and their hashes are
  retained. `persistence_receipts/` preserves native manifests/acknowledgments
  against the canonical campaign layout, not this selective portable copy.
  Excerpts retain full configuration, normalization and provenance fields,
  identify omitted verbose dataset details, and bind the complete original
  metadata files by SHA-256. No canonical run files are replaced by excerpts.
- `unverified_units/`: completed training artifacts that failed independent
  replay certification. They are excluded from every verified score table and
  contrast. The GVP six-class entry includes both replay prediction sets,
  the exact comparison diagnostics, original checkpoint hash and configuration
  excerpt. Its first failed 70-file transfer was preserved separately before
  retry; the second 72-file transfer includes the runner's archived first replay.
  Historical manifests refer to those original transfer snapshots. See
  [TECH-023](../../../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-023--full-fit-gvp-independent-replay-exceeds-the-frozen-probability-tolerance).

Canonical checkpoint/data/backup location:
`/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v2_context`.
Scientific source hash remains
`adc95c42261448dc9d35572a138a3b8349de79124d9708618f511c27a848dd23`.
The reporting helper lives outside the frozen scientific source tree.

Results remain single-fold, single-seed exploratory validation, with native
checkpoint selection. The common-four view is used for cross-target comparison;
six-class native metrics and Fe/Co/Ni recalls are retained. No model promotion,
final refit, test access, or claim of reproducing PMM's published protocol.
