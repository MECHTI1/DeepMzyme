# Execution v5 — Runpod G2 through Fold 0

Status: draft, 2026-11-04; CPU implementation only. All launch attempts are paused.
Amendment 2 and this execution package must be published before a separate explicit launch authorization.
There is no current fold-0 cost approval. The previous G2-only limits and launch requests do not authorize v5.

## Execution contract

One Runpod pod and one allocation clock cover G2 and the subsequent approved fold-0 work. Session limits
come from the approved configuration. The same owner maintains the heartbeat, stop deadline and independent
preservation throughout both phases. Two further failed launch windows exhaust the permitted retry count;
changing a pass directory or configuration must not reset it. GCP launches and fallback routes are excluded.

CLEAN fits, ESM-1b builds and ESMC builds use the pod GPU with TF32 explicitly disabled inside each child.
CUDA-trained model predictions are replayed on CPU with absolute tolerance `1e-5`; same-device CPU replay
retains `1e-6`. The fold-0 view keeps its lexical campaign path while its storage can use the verified local
POSIX placement. Runtime and hot-work capacity must be checked before admission.

G2 timing fits write inner-fold-0 predictions for proteins that appear in other official folds' test sets.
Every scientific design change therefore must be recorded before the first G2 run. The pending amendment
owns scientific choices, unit lists and selection rules; execution code must not infer them from timing results.

## Publication dependencies

The amendment-2 interface contract is available in the separate science worktree. Its chain verifier and
scientific rules remain owned there. Manifest v5 is constructed from a published chain receipt; a draft link
or a chain below sequence 2 cannot authorize real work. The published amendment, its five dry-run-v2 receipts
and the execution manifest are still required before launch. A complete cost forecast also remains pending.

The staged POSIX correction is included here, with its original bytes preserved as
[component manifest](posix_component_manifest.json) and [component verification](posix_component_verification.json).
These record the earlier component checks, using their original filenames inside the records; they do not
authenticate the later v5 changes. The [configuration draft](workstation_config.example.json) is not launch
authority: its session ceilings are unset and the final manifest does not yet exist. The
[published execution-v4 manifest](../execution_v4/manifest.json) and its companion files remain unchanged.

## CPU implementation state

[Verification and changed-file inventory](cpu_verification.json) records the component checks and source pins.
The transport, configurable session limits, CPU replay, subprocess read guard, child precision policy,
local CLEAN working storage and preservation checks are implemented. The explicit v5 dispatcher binds the
two-failed-window ledger to one retry authorization across configuration changes. It requires a separate launch
authorization and an approved numerical envelope before provider access.
No live GPU qualification or scientific experiment is established by these checks.

The Runpod transport exposes the same owned session to an explicit sequence of queues and preserves G2
and Phase-4 outputs independently. Preparation callbacks create local metadata only; remote work starts after
its output root is registered for preservation. The dispatcher materializes G3 from a preapproved numerical
envelope and measured Freeze 1c v2, without expanding the approved session. The legacy entry point refuses
the v5 execution scope. The explicit entry point is `workstation --run-v5`, with a configuration, launch
authorization and window identifier; the example configuration deliberately remains incomplete.

The workstation calls the science-owned receipt writer once per session:

```python
amendment2.write_chain_receipt(tip, freeze1_sha256, out,
    draft=False, expected_chain=expected_chain, repo=repo)
```

Pod and unit checks call `amendment_chain.check_chain_receipt(receipt_path, repo=repo,
expected_chain=expected_chain)`. Full chain verification stays on the workstation's full clone.
The returned `fit_chain` and `scientific_identity_sha256` bind G2, Freeze 1c v2, G3 and unit records;
execution-manifest hashes remain operational provenance. Each real view binding also checks its fold's
amendment-2 dry-run receipt and the unchanged Freeze-1 in-view receipt. The pinned replay function is
`phase4.replay_cpu_logits`: CPU, absolute tolerance `1e-5`, near-tie threshold `2e-5`, mandatory argmax agreement.
The existing `freeze.verify_s1_amendment` remains unchanged for historical authentication.

The implemented order is full-union ESMC fp32 cache, fold-0 F6 attachment, CPU sample parity, G2 v2 and
Freeze 1c v2. Plan-v2 queues come from the scientific plan: fold 0 has 112 S1 and 5 S4 units (cap 117),
and each later fold has 96 S1 and 5 S4 units. Refit recipes come from `selector.verified_refit_recipe`.
CLEAN in-sample and held-out sanity records precede selection. A verified futility continue is required before
fold-0 S4, every fold 1–4 unit and every final-arm CLEAN fit; any stop pointer wins. The five folds retain
separate view bindings. No final test export is added here.

F6's `whole_protein_pretrain/` is created empty and writable outside the frozen view manifest. Attachment
and scientific-output writes first create durable pending markers. Their snapshots and inventories are fsynced
only after the owned child is reaped; unresolved markers block automatic acknowledgment and deletion. Both
independent copies are verified before closeout. Scientific snapshots include every applicable futility pointer,
including partial records, so an interrupted decision cannot disappear during cleanup.

The science publication still must register `phase4.replay_cpu_logits` in `PINNED_EXECUTION_FUNCTIONS` and
expose public replacements for `views._set_permissions`, `views._remove_tree`, and
`ec_prediction_export._graph_options`, `_prediction_loader`, `_predict`. The existing private reads remain a
C7 publication blocker until that handoff; they are not an approved exception. The F6 input role
`inputs.f6_corpus.tables.<fold>` must bind both `corpus.tsv` and `corpus_table_receipt.json` for authenticated
final-arm CLEAN input preparation. None of these draft changes grants launch or spending authority.

## Provisional cost scenarios

No GPU completion-time measurement is available: G2 remains 0/12. The last verified RTX 4090 rate was
$0.89/hour plus $0.011506849315/hour running storage; this is historical pricing, not a refreshed quote.
Using the existing 15% cost margin and $0.30 retained-storage reserve:

`gross(H) = reconciled prior charges + 1.15 × (0.901506849315 × H + 0.30 + egress reserve)`

| Total billed session | Gross, excluding prior charges and assuming zero egress |
|---|---:|
| 8 hours | $8.64 |
| 12 hours | $12.79 |
| 24 hours | $25.23 |
| 36 hours | $37.67 |
| 48 hours | $50.11 |

These are cost scenarios with the historical storage configuration, not completion forecasts or approved ceilings.
Final hot/canonical storage capacity must also be priced. The existing rolling $10 policy
has no approved override, and the 8-hour scenario already crosses its $8 warning threshold before prior costs.
The prepared CLEAN closure contains 8,115 base and 20,120 masked embedding sequences and eight 2,500-epoch
fits. Historical CPU timing cannot establish their CUDA throughput. Full-training ESMC preparation and F6
pretraining add work whose CUDA throughput is unmeasured. Forecasts must count CPU replay/export on the billed pod,
setup, transfer and closeout, without assuming a parallel speedup.

Evidence: [evc-034](../log.md#evc-034), the preserved private `fold0_clean_inputs/clean_inputs.json`,
`phase3/prebilling_20261102/phase4_requests/README.md` and
`phase3/runpod_bootstrap_posix_prepared_20261103/preflight.json` beneath the campaign output root.
