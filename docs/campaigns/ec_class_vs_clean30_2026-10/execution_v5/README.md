# Execution v5 — Runpod G2 through Fold 0

Status: published, 2026-11-12, with the [execution corrections](#execution-corrections-of-2026-11-12) of the
same day, the [host replacement](#host-replacement-within-a-launch-window) that followed the first launch window and the [supervised caps](#supervised-caps-on-hosts-with-read-only-cgroup-controls) that followed the second; CPU implementation only. The [fold-0 ceiling](#fold-0-ceiling-a2-d8) is approved. Neither publication
nor the ceiling grants launch authorization: each launch needs its own explicit authorization. The previous
G2-only limits and launch requests do not authorize v5.

## Execution contract

The initial Runpod pod covers G2 and the subsequent approved fold-0 work. An interrupted fold can continue
in a separately authorized session under A2-D6, with the same GPU type and pinned environment. Session limits
come from the approved configuration. The same owner maintains the heartbeat, stop deadline and independent
preservation throughout both phases. Two further failed launch windows exhaust the permitted retry count;
changing a pass directory or configuration must not reset it. GCP launches and fallback routes are excluded.

CLEAN fits, ESM-1b builds and ESMC builds use the pod GPU with TF32 explicitly disabled inside each child.
CUDA-trained model predictions are replayed on CPU with absolute tolerance `1e-5`; same-device CPU replay
retains `1e-6`. The fold-0 view keeps its lexical campaign path while its storage can use the verified local
POSIX placement. Runtime and hot-work capacity must be checked before admission.

G2 timing fits write inner-fold-0 predictions for proteins that appear in other official folds' test sets.
Every scientific design change therefore must be recorded before the first G2 run. The published amendment
owns scientific choices, unit lists and selection rules; execution code must not infer them from timing results.

## Publication dependencies

The published [amendment-2 interface contract](../amendment2/interface_contract.md) owns the chain verifier
and scientific rules. Manifest v5 is constructed from a published chain receipt; a draft link
or a chain below sequence 2 cannot authorize real work. Link 2 is published. Its five private dry-run-v2 receipts stay on the workstation; the pod uses the
chain receipt and the bound view-manifest digests. After the [execution corrections](#execution-corrections-of-2026-11-12),
the published chain passed full verification at `0d4cc56`, including C7. The re-pinned [v5 manifest](manifest.json)
pins all 11 closed execution files, the runtime locks and the existing timing-input closure. After the
[probe time limit](#probe-time-limit-before-admission) it was rebuilt from a fresh published chain receipt at
`31c10d5`; only the `workstation.py` pin changed. Its draft verification passed on that clean published source tree,
and published-mode verification passed at `da30cd3` with a fresh chain receipt. It supersedes `205d1303…` (verified at
`cc7daed`), `e023e955…` (verified at `1f4404a`), `81e18aec…` (verified at `f57effa`), `72a5dadf…` (verified at
`bd016f2`) and the first v5 manifest (`bb8f50fd…`, verified at `2bff344`).
[Verification](cpu_verification.json) records the private receipt, build and publication evidence hashes.
No GPU completion time is measured yet; the approved ceiling is recorded [below](#fold-0-ceiling-a2-d8).

The POSIX correction is included here, with its original bytes preserved as
[component manifest](posix_component_manifest.json) and [component verification](posix_component_verification.json).
These record the earlier component checks, using their original filenames inside the records; they do not
authenticate the later v5 changes. The [configuration draft](workstation_config.example.json) is not launch
authority: its session ceilings are unset. The
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
execution-manifest hashes remain operational provenance. Each real view binding compares its manifest SHA-256 with the authenticated fold evidence and checks
the unchanged Freeze-1 in-view receipt; it does not open a private dry-run-v2 receipt on the pod. The pinned replay function is
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

C7 passes on the execution source tree: scientific interfaces are public, and the pinned CPU replay function
is registered by amendment 2. The F6 corpus-table roles bind each table and its receipt. Final CLEAN input
preparation checks the complete official training file against that receipt's SHA-256 before creating the
G3-bound closure; accepting a synthetic file's basename in the comparator does not authorize real-fit inputs.

Final CLEAN preparation dispatches one ESM-1b base farm and one masked farm for each frozen seed, and equivalent
C-ESMC farms through the fp32 sequence producer followed by `esmc-farm`. The frozen per-arm seed lists determine
the queue. `farms_by_unit` must name the generated base and matching seed's masked farm; shared arm mappings
are refused by the v5 launch gate. Final preparation requires explicit `embedding_forecasts` and
`embedding_timeouts`, keyed by producer unit ID, plus `hf_home` and `esmc_weights` in the worker template.

Sanity dispositions for individual fits use `clean_sanity_dispositions_by_unit`. Scientific input dispositions
are filtered by the SHA-256 of the consumed record: verify-all receives its whole record set, while S1 and FUT
receive only the held-out record's dispositions. S2 and S4 receive no CLEAN sanity flags.

## Resume and later-fold dispatch

A reviewed configuration may bind a `resume` object containing `config_path`, `config_sha256`, `state_path`,
`state_sha256`, `preservation_path` and `preservation_sha256` for the prior session. Admission verifies terminal
provider evidence, the preserved inventories on both devices, the scientific identity, GPU type and runtime
pins. New driver, controller and independent-copy destinations preserve the previous session's records unchanged.
The same shared failed-launch ledger remains in force; a preserved session that admitted scientific work is not
a failed launch window. Each new allocation still needs its own explicit authorization and numerical envelope.

Restoration uses the original remote paths, including F6 attachment snapshots and futility pointers. Finished
results are verified and reused. A unit with no preserved result gets at most one retry of the identical planned
unit; another failure, conflicting results or partial scientific decision outputs stop for diagnosis. G2 receipts
and preparation are reused as well. No lost attempt contributes a checkpoint or prediction to selection.

Later folds use `execution_scope: fold_execution_v5` and `execution_fold: 1` through `4`, with a matching reviewed
Phase-4 request and `fold_cost_ceiling_usd` in the launch authorization. They require preserved G2 and fold-0
records, the corresponding frozen view archive, and a binding futility continue. Their F6 attachment uses the
preserved union cache and the fold's bound `corpus_table`, `holdout_receipt`, `identity_work` and
`official_split_dir`; `attachment_forecast_seconds` and `attachment_timeout_seconds` are explicit reviewed bounds.
The route then runs the complete canonical S1, S2 and S4 sequence and optional authorized final CLEAN queue.

Pending launch requirements: the reviewed launch inputs, a fresh quote and a separate explicit launch
authorization. Each session requires a fresh workstation chain receipt on its clean published tree. The published
manifest and receipt checks establish provenance; they do not satisfy the numerical or launch-authorization gates.
None of the CPU checks grants launch or spending authority.

## Execution corrections of 2026-11-12

Offline reproductions with the real functions showed seven execution defects before any launch. All are class (a)
under A2-GOV-7: no scientific file, rule, unit, chain link or pinned function changed ([evc-039](../log.md#evc-039)).

- **Runtime gate.** The v5 unit check compared the stop-request deadline with the session duration, so it refused
  every G2 unit of a real controller session. It now requires hard stop − start = session duration, with the
  deadline between start and hard stop.
- **Selector input.** Phase-4 `components` is the frozen overlay's identity copy,
  `<remote campaign>/overlay/<overlay id>/identity/fold<K>`, which the general restricted-name guard refused. It is
  now the one accepted restricted-name input. Exactly one reviewed stage tar must carry this fold's copy and the
  overlay manifest to `<remote campaign>/overlay/<overlay id>`; the launch check refuses anything else.
- **Closeout inventory.** The pod writes the inventory as a hashed control file, which is downloaded and verified
  before use. It no longer passes through SSH output, which is bounded at 4 MiB.
- **Refusal before a launch intent.** A refusal inside the controller start before its launch intent is written
  (stale quote, another active GPU, pod-name collision, exhausted stock search) now closes the window as failed.
  Before, the window stayed open and blocked every later launch. Each such refusal uses one of the two failed windows.
- **Same-window recovery.** A re-run of the exact open window may hold an aged quote only to reach the allocation it
  already started; a window that never attempted a start, and every create, still needs a fresh quote. A closed
  driver record whose ledger row is still open is recorded without provider access.
- **Resume.** A resumed fold's G3 carries the allocations of earlier sessions admitted to Phase 4. Inner CLEAN fits
  read the farm of the attempt that holds the preserved embedding result. Preserved files are restored in a few
  verified archives, also through the POSIX storage aliases, and a remote file is never replaced.

Two operating limits remain. After a workstation or coordinator crash, the same `--run-v5` command must be re-run
within the 900-second heartbeat timeout, on the workstation whose watchdog is still running. Otherwise the watchdog
stops the pod, closeout cannot preserve its outputs, no A2-D6 resume is possible, and the stopped pod volume bills
until it is cleaned up. A crash inside the controller start before its session record exists leaves the ledger
window open; the maintainer closes it by hand after confirming with the provider that no pod exists.

## Host replacement within a launch window

The first launch window (`ec30v5-g2f0-w1`, 2026-11-12) ended at its first remote step: the allocated host exposed
read-only cgroup v1 controls, so the container caps helper refused kernel delegation before any staging. The pod was
deleted with provider absence verified after about one minute, and the window closed as failed. Earlier qualified
hosts of the same pool delegated cgroup v2, and the provider offers no host selection, so the refusal belongs to the
host, not to the inputs ([evc-040](../log.md#evc-040)). Two class (a) execution changes follow (A2-GOV-7):

- **Same-window host replacement.** A container caps refusal of `prepare` or `probe` during bootstrap, before any
  staging or workload admission, no longer ends the window. The coordinator allocates a replacement host under its
  own controller session (`host_try_<n>` receipts, its own watchdog, pod name `<pod name>-h<n>`, full session
  time-box) while the rejected pod stays allocated, then deletes the rejected pod and verifies its absence. Nothing
  ran on it, so nothing is preserved. At most three hosts are tried per window; the third refusal closes the
  window as failed. Every other refusal, and any refusal after staging begins, keeps the previous behavior.
- **Input proof across passes.** The prepared input proof no longer binds the pass-local paths (`state_root`,
  `durable_root`, `mirror_root`, `runpod.state_root`) or the pod name, so a new pass for a retry or a resume can
  reuse it at the same published source; the local storage check still runs on the new roots.

A coordinator crash during a replacement leaves both sessions under their own watchdogs; the ledger window then
needs the manual closure described above.

## Supervised caps on hosts with read-only cgroup controls

The second launch window (`ec30v5-g2f0-w2`) met the same read-only cgroup v1 controls on its first host, and its
replacement create was refused for capacity while the rejected pod was held. A third host of another GPU type and an
earlier pod showed the same read-only controls, so this is the provider's normal container setup, not a rare host
([evc-041](../log.md#evc-041)). The maintainer therefore approved a third backend in `container_caps.py`,
`cgroup_v1_readonly`, used only where the container-scoped v1 controls exist but cannot be written:

- **CPU.** Each capped command runs on whole cores set by kernel scheduler affinity (2 for the fit profile, 4 for
  the ESM-1b, CLEAN and science profiles). Concurrent commands claim distinct cores; a command started inside
  another capped command stays on its parent's cores. The supervisor refuses a run whose process widens its affinity.
- **Memory.** The wrapper becomes the reaper of every descendant, sums the memory of the whole process tree four
  times per second and kills the tree above the cap; the run then fails as an out-of-memory run does.
- **Swap.** The host must have no swap backing, checked from procfs before, during and after each command, as for
  the writable v1 backend.

This backend differs from the kernel cgroup backends, and the difference is part of the record. The memory cap is
supervised, not a kernel limit: it counts process memory, not page cache, and a spike shorter than one poll interval
can pass; the provider's own container limit remains the kernel bound. The CPU cap is a count of whole cores, not a
bandwidth quota, so G2 timings on such a host are measured on pinned cores. Every probe and unit record names its
backend. Hosts with writable cgroup v2 or v1 controls keep their previous backends unchanged. The change is
execution-only (A2-GOV-7 class (a)): no scientific file, rule, unit, chain link or pinned function changes.

## Official weight download through an expired certificate

Window `ec30v5-g2f0-w4` was the first to pass the caps step, on the supervised backend, and then stopped in
staging: the official ESM-1b weight host served a TLS certificate that had expired on 2026-11-11, so the pod's
download failed and the cached route has no upload fallback ([evc-042](../log.md#evc-042)). The bounded download
of the two approved ESM-1b files now retries without certificate verification in exactly one case: the only
verification error is an expired certificate. Every other TLS failure still refuses. The approved URL, the redirect
check, the pinned size and the pinned SHA-256 are unchanged and decide whether the bytes are accepted; a rejected
partial file is removed. Each official download writes a staging receipt that states whether TLS was verified.
The change is execution-only (A2-GOV-7 class (a)).

## Probe time limit before admission

Window `ec30v5-g2f0-w5` passed the caps step, all staging (both ESM-1b files through the expired certificate, with
matching pinned hashes), the runtime locks, the CUDA checks and the model probes, and then stopped at the last
step before workload admission: the relocated dry-run probe exceeded its fixed 300-second limit
([evc-043](../log.md#evc-043)). That probe verifies the view and runs two cold-cache preparation probes under the
two-core cap; the same two probes took 285 and 293 seconds in the workstation dry run of fold 0. The limit is now
1,800 seconds (`VM_PROBE_SECONDS`). Nothing else changes; the session deadline still bounds every remote command.
The change is execution-only (A2-GOV-7 class (a)).

## Fold-0 ceiling (A2-D8)

On 2026-11-12 the maintainer approved the fold-0 spending ceiling that A2-D8 requires before G2:

| Item | Approved value |
|---|---|
| Fold-0 gross ceiling | $85: `fold0_cost_ceiling_usd`, `runpod.gross_ceiling_usd` and the G3 preapproval `cost_usd` |
| Session time-box | 72 hours: `runpod.requested_hours` 72, `runpod.max_runtime_seconds` and G3 `gpu_seconds` 259,200 |
| Failed launch windows | at most two further failed windows in the shared ledger |
| Rolling $10 GPU policy | overridden for the fold-0 window only, up to the $85 ceiling |

The ceiling covers all of fold 0, G2 included. The launch check applies `gross(72)`, from the formula below, to the
fresh quote and requires at most $85. At the historical rate, with $5 of prior charges and a $2 egress reserve,
72 hours come to $82.29. If the fresh quote does not fit 72 hours, the session is shortened; the ceiling does not
rise. Billing ends with the session.

If fold 0 continues in a later session under A2-D6, that session keeps the $85 ceiling and counts the reconciled
charges of every earlier fold-0 session as prior charges. After Phase-4 admission, its G3 preapproval must equal
the sum of the session ceilings, as the resume check requires; that sum grants no further spending. The CLEAN
scope, inputs and caps bound by the first G3 stay fixed, except that a later session may add the final CLEAN scope.

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
| 72 hours | $74.99 |

These are cost scenarios with the historical storage configuration, not completion forecasts. The approved
ceiling is recorded [above](#fold-0-ceiling-a2-d8). Final hot/canonical storage capacity must also be priced.
The rolling $10 policy is overridden only for the fold-0 window, up to that ceiling.
The prepared CLEAN closure contains 8,115 base and 20,120 masked embedding sequences and eight 2,500-epoch
fits. Historical CPU timing cannot establish their CUDA throughput. Full-training ESMC preparation and F6
pretraining add work whose CUDA throughput is unmeasured. Forecasts must count CPU replay/export on the billed pod,
setup, transfer and closeout, without assuming a parallel speedup.

Evidence: [evc-034](../log.md#evc-034), the preserved private `fold0_clean_inputs/clean_inputs.json`,
`phase3/prebilling_20261102/phase4_requests/README.md` and
`phase3/runpod_bootstrap_posix_prepared_20261103/preflight.json` beneath the campaign output root.
