# Bounded metal architecture pilot — 2026-09-15

Experiment ID: `metal/nonoverlap/architecture-pilot/2026-09-15`.
Profile: `metal_architecture_pilot_10h_v1`.

**Historical scope:** this record describes the first allocation and its
partial state at teardown. Later recovery and completed training belong to
the [continuation report](summary_metal_architecture_pilot_continuation_20260915.md).
The first-allocation counts and raw evidence below are preserved unchanged.

## State and interpretation

**The allocation is stopped and verified; the campaign remains incomplete.**
Seven model smokes and three complete 50-epoch A1 runs (Only-GVP, Only-ESM,
early fusion) are archived and verified locally and in Drive. The three full
runs are experimentally evaluated single-seed results; the planned comparison
matrix is still partial. Late-fusion `attempt_012` was last observed at epoch
3 before session access failed, and its final artifacts were not recovered or
verified. It is not counted as complete or rejected for model performance.

Campaign evidence grade: **6 — partial/incomplete**; the three completed fits
are single-seed validation evidence (Grade 5 individually). A1 is incomplete,
A2 has not run, and no architecture winner, promotion, or held-out evaluation
is established. Teardown was confirmed at **03:56:01.801127 UTC**, with no active
Colab sessions remaining on the server.

The exact executable recipe is in the
[metal playbook](../../METAL_TRAINING_PIPELINE_PLAYBOOK.md#bounded-metal-architecture-pilot--stage-0-through-stage-2b).
The [current status](../../../EXPERIMENT_STATUS.md) owns the latest next action.
This record distinguishes preparation and runtime checks from later model
results; it does not establish a confirmed architecture advantage.

## Frozen experiment design

- Non-overlap PinMyMetal **training membership only**, with internal `pdbid`
  grouping, split seed 42, validation fraction 0.15, and metal-site-symbol
  stratification. All target arms use the same native-six-eligible examples.
- First screen: direct-four Only-GVP, Only-ESM, early fusion, and graph-level
  late fusion. Required four-/six-class and additional five-class comparisons
  cover the three core families before optional hybrid work.
- Native `val_metal_balanced_acc` selects every checkpoint. Native and
  probability-summed collapsed-four metrics come from that same checkpoint;
  compare target formulations on the common-four endpoint.
- Frozen ESMC-300m residue embeddings, conservative features, repaired PROPKA
  overlay, 10 Å extraction, 6 Å radius edges, zero additional pooling-distance
  cutoff, residue-only readout, no explicit metal nodes, no RING/augmentation.
- One GPU worker with persistent allocation/attempt accounting; seven model
  smokes precede forecast-admitted full-training blocks. The pilot's conditional
  hybrid rule controls priority and cannot establish architectural rejection.
- No HPO, Stage 6/6B, or Stage 7 runs are included. Exact PinMyMetal remains a
  later separately labeled reference benchmark; its shared historical test
  access is not reset by this campaign.

## CPU feature preparation and verification

All **1,304 non-overlap training structures** received a separately stored
PROPKA external-feature overlay. The original published v12 caches and geometry
features were preserved. The all-file audit verified source/structure hashes,
overlay checksums, and unchanged geometry.

| Overlay audit | Count/result |
|---|---:|
| Training structures | 1,304 |
| Residue rows | 484,682 |
| Rows with computed pKa-derived features | 148,607 |
| Rows retaining missing pKa-derived features | 336,075 |
| Measured zero-valued pKa-derived rows | 13,182 |
| Refreshed structures after parser/alignment fixes | 47 |
| Original geometry preserved | Yes |

Missing rows include non-titratable and structurally incomplete residues; they
are not all evidence of failed PROPKA execution. A measured zero is distinct
from a missing value. Forty-six structures required the compact wide-residue
token parser correction, and one additional structure (`3q6v`) required
insertion-code-aware alignment. The complete overlay and its masks were
audited after these corrections.

Local verification passed **95 focused tests**, with zero failures and three
warnings in **70.47 seconds**. These covered the pilot runner and scientific
contracts, overlay/parser behavior, and retained metal/EC standalone workflows.
Relevant syntax checks and `git diff --check` passed. This is code/preparation
verification, not a GPU model smoke.

After the first remote planning failure, an additional regression exercising
the exact headless Colab context passed in **8.09 seconds**. This checks the
source-v2 context fix; it does not replace the model smokes.

Evidence:
[local verification](../raw/metal_architecture_pilot_20260915/preparation/local_verification.json),
[overlay audit](../raw/metal_architecture_pilot_20260915/preparation/feature_overlay_audit.json),
[complete overlay manifest](../raw/metal_architecture_pilot_20260915/preparation/feature_overlay_manifest.json).

## Artifact identity and persistence

| Artifact | SHA256 |
|---|---|
| Active working-tree code archive `deepmzyme-metal-pilot-code-v2.tar.gz` | `da596b8842fd72a0c3b1649eab53322225a1a6f3791add326ab0907d02faa516` |
| Superseded pre-training archive `deepmzyme-metal-pilot-code.tar.gz` | `e7a7428902529d323b3ebf0b6740a2ccb9cc31fafe2ede194a21b566fb3753d2` |
| Feature archive `metal-pilot-feature-overlay.tar.gz` | `a4a5347a2194800b448362bd27094ead37999d9e25aa8cba701d3291b6b14dd8` |
| Feature overlay manifest | `41ef6627ee3c1d1595de8d1ca1fe0f03f8453cea7d6d7fff2376f8ddfd3fdef7` |
| Unchanged v12 base bundle | `90c0899829e0ac5ca94a5ef34484b74ca1014d3e9b6b1fba90bd338ee3440dee` |
| Actual GPU campaign manifest | `14010a873801cdbc2d067b8e61137f5469afb4f5d9f833cdd2251fa84080cdd5` |

The code archive is a **working-tree snapshot including uncommitted files**,
not a clean committed release. Its
[active v2 source manifest](../raw/metal_architecture_pilot_20260915/preparation/code_snapshot_manifest_v2.json)
records the base commit, working-tree status, and file hashes. The base dataset
bundle and repaired feature overlay have separate identities.

The [Drive campaign folder](https://drive.google.com/drive/folders/1slTff0joKjL-gZJDhYSGbHzOPnYhGk6I)
contains the
[active v2 source archive](https://drive.google.com/file/d/1QlzmGosr_WOtHFvVNXm-O_njzEdP0SO_/view),
[superseded pre-training source archive](https://drive.google.com/file/d/1YECGn0Qn4j39Z7vp-cMQb1JFaQR0vld7/view),
and [feature overlay archive](https://drive.google.com/file/d/1thEazEdmE6EXJL5AKmXCUF4tog-D_rbh/view).
Preparation receipts verify local SHA256 values and uploaded Drive file IDs,
parents, and byte sizes. Those upload checks do not themselves establish a
downloaded Drive SHA256; remote bootstrap must verify the downloaded artifacts
against the frozen hashes before execution.

Evidence:
[active v2 source transfer receipt](../raw/metal_architecture_pilot_20260915/preparation/source_transfer_receipt_v2.json),
[original source transfer receipt](../raw/metal_architecture_pilot_20260915/preparation/source_transfer_receipt.json),
[overlay transfer receipt](../raw/metal_architecture_pilot_20260915/preparation/overlay_transfer_receipt.json).

## Headless bootstrap correction and GPU readiness

The first bootstrap stopped in notebook command planning with
`NameError: name 'DRIVE_ROOT_PATH' is not defined`. The headless namespace
needed the Drive-root context expected by the notebook when `/content` exists.
Source v2 adds that context before executing the planner. This happened
**before any model training**, so it does not mix model results from two code
versions. The original failed bootstrap and source snapshot remain preserved;
its allocated runtime still counts against the same campaign budget.

Evidence: [original bootstrap failure log](../raw/metal_architecture_pilot_20260915/preparation/bootstrap_v1.log),
[versioned source receipt](../raw/metal_architecture_pilot_20260915/preparation/source_transfer_receipt_v2.json),
[v2 runtime setup](../raw/metal_architecture_pilot_20260915/preparation/pilot_setup_v2.json).

Source-v2 planning and GPU readiness passed. The readiness attempt took
**124.15 seconds**, and its extracted cohort passed feature, identity,
class-support, and pooling checks:

| Readiness field | Training | Validation |
|---|---:|---:|
| Retained pockets | 1,181 | 208 |
| PDB groups | 1,151 | 110 |
| Shared PDB groups between splits | 0 | 0 |

Native-six validation supports are **Mn 97, Cu 15, Zn 32, Fe 44, Co 13,
Ni 7**. The zero-cutoff pooling audit passed on all **1,389 pockets**: all
extracted residue nodes are retained for the residue-only readout. These
checks verify execution inputs and geometry semantics; they are not learned
model performance or architecture comparisons. Readiness `attempt_001` has
been archived and verified locally and in
[Drive](https://drive.google.com/file/d/1N75xKIxuxeiRU2r_TjV7RPA0dtpB-QAb/view).
The subsequent model-smoke results are recorded below.

Portable evidence:
[GPU readiness](../raw/metal_architecture_pilot_20260915/readiness/readiness.json),
[expected split](../raw/metal_architecture_pilot_20260915/readiness/expected_split.json),
[pooling diagnostics](../raw/metal_architecture_pilot_20260915/readiness/pooling_diagnostics.json),
[actual campaign manifest](../raw/metal_architecture_pilot_20260915/readiness/campaign_manifest.json),
[attempt ledger](../raw/metal_architecture_pilot_20260915/readiness/campaign_attempt_ledger.json).

## Seven completed model smokes

All planned one-epoch smoke variants completed successfully with native
validation checkpoint selection, the fixed shared cohort, and no held-out
evaluation. Each selected checkpoint is epoch 1. The seven archive receipts
verify local checksums and Drive storage and bind to campaign manifest
`14010a873801cdbc2d067b8e61137f5469afb4f5d9f833cdd2251fa84080cdd5`.

| Attempt | Model and target | Total attempt time | Outcome |
|---|---|---:|---|
| 002 | Only-GVP, four-class | 98.13 s | Passed |
| 003 | Only-GVP, five-class | 98.13 s | Passed |
| 004 | Only-GVP, six-class | 98.13 s | Passed |
| 005 | Only-ESM, four-class | 112.15 s | Passed |
| 006 | Early fusion, four-class | 114.15 s | Passed |
| 007 | Graph-level late fusion, four-class | 114.15 s | Passed |
| 008 | Hybrid fusion, four-class | 114.15 s | Passed |

These elapsed times include process/data setup, training, validation, and
saving. They are operational profile inputs. Smoke accuracies do not rank
architectures or justify the conditional hybrid full-training gate.

The whole-device GPU-memory monitor samples every 10 seconds. It missed the
short Only-GVP training phases: their recorded **3 MiB is not an actual
training-memory peak**. Preserve those raw values with this limitation; do not
infer memory efficiency or comparative peak usage from them.

The [portable smoke snapshot](../raw/metal_architecture_pilot_20260915/smokes/)
contains the ledger/coverage/configuration state through attempt 008, all seven
[archive receipts](../raw/metal_architecture_pilot_20260915/smokes/transfer_receipts/),
and each smoke's exact config, metadata, dataset/split summaries, preparation
status, and epoch/train/validation metrics under `smokes/runs/`. Copied files
were checked byte-for-byte against the local campaign artifacts. Checkpoint
tensors remain in the verified full archives and were not copied into Git.

The raw snapshot's `awaiting_archive_transfer` state is the runner state saved
at smoke completion; the accompanying receipts show that all seven transfers
subsequently passed. The three verified full A1 results below are newer than
that portable smoke snapshot.

## Verified partial A1 results

The saved local report, generated **2026-09-15 03:54:29 UTC**, verified three
completed 50-epoch runs. All use direct-four training, LR `3e-5`, model seed 42,
the same 1,181/208 pocket split, and native validation balanced-accuracy
checkpoint selection. Values below describe individual runs; the incomplete
block is excluded from architecture ranking.

| Attempt | Family | Selected epoch | Validation balanced accuracy | Minimum class recall |
|---|---|---:|---:|---:|
| 009 | Only-GVP | 44 | 0.647114 | 0.406250 |
| 010 | Only-ESM | 22 | 0.721228 | 0.593750 |
| 011 | GVP + early fusion | 40 | 0.693667 | 0.468750 |

Native and common-four metrics are identical here because these three runs
train directly on four classes. Validation support is Mn 97, Cu 15, Zn 32, and
Class VIII 64. No confidence intervals or promotion decision were produced.
The second learning rate, target-formulation full runs, and seed repeats remain
pending; these values do not establish the best architecture or learning rate.

Portable evidence:
[verified report snapshot](../raw/metal_architecture_pilot_20260915/partial_a1/local_report/pilot_local_report.json),
[full-run CSV](../raw/metal_architecture_pilot_20260915/partial_a1/local_report/pilot_local_full_runs.csv),
[per-run configs and metrics](../raw/metal_architecture_pilot_20260915/partial_a1/runs/),
[archive receipts](../raw/metal_architecture_pilot_20260915/partial_a1/transfer_receipts/).
These copies were verified byte-for-byte. The separate large dataset summaries
were not duplicated in this partial-result snapshot; shared cohort evidence
remains in the readiness/smoke records and per-run metadata. Checkpoints remain
in the verified local/Drive archives.

## Session interruption and verified teardown

At **03:53:09 UTC**, CLI session access failed with mixed 404/401 responses.
The authentication-status command still reported an OAuth credential valid
for about 59 minutes with Colab scope. The original runtime remained
server-listed, but its local session mapping was deleted. The underlying
cause remains unproven; these observations do not demonstrate token expiry.

The 401 stop condition suspended further training orchestration. Cleanup
verified the original allocation's ownership from CLI creation history,
restored only that owned session name with empty runtime credentials, and
stopped it by name. No reauthentication, new allocation, or resumed training
was performed during cleanup. `colab stop` confirmed termination, and the
subsequent session listing returned **no active sessions on the server**. The
allocation watchdog also exited after the stop marker.

Late-fusion `attempt_012` was last seen at epoch 3. Its partial remote status
is preserved as interruption evidence, not a completed validation result or
architecture rejection. The last archived training ledger stops at completed
`attempt_011`; its lack of an interrupted row does not mean no interruption
occurred. The separate closeout receipt records that unresolved attempt.

Portable evidence:
[stop receipt](../raw/metal_architecture_pilot_20260915/closeout/session_stop_receipt.json),
[last remote status](../raw/metal_architecture_pilot_20260915/closeout/last_remote_status.json),
[closed allocation ledger](../raw/metal_architecture_pilot_20260915/closeout/allocation_ledger_closed.json),
[frozen-source verification](../raw/metal_architecture_pilot_20260915/closeout/frozen_source_verification.json).
All **73 campaign source files** remained unchanged, with zero mismatches.
See [TECH-012](../../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-012--colab-session-access-loss-with-valid-authentication-status)
for the remaining CLI/session-recovery issue.

The [final derived report](../raw/metal_architecture_pilot_20260915/closeout/local_report/pilot_local_report.json),
generated at 04:00:21 UTC, combines the archived eleven-attempt ledger with the
separate stop/last-remote receipts. It records the missing twelfth attempt as
unverified without fabricating a completed or timed ledger entry. It confirms
seven completed smokes, three full runs, A1 coverage 3/4, and verified teardown.

The [Drive closeout archive](https://drive.google.com/file/d/1Xmtpx1783uEwtm6W7fhgmr02LjETaIwn/view)
preserves the final host receipts, derived report, original remote metadata
snapshots, per-attempt archive references, and document snapshot. Its 54 files
were verified during packaging; the 475,774-byte archive has SHA256
`16a7b31f9fda7337522b8573fdbf985a1f88ed691b13fa990cf0ac70ce82e9bf`.
The [upload receipt](../raw/metal_architecture_pilot_20260915/closeout/closeout_transfer_receipt.json)
records the local hash and verified Drive file ID, parent, and byte size.
The archive predates its own upload receipt and this link addition.

## Allocated runtime and next gate

| Field | Recorded value |
|---|---|
| Session | `deepmzyme-metal-pilot-20260915` |
| GPU | G4, NVIDIA RTX PRO 6000 Blackwell |
| Python | 3.13.15 |
| PyTorch/CUDA build | 2.11.0+cu128, includes `sm_120` |
| CUDA probe | Passed |
| Allocation start, UTC | 2026-09-15 02:52:13 UTC |
| Allocation start, Unix seconds | `1789440733.8320558` |
| Verified allocation stop, UTC | 2026-09-15 03:56:01.801127 UTC |
| Total allocated time | 3,827.969 seconds = 1.063325 hours |
| Unspent original campaign allowance | 8.936675 hours |
| Runtime output | `/content/metal_architecture_pilot_10h_v1` |
| Active runtime repository | `/content/DeepMzyme_metal_pilot_v2` |
| Runtime data root | `/content/deepmzyme_bundle/DeepMzyme_Data` |

The [active setup record](../raw/metal_architecture_pilot_20260915/preparation/pilot_setup_v2.json)
fixes the allocation timestamp, roots, source/overlay hashes, and base-bundle
URL. The [original setup](../raw/metal_architecture_pilot_20260915/preparation/pilot_setup.json)
is retained for the failed pre-training bootstrap. Both preserve allocation
start `1789440733.8320558`; the source correction did not reset the budget.
The completed readiness check does not establish model-training throughput.

The closed interval includes bootstrap, the pre-training context-fix recovery,
readiness, all training, transfers, the access failure, and cleanup. The
remaining allowance is not a new allocation or permission to reset the budget.

Next: resolve CLI connection-loss handling before another GPU allocation.
Any continuation must preserve completed runs and cumulative allocated time,
record a linked retry for the unverified late-fusion attempt, and finish A1/A2
before drawing architecture conclusions. **Cross-session recovery has not
been tested**; the preserved evidence is not a certified one-command resume
package. Reconcile and test the missing-attempt/closed-allocation state before
another allocation. No further training was launched.

No held-out inference or metrics have been produced. Downloading a bundle that
contains protected test assets does not authorize their use in training,
validation selection, or model comparison.
