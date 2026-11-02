# G2 readiness and the first Fold-0 comparison

Status: CPU implementation and required verification complete, 2026-11-02; publication and G2 pending.
This is a bounded execution proposal, not GPU authorization.
The [continuation](continuation.md) owns the full sequence; the
[EC playbook](../../EC_TRAINING_PIPELINE_PLAYBOOK.md#ec-class-versus-clean30-campaign-recipe) owns commands.
The S1 amendment is published at `3154a7f`; its publication check passed before WP-13 development.
Selection, scoring, simulations and all original freeze bytes remain unchanged.

## Completed CPU preparation

The [execution manifest](wp13/execution_manifest.json) binds WP-13 source, both original freezes, the published
S1 amendment and the deterministic 1,000-row training-only timing inputs. The Fold-0 transfer archive contains
4,366 files and is 1,485,957,120 bytes; SHA-256:
`a036a782e3f0de041e896b7e2ff3a79dd13eb6d827629b0eca28dfcbc9b90334`.

All 98 focused tests passed. A five-epoch synthetic F4 CPU fit completed training, terminal verification,
independent replay, host-copy acknowledgment and verified reuse. This is engineering evidence, not an empirical
S1 result. The [verification record](wp13/verification.json) also records 3,311 regression passes (18 existing skips,
15 subtest passes), 44 CPU smoke passes (one existing skip) and zero strict documentation failures.

Before G2, publish the tested package; prepare and hash the private local weights, CUDA embedding environment
and relocatable package; and approve the exact session, resource limits and output destinations below. After
start, verify the actual runtime and free space, perform the separate relocated-view probe, and bind requests to
the controller's actual allocation/deadline. No paid preparation or measurement has run.

## Proposed G2 scope

Use the existing stopped `g2-standard-8` VM in `us-central1-a`: one L4, 8 vCPU and 32 GiB RAM.
Read-only controller status, price and preflight checks passed on 2026-11-02. This establishes the current
resource state and price, not start-time capacity or the installed runtime's readiness.

| Measurement | Fixed work |
|---|---|
| Family timings | F1, F2, F3, F4, F5 and C-G: five epochs each |
| L3 timing | F4 with the existing L3 recipe: five epochs |
| Lane diagnostic | F4: two serial 30-epoch repeats and one batch of three 30-epoch repeats |
| Embedding timings | ESMC and ESM-1b, each on the same 1,000 official Fold-0 training rows |

All fits use Fold 0, inner fold 0, seed 42 and LR-grid index 1, with separate `ecc30timing` identities.
They cannot become S1 selection inputs. Transfer only the certified Fold-0 view with both existing chain-mean
caches, the deterministic timing inputs, published code and the separately approved private CLEAN setup.
Exclude old probes/caches, other folds, full overlays, private freeze files and the campaign identity tree.

Start with one training process. The only concurrency exception is the declared three-process lane batch on the
same L4. Its fixed comparison uses classes with at least ten validation proteins, at most one BA point and
three recall points of disagreement, at least 95% top-1 agreement and at least 1.2 times equal-work speedup.
A failed lane gate keeps subsequent execution serial. No lane-count or model search is added.

Proposed per-child caps, checked against actual hardware before measurement: a fit or ESMC timing gets two
CPU cores and 8 GiB RAM; ESM-1b timing gets four CPU cores and 24 GiB RAM, with zero swap allowed in each scope.
Each of the three lane children uses the fit cap. The lane gate checks sampled whole-host use against 600%
CPU, 24 GiB RAM and 21 GiB VRAM, and requires zero observed swap, no OOM and intact read guards. These aggregate
measurements are acceptance checks, not a shared hard cgroup limit. CPU checks are not L4 performance evidence.

Before the paid start, prepare the ESM-1b weights and companion on Data1 and an isolated GPU embedding
environment. The existing private CLEAN fit environment has `torch==1.11.0+cpu`; it cannot supply CUDA timings.
Keep that environment unchanged. The proposed embedding environment retains Python 3.10.4, fair-esm 2.0.0,
NumPy 1.22.3 and the other locked packages, replacing only Torch with `1.11.0+cu113` without dependency upgrades.
The [official wheel index](https://download.pytorch.org/whl/cu113/torch/) supplies the CPython 3.10 Linux build;
[NVIDIA's compatibility guide](https://docs.nvidia.com/cuda/ada-compatibility-guide/#applications-built-using-cuda-toolkit-11-0-through-11-7)
supports testing compatible Ampere binaries on Ada. This is a candidate, not measured L4 compatibility.
Record fresh environment locks/hashes and verify a relocatable package before starting the VM.

Weights, this GPU environment and its relocatable package are not prepared by the CPU implementation task.
Their preparation belongs in the bounded G2 authorization and should precede billing. Do not install fair-esm
over the separate ESMC environment. On the VM, require CUDA proof from the actual private embedding interpreter
and evidence that the extractor moved the model to the GPU; its requested-device field alone is insufficient
because the pinned extractor can fall back to CPU. Unknown runtime/setup time prevents a guarantee that all
measurements will finish within the proposed session.

## Budget and persistence

The proposal remains one requested two-hour session with a **$3 gross ceiling**. The existing-VM start preview
currently gives a **117-minute automatic STOP**. At the verified $0.879172/hour gross rate, the controller
estimates $1.71, or $1.97 with its 15% margin. Allocation, setup, transfer, verification, fits and replay all
consume this session. The actual allocation timestamp and provider deadline must come from the controller.

Reserve 900 seconds for persistence/closeout and apply the existing 1.25 admission multiplier. Give the first
probe an explicit bounded timeout within the approved session; do not present an unmeasured runtime as a
throughput estimate. Subsequent timing evidence supplies forecasts. Admit the entire parallel batch without
assuming the speedup being tested. A partial session is reported as partial.

For transfer accounting, reserve at most 4 GiB of outbound traffic including retries and overhead: $0.60 at
the published North America-to-Middle East rate of $0.15/GiB, without assuming a free allowance. Inbound transfer
is free under [Google's network pricing](https://cloud.google.com/vpc/network-pricing). Keep cumulative
host-pull payloads below 3 GiB to leave 1 GiB for overhead/retries; inspect each persistence manifest before
copying. Adding approximately $0.04 for the retained snapshots during 117 minutes gives a planning total of
about **$2.61 including margins**. Recheck rates and the remaining $3 allowance before starting and admitting work.

Each terminal unit or lane batch is copied to the independent workstation destination on Data1, verified by
reading its bytes, and acknowledged before another unit starts. Include actual transfer/readback time in the
timing comparison. The controller owns provider STOP; the execution helper owns worker interruption and recovery.
Confirm provider shutdown after closeout. Existing retained storage continues separately: approximately $15/month
for the stopped boot disk and $15/month for retained snapshots; this proposal creates no additional disk/snapshot.

## Storage placement

The first check found 7.015 GB free on internal ext4 and 59.755 GB on Data1. Completed synthetic pytest temporary
directories from S1 verification were then removed after their logs had been preserved, increasing internal free
space to **9.023 GB (8.40 GiB)**. No scientific artifact, archive or independent backup was removed.

| Content | Placement |
|---|---|
| ESM-1b weights, companion regression file and inactive archives | Directly under the Data1 campaign root; no intermediate ext4 copy |
| Fold-0 view and its transfer archive | Data1; both are prepared; archive size and checksum are recorded above |
| Private CLEAN environment and static embeddings | Data1; the existing environment is already there |
| One active CLEAN farm, map, fit and temporary files | Internal ext4; copy only that fit's hot payloads, then use local links |
| Completed outputs and independent host-pull backups | Data1, verified before reclaiming an active working directory |

Links from an ext4 farm back to Data1 would retain the observed FUSE I/O slowdown. Keep the hot vector payloads
on ext4, as well as the farm's links. Do not copy the entire Fold-0 view or the 7.8 GB weights onto ext4.

The official training table and certified inner exclusions give a worst one-fit planning allowance of
1,105,377,344 bytes, or **1,381,721,680 bytes (1.29 GiB)** with 25% margin. The allowance includes vector files,
the distance map, annotation embeddings, checkpoints, metadata and temporary space. This is a forecast;
measure the generated files before each fit. Require `free bytes >= 5 GiB + 1.25 * active peak bytes` before
staging the next fit, preserve the reserve during work, and keep fits serial. VM extraction capacity must be
checked after the authorized start; its free space was not inspected while stopped.

## Path to the empirical milestone

Publish WP-13 and its execution pins, approve the bounded G2 scope, and finish measured
timing/replay/persistence checks. Freeze 1c records those measurements and the numerical L3 ceilings; G3 then
authorizes the measured execution budget. Phase-4 real-unit orchestration still needs completion using these
same hooks before reportable fits; no GPU should remain billed while that code is written.

The next scientific workload is exactly **96 DeepMzyme S1 fits plus eight CLEAN inner fits**: all six families,
two LRs, four inner folds and seeds 42/43; CLEAN R1-inner and R2-inner each use four folds and seed 42.
Prepare only the embeddings needed by these eight CLEAN fits first. Run their serial CPU fits alongside remote
S1 when authorized, then apply the unchanged G4 gate. Do not launch S2 or folds 1–4 before that decision.
