# G2 readiness and the first Fold-0 comparison

Status: 2026-11-03; GCP recovery is closed after stockout ([evc-023](log.md#evc-023)).
The initial Runpod setup was stopped after verified preservation ([evc-024](log.md#evc-024)).
An earlier host passed kernel caps; migration reached a host without delegated controls. The isolated
runtime repair is CPU-verified ([evc-027](log.md#evc-027)). The reproducible cache/bootstrap extension is
implemented ([evc-028](log.md#evc-028)); actual G2 qualification remains pending. Zero events completed.
The GCP two-hour request and 117-minute STOP do not establish a Runpod deadline or spending ceiling.
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

Strict `verify-execution` passed without draft mode on published WP-13 before extending it. The
[extended manifest](execution_v2/manifest.json) authenticates the workstation queue and Fold-0 dispatcher against
that immutable base. Its publication is required before the launch command can allocate compute. After start,
the queue verifies the actual runtime and free space, performs the relocated-view probe and binds all requests
to the controller's actual allocation/deadline. No GCP remote staging or measurement ran in the closed passes;
the later Runpod setup is recorded separately below. [evc-022](log.md#evc-022) preserves cross-region charges.

## Closed GCP recovery and retained preparation

The private launch configuration now binds `same-region-v1`: one source start in `us-central1-a`, then
`us-central1-b` and `us-central1-c` only, in a fresh driver namespace. Cross-region recovery is excluded.
The requested runtime stays two hours with a 117-minute provider STOP. The **$3 gross ceiling** includes
the **$0.30 same-region recovery-storage reserve**, giving a conservative envelope of **$2.9640956**.
The controller's $25 daily guard is unchanged; it does not expand campaign authority.

The authorized pass returned stockout in all three zones. No remote staging, HostPull, embedding or fit began.
Evidence-gated abandonment removed both restored disks, the temporary snapshot and its recycle-bin copy.
The source VM, its disk and two pre-existing snapshots remain; provider status confirmed `TERMINATED`.
Incremental temporary-storage accounting is approximately **$0.02677**, not provider-confirmed billed spend.
The published execution manifest and all scientific preparation are reused unchanged.

[evc-024](log.md#evc-024) owns the separate Runpod setup and closeout evidence;
[evc-023](log.md#evc-023) owns the same-region authorization, attempts and cleanup evidence;
[evc-022](log.md#evc-022) preserves the earlier exhausted $19 cross-region pass. Both GCP passes are closed.
Another GCP allocation requires a fresh decision. Successful recovery still requires fit/replay/independent-backup
evidence before adoption; failed recovery uses the [abandonment gate](../../GCP_GPU_RUNBOOK.md#recover-a-stopped-vm-in-the-same-region).

## Runpod execution qualification

[evc-024](log.md#evc-024) records the allocation, observed environment, ten checked staged files, verified
off-pod preservation and provider-confirmed shutdown. The pod has no persistent volume; recovery uses
retained local inputs and source records, not an assumption that its container files survive.

The original [execution-v3 extension](execution_v3/manifest.json) remains preserved. The additive
[compatibility manifest](execution_v3/manifest_cgroup_v1.json) remains preserved. The current
[runtime-v2 manifest](execution_v3/manifest_runtime_v2.json) preserves isolated execution and its closed
replacement window; [evc-027](log.md#evc-027) owns its verification. The additive
[execution-v4 manifest](execution_v4/manifest.json) authenticates the bootstrap extension against that published ancestor. The corrected
[runtime locks](execution_v3/runtime_locks_v2.json) retain the same complete package maps.
The old pod is deleted. Full runtime qualification remains pending. Before any Runpod G2 fit:

- Preserve a clean, published checkout and authenticate its source/input manifests. The copied tree excluded
  `.git`; code inspection establishes the missing prerequisite, not a measured verifier failure.
- Qualify the actual L4 or RTX 4090 hardware, delegated child resource controls and both locked Python
  runtimes with user-site packages disabled. GPU identity alone does not establish delegation. Require full-model ESM-1b
  CUDA transfer and the source-bound CPU/CUDA qualification receipts; old container imports are insufficient.
- Provide a validated equivalent for the GCP-bound transport/session checks and the declared CPU/RAM/no-swap
  caps. Keep the exact queue, admission margin, closeout reserve, input/read guards and HostPull acknowledgments.
- Enforce the approved $5 gross envelope and 117-minute maximum allocation with the independent workstation
  stop process armed before creation or named adoption. Preserve the original allocation clock across
  container restarts. Verify its heartbeat and provider STOP; preserve the 900-second reserve.
- Complete extraction/view checks, private-runtime checks and independent output persistence on the qualified
  runtime. A named zero-volume handoff requires per-unit HostPull to independent Data1 and NVMe
  devices before advancement; closeout stops the pod without deleting it. The ten matching staged-file
  hashes establish input preservation, not execution readiness.

The v3 manifest preserves the original source records and scientific guards. The fixed G2 scope below remains
the reviewed timing recipe. Zero measured G2 results are available;
Freeze 1c and numerical G3 still precede the 96 S1 plus eight CLEAN inner fits.

## Reproducible Runpod bootstrap

The committed workstation entry point now owns local preparation, cache seeding and bounded execution;
the [EC recipe](../../EC_TRAINING_PIPELINE_PLAYBOOK.md#automated-g2-and-the-fold-0-queues) owns its commands.
The extension preserves the original execution manifests and runtime/archive pins. Its warm path is designed
to avoid repeated bulk transfers from the workstation; actual cold/warm throughput remains unmeasured.
The [private-configuration template](execution_v4/workstation_config.example.json) supplies the complete
staging closure and runtime maps. Replace every local path and resource placeholder, reconcile prior spending
and refresh the quote; its deliberately expired quote and ceiling-sized prior reserve refuse accidental launch.
The [verification record](execution_v4/verification.json) binds the CPU checks to unchanged source. Both complete
runtime maps and archive pins are checked against the published lock document before large input reads.

The approved cache is a private **20-GB Standard network volume in EU-RO-1**, accessed as an S3 origin rather
than mounted on a GPU pod. It holds exactly the two locked runtime archives and compressed Fold-0 transfer
archive. This leaves GPU placement independent of the cache datacenter. The original view archive is restored
and checked against its existing full checksum. Both public ESM-1b files download from their pinned official
origins; the ESMC model downloads from its pinned published revision and reconstructs the original tar bytes.
Every model and final archive must match its full source-bound SHA-256 before use.

The S3 profile `runpod-artifact-cache` stays on the workstation. Configuration and receipts contain no keys;
the pod receives only temporary exact-GET request headers in protected files. Seeding first verifies a tiny
signed-header PUT/GET round trip, then uploads SHA-addressed 64-MiB chunks with at most two workers.
Chunk and descriptor readback precede the completion record. A warm start reads the small descriptors before
allocation; the destination still verifies every downloaded chunk and assembled archive. Interrupted transport
resumes only at verified chunk boundaries. Cache preparation allocates no GPU and does not waive pricing or
storage accounting.

[evc-029](log.md#evc-029) records S3 authentication, local preflight timings and transport corrections.
[evc-030](log.md#evc-030) records verified seeding and the closed stockout with provider absence verified;
actual GPU qualification remains pending.

After allocation, only the authenticated cap helper transfers initially. It must prepare private delegated
controls without changing provider limits and pass the actual kernel probe before checkout or bulk staging.
The clean published `.git` checkout is staged once. The main runtime is restored first, its complete package
lock is checked with user site disabled, and an actual capped CUDA workload must pass before the remaining
inputs stage with at most two workers. Full ESM-1b model transfer and independent CPU/GPU replay remain
mandatory after staging. Default failures close out the pass. Explicit `runpod.keep_alive_for_operator:true`
retains a qualified ownership lease for diagnosis or completed-result review, bounded by the original deadline
minus the 900-second closeout reserve. The [EC recipe](../../EC_TRAINING_PIPELINE_PLAYBOOK.md#automated-g2-and-the-fold-0-queues)
owns its controls. Only unchanged published-source qualification can retry before workload admission;
no failed scientific unit is rerun by this control. GPU identity never substitutes for kernel-cap verification.

A continuous workstation heartbeat protects long transfers and qualification; the independent stop process
and original allocation clock remain authoritative. The **$5 pass gross ceiling**, **117-minute hard stop**,
two-hour requested maximum, 1.25 admission multiplier, 900-second closeout reserve, exact twelve-event queue,
F9 guards, replay and per-unit dual-device HostPull acknowledgments are unchanged. The separate local
**$10 gross rolling-24-hour policy** requires a decision near its warning threshold; it is not a provider hard
cap and does not expand the narrower pass ceiling. Cache storage counts toward spending.

The prior absolute windows are closed and must not be reset. A new authorized pass uses a fresh state namespace
and actual allocation clock. Before it starts, publish execution-v4, verify the seeded cache, refresh the price
and spending bounds, confirm GCP remains `TERMINATED`, and recheck the independent Data1/NVMe reserves.
No new GPU qualification or G2 measurement is established by this implementation record. Freeze 1c and numerical
G3 still require measured outputs; retained-resource shutdown is verified at its bounded closeout.
Phase 4 and held-out reads stay gated.

## Previous start attempts

The maintainer authorized one existing-VM start with the $3 ceiling and 117-minute automatic STOP. Local
preflight and the refreshed controller preview passed. Google rejected the start with
`ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS` in `us-central1-a`; the closed attempt records zero completed units.
Independent provider status confirmed `TERMINATED` at 11:12 UTC and zero recorded running time that day.
The closeout receipt, original configuration and error are preserved on Data1 and independently mirrored on NVMe.

Freeze 1c cannot be assembled without its ten verified measurements. The five private Phase-4 requests remain
pending with an explicit capacity blocker; numerical forecasts and G3 ceilings remain unset. The existing closed
attempt must not be reset. The authorized retry used a new namespace and tried `us-central1-b` and
`us-central1-c` after another confirmed source stockout. Both returned capacity failures. The controller verified
the failed operations, removed both destination disks and the temporary snapshot/recycle copy, and archived
the pass. The original VM and boot disk remain intact. Further allocation needs a fresh start decision;
all prepared inputs remain reusable. No Freeze 1c or numerical G3 proposal exists without measurements.

The previous same-region recovery configuration reserved $0.30 beyond the original $2.6641 envelope,
giving **$2.9641** within $3.
It prices up to four hours of overlapping storage through cleanup, including two possible destination disks,
the source disk, the temporary snapshot, its recycle-bin minimum and retained snapshots. This storage window
does not extend the 117-minute GPU STOP. Fresh controller prices and the explicit zone filter must pass before
allocation. Successful fit/replay/independent-backup evidence permits authorized finalization; an exhausted pass
uses the separate abandonment evidence gate. No further start or region is included.

## Fixed G2 scope

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
The prepared per-unit timeouts are admission bounds, not measured throughput predictions: 360 seconds for each
family/L3 probe and ESMC, 480 for ESM-1b, and 900 for each lane event. The lane forecast also includes three
times the slower measured serial repeat. Each next unit must still fit with the margin and closeout reserve.
The ESM-1b admission threshold is now 1,500 seconds (25 minutes), including the unchanged 900-second reserve.

The ESM-1b weights and companion were downloaded directly to Data1 and independently read back. Weight SHA-256:
`0569754efaff7dcb7e068c27367bc73f10afb4b450ea30aac30d9bc60783a8b1` (7,828,576,466 bytes); companion SHA-256:
`77193a8814f0db0b36a03aebb1a311adc6b4745f463c04839defc15407bbb28a` (3,174 bytes).
The existing private CLEAN fit environment retains `torch==1.11.0+cpu`. The separate embedding environment retains
Python 3.10.4, fair-esm 2.0.0,
NumPy 1.22.3 and the other locked packages, replacing only Torch with `1.11.0+cu113` without dependency upgrades.
The [official wheel index](https://download.pytorch.org/whl/cu113/torch/) supplies the CPython 3.10 Linux build;
[NVIDIA's compatibility guide](https://docs.nvidia.com/cuda/ada-compatibility-guide/#applications-built-using-cuda-toolkit-11-0-through-11-7)
supports testing compatible Ampere binaries on Ada. This is a candidate, not measured L4 compatibility.
CPU imports, tokenization, a small synthetic ESM forward/contact calculation and exact package comparisons passed
both before and after fresh extraction and `conda-unpack`. The archive is 1,924,660,777 bytes, SHA-256
`3a7706486611280cf3c17ce3b1b7be3f4a6ff60252e38d13a484e30f56df7b9d`; it expands to 3,882,102,310 regular-file bytes.
The [verification record](execution_v2/verification.json) pins the locks, preparation receipt, CPU checks and
local-only launch preview. Actual L4 kernels and a full ESM-1b model load remain G2 checks.

The offline ESMC-300m cache is also prepared and pinned; its original frozen weights are reused. Do not install fair-esm
over the separate ESMC environment. On the VM, require CUDA proof from the actual private embedding interpreter
and evidence that the extractor moved the model to the GPU; its requested-device field alone is insufficient
because the pinned extractor can fall back to CPU. Unknown runtime/setup time prevents a guarantee that all
measurements will finish within the proposed session.

Both public ESM-1b files have pinned official HTTPS URLs. The VM attempts a direct download into a partial file,
verifies its size and SHA-256, and only then renames it. A known terminal download failure permits the existing
workstation-upload fallback; ambiguous SSH failure stops staging. Local preflight still fully verifies the Data1
copies. The normal workstation upload is 4,743,701,970 bytes; direct download avoids sending the 7,828,579,640
weight-file bytes through the workstation uplink. The full allocation deadline still includes all preparation.

## Budget and persistence

The latest G2 envelope retained a two-hour request and **$3 gross ceiling**, including the $0.30 recovery
reserve described above. Its single same-region pass is consumed. The existing-VM start preview
gives a **117-minute automatic STOP**. At the verified $0.879172/hour gross rate, the controller
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
about **$2.67 including margins** when conservatively charging two complete hours. Recheck rates and the remaining
authorized allowance before starting and admitting work; the launcher rejects stale quotes.

The workstation queue automates all twelve events, including the two serial lane references, and produces the
ten measurements required by Freeze 1c. Each terminal unit or lane batch is copied to the independent workstation
destination on Data1, verified by reading its bytes, and acknowledged before another unit starts. Include actual transfer/readback time in the
timing comparison. The controller owns provider STOP; the execution helper owns worker interruption and recovery.
Confirm provider shutdown after closeout. Existing retained storage continues separately: approximately $15/month
for the stopped boot disk and $15/month for retained snapshots. The recovery pass's additional resources are removed.

## Storage placement

The first check found 7.015 GB free on internal ext4 and 59.755 GB on Data1. Completed synthetic pytest temporary
directories from S1 verification were then removed after their logs had been preserved, increasing internal free
space to **9.023 GB (8.40 GiB)**. No scientific artifact, archive or independent backup was removed.
After this preparation and temporary-test cleanup, approximately **8.8 GB remained on ext4 and 36.1 GB on Data1**.
The verified G2 staging peak is 19,269,249,867 bytes; the VM must have that space plus the 5 GiB reserve before
transfer. Its capacity is checked after the deliberate start, without assuming stopped-VM free space.

| Content | Placement |
|---|---|
| ESM-1b weights, companion regression file and inactive archives | Directly under the Data1 campaign root; no intermediate ext4 copy |
| Fold-0 view and its transfer archive | Data1; both are prepared; archive size and checksum are recorded above |
| Private CLEAN environment and static embeddings | Data1; the existing environment is already there |
| One active CLEAN farm, map, fit and temporary files | Internal ext4; copy only that fit's hot payloads, then use local links |
| Completed outputs and independent host-pull backups | Data1, verified before reclaiming an active working directory |

Links from an ext4 farm back to Data1 would retain the observed FUSE I/O slowdown. Keep the hot vector payloads
on ext4, as well as the farm's links. Do not copy the entire Fold-0 view or the 7.8 GB weights onto ext4.
Local CLEAN uses two CPU cores and zero swap. Its separate map/coordinator envelope is 4,500,000,000 bytes;
training and export children each retain the ordinary 3 GiB cap. Authenticate the completed map before training.

The official training table and certified inner exclusions give a worst one-fit planning allowance of
1,105,377,344 bytes, or **1,381,721,680 bytes (1.29 GiB)** with 25% margin. The allowance includes vector files,
the distance map, annotation embeddings, checkpoints, metadata and temporary space. This is a forecast;
measure the generated files before each fit. Require `free bytes >= 5 GiB + 1.25 * active peak bytes` before
staging the next fit, preserve the reserve during work, and keep fits serial. VM extraction capacity must be
checked after the authorized start; its free space was not inspected while stopped.

## Path to the empirical milestone

Publish the extended execution package, then deliberately launch the bounded G2 queue using the
[playbook](../../EC_TRAINING_PIPELINE_PLAYBOOK.md#automated-g2-and-the-fold-0-queues). Freeze 1c records the complete
measurements and numerical L3 ceilings; G3 then authorizes the measured execution budget. The Phase-4 dispatcher
is implemented before G2: it provides the 96-fit queue, two required CLEAN embedding farms, eight serial CLEAN
fits, terminal verification/replay, independent persistence and the existing S1 selection/G4 route. Subsequent
GPU queues attach only to an already authorized, protected allocation and stop their own allocation on exit.
No additional Phase-4 framework is required between G2 and this first milestone.

The next scientific workload is exactly **96 DeepMzyme S1 fits plus eight CLEAN inner fits**: all six families,
two LRs, four inner folds and seeds 42/43; CLEAN R1-inner and R2-inner each use four folds and seed 42.
Prepare only the embeddings needed by these eight CLEAN fits first. Run their serial CPU fits alongside remote
S1 when authorized, then apply the unchanged G4 gate. Do not launch S2 or folds 1–4 before that decision.

Training-only input preparation is complete: 8,115 base sequences, 20,120 distinct masked sequences and eight
certified train/query units. R1-inner training counts are 7,898/7,926/7,840/7,678; R2-inner counts are
471/469/469/470; their paired query counts are 151/150/151/151. No embeddings or scientific fits have run.
The dispatcher currently runs S1 serially; price G3 with measured serial time even if the lane diagnostic passes.
Later S2/S3 and refit dispatch remain separate work after the Fold-0 decision.

The preview verifies all ten staging inputs and the exact twelve-event queue. The current same-region
configuration adds $0.30 recovery storage to the $2.6641 source-only envelope and enforces the $3 ceiling.
Strict source authentication refuses an unpublished start; the launch-time preflight must pass before billing.
The private G2 configuration and five Phase-4 request templates are preserved on Data1; the closed pass cannot
restart automatically. Phase-4 measurements and numerical G3 authority remain
explicitly pending. Pricing evidence expires after 24 hours and must be refreshed if launch occurs later.
