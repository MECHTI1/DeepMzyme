# Bounded single-GPU concurrency probe

This operational diagnostic examines whether two independent GVP processes can
complete disposable optimizer steps faster together on one GPU. It does not
enable concurrent campaign fits. The normal one-worker campaign lock stays in
force, and only the coordinator may start/stop the provider resource through
the [GPU skill](../../.agents/skills/gpu-use-skill/SKILL.md).

Run the root `benchmark_pmm_gpu_concurrency.py` from the frozen scientific
checkout, using the existing audited GVP6 `prepared_recovered` input snapshot
and its original checkpoint. `--plan` verifies source, checkpoint,
normalization and every input batch on CPU and prints the plan without GPU
access or output writes. All pickle inputs must be trusted project artifacts.

Required CLI arguments are `--campaign-root`, `--prepared-dir`, `--checkpoint`
and a new `--output-dir`. Defaults are four evenly spaced complete 16-example
batches, ten warmup steps, 100 measured steps per process, two CPU threads per
process, and `--max-seconds 600` for GPU execution. Two separate serial workers
are followed by the same two workers concurrently, all using `cuda:0`. A common
start barrier and CUDA synchronization bound measured steps. Each worker resets
checkpoint weights, optimizer and its seed after warmup. The two workers use
seeds 42 and 43 consistently across serial/concurrent conditions. AdamW,
learning rate, weight decay, gradient clipping, FP32 and native loss come from
the saved supported GVP recipe. No MPS or precision changes are introduced.

The snapshot contains **development validation graphs** reused from a prior
input audit. Disposable optimizer updates on these tensors are solely a timing
diagnostic: no validation accuracy, target comparison, model selection,
scientific certification or publishable fitted model results. Original files,
caches, predictions and receipts are read-only. No normalization is refitted.
Temporary flat state vectors are compared, then deleted even on ordinary
failure; no trained checkpoint is retained or promoted. Reusing this existing
snapshot avoids rebuilding graphs or manufacturing a new scientific run.

The parent takes the campaign's actual `execution.lock` and refuses unresolved
worker/persistence state or any existing GPU compute process. Children inherit
the lock, receive Linux parent-death signals and a bounded lifetime. Parent
timeout, errors and interruption terminate/reap their process groups. At most
two child processes run; this is not a way around a production worker's lock.
Provider automatic shutdown and session budget controls remain necessary.

`plan.json`, per-worker specs/logs/results, `gpu_telemetry.csv` and `report.json`
record source/input fingerprints, actual loss finiteness, initial/final state
hashes, unchanged graph hashes, synchronized step throughput, launch-to-finish
wall time, process VRAM peaks and RSS peaks. `failure.json` records failures.
Numerical state comparison uses **diagnostic** `atol=1e-5, rtol=1e-4`; these
values never certify a fit or replace the scientific replay policy. Differences
are reported even when the timing probe finishes successfully.

Interpretation remains limited: this is one GVP pair and a small already loaded
graph subset. It excludes full graph datasets, CPU preparation, independent
replay and backup transfer. Two CPU threads per worker apply to both probe
conditions and may differ from the production default. Summed process peaks
are upper bounds, not simultaneous peak RSS/VRAM measurements. Full-training
RAM requirements can be substantially larger. Small timing gains, very short
timings, numerical divergence or resource pressure do not justify parallel
production. A positive result warrants a separately bounded representative
end-to-end check and explicit orchestration/ownership changes before adopting
two simultaneous fits. Existing single-worker execution remains the default.
