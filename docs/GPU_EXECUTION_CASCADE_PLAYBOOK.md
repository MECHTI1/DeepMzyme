# GPU execution routing for the PMM ion campaign

This guide chooses an execution route. The [PMM ion campaign recipe](METAL_TRAINING_PIPELINE_PLAYBOOK.md#pmm-ion-level-metal-comparison-campaign-pmm_ion_metal_v2_context)
owns scientific configuration and exact commands. The [GCP runbook](GCP_GPU_RUNBOOK.md)
and [Colab runbook](COLAB_GPU_RUNBOOK.md) own runtime procedures;
[`EXPERIMENT_STATUS.md`](../EXPERIMENT_STATUS.md) owns current progress.

## One operator, a bounded search

Only the coordinating operator allocates, starts, submits work to, or stops a GPU.
Other agents can prepare inputs, review code and analyze completed development
artifacts independently. Begin with one allocation and one training process.

1. Finish local contract tests and freeze the source snapshot, training-only
   bundle and output destination before allocating. Preserve current controller
   caps and recorded user authorization. Earlier explicit authorization remains
   usable within its scope; do not ask for it again.
2. Through `~/deepmzyme-vm/bin/*`, check ownership, gross price, quota, machine
   availability, subnet and provider hard-stop settings. Try at most three
   eligible L4 candidates: `us-central1-a`, `us-east4-a`, then `us-west1-a`.
   Virginia uses different verified SKUs; the GCP runbook lists the mapping.
   Allow at most ten minutes for starting new attempts. Check that deadline
   between attempts; reconcile an in-flight request before considering another.
3. Continue only for confirmed capacity exhaustion. Quota, authentication,
   price, lock or configuration failures need diagnosis; do not label them
   stockouts or continue through a controller refusal. Stop after success.
   Restore the exact saved controller configuration if all attempts fail or the
   search is abandoned. Keep the successful configuration while its VM exists
   so status and stop commands address the right resource.
4. If this pass cannot provide a VM, check the existing Colab entitlement and
   compute units, then request one explicitly named L4 runtime through the
   existing Colab interface. Use A100 only as the reviewed fallback when L4 is
   unavailable or the measured memory envelope requires it. Verify the actual
   assigned device and rate. Never silently substitute hardware or allocate a
   second runtime after a transport failure.

The retired `scripts/gpu_provisioning_cascade.py` supplied its own authorization,
modified configuration even on dry-run, and duplicated lifecycle decisions.
The existing controller is the only GCP lifecycle implementation.

## Hardware decisions use measurements

The installed ESMC SDK uses BF16 on CUDA. L4 and A100 support that inference
path; test ESMC-600M on representative training sequences, including the longest,
before generating all embeddings. More VRAM does not guarantee freedom from OOM.
The cached-feature campaign currently trains in FP32 unless its frozen
configuration explicitly says otherwise. T4/V100 are not categorically invalid
for that training workload, but changing the BF16 embedding path or machine
needs compatibility and cost evidence. Do not change scientific batch sizes
or precision silently.

L4 BF16 and memory specifications are documented by [NVIDIA](https://www.nvidia.com/en-au/data-center/l4/),
and Ampere BF16 support by [NVIDIA's tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/contents.html).
Quota is distinct from changing hardware capacity ([Google troubleshooting](https://docs.cloud.google.com/compute/docs/troubleshooting/troubleshooting-vm-creation)).
Paid Colab also has variable hardware availability and runtime limits
([Colab FAQ](https://research.google.com/colaboratory/faq.html)).

## Prepare, measure, persist, close

- Stage inputs and caches on runtime-local disk. Verify the mounted or
  acknowledged host-pull persistence route before unattended work. Synchronize generated
  embeddings and certify their content before freezing the feature inventory.
- Run the nine-configuration training-only smoke, then one complete canonical
  fit per family. Measure preparation, graph construction, training, validation,
  checkpoint/export and transfer separately, with CPU RSS and peak GPU memory.
  A single `nvidia-smi` utilization sample is not an efficiency gate.
- Forecast each next complete unit with a 1.25 multiplier and a 15-minute
  artifact/closeout reserve against actual remaining allocation time. Include
  setup/idle time from the real allocation start. Stop admission when it no
  longer fits; preserve the scientific grid and obtain a recorded budget
  decision only if the existing allowance must increase.
- The existing loop uses `benchmarking.pmm_execution` for exclusive ownership,
  admission, stop-on-failure and independently read-back artifact hashes.
  Persist each terminal unit's checkpoint, predictions, command, log and state
  before another launch. GCP can use the runbook's host-pull acknowledgment
  without mounting a new filesystem. Streaming supplements that boundary; a polling
  interval is not a guarantee of zero data loss.
- Preserve failed attempt logs and stop the queue on the first failed unit.
  Reuse only verified completed fits. Interrupted fits restart from their
  original seed; exact optimizer/RNG/epoch continuation is not implemented.
- On success, exception or exhausted allowance, verify durable artifacts, stop
  the owned provider resource and confirm provider state. Worker exit or a
  closed local receipt alone does not prove that billing stopped.

Reuse existing parse/feature caches. Add raw-graph caching or tune DataLoader
workers only after timing identifies a material bottleneck. Keep each fold's
normalization fitted to its training partition. Consider two disjoint training
processes only after serial measurements demonstrate improved completed-fit
throughput and safe combined host/GPU memory use.
