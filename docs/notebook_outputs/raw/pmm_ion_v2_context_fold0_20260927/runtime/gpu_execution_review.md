# GPU execution review: frozen fold-0 continuation

The existing L4 runs the frozen CUDA training and checkpoint replay correctly.
Its already generated ESMC embeddings are reused; these runs train the declared
readout/GVP models, not the ESMC foundation model. The source identity, split,
50 epochs, batch settings and seed remain fixed. Successful independent replay
and host-backup receipts are required before each next submission.

Measured profile examples (seconds; see each completed unit's
`runtime_profile.json` and the earlier ESMC-pair evidence):

| Unit | Preparation | Training epochs | Validation | Checkpoint saves |
|---|---:|---:|---:|---:|
| Ordinary ESMC, cold graphs | 1305.4 | 486.4 | 108.1 | 6.1 |
| Binding-aware ESMC, warm graphs | 102.0 | 507.8 | 112.1 | 6.1 |
| Ordinary GVP, cold graphs | 1294.5 | 986.2 | 156.0 | 8.9 |
| Ordinary late fusion, warm graphs | 107.7 | 1026.0 | 157.9 | 8.1 |
| Binding-aware GVP, warm graphs | 85.4 | 961.3 | 152.1 | 8.9 |

Raw graph-cache reuse removes most repeated preparation cost. The GVP pair's
preparation differs by about 20.2 minutes; its overall runtime difference must
not be attributed to binding awareness because the cache states differ.
Training-fold normalization is still fitted separately; raw-cache reuse does
not reuse validation-derived normalization. Existing feature certification,
embeddings, smokes, PMM folds and completed neural units are not rerun.

Independent replay still spends several minutes parsing structures on CPU
before evaluating the selected checkpoint. A temporarily idle GPU during that
step is expected. Checkpoint writing is only seconds per fit and is not the
main measured bottleneck. Reported phase times omit some orchestration/I/O and
do not synchronize CUDA; they are approximate wall-clock components, not
GPU-kernel benchmarks. Admission forecasts also include replay and transfer.

Peak allocated CUDA memory is approximately 29 MiB for the ESMC readout and
327–330 MiB for GVP/late fusion. These are PyTorch allocator measurements,
not total device usage or GPU utilization percentages. They do not justify
claims that a different GPU would be faster or cheaper. This campaign does not
change hardware, batch sizes, precision or scientific code mid-comparison.

One coordinator owns the existing VM, serial job submission and shutdown.
One focused read-only reviewer checked scientific/reporting contracts; it did
not allocate resources or submit another worker. Unique per-unit status tags
avoid mutating status files bound into earlier backup manifests. Each fit keeps
the 1.25 admission multiplier and 900-second persistence/closeout reserve;
the provider's fixed STOP deadline is recorded in `execution.json`.

The larger budget proposal remains separate from the current allocation.
Final resource state and actual session accounting belong in the closeout
receipt and the project's `EXPERIMENT_STATUS.md`, not this timing comparison.
