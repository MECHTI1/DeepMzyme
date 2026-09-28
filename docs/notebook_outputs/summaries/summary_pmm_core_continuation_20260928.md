# PMM core readiness and single-GPU concurrency — 2026-09-28

Subsequent user decision: retain the existing fold-0 results and defer the
36 remaining fits for future work. The budget proposal below remains an
unapproved historical proposal; no immediate continuation decision is pending.
The implementation/readiness snapshot is preserved, with fresh verification
required after an explicit resume. Current status is in `EXPERIMENT_STATUS.md`.

All nine trained core fold-0 fits now qualify under the explicitly retrospective
`pmm-core-replay-v1` integration. Seven retain original strict replay passes;
GVP5 and GVP6 retain their original strict failures. No fit or checkpoint was
replaced. The fivefold grid remains **9/45 trained, 36 untrained**.

The root continuation runner, separate core assessor and guarded final-refit
preview are implemented. **195 CPU tests passed**, followed by **42 passing
runner tests** after the scope queue was updated to preserve all fold-0 fits.
The actual host readiness action verified all nine units, their complete
replay/diagnostic evidence, source/configuration identities and frozen inputs.
This is implementation/readiness evidence, not a completed fivefold experiment.

## GVP5 diagnosis and unchanged screen results

Fresh reconstruction verified all predictive graph tensors for 1,492 validation
ions in 94 batches against checksummed original-window cache contents. The only
raw mismatch was the already documented inactive EC target. Its exclusion used
the unchanged v1.1 recovery audit and exact CPU logits/loss equivalence on the
first complete batch. Historical cache matching remains retrospective evidence.

Twenty predeclared evaluations completed: two fresh processes with five passes
each for original and strict deterministic settings. All native/common-four
classes remained unchanged. Original-setting passes differed by up to
`2.622604e-6`; ten strict passes were bitwise identical across both processes.
The largest difference against preserved exports was `3.515757e-6`. This supports
numerical variation on fixed inputs, without rewriting the original `1e-6`
failure. The separately disclosed core bound is absolute `1e-5`, relative zero,
with exact classes and reconciled metrics, fixed before any future fold fit.

Common-four balanced accuracy on the same fold-0/seed-42 validation ions:

| Family | Four-class training | Five-class training | Six-class training | Highest on this fold |
|---|---:|---:|---:|---|
| Only-ESMC | 88.3945% | 89.4354% | 85.8393% | Five |
| Only-GVP | 86.0232% | 78.2397% | 82.8399% | Four |
| GVP + ESMC, graph-level late fusion | 89.4653% | 87.6608% | 88.8124% | Four |

All values reuse each fit's native-BA-selected checkpoint. The five/six GVP
qualifications above remain attached. This single-fold table does not select
a final formulation or establish fivefold superiority. No held-out data was
accessed; no final refit or promotion occurred.

## Two processes on one L4

The disposable probe reused four audited GVP6 batches, batch size 16, FP32,
AdamW and two CPU threads per worker. After ten warmup steps and a reset,
each worker performed 100 measured steps. Two serial workers were compared
with the same seeds/initial weights in two concurrent processes on the same L4.

| Timing | Serial pair | Concurrent pair |
|---|---:|---:|
| Measured step window | 5.8619 seconds total | 3.8617 seconds maximum worker time |
| Aggregate throughput | 34.12 steps/second | 51.79 steps/second |

Measured throughput improved **1.518×** (about **34.1% less elapsed step time**).
Launch-to-finish speedup was 1.823×. Summed per-worker peak reserved VRAM was
828 MiB; summed peak RSS was about 3.63 GiB. These are upper bounds from a small
snapshot, not measurements of two complete resident training datasets.

Final weights were not bitwise equal, and both seed comparisons exceeded the
probe's diagnostic `atol=1e-5, rtol=1e-4` test: maximum absolute differences
`5.6773e-5` and `1.6849e-4`. Original input/checkpoint hashes stayed unchanged;
disposable state vectors were deleted. There was no serial-versus-serial
repeat control, so these differences cannot be attributed specifically to
concurrency rather than ordinary nondeterministic training.

This short, single-pair microbenchmark excludes full data preparation, replay
and backup costs. It is promising throughput evidence, not end-to-end speedup
or accuracy equivalence. Keep production at **one training worker per GPU**.
Any later adoption needs a bounded representative check including serial
repeat variability, full-dataset RAM/CPU pressure and independent replay, then
explicit orchestration changes. No MPS or scientific recipe change was made.

## Closeout and remaining boundary

All 72 remote files were independently copied and hash-verified. The coordinator
also consumed the previous late-fusion backup acknowledgment under the campaign
lock before the probe. The controller recorded **TERMINATED at 06:00:08 UTC**;
an independent status check confirmed it at **06:00:18 UTC**. Session
`session-20260928T053830Z-efe0130c` used **1,292 seconds / $0.3156 estimated gross**.
Historical running use is approximately **11.8358 hours / $10.4057**, with
additional storage. One 150-GB disk remains, approximately $15/month.

The [serial forecast](../raw/pmm_core_continuation_20260928/remaining_budget_forecast.json)
estimates 21–33 additional VM hours for the 36 missing fits, including persistence,
forecasting cushion and per-allocation closeout. Parallel savings are not assumed.
The proposed **34 additional hours / $34 additional gross ceiling** is pending
user approval. Existing four-hour/$6 session and six-hour/$10 UTC-day limits
remain unchanged. Final refits and held-out testing are outside that proposal.
The previous 30-hour/$34 total proposal also remains unapproved.

See the [portable evidence and canonical paths](../raw/pmm_core_continuation_20260928/README.md)
and [execution contract](../../plans/pmm_core_execution_v1.md). Recompute readiness
in the execution environment, synchronize the historical diagnostic index and
admit only absent fold-1–4 units after budget approval. Reuse all completed fits,
embeddings, caches, PMM folds and diagnostics.
