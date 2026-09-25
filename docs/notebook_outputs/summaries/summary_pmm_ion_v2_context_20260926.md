# PMM ion campaign: context correction, comparator and GPU preflight — 2026-09-26

**The PMM comparator completed five PDB-grouped folds: mean fold balanced
accuracy 70.2885%, pooled out-of-fold balanced accuracy 70.0766%.** The corrected
cohort contains **7,398 ions across 3,992 PDB groups**. This snapshot contains no
neural-model fits, model-superiority result, promotion, final refit or held-out
evaluation. PMM has **Grade 2** grouped-fold evidence; the unfinished neural grid
and operational preflight remain **Grade 6**.

Snapshot: 2026-09-25 21:56 UTC (2026-09-26 local). Exact receipts and all five
validation prediction files are in the [portable evidence](../raw/pmm_ion_v2_context_20260926/README.md).
Feature-generation completion and implementation checks were added at 22:24 UTC below.
[Current status](../../../EXPERIMENT_STATUS.md) owns subsequent GPU outcomes and
remaining execution gates; the [campaign plan](../../plans/metal_level_metal_task_compared_PMM_final_plan.md)
owns the intended comparison.

## Cohort and scientific scope

The training-only audit excluded **503 ions with explicit protein symmetry
links** whose partner protein coordinates were absent from the declared input
context. The versioned `pmm_ion_metal_v2_context` cohort preserves v1 as historical
evidence. Nine source rows lack reconstruction and ten have unsupported target
residues; the 7,920-row source therefore yields 7,398 retained ions.

The declared neural input uses the first coordinate model, all protein chains
within a 10 Å pocket, and no crystallographic symmetry expansion. Its bounded
audit found no remaining explicit target-to-protein symmetry links, model
ambiguities or ungrouped exact-content aliases. Thirteen retained ions have
nonprotein symmetry links. This certifies the declared input contract, **not
historical source/paper assembly parity**: absent LINK annotations do not prove
absent symmetry contacts. PMM retains its released features. The comparison is
classification at known metal sites on matched ions/folds, not a reconstruction
of every published input or a site-detection benchmark.

Native counts are Mn 2,528, Cu 343, Zn 2,032, Fe 1,737, Co 283 and Ni 475. The
common endpoint is Mn, Cu, Zn and Class VIII = Fe+Co+Ni. All five training and
validation partitions contain all six native elements; whole PDB groups stay
within a validation fold.

## Completed PMM comparator

Each fold freshly fit the released soft-voting LR, BalancedRF, MLP, SVC and
EasyEnsemble recipe, with `random_state=1`, 93 released features and no scaling.
Resampling stayed inside each training fold. The source's absent `source` column
was omitted from the drop list; `dropna()` removed no rows. The campaign pins
Python 3.11.5, scikit-learn 1.3.0, NumPy 1.23.5, pandas 2.1.4, joblib 1.2.0 and
imbalanced-learn 0.11.0; the last is an explicit campaign pin where the released
environment left it unpinned.

All 7,398 prediction rows have verified identities, folds, labels, probabilities
and file hashes. Their fivefold mean and pooled metrics above are different
aggregations of the same predictions. Total recorded fit time was 565.79 seconds
on local CPU. Released-recipe convergence warnings remain in the copied log;
parameters were not tuned in response. There are no repeated-seed estimates or
paired neural-versus-PMM confidence intervals yet.

## Measured GPU readiness and pending work

The real NVIDIA L4 preflight completed ESMC inference for sequence lengths 128,
512 and 3,715 using PyTorch 2.11.0+cu128/CUDA 12.8. Parameters were bfloat16 and
emitted embeddings float32. Model loading took 29.91 seconds; sample calls took
0.064–0.245 seconds. Maximum observed allocated/reserved GPU memory was
1.82/5.29 GB. An earlier check rejected the absent exact `sm_89` architecture
string; the successful receipt records compatible `sm_80`/`sm_86` cubins and
actual execution. The failed check is preserved separately.

These are bounded inference measurements, not complete feature generation or a
training-throughput benchmark. The ESM preparation plan contains 7,664 chain
files, 4,972 unique sequences and an estimated 10.36 GB of float32 output. Full
feature certification, neural smokes/timing, the matched validation grid,
selection, and both final refits remain pending at this snapshot. No primary
final-test claim is supported.

One coordinator owns GPU execution. Three bounded agents handled independent
scientific/cohort, training/integrity and GPU/runtime audits and tests; additional
workers are used only when an independent task warrants them. CPU comparator
work remained separate from GPU preparation. Source/setup implementation and
passing engineering checks do not constitute completed model experiments.

## Feature generation completed; training gate still pending

At 22:08 UTC, generation completed all **7,664 chain payloads**, inferring each
of the **4,972 unique sequences** once. Outputs preserve FP32 precision. Total
process wall time was **860.91 seconds**; the internal inference/writing phase
was **308.54 seconds**, so structure/sequence preparation and process overhead
accounted for roughly 552 seconds. Peak process RSS was **8.95 GB**. These times
do not include the separate full certification or host backup.

The exact ESMC repository revision and weight SHA-256 are preserved in
[`esmc_checkpoint_identity.json`](../raw/pmm_ion_v2_context_20260926/runtime/esmc_checkpoint_identity.json).
Before any neural fit, the source was revised to remove redundant per-fit
semantic parsing: admission still rehashes structures, plans, embeddings and
sidecars and checks exact coverage; full certification remains mandatory.
All **114 targeted tests passed**. The source-revision receipt distinguishes
the generation snapshot from the certification/training snapshot.

At this supplement's boundary, full graph certification and the verified
workstation backup were still in progress; no neural grid fit had started.
