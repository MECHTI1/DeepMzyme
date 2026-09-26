# PMM ion campaign: comparator, nine smokes and first full fit — 2026-09-26

**The PMM comparator completed five PDB-grouped folds: mean fold balanced
accuracy 70.2885%, pooled out-of-fold balanced accuracy 70.0766%.** The corrected
cohort contains **7,398 ions across 3,992 PDB groups**. The completed Only-ESM,
direct-four fold-0 fit has **88.3945% validation balanced accuracy** at selected
epoch 36, with independent replay and verified backup. This is **Grade 5**
single-seed validation evidence. PMM has **Grade 2** grouped-fold evidence;
the unfinished 45-fit neural grid remains **Grade 6**. There is no model-superiority
result, promotion, final refit or held-out evaluation.

Initial snapshot: 2026-09-25 21:56 UTC (2026-09-26 local), supplemented below through
the first full fit and budget decision on 2026-09-26. Exact receipts and all five
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

## Certification and execution recovery

Certification finished at 22:41 UTC: all **7,398 ion graphs** passed, with four
empty first-shell masks and no missing retained ESM residues. It took **32m28s**,
including structure/alignment checks, parsing and graph construction; peak main
process RSS was **5.23 GB**. The
[certificate summary](../raw/pmm_ion_v2_context_20260926/runtime/feature_certification_summary.json)
records the full inventory hash. After the interrupted agent session, the
existing L4 VM was recovered at 23:55 UTC and the unmounted workstation data
volume was mounted again. No embeddings or certification were regenerated.
All **15,339 input and receipt files** then passed host-side hash verification.

The remaining CPU graph work justified an optional raw-graph cache before the
first neural smoke. It binds complete input values, target scheme, graph options,
source and library versions; normalization still fits only training graphs.
The final source is
`adc95c42261448dc9d35572a138a3b8349de79124d9708618f511c27a848dd23`.
All **114 campaign regressions and 23 cache tests passed**, plus the existing
tiny training/replay regression repeated with caching enabled. These engineering
checks do not complete the 45-fit neural grid.

At 00:17 UTC, all **nine one-epoch GPU smoke cases** passed independent
checkpoint replay and verified host backup. Repeated GVP graph preparation
used 151/151 cached entries and fell from 19.47 seconds to 0.72 seconds; the
matching late-fusion case also reused the ESM-backed graphs. Cold/warm smoke
unit times were roughly 46–47/19–20 seconds, excluding host transfer.
Independent replay keeps a second cache copy because its unused EC target is
unset while the normal loader assigns the compatibility placeholder. The full
cache key preserves that difference; verified metal predictions are unchanged.
The [smoke receipt](../raw/pmm_ion_v2_context_20260926/runtime/smoke_validation_summary.json)
contains identities, selected-checkpoint hashes and measured profiles, not
evidence for selecting between model families. The first full 50-epoch Only-ESM
fit was then admitted with a 40-minute initial forecast, 25% forecast margin,
and 15-minute closeout reserve inside the original provider deadline.

## First full fit and measured continuation cost

The `only_esm__four_class__none__fold0__seed42` unit completed all 50 epochs and
independent validation replay in **2,417 seconds (40m17s)**. Its 71-file host
backup was verified and acknowledged at 01:00 UTC. The fit used **5,906 training
and 1,492 validation ions**. Validation selected epoch **36**, with common-four
balanced accuracy **88.3945%**, accuracy **84.7185%**, and recalls Mn **85.3659%**,
Cu **97.4359%**, Zn **89.5028%**, Class VIII **81.2734%**. These are one-fold
development results; comparison with PMM or other neural families awaits the
complete matched grid and predeclared paired analysis.

The [full-fit receipt](../raw/pmm_ion_v2_context_20260926/runtime/first_full_fit_summary.json)
binds the source, cohort, folds, checkpoint, 50-epoch history, independent replay
and host acknowledgment. Preparation took **1,305.36 seconds**; training,
train-metric evaluation, validation, checkpoint saves and selected export took
**648.28 seconds**. Peak main-process RSS was **7.76 GB**, with CUDA allocator
peaks of **30.62 MB allocated / 54.53 MB reserved**. The separate replay rebuilt
its validation graphs in **214.48 seconds** after roughly four minutes of serial
structure parsing. CUDA peaks exclude driver/context memory.

After the full fit and host acknowledgment, two bounded operational probes used
copies of smoke checkpoints and one representative 16-graph batch (602 nodes,
9,832 edges): five warmup batches and twenty timed batches. GVP averaged
**48.69 ms/batch** and late fusion **53.61 ms/batch**, with allocated GPU peaks
of **280.94/282.46 MB**. The copied checkpoint hashes remained unchanged; no
probe weights or comparison result were saved. These measurements include
collation/transfer but do not establish full-fit throughput. The
[probe script and receipts](../raw/pmm_ion_v2_context_20260926/README.md)
make their scope explicit.

GPU correctness is established by real ESMC and neural execution. Utilization
is limited here by CPU preparation and small batches over frozen embeddings.
The tested cache materially reduces repeated graph work, but serial replay
parsing remains a recurring cost. About **74 minutes** between certification
completion and recovery were interrupted-session overhead, not training time;
that allocation time remains charged. Completed work was recovered, not rerun.

The [remaining-budget forecast](../raw/pmm_ion_v2_context_20260926/runtime/remaining_budget_forecast.json)
estimates **22.3 additional VM hours before contingency** for 44 remaining fits
and final refits. Warm loading, evaluation cost and the PMM refit include explicit
proxies/allowances; reference-report preparation remains unmeasured. The user
was asked to close out or approve a **30 total VM-hour / $34 gross** continuation
ceiling, about 19% time reserve relative to the estimate at the decision point.
That proposal is **not approved** and does not guarantee full completion.
Existing four-hour session and six-hour daily limits remain unchanged, and
future VM starts require authorization. No scientific arm, fold or epoch budget
was reduced. The original session was stopped after artifact backup rather than
left idle pending a budget reply. The controller recorded **TERMINATED at
01:06:50 UTC**, **12,878 running seconds** and **$3.145 estimated gross**;
the [closeout receipt](../raw/pmm_ion_v2_context_20260926/runtime/session_closeout.json)
preserves that accounting. The disk and caches remain available for an authorized
continuation.

## Approved smaller screen and capacity failure

The user subsequently approved **ESMC pair first**: reuse ordinary Only-ESM,
direct-four, fold 0 and add only the matching `first_shell_bias` fit. The same
fold, model seed, 50-epoch recipe and frozen inputs/source remain fixed. This
supersedes immediate full-grid execution; the 30-hour/$34 proposal remains
unapproved. The written plan and playbook now specify the bounded single-fold
screen and its paired exploratory report, with no model promotion or test access.

The controller accepted a requested one-hour allocation (57-minute provider
limit, maximum estimated running cost $0.8352), but Google rejected the start
at **01:27 UTC** because L4 capacity was unavailable in `us-central1-a`.
The user then authorized one same-VM retry; Google rejected it at **02:12 UTC**
for the same L4 stockout.
Subsequent cloud status confirmed **TERMINATED**. There was no new running
session, fit or result. The ordinary baseline's completion/replay/source were
reverified without retraining. The
[execution receipt](../raw/pmm_ion_v2_context_20260926/runtime/esm_binding_screen_execution.json)
records the attempts. A later explicit resumption request authorized a third
start at **02:52:20 UTC**, rejected at **02:52:42 UTC**. Google suggested
`us-central1-c`; that is a transient capacity hint, not a guarantee. Status at
**02:53:12 UTC** again confirmed **TERMINATED**.

The user questioned the alternate-zone block. An offline call to the actual
controller guard reproduced its refusal for another managed VM in TERMINATED
state. This is a local controller limitation, not a Google restriction caused
by the preserved disk. The controller's budget also assumes one boot disk, so
removing only the duplicate check would be insufficient. A deliberate fallback
preserving old data, allowing only one running GPU and counting extra storage
was proposed at that boundary. The implementation and live verification below
supersede that earlier operational block.

## GPU recovery verified; fit admission refused

The approved controller recovery now enforces one active GPU while allowing a
positively stopped source to coexist temporarily. It snapshots/restores the
source boot disk, bounds destination attempts, verifies provider automatic STOP
before selecting SSH, and counts overlapping disks, snapshots and recycle-bin
storage. Both GPU skills and the related runbooks were updated. The initial
offline regression suite passed **63 tests**. A subsequent focused suite passed
**31 fallback tests**, including nine new cases for evidence from a later
explicitly authorized session on the same immutable replacement. Original
allocation history remains unchanged; unknown/wrong-instance sessions and
completion outside execution or stop bounds are rejected.

Recovery pass `e9ff994df82d` succeeded on its first destination,
`us-central1-c`, at **2026-09-26 03:42:02 UTC**. The restored NVIDIA L4 executed
a real CUDA matrix operation; the scientific source hash remained
`adc95c42261448dc9d35572a138a3b8349de79124d9708618f511c27a848dd23`.
The existing runner verified frozen inputs and accepted the completed ordinary
baseline's host acknowledgment. It did not repeat feature generation or smokes.

At **03:47:25 UTC**, admission refused the new binding-aware fit: its
1,800-second forecast × 1.25 plus a 900-second closeout reserve required
**3,150 seconds**, while **3,090 seconds** remained before automatic STOP.
Startup, connection, coordinator delay and verification consumed the available
setup allowance. Neither the forecast nor safety margins were lowered to force
admission. **No new fit started and no new scientific result exists.**

The coordinator stopped the replacement; the controller recorded
**TERMINATED at 03:49:17 UTC** (cloud stop 03:49:13),
about **7m12s / $0.11 estimated running gross**.
The stopped source and replacement disks plus temporary snapshot remain
preserved and bill for storage. The fit/replay/independent-backup cleanup gate
has not passed. A 1.5-hour total screen allocation ceiling, including this
session, was requested; it is not authorized without a user reply. No restart
was performed. The [recovery receipt](../raw/pmm_ion_v2_context_20260926/runtime/gpu_fallback_execution.json)
and admission log preserve this boundary; current status owns later decisions.
Colab authentication is valid and its session list is empty. Its installed CLI
does not expose the CU balance/rate, browser access was unavailable, and no
Colab allocation was attempted. The scientific comparison remains one completed
arm until the single new fit can run.
