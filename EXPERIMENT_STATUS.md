# DeepMzyme Current Experiment Status

This is the sole concise answer to: **Where am I now, and what should I do
next?** It is mutable. Scientific policy is in [`Plan.md`](Plan.md); exact
experiment history is in the [experiment index](docs/notebook_outputs/README.md).

Last historical experiment-evidence audit: 2026-08-20. Last execution audit: 2026-09-15.
Last scientific-policy documentation update: 2026-09-24.

**2026-09-24 ion-unit correction & Zenodo exact benchmark (ACTIVE RUNNING):**
- **Contract & Architecture:** `--metal-example-unit ion` is fully implemented for standalone metal training in CLI, runner scripts, and Colab notebook, ensuring multi-nuclear sites (e.g. `1a0e` 3-Zn center) receive independent coordinates and residue microenvironments while grouping sibling ions under parent pocket IDs.
- **Dataset Fidelity:** Reconstructed from published Zenodo source rows (`classmodel_train_set.csv` and `classmodel_test_set.csv`) at **99.89% exact fidelity** (9,398 / 9,408 source rows: 7,911 train sites across 6,443 PDBs, 1,487 test sites across 1,281 PDBs).
- **Hugging Face Hosting:** Published permanently to [`GMBioinformatics/DeepMzyme`](https://huggingface.co/datasets/GMBioinformatics/DeepMzyme) (`train_and_test_sets_structures_zenodo_pmm_exact.tar.gz`, SHA-256: `24903f120462aacb90b43c4af97f4f08d61f1847d12636606fcc66e6351db296`).
- **Automated Verification:** Verified via `scripts/verify_zenodo_pmm_ion_dataset.py` (passes 100% with 0 errors).
- **1-Command Reproducibility:** Automated via `scripts/reproduce_zenodo_pmm_benchmark.sh` (downloads bundle if missing, verifies SHA-256, runs pre-flight tests, launches cross-validation, and compiles comparison table).
- **Active Colab GPU Execution:**
  - Session: `pmm-zenodo` (NVIDIA L4 GPU, 24 GB VRAM)
  - Active Target: Fold 0 of `benchmark_enhanced_only_gvp` (50 epochs, lr=3e-4, raw RBF) followed by held-out test evaluation.
  - Process: Runner PID `5303`, Training PID `5353`
  - Started: 2026-09-24 10:40:27 UTC (13:40:27 local)
  - Estimated Finish Time: 2026-09-24 11:24 UTC (14:24 local) (~42-44 minutes total: ~10m CPU graph construction + ~32m GPU training + ~2m test evaluation)
  - Log File: `/content/zenodo_single_fold_execution.log`
  - Master Guide: [`docs/ZENODO_PINMYMETAL_EXACT_ION_LEVEL_REPRODUCIBILITY.md`](docs/ZENODO_PINMYMETAL_EXACT_ION_LEVEL_REPRODUCIBILITY.md)

## Current objective

**Exact PinMyMetal 5-Fold Cross-Validation Benchmark COMPLETED (2026-09-23):** Full replication of the Nature Communications (2025) PinMyMetal 5-fold cross-validation protocol (`--train-val-split-by pocket_id --n-folds 5`) across three comparative architectures: Sequence-Only ESM-C (`benchmark_only_esm`), Structure-Only Enhanced GVP (`benchmark_enhanced_only_gvp`), and Multimodal Late Fusion (`benchmark_enhanced_gvp_esmc`).
- **Table 1 (5-Fold CV Val Bal Acc vs PinMyMetal Fig 2a: 75.08%)**:
  - `benchmark_only_esm`: **80.29% ± 3.26%** (+5.21 pp)
  - `benchmark_enhanced_only_gvp`: **74.42% ± 2.39%** (-0.66 pp)
  - `benchmark_enhanced_gvp_esmc`: **80.22% ± 3.31%** (+5.14 pp)
- **Table 2 (Held-Out Test Set 352 Pockets vs PinMyMetal Fig 2b: 67.85% & Metal3D Fig 2c: 61.70%)**:
  - `benchmark_only_esm` 5-Fold Ensemble: **77.37%** (+9.52 pp vs PMM, +15.67 pp vs Metal3D)
  - `benchmark_enhanced_only_gvp` 5-Fold Ensemble: **78.03%** (+10.18 pp vs PMM, +16.33 pp vs Metal3D)
  - `benchmark_enhanced_gvp_esmc` 5-Fold Ensemble: **79.92%** (+12.07 pp vs PMM, +18.22 pp vs Metal3D)
  - Multimodal per-class recalls: Cu: **89.7%** (+30.3 pp vs PMM 59.4%), Zn: **88.0%** (+22.1 pp vs PMM 65.9%), Group VIII: **74.6%** (+17.1 pp vs PMM 57.5%).
Complete documentation is available in [`docs/notebook_outputs/summaries/summary_pinmymetal_5fold_three_models_20260923.md`](docs/notebook_outputs/summaries/summary_pinmymetal_5fold_three_models_20260923.md) and [`docs/EXACT_PINMYMETAL_5FOLD_CV_REPRODUCIBILITY.md`](docs/EXACT_PINMYMETAL_5FOLD_CV_REPRODUCIBILITY.md).

**Next campaign implementation (2026-09-16):** the separately named
[`metal_single_gpu_20h_v2`](docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md#single-gpu-metal-campaign)
adds bounded learning-rate/capacity screening, an initial hybrid arm, one
mixed diagnostic block per family, bounded paired numeric continuation, and
protected grouped-fold confirmation. Both top screened recipes receive a
second seed; the fixed larger late-fusion pair and historical-recipe late-five
challenger retain separate identities. The revised grid is a ceiling, not a
promise that every fit fits the original allocation caps.
Its dedicated serial CLI preserves prior pilot evidence and studies. This is
an implementation/preparation change: **no new GPU allocation, training fit,
hardware readiness result or model promotion is implied**. Exact budgets,
commands and parameter values belong to the playbook. The next runtime action
is measured readiness and cost admission on the actually assigned single GPU,
after freezing inputs and durable storage. Confirmation remains exploratory
until the primary final-test route is resolved.

**Local implementation checks (2026-09-16):** 229 campaign and retained-pilot
regression tests passed. The v2 local plan is at
`DeepMzyme_Data/notebook_outputs/plans/metal_single_gpu_20h_v2_prepared/`, with 65 initial
commands and the full budget preview. CPU preparation passed for **1,389**
eligible pockets and **5,216** cached input files; all six native classes are
present in every declared training/validation fold. Its historical proxy is **21.540 hours**
for complete coverage, including the training margin and four operations hours,
before additional large-capacity cost or optional tuning. Required discovery
alone projects to **6.712 hours** against its six-hour cap. Full training is
therefore **not admitted**; zero historical reuse cells are certified. The
cost-only confirmation order is core, fixed late-five, early/hybrid, then RING.
No new GPU session or training fit was launched; a fresh server check found no
active Colab sessions. The old closed allocation identities remain unchanged.
Earlier unexecuted local previews are retained. The prepared directory binds
the final source and documentation and is the execution-plan source. No local
preview directory contains a training attempt.

**Authorized RING continuation completed (2026-09-15):** all four smokes
and sixteen full 50-epoch fits in the
[bounded matched comparison](docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md#bounded-matched-ring-continuation--stage-2b)
are verified locally and in Drive. Both complete family blocks passed their
budget gates. Terminal checkpoint/configuration/normalization binding passed;
the owned G4 session `deepmzyme-metal-ring-20260915` stopped at
**16:20:52.944619 UTC**, and Colab reported no active sessions.
The original cumulative allocation is **7.312911 hours**, leaving
**2.687089 hours of the original ten-hour cap**. The two prior closed
allocations remain unchanged; the budget was not reset.

Only-GVP's mean RING-on-minus-off BA differences are **+1.426 pp** at
`3e-5` and **+0.387 pp** at `1e-4`; all four LR/seed differences are positive,
but mean Class VIII recall falls by **3.125 / 4.688 pp** respectively.
All four graph-level late-fusion pairs have identical selected BA and class
recalls, not proven identical models or predictions. This is **Grade 3** on
one shared validation split and two training seeds, with no promotion.
The audit found annotations on existing edges, no topology expansion and
matching normalization within all trained pairs. The existing local CPU
audit was not restarted; frozen source, previous pilots and Phase-3 analysis
were preserved. No held-out evaluation or auxiliary training was added.
See the [completed RING summary](docs/notebook_outputs/summaries/summary_metal_ring_pilot_20260915.md),
[portable evidence](docs/notebook_outputs/raw/metal_ring_pilot_20260915/README.md)
and [actual stop receipt](docs/notebook_outputs/raw/metal_ring_pilot_20260915/host_closeout_allocation3/session_stopped.json).

**Phase 3 completed locally (2026-09-15):** separate retained-training
PinMyMetal and CARE panels now have native-six/common-four counts, conditional
probabilities, protein-group-weighted measures and 9,999 whole-group
permutations. PinMyMetal contributes 1,163 usable pairs from 1,136 groups;
CARE contributes 983 common-four pairs from 743 groups, or 982 native-six
pairs from 742 groups. Common-four group-weighted Cramér's V is 0.32667 and
0.50600 respectively, with no finite-sample bias correction. These are
descriptive within-source associations, not evidence of auxiliary-learning
benefit. No validation statistics, external test inputs or model training
entered this analysis. Cross-source identity/homology and shared-training
holdout certification remain required. See the
[association summary](docs/notebook_outputs/summaries/summary_metal_ec1_association_20260915.md)
and [verified portable evidence](docs/notebook_outputs/raw/metal_ec1_association_20260915/README.md).

**Authorized metal pilots completed (2026-09-15):** all 30 original architecture
fits and 15 geometry fits, each 50 epochs, plus 12 model smokes are verified
locally and in Drive. Both queues passed genuine terminal-state verification;
all 15 geometry prediction exports are verified. The owned G4 session
`deepmzyme-metal-geometry-20260915` stopped at **11:37:34.517659 UTC**, and
the server reported no active sessions. Total allocation across both sessions
was **5.350957 hours of the original ten-hour cap**, including recovery,
setup, transfer and analysis.

In the original architecture pilot, selected-LR direct-four means ± sample SD
across training seeds 42/43 are
Only-ESM **74.342 ± 3.138%**, late fusion **72.437 ± 0.177%**, Only-GVP
**72.073 ± 1.663%**, and early fusion **65.926 ± 4.866%**. These are Grade-3
results on the same validation data, with no promotion. Late-five's
common-four mean is **74.718 ± 2.486%**, an exploratory target challenger
with class tradeoffs; native and common-four results come from each
native-selected checkpoint. See the [completed continuation report](docs/notebook_outputs/summaries/summary_metal_architecture_pilot_continuation_20260915.md)
and [verified stop receipt](docs/notebook_outputs/raw/metal_architecture_pilot_20260915/continuation/closeout_allocation2/post_stop/session_stopped.json).

All geometry arms selected LR `1e-4`; increasing LR improved seed-42 BA by
7.6–10.4 percentage points. Selected-LR means across training seeds 42/43 are
A 69.641%, B 70.186%, C 70.007%, D 68.865%, and E 68.030%. B−A and C−B
advantages reverse between seeds; D−B and E−D are negative in both high-LR
seeds. This is Grade-3 repetition on the **same validation data**, with no
architecture promotion or universal rejection. E's worst Zn recall is
15.625% (5/32), illustrating the class tradeoffs hidden by aggregate BA.
The fresh A control masks only the added coordination-count/angle slots: it
retains the original GVP geometry and four base metal-site statistics
(multinuclear flag, metal count, minimum/mean intermetal distances). Its
matched explicit machinery differs from the original legacy GVP baseline.
See the [completed geometry summary](docs/notebook_outputs/summaries/summary_metal_coordination_geometry_pilot_20260915.md)
and its verified prediction/paired-error evidence.

Fitted-normalization hashes match within A/B/C and within D/E, as required.
Adding metal nodes changes representation, connectivity and fitted edge
normalization together; those contrasts do not isolate topology alone.

Recovery verified the original frozen source and manifest, all 1,389 retained
pockets (1,181 train / 208 validation), unchanged feature contents and fresh
cache timestamps. Seven original smokes and all eight A1/A2 results are now
verified, including completed late-fusion linked retry `attempt_013`.
The original interrupted `attempt_012` is reconciled as incomplete; its exact
attempt duration is unknown and was not invented. The full first allocation
of 3,827.969 seconds remains charged to the shared 600-minute cap, together
with the second allocation and recovery. The prior two-allocation closed
total was 19,263.446 seconds, leaving 16,736.554 seconds before the RING
continuation; the current three-allocation total is reported above.
The geometry implementation and recovery controls passed 134 focused tests
in 30.37 seconds; 11 separate owned-teardown tests also passed. Legacy model
outputs remain bitwise unchanged in the checked compatibility comparison.
These are implementation/readiness checks, not geometry-model results.
The largest A1/A2 balanced accuracies are late fusion **0.723121** and
Only-ESM **0.721228**, a gap of only **0.001893**. These are experimentally
evaluated single-seed results, with no established architecture winner or
promotion. Hybrid is **deferred under the budget scheduling gate**, not
rejected: best early fusion did not exceed best Only-GVP by the required
margin. The bounded target/seed matrix is complete; full hybrid comparison
and grouped-fold/paired-CI confirmation, including RING, remain absent.
The separately completed bounded RING pilot does not fill those gates.
No held-out evaluation occurred;
historical multi-seed anchors remain separately labeled.
See the [first-allocation summary](docs/notebook_outputs/summaries/summary_metal_architecture_pilot_20260915.md)
and [geometry recipe](docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md#bounded-coordination-geometry-pilot--stage-2b).

**Completed EC reference (2026-09-14):** the v12 EC1 standalone campaign
