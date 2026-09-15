# Matched RING pilot — 2026-09-15

Experiment ID: `metal/nonoverlap/ring-pilot/2026-09-15`.
Profile: `metal_ring_pilot_v1`.

## Outcome and evidence level

All **four one-epoch smokes and sixteen full 50-epoch fits** completed.
Every attempt, selected-checkpoint binding, terminal capture and actual GPU
teardown was verified. No model is promoted and no held-out evaluation occurred.

RING increased selected validation balanced accuracy (BA) in all four
Only-GVP LR/seed pairs, by **0.024–1.812 percentage points**, with class
tradeoffs. All four graph-level late-fusion pairs have **identical selected
BA and per-class recalls** between RING off and on. Equal selected metrics
do not establish identical models or site predictions.

This is **Grade 3: two model seeds on one shared validation partition**.
There are no grouped-fold confidence intervals, new-data confirmation,
Stage 6 promotion, target-formulation conclusion or auxiliary-learning claim.
Both LRs are retained; no LR was dropped after observing results.

## Matched recipe and retained cohort

The [bounded Stage 2B recipe](../../METAL_TRAINING_PIPELINE_PLAYBOOK.md#bounded-matched-ring-continuation--stage-2b)
owns executable parameters. The [frozen manifest](../raw/metal_ring_pilot_20260915/finalization/campaign_manifest.json)
and [commands](../raw/metal_ring_pilot_20260915/finalization/commands.txt)
record what ran.

- Only-GVP and GVP + graph-level late fusion each receive RING off/on ×
  LR `3e-5`/`1e-4` × model seeds 42/43, with 50 epochs per fit.
  All controls are fresh runs; no historical baseline fills a matrix cell.
- Direct `four_class`, canonical `merge_fe_class_viii`: Mn, Cu, Zn and
  Class VIII = Fe+Co+Ni. Native-six eligibility preserves the parent cohort.
- Only non-overlap PinMyMetal external-training membership is used:
  1,181 training pockets / 1,151 PDB groups and 208 validation pockets /
  110 PDB groups. Split by `pdbid`, fraction 0.15, split seed 42,
  `metal_site` strata; no training/validation PDB overlap.
- Conservative features, certified training-only PROPKA overlay, residue
  radius 6 Å, extraction radius 10 Å, pooling cutoff 0, residue-only readout,
  no explicit metal nodes, legacy site geometry and no augmentation.
- Batch 8, weight decay `1e-4`, fixed learning rate and inverse-frequency
  cross entropy; no label smoothing or collapsed auxiliary loss.
- Native `val_metal_balanced_acc` selects each checkpoint independently
  within its 50-epoch history. No held-out score enters selection.
- `--shell-role-source geometry` in both arms freezes node shell annotations.
  RING on uses `--use-ring-edges --require-ring-edges`; generation of
  missing RING files is disabled in both arms.

## What changed in the inputs

The [whole-cohort audit](../raw/metal_ring_pilot_20260915/finalization/ring_input_audit.json)
passed, including identical node/site tensors and unchanged shared-radius
geometry. Every retained pocket has RING annotations:

| Partition | Pockets | Annotated existing undirected pairs | Added undirected pairs |
|---|---:|---:|---:|
| Training | 1,181 | 34,573 | 0 |
| Validation | 208 | 6,057 | 0 |

The general recipe tests RING edges/features with independently training-fitted
normalization; it is not a topology-only recipe. **The observed intervention
on this cohort is annotations on existing edges, with no topology expansion
and no on/off normalization difference.** Raw RING `Angle` is not consumed;
this experiment adds neither coordination-angle summaries nor metal nodes.

All sixteen full-fit normalization objects share SHA-256
`e541b2eea3b00528684e80457706d292da00c17fbe5953d6b1a5f37dbeb6f2be`.
The remote readiness audit also has matching off/on normalization, but its
hash differs from training: two non-edge means differ by at most
1.49×10⁻⁸. Local CPU versus remote readiness normalization also differs
slightly (three non-edge fields, at most 1.91×10⁻⁶). Do not claim bitwise
normalization equality across machines or all preparation modes.

The [cache audit](../raw/metal_ring_pilot_20260915/finalization/training_cache_audit.json)
verifies all 5,216 pinned feature records: 1,304 RING files and 3,912
ESM/external-feature/sidecar records. Both late-fusion RING modes passed
actual-data CUDA forward/backward readiness before training.

## Complete paired BA results

BA values are percentages; differences are percentage points (on minus off).
Mean ± SD uses sample SD across model seeds 42/43 on the same validation
proteins, **not** a confidence interval.

| Family | LR | Off mean ± SD (%) | On mean ± SD (%) | Δ seed 42 (pp) | Δ seed 43 (pp) | Mean Δ (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Only-GVP | 3e-5 | 67.609 ± 4.098 | 69.034 ± 3.551 | +1.812 | +1.039 | +1.426 |
| Only-GVP | 1e-4 | 72.073 ± 1.663 | 72.459 ± 2.176 | +0.024 | +0.749 | +0.387 |
| Graph-level late | 3e-5 | 73.538 ± 2.475 | 73.538 ± 2.475 | 0.000 | 0.000 | 0.000 |
| Graph-level late | 1e-4 | 72.437 ± 0.177 | 72.437 ± 0.177 | 0.000 | 0.000 | 0.000 |

No aggregate across the two LRs is used as an architecture score. The new
low-LR late-fusion seed-43 opportunity was not present in the original
selected-LR-repeat pilot; do not use unmatched historical maxima or differing
search opportunities to declare a family winner.

### Selected epochs and individual scores

| Family | LR | Seed | Selected epoch off / on | Off BA (%) | On BA (%) | Δ (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Only-GVP | 3e-5 | 42 | 44 / 44 | 64.711 | 66.524 | +1.812 |
| Only-GVP | 3e-5 | 43 | 11 / 46 | 70.506 | 71.545 | +1.039 |
| Only-GVP | 1e-4 | 42 | 35 / 30 | 70.897 | 70.921 | +0.024 |
| Only-GVP | 1e-4 | 43 | 23 / 18 | 73.249 | 73.998 | +0.749 |
| Graph-level late | 3e-5 | 42 | 17 / 17 | 71.789 | 71.789 | 0.000 |
| Graph-level late | 3e-5 | 43 | 16 / 16 | 75.288 | 75.288 | 0.000 |
| Graph-level late | 1e-4 | 42 | 7 / 7 | 72.312 | 72.312 | 0.000 |
| Graph-level late | 1e-4 | 43 | 8 / 8 | 72.562 | 72.562 | 0.000 |

These are native-BA-selected checkpoints, not final-epoch scores.
Unrounded metrics, sample SDs, paired deltas and input hashes are in the
[post-stop JSON](../raw/metal_ring_pilot_20260915/analysis/ring_post_stop_analysis.json)
and [CSV](../raw/metal_ring_pilot_20260915/analysis/ring_post_stop_analysis.csv).

## Class recalls and rare-class protection

Cells show correctly classified sites **off → on / class support** at the
selected checkpoint. Divide by support for recall; supports are identical
in every fit. These are marginal counts, not case-level transitions.

| Family | LR | Seed | Mn | Cu | Zn | Class VIII |
|---|---:|---:|---:|---:|---:|---:|
| Only-GVP | 3e-5 | 42 | 59 → 63 / 97 | 14 → 14 / 15 | 13 → 14 / 32 | 41 → 41 / 64 |
| Only-GVP | 3e-5 | 43 | 83 → 84 / 97 | 14 → 14 / 15 | 10 → 13 / 32 | 46 → 42 / 64 |
| Only-GVP | 1e-4 | 42 | 83 → 74 / 97 | 14 → 14 / 15 | 11 → 15 / 32 | 45 → 43 / 64 |
| Only-GVP | 1e-4 | 43 | 80 → 92 / 97 | 14 → 14 / 15 | 15 → 14 / 32 | 45 → 41 / 64 |
| Graph-level late | 3e-5 | 42 | 71 → 71 / 97 | 11 → 11 / 15 | 27 → 27 / 32 | 36 → 36 / 64 |
| Graph-level late | 3e-5 | 43 | 77 → 77 / 97 | 11 → 11 / 15 | 27 → 27 / 32 | 41 → 41 / 64 |
| Graph-level late | 1e-4 | 42 | 70 → 70 / 97 | 11 → 11 / 15 | 28 → 28 / 32 | 36 → 36 / 64 |
| Graph-level late | 1e-4 | 43 | 74 → 74 / 97 | 11 → 11 / 15 | 24 → 24 / 32 | 42 → 42 / 64 |

Only-GVP's mean Class VIII recall falls by **3.125 pp** at `3e-5` and
**4.688 pp** at `1e-4`. Cu remains 14/15 in all GVP fits. At `1e-4`,
seed 42's tiny BA gain accompanies Mn −9, Zn +4 and VIII −2 correctly
classified sites; seed 43 has Mn +12, Zn −1 and VIII −4. Its minimum class
recall falls from 46.875% to 43.750%. Aggregate BA gains therefore do not
establish rare-class protection or justify promotion.

All four late-fusion pairs have zero selected class-recall differences.
Their training trajectories need not match: the reviewed lower-LR seed-42
pair has different training losses in every epoch and different validation
metrics in 22 epochs despite equal selected metrics. No RING validation
prediction export or inference replay was performed; matching aggregate
metrics/confusion matrices cannot establish identical per-site predictions.

The same 208 sites and 110 PDB groups occur in every validation run.
Within-PDB sites are not independent; Cu has only 15 validation sites and
Zn only 32. Direct-four training does not provide separate Fe/Co/Ni recalls.

## Execution, original budget and verified teardown

The existing local CPU audit was allowed to finish and passed in 1,123.68
seconds; its restart helper was not rerun. The completed original architecture,
geometry and Phase-3 association campaigns were not rerun; this RING recipe
deliberately trains fresh matched controls.

One owned **G4 / NVIDIA RTX PRO 6000 Blackwell Server Edition** session ran
this continuation using stock Torch 2.11.0+cu128, CUDA 12.8 and Python 3.13.15.
The frozen source and runtime operators were checked again at final capture.
Actual preparation took 213.79 seconds for bootstrap and 418.52 seconds for
readiness. All 21 ledger attempts completed (one readiness, four smokes,
sixteen full fits), with **no retries**. Full-fit elapsed times, including
per-fit preparation, were 270.34–280.35 seconds for GVP and 306.39–314.40
seconds for late fusion. These are wall times, not kernel timings or
GPU-memory peak measurements.

Both family blocks passed their measured complete-block admission gate:
setup + 50 epochs for every remaining fit, multiplied by 1.25, fitting both
the original work category and training-allocation clock. Admission was
rechecked as the queue advanced. See
[GVP](../raw/metal_ring_pilot_20260915/finalization/gvp_admission.json) and
[late-fusion](../raw/metal_ring_pilot_20260915/finalization/late_admission.json)
receipts. No original allowance was renewed.

| Allocation | Started epoch | Stopped epoch | Allocated seconds |
|---|---:|---:|---:|
| Original architecture | 1789440733.8320558 | 1789444561.801127 | 3827.969071149826 |
| Recovery + geometry | 1789456819.0405653 | 1789472254.5176592 | 15435.477093935013 |
| This RING continuation | 1789482189.910627 | 1789489252.944619 | 7063.033992052078 |

**Original cumulative use: 26,326.480157 seconds = 7.312911 hours.
Remaining: 9,673.519843 seconds = 2.687089 hours of the original ten-hour cap.**

The queue first reached true `completed`, then finalization verified saved
checkpoint epoch/configuration/normalization and native-history maxima.
The finalization archive was verified locally and in Drive before stopping
`deepmzyme-metal-ring-20260915` at **16:20:52.944619 UTC**.
Colab subsequently reported **no active sessions**. The final capture's
`terminal_metadata_verified_gpu_not_stopped` label is correctly pre-stop;
the separate [actual stop receipt](../raw/metal_ring_pilot_20260915/host_closeout_allocation3/session_stopped.json)
and [post-stop closeout](../raw/metal_ring_pilot_20260915/host_closeout_allocation3/post_stop_receipt.json)
certify actual teardown. Post-stop analysis and report work use no allocated GPU.

## Provenance and preservation

The [portable package README](../raw/metal_ring_pilot_20260915/README.md)
maps exact copied evidence, archive descriptors and Drive readbacks.
The source archive was not rebuilt. Its archived base commit remains
`b45b893cff00b5ae4d10448dbc0671fc6f719071`; the resumed checkout was already
at `743d9a225b737156b521d3a9d312c6d0660ef2a2`, with all 141 compared
source/notebook/requirements/script/test/doc files matching the snapshot.
No commit or push was performed in this continuation.

| Identity | SHA-256 |
|---|---|
| Frozen code archive | `7ad27026c2baaefe9b566d80025455d852c63d3acf8198113743f5685ae6523a` |
| Pinned v12 data bundle | `90c0899829e0ac5ca94a5ef34484b74ca1014d3e9b6b1fba90bd338ee3440dee` |
| Executed campaign manifest | `7ef050d611bc7eec7c0638ec4c8a76b1a32a1e90690bb135c33546e460c3127c` |
| Retained cohort | `60b8af63454883579a2e851fa9a1cc7ddecdc95c53d996240241a099c35bbbe1` |
| Finalization archive | `e6ee056fb5e10831ded41eaff2cd31b69bd879273d7d54c0d149083736da5aeb` |
| Actual post-stop closeout archive | `3d8d6bf51353ddb01de0203e0ff9aeb49cd466eaf64516560dab5f6b44bfe625` |

Every attempt archive retains run/checkpoint evidence locally and in
[the authorized Drive folder](https://drive.google.com/drive/folders/1slTff0joKjL-gZJDhYSGbHzOPnYhGk6I).
Finalization is Drive file `1xdUe0_hOCcZaA9yOB_uODlOpB4h84OfL`;
actual post-stop closeout is `1XJBAtdmLw2AdJQz-vVWI4D3WtAGiMWcQ`.
Drive verification checked file ID, byte size and parent; SHA-256 verification
is local, not a claimed Drive-side checksum.

## Decision and remaining scientific work

Retain this complete bounded comparison as validation evidence. RING's
GVP BA gains are modest and include class losses; no selected-metric gain
appears for the tested late-fusion recipe. Neither result establishes a
universal RING benefit or ineffectiveness.

The broader research plan is unfinished: hybrid comparison, shared grouped
folds/seeds with paired CIs and rare-class gates, Stage 6B final refit,
scientifically resolved final-test route, and certified EC-primary auxiliary
learning remain outstanding. Phase-3 descriptive metal–EC1 association is
already completed separately and is not evidence that auxiliary training helps.
Current next actions belong in [EXPERIMENT_STATUS.md](../../../EXPERIMENT_STATUS.md);
do not automatically spend the remaining allocation on an unplanned run.
