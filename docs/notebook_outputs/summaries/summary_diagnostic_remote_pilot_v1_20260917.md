# Bounded diagnostic coordinate-chain sequence-remote pilot — 2026-09-17

The two authorized 50-epoch fits completed. Only-GVP outer balanced accuracy
was **44.61%**, compared with **54.29%** for Only-ESM. The GVP−ESM contrast
changed from −1.23 percentage points on ordinary development validation to
−9.68 points on the diagnostic outer partition: **directional shift −8.45
points**. GVP recalled **0/14 Cu pockets**. Apply the user's practical stop
rule: **NOT SUPPORTED — stop the remote-GVP direction**. The wide component
bootstrap interval crosses zero; this is not proof of a negative population
effect or evidence for a publication-grade remote-generalization claim.

Canonical evidence is in
[`diagnostic_pilot_v1/`](../../../DeepMzyme_Data/notebook_outputs/remote_homology_v1/diagnostic_pilot_v1/).
The machine-readable owners are `diagnostic_report.json`, `verification.json`,
`frozen_pilot.json`, the two run directories, and `execution/` receipts.
Research scope remains in the [remote addendum](../../REMOTE_HOMOLOGY_ADDENDUM.md);
mutable status remains in [EXPERIMENT_STATUS.md](../../../EXPERIMENT_STATUS.md).

## Membership and scientific scope

Endpoint: **diagnostic coordinate-chain sequence-remote**. Only the historical
non-overlap PinMyMetal **training** cohort is used. Protected test tables were
not opened. The task is direct four-class metal prediction: Mn, Cu, Zn, and
Class VIII = Fe+Co+Ni, canonical scheme `merge_fe_class_viii` (`four_class`).

One split seed, **20260917**, was used. The deterministic label/component-only
rebalance moved `component_4tm7` and `component_5hnm` from outer to inner:
three pockets, comprising two Cu and one Zn. It minimizes moved components,
then pockets, then uses a seeded SHA256 tie-break. Training membership is
unchanged; no predictions or performance values informed assignment. The
accepted manifests were already present at task entry, were reproduced from
the prior artifacts, and retained their original frozen hashes throughout.

Counts are **pockets / PDB groups / sequence components**. `protein_id` is a
PDB grouping proxy, not a certified biological-protein accession. Class-specific
group/component counts can overlap across classes and do not sum to totals.

| Partition | Mn | Cu | Zn | Class VIII | Total |
|---|---:|---:|---:|---:|---:|
| Train | 454 / 400 / 59 | 55 / 38 / 5 | 145 / 134 / 89 | 314 / 309 / 69 | 968 / 872 / 164 |
| Inner validation | 97 / 92 / 34 | 16 / 16 / 3 | 32 / 28 / 23 | 67 / 64 / 27 | 212 / 200 / 76 |
| Diagnostic outer | 97 / 89 / 40 | 14 / 11 / 7 | 30 / 29 / 23 | 68 / 64 / 30 | 209 / 189 / 75 |

All PDB/proxy-protein groups and sequence components are intact; zero detected
qualifying >20%-identity edges cross partitions. Reused MMseqs2 release
`18-8cc5c` / version `8cc5ce367b5638c4306c2d7cfc652dd099a4643f`, sensitivity
7.5, E≤1e−3, ≥50 residue-residue positions, and ≥50% shorter-sequence coverage
define the diagnostic edges; the prior primary annotation uses 80% coverage.
Alignment identity is recounted with internal alignment gaps included.
`metal_mmseqs_reuse_receipt.json` preserves the exact command, binary SHA256,
FASTA/alignment hashes and protocol SHA256. Existing search outputs were reused;
no search or feature/cache regeneration was needed.

All diagnostic inner/outer examples have **no qualifying training hit** under
the frozen annotations. This is not measured ≤20% identity. Coordinate chains
do not certify complete biological proteins or all site-participating chains.
No family, superfamily, structural, or homolog-free separation is claimed.

| Frozen artifact | SHA256 |
|---|---|
| `train.csv` | `ee95fc6f103621be4a90c0d6192e6b6ea52ff0635d1b2c3131d65dd4db566272` |
| `inner_validation.csv` | `c4ba4285c9e122e7dd1a55ccec5a3e3d649f06bab76ec22a95cf3357adac180a` |
| `outer_evaluation.csv` | `6355a7880169ab941fd08278eafcd6e2b94cee15a2c12c6244f24b40d8701309` |
| `split_freeze.json` | `fd3dba5ece075359feb95ea126a34d25c711148b64778abd7f06ec5e609f5e6c` |
| `frozen_pilot.json` | `66229a0db378872f4810e43a22562d0e617fd8b268e64e671301da181a5c91b9` |
| Only-ESM configuration | `4aa9268afcaffe7fda0e4cdf7a46ac6db7ebe8030b9720879a9f3894b7555b84` |
| Only-GVP configuration | `c75631ee4c4a59266f54ac4720bc8be9f1e3a5d94b2b17005812613575d63461` |
| Only-ESM selected checkpoint | `67d92c26a74925d697e732b9b3a1b7be9bdb9b1be5a42d6ca2ce7ac145f384e2` |
| Only-GVP selected checkpoint | `ac2925874e816c7af8a69112a8d522c412f757a834c704418cdd07bfa7b04f68` |

`manifest_checksums.json` binds the remaining inputs. Sequences, detected edges,
components, the development allowlist, class counts, overlap audit, feature
inventory/audit, split-generation receipt and unresolved provenance limitations
are retained beside the manifests. All 1,270 required structures and existing
cached inputs passed the audit. Source commit was
`4acdf654f6e87aeb860f912e5bd8b2d3f82bf260` with a dirty worktree; executed source
hashes are bound by `execution/executed_source_manifest.json` (SHA256
`6c68cf93b53537be69d91e98b2e205a058fe79c77cc33d86bfeba56f9782ee4e`).

## Implementation, tests and admission

The existing unified trainer accepts optional `--explicit-membership-manifest`
and `--explicit-membership-sha256` arguments. The descriptor binds three CSVs;
without it, the automatic split path is unchanged. Validation rejects duplicate,
overlapping, unknown or unresolved examples; target/dataset/scheme/hash
mismatches; crossing groups/components/detected edges; protected membership;
missing inputs/classes; and inadequate class-component support. Exact loaded
membership is checked again after data preparation.

Only train/inner structures enter fitting. Training alone supplies normalization
and inverse-frequency class weights; only inner metal balanced accuracy selects
checkpoints. Outer metadata support and file hashes are isolated in preflight;
no outer graph/loader is constructed during fitting. `run_test_eval=False`
implements `INCLUDE_HELD_OUT_TEST_DURING_TRAINING=False` here.

`freeze-checkpoints` verifies both complete 50-epoch histories, exact canonical
memberships, frozen configurations, finite selection scores, and checkpoint/run
metadata agreement. The separate `evaluate` action revalidates both frozen
checkpoints and refuses an existing outer output directory. The real outer
directory did not exist before the two-model freeze. Both checkpoint and source
hashes were verified again after evaluation.

The final focused/regression command was:

```bash
/home/mechti/miniconda3/envs/DeepMzyme/bin/python -m pytest -q \
  tests/test_explicit_membership.py tests/test_standalone_baselines.py \
  tests/test_evaluation_protocols.py tests/test_report_remote_homology.py \
  tests/test_export_validation_predictions.py tests/smoke_checks.py
```

**152 passed, 1 skipped, 6 subtests passed in 46.52 seconds.** The skip was an
existing absent local multi-metal fixture; no relevant membership or
training-core test was skipped. Three existing deprecation warnings were emitted.
The earlier five-file baseline (same command without `tests/smoke_checks.py`)
passed 106 tests and 6 subtests in 12.55 seconds; an intermediate six-file run
passed 149 tests, skipped one, and passed 6 subtests in 26.37 seconds. Logs and
the exact final command/source binding are in `execution/test_receipt.json` and
`execution/tests*.log`. Syntax compilation passed for all task source/test
files. The entire repository suite was not run.

Focused tests cover all 15 requested behaviors, including default splitting,
exact common membership, training-only statistics, inner checkpoint selection,
absence of outer loading, every exclusion/support/hash gate, the two-checkpoint
lock, post-freeze tampering, and one-shot evaluation with saved normalization.
Full CPU preparation plus a finite untrained forward passed for each family
(ESM 243.10 seconds; GVP 223.60 seconds), with **zero optimizer steps**. The
first smoke harness used the wrong output dictionary key; correcting that
harness resolved it before GPU allocation. Original selected checkpoint shapes
also matched the maintained architectures strictly.

The following table was printed and frozen **before GPU allocation** in
`gpu_admission_gate.json`:

| Gate | Result | Evidence |
|---|---|---|
| Source implementation and focused tests | PASS | 152 tests + 6 subtests; one unrelated fixture skip |
| Default splitting unchanged | PASS | Default-path regression and training-core smoke |
| Frozen manifests and checksums | PASS | Deterministic reproduction and unchanged original hashes |
| Four classes in every partition | PASS | Class counts above |
| At least three inner components/class | PASS | Mn/Cu/Zn/VIII: 34/3/23/27 |
| At least five outer components/class | PASS | 40/7/23/30 |
| No detected edge/component/PDB/proxy-protein crossing | PASS | All crossing counts zero |
| Every selected sample has existing required inputs | PASS | Hash inventory and both full CPU preparations |
| Protected tests excluded | PASS | Bound development-only allowlist; no protected route |
| Outer unavailable during fitting/selection | PASS | Filtered loading and separate two-checkpoint lock |
| Both selected configurations frozen | PASS | Exact saved recipes, seed 42, 50 epochs, batch 8 |
| Both full fits forecast within bounded allowance | PASS | 3,835.86-second forecast < 4,500-second retained cap |

## Selected systems and execution

Full configurations are preserved as
[`only_esm_configuration.json`](../../../DeepMzyme_Data/notebook_outputs/remote_homology_v1/diagnostic_pilot_v1/only_esm_configuration.json)
and
[`only_gvp_configuration.json`](../../../DeepMzyme_Data/notebook_outputs/remote_homology_v1/diagnostic_pilot_v1/only_gvp_configuration.json),
with original saved configs/metadata alongside them. Recipes originate from
`A1_1_four_class_lr3e-05_s42` and `A2_0_four_class_lr0.0001_s42` respectively.

Both use task `metal`, direct four-class targets, seed **42**, **50 epochs**,
batch **8**, AdamW, weight decay **1e−4**, fixed learning rate, cross-entropy,
inverse-frequency training-only weights, zero label smoothing, all class loss
multipliers **1**, gradient clip **1**, no AMP, and no early stopping. Both
completed all 50 epochs. The resulting weights in Mn/Cu/Zn/VIII order were
0.289196223, 2.387183428, 0.905483305, 0.418137223.

| Setting | Only-ESM | Only-GVP |
|---|---|---|
| Learning rate | 3e−5 | 1e−4 |
| Representation | Cached ESMC, 960→128 projection | Four GVP layers, scalar 128/vector 16, edge hidden 64 |
| Pooling | Mean + attention | Mean + attention, residue-only readout |
| Encoder/input settings | ESM encoder dropout 0.1 | Conservative node features; node/edge RBF sigma 0.75; shell roles from edge mode |
| Head | Two layers, dropout 0.2 | Two layers, dropout 0.2 |
| Site geometry | Legacy | Legacy; no metal node |
| Trainable parameters | 186,085 | 774,562 |
| Fit wall time | 162.61 seconds | 224.10 seconds |
| Selected epoch | 46 | 15 |
| Selected inner balanced accuracy | 64.73% | 42.35% |

Both retain the same **10 Å pocket radius**, **6 Å radius edges**, and pooling
cutoff **0** (all retained pocket residues). Only-GVP uses no ESM branch; neither
system uses RING, fusion, EC, or feature redesign. Inactive serialized settings
such as `fusion_mode="late_fusion"` do not activate fusion in these standalone
architectures. `metal_eligibility_scheme="six_class"` is the preserved source
cohort filter, not a six-class training target; each model has four output classes.

The ESM residues carry represented-chain language-model context and legacy site
descriptors, while GVP uses geometric/physicochemical residue information.
Capacity and learning rate also differ. This compares the **selected systems**;
it does not isolate a causal effect of representation type.

Exactly one G4 session used an NVIDIA RTX PRO 6000 Blackwell Server Edition,
PyTorch 2.11.0+cu128/CUDA 12.8, Python 3.13.15, verified sm_120 support. Fit wall
times include preparation, persistence and checkpoint verification. Total GPU
allocation including transfer/setup was **1,039.89 seconds (17.33 minutes)**,
below the 75-minute retained cap. Both artifacts were retrieved and verified;
the provider confirmed the session stopped and absent. Setup preserved Colab's
PyTorch. No fit failed or was retried.

Only after both completed checkpoints were frozen did one CPU outer-evaluation
action run, taking **74.06 seconds** for both models. It saved per-pocket labels,
predictions, raw logits/probabilities, PDB/proxy-protein and component identifiers.
There was no split, checkpoint, threshold or configuration change after seeing
results. No calibration was fitted. Protected tests remained unopened throughout.

## Results and uncertainty

Metal labels are site-level, so point metrics use pockets. PDB/protein-proxy and
component identifiers remain available; uncertainty resamples sequence
components, not individual pockets. The ordinary reference reuses existing
verified seed-42 development predictions, without additional model inference.
Its 208 pockets comprise 110 PDB groups and 58 components. Per-class ordinary
pocket/group/component counts are Mn 97/41/16, Cu 15/8/3, Zn 32/21/15, VIII
64/52/35. Diagnostic support is in the partition table above.

| Evaluation | Model | Balanced accuracy | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| Ordinary development | Only-ESM | 72.12% | 77.40% | 74.76% |
| Ordinary development | Only-GVP | 70.90% | 73.56% | 67.22% |
| Diagnostic outer | Only-ESM | 54.29% | 60.77% | 54.87% |
| Diagnostic outer | Only-GVP | 44.61% | 56.94% | 41.73% |

| Class | Ordinary ESM recall | Ordinary GVP recall | GVP−ESM, pp | Outer ESM recall | Outer GVP recall | GVP−ESM, pp |
|---|---:|---:|---:|---:|---:|---:|
| Mn | 94.85% | 85.57% | −9.28 | 64.95% | 61.86% | −3.09 |
| Cu | 73.33% | 93.33% | +20.00 | 28.57% | **0.00%** | **−28.57** |
| Zn | 59.38% | 34.38% | −25.00 | 63.33% | 53.33% | −10.00 |
| Class VIII | 60.94% | 70.31% | +9.38 | 60.29% | 63.24% | +2.94 |

GVP's only outer class advantage is Class VIII (+2.94 points). Its Zn recall
remains ten points below ESM; Cu falls to 0/14 correct versus ESM's 4/14. The
aggregate result does not mask these class-specific failures.

| Contrast | Estimate, pp | Paired component 95% interval, pp | Valid / 10,000 draws |
|---|---:|---:|---:|
| `ordinary_delta = BA_GVP − BA_ESM` | −1.23 | [−18.56, +21.25] | 9,530 |
| `remote_delta = BA_GVP − BA_ESM` | −9.68 | [−23.40, +3.59] | 9,996 |
| `directional_shift = remote_delta − ordinary_delta` | **−8.45** | **[−31.08, +13.50]** | 9,526 |

The pre-inference reporting protocol reused the prior count-tensor component
bootstrap, with 10,000 resamples, seed 20260917 and percentile 95% intervals.
Common draws over the 122-component union pair both models and both cohorts.
Missing-class draws are excluded and counted (470 ordinary, 4 remote, 474
interaction); the predeclared ≥95% valid-draw requirement passed. These intervals
are conditional on the frozen split/fitted checkpoints and have substantial
uncertainty; they do not incorporate new training seeds or split variability.

| Paired correctness | Both correct | GVP only | ESM only | Both wrong |
|---|---:|---:|---:|---:|
| Ordinary | 136 | 17 | 25 | 30 |
| Diagnostic outer | 88 | 31 | 39 | 51 |

Confusion matrices use rows = truth, columns = prediction, ordered
**Mn, Cu, Zn, Class VIII**:

```text
Ordinary Only-ESM       Ordinary Only-GVP
[[92, 0,  1,  4],      [[83, 0,  5,  9],
 [ 0,11,  4,  0],       [ 0,14,  1,  0],
 [ 5, 0, 19,  8],       [10, 8, 11,  3],
 [17, 0,  8, 39]]       [14, 3,  2, 45]]

Diagnostic Only-ESM     Diagnostic Only-GVP
[[63, 0, 11, 23],      [[60, 0, 21, 16],
 [ 0, 4,  9,  1],       [ 2, 0, 10,  2],
 [ 6, 0, 19,  5],       [ 6, 0, 16,  8],
 [ 8, 0, 19, 41]]       [10, 2, 13, 43]]
```

## Worktree and limitations

At entry, `src/training/config.py`, `data.py`, and `run.py` were already modified
(43 insertions, 5 deletions), and `src/diagnostic_metal_pilot.py`,
`src/training/explicit_membership.py`, and `tests/test_explicit_membership.py`
were already untracked. All six originals and the diff were backed up. Existing
work was preserved; the pilot/validator/test drafts were hardened rather than
replaced. The three pre-existing tracked source patches remained unchanged.

The completed task additionally changes only `EXPERIMENT_STATUS.md`,
`docs/REMOTE_HOMOLOGY_ADDENDUM.md`, `docs/notebook_outputs/README.md`, and this
new summary. Canonical run/evidence files are under the requested local data
directory. Before/after status and diffs are retained in `execution/`.
No staging, commit, push, reset, checkout, stash, PR, or history change occurred.

This is one split and one model seed. Inner Cu has only three components,
diagnostic outer Cu seven, and the ordinary reference Cu three. Full-protein and
site-chain provenance remain uncertified. Previously selected ordinary recipes,
the smaller diagnostic training cohort (968 versus 1,181 ordinary pockets), and
different evaluation/checkpoint-selection composition confound a causal
interpretation of the directional interaction. The bootstrap does not remedy
these design limits. This is exploratory coordinate-chain diagnostic evidence,
not Stage 6 evidence, a final refit, model promotion, or a final held-out result.

The negative observed interaction satisfies the user's stop policy despite its
wide interval. No further seed, fold, HPO, RING, EC, fusion, refit, or protected
evaluation is authorized by this result.

**NOT SUPPORTED — stop the remote-GVP direction**
