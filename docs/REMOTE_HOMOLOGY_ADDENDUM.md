# Remote homology addendum

This extends the existing validation campaign with sequence-remoteness reporting.
It does not select a new model, construct a new split, or repeat completed
training. [Plan.md](../Plan.md) owns scientific policy;
[current status](../EXPERIMENT_STATUS.md) owns execution and completion state.
The supplied Phase A audit is the starting evidence, not proof that later
implementation or experimental gates have passed.

## Questions and endpoints

Analyse metal and EC1 separately. The primary contrast is Only-GVP minus
Only-ESM. Its interaction is the difference between that contrast at maximum
detected training identity **≤20%** and at **>30%**. The **≤15%** endpoint is
secondary. The disjoint bins are >30%, (20%,30%], (15%,20%], and ≤15%.
Below 30% alone is not designated extremely remote.

Report matched GVP RING-on minus RING-off separately. Graph-level late fusion
is optional existing-artifact context. Do not add HPO, geometry grids,
threshold-specific training, or extra early/hybrid fits for this analysis.
Keep direct four-class metal and EC depth 1 separate; historical five/six-class
results are not relabeled. Remoteness never chooses a checkpoint or recipe.

Sequence identity is an operational measurement, not proof of absence of
homology. No ESMC pretraining-overlap or memorization claim follows from it.
Only-ESM pools structurally selected pocket residues, while GVP also consumes
residue identities and other features; this is not a pure causal geometry test.

## Reuse and exclusion contract

Preserve the completed 30 architecture, 15 geometry, 16 RING and 12 EC1 fits,
their smokes, feature preparation, and development-only metal–EC1 association.
Preserve the separate completed compact-GVP discovery fit. No completed fit
returns to a new queue merely to obtain predictions.

The initial inference matrix has twelve selected checkpoints:

| Task | Comparison | Existing recipes | Validation passes |
|---|---|---|---:|
| Metal | Original Only-GVP / Only-ESM | GVP LR 1e-4; ESM LR 3e-5; seeds 42/43 | 4 |
| Metal | Matched GVP RING off/on | Geometry shell roles, LR 1e-4, seeds 42/43 | 4 |
| EC1 | Only-GVP / Only-ESM | LR 1e-4, seeds 42/43 | 4 |

The geometry campaign already has fifteen validation prediction CSVs. Those
are optional context; their probabilities do not constitute saved raw logits.
Original and matched RING-off GVP recipes retain distinct identities even
when aggregate scores match. EC GVP+RING has no completed matched arm.

Each reusable run must bind its selected checkpoint, configuration, metadata,
dataset summary, normalization and frozen source files to recorded hashes.
The ledger preserves exact run, family, seed, target and split identities.
Saved source archives take priority for replay; current-source compatibility
must not be inferred solely from successful checkpoint loading.

## Sequence provenance and search

The preparation tool reads only saved development memberships and explicitly
training-only sequence tables. PDB grouping protects identifiers, not sequence
separation. CARE's upstream cluster name does not certify an independently
constructed internal validation split.

Two endpoints remain distinct:

- `represented_coordinate_chain`: a diagnostic based on the chains represented
  by existing model inputs. It does not certify full biological sequence or
  completeness of a multichain metal site.
- `full_protein`: requires resolved sequence provenance and reviewed mappings
  for every relevant training/evaluation protein and contributing chain.
  Preparation fails with an explicit blocker report when certification is
  missing; coordinate sequences cannot silently replace this endpoint.

CARE full source sequences are available. Differences between source and
coordinate sequences require explicit adjudication. Metal chain-isolation files
lack complete biological-sequence provenance. Family/domain annotation coverage
is not certified, so neither family-disjoint nor structure-remote status is
claimed. The exact current discrepancies belong in generated preparation
reports and [DATASETS.md](DATASETS.md).

Freeze a protocol before predictions are joined. Use CPU MMseqs2
`18-8cc5c`, recording the binary checksum and full version output (the official
binary prints commit `8cc5ce367b5638c4306c2d7cfc652dd099a4643f`).

| Search property | Frozen definition |
|---|---|
| Sensitivity | 7.5 |
| Significance | E-value ≤1e-3 |
| Identity | Identical residues / alignment columns, including internal gap columns |
| Minimum length | 50 residue-to-residue aligned positions |
| Primary coverage | At least 80% of the shorter sequence |
| Domain-sensitive audit | At least 50% of the shorter sequence |
| Database convention | One fixed development database per task; save its identity |
| Training maximum | Restrict retained hits to each run/fold's exact training membership |
| No qualifying hit | Explicit unclassified status; never 0% identity |

Export alignment strings and compute length/coverage directly. Enable explicit
identity calculation and retain enough search hits to cover the development
database. The pinned [MMseqs2 documentation](https://github.com/soedinglab/MMseqs2/wiki)
and [release](https://github.com/soedinglab/MMseqs2/releases/tag/18-8cc5c) are the
implementation references. Search sensitivity is still finite.

The pinned [alignment converter](https://raw.githubusercontent.com/soedinglab/MMseqs2/18-8cc5c/src/util/convertalignments.cpp)
reconstructs exported `nident` from a stored identity fraction. The audit instead
recounts identical residues from sequence-validated alignment strings. It
records observed one-residue undercounts and rejects unexplained discrepancies
or identical ambiguous symbols, so export rounding cannot change a threshold
assignment. The search runner writes exact commands and independent checksums.

Take the maximum over all represented contributing chains within a protein/PDB
group. A missing chain or incomplete training mapping invalidates its certified
bin. Preserve exact-sequence aliases. For conservative dependence accounting,
components join protein groups, exact sequence duplicates and detected pairs
above 20% at the 50% coverage audit rule. These components are uncertainty units;
their construction does not change any training split.

## Validation replay and reporting safeguards

The validation exporter constructs inputs and the saved model directly. It
must not invoke an optimizer, regenerate external features, refit normalization,
reselect an epoch, route validation through the held-out evaluator, or write
inside original run directories. It verifies whole-validation agreement before
marking an export reproducible. EC predictions average raw logits by protein
before argmax. Both pocket and protein exports retain identifiers and targets.

Reporting requires hashes binding the protocol, sequence/membership manifests,
search receipts, frozen counts and prediction files. Freeze counts before
inference and bind that freeze to replay receipts. Reject missing/duplicate
examples, changed groups/targets, unmatched seeds/folds, incompatible vocabularies
and repeated OOF examples. A run is not usable merely because its file exists.

Report balanced accuracy, macro-F1, full-vocabulary recall/confusion matrices,
pocket counts, protein/PDB counts, component counts, and excluded/unmapped/no-hit
cases. If a bin lacks a true class, its full-task BA and macro-F1 are undefined;
do not average across a smaller class set.

For each contrast, use 10,000 paired component-bootstrap replicates, seed
20260917, and 95% percentile intervals. Each sampled component is shared across
architectures, seeds, folds and strata. Average seed-specific metrics; do not
create an ensemble. Estimate the interaction directly:

`(BA_GVP,≤20 − BA_ESM,≤20) − (BA_GVP,>30 − BA_ESM,>30)`.

Require at least ten independent components per class in both primary strata
for a supportable full-task interaction. This is an adequacy floor, not a power
calculation. Report missing-class bootstrap draws and withhold an interval if
fewer than 95% of draws are valid. Failure of either support or stability leaves
descriptive/inconclusive evidence. Fixed-split EC5/EC7 and metal Cu support fail
this floor even before remoteness stratification. Across the full EC development
cohort there are only six EC7 proteins: different seeds or folds cannot create
additional independent proteins.

Component intervals supplement existing Stage 6 fold-paired intervals and
rare-class promotion gates. They do not replace them. All results remain
conditional on earlier validation-based recipe/checkpoint selection.

## Execution order and outputs

1. Verify reusable artifacts and write `reuse_ledger.json`; preserve frozen
   source snapshots and map archived paths explicitly to local inputs.
2. Write `protocol.json` with search/statistical settings and named comparisons.
3. Run sequence preparation. Inspect `preparation_report.json`,
   `sequence_manifest.json`, `memberships.json` and per-task FASTA files.
4. Run the pinned CPU search once per task. Save exact arguments, version,
   checksums, timings, alignment TSV and `search_receipt.json`.
5. Annotate exact run/fold membership and freeze `remoteness_manifest.json`
   and `counts_freeze.json`. Record support failures before loading predictions.
6. Replay only the selected existing checkpoints, serially. Save pocket/protein
   prediction CSVs, reproduction receipts and `predictions/manifest.json`.
7. Produce separate metal and EC `remote_homology_report.json` and `.md` files.
   Inspect failed reproduction/support gates before interpretation.

The source interfaces are [`audit_sequence_remoteness.py`](../src/audit_sequence_remoteness.py),
[`export_validation_predictions.py`](../src/export_validation_predictions.py),
and [`report_remote_homology.py`](../src/report_remote_homology.py).
Use `--help` for their current required fields; exact workflow commands belong
in the task playbooks. Generated data belongs under
`DeepMzyme_Data/notebook_outputs/remote_homology_v1/`; portable receipts and a
short verified summary belong in the notebook evidence index.

## Existing confirmation and later gates

Attach metal reporting to the already-prepared Stage 6 folds when those fits
are authorized and admitted. The exact existing matrix, seeds, epochs, selection
metric and allocation ceilings remain owned by the
[metal playbook](METAL_TRAINING_PIPELINE_PLAYBOOK.md#single-gpu-metal-campaign).
No additional training fit is needed for this attachment. EC confirmation
requires its own certified executable recipe; remote EC reporting remains
descriptive under the current independent-protein support limit.

A new ≤20% blocked split is conditional on both insufficient existing/planned
remote support and adequate component-level class feasibility. Join qualifying
identity edges, all pockets of a protein/PDB and verified aliases before
assigning whole components to folds. Verify no blocked edge crosses folds.
Never break a component to improve class balance. No new blocked split or
threshold-specific training campaign is currently justified.

Keep `INCLUDE_HELD_OUT_TEST_DURING_TRAINING = False`, current task-specific
selection metrics and incompatible-study reuse protection. The direct metal
arm remains `four_class`; paired six-class work belongs to the separate
target-formulation campaign. Stage 6B needs completed comparison evidence,
promotion gates and a full non-test refit. Stage 7 additionally needs a resolved
scientific final-test route. Neither is activated by this addendum.

The remaining runtime work belongs to the
[GPU efficiency plan](GPU_RUNTIME_EFFICIENCY_PLAN.md). An operational code
change does not clear a durable user pause, reset allocation accounting or
amend a frozen worker. Preserve the completed campaign evidence when preparing
a versioned continuation.
