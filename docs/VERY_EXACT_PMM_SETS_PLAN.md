# `very_exact-pmm_sets`: optional PMM source-site benchmark plan

**Status: planned.** This document specifies an additive implementation route; it
does not certify a dataset, provide a runnable command, or authorize a test run.
The existing catalytic PinMyMetal, non-overlap, CLEAN, and CARE paths retain
their present behavior. `Plan.md` owns scientific policy, `docs/DATASETS.md`
owns materialized dataset identity and test-use history, and the metal playbook
owns executable experiment recipes once this route is certified.

## Purpose and meaning of “exact”

Offer `very_exact-pmm_sets` as an explicit **metal-only, source-row-faithful**
option for comparing DeepMzyme's known-site metal classifier with PMM's
metal-type classifier. The PMM train/test source tables, rather than PDB-ID
membership or rebuilt MAHOMES predictions, must decide which examples belong
to each side. A PMM record is the unit of prediction even when two ions form
one multinuclear pocket in the current DeepMzyme representation.

Keep four claims separate in every manifest and report:

1. **Source membership:** each included example retains its PMM train/test row
   and original side. No PDB-level expansion or reassignment occurs.
2. **Site reconstruction:** the PMM row maps unambiguously to a particular
   structural ion or documented candidate-site coordinate. The complete and
   incomplete mapping cohorts have different names and denominators.
3. **Validation-fold identity:** a PMM-published fold is used only if its
   membership can be recovered and verified. A newly generated fold manifest
   is a paired DeepMzyme/PMM comparison, not PMM's original Fig. 2a fold.
4. **Task/input parity:** both systems predict metal type at the same site
   under a declared known-site input protocol. This does not imply replication
   of PMM's end-to-end site discovery or localization performance.

The raw GitHub class-model files contain 7,920 train and 1,488 test rows, but
those are **source-file counts**, not yet certified effective training/evaluation
counts. PMM's released training script applies `dropna()`, and the paper
describes additional data selection and oversampling. The exact effective
cohort and numeric-label mapping must be resolved before a published-score
comparison. The paper reports four metal classes (Mn, Cu, Zn, Fe+Co+Ni) and
fivefold validation on its training data, but the released row files do not
declare five fold IDs. See PMM's
[train](https://github.com/hhz-lab/PinMyMetal/blob/main/data_model/classmodel_train_set)
and [test](https://github.com/hhz-lab/PinMyMetal/blob/main/data_model/classmodel_test_set) files,
[released training script](https://github.com/hhz-lab/PinMyMetal/blob/main/data_model/train_chedhclassmodel.py),
[paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11953438/), and
[source-data deposition](https://springernature.figshare.com/articles/dataset/PinMyMetal_model_metal_binding_sites_source_data_2024/25011212).

## Why the existing `exact_pinmymetal` path cannot be repurposed

`step1a_download_structures.py` reduces source rows to unique PDB IDs. Later
steps choose EC-annotated, transition-metal-containing chains and rebuild
MAHOMES-positive ion rows. The loader clusters ions up to 4.5 Å apart and
labels each cluster. Thus a shared PDB brings its whole processed structure
into both local sides, while a PMM train/test source record may identify
different ions. The local site CSVs do not retain `residueid_ion` or `metalid`.
The existing `exact_pinmymetal` directories remain a separate historical
PDB-projection benchmark.

The new path needs a row-to-ion crosswalk and a row-anchored loader. Merely
renaming the existing directories, filtering their CSVs by PDB, or changing
`--train-val-split-by` cannot make their membership site-exact.

## Proposed on-disk and CLI contract

Use a new root, `DeepMzyme_Data/very_exact-pmm_sets/`, with versioned
`split_metadata.json`, `coverage.json`, `site_crosswalk.csv`,
`ambiguous_sites.csv`, `unmatched_sites.csv`, and train/test
`site_manifest.csv` files. Each side also has a `structure_manifest.csv` that
references immutable coordinate files by checksum. Existing structure-store
objects may be referenced read-only; new coordinate versions and generated
features live in the new root's namespace. The original PMM source CSVs remain
unchanged under `prepare_training_and_test_set/pinmymetal_files/`.

The proposed user-facing switch is
`--dataset-profile very_exact-pmm_sets --dataset-root <root>` on `src/train.py`.
With the switch absent, all current defaults, the MAHOMES summary loader, and
existing run identities behave exactly as before. The profile is metal-only;
it reads train/test site manifests and provides one `PocketRecord` per included
PMM row to the existing model, optimization, and metric code. It fails
preflight if a required manifest or source hash is missing or if legacy
`--summary-csv`/`--test-summary-csv` flags would silently select a different
cohort. An `audit`/`dry-run` builder command that reads existing datasets and
writes only a new, isolated audit output should be available
before any feature generation or training. This syntax is a design target,
not a command that works today.

The builder belongs in a focused module such as `src/build_pmm_site_dataset.py`;
the alternate loader belongs in `src/training/pmm_site_data.py`. Configuration
and run dispatch gain only an opt-in branch. Bundle packing and `report_runs.py`
gain a profile-aware branch after the local path is certified. New runs,
feature caches, comparison tables, and studies use dedicated
`very_exact-pmm_sets` names; the currently modified/running exact-PMM
orchestrator is not an implementation dependency.

The manifest schema must retain at least:

| Field family | Required information |
| --- | --- |
| Source identity | PMM file checksum, side, 1-based CSV row number, `pdbid`, `residueid_ion`, `metalid`, original `label_metal`, and a stable source-row UID |
| Target | Verified PMM numeric-label mapping and the existing `four_class` / `merge_fe_class_viii` target; never infer the target from the observed PDB ion element |
| Structural match | Structure accession/version/hash, model, author and label chain IDs, ion residue number and insertion code, alternate location, coordinate, observed element, and match method |
| Audit | Exact/ambiguous/unmatched status, reason, reviewer override with provenance if needed, physical-site group, and any cross-side structural relationship |

Use the source file checksum plus row number (and preserve the raw identifiers)
as the canonical UID. PDB ID alone, or `residueid_ion` alone, is insufficient.
No deduplication by PDB, ion type, 2.5 Å, or DeepMzyme's 4.5 Å pocket cluster
may silently delete a PMM source record. Physical grouping is recorded
separately for overlap analysis and fold design.

## Site reconstruction and representation

1. Freeze the PMM GitHub release/version and source hashes. Inspect the paper's
   Figshare source-data archive and, if needed, the PMM/NEIGHBORHOOD provenance
   to determine how `residueid_ion` and `metalid` map to PDB ions. Verify the
   numeric class-code mapping from an authoritative PMM source. Record raw
   rows, rows surviving the PMM script's filtering, class support, duplicate
   IDs, shared PDB IDs, and any published fold identifiers. A mismatch between
   release code, deposited data, and the paper is a documented finding, not a
   value to guess.
2. Resolve each PMM row against a pinned coordinate version. Prefer a direct
   identifier mapping. Otherwise require a unique match supported jointly by
   chain/model, residue and insertion code, element, coordinate, and/or
   coordinating ligands. Record author-versus-label numbering and crystal
   symmetry where relevant. Nearest-ion matching by PDB ID alone is
   insufficient. Ambiguous and absent rows go to separate ledgers rather than
   being assigned to a plausible-looking ion.
3. Build one target-anchored sample per mapped PMM row. Nearby ions may be
   recorded as context and as a physical-site group, but cannot determine or
   overwrite that row's target. In particular, two mapped ions in one 4.5 Å
   cluster remain two source-row examples with distinguishable anchors.
   `1a0e` (PMM rows on opposite sides of a PDB containing nearby Co ions,
   with their exact ion crosswalk still unresolved) and `4d8f` (nearby Fe/Mn
   ions in the local structures) are required regression cases. The current
   multinuclear-cluster loader remains the default for all other datasets.
4. Retain the full required protein-chain context for non-EC proteins and
   crystal-interface sites. Do not fabricate an EC label to satisfy filename
   parsing: the opt-in metal loader obtains identity from the manifest and
   leaves EC supervision absent. Where a site needs symmetry mates or a
   historical structure version not available locally, mark the structural
   context as incomplete until it is reconstructed and verified.
5. Audit every input tensor for direct metal-identity leakage from PDB ion
   names, observed elements, source labels, external feature files, or
   metadata. The row's target is supervision only. Freeze a single anchor and
   context policy across train and test; record it in the profile version.
   Existing feature caches are reused only after coordinate/sequence/feature
   identity checks pass. New ESM/external/RING assets live under the new
   profile's own paths.

`Plan.md` currently describes a catalytic-CSV rule requiring structure metals
to match its labels. At implementation time, add a narrowly scoped scientific
policy for this source-row profile: the *labeled anchor* must match the PMM
record, while explicitly marked nearby ions can be unlabeled context. Keep
the current rule unchanged for the legacy catalytic path.

## Comparison and test protocol

- **Primary reference endpoint:** direct four-class training with the existing
  `four_class` scheme; report balanced accuracy, macro F1, per-class recall,
  support, confusion matrix, and row-level predictions keyed by source UID.
  Five- or six-class training is a separately named challenger, never relabeled
  as PMM's direct four-class model.
- **Validation:** if PMM's original fold membership is recovered, reproduce it
  exactly and disclose physical-site/PDB overlap. Otherwise freeze a new
  fivefold manifest on the eligible PMM **train rows** and use identical rows,
  folds, labels, and metric definitions for both DeepMzyme and a rerun PMM
  classifier baseline. Prefer PDB/physical-site grouping for the project's
  primary paired validation comparison; label it `paired_grouped5`, not
  “PMM Fig. 2a folds.” A PMM-like row-stratified sensitivity analysis may be
  separately labeled if scientifically useful.
- **Held-out reference test:** the original PMM test side remains unchanged at
  the source-row level, including PDBs shared with train. Log exact row,
  physical-site, PDB, sequence, and coordinate overlap. Select models using
  validation only, freeze the model/reporting rule, refit once on the allowed
  full train side, then evaluate the reference test once for that fixed run.
  The existing PMM-related test history remains disclosed; this route does
  not create a newly pristine final test.
- **Fair comparator:** rerun the released PMM metal-type classifier, or a
  faithfully reconstructed version, on the same traceably mapped row cohort
  and frozen validation folds/test rows. Present the paper's Fig. 2 numbers
  as direct numerical comparators only after verifying its effective cohort,
  four-class labels, metric denominator, and fold/test protocol. Otherwise
  present them as historical context. State that DeepMzyme receives a known
  target-site coordinate/pocket; this comparison does not include PMM's
  site-finding stage.
- **Partial coverage:** if any source rows are unmatched or context-incomplete,
  produce a `matched_subset` report with an independently rerun PMM baseline
  on those very same rows. Display per-side and per-class coverage and reasons.
  Do not call the subset result a reproduction of PMM's full published test.

## Implementation order and decision gates

| Phase | Deliverable | Gate before proceeding |
| --- | --- | --- |
| 0. Source audit | Versioned source hashes, PMM code/deposition inventory, effective-row and label-code audit, fold-availability finding | Cohort and label semantics are documented; unknowns block “published-exact” claims |
| 1. Site crosswalk | Row-wise mapping plus ambiguous/unmatched ledgers and physical overlap report | Every included row has one unique, reviewed coordinate anchor; train/test sides remain source-faithful |
| 2. Isolated dataset | Versioned site and structure manifests, anchored pockets, feature provenance, coverage report | One graph per included row; no catalytic/EC/native filtering or label leakage; source-to-graph reconciliation passes |
| 3. Opt-in integration | Profile flag, alternate loader, bundle support, profile-aware reporting | Default dataset/split/checkpoint behavior and its identity hashes are unchanged |
| 4. Paired validation | Frozen official or explicitly new fold manifest, matched PMM baseline and DeepMzyme configurations | Complete shared validation predictions, class support, and selection rule; no test-based choice |
| 5. Reference test | One fixed full-train refit and a separately labeled secondary PMM test report | Frozen cohort/model/report rule, completed overlap ledger, and accurate coverage/role labels |

The work can stop after any audit gate without affecting a current run. Only
after Phase 3 is certified should the metal playbook contain a runnable
`very_exact-pmm_sets` recipe; dataset availability then belongs in
`docs/DATASETS.md`, current progress in `EXPERIMENT_STATUS.md`, and executed
evidence in `docs/notebook_outputs/`. Reconcile the current exact-PMM benchmark
document's “identical dataset/folds” wording with this source-site audit before
using either benchmark for a direct PMM performance claim.

## Required verification

- Assert that the raw and effective PMM row counts, class counts, source
  hashes, row UIDs, and side memberships reconcile exactly. Fail on duplicated
  or silently dropped source rows.
- For a shared PDB, prove that each side emits only its mapped source rows;
  a copied structure must not create a copied label. Verify both same-metal
  binuclear and mixed-metal examples, including `1a0e` and `4d8f`.
- Verify non-EC and MAHOMES-negative source-row fixtures are accepted by this
  profile when they are valid PMM records; legacy catalytic loading retains
  its current filters.
- Change a source label or observed ion symbol in a controlled fixture while
  holding approved geometric inputs fixed; check that no model input tensor
  discloses the answer. Check feature alignment and checksum mismatches fail
  loudly.
- Compare the old paths' dataset counts, split identity hashes, default CLI
  arguments, and run metadata before and after the opt-in code change. Test
  every profile-specific failure without writing to active runs, Optuna
  studies, Colab/Drive assets, or existing dataset directories.
- Require machine-readable `membership_exact`, `site_mapping_coverage`,
  `fold_identity`, `input_protocol`, `overlap`, `test_access_history`, and
  `comparison_scope` fields in every final report. An exactness label is
  earned by these audits, not inferred from the directory name.

This plan is confined to documentation. Source files, ongoing jobs, Colab
sessions, external connections, and existing experiment outputs were not
changed to create it.
