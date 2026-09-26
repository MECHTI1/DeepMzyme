# DeepMzyme — Research Design and Technical Authority

DeepMzyme is a deep-learning framework with two separate primary missions:
transition-metal classification and EC/function classification. Both may use
metalloenzyme structural pocket graphs and optional ESMC residue embeddings.
They share infrastructure and may share learned representations in controlled
experiments, but each task remains independently trainable, selectable,
reportable, publishable, and scientifically complete. Neither task is defined
merely as a helper for the other.

This document is the primary design and policy authority. Source code and run
outputs are evidence of implemented behavior; this document states the intended
architecture, training policy, and experiment governance. When any other file
conflicts with this document, prefer this document unless the source code
clearly contains newer working logic that should be preserved.

**Where to find related information:**

| Need | Go to |
| --- | --- |
| Executable orientation, local setup limits, and project navigation | `docs/GETTING_STARTED.md` |
| Colab browser/CLI runtime and environment procedure | `docs/COLAB_GPU_RUNBOOK.md` |
| Locked environment contracts | `requirements/README.md` |
| Benchmark schemas, commands, and artifact inventory | `bench/README.md` |
| Reproducibility remediation decisions and verification | `docs/REPRODUCIBILITY_REMEDIATION_PLAN.md` |
| Documentation index, validation/testing order, and output folder map | `docs/README.md` |
| Current experiment progress and next planned action | `EXPERIMENT_STATUS.md` |
| Dataset/split identity, bundle provenance, and test-use history | `docs/DATASETS.md` |
| Zenodo PinMyMetal exact ion-level dataset & 5-fold CV reproducibility | `docs/ZENODO_PINMYMETAL_EXACT_ION_LEVEL_REPRODUCIBILITY.md` |
| Validation/HPO parameter findings | `docs/PARAMETER_FINDINGS.md` |
| Experiment-batch index and evidence links | `docs/notebook_outputs/README.md` |
| Verified but deliberately unfixed technical issues | `docs/FOLLOW_UP_TECHNICAL_ISSUES.md` |
| Notebook workflow and option reference | `docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md` |
| Copy-paste-ready metal training stages | `docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md` |
| G4-class Optuna policy and exact stage budgets | `docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md` ("G4-Class Optuna Policy") |
| Bounded single-GPU discovery and confirmation | `docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md` ("Single-GPU metal campaign") |
| EC recipe intent and current compatibility status | `docs/EC_TRAINING_PIPELINE_PLAYBOOK.md` |
| Raw experiment results | `docs/notebook_outputs/raw/` |
| Historical documents and commands | `docs/archive/` |
| Public-facing overview and quick-start | `README.md` |

---

## 1) Training/Test Sets

The primary data source is the PinMyMetal train/test split, converted to
site-level summary CSVs compatible with the MAHOMES format. Each row represents
one catalytic metal site.

CSV format: structure name, EC number(s), metal type.

Site-level MAHOMES summary CSVs are the training source of truth. Structure-level
CSV artifacts may contain semicolon-joined metal labels for structures with
multiple catalytic metal sites; these are for inspection only and must not
replace site-level labels for single-label metal training.

Data integrity rule: the only metal types present in structure files must match
those in the CSV exactly, and vice versa.

The PMM known-ion comparison has a narrowly scoped source-row exception:
its frozen `train_cohort.csv` binds released PMM rows to individual physical
ions, and retains otherwise eligible noncatalytic examples. In that profile,
`whether_catalytic=1` and `EC_0.0.0.0` are compatibility placeholders, not
biological annotations. Only bound target ions supply labels; neighboring ions
do not become extra training examples. Explicit missing protein symmetry
context is excluded by the same frozen rule in both systems and in any later
authorized secondary reference report. No EC supervision is inferred from
these records. The [PMM plan](docs/plans/metal_level_metal_task_compared_PMM_final_plan.md)
defines this separate comparison and its validation-to-refit boundary.

Preparation scripts live under `prepare_training_and_test_set/`. These scripts
download structures, create non-redundant chain-level files, and run MAHOMES
activation to produce the site-level summary CSVs used for training.

The authoritative descriptive inventory of PinMyMetal, CLEAN, CARE, split
membership, current materialization, bundles, and historical test access is
`docs/DATASETS.md`. This section owns scientific data policy, not mutable
dataset readiness or counts.

Physical structure storage is content-addressed. Each distinct byte sequence is
stored once under `DeepMzyme_Data/structure_store/objects/`; dataset and fold
directories declare membership with `structure_manifest.csv`. Hash-qualified
paths preserve same-filename/different-content variants. This organization must
not alter split membership or hide train/test overlap. The layout, schema,
migration audit, and maintenance commands are documented in
`docs/STRUCTURE_STORE.md`.

Colab data bundles are built with `src/build_colab_bundle.py`. A bundle can pack
one or more named split roots, their site-level summary CSVs, referenced
structure-store objects, and optional shared feature assets into a compressed
archive for upload to HuggingFace or for local use. The notebook consumes this
bundle via `COLAB_DATA_SOURCE`.

---

## 2) Train the Metal-Classification Model

The intended primary PinMyMetal-compatible reporting target is four-class
classification:

- Mn
- Cu
- Zn
- Class VIII = Fe + Co + Ni

Use the implemented `four_class` label-scheme alias, whose canonical internal
name is `merge_fe_class_viii`. `src/label_schemes.py` maps Fe, Co, and Ni to
the same Class VIII target under that scheme. Do not create a new label scheme
for this target.

For the direct-four arm, select the best checkpoint by highest validation
balanced accuracy (`val_metal_ba



lanced_acc`) over the active four-class target.

The metal program must examine two training formulations for this four-class
endpoint:

1. direct four-class training with `four_class`; and
2. six-class training with `six_class` / `split_all_metals`, followed by a
   deterministic four-class evaluation that merges Fe, Co, and Ni into Class
   VIII.

Direct four-class training is the primary baseline. The six-class-trained,
collapsed-four arm is a required challenger that tests whether finer Fe/Co/Ni
supervision helps the four-class endpoint. It is not optional, and it must not
be relabeled as direct four-class training.

Run the two formulations as a controlled validation comparison across the
initial metal baseline families: Only-GVP, Only-ESM, and GVP + graph-level late
fusion. Within each family, keep eligible samples, grouping, folds, seeds,
features, architecture capacity, optimization budget, and HPO opportunity
matched; only the target formulation and unavoidable output/loss dimensions
may differ. Keep separate run names and Optuna studies. Select checkpoints from
validation only, using `val_metal_balanced_acc` over each arm's active target,
then compare the direct-four metric with
`val_metal_collapsed4_balanced_acc` from the six-class-trained arm on the same
validation units. Use paired confidence intervals and four-class per-class
recall protection before promotion. The six-class arm must also retain its
native six-class metrics and separate Fe/Co/Ni recalls.

Do not use held-out test results to select between these formulations. After
validation confirmation, freeze the selected formulation and complete Stage 6B
before its one-shot Stage 7 report. If both formulations are ever intended as
final report models, that reporting set and its interpretation must be fixed
before any held-out data are opened.

The implemented five-class scheme (`five_class`) remains valid for explicitly
labeled alternative experiments and for preserving historical evidence. It
keeps Mn, Cu, Zn, and Fe separate while grouping Co and Ni. Use a separate run
name and Optuna study whenever the target scheme changes.

Metal training also has an optional **example unit**: `pocket` (the default)
or `ion`. The [metal example terminology](#metal-example-terminology) below
defines both modes and distinguishes example identity from split grouping.

The bounded architecture pilot includes that five-class scheme as an additional
matched challenger across the same three core families. For this pilot every
arm selects checkpoints and its learning rate by native
`val_metal_balanced_acc`; native and collapsed-four reports must come from that
same selected checkpoint. Sum class probabilities before the collapsed-four
argmax. Native four-, five-, and six-class balanced accuracies are different
objectives and must not be ranked against one another. The common-four view
answers the target-formulation comparison; native recalls retain the finer
class diagnostics. Earlier explicitly collapsed-four-selected recipes and
their outputs retain that historical identity.

Historical six-class and five-class runs must retain their original target
scheme in every table, comparison, and interpretation. Their scores are not
direct four-class training scores. In particular, these are different
experiments:

1. training a four-class model directly; and
2. training a six-class model and then collapsing its predictions and metrics to
   Mn, Cu, Zn, and Class VIII.

The direct-four arm reports its active four-class metrics, confusion matrix,
and per-class recall. The required six-class challenger reports both its native
six-class results and a deterministic collapsed-four view. That view must remain
labeled as collapsed reporting from a six-class-trained model. If the
six-class-trained arm is promoted as the source of the primary four-class
endpoint, the final report must state that training formulation explicitly.
The optional collapsed-four auxiliary loss is a third, separate six-class
objective experiment; it is not the required standard six-class challenger and
is not part of direct four-class training.

Executable notebook or playbook defaults may temporarily lag this scientific
policy. Record that mismatch in `EXPERIMENT_STATUS.md` and
`docs/FOLLOW_UP_TECHNICAL_ISSUES.md`; do not reinterpret a six-class or
five-class recipe as a direct four-class run.

For the staged training pipeline (smoke, baseline, HPO, grouped-fold
confirmation, final test) with copy-paste notebook configuration blocks, use
`docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`.

### Metal example terminology

These definitions apply throughout the project. The word **pocket** alone does
not identify the prediction unit; check `metal_example_unit` in the saved
`run_config.json` / `run_metadata.json` and the loader used by that run.

| Term | Meaning |
|---|---|
| Clustered pocket / parent pocket | A residue neighborhood associated with a parsed cluster of one or more metal ions. It may represent a multinuclear binding site. |
| Pocket-level example (`metal_example_unit="pocket"`) | One graph and one metal-class target per eligible clustered pocket. Multiple metal atoms do not make the target multi-label; eligibility and class assignment follow the active label scheme. |
| Ion-centered example (`metal_example_unit="ion"`) | One target ion, its coordinate, and its own residue neighborhood. A retained three-ion site contributes three examples, even when their neighborhoods overlap or are identical. The physical site can still be multinuclear. |
| `PocketRecord` / record `pocket_id` | The existing container and example identifier, reused in both modes. Ordinary ion loading appends `__ION_<index>`; source-cohort loading uses `<structure_stem>__SRC_<source_row>`. Neither the field name nor an assumed suffix establishes the example unit. |
| `parent_pocket_id` | Metadata identifying the original clustered pocket for an ion example. Sibling examples share this parent identity. |
| Validation group | The identity used to assign examples to folds, separately from the record ID. `split_by="pocket_id"` resolves to `parent_pocket_id` when present; `pdbid` groups all sites/ions from a PDB together. Frozen campaign groups may additionally bind exact aliases. |

The implemented default extraction radius is **10 Å**, measured from any
residue atom to the target ion in ion mode, or to any ion in the cluster in
pocket mode. Clustered-pocket extraction is not a sphere around a centroid.
Metal clustering and residue extraction are separate operations; the current
default cluster merge distance is 4.5 Å. First-shell assignment and classifier
pooling are separate controls and do not redefine the example unit.

Only-ESM also uses the extracted residue neighborhood: selecting residues
around a known ion supplies site context even with
`binding_residue_pooling="none"`. The separate `first_shell_bias` option adds
explicit target-shell conditioning to readout. An ion-centered example does
not by itself imply that the model consumes coordinates or explicit metal
nodes; check its architecture and feature configuration.

Ion mode currently applies only to standalone metal training. Keep sibling
ions in one validation group; the PMM ion comparison requires the stricter
PDB-grouped frozen folds. Report **ion examples**, **parent pockets**, and
**PDB groups** as distinct counts. Preserve the recorded unit for historical
runs, and use fresh matched controls when changing units. Ion mode does not
replace the repository's default pocket mode or change held-out-test policy.

Implementation references: [`metal_examples.py`](src/training/metal_examples.py),
[`source_cohort.py`](src/training/source_cohort.py),
[`structure_parsing.py`](src/graph/structure_parsing.py), and
[`splits.py`](src/training/splits.py).

### Canonical Colab metal-training pipeline

The canonical metal-training workflow is
`notebooks/DeepMzyme_training_colab.ipynb` driven by the staged blocks in
`docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`. The pipeline has eight stages with
explicit decision gates: Stage 0 (environment/data readiness), Stage 1
(1-epoch smoke), Stage 2A (Only-GVP validation anchor), Stage 2B (baseline
family comparison), Stage 3 (Optuna plumbing debug), Stage 4 (optional medium
per-family Optuna), Stage 5A-5F (serious per-family HPO), Stage 5G
(RING/radius-only ablation), Stage 6 (top-K seed/split confirmation),
Stage 6B (promotion gates and final full-train refit), and Stage 7
(one-shot held-out test).

Authoritative rules for the pipeline:

- One `MODEL_PRESET` per Optuna study. Optuna never compares model families.
- Prefer a verified G4-class GPU when available; measure the actual accelerator,
  memory and throughput on each Colab allocation. Neither a particular GPU nor
  uninterrupted runtime is guaranteed. The playbook defines exact budgets,
  storage, search spaces, seed lists, and decision gates.
- No held-out test evaluation before Stage 7 and no Stage 7 launch without
  Stage 6 grouped-fold confirmation evidence plus a completed Stage 6B final
  full-train refit selected from that evidence.
- Stage 6 selects candidate configurations from validation/CV evidence only.
  Stage 6B is the mandatory bridge to final training: it ranks candidates by
  mean `val_metal_balanced_acc`, applies the predeclared paired-CI,
  rare-class recall, and tie-breaker promotion policy, then trains/refits one
  final model using the frozen selected configuration before opening the
  held-out test. This final-training run must keep the model family,
  hyperparameters, feature policy, label scheme, fixed final-refit seed, epoch
  budget, checkpoint-selection rule, calibration rule, and optional ensemble
  rule fixed from validation evidence.
- Stage 7 remains a one-shot held-out test event for the fixed final-training
  run derived from the validation-selected configuration. Optional ensemble, calibration,
  temperature-scaling, plot, or confidence-interval outputs are reporting
  additions only and must not feed back into model/configuration/checkpoint
  selection.
- Stage 6/6B promotion uses paired comparisons over shared validation
  folds/splits and, when configured, shared model-seed repeats. The notebook's
  explicit fold-plus-seed mode is `group_kfold_seed_repeat`; plain
  `group_kfold` is a one-seed grouped-fold option. Stage 6B candidate ranking
  uses validation means over the declared fold/seed units; paired bootstrap 95%
  confidence intervals stay fold-level, averaging seeds within each fold when
  multiple seeds are configured. If a top candidate is CI-supported, promote it;
  if top candidates are within the predeclared tie band, use the predeclared
  validation tie-breakers. Rare-class recall protection is required. Raw
  validation deltas alone are not sufficient promotion evidence.
- If an interrupted Stage 6 leaves only a subset of complete candidate/seed
  blocks, an explicitly labeled recovery mode may drop incomplete
  candidate/seed blocks and rank/refit from the remaining complete fold sets.
  The decision artifact must record that complete-seed-block subset basis before
  any held-out test is opened.
- Optional multi-objective HPO may be used as validation-only rare-class
  protection tooling. Its primary objectives are `val_metal_balanced_acc` and
  active metal-scheme `val_metal_min_recall`; for the direct-four arm, that is
  minimum recall across Mn, Cu, Zn, and Class VIII. In the required six-class
  challenger, collapsed-four recall is part of the common-endpoint comparison,
  but it must not replace native six-class minimum recall or hide separate
  Fe/Co/Ni failures.
- Serious validation-only metal Optuna searches should keep the current
  validated batch size in scope and compare the next larger practical batch
  size; reserve very small batches for smoke/debug or memory fallback, and
  reserve much larger batches for explicitly labeled ablations.
- The advanced fusion order is Stage 5C -> Stage 5D -> Stage 5E -> Stage 5F,
  gated by validation evidence and thresholds defined in the playbook.
- This serious-HPO order does not prohibit bounded early/hybrid manual screens
  during Stage 2B. The architecture pilot includes early fusion alongside the
  three core families, then completes the required target-formulation coverage
  before an optional hybrid screen. A weak early result may defer hybrid under
  that historical pilot's budget; it cannot establish that hybrid is ineffective.
  The separate single-GPU campaign below gives hybrid its own initial tuning
  opportunity without an early-fusion performance gate.
- The metal campaign must complete the controlled comparison matrix in
  Section 7 before making publication claims about the value of ESMC, fusion
  position, or RING edges. Separate HPO winners or unmatched historical runs
  do not satisfy this requirement.

### Bounded architecture exploration

The `metal_architecture_pilot_10h_v1` profile is a separately budgeted Stage
0–2B campaign. Its exact schedule, values, runnable block, outputs, and decision
gates belong to the historical bounded pilot section of the metal playbook. It does not
automatically expand into the serious Optuna stages or Stage 6 confirmation.

Use one GPU worker and persistent attempt accounting. Charge setup, training,
failures, interruption losses, and artifact handling to the declared allocation
budget. Forecast complete comparison blocks from measured costs on the current
GPU before admitting them. Preserve completed attempts across reconnects;
partial blocks are coverage evidence, not complete paired comparisons.

Explore architecture families with equal small learning-rate opportunities
before adding capacity, regularization, loss, or graph changes. Treat each
group of changes as a separately budgeted question instead of multiplying all
axes into one search. A later tuning cycle permits one specifically diagnosed
rescue comparison within its budget; after that, park unsuccessful directions
as not competitive under the tested budget. No architectural rejection follows
from smoke results, one learning rate, or one seed. Freeze one configuration
per required arm before separately costing grouped-fold confirmation; do not
automatically send a large HPO shortlist into confirmation.

The pilot uses internal `pdbid`-grouped validation from non-overlap PinMyMetal
training membership. PDB-ID separation does not establish homology separation.
Exact PinMyMetal is a later, separately labeled reference benchmark after
validation choices and the reporting cohort are frozen. Its overlap and the
historical use of the shared non-overlap test must remain visible; selecting
this development route does not resolve the primary final-test route.

Sequence-remoteness reporting is a supplementary validation analysis. The
[remote-homology addendum](docs/REMOTE_HOMOLOGY_ADDENDUM.md) specifies the
primary maximum detected training-identity endpoint of ≤20% and secondary ≤15%
endpoint; below 30% alone is not designated extremely remote. Preserve exact
run/fold training membership, distinguish represented coordinate chains from
certified full proteins, and keep no-hit cases unclassified. Reuse completed
checkpoints, freeze support counts before prediction joins, retain the complete
task vocabulary and report an unestimable interaction when support is absent.
This analysis neither changes model selection nor authorizes new splits,
training, promotion or held-out access. It cannot prove absence of homology.

Keep geometry controls distinct: pocket extraction selects residues around
the supplied metal coordinates; edge radius determines residue connections;
classifier pooling decides which graph states enter the final readout. A zero
pooling-distance cutoff disables only that additional filter. Residue-only
graphs without explicit metal nodes still use metal coordinates for pocket
construction and geometric features. Localized pooling remains a separate
matched ablation after the initial architecture screen.

### Bounded single-GPU discovery and confirmation

The separately named `metal_single_gpu_20h_v2` campaign extends the bounded
exploration policy with a protected confirmation allocation. Its exact recipe,
capacity profiles, diagnostic follow-ups, runtime caps and executable controls
belong to the [metal playbook](docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md#single-gpu-metal-campaign).
It does not replace the historical pilot or rewrite serious Optuna studies.

Compute ceilings are operational controls around the accepted scientific plan.
If a measured plan exceeds its recorded ceiling, ask the user to choose between
stopping at the current evidence boundary and continuing under a quantified,
durably recorded higher ceiling. When the user has already authorized the
increase, continue without asking again. A compute increase does not authorize
changes to model arms, folds, seeds, epochs, validation selection, promotion
gates, or held-out-test policy.

Use one allocated session and one training process at a time. Count allocation
time from provisioning, including setup, failed attempts, interruptions,
transfers and shutdown; changing machines or reconnecting cannot reset it.
Persist completed fits and restart an interrupted fit once from its original
seed. This is attempt recovery, not exact epoch continuation. Admit complete
blocks against measured hardware-specific costs and a safety margin, protecting
confirmation before discretionary tuning. No completion guarantee follows from
a historical minutes-per-fit estimate.

Give all required families an initial learning-rate by capacity screen,
including early and hybrid fusion. Repeat the two best screened recipes per
arm before choosing among them. Preserve matched target-formulation
opportunities across Only-GVP, Only-ESM and graph-level late fusion, including
one predeclared larger late-fusion recipe for both targets and seeds. Diagnose
more than one plausible failure mode before allocating additional experiments;
distinct diagnostic categories receive priority within the small fixed menu.
Treat the larger recipe as a fresh candidate: overlap with historical hidden
widths does not establish reproduction of a historical model.
Bound any further numeric continuation by paired improvement, class-recall
protection, fixed parameter boundaries and a per-arm fit cap including controls.
Only one chain per family is active, and a favorable result on one core target
earns the same opportunity for its paired target. Do not keep tuning after confirmation.
Unresolved learning curves, search boundaries or incomplete blocks mean
budget-limited uncertainty, not universal architectural rejection. Capacity
bundles and separately tuned recipes compare achievable configurations under
the stated opportunity; they do not isolate every individual hyperparameter.

Freeze one configuration per arm before shared grouped-fold/seed confirmation.
Choose any cost-based fallback coverage before inspecting fold results. Use
common-four paired comparisons and class-recall protection, retaining native
six-class diagnostics. A matched Only-GVP RING contrast fixes geometry-derived
shell roles in both controls; a base fit can serve as the off control only if
its full scientific identity matches. Combined models remain RING-off in this
profile. A fixed historical-recipe late-five challenger has its own labeled
confirmation block without opening a new five-class search. Missing comparisons stay incomplete and cannot support corresponding
publication claims.

Until the primary final-test route is resolved, this campaign's confirmation
is exploratory validation. It neither creates Stage 6B promotion evidence nor
launches final refitting or test evaluation. A reportable selection cycle must
first freeze the final-test route, then complete the applicable Stage 6,
Stage 6B and Stage 7 gates. EC remains an independent primary mission with a
separately costed confirmation and certified auxiliary-learning cycle.

### Controlled coordination-geometry exploration

The separate `metal_coordination_geometry_pilot_v1` profile tests whether
candidate-ligand counts, angular summaries, and explicit generic metal nodes
help direct-four Only-GVP prediction. Complete the original pilot's two
learning-rate architecture blocks first, then run this geometry comparison,
then return to the original remaining blocks. Both profiles share the original
cumulative allocation cap, including recovery and setup across sessions; the
geometry profile does not authorize a fresh budget. The metal playbook owns
the [exact five-arm recipe](docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md#bounded-coordination-geometry-pilot--stage-2b),
admission forecast, and execution commands.

Decouple site geometry inputs from metal-node construction. Compare masked
geometry inputs, counts alone, and counts plus angles without metal nodes;
then compare the latter two settings with metal nodes. All explicit geometry
modes must use the same feature dimensions and generic node-type machinery.
Hold the residue-only readout fixed so that adding metal nodes tests their
effect without also changing the pooled node population. The existing pipeline
fits edge-distance and sequence-distance normalization over all training edges;
adding metal edges therefore changes those fitted statistics. Interpret the
metal-node contrast as representation, connectivity, and edge-normalization
changes together, not as a topology-only effect. Angle-only contrasts keep
graphs and their normalization fixed.

Retain the legacy mode for historical configurations; its earlier results are
context, not a replacement for the new explicit masked control.

Interpret the geometry summaries as features of a candidate-ligand heuristic,
not chemically certified coordination numbers or coordination-shape labels.
The existing helper omits water, cofactors, and noncanonical residues and can
use a residue-centroid fallback. Aggregating within-center angles across metal
centers loses center-specific detail. Keep those limitations and missing or
degenerate geometry visible in diagnostics. Neither supplied metal coordinates
nor generic metal-node types may encode the target element identity.

Use matched learning-rate opportunities and a second model seed to assess
whether each added component warrants further work. These fixed-split results
do not promote an architecture or settle the required grouped-fold comparison.
Keep geometry exploration separate from EC, joint learning, RING, fusion, and
held-out reporting; do not multiply those axes into this pilot.

### Controlled RING continuation

After the bounded architecture and geometry queues close, a separately
identified RING comparison may use the remaining original budget. Its
[exact Stage 2B recipe](docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md#bounded-matched-ring-continuation--stage-2b)
uses fresh matched controls and complete per-family blocks. Fix residue shell
annotations geometrically across RING on/off; otherwise RING changes both
node features and edges. Report the contrast as RING edges/interaction
features with their training-fitted edge normalization. Check full-cohort
node-input equality and distinguish added pairs from annotations on existing
radius edges. These fixed-split runs guide further evaluation; they do not
replace grouped-fold confirmation, certify angle-specific benefit, or expand
the initial metal–EC auxiliary experiment.

### Metal Colab Parameter Ownership Rule

Exact executable stage-block values for the metal-training pipeline must be
owned by `docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`. The notebook may expose
coordinated UI defaults for the current canonical workflow, but those defaults
are not a substitute for the stage block being run.

`Plan.md` defines the research policy and stage ordering, including
validation-only selection, held-out-test protection, `val_metal_balanced_acc`
as the metal-selection metric, grouped validation splitting by `pdbid`, one
`MODEL_PRESET` per Optuna study, baseline-first family promotion, and the
advanced-fusion gate. It must not duplicate full stage configuration blocks or
exact stage values.

When a stage budget, Optuna search space, Stage 6 confirmation policy, or
final-test configuration changes, update the files in this order:

1. `docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md` - exact executable stage values.
2. `notebooks/DeepMzyme_training_colab.ipynb` - implemented notebook defaults
   and behavior when the live UI/defaults need to match the canonical workflow.
3. `docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md` - option explanation or
   crosswalk, only if the meaning or stage mapping changed.
4. `AGENTS.md` - agent instructions, only if the expected agent behavior
   changed.
5. `Plan.md` - only if the research policy, ownership policy, or stage ordering
   changed.

Do not add full Stage 0-7 blocks to `Plan.md`.

---

## 3) Train the EC-Number Classification Model

EC/function classification is the second primary DeepMzyme mission. Train and
validate on all EC first-digit classes, and progressively on deeper EC digits.
The decision on maximum training depth is open; start with depth 1
(`--ec-label-depth 1`) and expand to deeper digits only after the depth-1
standalone baseline is stable.

Keep these scopes distinct:

- EC depth-1 classification predicts one first-digit class.
- Deeper hierarchical EC classification predicts at a separately declared
  depth and restarts the staged validation workflow for that target.
- A protein may carry multiple EC annotations in the source data.
- Full multi-label EC prediction would predict multiple distinct EC targets for
  one protein and is not the current implemented objective.

The current label path is single-label at the selected depth. Multiple EC
annotations can share one prefix at that depth and therefore map to one class;
if they produce more than one distinct prefix, the current implementation does
not assign a target for that sample. Do not describe this behavior as a solved
full multi-label EC problem.

Use supervised contrastive learning as a secondary loss. Start with
`--ec-contrastive-weight 0.0` for the clean baseline; explore non-zero
contrastive weight only after the supervised baseline is validated.

EC supervision is structure/protein-level even when extraction creates multiple
separated metal-pocket samples for the same structure. Use group weighting at
`structure_id` to avoid over-counting such structures.

Final test reporting: level-1 balanced accuracy, macro F1, and per-class recall
at each trained depth. Report deeper-level metrics when deeper depths are trained.

For the staged training pipeline with copy-paste notebook configuration blocks,
use `docs/EC_TRAINING_PIPELINE_PLAYBOOK.md`.

### Metal-EC relationship and auxiliary-learning policy

Transition-metal identity and enzymatic chemistry are biologically related,
but the relationship is not deterministically one-to-one:

```text
metal type != EC number
```

This relationship motivates descriptive analysis and machine-learning
experiments. It does not establish that one target determines the other, or
that combining their supervision will improve either prediction task.

Before claiming a benefit from shared learning, establish strong standalone
depth-matched models for both primary missions. The initial model-family set for
each task is:

1. Only-GVP
2. Only-ESM
3. GVP + graph-level late fusion

The first experiment connecting the tasks is auxiliary multi-task learning:

```text
                     -> Metal head -> metal prediction
Input -> shared encoder
                     -> EC head ----> EC prediction
```

This is **independent predictions with shared learning**. The output of one
head is not fed into the other head. Each loss influences the shared trainable
parameters, while each prediction head remains independently evaluated. The
first controlled question is:

> Does metal supervision improve EC prediction?

Compare EC-only against EC plus auxiliary metal using matched data eligibility,
model family, encoder capacity, features, split units, seeds/folds, training
budget, and EC selection metric. Auxiliary learning is a challenger, not an
assumed winner. Negative transfer is possible, and the standalone model for
either primary task remains a valid final model when auxiliary learning does not
improve its validation performance. The reverse question, “Does EC supervision
improve metal prediction?”, is secondary and may be tested later; it is not
required before DeepMzyme can be completed.

Do not add hybrid fusion, node-level late fusion, cross-attention, or other
interaction mechanisms to this first metal-EC experiment. Those remain
separate architecture investigations. Do not make predicted metal a mandatory
EC input or introduce a hard `protein -> predicted metal -> EC` cascade. Soft
predicted-metal probabilities or learned metal embeddings may be studied later
as optional ablations, after the simpler auxiliary-loss comparison is complete.

Known-metal conditioning for EC is an optional diagnostic or special inference
mode only when metal identity is actually available. Record its provenance as
one of:

- experimentally observed metal;
- curated database annotation;
- computationally transferred AlphaFill/MAHOMES assignment; or
- model-predicted metal.

Computationally transferred assignments must not be called perfect ground
truth. Results from these provenance categories must remain separately labeled.

Metal supervision is per clustered pocket or per ion according to the
[example unit](#metal-example-terminology), whereas EC supervision is
protein/structure-level. Multiple pockets from one protein are repeated views
of one EC annotation, not independent EC labels. Preserve structure/protein
grouping, `pdbid` split protection where applicable, and EC group weighting
such as `structure_id` in standalone and auxiliary experiments.
When using ion-level metal examples, keep ions from one parent pocket together
in every split; do not duplicate structure-level EC supervision per ion.

Cross-task leakage protection is mandatory. A protein held out for either
primary task must be excluded from all shared-encoder training, including loss
terms supplied by the other task. For example, a protein held out for EC
validation or test cannot enter joint training because its metal label is
known. Apply the same rule in the reverse direction. Build split membership at
the protein/structure grouping level across the union of label sources before
constructing any shared-learning training set.

Before expensive auxiliary experiments, perform a descriptive association
analysis using permitted training/development data only. Predeclare and report:

- the metal x EC1 contingency table and sample counts;
- `P(EC | metal)` and `P(metal | EC)`;
- Cramer's V;
- a chi-square analysis where its assumptions are appropriate; and
- mutual information and normalized mutual information where appropriate.

This analysis is descriptive. It cannot prove that auxiliary learning will
help, and it must never use held-out test data.
The exact training-only analysis recipe and its source, weighting, and
statistical checks belong in the
[EC playbook](docs/EC_TRAINING_PIPELINE_PLAYBOOK.md#phase-3--training-only-metal--ec1-association).
Completing it does not replace the separate cross-source exclusion certificate
required for shared training.

The approximate cross-task experimental order is:

1. **Phase 1:** establish or reconcile the paired metal-only target-formulation
   baselines: direct four-class training and matched six-class training with
   collapsed-four evaluation.
2. **Phase 2:** establish or reconcile EC depth-1 standalone baselines.
3. **Phase 3:** measure the metal-EC association in permitted development data.
4. **Phase 4:** run one controlled EC-primary comparison: EC-only versus EC
   plus auxiliary metal.
5. **Phase 5:** only if useful or scientifically worthwhile, compare metal-only
   versus metal plus auxiliary EC.
6. **Phase 6:** only after those experiments, consider soft metal conditioning,
   cross-task attention, or more complicated interaction mechanisms.

Use status terms precisely. **Planned** means specified but not implemented;
**implemented** means a code path exists; **smoke-tested** means only execution
or plumbing was checked; **experimentally evaluated** means a declared
validation experiment completed; and **promoted** means its predeclared
selection gate passed. Code existence alone is not scientific validation.

---

## 4) Make Important Parameters and Model Types Configurable

The main training entry point is `src/train.py`, which delegates to
`src/training/config.py` and `src/training/task_entrypoint.py`. Task-specific
thin wrappers `src/train_metal.py` and `src/train_ec.py` are available for
single-task invocations that bypass the joint-task dispatch.

The Colab notebook (`notebooks/DeepMzyme_training_colab.ipynb`) exposes the
commonly used controls and documents the rest clearly enough that advanced users
can reproduce a command-line run.

#### Supported configurable training parameters

| Area | Parameter / CLI flag | Supported values or default | Plain-language meaning | Colab status |
| --- | --- | --- | --- | --- |
| Data paths | `--structure-dir` | train structure directory | Directory containing training `.pdb`, `.cif`, or `.mmcif` structures. | Expose |
| Data paths | `--summary-csv` | train site-level summary CSV | MAHOMES-style site-level labels used for training. | Expose |
| Data paths | `--test-structure-dir` | optional test structure directory | Held-out test structures, used only when `--run-test-eval` is enabled. | Expose |
| Data paths | `--test-summary-csv` | optional test summary CSV | Held-out site-level labels paired with `--test-structure-dir`. | Expose |
| Output/reporting | `--runs-dir` | output root | Parent directory for all run folders and reports. | Expose |
| Output/reporting | `--run-name` | optional | Human-readable run folder name; auto-generated if blank. | Expose |
| Output/reporting | `--dataset-bundle-id` | optional | Stable bundle filename or identifier recorded for source provenance. | Notebook-generated |
| Output/reporting | `--dataset-bundle-sha256` | optional 64-character SHA256 | Declared bundle checksum recorded for source provenance; never backfilled into historical runs. | Notebook-generated |
| Output/reporting | `--run-test-eval` | off by default in CLI | Runs held-out test reporting for a validation-selected checkpoint. For reportable final runs, the checkpoint must come from the frozen final-training/refit run derived from validation/CV selection. | Expose with warnings |
| Output/reporting | `--selection-metric` | task-dependent default | Metric used to select the best checkpoint. Use validation metrics for real comparisons. | Expose |
| Output/reporting | `--save-epoch-checkpoints` | false | Save every epoch checkpoint, not only the selected/best checkpoint. | Advanced |
| Output/reporting | `--allow-train-loss-test-eval-debug` | false | Debug-only override allowing held-out test evaluation without validation selection. | Advanced warning |
| Runtime | `--device` | `cpu` | PyTorch device such as `cuda` or `cpu`. | Expose |
| Runtime | `--deterministic` | false | Enables stricter deterministic PyTorch behavior for reproducibility, possibly slower. | Expose |
| Runtime | `--num-workers` | `0` | Number of DataLoader worker processes. Default preserves single-process loading. | Advanced |
| Runtime | `--pin-memory` | false | Enables pinned DataLoader host memory only for CUDA runs. CPU runs ignore it. | Advanced |
| Task | `--task` | `joint`; choices `joint`, `metal`, `ec` | Selects metal-only, EC-only, or joint prediction heads and losses. The raw CLI default is an implementation default, not scientific preference for joint training. | Expose |
| Target labels | `--metal-label-scheme` | raw code default `split_all_metals`; aliases `six_class`, `five_class`, `four_class` | Selects metal target classes. The primary reporting endpoint is four-class: the direct arm uses `four_class` / `merge_fe_class_viii`, and the required challenger uses standard `six_class` training with collapsed-four evaluation. Raw defaults may lag the paired recipe. `five_class` means Mn/Cu/Zn/Fe plus grouped Co/Ni. | Expose |
| Data policy | `--metal-example-unit` | `pocket`; choices `pocket`, `ion` | For standalone metal training, use one clustered-pocket label or one example and target per matched metal ion. Ion examples use their own coordinate and 10 Å residue neighborhood; the configured split policy groups siblings by parent pocket or a broader group. See [terminology](#metal-example-terminology). | Expose for metal diagnostics |
| Training | `--epochs` | `10` | Maximum number of training epochs. | Expose |
| Training | `--batch-size` | `8` | Number of example graphs per mini-batch: clustered pockets or ion-centered examples, according to `--metal-example-unit`. | Expose / sweep |
| Training | `--learning-rate` | `3e-4` | Optimizer step size. Previous serious baselines often start at `3e-5`. | Expose / sweep |
| Training | `--grad-clip-norm` | `1.0` | Gradient clipping max norm. Values `<= 0` disable clipping. | Advanced |
| Training | `--amp` | false | Optional CUDA automatic mixed precision for training only; evaluation stays FP32. | Advanced |
| Training | `--grad-accum-steps` | `1` | Number of mini-batches accumulated before each optimizer step. | Advanced |
| Training | `--weight-decay` | `1e-4` | L2-style optimizer regularization. | Expose / sweep |
| Training | `--seed` | `42` | Random seed for split/sampling/model initialization. | Expose / sweep |
| Training | `--lr-schedule` | `fixed`; choices `fixed`, `cosine`, `step` | Learning-rate schedule. | Expose / sweep |
| Training | `--lr-step-size` | `0`; required positive for `step` | Epoch interval for step LR decay. | Expose |
| Training | `--lr-decay-gamma` | `0.5` | Multiplicative LR decay for step schedule. | Expose |
| Split/validation | `--val-fraction` | `0.0` in CLI | Fraction of training data reserved for validation when not using folds. Real model selection should use validation. | Expose |
| Split/validation | `--train-val-split-by` (`--split-by` legacy alias) | `pdbid`; choices `pdbid`, `pdbid_chain`, `structure_id`, `pocket_id` | Group identity used only to avoid leakage when splitting the configured training source into train/validation. It does not change the explicit held-out test directory or CSV. Default `pdbid` is stricter than `pdbid_chain`, so same-chain repeated or binuclear metal sites cannot cross train/validation. | Expose |
| Split/validation | `--n-folds`, `--fold-index` | optional pair | Enables one fold of grouped cross-validation instead of a simple validation fraction. | Advanced |
| Data policy | `--unsupported-metal-policy` | `error`; choices `error`, `skip` | Whether unsupported metal labels should fail or be skipped during loading. | Advanced |
| Data policy | `--invalid-structure-policy` | `skip`; choices `error`, `skip` | Whether unreadable/invalid structures should fail or be skipped. | Advanced |
| Data policy | `--require-all-task-classes` | false | Fail if the training split lacks a class needed by the selected task. | Advanced |
| Model family | `--model-architecture` | `gvp`; choices `gvp`, `only_esm`, `only_gvp`, `simple_gnn_esm` | Selects the graph/ESM architecture family. | Expose |
| Model size | `--hidden-s` | `128` | Scalar hidden channel width used by GVP/GNN and classifier projections. | Expose / sweep |
| Model size | `--hidden-v` | `16` | Vector hidden channel width for GVP models. Ignored by non-GVP variants. | Expose / sweep |
| Model size | `--edge-hidden` | `64` | Hidden width for encoded edge features. | Expose / sweep |
| Model size | `--gvp-layers` | `4` | Number of graph message-passing layers. The raw CLI default is independent of the canonical notebook Optuna search space; use the playbook for current reportable metal HPO capacity ceilings and label any outside-space depth check explicitly. | Expose / sweep |
| Model size | `--head-mlp-layers` | `2` | Number of linear layers in metal/EC classifier heads. | Expose / sweep |
| Model regularization | `--head-mlp-dropout` | `0.2` | Dropout between hidden layers in classifier heads. Default preserves the previous hardcoded head dropout. | Expose / optional sweep |
| Graph construction | `--edge-radius` | project default currently `8.0` in code | Residue-neighbor radius for graph edges before optional RING edges. | Expose / sweep |
| Graph construction | `--metal-node-mode` | `none`; choices `none`, `per_metal` | Opt-in GVP graph variant that appends one generic metal anchor node per metal coordinate and promotes metal-ligand edges into message passing. Explicit site-geometry modes control summary inputs independently; `legacy` retains the historical coupling. Must not encode the true metal element. | Advanced / validation-only ablation |
| Site geometry | `--site-geometry-features` | `legacy`; choices `legacy`, `none`, `counts`, `counts_angles` | GVP summary-input control. Explicit modes share eight masked geometry slots and generic node-type embeddings; counts use `log1p`, and six angular summaries are divided by 180. `legacy` preserves historical behavior. | Advanced / validation-only ablation |
| Node/edge encoders | `--node-feature-set` | `conservative` only | Named set of residue/node features. Only `conservative` is currently implemented. | Expose |
| Node/edge encoders | `--node-rbf-sigma` | `0.75` | Width of distance radial-basis features for node distance features. | Advanced |
| Node/edge encoders | `--edge-rbf-sigma` | `0.75` | Width of distance radial-basis features for edge distance features. | Advanced |
| Node/edge encoders | `--node-rbf-use-raw-distances` | false | Uses raw, unnormalized node distances for node RBF expansion when available. | Advanced |
| Classifier pooling | `--classifier-pool-distance-cutoff` | `0.0` | If positive, pools only residues within this CA-to-metal Angstrom cutoff before the final classifier head; `0.0` keeps all residues. | Advanced |
| Classifier pooling | `--structural-readout-scope` | `auto`; choices `auto`, `residue_only`, `residue_and_metal`, `metal_only` | Controls which GVP structural nodes are pooled. `auto` preserves residue-only readout for standard graphs and uses residue-plus-metal readout when `--metal-node-mode per_metal` is enabled. | Advanced |
| Training augmentation | `--position-noise-std` | `0.0` | Training-only Gaussian coordinate noise. Validation and held-out test graphs stay unaugmented. | Advanced / optional sweep |
| Training augmentation | `--second-shell-dropout` | `0.0` | Training-only dropout probability for second-shell residues. Labels and cached source structures are unchanged. Canonical metal HPO keeps this fixed off; use only for explicitly labeled out-of-search-space ablations. | Advanced / manual ablation only |
| Training augmentation | `--outer-residue-dropout` | `0.0` | Training-only dropout probability for pocket residues that are neither first-shell nor second-shell. Labels and cached source structures are unchanged. This is the canonical residue-dropout sweep axis for metal HPO. | Expose / canonical Optuna sweep |
| ESM inputs | `--esm-embeddings-dir` | optional path | Directory containing precomputed ESMC residue embeddings. Needed by ESM-using models unless generation/missing behavior is enabled. | Expose |
| ESM inputs | `--esm-dim` | code default ESMC dimension | Expected dimension of residue ESM embeddings. | Advanced |
| ESM inputs | `--allow-missing-esm-embeddings` | false | Allows ESM-using runs to continue when embeddings are missing; use only for explicit debugging/ablation. | Expose with warning |
| ESM inputs | `--no-prepare-missing-esm-embeddings` | false | Disables automatic generation of missing ESM embeddings. | Expose as prepare-missing toggle |
| ESM inputs | `--disable-esm-branch` | false | Disables late ESM branch for compatible graph models. Usually prefer `only_gvp` for graph-only baseline. | Advanced |
| External features | `--external-features-root-dir` | optional path | Root directory for residue-level external features such as updated SASA/electrostatics. | Advanced |
| External features | `--external-feature-source` | `auto`; choices `auto`, `bluues_rosetta`, `updated` | Selects which external feature layout/source to read. | Advanced |
| External features | `--allow-missing-external-features` | false | Allows training if external feature files are missing, filling defaults where possible. | Expose |
| ESM fusion | `--fusion-mode` | `late_fusion`; choices `late_fusion`, `early_fusion`, `node_level_late_fusion`, `hybrid`, `cross_modal_attention` | Controls where ESM information is combined with graph states. | Expose via presets |
| ESM fusion | `--esm-fusion-dim` | `128` | Projection width for graph-level ESM pooling/fusion. | Expose / sweep |
| ESM fusion | `--esm-graph-encoder-dropout` | `0.1` | Dropout inside the graph-level ESM encoder branch. Default preserves the previous hardcoded ESM encoder dropout. | Expose / optional sweep |
| Early ESM | `--use-early-esm` | false | Adds residue-level ESM features before graph message passing. Automatically implied by early/hybrid fusion presets. | Preset/advanced |
| Early ESM | `--early-esm-dim` | `32` | Bottleneck dimension for early residue-level ESM projection. | Expose |
| Early ESM | `--early-esm-dropout` | `0.2` | Dropout in the early ESM projection. | Expose |
| Early ESM | `--early-esm-raw` | false | Uses raw full-size ESM vectors as early node features; high-dimensional ablation. | Advanced warning |
| Early ESM | `--early-esm-scope` | `all`; choices `all`, `first_shell`, `first_second_shell` | Limits early ESM injection to all residues or selected shell residues. | Advanced |
| Cross-attention | `--cross-attention-layers` | `1` | Number of cross-modal attention blocks. Only active for cross-modal attention fusion. | Expose / sweep |
| Cross-attention | `--cross-attention-heads` | `4` | Number of attention heads per cross-modal block. | Expose / sweep |
| Cross-attention | `--cross-attention-dropout` | `0.1` | Dropout inside cross-modal attention blocks. | Expose |
| Cross-attention | `--cross-attention-neighborhood` | `all`; choices `all`, `first_shell`, `first_second_shell` | Which residues participate in localized cross-attention. | Expose |
| Cross-attention | `--cross-attention-bidirectional` | false | Allows ESM states to also attend back to structure states. | Expose |
| RING edges | `--ring-features-dir` | optional path | Directory containing RING edge files, or output directory for generated RING files. | Expose |
| RING edges | `--use-ring-edges` | false in raw CLI; notebook default is RING-enabled | Adds RING interaction edges in addition to radius edges when files are available. | Expose via mode |
| RING edges | `--require-ring-edges` | false | Fails if RING edge files are missing for requested structures. | Expose with warning |
| RING edges | `--prepare-missing-ring-edges` | false flag, but current config prepares by default unless disabled | Generate missing RING edge files during preflight when RING is active. Notebook default is `with_ring`, with `REQUIRE_RING_EDGES=False` and missing-edge preparation enabled. | Expose |
| RING edges | `--no-prepare-missing-ring-edges` | false | Prevents RING generation during preflight. | Expose as prepare-missing toggle |
| Metal loss | `--balance-metal-site-symbols` | false | Uses a weighted sampler to balance metal classes and, when Co/Ni are grouped, Co/Ni site symbols inside the grouped class. | Expose |
| Metal loss | `--metal-loss-function` | `cross_entropy`; choices `cross_entropy`, `focal` | Loss function for metal classification. | Expose |
| Metal loss | `--metal-focal-gamma` | `2.0` | Focal-loss gamma when focal loss is selected. | Expose |
| Metal loss | `--metal-label-smoothing` | `0.0` | Label smoothing for metal cross-entropy. | Expose |
| Metal loss | `--metal-collapsed-loss-weight` | `0.0` | Optional validation-only collapsed-four auxiliary loss for an explicitly labeled six-class objective. It is not part of direct four-class training. | Advanced |
| Metal loss | `--metal-class-weight-mode` | `inverse_frequency`; choices `none`, `manual`, `inverse_frequency`, `inverse_sqrt_frequency`, `effective_number` | Controls class weights for the metal loss. `manual` starts from `1.0` for every class and uses the per-class loss multipliers as exact class weights. | Expose |
| Metal loss | `--mn-loss-multiplier`, `--cu-loss-multiplier`, `--zn-loss-multiplier`, `--fe-loss-multiplier`, `--co-loss-multiplier`, `--ni-loss-multiplier`, `--class-viii-loss-multiplier` | `1.0` each | Per-class multipliers applied to computed metal class weights; with `--metal-class-weight-mode manual`, they are the exact manual class weights. | Advanced |
| Joint loss | `--joint-loss-weighting` | `auto`; choices `auto`, `fixed`, `uncertainty` | Controls task-level metal/EC loss balancing. `auto` uses learned uncertainty weighting for joint runs and fixed weighting for single-task runs. | Expose |
| Joint loss | `--metal-loss-weight` | `1.0` | Base task-level multiplier for the metal loss; mainly useful with `--joint-loss-weighting fixed` or deliberate ablations. | Expose |
| Joint loss | `--ec-loss-weight` | `1.0` | Base task-level multiplier for the EC loss; mainly useful with `--joint-loss-weighting fixed` or deliberate ablations. | Expose |
| EC labels/loss | `--ec-label-depth` | `1` | EC hierarchy depth used to build EC labels. | Expose / sweep |
| EC labels/loss | `--ec-group-weighting` | `structure_id`; choices `none`, `structure_id`, `pdbid_chain`, `pdbid` | Weights EC loss so multiple pockets from the same structure/group do not over-count one protein. | Expose |
| EC labels/loss | `--ec-contrastive-weight` | `0.0` | Optional supervised contrastive loss weight for EC representations. Keep `0.0` for the clean baseline. | Expose / sweep |
| EC labels/loss | `--ec-contrastive-temperature` | `0.1` | Temperature used by EC contrastive loss. | Expose |

#### Supported model families and fusion modes

- `only_gvp`: graph-only GVP baseline. It should not require ESM embeddings.
- `only_esm`: ESM-only baseline. It requires ESM embeddings unless missing embeddings are explicitly allowed or generated.
- `gvp`: GVP structure model with optional ESM branch/fusion.
- `simple_gnn_esm`: non-GVP graph + ESM comparison model.

For `gvp` and `simple_gnn_esm`, supported fusion modes are:

- `late_fusion`: pool graph states and ESM states separately, then fuse near the classifier head.
- `early_fusion`: inject residue-level ESM features before graph message passing and disable the late ESM branch.
- `node_level_late_fusion`: inject ESM into node states after graph message passing and before pooling.
- `hybrid`: use both early residue-level ESM and late graph-level ESM.
- `cross_modal_attention`: advanced graph/ESM attention fusion; use only after simpler baselines are stable.

The `--metal-node-mode per_metal` option is currently a GVP-only validation
ablation (`gvp` and `only_gvp`). It adds generic metal anchor nodes and
metal-ligand message-passing edges. `--site-geometry-features legacy` preserves
the historical ligand-angle-summary path with metal nodes, including its
training-pipeline normalization; the
explicit `none`, `counts`, and `counts_angles` modes control these inputs
independently with fixed feature dimensions. These graph features must remain
identity-safe: they may use geometry and a generic metal-node type, but not the
true metal symbol, atomic number, or class-specific chemistry.

#### Desired future work not currently supported

- Additional `node_feature_set` values beyond `conservative`.
- A general EC loss-function selector equivalent to `--metal-loss-function`; EC currently uses cross-entropy plus optional contrastive loss.
- Generic class-loss multiplier flags for EC classes. Current per-class multipliers are metal-specific.
- Additional LR schedules beyond `fixed`, `cosine`, and `step`.

#### Practical notebook-ready training pipelines

`Plan.md` is the high-level research and design authority. The concrete,
copy-paste-ready notebook pipelines live in task-specific playbooks:

- Metal classification: `docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`
- EC-number classification: `docs/EC_TRAINING_PIPELINE_PLAYBOOK.md`

Each playbook covers staged notebook configuration blocks for:

- smoke/readiness checks
- baseline model comparison
- controlled medium Optuna searches
- large controlled Optuna searches
- top-K grouped-fold confirmation
- final held-out test evaluation after validation-based selection only

For the metal notebook pipeline, the playbook must keep exact values for:

- `TASK`, `RUN_MODE`, `RECOMMENDED_RUN_SET`, `MODEL_PRESET`
- split and selection controls: `DATASET_NAME`, `VAL_FRACTION`, `SPLIT_BY`,
  `SELECTION_METRIC`, `OPTUNA_SELECTION_METRIC`
- feature controls: `RING_EDGE_MODE`, RING preparation/requirement flags,
  ESM embedding flags, and external-feature strictness
- baseline budgets: epochs, batch sizes, learning rates, weight decay, seeds,
  and maximum planned rows
- Optuna budgets and sampler controls: `OPTUNA_TARGET_COMPLETE_TRIALS`,
  `MAX_EPOCHS_PER_TRIAL`, `OPTUNA_N_STARTUP_TRIALS`,
  `OPTUNA_TPE_MULTIVARIATE`, `OPTUNA_TPE_GROUP`,
  `OPTUNA_TPE_CONSTANT_LIAR`, `OPTUNA_PARALLEL_WORKERS`,
  `OPTUNA_PARALLEL_STARTUP_STAGGER_SECONDS`,
  `OPTUNA_STOP_ON_PARALLEL_CUDA_OOM`, `OPTUNA_AUTO_CONFIGURE_BUDGET`, storage,
  search preset, and search ranges
- Optional validation-only objective controls:
  `METAL_COLLAPSED_LOSS_WEIGHTS_CSV` and
  `OPTUNA_MULTIOBJECTIVE`
- Stage 6 confirmation controls: top-K, grouped-fold count, split seed, model
  seed list for fallback repeats, mismatch guard, paired-bootstrap comparison,
  and final-test exclusion
- final-test controls: preview mode first, explicit one-shot confirmation,
  source-run selection, predeclared primary report, optional five-checkpoint
  softmax-mean ensemble reporting, calibration settings, bootstrap confidence
  intervals, and repeat/mixed-batch guards

Keep current best-result notes and mutable next-step status in
`EXPERIMENT_STATUS.md`. Keep raw and summarized run evidence in
`docs/notebook_outputs/`.

Pipeline governance:

- The playbook owns the exact parameter values for every stage. The notebook may
  expose coordinated defaults for convenience, but the playbook remains the
  canonical copy-paste recipe. Plan.md does not duplicate stage blocks.
- Current notebook defaults, conservative first-pass GVP/HPO profiles, and
  canonical extended G4 HPO budgets are separate concepts. Treat notebook
  defaults as the live launch surface, not as held-out-test-selected evidence.
  Keep exact conservative-profile values and exact stage budgets in the metal
  playbook.
- Changes to stage budgets, search spaces, or stage ordering must update the
  playbook first, then the notebook if live defaults should match the canonical
  workflow, then `METAL_NOTEBOOK_CONFIGURATION_GUIDE.md`'s crosswalk if the
  option meanings or stage map changed, and this section only if governance or
  stage policy changed.
- Changes to the held-out test policy must update Plan.md first and propagate
  to the playbook's Stage 7.

### Pipeline design trade-offs

The sequential baseline-first architecture search is publication-safe because
each added modeling component has validation evidence against a stable simpler
anchor. The trade-off is that it may miss global optima that would appear only
from a joint architecture, capacity, feature, and loss search.

Separating Optuna discovery from Stage 6 grouped-fold stability checks costs
more compute than selecting the single best trial directly, but it makes the
analysis cleaner: HPO finds candidates, while shared-fold validation estimates
whether a candidate is stable enough to promote.

The non-overlapped PinMyMetal split remains useful as a historical reference,
but its held-out test was accessed in seven early runs and is not the current
final-reporting route. A separate scientific decision may select a newly
protected route such as a temporal, sequence-identity-clustered, or
EC-stratified split so that generalization is not tied to one historical
benchmark construction.

---

## 5) Colab Notebook and Data Bundle

The interactive training workflow is in `notebooks/DeepMzyme_training_colab.ipynb`.
The notebook supports run planning, training execution, result summarization, and
final held-out test evaluation in a staged, guarded workflow.

Colab data input modes (controlled by `COLAB_DATA_SOURCE`):

- `huggingface_link`: downloads and verifies the bundle from the project
  HuggingFace repository. Recommended default for cloud use.
- `upload_file`: prompts for a local `.tar.gz` or `.tar.zst` upload in the Colab runtime.
- `drive`: uses the configured Google Drive path after Drive is mounted.

The notebook's `BUNDLE_FILENAME`, `BUNDLE_URL`, and `BUNDLE_SHA256` fields are
the executable source of truth for the currently configured bundle. Keep the
matching tracked bundle name, checksum, contents, and upload provenance in
`docs/DATASETS.md` instead of pinning a versioned bundle filename in this
design document. The historical design preference was the non-overlapped
PinMyMetal route, but its test was accessed in seven early runs. Its bundle
availability does not restore pristine status. The primary final-test route is an
unresolved scientific decision; do not silently substitute exact PinMyMetal or
another dataset. The current availability and access record is owned by
`docs/DATASETS.md`.

Browser and CLI access to Colab, including same-VM attachment, stock-PyTorch
preservation, CUDA architecture preflight, Drive authorization, artifact
transfer, and mandatory CLI teardown, are documented in
`docs/COLAB_GPU_RUNBOOK.md`. These operational steps do not own stage budgets
or scientific selection policy.

Colab bundles include:

- site-level MAHOMES train and test summary CSVs (training source of truth)
- training and test structure files (`.pdb`, `.cif`)
- structure-level CSV artifacts (for inspection; not used for training labels)
- optional shared ESM embeddings, updated external features, RING features, and
  the RING runtime when included at bundle-build time

Comparison table and professional figure output are generated by the summarize
cell at the end of each run batch. Detailed notebook option reference is in
`docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md`.

---

## 6) Experiment Tracking and Reproducible Run Summaries

Every training run should save enough information to reproduce and compare the
result.

Each run should save:

- full config / hyperparameters
- metal label scheme
- random seed
- dataset paths and split identity
- dataset bundle identifier and checksum for serious validation or final-test
  runs, when the run uses a bundle
- key library versions for serious validation or final-test runs, including
  PyTorch, torch-geometric, ESM/ESMC, Optuna, NumPy, and scikit-learn when
  available
- model architecture
- fusion mode
- node feature set
- EC label depth, if relevant
- contrastive-learning settings, if relevant
- validation metric used for checkpoint selection
- selected checkpoint path
- final held-out test metrics, if test evaluation was requested
- git commit hash, if available

`src/report_runs.py` summarizes multiple run directories into one CSV table.

Run tiers:

| Tier | Purpose | Evidence status | Required record |
| --- | --- | --- | --- |
| Debug | Path, syntax, smoke, and plumbing checks | Not model-selection evidence | Enough config to reproduce the failure/smoke behavior |
| Serious validation | Baseline comparison, HPO, grouped-fold confirmation, or validation ablation | Eligible for model-selection discussion if gates pass | Full config, split/group policy, seeds/folds, dataset bundle checksum, git commit, key library versions, and validation artifacts |
| Final test | One-shot held-out reporting for the fixed final-training run derived from a validation-selected configuration | Final report only; never feeds back into selection | All serious-validation records plus Stage 6 selection evidence, final-training source-run identity, checkpoint, primary-report declaration, calibration/CI settings, and no-test-selection statement |

The repository has a solved Linux x86_64/Python 3.12 CPU development/test
contract in `pyproject.toml` and `uv.lock`, plus optional test, reporting, and
ESM groups documented in `requirements/README.md`. This lock is not a CUDA
contract. Colab uses `requirements/colab-overlay.txt`, which deliberately omits
PyTorch so the managed stock PyTorch/CUDA stack remains intact and is checked
with the architecture preflight in `docs/COLAB_GPU_RUNBOOK.md`. A separately
locked local CUDA environment is not currently supported. Per-run environment
capture remains mandatory because a lock complements rather than replaces
run-specific metadata, GPU/driver identity, commit/dirty state, and source
artifact checksums.

The summary table should include, when available:

- run name
- task
- metal label scheme
- model architecture
- fusion mode
- seed
- node feature set
- EC label depth
- selection metric
- best validation metrics
- final held-out test metrics
- final held-out calibration metrics and bootstrap confidence intervals
- metal active-scheme metrics, including direct four-class metrics, native
  six-class metrics for the required challenger, and scheme-labeled five-class
  metrics for alternatives
- metal collapsed-four metrics labeled as reporting from a six-class-trained
  model when applicable
- EC level-1 / level-2 metrics
- split name/type used for the run
- whether train/test overlap was detected

Important rules:

- Validation metrics are used for checkpoint selection and hyperparameter choice.
- After cross-validation or Stage 6 grouped-fold confirmation, select exactly
  one configuration from validation/CV evidence, train or refit the final model
  with that frozen configuration, and only then run the held-out test once.
- Held-out test metrics are used only for final reporting.
- Do not choose models by repeatedly checking the held-out test set.
- Stage 7 may report a predeclared five-checkpoint softmax-mean ensemble, but
  the ensemble source runs, averaging rule, and primary result label must be
  fixed before opening the held-out test.
- Stage 7 may include calibration metrics, validation-fitted temperature
  scaling, reliability/confidence plots, and bootstrap confidence intervals.
  Temperature fitting must use validation logits only.
- Primary final reports and secondary/diagnostic reports must be labeled
  clearly, and held-out test metrics must never be used to switch the primary
  report after evaluation.
- For a new check or fresh experiment request, previous raw notebook outputs are
  context and guardrails by default, not the main source for narrowing the new
  search. Use previous raw outputs heavily only when the request explicitly asks
  to rely on prior runs/results/raws.
- If the request is a fresh Optuna check and no narrower continuation is
  requested, use the largest sensible validation-only search space for the
  selected task/model family. Keep common-sense limits: one named model family
  or fusion mode per study, no held-out test use, plausible runtime, fixed split
  policy, fixed EC depth per EC study, and only features that are available or
  deliberately prepared.

Statistical methodology:

- Model and hyperparameter selection must be validation-only.
- Single-split validation is useful for screening but should not be the final
  promotion criterion when Stage 6 grouped-fold confirmation is available.
- Stage 6 comparisons should use shared validation units and shared seed lists
  when configured. Stage 6B then uses fold-level paired confidence intervals,
  rare-class recall protection, and predeclared tie-breakers for tie-band cases
  before promoting a candidate.
- Stage 6B promotes a configuration and produces the final full-train refit,
  not a held-out-test-ready score. The final-training run cannot change model
  family, hyperparameters, feature set, split policy, fixed final-refit seed,
  epoch/checkpoint rule, calibration rule, ensemble membership, or primary
  report based on test data.
- Calibration, temperature scaling, ensemble membership, thresholds, and primary
  report choice must be fixed from validation evidence before Stage 7.
- Stage 7 reports uncertainty, calibration, and diagnostic views after the
  held-out test is opened, but those reports must not change the selected
  configuration.

Limited-compute fallback:

- Debug and medium validation runs may be used to identify promising directions,
  but they are provisional unless they satisfy the relevant playbook decision
  gate.
- If compute is insufficient for the full serious route, stop at a labeled
  validation-only result instead of launching Stage 7 from incomplete evidence.
- A final held-out report still requires one fixed validation-selected
  configuration, a frozen Stage 6B final-training run derived from it, and the
  one-shot Stage 7 policy.


---

## 7) Baseline-First Model Comparison Policy

Before testing complex fusion models or cross-task learning, establish clean
standalone baselines for each primary task.

The initial baseline family set for both metal target formulations and for EC
depth-1 prediction is:

1. Only-GVP
2. Only-ESM
3. GVP + graph-level late ESM fusion

Use this same limited family set for the first EC-primary auxiliary-learning
comparison. Keep its single-task and auxiliary variants matched within one
family. Do not introduce early fusion, node-level late fusion, hybrid fusion,
cross-modal attention, predicted-metal conditioning, or cross-task attention
into that first relationship experiment.

### Separate metal architecture investigations

The broader metal-only architecture campaign may screen early fusion alongside
the three standalone baselines in a bounded Stage 2B comparison. Additional
complexity is conditional on the declared budget and validation priority gates.
That architecture campaign is scientifically separate from the initial
metal-EC auxiliary experiment and must not be used to smuggle extra complexity
into its comparison.

`GVP + early fusion` is a supported preset. It is required in the controlled
fusion-position comparison below. The bounded architecture pilot provides its
manual recipe in `docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`; a dedicated serious
early-fusion HPO block remains absent. Do not assign it the budget of another
fusion stage or describe the pilot as a serious HPO campaign.

### Required controlled metal-model comparison matrix

The publication-facing metal campaign must answer all four questions below
before making the corresponding scientific claims:

1. **Which training target best supports the four-class endpoint?** Compare
   direct four-class training against six-class training followed by
   deterministic collapsed-four evaluation across the three initial baseline
   families.
2. **Where should ESMC enter the structural model?** Compare GVP + early ESMC
   fusion, GVP + graph-level late ESMC fusion, and GVP + hybrid early-and-late
   ESMC fusion.
3. **Which input modality contributes useful information?** Compare the
   `Only-ESM` preset using ESMC embeddings, Only-GVP, and a predeclared
   combined GVP + ESMC model.
4. **Do RING interaction edges help GVP?** Compare the same GVP-capable
   configuration with radius-only edges and with radius + RING edges. At
   minimum this comparison must be completed for Only-GVP. If the proposed
   final combined GVP + ESMC model uses RING, repeat the same matched RING
   ablation in that combined architecture.

These are controlled ablations, not a ranking of unrelated historical maxima.
For the target-formulation comparison, the label scheme and output dimension
are the intended differences; compare both arms on the common four-class
validation view while preserving the six-class arm's native metrics. For the
other three comparisons, keep the direct four-class label scheme fixed. Within
every comparison, keep the dataset, ESMC model and embedding coverage, GVP
backbone, non-target features, training budget, grouped folds, and active model
seeds identical wherever the question permits. Give separately tuned families
comparable HPO budgets, then compare their frozen candidates on the shared
Stage 6 fold/seed grid. Use paired bootstrap confidence intervals over shared
folds and four-class rare-recall protection. A numerical difference from one
split or one selected Optuna trial is not evidence of a significant advantage.

The current completion state and evidence gaps for this matrix belong in
`EXPERIMENT_STATUS.md`; exact runnable blocks and stage mapping belong in
`docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`.

Complex fusion modes include:

- node-level late fusion
- hybrid fusion
- cross-modal attention

The comparison should be sequential across model families, not a free search
over every architecture at every stage:

1. Tune and validate the simplest relevant model first.
2. Select a stable validation-best anchor from multiple seeds where possible.
3. When deliberately continuing from an anchor, carry forward shared settings
   from the simpler anchor when adding one new source of complexity. Shared
   settings include the split policy, epoch budget, Stage 6 fold plan, graph
   radius, GVP capacity, class-weighting policy, and validation selection
   metric.
4. When the user instead asks for a new check or fresh Optuna sweep, do not
   over-constrain the search to prior raw outputs. Search broadly within the
   selected model family/fusion mode, while keeping the validation/test and
   runtime safeguards above.
5. Move to the next more complex model only when validation evidence justifies the added parameters.

For metal GVP/ESM fusion, the advanced-fusion order should be:

1. Node-level late fusion after the late-fusion baseline is stable.
2. Serious hybrid HPO only after early or late fusion shows useful validation
   signal. The separately bounded single-GPU screen includes hybrid directly.
3. Cross-modal attention last, starting with a narrow one-layer configuration, because it has the most tuning degrees of freedom and the greatest overfitting risk.

`simple_gnn_esm` should be treated as an auxiliary architecture ablation, not the main next step in the best-pipeline search. Use it after the GVP and ESM baselines are stable when the question is whether GVP vector geometry is actually helping compared with a simpler scalar graph model.
It is supported by the notebook/model stack, but it is not a required metal HPO
stage unless the metal playbook adds a canonical executable block for it.

For each standalone task:

- compare models using validation metrics first
- select checkpoints using validation metrics
- evaluate the selected model once on the held-out test set for final reporting

The goal is to avoid adding complex architecture before proving that it
improves over simple baselines. A positive metal-only architecture result is
not evidence that multi-task learning helps EC, and an exploratory joint result
is not a substitute for a matched standalone comparison.

---

## 8) Data Leakage and Split Policy

This section preserves the intended scientific policy. Actual dataset
availability, overlap counts, and historical access are recorded in
`docs/DATASETS.md`. The legacy non-overlap test was historically evaluated in
seven early runs, so “historically trusted” below must not be read as pristine
or unopened. Those test values are not eligible current selection evidence.

> **Primary final-test route: unresolved scientific decision required before final reporting.**

This documentation update does not designate a replacement test. Until a
separate scientific decision resolves the route, no named PinMyMetal split is
the current primary final held-out test. The legacy non-overlap split remains a
historical reference and is not pristine. Current materialization and bundle
availability are recorded in `docs/DATASETS.md`.

The metal Colab notebook currently defaults `DATASET_NAME` to the exact
PinMyMetal split for new serious metal-validation workflows, and the playbook
owns that executable default. That default changes the external dataset split
used by the notebook; it does not select a final-reporting route or weaken the
held-out-test policy below. Exact-split results must remain labeled as
exact/possibly-overlapped when train/test PDB-ID overlap exists.

Named split definitions, exact counts, construction evidence, and current
availability are maintained once in `docs/DATASETS.md`. The policy distinctions
remain: exact PinMyMetal may overlap; non-overlapped and harsh variants are
zero-overlap constructions with different assignment rules; Common-PDBID 70/30
is a custom comparison rather than an automatically selected final test.

For the EC-number classification task:

- No current primary final held-out route is designated. Resolve it in a
  separate scientific decision before Stage 7.
- The exact PinMyMetal split should not be used as the final EC held-out test split if train/test structures overlap.
- EC supervision is structure/protein/chain-level even when extraction creates multiple separated metal-pocket samples for the same structure. EC cross-entropy should use group weighting, by default at `structure_id`, so such structures are not over-counted; this does not divide by raw metal atom count and does not downweight true multinuclear pockets.

For the metal-type classification task:

- No current primary final held-out route is designated. Resolve it in a
  separate scientific decision before Stage 7.
- The exact PinMyMetal split may be kept as an optional secondary metal-testing mode.
- If the exact PinMyMetal split is used for metal testing, the result must be clearly labeled as using the exact/possibly-overlapped split.
- Metal results from the exact/possibly-overlapped split should not be presented as the main final held-out result if train/test overlap exists.

For shared-encoder metal-EC experiments:

- Construct train/validation/test membership across the union of all metal and
  EC label sources before training.
- If a protein or structure group is held out for metal or EC, exclude it from
  every shared-encoder training loss, even when a label for the other task is
  available.
- Keep site/pocket metal labels nested under their protein/structure group so
  repeated pockets cannot cross a primary-task boundary.
- Apply the same group assignments, active fold definitions, and eligible
  sample rules to the standalone and auxiliary variants being compared.
- Treat any cross-task overlap as leakage and invalidate the affected
  comparison until it is rebuilt.

The code and/or result summary files should clearly record which split was used:

- non-overlapped PinMyMetal split
- exact PinMyMetal split
- any other custom split

If the exact PinMyMetal split is used as an optional metal-testing mode, the output summary should explicitly warn that this split may contain train/test overlap and should be interpreted only as a secondary/reference result.

Before final training/evaluation, validate train/test overlap by:



- full structure filename
- PDB ID
- preferably PDB-chain or pocket ID when available

The held-out test set must remain separate from model selection.

Use only validation or cross-validation for:

- checkpoint selection
- hyperparameter choices
- model architecture choices
- fusion-mode choices
- auxiliary-loss weights and joint-versus-single-task decisions
- feature and conditioning choices
- classification thresholds
- temperature or calibration-method choices
- ensemble membership or ensemble weighting

Use the held-out test set only for final reporting of selected models.
