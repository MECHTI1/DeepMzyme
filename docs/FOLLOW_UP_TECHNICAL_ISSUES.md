# Follow-Up Technical Issues

This register contains verified implementation/documentation problems found
during the documentation/provenance cleanup. Resolved items remain here as an
audit trail; open items still require separate authorization.

It is not the current-status document and does not change scientific policy.
Each item requires a separate, explicitly authorized technical or scientific
task.

## TECH-001 — Stage 7 can fall back when Stage 6B evidence is absent

**Status:** Resolved 2026-08-20

**Historical observed behavior**

The notebook's primary workflow value is
`FINAL_TEST_WORKFLOW = "evaluate_stage6_selected_candidate"`. Its own Markdown
and code state that it loads
`stage6b_selected_final_refit_candidate.json` when present, but otherwise falls
back to `stage6_selected_final_candidate.json`.

Audit evidence in the serialized notebook:

- lines around 13,190–13,204 describe the fallback;
- lines around 13,230–13,274 repeat it in configuration comments;
- Stage 7 source-discovery code searches both Stage 6B and Stage 6 artifacts;
- accepted workflow values are
  `evaluate_stage6_selected_candidate` and
  `exploratory_evaluate_all_stage6_ranked_candidates`.

**Conflicting documentation/policy**

`Plan.md`, `AGENTS.md`, and the metal playbook describe Stage 6B final
full-train refit as the required bridge between Stage 6 selection and reportable
Stage 7 evaluation.

**Risk**

A user may treat a direct Stage-6 checkpoint evaluation as the primary final
report even though the documented policy expects a completed/frozen Stage 6B
refit.

**Resolution**

The canonical primary Stage 7 workflow now accepts only
`evaluate_stage6_selected_candidate` and validates artifact semantics, not the
filename alone. It requires a completed/reused Stage 6B full non-test refit,
matching Stage 6 and Stage 6B decision IDs, source `run_config.json` and
`run_metadata.json`, the frozen checkpoint, no refit-time test evaluation, and
no existing source `test_report.json`. Missing or explicit Stage 6-only JSON is
hard-blocked before held-out input paths or inference are reached.

The all-ranked-candidate primary-test option was disabled instead of retained
as a peer workflow. Synthetic/static smoke checks cover valid Stage 6B evidence,
Stage 6-only rejection, missing-Stage-6B fallback rejection, already-tested
source rejection, and the single accepted workflow value.

## TECH-002 — EC playbook assignments do not match the notebook surface

**Status:** Open

**Standalone baseline reconciliation:** The opening Stage 0–2B block in the
EC playbook now uses current notebook controls and the generic training CLI.
Synthetic expansion tests cover all three families in smoke/baseline modes,
with EC1 group-level selection, group-count class weights, and launch/test
switches off. This is execution preparation, not measured EC model evidence.
Legacy HPO controls and the complete EC Stage 6/6B/7 migration remain open;
the opening recipe explicitly excludes those blocks.

**Pre-remediation observed behavior**

Static comparison found EC playbook assignments absent from the current
notebook assignment surface:

- `CONFIRM_ONE_SHOT_POLICY`
- `OPTUNA_BATCH_SIZES_CSV`
- `OPTUNA_EDGE_HIDDEN_VALUES_CSV`
- `OPTUNA_EDGE_RADIUS_VALUES_CSV`
- `OPTUNA_ESM_FUSION_DIM_VALUES_CSV`
- `OPTUNA_GVP_LAYERS_VALUES_CSV`
- `OPTUNA_HEAD_MLP_LAYERS_VALUES_CSV`
- `OPTUNA_HIDDEN_S_VALUES_CSV`
- `OPTUNA_HIDDEN_V_VALUES_CSV`
- `OPTUNA_WEIGHT_DECAYS_CSV`

Current audited occurrences are at lines 449–450, 527–528, 606–614, 664–673,
and 789/808 of the annotated playbook. The variable names themselves are the
stable evidence if later documentary insertions shift line numbers.

The EC Stage 7 examples use:

- `FINAL_TEST_WORKFLOW = "preview_only"` at line 787;
- `FINAL_TEST_WORKFLOW = "evaluate_selected_checkpoint"` at line 806.

The current notebook accepts only
`evaluate_stage6_selected_candidate`.

The EC playbook also describes seed-repeat Stage 6 without the current
metal-style named Stage 6B bridge.

**Conflicting documentation/policy**

The EC playbook presents copy-paste blocks, while the notebook rejects or does
not expose some of their controls. Existing EC budgets, ranges, label-depth
progression, weighting intent, and contrastive-loss intent remain scientifically
important and were preserved.

**Risk**

An EC block may fail immediately or silently fail to define the intended search
space. A superficial variable rename could also change scientific execution.

**Future dedicated fix**

Audit every EC block against actual notebook command expansion and training CLI
semantics. Design the EC Stage 6/6B/7 flow explicitly, then update notebook and
playbook together. Preserve a search-space migration record.

**Required future tests**

- Static assignment-name validation for every block.
- Dry-run command expansion for every EC stage and label depth.
- Verification that each declared range reaches the intended CLI field.
- Persistent-study compatibility checks.
- Validation-only Stage 6 behavior.
- Stage 6B/final-test safety tests after a design decision.

The cleanup added warnings only; it did not repair or modernize these blocks.

## TECH-003 — Notebook live values and notebook prose can disagree

**Status:** Open

**Observed behavior**

The audited live configuration cells currently contain:

- `TASK = "joint"`;
- `METAL_LABEL_SCHEME = "five_class"`;
- `RUN_MODE = "single"`;
- `MODEL_PRESET = "GVP + hybrid fusion"`;
- `DATASET_NAME = "CARE_task1_30_clusterRes30_train_test_metallo"`;
- `VAL_FRACTION = 0.18`;
- `SELECTION_METRIC = "task_default"`;
- `PREPARE_MISSING_ESM_EMBEDDINGS = True`;
- `REQUIRE_RING_EDGES = True`.

Notebook prose near the top and option tables still says, for example,
`PREPARE_MISSING_ESM_EMBEDDINGS = False` and calls both ESM preparation and
RING requirements false by default.

**Conflicting documentation/policy**

Stable docs previously copied a different mutable default snapshot. The cleanup
removed those external copies and now treats notebook cells as implemented
resume state, but it did not modify notebook prose.

**Risk**

A user may read a Markdown cell instead of the executable cell and misunderstand
what preparation or preflight will occur.

**Future dedicated fix**

Perform a notebook-only documentation consistency pass that derives displayed
defaults from the live cell or clearly labels all examples as non-live.

**Required future tests**

- Parse notebook JSON.
- Extract displayed defaults and executable assignments.
- Fail on mismatched values for safety-sensitive fields.
- Confirm no command expansion or training behavior changes.

No notebook cell was changed during cleanup.

## TECH-004 — Late-fusion Round-4 artifacts were removed

**Status:** Open — lightweight metadata recovered; checkpoint availability unresolved

**Observed behavior**

The Round-4 summary cited:

`docs/notebook_outputs/raw/GVP + late fusion/metal_late_fusion_optuna_top3_seedrepeat_50epoch_v1/`

The directory was absent before cleanup. Git history showed:

- commit `783acae` added 75 JSON artifacts and 30 checkpoint binaries;
- commit `20b4d64` removed them.

The cleanup restored all 75 JSON artifacts byte-for-byte and verified their Git
blob identities. It did not restore the 30 large checkpoint binaries.
`MISSING_CHECKPOINT_GIT_BLOBS.tsv` records their historical blob IDs, sizes, and
paths.

**Risk**

The configuration, split, trajectory, and selected-checkpoint provenance is now
portable, but exact checkpoint reuse is unavailable from the working tree.

**Future dedicated fix**

Decide whether checkpoints belong in an external immutable artifact store.
Record stable URLs/checksums without reintroducing large binaries to normal Git
history.

**Required future tests**

- Verify all 15 run configs/metadata records.
- Verify checkpoint artifact checksums against recorded Git blobs or external
  objects.
- Recompute the aggregate table from restored JSON.
- Check all summary links.

## TECH-005 — Hybrid Round-1 provenance is incomplete

**Status:** Open

**Observed behavior**

The copied Hybrid Round-1 output preserves LR/WD, metrics, trial IDs, run names,
and Drive paths, but not the complete architecture configuration or HPO search
space. Repository and Git-object searches found no corresponding
`run_config.json`, `run_metadata.json`, Optuna database, or study JSON.

**Risk**

The batch cannot serve as a fully reproducible anchor, and its metric is a joint
selection metric that is not directly comparable with metal-only results.

**Future dedicated fix**

Recover the named Drive artifacts if they still exist, copy lightweight configs
and metadata into raw evidence, hash them, and update the experiment index.
Otherwise retain `MISSING — recovery required`.

**Required future tests**

- Match run names and trial IDs between recovered configs and copied output.
- Recover complete model/search-space parameters.
- Verify task, dataset, label scheme, seeds, and selection metric.
- Confirm held-out evaluation remained disabled.

## TECH-006 — Final-dataset implementation and availability conflict

**Status:** Open; scientific decision required

**Observed behavior**

- `Plan.md` leaves the primary final route unresolved and records
  non-overlapped PinMyMetal only as a historical reference.
- The metal playbook's current common defaults use exact PinMyMetal.
- Exact PinMyMetal contains 177 overlapping PDB IDs.
- The notebook dropdown omits non-overlapped and harsh roots. The current
  bundle includes historical non-overlapped data, accessible through an explicit
  dataset-root override; harsh data is still unavailable.
- The historical non-overlap test was evaluated seven times and is not pristine.

**Conflicting documentation/policy**

No current dataset simultaneously satisfies every described availability,
overlap, historical-access, and final-reporting expectation.

**Risk**

A user could silently substitute exact PinMyMetal, misdescribe the historically
accessed test as pristine, or launch a final report without a resolved
scientific basis.

**Future dedicated task**

Make an explicit scientific decision about the primary final-test route,
disclosure language, bundle support, and any new split. Update Plan, DATASETS,
playbooks, notebook availability, and publication protocol together.

**Required future tests**

- Exact membership/overlap audit.
- Historical test-access audit.
- Bundle-content verification.
- Notebook dataset-resolution test.
- Predeclared final-report protocol review.
- Confirmation that no test value is used for selection.

Current documentation states:

> **Primary final-test route: unresolved scientific decision required before final reporting.**

The 2026-08-20 safety corrections did not designate a replacement, alter split
membership, or resolve this scientific decision. The canonical Stage 7 cell is
fail-closed while its internal primary-route status remains `unresolved`.

## TECH-007 — Smoke suite references a removed root document

**Status:** Resolved in source 2026-08-23; CI run still external

**Observed behavior**

Running the documented smoke command with the configured interpreter prints 37
passing checks, then fails in
`check_docs_do_not_use_broken_training_command()` with:

```text
FileNotFoundError: .../DeepMzyme/list_train_commands.md
```

The check loops over `README.md` and the removed root path
`list_train_commands.md`. The command reference was archived at
`docs/archive/workflows/list_train_commands_legacy.md`, but the smoke check was
not repointed. Because the exception stops the suite, the final optional
multi-metal granularity check is not reached.

**Risk**

The repository advertises a fast smoke command that cannot finish even when the
preceding implementation checks pass. A user can misinterpret the stale path as
a training failure, and later smoke checks receive no result.

**Implemented resolution**

`tests/smoke_checks.py` now checks the archived command-document path, preserves
the broken-command assertion, continues through all checks after individual
failures, reports optional local-data absence as an explicit skip, and returns
nonzero when any check fails. The same checks are exposed as isolated pytest
cases, and `.github/workflows/cpu-ci.yml` runs CLI help, pytest, and the
compatibility wrapper in the pinned CPU environment.

**Validation status**

The compatibility wrapper completed all 43 checks locally: 42 passed and the
final optional multi-metal check reported its intended data-based skip. Pytest
collected the same 43 isolated cases with the same result. The archived command
document passed the retained broken-command assertion. The external GitHub
Actions result is not available from this local change; the detailed local
verification record belongs in the remediation plan.

## TECH-008 — Interactive Drive mount blocks unattended CLI execution

**Status:** Open; operational workaround documented 2026-08-22

**Observed behavior**

The notebook's editable live configuration currently has
`MOUNT_DRIVE = True`. Its data setup cell calls
`google.colab.drive.mount("/content/drive")` whenever mounting is enabled or
the selected data source is `drive`.

That authorization is appropriate in an interactive browser notebook, but an
unattended `colab exec -f notebooks/DeepMzyme_training_colab.ipynb` can wait for
human input. The CLI's `colab drivemount` command is also interactive and is not
a headless repair.

**Risk**

A terminal user can mistake the wait for a failed download, hung kernel, or
training stall. Long automated setup becomes unreliable, and a CLI transport
timeout can obscure the real prompt.

**Current workaround**

Follow `docs/COLAB_GPU_RUNBOOK.md`: create the session with the CLI, use
`colab url` to attach a browser to the same VM/kernel, authorize Drive once in
the browser when persistence is required, and prevent a second mount attempt in
the run-specific editable configuration. For ephemeral smoke work, use the
Hugging Face data source with `MOUNT_DRIVE = False` and download artifacts
before teardown.

**Proposed implementation fix**

In a separately authorized notebook change, make interactive mounting an
explicit mode with a clear preflight failure for unattended execution. A
headless path should accept an already-mounted Drive or local/Hugging Face
output root without prompting. Coordinate notebook prose, live controls, and
the Colab runbook.

**Required future tests**

- Browser run with interactive Drive authorization.
- CLI planning run with Drive disabled and no prompt.
- CLI planning run against an already-mounted same-kernel Drive.
- Clear failure when persistent storage is required but unavailable.
- Confirmation that serious Stage 4/5 Optuna still requires persistent Drive
  SQLite and blocks incompatible study reuse.

No notebook cell was changed in the 2026-08-22 documentation task.

## TECH-009 — Environment specification and Colab PyTorch contract are incomplete

**Status:** Partially resolved 2026-08-23; local CPU contract implemented, GPU runtime checks remain

**Observed behavior**

Before the 2026-08-23 remediation, `src/requirements.txt` was a short
direct-dependency list. It did not define the Python version, CUDA wheel source, transitive resolution,
ESM/ESMC version, Optuna, NumPy, scikit-learn, or the complete
notebook/reporting environment. It pins `torch==2.5.1` for the local project
environment.

On an audited Colab G4 runtime, an unfiltered requirements installation resolved
that line to `torch==2.5.1+cu124`. Its compiled CUDA architectures stopped at
`sm_90`, while the assigned NVIDIA RTX PRO 6000 Blackwell Server Edition
required `sm_120`; GPU execution failed with `no kernel image is available for
execution on the device`. A fresh stock Colab build, `2.11.0+cu128`, included
`sm_120` and ran the audited workload. The A100 audit also succeeded with the
stock build.

The separately installed host CLI was audited as
`google-colab-cli==0.6.0` with `jupyter-kernel-client==0.15.0`; the documented
installation constrains `jupyter-kernel-client<1.0` for that CLI release. This
host tool environment is not part of `src/requirements.txt`.

**Pre-remediation risk**

- A fresh local installation cannot be reproduced exactly from the repository.
- Installing the local PyTorch pin in Colab can make an assigned GPU unusable.
- A future stock Colab image may change, so a hard-coded historical version is
  not a sufficient compatibility check.

**Implemented resolution**

`pyproject.toml` and `uv.lock` now define the Linux x86_64/Python 3.12 CPU
development/test environment. Test, reporting, and optional ESM dependency
groups are explicit. `requirements/colab-overlay.txt` is a separate managed
Colab overlay that omits PyTorch, and the notebook now installs that overlay
instead of filtering the local requirements file. `requirements/README.md`
documents that the existing workstation prefix is real but non-canonical and
that no separately locked local CUDA environment is supported yet.

Future training runs automatically record Python/platform/package versions,
PyTorch/CUDA/GPU details, exact invocation, commit/dirty state, summary-CSV
checksums, and optional dataset bundle ID/SHA256 in additive
`runtime_environment`, `source_control`, and `source_artifacts` objects. The
notebook passes its existing bundle values to every generated training command.

**Remaining runtime tests**

- Colab T4/L4/G4/A100 preflight that compares device capability to
  `torch.cuda.get_arch_list()`.
- Verification that the Colab install never replaces stock PyTorch unless an
  explicitly tested compatibility path is selected.
- Capture of exact versions in a serious-run metadata artifact.

Fresh CPU-lock reconstruction, CLI help, core dependency imports, reporting
dependencies, and the optional ESM `3.2.3` imports are recorded in the
remediation plan. Colab T4/L4/G4/A100 validation and a serious training-run
metadata capture remain runtime actions and were not performed by this
non-training remediation.

## TECH-010 — Four-class endpoint and paired metal target recipes are not reconciled

**Status:** Open; policy recorded 2026-09-14

**Bounded single-GPU implementation (2026-09-16):** the separate
[`metal_single_gpu_20h_v2` recipe](METAL_TRAINING_PIPELINE_PLAYBOOK.md#single-gpu-metal-campaign)
adds matched core four-/six-class learning-rate/capacity discovery, unconditional
early/hybrid coverage, top-two seed repeats, mixed diagnostics, bounded numeric
continuation, fixed larger late-fusion and late-five comparisons, a serial allocation ledger and
frozen exploratory grouped-fold comparisons. It keeps native checkpoint
selection and common-four comparison distinct. This does not certify the
retained serious paired HPO/Stage 6B/Stage 7 recipes or resolve final-test data
policy. Exact epoch resume remains unsupported: interrupted fits restart once,
linked to the original attempt, within the same cumulative allocation budget.
Implementation checks are not new GPU readiness or experimental evidence;
current state belongs to `EXPERIMENT_STATUS.md`.

**Bounded pilot reconciliation (2026-09-15):** The separately named
`metal_architecture_pilot_10h_v1` profile covers direct-four early/late and
unimodal screens, and matched four-/five-/six-class core-family screens before
an optional hybrid screen. It uses non-overlap train membership, native-six
eligibility, native `val_metal_balanced_acc` checkpoint selection in every arm,
and same-checkpoint collapsed-four reporting. Its exact recipe is at the start
of the metal playbook. The older Common70 standalone block remains explicitly
historical and retains its different six-class collapsed-four-selection rule.
Pilot implementation, smoke outcomes, and experimental completion must be
reported separately in `EXPERIMENT_STATUS.md`; this note is not run evidence.

Early fusion now has bounded manual recipes. A dedicated serious early HPO
block and reportable paired HPO/Stage 6/6B/7 recipes remain open; the new
single-GPU profile supplies exploratory paired grouped-fold execution only.
The historical pilot's conditional hybrid budget gate is not the serious
Stage 5E promotion gate and does not apply to the new initial hybrid screen.

**Feature-readiness qualification:** Cache-file coverage alone does not verify
measured external channels. The pilot requires a separately hashed,
training-only PROPKA overlay and checks its provenance rather than silently
accepting unavailable pKa defaults. Preserve the published bundle and track
overlay preparation/validation outcomes in the campaign evidence. No feature
repair may silently change the matched cohort.

**Completed pilot repair; broader cache audit remains open (2026-09-15):**
The isolated overlay is complete and verified for all 1,304 non-overlap
**training** structures. The underlying PROPKA path now handles compact
wide-residue tokens and insertion-code alignment. The final overlay refresh
covered 46 wide-number structures plus `3q6v`; all 47 were rechecked before
freezing the overlay. File hashes, structure identities, and unchanged geometry
passed the all-file audit. The original v12 caches were not rewritten.
Other legacy caches were not audited or repaired by this scoped work; the
parser fix does not retroactively correct cached values. Missing pKa masks
remain explicit for non-titratable/incomplete residues rather than being
described as fully measured channels. See the
[preparation audit](notebook_outputs/raw/metal_architecture_pilot_20260915/preparation/feature_overlay_audit.json),
[manifest](notebook_outputs/raw/metal_architecture_pilot_20260915/preparation/feature_overlay_manifest.json),
and [pilot summary](notebook_outputs/summaries/summary_metal_architecture_pilot_20260915.md).

**Standalone baseline reconciliation:** The opening metal Stage 0–2B block
now covers direct-four and six-trained/collapsed-four arms for all three
initial families, with separate identities and a shared metal-site-symbol
stratification mode. A native-six eligibility requirement keeps mixed sites
that become eligible only after merging out of both arms. Synthetic tests
verify this eligibility boundary and equal validation membership under
target relabeling and current notebook/CLI command expansion. This does not
establish model evidence or certify later HPO/Stage 6/6B/7 paired recipes.

**Observed state**

- `src/label_schemes.py` implements `four_class` as the alias for
  `merge_fe_class_viii`, mapping Mn, Cu, Zn, and Fe+Co+Ni to Class VIII.
- `Plan.md` now designates the four-class endpoint as primary and requires a
  matched comparison between direct four-class training and six-class training
  with collapsed-four evaluation.
- The raw source default still resolves to the six-class
  `split_all_metals` scheme.
- The metal playbook's retained common recipe still assigns
  `METAL_LABEL_SCHEME = "six_class"`.
- The audited notebook live value recorded in TECH-003 is a separate
  `five_class` resume value.
- Historical performance anchors remain labeled six-class. The subsequent
  [completed bounded pilot](notebook_outputs/summaries/summary_metal_architecture_pilot_continuation_20260915.md)
  adds direct-four baselines and matched target-formulation screening across
  the three core families, with selected-LR repeats on one validation split.
  This initial evidence does not certify the later grouped-fold/HPO recipes
  or establish target superiority.

**Risk**

A future run may be launched under a stale six-class or five-class value and
then described as direct four-class training. Conversely, an agent may run only
the direct-four arm and omit the now-required six-class-trained/collapsed-four
challenger, or treat unmatched historical six-class evidence as that challenger.

**Future dedicated fix**

Reconcile the metal playbook's exact Stage 0-7 recipes and the notebook launch
surface together. Define paired direct-four and standard-six-class arms for
Only-GVP, Only-ESM, and GVP + graph-level late fusion. Give them separate study
and run identities, matched development splits/folds/seeds/features/budgets,
and a common collapsed-four comparison table while retaining native six-class
metrics for the six-class arm. Coordinate active-class metrics, rare-class
gates, and Stage 6/6B/7 provenance checks. Preserve historical and five-class
routes under clearly separate names. Do not change only one default or reuse an
incompatible Optuna study.

**Required future tests**

- Static verification that each direct arm resolves `four_class` to
  `merge_fe_class_viii` with exactly four outputs and each matched challenger
  resolves `six_class` with exactly six outputs.
- Planning/dry-run checks for all affected stage blocks without launching
  training or held-out evaluation.
- Separate study/run identity checks across four-, five-, and six-class tasks.
- Matched-design checks for the required direct-four and
  six-class-trained/collapsed-four arms in all three initial metal families.
- Paired comparison checks using direct-four active metrics and the six-class
  arm's collapsed-four metrics on identical validation units.
- Active-scheme metric, class-weight, rare-recall, and report-column checks.
- Confirmation that collapsed reporting from a six-class model cannot be
  labeled as direct four-class training.

The original policy-only update did not change executable recipes. The later
pilot adds a separate execution profile while preserving those retained
recipes and historical evidence. Later-stage reconciliation remains open.

### Remaining PinMyMetal reference-benchmark integration

Dual-protocol validation helpers do not establish an integrated, scientifically
certified final benchmark. A later dedicated change must wire frozen refit
identities, auditable original-site matching, overlap labels, the shared-test
access history, and one-shot reporting into the notebook/reporting route.
The primary final-test route is still unresolved. The bounded pilot reads
non-overlap training membership only and does not implement or execute that
final benchmark.

## TECH-011 — Controlled EC-primary auxiliary-learning protocol is not certified

**Status:** Open; scientific recipe and safeguards required

**Minimum implementation update:** `--controlled-ec-auxiliary` now enforces
the EC-primary, fixed-weight, direct-four, graph-level late-fusion intersection
comparison with validation-only selection. Ordered retained-example identities
and counts are saved for pair verification. A zero EC task weight now also
disables EC contrastive supervision in both model loss paths. See the
[minimum CLI controls](EC_TRAINING_PIPELINE_PLAYBOOK.md#minimum-cli-controls-for-a-matched-intersection-comparison).
The notebook recipe, actual dataset/holdout certification, multi-source
exclusion and experimental evaluation remain open; no partial-label loader
redesign is included.

**Observed state**

- The current model path can form one shared pocket representation and send it
  to independent metal and EC heads. The losses can both update shared
  parameters; neither head's prediction is fed into the other head.
- The current joint loader uses one structure root, requires both targets for a
  joint sample, and applies one grouped train/validation split to that loaded
  pocket set.
- Historical joint Hybrid and Hybrid+RING results are exploratory and were not
  a matched EC-only versus EC-plus-auxiliary-metal comparison.
- No exact EC-playbook recipe currently defines that first controlled
  auxiliary experiment.
- The current single-root path does not by itself certify cross-task exclusion
  if future work combines separate metal and EC label sources or task-specific
  holdouts.

**Risk**

An unmatched joint result could be presented as proof of multi-task benefit, or
a protein held out for one task could enter shared-encoder training through the
other task when distinct label sources are combined.

**Future dedicated fix**

Define one EC-primary auxiliary recipe in the EC playbook after its existing
notebook compatibility issues are reconciled. Compare EC-only with EC plus
auxiliary metal in one of the initial standalone families (Only-GVP, Only-ESM,
or GVP + graph-level late fusion), using matched eligibility, encoder capacity,
features, folds/seeds, budget, and EC selection metric. Build group membership
across the union of label sources and fail closed on cross-task overlap. Do not
add a predicted-metal cascade or advanced interaction architecture to this
first comparison. The reverse metal-primary experiment remains optional.

**Required future tests**

- Dry-run proof that the EC-only and auxiliary commands differ only in the
  declared auxiliary-task controls.
- Group-level overlap checks across every label source and primary-task split.
- Verification that repeated pockets receive EC group weighting and stay in
  one protein/structure split unit.
- Validation-only selection and HPO checks with no held-out paths or reports.
- Independent metal/EC head-output checks and explicit negative-transfer
  reporting.

No auxiliary experiment, association analysis, training, or held-out access was
performed while recording this issue.

## TECH-012 — Colab session access loss with valid authentication status

**Status:** Partially resolved 2026-09-15. Owned-session teardown and
cross-session recovery are implemented and tested; the original connection-loss
cause remains under investigation. Both allocations are verified stopped;
the authorized continuation and both bounded pilots are complete.

During the bounded metal pilot, CLI access returned mixed 404/401 responses at
03:53:09 UTC and removed the local session mapping, while the original runtime
remained server-listed. `whoami` still reported OAuth2 credentials with Colab
scope and about 59 minutes of validity. Subsequent CLI inspection established
that `whoami` refreshes OAuth before reporting, so that response does not
certify the token state before the failure. CLI proxy-token lifetime is a
plausible mechanism under investigation, not a proven cause. Neither the
mixed errors nor the later authentication report establish an upstream root
cause.

Further training stopped being orchestrated under the 401 stop rule. Cleanup
verified ownership using the original CLI creation-history entry, restored
only that owned session record with empty runtime credentials, then performed
a named stop. During that cleanup, no reauthentication, new runtime allocation,
or training restart occurred. Stop succeeded at 03:56:01.801127 UTC and a subsequent server session
listing found no active sessions. The watchdog exited after the stop marker.

The host watchdog originally knew only the local session name. Once the CLI
pruned that mapping, its initial named stop could not find the session. Cleanup
therefore required restoring the exact owned mapping from creation history.
The updated host controls retain ownership-verified endpoint/name information
and retry teardown with a five-minute margin. Eleven offline owned-teardown
cases passed. These tests verify guarded behavior under simulated conditions;
the continuation's subsequent teardown also passed actual server verification.

Seven model smokes and three full A1 runs were already archived and verified.
Late-fusion `attempt_012` was last seen at epoch 3 and is now reconciled as
interrupted. Its final artifacts remain unverified; its exact attempt timing
is unknown and was not fabricated. It cannot count as a completed result or
architecture rejection. The entire first allocated interval, including the
bootstrap failure and cleanup, remains charged at 3,827.969 seconds.

**Verified continuation:** `scripts/colab_metal_pilot_resume.py` restored the
frozen original source/manifest and verified all 1,389 retained pockets and
unchanged feature content after fresh cache-timestamp auditing. Its
`cross_session_recovery/readiness.json` passed; seven original smoke runs and
three full runs were reverified. New session
`deepmzyme-metal-geometry-20260915` began at epoch `1789456819.0405653` under
the same cumulative budget. Linked late-fusion retry `attempt_013` completed
and its archive was verified locally and in Drive. This recovery followed explicit
authorization; restoring a mapping alone does not authorize a new allocation
or training.

**Proxy-credential mitigation:** Inspection of the installed CLI confirmed
that ordinary session/status reads do not refresh its stored runtime proxy
credentials. The owned-session helper now refreshes the exact endpoint's
proxy URL/token from authenticated assignment metadata before expiry,
preserving session and kernel identities. Fourteen offline checks passed,
and scheduled refreshes succeeded on the completed continuation. This
mitigates a verified credential-lifetime limitation; it does not establish
the cause of the earlier mixed 404/401 failure. No interactive OAuth
reauthentication is part of this helper, and a real 401/403 still stops
orchestration under the existing rule.

The combined geometry/recovery checks passed 134 focused tests in 30.37
seconds, alongside the separate 11 owned-teardown cases. The new explicit
geometry controls preserve legacy model outputs bitwise in the tested
compatibility comparison. These are implementation checks, not evidence that
a geometry arm improves validation performance.

**Verified final closeout:** All 30 original full fits, 15 geometry full fits,
12 smokes, and 15 selected-checkpoint geometry prediction exports were
verified before teardown. The final capture preserves both terminal campaign
states and was verified locally and in Drive. The owned continuation was
stopped at epoch `1789472254.5176592`; the server then reported no active
sessions. Both closed allocation intervals total 19,263.446 seconds
(5.350957 hours), including the full interrupted first allocation. The
post-stop package separately binds the actual stop receipt, allocation
ledger, and final capture receipts. The host watchdog has exited.

**2026-09-16 recurrence:** During `metal_single_gpu_20h_v2` profiling, the
provider session mapping was again lost while a RING-on retry was in flight.
The attempt was conservatively reconciled as interrupted from the last
observed launch time, the exact owned endpoint was stopped, and a fresh
provider listing showed no active sessions. The interrupted attempt did not
become timing evidence. A later fresh session completed its mandatory hardware
anchor and was also provider-verified stopped. This recurrence strengthens the
need for diagnosis but still does not establish OAuth or proxy-token expiry as
the cause. See the
[v2 profile summary](notebook_outputs/summaries/summary_metal_single_gpu_20h_v2_profile_20260916.md)
and its [stop receipts](notebook_outputs/raw/metal_single_gpu_20h_v2_profile_20260916/README.md).

**Remaining work:** diagnose the original CLI/runtime mapping-loss path and
distinguish OAuth credentials from runtime proxy credentials. Do not describe
proxy-token expiry as
the established cause or treat a refreshed `whoami` response as evidence of
the pre-failure authentication state. Preserve both allocation intervals and
linked-attempt identities in subsequent reports.

Evidence: [verified stop receipt](notebook_outputs/raw/metal_architecture_pilot_20260915/closeout/session_stop_receipt.json),
[last remote status](notebook_outputs/raw/metal_architecture_pilot_20260915/closeout/last_remote_status.json),
[closed allocation ledger](notebook_outputs/raw/metal_architecture_pilot_20260915/closeout/allocation_ledger_closed.json),
and [first-allocation summary](notebook_outputs/summaries/summary_metal_architecture_pilot_20260915.md).
The continuation adds its [verified stop receipt](notebook_outputs/raw/metal_architecture_pilot_20260915/continuation/closeout_allocation2/post_stop/session_stopped.json),
[complete allocation closeout](notebook_outputs/raw/metal_architecture_pilot_20260915/continuation/closeout_allocation2/post_stop/post_stop_closeout.json),
[sanitized proxy-refresh receipt](notebook_outputs/raw/metal_architecture_pilot_20260915/continuation/closeout_allocation2/post_stop/runtime_proxy_refresh.json),
and [completed continuation summary](notebook_outputs/summaries/summary_metal_architecture_pilot_continuation_20260915.md).

## TECH-013 — Serial v2 profiling integration defects

**Status:** Resolved 2026-09-16; measured campaign admission still rejected

Three integration defects appeared while resuming the frozen
`metal_single_gpu_20h_v2` profiling stage:

- the controller created `execution.log` before launching the trainer, while
  the trainer treated that controller-owned file as foreign prelaunch output;
- generated RING-on commands enabled and required RING edges without passing
  the frozen bundle's `RING_features` directory; and
- after the failed RING-on probe and its single interrupted retry, the
  operations dispatcher kept selecting the exhausted probe instead of the
  mandatory fresh-session timing anchor.

The trainer now permits only the known controller-owned `execution.log` in
this prelaunch path. RING-on run generation passes the explicit cache root.
The dispatcher skips an operations probe after two non-completed attempts so
a later session can run its fresh anchor; measured forecasting still fails
closed if the missing probe is required for compatible pricing. Focused serial
campaign tests cover all three cases and pass 165 tests.

These fixes do not change the frozen scientific recipes or override cost
admission. The measured discovery and host-reconciled operations budgets both
fail independently, and the repaired RING path was not re-profiled under the
closed campaign. No 50-epoch or held-out work was run. Evidence is in the
[measured profile summary](notebook_outputs/summaries/summary_metal_single_gpu_20h_v2_profile_20260916.md).

## TECH-014 — Serial campaign orchestration and preparation overhead

**Status:** Partially implemented 2026-09-17; local controller tests passed,
live transport and GPU performance remain unvalidated.

The audited continuation allocation spent 30.56 minutes allocated and 7.39
minutes inside its two probes and one full fit. The 23.18-minute remainder
includes setup, transfer, verification and orchestration; this is not a measured
GPU-utilization percentage. The full fit's profile separately records 91.66
seconds of preparation and 143.76 seconds in its training region.

Constraints recorded before the Release 1 implementation:

- `workflow.execute()` owns one fit per call. A maintained automatic host
  supervisor is needed to replace the temporary transport operator and remove
  conversation turns from the routine next-fit path.
- `runtime.authorize_budget()` requires all allocations closed, and the host's
  worker receipts bind the authorization digest frozen during preparation.
  Increasing that ceiling currently requires a fresh session; silently editing
  an active host configuration is not a valid optimization.
- Artifact/state persistence requires independent byte readback. Batching the
  existing handshake is possible; eliminating re-uploads requires a supported,
  tested acknowledgement protocol.
- `prepare_run()` rebuilds preparation per process; caching opportunities need
  phase timing, exact identity keys, equivalence checks and fold-local fitted
  preprocessing. The existing pinned-memory and optional AMP/loader-worker
  controls must not be described as absent.

The [GPU runtime efficiency plan](GPU_RUNTIME_EFFICIENCY_PLAN.md) owns the
prioritized work, performance targets, provenance boundary and failure tests.
The [current status](../EXPERIMENT_STATUS.md) owns the user-requested pause and
allocation accounting. Durable pause/lease controls and exactly-once accounting
have local failure tests; session 9's verified host-only interval was reconciled
without clearing the pause. The maintained queue passed CPU failure tests.
The first release preserves scientific settings and the shutdown, persistence
and held-out safeguards. Runtime measurement and later preparation caching
remain distinct gates.

## TECH-015 — Stage 6 candidate imports dropped scientific configuration

**Status:** Resolved in source 2026-09-17; focused regression tests passed.

The generic Stage 6 command builder and summary/Optuna CSV candidate import
omitted `split_stratify_by`, `metal_eligibility_scheme`, `shell_role_source`,
`site_geometry_features`, and `ec_class_weight_unit`. Falling back to defaults
could change cohort, graph semantics or weighting during confirmation.
Both import and command paths now retain these fields. Tests expand JSON,
summary-CSV and Optuna-CSV candidates through the real CLI parser. Existing
frozen campaign commands and historical evidence remain unchanged.

## TECH-016 — Validation-only checkpoint replay and fixed-vocabulary bin metrics

**Status:** Implemented and twelve validation exports reproduced 2026-09-17.

The saved-checkpoint training entry point is held-out-specific. The separate
`export_validation_predictions.py` reconstructs validation inputs with verified
frozen source, exact saved membership and saved normalization. It avoids
training preparation and emits independent prediction/logit sidecars only after
whole-validation metric agreement. No original run is rewritten.

The remoteness reporter requires a pre-prediction counts freeze and artifact
bindings. It retains all task classes in confusion matrices and leaves full-task
BA/macro-F1 undefined when a true class is absent. Existing training metrics are
unchanged to preserve historical checkpoint selection. See the
[remote addendum](REMOTE_HOMOLOGY_ADDENDUM.md) for interpretation and remaining
full-protein/support limits.

## TECH-017 — Ion-unit loading of the Zenodo PinMyMetal reconstruction

**Status:** Resolved for the opt-in source-cohort path 2026-09-25; the legacy
summary-matching ion path is unchanged and remains affected.

Every `{pdbid}__chain_{X}__EC_0.0.0.0.pdb` file of the reconstruction is a hard
link to the full PDB entry. Summary-key ion matching (`pdbid`, EC,
`chain_resi`) ignores which file an ion came from and ignores insertion codes,
so an ion can be ingested once per chain alias of its entry, and
`load_allowed_site_metal_labels()` silently overwrites duplicate keys.
`collect_structure_residues_and_metals()` traverses every model, and features are
attached to every pocket of a structure before ion filtering, so a strict ESM
requirement also fails on unused remote chains. Per-chain ESM files cover only
the file's named chain, although 32 % of retained ion contexts include another
chain.

The campaign's `--source-cohort-csv` path binds each source UID to one atom
(canonical file, content hash, model 0, chain, residue, insertion code, atom,
altloc, coordinate), builds one example per binding from model 0 only,
attaches features to retained ions only, and fails on any missing or repeated
binding. `scripts/run_zenodo_pmm_exact_5fold_cv.py` and the generalized
runner's legacy (non-campaign) path still use summary matching, five-class
defaults, `--allow-missing-esm-embeddings` and per-fold test evaluation; do not
use them for this campaign.

## TECH-018 — Legacy 5-fold runner summary selects independent maxima

**Status:** Open for the legacy runner path; not used by the PMM campaign profiles.

`summarize_single_fold()` in `scripts/run_metal_5fold_cv.py` reports the maximum
native and the maximum collapsed-four validation BA from possibly different
epochs, and `compute_oof_cv_metrics()` averages those scalar maxima under an
"OOF" name. Neither describes one selected checkpoint or real out-of-fold
predictions. The campaign path reports both views from the same selected
checkpoint (`selected_checkpoint.json`, `val_predictions.csv`) and pools actual
UID-keyed predictions (`src/benchmarking/pmm_ion_analysis.py`).

## TECH-019 — Stale explicit-membership loader test

**Status:** Open (pre-existing on `main`, found 2026-09-25).

`tests/test_explicit_membership.py::test_outer_loader_not_deserialized` patches
`training.data.load_structure_pockets`, which `training/data.py` has not
imported since parallel parsing moved per-structure loading to
`training/parallel_loading.py` (commit `c3728b0`). The test fails with
`AttributeError` before exercising its guard. The guard itself is covered by
the other explicit-membership tests; the stale test needs retargeting to
`training.parallel_loading.load_structure_pockets`.

## TECH-020 — Greedy k-fold assignment concentrates large groups in fold 0

**Status:** Open; observed 2026-09-25, deliberately not changed.

`split_pockets_k_fold()` places each group (largest first) into the fold with
the lowest *absolute* penalty after assignment. A partially filled fold already
has a smaller label deviation than an empty one, so large groups accumulate in
fold 0 until its size overshoots. In the frozen `pmm_ion_metal_v1` cohort all of
the largest PDB groups (for example two 48-Mn entries) are in fold 0, and
validation Mn counts range from 340 (fold 4) to 729 (fold 0). Grouping and
per-fold class presence are correct; only stratification balance is weak. The
campaign plan requires reusing this splitter unchanged, and changing it would
alter every existing fold definition, so any fix needs a separate, versioned
split identity.

## TECH-021 — PMM input, replay and runtime certification gaps

**Status:** Implemented and tested 2026-09-26; complete input certification and all
nine GPU smoke/replay cases passed. See `EXPERIMENT_STATUS.md` for full-grid progress.

The original transfer builder could include the tracked held-out PMM source;
embedding metadata did not fully bind live payload content, sequence and residue
order; selected-checkpoint reconciliation could be recorded without failing
completion; train-metric evaluation shared training RNG; and the provisioning
cascade duplicated the VM controller while changing configuration during dry-run.
The revised implementation uses allowlisted archives, schema-v2 content
certification, independent checkpoint replay, separate evaluation RNG, worker
read guards, strict matched prediction validation, and the existing VM controller.
Execution ownership, complete-unit admission and independently verified transfer
receipts gate the next fit. The redundant cascade script was retired.

A training-only context scan also found 503 ions with explicit missing protein
symmetry context. The versioned v2 matched subset consistently excludes them;
see `DATASETS.md` for counts and interpretation. The original v1 artifacts remain
historical evidence. These implementation checks do not establish GPU efficiency,
model superiority, or published PMM context parity.
