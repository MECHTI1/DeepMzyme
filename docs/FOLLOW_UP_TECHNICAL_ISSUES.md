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

**Observed behavior**

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

- `Plan.md` describes non-overlapped PinMyMetal as the historically preferred
  primary final route.
- The metal playbook's current common defaults use exact PinMyMetal.
- Exact PinMyMetal contains 177 overlapping PDB IDs.
- The notebook dropdown and current v10 bundle omit non-overlapped and harsh
  roots.
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

**Status:** Open; documentation workaround recorded 2026-08-22

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

**Current workaround**

Treat the first 37 `PASS` lines as completed checks and the final traceback as
this known documentation-path defect. `docs/GETTING_STARTED.md` records the
expected outcome. Do not describe the full smoke suite as green.

**Proposed implementation fix**

Update `tests/smoke_checks.py` in a separately authorized code/test change so
the documentation check references the archived command file, or deliberately
scopes the assertion to active documentation only. Preserve the check for the
broken `src.training.run` command pattern.

**Required future tests**

- Run the complete smoke suite to exit code 0.
- Confirm the archived command document contains no broken active command
  pattern if it remains in scope.
- Confirm the final multi-metal check runs or reports its intended data-based
  skip.

No test source was changed in the 2026-08-22 documentation task.

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

**Status:** Open; runtime contract documented 2026-08-22

**Observed behavior**

`src/requirements.txt` is present, but it is a short direct-dependency list. It
does not define the Python version, CUDA wheel source, transitive resolution,
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

**Risk**

- A fresh local installation cannot be reproduced exactly from the repository.
- Installing the local PyTorch pin in Colab can make an assigned GPU unusable.
- A future stock Colab image may change, so a hard-coded historical version is
  not a sufficient compatibility check.

**Current workaround**

Use the existing absolute interpreter on the project workstation. In Colab,
preserve an importable stock PyTorch, filter only the top-level `torch`
requirement, and run the version/CUDA/compute-capability/architecture preflight
before GPU work and again after installation. Exact commands are in
`docs/COLAB_GPU_RUNBOOK.md`.

**Proposed implementation fix**

Create separate, validated environment contracts for local development and
Colab instead of forcing one PyTorch pin across both hardware contexts. Record
the Python version and solve/lock the non-hardware-dependent dependencies;
document the CUDA/PyTorch selection boundary explicitly. Keep serious-run
library metadata even after locks are introduced.

**Required future tests**

- Fresh local environment creation from the proposed lock/specification.
- CPU import and CLI-help check in that environment.
- Colab T4/L4/G4/A100 preflight that compares device capability to
  `torch.cuda.get_arch_list()`.
- Verification that the Colab install never replaces stock PyTorch unless an
  explicitly tested compatibility path is selected.
- Imports for `torch`, `torch_geometric`, ESMC when requested, Optuna, NumPy,
  scikit-learn, notebook reporting dependencies, and DeepMzyme training.
- Capture of exact versions in a serious-run metadata artifact.

No dependency file or executable setup cell was changed in the 2026-08-22
documentation task.
