# DeepMzyme Documentation Cleanup and Scientific-Provenance Plan — Revised

> Intended path: `docs/PROJECT_CLEANUP_PLAN.md`  
> Status: complete replacement for the previous plan. The repository remains unchanged in this read-only planning session.

## Revision delta

The approved architecture remains, with these binding changes:

- The cleanup is documentation/provenance-only.
- No notebook, training code, model-selection, Stage 6B, Stage 7, or evaluation behavior changes.
- Stage 7 and EC workflow problems move to `docs/FOLLOW_UP_TECHNICAL_ISSUES.md`.
- The EC playbook receives warnings and an incompatibility inventory, not a pipeline redesign.
- `experiment_notes.md` moves intact to the archive.
- `docs/PARAMETER_FINDINGS.md` is a new structured synthesis that excludes historical test metrics from parameter conclusions.
- No new final-test policy is chosen. Active docs use the exact status:  
  **“Primary final-test route: unresolved scientific decision required before final reporting.”**
- `AGENTS.md` receives only conservative deduplication.
- Existing uncommitted changes are protected, inventoried, and never overwritten speculatively.

## 1. Current problems

1. `EXPERIMENT_STATUS.md` is 490 lines and mixes status, policy, experiment history, parameters, bundles, and execution guidance.

2. The legacy non-overlapped PinMyMetal test is incorrectly described as unused:

   - Seven early Only-GVP runs produced `test_report.json`.
   - Four recorded commands explicitly contain `--run-test-eval`.
   - Therefore it is not pristine or unopened.
   - Whether these results influenced later model selection is not established.

3. Unique evidence is only present under Git-ignored `DeepMzyme_Data`, including:

   - Seven historical test evaluations.
   - CLEAN conservative provenance and fold counts.
   - CARE clusterRes30 metadata and audit details.

4. Dataset/final-reporting records conflict:

   - Non-overlapped PinMyMetal is named by policy but was historically accessed and is absent from the current bundle/local data.
   - Exact PinMyMetal has PDB overlap.
   - Other prepared datasets have different scientific purposes and protection states.
   - Cleanup must expose this conflict without choosing a replacement final test.

5. Current-stage guidance conflicts:

   - Trial 49 has fixed-split five-seed evidence, not current grouped-fold confirmation.
   - Status suggests direct Stage 6B progression despite the documented grouped-fold policy.

6. Mutable notebook values are copied into multiple documents and already disagree with the notebook.

7. The EC playbook contains obsolete or unsupported notebook variables and Stage 7 workflow values. Editing these recipes without a dedicated reconciliation could unintentionally alter scientific execution.

8. `LEADERBOARD.md` is stale, omits Hybrid+RING, contains incorrect held-out-test wording, and mixes incomparable reliability levels.

9. Important late-fusion and Hybrid provenance is incomplete:

   - A cited 15-run late-fusion artifact directory is absent.
   - Hybrid Round 1 lacks complete architecture/search-space evidence.
   - Some summaries point to paths rather than durable tracked artifacts.

10. Dataset identities are unclear:

   - CARE legacy and clusterRes30 preparations are blurred.
   - CLEAN materialized fold views do not unambiguously identify their source variant.
   - CLEAN10 was not found.
   - CARE source citation/URL is missing.

11. Measured findings, future search spaces, notebook values, and historical defaults are mixed together.

12. `AGENTS.md`, README, Plan, docs index, guide, playbooks, and command cookbook duplicate mutable facts.

## 2. Current documentation map

| File | Current role | Classification | Problems | Unique information? | Proposed final role |
|---|---|---|---|---|---|
| `README.md` | Overview, quick start, defaults, datasets, bundles, commands | `MIXED / NEEDS SPLITTING` | Stale mutable defaults and duplicated detail | Yes: public overview, quick start, bundle/build information | Short public overview with links |
| `Plan.md` | Architecture, policy, options, datasets, experiment design | `AUTHORITATIVE` | Overloaded with implementation/default inventories | Yes: scientific policy, leakage rules, selection/refit principles | Sole scientific/design policy |
| `EXPERIMENT_STATUS.md` | Status, anchors, history, parameters, bundles, next actions | `MIXED / NEEDS SPLITTING` | Too long; false test wording; stage inconsistency | Yes: current campaign and blockers | Sole concise “where am I and what next?” document |
| `experiment_notes.md` | Early LR/epoch diary | `EVIDENCE` | Looks active; contains unique test-access history | Yes: exact early validation/test results | Move intact to historical archive |
| `list_train_commands.md` | Direct CLI cookbook | `MIXED / NEEDS SPLITTING` | Competes with notebook playbooks | Yes: exact historical CLI recipes | Archive intact |
| `AGENTS.md` | Agent behavior plus copied project facts | `AUTHORITATIVE` | Some mutable facts are duplicated | Yes: extensive operational and safety instructions | Preserve conservatively; remove only proven duplication |
| `CLAUDE.md` | Compatibility pointer | `REDUNDANT` | Broken line wrapping | No scientific content | One-line compatibility shim |
| `docs/README.md` | Documentation index plus copied policy/defaults | `MIXED / NEEDS SPLITTING` | Repeats mutable facts | Yes: ownership/navigation map | Sole detailed documentation index |
| Metal configuration guide | Option semantics plus stages, values, budgets, policy | `MIXED / NEEDS SPLITTING` | Heavy playbook/notebook duplication | Yes: non-obvious stable semantics | Shorter semantics reference; no behavior changes |
| Metal playbook | Exact metal execution recipes | `AUTHORITATIVE` | Dataset/final-reporting conflict | Yes: exact blocks, ranges, seeds, outputs, gates | Preserve recipes; add documentary conflict warning only |
| EC playbook | Intended EC execution recipes | `MIXED / NEEDS SPLITTING` | Unsupported names, obsolete workflow values, old stage model | Yes: all EC budgets, ranges, and scientific intent | Preserve with compatibility warning; defer redesign |
| Manual note | Session scratch note | `ARCHIVE` | Active-looking location | Yes: small historical context | Dated archive note |
| Agent audit report | Historical audit | `ARCHIVE` | Superseded findings appear current | Yes: review chronology | Dated archive audit |
| Notebook-output README | Evidence guide plus mutable claims | `MIXED / NEEDS SPLITTING` | No complete experiment index; incorrect test wording | Yes: evidence-storage rules | Experiment index and evidence contract |
| `LEADERBOARD.md` | Cross-family leaderboard | `MIXED / NEEDS SPLITTING` | Stale, incomplete, and misleading reliability hierarchy | Yes: dated comparison snapshot | Archive intact after indexing |
| Only-GVP history | Historical configs and decisions | `MIXED / NEEDS SPLITTING` | Stale recommendations and config identity conflicts | Yes: six exact configurations and aggregates | Extract findings, then archive intact |
| 19 `summary_run_*.md` files | Batch summaries | `EVIDENCE` | Historical recommendations can look current | Yes | Preserve unchanged; contextualize through index |
| 19 tracked raw output files | Copied notebook outputs | `EVIDENCE` | Navigation and provenance gaps | Yes | Preserve byte-for-byte |
| Main training notebook | Implemented notebook workflow | `AUTHORITATIVE` | Documented Stage 7/Stage 6B safety concern and stale copied descriptions | Yes: actual behavior | Do not modify during cleanup; record issues separately |
| CLEAN baseline notebook | CLEAN execution workflow | `AUTHORITATIVE` | Defaults can be mistaken for completed results | Yes | Preserve unchanged |
| `CLEAN/README.md` | CLEAN baseline guide | `AUTHORITATIVE` | Minor duplicated dataset facts | Yes | Keep procedure-focused |
| CLEAN preparation README | CLEAN preparation provenance | `AUTHORITATIVE` | Current conservative derivative under-documented | Yes | Keep pipeline-focused; add documentary provenance |
| CARE preparation README | Legacy/current CARE preparation | `MIXED / NEEDS SPLITTING` | Current and legacy identities blurred | Yes | Clarify both tracks without changing scripts |
| Feature-extraction README | Feature contract | `AUTHORITATIVE` | No material problem | Yes | Keep as-is |
| PinMyMetal source membership and preparation material | Source/executable provenance | `EVIDENCE` | Some terminology is inconsistent | Yes | Preserve in place |
| Ignored local metadata and runs | Local evidence | `EVIDENCE` | Not durable in the tracked repository | Yes, critically | Promote lightweight exact evidence |

## 3. Proposed authoritative documentation hierarchy

| Responsibility | Authority after cleanup |
|---|---|
| Project overview | `README.md` |
| Documentation navigation | `docs/README.md` |
| Current status and next actions | `EXPERIMENT_STATUS.md` |
| Scientific/design policy | `Plan.md` |
| Datasets, splits, bundles, test-use history | `docs/DATASETS.md` |
| Validation/HPO parameter findings | `docs/PARAMETER_FINDINGS.md` |
| Experiment index | `docs/notebook_outputs/README.md` |
| Exact metal recipes | Metal playbook, unchanged executable blocks |
| EC recipe intent | EC playbook, with affected sections explicitly marked unverified |
| Notebook option semantics | Metal configuration guide |
| Implemented behavior | Existing notebook and training source, unchanged |
| Unresolved technical/workflow problems | `docs/FOLLOW_UP_TECHNICAL_ISSUES.md` |
| Raw/summarized evidence | Existing evidence tree plus promoted provenance |
| Historical records | `docs/archive/` |

`docs/FOLLOW_UP_TECHNICAL_ISSUES.md` is not policy or status. It prevents identified implementation defects from being silently fixed or forgotten.

## 4. Files to keep largely as-is

- All tracked raw outputs.
- All 19 batch summaries.
- Main training notebook.
- CLEAN baseline notebook.
- Training and model source code.
- Tests and preparation/build scripts.
- Metal and EC executable configuration blocks.
- Original PinMyMetal membership files.
- CLEAN, CARE, and PinMyMetal pipeline paths.
- `DeepMzyme_Data` structure, symlinks, features, bundles, and run outputs.
- `CLEAN/README.md` and feature-extraction README.
- Existing ignored raw evidence after portable copies are created.

No `.ipynb`, `.py`, executable configuration behavior, selection logic, Stage 6B logic, or Stage 7 logic is to be changed.

## 5. Files to shorten or consolidate

| File/action | Unique information to preserve | Destination | Preservation check |
|---|---|---|---|
| Shorten `README.md` | Overview, minimal quick start, bundle/build facts | Overview stays; data facts to `DATASETS.md`; recipes remain in playbooks | Compare quick start, bundle hashes/commits, and build command |
| Shorten `Plan.md` | All normative scientific, validation, leakage, refit, and reporting policy | Remain in Plan; descriptive inventories link elsewhere | Policy-heading and sentinel comparison |
| Rewrite status | Current anchors, challengers, evidence grades, readiness, blockers, next actions | Current facts remain; histories/findings/datasets move to owners | Verify every current fact has a destination |
| Conservatively deduplicate `AGENTS.md` | Every operational, behavioral, safety, testing, routing, and response instruction | Remain in AGENTS | Semantic checklist; when uncertain, retain text |
| Fix `CLAUDE.md` | Compatibility pointer | Same file | Pointer resolves |
| Simplify docs index | Ownership/navigation | Same file | Every authority reachable exactly once |
| Shorten metal guide cautiously | Stable option semantics and precedence | Same file; exact recipes stay in playbook | Compare each removed passage with destination |
| Annotate metal playbook | Existing exact recipes | Same file, unchanged blocks | Hash or structured diff of configuration blocks |
| Annotate EC playbook | All existing EC ranges, budgets, intent, and blocks | Same file plus technical-issue record | Hash or structured diff of all existing blocks and numeric values |
| Expand notebook-output README | Evidence-storage contract | Same file becomes experiment index | Account for every summary/raw artifact |
| Clarify CLEAN preparation README | Commands, thresholds, folds, historical/current variants | Same file plus `DATASETS.md` | Preserve every command and threshold |
| Clarify CARE preparation README | Legacy and clusterRes30 procedures | Same file plus `DATASETS.md` | Preserve every path, command, threshold, count, and alias |

### EC playbook treatment

The cleanup may only:

- Add a prominent warning that affected sections are not verified against the current notebook.
- Add a table of exact discovered inconsistencies.
- Link to `FOLLOW_UP_TECHNICAL_ISSUES.md`.
- Label obsolete blocks as historical/intended configuration pending reconciliation.

The cleanup must not:

- Rename variables inside executable blocks.
- invent replacement search-space controls;
- add a redesigned Stage 6B;
- change evaluation workflow values;
- modernize EC Stage 6/6B/7;
- imply that affected blocks have become executable.

Known inconsistencies to record include:

- `preview_only` and `evaluate_selected_checkpoint` versus the notebook’s current recognized values.
- Missing notebook variables including `OPTUNA_WEIGHT_DECAYS_CSV`, `OPTUNA_BATCH_SIZES_CSV`, `OPTUNA_HIDDEN_S_VALUES_CSV`, and `OPTUNA_GVP_LAYERS_VALUES_CSV`.
- EC Stage 6 being seed-repeat-oriented and lacking the current metal-style Stage 6B bridge.
- Any additional mismatched assignments found by a complete static comparison.

## 6. Files to archive

All archival operations use `git mv`.

| Current file | Reason | Unique information | Final path | Preservation check |
|---|---|---|---|---|
| `experiment_notes.md` | Historical diary, not active parameter authority | Exact early validation and historical test metrics | `docs/archive/experiments/experiment_notes_legacy.md` | Whole-file hash and rename detection |
| `list_train_commands.md` | Historical second command surface | Every exact CLI recipe | `docs/archive/workflows/list_train_commands_legacy.md` | Whole-file hash |
| Manual 20 Aug note | Session scratch material | G4 preference and CLEAN question | `docs/archive/session_notes/2026-08-20.md` | Whole-file hash |
| Agent audit report | Superseded audit | Historical findings and chronology | `docs/archive/audits/2026-05-prompt-0-audit.md` | Whole-file hash |
| `LEADERBOARD.md` | Stale current ranking | Dated cross-family comparison | `docs/archive/experiments/leaderboard_snapshot_2026-05-16.md` | Index every row, then verify hash |
| Only-GVP history | Stale historical narrative | Exact configs, aggregates, decisions | `docs/archive/experiments/metal_only_gvp_round3_history.md` | Findings/index comparison and hash |
| Notebook-output changelog | Historical chronology | Dates and rationale | `docs/archive/changelogs/notebook_outputs_changelog.md` | Whole-file hash |
| Completed cleanup plan | Prevent it becoming another active authority | Cleanup rationale and record | `docs/archive/plans/PROJECT_CLEANUP_PLAN_completed.md` | Move only after approval and verification |

If an archive candidate overlaps an uncommitted change whose intent remains unclear after inspecting the diff, stop and report that file rather than moving it.

## 7. Files that can safely be removed

None in the first cleanup pass.

No scientific record, raw output, batch summary, notebook, source membership file, preparation file, provenance record, or compatibility shim will be deleted.

## 8. New files genuinely needed

### `docs/DATASETS.md`

Needed because there is no tracked authority for dataset identity, preparation status, split relationships, bundles, provenance, and test-use history. It replaces that responsibility across README, Plan, status, AGENTS, playbooks, preparation READMEs, and ignored metadata.

### `docs/PARAMETER_FINDINGS.md`

A clean new synthesis. It replaces active parameter-summary fragments in status and other general docs, but does not replace or embed the historical diary.

Every conclusion links to:

- Archived `experiment_notes.md`, when relevant.
- A batch summary.
- Exact config/raw evidence.
- An explicit provenance-gap marker when raw evidence is missing.

### `docs/FOLLOW_UP_TECHNICAL_ISSUES.md`

Needed because no existing maintenance location cleanly owns verified-but-unfixed workflow problems.

Initial issues:

1. Stage 7 may fall back to a Stage 6-selected candidate without the documented Stage 6B refit.
2. EC playbook variables and workflow values do not match the current notebook.
3. Notebook/documentation live-default drift.
4. Missing late-fusion Round-4 artifact directory and incomplete Hybrid provenance.
5. Final-dataset route implementation/availability mismatch, distinct from the unresolved scientific policy decision.

Each issue records:

- Exact file/section evidence.
- Current observed behavior.
- Documentation/policy conflict.
- Risk.
- Recommended scope for a separate future task.
- Tests required for that future task.
- Status: open, without modifying current behavior.

### Durable evidence packages

- `docs/notebook_outputs/raw/legacy_nonoverlap_test_access/`
- `CLEAN_prepare_training_and_test_set/provenance/`
- `CARE_prepare_training_and_test_set/provenance/`

These are evidence storage, not competing human-facing authorities.

No additional experiment index, status file, command index, or final-test policy document will be created.

## 9. Parameter/HPO knowledge preservation plan

`PARAMETER_FINDINGS.md` will use these evidence grades:

1. Grouped folds × seeds with paired CI.
2. Grouped folds only.
3. Fixed validation split across seeds.
4. HPO discovery on one validation split.
5. Single-seed validation.
6. Exploratory/smoke/partial/incomplete.
7. Superseded historical evidence.

Every finding records:

- Namespaced experiment/study identity.
- Task, model, fusion, dataset, label scheme, and node mode.
- Exact parameters or search space.
- Validation folds and seeds.
- Selection metric.
- Validation mean/std/min/max and per-class recall when available.
- Promotion, rejection, or inconclusive decision.
- Negative findings and settings not worth repeating without reason.
- Links to summary, config, and raw evidence.
- Provenance completeness.

Historical held-out-test results are explicitly excluded from:

- HPO recommendations.
- Model promotion or rejection.
- Parameter-range conclusions.
- Current model ranking.
- “Do not repeat” guidance.

The early validation result may be listed as single-seed validation evidence, but its test metrics appear only in the dataset/test-use ledger, archived notes, experiment index access flag, and raw evidence.

Required findings include:

- Only-ESM anchor: LR `3e-5`, WD `1e-4`, batch 8, inverse-frequency, mean `0.625325230595`, SD `0.031449451169`, seeds `42,123,2026,43,44`.
- Only-ESM Round 2: 24/36 rows completed; `5e-5` never ran.
- Only-GVP trials 7/12/13 and both conflicting “trial 12” LR identities.
- Late-fusion trials 49/32/15 and exact fixed-split aggregates.
- Node-level late-fusion negative result `0.606599196822`.
- Hybrid exploratory findings with incomplete architecture provenance.
- Hybrid+RING trial 114 exact configuration and single-seed evidence grade.
- RING contribution recorded as inconclusive.
- One-epoch late-fusion repeats recorded as smoke, not rejection evidence.
- Trial 177 recorded as incomplete.
- Older ESM identity recorded as unavailable.

Trial IDs must be namespaced by family, batch/date, study/storage identity, and trial number.

## 10. Dataset and test-set documentation plan

`DATASETS.md` will distinguish:

- Membership/labels materialized.
- Model evaluation artifacts found.
- Evidence of selection use.
- Current protection status.
- Scientific purpose.
- Current local and bundle availability.

| Dataset | Essential record |
|---|---|
| Original PinMyMetal | Preserve original train/test files, 7,920/1,488 rows, 4,195/1,179 PDBs, 668 overlapping PDBs, original site identifiers |
| Exact PinMyMetal | 1,472 train and 313 test PDBs, 177 overlaps, 2,144/490 site rows; present locally/v10; no completed evaluation found in inspected evidence |
| Non-overlapped PinMyMetal | Intended zero overlap; absent locally/v10; evaluated in exactly seven early runs; not pristine |
| Harsh PinMyMetal | Zero-overlap severe variant; currently absent; no evaluation evidence found |
| Common 70/30 | Seed 42, 1,419/189 PDBs, zero overlap; comparison split, not established primary final split |
| CLEAN30 original | Official five-fold multi-donor reference, 740 structures |
| CLEAN30 conservative | Current `CLEAN_30_main`; deterministic single donor, supported-metal filtering, 2.0 Å donor dedup, exact fold counts |
| CLEAN10 | Not present or documented in the inspected repository |
| CARE legacy | Historical base `30_identity` preparation |
| CARE clusterRes30 | Current prepared route; 817/34 structures and 1,520/76 sites; exact preparation thresholds and audit links |

Required active wording:

> The non-overlap PinMyMetal test was historically evaluated in seven early runs and is therefore not pristine or unopened. Whether those values influenced subsequent selection is not established by repository evidence. These test metrics must not be used for current HPO recommendations or model selection.

And:

> **Primary final-test route: unresolved scientific decision required before final reporting.**

Cleanup will not:

- Designate a replacement test.
- Change Plan’s scientific policy.
- Declare another dataset the primary final test.
- Treat exact PinMyMetal as an automatic substitute.
- Change evaluation code or notebook behavior.

Bundle names, SHA256 hashes, Hub commits, source scripts, thresholds, fold counts, catalytic filtering, aliases, and availability remain fully documented.

## 11. Proposed final human-facing tree

```text
README.md
EXPERIMENT_STATUS.md
Plan.md
AGENTS.md
CLAUDE.md

docs/
├── README.md
├── DATASETS.md
├── PARAMETER_FINDINGS.md
├── FOLLOW_UP_TECHNICAL_ISSUES.md
├── METAL_NOTEBOOK_CONFIGURATION_GUIDE.md
├── METAL_TRAINING_PIPELINE_PLAYBOOK.md
├── EC_TRAINING_PIPELINE_PLAYBOOK.md
├── notebook_outputs/
│   ├── README.md
│   ├── summaries/
│   └── raw/
└── archive/
    ├── audits/
    ├── changelogs/
    ├── experiments/
    ├── plans/
    ├── session_notes/
    └── workflows/

CLEAN/
├── README.md
└── train_clean_predictor_baselines.ipynb

CLEAN_prepare_training_and_test_set/
├── README.md
└── provenance/

CARE_prepare_training_and_test_set/
├── README.md
└── provenance/

prepare_training_and_test_set/    # unchanged
notebooks/                         # unchanged
src/                               # unchanged
DeepMzyme_Data/                    # unchanged
```

## 12. Link and provenance strategy

Stable chain:

```text
EXPERIMENT_STATUS
  → experiment index or parameter finding
    → immutable experiment summary
      → exact configuration
        → raw evidence
```

Dataset chain:

```text
EXPERIMENT_STATUS
  → DATASETS dataset ID / test-use ledger
    → preparation README/script
      → tracked metadata or original membership
        → bundle/hash or local-source provenance
```

Technical issue chain:

```text
EXPERIMENT_STATUS or documentation warning
  → FOLLOW_UP_TECHNICAL_ISSUES issue ID
    → exact notebook/playbook evidence
      → recommended separate implementation task
```

Every experiment-index row includes:

- Namespaced ID.
- Task/model/dataset/label scheme.
- Selection metric.
- Planned/completed runs.
- Seeds/folds.
- Evidence grade.
- Result and historical decision.
- Held-out-access flag.
- Summary/raw/config links.
- Artifact-completeness status.

Missing artifacts are labeled explicitly, never replaced by unsupported reconstruction.

## 13. Exact execution sequence

1. Save this revised plan at `docs/PROJECT_CLEANUP_PLAN.md`.

2. Protect the existing working tree:

   - Record `git status --short`.
   - Save `git diff --stat`.
   - Save full diffs for every file that may be edited.
   - Hash current contents of every edit/archive target.
   - Record current modified paths, including README, Plan, status, both metal docs, CLEAN files, notebook, tests, and the manual note.
   - Never reset, checkout, overwrite, or normalize unrelated changes.
   - If an intended cleanup edit conflicts with an unclear uncommitted change, halt and report the exact file/hunk.

3. Establish a strict allowed-path set:

   Documentation/provenance files only.

   Explicitly forbidden modifications:

   - `notebooks/*.ipynb`
   - `CLEAN/*.ipynb`
   - `src/**`
   - `tests/**`
   - preparation/build `.py` files
   - `DeepMzyme_Data/**`
   - training/configuration behavior
   - Stage 6B/7 or evaluation logic

4. Freeze evidence:

   - Hash all existing raw outputs and summaries.
   - Build the numeric/trial/dataset sentinel manifest.
   - Inventory ignored test reports and CARE/CLEAN metadata.

5. Promote historical test evidence:

   - Copy the seven exact configs, metadata files, and test reports.
   - Copy diagnostic/sweep CSVs.
   - Preserve one canonical dataset summary and hashes for identical copies.
   - Leave ignored originals unchanged.

6. Recover portable provenance:

   - Recover lightweight late-fusion JSON artifacts from current/local sources, Drive paths, or the identified historical commit.
   - Do not restore checkpoint binaries during documentation cleanup.
   - Preserve checkpoint names/hashes where available.
   - Recover Hybrid configs if available; otherwise record the gap.
   - Promote CARE/CLEAN metadata into tracked provenance directories.

7. Create `DATASETS.md` and its test-use ledger.

8. Archive `experiment_notes.md` intact with `git mv`.

9. Create `PARAMETER_FINDINGS.md` from validation/HPO evidence only.

10. Expand the notebook-output README into the experiment index.

11. Create `FOLLOW_UP_TECHNICAL_ISSUES.md` with exact evidence for deferred Stage 7, Stage 6B, EC, default-drift, and provenance issues.

12. Archive stale leaderboard/history/changelog files with `git mv`.

13. Rewrite `EXPERIMENT_STATUS.md` concisely, including:

   - Current objective.
   - Trusted fixed-split anchor and challengers.
   - Evidence grades.
   - Dataset readiness.
   - Accurate historical test-access statement.
   - Exact unresolved-final-test-route statement.
   - Immediate and next few documentation/scientific actions.
   - Links to details.

14. Simplify README, Plan, docs index, and guide without altering scientific policy or executable values.

15. Update playbooks documentary-only:

   - Metal: add final-route conflict link/warning without altering blocks.
   - EC: add compatibility warning and exact inconsistency inventory without repairing blocks.

16. Deduplicate AGENTS conservatively:

   - Only after destination authorities exist.
   - Preserve every operational/safety/behavioral instruction.
   - Keep uncertain content.
   - Do not optimize for line count.

17. Clarify CLEAN/CARE preparation READMEs without changing scripts or paths.

18. Archive command cookbook, audit, and scratch note, subject to protected-worktree checks.

19. Verify no prohibited file changed.

20. Present the cleanup diff for review. Do not commit, push, or create a PR.

21. Only after cleanup acceptance, archive the completed cleanup plan.

## 14. Verification checklist

### Protected worktree

- Pre-edit status, stat, full diffs, and hashes exist.
- Every pre-existing user change remains present.
- No unrelated hunk changed.
- Any ambiguous overlap caused a halt rather than a guess.

### Scope boundary

`git diff --name-only` must contain no changes under:

- `notebooks/`
- `CLEAN/*.ipynb`
- `src/`
- `tests/`
- preparation/build scripts
- `DeepMzyme_Data/`

No Stage 6B, Stage 7, model-selection, training, or evaluation behavior changed.

### Evidence preservation

Search for:

- Trials `7,12,13,15,17,24,32,49,84,105–177`, especially `114`.
- LRs:
  - `6.464669746492395e-05`
  - `4.735385769610685e-05`
  - `4.752317377508605e-05`
  - `1.6801503587890522e-05`
  - `3.705631497756492e-05`
- Anchor and negative-result aggregates:
  - `0.625325230595`
  - `0.635468206972`
  - `0.606599196822`
  - `0.7303469775006777`
  - `0.6992755722978847`
- Seeds `42,123,2026,43,44`.
- Every search space, fold definition, per-class recall, exact output name, and caveat.

### Historical test handling

- Exactly seven non-overlap test reports are tracked.
- Each covers 352 pockets.
- All exact metrics remain in raw evidence and archived notes.
- Active docs state the test was historically accessed.
- Active docs do not use test metrics to recommend parameters or select models.
- Search active parameter findings for test-derived rankings or recommendations and require none.
- No active document calls the split pristine, unopened, untouched, or never evaluated.
- The exact unresolved-final-test-route sentence appears in status and DATASETS.

### Dataset provenance

- Original PinMyMetal files remain byte-identical.
- Exact/common70/CLEAN/CARE counts match metadata.
- CLEAN donor selection, thresholds, fold counts, and materialized source variants remain recoverable.
- CARE counts, thresholds, missing lists, and audit details remain recoverable.
- Bundle names, hashes, and commits remain exact.
- CLEAN10 remains labeled absent/not documented.
- CARE citation gap remains explicit.

### Documentation ownership

- Each concept has one authority.
- README and docs index contain no live-default tables.
- Status contains no long experiment diary.
- Plan remains policy, not dataset inventory.
- Parameter findings contain validation/HPO conclusions, not test-derived advice.
- Notebook-output README indexes every batch.
- Follow-up technical issues contain all deliberately deferred behavior problems.
- EC playbook warnings do not imply reconciliation is complete.

### Archive and links

- All archive operations appear as renames.
- Archived file hashes match originals.
- All Markdown links resolve.
- Every important status/finding claim reaches summary/config/raw evidence or an explicit missing marker.
- No raw or summary evidence was deleted.

## 15. Expected net simplification

Current:

- 49 tracked Markdown files.
- 17 active human-facing/control files outside the notebook-output evidence tree.
- Approximately 13 general project/control documents plus four scoped component READMEs.

After cleanup:

- Approximately 16 active human-facing/control files:
  - 12 general authorities, including the new technical-issue register.
  - The same four scoped component READMEs.
- The cleanup plan is temporary and later archived.
- Raw/provenance file count increases because unique ignored evidence becomes durable.
- Historical documents remain physically present under `docs/archive/`.

Conceptual simplification remains material:

- One current-status authority.
- One dataset/test-use authority.
- One validation/HPO findings authority.
- One experiment index.
- One technical-issues register for verified but deliberately unfixed behavior problems.
- No active historical diary.
- No active stale leaderboard.
- No root command cookbook competing with playbooks.
- No test-derived parameter advice.
- No silent final-test policy decision.
- No executable behavior changed.
- No useful AGENTS behavior sacrificed for line-count reduction.
