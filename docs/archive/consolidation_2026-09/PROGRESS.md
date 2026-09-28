# PROGRESS — docs consolidation 2026-09

Read this first after a context compaction or in a new session. Never redo finished steps.
Plan: [`PLAN_v2.md`](PLAN_v2.md), a copy of `~/.claude/plans/docs-consolidation-plan-v2.md`.

## User decisions received (2026-09-28)
- Worktree on `/media/mechti/Data1/DeepMzyme_worktrees/docs`; verify clean after checkout.
- Doc freeze confirmed: Codex idle with no active threads, covering Job A **and** Job B
  while the campaign is paused.
- Ledger: record all six 2026-09-18 non-overlap test reports.
- Zenodo relaunch: record "unknown, possibly evaluated".
- The agent may run the rebase and `git merge --ff-only`, but must **stop and ask before
  executing the merge**.
- gitleaks is not installed: use the fallback credential scan.
- Job B step-2 review (user's written answers): 1A “remove "primary",
  everywhere”, keeping collapsed four as the reporting endpoint, direct four as
  distinct from six-then-collapse, and historical labels; architecture
  comparisons keep one fixed training target named per campaign. 2B “any mix of
  four, five and six per campaign”, with the neutral-test rules in RULES B-M01.
  “Approve B-M03 to B-M10 and B-D01”; B-M02 approved with changes (a)–(d).
  Repeat log-only resume rules in the PMM README. AGENTS (15.8 KB) and
  docs/README (22 KB) accepted over target until Job C. Show ANSWER_KEY Q1 and
  the STATUS layout for approval; push the backup branch after applying.
  `docs/plans/` stays unedited (the PMM plan's line-312 tie rule stays).
- Review of `e261680` (user): fix remaining target wording (A) and the Stage 6B
  text (B: the notebook Stage 6/6B route is single-scheme; cross-scheme
  comparison uses a campaign assessor); record the Stage 6B gap now in a dated
  TECH-010 note and the STATUS caveats slot (C; FOLLOW_UP editing allowed for
  this only); “only the rows I named are approved”; AGENTS 16.8 KB and a
  25.8-KB default read accepted until Job C (D). ANSWER_KEY Q1 approved with
  the user's element 4 (E). STATUS layout approved with fixed campaign, stage,
  best-result, authorization and GPU/VM lines and a caveats slot (F). Then push
  and continue the remaining Job B steps, stopping before the final commit.
- After fresh-agent run 1 (user): commit and push the staged work as a backup
  with the two listed commands, no merge (done: `a205658`). Q1 fix approved:
  the PMM README endpoint sentence says five- or six-class training followed by
  collapse differs from direct four-class training and that comparing
  objectives is a neutral test where better, no difference and worse are all
  valid; run one new fresh agent with the same setup and report both runs; if
  it still fails, stop rather than rewrite docs to pass. Sizes accepted for now:
  default read about 27.0 KB, AGENTS 17.0 KB, PMM README over 60 lines; safety
  text comes first and tightening is Job C. Record the `remote_SHA256SUMS`
  finding for Job C/D; raw evidence unchanged. Local pytest and `smoke_checks`
  with the Conda interpreter run on the workstation before any merge. Then
  commit, push and stop.

## State
| Field | Value |
|---|---|
| Job | B |
| Phase | Job B cloud work done (2026-09-28): fresh-agent run 1 not accepted; run 2, after the approved Q1 fix, matches all six under the recorded grading; sizes accepted for now; workstation pytest/smoke and any merge await the user |
| Cloud branch | `claude/happy-ritchie-99xmhd` (backup pushes only, `29e7460` through `a205658` and the commit carrying this update; nothing merged into `metal-pmm-ion-campaign`) |
| Branch / base | `docs-consolidation` and `metal-pmm-ion-campaign` @ `3c0f80c6033a332b65e91302eea16c81add13e1d` |
| Worktree | `/media/mechti/Data1/DeepMzyme_worktrees/docs` (clean at Job B start) |

## Done (Phase 0)
1. Pre-flight on the shared checkout (read-only): status empty; HEAD = origin = `3e912a6`;
   `source_tree_sha256` = `adc95c42…`; `docs/plans/*` hashes recorded; `/` 2.6 GB free.
2. Worktree created and verified clean.
3. `before_hashes.txt` (1,521 files); `baseline/` (see `baseline/README.md`).
4. Secret scan: 0 hits. Code↔doc inventory: `inventory/code_coupling.md` (no new consumer
   affects Job A). Job A pre-checks B0.1–B0.7: `inventory/job_a_prechecks.md`.
   Test-access artifacts: `inventory/test_access_artifacts.tsv`.
5. Saved inventory evidence: `inventory/code_coupling.md` and
   `inventory/job_a_prechecks.md`. Separate per-group subagent files are not
   present. Baseline fresh-agent answers: `baseline/fresh_agent_answers.md`.
6. `ANSWER_KEY.md` drafted.

7. Answer Key approved by user.
8. Job A edits applied per PLAN_v2 Appendix B (B4-B7):
   - B4 (D1 compute route): AGENTS.md, Plan.md, docs/GETTING_STARTED.md, METAL_TRAINING_PIPELINE_PLAYBOOK.md.
   - B5 (D2 test ledger & wording): docs/DATASETS.md, docs/EXACT_PINMYMETAL_5FOLD_CV_REPRODUCIBILITY.md, EXPERIMENT_STATUS.md.
   - B6 (fold table & index rows): Plan.md (fold table), docs/notebook_outputs/README.md, docs/PARAMETER_FINDINGS.md.
   - B7 (records): docs/archive/consolidation_2026-09/RULES.md created with all 12 Job A modifications documented.

9. Independent review corrections applied (2026-09-28):
   - AGENTS.md: Restored `gpu-use-skill` tag and clarified persistent storage policy across both VM and Colab routes.
   - Evidence grading: Corrected benchmark evidence grade from Grade 2 to Grade 6 (exploratory) in both PARAMETER_FINDINGS.md and docs/notebook_outputs/README.md due to pocket-stratified fold leakage.
   - Link fix: Directed L4 continuation row in docs/notebook_outputs/README.md to `docs/agents_report/GVP_FUSION_EXACT_POCKET_L4_V2_EXECUTION.md`.
   - Metric labeling: Labeled Table 1 in EXACT doc and EXPERIMENT_STATUS.md explicitly as unadjusted epoch maxima, cross-referencing selected-checkpoint OOF values.
   - Ledger path: Linked full TSV path `docs/archive/consolidation_2026-09/inventory/test_access_artifacts.tsv` in DATASETS.md.
   - Audit registers: Appended RULE-A13 and RULE-A14 to RULES.md.
   - Full tree verification against HEAD: Checked `git diff HEAD --name-only -- src scripts tests notebooks docs/plans` (completely empty).

## Next
- User: accept or reject the fresh-agent result
  ([both runs](verification/job_b_fresh_agent_test.md)).
- On the workstation, before any merge: the checker and hook with the Conda
  interpreter, and pytest and `smoke_checks` against the local baseline.
- Ask before commit, merge or push. Shared checkout remains read-only.

## Job B completed at the rule-review checkpoint

- Reconciled the integrated Job A commit with both local tips and the remote.
- Repeated code/document consumer inventory and recorded 1,869 protected-file
  hashes before rewrites; no new executable consumer requires a path change.
- Preserved all four requested STATUS revisions (current `3c0f80c`,
  `9a25f36^`, `9a069d2^`, full `b45b893`, including lines 72–86) in nine
  README/log folders before replacing STATUS. The 91 source spans deduplicate
  to 60 verbatim blocks; reconstructing each full source is byte-identical.
- Inventoried 466 content blocks in RULES (conservative superset of normative
  statements): 248 kept, 140 merged, 71 modified, seven dropped. These are
  proposed dispositions; the only intentional drops are seven invalid CLI
  examples. The short substantive review list is separate from the full table.
- Prepared concise root AGENTS/STATUS and PMM overview. Moved the navigation/
  coordination text to docs/README and stage-answer rules to the metal
  playbook; all executable fences remain byte-identical. MAP/MOVED record
  copies and section relocations; no existing source/evidence file moved.
- Verification: [review invariants](verification/job_b_review_invariants.json).
  Both source hashes remain `adc95c42…`; 1,869 protected files unchanged;
  247 local links/anchors checked without errors; shared checkout clean.
  Default read path: 23,225 bytes. AGENTS exceeds the 12-KB target; no rule
  was dropped to meet a cap and no checker cap was raised.
- The v1 STATUS template referenced by PLAN_v2 was not found in the supplied
  files or local plan directory. Draft fields follow the recovered Update rule;
  user review is of this concrete layout, not a claimed copy of absent v1 text.
- Full CPU regression, checker/hook, ignore fixes and fresh-agent acceptance
  await the plan's post-review steps. No fresh-agent pass is claimed.
- The completed history/inventory work and concrete rewrite proposals are staged
  for review; staging is not approval or a completed Job B integration.
- No training, GPU action, final refit or held-out evaluation occurred. No
  commit, merge or push occurred during Job B.

## Job B integration reconciliation (2026-09-28)

- The user reports Job A committed, fast-forwarded and pushed. Read-only Git
  verification confirms both local branch tips and remote
  `refs/heads/metal-pmm-ion-campaign` equal `3c0f80c6033a332b65e91302eea16c81add13e1d`;
  both worktrees were clean before this edit. Commands: `git status --porcelain=v1`,
  `git rev-parse HEAD`, `git branch -vv`, `git worktree list`, and
  `git ls-remote --heads origin docs-consolidation metal-pmm-ion-campaign`.
- The remote has no `docs-consolidation` head; pushing that optional backup
  branch is not necessary to establish campaign integration.
- Job A verification files remain dated pre-integration records, not current
  branch-state assertions. No baseline is rerun merely to reconcile Git.
- Job B is authorized only in this documentation worktree. PMM core v2 stays
  paused: nine ordinary-readout fold-0 fits retained; 36 fits deferred. Original
  strict replay failures and separate retrospective qualifications stay distinct.
  No training, GPU work, final refit or held-out evaluation is authorized.

## Job B cloud continuation (2026-09-28)

- Environment: Claude Code cloud container, fresh clone. No workstation, shared
  checkout, frozen checkout, controller or VM access; workstation paths above are
  historical context. Branch `claude/happy-ritchie-99xmhd` at `3c0f80c`.
- The user's staged Job B patch (31 files, SHA-256 `f8ac2f27…`) passed
  `git apply --check` and was applied to the index and worktree.
- Independently reproduced: byte-identical rebuild of all four STATUS
  revisions, inventory fidelity/coverage, 1,869 unchanged protected files, the
  `adc95c42…` source hash in this checkout, playbook fence invariants and links.
  See [cloud review record](verification/job_b_cloud_review.json).
- Review corrections restore merged-row substance lost in compression, keep
  commit/push suggestions in final responses, and repair two relocated
  navigation lines.
- Backup `29e7460` (patch + cloud corrections, before the approved edits) was
  pushed to `claude/happy-ritchie-99xmhd` at the user's request.
- `e261680` applied the user's step-2 answers only partly; the user's review
  found remaining target wording and an inaccurate Stage 6B claim. After the
  review fixes, no training objective is called primary anywhere in the active
  docs; the places changed and the exceptions (other meanings of “primary”;
  historical recipe wording; `docs/plans/`) are listed in RULES B-M11. Any mix
  of four/five/six with the neutral test; B-M02 changes; PMM README resume
  rules. Counts: 244 kept, 139 merged, 76 modified, seven dropped.
- Stage 6B code check: every metal run computes
  `val_metal_collapsed4_balanced_acc` (`src/training/run.py:776-807`);
  `STAGE6B_RANK_BY_METRIC = "auto"` resolves to native
  `mean_val_metal_balanced_acc` (`src/training/pipeline_metric_policy.py:109-118,
  159-161`), whose rare-recall and tie-breaker metrics are native. Stage 6
  skips imported candidates whose `metal_label_scheme` differs (notebook cell
  `a408dcb9`), and Stage 7 blocks mixed batches unless
  `ALLOW_MIXED_FINAL_TEST_BATCH` (final held-out cell). So the notebook route is
  single-scheme; `pmm_core_assessment.py:95-146` ranks on mean common-four BA.
  Recorded in TECH-010 and the STATUS caveats; no code change.
- STATUS follows the approved layout (5,290 bytes). The user accepted AGENTS at
  about 16.8 KB and a default read of about 25.8 KB until Job C; after the
  approved STATUS layout and review fixes they are 17,027 and 27,026 bytes,
  which exceeded that acceptance; the user later accepted the current sizes
  for now (see the decisions above).
- No training, GPU action or held-out evaluation occurred.

## Job B remaining steps (cloud, 2026-09-28)

Details: [cloud review record](verification/job_b_cloud_review.json),
`remaining_job_b_steps`.
- Added `tools/check_docs_contract.py` (stdlib) and `.githooks/pre-commit`,
  enabled per command only; `core.hooksPath` is not set. Checker: 188 Markdown
  files, 792 links, 0 strict, 13 warnings (caps, six resolved TECH issues kept
  in full, two duplicated paragraphs). 18 of 18 seeded violations fail as
  expected; the hook blocks a broken-link commit in a throwaway clone.
- `a205658` used the approved command `DEEPMZYME_PYTHON=python3 git -c
  core.hooksPath=.githooks commit …`; the hook skipped because it required a
  path (the checker had run directly just before: 0 strict). The hook now
  resolves a command name on PATH and then requires an executable; retested,
  a missing interpreter still skips with a message and a broken-link commit is
  refused.
- `.ignore` added; `.aiignore` replaced by the same list (PLAN_v2
  contradiction 11); `prepare_training_and_test_set` stays visible because the
  user did not ask to hide it. `GEMINI.md` points to AGENTS. The gpu-use-skill
  Handoff follows the STATUS/log rule.
- Fresh-agent test ([both runs](verification/job_b_fresh_agent_test.md)):
  run 1 **not accepted** (Q1 missed part of element 4; Q3 borderline). After
  the approved PMM README fix (lines 10–12; still 63 lines), run 2 matches all
  six under the same grading (Q4 counting the whole response), with wording
  notes. One cold-start subagent per run, read-only by instruction only.
- CPU CI (stand-in for pytest): base `3c0f80c` and backups `042e93d` and
  `a205658` all show 3 failed, 991 passed, 40 skipped, with the same three
  failures; CI never reaches `smoke_checks`. No local pytest or smoke run (no
  torch/numpy here; no installs). No match to the local baseline is claimed.
- Manifests: 10 of the 11 Phase 0 manifests pass; the same three legacy CRLF
  CSVs fail. New observation: `pmm_replay_diagnostic_20260928/remote_SHA256SUMS`
  lists 34 VM files absent from the repository (8 present entries match); not
  caused by Job B; recorded for Job C/D below.
- `before_hashes.txt`: 1,521 entries, 0 missing, 14 changed, all documents.
  Protected paths unchanged; source hash `adc95c42…`; playbook invariants hold.
- Sizes accepted by the user for now: after the Q1 fix the default read is
  27,112 bytes (AGENTS 17,031 + STATUS 5,290 + PMM README 4,791) and the PMM
  README 63 lines; the checker keeps warning.

## Job A closeout verification

- Final report: [verification/README.md](verification/README.md).
- Full CPU suite: 992 passed, 2 failed, 32 errors, 8 skipped, 15 subtests passed.
  All 34 failure/error identities and counts match Phase 0. Smoke: 43 passed,
  one known absent-fixture skip, exit 0.
- Both frozen source hashes match; first playbook blocks are unchanged; only
  nine intended documents differ among the 1,521 pre-job hash entries.
- All 36 local report hashes match. Ten of eleven evidence manifests pass;
  the same three legacy CSV hash failures remain. No evidence was rewritten.
- Twenty-two added local links/anchors pass. No protected path changed.
- At the Job A verification checkpoint no commit, merge, push, Job B rewrite,
  or GPU action had been performed. Job A was subsequently integrated as above.

## Open questions
- Acceptance of fresh-agent run 2 (wording notes recorded with it).
- `.aiignore`: `prepare_training_and_test_set` is visible (the plan's
  default); hiding it needs the user's word.
- Job C/D decisions remain outside this checkpoint.

## Recorded for Job C (user, 2026-09-28)
All line numbers refer to `3c0f80c`.
- `docs/DATASETS.md:136` says no completed exact-PMM test evaluation was found;
  the ledger at `:515` records the 2026-09-22/23 openings.
- “Seven early runs” without the six 2026-09-18 reports:
  `docs/DATASETS.md:16-17` and `:137`, `Plan.md:917`, `:943` and `:1223`,
  `README.md:43`, `docs/PARAMETER_FINDINGS.md:33`; the ledger at
  `docs/DATASETS.md:516` has seven plus six.
- `Plan.md:113-117` splits `val_metal_balanced_acc` across blank lines.
- `docs/PARAMETER_FINDINGS.md:49`: “intended primary metal reporting endpoint”.
- RULES table destination references are repo-root-relative and do not resolve
  from its folder.
- Tighten AGENTS (12-KB target), docs/README (5-KB target), the default read
  path (25-KB target) and the PMM README (60-line target). Accepted for now
  (user, 2026-09-28): default read about 27.0 KB, AGENTS 17.0 KB, PMM README
  over 60 lines (27,112 bytes, 17,031 bytes and 63 lines at this record).
- Job C/D: `docs/notebook_outputs/raw/pmm_replay_diagnostic_20260928/remote_SHA256SUMS`
  lists 42 entries; 34 are VM-side files absent from the repository and the 8
  present entries match. Not caused by Job B; raw evidence stays unchanged.
- Historical “challenger” wording left as run: single-GPU and pilot recipe
  sections of the playbook, FOLLOW_UP issue text, `docs/VERY_EXACT_PMM_SETS_PLAN.md`.

## Findings to report (not ours to fix)
- `raw/legacy_nonoverlap_test_access/SHA256SUMS` fails in every fresh checkout (CRLF
  stripped at commit by `core.autocrlf=input`). See `baseline/README.md`.
- 32 pytest errors in `tests/test_pmm_core_assessment.py` in full-suite runs only (test
  order); the file passes alone.
