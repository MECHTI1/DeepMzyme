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
- Target schemes (Job B step-2 review): each new metal campaign trains six only,
  four + six, or four + five + six, chosen by the user and recorded; apply a
  small Plan edit now (RULES B-M01, B-M10). Existing scopes are unchanged.

## State
| Field | Value |
|---|---|
| Job | B |
| Phase | Job B step 2: concrete rewrites and preservation verified; stopped for modified/dropped-rule review |
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
- **Real review stop: PLAN_v2 §11, Job B step 2.** Review [RULES](RULES.md#modified-and-dropped-rules-for-user-review)
  and the concrete AGENTS/STATUS rewrites. No approval has been inferred.
- After approval: finish skill-handoff/ignore/navigation corrections, checker
  and per-command hook; run the fresh-agent test against the approved answer
  key, full Job B verification and final staging. These are not yet complete.
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
  navigation lines. The user's target-scheme decision replaced the five-class
  wording and added a Plan subsection. RULES lists them; counts are now 246
  kept, 139 merged, 74 modified, seven dropped. The rest of the list still
  awaits approval.
- Stopped at the PLAN_v2 §11 Job B step-2 review. No approval is inferred. No
  commit, merge, push, training, GPU action or held-out evaluation occurred.

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
- Await user approval of the concrete modified/dropped-rule proposals.
- Later Job B steps include the `.aiignore` preparation-directory decision;
  default proposal is to expose it and match the planned `.ignore` list.
- Job C/D decisions remain outside this checkpoint.

## Findings to report (not ours to fix)
- `raw/legacy_nonoverlap_test_access/SHA256SUMS` fails in every fresh checkout (CRLF
  stripped at commit by `core.autocrlf=input`). See `baseline/README.md`.
- 32 pytest errors in `tests/test_pmm_core_assessment.py` in full-suite runs only (test
  order); the file passes alone.
