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

## State
| Field | Value |
|---|---|
| Job | A |
| Phase | Job A final corrections and post-edit verification complete; ready for staged commit/integration checkpoint |
| Branch / base | `docs-consolidation` from `metal-pmm-ion-campaign` @ `3e912a6` |
| Worktree | `/media/mechti/Data1/DeepMzyme_worktrees/docs` (clean at checkout) |

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
- Stop for the Job A commit/integration boundary. Commit and push remain with
  the user unless explicitly delegated; **ask before** `git merge --ff-only`.
- After Job A integration and a fresh agent session, begin Job B's campaign
  history preservation and rule inventory before rewriting STATUS or AGENTS.

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
- No commit, merge, push, Job B rewrite, or GPU action has been performed.

## Open questions
- None blocking. Job B–D decisions are listed in PLAN_v2 §15.

## Findings to report (not ours to fix)
- `raw/legacy_nonoverlap_test_access/SHA256SUMS` fails in every fresh checkout (CRLF
  stripped at commit by `core.autocrlf=input`). See `baseline/README.md`.
- 32 pytest errors in `tests/test_pmm_core_assessment.py` in full-suite runs only (test
  order); the file passes alone.
