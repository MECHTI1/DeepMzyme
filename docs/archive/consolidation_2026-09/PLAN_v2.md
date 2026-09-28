# DeepMzyme docs consolidation — Plan v2

- **Version:** v2, 2026-09-28 (~14:00 IDT). It replaces v1, the plan pasted into the
  05:22 UTC review session, and merges in all 33 review amendments
  (`~/.claude/plans/system-reminder-the-user-started-pure-avalanche.md`, section
  "For later"). Appendix A maps each amendment to its place here.
- **Re-verified:** the facts below were re-measured read-only at 13:50 IDT on `3e912a6`.
  Where a re-check changed an amendment, the text says **re-verified**.
- **Repo untouched:** writing this plan changed nothing in the repo. Job A has not
  started. Appendix B is its prepared work order.

---

## 0. Safety invariants (user, 2026-09-28; they override everything below)

1. **Never add or modify any file in `src/` or `scripts/`.**
   - `source_tree_sha256` hashes the path and bytes of every `src/**/*.py` and
     `scripts/*.py` file, untracked files included (`src/benchmarking/pmm_ion_campaign.py:362-369`,
     `run_pmm_core_campaign.py:34-39`).
   - It is pinned to `adc95c42…` in `run_pmm_core_campaign.py:20` and in all five
     `docs/plans/*.json` scopes.
   - Recomputed at 13:50: equal. Even a scratch `.py` file there breaks every PMM runner.
2. **Never move or rename `docs/plans/*.md` or anything under `docs/notebook_outputs/raw/`.**
   Checksummed evidence links to them. In addition, v2 treats every file in
   `docs/plans/` as **no-edit**: Codex owns them, and evidence records the SHA-256 of
   `pmm_replay_diagnostic_v1*.md`.
3. **Never add tests under `tests/`.**
   - `pyproject.toml:46` sets `testpaths=["tests"]`.
   - `.github/workflows/cpu-ci.yml` runs `pytest` on every push to every branch.
   - Standalone checkers go in `tools/`. That folder does not exist yet, is outside
     the source hash, and pytest does not collect it.
4. **Job D (raw evidence out of git) is deferred until the PMM campaign closes.**
   Replanning it later needs your explicit lift of invariant 2.
5. **Stay inside doc-consolidation scope.**

---

## 1. State at 13:50 IDT (re-verified)

- **Shared checkout** (`/home/mechti/PycharmProjects/DeepMzyme`):
  - branch `metal-pmm-ion-campaign` at `3e912a6`, equal to its origin;
  - `git status --porcelain` is empty, so v1's precondition (amendment 2) holds now;
  - 29 commits ahead of `main` (`34633fa` = `origin/main`), 0 behind. The review said 25.
- **Codex:** idle since 11:14 IDT. Its last action removed the incomplete Antigravity
  GVP draft (`3e912a6`).
- **PMM campaign: paused, not closed** (STATUS:10-29, your decision of 2026-09-28):
  - the nine fold-0 fits are kept;
  - the 36 fold-1–4 fits are deferred;
  - no fits are authorized;
  - the last verified VM state is TERMINATED.

  Codex's doc-edit rate is therefore zero now. It comes back when the campaign resumes.
- **Antigravity GVP draft:** the code draft is gone and backed up in
  `~/gvp_wip_2026-09-28*`. The plan is committed at
  `docs/plans/gvp_and_esmc_evidence_ranked_improvement_plan.md` (`6d4e674`).
- **Disk:**
  - `/` has 2.7 GB free (99% used). `.git` (490 MB) is on `/`.
  - `/media/mechti/Data1` (ntfs3) is mounted with 66 GB free.
  - A worktree holds about 322 MB of tracked files (2,042 files).
- **Worktrees:**
  - the shared checkout;
  - `DeepMzyme_pmm_source_audit`;
  - two frozen checkouts under `/media/.../campaigns/_code/`;
  - `/tmp/deepmzyme_bundle_release_final`, marked "prunable" (leave it; see amendment 31).
- **Git config:** `core.hooksPath` and `extensions.worktreeConfig` are unset, and no
  active hooks exist in `.git/hooks`.
- **Default read path today: 254,445 bytes (~64k tokens):**
  AGENTS 43,624 + Plan 83,267 + STATUS 42,062 (547 lines) + `docs/README.md` 8,320 +
  DATASETS 46,801 + evidence index 30,371.

---

## 2. Goal (v1, numbers updated)

Agents read too much old, duplicated and conflicting text before they act. The default
list costs ~254 KB. The 2026-08-20 cleanup
(`docs/archive/plans/PROJECT_CLEANUP_PLAN_2026-08-20.md`) failed because nothing
enforced it. This plan has four parts:

1. Fix the contradictions.
2. Give every fact one owner file.
3. Move campaign-specific text into per-campaign folders.
4. Add a check that stops the text from growing back.

Do not add an "AI index" document on top of the existing ones.

---

## 3. Where and how to work

### 3.1 Base and branch (amendment 1)
- Branch from the campaign tip:
  `git -C <repo> worktree add <wt> -b docs-consolidation metal-pmm-ion-campaign`.
- Sync with the local ref `metal-pmm-ion-campaign`; no fetch is needed.
- `main` gets the work when the campaign branch merges.

### 3.2 Worktree location (amendment 5)
- `<wt>` = `/media/mechti/Data1/DeepMzyme_worktrees/docs`. It is outside `campaigns/`.
- **Right after checkout,** `git -C <wt> status --porcelain` must be empty.
  - If only file modes differ (NTFS), use `git -c core.fileMode=false` on every
    command in `<wt>`. Never change the repo config.
  - If any content differs (for example CRLF), stop: run `git worktree remove <wt>`,
    then ask. The fallback is a worktree on `/`: 322 MB, leaving ~2.4 GB. It needs your OK.
- **Never** run `uv sync` in `<wt>`.
- Copy without dereferencing symlinks.

### 3.3 The shared checkout is read-only for the agent (amendment 8)
- The PMM code-bundle builder ships untracked, non-ignored files
  (`src/benchmarking/pmm_campaign_bundle.py:161`, `git ls-files --others --exclude-standard`).
- Run metadata records dirty state.
- A plain `git status` takes `index.lock`.
- So every agent command there is a `git --no-optional-locks` read or a plain file read.
  All writing happens in `<wt>`.
- The only exceptions are `git worktree add` and the branch ref, which write inside `.git/`.

### 3.4 Two kinds of work (amendment 3)
- **Codex-owned files.** Edit them only inside a doc freeze (3.5):
  - `EXPERIMENT_STATUS.md` and `AGENTS.md`;
  - the metal playbook's PMM section;
  - `FOLLOW_UP_TECHNICAL_ISSUES.md`, `PARAMETER_FINDINGS.md` and `DATASETS.md`;
  - `docs/notebook_outputs/README.md` and `docs/README.md`;
  - `.gitattributes`;
  - `docs/plans/*` (never edited; see invariant 2).
- **Codex-independent work.** Prepare it in `<wt>` at any time:
  - closed campaigns;
  - the GPU/CV doc merges;
  - archive moves;
  - the checker;
  - ANSWER_KEY.

### 3.5 Doc freeze
A freeze starts when you confirm all three:
1. no Codex thread is running;
2. none will start until the freeze ends;
3. the shared checkout is clean.

It ends when you have fast-forwarded and pushed. Because the campaign is paused, one
freeze may cover Job A and Job B if you want.

### 3.6 Merge-back: one path only (amendment 4)
1. The agent stages in `<wt>`. You commit in `<wt>`.
2. If the campaign tip moved since the branch point, rebase:
   `git -C <wt> rebase metal-pmm-ion-campaign`. On a conflict in a Codex-owned file, stop and ask.
3. In the shared checkout, with Codex idle, you run
   `git merge --ff-only docs-consolidation && git push`.
4. No PR merges into the campaign branch while Codex is live.
5. Pushing `docs-consolidation` itself as a backup is optional. CI runs on it; that is
   harmless because `tests/` does not change.

### 3.7 Git
- Stage only. Never commit, push, reset, stash or rewrite history unless you ask for it.
- Suggest `git commit -m`, never `-a`.

### 3.8 Rollback
- Undo a merged job with `git revert <commit>`.
- To abandon everything: `git worktree remove <wt>`, then delete the branch.
- The campaign branch is untouched until you fast-forward it.

### 3.9 Codex handover (amendment 32)
A running Codex thread keeps the `AGENTS.md` it loaded at start. `AGENTS.md:83` reads
STATUS only "if present at the repository root". After each fast-forward:
1. stop the old thread;
2. start a new one;
3. paste the handover note (section 14).

---

## 4. Hard constraints

1. **Text only.** Never edit:
   - `src/`, `scripts/` or `tests/`;
   - root `*.py` files;
   - `notebooks/`;
   - the `prepare_*`, `CARE_*` or `CLEAN*` code;
   - `requirements/`, `pyproject.toml`, `uv.lock` or `.github/`;
   - `docs/plans/*`.

   The only non-doc files allowed, all new or fixed:
   - `tools/check_docs_contract.py`;
   - `.githooks/pre-commit`;
   - `.ignore` and `.aiignore`;
   - `GEMINI.md`;
   - the text of `.agents/skills/gpu-use-skill/`.
2. **Files that code reads.** Before moving, renaming or editing any file, run
   `rg --no-ignore -n "<basename>" src scripts tests notebooks *.py`.
   Re-run the full inventory at the start of every job (amendment 16). Known at 13:50:
   - **Both playbooks keep their paths.** The first ```` ```python ```` block of each
     is exec'd by:
     - `tests/test_standalone_baselines.py:185-235`;
     - `src/verify_colab_notebook_smoke.py`;
     - notebook cell `b16d9206c7e17d9e`;
     - `src/run_ec_baselines.py:83-94` (EC).

     The metal playbook is also split on `### Exact standalone notebook block`
     (L1512; fence at L1514). Other readers and their CI tests:
     - `src/run_metal_architecture_pilot.py`, `run_metal_ring_pilot.py` and
       `run_metal_coordination_geometry_pilot.py`;
     - `src/serial_metal_campaign/profile.py` (`_templates`).

     **Invariants:**
     - LF line endings;
     - the heading stays unique and before the first python fence;
     - no new python fence goes above it;
     - that block stays byte-identical.
   - **`docs/plans/*`** stays as is:
     - `run_pmm_core_campaign.py`, `pmm_core_assessment.py`,
       `run_pmm_five_class_screen.py` and `audit_pmm_screen_agreement.py` read and
       hash-bind the JSONs;
     - `tests/test_pmm_screen_agreement.py` hash-checks `pmm_screen_agreement_v2_1.json`;
     - `scripts/run_metal_5fold_cv.py:1068` names the PMM final-plan file.
   - **Keep at the root:**
     - `Plan.md`: notebook cell `0083d7bd` and the CLEAN notebook's
       `find_project_root` use it as the repo-root marker (amendment 13);
     - `EXPERIMENT_STATUS.md` (amendment 11).
   - **Snapshot builder** (amendment 14). `scripts/prepare_colab_smoke_snapshot.py:26-32`
     ships `docs/`, `Plan.md`, `README.md` and `EXPERIMENT_STATUS.md`, and excludes
     `docs/archive/` and `raw/`. So never archive a playbook.
   - **`tests/smoke_checks.py`** needs (amendment 15):
     - `README.md` and `docs/archive/workflows/list_train_commands_legacy.md` to exist;
     - `bench/README.md`, `bench/schemas/*.json` and the root `benchmark_step*.py` to exist;
     - the bare string `src.training.run` never to appear (`:2663-2669`). Don't write
       it, even in a warning.
   - **Raw evidence read by tests or `src/`:**
     - `raw/colab_care_cache_smoke_20260914/`;
     - `raw/metal_architecture_pilot_20260915/readiness/expected_split.json`;
     - `raw/ec1_standalone_v12_20260914/expected_split.json`.
   - **Other links from code:**
     - `src/serial_metal_campaign/reporting.py` → a summary file (immutable anyway);
     - `src/project_paths.py` and `src/manage_structure_store.py` → `docs/STRUCTURE_STORE.md` (unchanged);
     - the notebook's markdown → the playbook, the notebook guide, DATASETS, STATUS and Plan;
     - `scripts/colab_artifact_streamer.py` (comment) → `docs/agents_report/HANDOFF_ZENODO_PMM_EXACT_BENCHMARK.md`.
       That file may move; record it in `docs/MOVED.md`.
   - **FOLLOW_UP `TECH-###` anchors stay resolvable** (amendment 11). An archived issue
     keeps a one-line stub under its old heading if any file links to its anchor.

   Every move you skip goes in the report under "blocked by code".
3. **Evidence is immutable.**
   - Never edit `docs/notebook_outputs/raw/` or `docs/notebook_outputs/summaries/`.
   - Keep `.gitattributes`-protected directories and hash-bound files byte-identical.
   - Exclude every `*/provenance/SHA256SUMS` tree from every edit or move sweep
     (amendment 17). Verification may read them.
4. **No science changes** except D1–D2.
   - Carry every number, hash, seed, run ID, fold definition and gate over exactly.
   - "Remove" means `git mv` to `docs/archive/`. Nothing is deleted.
5. **Don't trust earlier AI summaries.** Work from the files.
   - Antigravity's "AI_AGENT_INDEX" and "Rosetta Stone" prompts contain invented facts.
   - Never cite private paths such as `~/.gemini/...` in this public repo (amendment 23).
6. **No new prose.**
   - New text is allowed only for templates, the fold-regime table, the D1/D2
     corrections, campaign READMEs and links.
   - Every new factual sentence cites a file and line, or a command.
   - Use short, plain sentences and no marketing words.
7. **Never drop a rule silently.**
   - Before rewriting AGENTS, STATUS or Plan, list every normative statement in
     `docs/archive/consolidation_2026-09/RULES.md` with its source line.
   - After the rewrite, mark each one kept, merged, modified or dropped. You review the
     dropped and modified lists.
   - Job A records the few rules it modifies.
   - Job B also inventories the normative statements of `9a25f36^:EXPERIMENT_STATUS.md`,
     which were dropped silently once already (amendment 24).
8. **Resume safely.**
   - After every step, update `docs/archive/consolidation_2026-09/PROGRESS.md` (job,
     phase, done, next, open questions). Read it first after a compaction or in a new session.
   - Phase 0 copies this file there as `PLAN_v2.md`.
9. **Keep your context small.** Work from inventory tables and open single sections.
10. **Search with `rg --no-ignore`** for every safety or verification search.
11. **Never do any of these:**
    - run `~/deepmzyme-vm/bin/*`, `colab`, training, HPO or any GPU command;
    - touch `/media/mechti/Data1/DeepMzyme_Data/campaigns/`;
    - touch the `DeepMzyme_pmm_source_audit` worktree;
    - run a blanket `git worktree prune` (amendment 31);
    - make a repo-wide `git config` change while the campaign is open.

---

## 5. Out of scope

- **The GVP-improvement work.** Its plan stays at
  `docs/plans/gvp_and_esmc_evidence_ranked_improvement_plan.md`. Don't move, index,
  summarize, plan or cite it as a contradiction. The code draft no longer exists
  (`3e912a6`).
- **`docs/agents_report/RESUMED_REVIEWS_HANDOFF_20260923.md`** stays at its path. It is
  allowed in the file list and may be cited as D2 evidence.
- Exception: the D2 ledger records data-integrity facts, even where they touch this work.

---

## 6. User decisions

- **D1: the primary route is the CLI runners on the GCP L4 VM**, run through
  `~/deepmzyme-vm/bin/*` under `gpu-use-skill`.
  - Colab is the authorized fallback, and the notebook is a secondary interface.
  - Rewrite the Colab-first and Drive-mandatory phrases.
  - Keep the G4 budgets, labeled as historical.
  - Exact targets are in Appendix B (step B4).
- **D2: the test ledger and the benchmark wording** (amendments 18–23, **re-verified**):
  1. **The exact-PinMyMetal test was opened in two rounds.** All files are local and
     git-ignored under the repo-root `runs/`.
     - **2026-09-22:** 3 single-split reports in `runs/benchmark_exact_pinmymetal/*/test_report.json`.
       Byte-identical copies are in `DeepMzyme_Data/notebook_outputs/benchmark_exact_pinmymetal_recovery_20260922/`.
       Commit `9a069d2` reports fold-0/1 test scores (77.62% / 74.21%, collapsed-4).
     - **2026-09-23:** 15 fold reports and 3 ensemble reports in `runs/benchmark_exact_pinmymetal_5fold/`.
     - The ledger records each path and its sha256.
  2. **Non-overlap test, 2026-09-18 — corrected: 6 distinct reports, not 3.** All show
     `split_type: non_overlapped_pinmymetal` and 352 pockets, and no doc mentions them.
     - `DeepMzyme_Data/notebook_outputs/benchmark_50epochs/{only_esm,enhanced_only_gvp,enhanced_gvp_esmc}/test_report.json`
       (02:23–03:01). The `…_ARCHIVE_successful_and_failed/` copy has identical hashes.
     - `runs/benchmark_replicated_72pct/benchmark_{…}/test_report.json` (04:26–04:51).
       `DeepMzyme_Data/notebook_outputs/benchmark_replicated_72pct/` has identical copies.
     - So `DATASETS.md:294` ("exactly seven"), `:300` and `:516` are wrong. **Needs your yes (Q3).**
  3. **Zenodo relaunch `pmm-zenodo-v2`: partly answered.**
     - The last mirrored state (`~/zenodo_pmm_artifacts`, 2026-09-24 15:34 IDT) shows
       fold-0 training with `run_test_eval: false`, no test paths and no test report.
     - But the runner evaluates the test after each finished fold by default
       (`scripts/run_zenodo_pmm_exact_5fold_cv.py:794`). If fold 0 finished after the
       mirror stopped, the Zenodo test was opened.
     - The ledger says "unknown" unless you know. **(Q4)**
  4. **Wording (amendment 21):**
     - "316 structures" = 316 structure files = 313 PDB IDs.
     - 1,597 is DeepMzyme's catalytic-pocket subset of the exact train membership
       (`EXACT…:62`, `playbook:610`).
     - 7,920 is the number of train rows in PinMyMetal's released source file; the
       effective cohort is uncertified.
     - 177 = shared PDB IDs (179 identical filenames).
  5. **The sentence, placed once** in `docs/EXACT_PINMYMETAL_5FOLD_CV_REPRODUCIBILITY.md`
     (later that campaign's README):
     > "Not a like-for-like comparison with PinMyMetal: this cohort has 1,597 training
     > pockets (DeepMzyme's catalytic-pocket subset of the exact train membership;
     > PinMyMetal's released source file has 7,920 training rows, and its effective
     > training cohort is not certified), PinMyMetal's fold assignments were not released,
     > and the exact-PinMyMetal test set shares 177 PDB IDs (179 identical structure
     > filenames) with its training set (see `docs/DATASETS.md`, Exact PinMyMetal)."

     Elsewhere (STATUS, the ledger, the index row), write one short line and link to it.
  6. **Selection influence: yes, exploratory.**
     - Test deltas were used to rank models and to claim "beats PMM".
     - `gvp_fusion_exact_pocket_l4_v2` is described as a continuation of this benchmark.
     - A 50/50 blend was argued from a test score of 81.17%. On validation it is worse
       (77.70 vs 79.84 BA4) and "must not be adopted"
       (`RESUMED_REVIEWS_HANDOFF_20260923.md:166-169`).
     - **No model was promoted** (confirmed).
  7. **More EXACT-doc errors (amendment 22):**
     - `:23` attributes Fig 2b to "the independent 352 test pockets". PinMyMetal's test
       side has 1,488 rows; 352 is DeepMzyme's exact-test pocket count.
     - `:29` calls collapsed-4 accuracy "raw accuracy".
     - The Table 1 CV values are maxima over epochs. The selected-checkpoint out-of-fold
       values are 78.98 / 73.87 / 79.84 (`RESUMED_REVIEWS_HANDOFF_20260923.md:170-174`).
  8. **Clean-up:**
     - Remove every "identical dataset / identical 5-fold stratification / full
       replication / parity" claim and every hype word. Keep all numbers.
     - Don't edit summaries. Correct them through their index row.
     - The ledger notes that the exact test structures also occur in the historically
       opened non-overlap test.
- **D3: raw evidence — deferred** (invariant 4). See the Job D notes in 12.4.
- **D4: text only.** List code redundancy for a later pass; don't fix it.

---

## 7. Target structure (amended)

```
AGENTS.md              ≤12 KB  rules only
CLAUDE.md              1 line → AGENTS.md (unchanged)
GEMINI.md              1 line → AGENTS.md (new)
README.md              public overview + ≤30-line quick start
Plan.md                ≤35 KB  science/design policy (stays at root: repo-root marker)
EXPERIMENT_STATUS.md   ≤6 KB   overwritten in place, never appended (stays at root)
docs/README.md         ≤5 KB   owner table + link to docs/MOVED.md
docs/MOVED.md                  old path → new path (visible: not under an .ignore path; amendment 9)
docs/DATASETS.md               + "Storage and backups"; corrected ledger
docs/PARAMETER_FINDINGS.md     validation-only findings (name kept)
docs/FOLLOW_UP_TECHNICAL_ISSUES.md   open issues + anchor stubs (name kept)
docs/COMPUTE.md                GCP primary, Colab fallback, local CPU limits
docs/CV_RUNNERS.md             5-fold runners, fold regimes, which campaign used which
docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md   reusable stages + the code-read Common70 section
docs/EC_TRAINING_PIPELINE_PLAYBOOK.md      path fixed by code
docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md notebook option meanings (secondary route)
docs/STRUCTURE_STORE.md        unchanged (read by src/project_paths.py)
docs/plans/                    unchanged, never moved or edited; campaign READMEs link here
docs/campaigns/<id>/           active, paused and planned campaigns
docs/archive/campaigns/<id>/   closed campaigns
docs/notebook_outputs/         evidence index, summaries/, raw/ (raw stays in git until Job D)
docs/archive/consolidation_2026-09/   PLAN_v2, MAP, RULES, PROGRESS, ANSWER_KEY, hashes, baselines, inventory/
tools/check_docs_contract.py   the checker (not under tests/)
.githooks/pre-commit           runs the checker (per-command during the campaign)
```

- The component READMEs stay as they are.
- **Default read path:** `AGENTS.md` → `EXPERIMENT_STATUS.md` →
  `docs/campaigns/<active>/README.md`, ≤25 KB in total.
- **Campaign README** (≤60 lines, v1 template):
  - `Status: active|paused|closed|planned (date)`;
  - goal, target schemes trained, evaluation endpoint, example unit, fold regime,
    dataset and compute route;
  - validation result and evidence grade, and held-out access;
  - files here, evidence links, supersedes / superseded by, and "How to continue".

  Campaigns with plans in `docs/plans/` link to them there.
- **STATUS template:** the v1 template, unchanged.

---

## 8. File-by-file map (amended; verify each before acting)

Closed campaigns go to `docs/archive/campaigns/<id>/`; the rest go to
`docs/campaigns/<id>/`. Record every move in `docs/MOVED.md`.

| Source | Action | Job |
|---|---|---|
| STATUS dated entries | Copy verbatim to each campaign's `log.md` **before** overwriting STATUS. Recover lost history from `9a25f36^` (437 lines cut on 09-24, incl. *Test-use status*, *Current blockers*, *Immediate next action*, *Update rule*), `9a069d2^` (09-16/17 entries) and `b45b893:72-86` (amendment 24) | B |
| PMM ion work (amendment 26) | One folder, `docs/campaigns/pmm_ion_metal/`, with README sub-batches: v1 audit (closed) · five-class screen v1 (closed) · replay diagnostic (closed) · core v1 (superseded) · **core v2 (paused: 9/45 fold-0 fits kept, 36 deferred; STATUS:10-29)** · binding-aware arms (paused). It links to `docs/plans/metal_level_metal_task_compared_PMM_final_plan.md`, `pmm_core_execution_v1.md`, `pmm_replay_diagnostic_v1*.md`, `pmm_gpu_concurrency_probe.md` and the JSONs, all left in place | B (folder) |
| Metal playbook "PMM ion-level metal comparison campaign" (starts ~L33, ~433 lines; no python fence) | → `campaigns/pmm_ion_metal/recipe.md`, **only in a freeze or after the campaign closes** | C |
| `docs/GPU_EXECUTION_CASCADE_PLAYBOOK.md`; GCP runbook "PMM campaign setup" | → `campaigns/pmm_ion_metal/compute.md` (same freeze rule) | C |
| `docs/EXACT_PINMYMETAL_5FOLD_CV_REPRODUCIBILITY.md` | Job A corrects it in place. Job C: results + D2 sentence → `archive/campaigns/exact_pmm_5fold_2026-09-23/README.md`; runner usage → `CV_RUNNERS.md` | A → C |
| Zenodo docs (`ZENODO_…md`, `agents_report/HANDOFF_ZENODO…`, `…EXECUTION_LOG.md`) | Dataset facts → `DATASETS.md` (merge; freeze); runner usage → `CV_RUNNERS.md`; the rest → `zenodo_pmm_exact_2026-09-24/`. The streamer comment path goes stale (can't edit `scripts/`) → report it | C |
| `docs/GENERALIZED_METAL_5FOLD_CV_GUIDE.md` | → `CV_RUNNERS.md` | C |
| GVP-fusion L4 work (amendment 25) | **Two** folders: `gvp_fusion_diagnostic_l4_v1` (stopped 09-24; `agents_report/GVP_FUSION_L4_EXECUTION_20260924.md`, `archive/experiments/gvp_fusion_l4_v1_stopped_plan.md`) and `gvp_fusion_exact_pocket_l4_v2` (stopped at 3/20, validation-only; playbook sections + `agents_report/GVP_FUSION_EXACT_POCKET_L4_V2_EXECUTION.md`) | C |
| `agents_report/TEMP_CHAT_SUMMARY_20260924.md` | → archive | C |
| Single-GPU text (playbook section, Plan section, guide/runbook parts, `GPU_RUNTIME_EFFICIENCY_PLAN.md`) | → `metal_single_gpu/` with **two** items: 20h_v2 admission (closed; full training rejected) and the authorized continuation (paused 09-17). Fix playbook L3-4 "the new campaign" (amendment 25) | C |
| Pilot sections (architecture, coordination-geometry, matched RING, frozen GVP capacity recovery) | → `metal_pilots_2026-09-15/` (**the Job C pilot**). Plan keeps only the generic geometry-control rules | C |
| Metal×EC1 association analysis (CPU, 09-15) | Its own closed folder, `metal_ec1_association_2026-09-15/`, separate from the pilots (amendment 25) | C |
| `docs/REMOTE_HOMOLOGY_ADDENDUM.md` + sequence-remoteness sections | → `remote_homology_v1/` (closed). Plan keeps its policy paragraph. `raw/remote_homology_v1_20260917/` stays in place | C |
| `docs/plans/pocket_level_EC_task_compare_clean30_final_plan.md` | **Stays** (invariant 2). Create `campaigns/ec1_clean30/README.md` (planned) linking to it | C |
| `docs/VERY_EXACT_PMM_SETS_PLAN.md` | "Planned; partly realized via `--source-cohort-csv` (train side only)". Ask before calling it superseded (amendment 26) | C |
| `docs/GCP_GPU_RUNBOOK.md` + `docs/COLAB_GPU_RUNBOOK.md` | → `docs/COMPUTE.md`. Update the skill links and `GETTING_STARTED.md:19`'s anchor link. `~/deepmzyme-vm/AGENTS.md:147` links to `docs/GCP_GPU_RUNBOOK.md`: leave a one-line stub there, or you update the controller (amendment 28) | C |
| `docs/GETTING_STARTED.md` | Essentials → README + `COMPUTE.md`; file → archive | C |
| `docs/BACKUP_AND_DATA_INVENTORY.md` | → DATASETS "Storage and backups" (freeze); file → archive | C |
| `docs/REPRODUCIBILITY_REMEDIATION_PLAN.md` | → archive | C |
| FOLLOW_UP resolved issues | **7**, not 8: TECH-001, 007, 013, 015, 016, 021, 024. **TECH-017 stays open**; the legacy path is still affected (amendment 28). → `docs/archive/issues_resolved.md`, with anchor stubs where linked. Never reuse TECH numbers (freeze) | C |
| DATASETS bundle v10/v11 details | → `docs/archive/datasets_bundle_history.md` (freeze) | C |
| Playbook warning sections | Replace with 2-line links to Plan / FOLLOW_UP | C |
| `docs/notebook_outputs/README.md` | Keep **every** row and link. Job A adds the exact-PMM 5-fold and exact-pocket L4 v2 rows. Job C groups rows by campaign and adds any rows still missing (`gvp_fusion_diagnostic_l4_v1`, PMM v1 audit, metal×EC1 association) | A, C |
| `docs/PARAMETER_FINDINGS.md` | Job A adds validation-only entries. Job C removes status-like preambles | A, C |

---

## 9. Contradictions (amended)

1. **Test ledger vs the 09-18/22/23 test evaluations** → D2 (Job A).
2. **Target schemes** (Job B). The project trains all three schemes:
   - `four_class`: the primary direct arm;
   - `six_class`: the paired challenger, reported as collapsed-four;
   - `five_class`: added by your decision in `docs/plans/pmm_core_scope_v2.json` and
     trained alone in the five-class screen.

   All are compared on the common-four endpoint.
   - AGENTS' "treat `five_class` only as an explicitly labeled alternative" is out of date.
   - Keep `four_class` as the primary formulation, and point to the v2 scope for
     `five_class`.
   - Record the change as *modified* in `RULES.md` for your review.
   - Every campaign README states its trained schemes and its endpoint.
3. **"5-fold" means several regimes** (Job A). Add one table to Plan after "Metal example
   terminology" (`Plan.md:199`):

   | Regime | Grouping | Campaigns |
   |---|---|---|
   | `pocket_id`-stratified 5-fold | One PDB can appear in several folds. Fold 0 has 58 val pockets whose PDB ID is also in train (`RESUMED_REVIEWS_HANDOFF_20260923.md:175-177`) | Exact-PMM benchmark; exact-pocket L4 v2; Zenodo runner (parent pocket) |
   | Frozen PDB-grouped 5-fold | Fold files frozen | PMM ion campaign |
   | Single train/val split | `pdbid` groups; `VAL_FRACTION` 0.15 in the recipe, **0.18** as the notebook's live value (notebook L222; TECH-003) | Notebook stages 0–5 |
   | Stage 6 grouped k-fold × seeds | `pdbid` groups | Notebook Stage 6 (`group_kfold_seed_repeat`) |

   - The Stage 6 row is new (**re-verified**; v1 listed three regimes).
   - The splitter is a custom greedy, label-balanced assignment (`src/training/splits.py`:
     `split_pockets_k_fold` at 210, `val_assignment_penalty` at 397), not sklearn (amendment 27).
4. **Compute wording** → D1 (Job A).
5. **Stale STATUS blocks** (Job B):
   - "Current objective" (09-23);
   - "Next campaign" (09-16);
   - L7 "Last execution audit: 2026-09-15" (amendment 28);
   - the file currently ends mid-sentence (re-check).
6. **Missing index entries** → Job A, for the exact-PMM 5-fold benchmark and exact-pocket
   L4 v2 (re-verified absent in the index, FINDINGS and DATASETS). Test metrics stay out
   of FINDINGS.
7. **Wrong CLI flags in AGENTS.md:688-696** (Job B). Of the seven, 3 don't exist and 4 are
   underscore misspellings of real dash flags (amendment 28). Delete them and point to
   `src/train.py --help`.
8. **The Plan.md CLI table** lacks about 51 committed flags (amendment 28; re-count in
   Job C). Remove it: `config.py` / `--help` is the authority. The "seven 09-28 flags"
   belonged to the removed GVP draft, so that point is moot.
9. **"Group VIII" → "Class VIII"** in active docs (Job C): 7 occurrences in 3 files, plus
   the EXACT doc and STATUS; re-count. Summaries stay unedited.
10. **The skill's "Handoff" section** → reword it to the STATUS template and the `log.md`
    rule (Job B).
11. **`.aiignore`** (Job B):
    - The bare `README.md` entry hides all 25 READMEs.
    - `.data`, `.md_files` and `Documenation` match nothing.
    - Replace them with the `.ignore` list. `prepare_training_and_test_set` stays hidden
      only if you say so.
12. **Stale counts and labels** (amendment 28):
    - `playbook:97-99`;
    - playbook L3-4 "the new campaign";
    - STATUS L7 (Job B/C).

---

## 10. AGENTS.md content (≤150 lines) and documentation rules

**Content:** as in v1, with these changes:
- **Compute:** the D1 route, plus "one active GPU; one coordinator".
- **Search:** "`.ignore` hides `docs/archive/` and `raw/`; use `rg --no-ignore`".
- **Codex note:** "Restart your thread after AGENTS.md changes."

What moves out is as in v1:
- the navigation map → `docs/README.md`;
- the stage-answer format → the metal playbook (check first);
- the review-only guidance → condensed to 5 lines;
- the flag list → deleted.

**Documentation rules** (paste into AGENTS.md):
1. Each fact has one owner file (see `docs/README.md`). Other files link to it; they never copy it.
2. `EXPERIMENT_STATUS.md` is overwritten, never appended. Copy history to
   `docs/campaigns/<id>/log.md` first.
3. Campaign-specific text lives only in `docs/campaigns/<id>/`. The exception is
   `docs/plans/`, which is never moved or edited; the campaign README links to it.
   To close a campaign:
   - set `Status: closed`;
   - `git mv` its folder to `docs/archive/campaigns/<id>`;
   - add one line to `PARAMETER_FINDINGS.md`;
   - update STATUS.
4. A new top-level doc is allowed only if an existing one is merged or archived in the
   same change.
5. Resolved issues move to `docs/archive/issues_resolved.md`. If anything links to the
   issue's anchor, keep a one-line stub under its `TECH-###` heading.
6. After editing any `.md` file, run
   `/home/mechti/miniconda3/envs/DeepMzyme/bin/python tools/check_docs_contract.py`.
   Fix the text, not the checker.

---

## 11. Jobs and sequencing (amended)

**After each job:**
1. Sync: rebase onto the campaign tip if it moved, port Codex's doc changes into the new
   structure, and list what was ported.
2. Verify.
3. Report.
4. **Stop.** You commit, fast-forward, push and restart Codex.

Every stop is a real wait.

- **Now:** nothing in the repo until you approve v2 and open freeze window 1.
- **Freeze window 1:**
  1. Create the worktree.
  2. Run Phase 0.
  3. Write `ANSWER_KEY.md`. **Stop for approval.**
  4. Do Job A (D1; D2 with 18–23; the fold table; the index rows).
  5. Verify, then stage. **Stop.**
  6. You commit, fast-forward, push and start a new Codex thread (Appendix B).
- **Freeze window 2 — Job B:**
  1. Create the campaign `README.md` + `log.md` folders needed for the STATUS entries;
     closed ones go straight to the archive.
  2. Rewrite STATUS and AGENTS via `RULES.md`. **Stop on the dropped/modified list.**
  3. Fix contradictions 2, 5, 7, 10 and 11.
  4. Enforcement (Phase 5).
  5. Run the fresh-agent test.
  6. Fast-forward and hand over.

  Link only to files that exist at the end of Job B.
- **Job C:**
  1. **Pilot:** move `metal_pilots_2026-09-15` completely. **Stop for your OK.**
  2. Do the Codex-independent rest in `<wt>` at any time.
  3. Do the Codex-owned parts only in a freeze: the PMM playbook section, FOLLOW_UP,
     FINDINGS, DATASETS, the evidence index and `docs/README.md`.
  4. Show the Plan outline. **Stop.** Then rewrite Plan. **Stop on the dropped list.**
  5. Write README, `docs/README.md`, `COMPUTE.md` and `CV_RUNNERS.md`, and fix
     contradictions 8, 9 and 12.
  6. Set caps to the achieved size + 10%, then make all checks strict.
- **Job D:** only after the campaign closes, and replanned then (12.4).

**Caps are targets, not orders.** Never drop a rule, number or caveat to meet a cap.

---

## 12. Phases

### 12.0 Phase 0 — Snapshot (freeze window 1)
- **Pre-flight** (shared checkout, `--no-optional-locks` reads only):
  - status is empty;
  - HEAD equals `origin/metal-pmm-ion-campaign`;
  - `source_tree_sha256` = `adc95c42…` (read-only inline Python);
  - record `sha256sum docs/plans/*`;
  - check `df -h / /media/mechti/Data1`.
- **Worktree:** create it (3.1/3.2), confirm it is clean, copy this plan to
  `PLAN_v2.md`, and start `PROGRESS.md`.
- **`before_hashes.txt`:**
  - sha256 of every tracked `.md`/`.json`/`.tsv`/`.csv` under `docs/`, plus the root docs;
  - the `sha256sum -c` result of every `SHA256SUMS` under `docs/`;
  - the sha256 of the first python block of each playbook.
- **Baseline tests, in `<wt>` only** (amendment 30, plus the local resource cap):
  ```
  systemd-run --user --scope --quiet -p CPUQuota=200% -p MemoryMax=3G -p MemorySwapMax=0 \
    env GIT_OPTIONAL_LOCKS=0 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    taskset -c 0,1 nice -n 19 /home/mechti/miniconda3/envs/DeepMzyme/bin/python \
    -m pytest -p no:cacheprovider -q -rs
  ```
  Then run `tests/smoke_checks.py` the same way. Save both outputs. Compare worktree runs
  only with each other. Skips and known failures (for example TECH-019 and
  `test_explicit_membership.py::test_outer_loader_not_deserialized`) are not ours.
- **Secret scan** (amendment 29):
  - Preferred: `gitleaks git --log-opts=--all --redact`.
  - If gitleaks is absent, run a credential-format regex over the tree and `git log -p --all`,
    excluding the protein-sequence "AKIA" hits in the CARE CSVs. Installing gitleaks is your call.
  - Report masked results only. The review found 0 real hits in 343 commits.
- **Code↔doc inventory re-run** (amendment 16): basename and path searches with
  `rg --no-ignore`. Write it to `inventory/code_coupling.md` and diff it against 4.2.
  A new consumer updates 4.2 before any edit.
- **Per-group inventory subagents** (v1 section 13) → `inventory/<group>.md`.
- **Baseline fresh-agent run:** a fresh agent reads today's default set and answers the
  six questions. Save its answers.
- **`ANSWER_KEY.md`:** the six answers with sources (draft in Appendix C). **Stop.**

### 12.1 Phase 1 — Contradiction fixes
Small, separate edits.

### 12.2 Phase 2 — Campaign folders
`git mv` whole files. Split sections by moving text verbatim, under 4.2.

### 12.3 Phase 3 — Rewrites
STATUS, AGENTS, Plan, README and `docs/README.md` to their caps, plus the `COMPUTE.md`
and `CV_RUNNERS.md` merges.

### 12.4 Phase 4 — Raw evidence (Job D, DEFERRED)
When the campaign closes, replan Job D with these points (amendments 17 and 33):
- **Your lift of invariant 2 is required first.**
- The keep list is "every raw dir of the active campaign", not a fixed list. It already
  includes `raw/pmm_core_continuation_20260928/`.
- `.gitattributes` protects `raw/remote_homology_v1_20260917/`, which D3 would remove.
  You decide which wins.
- `EVIDENCE_MANIFEST.tsv` is per file, because kept summaries publish SHA-256 values of
  manifests inside dirs Job D would remove.
- Exclude every `*/provenance/SHA256SUMS` tree.
- Staging goes on `/media`, never `/`.
- Job D does not serve the token goal:
  - agents don't read `raw/` by default;
  - history keeps the blobs, so clone size doesn't shrink.
- Its costs: about 175 dangling links, disk space, and an HF upload that needs your confirmation.

### 12.5 Phase 5 — Enforcement (Job B; amendments 6, 7, 9)
- **`tools/check_docs_contract.py`:**
  - stdlib only, no reads at import time, and `main()` under `__main__`;
  - exit 1 on strict failures; warnings print only;
  - it prints the default read path's size;
  - a header comment says that raising a cap or allowing a new file needs your approval;
  - every message names the file, the rule and a one-line fix.

  It checks v1's list:
  - byte caps;
  - the allowed-file list;
  - STATUS format;
  - FOLLOW_UP statuses, allowing anchor stubs;
  - campaign `Status:` lines;
  - relative links and `#anchors`;
  - no 200-character paragraph duplicated across core docs.

  It adds the playbook invariants:
  - LF only;
  - the heading unique and before the first python fence;
  - that block's sha256 unchanged from Phase 0.

  Staged strictness is as in v1.
- **Not in CI** while the campaign is open. Afterwards, with your approval, add a
  separate non-blocking job.
- **`.githooks/pre-commit`:**
  - it runs the checker only when the staged files include a `.md` outside
    `docs/notebook_outputs/raw/**`;
  - it exits 0 if the interpreter is missing.

  **During the campaign, use it only per command:** `git -c core.hooksPath=.githooks commit …`
  in `<wt>`. Enable it repo-wide (`git config core.hooksPath .githooks`) only after the
  campaign, since without `extensions.worktreeConfig` that setting reaches all worktrees,
  frozen checkouts included. Today `.git/hooks/` has no active hooks to copy.
- **`.ignore`:** `docs/archive/`, `docs/notebook_outputs/raw/`, `uv.lock`. `docs/MOVED.md`
  stays visible.
- **`.aiignore`:** see contradiction 11.
- **`GEMINI.md`:** one line → AGENTS.md.
- **Optional, ask first:** a symlink `.claude/skills/gpu-use-skill` → `.agents/skills/gpu-use-skill`.

### 12.6 Phase 6 — Verify (every job, the parts that apply)
- `git -C <wt> diff --name-only` touches only allowed paths, and
  `git diff --name-only -- src scripts tests notebooks docs/plans` is empty.
- `source_tree_sha256` still equals `adc95c42…`, in `<wt>` and in the shared checkout,
  before and after each job.
- `docs/plans/*` hashes are unchanged.
- Every kept `SHA256SUMS` passes `sha256sum -c`.
- Evidence hashes match `before_hashes.txt`.
- The shared checkout has no untracked files.
- pytest and `smoke_checks` match the worktree baseline.
- The playbook invariants hold.
- MAP/MOVED are complete, and RULES has no rule without a status.
- The fresh-agent test (Job B onward): all six answers match `ANSWER_KEY.md`.

---

## 13. Subagents

As in v1:
- read-only inventory in Phase 0, one subagent per file group;
- `rg --no-ignore` checks in Phase 6;
- one subagent for the fresh-agent test.

Do all writing yourself, in sequence.

---

## 14. Final report after each job (amended)

v1's items 1–8 apply, with these additions:
- **Blocked by code** now also covers:
  - the runner's overclaims (amendment 10): `scripts/run_exact_pinmymetal_5fold_cv.py:4`
    ("identical 5-fold CV protocol") and `:56` ("PinMyMetal Fig 2a (5-fold CV on 1,597
    pockets)"; PinMyMetal's CV was not run on DeepMzyme's 1,597-pocket cohort);
  - the stale comment path in `scripts/colab_artifact_streamer.py`;
  - the checker's CI job (after the campaign).
- **Handover note** (≤5 lines, for a *new* Codex thread):
  - where the campaign text lives;
  - STATUS is overwrite-only, and history goes to `log.md`;
  - which sections moved;
  - `docs/plans/` is unchanged;
  - the frozen checkouts keep the old docs, which is expected.

---

## 15. Open decisions for you

1. **Worktree location.** `/media/mechti/Data1/DeepMzyme_worktrees/docs` (NTFS; clean
   check first), or `/` (322 MB, leaving ~2.4 GB)?
2. **Freeze window 1.** Can I treat "Codex idle, no new thread until you fast-forward
   Job A" as confirmed? Should the freeze extend through Job B while the campaign is paused?
3. **Ledger (D2.2).** Record the **six** 2026-09-18 non-overlap test reports (corrected
   from three)?
4. **Zenodo `pmm-zenodo-v2`.** Do you know whether fold 0 finished? If not, the ledger
   says "unknown — possible test evaluation; last mirrored state shows training only".
5. **Rebase and fast-forward.** You run them (default), or I run them when you say so
   each time?

Later, not blocking Job A:
- the target-scheme wording (Job B);
- the `.aiignore` `prepare_training_and_test_set` entry (Job B);
- very-exact PMM sets (Job C);
- the controller link stub (Job C);
- the `remote_homology` `.gitattributes` vs D3 question (Job D).

---

## Appendix A — Amendment trace (all 33)

| # | Amendment | Where in v2 | Status |
|---|---|---|---|
| 1 | Wrong base branch | 3.1 | Adopted |
| 2 | Precondition failed | 1, 3.5 | Adopted; holds at 13:50 |
| 3 | Codex edit rate | 3.4, 3.5, 11 | Adopted; rate is zero while the campaign is paused |
| 4 | Merge-back path | 3.6 | Adopted |
| 5 | Disk | 1, 3.2 | Adopted; re-measured |
| 6 | Test would turn CI red | 0.3, 12.5 | Adopted (`tools/`) |
| 7 | `core.hooksPath` is repo-wide | 12.5 | Adopted |
| 8 | Shared checkout untouched | 3.3 | Adopted |
| 9 | `.ignore` hides MOVED.md | 7, 12.5 | Adopted (`docs/MOVED.md`) |
| 10 | Frozen source hash; runner overclaims | 0.1, 4.2, 14 | Adopted; hash re-verified |
| 11 | `docs/plans` in place; STATUS at root; TECH anchors | 0.2, 4.2, 8, 10 | Adopted; also no edits to `docs/plans/*` |
| 12 | More playbook consumers + invariants | 4.2, 12.5 | Adopted |
| 13 | `Plan.md` root marker | 4.2, 7 | Adopted |
| 14 | Snapshot builder | 4.2 | Adopted |
| 15 | smoke_checks strings/files | 4.2 | Adopted |
| 16 | New consumers; re-run inventory | 4.2, 12.0 | Adopted; 13:50 re-run added 4 read-only couplings |
| 17 | Evidence-dir details | 4.3, 12.4 | Adopted (mostly Job D) |
| 18 | Exact test opened in two rounds | 6 D2.1, B | Adopted; paths corrected (repo-root `runs/`) |
| 19 | Undocumented 09-18 non-overlap runs | 6 D2.2 | **Corrected: 6 reports, not 3**; Q3 |
| 20 | Zenodo relaunch unknown | 6 D2.3 | Partly answered; Q4 |
| 21 | Wording fixes | 6 D2.4–5 | Adopted in the sentence |
| 22 | More EXACT-doc errors | 6 D2.7, B | Adopted |
| 23 | Selection influence; no private path | 6 D2.6, 4.5 | Adopted |
| 24 | STATUS history partly lost | 4.7, 8 | Adopted (Job B) |
| 25 | Campaigns missing from the map | 8 | Adopted |
| 26 | PMM ion campaign as one folder | 8 | Adopted; core v2 now paused |
| 27 | Fold regimes | 9.3, B | Adopted; Stage 6 row added |
| 28 | Smaller corrections | 8, 9 | Adopted; the 09-28-flag point is moot |
| 29 | Secret scan | 12.0 | Adopted |
| 30 | Baseline pytest in worktree | 12.0 | Adopted, plus the resource cap |
| 31 | No blanket worktree prune | 4.11 | Adopted |
| 32 | Codex handover | 3.9, 14 | Adopted |
| 33 | Job D after the campaign | 0.4, 11, 12.4 | Adopted (user invariant) |

---

## Appendix B — Job A work order (prepared; not started)

Line numbers are for `3e912a6`. Re-locate each edit by its text before editing.

**B0. Pre-checks.** Bash was blocked by transient classifier errors at 14:00, so these
still need running:
1. Inbound links to `Plan.md#canonical-colab-metal-training-pipeline` and to the
   playbook's "G4-Class Optuna Policy" anchor, across all files, evidence included. If
   any exist, keep the old anchor with an explicit `<a id="…"></a>` when renaming.
2. Remaining parity and hype claims in active docs:
   `rg --no-ignore -n -i "identical (dataset|5-fold|cohort|protocol)|full replication|massive|definitive|breakthrough|beating" AGENTS.md Plan.md README.md EXPERIMENT_STATUS.md docs/*.md docs/notebook_outputs/README.md`.
3. Other "Colab recommended" phrases: README, `COLAB_GPU_RUNBOOK.md`, the notebook guide.
4. Whether `docs/GCP_GPU_RUNBOOK.md` says where Optuna storage lives on the VM route
   (needed for the AGENTS:469/552 rewrite).
5. The PMM ion campaign's fold grouping key and source, for the fold table.
6. Whether the exact test's structures are byte-identical to the non-overlap test side.
7. Whether gitleaks is installed.

**B1–B3.** Phase 0 (12.0) → `ANSWER_KEY.md` → **stop for approval.**

**B4. D1 (compute route):**

| Target | Now | Change |
|---|---|---|
| `AGENTS.md:231-234` | "For G4-class GPU planning, this is where serious/custom Optuna budgets…" | Say its G4-class budgets are historical |
| `AGENTS.md:465-470` | "Hardware: G4-class GPU… Persistent Optuna storage in Drive is mandatory for Stage 4 and Stage 5." | The D1 route in one line. The Drive rule applies to the Colab route; the VM route per B0.4 |
| `AGENTS.md:551-552` | "Confirm persistent Drive SQLite storage for serious Optuna stages." | Make it route-specific, as above |
| `Plan.md:252-256` | heading "Canonical Colab metal-training pipeline" + "The canonical metal-training workflow is `notebooks/…`" | The heading names the staged pipeline; the old anchor is kept if B0.1 finds links. The text: the playbook's staged blocks are canonical; the primary route runs them as CLI runners on the GCP L4 VM; the notebook is secondary and Colab the fallback |
| `Plan.md:268-271` | "Prefer a verified G4-class GPU…on each Colab allocation" | Measure the accelerator on each allocation (the VM on the primary route, Colab on the fallback) |
| `docs/GETTING_STARTED.md:26` | "Recommended cloud entry point" | "Authorized fallback; secondary interface". Add a GCP-VM row pointing to `gpu-use-skill` |
| `playbook:2689`, `:3834` | G4-class budget text | Prefix "Historical (G4-class):". Both are below the python fence at L1514 |

**B5. D2 (test ledger and wording):**

| Target | Change |
|---|---|
| `DATASETS.md:216-229` (Exact PMM status) | "Completed test evaluation found: no" → yes, with the dates and a ledger link. "Selection use established: no" → yes, exploratory. Rewrite `:225-227` to match |
| `DATASETS.md:291-305` (non-overlap) | "exactly seven" → seven early + six on 2026-09-18 (**if Q3 = yes**). Blockquote `:300` to match |
| `DATASETS.md:513-518` (ledger) | Update the Exact PMM and Non-overlap rows. Add an Exact Zenodo PMM row (Q4). Add a subsection, "Local test-access artifacts (git-ignored)", listing each report path and sha256: 3 + 15 + 3 exact and 6 non-overlap, with duplicates noted |
| `EXACT…:15-33`, Table 1 section | Remove the parity claims (`:25`) and hype (`:15`, `:30` "Massive Breakthrough", "beating"). Fix `:23` (1,488 rows vs 352 pockets) and `:29` ("raw" → collapsed-4 accuracy). Add the D2 sentence once, and the max-over-epochs note with the selected-checkpoint OOF values. Label the test results "test set opened 2026-09-22/23; see ledger". Keep every number |
| STATUS "Current objective" (~`:402-411`) | "Full replication of the … protocol" → a one-line caveat plus a link to the sentence. The Job B rewrite replaces this block later |

**B6. Fold table and index rows:**

| Target | Change |
|---|---|
| `Plan.md`, after the "Metal example terminology" section (starts `:199`) | The fold-regime table (9.3) |
| `docs/notebook_outputs/README.md` | Two rows: exact-PMM 5-fold 2026-09-22/23 (held-out opened; not like-for-like; max-over-epochs caveat; links to both summaries), and exact-pocket L4 v2 (validation-only, stopped at 3/20; link to its execution doc) |
| `docs/PARAMETER_FINDINGS.md` | Validation-only entries for both, with FINDINGS' own grading. No test metrics |

**B7. Records:** `RULES.md` lists every normative statement Job A modified (old → new,
with its line). `PROGRESS.md` is updated. No `MOVED.md` entries: Job A moves nothing.

**B8. Verify** (12.6), **stage** with `git -C <wt> add -- <explicit paths>`, then **stop**.
- Expected diff: about 9 files and ~150 lines, plus new files under
  `docs/archive/consolidation_2026-09/`.
- Suggested commit message: `git commit -m "Correct test ledger, compute route and fold terminology"`.
- Then you run: rebase if needed → `git merge --ff-only docs-consolidation` in the shared
  checkout → `git push` → a new Codex thread with the handover note.

---

## Appendix C — ANSWER_KEY draft (finalize in Phase 0)

1. **Which schemes are trained, and what is the endpoint?**
   - Active PMM core v2: `four_class`, `five_class` and `six_class` (`run_pmm_core_campaign.py:22`;
     `docs/plans/pmm_core_scope_v2.json`).
   - The endpoint is common-four: Mn, Cu, Zn, and Class VIII = Fe+Co+Ni.
   - Direct `four_class` is the primary formulation (AGENTS 1c).
   - The five-class screen trained `five_class` only.
2. **What does "5-fold" mean in the active campaign?**
   - Five frozen PDB-grouped folds over ion examples (evidence-index row
     `metal/pmm-ion-v2-context/2026-09-26`; `EXPECTED["folds"]`).
   - Not the `pocket_id`-stratified folds of the 09-22/23 benchmark.
   - Fold 0 is trained; folds 1–4 are deferred.
   - Phase 0 verifies the grouping key.
3. **Where do runs execute, and who owns the GPU lifecycle?**
   - CLI runners on the GCP L4 VM via `~/deepmzyme-vm/bin/*` under `gpu-use-skill` (D1).
     Colab is the fallback.
   - One coordinator owns allocation, submission and shutdown. A monitor is read-only.
     There is one active GPU (`AGENTS.md:38-51`).
4. **What is the next action?**
   - None is authorized. The core campaign is paused: 9/45 fold-0 fits are kept and the
     36 fold-1–4 fits are deferred.
   - Resume only on your explicit request, with refreshed budget authority and new
     readiness verification (STATUS:14-21).
5. **Which held-out test sets have been opened?**
   - Non-overlap PMM: seven early reports, plus six on 2026-09-18 (Q3).
   - Exact PMM: 2026-09-22 and 2026-09-23. These are the same structures as the
     non-overlap test.
   - Zenodo PMM: unknown (Q4).
   - CARE clusterRes30: an incidental metadata exposure on 2026-09-15, with no evaluation.
   - Others: no evaluation found (the DATASETS ledger).
6. **What is forbidden before Stage 7?**
   - Any held-out test evaluation.
   - Using test metrics for HPO, ranking, promotion or rejection.
   - Stage 7 requires Stage 6 grouped-fold confirmation and a completed Stage 6B frozen
     final full-train refit (`AGENTS.md:405-413`; Plan).
