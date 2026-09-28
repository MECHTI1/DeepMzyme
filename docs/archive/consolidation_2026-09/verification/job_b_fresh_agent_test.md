# Job B fresh-agent test — runs 1 and 2 (2026-09-28, cloud)

| Run | Documents read | Result |
|---|---|---|
| 1 | tip `042e93d` plus the uncommitted checker/hook/ignore/skill edits | **Not accepted.** Q2, Q5 and Q6 match; Q4 matches counting the whole response; Q1 misses part of element 4; Q3 element 3 is a borderline miss. |
| 2 | `a205658` plus the user-approved Q1 fix to the PMM README | All six match under the same grading (Q4 again counting the whole response), with the wording notes in its table. |

PLAN_v2 §12.6 requires all six answers to match [ANSWER_KEY](../ANSWER_KEY.md).
Run 2 meets that under this grading. It is a single run after a document fix
aimed at run 1's miss, not an independent repeat; acceptance is the user's call.

## Setup and limits (both runs)

- Each run used a new cold-start subagent (Claude Code `Explore` type) in the
  cloud container on branch `claude/happy-ritchie-99xmhd`, with the same prompt.
- It was told to use only Read, open only `AGENTS.md`, `EXPERIMENT_STATUS.md`
  and the campaign README named on STATUS's `Current campaign:` line, and to
  answer the six ANSWER_KEY questions. The questions were given without the key.
- Read-only was by instruction. The agent type also has Bash, Grep and Glob,
  and its tool calls are not logged to this session. For run 2 the harness
  reported 4 tool uses, consistent with the three Reads and the hand-back the
  agent reported; no count was shown for run 1.
- Each agent's specific claims were checked against the three files and are
  grounded there.
- Not a workstation or Codex run; model and tooling differ from the local
  environment.

## Run 1

Documents: working tree after the review fixes (tip `042e93d` plus uncommitted
checker/hook/ignore/skill edits; the three read files differ from `042e93d`
only in AGENTS.md's checker/hook lines). Claims checked include
`merge_fe_class_viii` at AGENTS.md:16, the comparator's Grade 2 at the PMM
README:23-24 and the disk cost at STATUS:61.

### Grading, run 1

| Q | Result | Notes |
|---|---|---|
| Q1 | **Miss** | Elements 1–3 present. Element 4: states that no objective is primary, that every arm is scored on collapsed four, and the tie rule; says six-then-collapse differs from direct four but does not mention five, and omits that comparing objectives is a neutral test in which better, no difference and worse are all valid. AGENTS.md:17 and :19 contain both points; the PMM README:9-12 names only six-then-collapse and does not state the neutral test. |
| Q2 | Match | All four elements; the nine-fit/36-deferred state appears in the Q1 answer. |
| Q3 | **Borderline miss** | Elements 1, 2 and 4 present; element 3 has one coordinator, a read-only monitor and one active GPU, but not “read `gpu-use-skill` before any GPU action” (AGENTS.md:41). |
| Q4 | Match (whole response) | Elements 1 and 3 in the Q4 answer; pause date in the preamble, nine kept/36 deferred in Q1, TERMINATED VM and disk cost in Q3. The word “exploratory” for the nine fits is not used (Grade 5 is). |
| Q5 | Match | All four elements. |
| Q6 | Match | All three elements. |

No contradicting claim was found. Direct four-class is not called primary,
required or main.

### Remedy after run 1

Q1: extend the PMM README endpoint sentence to “five- or six-class training
followed by collapse differs from direct four-class training; the comparison is
a neutral test (better, no difference and worse are all valid)”, restating
approved AGENTS.md:17/:19 text; the README is already 63 lines against the
60-line checker warning. Q3 needs no document change: AGENTS.md:41 states the
rule. A rerun needs a new cold-start agent, and every run is to be reported.

The user approved the wording on 2026-09-28 (“five- or six-class training
followed by collapse differs from direct four-class training; comparing
objectives is a neutral test where better, no difference and worse are all
valid”). It was applied to PMM README lines 10–12 before run 2; the README
stays at 63 lines, so ANSWER_KEY line citations are unchanged. Q3 needed no
document change.

### Agent answers, run 1 (verbatim)

The active campaign is **pmm_ion_metal**, PMM core scope v2 (paused). Its README scope ID is `pmm-core-v2`, campaign `pmm_ion_metal_v2_context`. STATUS says it has been paused since the user's decision on 2026-09-28.

**Q1. Target schemes and evaluation endpoint**
- **Training schemes:** `four_class`, `five_class` and `six_class`. Each is trained separately with three architectures: Only-ESMC, Only-GVP and graph-level late fusion.
- **Current state:** nine ordinary-readout fits on fold 0 are kept, and 36 more fits are deferred.
- **No primary objective:** none of the three training schemes is primary. Training six-class and then collapsing it is a different method from training four-class directly, so each keeps its own run/study identity. The direct four-class scheme's canonical name is `merge_fe_class_viii`.
- **Endpoint:** every arm is scored on the collapsed four classes: Mn, Cu, Zn, and Class VIII = Fe+Co+Ni.
  - The five- and six-class arms also keep their native metrics and Fe/Co/Ni recalls (for five-class: Fe, and Co+Ni).
  - Ranking across schemes uses collapsed-four balanced accuracy only. It needs a campaign assessor, because the notebook's Stage 6/6B route handles only one scheme.
- **Tie rule:** set in advance. Without a clear improvement, keep direct four-class.

**Q2. What "5-fold" means here**
- It means five folds grouped by PDB ID, frozen once in `fold_membership.csv` with `split_seed=42` and `split_stratify_by=metal_site`. The model seed is 42 only.
- The data is a frozen, training-only PMM source cohort: 7,398 ions in 3,992 PDB groups, with `metal_example_unit=ion`.
- The README says this is a one-seed grouped-fold design. It is **not** seed-repeat confirmation, and it is **not** the pocket-stratified benchmark from 2026-09-22/23.
- Only fold 0 has been run for the neural core. The individual qualified fits are evidence Grade 5; the incomplete five-fold grid is Grade 6. No completed five-fold neural confirmation is claimed.
- The released PMM comparator did complete all five folds (Grade 2).

**Q3. Where runs execute and who owns the GPU lifecycle**
- **Main route:** CLI runners on a GCP L4 VM, run only through `~/deepmzyme-vm/bin/*` under the gpu-use-skill. Colab is the authorized fallback, and its notebook is the secondary interface.
- **Ownership:** one coordinator owns allocation, submission and shutdown. An optional monitor is read-only and never replaces the stop controls.
- **Limits:** at most one active GCP GPU, and one training worker per GPU in production. Invoking the skill gives no authority to spend money or change providers.
- **VM state:** TERMINATED as of 2026-09-28 06:00:18 UTC. That is historical evidence, not a live check. The kept 150-GB disk costs about $15/month.
- **Not stated:** the files do not name which person or agent is the coordinator.

**Q4. Next action**
- Nothing is authorized now: no experiment, GPU work, final refit or held-out evaluation.
- **To resume:** the user must explicitly ask, budget authority must be refreshed, and readiness must be checked again against the execution environment, under the v2 scope and the playbook's guarded recipe ("active core-only continuation").
- **On any resume:**
  - Reuse all completed fits and run only the missing selectors.
  - Do not rerun completed units.
  - Do not retry or retrain GVP5/GVP6, or repeat their replay, hoping for a chance strict pass.
- **Budget:** the proposed extra 34 hours / $34 is not approved and no decision on it is pending. No extra fold-0 fits or final refits are authorized.
- **Before Stage 7:** Stage 6, then Stage 6B, then frozen report/checkpoint rules, then a resolved final-test route.

**Q5. Held-out test sets that have been opened** (STATUS says docs/DATASETS.md owns the full ledger; I did not read it)
- **Non-overlap PMM test (352 pockets):** opened. Seven early reports, plus six on 2026-09-18 (three `benchmark_50epochs` and three `benchmark_replicated_72pct`).
- **Exact PMM test (same structure set: 352 pockets / 316 structure files / 313 PDB IDs):** opened.
  - 2026-09-22: three single-split reports.
  - 2026-09-23: 15 fold reports and three ensemble reports.
  - These reports did influence exploratory selection; no model was promoted.
- **Exact Zenodo PMM test:** unknown, possibly evaluated. The relaunch outcome was never recorded.
- **No evaluation artifacts found:** Harsh, Common-PDBID 70/30, CLEAN30 and CARE clusterRes30. CARE's test metadata was seen by accident on 2026-09-15, but it was not evaluated. Not finding artifacts does not prove no outside run happened.
- **Active campaign core:** no held-out evaluation took place.

**Q6. What is forbidden before Stage 7**
- Any held-out (test) evaluation.
- Using test metrics for tuning, ranking, promotion, rejection or checkpoint choice. Selection uses validation only, and the neutral target test also forbids held-out use.
- Opening the test set before all of these are in place:
  - Stage 6 grouped-fold selection, or an explicitly labeled fallback.
  - A completed or reused Stage 6B full non-test refit, frozen as the one-shot Stage 7 source.
  - A frozen primary report, source checkpoint/refit, ensemble list and averaging, and calibration rule.
  - A scientifically resolved main final-test route.
- Using the test set without first reading docs/DATASETS.md.
- After opening, test scores cannot pick a different report or configuration. A test set that was opened before is never pristine.

Files read, as reported by the agent: `AGENTS.md`, `EXPERIMENT_STATUS.md`,
`docs/campaigns/pmm_ion_metal/README.md`; it reported using only Read.

## Run 2

Documents: `a205658` with the approved PMM README fix (lines 10–12, then
uncommitted); `AGENTS.md` and `EXPERIMENT_STATUS.md` as in `a205658`. Claims
checked include the Core v1 four/six scope at README:37, one training worker per
GPU at README:44 and “never pristine” at AGENTS.md:33.

### Grading, run 2

| Q | Result | Notes |
|---|---|---|
| Q1 | Match | Elements 1–4: `pmm-core-v2`; four/five/six trained separately; the collapsed-four endpoint with native metrics kept; no objective primary, every arm evaluated on collapsed four, five- or six-then-collapse differs from direct four, and “the comparison is a neutral test”, with the tie rule (consistent per the grading note). The gloss “better, no difference and worse are all valid” is not restated; the term is used as defined in AGENTS.md:19 and PMM README:10-12. |
| Q2 | Match | All four elements; nine kept/36 deferred appears in Q1. Folds 1–4 for the 36 fits is the agent's stated arithmetic; the three files say only that 36 fits are deferred. |
| Q3 | Match | All four elements, including reading `gpu-use-skill` before any GPU action (AGENTS.md:41). |
| Q4 | Match (whole response) | Elements 1 and 3 in the Q4 answer; the pause in the preamble and Q4; nine kept/36 deferred in Q1; TERMINATED VM and disk cost in Q3. “Exploratory” is not used for the nine fits (Grade 5 is). Same reading as run 1. |
| Q5 | Match | All four elements. |
| Q6 | Match | All three elements; “selection is validation-only” is expressed as the ban on test metrics for tuning, ranking, promotion, rejection or checkpoint choice. |

No contradicting claim was found. Direct four-class is not called primary,
required or main. Observation for Job C: the agent noted that the three files do
not say the pocket-stratified 2026-09-22/23 benchmark is the one whose exact-PMM
test reports STATUS lists for those dates.

### Agent answers, run 2 (verbatim)

The active campaign is `pmm_ion_metal` and it is paused, so nothing is authorized right now. STATUS's `Current campaign:` line (S:8) reads "pmm_ion_metal, PMM core scope v2 (paused)" and links `docs/campaigns/pmm_ion_metal/README.md`. Against the repo root that is `/home/user/DeepMzyme/docs/campaigns/pmm_ion_metal/README.md`. The README calls the scope `pmm-core-v2` and the campaign `pmm_ion_metal_v2_context` (R:5). Both files say it was paused by a 2026-09-28 user decision (S:3, R:3).

Line references: A = AGENTS.md, S = EXPERIMENT_STATUS.md, R = the campaign README.

**Q1. Target schemes and evaluation endpoint**
- Three objectives are each trained separately: `four_class`, `five_class` and `six_class` (R:6-8). None is primary (R:10, A:17). Direct four-class has the canonical name `merge_fe_class_viii` (A:16).
- Each objective is run with three model families: Only-ESMC, Only-GVP and graph-level late fusion. That makes nine "ordinary-readout" fits, all on fold 0; 36 more are deferred (R:6-8, S:12).
- **Endpoint:** the common (collapsed) four classes: Mn, Cu, Zn and Class VIII (Fe+Co+Ni). Every arm is evaluated on it (R:9-10, A:16).
  - Five- and six-class arms also keep their native metrics (R:10).
  - AGENTS adds Fe/Co/Ni recalls; for five-class these are Fe and Co+Ni (A:17).
- **Comparing targets:**
  - Rank only by collapsed-four balanced accuracy, never by native balanced accuracy (A:19, S:36-39).
  - Training on five or six classes and then collapsing is a different identity from direct four-class (A:17, R:10-11).
  - The comparison is a neutral test. Its tie rule is "without clear improvement, keep direct four-class" (R:11-12, R:61-62, A:20).
- **Context:** v2 replaced Core v1, which trained only four and six classes (R:37). The closed five-class screen v1 and the paused binding-aware arms are separate sub-batches (R:35, R:39). The files do not say which targets the binding-aware arms used.

**Q2. What "5-fold" means here**
- The cohort is the frozen, training-only PMM source cohort: 7,398 ions in 3,992 PDB groups, with `metal_example_unit=ion` (R:14-15).
- It is split into five folds grouped by PDB. They were frozen once in `fold_membership.csv` with `split_seed=42` and `split_stratify_by=metal_site`, and only model seed 42 is used (R:15-17).
  - The files do not give the CSV's path.
  - The fold contract is in `docs/plans/metal_level_metal_task_compared_PMM_final_plan.md`, which I did not open.
- The README calls this a "one-seed grouped-fold design". It says this is not seed-repeat confirmation and not "the pocket-stratified 2026-09-22/23 benchmark" (R:17-19).
  - STATUS records Exact PMM test openings on those same dates (S:48-51), but the files never say they are the same thing.
- **Progress:** the neural core has run fold 0 only. Individual fits are Grade 5; the incomplete five-fold grid is Grade 6 (R:22-23). No completed five-fold neural confirmation is claimed (S:13).
- The released PMM comparator did complete all five folds (Grade 2) (R:23-24).
- 9 x 5 = 45, so 36 deferred fits would fit folds 1-4. That is my arithmetic; the files do not list the deferred fits.

**Q3. Where runs execute and who owns the GPU lifecycle**
- **Primary route:** CLI runners on the GCP L4 VM, operated only through `~/deepmzyme-vm/bin/*` under gpu-use-skill (`.agents/skills/gpu-use-skill/SKILL.md`) (A:42, R:41-42).
- **Fallback:** Colab is the authorized fallback, and the notebook is a secondary interface. Historical G4 budgets stay labeled historical (A:42, R:42-43).
- **Limits:** at most one active GCP GPU (A:43), and one training worker per GPU in production (R:44). The skill must be read before any GPU planning, connection, monitoring or training (A:41).
- **Ownership:** "One coordinator owns allocation/submission/shutdown". An optional monitor is read-only and never replaces stop controls (A:43). This is a role only; the files do not say which person or agent is the coordinator.
- **Authority:** permission to resume or spend comes from an explicit user request plus refreshed budget authority (S:63-64, R:51-53). Invoking the skill grants no spending, migration, provider or larger-budget authority (A:43).
- **Last recorded state:** the VM was TERMINATED at the 2026-09-28 06:00:18 UTC check. A retained 150-GB disk costs about $15/month (S:61, R:46-47). STATUS says this is historical, not a live check.

**Q4. Next action**
- Nothing is authorized: no experiment, GPU work, final refit or held-out evaluation (S:60, R:51). The pause holds until the user explicitly resumes within verified authorization (A:47).
- **Resuming requires** (S:63-64, R:51-54):
  - an explicit user request
  - refreshed remaining-budget authority
  - new readiness checks against the execution environment, under the v2 scope and the guarded recipe (playbook section `#active-core-only-continuation`, which I did not open)
- The proposed extra 34 hours / $34 is unapproved and no longer waiting on an immediate decision. No extra fold-0 fits or final refits are authorized (S:64-67).
- **Rules on any authorized resume** (R:55-63):
  - Reuse all completed fits and "run only missing selectors". The files do not define "selectors". Do not rerun completed units.
  - An old scope or implementation hash cannot authorize a run.
  - Do not retry or retrain GVP5 or GVP6, or repeat their replay.
  - Old forecasts and historical next actions in the log give no authorization.

**Q5. Held-out test sets opened**
This comes from STATUS's short reminder (S:43-56). The authoritative ledger is `docs/DATASETS.md#test-use-ledger`, which I did not read.
- **Non-overlap PMM test (352 pockets):** opened. Seven early reports, plus six on 2026-09-18 (`benchmark_50epochs` x3 and `benchmark_replicated_72pct` x3).
- **Exact PMM test:** opened. It is the same structure set: 352 pockets / 316 structure files / 313 PDB IDs.
  - 2026-09-22: three single-split reports.
  - 2026-09-23: 15 fold reports and three ensemble reports.
  - Test results did influence exploratory selection; no model was promoted.
- **Exact Zenodo PMM test:** unknown, possibly evaluated; the relaunch outcome is unrecorded (see also S:9).
- **Harsh, Common-PDBID 70/30, CLEAN30, CARE clusterRes30:** no evaluation artifacts found.
  - CARE had test metadata exposed by accident on 2026-09-15; that was not an evaluation.
  - STATUS warns that finding no artifacts does not prove no run happened elsewhere.
- The active `pmm_ion_metal` core opened no test (R:30).
- The primary final-test route is still unresolved (S:43), and a test set opened in the past is never pristine (A:33).

**Q6. What is forbidden before Stage 7**
- Any held-out evaluation (A:32). Right now none is authorized at all (S:60).
- Using test metrics for tuning, ranking, promotion, rejection or checkpoint choice. This is banned at every stage, not only before Stage 7 (A:32, S:71-72). The neutral target comparison also allows no held-out use (A:19).
- Opening the test before these are frozen: the primary report, the source checkpoint or refit, the ensemble list and averaging, and the calibration rule. After opening, test scores cannot choose a different report or configuration (A:34).
- Starting the one-shot Stage 7 before all of the following are done (A:33, S:69-71):
  - Stage 6 grouped-fold selection, or an explicitly labeled fallback
  - a completed or reused Stage 6B full non-test refit
  - frozen report and checkpoint rules
  - a scientifically resolved final-test route
- `docs/DATASETS.md` must also be read before any test-set use (A:32).

**Files read** (each in full, one Read call each):
1. `/home/user/DeepMzyme/AGENTS.md` (92 lines)
2. `/home/user/DeepMzyme/EXPERIMENT_STATUS.md` (85 lines)
3. `/home/user/DeepMzyme/docs/campaigns/pmm_ion_metal/README.md` (64 lines)

**Tool confirmation:** I used only the Read tool, three calls in total. I ran no Bash, Grep or Glob, opened no linked file, used no git history and changed nothing. The only other call is the required SubagentHandback that delivers this report.
