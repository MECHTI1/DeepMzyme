# Job B fresh-agent test — run 1 (2026-09-28, cloud)

Result: **not accepted.** Q2, Q5 and Q6 match [ANSWER_KEY](../ANSWER_KEY.md). Q4
matches only when facts stated elsewhere in the same response are counted. Q1
misses part of element 4. Q3 element 3 is a borderline miss. PLAN_v2 §12.6
requires all six answers to match, so no pass is claimed.

## Method and limits

- Agent: one cold-start subagent (Claude Code `Explore` type) in the cloud
  container at branch `claude/happy-ritchie-99xmhd`, working tree after the
  review fixes (tip `042e93d` plus uncommitted checker/hook/ignore/skill edits;
  the three read files differ from `042e93d` only in AGENTS.md's checker/hook
  lines).
- It was told to use only Read, open only `AGENTS.md`, `EXPERIMENT_STATUS.md`
  and the campaign README named on STATUS's `Current campaign:` line, and to
  answer the six ANSWER_KEY questions. The questions were given without the key.
- Read-only was by instruction. The agent type also has Bash, Grep and Glob;
  the harness does not log its tool calls to this session, so its report is the
  only evidence of the three files it read and of Read-only use.
- The agent's claims below were checked against the three files; each is
  grounded there (for example `merge_fe_class_viii` at AGENTS.md:16, the
  comparator's Grade 2 at the PMM README:23-24 and the disk cost at
  STATUS:61).
- One run. It is not a workstation or Codex run; model and tooling differ from
  the local environment.

## Grading against ANSWER_KEY

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

## Possible remedy (not applied; user decision)

Q1: extend the PMM README endpoint sentence to “five- or six-class training
followed by collapse differs from direct four-class training; the comparison is
a neutral test (better, no difference and worse are all valid)”, restating
approved AGENTS.md:17/:19 text; the README is already 63 lines against the
60-line checker warning. Q3 needs no document change: AGENTS.md:41 states the
rule. A rerun needs a new cold-start agent, and every run is to be reported.

## Agent answers (verbatim)

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
