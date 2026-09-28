# ANSWER_KEY — fresh-agent test (docs consolidation 2026-09)

Status: **approved in Phase 0**, as recorded in [PROGRESS.md](PROGRESS.md)
(2026-09-28). Scientific facts are as of `3e912a6`; Job A corrects their
documentation. The six required answers are unchanged.

After Job B, a fresh agent that reads only `AGENTS.md`, `EXPERIMENT_STATUS.md` and the
active campaign README must give every **required element** below. Wording may differ.
An answer that misses an element, or adds a contradicting claim, fails. If the
underlying facts change before the test (for example Codex resumes the campaign),
update this key first, with sources, and get approval again.

---

## Q1. Which metal target schemes are trained in the active campaign, and what is the evaluation endpoint?

**Required elements**
1. Active campaign: the PMM ion-level metal comparison, core scope v2 (`pmm-core-v2`).
2. Trained schemes: `four_class`, `five_class` and `six_class`, as separate identities.
3. Endpoint: the common-four view (Mn, Cu, Zn, Class VIII = Fe+Co+Ni), on which all three
   are compared. The five- and six-class arms also keep their native metrics.
4. Direct `four_class` training remains the primary formulation. It is not equivalent to
   training on six classes and collapsing to four.

**Sources:** `run_pmm_core_campaign.py:18-26` (`targets`); `docs/plans/pmm_core_scope_v2.json`
(`targets`); `docs/notebook_outputs/README.md:54` ("common-four endpoint"); `AGENTS.md:125-137`.

## Q2. What does "5-fold" mean in the active campaign?

**Required elements**
1. Five **PDB-grouped** folds over **ion** examples (`metal_example_unit=ion`), frozen once
   in a single `fold_membership.csv` (`split_seed=42`, `split_stratify_by=metal_site`).
2. One model seed (42): a one-seed grouped-fold comparison, not seed-repeat confirmation.
3. State: fold 0 is trained for all nine core arms; the 36 fold-1–4 fits are deferred.
4. It is **not** the `pocket_id`-stratified 5-fold of the 2026-09-22/23 exact-PinMyMetal
   benchmark, where one PDB can appear in several folds.

**Sources:** `docs/plans/metal_level_metal_task_compared_PMM_final_plan.md:239-242, :259`;
`docs/plans/pmm_core_scope_v2.json` (`folds`, `model_seeds`); `EXPERIMENT_STATUS.md:10-16`;
`docs/EXACT_PINMYMETAL_5FOLD_CV_REPRODUCIBILITY.md:9`.

## Q3. Where do runs execute, and who owns the GPU lifecycle?

**Required elements**
1. Primary route: the CLI runners on the GCP L4 VM, operated only through
   `~/deepmzyme-vm/bin/*` under `gpu-use-skill` (user decision D1, 2026-09-28).
2. Colab is the authorized fallback; the Colab notebook is a secondary interface.
3. Read `gpu-use-skill` before any GPU action. One coordinator owns allocation, submission
   and shutdown; an optional monitor is read-only. At most one **active** GPU.
4. A skill invocation does not authorize spending, another provider or a larger budget.

**Sources:** D1 (PLAN_v2 §6); `AGENTS.md:29-51`.
Baseline note: before Job A, `AGENTS.md:467` ("Hardware: G4-class GPU") and
`docs/GETTING_STARTED.md:26` ("Recommended cloud entry point") contradicted D1.
Job A fixes these statements in the isolated worktree.

## Q4. What is the next action?

**Required elements**
1. No experiment or GPU work is authorized now.
2. The PMM core campaign is paused (2026-09-28 user decision). The nine fold-0 results are
   kept as an exploratory comparison, and the 36 fold-1–4 fits are deferred.
3. Resume only on an explicit user request, with refreshed budget authority and new
   readiness verification.
4. The last verified VM state is TERMINATED; its disk storage still costs money.

**Sources:** `EXPERIMENT_STATUS.md:10-29`.

## Q5. Which held-out test sets have been opened?

**Required elements**
1. **Non-overlapped PinMyMetal test** (352 pockets): seven early reports, plus **six**
   reports on 2026-09-18 (`benchmark_50epochs` ×3 and `benchmark_replicated_72pct` ×3).
2. **Exact PinMyMetal test** (352 pockets; 316 structure files; 313 PDB IDs): opened on
   2026-09-22 (3 single-split reports) and 2026-09-23 (15 fold and 3 ensemble reports). It
   is the same structure set as the non-overlap test. Selection influence: yes,
   exploratory. No model was promoted.
3. **Exact Zenodo PinMyMetal test:** unknown, possibly evaluated. The outcome of the
   relaunch `pmm-zenodo-v2` is unrecorded.
4. No evaluation artifacts were found for Harsh, Common-PDBID 70/30, CLEAN30 or CARE
   clusterRes30. CARE had an incidental test-metadata exposure on 2026-09-15, which was
   not an evaluation.

**Sources:** `docs/DATASETS.md:280-305, :511-533` (updated by Job A);
`docs/archive/consolidation_2026-09/inventory/test_access_artifacts.tsv`;
`inventory/job_a_prechecks.md` (B0.6); `docs/agents_report/RESUMED_REVIEWS_HANDOFF_20260923.md:166-169`;
`~/zenodo_pmm_artifacts/benchmark_enhanced_only_gvp_fold0/prepare_status.json` (last mirrored
state: fold 0 training, `run_test_eval: false`); `scripts/run_zenodo_pmm_exact_5fold_cv.py:794`
(the runner tests after each finished fold by default).

## Q6. What is forbidden before Stage 7?

**Required elements**
1. Any held-out test evaluation.
2. Using test metrics for HPO, ranking, promotion, rejection or checkpoint choice.
   Selection is validation-only.
3. Stage 7 needs Stage 6 grouped-fold confirmation (or an explicitly labeled fallback) that
   selects one configuration, plus a completed Stage 6B final full-train refit of it, frozen
   as the Stage 7 source. Stage 7 is one-shot.

**Sources:** `AGENTS.md:405-413, :474-486`; `Plan.md:272-275`.
