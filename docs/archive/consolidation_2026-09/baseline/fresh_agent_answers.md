# Fresh-agent baseline answers (default reading path only)

Worktree: `/media/mechti/Data1/DeepMzyme_worktrees/docs` (W). Date: 2026-09-28.

## Files read (in full) and byte counts

| File | Bytes |
|---|---:|
| `AGENTS.md` | 43,624 |
| `Plan.md` | 83,267 |
| `EXPERIMENT_STATUS.md` | 42,062 |
| `docs/README.md` | 8,320 |
| `docs/DATASETS.md` | 46,801 |
| `docs/notebook_outputs/README.md` | 30,371 |
| **Total** | **254,445** |

Nothing else was opened. `AGENTS.md:371-373` defines the "Read by default" set.

Reading-path observations:
- `EXPERIMENT_STATUS.md` ends mid-sentence at line 547 ("the v12 EC1 standalone campaign"). The file may be truncated.
- `Plan.md:113-117` splits `val_metal_balanced_acc` into "val_metal_ba … lanced_acc" with blank lines between the two parts.
- The "Read by default" list sits at `AGENTS.md:371`, deep in the file.

## 1. Metal target schemes in the active campaign; evaluation endpoint

**Answer:** The active campaign is the PMM ion-level core v2 scope. It trains ordinary-readout Only-ESMC, Only-GVP and graph-level late fusion under three schemes: `four_class` (`merge_fe_class_viii`), `six_class`, and the user-requested `five_class` (Mn/Cu/Zn/Fe/Co+Ni). Each arm selects its checkpoint by native `val_metal_balanced_acc`. The endpoint is common-four balanced accuracy (Mn, Cu, Zn, Class VIII = Fe+Co+Ni), and native metrics plus Fe/Co/Ni recalls are kept. There is minor tension between the files: AGENTS treats `five_class` only as a labeled alternative, while Plan includes it explicitly.

**Sources:** Plan.md:228-238, 99-117, 151-154, 160-169; EXPERIMENT_STATUS.md:10-16, 108-116; docs/README.md:32-41; AGENTS.md:125-137, 528-534.

**Confidence:** high

## 2. Meaning of "5-fold" in the active campaign

**Answer:** The campaign uses five frozen PDB-grouped folds over the training-only PMM cohort, which is 7,398 ions / 3,992 PDB groups after exclusions. Every fold has every native class on both sides. Examples are ion-centered, and sibling ions stay in one group. The fold-0 validation set is 1,492 ions / 1,346 parent pockets / 195 PDB groups, with seed 42. Folds 1–4 are deferred. The files conflict: the status "Current objective" describes a separate exact-PMM 5-fold run split by `pocket_id`, and AGENTS' Stage 6 default is `pdbid` folds × seeds. The fold-assignment algorithm is not stated.

**Sources:** EXPERIMENT_STATUS.md:314-316, 265-266, 14-16; DATASETS.md:256-261; Plan.md:209-212, 240-242. Conflicting: EXPERIMENT_STATUS.md:397-399; AGENTS.md:487-494, 539-542.

**Confidence:** medium-high

## 3. Where GPU runs execute; who owns the GPU lifecycle

**Answer:** Recent runs used a GCP L4 VM (`deepmzyme-l4`, us-central1-c, with a recovery in us-central1-a). The rules are one active GPU, one training worker per GPU, and caps of 4h/$6 per session and 6h/$10 per UTC day. Earlier runs used Colab. `gpu-use-skill` is the mandatory entry point. One coordinator owns allocation, submission and shutdown; an optional monitor is read-only; the runbook and controller own the commands. The default path does not say who the coordinator is or give the controller commands. AGENTS' "G4-class GPU" and Drive-storage assumptions are Colab-era and stale.

**Sources:** AGENTS.md:29-51, 467-469; EXPERIMENT_STATUS.md:26-27, 55-69, 91-99, 238-245, 256-257, 302-303; Plan.md:268-271, 394-395; docs/README.md:65-67, 84-94.

**Confidence:** medium

## 4. Next action for the project right now

**Answer:** No compute action is authorized. The latest entry (2026-09-28) keeps the nine fold-0 fits and defers the remaining 36. Resuming needs an explicit user request, refreshed budget authority and a new readiness check. The GVP/ESMC improvement plan is saved for separately scoped future work, so the effective next step is a user decision. Older "Next:" lines further down the same file have not been marked superseded and could mislead: a pending 34h/$34 decision, a recovery-pass decision, and a 2026-09-16 readiness admission.

**Sources:** EXPERIMENT_STATUS.md:10-37; docs/README.md:32-34; docs/notebook_outputs/README.md:46-49. Stale: EXPERIMENT_STATUS.md:65-69, 136-138, 172-175, 411-425; DATASETS.md:12.

**Confidence:** medium

## 5. Held-out test sets already opened/evaluated

**Answer:**
- **Non-overlap PinMyMetal test:** evaluated in seven early Only-GVP runs (352 pockets each), so it is not pristine.
- **Exact PinMyMetal:** shares those test structures, so it is not untouched, although no exact-trained evaluation is recorded.
- **CARE clusterRes30:** test metadata was exposed by accident. This was not a model evaluation.
- **PMM campaign:** no held-out access.

**Conflict:** the status "Current objective" reports 2026-09-23 held-out results (352 pockets, 5-fold ensembles for three models). The DATASETS.md test-use ledger (last audited 2026-09-14) does not record them.

**Sources:** DATASETS.md:12-20, 68-73, 219-226, 294-308, 511-534; docs/notebook_outputs/README.md:223; Plan.md:902-904, 1208-1210; EXPERIMENT_STATUS.md:104, 212-213. Conflicting: EXPERIMENT_STATUS.md:404-409.

**Confidence:** medium

## 6. What is forbidden before Stage 7

**Answer:** The following are forbidden before Stage 7:
- Any held-out test evaluation, or any selection that uses test data. `INCLUDE_HELD_OUT_TEST_DURING_TRAINING` must be `False` in every non-final stage.
- Recommending or launching Stage 7 unless all of these exist: a configuration selected by Stage 6 grouped folds (or a labeled fallback), a completed Stage 6B full-train refit, and that refit fixed as the Stage 7 source.
- Leaving the primary report, ensemble, calibration/temperature or thresholds unfixed.
- Running Stage 7 while the primary final-test route is unresolved.

Exploratory single-GPU confirmation does not count as Stage 6B evidence.

**Sources:** AGENTS.md:405-413, 474-486, 525-527, 861-865; Plan.md:272-288, 431-435, 1028-1041, 1062-1068, 1073-1082, 1235-1236, 1242-1243.

**Confidence:** high
