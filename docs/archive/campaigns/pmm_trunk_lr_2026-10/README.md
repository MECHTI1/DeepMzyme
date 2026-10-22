# PMM trunk-learning-rate comparison

Status: closed (2026-10-13 reference retained; lower trunk rate kept as an unpromoted candidate; log tlr-004)

One bounded, validation-only comparison:
- **Candidate:** the v3-selected four-class late fusion with GVP-trunk learning rate 1e-4 instead of 3e-4.
- **Unchanged:** base rate 3e-5, groups, 50-epoch cosine schedule, terminal checkpoint, data and seed 42.
- **Fits:** five candidate fits against the five verified v3 controls (reference mean common-four BA 74.28%).
- **Objective:** `four_class` only (maintainer decision, 2026-10-13).

**Rule** (frozen before any fit in the [specification](trunklr_assessment_spec.json)):
- **Primary, folds 0–4:** mean gain above 0.2 points.
- **Intervals:** fold-bootstrap and t lower 95% bounds above 0.
- **Recall:** no class mean recall drop above 3 points; no zero recall in any fold.
- **Guard:** folds 1–4 mean gain above 0; folds 1–4 are also reported as sensitivity.
- **Outcome:** passing means a promising candidate for further confirmation; otherwise the reference is retained.

**Result** ([log tlr-003](log.md#tlr-003)): mean +1.16 points (75.45 vs 74.28), bootstrap [−0.65, +3.11]; rule (b)
not met, so the reference is retained. One session, $1.22 of $2.50 compute.

**Files:** `run_pmm_trunklr.py` (VM), `pmm_trunklr.py` (launcher), `pmm_trunklr_assessment.py`,
`tests/test_trunklr.py`; records on Data1 `campaigns/pmm_trunk_lr_2026-10/`. v3 results unchanged; no test, refit
or sweep.

**Closed** ([log tlr-004](log.md#tlr-004)): archived here; the old path `docs/campaigns/pmm_trunk_lr_2026-10` is a
registered alias for the frozen tools. Retrospective completion check and additive v2 safeguards:
[log tlr-005](log.md#tlr-005).

History and evidence: [log](log.md).
