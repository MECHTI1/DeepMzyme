# PMM trunk-learning-rate comparison

Status: active (2026-10-13 CPU preparation complete; GPU start not authorized)

One bounded, validation-only comparison:
- **Candidate:** the v3-selected four-class late fusion with GVP-trunk learning rate 1e-4 instead of 3e-4.
- **Unchanged:** base rate 3e-5, groups, 50-epoch cosine schedule, terminal checkpoint, data and seed 42.
- **Fits:** five candidate fits against the five verified v3 controls (reference mean common-four BA 74.28%).
- **Objective:** `four_class` only (user, 2026-10-13).

**Rule** (frozen before any fit in the [specification](trunklr_assessment_spec.json)):
- **Primary, folds 0–4:** mean gain above 0.2 points.
- **Intervals:** fold-bootstrap and t lower 95% bounds above 0.
- **Recall:** no class mean recall drop above 3 points; no zero recall in any fold.
- **Guard:** folds 1–4 mean gain above 0; folds 1–4 are also reported as sensitivity.
- **Outcome:** passing means a promising candidate for further confirmation; otherwise the reference is retained.

**Budget** (pending authorization): one L4 session, $2.50 compute ceiling, 2.5-hour hard stop.

**Files:** `run_pmm_trunklr.py` (VM), `pmm_trunklr.py` (launcher), `pmm_trunklr_assessment.py`,
`tests/test_trunklr.py`; records on Data1 `campaigns/pmm_trunk_lr_2026-10/`. v3 results unchanged; no test, refit
or sweep.

History and readiness evidence: [log](log.md).
