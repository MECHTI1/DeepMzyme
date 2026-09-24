# Full PMM source-row feature crosswalk audit

Status on 2026-09-23: **all 9,408 released class-model rows were audited;
6,791 have one feature-supported structural ion, 153 remain ambiguous, and
2,464 are unavailable under the fixed rule.** This is **72.18% matched
coverage**, not a near-complete or representative reconstruction. These are
source-row audit records, **not training examples**. No `very_exact-pmm_sets`
training dataset, bundle, notebook option, or run was created; Part 1 is **not
ready for Part 2**.

The versioned isolated output is
`/home/mechti/PycharmProjects/DeepMzyme_pmm_source_audit_output/pmm_feature_crosswalk_full_v1/`.
`row_crosswalk.csv` has exactly one record per source UID with the original
train/test side and numeric row ID. `exclusion_ledger.csv` contains every
ambiguous or unavailable row. All eligible structural ions, their identities,
feature differences, and pass/fail results are preserved in the 95 compressed
`chunks/chunk_*.jsonl.gz` files, referenced from each CSV row.
`coverage.json`, `materialization_gate.json`, `independent_review_sample.csv`,
and `independent_review.json` provide the aggregate and review evidence.
The scripts are
[`src/audit_pmm_feature_crosswalk.py`](../src/audit_pmm_feature_crosswalk.py),
[`src/summarize_pmm_feature_crosswalk.py`](../src/summarize_pmm_feature_crosswalk.py),
and [`src/review_pmm_feature_matches.py`](../src/review_pmm_feature_matches.py).

## Fixed matching rule

The source train/test file SHA-256 values are pinned in the auditor, as is the
Figshare v1 metal-type file. Each row retains its source checksum-derived UID,
original side, 1-based row number, `residueid_ion`, `metalid`, and PMM numeric
label. All 4,706 distinct source PDB IDs were processed serially from current
RCSB PDB gzip files, with an 8 MB compressed and 32 MB expanded per-file cap;
no row failed retrieval or either size cap. The source PDB byte version used
by PMM remains unavailable. Current structure SHA-256 values and URLs are
recorded per row. There is no catalytic or enzyme annotation filter.

The check uses the released PMM chemical-class tests and requires **exact
equality of all 24 atom-class histogram counts** in the 2–3, 3–4, and 4–5 Å
shells. It independently requires the source `coordnum_inner` nearest
O/N/S/Se donor distances (including waters) within 2.8 Å to reproduce
`distance_min`, `distance_avg`, and `distance_max`, each within **0.001 Å**.
The deposited CH/ED/H ligand-type counts must agree when available. Deposited
chain/residue/coordinates within **0.005 Å** corroborate a candidate but
their absence does not eliminate it. Every structural ion of an eligible
element is retained and scored; an accepted row must have **exactly one**
passing ion. The eligible element comes from the deposited source-ID type
record when present, otherwise from PMM's four-class label (Fe/Co/Ni are all
considered for Class VIII). PDB ID, element, row order, or nearest-ion
distance alone never assigns a row. The first-shell reconstruction is
empirical rather than the missing historical NEIGHBORHOOD export; its control
validation is documented in the preceding
[bounded feasibility check](PMM_FEATURE_MATCH_FEASIBILITY.md). No threshold
was changed during this full scan.

## Coverage by original side and metal class

| Side / class | Source rows | Unique | Ambiguous | Unavailable | Unique coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train Mn | 2,590 | 1,732 | 75 | 783 | 66.87% |
| Train Class VIII | 2,629 | 2,030 | 14 | 585 | 77.22% |
| Train Cu | 400 | 285 | 12 | 103 | 71.25% |
| Train Zn | 2,301 | 1,644 | 36 | 621 | 71.45% |
| **Train total** | **7,920** | **5,691** | **137** | **2,092** | **71.86%** |
| Test Mn | 167 | 131 | 3 | 33 | 78.44% |
| Test Class VIII | 252 | 182 | 3 | 67 | 72.22% |
| Test Cu | 64 | 52 | 1 | 11 | 81.25% |
| Test Zn | 1,005 | 735 | 9 | 261 | 73.13% |
| **Test total** | **1,488** | **1,100** | **16** | **372** | **73.92%** |
| **Both sides** | **9,408** | **6,791** | **153** | **2,464** | **72.18%** |

Of the unavailable rows, **2,462** had eligible structural ions but no
candidate passed the fixed feature rule; **2** had no eligible structural ion.
Thus retrieval failures and the 153 ambiguous rows do not account for the
large missing cohort. Unavailable does **not** mean the original PMM site was
wrong or absent; historical structure preprocessing, feature generation, or
site identity may differ from this public-data reconstruction. The precise
reason and every scored candidate remain in the audit output.

## Representation and structural overlap

Coverage is strongly site-type dependent, so the 72.18% total cannot stand
for the released cohort. Among deposited types with at least 100 source rows:

| Site type | Unique / source rows | Coverage |
| --- | ---: | ---: |
| ED2 | 189 / 435 | 43.45% |
| H1ED1 | 205 / 427 | 48.01% |
| H1ED2 | 184 / 359 | 51.25% |
| ED3 | 136 / 237 | 57.38% |
| H2 | 707 / 995 | 71.06% |
| C4 | 606 / 663 | 91.40% |
| C3H1 | 441 / 479 | 92.07% |
| C2H2 | 214 / 231 | 92.64% |

The deposited type is unavailable for 4,442 source rows; 3,222 of those
matched (72.54%). The source `ched_count=2` group matched 549/1,145
(47.95%), versus 3,157/4,253 (74.23%) for count 3 and 2,934/3,832
(76.57%) for count 4. `coverage.json` retains every type, count, side, and
class. No two uniquely mapped source rows were assigned the same full
structural-ion identity, and no uniquely mapped ion crosses the original
train/test sides. This does not resolve structural overlap involving excluded
rows. Source rows stay distinct even if ions belong to one physical pocket.

A deterministic review sample selected 26 accepted rows across original
sides, metal classes, and deposited-type presence. An independent Biopython
parse of freshly downloaded, SHA-256-matched PDB bytes reproduced the ion
identity, 24-bin fingerprint, first-shell summaries, and applicable type
counts for **26/26**. The sample includes accepted ED2 and H1ED1 sites; it
does not validate all 6,791 matches individually.

## Materialization decision

The fixed materialization gate required at least 95% unique coverage overall,
90% in every side/class group, 85% for each deposited site type with at least
100 rows, zero cross-side ion collisions, and a passing independent review.
The first three gates fail; the collision and review gates pass. The
`row_crosswalk.csv` and exclusion ledger are **audit artifacts**, not a mapped
training dataset. `very_exact-pmm_sets` was not materialized and Part 1 is
not ready for Part 2 under this rule. The exact effective published PMM
classification cohort and original fivefold row assignments are also still
unverified.

Any future PMM comparison on a defensible mapped cohort must rerun the PMM
metal-classification baseline on **those same rows and folds**. The paper's
reported scores remain historical context unless cohort identity is
established; the existing [implementation plan](VERY_EXACT_PMM_SETS_PLAN.md)
owns that comparison policy.
