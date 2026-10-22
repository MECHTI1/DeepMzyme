# Stronger EC evidence: route proposal

Campaign [README](README.md); evidence in [log ec1s-001](log.md#ec1s-001). The generated copy is on the secondary drive at
`DeepMzyme_Data/campaigns/ec1_independent_support_2026-10/report_20261013T191929Z/next_scientific_decision.md` (SHA-256 `47d9e065…`,
bound by `artifact_manifest.json` `6674d052…`). This page has the same substance with repository links.

Evidence boundary: counts from the saved EC1 development membership and the recorded CARE training-side build tables.
Nothing here trains, promotes, downloads or changes eligibility. Neither route is claimed to improve performance, and
neither merges new examples into the existing cohort.

## What limits EC7 now

- **Support.** 6 EC7 proteins in one dependence component (with one EC1 member, Q59647); 11 qualifying EC7–EC7
  pairs keep the six connected without it. Used as indivisible split units, these components would prevent
  seven-class train/validation coverage; they were built as uncertainty units, not as a mandatory split rule.
  Accession-grouped validation can hold EC7 only with related training sequences, which must be disclosed.
- **The training source is exhausted under current eligibility.** Of 179 unambiguous EC7 CARE clusterRes30 training
  proteins: 141 lack a UniProt-supported transition metal, 24 lack assets, 8 were prepared but carry only
  noncatalytic-predicted sites (9 site rows; the baseline cohort table is the only-catalytic table, and these must
  never be relabelled), and 6 are in the baseline.
- **The missing assets are deliberate.** 22 CIFs were skipped because the JSON had no annotation-compatible elemental
  candidate (re-derived with the recorded policy: 0), and 2 JSONs returned HTTP 404 (P08748, Q00227, both
  Cu-annotated). Fetching the 22 skipped CIFs would add no eligible site.
- **Rejections.** 7 elemental candidates rejected by annotation; 12 only metal-containing compounds outside the
  exact-element mapping; 3 no qualifying donor/transplant; 2 recorded missing JSON; 0 unexpected changes.

## Route 1: preserve current eligibility

- **Population:** EC depth-1 proteins with an annotation-compatible elemental transition-metal pocket (Mn, Fe, Co,
  Ni, Cu, Zn) predicted catalytic, as in the v12 EC1 cohort.
- **Input/eligibility changes:** none.
- **Evidence available:** the CARE clusterRes30 training source adds no EC7 example under this definition; only the
  two HTTP-404 JSONs are unresolved.
- **Unknowns:** whether other permitted, non-test sources hold independent EC7 (and EC5/EC6) transition-metal
  catalytic pockets; their identity/homology to the development set and every held-out set; whether the two 404
  entries now exist.
- **Code/data preparation:** a CPU inventory of already-local permitted training-side sources with the same filters;
  cross-source identity/homology and held-out exclusion certification; new structures only through the unchanged
  `care_prepare` pipeline; a new cohort version identity.
- **Validation and rare classes:** accession-grouped; EC7 recall with intervals and disclosed related training
  sequences; component-grouped EC7 assessment needs at least two independent EC7 components.
- **Reuse:** `care_prepare`, the remoteness tooling, this reporter and the v12 membership.
- **Authorization/choice:** which sources are permitted; any AlphaFill/UniProt/PDB download needs the maintainer's
  authorization.

## Route 2: separately versioned cofactor-inclusive EC cohort

- **Population:** a different one: enzymes whose metal is held in a cofactor (Fe–S clusters, hemes, cobalamins) as
  well as elemental sites.
- **Input/eligibility changes:** a reviewed compound-to-metal mapping, a cofactor-pocket centre definition, annotation
  matching for cofactors, and a decision whether MAHOMES catalytic prediction applies to cofactor sites; the metal
  task stays out of this cohort unless separately defined.
- **Evidence available:** 19 of the 24 allowlisted EC7 accessions have qualifying AlphaFill donors carrying
  metal-containing compounds (FES, SF4, HEM/HEC/HEB/HDD); no CIF was downloaded for them.
- **Unknowns:** scientific appropriateness (cofactor identity may act as an EC shortcut), independence (complex
  subunits, homologs), transplant quality for large cofactors, MAHOMES applicability, and whether heme-only
  annotations hide further proteins among the 141 without a supported annotation (not inspected).
- **Code/data preparation:** an additive, versioned compound mapping in `care_prepare`; CIF downloads for candidates;
  a separate work root and cohort; the same support/remoteness analysis; held-out exclusion.
- **Validation and rare classes:** a separate cohort identity, never merged into v12; matched comparison inside the
  cohort only; cofactor-stratified metrics as a shortcut control.
- **Reuse:** the fetch/build pipeline (with the additive option), this reporter, the remoteness tooling.
- **Authorization/choice:** the scientific decision to define this population, then download authorization.

## Missing interface (documented, not implemented)

EC training cannot take an explicit frozen train/validation membership: `--fold-split-source membership` needs
`--source-cohort-csv`, which [`src/training/config.py`](../../../src/training/config.py) restricts to
`--task metal --metal-example-unit ion`. The smallest later bridge: accept an EC membership CSV (structure/pocket id,
protein group, fold) with its SHA-256 for `--task ec`, refuse rows outside the cohort, keep `structure_id` EC group
weighting, and prove parity by reproducing the saved v12 fixed split exactly. Its cohort and validation contract
depend on the route chosen.

## Recommendation and remaining decision

Route 1 first: keep the v12 population, which stays comparable with the metal task's elemental-site definition, and
run a CPU-only inventory of already-local permitted training-side sources for independent EC7/EC6/EC5 pockets. Treat
Route 2 as a separately versioned exploratory cohort, opened only after an explicit scientific decision on cofactor
pockets.

**Remaining decision (the maintainer's):** which training-side sources are permitted for the Route 1 inventory, and whether
Route 2 should be scoped at all. No download, training or GPU use follows from this proposal. Related policy:
[Plan, EC classification](../../../Plan.md#3-train-the-ec-number-classification-model); the
[EC/CLEAN30 plan](../../plans/pocket_level_EC_task_compare_clean30_final_plan.md).

Decided 2026-10-13: the maintainer chose the Route 1 inventory and its permitted sources (result below); Route 2 stays
unscoped.

## Route 1 inventory result (ec1s-002)

The Route 1 inventory ran on the permitted training sources ([log ec1s-002](log.md#ec1s-002); output on the secondary drive at
`DeepMzyme_Data/campaigns/ec1_independent_support_2026-10/inventory_20261013T211519Z/`, `recommendation.md`).

**Already usable:** nothing beyond the current membership.
- No EC5, EC6 or EC7 protein outside the development membership has documented eligibility in any source.
- The CARE clusterRes30 cohort is exhausted for these classes: every built protein with a predicted catalytic site is
  already a development member.
- The non-overlap PinMyMetal training chains fall outside the definition (PDB EC labels, experimental structures).
  Those with compatible annotations are all in or beside the development membership.

**Uncertain** (non-representative CARE accessions, unclustered work root):

| Class | Partially prepared, not built (units / clusters / clusters without a known relation) | Supported metal, no local AlphaFill (units without a known relation / clusters) |
|---|---|---|
| EC7 | 97 / 5 / 3 (58 units: Q00227, O30085, A4VXE1) | 629 / 26 |
| EC6 | 1,922 / 34 / 18 (625 units) | 1,162 / 45 |
| EC5 | 532 / 28 / 10 (111 units) | 214 / 21 |

"Without a known relation" means no exact-sequence or same-cluster link was found. It is not independence: the six
development EC7 proteins lie in six clusterRes30 clusters but form one dependence component.

None of these candidates is certified for held-out exclusion, and none is built or has a catalytic prediction.

**Worthwhile next steps, in order** (each needs the maintainer's authorization; none is implied):
1. **Training-side relationship search (CPU, no download).** The frozen MMseqs2 protocol over the EC development set
   plus the candidates, all-vs-all, so that candidate–candidate dependence is counted; EC7 first. It prunes the most
   before anything else is spent.
2. **Label-blind held-out exclusion check.** Identifiers and exact sequences of the surviving candidates against the
   CARE test, CLEAN30 folds and PinMyMetal/Zenodo tests, plus the 128 absent representatives. It reads test data, so
   it must be authorized and disclosed.
3. **Training-only build and MAHOMES (CPU) of the surviving partially prepared candidates.** The pipeline needs:
   - a restricted training-only manifest (the build has no accession allowlist);
   - `--splits train --no-fetch-missing-uniprot`;
   - RCSB entry metadata for 185 donor PDB entries: 80 copyable from the clusterRes30 cache and 105 to download.
4. **Targeted AlphaFill downloads.** JSON first, then CIF where a compatible candidate exists, for the surviving
   supported-metal candidates (`download_candidates.csv`), plus a re-fetch of the four HTTP-404 JSONs.
5. **Only then:** a separately versioned cohort decision and the EC frozen-membership training interface (still
   deferred).

Expected value: EC7 gains at most three new clusters from existing assets (58 proteins), and only if the search finds
them independent of the one development EC7 component and MAHOMES predicts catalytic sites. In the cohort, 6 of the
14 built EC7 proteins had a predicted catalytic site, all of them development members. Most of the potential EC7 gain
(26 clusters) needs downloads.

**Remaining decision (the maintainer's):** whether to authorize step 1, the training-side relationship search.
