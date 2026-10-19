# Bounded PMM feature-to-ion feasibility check

Status on 2026-09-23: **7 unique feature-supported row-to-ion matches in a
9-row, 6-structure check; 2 rows remain ambiguous.** This is a source-row
audit, **not a training dataset or the effective published classification
cohort**. The source files, older coordinate-only crosswalk, and all existing
datasets and runs are unchanged. No full-cohort matching or Parts 2–3 work was
started at the time of this feasibility check. The later
[complete feature audit](PMM_FULL_FEATURE_CROSSWALK.md) processed all 9,408
rows and found 6,791 unique matches, but failed its coverage and representation
gates; no training dataset was created.

The reproducible, lightweight check and its machine-readable output are at
`/home/mechti/PycharmProjects/DeepMzyme_pmm_source_audit_output/pmm_feature_feasibility_v1/`:
`check.py`, `candidate_feature_checks.json`, `reviewed_source_rows.csv`, and
`bounded_coverage.json`. It reads the pinned V1.0 class-model files, the
Figshare v1 site-type and coordinate tables, and six current RCSB PDB files.
The JSON records SHA-256 hashes of the inputs, all competing same-element ions,
per-candidate feature differences, deposited coordinate references, and the
source UID and original side. It contains audit records, not training examples.

## Matching rule and control check

The released
[`classmodel_features.py`](https://github.com/hhz-lab/PinMyMetal/blob/59ef46795920322c798db4e5ec500b04451f7904/metal_prediction/script/hybrid_algorithm/classmodel_features.py)
defines 24 chemical-class counts in 2–3, 3–4, and 4–5 Å bins;
[`chemfeatures.py`](https://github.com/hhz-lab/PinMyMetal/blob/59ef46795920322c798db4e5ec500b04451f7904/metal_prediction/script/hybrid_algorithm/chemfeatures.py)
defines the atom classes. The check ports those class tests and excludes water
from that histogram, as
[`feature_pre_chedh.py`](https://github.com/hhz-lab/PinMyMetal/blob/59ef46795920322c798db4e5ec500b04451f7904/metal_prediction/script/hybrid_algorithm/feature_pre_chedh.py)
does. It includes all PDB chains, which is necessary for cross-chain ligands
in `4onx`. The second feature family reconstructs the source
`distance_min/avg/max` from the `coordnum_inner` nearest O/N/S/Se atoms within
2.8 Å, including coordinating waters. This nearest-donor rule is an
**empirical reconstruction**, not a verbatim released PMM training-data
generator. An ion passes only if all 24 counts agree exactly and each of the
three first-shell distance summaries differs by at most 0.001 Å. A deposited
`resi_type` ligand fingerprint must agree when present. Deposited coordinates
are checked against PDB chain, residue, and x/y/z within 0.005 Å; their
absence cannot eliminate a candidate. A row is a unique feature match only
when exactly one eligible ion in the complete reviewed PDB model passes.

The four controls were selected because the deposition names an exact ion
coordinate and the PDB contains competing same-element ions. Each source row
matched the named ion with zero histogram-count difference, first-shell
distance differences below 0.000006 Å, and the deposited ligand fingerprint:

| Control source row | Named ion | Other same-element ions | Result |
| --- | --- | ---: | --- |
| Train 22, `4onx` `(765,2016)` | Zn A201 | 2 | Unique; nearest competing histogram L1 = 8 |
| Train 37, `1ozj` `(280,317)` | Zn A145 | 1 | Unique; competitor L1 = 12 |
| Train 40, `4nj4` `(425,3666)` | Mn A301 | 1 | Unique; competitor L1 = 2 |
| Train 75, `1vj2` `(228,2326)` | Mn A115 | 1 | Unique; competitor L1 = 3 |

The paired `4nj4` train row 4826 `(430,8572)` and `1vj2` train row 6065
`(230,7209)` uniquely match Mn B301 and Mn B115, respectively, with exact
histograms and all three first-shell summaries within 0.000008 Å. The
deposition does not give separate coordinates or type fingerprints for these
two rows; they are supported by the two distinct structural feature families.

Two simpler reconstructions failed on controls. Restricting the histogram to
the ion's own chain gives L1 = 26 for `4onx` A201 because its environment
includes another chain; using all chains gives L1 = 0. Excluding coordinating
waters from the first-shell calculation overcounts distance differences or
undercounts donors. These checks are recorded as sensitivity modes in the
JSON; neither failed approximation was used to certify a row. The released
files do not identify the historical ligand-chain filter or source PDB byte
version, so this rule is evidence for these inspected structures only.

## Challenging cases

`4d8f` train row 5698 `(residueid_ion=1316, metalid=3761)` uniquely matches
**Mn model 1, chain A, residue 402, atom serial 10808**, at
`(3.641, 19.721, -10.562)`. Its histogram L1 is zero; all three first-shell
summaries agree within 0.000003 Å; the deposited `H1ED2` type agrees with
one His and two Glu coordinating residues; and the deposited coordinate is
identical to that ion. Competing Mn B401, C402, and D402 have histogram L1
differences of 10, 6, and 13, respectively, and do not pass the rule.

`1a0e` train row 4547 `(887,4637)` passes for **Co A492 and Co D492**.
`1a0e` test row 1 `(886,4638)` passes for **Co A491 and Co D491**. Each pair
has the same 24-bin histogram and first-shell summaries within 0.001 Å;
their deposited `H1ED1` and `ED4` ligand types also agree. The deposition
names only chain-A coordinates but does not link them to the source internal
IDs or rule out the symmetry-related chain-D ions. Both source rows remain
**ambiguous**, with no assigned coordinate or training example.

## Coverage after this bounded check

These are lower bounds against the full released source sides; unreviewed rows
are **not** called unmatched or incomplete by this test.

| Original side / class | Unique feature matches / source rows | Ambiguous in this check | Not reviewed here |
| --- | ---: | ---: | ---: |
| Train Mn | 5 / 2,590 | 0 | 2,585 |
| Train Class VIII | 0 / 2,629 | 1 | 2,628 |
| Train Cu | 0 / 400 | 0 | 400 |
| Train Zn | 2 / 2,301 | 0 | 2,299 |
| Test Mn | 0 / 167 | 0 | 167 |
| Test Class VIII | 0 / 252 | 1 | 251 |
| Test Cu | 0 / 64 | 0 | 64 |
| Test Zn | 0 / 1,005 | 0 | 1,005 |
| **Total** | **7 / 9,408** | **2** | **9,399** |

The initial coordinate-only [deposition crosswalk](PMM_DEPOSITION_CROSSWALK.md)
still reports zero matches under its stricter direct-coordinate rule. This
bounded feature check raises the **observed matched lower bound**, but does not
establish full-cohort coverage, the historical NEIGHBORHOOD ID export, the
effective PMM fitted cohort, or published fivefold row IDs. It cannot support
`very_exact-pmm_sets` construction or training at the time of the pilot. The
later [complete feature audit](PMM_FULL_FEATURE_CROSSWALK.md) establishes
full-source-table coverage and confirms that materialization gates still fail.
