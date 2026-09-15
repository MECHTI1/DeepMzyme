# Training-only metal × EC1 association — 2026-09-15

**Role:** Phase-3 descriptive development analysis, with separate PinMyMetal and
CARE panels. This is Grade-6 exploratory evidence, not a model comparison or a
certificate for shared metal–EC training. The exact recipe is in the
[EC playbook](../../EC_TRAINING_PIPELINE_PLAYBOOK.md#phase-3--training-only-metal--ec1-association).

## Inputs and units

The analysis uses the exact retained **training** pockets from the
[metal readiness artifact](../raw/metal_architecture_pilot_20260915/readiness/expected_split.json)
and [EC standalone readiness artifact](../raw/ec1_standalone_v12_20260914/expected_split.json).
All 2,174 retained pocket labels were reconstructed from canonical metal
clusters and the catalytic-site training summaries, then checked against the
saved source-scheme metal and EC1 targets. Training structure hashes passed;
no conflicting duplicate summary-site labels were found.

Metal labels describe pockets/sites. EC1 labels describe the protein or
structure. Each eligible protein group therefore contributes total weight one,
divided among its eligible pockets, alongside the raw pocket-count view. The
group key is the PDB ID in PinMyMetal and the full UniProt accession in CARE.
No group had conflicting retained EC1 targets. These groups do not establish
sequence-homology independence.

| Source and view | Retained training pockets | Usable pairs | Usable groups | Excluded |
|---|---:|---:|---:|---|
| PinMyMetal, native six or common four | 1,181 | 1,163 | 1,136 | 18 lack a single EC1 target |
| CARE, native six | 993 | 982 | 742 | 11 lack a single native metal target |
| CARE, inclusive common four | 993 | 983 | 743 | 10 lack a single common-four metal target |
| CARE, common four on native-six-eligible pockets | 993 | 982 | 742 | Same 10, plus one Fe+Ni pocket |

Class VIII means Fe+Co+Ni in every common-four view. The Fe+Ni pocket has a
valid common-four target but no single native-six target. Each view records its
own exclusion counts; no majority-metal rule replaces ambiguous labels.

The saved internal-validation groups—110 PDB groups and 42 CARE groups—were
excluded from all statistics. Only their group IDs were used to check separation
from training. The analysis opens no external test files and inherits external
training membership from the standalone artifacts. It does not independently
re-certify that membership against external test identities. A preceding broad
code/provenance search incidentally printed test candidate metadata; that
separate access belongs in the [dataset access record](../../DATASETS.md), and
those labels were not used as association inputs.

## Association measures

Each table below uses **one total weight per eligible group**. MI is mutual
information in natural-log units; NMI is `2 MI / (H(metal) + H(EC1))`. Cramér's V
is the uncorrected Pearson-statistic normalization defined in the frozen recipe;
its finite-sample bias is not corrected.
The table describes each selected source population separately; differences
between rows are not a ranking of biological reliability or model utility.

| Panel | Metal view | Cramér's V | MI, nats | NMI |
|---|---|---:|---:|---:|
| PinMyMetal | Native six | 0.28652 | 0.20288 | 0.14319 |
| PinMyMetal | Common four | 0.32667 | 0.15777 | 0.12473 |
| CARE | Native six | 0.41315 | 0.31171 | 0.22935 |
| CARE | Inclusive common four | 0.50600 | 0.27649 | 0.21251 |
| CARE | Common four, native-six eligible | 0.50539 | 0.27535 | 0.21168 |

The PinMyMetal matched-cohort common-four view is identical to its inclusive
view. In CARE, excluding the additional Fe+Ni pocket changes common-four V
from 0.50600 to 0.50539; the cohort distinction remains explicitly recorded.

All native-six and common-four tables fail the conventional expected-cell
heuristic: some expected counts are below one and more than 20% are below five.
Even adequate expected counts would not make repeated pockets independent.
Pearson chi-square statistics and diagnostics are saved, with **no asymptotic
site-level p-values**.

The four predeclared primary tests permute whole-group EC1 labels, preserving
each group's metal profile, with 9,999 permutations and seed 42. None reached
the observed group-weighted MI. Each plus-one Monte Carlo p estimate is 0.0001
(the available resolution), or 0.0004 after Holm adjustment across the four
tests. These are exploratory permutation results conditional on exchangeability;
homology and source selection remain uncontrolled. The matched-cohort
sensitivity has no additional permutation test.

## Common-four pocket counts

These integer counts are descriptive site-level counts, not independent-protein
sample sizes. The portable CSV contains every native-six and common-four table,
both weighting views, `P(EC1 | metal)`, and `P(metal | EC1)`.

| PinMyMetal | EC1=1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mn | 163 | 144 | 172 | 7 | 48 | 6 | 0 |
| Cu | 56 | 3 | 4 | 0 | 0 | 0 | 1 |
| Zn | 13 | 26 | 99 | 22 | 8 | 6 | 0 |
| VIII | 259 | 24 | 69 | 17 | 13 | 3 | 0 |

| CARE | EC1=1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mn | 13 | 86 | 195 | 17 | 17 | 16 | 0 |
| Cu | 29 | 0 | 0 | 0 | 0 | 0 | 5 |
| Zn | 38 | 33 | 264 | 58 | 9 | 14 | 0 |
| VIII | 128 | 9 | 35 | 8 | 7 | 2 | 0 |

For example, the site-level `P(EC1=1 | Cu)` is 56/64 = 87.5% in PinMyMetal
and 29/34 = 85.3% in CARE. The reverse conditional is different: Cu accounts
for 56/491 = 11.4% and 29/208 = 13.9% of EC1=1 pockets, respectively. Sparse
cells, such as EC1=7, should not be generalized from these small counts.

## Interpretation and next boundary

Metal and EC1 annotations are associated within both permitted development
panels. This does not show that adding metal supervision improves EC prediction:
the encoder may already capture the same information, the source labels may be
biased, or the auxiliary objective may cause negative transfer.

PinMyMetal-derived labels on experimentally resolved structures remain distinct
from CARE's computational AlphaFill/MAHOMES transfers and UniProt/catalytic-site
selection. The sources were never pooled. A cross-source PDB-to-UniProt identity
union, sequence-overlap audit, and shared-training held-out exclusion protocol
are **not certified** by this analysis. Those checks still precede any matched
EC-only versus EC-plus-metal experiment. No model training, model promotion,
hyperparameter selection, or held-out evaluation occurred in this analysis.

## Reproducibility

The [portable evidence](../raw/metal_ec1_association_20260915/README.md) contains
the predeclared source/analysis manifest and hash, exact frozen training-pair
CSV, structure/eligibility audit, all contingencies and conditionals, statistical
results, and execution receipt. The recipe was written before execution; the
retained labels and source hashes were frozen before statistics. The local
runner verifies both input and source hashes before calculating results.
