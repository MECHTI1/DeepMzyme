# PMM source-row audit for `very_exact-pmm_sets`

Status on 2026-09-23: **source audit implemented; structural dataset not yet certified**.
The isolated outputs are source-row audit records, **not training examples**
or a reproduction of PMM's reported score. The initial audit is at
`/home/mechti/PycharmProjects/DeepMzyme_pmm_source_audit_output/very_exact_pmm_sets_source_audit_v1/`.
The follow-up [deposited-data crosswalk](PMM_DEPOSITION_CROSSWALK.md) is at
`/home/mechti/PycharmProjects/DeepMzyme_pmm_source_audit_output/pmm_deposition_crosswalk_v3/`.
A separate [bounded feature check](PMM_FEATURE_MATCH_FEASIBILITY.md) later
identified seven unique ion matches among nine reviewed source rows; the full
cohort was subsequently reviewed in the
[complete feature audit](PMM_FULL_FEATURE_CROSSWALK.md), which found 6,791
unique matches but did not materialize a structural training dataset.

## Pinned source and cohort

The local PMM files are byte-identical to the [PMM GitHub release at commit
`59ef46795920322c798db4e5ec500b04451f7904`](https://github.com/hhz-lab/PinMyMetal/tree/59ef46795920322c798db4e5ec500b04451f7904).

| File | SHA-256 | Raw rows | Rows after released `dropna()` |
| --- | --- | ---: | ---: |
| `classmodel_train_set` | `4748babd2b6ac0706cd9ed4bcfd4855c8d2c5535f01813fb2e10b68b31a24c0f` | 7,920 | 7,920 |
| `classmodel_test_set` | `ec6427ad0b8a18261dbc1d6822bdcfe7194730c7da4a36c208957bcb00f6a0f2` | 1,488 | 1,488 |

The [released training script](https://github.com/hhz-lab/PinMyMetal/blob/59ef46795920322c798db4e5ec500b04451f7904/data_model/train_chedhclassmodel.py)
has SHA-256 `9b5463fc7623ed72468ebd33b43d26def50d3ead3f33672511df9feef0603445`.
It calls `dropna()` on both tables, then asks `DataFrame.columns.drop()` to
remove a `source` column. Neither released table has that column, so the script
does not run unchanged on those files. The known post-`dropna` counts are **not**
a verified effective training/evaluation cohort for the published model; the
paper's further selection and oversampling remain unreconstructed.

The [released classification SQL](https://github.com/hhz-lab/PinMyMetal/blob/59ef46795920322c798db4e5ec500b04451f7904/metal_prediction/script/hybrid_algorithm/script10_clasmodel_result.sql)
explicitly maps codes `1→Mn`, `2→Fe+Co+Ni` (`FECONI`), `6→Cu`, and `7→Zn`.
These are the only codes in the two class-model tables. `2` is the existing
`four_class` Class VIII target, not a single observed element.

| Side | Mn (`1`) | Class VIII (`2`) | Cu (`6`) | Zn (`7`) | Unique PDB IDs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 2,590 | 2,629 | 400 | 2,301 | 4,195 |
| Test | 167 | 252 | 64 | 1,005 | 1,179 |

There are 668 PDB IDs on both source sides. No source row UID or
`(pdbid, residueid_ion, metalid)` key is duplicated within a side. The row
UID includes the file SHA-256 and 1-based data-row number, so an overlapping
PDB never transfers a label to the other side.

The [Figshare v1 deposition](https://doi.org/10.6084/m9.figshare.25011212.v1)
was downloaded through Figshare's public API after its advertised downloader
returned HTTP 403. The complete 99,424,422-byte archive matches the published
MD5 `dcb73c3520dcb0c1df07b5cd9126f147`; its SHA-256 is
`e8efd7aa463f217c0ada67a538c3b4eb20f83927936d73cc9be2a1822696dfb8`.
Deposited train/test files are byte-identical to the pinned local files. The
deposited Figure 2 notebook specifies `StratifiedKFold(n_splits=5)` on source
row order with default `shuffle=False`, but no serialized published row-fold
IDs were found. See the [crosswalk review](PMM_DEPOSITION_CROSSWALK.md).

## Structural crosswalk and input boundary

`residueid_ion` is a NEIGHBORHOOD internal residue key; `metalid` has no
verified PDB-ion mapping. Neither is demonstrated to be a PDB chain-residue
identifier. In the source tables, `1a0e` train row 4,547
is `(887, 4637, Class VIII)` and test row 1 is `(886, 4638, Class VIII)`.
The available local `1a0e` chain-A structure contains nearby Co ions A491
and A492 (and Co A493). Their PMM identifier-to-ion pairing is not documented
in the inspected material. `4d8f` train row 5,698 is
`(1316, 3761, Mn)`; the local chain-A structure contains Fe A401 and Mn A402,
but the source identifier's coordinate link is likewise unverified. These
examples remain separate rows. The deposited-data review classifies them as
**ambiguous**, not as verified ion matches; see the
[case review](PMM_DEPOSITION_CROSSWALK.md#case-reviews).

The builder accepts a reviewed evidence CSV with PMM row UID and identifiers,
coordinate file/version/hash, author and label chain, ion model/residue/altloc,
symmetry/context declaration, and provenance. It checks the file checksum,
unique ion identity, coordinates when provided, and observed element against
the verified class code. An ambiguous candidate, missing structure/context,
checksum mismatch, or label conflict cannot enter a site manifest. Verified
rows retain source side and one anchor per row even when their ions share a
4.5 Å physical group. The initial output has **0 exact, 0 ambiguous, 0
unmatched, and 9,408 incomplete mappings** because deposited evidence was not
yet incorporated. The follow-up crosswalk has **0 exact, 3,409 ambiguous, 0
unmatched, and 5,999 incomplete** records. The initial `train/` and `test/`
site and structure manifests remain headers only. Neither output is a
training-ready dataset. See the [per-class coverage](PMM_DEPOSITION_CROSSWALK.md#coverage).

The optional prototype pocket JSON uses only protein CA residue identity and
geometry relative to the verified anchor; observed ion symbol and PMM label
stay out of the graph. Focused mutation tests show that changing an ion symbol
with geometry fixed leaves that input unchanged, and a checksum mismatch stops
mapping. It does not use existing ESM, external, or RING caches. The legacy
loader clusters ions, derives targets from observed/summary metal information,
and retains metal identity in metadata. No claim of label-safe integration with
the full DeepMzyme model is made until the opt-in loader and all tensor channels
are audited and tested. The source-row path never applies the legacy catalytic
or EC filter; a non-EC/MAHOMES-negative fixture is retained by its PMM record.

## Reproduce the lightweight audit

From the isolated worktree, after the interpreter check required by `AGENTS.md`:

```bash
/home/mechti/miniconda3/envs/DeepMzyme/bin/python -c "import sys; print(sys.executable)"
/home/mechti/miniconda3/envs/DeepMzyme/bin/python src/build_pmm_site_dataset.py \
  --output-root /home/mechti/PycharmProjects/DeepMzyme_pmm_source_audit_output/new_audit_version
```

The output root must not exist. The command reads the pinned local PMM tables
and writes only under that root. It performs no coordinate sweep, feature
build, training, GPU work, Colab action, or test evaluation. To build an actual
matched subset, first obtain and review a PMM row-to-coordinate source record
for each included row and a pinned full-context coordinate version. Supply the
reviewed evidence CSV and read-only structure root; only then use
`--emit-graphs` for the verified subset. The resulting prototype graph format
is not yet wired into `src/train.py`.

The remaining scientific gates are the PMM identifier crosswalk, symmetry and
full-chain context verification, independent PMM baseline on the same eventual
matched cohort, and validation fold identity (published IDs if recovered,
otherwise a newly named grouped manifest). No held-out test is opened here.
