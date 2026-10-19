# PMM deposited-data row-to-ion crosswalk review

Status on 2026-09-23: **the coordinate-only review below found zero certified
row-to-ion matches**. A later, bounded
[feature-matching check](PMM_FEATURE_MATCH_FEASIBILITY.md) found seven unique
matches among nine sampled rows. The subsequent
[complete feature audit](PMM_FULL_FEATURE_CROSSWALK.md) found 6,791 unique
matches among all 9,408 rows; neither result changes the coordinate-only
counts below. The versioned output at
`/home/mechti/PycharmProjects/DeepMzyme_pmm_source_audit_output/pmm_deposition_crosswalk_v3/`
contains audit records, **not training examples**. It does not populate a
DeepMzyme dataset, and neither PMM side has been evaluated here.

## Source and identifier evidence

The source cohort, row UID, side, raw and `dropna` row counts, source hashes,
and label-code mapping are described in the [source audit](PMM_SOURCE_AUDIT.md).
PMM's [classification SQL](https://github.com/hhz-lab/PinMyMetal/blob/59ef46795920322c798db4e5ec500b04451f7904/metal_prediction/script/hybrid_algorithm/script10_clasmodel_result.sql)
maps `1=Mn`, `2=Fe+Co+Ni`, `6=Cu`, and `7=Zn`. The class-model rows contain
`metalid`, but the published Figure 2 notebook and released training script
drop it before modeling; neither establishes a `metalid`→PDB ion identity.

The pinned [PMM preprocessing SQL](https://github.com/hhz-lab/PinMyMetal/blob/59ef46795920322c798db4e5ec500b04451f7904/metal_prediction/script/script1_predict.sql)
shows the missing authoritative link: it joins
`ion_bindingsites.residueid_ion` to `neighborhood.residues.residueid` using
`pdbfileid` for `chainid,resseq`, then joins `neighborhood.atoms` on
`pdbfileid,residueid_ion,atomid_ion` for `x_ion,y_ion,z_ion`.
[`get_ionbinding_data.py`](https://github.com/hhz-lab/PinMyMetal/blob/59ef46795920322c798db4e5ec500b04451f7904/metal_prediction/script/hybrid_algorithm/get_ionbinding_data.py)
also exports a direct `chainid_ion,resseq_ion,pdbfileid,residueid` relationship
from runtime `RESIDUE.data` and `ION_BINDINGSITE.data` for its `_ZN` candidate
path. That path documents identifier semantics; it is not a source-cohort
crosswalk. These database
tables/files and their direct-key export are absent from the released class
files and the inspected Figshare v1 archive. The archive contains no confirmed
crosswalk for `metalid` either. Numeric proximity, source order, ligand
proximity, and nearest-ion distance are not substitutes for this link.
The archive inventory was reviewed; 77 text/code/notebook members totaling
13.9 MB were scanned for combined internal-ID and chain/coordinate fields.
None contained them. The one larger text member (11.9 MB) is a chain-level
experimental-method summary by its header, with no internal IDs.

## Deposition retrieval and checked files

The [Figshare v1 archive](https://doi.org/10.6084/m9.figshare.25011212.v1)
was retrieved with the documented [public file-download API route](https://www.niddk.nih.gov/-/media/Files/News/Meetings/DMS-4-Figshare-R-in-FAIR-v2_508.pdf)
on [Figshare API v2](https://docs.figshare.com/v2/), using
`https://api.figshare.com/v2/file/download/52224257` after the advertised
`ndownloader` URL returned HTTP 403. Its size is 99,424,422 bytes; its MD5
`dcb73c3520dcb0c1df07b5cd9126f147` matches Figshare's metadata and its
SHA-256 is
`e8efd7aa463f217c0ada67a538c3b4eb20f83927936d73cc9be2a1822696dfb8`.
The entire archive remains under the isolated output's `upstream/` directory;
only relevant members were extracted. The archive inventory contains 272 ZIP
entries and is preserved alongside it.

| Deposited archive member | SHA-256 | Useful content |
| --- | --- | --- |
| `Figure2/classmodel_train_set` | `4748babd2b6ac0706cd9ed4bcfd4855c8d2c5535f01813fb2e10b68b31a24c0f` | Byte-identical source train side |
| `Figure2/classmodel_test_set` | `ec6427ad0b8a18261dbc1d6822bdcfe7194730c7da4a36c208957bcb00f6a0f2` | Byte-identical source test side |
| `Figure2/Figure2.ipynb` | `23b0a6a2db1bde5670d8ed8831b4159fd39e7d6485e0672328df267edf7085e5` | Figure 2 analysis; `StratifiedKFold(n_splits=5)` on source order, default `shuffle=False` |
| `Figure_S9/transition_metal_type.csv` | `67a75598d248e099e2f7b245592aa821971801d67955b65b1953501211c6ccf2` | 5,665 `(pdbid,residueid_ion)` metal atomic numbers and ligand fingerprints; no coordinates |
| `Figure_S1_Abstract/exp_pre_sites_CH.txt` | `48cf05bee5254de85bd7cc67033a448593ec6ecfcdb44d466514ea6a83f9743c` | Experimental ion chain/residue/coordinates; no internal ID |
| `Figure_S1_Abstract/exp_pre_sites_EDH.txt` | `29aae32d959ae5bd0c016d2bfd32189e878a4eb0a328ed3abd330ce4f88cb041` | Same coordinate fields; no internal ID |

The source train/test tables have 7,920/1,488 raw rows and the same counts
after `dropna()`. The deposited Figure 2 notebook omits the nonexistent
`source` column from its feature-drop list, unlike the GitHub training
script. This does not establish the exact effective fitted cohort or saved
fold membership of the published model. There is **no serialized published
validation-fold ID manifest** in the inspected class files or archive; the
notebook's fold algorithm could generate a candidate assignment from the
verified row order, but those IDs have not been asserted as published IDs.

## Review rule and output

`src/audit_pmm_deposition_crosswalk.py` preserves every source UID and its
original side, resolves an element only by exact `(pdbid,residueid_ion)` in
the deposited metal-type table, and validates it against the class code.
The separate coordinate tables give candidate sites by `pdbid,element`.
An exact match requires **one deposited candidate coordinate, one ion of that
element in a reviewed complete PDB entry, and exact agreement in chain,
residue number, and all three coordinates**. This is a conservative sufficient
condition, not a claim that all unresolved rows are wrong. No distance-based
assignment is made. The output includes `deposition_crosswalk.csv`, one CSV
per status, and `coverage.json`; candidate coordinate references and full PDB
checksums are retained per reviewed row. Full current RCSB PDB files for 22
targeted entries were downloaded into the isolated output and reviewed at
lightweight scale. All selected entries had multiple ions of the relevant
element; none satisfied the sufficient condition. No exhaustive PDB download
or local structure-store sweep was performed while other PC work was active.

The deposited type table gives an element for **4,966** source rows. Of those,
**4,951** have at least one deposited coordinate candidate. Neither number
is matched coverage. A candidate is `ambiguous` when multiple deposited
coordinates or multiple same-element ions in a reviewed full structure leave
the PMM internal key unresolved. A record is `incomplete` when no type or
coordinate is deposited, or a complete structure has not been reviewed.
`unmatched` is reserved for positive contradictory evidence. The present
exact matched coverage **under this deposition-coordinate-only rule** is
**0/7,920 train and 0/1,488 test**, with no training-ready rows. Missing
runtime join data, not a choice of matching radius, is the blocker for that
rule. The later bounded feature check is reported separately.

## Coverage

| Original side / metal class | Raw = `dropna` records | Exact | Ambiguous | Unmatched | Incomplete |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train Mn | 2,590 | 0 | 247 | 0 | 2,343 |
| Train Class VIII | 2,629 | 0 | 333 | 0 | 2,296 |
| Train Cu | 400 | 0 | 115 | 0 | 285 |
| Train Zn | 2,301 | 0 | 1,665 | 0 | 636 |
| Test Mn | 167 | 0 | 103 | 0 | 64 |
| Test Class VIII | 252 | 0 | 154 | 0 | 98 |
| Test Cu | 64 | 0 | 51 | 0 | 13 |
| Test Zn | 1,005 | 0 | 741 | 0 | 264 |
| **Total** | **9,408** | **0** | **3,409** | **0** | **5,999** |

There are 668 shared PDB IDs across the original sides; source UIDs do not
cross sides. Since no ion has been certified, structural train/test site
overlap cannot be concluded or measured. A shared PDB ID does not itself
prove that the same ion occurs on both sides.

## Case reviews

`1a0e`: source **train row 4,547** is
`(residueid_ion=887,metalid=4637,label=Class VIII)`; source **test row 1**
is `(886,4638,Class VIII)`. The deposited type table calls both Co, with
`H1ED1` and `ED4` ligand fingerprints. The deposited coordinate table names
Co **A491** `(55.423,-11.530,10.398)` and **A492**
`(54.251,-8.229,8.167)`. The [complete PDB entry](https://www.rcsb.org/structure/1A0E)
contains those two, A493, and three chain-D Co ions; its downloaded PDB
SHA-256 is
`c0476fba37b4537a67c9df8a35ea61e1eb2f61ef42c8fc6f784fc1dd44a173df`.
The ligand patterns suggest possible pairings, but do not supply the missing
internal-key-to-chain link or rule out symmetry-related alternatives.
**Both records remain ambiguous; neither coordinate is assigned to a side.**

`4d8f`: source **train row 5,698** is
`(residueid_ion=1316,metalid=3761,label=Mn)`. The deposited type table calls
it Mn (`H1ED2`); the coordinate table documents Mn **A402**
`(3.641,19.721,-10.562)`. The [complete PDB entry](https://www.rcsb.org/structure/4D8F)
also has Mn B401, C402, and D402, alongside Fe ions. The downloaded PDB
SHA-256 is
`d1cf606d59e722cf359736aea320a2ef32aeace4eb2e0cbb2fb798dfab5c8cff`.
The deposited coordinate confirms that A402 is a real site, but does not
prove `residueid_ion=1316` is A402. **The source record remains ambiguous
under this coordinate-only rule.** The later
[bounded feature check](PMM_FEATURE_MATCH_FEASIBILITY.md) supplied independent
source-feature evidence that uniquely identifies A402.

The unresolved work is to obtain a pinned PMM/NEIGHBORHOOD
`(pdbfileid,residueid_ion,chainid,resseq,atomid_ion,x,y,z)` export (or
the underlying `residues`/`atoms`/`ion_bindingsites` tables) and verify its
provenance against the published train/test tables. A distinct feature-based
route is evaluated in the later bounded check; neither route has produced a
training-ready cohort. Existing datasets,
runs, and shared environments remain unchanged.

## Targeted provenance recovery, 2026-09-23

An additional read-only check of the official [Zenodo V1.0 software
archive](https://zenodo.org/records/14830978) found the same source commit
(`59ef467`) and byte-identical train/test class files. Its single
30,470,634-byte ZIP has published MD5 `d384b78605fd4eb16453b69dc1df3841`
and locally checked SHA-256
`552c677d34d1a867c2025ce8b0d7a2c5d3786e2d10c2ac4fa386f35cc0b1b557`.
Its 186-entry inventory has schema SQL and the NEIGHBORHOOD executable, but
no populated `RESIDUE.data`, `ATOM.data`, `ION_BINDINGSITE.data`,
`copyNeighborhoodData.sql`, database dump, or original per-row export. A
targeted scan of its 59 text/code members found `metalid` only in model
training scripts, where it is dropped from inputs; no source-ID assignment
or row-to-ion export is present. The [GitHub V1.0
release](https://github.com/hhz-lab/PinMyMetal/releases/tag/V1.0) points to
that commit and has no separate release assets. The Figshare article API
lists only version 1.

The released [`script.sh`](https://github.com/hhz-lab/PinMyMetal/blob/59ef46795920322c798db4e5ec500b04451f7904/metal_prediction/script/script.sh)
invokes NEIGHBORHOOD separately on each input PDB, and
[`import_neighborhood.py`](https://github.com/hhz-lab/PinMyMetal/blob/59ef46795920322c798db4e5ec500b04451f7904/metal_prediction/script/import_neighborhood.py)
loads the generated `copyNeighborhoodData.sql`. The workflow later drops the
per-PDB PostgreSQL database. The source code describes how a fresh import
would run; it does not preserve the April 2024 PDB inputs, generated import
records, or the database state that assigned identifiers to the released
class-model rows. A fresh import cannot certify the original internal IDs.
No compatible public snapshot or direct export was found in the inspected
official releases and documentation.

At the time of this provenance search, **no original database-ID mapping was
found**. The later [complete feature audit](PMM_FULL_FEATURE_CROSSWALK.md)
established 6,791 unique matches under an inferential rule, but only 72.18%
coverage with severe site-type bias; it did not create a training dataset.
The coordinate-only crosswalk and its per-side/class counts above remain
unchanged. `very_exact-pmm_sets` has not been built or passed to Parts 2–3.
A short [request for the missing provenance](PMM_AUTHOR_PROVENANCE_REQUEST_DRAFT.md)
remains unsent.
