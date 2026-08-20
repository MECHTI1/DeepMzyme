# CLEAN Shared Fold Layout

This compact CLEAN layout stores structures once and fold membership as CSV files.
The regular DeepMzyme training code still expects `train/` and `test/` directories; the Colab notebook materializes those views at runtime for the selected fold.

## Which CLEAN Dataset To Use

For new CLEAN-30 runs, use:

```text
DeepMzyme_Data/CLEAN_30_main
```

`CLEAN_30_main` is a stable alias for the conservative single-donor dataset:

```text
DeepMzyme_Data/CLEAN_30_shared_single_donor_supported_metal_conservative
```

This original `CLEAN_30_shared` directory is kept as the source/reference
multi-donor CLEAN layout. It can contain multiple AlphaFill donor PDB sources
for the same UniProt accession in the fold CSVs. That is useful for provenance
and comparison, but it is not the preferred default for new conservative
training runs.

The conservative dataset selects one `alphafill_pdb_id` donor per UniProt
accession within each fold split, keeps that donor's selected metal-site rows,
and records audit columns. It uses UniProt-supported metal identities from the
current CSVs; exact UniProt metal stoichiometry counts were not available in
these inputs.

## Contents

- `structures/`: 740 unique PDB structures
- `folds/`: site-level train/test CSVs for each CLEAN fold
- `metadata/structure_sources.csv`: source path for each shared structure
