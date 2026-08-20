# Prefer CLEAN_30_main For New Runs

Use this path for new CLEAN-30 training/evaluation runs:

```text
DeepMzyme_Data/CLEAN_30_main
```

`CLEAN_30_main` points to:

```text
DeepMzyme_Data/CLEAN_30_shared_single_donor_supported_metal_conservative
```

This `CLEAN_30_shared` directory is the original multi-donor source/reference
layout. It is kept for provenance and comparison, but it is not the preferred
default for new conservative runs because fold CSVs can include multiple
AlphaFill donor PDB sources for the same UniProt accession.
