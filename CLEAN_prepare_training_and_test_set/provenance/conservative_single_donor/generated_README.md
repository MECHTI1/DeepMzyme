# CLEAN 30 Single-Donor Supported-Metal Conservative Layout

This dataset was derived from `/home/mechti/PycharmProjects/DeepMzyme/DeepMzyme_Data/CLEAN_30_shared` by selecting one AlphaFill donor
structure (`alphafill_pdb_id`) per UniProt target within each CLEAN fold split,
then keeping only that donor's site rows after close-metal deduplication.

The original `CLEAN_30_shared` directory is not modified.

For new CLEAN-30 training/evaluation runs, prefer the stable alias
`DeepMzyme_Data/CLEAN_30_main`, which points to this conservative dataset.
Keep `CLEAN_30_shared` as the original multi-donor reference/source dataset.

## Selection Rule

1. Group rows by target protein (`uniprot_id`).
2. Group each target's rows by `alphafill_pdb_id`.
3. Deduplicate close metal rows within each donor group at
   `2.0` Angstrom using coordinates from the
   shared PDB files.
4. If exact stoichiometry is supplied through `--stoichiometry-csv`, prefer
   donor groups whose deduplicated metal counts match it.
5. If exact counts are unavailable but UniProt-supported metal identities are
   present, choose the best-quality single donor group and mark rows as
   `metal_supported_but_count_unknown`.
6. If neither exact counts nor supported-metal identities are available, choose
   the best-quality donor group and mark rows as `no_clear_supported_metal`
   unless that exclusion option was enabled.

Quality tie-breakers are donor PDB resolution, AlphaFill identity, alignment
length, binding-site/local RMSD fields when present, local RMSD, PAE, and donor
ID for deterministic ordering.

## Contents

- `structures/`: selected shared structure files.
- `folds/`: conservative site-level fold CSVs with the original CLEAN fold file
  names and additional audit columns.
- `metadata/structure_sources.csv`: source path and link/copy status for each
  structure file.
- `metadata/single_donor_selection_audit.csv`: one decision row per target per
  fold split.
- `split_metadata.json`: generation settings and per-fold source/output counts.

## Supported-Metal / Stoichiometry Caveat

The source CLEAN CSVs used here contain `uniprot_supported_transition_metals`,
not exact UniProt metal counts. That is why this dataset name uses
`supported_metal` rather than `stoich`.

Unless a `--stoichiometry-csv` is supplied, this builder does not invent exact
stoichiometry. It records the count status in `stoichiometry_status` and
`stoichiometry_mismatch`.
