# Tracked CLEAN Dataset Provenance

These files are byte-for-byte copies of lightweight generated metadata from
Git-ignored `DeepMzyme_Data` dataset roots. They make current dataset identity
and fold construction recoverable without moving or tracking structures.

## Sources

- `original_shared/`: copied from `DeepMzyme_Data/CLEAN_30_shared/`.
- `conservative_single_donor/`: copied from
  `DeepMzyme_Data/CLEAN_30_shared_single_donor_supported_metal_conservative/`.

The local `DeepMzyme_Data/CLEAN_30_main` symlink pointed to the conservative
single-donor root at the time of this snapshot.

The original shared dataset contains 740 structures and preserves the
multi-donor reference. The conservative derivative selects one AlphaFill donor
per UniProt target per fold, applies deterministic quality tie-breakers,
retains supported metals, and performs within-donor deduplication at `2.0 Å`.
Exact metal stoichiometry is unavailable for the selected CLEAN targets.

Conservative site counts by fold are:

| Fold | Train source sites | Train retained sites | Test source sites | Test retained sites |
|---:|---:|---:|---:|---:|
| 0 | 1102 | 743 | 229 | 139 |
| 1 | 1063 | 698 | 179 | 128 |
| 2 | 1024 | 668 | 220 | 148 |
| 3 | 1034 | 696 | 208 | 119 |
| 4 | 1101 | 723 | 187 | 124 |

See `SHA256SUMS` for the copied-file hashes and
[`docs/DATASETS.md`](../../docs/DATASETS.md) for the authoritative dataset
interpretation.
