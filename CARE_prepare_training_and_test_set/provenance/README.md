# Tracked CARE clusterRes30 Provenance

`clusterRes30/` is a byte-for-byte copy of lightweight generated metadata from
the Git-ignored
`DeepMzyme_Data/CARE_task1_30_clusterRes30_train_test_metallo/` root.
Structures and runtime features remain outside Git.

The snapshot preserves:

- split metadata and build parameters;
- the exact CARE Task 1 audit JSON/CSV;
- train/test protein and pair membership tables;
- fetch and build summaries;
- MAHOMES candidate, prediction, and summary records.

The generated preparation produced 817 train structures with 1,520 catalytic
sites and 34 test structures with 76 sites. The selected source rows were
10,321 train and 432 test rows; these are distinct from the larger 184,529-row
full-train input. The AlphaFill/MAHOMES preparation used identity threshold
`0.30`, minimum alignment length `85`, `1.0 Å` site deduplication, and
supported-metal filtering.

No repository/source URL or formal citation for the upstream CARE files was
found during the audit; this remains a provenance gap.

See `SHA256SUMS` for copied-file hashes and
[`docs/DATASETS.md`](../../docs/DATASETS.md) for the authoritative dataset
interpretation.
