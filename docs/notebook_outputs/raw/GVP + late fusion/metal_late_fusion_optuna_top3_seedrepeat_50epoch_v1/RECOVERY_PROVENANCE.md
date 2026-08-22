# Round-4 Late-Fusion Artifact Recovery

This directory restores the 75 lightweight JSON artifacts for the 15-run
validation-only late-fusion confirmation batch:

- source trials `49`, `32`, and `15`;
- seeds `42`, `123`, `2026`, `43`, and `44`;
- 50 epochs per run;
- no held-out-test evaluation.

The artifacts were recovered byte-for-byte from Git commit `783acae` during the
documentation/provenance cleanup. They had later been removed in commit
`20b4d64`, leaving summaries that pointed to an absent directory.

`SHA256SUMS` records every restored JSON file. Git blob comparison confirmed
that all 75 files match their historical blobs.

The 30 historical `best_model_checkpoint.pt` and `last_model_checkpoint.pt`
files were deliberately not restored because this is a lightweight
documentation cleanup. `MISSING_CHECKPOINT_GIT_BLOBS.tsv` records their
historical Git blob identities, sizes, and paths. Their absence does not change
the validation results, but checkpoint binaries are unavailable in the current
working tree.
