# Common-PDBID 70/30 Split PinMyMetal

Generated from `train_and_test_sets_structures_exact_pinmymetal`.

Train-only PDB IDs stay in train, test-only PDB IDs stay in test, and PDB IDs that appear in both exact train and exact test are assigned as whole PDB-ID groups: 70% of common PDB IDs to train and 30% to test.

This split is a custom comparison split, not the current trusted final held-out
split defined in `Plan.md`.

Seed: `42`
Assignment scope: `common_exact_split_pdbids`
Test common-PDB-ID fraction: `0.3`
Final PDB IDs assigned to train: `1419`
Final PDB IDs assigned to test: `189`
Common exact-split PDB IDs assigned to train: `124`
Common exact-split PDB IDs assigned to test: `53`
Final train/test PDB-ID overlap: `0`
