# CARE Task 1 30% AlphaFill-MAHOMES Metalloenzyme Subset

This is a CARE-derived AlphaFill-MAHOMES catalytic metalloenzyme subset for DeepMzyme.
It is not the full CARE benchmark.

AlphaFill transferred metals are computational hypotheses.
MAHOMES catalytic filtering is computational catalytic-site evidence.
These structures should not be called experimentally validated metalloenzymes without independent evidence.

Do not tune or select models on the exported CARE test split. Use only the exported CARE train split for internal train/validation splitting, HPO, seed repeats, and model selection.

## Contents

- `train/`: 817 structures, 1520 catalytic site rows
- `test/`: 34 structures, 76 catalytic site rows

See `split_metadata.json` and `metadata/` for source evidence.
