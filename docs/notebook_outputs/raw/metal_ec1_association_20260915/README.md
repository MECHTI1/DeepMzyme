# Portable training-only metal × EC1 association evidence

Read the [summary](../../summaries/summary_metal_ec1_association_20260915.md)
first. Exact execution rules are in the
[EC playbook](../../../EC_TRAINING_PIPELINE_PLAYBOOK.md#phase-3--training-only-metal--ec1-association).
The runner is [analyze_metal_ec_association.py](../../../../src/analyze_metal_ec_association.py).

These are exact copies from
`DeepMzyme_Data/notebook_outputs/analyses/metal_ec1_development_association_v1_20260915/`.
`portable_manifest.tsv` records the SHA-256 and byte size of every copied file;
this README is an authored navigation document.

| File | Role |
|---|---|
| `analysis_manifest.json`, `analysis_manifest.sha256` | Predeclared source identities, separate panels, training partition, target views, weighting/statistic/permutation rules, and frozen pair/audit hashes |
| `retained_training_pairs.csv` | Exact retained training-pocket labels, native metal symbols, grouping and matched catalytic-site keys; includes explicitly excluded missing/ambiguous targets |
| `eligibility_audit.json` | Saved-source target equality, structure content hashes, internal-validation group exclusion, and preparation counts |
| `association_results.json` | Native-six/common-four measures and exclusion counts, expected-cell checks and exploratory group-permutation results |
| `contingencies_and_conditionals.csv` | Complete raw-count and protein-group-weighted tables; both conditional probabilities; matched native-six-eligible common-four sensitivity |
| `execution_receipt.json` | Completed analysis output hashes, no training/test input, and explicit absence of cross-source identity certification |
| `local_verification.json` | Focused test result and local interpreter/library versions |

The two source populations are never pooled. The runner opens no external
test files; it inherits external-training membership and excludes saved
internal-validation groups. This is descriptive development evidence, not
model-quality evidence or a certificate for shared training. The dataset
authority separately owns historical and incidental test-metadata access.

The ignored `_initial` local output preceded an additional exclusion diagnostic
and a stricter group-label guard. It is not copied here. The final preparation
freezes the complete current analysis source before execution; the retained
pair and contingency files are checked against that initial computation for
exact equality. No model checkpoints, structures, feature caches, or archives
are included in this portable batch.
