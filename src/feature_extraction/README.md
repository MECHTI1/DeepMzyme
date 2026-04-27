# feature_extraction

This package replaces the old Bluues and Rosetta-derived residue features with
a lighter workflow that is easier to install and maintain.

Chosen tools:

- `biotite` for structure parsing, solvent-accessibility calculation, residue
  geometry, and contact-based electrostatic proxies.
- `propka` for an optional titration/electrostatic contribution to `dpka_titr`.

The output now keeps only the DeepMzyme external feature names that are still
consumed by the model:

- `biotite_residue_sasa`
- `custom_charge_distance_proxy`
- `dpka_titr`

Important note:

`custom_charge_distance_proxy` is a heuristic q1*q2/r-style charge-distance
feature, while `dpka_titr` is parsed from PROPKA. They are kept separate
because they are not assumed to share a common physical scale. These values are
compatible replacements, not byte-for-byte recreations of Rosetta/Bluues
output. The package now emits only the reduced external-feature contract used
by the current model.

Default output location:

- `.data/updated_feature_extraction/<structure_id>/residue_features.json`

Example:

```bash
/home/mechti/miniconda3/envs/DeepMzyme/bin/python -m feature_extraction.generate_features \
  --structure-dir /media/Data/pinmymetal_sets/mahomes/train_set \
  --output-root /home/mechti/PycharmProjects/DeepMzyme/.data/updated_feature_extraction \
  --skip-existing
```

The training loader reads these JSON files directly. In the default `auto`
mode, DeepMzyme now prefers the updated external-feature directory rather than
legacy Bluues/Rosetta-style sidecar files.
