# Metal inference and feature reference

This page documents the implementation behind the [public overview](../../README.md).
The historical [release audit](../../docs/campaigns/phase2_unified_2026-10/release_audit/README.md)
owns ensemble comparisons, parity evidence and timing.

## Public checkpoint inference

Run these commands from the repository root after the locked CPU installation in
[Getting started](../../docs/GETTING_STARTED.md). The standalone candidate
`unified_p5_trunklr1e4_5fold_s42` averages the five public
`unified__P_five_class_trunklr1e4__fold{0..4}__seed42` terminal checkpoints.
It trains on five classes and sums Fe and Co+Ni probabilities into Class VIII
before averaging common-four scores. It is an available research configuration;
no deployment default has been chosen.

Download and verify the public archive, then extract only its training-run subtree.
The archive also contains historical test outputs; they are not inputs to binding
or prediction. The [dataset inventory](../../docs/DATASETS.md#benchmark-runs-and-checkpoint-releases)
records the archive provenance.

```bash
mkdir -p DeepMzyme_Data/public_metal
curl --fail --location \
  https://huggingface.co/datasets/GMBioinformatics/DeepMzyme/resolve/main/benchmarks/DeepMzyme_phase2_unified_and_180ckpt_eval_20261019.tar.gz \
  --output DeepMzyme_Data/public_metal/phase2.tar.gz
echo 'd4d045b5dd9de56a11504639961c3e30e6a4fda5e7e6e60f5c63a563af8951fe  DeepMzyme_Data/public_metal/phase2.tar.gz' | sha256sum --check -
tar -xzf DeepMzyme_Data/public_metal/phase2.tar.gz \
  -C DeepMzyme_Data/public_metal \
  phase2_unified_and_180ckpt_eval_20261019T0317Z/runs

uv run --frozen --no-sync python src/metal_release_cli.py bind \
  --candidate unified_p5_trunklr1e4_5fold_s42 \
  --run-root DeepMzyme_Data/public_metal/phase2_unified_and_180ckpt_eval_20261019T0317Z/runs \
  --fold-membership docs/campaigns/phase2_unified_2026-10/reproduction/fold_membership.csv \
  --output DeepMzyme_Data/public_metal/bound_manifest.json
```

Binding writes the manifest and a sibling `.inventory.json` report. Successful
binding verifies the five members' completed-fit receipts, terminal checkpoint
hashes, run configurations, validation predictions and fold-file
identity. Missing or inconsistent required artifacts block the candidate.
Unrelated catalog members are outside this explicit scope; requesting them from
this manifest fails. Omitting `bind --candidate` retains the developer workflow
that requires the complete local catalog. Existing output files are never overwritten.

For a structure and site of your own, install the ESM group and replace the
example structure path and atom locator below. The locator fields are
`CHAIN:RESSEQ:ICODE:ATOM[:ALTLOC]`; `_` denotes a blank field. Alternatively,
replace `--atom` with `--center X Y Z` in the structure's coordinate frame.

```bash
uv sync --frozen --group esm
uv run --frozen --no-sync python src/metal_release_cli.py predict \
  --manifest DeepMzyme_Data/public_metal/bound_manifest.json \
  --candidate unified_p5_trunklr1e4_5fold_s42 \
  --structure path/to/holo_structure.pdb \
  --atom A:402:_:ZN:_ \
  --embedding-cache ~/.cache/deepmzyme/inference_esmc600m \
  --generate-embeddings \
  --device cpu \
  --output DeepMzyme_Data/public_metal/site_prediction.json
```

The first prediction downloads ESM-C weights if absent and computes full-chain
embeddings; later calls reuse matching cached embeddings. The inference cache
requires `bfloat16` model-dtype metadata that the training helper's cache does
not supply. Keep these caches separate. Per-site outputs under `sites` include `predicted_class`,
`class_order: ["Mn", "Cu", "Zn", "Class VIII"]`, `class_scores`,
`calibration: "none"`, and `quality_flags`.

Binding checks artifact identity; it does not establish prediction parity or
scientific validity on a new environment or structure. The historical audit's
parity measurements retain their recorded source/environment scope. Predictions
assume a metal-binding site and are not calibrated binding probabilities.

## Input features

The benchmark and public candidate use an 8 Å ion-centered residue neighborhood.
This is a saved recipe choice; the general training parser's default radius is
separate. Feature dimensions below describe constructed inputs before learned
projection and distance RBF expansion. Configured omissions zero their channels.

| Input | Constructed representation | Implementation |
|---|---|---|
| Node chemistry | 20 amino-acid indicators, five donor/acceptor/aromatic/acidic/basic flags, one Kyte–Doolittle hydrophobicity value | [`featurization.py`](../featurization.py) |
| Node coordination | Two shell indicators, three nearest-metal distances (Cα, functional-group centroid, minimum donor), one ligand-orientation cosine | [`featurization.py`](../featurization.py) |
| Node vectors | Two raw 3D displacements: Cβ→functional-group centroid (Cα fallback if Cβ is absent) and nearest metal→Cα | [`featurization.py`](../featurization.py) |
| External node channels | SASA, custom charge-distance proxy and PROPKA ΔpKa; omitted in the public candidate | [`feature_extraction`](../feature_extraction/README.md) |
| Edge scalars | Closest inter-atom contact and Cα–Cα distances (RBF encoded), absolute residue-number separation, same-chain indicator | [`edge_geometry.py`](../graph/edge_geometry.py), [`model.py`](../model.py) |
| Edge vectors | One raw 3D displacement between the closest atoms of the directed residue pair | [`edge_geometry.py`](../graph/edge_geometry.py) |
| Sequence | Frozen ESM-C 600M full-coordinate-chain embeddings, 1,152 values/residue, indexed onto pocket residues and pooled by attention plus mean | [`model.py`](../model.py) |
| Site summary | Four values: multinuclear indicator, metal count, minimum and mean inter-metal distances | [`featurization.py`](../featurization.py) |

The public candidate uses radius-based edges and geometry-derived shell roles;
other training configurations can enable RING features. Each example here is
centered on one target ion. Site-summary fields do not establish that a model
was trained to predict an entire multinuclear pocket.
