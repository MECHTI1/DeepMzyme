# Content-Addressed Structure Store

DeepMzyme can store distinct PDB/CIF/mmCIF byte contents under
`DeepMzyme_Data/structure_store/objects/`. Dataset and fold directories record
membership in `structure_manifest.csv` instead of keeping per-split copies.
This changes physical storage layout, not scientific train/test membership,
labels, or the meaning of a split. Deduplication does not remove scientific
overlap between splits.

## Layout and identity

```text
DeepMzyme_Data/
├── structure_store/
│   ├── objects/<sha256-prefix>/<sha256>/<original-filename>.pdb
│   ├── catalog.csv
│   ├── audit_report.json
│   └── migration_audit_before.json
└── <dataset>/<train-or-test>/
    ├── structure_manifest.csv
    └── <site-level-summary>.csv
```

SHA-256 identifies the complete file bytes. Same-named files with different
bytes occupy different hash directories. Original filenames remain intact
because downstream code derives structure, chain, and EC identifiers from them.
Identical bytes under different logical filenames are rejected rather than
silently changing those identifiers.

| Manifest column | Meaning |
|---|---|
| `structure_name` | Logical filename used by training and feature lookup |
| `relative_path` | Path from the manifest directory to the stored object |
| `sha256` | Expected SHA-256 of the complete structure bytes |
| `size_bytes` | Expected file size |

Manifests are authoritative: extra legacy files are not added to membership by
runtime resolution. Without a manifest, legacy PDB/CIF/mmCIF directory scanning
remains available for older datasets and bundles. Relative paths allow a dataset
and its store to move together. Absolute manifest paths, invalid logical names,
duplicate names, mismatched physical filenames, and missing targets are rejected.
Relative parent-directory components are needed to reference the shared store.
The maintenance `verify` command additionally requires targets inside the
configured store's `objects/` directory.

## Integrity verification

Every manifest read verifies target size and SHA-256 before returning structure
paths. This includes the training loader and manifest-backed path/index helpers.
Same-size corruption fails closed; an invalid manifest never triggers a fallback
scan of legacy files. The compatibility keyword `verify_hashes=False` cannot
disable verification on reads. Manifest write helpers retain their separate
optional hash-check setting; subsequent reads always verify integrity.

Object creation still prefers a hardlink, falling back to copying when linking
is unavailable. New and reused stored objects are verified. A hardlinked source
shares bytes with the object: editing it in place can corrupt the object.
Mandatory read verification detects that corruption rather than silently using
it under the old hash. Treat sources and objects as immutable while resolving
and consuming them; verification is not a lock against concurrent writers.
Repeated manifest reads incur hashing I/O; hashes are not cached across reads.

## Migration and retries

Migration inventories logical membership and validates every legacy cleanup
candidate against the expected size and SHA-256 before writing objects,
manifests, or reports. A same-named file with different bytes raises a clear
conflict error and is preserved. Files absent from an existing manifest and
unreadable legacy entries, including broken symlinks, also block cleanup.

For valid inputs, migration creates/reuses objects, writes manifests, verifies
their membership and stored content, and rechecks all cleanup candidates before
unlinking the validated legacy entries. It does not discover additional files
for deletion after that validation. Run migration with exclusive access to the
input directories; it is not a concurrent-writer transaction.

Already-correct layouts can be migrated repeatedly without changing membership
or structure bytes. Matching readable leftovers can be cleaned on retry. A
retry with conflicting bytes or an unreadable leftover fails closed and requires
manual investigation; safe recovery is not claimed for every interrupted state.
Audit reports can be refreshed on successful runs.

## Commands

Use the project's configured Python interpreter from the repository root. The
examples below use the local `.venv`; substitute the configured environment's
Python executable if it differs.

```bash
# Read logical references and duplication; no data writes without --report.
.venv/bin/python src/manage_structure_store.py audit --data-root DeepMzyme_Data

# Preview only; --apply is required for migration writes and legacy cleanup.
.venv/bin/python src/manage_structure_store.py migrate --data-root DeepMzyme_Data
.venv/bin/python src/manage_structure_store.py migrate --data-root DeepMzyme_Data --apply

# Verify manifests, content identities, store containment, and absence of
# legacy entries; on success this refreshes the store's audit_report.json.
.venv/bin/python src/manage_structure_store.py verify --data-root DeepMzyme_Data
```

These commands inspect structure data and membership, potentially including
held-out membership. Use them only within the authorized dataset scope; they
are not training or held-out performance evaluation commands. For a different
data root, specify its contained `--store-root` as well.

`src/build_colab_bundle.py` includes objects referenced by bundled manifests.
For direct `tar` packaging, include the referenced store objects alongside the
dataset roots while preserving relative layout. Use a new bundle identity and
checksum when changing layout.

## Validation scope

The accompanying tests use temporary synthetic structures. They cover manifest
resolution, integrity failures, migration conflict preservation, matching retry,
idempotence, symlink cleanup, relocation, and bundle dependency discovery.
The v11 bundle publication additionally verified every archived file hash and
all relocated structure-manifest targets, then exercised the notebook's CLEAN
fold materialization. See [DATASETS.md](DATASETS.md#main-colab-bundle-v11-historical-hosted-release)
for that release's validation scope and cache gaps. This publication did not
rerun migration, train a model, or evaluate held-out performance.
