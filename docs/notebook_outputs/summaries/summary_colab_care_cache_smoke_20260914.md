# CARE cache completion and Colab GPU smoke — 2026-09-14

CARE cache preparation is complete locally. All three standalone EC1 notebook
smoke runs passed on an actual Colab G4 GPU. This is **Grade 6 execution
evidence**, not a baseline comparison, model promotion, or held-out result.
The cache/GPU preparation phase ended without a repository upload. A later
Hugging Face publication is recorded below; no GitHub push was performed.
The session was stopped and `colab sessions` confirmed no active runtimes.

## Cache completion

| CARE split | Structures | Missing ESM | Missing external | Missing RING |
|---|---:|---:|---:|---:|
| Train | 817 | 0 | 0 | 0 |
| Test | 34 | 0 | 0 | 0 |

- Generated 586 exact-name ESMC-300m caches and their metadata sidecars.
  The earlier count of 585 missing structures treated
  `Q8G7X1__chain_A__EC_2.5.1.3,4.1.99.17_chain_A_esmc.pt` as a cache for the
  shorter `Q8G7X1__chain_A__EC_2.5.1.3` identity. The repaired data also contains
  the exact-name cache; the older cache remains available for its own identity.
- Generated 586 previously absent external-feature files and 586 RING outputs.
- Regenerated all 265 older CARE external files: their metadata explicitly
  recorded `pka: unavailable` because the original interpreter lacked PROPKA.
  All 851 CARE files now record successful PROPKA processing. The scientific
  missing-value masks remain; this does not assert that every residue has an
  experimentally observed pKa.
- Audited all 851 structures, including all embedding dimensions, finite
  tensors, sequence hashes, residue alignment, external schemas/masks and
  finite values, RING syntax, and the actual ESM filename resolver.
- Downloaded and verified 3,195 cache artifacts. A fresh local audit reproduced
  zero missing caches and zero validation failures. Structures, labels, and
  train/test membership were not changed. Test-side work was feature
  extraction and integrity checking only.

The repaired external files are shared by other dataset views using those
same structure identities. Those views receive the repaired features too;
historical outputs remain evidence of their original feature state.

The initial audit also reported 586 schema mismatches because the new checker
omitted the existing `*_missing` columns. The checker was corrected to the
project schema; those columns were retained, and the final audits passed.

Provenance: [generation runtime](../raw/colab_care_cache_smoke_20260914/generation_runtime.json),
[initial audit](../raw/colab_care_cache_smoke_20260914/audit_initial.json),
[external repair](../raw/colab_care_cache_smoke_20260914/external_repair.json),
[final remote loader audit](../raw/colab_care_cache_smoke_20260914/audit_runtime.json),
[local audit](../raw/colab_care_cache_smoke_20260914/audit_local.json),
[delta manifest](../raw/colab_care_cache_smoke_20260914/delta_manifest.json).

## GPU execution

The final runs used stock Colab Python **3.13.15**, PyTorch **2.11.0+cu128**,
CUDA **12.8**, and NVIDIA **RTX PRO 6000 Blackwell Server Edition**, compute
capability **12.0**. The overlay preserved stock PyTorch. ESM generation used
an isolated Python **3.12.14** environment because `esm==3.2.3` excludes
Python 3.13. It used the legacy ESMC-300m weights pinned to Hugging Face revision
`7f10b20ae75017b2dbc884070e03434515709a8d` in
`biohub/esmc-300m-2024-12`; sequence and checkpoint hashes are retained.

`src/verify_colab_notebook_smoke.py` reads the current notebook's main
configuration, central CONFIG, and command-planning cells, applying the exact
first standalone smoke block from the EC playbook. It then executes the
generated command in a fresh process. Only runtime paths, output transfer,
prepared-data identity, and Python executable differ from browser/Drive
plumbing. The browser UI and Drive mount were not part of this verification.

All three runs used one epoch, seed/split seed 42, batch size 4, EC depth 1,
protein grouping, and the playbook's 15% **pocket** validation target. Actual
retained membership was **993 training pockets / 175 validation pockets**, from
**751 training proteins / 42 validation proteins**. The ordered membership
hashes match across all families, all seven EC1 classes are present on both
sides, and protein/structure/pocket overlap is zero. The single-label parser's
existing ambiguous-EC exclusion policy was retained.

| Family | Train loss | Validation loss | EC1 group balanced accuracy | Result |
|---|---:|---:|---:|---|
| Only-GVP | 0.405554 | 0.472342 | 0.166667 | Passed |
| Only-ESM | 0.401617 | 0.465846 | 0.203571 | Passed |
| GVP + graph-level late fusion | 0.405692 | 0.478507 | 0.269048 | Passed |

These one-epoch values are plumbing checks and must not rank families. Minimum
group recall is zero in each run; no performance or rare-class promotion gate
has passed. Metal, joint, HPO, grouped-fold confirmation, refit, and final-test
runs were not performed by this task.

Every run has finite losses, full loaded ESM/external residue coverage,
config/runtime/source hashes, split diagnostics, epoch/train/validation CSVs,
and best/last checkpoints. `run_test_eval=False`, both test input paths are
null, and no `test_report.json` exists.

The first Python 3.12 attempt completed Only-GVP, then stopped at Only-ESM on a
duplicate-residue cache-alias error. The loader fix prefers exact cache names
and retains the historical alias fallback only when no exact cache exists.
All three final stock-runtime runs were restarted with that fix. Their data
identity explicitly records `v11+care-complete-20260914:care-audit-manifest`;
they must not be attributed to the unmodified hosted v11 archive. Source
baseline is `d75219e` plus the recorded local patch. The later local Git changes
through `89f0771` changed Git policy only, not the training source.

Evidence: [artifact and cohort checks](../raw/colab_care_cache_smoke_20260914/smoke_validation.json),
[complete portable run evidence](../raw/colab_care_cache_smoke_20260914/care_smoke_stock/).
Checkpoint binaries and the complete initial/final run archive remain under
`DeepMzyme_Data/notebook_outputs/colab_care_cache_smoke_20260914/` and are not
Git payloads. Full run-archive SHA256:
`ad16c1afafd9c0dbd6371c242241edb3c602784670b38e032f010dd676b7dca8`.

## Release and reproduction

[DATASETS.md](../../DATASETS.md) owns the v12 filename, checksum, and publication
state. The complete v12 archive and its three sidecars were subsequently
published on Hugging Face; the [publication receipt](../raw/colab_care_cache_smoke_20260914/v12_publication.json)
records remote checksum/size and public-download verification. The local
notebook selects the pinned release, with optional ESM generation disabled.
The original v11 archive remains unchanged and incomplete for CARE. Immutable
build receipts retain their pre-publication state. Archive, embedding, and
checkpoint binaries remain outside Git; GitHub changes await the user's push.

Run the cache utility with `--phase inventory`, then independent `external`,
`ring`, and `esm` phases, and finally `audit`. Use `--model-revision` for the
immutable weight revision, `--source-commit` for the actual source identity,
and a separate `--report` per phase. The external phase accepts
`--repair-audit` to regenerate pre-existing files that an audit identifies as
having unavailable PROPKA. [The captured setup](../raw/colab_care_cache_smoke_20260914/setup.sh)
records the environment and pinned base-bundle download used here.

For GPU reproduction, invoke `src/verify_colab_notebook_smoke.py` once per
family with the data root, output directory, source identity, prepared-data
identity/checksum, and `--execute`. It fails if a command references held-out
test inputs. Exact scientific values remain owned by the
[EC playbook](../../EC_TRAINING_PIPELINE_PLAYBOOK.md), not this evidence note.

`src/rebuild_care_complete_bundle.py` builds the local gzip successor from the
v11 manifest and completed local CARE audit. It verifies audited cache hashes,
preserves base structure/membership bytes, records repaired external files,
and checks all archive member hashes and symlinks. It performs no upload.

Local regression validation: **40 tests passed** across cache integrity,
exact-versus-alias lookup, standalone notebook expansion/split safeguards, and
gzip bundle handling.
