# Reproducibility Remediation Plan

Status: local implementation completed 2026-08-23; external/GPU actions remain
authorization-gated.

This document records the decisions and execution status for the reproducibility
audit. It does not change model architecture, scientific conclusions, dataset
splits, selection metrics, or Stage 6/6B/7 policy.

## Owner decisions applied

1. Canonical fresh environment: Linux x86_64/Python 3.12 with a locked CPU
   development/test stack. Colab retains its managed stock PyTorch and uses a
   separate overlay. A separately locked local CUDA stack is not supported yet.
2. Benchmark v1: immutable historical retention. V2 uses new schemas and paths;
   no v1 hash is overwritten.
3. Historical validation anchor: retained as explicitly historical and
   non-rerunnable because its exact source bundle/checksum and checkpoint
   binaries are absent.
4. External publication: not inferred from the implementation request. Hugging
   Face changes and v2 uploads remain pending explicit authorization.
5. `src/training/smoke_test.py`: internal/dormant, not a supported entry point,
   until a caller, CLI contract, and data-fixture policy are separately defined.

## Immutable v1 record

| Artifact/source | Pre-remediation SHA256 |
| --- | --- |
| `bench/realistic_subset.pt` | `84e7e039f1df5b3a7b32dc3d4ac1b8fa21bba2827679b4d3f1650d394e2754bf` |
| `bench/realistic_subset.json` | `f4660b80ffeeb4e6e158791943a0dc5ba771461b5cf2a0080fff158ab1e7e6b5` |
| `bench/g4_realistic.json` | `63ec7627d0e645058e578cf13fadd9d6166796a898b4f3c95e97c61d2dd232cb` |
| `bench/a100_realistic.json` | `79848832c2c6fe2d7024f9f143b5d2af4fd1faf8d172ec4f0e963e1fe6d85aec` |
| `bench/g4.json` | `72b2cbb95cb09f946fb8010b6ca71cc83557417f3045af44219cb34f4c4bbfdb` |
| `bench/a100.json` | `d27416211f9eac65e18a87203debef353b6178a3ae54bee82c94139bf2a02988` |
| legacy `benchmark_step_realistic.py` | `8bd0e040f873b8783379c0693970c9284d84506155199060bd83ebe1274cf633` |
| legacy `benchmark_step.py` | `968dbd230b9ed49535047e83470393048a2a4a20fb024ca7ed3b067e8cdc5d0e` |
| legacy `build_realistic_benchmark_subset.py` | `ce541713866a0d5b87f9b4c806d69abad2c2a99a1b6f7d7565a98daeb9da4d61` |

The tracked v1 JSON files were not modified. The local untracked 50 MB subset
was not regenerated. Full inventory and interpretation are in
[`bench/README.md`](../bench/README.md).

## Implemented remediation

| Finding | Implementation |
| --- | --- |
| R1 environment | Added `pyproject.toml`, `uv.lock`, explicit test/reporting/ESM groups, `requirements/colab-overlay.txt`, and the environment contract. The notebook installs only the overlay in Colab. |
| R2 portability | Added logical subset schema v2 and executable validation. The builder serializes only mappings, lists, scalars, and tensors, proves `weights_only=True` loading, then the runner reconstructs `PocketData`. |
| R3 traceability | Both runners now have argparse/main guards, result schema v2, exact invocation/source/runner/environment/optimizer/seed/timing fields, and nonzero failure exits after diagnostic JSON is written. |
| R4 historical claims | Current status and evidence index explicitly label the anchor historical/non-rerunnable; no metadata was backfilled. |
| R5 run metadata | Added optional `--dataset-bundle-id` and `--dataset-bundle-sha256`; future run outputs include additive `runtime_environment`, `source_control`, and `source_artifacts` objects. Notebook commands pass existing bundle values. |
| R6 tests | Repaired the archived documentation path, isolated smoke checks under pytest, made the wrapper continue after failures, added explicit skips, and added pinned CPU CI plus benchmark contract/failure tests. |
| R7 cohort provenance | V2 builder records source-site count, feature-complete eligible population, eligibility rules, load/skip report, label scheme, bundle identity, graph-builder source state, and sampling algorithm. |
| R8 inventory | Added `bench/README.md`, v1 hashes/limitations, v2 schemas, exact local/GPU commands, failure semantics, and timing interpretation. |

## Verification record

Final local results:

- configured workstation interpreter verified as
  `/home/mechti/miniconda3/envs/DeepMzyme/bin/python`; the canonical lock was
  also created independently at `.venv/bin/python` by PyCharm/uv;
- locked environment: Python `3.12.3`, PyTorch `2.5.1+cpu`,
  torch-geometric `2.7.0`, NumPy `2.4.4`, scikit-learn `1.8.0`, Optuna `4.8.0`,
  and pytest `8.4.2` imported successfully;
- optional ESM contract: `uv sync --frozen --group esm` installed the locked
  group, and ESM `3.2.3` model/API imports succeeded;
- clean-copy reconstruction: `uv sync --frozen --no-default-groups --group
  test` created a separate temporary environment with 51 locked packages;
  core imports and `src/train.py --help` passed, then the temporary copy was
  deleted;
- syntax: `py_compile` passed for the main model/training files, new
  reproducibility/benchmark modules, both benchmark runners, builder, and test
  wrapper;
- CLI: `src/train.py --help` passed and exposes both dataset-bundle fields;
- compatibility wrapper: all 43 checks completed; 42 passed and the optional
  historical multi-metal fixture check skipped explicitly because the audited
  local data is absent;
- pytest: 43 isolated cases collected; 42 passed and the same one skipped;
- benchmark failure test: realistic and synthetic runners both wrote valid v2
  diagnostic JSON and returned nonzero with CUDA hidden/unavailable;
- portability test: a synthetic v2 subset loaded under `python -I` with
  `weights_only=True`, without the repository on `sys.path`, then reconstructed
  `PocketData` inside the project;
- v1 preservation: all six recorded data/result hashes matched, and tracked v1
  JSON files had no Git diff;
- lock/notebook: `uv lock --check`, `uv sync --frozen`, `git diff --check`, and
  notebook JSON parsing (`nbformat=4`, 19 cells) passed.

## Authorization-gated remainder

The following actions were deliberately not performed:

- generating the full v2 realistic subset from project data;
- running training, Optuna, Stage 6, Stage 6B, or Stage 7;
- rerunning G4 or A100 benchmarks;
- uploading versioned artifacts or changing the Hugging Face dataset card;
- recovering or fabricating historical checkpoints, bundle hashes, or old
  environment identities;
- selecting a final-test dataset route.

After explicit authorization, generate the v2 subset, validate it in a clean
environment, run the identical pinned command on G4 and A100, verify schema,
hash, configuration, and throughput arithmetic, then upload at a new versioned
path and mirror `bench/README.md` into the dataset card.
