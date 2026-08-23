# DeepMzyme benchmark artifact inventory

This directory is the authority for DeepMzyme compute-benchmark files,
schemas, commands, checksums, and interpretation. These workloads measure
training-step compute behavior; they do not measure model quality and do not
authorize held-out-test access.

## Immutable historical v1 inventory

The v1 files remain unchanged. Their limitations are part of the record: the
realistic subset pickles `graph.construction.PocketData`, requires DeepMzyme
during unpickling, and cannot be loaded with `weights_only=True`. The realistic
result JSON files predate result schema v2 and do not contain enough runner and
environment identity to prove that they came from the currently shipped
runner.

| File | SHA256 | Status |
| --- | --- | --- |
| `realistic_subset.pt` | `84e7e039f1df5b3a7b32dc3d4ac1b8fa21bba2827679b4d3f1650d394e2754bf` | Untracked local copy; byte-identical hosted v1 artifact; project-class-dependent |
| `realistic_subset.json` | `f4660b80ffeeb4e6e158791943a0dc5ba771461b5cf2a0080fff158ab1e7e6b5` | Tracked v1 manifest; cohort field is historically mislabeled |
| `g4_realistic.json` | `63ec7627d0e645058e578cf13fadd9d6166796a898b4f3c95e97c61d2dd232cb` | Historical realistic G4 evidence; legacy result schema |
| `a100_realistic.json` | `79848832c2c6fe2d7024f9f143b5d2af4fd1faf8d172ec4f0e963e1fe6d85aec` | Historical realistic A100 evidence; legacy result schema |
| `g4.json` | `72b2cbb95cb09f946fb8010b6ca71cc83557417f3045af44219cb34f4c4bbfdb` | Historical synthetic G4 evidence; legacy result schema |
| `a100.json` | `d27416211f9eac65e18a87203debef353b6178a3ae54bee82c94139bf2a02988` | Historical synthetic A100 evidence; legacy result schema |

The pre-remediation source hashes associated with these legacy artifacts were:

- `benchmark_step_realistic.py`: `8bd0e040f873b8783379c0693970c9284d84506155199060bd83ebe1274cf633`;
- `benchmark_step.py`: `968dbd230b9ed49535047e83470393048a2a4a20fb024ca7ed3b067e8cdc5d0e`;
- `build_realistic_benchmark_subset.py`: `ce541713866a0d5b87f9b4c806d69abad2c2a99a1b6f7d7565a98daeb9da4d61`.

These hashes document history; they do not repair the missing command,
commit, input-artifact, and environment fields in the old result JSON.

Hosted v1 files:

- [manifest](https://huggingface.co/datasets/GMBioinformatics/DeepMzyme/resolve/main/benchmarks/gvp_esm_hybrid_realistic_subset_v1/realistic_subset.json)
- [subset](https://huggingface.co/datasets/GMBioinformatics/DeepMzyme/resolve/main/benchmarks/gvp_esm_hybrid_realistic_subset_v1/realistic_subset.pt)

## Version 2 contracts

Version 2 is additive and does not overwrite v1:

- [`schemas/realistic-subset-v2.schema.json`](schemas/realistic-subset-v2.schema.json)
  describes the logical tensor-only subset payload;
- [`schemas/benchmark-result-v2.schema.json`](schemas/benchmark-result-v2.schema.json)
  describes benchmark results;
- [`../src/benchmarking/artifacts.py`](../src/benchmarking/artifacts.py)
  performs executable validation, safe loading, and in-runner `PocketData`
  reconstruction;
- [`../build_realistic_benchmark_subset.py`](../build_realistic_benchmark_subset.py)
  builds a versioned v2 subset and provenance manifest;
- [`../benchmark_step_realistic.py`](../benchmark_step_realistic.py) runs the
  realistic v2 workload and returns nonzero after writing diagnostics on any
  failure;
- [`../benchmark_step.py`](../benchmark_step.py) provides the synthetic v2
  diagnostic workload with the same failure semantics.

No v2 subset or GPU result is tracked or published yet. Artifact generation,
G4/A100 execution, and Hugging Face upload require separate authorization.

## Build and validate a v2 subset

Run only after artifact-regeneration authorization and with the documented
CARE/ESM/RING/external-feature inputs present:

```bash
uv run --frozen --no-sync python build_realistic_benchmark_subset.py \
  --source-bundle-id DeepMzyme_Data_v10_exact_common70_clean30main_clean30x5_care30_esm_ring_external.tar.zst \
  --source-bundle-sha256 09525aad00d6c35e32a3601ff3ecf338978c465cec1ccfc18e47b9222b220aba \
  --output-dir bench/gvp_esm_hybrid_realistic_subset_v2
```

The builder records the source CSV row count, eligible feature-complete
population, all eligibility rules, load/skip report, label scheme, graph
configuration, source bundle identity, graph-builder commit/dirty state, and
sampling algorithm. It then reloads the result with
`torch.load(..., weights_only=True)` before writing the manifest.

Validate after generation:

```bash
sha256sum bench/gvp_esm_hybrid_realistic_subset_v2/realistic_subset_v2.pt
uv run --frozen --no-sync python -I -c "import torch; torch.load('bench/gvp_esm_hybrid_realistic_subset_v2/realistic_subset_v2.pt', map_location='cpu', weights_only=True); print('safe load passed')"
```

## Prepare a pinned source archive

Build the archive from the exact commit that will be recorded in the result:

```bash
git status --short
git rev-parse HEAD
tar -czf /tmp/deepmzyme_benchmark_src.tar.gz -C src .
sha256sum /tmp/deepmzyme_benchmark_src.tar.gz benchmark_step_realistic.py
```

Do not use a dirty checkout for a reportable GPU result. Transfer the runner,
source archive, and v2 subset to the GPU runtime without renaming their
contents.

## Run the realistic v2 benchmark

Replace all angle-bracket placeholders with values from the generated v2
manifest and pinned clean checkout:

```bash
python benchmark_step_realistic.py \
  --subset /content/realistic_subset_v2.pt \
  --subset-url <V2_HUGGING_FACE_URL> \
  --subset-sha256 <V2_SHA256> \
  --source-commit <GIT_COMMIT> \
  --result /content/g4_realistic_v2.json \
  --batch-size 12 \
  --warmup-steps 3 \
  --measured-steps 20 \
  --num-workers 4 \
  --seed 42 \
  --learning-rate 3.705631497756492e-5 \
  --weight-decay 1e-5
```

Use the same command for A100 except for the output filename. The runner
records the exact invocation, source commit/dirty state, runner hash,
input checksum and provenance, Python/platform/package versions, GPU/CUDA and
driver details, optimizer settings, seed policy, raw step timings, median, and
throughput.

## Result validation and interpretation

Successful v2 results have `status: "ok"`. Any caught setup, checksum, loading,
CUDA, OOM, or runtime error is written with `status: "error"`, diagnostics, and
a nonzero process exit. Automation must require both exit code 0 and
`status: "ok"`.

Throughput is calculated as `batch_size / median_step_time_seconds`. Timing is
hardware- and runtime-dependent and is not expected to be bit-identical across
runs. Compare results only when the input hash, runner hash, source commit,
model/optimizer configuration, batch size, precision, warm-up/measured-step
counts, and seed policy match.
