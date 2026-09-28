# Preserved STATUS history

Historical text, copied verbatim before the Job B rewrite. Old next actions,
permissions, counts and claims are dated evidence, not current instructions.
Current authority: [EXPERIMENT_STATUS.md](../../../EXPERIMENT_STATUS.md).
Source ranges and hashes: [preservation manifest](../../../docs/archive/consolidation_2026-09/inventory/job_b_status_preservation.json).
Literal blocks preserve original relative link spelling; use this folder’s
README for navigable evidence links. No historical error has been silently fixed.

## zenodo-001

Source: `3c0f80c:EXPERIMENT_STATUS.md:353-396`.

```text
**2026-09-24 ion-unit correction & Zenodo exact benchmark (RELAUNCHED after total run loss):**
- **Contract & Architecture:** `--metal-example-unit ion` is fully implemented for standalone metal training in CLI, runner scripts, and Colab notebook, ensuring multi-nuclear sites (e.g. `1a0e` 3-Zn center) receive independent coordinates and residue microenvironments while grouping sibling ions under parent pocket IDs.
- **Dataset Fidelity:** Reconstructed from published Zenodo source rows (`classmodel_train_set.csv` and `classmodel_test_set.csv`) at **99.89% exact fidelity** (9,398 / 9,408 source rows: 7,911 train sites across 6,443 PDBs, 1,487 test sites across 1,281 PDBs).
- **Hugging Face Hosting:** Published permanently to [`GMBioinformatics/DeepMzyme`](https://huggingface.co/datasets/GMBioinformatics/DeepMzyme) (`train_and_test_sets_structures_zenodo_pmm_exact.tar.gz`, SHA-256: `24903f120462aacb90b43c4af97f4f08d61f1847d12636606fcc66e6351db296`).
- **Automated Verification:** Verified via `scripts/verify_zenodo_pmm_ion_dataset.py` (passes 100% with 0 errors).
- **1-Command Reproducibility:** Automated via `scripts/reproduce_zenodo_pmm_benchmark.sh` (downloads bundle if missing, verifies SHA-256, runs pre-flight tests, launches cross-validation, and compiles comparison table).
- **Colab GPU Execution (LOST - must be relaunched):**
  - Session `pmm-zenodo` (endpoint `gpu-l4-s-kkb-ass1b1-c4x86vhbzxfb`, NVIDIA L4) was
    **reaped by the Colab backend and no longer exists**. `colab sessions` reports no
    active sessions; local session state was pruned at 2026-09-24 11:48 UTC.
  - **Cause:** the workstation rebooted at 2026-09-24 11:42 UTC (14:42 local). The
    Colab CLI keep-alive daemon runs *locally*, so it died with the machine and the
    backend reclaimed the VM. The same pruning had already happened once earlier to
    the first `pmm-zenodo` VM (created 09:20 UTC, pruned 10:21 UTC).
  - **Result: zero training artifacts survive.** Fold 0 of
    `benchmark_enhanced_only_gvp` (started 10:40:27 UTC) had produced only
    `prepare_status.json` in its run directory. At the last successful telemetry poll
    (10:51:48 UTC, 11m12s of CPU time) `nvidia-smi` reported **0% GPU utilisation and
    3 MiB of 23,034 MiB VRAM in use**, i.e. the job was still in the single-threaded
    CPU graph-construction phase and **had not begun epoch 1**. No checkpoint, no
    `val_metrics.csv`, no `test_report.json` was ever written, and nothing was
    downloaded off the VM before it was reclaimed.
  - **No published benchmark numbers exist for this dataset yet.** Any earlier note
    giving an expected completion time (~11:24 UTC / 14:24 local) is superseded.
  - **Before relaunching**, see the durability gaps recorded in
    [`docs/agents_report/HANDOFF_ZENODO_PMM_EXACT_BENCHMARK.md`](docs/agents_report/HANDOFF_ZENODO_PMM_EXACT_BENCHMARK.md)
    (section "Post-mortem"): artifacts are written only to ephemeral VM disk, the best
    checkpoint is held in memory until the run ends, and the ~11-minute graph
    construction phase emits no progress output and is repeated for every fold.
- **Relaunch in flight (session `pmm-zenodo-v2`, started 2026-09-24 12:05 UTC / 15:05 local):**
  - Fold 0 of `benchmark_enhanced_only_gvp`, 50 epochs, lr=3e-4, raw RBF, seed 42,
    now launched with **`--save-epoch-checkpoints`** so an interrupted run keeps its weights.
  - Artifacts are mirrored off the VM every 5 minutes by
    `scripts/colab_artifact_streamer.py` into `~/zenodo_pmm_artifacts`, so a VM reclaim
    costs at most one poll interval instead of the whole fold.
  - **Measured** structure-parsing throughput: **3.0 structures/s**, i.e. **~36 minutes**
    to parse the 6,443 training structures. The previously documented "~9-11 minutes"
    for this phase was an estimate and is wrong by roughly 3x; budget fold timings
    accordingly (~36 min parse + ~32 min train + test parse/eval).
  - Two runner defects found and fixed before relaunching (see below), either of which
    would have produced an empty comparison table even from a fully successful run.

  - Master Guide: [`docs/ZENODO_PINMYMETAL_EXACT_ION_LEVEL_REPRODUCIBILITY.md`](docs/ZENODO_PINMYMETAL_EXACT_ION_LEVEL_REPRODUCIBILITY.md)

```
