# Handoff & Continuation Playbook: Zenodo PinMyMetal Exact Ion-Level Benchmark

> **STATUS 2026-09-24 11:50 UTC (14:50 local): THE RUN DESCRIBED BELOW IS DEAD AND
> PRODUCED NO RESULTS.** Colab session `pmm-zenodo` was reaped by the backend;
> `colab sessions` reports no active sessions. Fold 0 never reached epoch 1 (GPU was
> at 0% / 3 MiB VRAM at the last poll) and no checkpoint or metrics file survives.
> Sections 1-4 below describe the *intended* run and are retained for reference only;
> the monitoring commands in section 2 will all fail with "Session not found".
> **Read section 7 (Post-mortem) before relaunching anything.**

## Purpose
This document provides exact, turn-key instructions for any subsequent autonomous agent or developer to inspect, monitor, retrieve results from, or resume/continue the **Exact Zenodo PinMyMetal Benchmark** from the exact point of interruption.

---

## 1. Quick Status & Context
- **Dataset:** `train_and_test_sets_structures_zenodo_pmm_exact` (99.89% exact reconstruction of Zenodo PMM benchmark, 9,398 total sites: 7,911 train in 6,443 PDBs, 1,487 test in 1,281 PDBs).
- **Hosting:** Hugging Face [`GMBioinformatics/DeepMzyme`](https://huggingface.co/datasets/GMBioinformatics/DeepMzyme/resolve/main/train_and_test_sets_structures_zenodo_pmm_exact.tar.gz) (SHA-256: `24903f120462aacb90b43c4af97f4f08d61f1847d12636606fcc66e6351db296`).
- **Supervision Unit:** `--metal-example-unit ion` (coordinates and microenvironments preserved per individual metal ion; multi-nuclear centers like `1a0e` 3-Zn center are separated into distinct examples while grouped under parent pocket IDs).
- **Active Job:** Fold 0 of `benchmark_enhanced_only_gvp` (50 epochs, lr=3e-4, raw RBF) on Google Colab session `pmm-zenodo` (NVIDIA L4 GPU, 24 GB VRAM).
- **Execution Start:** 2026-09-24 10:40:27 UTC (13:40:27 local time).
- **Estimated Completion:** 2026-09-24 11:24 UTC (14:24 local time).
- **Active Process IDs:** Runner PID `5303`, Training PID `5353`.

---

## 2. Monitoring & Diagnostic Commands

Run these commands from the local repository root (`/home/mechti/PycharmProjects/DeepMzyme`):

### A. Check Session Status
```bash
/home/mechti/.local/bin/colab status -s pmm-zenodo
```

### B. Inspect Running Processes & GPU Utilization
```bash
echo "import subprocess; print(subprocess.check_output('ps aux | grep train.py || true; nvidia-smi || true', shell=True, text=True))" | /home/mechti/.local/bin/colab exec -s pmm-zenodo
```

### C. Download and View the Latest Execution Log
```bash
/home/mechti/.local/bin/colab download -s pmm-zenodo /content/zenodo_single_fold_execution.log /tmp/zenodo_single_fold_execution.log
tail -n 50 /tmp/zenodo_single_fold_execution.log
```

### D. Check if Fold 0 Checkpoints and Test Report are Generated
```bash
echo "import os; print(os.listdir('/content/runs/runs_zenodo_pmm_exact/benchmark_enhanced_only_gvp_fold0') if os.path.exists('/content/runs/runs_zenodo_pmm_exact/benchmark_enhanced_only_gvp_fold0') else 'NOT CREATED')" | /home/mechti/.local/bin/colab exec -s pmm-zenodo
```

---

## 3. Retrieving Completed Results

When Fold 0 completes:
1. **Download the Run Artifacts to Local Storage:**
   *(Always target `/media/mechti/Data1` to preserve `/home` partition space)*
   ```bash
   mkdir -p /media/mechti/Data1/DeepMzyme_Data/runs_zenodo_pmm_exact
   /home/mechti/.local/bin/colab download -s pmm-zenodo \
     /content/runs/runs_zenodo_pmm_exact/benchmark_enhanced_only_gvp_fold0 \
     /media/mechti/Data1/DeepMzyme_Data/runs_zenodo_pmm_exact/benchmark_enhanced_only_gvp_fold0
   ```
2. **Download the Comparative Markdown Summary Table:**
   ```bash
   /home/mechti/.local/bin/colab download -s pmm-zenodo \
     /content/runs/runs_zenodo_pmm_exact/zenodo_pmm_exact_5fold_comparison_table.md \
     /media/mechti/Data1/DeepMzyme_Data/runs_zenodo_pmm_exact/zenodo_pmm_exact_5fold_comparison_table.md
   cat /media/mechti/Data1/DeepMzyme_Data/runs_zenodo_pmm_exact/zenodo_pmm_exact_5fold_comparison_table.md
   ```
3. **Verify Expected Output Files:**
   - `best_checkpoint.pt`: Best model weights based on `val_metal_balanced_acc`
   - `val_metrics.csv`: 50-epoch validation log with balanced accuracy and per-metal recalls
   - `test_predictions.pt`: Model logits and predictions on the 1,487 held-out test sites
   - `test_report.json`: Metrics report (Balanced Accuracy, Macro F1, Recall for Mn, Zn, Cu, Class VIII)

---

## 4. Continuing with the Remaining Folds or Models

To continue training remaining folds (Folds 1, 2, 3, 4) or models on Colab:

```bash
cat << 'EOF' | /home/mechti/.local/bin/colab exec -s pmm-zenodo
import subprocess
cmd = (
    "nohup python3 -u /content/DeepMzyme/scripts/run_zenodo_pmm_exact_5fold_cv.py "
    "--models benchmark_enhanced_only_gvp "
    "--folds 1 2 3 4 "
    "--n-folds 5 "
    "--data-root /content/DeepMzyme_Data/DeepMzyme_Data "
    "--runs-dir /content/runs/runs_zenodo_pmm_exact "
    "--epochs 50 "
    "--batch-size 16 "
    "--device cuda > /content/zenodo_folds1_4_execution.log 2>&1 & echo $!"
)
pid = subprocess.check_output(cmd, shell=True, text=True).strip()
print(f"LAUNCHED REMAINING FOLDS PID: {pid}")
EOF
```

To run the Sequence-Only ESM-C (`benchmark_only_esm`) or Multimodal Fusion (`benchmark_enhanced_gvp_esmc`):
```bash
# In the command above, specify:
# --models benchmark_only_esm benchmark_enhanced_gvp_esmc
```

---

## 5. 1-Command Clean Reproduction from Scratch

On ANY clean Linux workstation or fresh Colab instance:

```bash
git clone https://github.com/GMBioinformatics/DeepMzyme.git
cd DeepMzyme
bash scripts/reproduce_zenodo_pmm_benchmark.sh
```

To run only a specific fold (e.g. Fold 0) or custom epochs:
```bash
FOLDS="0" EPOCHS=50 bash scripts/reproduce_zenodo_pmm_benchmark.sh
```

---

## 6. If the Colab Session is Recycled / Disconnected

If the Colab VM disconnects:
1. Re-provision an NVIDIA L4 GPU instance:
   ```bash
   /home/mechti/.local/bin/colab new -s pmm-zenodo --gpu L4
   ```
2. Upload the latest code archive:
   ```bash
   git archive -o /tmp/deepmzyme_code_latest.tar.gz HEAD
   /home/mechti/.local/bin/colab upload -s pmm-zenodo /tmp/deepmzyme_code_latest.tar.gz /content/deepmzyme_code_latest.tar.gz
   ```
3. Run the 1-command bootstrap script:
   ```bash
   # Executes download from HF, SHA-256 verification, and benchmark execution automatically
   bash scripts/reproduce_zenodo_pmm_benchmark.sh
   ```

---

## 7. Post-mortem: why the run was lost, and what to change before relaunching

### What actually happened
| Time (UTC) | Event |
| :--- | :--- |
| 09:20:34 | First `pmm-zenodo` VM created (`gpu-l4-s-kkb-usw4a0-mryee7671hft`). |
| 10:21:36 | First VM **pruned** by the backend. |
| 10:22:35 | Second VM created (`gpu-l4-s-kkb-ass1b1-c4x86vhbzxfb`). |
| 10:40:27 | Fold 0 of `benchmark_enhanced_only_gvp` launched (training PID 5353). |
| 10:51:48 | Last successful telemetry poll: 11m12s CPU time, **GPU 0%, 3 MiB / 23,034 MiB VRAM**, run dir contains only `prepare_status.json`. Training had **not** started. |
| 11:42 | **Workstation rebooted** (`who -b`: 2026-09-24 14:42 local). The Colab CLI keep-alive daemon is a *local* process, so it died here. |
| 11:48:32 | Second VM found gone; local session state **pruned**. All VM-side artifacts lost. |

### Root cause
The Colab keep-alive daemon runs on the local workstation. When the workstation went
down, nothing renewed the VM lease and the backend reclaimed it. Everything the job had
produced lived only on that VM's ephemeral disk.

### Durability gaps that turned a reboot into total data loss
1. **No off-VM persistence.** `--runs-dir /content/runs/...` is ephemeral VM disk.
   Nothing was mirrored to Google Drive (`colab drivemount`) or downloaded during the
   run, so a reap destroys all output. *Fix: mount Drive and point `--runs-dir` at it,
   or download artifacts on a timer.*
2. **Best checkpoint is held in RAM until the run ends.** `train_and_select_checkpoint`
   (`src/training/run.py:1612`) deep-copies the best state into memory; it is only
   written to disk by `persist_run_outputs` (`src/training/run.py:1886`) after the final
   epoch. A kill at epoch 49 of 50 leaves no weights. Per-epoch metric CSVs *are*
   written each epoch (`src/training/run.py:1629`), and `--save-epoch-checkpoints`
   exists but was not enabled. *Fix: enable per-epoch checkpoints for long runs.*
3. **The ~11-minute graph-construction phase is silent.** Neither
   `src/training/data.py` nor the loaders print progress, so the log shows only the
   runner banner for the first ten-plus minutes. This is what made a stalled/never-started
   job look identical to a healthy one, and led the previous session to report "50 epochs
   training, finishing ~11:24 UTC" while the GPU was in fact idle at 0%.
   *Fix: emit a structure-parsing progress line, and always check `nvidia-smi`
   utilisation - not just CPU time - before claiming training is underway.*
4. **No parsed-graph cache.** There is no caching in `src/training/graph_dataset.py`
   or `src/graph/structure_parsing.py`, so all 6,443 train PDBs are re-parsed from
   scratch for every fold and every model. A full 3-model x 5-fold campaign repeats
   this 15 times (~2.75 h of pure re-parsing). *Fix: cache built graphs to disk keyed
   by structure set + feature config.*
5. **`/media/mechti/Data1` does not exist on this workstation.** Both this document and
   the local-path fallback in `scripts/reproduce_zenodo_pmm_benchmark.sh` reference it;
   the actual 1.8 TB disk is `/dev/sda1` (NTFS, label `Data`) and is currently
   **unmounted**. The root filesystem has only ~7.4 GB free (97% full), so a download
   target must be chosen and mounted before retrieving run artifacts.

### Pre-flight checklist for the relaunch
- [ ] Mount Drive on the VM and write run outputs there (survives a VM reap).
- [ ] Enable per-epoch checkpointing for any run longer than a few minutes.
- [ ] Confirm the workstation will stay powered on, or accept that the VM dies with it.
- [ ] Verify training actually started by checking GPU utilisation is non-zero, not CPU time.
- [ ] Mount `/dev/sda1` (or pick another target) before downloading artifacts; `/` is 97% full.
- [ ] `colab stop -s <name>` when finished - idle VMs keep burning compute units.
