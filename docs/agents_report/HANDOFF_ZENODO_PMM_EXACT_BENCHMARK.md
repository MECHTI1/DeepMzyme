# Handoff & Continuation Playbook: Zenodo PinMyMetal Exact Ion-Level Benchmark

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
