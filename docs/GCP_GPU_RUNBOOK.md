# Google Cloud Platform (GCP) GPU VM Runbook

This runbook documents how to provision, connect, operate, and optimize the dedicated DeepMzyme GPU Virtual Machine on Google Cloud Platform.

For Google Colab interactive runs, consult [`docs/COLAB_GPU_RUNBOOK.md`](COLAB_GPU_RUNBOOK.md). This document owns the GCP Virtual Machine environment, PyCharm remote development, and autonomous batch training.

---

## 1. Overview & Architecture

The GCP GPU environment provides a dedicated, persistent Linux development host paired with enterprise NVIDIA GPUs for training, evaluating, and embedding representations in DeepMzyme.

| Component | Specification |
| :--- | :--- |
| **Instance Name** | `deepmzyme-l4` |
| **GCP Project** | `deepmzyme-gpu-vm` (Project Number: `589279151828`) |
| **Machine Type** | `g2-standard-8` (8 vCPU, 32 GB RAM) |
| **Accelerator** | 1x NVIDIA L4 (24 GB GDDR6, Ada Lovelace compute capability 8.9) |
| **Boot Disk** | 150 GB `pd-balanced` (Persistent Disk Balanced) |
| **Default Zone / Region** | `us-central1-a` (`us-central1` Iowa) |
| **Security & Network** | Private subnet (`deepmzyme-subnet`), Identity-Aware Proxy (IAP) SSH only (no open public ports) |
| **Remote Interpreter** | Python 3.12 venv at `/home/mechti/deepmzyme-venv` |
| **CUDA / PyTorch Build** | PyTorch 2.11+cu128 (matches Colab verified builds; sm_89 compatible) |
| **Gross Commercial Rate** | **$0.87917/h** ($0.85362/h compute + $0.005/h IP + $0.02055/h disk capacity) |

---

## 2. Non-Negotiable Safety & Billing Guardrails

All operations are governed by the controller in `~/deepmzyme-vm/` (tracked in repository `MECHTI1/deepmzyme-vm-manager`).

1. **Never use raw `gcloud compute instances create/start` directly.** Always manage the instance via `~/deepmzyme-vm/bin/*`.
2. **Explicit User Authorization Gate:** Starting or creating a VM requires the exact authorization phrase `AUTHORIZE VM START`. AI agents cannot start a VM autonomously without user instruction.
3. **Hard Google-Side Termination Limits:** Every creation and start sets `--max-run-duration` with `--instance-termination-action=STOP`. Even if your laptop sleeps or your connection drops, Google's datacenter automatically shuts down the instance at the cutoff time.
4. **Gross Commercial Accounting:** All cost limits, session caps, and reports reflect gross commercial list prices. Promotional credits (such as the \$300 trial) are never used to relax budget ceilings.
5. **No Idle Billing:** The VM must always be stopped (`bin/vm-stop`) immediately when work completes.

---

## 3. Quickstart CLI Reference

All commands are run from your local terminal:

```bash
# Check status, running state, and daily spend ledger
~/deepmzyme-vm/bin/vm-status

# Non-billable preflight check (verifies APIs, billing, and GPU quotas)
~/deepmzyme-vm/bin/vm-preflight

# List available zones offering the machine type (useful during stockouts)
~/deepmzyme-vm/bin/vm-zones

# Create the VM under a capped duration (e.g., 0.5 hours for testing)
~/deepmzyme-vm/bin/vm-create --hours 0.5 --authorize 'AUTHORIZE VM START'

# Provision remote environment (installs NVIDIA driver, PyTorch+CUDA, datasets, smoke test)
~/deepmzyme-vm/bin/vm-setup

# Configure SSH config and get PyCharm connection parameters
~/deepmzyme-vm/bin/vm-pycharm-info --write-ssh-config

# Connect directly via interactive SSH over secure IAP
~/deepmzyme-vm/bin/vm-connect

# Stop the VM immediately (stops all compute and IP billing)
~/deepmzyme-vm/bin/vm-stop

# Detailed gross ledger and financial usage report
~/deepmzyme-vm/bin/vm-report
```

---

## 4. PyCharm Remote Development Setup

You can use PyCharm Professional to develop locally and execute seamlessly on the remote L4 GPU:

1. **Write SSH configuration:**
   ```bash
   ~/deepmzyme-vm/bin/vm-pycharm-info --write-ssh-config
   ```
2. **Add Remote Interpreter in PyCharm:**
   * Open **Settings / Preferences** $\rightarrow$ **Project: DeepMzyme** $\rightarrow$ **Python Interpreter**.
   * Click **Add Interpreter** $\rightarrow$ **On SSH...**
   * Select **Existing SSH configuration** and choose `deepmzyme-vm` (or Host `localhost`, Port `2222` if using `bin/vm-connect --tunnel`).
3. **Select Environment Paths:**
   * **Python Interpreter path:** `/home/mechti/deepmzyme-venv/bin/python`
   * **Remote Project Path:** `/home/mechti/DeepMzyme`
4. **Deployment & Sync Rules:**
   * **IMPORTANT:** Do **NOT** enable automatic deployment/upload for `DeepMzyme_Data`, `deepmzyme_cache`, or `deepmzyme_runs`. Large scientific data and embeddings must remain on the VM.
   * `DEEPMZYME_LOAD_WORKERS` and `DEEPMZYME_PARSE_CACHE_DIR` are preconfigured on the VM in `/etc/environment`.

---

## 5. GPU Efficiency Best Practices for DeepMzyme

To maximize training throughput and make your \$300 credit last as long as possible:

### A. Automatic Mixed Precision (BF16 / FP16)
The NVIDIA L4 features 4th-generation Tensor Cores optimized for `bfloat16`. Always wrap your training iterations with PyTorch AMP:
```python
import torch

scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
```
*Effect:* Approximately **2x speedup** in forward/backward passes and **50% lower VRAM usage**, allowing larger batch sizes.

### B. DataLoader Workers & Memory Pinning
Ensure the 8 vCPUs on `g2-standard-8` continuously supply batches to the GPU without stalling:
```python
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)
```

### C. Precomputed Feature & ESM Embedding Caching
Embedding protein sequences with large ESM models is computationally expensive.
* Never run raw ESM sequence embedding repeatedly inside training epochs.
* Precalculate embeddings once and cache them in `REMOTE_CACHE_DIR` (`/home/mechti/deepmzyme_cache/parse`).
* The training loop should read precomputed tensors directly from disk to keep GPU utilization above 90%.

### D. Unattended Batch Execution with Automatic Poweroff
Do not leave an interactive terminal or PyCharm session open while waiting for long runs to finish. Instead, launch your script detached and power down upon completion:
```bash
# Example batch launch via SSH:
nohup bash -c "python -m scripts.run_metal_5fold_cv --config configs/production.yaml && sudo poweroff" > train.log 2>&1 &
```
When the script exits, `sudo poweroff` shuts down the VM. The local controller's watchdog detects the stop and marks the session closed in the ledger, preventing accidental overnight charges.

---

## 6. Long-Term Storage Optimization (Disk Snapshots)

* While a stopped VM incurs **$0.00/h for compute**, the 150 GB persistent boot disk bills **~$15/month (~$0.02/h)** as long as it exists.
* **If pausing GPU experimentation for more than a few days:**
  1. Take a snapshot of the boot disk:
     ```bash
     gcloud compute disks snapshot deepmzyme-l4 --zone=us-central1-a --snapshot-names=deepmzyme-l4-backup-$(date +%Y%m%d) --project=deepmzyme-gpu-vm
     ```
  2. Delete the VM and disk:
     ```bash
     ~/deepmzyme-vm/bin/vm-delete --delete-disk --confirm 'DELETE deepmzyme-l4'
     ```
  3. When ready to resume, recreate the disk from the snapshot in under 60 seconds. Storage cost for compressed snapshots is only ~$0.026/GB-month, reducing idle holding costs by >70%.

---

## 7. Troubleshooting & Stockout Handling

### `ZONE_RESOURCE_POOL_EXHAUSTED` (Datacenter Capacity Stockout)
* **Cause:** Google has temporarily rented out all physical NVIDIA L4 GPUs in the requested zone. This is common during US business peak hours (14:00–23:00 UTC).
* **Diagnosis:**
  Run `~/deepmzyme-vm/bin/vm-zones` to see all zones in `us-central1` that offer `g2-standard-8`:
  * `us-central1-a`
  * `us-central1-b`
  * `us-central1-c`
* **Resolution:**
  1. Switch to an alternative zone by editing `ZONE` in `~/deepmzyme-vm/config.env` (e.g. `ZONE=us-central1-c`).
  2. Or wait and launch during US off-peak hours (evenings / early mornings).
