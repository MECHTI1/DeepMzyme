#!/usr/bin/env python3
"""Colab 5-Fold Benchmark Watchdog Daemon.

Monitors Colab remote session health every interval:
1. Pings the active session to verify connectivity and prevent idling.
2. Checks training progress across all folds of benchmark models.
3. Automatically detects newly completed folds and streams their artifacts
   (test_report.json, val_metrics.csv, best_checkpoint.pt) to local disk.
4. If a tunnel prune / session termination is detected:
   - Terminates previous dead session cleanly via 'colab stop'
   - Automatically provisions a fresh NVIDIA L4 GPU instance
   - Sets up code and dataset environment
   - Restores completed fold archives
   - Automatically relaunches the runner script targeting benchmark_enhanced_gvp_esmc
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time


COLAB_CLI = "/home/mechti/.local/bin/colab"
SESSION_NAME = "pmm-5fold"
REPO_ROOT = Path("/home/mechti/PycharmProjects/DeepMzyme")
LOCAL_RUNS_DIR = REPO_ROOT / "runs" / "benchmark_exact_pinmymetal_5fold"
CHECK_INTERVAL_SECONDS = 120


def log(msg: str) -> None:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def run_cmd(
    cmd: list[str],
    *,
    stdin_text: str | None = None,
    timeout: int = 60,
) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            input=stdin_text,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "command timed out"
    except Exception as exc:
        return 1, "", str(exc)


def is_session_alive() -> bool:
    code, stdout, _ = run_cmd([COLAB_CLI, "url", "-s", SESSION_NAME], timeout=20)
    return code == 0 and "https://colab.research.google.com" in stdout


def check_and_sync() -> None:
    check_code = """
import os, json

runs_dir = '/content/runs/benchmark_exact_pinmymetal_5fold'
folds = {}
if os.path.exists(runs_dir):
    for entry in sorted(os.listdir(runs_dir)):
        p = os.path.join(runs_dir, entry)
        if os.path.isdir(p):
            val_p = os.path.join(p, 'val_metrics.csv')
            test_p = os.path.join(p, 'test_report.json')
            has_val = os.path.exists(val_p)
            has_test = os.path.exists(test_p)
            epochs = 0
            if has_val:
                try:
                    with open(val_p) as f:
                        epochs = max(0, len(f.readlines()) - 1)
                except Exception:
                    pass
            folds[entry] = {'epochs': epochs, 'completed': has_val and has_test and epochs >= 50}

log_tail = ''
if os.path.exists('/content/benchmark_5fold_execution.log'):
    try:
        with open('/content/benchmark_5fold_execution.log') as f:
            lines = f.readlines()
            log_tail = ''.join(lines[-5:])
    except Exception:
        pass

print('STATUS_JSON:' + json.dumps({'folds': folds, 'tail': log_tail}))
"""
    code, stdout, stderr = run_cmd(
        [COLAB_CLI, "exec", "-s", SESSION_NAME, "--timeout", "45"],
        stdin_text=check_code,
        timeout=60,
    )
    if code != 0:
        log(f"Heartbeat check returned code {code}: {stderr[:200]}")
        return

    if "STATUS_JSON:" in stdout:
        json_str = stdout.split("STATUS_JSON:")[1].strip().split("\n")[0]
        try:
            data = json.loads(json_str)
            folds = data.get("folds", {})
            tail = data.get("tail", "").strip()
            log(f"Session alive. Active folds tracking: {folds}")
            if tail:
                log(f"Recent log tail: {tail[-200:]}")

            # Sync completed folds
            for fold_name, info in folds.items():
                if info.get("completed"):
                    local_fold_dir = LOCAL_RUNS_DIR / fold_name
                    if not (local_fold_dir / "test_report.json").exists():
                        log(f"Fold {fold_name} is complete on Colab! Downloading...")
                        archive_cmd = f"import subprocess; subprocess.run(['tar', '-czf', '/content/{fold_name}.tar.gz', '-C', '/content/runs/benchmark_exact_pinmymetal_5fold', '{fold_name}'])\n"
                        run_cmd([COLAB_CLI, "exec", "-s", SESSION_NAME, "--timeout", "60"], stdin_text=archive_cmd, timeout=75)
                        run_cmd([COLAB_CLI, "download", "-s", SESSION_NAME, f"/content/{fold_name}.tar.gz", str(LOCAL_RUNS_DIR / f"{fold_name}.tar.gz")], timeout=60)
                        if (LOCAL_RUNS_DIR / f"{fold_name}.tar.gz").exists():
                            subprocess.run(["tar", "-xzf", str(LOCAL_RUNS_DIR / f"{fold_name}.tar.gz"), "-C", str(LOCAL_RUNS_DIR)], check=True)
                            log(f"Successfully synced fold {fold_name} to local disk!")

            # Check and sync 5-fold ensemble report if generated
            ensemble_report_name = "benchmark_enhanced_gvp_esmc_5fold_ensemble_report.json"
            local_ensemble_path = LOCAL_RUNS_DIR / ensemble_report_name
            if not local_ensemble_path.exists():
                code, stdout, _ = run_cmd(
                    [COLAB_CLI, "exec", "-s", SESSION_NAME, "--timeout", "15"],
                    stdin_text=f"import os; print('EXISTS:' + str(os.path.exists('/content/runs/benchmark_exact_pinmymetal_5fold/{ensemble_report_name}')))\n",
                    timeout=20,
                )
                if "EXISTS:True" in stdout:
                    log(f"Ensemble report {ensemble_report_name} found on Colab! Downloading...")
                    run_cmd(
                        [COLAB_CLI, "download", "-s", SESSION_NAME, f"/content/runs/benchmark_exact_pinmymetal_5fold/{ensemble_report_name}", str(local_ensemble_path)],
                        timeout=30,
                    )
                    if local_ensemble_path.exists():
                        log(f"Successfully downloaded {ensemble_report_name}!")
        except Exception as exc:
            log(f"Error parsing status json: {exc}")


def recover_session_with_l4() -> bool:
    log("ALERT: Colab session lost. Stopping any previous session to release quota...")
    run_cmd([COLAB_CLI, "stop", "-s", SESSION_NAME], timeout=20)
    time.sleep(3)

    log("Allocating fresh session with GPU L4...")
    code, stdout, stderr = run_cmd([COLAB_CLI, "new", "-s", SESSION_NAME, "--gpu", "L4"], timeout=60)
    if code != 0:
        log(f"L4 GPU allocation failed ({stderr}). Falling back to T4...")
        code, stdout, stderr = run_cmd([COLAB_CLI, "new", "-s", SESSION_NAME, "--gpu", "T4"], timeout=60)
        if code != 0:
            log(f"Fatal: GPU allocation failed completely: {stderr}")
            return False

    log("Session ready. Setting up environment...")
    setup_code = """
import subprocess, os, csv
from pathlib import Path

subprocess.run(["git", "clone", "https://github.com/MECHTI1/DeepMzyme.git", "/content/DeepMzyme"], check=True)
subprocess.run(["pip", "install", "-q", "torch-geometric", "biopython", "biotite", "propka", "gemmi"], check=True)
url = "https://huggingface.co/datasets/GMBioinformatics/DeepMzyme/resolve/4c36d3bfbf5e0c892fb165cbf101184e4151dda2/DeepMzyme_Data_v11_manifest_exact_common70_nonoverlap_clean30_care30_esm_ring_external.tar.gz"
subprocess.run(["wget", "-q", "-c", url, "-O", "/content/data.tar.gz"], check=True)
os.makedirs("/content/DeepMzyme_Data", exist_ok=True)
subprocess.run(["tar", "-xzf", "/content/data.tar.gz", "-C", "/content/DeepMzyme_Data"], check=True)

base = Path("/content/DeepMzyme_Data/DeepMzyme_Data/train_and_test_sets_structures_exact_pinmymetal")
for split in ["train", "test"]:
    split_dir = base / split
    manifest = split_dir / "structure_manifest.csv"
    with open(manifest) as f:
        reader = csv.DictReader(f)
        for row in reader:
            target = (split_dir / row["relative_path"]).resolve()
            link = split_dir / row["structure_name"]
            if not link.exists():
                try:
                    os.link(target, link)
                except Exception:
                    os.symlink(target, link)
print("SETUP_DONE")
"""
    code, stdout, stderr = run_cmd([COLAB_CLI, "exec", "-s", SESSION_NAME, "--timeout", "600"], stdin_text=setup_code, timeout=660)
    if "SETUP_DONE" not in stdout:
        log(f"Setup failed: {stderr[:200]}")
        return False

    # Upload local runner script
    run_cmd([COLAB_CLI, "upload", "-s", SESSION_NAME, str(REPO_ROOT / "scripts" / "run_exact_pinmymetal_5fold_cv.py"), "/content/DeepMzyme/scripts/run_exact_pinmymetal_5fold_cv.py"], timeout=60)

    # Re-upload Model 3 Fold 0 (and any other completed folds of benchmark_enhanced_gvp_esmc)
    for fold_idx in range(5):
        fold_name = f"benchmark_enhanced_gvp_esmc_fold{fold_idx}"
        local_fold_dir = LOCAL_RUNS_DIR / fold_name
        if local_fold_dir.is_dir() and (local_fold_dir / "test_report.json").exists():
            log(f"Re-uploading completed fold {fold_name} to Colab...")
            tar_path = LOCAL_RUNS_DIR / f"{fold_name}.tar.gz"
            if not tar_path.exists():
                subprocess.run(["tar", "-czf", str(tar_path), "-C", str(LOCAL_RUNS_DIR), fold_name], check=True)
            run_cmd([COLAB_CLI, "upload", "-s", SESSION_NAME, str(tar_path), f"/content/{fold_name}.tar.gz"], timeout=60)
            untar_code = f"import subprocess, os; os.makedirs('/content/runs/benchmark_exact_pinmymetal_5fold', exist_ok=True); subprocess.run(['tar', '-xzf', '/content/{fold_name}.tar.gz', '-C', '/content/runs/benchmark_exact_pinmymetal_5fold'])\n"
            run_cmd([COLAB_CLI, "exec", "-s", SESSION_NAME, "--timeout", "60"], stdin_text=untar_code, timeout=75)

    # Launch benchmark runner detached for benchmark_enhanced_gvp_esmc
    launch_code = """
import subprocess, os
os.makedirs('/content/runs/benchmark_exact_pinmymetal_5fold', exist_ok=True)
cmd = '''nohup python3 -u /content/DeepMzyme/scripts/run_exact_pinmymetal_5fold_cv.py \\
  --models benchmark_enhanced_gvp_esmc \\
  --folds 0 1 2 3 4 \\
  --data-root /content/DeepMzyme_Data/DeepMzyme_Data \\
  --runs-dir /content/runs/benchmark_exact_pinmymetal_5fold \\
  --epochs 50 --batch-size 16 --device cuda > /content/benchmark_5fold_execution.log 2>&1 &'''
subprocess.run(cmd, shell=True, executable='/bin/bash')
print("LAUNCHED")
"""
    code, stdout, stderr = run_cmd([COLAB_CLI, "exec", "-s", SESSION_NAME, "--timeout", "30"], stdin_text=launch_code, timeout=45)
    log("Benchmark runner relaunched successfully!")
    return True


def step() -> None:
    if not is_session_alive():
        log("Session is dead! Initiating automatic recovery with L4 GPU...")
        recovered = recover_session_with_l4()
        if not recovered:
            log("Recovery attempt failed. Retrying in next cycle...")
    else:
        check_and_sync()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Perform one check/sync and exit")
    args = parser.parse_args()

    if args.once:
        step()
        return

    log(f"Starting Colab 5-Fold Watchdog Daemon (interval={CHECK_INTERVAL_SECONDS}s)...")
    while True:
        try:
            step()
        except Exception as exc:
            log(f"Watchdog loop exception: {exc}")

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
