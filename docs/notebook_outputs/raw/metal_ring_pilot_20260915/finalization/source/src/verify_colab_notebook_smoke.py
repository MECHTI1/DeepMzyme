"""Execute the current notebook's exact standalone smoke recipe on CUDA.

Reads notebook cells without editing the notebook. Only filesystem locations,
the Python executable, and output transfer to local disk replace Colab/Drive
plumbing. Scientific parameters come from the owning playbook. Never uses test
inputs or metrics. Every family runs in a fresh process.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import torch

from complete_care_caches import atomic_json, CARE_ROOT
from training.config import parse_args
from training.run import validate_training_configuration


def build_command(root, data, output, family, task, scheme, dataset_id, dataset_sha256):
    notebook = root / "notebooks/DeepMzyme_training_colab.ipynb"
    playbook = root / f"docs/{task.upper()}_TRAINING_PIPELINE_PLAYBOOK.md"
    cells = {cell["id"]: "".join(cell["source"]) for cell in json.loads(notebook.read_text())["cells"]}
    block = re.findall(r"```python\n(.*?)```", playbook.read_text(), flags=re.S)[0]
    block = block.replace('MODEL_PRESET = "Only-GVP"', f'MODEL_PRESET = "{family}"')
    block = block.replace('METAL_LABEL_SCHEME = "four_class"', f'METAL_LABEL_SCHEME = "{scheme}"')
    ns = {"IN_COLAB": False, "NOTEBOOK_START_CWD": root, "DEVICE": "cuda", "Path": Path}
    exec(cells["eb4db512"], ns)
    exec(block, ns)
    ns.update(DRIVE_ROOT=str(output / "drive"), RUNS_DIR=str(output / "runs"),
              ESM_EMBEDDINGS_DIR=str(data / "esm_embeddings"),
              EXTERNAL_FEATURES_ROOT_DIR=str(data / "updated_feature_extraction"),
              RING_FEATURES_DIR=str(data / "RING_features"),
              BUNDLE_FILENAME=dataset_id, BUNDLE_SHA256=dataset_sha256,
              COPY_OUTPUTS_TO_DRIVE=False, DEVICE="cuda")
    exec(cells["ba89d9f5"], ns)
    dataset = data / (CARE_ROOT if task == "ec" else
                      "train_and_test_sets_structures_common_pdbid_70_30_pinmymetal")
    train = dataset / "train"
    csv = train / "final_data_summarazing_table_transition_metals_only_catalytic.csv"
    if not csv.is_file():
        raise FileNotFoundError(csv)
    ns.update(REPO_DIR=root, SRC_DIR=root / "src", TRAIN_DIR=train,
              DRIVE_ROOT_PATH=output / "drive",
              TRAIN_SITE_SUMMARY_CSV=csv, TRAIN_CSV=csv,
              TEST_DIR=output / "unused_test", TEST_SITE_SUMMARY_CSV=output / "unused_test.csv",
              TEST_CSV=output / "unused_test.csv", TRAIN_STRUCTURES=[], TEST_STRUCTURES=[],
              DATA_ROOT=data, DATASET_ROOT=dataset, DRIVE_DATA_DIR=output / "drive_data")
    exec(cells["75e1f97ec96a047f"], ns)
    if len(ns["planned_runs"]) != 1:
        raise ValueError("Smoke recipe must expand to exactly one run per family/target")
    command = list(map(str, ns["planned_runs"][0]["command"]))
    command[0] = sys.executable
    if any(flag in command for flag in ("--run-test-eval", "--test-structure-dir", "--test-summary-csv")):
        raise ValueError("Smoke command must never reference held-out test inputs")
    config = parse_args(command[2:])
    validate_training_configuration(config)
    if config.epochs != 1 or config.run_test_eval or config.device != "cuda":
        raise ValueError("Expected one validation-only CUDA epoch")
    return command, {
        "notebook_sha256": hashlib.sha256(notebook.read_bytes()).hexdigest(),
        "playbook_sha256": hashlib.sha256(playbook.read_bytes()).hexdigest(),
        "recipe": block,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--dataset-id", required=True,
                        help="Exact prepared-data identity; do not label changed caches as the unmodified base bundle")
    parser.add_argument("--dataset-sha256", required=True,
                        help="SHA256 of the bundle or complete prepared-cache manifest")
    parser.add_argument("--task", choices=["ec", "metal"], default="ec")
    parser.add_argument("--metal-label-scheme", choices=["four_class", "six_class"], default="four_class")
    parser.add_argument("--family", choices=["Only-GVP", "Only-ESM", "GVP + late fusion"], required=True)
    parser.add_argument("--execute", action="store_true", help="Actually run the one-epoch command")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this verification")
    probe = torch.randn(32, 32, device="cuda")
    if not torch.isfinite(probe @ probe).all():
        raise RuntimeError("CUDA matrix multiplication failed")
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    command, provenance = build_command(root, args.data_root.resolve(), output,
                                        args.family, args.task, args.metal_label_scheme,
                                        args.dataset_id, args.dataset_sha256)
    report = dict(provenance, command=command, source_commit=args.source_commit,
                  source_file_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in sorted((root / "src").rglob("*.py"))},
                  started_at=datetime.now(timezone.utc).isoformat(),
                  gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                  cuda=torch.version.cuda, python=sys.version, task=args.task, family=args.family,
                  held_out_test_evaluated=False, status="planned",
                  versions={p: version(p) for p in ("torch", "torch-geometric", "numpy", "pandas", "scikit-learn", "optuna")})
    try:
        report["versions"]["esm"] = version("esm")
    except PackageNotFoundError:
        report["versions"]["esm"] = None  # Precomputed embeddings do not require the generation SDK.
    atomic_json(output / "verification.json", report)
    if args.execute:
        with (output / "training.log").open("w") as log:
            result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, PYTHONUNBUFFERED="1"))
        report.update(returncode=result.returncode, finished_at=datetime.now(timezone.utc).isoformat(),
                      status="completed" if result.returncode == 0 else "failed")
        if list(output.rglob("test_report.json")):
            report["status"] = "failed_unexpected_test_report"
        atomic_json(output / "verification.json", report)
        if report["status"] != "completed":
            raise SystemExit(result.returncode or 1)


if __name__ == "__main__":
    main()
