"""Run the authorized fixed-split EC1 grid using existing notebook commands.

No dataset generation, held-out evaluation, HPO, or training-code changes.
Planning, input preflight and execution are separate explicit operations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
import shutil
from dataclasses import replace
from uuid import uuid4
from importlib.metadata import version

import torch

from training.config import parse_args
from training.run import prepare_run, set_seed, validate_training_configuration
from structure_store import resolve_structure_files
from training.labels import parse_structure_identity

FAMILIES = ("Only-GVP", "Only-ESM", "GVP + late fusion")
METRIC = "val_ec_group_level_1_balanced_acc"
DATASET = "CARE_task1_30_clusterRes30_train_test_metallo"
BUNDLE = "DeepMzyme_Data_v12_manifest_exact_common70_nonoverlap_clean30_care30_complete_esm_ring_external.tar.gz"
BUNDLE_SHA = "90c0899829e0ac5ca94a5ef34484b74ca1014d3e9b6b1fba90bd338ee3440dee"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def storage_check(output):
    mount = Path("/content/drive/MyDrive")
    direct_drive = mount.is_dir() and output.resolve().is_relative_to(mount.resolve())
    if not direct_drive:
        receipt = json.loads(Path("/content/ec_persistence_receipt.json").read_text())
        assert receipt["method"] == "verified_archive_transfer" and receipt["drive_probe_verified"]
        assert receipt["local_write_verified"] and receipt["drive_folder_id"]
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w+", dir=output) as f:
        f.write("persistent EC campaign"); f.flush(); f.seek(0)
        assert f.read() == "persistent EC campaign"


def validate(config):
    validate_training_configuration(config)
    assert config.task == "ec" and config.ec_label_depth == 1
    assert config.epochs == 30 and config.batch_size == 4
    assert config.seed in (42, 43) and config.split_seed == 42
    assert config.val_fraction == .15 and config.train_val_split_by == "pdbid"
    assert config.split_stratify_by == "active_targets"
    assert config.ec_group_weighting == "structure_id" and config.ec_class_weight_unit == "group"
    assert config.selection_metric == METRIC and config.device == "cuda"
    assert config.learning_rate in (3e-5, 1e-4) and config.weight_decay == 1e-4
    assert config.require_all_task_classes and config.require_external_features
    assert not config.run_test_eval and config.test_structure_dir is None and config.test_summary_csv is None
    assert not config.prepare_missing_esm_embeddings and not config.prepare_missing_ring_edges
    assert not config.use_ring_edges and not config.use_early_esm and config.ec_contrastive_weight == 0


def plan(root, data, output, source_commit):
    storage_check(output)
    notebook = root / "notebooks/DeepMzyme_training_colab.ipynb"
    playbook = root / "docs/EC_TRAINING_PIPELINE_PLAYBOOK.md"
    cells = {c["id"]: "".join(c["source"]) for c in json.loads(notebook.read_text())["cells"]}
    block = re.findall(r"```python\n(.*?)```", playbook.read_text(), re.S)[0]
    block = block.replace('STANDALONE_PHASE = "smoke"', 'STANDALONE_PHASE = "baseline"')
    dataset = data / DATASET
    train = dataset / "train"
    summary = train / "final_data_summarazing_table_transition_metals_only_catalytic.csv"
    all_runs = []
    for family in FAMILIES:
        ns = {"IN_COLAB": False, "NOTEBOOK_START_CWD": root, "Path": Path}
        exec(cells["eb4db512"], ns)
        recipe = block.replace('MODEL_PRESET = "Only-GVP"', f'MODEL_PRESET = "{family}"')
        exec(recipe, ns)
        ns.update(REPO_ROOT=str(root), DRIVE_ROOT="/content/drive/MyDrive/DeepMzyme",
                  RUNS_DIR=str(output / "runs"), ESM_EMBEDDINGS_DIR=str(data / "esm_embeddings"),
                  EXTERNAL_FEATURES_ROOT_DIR=str(data / "updated_feature_extraction"),
                  RING_FEATURES_DIR=str(data / "RING_features"),
                  BUNDLE_FILENAME=BUNDLE, BUNDLE_SHA256=BUNDLE_SHA,
                  MOUNT_DRIVE=False, COPY_OUTPUTS_TO_DRIVE=False, DEVICE="cuda")
        exec(cells["ba89d9f5"], ns)
        # The independent preflight below checks real inputs. No held-out paths
        # are supplied to notebook command expansion or to training commands.
        ns.update(REPO_DIR=root, SRC_DIR=root / "src", TRAIN_DIR=train,
                  DRIVE_ROOT_PATH=Path(ns["DRIVE_ROOT"]), TRAIN_SITE_SUMMARY_CSV=summary,
                  TEST_DIR=output / "unused_test", TEST_SITE_SUMMARY_CSV=output / "unused_test.csv",
                  TRAIN_STRUCTURES=[], TEST_STRUCTURES=[], DATA_ROOT=data,
                  DATASET_ROOT=dataset, DRIVE_DATA_DIR=output / "unused_drive_data")
        exec(cells["75e1f97ec96a047f"], ns)
        runs = ns["planned_runs"]
        if len(runs) != 4:
            raise ValueError(f"Expected four fresh planned runs for {family}, got {len(runs)}")
        for row in runs:
            command = list(map(str, row["command"]))
            command[0] = sys.executable
            config = parse_args(command[2:]); validate(config)
            all_runs.append(dict(family=family, lr=config.learning_rate, seed=config.seed,
                                 command=command, env=row.get("env", {}), run_dir=row["run_dir"]))
    all_runs.sort(key=lambda r: (FAMILIES.index(r["family"]), r["lr"], r["seed"]))
    assert len({(r["family"], r["lr"], r["seed"]) for r in all_runs}) == 12
    manifest = dict(source_commit=source_commit, bundle=BUNDLE, bundle_sha256=BUNDLE_SHA,
                    notebook_sha256=digest(notebook), playbook_sha256=digest(playbook),
                    source_files={str(p.relative_to(root)): digest(p) for p in sorted((root / "src").rglob("*.py"))},
                    task="ec", metric=METRIC, held_out_evaluation=False, runs=all_runs)
    receipt = Path("/content/ec_persistence_receipt.json")
    if receipt.is_file(): manifest["persistent_storage"] = json.loads(receipt.read_text())
    save(output / "campaign_plan.json", manifest)
    (output / "commands.txt").write_text("\n\n".join(shlex.join(r["command"]) for r in all_runs) + "\n")
    with (output / "run_matrix.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["family", "lr", "seed", "epochs", "run_dir"])
        writer.writeheader()
        writer.writerows({**{k: r[k] for k in ("family", "lr", "seed", "run_dir")}, "epochs": 30} for r in all_runs)
    print("Exact authorized grid saved:", output / "campaign_plan.json", flush=True)


def assert_identity(summary, reference):
    assert summary["n_train_pockets"] == 993 and summary["n_val_pockets"] == 175
    assert summary["n_train_ec_groups"] == 751 and summary["n_val_ec_groups"] == 42
    assert {str(k): v for k, v in summary["ec_labels"].items()} == {str(k): v for k, v in reference["ec_labels"].items()}
    for part in ("train", "validation"):
        assert summary["retained_split_identity"][part]["examples"] == reference["retained_split_identity"][part]["examples"], part
    for field in ("ec_group_weighting", "ec_class_weight_unit", "ec_group_metric_mode", "eligibility"):
        assert summary[field] == reference[field], field


def audit_training_cache(data, path):
    """Check precomputed training caches without importing the generation SDK."""
    from training.esm_feature_loading import embedding_path_candidates
    from training.runtime_preparation import updated_external_feature_path_candidates
    embeddings = [p for p in embedding_path_candidates(data / "esm_embeddings", path) if p.is_file()]
    assert embeddings, path.name
    files = []
    for embedding in embeddings:
        payload = torch.load(embedding, map_location="cpu", weights_only=True)
        tensor = payload["embeddings"]
        assert tensor.ndim == 2 and tensor.shape[1] == 960 and torch.isfinite(tensor).all(), embedding
        sidecar = Path(str(embedding) + ".json")
        metadata = json.loads(sidecar.read_text())
        assert metadata["esm_model_name"] == "esmc_300m" and metadata["embedding_dim"] == 960
        files.extend((embedding, sidecar))
    external = next(p for p in updated_external_feature_path_candidates(
        path, structure_root=data / DATASET / "train", external_features_root_dir=data / "updated_feature_extraction") if p.is_file())
    payload = json.loads(external.read_text())
    assert payload["tooling"]["pka"] == "propka" and payload["residues"], external
    assert all(math.isfinite(float(value)) for residue in payload["residues"] for value in residue["features"].values())
    files.append(external)
    return [dict(path=str(p.relative_to(data)), bytes=p.stat().st_size, sha256=digest(p)) for p in files]


def preflight(root, data, output, reference_path):
    storage_check(output)
    assert torch.cuda.is_available()
    x = torch.ones(32, device="cuda", requires_grad=True)
    x.square().sum().backward(); torch.cuda.synchronize()
    plan_data = json.loads((output / "campaign_plan.json").read_text())
    assert json.loads(Path("/content/v12_ready.json").read_text())["sha256"] == BUNDLE_SHA
    train = data / DATASET / "train"
    paths = resolve_structure_files(train, recursive_legacy_scan=False)
    test_paths = resolve_structure_files(data / DATASET / "test", recursive_legacy_scan=False)
    assert not ({parse_structure_identity(p.stem)[0] for p in paths} &
                {parse_structure_identity(p.stem)[0] for p in test_paths})
    # Existing cache audit only; no generation, and training-side structures only.
    cached_audit = Path("/content/ec_training_cache_audit.json")
    if cached_audit.is_file():
        audit = json.loads(cached_audit.read_text())
        assert audit["bundle_sha256"] == BUNDLE_SHA and audit["structures"] == len(paths) and not audit["failures"]
        audit_files = audit["files"]
        for entry in audit_files:
            assert digest(data / entry["path"]) == entry["sha256"], entry["path"]
        print("Reused training cache audit after checking every cache hash", flush=True)
    else:
        audit_files = []
        for i, path in enumerate(paths, 1):
            audit_files.extend(audit_training_cache(data, path))
            if i % 100 == 0: print(f"Audited {i}/{len(paths)} training structures", flush=True)
    save(output / "training_cache_audit.json", dict(structures=len(paths), files=audit_files, failures=[]))
    # Exercise full ESM + graph loading without running an epoch.
    row = next(r for r in plan_data["runs"] if r["family"] == "GVP + late fusion")
    config = parse_args(row["command"][2:]); validate(config)
    config = replace(config, runs_dir=output / "preflight_runs", run_name=f"readiness_{uuid4().hex[:8]}")
    set_seed(config.seed, deterministic=config.deterministic)
    prepared = prepare_run(config)
    reference = json.loads(reference_path.read_text())
    assert_identity(prepared.dataset_summary, reference)
    assert next(prepared.model.parameters()).is_cuda
    save(output / "expected_split.json", prepared.dataset_summary)
    save(output / "readiness.json", dict(plan_sha256=digest(output / "campaign_plan.json"),
         gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
         python=sys.version, compiled_architectures=torch.cuda.get_arch_list(),
         versions={name: version(name) for name in ("torch", "torch-geometric", "numpy", "pandas", "scikit-learn", "optuna")},
         cuda_forward_backward=bool(x.grad.is_cuda), persistent_output=str(output),
         training_structures=len(paths), train_pockets=993, val_pockets=175,
         train_groups=751, val_groups=42, matches_successful_smoke=True,
         held_out_evaluation=False, status="passed"))
    print("READINESS PASSED; no epoch or held-out evaluation executed", flush=True)


def summarize(output):
    plan_data = json.loads((output / "campaign_plan.json").read_text())
    reference = json.loads((output / "expected_split.json").read_text())
    rows = []
    for run in plan_data["runs"]:
        directory = Path(run["run_dir"])
        if not (directory / "run_metadata.json").is_file(): continue
        meta = json.loads((directory / "run_metadata.json").read_text())
        assert (directory / "best_model_checkpoint.pt").is_file()
        assert (directory / "last_model_checkpoint.pt").is_file()
        assert meta["test_report"] is None and not (directory / "test_report.json").exists()
        assert_identity(json.loads((directory / "dataset_summary.json").read_text()), reference)
        with (directory / "epoch_metrics.csv").open() as f:
            assert len(list(csv.DictReader(f))) == 30
        records = json.loads((directory / "run_config.json").read_text())["history"]
        assert len(records) == 30
        record = next(r for r in records if int(r["epoch"]) == meta["selected_checkpoint_epoch"])
        score = float(record[METRIC]); assert math.isfinite(score)
        recalls = record["val_ec_group_per_class_recall"]
        rows.append(dict(family=run["family"], lr=run["lr"], seed=run["seed"],
                         selected_epoch=meta["selected_checkpoint_epoch"], balanced_accuracy=score,
                         per_class_recall=recalls, run_dir=str(directory)))
    groups = []
    for family in FAMILIES:
        for lr in (3e-5, 1e-4):
            pair = [r for r in rows if r["family"] == family and r["lr"] == lr]
            if len(pair) != 2: continue
            values = [r["balanced_accuracy"] for r in pair]
            groups.append(dict(family=family, lr=lr, seeds=[r["seed"] for r in pair],
                               mean=statistics.mean(values), sd=statistics.stdev(values), minimum=min(values),
                               per_class_recall_mean={k: statistics.mean(r["per_class_recall"][k] for r in pair)
                                                      for k in pair[0]["per_class_recall"]}))
    save(output / "validation_results.json", dict(completed_runs=len(rows), runs=rows, family_lr_summary=groups,
                                                  evidence="initial_fixed_split_two_seeds", promoted=False))
    return rows


def execute(root, output, run_index=None):
    storage_check(output)
    ready = json.loads((output / "readiness.json").read_text())
    assert ready["status"] == "passed" and ready["plan_sha256"] == digest(output / "campaign_plan.json")
    plan_data = json.loads((output / "campaign_plan.json").read_text())
    if plan_data.get("persistent_storage", {}).get("method") == "verified_archive_transfer":
        assert run_index is not None, "Transfer mode requires one run at a time and a verified archive receipt before advancing"
        if run_index > 1:
            previous = json.loads((output / "transfer_receipts" / f"run_{run_index - 1:02d}.json").read_text())
            assert previous["drive_verified"] and previous["local_sha256_verified"]
    for name, expected in plan_data["source_files"].items(): assert digest(root / name) == expected, name
    for index, run in enumerate(plan_data["runs"], 1):
        if run_index is not None and index != run_index: continue
        directory = Path(run["run_dir"])
        if (directory / "run_metadata.json").is_file():
            summarize(output)
            continue
        config = parse_args(run["command"][2:]); validate(config)
        from training.runtime_preparation import discover_missing_updated_external_features
        assert not discover_missing_updated_external_features(
            resolve_structure_files(config.structure_dir, recursive_legacy_scan=False),
            structure_root=Path(config.structure_dir), external_features_root_dir=Path(config.external_features_root_dir))
        save(output / "campaign_state.json", dict(status="running", index=index, total=12,
                                                  family=run["family"], lr=run["lr"], seed=run["seed"]))
        print(f"START {index}/12 {run['family']} lr={run['lr']} seed={run['seed']}", flush=True)
        log_path = output / "logs" / f"run_{index:02d}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log:
            result = subprocess.run(run["command"], cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, **{k: str(v) for k, v in run["env"].items()}, "PYTHONUNBUFFERED": "1"})
        if result.returncode:
            save(output / "campaign_state.json", dict(status="failed", index=index, returncode=result.returncode,
                                                      log=str(log_path)))
            raise SystemExit(result.returncode)
        shutil.copy2(log_path, directory / "training.log")
        summarize(output)
        print(f"COMPLETE {index}/12", flush=True)
    rows = summarize(output)
    save(output / "campaign_state.json", dict(status="completed" if len(rows) == 12 else "awaiting_archive_transfer",
                                              completed_runs=len(rows), held_out_evaluation=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["plan", "preflight", "execute", "summarize"])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--smoke-reference", type=Path, required=True)
    parser.add_argument("--run-index", type=int, choices=range(1, 13), help="Execute one planned run, then transfer its artifacts")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.phase == "plan": plan(root, args.data_root, args.output_dir, args.source_commit)
    elif args.phase == "preflight": preflight(root, args.data_root, args.output_dir, args.smoke_reference)
    elif args.phase == "execute": execute(root, args.output_dir, args.run_index)
    else: summarize(args.output_dir)


if __name__ == "__main__":
    main()
