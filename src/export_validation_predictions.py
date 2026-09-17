"""Replay frozen selected checkpoints on their exact saved validation membership.

The reuse ledger has ``schema_version: 1`` and a ``runs`` list. Each run needs
run_id, task, family, run_dir, source_root, source_files (relative source path
to SHA256), path_map (old absolute prefix to local prefix), and expected_sha256
for run_config.json and best_model_checkpoint.pt. dataset_summary_path defaults
to run_dir/dataset_summary.json. Ledger paths are relative to the ledger file.

Every run executes in a separate process against its verified source snapshot.
No training preparation, optimizer, feature generator, or held-out evaluator is
called. Saved normalization and class-weight buffers are loaded unchanged.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def resolve_path(value, base):
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def remap_path(value, path_map):
    """Remap whole path prefixes; never treat a substring as a directory."""
    if value is None:
        return None
    original = Path(value)
    for old, new in sorted(path_map.items(), key=lambda item: len(item[0]), reverse=True):
        try:
            relative = original.relative_to(old)
        except ValueError:
            continue
        return (Path(new) / relative).resolve()
    return original.resolve()


def validate_config(config):
    require(config.get("task") in {"metal", "ec"}, "Only standalone metal or EC checkpoints are supported")
    for name in ("run_test_eval", "allow_train_loss_test_eval_debug", "allow_final_refit_test_eval", "controlled_ec_auxiliary"):
        require(not config.get(name), f"Forbidden replay configuration: {name}")
    for name in ("test_structure_dir", "test_summary_csv", "final_test_checkpoint_paths", "final_test_source_run_dirs"):
        require(not config.get(name), f"Held-out paths are forbidden: {name}")
    require(config.get("train_val_split_by", config.get("split_by")) == "pdbid", "Replay requires saved protein/PDB grouping")
    require(config.get("selection_metric", "").startswith("val_"), "Checkpoint must have validation selection")
    require(config.get("ec_label_depth") == 1 or config["task"] == "metal", "This exporter supports EC depth 1")
    for key in ("structure_dir", "summary_csv"):
        path = Path(config[key])
        require("train" in path.parts, f"External-training membership must be explicit in {key}")
        require(not any(part.lower() in {"test", "held_out", "held-out"} for part in path.parts),
                f"Held-out path is forbidden: {key}")


def verify_membership(summary):
    identities = summary["retained_split_identity"]
    groups = {}
    for split in ("train", "validation"):
        item = identities[split]
        examples = item["examples"]
        require(bool(examples), f"Empty saved {split} membership")
        require(item["ordered_examples_sha256"] == fingerprint(examples), f"Corrupt saved {split} membership digest")
        require(item["n_examples"] == len(examples), f"Wrong saved {split} count")
        keys = [(row["structure_id"], row["pocket_id"]) for row in examples]
        require(len(keys) == len(set(keys)), f"Duplicate {split} pocket identity")
        require(len({row["pocket_id"] for row in examples}) == len(examples),
                f"Pocket IDs are not globally unique in {split}; a composite export ID is required")
        groups[split] = {row["group"] for row in examples}
    require(not groups["train"] & groups["validation"], "Saved training/validation groups overlap")
    return identities


def verify_source(source_root, source_files):
    require(isinstance(source_files, dict) and source_files, "Frozen source hashes are required")
    require("src/training/run.py" in source_files, "Frozen training/run.py must be captured")
    for relative, expected in source_files.items():
        path = (source_root / relative).resolve()
        require(path.is_relative_to(source_root), "Source manifest path escapes source root")
        require(path.is_file() and digest(path) == expected, f"Frozen source hash mismatch: {relative}")
    return {"mode": "verified_frozen_source", "source_root": str(source_root),
            "source_manifest_sha256": fingerprint(source_files), "verified_files": len(source_files)}


def verify_bound_run_files(entry, base):
    """Validate every supplied run artifact, including metadata and membership."""
    hashes = entry["expected_sha256"]
    allowed = {"run_config.json", "best_model_checkpoint.pt", "dataset_summary.json", "run_metadata.json"}
    require({"run_config.json", "best_model_checkpoint.pt"} <= hashes.keys(), "Missing required run artifact hashes")
    require(hashes.keys() <= allowed, "Unsupported run artifact in replay ledger")
    run_dir = resolve_path(entry["run_dir"], base)
    for name, expected in hashes.items():
        override = entry.get("checkpoint_path") if name == "best_model_checkpoint.pt" else (
            entry.get("dataset_summary_path") if name == "dataset_summary.json" else None)
        path = resolve_path(override, base) if override else run_dir / name
        require(digest(path) == expected, f"Artifact hash mismatch: {name}")


def factory_kwargs(source_path, config, checkpoint, metal_labels, ec_labels):
    """Read the frozen factory call without invoking its training preparation.

    A tiny expression interpreter accepts only configuration access, constants,
    class counts, task predicates, and saved weight buffers. An unfamiliar new
    construction expression fails instead of silently adopting current defaults.
    """
    tree = ast.parse(Path(source_path).read_text(encoding="utf-8"))
    prepare = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "prepare_run")
    calls = [node for node in ast.walk(prepare) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "build_pocket_classifier"]
    require(len(calls) == 1 and not calls[0].args, "Frozen model construction is ambiguous")
    names = {"METAL_TARGET_LABELS": metal_labels,
             **{name: checkpoint["model_state_dict"].get(name) for name in
                ("metal_class_weights", "metal_collapsed4_class_weights", "ec_class_weights")}}

    def value(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name) and node.id in names:
            return names[node.id]
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "config":
                require(node.attr in config, f"Frozen factory field absent from saved config: {node.attr}")
                return config[node.attr]
            if node.value.id == "load_result" and node.attr == "ec_index_to_label":
                return ec_labels
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.NotEq):
            return value(node.left) != value(node.comparators[0])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            functions = {"len": len, "max": max,
                         "task_predicts_metal": lambda task: task == "metal",
                         "task_predicts_ec": lambda task: task == "ec"}
            if node.func.id in functions:
                return functions[node.func.id](*(value(arg) for arg in node.args))
        raise ValueError(f"Unsupported frozen model expression: {ast.dump(node)}")

    require(all(keyword.arg is not None for keyword in calls[0].keywords), "Expanded model kwargs are unsupported")
    return {keyword.arg: value(keyword.value) for keyword in calls[0].keywords}


def compare_metrics(actual, selected, selection_metric):
    """Check all saved task validation metrics, including their confusion table."""
    require(selection_metric in actual and selection_metric in selected, "Missing selected validation metric")
    prefix = "val_metal_" if selection_metric.startswith("val_metal_") else "val_ec_"
    required_matrix = "val_metal_confusion_matrix" if prefix == "val_metal_" else "val_ec_group_confusion_matrix"
    require(required_matrix in actual and required_matrix in selected, "Missing saved whole-validation confusion matrix")

    def same(left, right):
        if isinstance(left, dict) and isinstance(right, dict):
            return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
        if isinstance(left, list) and isinstance(right, list):
            return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=0, abs_tol=1e-8)
        return left == right

    checked = [key for key in actual if key.startswith(prefix) and key in selected]
    mismatches = [key for key in checked if not same(actual[key], selected[key])]
    require(not mismatches, f"Validation reproduction failed for: {mismatches}")
    return {key: actual[key] for key in checked}


def prediction_rows(logits, targets, examples, labels):
    import torch

    require(tuple(logits.shape) == (len(examples), len(labels)), "Logit dimensions differ from saved membership")
    require(tuple(targets.shape) == (len(examples),), "Target dimensions differ from saved membership")
    require(bool(torch.isfinite(logits).all()), "Nonfinite prediction logits")
    probabilities = logits.softmax(-1)
    rows = []
    for index, example in enumerate(examples):
        target = int(targets[index])
        require(0 <= target < len(labels), "Invalid target index")
        rows.append({**example, "target": target, "prediction": int(logits[index].argmax()),
                     **{f"logit_{label_id}": float(logits[index, label_id]) for label_id in range(len(labels))},
                     **{f"probability_{label_id}": float(probabilities[index, label_id]) for label_id in range(len(labels))}})
    return rows


def aggregate_ec_logits(logits, targets, group_ids, group_names):
    import torch

    values, truths, examples = [], [], []
    for group_index in sorted(set(group_ids.tolist())):
        require(group_index >= 0 and group_index in group_names, "Missing EC group identity")
        mask = group_ids == group_index
        labels = targets[mask].unique()
        require(len(labels) == 1, "Conflicting EC labels within a protein; cannot silently exclude it")
        values.append(logits[mask].mean(0))
        truths.append(int(labels[0]))
        name = group_names[group_index]
        examples.append({"example_id": name, "group_id": name, "n_pockets": int(mask.sum())})
    return torch.stack(values), torch.tensor(truths), examples


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    require(bool(rows), "Cannot export an empty prediction table")
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"path": str(Path(path).resolve()), "sha256": digest(path)}


def install_read_only_guard(output_dir):
    """Block Python-level cache writes and external execution during replay."""
    input_roots, input_reads = [], set()

    def guard(event, args):
        if event in {"subprocess.Popen", "os.system", "socket.connect", "socket.bind"}:
            raise PermissionError(f"External operation forbidden during replay: {event}")
        if event in {"os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.truncate", "os.symlink", "os.link", "os.rename"}:
            checked = args[:2] if event in {"os.rename", "os.link", "os.symlink"} else args[:1]
            for item in checked:
                if isinstance(item, (str, bytes, os.PathLike)) and not Path(os.fsdecode(item)).resolve().is_relative_to(output_dir):
                    raise PermissionError(f"Filesystem mutation outside sidecars: {event}")
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            mode = args[1] or ""
            flags = args[2] or 0
            writing = any(token in mode for token in "wax+") or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
            if writing:
                if not path.is_relative_to(output_dir):
                    raise PermissionError(f"Replay write outside sidecar directory: {path}")
            elif any(path.is_relative_to(root) for root in input_roots):
                input_reads.add(path)
    sys.addaudithook(guard)
    return input_roots, input_reads


def run_worker(ledger_path, run_id, output_dir, device):
    executed_source_sha256 = digest(__file__)
    ledger = read_json(ledger_path)
    entries = [row for row in ledger["runs"] if row["run_id"] == run_id]
    require(len(entries) == 1, f"Missing or duplicate run ID: {run_id}")
    entry = entries[0]
    base = ledger_path.parent
    counts_freeze = ledger.get("counts_freeze")
    if counts_freeze:
        require(digest(resolve_path(counts_freeze["path"], base)) == counts_freeze["sha256"],
                "Pre-prediction counts freeze hash mismatch")
    run_dir = resolve_path(entry["run_dir"], base)
    source_root = resolve_path(entry["source_root"], base)
    source_files = entry["source_files"]
    source_receipt = verify_source(source_root, source_files)
    config_path = run_dir / "run_config.json"
    checkpoint_path = resolve_path(entry["checkpoint_path"], base) if entry.get("checkpoint_path") else run_dir / "best_model_checkpoint.pt"
    require(checkpoint_path.name == "best_model_checkpoint.pt", "Only the saved selected checkpoint is supported")
    verify_bound_run_files(entry, base)
    payload = read_json(config_path)
    config = payload["config"]
    validate_config(config)
    require(entry["task"] == config["task"], "Ledger task differs from saved task")
    summary = payload["dataset_summary"]
    identities = verify_membership(summary)
    if entry.get("dataset_summary_path"):
        require(read_json(resolve_path(entry["dataset_summary_path"], base)) == summary,
                "Standalone dataset summary differs from saved run configuration")
    require(not output_dir.exists(), f"Refusing to overwrite prediction sidecars: {output_dir}")
    require(not output_dir.is_relative_to(run_dir), "Output must be separate from the original run")
    output_dir.mkdir(parents=True)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source_root / "src"))
    os.environ["DEEPGM_METAL_LABEL_SCHEME"] = str(config["metal_label_scheme"])
    os.environ["OMP_NUM_THREADS"] = "1"
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch
    from torch_geometric.loader import DataLoader
    from label_schemes import configure_active_metal_label_scheme, METAL_TARGET_LABELS
    from model_variants import build_pocket_classifier
    from training.labels import assign_ec_targets
    from training.site_filter import resolve_allowed_site_metal_labels
    from training.structure_loading import find_structure_files, load_structure_pockets
    from training.graph_dataset import PocketGraphDataset, build_graph_data_list
    from training.splits import assign_ec_group_metadata, retained_split_identity
    from training.run import normalization_stats_from_payload, metrics_from_predictions, to_jsonable
    from training.loop import evaluate_epoch_with_predictions

    torch.set_num_threads(1)
    configure_active_metal_label_scheme(config["metal_label_scheme"])
    # Imports may initialize library caches; all scientific I/O below is guarded.
    input_roots, input_reads = install_read_only_guard(output_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(to_jsonable(checkpoint["config"]) == config, "Checkpoint configuration differs from saved run")
    require(to_jsonable(checkpoint["dataset_summary"]) == summary, "Checkpoint dataset summary differs")
    require(to_jsonable(checkpoint["normalization_stats"]) == payload["normalization_stats"], "Saved normalization differs")
    require(checkpoint["selection_metric"] == config["selection_metric"], "Checkpoint selection rule differs")
    selected = [row for row in payload["history"] if row["epoch"] == checkpoint["epoch"]]
    require(len(selected) == 1, "Selected epoch missing or duplicated in saved history")
    require(payload["selected_checkpoint_epoch"] == checkpoint["epoch"], "Checkpoint is not the recorded selected epoch")
    require(payload.get("test_report") is None, "A held-out report is outside validation-only replay")
    require(not checkpoint.get("is_final_refit", False), "Final refits are outside validation replay")
    ec_labels = {int(k): str(v) for k, v in checkpoint["ec_labels"].items()}
    metal_labels = {int(k): str(v) for k, v in checkpoint["metal_labels"].items()}
    require(metal_labels == METAL_TARGET_LABELS, "Active metal label vocabulary differs from checkpoint")
    labels = metal_labels if config["task"] == "metal" else ec_labels
    require(list(sorted(labels)) == list(range(len(labels))), "Noncontiguous label vocabulary")

    path_map = entry.get("path_map", {})
    paths = {key: remap_path(config.get(key), path_map) for key in
             ("structure_dir", "summary_csv", "esm_embeddings_dir", "ring_features_dir", "external_features_root_dir")}
    # Only-GVP historically had no ESM files in its implicit default directory.
    # Retain that observed absence; using the available ESM cache changes inputs.
    recorded_preparation = summary.get("runtime_preparation", {})
    if paths["esm_embeddings_dir"] is None:
        require(not config["require_esm_embeddings"]
                and recorded_preparation.get("esm_embedding_metadata", {}).get("embedding_files_found") == 0,
                "Implicit ESM cache has no certified empty-cache provenance")
        paths["esm_embeddings_dir"] = output_dir / "empty_saved_esm_cache"
        paths["esm_embeddings_dir"].mkdir()
    for key in ("structure_dir", "summary_csv", "esm_embeddings_dir", "external_features_root_dir"):
        require(paths[key] is not None and paths[key].exists(), f"Existing explicit data/cache path required: {key}")
    validate_config({**config, **{key: str(value) if value is not None else None for key, value in paths.items()}})
    input_roots.extend(path for path in paths.values() if path is not None)
    allowed_sites = resolve_allowed_site_metal_labels(paths["summary_csv"])
    structure_files = find_structure_files(paths["structure_dir"])
    structure_lookup = {}
    for path in structure_files:
        require(path.stem not in structure_lookup, f"Ambiguous structure filename: {path.stem}")
        structure_lookup[path.stem] = path
    require(all(row["structure_id"] in structure_lookup for split in identities.values() for row in split["examples"]),
            "Saved development membership is absent from the external-training structure manifest")
    examples = identities["validation"]["examples"]
    by_structure = {}
    for example in examples:
        by_structure.setdefault(example["structure_id"], []).append(example)
    pockets_by_key = {}
    source_inputs = []
    for structure_id, expected in by_structure.items():
        require(structure_id in structure_lookup, f"Validation structure absent from external training membership: {structure_id}")
        path = structure_lookup[structure_id]
        source_inputs.append({"structure_id": structure_id, "path": str(path.resolve()), "sha256": digest(path)})
        loaded, _, _ = load_structure_pockets(
            structure_path=path, structure_root=paths["structure_dir"], allowed_site_metal_labels=allowed_sites,
            esm_dim=config["esm_dim"], embeddings_dir=paths["esm_embeddings_dir"],
            require_esm_embeddings=config["require_esm_embeddings"], ring_features_dir=paths["ring_features_dir"],
            feature_root_dir=paths["external_features_root_dir"], external_feature_source=config["external_feature_source"],
            require_external_features=config["require_external_features"], unsupported_metal_policy=config["unsupported_metal_policy"],
            ec_label_depth=config["ec_label_depth"])
        assign_ec_targets(loaded, depth=config["ec_label_depth"], token_to_index={value: key for key, value in ec_labels.items()})
        wanted = {(row["structure_id"], row["pocket_id"]) for row in expected}
        for pocket in loaded:
            key = (pocket.structure_id, pocket.pocket_id)
            if key in wanted:
                require(key not in pockets_by_key, f"Duplicate loaded validation pocket: {key}")
                pockets_by_key[key] = pocket
    keys = [(row["structure_id"], row["pocket_id"]) for row in examples]
    require(set(pockets_by_key) == set(keys), "Loaded validation pockets differ from saved membership")
    pockets = [pockets_by_key[key] for key in keys]
    require(retained_split_identity(pockets, "pdbid") == identities["validation"],
            "Loaded validation identities, ordering, or labels differ from saved membership")
    if config["task"] == "ec":
        assign_ec_group_metadata(pockets, weighting_mode=config["ec_group_weighting"])
    graph_options = {key: config[key] for key in ("esm_dim", "edge_radius", "use_ring_edges", "require_ring_edges", "node_feature_set", "omit_node_features", "metal_node_mode")}
    if "shell_role_source" in config:
        graph_options["shell_role_source"] = config["shell_role_source"]
    graphs = build_graph_data_list(pockets, **graph_options)
    dataset = PocketGraphDataset(pockets, precomputed_data=graphs,
                                 normalization_stats=normalization_stats_from_payload(checkpoint["normalization_stats"]), **graph_options)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, drop_last=False, num_workers=0)
    kwargs = factory_kwargs(source_root / "src/training/run.py", config, checkpoint, metal_labels, ec_labels)
    model = build_pocket_classifier(**kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    predictions = evaluate_epoch_with_predictions(model, loader, device=device)
    prefix = config["task"]
    require(predictions[f"{prefix}_y"].tolist() == [int(row[f"y_{prefix}"]) for row in examples], "Inference targets or ordering differ")
    metrics = metrics_from_predictions(predictions, "val", task=config["task"], ec_label_map=ec_labels, ec_label_depth=config["ec_label_depth"])
    verified_metrics = compare_metrics(metrics, selected[0], config["selection_metric"])
    exported_examples = [{"example_id": row["pocket_id"], "group_id": row["group"],
                          "structure_id": row["structure_id"], "pocket_id": row["pocket_id"]} for row in examples]
    rows = prediction_rows(predictions[f"{prefix}_logits"], predictions[f"{prefix}_y"], exported_examples, labels)
    result = {key: entry[key] for key in ("run_id", "task", "family")}
    result.update(seed=int(config["seed"]), fold_id=entry.get("fold_id", "fixed"), validation_only=True,
                  reproduction_passed=True, target_vocabulary=[labels[index] for index in range(len(labels))])
    if counts_freeze:
        result["counts_freeze_sha256"] = counts_freeze["sha256"]
    result["pocket_predictions"] = write_csv(output_dir / "pocket_predictions.csv", rows)
    if config["task"] == "ec":
        names = {pocket.metadata["ec_group_id"]: row["group"] for pocket, row in zip(pockets, examples)}
        require(all(names[pocket.metadata["ec_group_id"]] == row["group"] for pocket, row in zip(pockets, examples))
                and len(set(names.values())) == len(names),
                "EC metric groups do not map one-to-one to protein remoteness groups")
        values, truths, group_examples = aggregate_ec_logits(predictions["ec_logits"], predictions["ec_y"], predictions["ec_group_id"], names)
        result["group_predictions"] = write_csv(output_dir / "group_predictions.csv", prediction_rows(values, truths, group_examples, labels))
    imported_sources = {}
    for module in list(sys.modules.values()):
        module_file = getattr(module, "__file__", None)
        if module_file and Path(module_file).resolve().is_relative_to(source_root / "src"):
            relative = Path(module_file).resolve().relative_to(source_root).as_posix()
            if relative.endswith(".py"):
                require(relative in source_files and digest(module_file) == source_files[relative], f"Imported source is not frozen: {relative}")
                imported_sources[relative] = source_files[relative]
    input_file_hashes = [{"path": str(path), "sha256": digest(path)} for path in sorted(input_reads) if path.is_file()]
    require(digest(__file__) == executed_source_sha256,
            "Exporter source changed during replay; refusing to certify its source identity")
    receipt = {"run_id": run_id, "task": config["task"], "validation_only": True, "reproduction_passed": True,
               "seed": result["seed"], "fold_id": result["fold_id"], "family": result["family"],
               "training_performed": False, "optimizer_created": False, "checkpoint_reselection": False,
               "feature_generation": False, "normalization_refitted": False, "checkpoint_sha256": digest(checkpoint_path),
               "run_config_sha256": digest(config_path), "reuse_ledger_sha256": digest(ledger_path),
               "selected_epoch": checkpoint["epoch"], "selection_metric": config["selection_metric"],
               "normalization_sha256": fingerprint(payload["normalization_stats"]),
               "validation_membership_sha256": identities["validation"]["ordered_examples_sha256"],
               "train_membership_sha256": identities["train"]["ordered_examples_sha256"],
               "source": source_receipt, "imported_source_files": imported_sources,
               "input_structure_hashes": source_inputs, "verified_whole_validation_metrics": verified_metrics,
               "input_file_hashes": input_file_hashes,
               "python": sys.version, "torch": str(torch.__version__), "device": device,
               "ec_aggregation": "mean_logits" if config["task"] == "ec" else None,
               "reproduction_scope": "saved whole-validation metrics and confusion matrices; archived logits were unavailable",
               "prediction_artifacts": {key: result[key] for key in ("pocket_predictions", "group_predictions") if key in result},
               "target_vocabulary": result["target_vocabulary"],
               "path_map": path_map, "exporter_sha256": executed_source_sha256,
               "exporter_source": {"path": str(Path(__file__).resolve()), "sha256": executed_source_sha256}}
    receipt["resolved_replay_paths"] = {key: str(value) if value is not None else None for key, value in paths.items()}
    if counts_freeze:
        receipt["counts_freeze_sha256"] = counts_freeze["sha256"]
    write_json(output_dir / "receipt.json", receipt)
    result["receipt"] = {"path": str(output_dir / "receipt.json"), "sha256": digest(output_dir / "receipt.json")}
    write_json(output_dir / "manifest_entry.json", result)
    print(json.dumps({"run_id": run_id, "status": "verified", "validation_examples": len(rows)}), flush=True)


def verified_completed_export(entry, ledger_path, output_dir):
    """Reopen completed sidecars only after binding every scientific input."""
    manifest_path, receipt_path = output_dir / "manifest_entry.json", output_dir / "receipt.json"
    require(manifest_path.is_file() and receipt_path.is_file(),
            f"Incomplete replay directory requires operator archival before retry: {output_dir}")
    result, receipt = read_json(manifest_path), read_json(receipt_path)
    ledger = read_json(ledger_path)
    base = ledger_path.parent
    require(result.get("receipt", {}).get("sha256") == digest(receipt_path), "Completed receipt hash changed")
    require(resolve_path(result["receipt"]["path"], base) == receipt_path, "Completed receipt path changed")
    for record in (result, receipt):
        require(record.get("validation_only") is True and record.get("reproduction_passed") is True,
                "Completed export lacks successful reproduction")
        for key in ("run_id", "task", "family", "seed", "fold_id"):
            expected = entry.get(key, "fixed" if key == "fold_id" else None)
            require(record.get(key) == expected, f"Completed export {key} differs from the ledger")
        expected_freeze = ledger.get("counts_freeze", {}).get("sha256")
        require(record.get("counts_freeze_sha256") == expected_freeze, "Completed counts freeze changed")
    require(receipt.get("reuse_ledger_sha256") == digest(ledger_path), "Completed export used a different ledger")
    if receipt.get("exporter_source"):
        artifact = receipt["exporter_source"]
        require(digest(resolve_path(artifact["path"], base)) == artifact["sha256"] == receipt["exporter_sha256"],
                "Completed exporter source snapshot changed")
    if ledger.get("counts_freeze"):
        freeze = ledger["counts_freeze"]
        require(digest(resolve_path(freeze["path"], base)) == freeze["sha256"], "Counts freeze file changed")
    run_dir = resolve_path(entry["run_dir"], base)
    verify_bound_run_files(entry, base)
    checkpoint = resolve_path(entry["checkpoint_path"], base) if entry.get("checkpoint_path") else run_dir / "best_model_checkpoint.pt"
    for key, path, field in (("run_config.json", run_dir / "run_config.json", "run_config_sha256"),
                             ("best_model_checkpoint.pt", checkpoint, "checkpoint_sha256")):
        require(digest(path) == entry["expected_sha256"][key] == receipt.get(field),
                f"Completed export scientific artifact changed: {key}")
    source = verify_source(resolve_path(entry["source_root"], base), entry["source_files"])
    require(receipt.get("source") == source, "Completed export frozen source identity changed")
    for name in ("pocket_predictions", "group_predictions"):
        if name == "group_predictions" and entry["task"] != "ec":
            continue
        require(name in result and receipt.get("prediction_artifacts", {}).get(name) == result[name],
                "Completed predictions are not bound to their receipt")
        artifact = result[name]
        require(digest(resolve_path(artifact["path"], base)) == artifact["sha256"], "Completed prediction hash changed")
    for item in receipt.get("input_file_hashes", []) + receipt.get("input_structure_hashes", []):
        require(digest(Path(item["path"])) == item["sha256"], f"Completed replay input changed: {item['path']}")
    return result


def persist_manifest_index(output, manifest):
    """Keep immutable index generations and atomically advance the small index."""
    destination = output / "manifest.json"
    if destination.is_file() and read_json(destination) == manifest:
        return
    generations = output / "manifest_generations"
    generations.mkdir(exist_ok=True)
    version = len(list(generations.glob("*.json"))) + 1
    write_json(generations / f"{version:04d}.json", manifest)
    pending = output / ".manifest.pending.json"
    write_json(pending, manifest)
    os.replace(pending, destination)


def snapshot_worker_source(output, source_path):
    """Launch from captured source, so repository edits cannot change a worker."""
    source = Path(source_path).read_bytes()
    source_sha256 = hashlib.sha256(source).hexdigest()
    compile(source, str(source_path), "exec")
    directory = output / "_exporter_sources"
    directory.mkdir(exist_ok=True)
    snapshot = directory / f"{source_sha256}.py"
    if snapshot.exists():
        require(digest(snapshot) == source_sha256, "Existing exporter source snapshot changed")
    else:
        with snapshot.open("xb") as handle:
            handle.write(source)
    return snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-ledger", type=Path, required=True)
    parser.add_argument("--validation-only", action="store_true", required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu", help="CPU replay only; no GPU allocation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--resume", action="store_true", help="Verify and reuse completed sidecars without repeating inference")
    parser.add_argument("--worker-run-id", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    ledger_path, output = args.reuse_ledger.resolve(), args.output_dir.resolve()
    if args.worker_run_id:
        run_worker(ledger_path, args.worker_run_id, output, args.device)
        return
    ledger = read_json(ledger_path)
    require(ledger.get("schema_version") == 1, "Unsupported reuse ledger schema")
    run_ids = [row["run_id"] for row in ledger["runs"]]
    require(len(run_ids) == len(set(run_ids)), "Duplicate run IDs in reuse ledger")
    require(set(args.run_id) <= set(run_ids), "Requested run IDs absent from reuse ledger")
    chosen = [run_id for run_id in run_ids if not args.run_id or run_id in args.run_id]
    require(bool(chosen), "No replay runs selected")
    require(not output.exists() or args.resume, "Choose a new immutable output directory or explicitly --resume")
    for run_id in chosen:
        require(re.fullmatch(r"[A-Za-z0-9_.+-]+", run_id) and run_id not in {".", ".."}, "Unsafe run ID")
    output.mkdir(parents=True, exist_ok=args.resume)
    worker_source = snapshot_worker_source(output, Path(__file__).resolve())
    entries = []
    by_run = {entry["run_id"]: entry for entry in ledger["runs"]}
    for run_id in chosen:
        if (output / run_id).exists():
            require(args.resume, "Completed replay needs --resume")
            entries.append(verified_completed_export(by_run[run_id], ledger_path, output / run_id))
            print(json.dumps({"run_id": run_id, "status": "verified_reused_without_inference"}), flush=True)
            continue
        command = [sys.executable, str(worker_source), "--reuse-ledger", str(ledger_path),
                   "--validation-only", "--device", args.device, "--output-dir", str(output / run_id), "--worker-run-id", run_id]
        subprocess.run(command, check=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""})
        entries.append(read_json(output / run_id / "manifest_entry.json"))
    manifest = {"schema_version": 1, "validation_only": True,
                "reproduction_passed": True, "reuse_ledger_sha256": digest(ledger_path), "runs": entries}
    if ledger.get("counts_freeze"):
        manifest["counts_freeze_sha256"] = ledger["counts_freeze"]["sha256"]
    persist_manifest_index(output, manifest)


if __name__ == "__main__":
    main()
