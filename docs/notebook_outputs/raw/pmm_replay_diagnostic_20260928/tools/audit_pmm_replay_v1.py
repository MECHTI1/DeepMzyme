"""Read-only TECH-023 input/numerical diagnosis; never certifies a campaign fit.

Kept outside src/ and scripts/ so that diagnostic code cannot change the frozen
training or graph-cache identity. Pickles/checkpoints must be trusted local files.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
from time import perf_counter
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
SCHEMA = "pmm-replay-diagnostic-v1"
EC_METADATA = frozenset({"ec_group_id", "ec_sample_weight"})


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(value):
    """Content identity preserves tensor type/shape/order and nested metadata."""
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise ValueError("Only dense tensors are supported by this diagnostic")
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        return {"type": "tensor", "dtype": str(value.dtype), "shape": list(value.shape),
                "sha256": hashlib.sha256(raw).hexdigest()}
    if isinstance(value, dict):
        return {str(key): descriptor(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "items": [descriptor(item) for item in value]}
    if value is None or isinstance(value, (str, int, float, bool)):
        return {"type": type(value).__name__, "value": value}
    raise TypeError(f"Unsupported input metadata: {type(value).__name__}")


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def graph_fields(graph):
    return {key: descriptor(graph[key]) for key in sorted(graph.keys())}


def graph_hash(graph, excluded=()):
    return stable_hash({key: value for key, value in graph_fields(graph).items() if key not in excluded})


def field_differences(left, right):
    return {key: {"fresh": left.get(key), "cached": right.get(key)}
            for key in sorted(left.keys() | right.keys()) if left.get(key) != right.get(key)}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def new_output_dir(path, protected=()):
    path = Path(path).resolve()
    for original in map(lambda item: Path(item).resolve(), protected):
        if path == original or path in original.parents or original in path.parents:
            raise ValueError(f"Diagnostic output overlaps protected input: {original}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def read_cache(path):
    """Validate native raw-cache framing before unpickling trusted local data."""
    from torch_geometric.data import Data

    path = Path(path)
    content = path.read_bytes()
    checksum, separator, payload = content.partition(b"\n")
    if not separator or checksum != hashlib.sha256(payload).hexdigest().encode():
        raise ValueError(f"Corrupt raw graph cache: {path}")
    key, graph = pickle.loads(payload)
    if key != path.stem or not isinstance(graph, Data):
        raise ValueError(f"Misidentified raw graph cache: {path}")
    return graph, hashlib.sha256(content).hexdigest()


def utc_timestamp(text):
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("Cache time bounds require explicit timezone")
    return value.timestamp()


def saved_rows(path, ordered_uids):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_uid = {row["source_uid"]: row for row in rows}
    if len(by_uid) != len(rows) or len(set(ordered_uids)) != len(ordered_uids) or set(by_uid) != set(ordered_uids):
        raise ValueError("Prediction UIDs are duplicated or differ from validation inputs")
    return [by_uid[uid] for uid in ordered_uids]


def verify_source(expected):
    from benchmarking.pmm_ion_campaign import source_tree_sha256

    actual = source_tree_sha256()
    if actual != expected:
        raise ValueError(f"Frozen scientific source mismatch: {actual} != {expected}")
    return actual


def prepare(args):
    from training import campaign_runtime
    from training.graph_dataset import PocketGraphDataset
    from torch_geometric.loader import DataLoader

    run = args.run_dir.resolve()
    payload = json.loads((run / "run_config.json").read_text())
    config = payload["config"]
    identity = json.loads(config["campaign_run_identity"])
    verify_source(identity["source_tree_sha256"])
    if config["task"] != "metal" or config["batch_size"] != 16:
        raise ValueError("This diagnostic is scoped to the saved metal campaign with batch size 16")
    after, before = utc_timestamp(args.cache_after), utc_timestamp(args.cache_before)
    if after >= before:
        raise ValueError("Cache time window is empty")
    output = new_output_dir(args.output_dir, (run, args.cache_namespace))
    captured = {}

    class CapturedInputs(Exception):
        pass

    def capture(**kwargs):
        captured.update(kwargs)
        raise CapturedInputs()

    # Reuse the frozen, guarded validation reconstruction; stop before any export.
    with patch.dict(os.environ, {"DEEPMZYME_GRAPH_CACHE_DIR": ""}), \
            patch.object(campaign_runtime, "export_selected_validation_predictions", capture):
        try:
            campaign_runtime.replay_campaign_run(run, device="cpu", output_dir=output / "capture")
        except CapturedInputs:
            pass
    if not captured:
        raise ValueError("Frozen validation input capture did not execute")
    model, loader = captured["model"], captured["val_loader"]
    if getattr(model, "predict_ec", None) is not False:
        raise ValueError("Cannot exclude EC metadata for a model with an EC prediction path")
    excluded = EC_METADATA if args.allow_ec_metadata_differences else frozenset()
    dataset = loader.dataset
    pockets, fresh = captured["val_pockets"], dataset.precomputed_data
    uids = [pocket.metadata["source_uid"] for pocket in pockets]
    rows = saved_rows(run / "val_predictions.csv", uids)
    fingerprint_index, position_index = defaultdict(list), defaultdict(list)
    cache_records = {}
    print("Indexing original-window cache graph contents", flush=True)
    for index, path in enumerate(sorted(args.cache_namespace.glob("*.pkl"))):
        stat = path.stat()
        if not after <= stat.st_mtime <= before:
            continue
        graph, checksum = read_cache(path)
        fields = graph_fields(graph)
        fingerprint = stable_hash({key: value for key, value in fields.items() if key not in excluded})
        record = {"path": str(path.resolve()), "sha256": checksum, "mtime_ns": stat.st_mtime_ns,
                  "size_bytes": stat.st_size, "fields": fields}
        cache_records[str(path)] = record
        fingerprint_index[fingerprint].append(str(path))
        if "pos" in fields:
            position_index[stable_hash(fields["pos"])].append(str(path))
        if index % 1000 == 0:
            print(f"Indexed {len(cache_records)} eligible cache entries", flush=True)
    if not cache_records:
        raise ValueError("No cache files occur in the specified original-fit time window")
    cached, example_audit, problems = [], [], []
    for uid, graph in zip(uids, fresh):
        fields = graph_fields(graph)
        candidates = fingerprint_index.get(graph_hash(graph, excluded), [])
        record = {"source_uid": uid, "fresh_fields": fields, "matching_cache_files": candidates}
        if not candidates:
            near = position_index.get(stable_hash(fields.get("pos")), [])
            record["candidate_field_mismatches"] = [
                {"cache_path": path, "differences": field_differences(fields, cache_records[path]["fields"])}
                for path in near]
            problems.append(uid)
        else:
            chosen = candidates[0]
            old, _ = read_cache(chosen)
            record["raw_field_differences"] = field_differences(fields, cache_records[chosen]["fields"])
            record["matching_cache_sha256"] = [cache_records[path]["sha256"] for path in candidates]
            cached.append(old)
        example_audit.append(record)
    manifest = {
        "schema": SCHEMA, "status": "input_mismatch" if problems else "raw_inputs_matched",
        "certifies_fit": False, "run_dir": str(run), "structure_dir": config["structure_dir"],
        "campaign_run_identity": identity, "tool_sha256": file_sha(__file__),
        "cache_namespace": str(args.cache_namespace.resolve()),
        "cache_time_window": {"after": args.cache_after, "before": args.cache_before, "basis": "filesystem mtime"},
        "retrospective_limit": "Original in-memory inputs were not saved. Fresh UID-keyed graphs match original-window cache contents; cache entries themselves do not contain UIDs. Identical-content candidates are all listed, not assigned distinct historical identities.",
        "excluded_from_predictive_equality": sorted(excluded),
        "exclusion_basis": "task=metal, model.predict_ec=False; ec_group_id is reporting metadata and ec_sample_weight is only consumed by the disabled EC loss branch. Their differences remain recorded; targets are never excluded.",
        "eligible_cache_files": len(cache_records), "examples": example_audit, "unmatched_uids": problems,
        "ambiguous_identical_content_examples": sum(len(row["matching_cache_files"]) > 1 for row in example_audit),
        "input_files": {str((run / name).resolve()): file_sha(run / name) for name in
                        ("best_model_checkpoint.pt", "run_config.json", "selected_checkpoint.json", "val_predictions.csv")},
        "cohort_file": {"path": config["source_cohort_csv"], "sha256": config["source_cohort_sha256"]},
        "fold_file": {"path": config["fold_membership_csv"], "sha256": config["fold_membership_sha256"]},
        "normalization_sha256": stable_hash(descriptor(captured["best_checkpoint"]["normalization_stats"])),
    }
    matched_paths = {path for row in example_audit for path in row["matching_cache_files"]}
    manifest["matched_cache_files"] = [cache_records[path] for path in sorted(matched_paths)]
    write_json(output / "input_manifest.json", manifest)
    if problems:
        raise ValueError(f"{len(problems)} fresh graphs lack exact original-window predictive-content matches")
    kwargs = {key: getattr(dataset, key) for key in
              ("esm_dim", "edge_radius", "use_ring_edges", "require_ring_edges", "node_feature_set",
               "omit_node_features", "metal_node_mode", "shell_role_source")}
    old_dataset = PocketGraphDataset(pockets, precomputed_data=cached,
                                    normalization_stats=dataset.normalization_stats, **kwargs)
    old_loader = DataLoader(old_dataset, batch_size=16, shuffle=False, num_workers=0,
                            generator=torch.Generator().manual_seed(int(config["seed"]) + 2))
    batches, batch_manifest = [], []
    for index, (new_batch, old_batch) in enumerate(zip(loader, old_loader)):
        difference = field_differences(graph_fields(new_batch), graph_fields(old_batch))
        prohibited = set(difference) - excluded
        batch_manifest.append({"index": index, "fresh_sha256": graph_hash(new_batch),
                               "cached_sha256": graph_hash(old_batch),
                               "predictive_sha256": graph_hash(new_batch, excluded),
                               "field_differences": difference})
        if prohibited:
            manifest.update(status="normalized_batch_mismatch", batches=batch_manifest)
            write_json(output / "input_manifest.json", manifest)
            raise ValueError(f"Normalized/collated batch {index} differs in {sorted(prohibited)}")
        batches.append(new_batch)
    if not batches or len(batches) != len(loader) or len(batches) != len(old_loader):
        raise ValueError("Batch count mismatch")
    # Do not delete even nonpredictive fields; preserve the exact fresh replay input.
    snapshot = output / "input_snapshot.pt"
    torch.save({"schema": SCHEMA, "batches": batches, "ordered_uids": uids, "saved_rows": rows}, snapshot)
    manifest.update(status="predictive_inputs_matched", batches=batch_manifest,
                    snapshot_sha256=file_sha(snapshot), n_examples=len(uids), n_batches=len(batches),
                    normalization_refitted=False, training_performed=False)
    write_json(output / "input_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "n_examples": len(uids), "output": str(output)}), flush=True)
    return manifest


def probability_views(logits):
    from training.final_test_reporting import collapse_metal_probabilities

    native = torch.softmax(logits.detach().cpu().float(), dim=-1)
    if not bool(torch.isfinite(native).all()):
        raise ValueError("Nonfinite model probabilities")
    return native, collapse_metal_probabilities(native)


def compare_probabilities(actual, reference):
    if actual.shape != reference.shape or not bool(torch.isfinite(reference).all()):
        raise ValueError("Invalid reference probabilities")
    difference = (actual.double() - reference.double()).abs()
    margin = actual.topk(2, dim=1).values
    # Keep the original CSV's eight-decimal conversion distinct from tensor drift.
    rounded = torch.tensor([[float(f"{float(value):.8f}") for value in row] for row in actual], dtype=torch.float64)
    return {"max_abs_difference": float(difference.max()),
            "mean_abs_difference": float(difference.mean()),
            "probability_fields_above_original_1e_6": int(((rounded - reference.double()).abs() > 1e-6).sum()),
            "changed_predictions": int((actual.argmax(1) != reference.argmax(1)).sum()),
            "minimum_class_margin": float((margin[:, 0] - margin[:, 1]).min())}


def repeated_forward(model, batches, *, device, repeats, on_pass=None):
    from training.loop import evaluate_epoch_with_predictions

    if repeats < 2:
        raise ValueError("At least two predeclared repeats are required")
    model.eval()
    original_inputs = [graph_hash(batch) for batch in batches]
    original_model = stable_hash(descriptor(model.state_dict()))
    result = []

    class ClonedBatches:
        def __len__(self):
            return len(batches)

        def __iter__(self):
            for original, expected in zip(batches, original_inputs):
                clone = original.clone()
                yield clone
                if graph_hash(clone) != expected:
                    raise ValueError("Model mutated an input batch during evaluation")

    for index in range(repeats):
        started = perf_counter()
        prediction = evaluate_epoch_with_predictions(model, ClonedBatches(), device=device)
        if [graph_hash(batch) for batch in batches] != original_inputs:
            raise ValueError("Stored fixed inputs were mutated")
        if stable_hash(descriptor(model.state_dict())) != original_model:
            raise ValueError("Model state changed during evaluation")
        native, common = probability_views(prediction["metal_logits"])
        item = {"logits": prediction["metal_logits"], "native": native, "common4": common,
                "y_native": prediction["metal_y"], "seconds": perf_counter() - started,
                "pred_native": native.argmax(1), "pred_common4": common.argmax(1)}
        for view, probabilities in (("native", native), ("common4", common)):
            top = probabilities.topk(2, dim=1).values
            item[f"class_margin_{view}"] = top[:, 0] - top[:, 1]
        result.append(item)
        if on_pass is not None:
            on_pass(index, item)
        print(f"Completed fixed-input pass {index + 1}/{repeats}", flush=True)
    return result, {"input_hashes_before_and_after": original_inputs,
                    "model_state_sha256_before_and_after": original_model}


def evaluate(args):
    from training.access_guard import install_forbidden_read_guard
    from benchmarking.pmm_ion_campaign import forbidden_read_roots
    from training.campaign_runtime import load_campaign_prediction_components

    prepared = args.prepared_dir.resolve()
    manifest_path = prepared / "input_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "predictive_inputs_matched" or manifest["schema"] != SCHEMA:
        raise ValueError("Input preparation did not pass")
    install_forbidden_read_guard(forbidden_read_roots(Path(manifest["structure_dir"])))
    verify_source(manifest["campaign_run_identity"]["source_tree_sha256"])
    if manifest["tool_sha256"] != file_sha(__file__):
        raise ValueError("Diagnostic implementation changed after input preparation")
    for path, expected in manifest["input_files"].items():
        if file_sha(path) != expected:
            raise ValueError(f"Original run artifact changed: {path}")
    for key in ("cohort_file", "fold_file"):
        if file_sha(manifest[key]["path"]) != manifest[key]["sha256"]:
            raise ValueError(f"Frozen input changed: {key}")
    snapshot_path = prepared / "input_snapshot.pt"
    if file_sha(snapshot_path) != manifest["snapshot_sha256"]:
        raise ValueError("Input snapshot changed")
    output = new_output_dir(args.output_dir, (Path(manifest["run_dir"]), prepared))
    snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    if snapshot["schema"] != SCHEMA or [graph_hash(batch) for batch in snapshot["batches"]] != [
            row["fresh_sha256"] for row in manifest["batches"]]:
        raise ValueError("Input snapshot tensor hashes differ")
    if args.condition == "strict" and str(args.device).startswith("cuda"):
        if torch.cuda.is_initialized():
            raise ValueError("Strict condition requires a fresh process before CUDA initialization")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in {":4096:8", ":16:8"}:
            raise ValueError("Strict condition requires a supported CUBLAS_WORKSPACE_CONFIG")
    torch.use_deterministic_algorithms(args.condition == "strict", warn_only=False)
    if args.condition == "strict":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    checkpoint = torch.load(Path(manifest["run_dir"]) / "best_model_checkpoint.pt", map_location="cpu", weights_only=False)
    if checkpoint["config"].get("deterministic", False):
        raise ValueError("Original condition currently implements the frozen deterministic=False fit only")
    model, _, _ = load_campaign_prediction_components(checkpoint, device=args.device)
    from label_schemes import METAL_TARGET_LABELS, COLLAPSED_METAL_LABELS
    report = {"schema": SCHEMA, "certifies_fit": False, "training_performed": False,
              "normalization_refitted": False, "condition": args.condition, "planned_repeats": args.repeats,
              "input_manifest_sha256": file_sha(manifest_path), "tool_sha256": file_sha(__file__),
              "device": args.device, "torch": torch.__version__, "cuda": torch.version.cuda,
              "python": sys.version, "gpu_library": __import__("torch_geometric").__version__,
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
              "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
              "cuda_device": torch.cuda.get_device_name() if str(args.device).startswith("cuda") else None,
              "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
              "cudnn_benchmark": torch.backends.cudnn.benchmark,
              "cudnn_deterministic": torch.backends.cudnn.deterministic}
    preserved_passes = []

    def preserve_pass(index, item):
        path = output / f"pass_{index:02d}.pt"
        torch.save(item, path)
        preserved_passes.append({"pass": index, "path": path.name, "sha256": file_sha(path)})

    try:
        passes, unchanged = repeated_forward(model, snapshot["batches"], device=args.device,
                                             repeats=args.repeats, on_pass=preserve_pass)
    except Exception as exc:
        report.update(status="evaluation_failed", error_type=type(exc).__name__, error=str(exc),
                      preserved_passes=preserved_passes, completed_repeats=len(preserved_passes))
        write_json(output / "evaluation_report.json", report)
        raise
    references = {}
    for view, labels, prefix in (("native", METAL_TARGET_LABELS, "p_native_"),
                                 ("common4", COLLAPSED_METAL_LABELS, "p_common4_")):
        references[view] = torch.tensor([[float(row[prefix + labels[key].replace(" ", "_")])
                                         for key in sorted(labels)] for row in snapshot["saved_rows"]], dtype=torch.float64)
    comparisons, pairwise = [], []
    for index, result in enumerate(passes):
        if result["y_native"].tolist() != [int(row["y_native"]) for row in snapshot["saved_rows"]]:
            raise ValueError("Target order differs from saved predictions")
        comparisons.append({"pass": index, "seconds": result["seconds"], **{
            view: compare_probabilities(result[view], references[view]) for view in references}})
    for left in range(len(passes)):
        for right in range(left + 1, len(passes)):
            pairwise.append({"left": left, "right": right, **{
                view: compare_probabilities(passes[left][view], passes[right][view]) for view in references}})
    arrays = output / "forward_passes.pt"
    torch.save({"ordered_uids": snapshot["ordered_uids"], "passes": passes,
                "saved_probabilities": references}, arrays)
    report.update(status="diagnosis_complete", completed_repeats=len(passes), **unchanged,
                  comparisons_to_saved=comparisons, pairwise_comparisons=pairwise,
                  forward_passes_sha256=file_sha(arrays), preserved_passes=preserved_passes)
    write_json(output / "evaluation_report.json", report)
    print(json.dumps({"status": report["status"], "output": str(output)}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--run-dir", type=Path, required=True)
    prep.add_argument("--cache-namespace", type=Path, required=True)
    prep.add_argument("--cache-after", required=True)
    prep.add_argument("--cache-before", required=True)
    prep.add_argument("--allow-ec-metadata-differences", action="store_true")
    prep.add_argument("--output-dir", type=Path, required=True)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--prepared-dir", type=Path, required=True)
    evaluation.add_argument("--output-dir", type=Path, required=True)
    evaluation.add_argument("--condition", choices=("original", "strict"), required=True)
    evaluation.add_argument("--device", default="cpu")
    evaluation.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    (prepare if args.phase == "prepare" else evaluate)(args)


if __name__ == "__main__":
    main()
