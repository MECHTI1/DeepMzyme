"""Bounded two-system diagnostic pilot using the unified training core.

The split/plan freeze precedes fitting. The separate evaluation action requires
both complete fits and their hash-bound selected checkpoints. It never invokes
the protected held-out evaluator, optimizer, or normalization fitting.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import fields
import itertools
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

from training.explicit_membership import (
    ENDPOINT, PARTITIONS, checked_path, counts, fixed_split, load_membership,
    read_json, read_rows, relabel_five_class_memberships, require, select_exact, sha256, validate_rows,
)

FAMILIES = ("only_esm", "only_gvp")
PATH_FIELDS = ("structure_dir", "summary_csv", "esm_embeddings_dir", "ring_features_dir",
               "external_features_root_dir", "runs_dir", "explicit_membership_manifest")


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def write_rows(path, rows):
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reference(path, base):
    return {"path": str(Path(path).relative_to(base)), "sha256": sha256(path)}


def rebalance(rows, edges, seed=20260917):
    """Minimum moved pockets, then seeded hash tie-break; never reads outcomes."""
    require(seed == 20260917, "Only the frozen split seed is permitted")
    rows = [dict(r, protein_id=r["group_id"]) for r in rows]
    for row in rows:
        row["target"] = int(row["target"])
    original = {r["component_id"]: r["partition"] for r in rows}
    require(all(original[r["component_id"]] == r["partition"] for r in rows), "Original component crossing")
    missing = {c: max(0, 3 - len({r["component_id"] for r in rows
                                  if r["partition"] == "inner_validation" and r["target"] == c})) for c in range(4)}
    candidates = sorted({r["component_id"] for r in rows
                         if r["partition"] == "outer_evaluation" and missing[r["target"]]})
    feasible = []
    for n_moves in range(sum(missing.values()) + 1):
        for move in itertools.combinations(candidates, n_moves):
            trial = [dict(r, partition="inner_validation") if r["component_id"] in move else r for r in rows]
            parts = {p: [{k: v for k, v in r.items() if k != "partition"}
                         for r in trial if r["partition"] == p] for p in PARTITIONS}
            development = [r for part in parts.values() for r in part]
            try:
                validate_rows(parts, development, edges)
            except ValueError:
                continue
            import hashlib
            moved = sum(r["component_id"] in move for r in rows)
            tie = hashlib.sha256(f"{seed}:{','.join(move)}".encode()).hexdigest()
            feasible.append(((moved, tie), parts, move))
        if feasible:
            break
    require(bool(feasible), "Cannot achieve three inner-validation Cu components with adequate training/outer support")
    _, parts, move = min(feasible, key=lambda x: x[0])
    return parts, {"seed": seed, "outcomes_used": False, "method": "minimum moved components, then pockets, then seeded SHA256 tie-break",
                   "moved_components_outer_to_inner": list(move),
                   "moved_pockets": sum(r["component_id"] in move for r in rows),
                   "training_membership_unchanged": True, "seeds_tried": [seed]}


def freeze_split(previous, output, root):
    require(not output.exists(), "Refusing to overwrite an accepted diagnostic split")
    old_rows = read_rows(previous / "metal_split_manifest.csv")
    edges = read_rows(previous / "metal_similarity_edges.csv")
    parts, receipt = rebalance(old_rows, edges)
    selected = [r for r in read_json(previous / "selected_configurations.json")
                if r["family"] in FAMILIES and r["seed"] == 42]
    require({r["family"] for r in selected} == set(FAMILIES) and len(selected) == 2, "Missing exact selected configurations")
    # Bind the entire allowed cohort to the saved development-only runs.
    source_payloads = {}
    for record in selected:
        directory = Path(record["source_run_dir"])
        for name, expected in record["expected_sha256"].items():
            require(sha256(directory / name) == expected, f"Selected source artifact changed: {name}")
        payload = read_json(directory / "run_config.json")
        require(payload.get("test_report") is None and not payload["config"]["run_test_eval"], "Source is not development-only")
        identities = payload["dataset_summary"]["retained_split_identity"]
        allowed = {r["pocket_id"]: r for part in identities.values() for r in part["examples"]}
        require(set(allowed) == {r["example_id"] for r in old_rows}, "Development allowlist differs from trusted source runs")
        for row in old_rows:
            source = allowed[row["example_id"]]
            require(source["structure_id"] == row["structure_id"] and source["group"] == row["group_id"]
                    and source["y_metal"] == row["target"], "Development source identity/target mismatch")
        source_payloads[record["family"]] = payload
    output.mkdir(parents=True)
    manifests = {}
    for part, rows in parts.items():
        write_rows(output / f"{part}.csv", sorted(rows, key=lambda r: r["example_id"]))
        manifests[part] = reference(output / f"{part}.csv", output)
    development = sorted([r for rows in parts.values() for r in rows], key=lambda r: r["example_id"])
    write_rows(output / "development_allowlist.csv", development)
    for name in ("metal_sequence_manifest.json", "metal_similarity_edges.csv", "metal_component_membership.csv",
                 "metal_mmseqs_reuse_receipt.json", "metal_feature_coverage.json", "unresolved_chain_report.json",
                 "metal_pocket_feature_adjudication.json", "selected_measured_runtimes.json", "selected_system_differences.json"):
        shutil.copyfile(previous / name, output / name)
    sequence_rows = read_json(output / "metal_sequence_manifest.json")["sequences"]
    (output / "development.fasta").write_text("".join(f">{r['sequence_id']}\n{r['sequence']}\n" for r in sequence_rows))
    original_fasta = root / "DeepMzyme_Data/notebook_outputs/remote_homology_v1/metal/development.fasta"
    require(sha256(original_fasta) == read_json(output / "metal_mmseqs_reuse_receipt.json")["fasta_sha256"], "Frozen search FASTA changed")
    shutil.copyfile(original_fasta, output / "searched_development.fasta")
    feature_rows = read_rows(previous / "metal_feature_coverage.csv")
    portable_features = [{k: r[k] for k in ("structure_id", "structure_sha256", "esm_sha256", "external_sha256")}
                         | {"esm_file": Path(r["esm_path"]).name} for r in feature_rows]
    write_rows(output / "feature_inventory.csv", portable_features)
    train_root = root / "DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train"
    summary = train_root / "final_data_summarazing_table_transition_metals_only_catalytic.csv"
    cfg0 = source_payloads["only_esm"]["config"]
    source = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "status": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True)}
    spec = {"schema_version": 1, "endpoint": ENDPOINT, "split_seed": 20260917,
            "label_scheme": "merge_fe_class_viii", "manifests": manifests,
            "development_allowlist": reference(output / "development_allowlist.csv", output),
            "detected_edges": reference(output / "metal_similarity_edges.csv", output),
            "search_protocol": reference(output / "metal_mmseqs_reuse_receipt.json", output),
            "feature_inventory": reference(output / "feature_inventory.csv", output),
            "dataset": {"scope": "external_training_only", "name": "non_overlapped_pinmymetal_development",
                        "bundle_id": cfg0["dataset_bundle_id"], "bundle_sha256": cfg0["dataset_bundle_sha256"],
                        "summary_sha256": sha256(summary), "structure_manifest_sha256": sha256(train_root / "structure_manifest.csv")},
            "counts": validate_rows(parts, development, edges), "source_control": source,
            "limitations": ["Coordinate chains only; full biological protein and site-chain completeness uncertified",
                            "protein_id uses the available PDB grouping, not a certified full-protein accession",
                            "No qualifying hit is unclassified, not measured <=20% identity",
                            "Not family-disjoint, superfamily-disjoint, structure-remote or homolog-free",
                            "One diagnostic split and one model seed; selected systems, not a causal representation test"]}
    write_json(output / "split_freeze.json", spec)
    receipt.update(previous_split_sha256=sha256(previous / "metal_split_manifest.csv"),
                   original_counts=read_json(previous / "metal_split_counts.json"),
                   split_freeze_sha256=sha256(output / "split_freeze.json"))
    write_json(output / "split_generation_receipt.json", receipt)
    write_json(output / "overlap_audit.json", {"pdb_crossings": 0, "protein_proxy_crossings": 0,
               "component_crossings": 0, "detected_edge_crossings": 0, "protected_test_opened": False,
               "exclusion_basis": "Exact hash-bound development allowlists from both original selected runs"})
    from training.config import TrainConfig, config_to_payload
    names = {f.name for f in fields(TrainConfig)}
    models = {}
    for record in selected:
        family = record["family"]
        payload = source_payloads[family]
        original = payload["config"]
        cfg = {k: v for k, v in original.items() if k in names}
        cfg.update(structure_dir=str(train_root.relative_to(root)), summary_csv=str(summary.relative_to(root)),
                   esm_embeddings_dir="DeepMzyme_Data/esm_embeddings" if family == "only_esm" else str((output / "empty_esm_cache").relative_to(root)),
                   external_features_root_dir="DeepMzyme_Data/notebook_outputs/plans/metal_architecture_pilot_10h_v1/external_overlay",
                   runs_dir=str((output / "runs").relative_to(root)), run_name=family, split_seed=20260917,
                   explicit_membership_manifest=str((output / "split_freeze.json").relative_to(root)),
                   explicit_membership_sha256=sha256(output / "split_freeze.json"))
        # Newly introduced fields preserve the archived legacy behavior.
        cfg = config_to_payload(TrainConfig(**cfg))
        write_json(output / f"{family}_configuration.json", cfg)
        write_json(output / f"{family}_source_run_config.json", payload)
        shutil.copyfile(Path(record["source_run_dir"]) / "run_metadata.json", output / f"{family}_source_run_metadata.json")
        models[family] = {"configuration": reference(output / f"{family}_configuration.json", output),
                          "source_run_id": record["run_id"], "source_hashes": record["expected_sha256"],
                          "source_selected_epoch": record["selected_epoch"]}
    (output / "empty_esm_cache").mkdir()
    write_json(output / "pilot_plan.json", {"endpoint": ENDPOINT, "split": reference(output / "split_freeze.json", output),
               "models": models, "epochs": 50, "model_seed": 42, "outer_observed": False,
               "fits_authorized": 2, "source_control": source})
    write_json(output / "manifest_checksums.json", {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()})
    return spec["counts"]


def load_config(output, family, root, device="cpu"):
    from training.config import TrainConfig
    plan = read_json(output / "pilot_plan.json")
    validate_execution_device(plan, device)
    checked_path(output, plan["split"])
    cfg = read_json(checked_path(output, plan["models"][family]["configuration"]))
    for key in PATH_FIELDS:
        if cfg.get(key) is not None:
            cfg[key] = str((root / cfg[key]).resolve())
    cfg["structure_dir"], cfg["summary_csv"] = Path(cfg["structure_dir"]), Path(cfg["summary_csv"])
    cfg["device"] = device
    return TrainConfig(**cfg)


def freeze_five_class(previous, output, root, source_runs):
    """Freeze a local-only relabeling of the completed four-class diagnostic."""
    require(not output.exists(), "Refusing to overwrite an accepted diagnostic split")
    parent_plan = read_json(previous / "pilot_plan.json")
    parent_path = checked_path(previous, parent_plan["split"])
    parent = read_json(parent_path)
    require(parent["label_scheme"] == "merge_fe_class_viii" and parent["split_seed"] == 20260917,
            "Expected the frozen direct-four diagnostic parent")
    for name, digest in read_json(previous / "manifest_checksums.json").items():
        require(sha256(previous / name) == digest, f"Parent artifact changed: {name}")
    parent_parts = {p: read_rows(checked_path(previous, parent["manifests"][p])) for p in PARTITIONS}
    run_ids = {"only_esm": "T1_1_five_class_lr3e-05_s42", "only_gvp": "T2_0_five_class_lr0.0001_s42"}
    payloads, source_hashes = {}, {}
    for family, run_id in run_ids.items():
        directory = source_runs / run_id
        payload = read_json(directory / "run_config.json")
        metadata = read_json(directory / "run_metadata.json")
        original = read_json(previous / f"{family}_source_run_config.json")["config"]
        require({k: v for k, v in payload["config"].items() if k not in ("run_name", "metal_label_scheme")}
                == {k: v for k, v in original.items() if k not in ("run_name", "metal_label_scheme")},
                "Five-class saved recipe differs beyond target/run name")
        require(metadata["config"] == payload["config"], "Five-class source metadata/config mismatch")
        source_hashes[family] = {name: sha256(directory / name) for name in
                                ("run_config.json", "run_metadata.json", "best_model_checkpoint.pt")}
        payloads[family] = payload
    parts = relabel_five_class_memberships(parent_parts, payloads)
    edges = read_rows(checked_path(previous, parent["detected_edges"]))
    support = validate_rows(parts, [r for rows in parts.values() for r in rows], edges, label_scheme="five_class")
    output.mkdir(parents=True)
    parent_copy = output / "parent_four_class"
    parent_copy.mkdir()
    shutil.copyfile(parent_path, parent_copy / "split_freeze.json")
    for part in PARTITIONS:
        shutil.copyfile(checked_path(previous, parent["manifests"][part]), parent_copy / parent["manifests"][part]["path"])
        write_rows(output / f"{part}.csv", parts[part])
    development = sorted([r for rows in parts.values() for r in rows], key=lambda r: r["example_id"])
    write_rows(output / "development_allowlist.csv", development)
    copied = ("feature_inventory.csv", "metal_sequence_manifest.json", "metal_similarity_edges.csv",
              "metal_mmseqs_reuse_receipt.json", "metal_feature_coverage.json", "unresolved_chain_report.json",
              "metal_pocket_feature_adjudication.json", "development.fasta", "searched_development.fasta")
    for name in copied:
        shutil.copyfile(previous / name, output / name)
    write_rows(output / "metal_component_membership.csv", development)
    for family in FAMILIES:
        for name in ("run_config.json", "run_metadata.json"):
            shutil.copyfile(source_runs / run_ids[family] / name, output / f"{family}_source_{name}")
    source = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "status": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True)}
    spec = dict(parent, schema_version=2, label_scheme="five_class", counts=support, source_control=source,
                manifests={p: reference(output / f"{p}.csv", output) for p in PARTITIONS},
                development_allowlist=reference(output / "development_allowlist.csv", output),
                parent_split=reference(parent_copy / "split_freeze.json", output),
                label_sources={f: reference(output / f"{f}_source_run_config.json", output) for f in FAMILIES},
                prior_four_class_outer_observed=True,
                limitations=parent["limitations"] + [
                    "Follow-up on an already observed diagnostic outer partition, not independent confirmation",
                    "Local CPU follow-up versus prior GPU fits; backend differences limit causal target claims"])
    write_json(output / "split_freeze.json", spec)
    write_json(output / "split_generation_receipt.json", {
        "seed": 20260917, "seeds_tried": [20260917], "method": "relabel exact parent membership; no reassignment",
        "moved_pockets": 0, "new_predictions_used": False, "all_partitions_unchanged": True,
        "parent_split": spec["parent_split"], "label_sources": spec["label_sources"],
        "split_freeze_sha256": sha256(output / "split_freeze.json"), "prior_four_class_outer_observed": True})
    audit = read_json(previous / "overlap_audit.json")
    write_json(output / "overlap_audit.json", dict(audit, label_scheme="five_class", all_partitions_unchanged=True))
    models = {}
    for family in FAMILIES:
        cfg = read_json(checked_path(previous, parent_plan["models"][family]["configuration"]))
        cfg.update(metal_label_scheme="five_class", device="cpu", pin_memory=False, num_workers=0,
                   runs_dir=str((output / "runs").relative_to(root)),
                   explicit_membership_manifest=str((output / "split_freeze.json").relative_to(root)),
                   explicit_membership_sha256=sha256(output / "split_freeze.json"))
        if family == "only_gvp":
            cfg["esm_embeddings_dir"] = str((output / "empty_esm_cache").relative_to(root))
        write_json(output / f"{family}_configuration.json", cfg)
        models[family] = {"configuration": reference(output / f"{family}_configuration.json", output),
                          "source_run_id": run_ids[family], "source_run_directory": str(source_runs / run_ids[family]),
                          "source_hashes": source_hashes[family],
                          "source_selected_epoch": payloads[family]["selected_checkpoint_epoch"]}
    (output / "empty_esm_cache").mkdir()
    write_json(output / "pilot_plan.json", {
        "endpoint": ENDPOINT, "split": reference(output / "split_freeze.json", output), "models": models,
        "epochs": 50, "model_seed": 42, "fits_authorized": 2, "outer_observed": False,
        "prior_four_class_outer_observed": True, "source_control": source,
        "execution": {"local_only": True, "device": "cpu", "torch_threads": 1, "interop_threads": 1}})
    write_json(output / "manifest_checksums.json", {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()})
    return support


def validate_execution_device(plan, device, *, fitting=False):
    if plan.get("execution", {}).get("local_only"):
        require(device == "cpu", "This pilot is authorized for local CPU execution only")
    elif fitting:
        require(device.startswith("cuda"), "The original pilot requires admitted GPU execution")


def verify_local_admission(output, root):
    path = output / "cpu_admission_gate.json"
    require(path.is_file(), "Local fitting requires a completed CPU admission gate")
    gate = read_json(path)
    require(gate["pilot_plan_sha256"] == sha256(output / "pilot_plan.json"), "CPU admission plan changed")
    require(gate["gates"] and all(g["status"] == "PASS" for g in gate["gates"]), "CPU admission gate failed")
    require(gate["source_files"], "CPU admission requires tested source hashes")
    for name, digest in gate["source_files"].items():
        require(sha256(root / name) == digest, f"Tested source changed before fitting: {name}")
    return gate


def verify_completed_run(directory, family, split_hash):
    import torch
    from training.run import to_jsonable
    payload = read_json(directory / "run_config.json")
    metadata = read_json(directory / "run_metadata.json")
    cfg = payload["config"]
    require(metadata.get("fit_status") == "completed" and metadata.get("outer_observed_during_training") is False,
            "Both required fits must be completed without outer observation")
    require(cfg["model_architecture"] == family and cfg["seed"] == 42 and cfg["epochs"] == 50,
            "Completed run identity/budget mismatch")
    require(cfg["explicit_membership_sha256"] == split_hash and metadata["config"] == cfg,
            "Runs do not share the frozen scientific split/configuration")
    require(not cfg["run_test_eval"] and payload.get("test_report") is None and metadata.get("test_report") is None,
            "Protected-test route was activated")
    history = payload["history"]
    require([r["epoch"] for r in history] == list(range(1, 51)), "Incomplete 50-epoch fit")
    require(all(math.isfinite(r["val_metal_balanced_acc"]) for r in history),
            "Nonfinite inner-validation checkpoint metric")
    selected = max(history, key=lambda r: r["val_metal_balanced_acc"])
    checkpoint_path = directory / "best_model_checkpoint.pt"
    require(Path(metadata["selected_checkpoint"]).name == checkpoint_path.name, "Selected checkpoint path mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(checkpoint["epoch"] == selected["epoch"] == metadata["selected_checkpoint_epoch"] == payload["selected_checkpoint_epoch"],
            "Checkpoint was not selected by inner validation")
    for key in ("config", "normalization_stats", "dataset_summary"):
        require(to_jsonable(checkpoint[key]) == payload[key] == metadata[key], f"Checkpoint {key} mismatch")
    require(checkpoint["selection_metric"] == "val_metal_balanced_acc"
            and checkpoint["selection_metric_value"] == selected["val_metal_balanced_acc"], "Selected metric mismatch")
    return {"family": family, "seed": 42, "selected_epoch": checkpoint["epoch"],
            "run_directory": f"runs/{family}", "selected_checkpoint": checkpoint_path.name,
            "inner_balanced_accuracy": selected["val_metal_balanced_acc"],
            "checkpoint_sha256": sha256(checkpoint_path),
            "files": {name: sha256(directory / name) for name in
                      ("best_model_checkpoint.pt", "run_config.json", "run_metadata.json", "dataset_summary.json")}}


def freeze_checkpoints(output):
    require(not (output / "outer_evaluation").exists(), "Outer prediction artifact already exists")
    plan = read_json(output / "pilot_plan.json")
    checked_path(output, plan["split"])
    models = {family: verify_completed_run(output / "runs" / family, family, plan["split"]["sha256"]) for family in FAMILIES}
    for family in FAMILIES:
        expected = read_json(checked_path(output, plan["models"][family]["configuration"]))
        actual = read_json(output / "runs" / family / "run_config.json")["config"]
        for key, value in expected.items():
            if key not in (*PATH_FIELDS, "device"):
                require(actual[key] == value, f"Fit differs from frozen configuration: {family}/{key}")
    membership = [read_json(output / "runs" / f / "dataset_summary.json")["retained_split_identity"] for f in FAMILIES]
    split_spec = read_json(checked_path(output, plan["split"]))
    for part in ("train", "validation"):
        left, right = [m[part]["examples"] for m in membership]
        keys = ("structure_id", "pocket_id", "group", "y_metal")
        require([{k: r[k] for k in keys} for r in left] == [{k: r[k] for k in keys} for r in right],
                "Models received different scientific split membership")
        manifest_part = "inner_validation" if part == "validation" else part
        canonical = read_rows(checked_path(output, split_spec["manifests"][manifest_part]))
        require([(r["structure_id"], r["pocket_id"], r["group"], r["y_metal"]) for r in left]
                == [(r["structure_id"], r["example_id"], r["group_id"], r["target"]) for r in canonical],
                "Completed fitting membership differs from frozen manifests")
    frozen = {"status": "both_required_fits_completed", "models": models, "model_seed": 42,
              "split": plan["split"], "pilot_plan_sha256": sha256(output / "pilot_plan.json"),
              "configurations": {f: plan["models"][f]["configuration"] for f in FAMILIES},
              "manifest_sha256": {p: split_spec["manifests"][p]["sha256"] for p in PARTITIONS},
              "outer_observed_during_training": False, "protected_test_access": False,
              "frozen_at_unix": time.time()}
    if plan.get("prior_four_class_outer_observed"):
        frozen["prior_four_class_outer_observed"] = True
    write_json(output / "frozen_pilot.json", frozen)
    (output / "frozen_pilot.sha256").write_text(sha256(output / "frozen_pilot.json") + "\n")
    return frozen


def verify_outer_lock(output, frozen_sha256):
    require((output / "frozen_pilot.json").is_file(), "Outer evaluation locked: both frozen checkpoints are required")
    require(sha256(output / "frozen_pilot.json") == frozen_sha256, "Frozen pilot checksum mismatch")
    frozen = read_json(output / "frozen_pilot.json")
    require(frozen["status"] == "both_required_fits_completed" and set(frozen["models"]) == set(FAMILIES)
            and frozen["outer_observed_during_training"] is False, "Outer evaluation locked: incomplete pilot")
    require(sha256(output / "pilot_plan.json") == frozen["pilot_plan_sha256"], "Pilot configurations changed after freeze")
    checked_path(output, frozen["split"])
    for family, model in frozen["models"].items():
        directory = output / "runs" / family
        for name, expected in model["files"].items():
            require((directory / name).is_file() and sha256(directory / name) == expected,
                    f"Outer evaluation locked: missing/changed frozen checkpoint or run artifact: {family}/{name}")
        verified = verify_completed_run(directory, family, frozen["split"]["sha256"])
        require(verified == model, "Frozen completed-run verification changed")
    return frozen


def evaluate_outer(output, root, frozen_sha256, device):
    frozen = verify_outer_lock(output, frozen_sha256)
    destination = output / "outer_evaluation"
    destination.mkdir()  # Exclusive one-shot marker survives partial failure.
    write_json(destination / "started.json", {"frozen_pilot_sha256": frozen_sha256, "started_at_unix": time.time()})
    import torch
    from torch_geometric.loader import DataLoader
    from label_schemes import configure_active_metal_label_scheme, METAL_TARGET_LABELS
    from model_variants import build_pocket_classifier
    from training.data import load_training_pockets_with_report_from_dir
    from training.graph_dataset import build_graph_data_list, PocketGraphDataset
    from training.loop import evaluate_epoch_with_predictions
    from training.run import normalization_stats_from_payload, metrics_from_predictions, to_jsonable
    from export_validation_predictions import factory_kwargs, prediction_rows
    results = {}
    for family in FAMILIES:
        config = load_config(output, family, root, device)
        membership = load_membership(config)
        rows = membership["partitions"]["outer_evaluation"]
        configure_active_metal_label_scheme(config.metal_label_scheme)
        checkpoint = torch.load(output / "runs" / family / "best_model_checkpoint.pt", map_location="cpu", weights_only=False)
        loaded = load_training_pockets_with_report_from_dir(
            structure_dir=config.structure_dir, summary_csv=config.summary_csv, required_targets=("metal",),
            metal_eligibility_scheme=config.metal_eligibility_scheme, esm_dim=config.esm_dim,
            esm_embeddings_dir=config.esm_embeddings_dir, require_esm_embeddings=config.require_esm_embeddings,
            external_features_root_dir=config.external_features_root_dir, external_feature_source=config.external_feature_source,
            require_external_features=config.require_external_features, invalid_structure_policy="error",
            allowed_structure_ids={r["structure_id"] for r in rows})
        require({p.pocket_id for p in loaded.pockets} == {r["example_id"] for r in rows}, "Outer loaded membership mismatch")
        pockets = select_exact(loaded.pockets, rows)
        options = {key: getattr(config, key) for key in ("esm_dim", "edge_radius", "use_ring_edges", "require_ring_edges",
                   "shell_role_source", "node_feature_set", "omit_node_features", "metal_node_mode")}
        graphs = build_graph_data_list(pockets, **options)
        dataset = PocketGraphDataset(pockets, precomputed_data=graphs,
                  normalization_stats=normalization_stats_from_payload(checkpoint["normalization_stats"]), **options)
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
        ec_labels = {int(k): v for k, v in checkpoint["ec_labels"].items()}
        kwargs = factory_kwargs(root / "src/training/run.py", checkpoint["config"], checkpoint, METAL_TARGET_LABELS, ec_labels)
        model = build_pocket_classifier(**kwargs).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        predictions = evaluate_epoch_with_predictions(model, loader, device=device)
        require(predictions["metal_y"].tolist() == [r["target"] for r in rows], "Outer inference target/order mismatch")
        exported = prediction_rows(predictions["metal_logits"], predictions["metal_y"], rows, METAL_TARGET_LABELS)
        write_rows(destination / f"{family}_predictions.csv", exported)
        results[family] = to_jsonable(metrics_from_predictions(
            predictions, "outer", task="metal", ec_label_map=ec_labels,
            ec_label_depth=config.ec_label_depth))
        del model, loader, graphs, dataset, loaded, pockets, checkpoint
    write_json(destination / "metrics.json", results)
    write_json(destination / "completed.json", {"frozen_pilot_sha256": frozen_sha256, "completed_at_unix": time.time(),
               "models": list(FAMILIES), "protected_test_access": False,
               "prediction_sha256": {f: sha256(destination / f"{f}_predictions.csv") for f in FAMILIES}})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze-split", "freeze-five-class", "preflight", "train", "freeze-checkpoints", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frozen-sha256")
    parser.add_argument("--source-runs", type=Path)
    args = parser.parse_args()
    output, root = args.output.resolve(), args.root.resolve()
    if args.action == "freeze-split":
        result = freeze_split(args.previous, output, root)
    elif args.action == "freeze-five-class":
        require(args.previous is not None and args.source_runs is not None, "Parent pilot and saved source runs are required")
        result = freeze_five_class(args.previous.resolve(), output, root, args.source_runs.resolve())
    elif args.action == "freeze-checkpoints":
        result = freeze_checkpoints(output)
    elif args.action == "evaluate":
        result = evaluate_outer(output, root, args.frozen_sha256, args.device)
    else:
        require(args.family in FAMILIES, "An explicit model family is required")
        config = load_config(output, args.family, root, args.device)
        if args.action == "preflight":
            result = load_membership(config)["receipt"]
        else:
            plan = read_json(output / "pilot_plan.json")
            validate_execution_device(plan, args.device, fitting=True)
            if plan.get("execution", {}).get("local_only"):
                verify_local_admission(output, root)
                import torch
                torch.set_num_threads(plan["execution"]["torch_threads"])
                torch.set_num_interop_threads(plan["execution"]["interop_threads"])
            from training.run import run_training
            require(not (output / "frozen_pilot.json").exists() and not (output / "outer_evaluation").exists(), "Pilot already frozen/opened")
            write_json(output / f"{args.family}_fit_started.json", {"started_at_unix": time.time(), "pilot_plan_sha256": sha256(output / "pilot_plan.json")})
            start = time.monotonic()
            directory = run_training(config)
            result = verify_completed_run(directory, args.family, config.explicit_membership_sha256)
            result["elapsed_seconds"] = time.monotonic() - start
            write_json(output / f"{args.family}_fit_completed.json", result)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
