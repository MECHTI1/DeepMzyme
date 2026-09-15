"""Certify a retained-cohort RING comparison without training or test access.

The caller supplies matched planned Only-GVP smoke configurations and the
parent's expected_split.json payload. Set both devices to CPU for a local
audit. ESM caches are hashed for a later fusion arm, never deserialized here.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, replace
import gc
from pathlib import Path
import time

import torch

from data_structures import (
    EDGE_SOURCE_TO_INDEX, GRAPH_EDGE_TENSOR_FIELDS, GRAPH_NODE_MASK_FIELDS,
    GRAPH_NODE_TENSOR_FIELDS, GRAPH_SITE_GEOMETRY_TENSOR_FIELDS,
    GRAPH_SITE_TENSOR_FIELDS, GRAPH_TARGET_FIELDS,
)
from graph.edge_sources import normalize_ring_interaction_type
from graph.ring_edges import parse_ring_node_id, resolve_ring_edges_path, ring_edges_output_path
from label_schemes import metal_labels_for_scheme, normalize_metal_label_scheme_name
from run_metal_architecture_pilot import cohort_identity, digest, fingerprint, require, save
from structure_store import resolve_structure_files
from training.esm_feature_loading import embedding_path_candidates
from training.run import normalization_stats_payload, prepare_run, set_seed, to_jsonable
from training.runtime_preparation import updated_external_feature_path_candidates


NODE_AND_SITE_FIELDS = (
    *GRAPH_NODE_TENSOR_FIELDS, *GRAPH_NODE_MASK_FIELDS, *GRAPH_SITE_TENSOR_FIELDS,
    *GRAPH_SITE_GEOMETRY_TENSOR_FIELDS, *GRAPH_TARGET_FIELDS,
)
EDGE_NORMALIZATION_FIELDS = {"edge_dist_raw", "edge_seqsep"}


def validate_configs(off, on):
    """Reject unsafe or scientifically unmatched inputs before preparation."""
    for config in (off, on):
        require(not config.run_test_eval and config.test_structure_dir is None
                and config.test_summary_csv is None, "RING audit forbids held-out inputs/evaluation")
        require(config.task == "metal" and config.model_architecture == "only_gvp"
                and not config.use_esm_branch and not config.require_esm_embeddings
                and config.esm_embeddings_dir is None, "Input audit requires graph-only preparation without ESM loading")
        require(normalize_metal_label_scheme_name(config.metal_label_scheme) == "merge_fe_class_viii",
                "RING audit requires direct four-class labels")
        require(config.shell_role_source == "geometry" and config.metal_node_mode == "none"
                and config.site_geometry_features == "legacy", "RING audit requires fixed geometric roles and legacy residue-only graphs")
        require(config.position_noise_std == config.second_shell_dropout == config.outer_residue_dropout == 0,
                "Audit configuration must disable augmentation")
        require(not config.prepare_missing_ring_edges and not config.prepare_missing_esm_embeddings,
                "Audit must not generate or replace feature caches")
        require(Path(config.structure_dir).name == "train", "Audit accepts the non-test train directory only")
        require(config.summary_csv is not None and Path(config.summary_csv).parent.resolve() == Path(config.structure_dir).resolve(),
                "Site summary must belong to the same train directory")
        require(config.external_features_root_dir is not None and config.require_external_features,
                "Certified external features are required")
    require(not off.use_ring_edges and not off.require_ring_edges, "The off configuration must disable RING")
    require(on.use_ring_edges and on.require_ring_edges and on.ring_features_dir is not None,
            "The on configuration must require the pinned RING cache")
    allowed = {"run_name", "runs_dir", "use_ring_edges", "require_ring_edges", "ring_features_dir"}
    left, right = asdict(off), asdict(on)
    mismatched = [key for key in left if key not in allowed and left[key] != right[key]]
    require(not mismatched, "Non-RING configuration mismatch: " + ", ".join(mismatched))


def _check_expected(path, root, record, expected, *, label):
    if expected is None:
        return
    candidates = (str(path), str(path.resolve()), str(path.relative_to(root)))
    matches = [expected[key] for key in candidates if key in expected]
    require(matches, f"{label} provenance is missing {path}")
    for item in matches:
        checksum = item if isinstance(item, str) else item["sha256"]
        require(checksum == record["sha256"], f"{label} checksum mismatch: {path}")
        if isinstance(item, dict):
            size = item.get("bytes", item.get("size"))
            require(size is None or int(size) == record["bytes"], f"{label} size mismatch: {path}")


def _ring_syntax(path):
    counts = {"rows": 0, "supported_rows": 0, "unsupported_interaction_rows": 0}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        require({"NodeId1", "NodeId2", "Interaction", "Atom1", "Atom2"}.issubset(reader.fieldnames or []),
                f"Invalid RING header: {path}")
        for row in reader:
            counts["rows"] += 1
            if normalize_ring_interaction_type(row.get("Interaction", "")) is None:
                counts["unsupported_interaction_rows"] += 1
                continue
            for field in ("NodeId1", "NodeId2"):
                try:
                    parse_ring_node_id(row[field])
                except (TypeError, ValueError, KeyError) as exc:
                    raise ValueError(f"Malformed RING node at row {counts['rows']} in {path}") from exc
            counts["supported_rows"] += 1
    require(counts["rows"] > 0, f"Empty RING edge file: {path}")
    return counts


def collect_training_cache(config, *, expected_ring_files=None, expected_feature_files=None):
    """Hash all train-structure caches; do not load ESM tensors or held-out files."""
    train = Path(config.structure_dir).resolve()
    data_root = train.parent.parent
    ring_root = Path(config.ring_features_dir).resolve()
    external_root = Path(config.external_features_root_dir).resolve()
    esm_root = data_root / "esm_embeddings"
    records, ring_reports, structures = {}, {}, resolve_structure_files(train, recursive_legacy_scan=False)
    require(structures, "No training structures found for the cache audit")

    def add(path, kind, root, expected=None):
        path = Path(path).resolve()
        require(path.is_file(), f"Missing {kind} cache: {path}")
        require(path.is_relative_to(root), f"{kind} cache escaped its declared root: {path}")
        if str(path) not in records:
            before = path.stat()
            record = dict(path=str(path), bytes=before.st_size, mtime_ns=before.st_mtime_ns,
                          sha256=digest(path), kind=kind)
            after = path.stat()
            require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                    f"Input changed during hashing: {path}")
            require(record["bytes"] > 0, f"Empty {kind} cache: {path}")
            records[str(path)] = record
        _check_expected(path, root, records[str(path)], expected, label=kind)
        return path

    for structure in structures:
        external = next((path for path in updated_external_feature_path_candidates(
            structure, structure_root=train, external_features_root_dir=external_root,
        ) if path.is_file()), None)
        require(external is not None, f"Missing external cache: {structure.name}")
        add(external, "external", external_root, expected_feature_files)
        embeddings = [path for path in embedding_path_candidates(esm_root, structure) if path.is_file()]
        require(embeddings, f"Missing ESM cache: {structure.name}")
        for embedding in embeddings:
            add(embedding, "esm", esm_root, expected_feature_files)
            add(Path(str(embedding) + ".json"), "esm_metadata", esm_root, expected_feature_files)
        ring = add(ring_edges_output_path(ring_root, structure), "ring", ring_root, expected_ring_files)
        if str(ring) not in ring_reports:
            ring_reports[str(ring)] = _ring_syntax(ring)
    return dict(
        structures=len(structures), files=sorted(records.values(), key=lambda item: item["path"]),
        ring_files=ring_reports, missing_esm=0, missing_external=0, missing_ring=0,
        ring_provenance_pinned=expected_ring_files is not None,
        external_and_esm_provenance_pinned=expected_feature_files is not None,
        esm_tensor_loading=False,
    )


def _finite(value, label):
    require(isinstance(value, torch.Tensor), f"Missing tensor: {label}")
    require(bool(torch.isfinite(value).all()), f"Nonfinite tensor: {label}")


def _graph_edges(graph, label):
    for field in GRAPH_EDGE_TENSOR_FIELDS:
        _finite(getattr(graph, field, None), f"{label}/{field}")
    index = graph.edge_index
    require(index.dtype == torch.long and index.ndim == 2 and index.shape[0] == 2 and index.shape[1] > 0,
            f"Invalid edge index shape/type: {label}")
    require(int(index.min()) >= 0 and int(index.max()) < graph.num_nodes, f"Out-of-range edge endpoint: {label}")
    pairs = list(zip(index[0].tolist(), index[1].tolist()))
    pair_set = set(pairs)
    require(len(pair_set) == len(pairs), f"Duplicate directed edges: {label}")
    require(all(left != right for left, right in pairs), f"Unexpected self edge: {label}")
    require(all((right, left) in pair_set for left, right in pairs), f"Missing reverse edge: {label}")
    for field in GRAPH_EDGE_TENSOR_FIELDS[1:]:
        require(getattr(graph, field).shape[0] == len(pairs), f"Misaligned edge feature: {label}/{field}")
    return {pair: index for index, pair in enumerate(pairs)}


def compare_graphs(off, on, label):
    """Compare the actual unnormalized model inputs for one retained pocket."""
    require(off.num_nodes == on.num_nodes and off.num_nodes > 0, f"Node count changed: {label}")
    for field in NODE_AND_SITE_FIELDS:
        left, right = getattr(off, field, None), getattr(on, field, None)
        _finite(left, f"{label}/off/{field}")
        _finite(right, f"{label}/on/{field}")
        require(torch.equal(left, right), f"Node/site input changed: {label}/{field}")
    require(bool(off.residue_node_mask.all()) and not bool(off.metal_node_mask.any()),
            f"Unexpected explicit metal node: {label}")
    off_edges, on_edges = _graph_edges(off, label + "/off"), _graph_edges(on, label + "/on")
    require(off_edges.keys() <= on_edges.keys(), f"RING removed radius edges: {label}")
    shared_indices = torch.tensor([on_edges[pair] for pair in off_edges], device=on.edge_index.device)
    for field in ("edge_dist_raw", "edge_seqsep", "edge_same_chain", "edge_vector_raw"):
        require(torch.equal(getattr(off, field), getattr(on, field)[shared_indices]),
                f"RING replaced existing radius geometry: {label}/{field}")
    ring_column = EDGE_SOURCE_TO_INDEX["ring"]
    radius_column = EDGE_SOURCE_TO_INDEX["radius"]
    require(not bool(off.edge_source_type[:, ring_column].any()), f"RING edges present in off graph: {label}")
    require(bool(off.edge_source_type[:, radius_column].all()), f"Off graph is not radius-only: {label}")
    ring_flags = (on.edge_source_type[:, ring_column] > .5).tolist()
    ring_pairs = {pair for pair, index in on_edges.items() if ring_flags[index]}
    added = on_edges.keys() - off_edges.keys()
    require(added <= ring_pairs, f"Added edge lacks RING provenance: {label}")
    existing = ring_pairs & off_edges.keys()
    return dict(nodes=off.num_nodes, radius_undirected_pairs=len(off_edges) // 2,
                ring_added_undirected_pairs=len(added) // 2,
                ring_annotated_existing_undirected_pairs=len(existing) // 2,
                ring_undirected_pairs=len(ring_pairs) // 2,
                no_ring_effect=not ring_pairs)


def compare_normalizers(off, on):
    left, right = to_jsonable(normalization_stats_payload(off)), to_jsonable(normalization_stats_payload(on))
    require(off.clamp_value == on.clamp_value, "Normalization clamp changed")
    changes = {}
    for group in ("means", "stds"):
        first, second = getattr(off, group), getattr(on, group)
        require(first.keys() == second.keys(), "Normalizer fields changed")
        for field in first:
            _finite(first[field], f"off/{group}/{field}")
            _finite(second[field], f"on/{group}/{field}")
            require(first[field].shape == second[field].shape, f"Normalizer dimensions changed: {field}")
            if not torch.equal(first[field], second[field]):
                require(field in EDGE_NORMALIZATION_FIELDS, f"Non-edge normalization changed: {group}/{field}")
                changes[f"{group}/{field}"] = float((first[field] - second[field]).abs().max())
    return dict(non_edge_normalization_identical=True, changed_edge_statistics=changes,
                off_sha256=fingerprint(left), on_sha256=fingerprint(right), off=left, on=right,
                interpretation="Per-arm training-edge normalization is part of the RING representation comparison.")


def audit(off_config, on_config, expected_split, output, *, expected_ring_files=None, expected_feature_files=None):
    """Prepare and audit both arms; return/save a receipt only after every check."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    report_path, cache_path = output / "ring_input_audit.json", output / "training_cache_audit.json"
    save(report_path, dict(status="running", started_epoch=started, held_out_evaluation=False))
    save(cache_path, dict(status="running", files=[], failures=["Input audit in progress"]))
    try:
        validate_configs(off_config, on_config)
        parent_scheme = normalize_metal_label_scheme_name(expected_split.get("metal_label_scheme", ""))
        require(parent_scheme == "split_all_metals", "Parent reference must use the native six-class metal scheme")
        parent_labels = metal_labels_for_scheme(parent_scheme)
        require([parent_labels[index] for index in range(6)] == ["Mn", "Cu", "Zn", "Fe", "Co", "Ni"],
                "Parent native metal label order changed")
        expected_identity = cohort_identity(expected_split)
        cache = collect_training_cache(on_config, expected_ring_files=expected_ring_files,
                                       expected_feature_files=expected_feature_files)
        off_audit_config = replace(off_config, runs_dir=output / "input_audit_runs", run_name="ring_off")
        on_audit_config = replace(on_config, runs_dir=output / "input_audit_runs", run_name="ring_on")
        set_seed(off_audit_config.seed, deterministic=off_audit_config.deterministic)
        off = prepare_run(off_audit_config)
        require(cohort_identity(off.dataset_summary) == expected_identity, "Retained cohort differs from parent")
        off_graphs = {}
        for part, loader_name in (("train", "train_loader"), ("validation", "val_loader")):
            loader = getattr(off, loader_name)
            require(loader is not None and loader.dataset.precomputed_data is not None,
                    f"Missing precomputed {part} graphs")
            off_graphs[part] = loader.dataset.precomputed_data
        off_normalization = off.normalization_stats
        # Graphs contain tensors only; release parsed atoms, pockets, and model
        # before the second full preparation on memory-limited local machines.
        del loader, off
        gc.collect()
        set_seed(on_audit_config.seed, deterministic=on_audit_config.deterministic)
        on = prepare_run(on_audit_config)
        require(cohort_identity(on.dataset_summary) == expected_identity, "Retained cohort differs from parent")
        known_ring = {item["path"] for item in cache["files"] if item["kind"] == "ring"}
        rows, splits = [], {}
        for part, loader_name, pocket_name in (("train", "train_loader", "train_pockets"),
                                                ("validation", "val_loader", "val_pockets")):
            right_loader = getattr(on, loader_name)
            require(right_loader is not None, f"Missing {part} loader")
            left, right = off_graphs[part], right_loader.dataset.precomputed_data
            pockets = getattr(on.split, pocket_name)
            require(left is not None and right is not None and len(left) == len(right) == len(pockets) == len(expected_identity[part]),
                    f"Incomplete precomputed {part} graphs")
            for index, (off_graph, on_graph, pocket) in enumerate(zip(left, right, pockets)):
                expected = expected_identity[part][index]
                require(pocket.pocket_id == expected["pocket_id"] and pocket.structure_id == expected["structure_id"],
                        f"Graph/pocket order differs: {part}/{index}")
                expected_target = int(expected_split["retained_split_identity"][part]["examples"][index]["y_metal"])
                require(0 <= expected_target < 6, f"Invalid parent native target: {pocket.pocket_id}")
                require(int(on_graph.y_metal.item()) == min(expected_target, 3), f"Common-four target differs: {pocket.pocket_id}")
                ring_path = resolve_ring_edges_path(pocket)
                require(ring_path is not None and str(ring_path.resolve()) in known_ring,
                        f"Pocket did not resolve its audited RING input: {pocket.pocket_id}")
                row = compare_graphs(off_graph, on_graph, pocket.pocket_id)
                rows.append(dict(part=part, **expected, **row))
            part_rows = [row for row in rows if row["part"] == part]
            splits[part] = dict(pockets=len(part_rows),
                               pockets_without_ring_effect=sum(row["no_ring_effect"] for row in part_rows),
                               pockets_with_added_pairs=sum(row["ring_added_undirected_pairs"] > 0 for row in part_rows),
                               pockets_with_existing_annotations=sum(row["ring_annotated_existing_undirected_pairs"] > 0 for row in part_rows),
                               ring_added_undirected_pairs=sum(row["ring_added_undirected_pairs"] for row in part_rows),
                               ring_annotated_existing_undirected_pairs=sum(row["ring_annotated_existing_undirected_pairs"] for row in part_rows))
        require(any(not row["no_ring_effect"] for row in rows), "RING produces no input effect across the retained cohort")
        normalization = compare_normalizers(off_normalization, on.normalization_stats)
        for record in cache["files"]:
            info = Path(record["path"]).stat()
            require((info.st_size, info.st_mtime_ns) == (record["bytes"], record["mtime_ns"]),
                    f"Feature changed during preparation: {record['path']}")
        save(cache_path, dict(status="passed", failures=[], **cache))
        report = dict(status="passed", started_epoch=started, elapsed_seconds=time.time() - started,
                      training_performed=False, held_out_evaluation=False, shell_role_source="geometry",
                      expected_cohort_sha256=fingerprint(expected_identity),
                      parent_metal_label_scheme=parent_scheme, parent_metal_labels=parent_labels,
                      off_config_sha256=fingerprint(to_jsonable(asdict(off_config))),
                      on_config_sha256=fingerprint(to_jsonable(asdict(on_config))),
                      audit_device=off_config.device, node_and_site_fields=list(NODE_AND_SITE_FIELDS),
                      all_node_and_site_tensors_identical=True, all_edge_tensors_finite=True,
                      edge_endpoints_valid=True, shared_radius_geometry_identical=True,
                      splits=splits, pockets=rows, normalization=normalization,
                      cache_audit_sha256=digest(cache_path),
                      cache_coverage={key: value for key, value in cache.items() if key not in {"files", "ring_files"}},
                      raw_ring_angle_column_consumed=False,
                      limitations=["RING annotations and added edges are tested together with per-arm fitted edge normalization.",
                                   "Zero in-pocket RING effects are reported, not grounds for dropping a pocket.",
                                   "ESM tensor values are not deserialized; supplied verified hashes provide prior integrity provenance.",
                                   "This input audit provides no validation performance or architecture promotion."])
        save(report_path, report)
        return report
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        save(report_path, dict(status="failed", started_epoch=started, elapsed_seconds=time.time() - started,
                               held_out_evaluation=False, failures=[failure]))
        save(cache_path, dict(status="failed", files=[], failures=[failure]))
        raise
