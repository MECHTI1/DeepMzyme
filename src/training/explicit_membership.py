"""Hash-bound, development-only diagnostic memberships. No inference or fitting.

Outer labels are read only for metadata integrity/support checks. Outer structures,
feature tensors and graph loaders are never deserialized here. A trusted frozen
development allowlist, not a test dataset, establishes the exclusion boundary.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

PARTITIONS = ("train", "inner_validation", "outer_evaluation")
ENDPOINT = "diagnostic coordinate-chain sequence-remote"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def checked_path(base, reference):
    path = (Path(base) / reference["path"]).resolve()
    require(path.is_file(), f"Missing frozen artifact: {path}")
    require(sha256(path) == reference["sha256"], f"Frozen checksum mismatch: {path}")
    return path


def read_rows(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if "target" in row and "example_id" in row:
            row["target"] = int(row["target"])
    return rows


def diagnostic_class_count(label_scheme):
    from label_schemes import METAL_LABEL_SCHEMES, normalize_metal_label_scheme_name
    scheme = normalize_metal_label_scheme_name(label_scheme)
    require(scheme in ("merge_fe_class_viii", "five_class"),
            "Explicit diagnostic membership requires four-class or five-class metal labels")
    return len(METAL_LABEL_SCHEMES[scheme][0])


def counts(rows, label_scheme="merge_fe_class_viii"):
    return {
        "pockets": len(rows),
        "groups": len({r["group_id"] for r in rows}),
        "components": len({r["component_id"] for r in rows}),
        "classes": {str(c): {
            "pockets": sum(r["target"] == c for r in rows),
            "groups": len({r["group_id"] for r in rows if r["target"] == c}),
            "components": len({r["component_id"] for r in rows if r["target"] == c}),
        } for c in range(diagnostic_class_count(label_scheme))},
    }


def validate_rows(partitions, development, edges, *, label_scheme="merge_fe_class_viii"):
    n_classes = diagnostic_class_count(label_scheme)
    allowed = {r["example_id"]: r for r in development}
    require(len(allowed) == len(development), "Duplicate scientific example in development allowlist")
    for row in development:
        require(str(row.get("protected_test", False)).lower() in ("false", "0", ""),
                f"Protected-test membership in development allowlist: {row['example_id']}")
    ownership = {key: {} for key in ("example_id", "group_id", "protein_id", "component_id", "structure_id")}
    seen = set()
    for part in PARTITIONS:
        rows = partitions[part]
        local = set()
        for row in rows:
            eid = row["example_id"]
            require(eid not in local, f"Duplicate scientific example in {part}: {eid}")
            local.add(eid)
            require(not row.get("protected_test", False) or str(row["protected_test"]).lower() in ("false", "0"),
                    f"Protected-test membership: {eid}")
            require(eid in allowed, f"Unknown ID or protected-test membership outside development allowlist: {eid}")
            for key in ("group_id", "protein_id", "structure_id", "target", "component_id"):
                require(row.get(key) == allowed[eid].get(key), f"Membership {key} mismatch: {eid}")
            require(row["target"] in range(n_classes), f"Target outside {label_scheme} scheme: {eid}")
            for key, owners in ownership.items():
                value = row.get(key)
                require(bool(value), f"Unresolved {key}: {eid}")
                require(value not in owners or owners[value] == part, f"Partition overlap/crossing for {key}: {value}")
                owners[value] = part
            seen.add(eid)
        support = counts(rows, label_scheme)["classes"]
        minimum = {"train": 5, "inner_validation": 3, "outer_evaluation": 5}[part]
        for c in range(n_classes):
            require(support[str(c)]["pockets"] > 0, f"Missing active class {c} in {part}")
            require(support[str(c)]["components"] >= minimum,
                    f"Insufficient {part} class-component support for {c}; requires {minimum}")
    require(seen == set(allowed), "Explicit membership does not exactly cover the frozen development allowlist")
    for edge in edges:
        left, right = edge["query_group"], edge["target_group"]
        require(left in ownership["group_id"] and right in ownership["group_id"], "Unresolved detected-edge group")
        require(ownership["group_id"][left] == ownership["group_id"][right],
                "Detected >20%-identity/exact-alias edge crosses partitions")
    return {part: counts(partitions[part], label_scheme) for part in PARTITIONS}


def relabel_five_class_memberships(parent_parts, source_payloads):
    """Recover native targets from two agreeing development-only saved runs."""
    require(set(source_payloads) == {"only_esm", "only_gvp"}, "Both five-class label sources are required")
    parents = [r for p in PARTITIONS for r in parent_parts[p]]
    identities = []
    for family, payload in source_payloads.items():
        cfg = payload["config"]
        require(cfg["task"] == "metal" and cfg["metal_label_scheme"] == "five_class"
                and cfg["model_architecture"] == family and cfg["seed"] == 42,
                "Five-class label source configuration mismatch")
        require(not cfg["run_test_eval"] and payload.get("test_report") is None,
                "Five-class label source must be development-only")
        split = payload["dataset_summary"]["retained_split_identity"]
        require(set(split) == {"train", "validation"}, "Unexpected label-source partition")
        examples = [r for part in split.values() for r in part["examples"]]
        lookup = {r["pocket_id"]: {k: r[k] for k in ("structure_id", "group", "y_metal")} for r in examples}
        require(len(lookup) == len(examples), "Duplicate scientific example in label source")
        require(set(lookup) == {r["example_id"] for r in parents}, "Five-class source membership differs from parent")
        for row in parents:
            native = lookup[row["example_id"]]
            require(native["structure_id"] == row["structure_id"] and native["group"] == row["group_id"],
                    "Five-class source structure/group differs from parent")
            require(native["y_metal"] in range(5) and min(native["y_metal"], 3) == row["target"],
                    "Five-class target does not collapse to parent target")
        identities.append(lookup)
    require(identities[0] == identities[1], "Five-class label sources disagree")
    return {part: [dict(row, target=identities[0][row["example_id"]]["y_metal"]) for row in parent_parts[part]]
            for part in PARTITIONS}


def verify_five_class_provenance(base, spec, parts):
    require("parent_split" in spec and "label_sources" in spec, "Five-class membership requires frozen label provenance")
    parent_path = checked_path(base, spec["parent_split"])
    parent = read_json(parent_path)
    require(parent["label_scheme"] == "merge_fe_class_viii", "Five-class parent must be direct four-class")
    require(parent["dataset"] == spec["dataset"], "Five-class parent dataset mismatch")
    for name in ("detected_edges", "search_protocol", "feature_inventory"):
        require(parent[name]["sha256"] == spec[name]["sha256"], f"Five-class parent {name} changed")
    parent_parts = {p: read_rows(checked_path(parent_path.parent, parent["manifests"][p])) for p in PARTITIONS}
    sources = {family: read_json(checked_path(base, ref)) for family, ref in spec["label_sources"].items()}
    require(relabel_five_class_memberships(parent_parts, sources) == parts,
            "Five-class manifests changed parent memberships or native labels")


def validate_mode(config):
    enabled = bool(config.explicit_membership_manifest)
    require(enabled == bool(config.explicit_membership_sha256), "Explicit membership requires descriptor AND frozen checksum")
    if not enabled:
        return
    require(config.task == "metal", "Explicit diagnostic membership requires the metal task")
    diagnostic_class_count(config.metal_label_scheme)
    require(not config.run_test_eval and config.test_structure_dir is None and config.test_summary_csv is None
            and not config.allow_final_refit_test_eval and not config.allow_train_loss_test_eval_debug,
            "Protected held-out test routes are forbidden for explicit diagnostic membership")
    require(config.selection_metric in (None, "val_metal_balanced_acc"), "Selection must use inner-validation metal balanced accuracy")
    require(config.n_folds is None and config.fold_index is None, "Automatic folds cannot modify explicit membership")
    require(config.split_seed == 20260917 and config.train_val_split_by == "pdbid", "Diagnostic split seed/group mismatch")
    require(not config.prepare_missing_esm_embeddings and not config.prepare_missing_ring_edges,
            "Feature/cache generation is forbidden in explicit diagnostic mode")
    require(config.model_architecture in ("only_esm", "only_gvp") and not config.use_ring_edges
            and not config.use_early_esm, "Diagnostic mode supports only the two standalone, RING-off systems")


def load_membership(config, *, verify_features=True):
    validate_mode(config)
    if not config.explicit_membership_manifest:
        return None
    path = Path(config.explicit_membership_manifest).resolve()
    require(sha256(path) == config.explicit_membership_sha256, "Frozen manifest checksum differs")
    spec = read_json(path)
    require(spec["endpoint"] == ENDPOINT and spec["split_seed"] == config.split_seed, "Diagnostic protocol identity mismatch")
    from label_schemes import normalize_metal_label_scheme_name
    scheme = normalize_metal_label_scheme_name(config.metal_label_scheme)
    require(spec["label_scheme"] == scheme, "Manifest label-scheme mismatch")
    require(spec["dataset"]["scope"] == "external_training_only", "Protected-test dataset is forbidden")
    require(spec["dataset"]["bundle_id"] == config.dataset_bundle_id
            and spec["dataset"]["bundle_sha256"] == config.dataset_bundle_sha256, "Dataset identity mismatch")
    require(sha256(config.summary_csv) == spec["dataset"]["summary_sha256"], "Dataset summary checksum mismatch")
    require(sha256(Path(config.structure_dir) / "structure_manifest.csv") == spec["dataset"]["structure_manifest_sha256"],
            "Dataset structure membership mismatch")
    paths = {part: checked_path(path.parent, spec["manifests"][part]) for part in PARTITIONS}
    parts = {part: read_rows(p) for part, p in paths.items()}
    development = read_rows(checked_path(path.parent, spec["development_allowlist"]))
    edges = read_rows(checked_path(path.parent, spec["detected_edges"]))
    checked_path(path.parent, spec["search_protocol"])
    support = validate_rows(parts, development, edges, label_scheme=scheme)
    if scheme == "five_class":
        verify_five_class_provenance(path.parent, spec, parts)
    require(support == spec["counts"], "Frozen class/component counts differ")
    if verify_features:
        verify_feature_files(config, spec, path.parent, development)
    receipt = {"enabled": True, "endpoint": ENDPOINT, "descriptor_path": str(path),
               "descriptor_sha256": config.explicit_membership_sha256,
               "manifest_paths": {k: str(v) for k, v in paths.items()},
               "manifest_sha256": {k: spec["manifests"][k]["sha256"] for k in PARTITIONS},
               "split_seed": spec["split_seed"], "counts": support, "dataset": spec["dataset"],
               "label_scheme": spec["label_scheme"], "search_protocol": spec["search_protocol"],
               "source_control_at_freeze": spec["source_control"],
               "outer_available_during_fitting": False, "protected_test_access": False}
    if scheme == "five_class":
        receipt.update(parent_split=spec["parent_split"], label_sources=spec["label_sources"],
                       prior_four_class_outer_observed=spec.get("prior_four_class_outer_observed", True))
    return {"partitions": parts, "receipt": receipt}


def verify_feature_files(config, spec, base, development):
    # Hash bytes only, including outer caches: no tensor/structure deserialization.
    from training.structure_loading import find_structure_files
    structures = {p.stem: p for p in find_structure_files(Path(config.structure_dir))}
    inventory = read_rows(checked_path(base, spec["feature_inventory"]))
    require(len({r["structure_id"] for r in inventory}) == len(inventory),
            "Duplicate structure in frozen feature inventory")
    require({r["structure_id"] for r in inventory} == {r["structure_id"] for r in development},
            "Feature audit does not cover exact development membership")
    for row in inventory:
        sid = row["structure_id"]
        require(sid in structures, f"Missing required structure: {sid}")
        files = [(structures[sid], row["structure_sha256"]),
                 (Path(config.external_features_root_dir) / sid / "residue_features.json", row["external_sha256"])]
        if config.require_esm_embeddings:
            files.append((Path(config.esm_embeddings_dir) / row["esm_file"], row["esm_sha256"]))
        for file, expected in files:
            require(file.is_file(), f"Missing required structure/cached feature: {file}")
            require(sha256(file) == expected, f"Required input checksum mismatch: {file}")


def select_exact(pockets, rows):
    lookup = {p.pocket_id: p for p in pockets}
    require(len(lookup) == len(pockets), "Duplicate loaded scientific example")
    selected = []
    for row in rows:
        require(row["example_id"] in lookup, f"Unknown/unresolved loaded ID: {row['example_id']}")
        p = lookup[row["example_id"]]
        require(p.structure_id == row["structure_id"] and p.y_metal == row["target"], "Manifest target/structure differs from loaded target")
        from training.splits import pocket_split_key
        require(pocket_split_key(p, "pdbid") == row["group_id"], "Loaded PDB/protein group mismatch")
        selected.append(p)
    return selected


def fixed_split(pockets, membership):
    from training.splits import PocketSplit
    parts = membership["partitions"]
    expected = parts["train"] + parts["inner_validation"]
    require({p.pocket_id for p in pockets} == {r["example_id"] for r in expected},
            "Loaded fitting membership changed; outer/unknown/missing examples are forbidden")
    return PocketSplit(select_exact(pockets, parts["train"]), select_exact(pockets, parts["inner_validation"]))
