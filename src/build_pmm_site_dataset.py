"""Audit PinMyMetal source rows and build an isolated, evidence-backed site cohort.

This command never infers a PMM row's ion from PDB ID alone. A training-ready
cohort requires an independently documented row-to-ion crosswalk.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


PROFILE_VERSION = "very_exact-pmm_sets-v1"
PMM_COMMIT = "59ef46795920322c798db4e5ec500b04451f7904"
SOURCE_HASHES = {
    "train": "4748babd2b6ac0706cd9ed4bcfd4855c8d2c5535f01813fb2e10b68b31a24c0f",
    "test": "ec6427ad0b8a18261dbc1d6822bdcfe7194730c7da4a36c208957bcb00f6a0f2",
}
TRAINING_SCRIPT_HASH = "9b5463fc7623ed72468ebd33b43d26def50d3ead3f33672511df9feef0603445"
# Verified in PMM's script10_clasmodel_result.sql at the pinned commit.
LABELS = {"1": "MN", "2": "CLASS_VIII", "6": "CU", "7": "ZN"}
ELEMENTS = {"MN": {"MN"}, "CLASS_VIII": {"FE", "CO", "NI"}, "CU": {"CU"}, "ZN": {"ZN"}}
IDENTITY_COLUMNS = ("pdbid", "residueid_ion", "metalid", "label_metal")
EVIDENCE_COLUMNS = (
    "source_uid", "pdbid", "residueid_ion", "metalid", "coordinate_path",
    "model", "chain", "resseq", "icode", "altloc", "element",
    "structure_accession", "structure_version", "label_chain", "symmetry_operator",
    "context_complete", "coordinate_sha256", "evidence_kind", "evidence_reference",
    "evidence_sha256", "reviewer",
)
CROSSWALK_COLUMNS = [
    "source_uid", "source_sha256", "original_side", "source_row_number", "pdbid",
    "residueid_ion", "metalid", "label_metal", "four_class_target",
    "effective_dropna", "mapping_status", "mapping_reason", "physical_site_group",
    "model", "author_chain", "chain", "resseq", "icode", "altloc", "element", "serial",
    "x", "y", "z", "coordinate_path", "coordinate_sha256", "evidence_kind",
    "evidence_reference", "evidence_sha256", "reviewer", "structure_accession",
    "structure_version", "label_chain", "symmetry_operator", "context_complete", "graph_path",
]
ACCEPTED_EVIDENCE = {"pmm_neighborhood_residue_record", "deposition_site_coordinate"}
DEFAULT_SOURCES = Path(__file__).resolve().parents[1] / "prepare_training_and_test_set" / "pinmymetal_files"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_source(path: Path, side: str) -> tuple[list[dict], dict]:
    checksum = sha256(path)
    if checksum != SOURCE_HASHES[side]:
        raise ValueError(f"PMM {side} source checksum mismatch: {checksum}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(IDENTITY_COLUMNS).issubset(reader.fieldnames):
            raise ValueError(f"Missing PMM identity columns in {path}")
        raw = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in raw):
        raise ValueError(f"Malformed CSV row in {path}")
    # Reproduce the released train_chedhclassmodel.py read_csv().dropna() path.
    frame = pd.read_csv(path)
    if len(frame) != len(raw):
        raise ValueError(f"pandas/raw row count mismatch for {path}")
    effective_indices = set(frame.dropna().index.tolist())
    # The released script also calls Index.drop on these columns. Its current
    # source tables omit `source`, so do not describe post-dropna as a completed
    # model-training cohort.
    script_drop_columns = {"metalid", "pdbid", "residueid_ion", "label_metal", "source", "ched_count"}
    missing_script_columns = sorted(script_drop_columns - set(frame.columns))
    rows = []
    for index, row in enumerate(raw):
        label = row["label_metal"].strip()
        if label not in LABELS:
            raise ValueError(f"Unknown PMM label {label!r} at {side} row {index + 1}")
        pdbid = row["pdbid"].strip().lower()
        if not pdbid or not row["residueid_ion"].strip() or not row["metalid"].strip():
            raise ValueError(f"Missing source identity at {side} row {index + 1}")
        rows.append({
            "source_uid": f"sha256:{checksum}:row:{index + 1}",
            "source_sha256": checksum,
            "original_side": side,
            "source_row_number": index + 1,
            "pdbid": pdbid,
            "residueid_ion": row["residueid_ion"].strip(),
            "metalid": row["metalid"].strip(),
            "label_metal": label,
            "four_class_target": LABELS[label],
            "effective_dropna": index in effective_indices,
        })
    if len({row["source_uid"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate source UID in {side}")
    duplicate_keys = Counter((row["pdbid"], row["residueid_ion"], row["metalid"]) for row in rows)
    return rows, {
        "sha256": checksum,
        "raw_rows": len(rows),
        "effective_dropna_rows": len(effective_indices),
        "raw_by_label": dict(sorted(Counter(row["label_metal"] for row in rows).items())),
        "effective_by_label": dict(sorted(Counter(row["label_metal"] for row in rows if row["effective_dropna"]).items())),
        "unique_pdbids": len({row["pdbid"] for row in rows}),
        "duplicate_identity_keys": sum(count - 1 for count in duplicate_keys.values() if count > 1),
        "released_script_missing_columns": missing_script_columns,
        "released_script_runs_unchanged": not missing_script_columns,
    }


def load_evidence(path: Path | None, source_rows: list[dict]) -> dict[str, list[dict]]:
    if path is None:
        return {}
    valid = {row["source_uid"]: row for row in source_rows}
    result: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(EVIDENCE_COLUMNS).issubset(reader.fieldnames):
            raise ValueError(f"Evidence CSV lacks columns {set(EVIDENCE_COLUMNS) - set(reader.fieldnames or [])}")
        for evidence in reader:
            uid = evidence["source_uid"]
            if uid not in valid:
                raise ValueError(f"Evidence references unknown source UID: {uid}")
            source = valid[uid]
            if any(evidence[key].strip().lower() != source[key].lower() for key in ("pdbid", "residueid_ion", "metalid")):
                raise ValueError(f"Evidence identity mismatch: {uid}")
            if evidence["evidence_kind"] not in ACCEPTED_EVIDENCE or not evidence["evidence_reference"] or not evidence["evidence_sha256"] or not evidence["reviewer"]:
                raise ValueError(f"Unreviewed or untraceable evidence: {uid}")
            result[uid].append(evidence)
    return result


def parse_ions(path: Path) -> list[dict]:
    ions = []
    model = 1
    with path.open(encoding="ascii", errors="replace") as handle:
        for line in handle:
            if line.startswith("MODEL "):
                model = int(line[10:14].strip())
            if not line.startswith("HETATM"):
                continue
            element = line[76:78].strip().upper()
            if element not in {"MN", "FE", "CO", "NI", "CU", "ZN"}:
                continue
            ions.append({
                "model": model, "chain": line[21].strip(), "resseq": line[22:26].strip(),
                "icode": line[26].strip(), "altloc": line[16].strip(), "element": element,
                "serial": line[6:11].strip(),
                "x": float(line[30:38]), "y": float(line[38:46]), "z": float(line[46:54]),
            })
    return ions


def verify_match(source: dict, evidence: dict, structure_root: Path) -> tuple[str, str, dict]:
    if evidence.get("structure_accession", "").strip().lower() != source["pdbid"].lower():
        return "incomplete", "structure_accession_mismatch", {}
    if not all(evidence.get(field, "").strip() for field in ("structure_version", "label_chain", "symmetry_operator")):
        return "incomplete", "structure_chain_or_symmetry_provenance_missing", {}
    if evidence.get("context_complete", "").strip().lower() != "yes":
        return "incomplete", "structural_context_not_certified", {}
    relative = Path(evidence["coordinate_path"])
    if relative.is_absolute() or ".." in relative.parts:
        return "incomplete", "coordinate_path_must_be_relative", {}
    filename = relative.name.lower()
    pdbid = source["pdbid"].lower()
    if not filename.startswith(pdbid) or (len(filename) > len(pdbid) and filename[len(pdbid)].isalnum()):
        return "incomplete", "coordinate_filename_does_not_identify_source_pdb", {}
    path = (structure_root / relative).resolve()
    if not path.is_relative_to(structure_root.resolve()) or not path.is_file():
        return "incomplete", "coordinate_file_unavailable", {}
    file_sha = sha256(path)
    expected_sha = evidence.get("coordinate_sha256", "").strip()
    if not expected_sha or file_sha != expected_sha:
        return "incomplete", "coordinate_checksum_missing_or_mismatch", {}
    try:
        ion_matches = [ion for ion in parse_ions(path) if all(
            str(ion[key]).upper() == evidence[key].strip().upper()
            for key in ("model", "chain", "resseq", "icode", "altloc", "element")
        )]
    except (ValueError, UnicodeError):
        return "incomplete", "coordinate_parse_failure", {}
    if not ion_matches:
        return "unmatched", "documented_ion_absent_from_coordinate_file", {}
    if len(ion_matches) != 1:
        return "ambiguous", "multiple_ions_match_documented_identity", {}
    ion = ion_matches[0]
    if ion["element"] not in ELEMENTS[source["four_class_target"]]:
        return "unmatched", "documented_ion_conflicts_with_source_label", {}
    for axis in ("x", "y", "z"):
        value = evidence.get(axis, "").strip()
        if value and abs(float(value) - ion[axis]) > 0.01:
            return "unmatched", "documented_coordinate_mismatch", {}
    return "exact", "documented_source_identifier_and_ion_verified", {
        **ion, "author_chain": ion["chain"], "coordinate_path": str(relative), "coordinate_sha256": file_sha,
        "evidence_kind": evidence["evidence_kind"],
        "evidence_reference": evidence["evidence_reference"],
        "evidence_sha256": evidence["evidence_sha256"],
        "reviewer": evidence["reviewer"],
        "structure_accession": evidence["structure_accession"],
        "structure_version": evidence["structure_version"],
        "label_chain": evidence["label_chain"],
        "symmetry_operator": evidence["symmetry_operator"],
        "context_complete": evidence["context_complete"],
    }


def crosswalk(rows: list[dict], evidence: dict[str, list[dict]], structure_root: Path | None) -> list[dict]:
    result = []
    for source in rows:
        candidates = evidence.get(source["source_uid"], [])
        if not source["effective_dropna"]:
            status, reason, match = "incomplete", "excluded_by_released_dropna", {}
        elif not candidates:
            status, reason, match = "incomplete", "source_to_ion_evidence_unavailable", {}
        elif len(candidates) > 1:
            status, reason, match = "ambiguous", "multiple_documented_candidates", {}
        elif structure_root is None:
            status, reason, match = "incomplete", "structure_root_unavailable", {}
        else:
            status, reason, match = verify_match(source, candidates[0], structure_root)
        result.append({**source, "mapping_status": status, "mapping_reason": reason,
                       "physical_site_group": "", **match})
    return result


def assign_physical_groups(rows: list[dict]) -> None:
    """Record proximity without merging target rows or changing side membership."""
    by_pdb: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["mapping_status"] == "exact":
            by_pdb[row["pdbid"]].append(row)
    for pdbid, group in by_pdb.items():
        parent = list(range(len(group)))

        def root(index: int) -> int:
            while parent[index] != index:
                index = parent[index]
            return index

        for i, first in enumerate(group):
            for j in range(i + 1, len(group)):
                second = group[j]
                if math.dist([first[axis] for axis in ("x", "y", "z")],
                             [second[axis] for axis in ("x", "y", "z")]) <= 4.5:
                    parent[root(j)] = root(i)
        for i, row in enumerate(group):
            row["physical_site_group"] = f"{pdbid}:{min(group[j]['source_uid'] for j in range(len(group)) if root(j) == root(i))}"


def pocket_graph(path: Path, anchor: dict, radius: float = 10.0) -> dict:
    """Minimal label-blind, target-centered protein CA graph for an approved row."""
    vertices = []
    center = (float(anchor["x"]), float(anchor["y"]), float(anchor["z"]))
    selected_model = int(anchor.get("model", 1))
    current_model = 1
    with path.open(encoding="ascii", errors="replace") as handle:
        for line in handle:
            if line.startswith("MODEL "):
                current_model = int(line[10:14].strip())
            if current_model != selected_model:
                continue
            if not line.startswith("ATOM  ") or line[12:16].strip() != "CA":
                continue
            if line[16].strip() not in {"", "A"}:
                continue
            point = tuple(float(line[start:start + 8]) for start in (30, 38, 46))
            vector = [round(point[i] - center[i], 4) for i in range(3)]
            if math.dist(point, center) <= radius:
                vertices.append({"chain": line[21].strip(), "resseq": line[22:26].strip(),
                                 "icode": line[26].strip(), "resname": line[17:20].strip(),
                                 "relative_ca": vector})
    edges = []
    for i, a in enumerate(vertices):
        for j in range(i + 1, len(vertices)):
            if math.dist(a["relative_ca"], vertices[j]["relative_ca"]) <= 8.0:
                edges.append([i, j])
    return {"nodes": vertices, "edges": edges, "anchor_policy": "verified_target_ion_coordinate_only",
            "input_protocol": "protein_CA_within_10A; relative_coordinates; 8A_CA_edges; no_ion_element_or_label"}


def build(args: argparse.Namespace) -> dict:
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite isolated output: {args.output_root}")
    source_rows = []
    source_stats = {}
    for side, path in (("train", args.source_train), ("test", args.source_test)):
        rows, stats = load_source(path, side)
        source_rows.extend(rows)
        source_stats[side] = stats
    if len({row["source_uid"] for row in source_rows}) != len(source_rows):
        raise ValueError("Duplicate source UIDs across sides")
    evidence = load_evidence(args.evidence_csv, source_rows)
    rows = crosswalk(source_rows, evidence, args.structure_root)
    assign_physical_groups(rows)
    train_pdbs = {row["pdbid"] for row in rows if row["original_side"] == "train"}
    test_pdbs = {row["pdbid"] for row in rows if row["original_side"] == "test"}
    train_source_ids = {(row["pdbid"], row["residueid_ion"], row["metalid"]) for row in rows if row["original_side"] == "train"}
    test_source_ids = {(row["pdbid"], row["residueid_ion"], row["metalid"]) for row in rows if row["original_side"] == "test"}
    side_class_coverage = {}
    for side in ("train", "test"):
        side_class_coverage[side] = {}
        for code in LABELS:
            subset = [row for row in rows if row["original_side"] == side and row["label_metal"] == code]
            side_class_coverage[side][code] = dict(Counter(row["mapping_status"] for row in subset))
            side_class_coverage[side][code]["raw"] = len(subset)
            side_class_coverage[side][code]["effective_dropna"] = sum(row["effective_dropna"] for row in subset)
    exact = [row for row in rows if row["mapping_status"] == "exact"]
    if args.emit_graphs:
        for row in exact:
            graph_name = hashlib.sha256(row["source_uid"].encode()).hexdigest() + ".json"
            row["graph_path"] = f"{row['original_side']}/pockets/{graph_name}"
    dataset_status = "site_mapping_complete" if len(exact) == sum(row["effective_dropna"] for row in rows) else "matched_subset" if exact else "audit_only_no_verified_sites"
    train_exact = [row for row in exact if row["original_side"] == "train"]
    test_exact = [row for row in exact if row["original_side"] == "test"]
    exact_uid_overlap = {row["source_uid"] for row in train_exact} & {row["source_uid"] for row in test_exact}
    if exact_uid_overlap:
        raise ValueError("Source UIDs crossed train/test sides")
    train_sites = {row["physical_site_group"] for row in train_exact}
    test_sites = {row["physical_site_group"] for row in test_exact}
    train_coords = {(row["pdbid"], row["model"], row["x"], row["y"], row["z"]) for row in train_exact}
    test_coords = {(row["pdbid"], row["model"], row["x"], row["y"], row["z"]) for row in test_exact}
    output = args.output_root
    output.mkdir(parents=True)
    write_csv(output / "site_crosswalk.csv", rows, CROSSWALK_COLUMNS)
    for status, filename in (("ambiguous", "ambiguous_sites.csv"), ("unmatched", "unmatched_sites.csv"), ("incomplete", "incomplete_sites.csv")):
        write_csv(output / filename, [row for row in rows if row["mapping_status"] == status], CROSSWALK_COLUMNS)
    for side in ("train", "test"):
        folder = output / side
        folder.mkdir()
        side_rows = [row for row in exact if row["original_side"] == side]
        write_csv(folder / "site_manifest.csv", side_rows, CROSSWALK_COLUMNS)
        structures = sorted({(row["coordinate_path"], row["coordinate_sha256"]) for row in side_rows})
        write_csv(folder / "structure_manifest.csv", [{"coordinate_path": p, "sha256": h} for p, h in structures], ["coordinate_path", "sha256"])
        if args.emit_graphs and side_rows:
            graph_dir = folder / "pockets"
            graph_dir.mkdir()
            for row in side_rows:
                graph = pocket_graph(args.structure_root / row["coordinate_path"], row)
                if not graph["nodes"]:
                    raise ValueError(f"No protein CA context for verified site: {row['source_uid']}")
                graph["source_uid"] = row["source_uid"]
                # Supervision stays in site_manifest.csv, never in the graph.
                write_json(output / row["graph_path"], graph)
    metadata = {
        "profile_version": PROFILE_VERSION,
        "builder_sha256": sha256(Path(__file__)),
        "status": dataset_status,
        "artifact_level": "prototype_pocket_graphs" if args.emit_graphs and exact else "site_manifests_only" if exact else "source_audit_only",
        "training_loader_certified": False,
        "pmm_github_commit": PMM_COMMIT,
        "pmm_training_script_sha256": TRAINING_SCRIPT_HASH,
        "published_effective_model_cohort_verified": False,
        "published_effective_model_cohort_note": "Post-dropna rows are known; released script requests absent source column and paper selection/oversampling are not reconstructed",
        "numeric_label_mapping": LABELS,
        "numeric_mapping_evidence": f"https://github.com/hhz-lab/PinMyMetal/blob/{PMM_COMMIT}/metal_prediction/script/hybrid_algorithm/script10_clasmodel_result.sql",
        "source": source_stats,
        "figshare": {"doi": "10.6084/m9.figshare.25011212.v1", "archive_md5": "dcb73c3520dcb0c1df07b5cd9126f147", "archive_inspected": False},
        "fold_identity": "PMM published row-fold IDs unavailable in inspected GitHub release; no fold assigned",
        "membership_exact": True,
        "membership_exact_scope": "source rows and sides only; no structural cohort implied",
        "site_mapping_coverage": {"status": dataset_status, "exact": len(exact),
                                  "effective_dropna": sum(row["effective_dropna"] for row in rows)},
        "input_protocol": "verified target anchor; protein CA within 10A; CA edges within 8A; selected model; no observed ion element or source label in graph",
        "comparison_scope": "source audit; no PMM score reproduction",
        "test_access_history": "Existing PMM-related tests were historically accessed; see docs/DATASETS.md",
        "overlap": {"source_shared_pdbids": len(train_pdbs & test_pdbs), "shared_pdbids": sorted(train_pdbs & test_pdbs),
                    "source_shared_identifier_triplets": len(train_source_ids & test_source_ids),
                    "matched_shared_physical_sites": len(train_sites & test_sites),
                    "matched_shared_coordinates": len(train_coords & test_coords),
                    "matched_shared_pdbids": len({r["pdbid"] for r in train_exact} & {r["pdbid"] for r in test_exact})},
    }
    coverage = {"status": dataset_status, "raw_rows": len(rows), "effective_dropna_rows": sum(row["effective_dropna"] for row in rows),
                "exact_rows": len(exact), "mapping_status": dict(Counter(row["mapping_status"] for row in rows)),
                "by_side_and_label": side_class_coverage}
    write_json(output / "split_metadata.json", metadata)
    write_json(output / "coverage.json", coverage)
    return coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-train", type=Path, default=DEFAULT_SOURCES / "classmodel_train_set")
    parser.add_argument("--source-test", type=Path, default=DEFAULT_SOURCES / "classmodel_test_set")
    parser.add_argument("--evidence-csv", type=Path, help="Reviewed PMM identifier-to-ion crosswalk")
    parser.add_argument("--structure-root", type=Path, help="Read-only root for coordinate_path in evidence")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--emit-graphs", action="store_true", help="Write minimal label-blind pocket graphs for verified rows")
    args = parser.parse_args()
    if args.emit_graphs and (args.evidence_csv is None or args.structure_root is None):
        parser.error("--emit-graphs requires --evidence-csv and --structure-root")
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
