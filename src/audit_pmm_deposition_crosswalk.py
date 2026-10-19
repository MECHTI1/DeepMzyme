"""Review PMM source identities against deposited metal and coordinate records.

This is an audit of source records, not a training dataset. A coordinate is
accepted only when the deposited element and coordinate uniquely identify the
ion in a complete PDB entry. No distance or row-order inference is used.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

if __package__:
    from .build_pmm_site_dataset import ELEMENTS, LABELS, PMM_COMMIT, load_source, parse_ions, sha256, write_csv, write_json
else:
    from build_pmm_site_dataset import ELEMENTS, LABELS, PMM_COMMIT, load_source, parse_ions, sha256, write_csv, write_json


VERSION = "pmm-deposition-crosswalk-v1"
FIGSHARE = {
    "doi": "10.6084/m9.figshare.25011212.v1",
    "file_id": 52224257,
    "archive_size": 99424422,
    "archive_md5": "dcb73c3520dcb0c1df07b5cd9126f147",
    "archive_sha256": "e8efd7aa463f217c0ada67a538c3b4eb20f83927936d73cc9be2a1822696dfb8",
}
PINNED_DEPOSITION_HASHES = {
    "figshare_transition_metal_type.csv": "67a75598d248e099e2f7b245592aa821971801d67955b65b1953501211c6ccf2",
    "figshare_exp_pre_sites_CH.txt": "48cf05bee5254de85bd7cc67033a448593ec6ecfcdb44d466514ea6a83f9743c",
    "figshare_exp_pre_sites_EDH.txt": "29aae32d959ae5bd0c016d2bfd32189e878a4eb0a328ed3abd330ce4f88cb041",
}
ATOMIC_NUMBERS = {"25": "MN", "26": "FE", "27": "CO", "28": "NI", "29": "CU", "30": "ZN"}
COLUMNS = [
    "source_uid", "original_side", "source_row_number", "pdbid", "residueid_ion",
    "metalid", "label_metal", "four_class_target", "effective_dropna",
    "deposited_element", "type_record_count", "coordinate_candidate_count",
    "coordinate_candidates_json", "structural_ion_count", "mapping_status", "mapping_reason", "model",
    "chain", "resseq", "icode", "altloc", "element", "serial", "x", "y", "z",
    "structure_path", "structure_sha256", "evidence_references_json",
]


def read_type_records(path: Path) -> dict[tuple[str, str], list[dict]]:
    records = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"pdbid", "residueid_ion", "which_metal", "resname_ion"}.issubset(reader.fieldnames or []):
            raise ValueError("Deposited metal-type table lacks required columns")
        for number, row in enumerate(reader, 2):
            element = ATOMIC_NUMBERS.get(row["which_metal"].strip())
            if element is None:
                continue
            resname = row["resname_ion"].strip().upper().lstrip("_")
            if not resname.startswith(element):
                raise ValueError(f"Metal name and atomic number disagree at line {number}")
            records[(row["pdbid"].strip().lower(), row["residueid_ion"].strip())].append(
                {"element": element, "line": number, "fingerprint": row.get("resi_type", "").strip()}
            )
    return records


def read_coordinate_records(paths: list[Path]) -> dict[tuple[str, str], list[dict]]:
    records = defaultdict(dict)
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not {"pdbid", "chainid", "resseq", "resname_ion", "exp_metal_coord"}.issubset(reader.fieldnames or []):
                raise ValueError(f"Deposited coordinate table lacks required columns: {path}")
            for number, row in enumerate(reader, 2):
                coordinate = row["exp_metal_coord"].strip()
                if not coordinate:
                    continue
                element = row["resname_ion"].strip().upper().removeprefix("_").rstrip("0123456789")
                if element not in ATOMIC_NUMBERS.values():
                    continue
                pdbid = row["pdbid"].strip().lower()
                chain = row["chainid"].strip()
                resseq = row["resseq"].strip()
                axes = tuple(float(value) for value in coordinate.split(","))
                if len(axes) != 3:
                    raise ValueError(f"Invalid deposited coordinate: {path}:{number}")
                key = (chain, resseq, axes)
                item = records[(pdbid, element)].setdefault(key, {
                    "chain": chain, "resseq": resseq, "coordinate": axes, "references": []
                })
                item["references"].append(f"{path.name}:{number}")
    return {key: list(value.values()) for key, value in records.items()}


def structure_info(path: Path) -> tuple[str, list[dict]]:
    return sha256(path), parse_ions(path)


def review_row(source: dict, type_records: dict, coordinates: dict, structures: Path) -> dict:
    row = {key: source[key] for key in (
        "source_uid", "original_side", "source_row_number", "pdbid", "residueid_ion",
        "metalid", "label_metal", "four_class_target", "effective_dropna"
    )}
    types = type_records.get((source["pdbid"], source["residueid_ion"]), [])
    row["type_record_count"] = len(types)
    elements = {item["element"] for item in types}
    row["coordinate_candidate_count"] = 0
    row["coordinate_candidates_json"] = "[]"
    if not source["effective_dropna"]:
        row.update(mapping_status="incomplete", mapping_reason="excluded_by_released_dropna")
        return row
    if not types:
        row.update(mapping_status="incomplete", mapping_reason="deposited_metal_type_absent")
        return row
    if len(elements) != 1:
        row.update(mapping_status="ambiguous", mapping_reason="conflicting_deposited_metal_types")
        return row
    element = next(iter(elements))
    row["deposited_element"] = element
    if element not in ELEMENTS[source["four_class_target"]]:
        row.update(mapping_status="unmatched", mapping_reason="deposited_element_conflicts_with_source_class")
        return row
    path = structures / f"{source['pdbid']}_rcsb_current.pdb"
    ions = []
    if path.is_file():
        file_sha, all_ions = structure_info(path)
        ions = [ion for ion in all_ions if ion["element"] == element]
        row.update(structure_path=path.name, structure_sha256=file_sha,
                   structural_ion_count=len(ions))
    candidates = coordinates.get((source["pdbid"], element), [])
    row["coordinate_candidate_count"] = len(candidates)
    row["coordinate_candidates_json"] = json.dumps(candidates, sort_keys=True)
    if not candidates:
        row.update(mapping_status="incomplete", mapping_reason="deposited_coordinate_absent")
        return row
    if len(candidates) > 1:
        row.update(mapping_status="ambiguous", mapping_reason="multiple_deposited_coordinates_without_identifier_link")
        return row
    candidate = candidates[0]
    if not path.is_file():
        row.update(mapping_status="incomplete", mapping_reason="full_structure_not_reviewed")
        return row
    if len(ions) != 1:
        row.update(mapping_status="ambiguous", mapping_reason="multiple_structural_ions_without_identifier_link")
        return row
    ion = ions[0]
    if (ion["chain"], ion["resseq"]) != (candidate["chain"], candidate["resseq"]) or any(
        abs(ion[axis] - candidate["coordinate"][index]) > 0.001
        for index, axis in enumerate(("x", "y", "z"))
    ):
        row.update(mapping_status="unmatched", mapping_reason="deposited_coordinate_not_in_reviewed_structure")
        return row
    row.update({key: ion[key] for key in (
        "model", "chain", "resseq", "icode", "altloc", "element", "serial", "x", "y", "z"
    )})
    row["evidence_references_json"] = json.dumps({
        "type_lines": [item["line"] for item in types],
        "coordinate_lines": candidate["references"],
        "structure_url": f"https://files.rcsb.org/download/{source['pdbid'].upper()}.pdb",
        "method": "unique_deposited_element_and_coordinate; unique_element_in_complete_structure",
    }, sort_keys=True)
    row.update(mapping_status="exact", mapping_reason="unique_element_and_deposited_coordinate_verified")
    return row


def audit(args: argparse.Namespace) -> dict:
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_root}")
    source = []
    source_stats = {}
    for side, path in (("train", args.source_train), ("test", args.source_test)):
        rows, source_stats[side] = load_source(path, side)
        source.extend(rows)
    coordinate_paths = [args.ch_coordinates, args.edh_coordinates]
    input_hashes = {path.name: sha256(path) for path in [args.type_table, *coordinate_paths]}
    if input_hashes != PINNED_DEPOSITION_HASHES:
        raise ValueError(f"Figshare v1 extracted-file checksum mismatch: {input_hashes}")
    types = read_type_records(args.type_table)
    coords = read_coordinate_records(coordinate_paths)
    reviewed = [review_row(row, types, coords, args.structures) for row in source]
    coverage = {}
    for side in ("train", "test"):
        coverage[side] = {}
        for code, target in LABELS.items():
            group = [row for row in reviewed if row["original_side"] == side and row["label_metal"] == code]
            coverage[side][target] = {
                "raw": len(group),
                "effective_dropna": sum(row["effective_dropna"] for row in group),
                **{status: sum(row["mapping_status"] == status for row in group)
                   for status in ("exact", "ambiguous", "unmatched", "incomplete")},
            }
    train_pdbs = {row["pdbid"] for row in reviewed if row["original_side"] == "train"}
    test_pdbs = {row["pdbid"] for row in reviewed if row["original_side"] == "test"}
    result = {
        "version": VERSION,
        "classification": "source-row audit records; not training examples",
        "pmm_source_commit": PMM_COMMIT,
        "figshare": FIGSHARE,
        "source": source_stats,
        "input_sha256": input_hashes,
        "coverage_by_side_and_class": coverage,
        "status_totals": dict(Counter(row["mapping_status"] for row in reviewed)),
        "deposited_type_link_rows": sum(bool(row.get("deposited_element")) for row in reviewed),
        "deposited_coordinate_candidate_rows": sum(row["coordinate_candidate_count"] > 0 for row in reviewed),
        "source_pdb_overlap": len(train_pdbs & test_pdbs),
        "exact_source_uid_overlap": len({row["source_uid"] for row in reviewed if row["original_side"] == "train" and row["mapping_status"] == "exact"} &
                                        {row["source_uid"] for row in reviewed if row["original_side"] == "test" and row["mapping_status"] == "exact"}),
        "reviewed_full_structures": sorted({row["pdbid"] for row in reviewed if row.get("structure_sha256")}),
    }
    args.output_root.mkdir(parents=True)
    write_csv(args.output_root / "deposition_crosswalk.csv", reviewed, COLUMNS)
    for status in ("exact", "ambiguous", "unmatched", "incomplete"):
        write_csv(args.output_root / f"{status}_records.csv",
                  [row for row in reviewed if row["mapping_status"] == status], COLUMNS)
    write_json(args.output_root / "coverage.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).resolve().parents[1] / "prepare_training_and_test_set" / "pinmymetal_files"
    parser.add_argument("--source-train", type=Path, default=default / "classmodel_train_set")
    parser.add_argument("--source-test", type=Path, default=default / "classmodel_test_set")
    parser.add_argument("--type-table", type=Path, required=True)
    parser.add_argument("--ch-coordinates", type=Path, required=True)
    parser.add_argument("--edh-coordinates", type=Path, required=True)
    parser.add_argument("--structures", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
