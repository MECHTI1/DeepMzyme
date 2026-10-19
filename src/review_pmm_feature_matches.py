"""Independently reparse a stratified PMM match sample with Biopython."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import re
import urllib.request
from pathlib import Path

from Bio.PDB import PDBParser


METALS = {"MN", "FE", "CO", "NI", "CU", "ZN"}


def csv_rows(path):
    with path.open(newline="") as handle:
        yield from csv.DictReader(handle)


def source_rows(upstream, sample):
    wanted = {"train": set(), "test": set()}
    for row in sample:
        wanted[row["original_side"]].add(int(row["source_row_number"]))
    result = {}
    for side in ("train", "test"):
        for number, row in enumerate(csv_rows(upstream / f"classmodel_{side}_set"), 1):
            if number in wanted[side]:
                result[(side, number)] = row
    return result


def classes(atom):
    residue = atom.get_parent().get_resname().strip()
    name = atom.get_name().strip()
    element = atom.element.upper()
    found = []
    if residue in {"HIS", "TRP", "TYR", "PHE"} and name in {
        "CG", "ND1", "CD2", "CE1", "NE2", "CD1", "NE1", "CE2", "CE3",
        "CZ2", "CZ3", "CH2", "CZ", "OH"}:
        found.append(1)
    if element == "C":
        found.append(2)
    if name == "N" or (residue in {"ASN", "GLN", "TRP", "MSE", "SER", "THR", "MET", "CYS"}
                       and name in {"ND2", "NE2", "NE1", "SG", "SE", "OG", "OG1"}):
        found.append(3)
    if name == "O" or (residue in {"ASP", "GLU", "HIS", "SER", "THR", "MSE", "CYS", "MET"}
                       and name in {"ND2", "NE2", "OE1", "OE2", "OD1", "OD2",
                                    "OG", "OG1", "SE", "SG"}):
        found.append(4)
    if residue in {"LYS", "ARG", "HIS"} and name in {"NZ", "NH1", "NH2", "ND1", "NE2", "NE"}:
        found.append(5)
    if residue in {"ASP", "GLU"} and name in {"OD1", "OD2", "OE1", "OE2"}:
        found.append(6)
    if name in {"ND1", "NE2", "SG", "OE1", "OE2", "OD2"}:
        found.append(7)
    return found or [0]


def load_structure(pdbid, expected_sha):
    request = urllib.request.Request(
        f"https://files.rcsb.org/download/{pdbid.upper()}.pdb.gz",
        headers={"User-Agent": "DeepMzyme-PMM-independent-review/1.0"})
    with urllib.request.urlopen(request, timeout=20) as response:
        packed = response.read(8_000_001)
    if len(packed) > 8_000_000:
        raise ValueError("compressed structure exceeds audit limit")
    with gzip.GzipFile(fileobj=io.BytesIO(packed)) as stream:
        raw = stream.read(32_000_001)
    if len(raw) > 32_000_000:
        raise ValueError("structure exceeds audit limit")
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(f"PDB bytes changed: {actual_sha} != {expected_sha}")
    return PDBParser(QUIET=True).get_structure(pdbid, io.StringIO(raw.decode("ascii", "replace")))


def review(row, source, structure):
    expected_hist = [int(float(source[f"bin{b}_class{c}"]))
                     for b in range(1, 4) for c in range(8)]
    model_num, chain, resseq = int(row["model"]), row["chain"], int(row["resseq"])
    serial, element = int(row["atom_serial"]), row["element"]
    model = next((m for m in structure if (getattr(m, "serial_num", None) or m.id + 1)
                  == model_num), None)
    if model is None:
        return False, "model_missing", None
    atom_list = list(model.get_atoms())
    ion = next((a for a in atom_list
                if a.get_serial_number() == serial and a.element.upper() == element
                and a.get_parent().get_parent().id == chain
                and a.get_parent().id[1] == resseq), None)
    if ion is None:
        return False, "ion_identity_missing", None
    if math.dist(tuple(map(float, ion.get_coord())),
                 (float(row["x"]), float(row["y"]), float(row["z"]))) > 0.005:
        return False, "ion_coordinate_changed", None
    histogram = [0] * 24
    donors = []
    ligand_residues = set()
    for atom in atom_list:
        if atom.get_serial_number() == serial or atom.element.upper() in METALS:
            continue
        if atom.get_altloc().strip() not in ("", "A"):
            continue
        d = math.dist(tuple(map(float, atom.get_coord())), tuple(map(float, ion.get_coord())))
        residue = atom.get_parent()
        if atom.element.upper() in {"O", "N", "S", "SE"} and d <= 2.8:
            donors.append(d)
            if residue.get_resname().strip() in {"CYS", "HIS", "ASP", "GLU"}:
                ligand_residues.add((residue.get_parent().id, residue.id,
                                     residue.get_resname().strip()))
        if residue.get_resname().strip() != "HOH" and 2 <= d < 5:
            for chemical_class in classes(atom):
                histogram[int(d-2)*8 + chemical_class] += 1
    donors.sort()
    n = int(float(source["coordnum_inner"]))
    if len(donors) < n:
        return False, "too_few_donors", None
    near = donors[:n]
    deltas = {
        "min": abs(min(near) - float(source["distance_min"])),
        "avg": abs(sum(near)/n - float(source["distance_avg"])),
        "max": abs(max(near) - float(source["distance_max"])),
    }
    hist_l1 = sum(abs(a-b) for a, b in zip(histogram, expected_hist))
    composition = {name: sum(resname == name for _, _, resname in ligand_residues)
                   for name in ("CYS", "HIS", "ASP", "GLU")}
    site_type = row["deposited_site_type"]
    tests = re.findall(r"(ED|C|H)(\d+)", site_type)
    type_ok = all({"C": composition["CYS"], "H": composition["HIS"],
                   "ED": composition["ASP"] + composition["GLU"]}[group] == int(count)
                  for group, count in tests)
    passed = hist_l1 == 0 and max(deltas.values()) <= 0.001 and type_ok
    return passed, "passed" if passed else "feature_disagreement", {
        "histogram_l1": hist_l1, "distance_deltas": deltas,
        "site_type_consistent": type_ok, "ligand_counts": composition,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    crosswalk = {row["source_uid"]: row for row in csv_rows(args.output / "row_crosswalk.csv")}
    sample = [crosswalk[row["source_uid"]] for row in csv_rows(
        args.output / "independent_review_sample.csv")]
    sample.sort(key=lambda row: (row["pdbid"], row["source_uid"]))
    sources = source_rows(args.upstream, sample)
    current_key = None
    current_structure = None
    reviewed = []
    for row in sample:
        key = (row["pdbid"], row["structure_sha256"])
        try:
            if key != current_key:
                current_structure = load_structure(*key)
                current_key = key
            passed, reason, details = review(
                row, sources[(row["original_side"], int(row["source_row_number"]))],
                current_structure)
        except Exception as exc:
            passed, reason, details = False, f"review_error:{type(exc).__name__}:{exc}", None
        reviewed.append({"source_uid": row["source_uid"], "side": row["original_side"],
                         "metal_class": row["metal_class"], "site_type": row["deposited_site_type"],
                         "pdbid": row["pdbid"], "passed": passed, "reason": reason,
                         "details": details})
        print(f"reviewed {len(reviewed)}/{len(sample)}", flush=True)
    artifact = {"reviewed": len(reviewed), "passed": sum(item["passed"] for item in reviewed),
                "rows": reviewed}
    (args.output / "independent_review.json").write_text(json.dumps(artifact, indent=2) + "\n")
    gate_path = args.output / "materialization_gate.json"
    gate = json.loads(gate_path.read_text())
    gate["independent_stratified_review"] = {
        "reviewed": len(reviewed), "passed": artifact["passed"],
        "all_passed": bool(reviewed) and artifact["passed"] == len(reviewed),
    }
    gate["ready_to_materialize"] = (
        gate["overall_at_least_95_percent"]
        and gate["each_side_class_at_least_90_percent"]
        and gate["each_site_type_with_at_least_100_rows_at_least_85_percent"]
        and gate["zero_cross_side_ion_collisions"]
        and gate["independent_stratified_review"]["all_passed"]
    )
    gate["note"] = ("All predeclared checks passed" if gate["ready_to_materialize"]
                    else "Mapped dataset not materialized: one or more checks failed")
    gate_path.write_text(json.dumps(gate, indent=2) + "\n")


if __name__ == "__main__":
    main()
