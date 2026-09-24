"""Serial, resumable PMM source-row to structural-ion feature audit.

This writes audit records only. It never changes existing datasets or trains a model.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


SOURCE_HASHES = {
    "train": "4748babd2b6ac0706cd9ed4bcfd4855c8d2c5535f01813fb2e10b68b31a24c0f",
    "test": "ec6427ad0b8a18261dbc1d6822bdcfe7194730c7da4a36c208957bcb00f6a0f2",
}
TYPE_HASH = "67a75598d248e099e2f7b245592aa821971801d67955b65b1953501211c6ccf2"
METALS = {"MN", "FE", "CO", "NI", "CU", "ZN"}
ELEMENTS = {"1": {"MN"}, "2": {"FE", "CO", "NI"}, "6": {"CU"}, "7": {"ZN"}}
CLASSES = {"1": "MN", "2": "CLASS_VIII", "6": "CU", "7": "ZN"}
COMPRESSED_LIMIT = 8_000_000
PDB_LIMIT = 32_000_000
CHUNK_SIZE = 50


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_csv(path: Path, **kwargs):
    with path.open(newline="", **kwargs) as handle:
        yield from csv.DictReader(handle)


def load_sources(upstream: Path):
    grouped = defaultdict(list)
    counts = Counter()
    for side in ("train", "test"):
        path = upstream / f"classmodel_{side}_set"
        actual = digest(path.read_bytes())
        if actual != SOURCE_HASHES[side]:
            raise ValueError(f"Unexpected source hash for {path}: {actual}")
        for number, raw in enumerate(read_csv(path), 1):
            row = {
                "source_uid": f"sha256:{actual}:row:{number}",
                "side": side, "row_number": number,
                "pdbid": raw["pdbid"].lower(),
                "residueid_ion": raw["residueid_ion"],
                "metalid": raw["metalid"],
                "label_metal": raw["label_metal"],
                "class": CLASSES[raw["label_metal"]],
                "ched_count": raw["ched_count"],
                "coordnum_inner": int(float(raw["coordnum_inner"])),
                "distance_min": float(raw["distance_min"]),
                "distance_avg": float(raw["distance_avg"]),
                "distance_max": float(raw["distance_max"]),
                "histogram": [int(float(raw[f"bin{b}_class{c}"]))
                              for b in range(1, 4) for c in range(8)],
            }
            grouped[row["pdbid"]].append(row)
            counts[(side, row["class"])] += 1
    if sum(counts.values()) != 9408:
        raise ValueError(f"Expected 9,408 pinned source rows, found {sum(counts.values())}")
    return grouped, counts


def load_deposition(upstream: Path):
    type_path = upstream / "figshare_transition_metal_type.csv"
    if digest(type_path.read_bytes()) != TYPE_HASH:
        raise ValueError("Unexpected deposited metal-type file hash")
    types = {}
    for row in read_csv(type_path, encoding="utf-8-sig"):
        types[(row["pdbid"].lower(), row["residueid_ion"])] = {
            "site_type": row["resi_type"],
            "element": row["resname_ion"].strip("_").upper(),
        }
    coordinates = defaultdict(list)
    for suffix in ("CH", "EDH"):
        path = upstream / f"figshare_exp_pre_sites_{suffix}.txt"
        with path.open(newline="") as handle:
            for number, row in enumerate(csv.DictReader(handle, delimiter="\t"), 2):
                try:
                    xyz = tuple(float(v) for v in row["exp_metal_coord"].split(","))
                    if len(xyz) != 3:
                        continue
                except ValueError:
                    continue
                coordinates[(row["pdbid"].lower(), row["resname_ion"].strip().upper())].append({
                    "chain": row["chainid"].strip(), "resseq": row["resseq"].strip(),
                    "xyz": xyz, "reference": f"{path.name}:{number}",
                })
    return types, coordinates


def get_pdb(pdbid: str):
    url = f"https://files.rcsb.org/download/{pdbid.upper()}.pdb.gz"
    request = urllib.request.Request(url, headers={"User-Agent": "DeepMzyme-PMM-source-audit/1.0"})
    last_error = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                packed = response.read(COMPRESSED_LIMIT + 1)
            if len(packed) > COMPRESSED_LIMIT:
                return None, "compressed_size_limit", url
            with gzip.GzipFile(fileobj=io.BytesIO(packed)) as stream:
                raw = stream.read(PDB_LIMIT + 1)
            if len(raw) > PDB_LIMIT:
                return None, "pdb_size_limit", url
            return raw, None, url
        except (urllib.error.URLError, TimeoutError, OSError, EOFError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (404, 410):
                break
            if attempt == 0:
                time.sleep(0.5)
    return None, f"retrieval_failed:{last_error}", url


def parse_atoms(raw: bytes):
    atoms = []
    model = 1
    for line in raw.decode("ascii", errors="replace").splitlines():
        if line.startswith("MODEL "):
            try:
                model = int(line[10:14])
            except ValueError:
                model = 1
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        try:
            atoms.append({
                "model": model, "record": line[:6].strip(),
                "serial": int(line[6:11]), "name": line[12:16].strip(),
                "altloc": line[16:17].strip(), "resname": line[17:20].strip(),
                "chain": line[21:22].strip(), "resseq": line[22:26].strip(),
                "icode": line[26:27].strip(),
                "xyz": tuple(float(line[i:i+8]) for i in (30, 38, 46)),
                "element": line[76:78].strip().upper(),
            })
        except ValueError:
            continue
    return atoms


def chem_classes(atom):
    """Direct category tests from pinned PMM chemfeatures.py."""
    res, name, element = atom["resname"], atom["name"], atom["element"]
    classes = []
    if res in {"HIS", "TRP", "TYR", "PHE"} and name in {
        "CG", "ND1", "CD2", "CE1", "NE2", "CD1", "NE1", "CE2", "CE3",
        "CZ2", "CZ3", "CH2", "CZ", "OH"}:
        classes.append(1)
    if element == "C":
        classes.append(2)
    if name == "N" or (res in {"ASN", "GLN", "TRP", "MSE", "SER", "THR", "MET", "CYS"}
                       and name in {"ND2", "NE2", "NE1", "SG", "SE", "OG", "OG1"}):
        classes.append(3)
    if name == "O" or (res in {"ASP", "GLU", "HIS", "SER", "THR", "MSE", "CYS", "MET"}
                       and name in {"ND2", "NE2", "OE1", "OE2", "OD1", "OD2",
                                    "OG", "OG1", "SE", "SG"}):
        classes.append(4)
    if res in {"LYS", "ARG", "HIS"} and name in {"NZ", "NH1", "NH2", "ND1", "NE2", "NE"}:
        classes.append(5)
    if res in {"ASP", "GLU"} and name in {"OD1", "OD2", "OE1", "OE2"}:
        classes.append(6)
    if name in {"ND1", "NE2", "SG", "OE1", "OE2", "OD2"}:
        classes.append(7)
    return classes or [0]


def signature(ion, atoms):
    histogram = [0] * 24
    donors = []
    for atom in atoms:
        if atom["model"] != ion["model"] or atom["serial"] == ion["serial"]:
            continue
        if atom["element"] in METALS or atom["altloc"] not in ("", "A"):
            continue
        distance = math.dist(atom["xyz"], ion["xyz"])
        if atom["element"] in {"O", "N", "S", "SE"} and distance <= 2.8:
            donors.append((distance, atom))
        if atom["resname"] == "HOH":
            continue
        if 2 <= distance < 5:
            for chemical_class in chem_classes(atom):
                histogram[int(distance - 2) * 8 + chemical_class] += 1
    donors.sort(key=lambda pair: pair[0])
    composition = Counter(
        resname for _, _, _, resname in {
            (atom["chain"], atom["resseq"], atom["icode"], atom["resname"])
            for _, atom in donors if atom["resname"] in {"CYS", "HIS", "ASP", "GLU"}
        }
    )
    return histogram, [distance for distance, _ in donors], dict(composition)


def type_consistent(site_type, composition):
    if not site_type:
        return None
    checks = re.findall(r"(ED|C|H)(\d+)", site_type)
    if not checks:
        return None
    counts = {"C": composition.get("CYS", 0), "H": composition.get("HIS", 0),
              "ED": composition.get("GLU", 0) + composition.get("ASP", 0)}
    return all(counts[group] == int(number) for group, number in checks)


def ion_identity(ion):
    return {key: ion[key] for key in ("model", "chain", "resseq", "icode", "altloc",
                                      "element", "name", "serial", "xyz")}


def review_pdb(pdbid, source_rows, types, coordinates):
    raw, error, url = get_pdb(pdbid)
    if error:
        return [{**row, "site_type": types.get((pdbid, row["residueid_ion"]), {}).get("site_type"),
                 "mapping_status": "unavailable", "reason": error,
                 "structure_url": url, "structure_sha256": None,
                 "structural_candidates": [], "passing_ions": []} for row in source_rows]
    pdb_sha256 = digest(raw)
    atoms = parse_atoms(raw)
    ions = [atom for atom in atoms if atom["record"] == "HETATM" and atom["element"] in METALS]
    cached = {id(ion): signature(ion, atoms) for ion in ions}
    results = []
    for row in source_rows:
        kind = types.get((pdbid, row["residueid_ion"]))
        allowed = ({kind["element"]} if kind and kind["element"] in METALS
                   else ELEMENTS[row["label_metal"]])
        if kind and not allowed.issubset(ELEMENTS[row["label_metal"]]):
            allowed = set()
        candidate_records = []
        passing = []
        for ion in ions:
            if ion["element"] not in allowed:
                continue
            observed_hist, donors, composition = cached[id(ion)]
            hist_l1 = sum(abs(a - b) for a, b in zip(row["histogram"], observed_hist))
            near = donors[:row["coordnum_inner"]]
            if len(near) == row["coordnum_inner"] and near:
                distance_delta = {
                    "min": abs(min(near) - row["distance_min"]),
                    "avg": abs(sum(near) / len(near) - row["distance_avg"]),
                    "max": abs(max(near) - row["distance_max"]),
                }
            else:
                distance_delta = None
            type_ok = type_consistent(kind["site_type"] if kind else None, composition)
            coord_refs = [point["reference"] for point in coordinates.get((pdbid, ion["element"]), [])
                          if point["chain"] == ion["chain"] and point["resseq"] == ion["resseq"]
                          and math.dist(point["xyz"], ion["xyz"]) <= 0.005]
            passes = (hist_l1 == 0 and distance_delta is not None
                      and all(value <= 0.001 for value in distance_delta.values())
                      and type_ok is not False)
            record = {"ion": ion_identity(ion), "histogram_l1": hist_l1,
                      "first_shell_distance_deltas": distance_delta,
                      "donors_within_2p8": len(donors), "ligand_counts": composition,
                      "deposited_site_type_consistent": type_ok,
                      "deposited_coordinate_refs": coord_refs, "passes": passes}
            candidate_records.append(record)
            if passes:
                passing.append(record["ion"])
        if len(passing) == 1:
            status, reason = "unique", "one_candidate_passes_fixed_rule"
        elif len(passing) > 1:
            status, reason = "ambiguous", "multiple_candidates_pass_fixed_rule"
        elif not allowed:
            status, reason = "unavailable", "deposited_element_conflicts_with_class"
        elif not candidate_records:
            status, reason = "unavailable", "no_eligible_structural_ion"
        else:
            status, reason = "unavailable", "no_candidate_passes_fixed_rule"
        results.append({**row, "site_type": kind["site_type"] if kind else None,
                        "mapping_status": status, "reason": reason,
                        "structure_url": url, "structure_sha256": pdb_sha256,
                        "structural_candidates": candidate_records,
                        "passing_ions": passing})
    return results


def run(args):
    upstream, output = args.upstream, args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "chunks").mkdir(exist_ok=True)
    grouped, _ = load_sources(upstream)
    types, coordinates = load_deposition(upstream)
    ids = sorted(grouped)
    config = {
        "version": "pmm_feature_crosswalk_v1", "source_sha256": SOURCE_HASHES,
        "type_sha256": TYPE_HASH, "structure_source": "RCSB current PDB gzip",
        "histogram": "exact 24-bin PMM chemical-class vector; all chains, altloc blank/A; exclude waters",
        "first_shell": "coordnum_inner nearest O/N/S/Se donors <=2.8 A including waters; min/avg/max <=0.001 A",
        "site_type": "deposited CH/ED/H ligand counts must agree when parseable",
        "coordinate": "deposited PDB chain/residue/xyz <=0.005 A corroborates but absence does not reject",
        "unique": "exactly one eligible structural ion passes all applicable checks",
        "compressed_byte_limit": COMPRESSED_LIMIT, "pdb_byte_limit": PDB_LIMIT,
        "chunk_size": CHUNK_SIZE, "pdb_count": len(ids), "source_rows": 9408,
    }
    config_path = output / "rule.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Existing output uses a different matching rule")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    for index in range((len(ids) + CHUNK_SIZE - 1) // CHUNK_SIZE):
        chunk = output / "chunks" / f"chunk_{index:04d}.jsonl.gz"
        if chunk.exists():
            continue
        selected = ids[index * CHUNK_SIZE:(index + 1) * CHUNK_SIZE]
        temp = chunk.with_suffix(".tmp")
        with gzip.open(temp, "wt", encoding="utf-8") as handle:
            for pdbid in selected:
                for record in review_pdb(pdbid, grouped[pdbid], types, coordinates):
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                time.sleep(args.delay)
        os.replace(temp, chunk)
        print(f"completed chunk {index + 1}/{(len(ids) + CHUNK_SIZE - 1) // CHUNK_SIZE}", flush=True)
        if args.max_new_chunks and index + 1 >= args.max_new_chunks:
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--max-new-chunks", type=int, default=0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
