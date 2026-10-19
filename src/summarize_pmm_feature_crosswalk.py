"""Summarize a complete serial PMM feature crosswalk into audit manifests."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


CLASSES = ("MN", "CLASS_VIII", "CU", "ZN")
SIDES = ("train", "test")


def load_records(root: Path):
    rule = json.loads((root / "rule.json").read_text())
    chunk_count = (rule["pdb_count"] + rule["chunk_size"] - 1) // rule["chunk_size"]
    chunks = [root / "chunks" / f"chunk_{index:04d}.jsonl.gz" for index in range(chunk_count)]
    missing = [path.name for path in chunks if not path.is_file()]
    if missing:
        raise ValueError(f"Incomplete scan: {len(missing)} missing chunks; first: {missing[:5]}")
    records = []
    for path in chunks:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                row = json.loads(line)
                row["candidate_record_ref"] = f"chunks/{path.name}:{number}"
                records.append(row)
    if len(records) != rule["source_rows"]:
        raise ValueError(f"Expected {rule['source_rows']} rows; got {len(records)}")
    uids = {row["source_uid"] for row in records}
    if len(uids) != len(records):
        raise ValueError("Duplicate source UID in chunk output")
    return rule, records


def group_counts(records, key):
    counts = defaultdict(Counter)
    for row in records:
        counts[key(row)][row["mapping_status"]] += 1
    return counts


def status_summary(counts):
    return {
        "source_rows": sum(counts.values()),
        "unique": counts["unique"],
        "ambiguous": counts["ambiguous"],
        "unavailable": counts["unavailable"],
        "unique_fraction": counts["unique"] / sum(counts.values()) if sum(counts.values()) else None,
    }


def candidate_key(row):
    ion = row["passing_ions"][0]
    return (row["pdbid"], row["structure_sha256"], ion["model"],
            ion["chain"], ion["resseq"], ion["icode"], ion["altloc"],
            ion["element"], ion["serial"])


def write_csv(path: Path, fieldnames, records):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def summarize(root: Path):
    rule, records = load_records(root)
    by_side_class = group_counts(records, lambda row: (row["side"], row["class"]))
    by_site_type = group_counts(records, lambda row: row["site_type"] or "UNKNOWN")
    by_ched_count = group_counts(records, lambda row: row["ched_count"] or "UNKNOWN")
    by_reason = Counter((row["mapping_status"], row["reason"]) for row in records)
    overall = Counter(row["mapping_status"] for row in records)

    ion_rows = defaultdict(list)
    for row in records:
        if row["mapping_status"] == "unique":
            ion_rows[candidate_key(row)].append(row)
    collisions = []
    for key, mapped in ion_rows.items():
        if len(mapped) > 1:
            collisions.append({"ion_key": key, "source_uids": [r["source_uid"] for r in mapped],
                               "sides": sorted({r["side"] for r in mapped})})
    cross_side_collisions = sum(len(item["sides"]) > 1 for item in collisions)

    coverage = {
        "classification": "complete-source-row feature audit; not training examples",
        "rule": rule,
        "overall": status_summary(overall),
        "by_side_class": {
            side: {cls: status_summary(by_side_class[(side, cls)]) for cls in CLASSES}
            for side in SIDES
        },
        "by_site_type": {name: status_summary(counts)
                         for name, counts in sorted(by_site_type.items())},
        "by_ched_count": {name: status_summary(counts)
                          for name, counts in sorted(by_ched_count.items())},
        "by_status_reason": {f"{status}:{reason}": count
                             for (status, reason), count in sorted(by_reason.items())},
        "unique_ion_collision_count": len(collisions),
        "cross_side_ion_collision_count": cross_side_collisions,
        "ion_collisions": collisions,
    }
    (root / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")

    fields = ["source_uid", "original_side", "source_row_number", "pdbid",
              "residueid_ion", "metalid", "label_metal", "metal_class", "ched_count",
              "deposited_site_type", "mapping_status", "reason", "structural_candidate_count",
              "passing_candidate_count", "model", "chain", "resseq", "icode", "altloc",
              "element", "atom_name", "atom_serial", "x", "y", "z",
              "structure_url", "structure_sha256", "candidate_record_ref"]
    flat = []
    for row in records:
        item = {
            "source_uid": row["source_uid"], "original_side": row["side"],
            "source_row_number": row["row_number"], "pdbid": row["pdbid"],
            "residueid_ion": row["residueid_ion"], "metalid": row["metalid"],
            "label_metal": row["label_metal"], "metal_class": row["class"],
            "ched_count": row["ched_count"],
            "deposited_site_type": row["site_type"] or "",
            "mapping_status": row["mapping_status"], "reason": row["reason"],
            "structural_candidate_count": len(row["structural_candidates"]),
            "passing_candidate_count": len(row["passing_ions"]),
            "structure_url": row["structure_url"],
            "structure_sha256": row["structure_sha256"] or "",
            "candidate_record_ref": row["candidate_record_ref"],
        }
        if row["mapping_status"] == "unique":
            ion = row["passing_ions"][0]
            item.update({"model": ion["model"], "chain": ion["chain"],
                         "resseq": ion["resseq"], "icode": ion["icode"],
                         "altloc": ion["altloc"], "element": ion["element"],
                         "atom_name": ion["name"], "atom_serial": ion["serial"],
                         "x": ion["xyz"][0], "y": ion["xyz"][1], "z": ion["xyz"][2]})
        flat.append(item)
    write_csv(root / "row_crosswalk.csv", fields, flat)
    write_csv(root / "exclusion_ledger.csv", fields,
              (item for item in flat if item["mapping_status"] != "unique"))

    strata = defaultdict(list)
    for row in records:
        if row["mapping_status"] == "unique":
            strata[(row["side"], row["class"], bool(row["site_type"]))].append(row)
    sample = []
    for stratum, candidates in sorted(strata.items()):
        ranked = sorted(candidates, key=lambda row: hashlib.sha256(
            ("PMM-independent-review-v1:" + row["source_uid"]).encode()).hexdigest())
        for row in ranked[:2]:
            sample.append({"source_uid": row["source_uid"], "side": row["side"],
                           "metal_class": row["class"], "deposited_type_present": stratum[2],
                           "pdbid": row["pdbid"], "structure_sha256": row["structure_sha256"],
                           "candidate_record_ref": row["candidate_record_ref"]})
    write_csv(root / "independent_review_sample.csv",
              ["source_uid", "side", "metal_class", "deposited_type_present", "pdbid",
               "structure_sha256", "candidate_record_ref"], sample)

    # Predeclared gate; sample review is a separate mandatory gate and is not assumed passed here.
    overall_pass = coverage["overall"]["unique_fraction"] >= 0.95
    class_pass = all(coverage["by_side_class"][side][cls]["unique_fraction"] >= 0.90
                     for side in SIDES for cls in CLASSES)
    common_types = {name: data for name, data in coverage["by_site_type"].items()
                    if data["source_rows"] >= 100}
    type_pass = all(data["unique_fraction"] >= 0.85 for data in common_types.values())
    gate = {
        "overall_at_least_95_percent": overall_pass,
        "each_side_class_at_least_90_percent": class_pass,
        "each_site_type_with_at_least_100_rows_at_least_85_percent": type_pass,
        "zero_cross_side_ion_collisions": cross_side_collisions == 0,
        "independent_stratified_review": "pending",
        "ready_to_materialize": False,
        "note": "Independent review must pass before a mapped dataset can be materialized.",
    }
    (root / "materialization_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps({"overall": coverage["overall"], "gate": gate,
                      "sample_rows": len(sample)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    summarize(parser.parse_args().output)


if __name__ == "__main__":
    main()
