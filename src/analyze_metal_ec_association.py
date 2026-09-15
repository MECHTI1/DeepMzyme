"""Frozen, development-only metal/EC1 association; no training or test inputs.

Preparation reconstructs only metal clusters for already retained training
pockets. Execution consumes the frozen labels, never structures or test labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROFILE = "metal_ec1_development_association_v1"
METALS = {"native6": ["Mn", "Cu", "Zn", "Fe", "Co", "Ni"],
          "common4": ["Mn", "Cu", "Zn", "VIII"],
          "common4_native6_eligible": ["Mn", "Cu", "Zn", "VIII"]}
PANELS = {
    "pinmymetal": {
        "dataset": "train_and_test_sets_structures_non_overlapped_pinmymetal",
        "split": "docs/notebook_outputs/raw/metal_architecture_pilot_20260915/readiness/expected_split.json",
        "provenance": "PinMyMetal-derived labels on experimentally resolved structures; catalytic-site selection",
    },
    "care": {
        "dataset": "CARE_task1_30_clusterRes30_train_test_metallo",
        "split": "docs/notebook_outputs/raw/ec1_standalone_v12_20260914/expected_split.json",
        "provenance": "CARE/UniProt annotations with computational AlphaFill/MAHOMES metal-site transfer and selection",
    },
}
SOURCE_FILES = ["src/analyze_metal_ec_association.py", "src/graph/structure_parsing.py",
                "src/data_structures.py", "src/featurization.py", "src/label_schemes.py",
                "src/structure_store.py", "src/training/labels.py", "src/training/site_filter.py"]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def certified_training_examples(summary, dataset):
    """Certify the saved internal split; do not open external test metadata."""
    if not Path(summary["structure_dir"]).as_posix().endswith(f"/{dataset}/train"):
        raise ValueError("Expected split must originate in the declared external train directory")
    if summary.get("test_structure_dir") or summary.get("test_summary_csv"):
        raise ValueError("Expected split includes a test input")
    if summary.get("ec_label_depth") != 1:
        raise ValueError("Only the retained EC1 task is supported")
    identity = summary["retained_split_identity"]
    train = identity["train"]["examples"]
    # Read only group identifiers from the saved internal-validation partition.
    validation_groups = {row["group"] for row in identity["validation"]["examples"]}
    if {row["group"] for row in train} & validation_groups:
        raise ValueError("Training/internal-validation group overlap")
    keys = [(row["structure_id"], row["pocket_id"]) for row in train]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate retained pocket identity")
    return train


def checked_site_labels(path):
    from training.site_filter import _iter_normalized_summary_rows
    rows, columns = _iter_normalized_summary_rows(path)
    labels, duplicates = {}, 0
    for key, row in rows:
        label = row[columns["metal residue type"]].strip().upper()
        if key in labels:
            duplicates += 1
            if labels[key] != label:
                raise ValueError(f"Conflicting summary metal annotations for {key}")
        labels[key] = label
    return labels, duplicates


def retained_label_rows(root, data, name):
    from data_structures import DEFAULT_MULTINUCLEAR_MERGE_DISTANCE
    from graph.structure_parsing import (parse_structure_file, metal_records_from_biopython_residue,
                                         cluster_metal_records)
    from label_schemes import map_site_metal_symbols_for_scheme
    from structure_store import read_structure_manifest
    from training.labels import ec_label_token_from_structure_id, parse_structure_identity
    from training.site_filter import matched_site_keys_for_pocket

    spec = PANELS[name]
    summary = json.loads((root / spec["split"]).read_text())
    examples = certified_training_examples(summary, spec["dataset"])
    train_dir = data / spec["dataset"] / "train"
    sites, duplicate_count = checked_site_labels(
        train_dir / "final_data_summarazing_table_transition_metals_only_catalytic.csv")
    references = {Path(ref.structure_name).stem: ref for ref in read_structure_manifest(train_dir)}
    by_structure = defaultdict(list)
    for example in examples:
        by_structure[example["structure_id"]].append(example)
    rows, structure_inventory = [], []
    for structure_id, retained in sorted(by_structure.items()):
        ref = references[structure_id]
        structure_inventory.append({"structure_id": structure_id, "sha256": ref.sha256})
        structure = parse_structure_file(str(ref.path), structure_id=structure_id)
        records = [record for model in structure for chain in model for residue in chain
                   for record in metal_records_from_biopython_residue(residue, chain.id)]
        clusters = cluster_metal_records(records, DEFAULT_MULTINUCLEAR_MERGE_DISTANCE)
        group = parse_structure_identity(structure_id)[0]
        ec1 = ec_label_token_from_structure_id(structure_id, depth=1)
        if ec1 is not None and ec1 not in list("1234567"):
            raise ValueError(f"Invalid EC1 token for {structure_id}")
        for example in retained:
            if example["group"] != group:
                raise ValueError("Retained protein identity differs from parsed identity")
            saved_ec = (None if example["y_ec"] is None else
                        str(summary["ec_labels"][str(example["y_ec"]) ]))
            if saved_ec != ec1:
                raise ValueError(f"Retained EC1 target mismatch: {structure_id}")
            prefix = f"{structure_id}_METAL_"
            pocket_id = example["pocket_id"]
            if not pocket_id.startswith(prefix):
                raise ValueError("Unexpected canonical pocket identity")
            cluster = clusters[int(pocket_id[len(prefix):])]
            pocket = SimpleNamespace(metadata={"metal_site_ids": [x.site_id for x in cluster]})
            matched = matched_site_keys_for_pocket(pocket, ref.path, sites)
            if not matched:
                raise ValueError(f"Retained pocket is not in training catalytic-site summary: {pocket_id}")
            symbols = sorted({sites[key] for key in matched})
            labels = {view: map_site_metal_symbols_for_scheme(symbols, scheme_name=scheme)
                      for view, scheme in (("native6", "six_class"), ("common4", "four_class"))}
            source_metal = map_site_metal_symbols_for_scheme(
                symbols, scheme_name=summary["metal_label_scheme"])
            if source_metal != example["y_metal"]:
                raise ValueError(f"Saved/reconstructed source-scheme metal target mismatch: {pocket_id}")
            rows.append({"panel": name, "partition": "train", "group": group,
                         "structure_id": structure_id, "pocket_id": pocket_id,
                         "ec1": ec1 or "", "native6": labels["native6"],
                         "common4": labels["common4"], "metal_symbols": "+".join(symbols),
                         "matched_site_keys": json.dumps(sorted(matched), separators=(",", ":"))})
    return rows, {"retained_training_pockets": len(rows),
                  "retained_training_groups": len({r["group"] for r in rows}),
                  "internal_validation_groups_excluded": summary["retained_split_identity"]["validation"]["n_groups"],
                  "identical_duplicate_summary_keys": duplicate_count,
                  "saved_metal_label_scheme": summary["metal_label_scheme"],
                  "all_saved_source_scheme_metal_and_ec1_labels_match": True,
                  "structure_inventory": structure_inventory}


def prepare(root, data, output, permutations=9999, seed=42):
    if permutations < 0:
        raise ValueError("Permutation count must be nonnegative")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Preparation requires a fresh output directory")
    sources = [root / path for path in SOURCE_FILES]
    for spec in PANELS.values():
        sources += [root / spec["split"], data / spec["dataset"] / "train" / "structure_manifest.csv",
                    data / spec["dataset"] / "train" / "final_data_summarazing_table_transition_metals_only_catalytic.csv"]
    manifest = {"profile": PROFILE, "phase": "predeclared", "created_at": datetime.now(timezone.utc).isoformat(),
                "sources": [{"path": str(p.resolve()), "sha256": digest(p)} for p in sources],
                "panels": PANELS, "partitions": ["train"], "views": list(METALS),
                "weighting": ["site", "protein_group"], "permutations": permutations, "seed": seed,
                "permutation_statistic": "group-weighted mutual information; EC1 labels shuffled between whole groups",
                "definitions": {"cramers_v": "sqrt(Pearson chi2 / (total weight * min(active rows-1, active cols-1)))",
                                "mi": "sum p(m,e) * ln[p(m,e)/(p(m)*p(e))], natural logarithms",
                                "nmi": "2*MI/(H(metal)+H(EC1)); null if either marginal is constant",
                                "group_weight": "one total weight per group divided among eligible pockets in this view",
                                "chi_square": "descriptive only; expected-cell audit does not establish independent pockets",
                                "multiplicity": "Holm correction across the four primary panel x native6/common4 permutation tests; matched sensitivity descriptive only"},
                "limitations": ["Separate source panels; no PDB-to-UniProt or sequence-homology union certificate",
                                "No shared-encoder split certification or inference of auxiliary-learning benefit",
                                "No test input opened by this analysis; existing train membership is inherited, not re-certified against external test identifiers",
                                "Source selection and homology can violate exchangeability even between protein groups"]}
    save(output / "analysis_manifest.json", manifest)  # Rules and sources precede label extraction/statistics.
    rows = []
    audits = {}
    for name in PANELS:
        panel_rows, audits[name] = retained_label_rows(root, data, name)
        rows.extend(panel_rows)
    fields = ["panel", "partition", "group", "structure_id", "pocket_id", "ec1", "native6", "common4", "metal_symbols", "matched_site_keys"]
    write_csv(output / "retained_training_pairs.csv", rows, fields)
    save(output / "eligibility_audit.json", audits)
    manifest.update(phase="prepared", pairs_sha256=digest(output / "retained_training_pairs.csv"),
                    eligibility_audit_sha256=digest(output / "eligibility_audit.json"))
    # Abort rather than bind labels extracted across a source mutation.
    verify_sources(manifest)
    save(output / "analysis_manifest.json", manifest)
    (output / "analysis_manifest.sha256").write_text(digest(output / "analysis_manifest.json") + "\n")
    return manifest


def verify_sources(manifest):
    for source in manifest["sources"]:
        if digest(source["path"]) != source["sha256"]:
            raise ValueError(f"Source changed: {source['path']}")


def association_statistics(table):
    table = np.asarray(table, dtype=float)
    if table.ndim != 2 or not np.isfinite(table).all() or (table < 0).any():
        raise ValueError("Contingency table must be finite and nonnegative")
    total = float(table.sum())
    active = table[table.sum(1) > 0][:, table.sum(0) > 0]
    if total == 0 or min(active.shape) < 2:
        return {"total_weight": total, "mi_nats": None, "nmi": None, "cramers_v": None,
                "chi_square": None, "reason": "empty or constant marginal"}
    expected = np.outer(active.sum(1), active.sum(0)) / total
    probability = active / total
    positive = active > 0
    mi = max(0.0, float(np.sum(probability[positive] * np.log(active[positive] / expected[positive]))))
    entropies = [-float(np.sum(p * np.log(p))) for p in (probability.sum(0), probability.sum(1))]
    chi2 = float(np.sum((active - expected) ** 2 / expected))
    return {"total_weight": total, "mi_nats": mi, "nmi": 2 * mi / sum(entropies),
            "cramers_v": math.sqrt(chi2 / (total * min(active.shape[0] - 1, active.shape[1] - 1))),
            "chi_square": chi2, "degrees_of_freedom": (active.shape[0]-1)*(active.shape[1]-1),
            "expected_min": float(expected.min()), "expected_fraction_below_5": float((expected < 5).mean()),
            "expected_any_below_1": bool((expected < 1).any()),
            "conventional_expected_cell_heuristic_passes": bool(expected.min() >= 1 and (expected < 5).mean() <= .2),
            "asymptotic_chi_square_p_value": None,
            "asymptotic_p_omission": "Repeated pockets/fractional weights are not independent multinomial counts"}


def build_tables(rows, view):
    retained_group_ec = defaultdict(set)
    for row in rows:
        if row["ec1"]:
            retained_group_ec[row["group"]].add(row["ec1"])
    if any(len(labels) > 1 for labels in retained_group_ec.values()):
        raise ValueError("Conflicting EC1 labels within retained protein group")
    field = "native6" if view == "native6" else "common4"
    eligible = [r for r in rows if r["ec1"] and r[field] not in (None, "")
                and (view != "common4_native6_eligible" or r["native6"] not in (None, ""))]
    groups = defaultdict(list)
    for row in eligible:
        groups[row["group"]].append(row)
    site = np.zeros((len(METALS[view]), 7))
    profiles, ec_labels = [], []
    for group in sorted(groups):
        group_rows = groups[group]
        ec = {int(r["ec1"])-1 for r in group_rows}
        if len(ec) != 1 or not ec <= set(range(7)):
            raise ValueError(f"Conflicting/invalid EC1 labels within protein group {group}")
        profile = np.bincount([int(r[field]) for r in group_rows], minlength=len(METALS[view])).astype(float)
        if len(profile) != len(METALS[view]):
            raise ValueError("Metal target outside declared vocabulary")
        label = ec.pop()
        site[:, label] += profile
        profiles.append(profile / len(group_rows))
        ec_labels.append(label)
    profiles = np.asarray(profiles).reshape(-1, len(METALS[view]))
    ec_labels = np.asarray(ec_labels, dtype=int)
    weighted = profiles.T @ np.eye(7)[ec_labels]
    audit = {"retained_pockets": len(rows), "eligible_pockets": len(eligible), "eligible_groups": len(groups),
             "excluded_missing_ec1": sum(not r["ec1"] for r in rows),
             "excluded_ambiguous_or_missing_view_metal": sum(r[field] in (None, "") for r in rows),
             "excluded_additional_native6_eligibility": sum(
                 bool(r["ec1"]) and r[field] not in (None, "") and r["native6"] in (None, "")
                 for r in rows) if view == "common4_native6_eligible" else 0,
             "excluded_total": len(rows)-len(eligible),
             "exclusion_counts_can_overlap": True}
    return site, weighted, profiles, ec_labels, audit


def permutation_mi(profiles, labels, count, seed):
    observed = association_statistics(profiles.T @ np.eye(7)[labels])["mi_nats"]
    if observed is None or count == 0:
        return {"permutations": count, "p_value": None, "reason": "degenerate table or permutations disabled"}
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(count):
        statistic = association_statistics(profiles.T @ np.eye(7)[rng.permutation(labels)])["mi_nats"]
        exceed += statistic >= observed - 1e-12
    return {"permutations": count, "seed": seed, "exceedances_including_ties": exceed,
            "p_value": (exceed + 1)/(count + 1), "unit": "whole protein group", "statistic": "group-weighted MI",
            "interpretation": "Exploratory conditional on exchangeability; sequence homology and selection remain uncontrolled"}


def holm_adjust(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result, running = [0.0]*len(values), 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values)-rank)*values[index])
        result[index] = min(1.0, running)
    return result


def execute(output):
    manifest_path = output / "analysis_manifest.json"
    if digest(manifest_path) != (output / "analysis_manifest.sha256").read_text().strip():
        raise ValueError("Frozen manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest["phase"] != "prepared" or manifest["profile"] != PROFILE:
        raise ValueError("Analysis has not been prepared")
    verify_sources(manifest)
    for filename, key in (("retained_training_pairs.csv", "pairs_sha256"), ("eligibility_audit.json", "eligibility_audit_sha256")):
        if digest(output / filename) != manifest[key]:
            raise ValueError(f"Frozen input hash mismatch: {filename}")
    with (output / "retained_training_pairs.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if any(r["partition"] != "train" or r["panel"] not in PANELS for r in rows):
        raise ValueError("Only separately declared training panels are permitted")
    results, cells, permutation_tests = [], [], []
    for panel in PANELS:
        panel_rows = [r for r in rows if r["panel"] == panel]
        for view in METALS:
            site, weighted, profiles, labels, eligibility = build_tables(panel_rows, view)
            result = {"panel": panel, "partition": "train", "view": view, "eligibility": eligibility,
                      "site": association_statistics(site), "protein_group": association_statistics(weighted)}
            if view != "common4_native6_eligible":
                result["permutation"] = permutation_mi(profiles, labels, manifest["permutations"], manifest["seed"])
                if result["permutation"]["p_value"] is not None:
                    permutation_tests.append(result["permutation"])
            results.append(result)
            for weighting, table in (("site", site), ("protein_group", weighted)):
                for i, metal in enumerate(METALS[view]):
                    for j in range(7):
                        cells.append({"panel": panel, "partition": "train", "view": view, "weighting": weighting,
                                      "metal": metal, "ec1": j+1, "count_or_weight": float(table[i,j]),
                                      "p_ec_given_metal": float(table[i,j]/table[i].sum()) if table[i].sum() else "",
                                      "p_metal_given_ec": float(table[i,j]/table[:,j].sum()) if table[:,j].sum() else ""})
    for test, adjusted in zip(permutation_tests, holm_adjust([t["p_value"] for t in permutation_tests])):
        test["holm_adjusted_p_value"] = adjusted
    save(output / "association_results.json", {"profile": PROFILE, "manifest_sha256": digest(manifest_path),
                                               "status": "completed_descriptive_development_analysis", "results": results,
                                               "limitations": manifest["limitations"]})
    write_csv(output / "contingencies_and_conditionals.csv", cells, list(cells[0]))
    save(output / "execution_receipt.json", {"completed_at": datetime.now(timezone.utc).isoformat(),
         "manifest_sha256": digest(manifest_path), "results_sha256": digest(output / "association_results.json"),
         "contingencies_sha256": digest(output / "contingencies_and_conditionals.csv"),
         "test_inputs": [], "model_training": False, "cross_source_union_certified": False})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "execute"))
    parser.add_argument("--data-root", type=Path, default=Path("DeepMzyme_Data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(Path(__file__).resolve().parents[1], args.data_root.resolve(), args.output_dir.resolve(), args.permutations, args.seed)
    else:
        execute(args.output_dir.resolve())
    print(f"{args.phase} complete: {args.output_dir}")


if __name__ == "__main__":
    main()
