"""Read-only EC1 component-support and EC7 data-attrition report (campaign ec1_independent_support_2026-10).

CPU evidence only. The reporter reads the saved EC1 development membership, the frozen sequence-remoteness
evidence and the recorded CARE clusterRes30 training-side build tables, and writes counts. It has no training,
inference, download, similarity-search, feature-generation or split-writing mode.

Relationships use the frozen remoteness component rule (``audit_sequence_remoteness.annotate_task``): same protein
group, identical sequence, or a qualifying alignment (E-value <= 0.001, >= 50 residue pairs, shorter-sequence
coverage >= 0.5, identity > 0.2). These components were built as conservative uncertainty units, not as a mandatory
split rule. AlphaFill candidates are re-derived only for the allowlisted EC7 training accessions with the recorded
build policy and the pipeline's own functions (``care_prepare.extract_candidate_dicts``); no structure is parsed and
nothing is fetched.

Test-side records are never used: the CARE fetch summary is filtered to ``split == "train"`` and the allowlist as it
is read, and every other training-side table must contain only training rows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "CARE_prepare_training_and_test_set")]

import audit_sequence_remoteness as asr  # noqa: E402
import care_prepare as cp  # noqa: E402

CAMPAIGN_ID = "ec1_independent_support_2026-10"
FOCAL_CLASS = "7"
REMOTENESS_FILES = {
    "protocol": "protocol.json", "preparation_freeze": "preparation_freeze.json",
    "sequence_manifest": "sequence_manifest.json", "memberships": "memberships.json",
    "ec_fasta": "ec/development.fasta", "ec_alignments": "ec/alignments.tsv",
    "ec_search_receipt": "ec/search_receipt.json",
    "full_protein_readiness": "full_protein_readiness/preparation_report.json",
}
CARE_METADATA_FILES = {
    "train_proteins": "care_task1_30_train_proteins.csv", "fetch_summary": "fetch_summary.csv",
    "build_summary": "train/build_summary.csv", "candidate_sites": "train/candidate_site_summary.csv",
    "prediction_results": "train/prediction_results_summary.csv",
    "final_sites": "train/data_summarazing_table_transition_metals_whether_catalytic.csv",
    "build_parameters": "build_parameters.json",
}
ANNOTATION_CACHE = Path("uniprot") / cp.UNIPROT_ANNOTATION_CACHE_NAME
MISSING_ASSETS = "MISSING_ALPHAFILL_JSON_OR_CIF"
DELIBERATE_CIF_SKIP = "SKIPPED_NO_JSON_SUPPORTED_CANDIDATE"
# Descriptive lookup only (never a filter): PDB chemical-component IDs of metal-containing cofactors that the
# exact-element mapping (care_prepare.canonicalize_metal) does not interpret as their constituent metal atoms.
METAL_CONTAINING_COMPOUNDS = {
    "FES": "FE", "F3S": "FE", "SF4": "FE", "FS4": "FE", "HEM": "FE", "HEC": "FE", "HEB": "FE", "HEA": "FE",
    "HAS": "FE", "HDD": "FE", "HEO": "FE", "B12": "CO", "CNC": "CO", "COB": "CO", "CUA": "CU",
}
REJECTION_CATEGORIES = (
    "no_qualifying_donor_or_transplant", "compound_ids_outside_exact_element_mapping",
    "elemental_candidates_rejected_by_annotation", "recorded_missing_json", "unexpected_change_requiring_review",
)
# Values stated in the handoff (section 4.4/4.5); compared, never enforced.
HANDOFF_EXPECTATIONS = {
    "proteins": 793, "pockets": 1168, "components": 319,
    "class_proteins": [175, 111, 382, 64, 26, 29, 6], "class_components": [83, 51, 131, 32, 22, 14, 1],
    "largest_component_sizes": [53, 25, 16, 15, 12], "focal_qualifying_unordered_pairs": 11,
    "focal_components": 1, "focal_connected_without_other_classes": True, "focal_sequence_discrepancies": 0,
    "focal_unambiguous_training_proteins": 179, "focal_in_baseline": 6, "focal_outside_baseline": 173,
    "build_outcomes": {"NO_UNIPROT_SUPPORTED_TRANSITION_METAL": 141, MISSING_ASSETS: 24, "OK": 14},
    "prepared_structures": 14, "prepared_candidate_sites": 17, "final_catalytic_sites": 8,
    "final_noncatalytic_sites": 9, "allowlist": 24, "allowlist_sequences": 24, "allowlist_json": 22,
    "allowlist_cif": 0, "allowlist_complete_pairs": 0, "deliberate_cif_skips": 22, "json_http_404": 2,
    "missing_json_accessions": ["P08748", "Q00227"], "annotation_compatible_candidates": 0,
}


class ReportError(ValueError):
    """A refused input or output."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportError(message)


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": asr.sha256(path), "bytes": path.stat().st_size}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ec_classes(value: str) -> set[str]:
    return {ec.split(".")[0] for ec in cp.normalize_ec_number_list(value).split(";") if ec}


# ---------------------------------------------------------------------------
# Baseline membership and the frozen remoteness evidence
# ---------------------------------------------------------------------------

def load_baseline(path: Path) -> dict[str, Any]:
    summary = asr.read_json(path)
    require(summary.get("task") == "ec" and summary.get("ec_label_depth") == 1,
            "The baseline split is not an EC depth-1 development membership")
    asr._development_source(summary)  # training-side structure_dir/summary_csv only
    return asr.saved_membership(summary, "ec", "ec1_baseline")


def check_ec_memberships(memberships: list[dict[str, Any]], baseline: dict[str, Any]) -> dict[str, Any]:
    """Every saved EC run membership must be the baseline membership; repeats are counted once."""
    ec_runs = [m for m in memberships if m["task"] == "ec"]
    require(bool(ec_runs), "The remoteness memberships contain no EC run")
    unique = set()
    for membership in ec_runs:
        digest = asr.digest({s: membership[s] for s in ("train", "val")})
        require(digest == membership["membership_sha256"], f"EC membership {membership['run_id']} digest changed")
        require(digest == baseline["membership_sha256"],
                f"EC membership {membership['run_id']} differs from the baseline membership")
        unique.add(digest)
    return {"ec_run_memberships": len(ec_runs), "unique_ec_memberships": len(unique),
            "all_equal_to_baseline": True, "membership_sha256": baseline["membership_sha256"]}


def load_remoteness(root: Path, baseline: dict[str, Any]) -> dict[str, Any]:
    protocol = asr.protocol_at(root / REMOTENESS_FILES["protocol"])
    require(protocol["component_coverage"] == protocol["audit_shorter_coverage"],
            "The protocol's component coverage and the rule's coverage differ")
    freeze = asr.read_json(root / REMOTENESS_FILES["preparation_freeze"])
    manifest = asr.read_json(root / REMOTENESS_FILES["sequence_manifest"])
    for name in ("sequence_manifest", "memberships"):
        require(freeze[f"{name}_sha256"] == asr.sha256(root / REMOTENESS_FILES[name]),
                f"Frozen {name} changed after preparation")
    require(freeze["protocol_sha256"] == asr.sha256(root / REMOTENESS_FILES["protocol"])
            == manifest["protocol_sha256"], "Protocol changed after preparation")
    require(manifest["memberships_sha256"] == asr.sha256(root / REMOTENESS_FILES["memberships"]),
            "Membership manifest checksum mismatch")
    membership_check = check_ec_memberships(asr.read_json(root / REMOTENESS_FILES["memberships"])["memberships"],
                                            baseline)
    records = [r for r in manifest["sequences"] if r["task"] == "ec"]
    baseline_groups = {r["group_id"] for s in ("train", "val") for r in baseline[s]}
    baseline_structures = {r["structure_id"] for s in ("train", "val") for r in baseline[s]}
    require({r["group_id"] for r in records} == baseline_groups,
            "The sequence records do not cover exactly the baseline proteins")
    require({r["structure_id"] for r in records} <= baseline_structures, "A sequence record is outside the baseline")
    alignment_path, _ = asr.verify_search_receipt(root, "ec", manifest, protocol)
    sequences = {r["sequence_id"]: r for r in records if r["status"] == "resolved"}
    fasta = "".join(f">{sid}\n{sequences[sid]['sequence']}\n" for sid in
                    [r["sequence_id"] for r in records if r["status"] == "resolved"])
    require((root / REMOTENESS_FILES["ec_fasta"]).read_text() == fasta,
            "The development FASTA disagrees with the frozen sequence records")
    diagnostics = {"alignment_rows": 0, "nident_recount_differs_from_export": 0}
    alignments = list(asr.read_alignments(alignment_path, sequences, diagnostics))
    readiness = asr.read_json(root / REMOTENESS_FILES["full_protein_readiness"])
    discrepant = {row["group_id"] for row in readiness.get("care_discrepancies", [])}
    discrepant |= {row["group_id"] for row in readiness.get("blockers", [])
                   if row.get("task") == "ec" and row.get("status") == "unresolved_mapping"}
    return {"protocol": protocol, "records": records, "alignments": alignments, "diagnostics": diagnostics,
            "membership_check": membership_check, "discrepant_groups": discrepant}


def dependence_components(records: list[dict[str, Any]], alignments: list[dict[str, Any]],
                          protocol: dict[str, Any]) -> tuple[asr.Components, list[dict[str, Any]]]:
    """The frozen component rule of audit_sequence_remoteness.annotate_task, with its links between groups."""
    components = asr.Components(sorted({r["group_id"] for r in records}))
    by_id = {r["sequence_id"]: r for r in records if r.get("sequence_id")}
    links = []
    first_group_of_sequence: dict[str, str] = {}
    for record in sorted(records, key=lambda r: (r["group_id"], r.get("sequence_id") or "")):
        if record["status"] != "resolved":
            continue
        other = first_group_of_sequence.setdefault(record["sequence_sha256"], record["group_id"])
        if other != record["group_id"]:
            components.join(record["group_id"], other)
            links.append({"a": min(other, record["group_id"]), "b": max(other, record["group_id"]),
                          "link_type": "exact_sequence", "identity": 1.0, "shorter_coverage": 1.0,
                          "residue_pairs": None, "evalue": None, "qualifies": True})
    for hit in alignments:
        a, b = by_id[hit["query"]]["group_id"], by_id[hit["target"]]["group_id"]
        qualifies = (hit["evalue"] <= protocol["max_evalue"] and hit["residue_pairs"] >= protocol["min_residue_pairs"]
                     and hit["shorter_coverage"] >= protocol["audit_shorter_coverage"]
                     and hit["identity"] > protocol["component_identity_threshold"])
        if qualifies:
            components.join(a, b)
        if a != b:
            links.append({"a": min(a, b), "b": max(a, b), "link_type": "alignment", "identity": hit["identity"],
                          "shorter_coverage": hit["shorter_coverage"], "residue_pairs": hit["residue_pairs"],
                          "evalue": hit["evalue"], "qualifies": qualifies})
    return components, links


def component_id(components: asr.Components, group: str) -> str:
    return "c_" + asr.digest(["ec", components.root(group)])[:20]  # as annotate_task names components


def protein_table(baseline: dict[str, Any], records: list[dict[str, Any]], components: asr.Components,
                  vocabulary: list[str]) -> dict[str, dict[str, Any]]:
    proteins: dict[str, dict[str, Any]] = {}
    for side in ("train", "val"):
        for row in baseline[side]:
            item = proteins.setdefault(row["group_id"], {"targets": set(), "pockets": set(), "structures": set(),
                                                         "sides": set()})
            item["targets"].add(row["target"])
            item["pockets"].add(row["example_id"])
            item["structures"].add(row["structure_id"])
            item["sides"].add(side)
    by_group = defaultdict(list)
    for record in records:
        by_group[record["group_id"]].append(record)
    out = {}
    for group, item in proteins.items():
        require(len(item["targets"]) == 1, f"Contradictory EC labels within protein {group}: {sorted(item['targets'])}")
        target = next(iter(item["targets"]))
        require(0 <= target < len(vocabulary), f"Target outside the EC vocabulary for {group}")
        chains = sorted(by_group[group], key=lambda r: r.get("sequence_id") or "")
        out[group] = {"group": group, "accession": group.upper(), "ec_class": vocabulary[target],
                      "split_side": "+".join(sorted(item["sides"])), "pockets": len(item["pockets"]),
                      "structures": len(item["structures"]), "component_id": component_id(components, group),
                      "care_relationship": ";".join(sorted({str(r.get("care_relationship")) for r in chains})),
                      "sequence_sha256": ";".join(r.get("sequence_sha256", "") for r in chains)}
    return out


def support_summary(proteins: dict[str, dict[str, Any]], vocabulary: list[str]) -> dict[str, Any]:
    members = defaultdict(set)
    for group, item in proteins.items():
        members[item["component_id"]].add(group)
    for item in proteins.values():
        item["component_size"] = len(members[item["component_id"]])
        item["component_classes"] = ";".join(sorted({proteins[g]["ec_class"] for g in members[item["component_id"]]}))
    per_class = {}
    for label in vocabulary:
        groups = [g for g, item in proteins.items() if item["ec_class"] == label]
        per_class[label] = {"proteins": len(groups), "pockets": sum(proteins[g]["pockets"] for g in groups),
                            "components": len({proteins[g]["component_id"] for g in groups})}
    sizes = sorted((len(v) for v in members.values()), reverse=True)
    single = [label for label, counts in per_class.items() if counts["proteins"] and counts["components"] < 2]
    return {"proteins": len(proteins), "pockets": sum(item["pockets"] for item in proteins.values()),
            "components": len(members), "per_class": per_class, "largest_component_sizes": sizes[:5],
            "component_size_distribution": dict(sorted(Counter(sizes).items())),
            "classes_in_fewer_than_two_components": single,
            "component_grouping_prevents_all_class_train_val_coverage": bool(single)}


def focal_dependence(proteins: dict[str, dict[str, Any]], links: list[dict[str, Any]], focal: str,
                     discrepant_groups: set[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    focal_groups = sorted(g for g, item in proteins.items() if item["ec_class"] == focal)
    component_ids = sorted({proteins[g]["component_id"] for g in focal_groups})
    co_members = sorted(g for g, item in proteins.items() if item["component_id"] in component_ids)
    # Pairs involving a focal protein, one row per unordered pair (best qualifying evidence first).
    pairs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for link in links:
        if proteins[link["a"]]["ec_class"] == focal or proteins[link["b"]]["ec_class"] == focal:
            pairs[(link["a"], link["b"])].append(link)
    rows = []
    for (a, b), evidence in sorted(pairs.items()):
        qualifying = [e for e in evidence if e["qualifies"]]
        best = max(qualifying or evidence, key=lambda e: (e["identity"], e["shorter_coverage"]))
        rows.append({"protein_a": a.upper(), "class_a": proteins[a]["ec_class"], "protein_b": b.upper(),
                     "class_b": proteins[b]["ec_class"],
                     "link_types": ";".join(sorted({e["link_type"] for e in evidence})),
                     "evidence_rows": len(evidence), "qualifying_rows": len(qualifying),
                     "qualifies_component_rule": bool(qualifying), "best_identity": round(best["identity"], 6),
                     "best_shorter_coverage": round(best["shorter_coverage"], 6),
                     "best_residue_pairs": best["residue_pairs"], "best_evalue": best["evalue"],
                     "same_component": proteins[a]["component_id"] == proteins[b]["component_id"]})
    focal_pairs = [r for r in rows if r["class_a"] == focal == r["class_b"] and r["qualifies_component_rule"]]
    within = asr.Components(focal_groups)
    for row in focal_pairs:
        within.join(row["protein_a"].lower(), row["protein_b"].lower())
    others = [g for g in co_members if proteins[g]["ec_class"] != focal]
    result = {
        "focal_class": focal, "proteins": [g.upper() for g in focal_groups], "components": len(component_ids),
        "component_ids": component_ids,
        "component_members": [{"protein": g.upper(), "ec_class": proteins[g]["ec_class"]} for g in co_members],
        "other_class_members": [{"protein": g.upper(), "ec_class": proteins[g]["ec_class"]} for g in others],
        "qualifying_unordered_focal_pairs": len(focal_pairs),
        "connected_using_only_focal_links": len({within.root(g) for g in focal_groups}) == 1 if focal_groups else None,
        "sequence_discrepancies": sorted(g.upper() for g in focal_groups
                                         if g in discrepant_groups or proteins[g]["care_relationship"] != "exact"),
        "split_sides": dict(sorted(Counter(proteins[g]["split_side"] for g in focal_groups).items())),
    }
    return result, rows


# ---------------------------------------------------------------------------
# CARE clusterRes30 training-side attrition
# ---------------------------------------------------------------------------

def training_rows(path: Path, *, split_columns: tuple[str, ...]) -> list[dict[str, str]]:
    """A training-side table: every row must be a training row (a non-training row refuses the input)."""
    rows = read_csv_rows(path)
    for row in rows:
        for column in split_columns:
            require(row.get(column) == "train", f"{path.name}: non-training record ({column}={row.get(column)!r})")
    return rows


def unique_by_accession(rows: list[dict[str, str]], name: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        accession = row["uniprot_id"].strip()
        require(accession not in out or out[accession] == row, f"{name}: conflicting rows for {accession}")
        out[accession] = row
    return out


def site_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return cp.candidate_lookup_key(row)


def care_attrition(meta: Path, baseline_groups: set[str], focal: str) -> dict[str, Any]:
    proteins = unique_by_accession(
        training_rows(meta / CARE_METADATA_FILES["train_proteins"], split_columns=("split", "source_split")),
        "train proteins")
    focal_accessions = sorted(a for a, row in proteins.items() if ec_classes(row["ecnumber"]) == {focal})
    mixed = sorted(a for a, row in proteins.items() if focal in ec_classes(row["ecnumber"])
                   and ec_classes(row["ecnumber"]) != {focal})
    build = unique_by_accession(training_rows(meta / CARE_METADATA_FILES["build_summary"], split_columns=("split",)),
                                "build summary")
    candidates = training_rows(meta / CARE_METADATA_FILES["candidate_sites"], split_columns=("source_split",))
    predictions = training_rows(meta / CARE_METADATA_FILES["prediction_results"], split_columns=("source_split",))
    finals = training_rows(meta / CARE_METADATA_FILES["final_sites"], split_columns=("source_split",))
    focal_set = set(focal_accessions)
    final_by_key: dict[tuple, int] = {}
    for row in finals:
        if row["uniprot_id"] in focal_set:
            key = site_key(row)
            require(key not in final_by_key, f"Duplicate final site {key}")
            final_by_key[key] = int(row["whether_catalytic"])
    for row in predictions:
        if row["uniprot_id"] in focal_set:
            label = cp.parse_prediction_label(row["prediction"])
            key = site_key(row)
            require(label is not None and final_by_key.get(key) == label,
                    f"Saved prediction and final site table disagree for {key}")
    sites_by_accession = defaultdict(list)
    for row in candidates:
        if row["uniprot_id"] in focal_set:
            sites_by_accession[row["uniprot_id"]].append(row)
    inventory, outcomes = [], Counter()
    totals = Counter()
    for accession in focal_accessions:
        record = build.get(accession)
        status = record["status"] if record else "NO_BUILD_RECORD"
        outcomes[status] += 1
        sites = sites_by_accession.get(accession, [])
        final = [final_by_key.get(site_key(s)) for s in sites]
        catalytic, noncatalytic = final.count(1), final.count(0)
        missing = sum(v is None for v in final)
        in_baseline = accession.lower() in baseline_groups
        if status == "OK":
            totals["prepared_structures"] += 1
            totals["prepared_candidate_sites"] += len(sites)
            totals["recorded_selected_sites"] += int(record["selected_site_count"])
            totals["final_catalytic_sites"] += catalytic
            totals["final_noncatalytic_sites"] += noncatalytic
            totals["sites_without_final_prediction"] += missing
            totals["placeholder_catalytic_values_nonzero"] += sum(s["whether_catalytic"] != "0" for s in sites)
        if in_baseline:
            reason = "in_baseline"
        elif status == "OK" and catalytic == 0 and noncatalytic and not missing:
            reason = "prepared_noncatalytic_prediction_only"
        elif status == "OK":
            reason = "prepared_other_review"
        else:
            reason = status.lower()
        sequence = proteins[accession].get("sequence", "")
        inventory.append({
            "accession": accession, "ecnumbers": cp.normalize_ec_number_list(proteins[accession]["ecnumber"]),
            "sequence_length": len("".join(sequence.split())), "in_baseline": in_baseline, "build_status": status,
            "build_detail": record["detail"] if record else "",
            "uniprot_supported_transition_metals": record["uniprot_supported_transition_metals"] if record else "",
            "recorded_selected_sites": int(record["selected_site_count"]) if record else 0,
            "candidate_sites": len(sites), "final_catalytic_sites": catalytic,
            "final_noncatalytic_sites": noncatalytic, "sites_without_final_prediction": missing,
            "attrition_reason": reason})
    by_reason = Counter(row["attrition_reason"] for row in inventory)
    baseline_focal_ok = sum(1 for row in inventory if row["in_baseline"] and row["build_status"] == "OK")
    allowlist = sorted(row["accession"] for row in inventory
                       if not row["in_baseline"] and row["build_status"] == MISSING_ASSETS)
    return {"inventory": inventory, "allowlist": allowlist, "proteins": proteins, "build": build,
            "summary": {"training_proteins": len(proteins), "focal_unambiguous_training_proteins": len(focal_accessions),
                        "focal_ambiguous_multi_class_proteins": len(mixed),
                        "focal_in_baseline": sum(row["in_baseline"] for row in inventory),
                        "focal_outside_baseline": sum(not row["in_baseline"] for row in inventory),
                        "build_outcomes": dict(sorted(outcomes.items())), "attrition_reasons": dict(sorted(by_reason.items())),
                        "baseline_focal_with_ok_build": baseline_focal_ok, **dict(sorted(totals.items())),
                        "catalytic_assignment": ("predictive: saved MAHOMES predictions in the final site table; the "
                                                 "candidate table's whether_catalytic=0 is a placeholder, not a result")}}


def alphafill_policy(meta: Path) -> SimpleNamespace:
    """The recorded build policy (build_parameters.json) with the build parser's optional filters (all None)."""
    recorded = asr.read_json(meta / CARE_METADATA_FILES["build_parameters"])
    parsed = cp.build_parser().parse_args([
        "build-mahomes-inputs", "--min-alphafill-identity", str(recorded["min_alphafill_identity"]),
        "--min-alignment-length", str(recorded["min_alignment_length"]),
        "--site-dedup-distance", str(recorded["site_dedup_distance"]),
        "--uniprot-metal-policy", recorded["uniprot_metal_policy"], "--use-uniprot-annotation-cache"])
    require(sorted(recorded["supported_transition_metals"]) == sorted(cp.SUPPORTED_TRANSITION_METALS),
            "The recorded supported metals differ from the pipeline's")
    return SimpleNamespace(**{key: getattr(parsed, key) for key in (
        "min_alphafill_identity", "min_alignment_length", "max_local_rmsd", "max_binding_site_rmsd",
        "max_local_environment_rmsd", "max_pae_mean", "min_binding_site_atom_count", "uniprot_metal_policy")})


def allowlisted_rows(path: Path, allowlist: set[str]) -> dict[str, dict[str, str]]:
    """Rows with split == 'train' for allowlisted accessions only; every other row is discarded as it is read."""
    kept: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") != "train" or row.get("uniprot_id", "").strip() not in allowlist:
                continue
            accession = row["uniprot_id"].strip()
            require(accession not in kept or kept[accession] == row, f"{path.name}: conflicting rows for {accession}")
            kept[accession] = row
    return kept


def qualifying_hits(json_data: dict[str, Any], policy: SimpleNamespace) -> list[dict[str, Any]]:
    """Hits passing the recorded identity/length policy (the same comparisons as extract_candidate_dicts)."""
    hits = []
    for hit in json_data.get("hits", []) or []:
        alignment = hit.get("alignment", {}) or {}
        identity, length = cp.as_float(alignment.get("identity")), cp.as_int(alignment.get("length"))
        if identity is None or identity < policy.min_alphafill_identity:
            continue
        if policy.min_alignment_length is not None and (length is None or length < policy.min_alignment_length):
            continue
        hits.append(hit)
    return hits


def asset_rejections(meta: Path, work: Path, allowlist: list[str], proteins: dict[str, dict[str, str]],
                     build: dict[str, dict[str, str]], policy: SimpleNamespace) -> tuple[list[dict[str, Any]], dict]:
    wanted = set(allowlist)
    fetch = allowlisted_rows(meta / CARE_METADATA_FILES["fetch_summary"], wanted)
    cache_path = work / ANNOTATION_CACHE
    cache = allowlisted_rows(cache_path, wanted) if cache_path.is_file() else {}
    keyed_cache = {("train", a): row for a, row in cache.items()}
    rows, inputs = [], {}
    for accession in allowlist:
        json_path = work / "alphafill" / "train" / "json" / f"{accession}.json"
        cif_path = work / "alphafill" / "train" / "cif" / f"{accession}.cif"
        uniprot_path = work / "uniprot" / "train" / f"{accession}.json"
        fetched = fetch.get(accession, {})
        built = build[accession]
        annotated = cp.cached_uniprot_supported_metals(keyed_cache, split="train", accession=accession)
        annotated = set(annotated or [])
        cofactor_names = cp.extract_uniprot_tsv_cofactor_names(cache.get(accession, {}).get("cofactor", ""))
        try:
            sequence_ok = bool(asr._sequence(proteins[accession].get("sequence", "")))
        except ValueError:
            sequence_ok = False
        json_status, cif_status = fetched.get("json_status", "NO_FETCH_RECORD"), fetched.get("cif_status", "")
        item: dict[str, Any] = {
            "accession": accession, "sequence_available": sequence_ok, "build_status": built["status"],
            "build_detail": built["detail"], "recorded_json_status": json_status, "recorded_cif_status": cif_status,
            "json_present": json_path.is_file(), "cif_present": cif_path.is_file(),
            "uniprot_json_present": uniprot_path.is_file(),
            "cif_skip_kind": ("deliberate_skip_no_json_supported_candidate" if cif_status == DELIBERATE_CIF_SKIP
                              else "not_requested" if cif_status == "NOT_REQUESTED"
                              else "download_failure" if cif_status.startswith(("HTTP_", "FAILED")) else cif_status),
            "json_fetch_kind": ("downloaded" if json_status in {"OK", "SKIPPED"}
                                else "download_failure_http_404" if json_status == "HTTP_404"
                                else "download_failure_other" if json_status.startswith(("HTTP_", "FAILED"))
                                else json_status.lower()),
            "annotated_metals_cache": ";".join(sorted(annotated)),
            "annotated_metals_recorded_fetch": fetched.get("uniprot_supported_transition_metals", ""),
            "annotated_metals_recorded_build": built["uniprot_supported_transition_metals"],
            "uniprot_cofactor_names": ";".join(cofactor_names),
            "recorded_json_candidate_count": fetched.get("json_candidate_count", ""),
            "recorded_json_supported_candidate_count": fetched.get("json_supported_candidate_count", ""),
            "hits": "", "qualifying_hits": "", "elemental_candidates": "", "annotation_compatible_candidates": "",
            "qualifying_compound_ids": "", "metal_containing_compound_ids": "", "json_sha256": ""}
        review = []
        if not sequence_ok:
            review.append("source sequence unavailable")
        if built["status"] != MISSING_ASSETS:
            review.append(f"build status {built['status']}")
        if item["cif_present"]:
            review.append("CIF now present")
        if not fetched:
            review.append("no training fetch record")
        if annotated != set(filter(None, item["annotated_metals_recorded_build"].split(";"))):
            review.append("annotation cache differs from the build record")
        if item["json_present"]:
            inputs[f"alphafill_json/{accession}"] = file_record(json_path)
            item["json_sha256"] = inputs[f"alphafill_json/{accession}"]["sha256"]
            data = asr.read_json(json_path)
            hits = qualifying_hits(data, policy)
            elemental = cp.extract_candidate_dicts(data, policy)
            compatible = [c for c in elemental if c["metaltype"] in annotated]
            compounds = sorted({str(t.get("compound_id") or t.get("analogue_id") or "").strip().upper()
                                for h in hits for t in (h.get("transplants") or [])} - {""})
            metal_compounds = [c for c in compounds if c in METAL_CONTAINING_COMPOUNDS]
            item.update(hits=len(data.get("hits", []) or []), qualifying_hits=len(hits),
                        elemental_candidates=";".join(sorted(c["metaltype"] for c in elemental)) or "",
                        annotation_compatible_candidates=len(compatible),
                        qualifying_compound_ids=";".join(compounds),
                        metal_containing_compound_ids=";".join(metal_compounds))
            if json_status not in {"OK", "SKIPPED"}:
                review.append(f"JSON present but recorded {json_status}")
            if str(len(elemental)) != item["recorded_json_candidate_count"]:
                review.append("elemental candidate count differs from the fetch record")
            if str(len(compatible)) != item["recorded_json_supported_candidate_count"]:
                review.append("annotation-compatible count differs from the fetch record")
            if compatible:
                review.append("annotation-compatible candidate found")
            if review:
                category = "unexpected_change_requiring_review"
            elif elemental:
                category = "elemental_candidates_rejected_by_annotation"
            elif metal_compounds:
                category = "compound_ids_outside_exact_element_mapping"
            else:
                category = "no_qualifying_donor_or_transplant"
        else:
            if json_status in {"OK", "SKIPPED"}:
                review.append("JSON recorded as downloaded but absent")
            category = "unexpected_change_requiring_review" if review else "recorded_missing_json"
        item["rejection_category"] = category
        item["review_notes"] = "; ".join(review)
        rows.append(item)
    present_json = [r for r in rows if r["json_present"]]
    summary = {
        "allowlist": len(rows), "sequences_available": sum(r["sequence_available"] for r in rows),
        "json_present": len(present_json), "cif_present": sum(r["cif_present"] for r in rows),
        "complete_json_cif_pairs": sum(r["json_present"] and r["cif_present"] for r in rows),
        "uniprot_json_present": sum(r["uniprot_json_present"] for r in rows),
        "deliberate_cif_skips": sum(r["cif_skip_kind"] == "deliberate_skip_no_json_supported_candidate" for r in rows),
        "json_http_404": sum(r["json_fetch_kind"] == "download_failure_http_404" for r in rows),
        "missing_json_accessions": sorted(r["accession"] for r in rows if not r["json_present"]),
        "missing_json_annotations": sorted(r["annotated_metals_cache"] or "none" for r in rows if not r["json_present"]),
        "annotation_compatible_candidates": sum(int(r["annotation_compatible_candidates"] or 0) for r in present_json),
        "elemental_candidates": sum(len([m for m in r["elemental_candidates"].split(";") if m]) for r in present_json),
        "accessions_with_metal_containing_compounds": sum(bool(r["metal_containing_compound_ids"]) for r in rows),
        "rejection_categories": {c: sum(r["rejection_category"] == c for r in rows) for c in REJECTION_CATEGORIES},
        "policy": {k: v for k, v in vars(policy).items()},
        "annotation_source": f"{ANNOTATION_CACHE} (allowlisted training rows only)",
        "compound_lookup_note": ("METAL_CONTAINING_COMPOUNDS is a descriptive lookup; the pipeline filter accepts only "
                                 "exact elemental symbols and is unchanged"),
    }
    return rows, {"summary": summary, "inputs": inputs, "annotation_cache_present": cache_path.is_file()}


# ---------------------------------------------------------------------------
# Reproduction check, outputs and the command line
# ---------------------------------------------------------------------------

def observed_values(support: dict, focal: dict, attrition: dict, assets: dict, vocabulary: list[str]) -> dict:
    s, a = attrition["summary"], assets["summary"]
    return {
        "proteins": support["proteins"], "pockets": support["pockets"], "components": support["components"],
        "class_proteins": [support["per_class"][label]["proteins"] for label in vocabulary],
        "class_components": [support["per_class"][label]["components"] for label in vocabulary],
        "largest_component_sizes": support["largest_component_sizes"],
        "focal_qualifying_unordered_pairs": focal["qualifying_unordered_focal_pairs"],
        "focal_components": focal["components"],
        "focal_connected_without_other_classes": focal["connected_using_only_focal_links"],
        "focal_sequence_discrepancies": len(focal["sequence_discrepancies"]),
        "focal_unambiguous_training_proteins": s["focal_unambiguous_training_proteins"],
        "focal_in_baseline": s["focal_in_baseline"], "focal_outside_baseline": s["focal_outside_baseline"],
        "build_outcomes": s["build_outcomes"], "prepared_structures": s.get("prepared_structures", 0),
        "prepared_candidate_sites": s.get("prepared_candidate_sites", 0),
        "final_catalytic_sites": s.get("final_catalytic_sites", 0),
        "final_noncatalytic_sites": s.get("final_noncatalytic_sites", 0), "allowlist": a["allowlist"],
        "allowlist_sequences": a["sequences_available"], "allowlist_json": a["json_present"],
        "allowlist_cif": a["cif_present"], "allowlist_complete_pairs": a["complete_json_cif_pairs"],
        "deliberate_cif_skips": a["deliberate_cif_skips"], "json_http_404": a["json_http_404"],
        "missing_json_accessions": a["missing_json_accessions"],
        "annotation_compatible_candidates": a["annotation_compatible_candidates"]}


def interpretation(support: dict, focal: dict) -> list[str]:
    lines = [
        "The conservative dependence components (same protein, identical sequence, or >20% identity at >=50% "
        "shorter-sequence coverage) were built as uncertainty units for remoteness reporting; they are not an "
        "already mandatory standalone EC split rule.",
        "Accession-grouped validation remains possible; validation proteins of a class that sits in one component "
        "then have related training sequences, which must be disclosed.",
        "The remoteness minimum-support requirement (components per class per stratum) belongs to that analysis and is "
        "not imported here.",
    ]
    if support["component_grouping_prevents_all_class_train_val_coverage"]:
        lines.insert(1, "Adopted as indivisible split units, these components would prevent train/validation coverage "
                        f"of all {len(support['per_class'])} classes: "
                        f"{', '.join('EC' + c for c in support['classes_in_fewer_than_two_components'])} "
                        "lie(s) in fewer than two components.")
    if focal["components"] == 1 and focal["connected_using_only_focal_links"]:
        lines.append(f"All EC{focal['focal_class']} proteins share one component and stay connected through "
                     f"EC{focal['focal_class']}-to-EC{focal['focal_class']} links alone, so the shared component does not "
                     "depend on its other-class member(s).")
    return lines


def write_once(path: Path, content: str) -> None:
    if path.exists():
        require(path.read_text(encoding="utf-8") == content, f"Refusing to overwrite differing result {path}")
        return
    path.write_text(content, encoding="utf-8")


def csv_text(rows: list[dict[str, Any]], fields: list[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in fields})
    return buffer.getvalue()


def check_output_dir(output: Path, roots: list[Path]) -> None:
    resolved = output.resolve()
    for root in roots:
        root = root.resolve()
        require(resolved != root and root not in resolved.parents,
                f"The output directory {output} lies inside the input/evidence root {root}")
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
            f"The output directory {output} exists and is not empty; use a fresh directory")


def collect_input_records(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    records = {"baseline_split": file_record(args.baseline_split)}
    records.update({f"remoteness/{k}": file_record(args.remoteness_root / v) for k, v in REMOTENESS_FILES.items()})
    records.update({f"care_metadata/{k}": file_record(args.care_metadata_root / v)
                    for k, v in CARE_METADATA_FILES.items()})
    cache = args.care_work_root / ANNOTATION_CACHE
    if cache.is_file():
        records["care_work/annotation_cache"] = file_record(cache)
    return records


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    """All computations; no file is written."""
    baseline = load_baseline(args.baseline_split)
    remote = load_remoteness(args.remoteness_root, baseline)
    vocabulary = list(remote["protocol"]["vocabulary"]["ec"])
    require(FOCAL_CLASS in vocabulary, "The focal EC class is not in the vocabulary")
    components, links = dependence_components(remote["records"], remote["alignments"], remote["protocol"])
    proteins = protein_table(baseline, remote["records"], components, vocabulary)
    support = support_summary(proteins, vocabulary)
    focal, link_rows = focal_dependence(proteins, links, FOCAL_CLASS, remote["discrepant_groups"])
    attrition = care_attrition(args.care_metadata_root, set(proteins), FOCAL_CLASS)
    policy = alphafill_policy(args.care_metadata_root)
    rejections, assets = asset_rejections(args.care_metadata_root, args.care_work_root, attrition["allowlist"],
                                          attrition["proteins"], attrition["build"], policy)
    observed = observed_values(support, focal, attrition, assets, vocabulary)
    discrepancies = [{"item": k, "expected": v, "observed": observed.get(k)}
                     for k, v in HANDOFF_EXPECTATIONS.items() if observed.get(k) != v]
    results = {
        "baseline_support": support, "membership_check": remote["membership_check"],
        "baseline_cohort_table": Path(asr.read_json(args.baseline_split)["summary_csv"]).name,
        "alignment_diagnostics": remote["diagnostics"], "focal_dependence": focal,
        "focal_attrition": attrition["summary"], "allowlist": attrition["allowlist"],
        "asset_rejections": assets["summary"], "interpretation": interpretation(support, focal),
        "component_rule": {k: remote["protocol"][k] for k in ("component_rule", "component_identity_threshold",
                                                              "audit_shorter_coverage", "max_evalue",
                                                              "min_residue_pairs")},
    }
    return {"results": results, "reproduction_check": {"expected": HANDOFF_EXPECTATIONS, "observed": observed,
                                                       "discrepancies": discrepancies,
                                                       "all_match": not discrepancies},
            "tables": {"proteins": sorted(proteins.values(), key=lambda r: r["accession"]),
                       "links": link_rows, "inventory": attrition["inventory"], "rejections": rejections},
            "extra_inputs": assets["inputs"]}


PROTEIN_FIELDS = ["accession", "ec_class", "split_side", "pockets", "structures", "component_id", "component_size",
                  "component_classes", "care_relationship", "sequence_sha256"]
LINK_FIELDS = ["protein_a", "class_a", "protein_b", "class_b", "link_types", "evidence_rows", "qualifying_rows",
               "qualifies_component_rule", "best_identity", "best_shorter_coverage", "best_residue_pairs",
               "best_evalue", "same_component"]
INVENTORY_FIELDS = ["accession", "ecnumbers", "sequence_length", "in_baseline", "build_status", "build_detail",
                    "uniprot_supported_transition_metals", "recorded_selected_sites", "candidate_sites",
                    "final_catalytic_sites", "final_noncatalytic_sites", "sites_without_final_prediction",
                    "attrition_reason"]
REJECTION_FIELDS = ["accession", "rejection_category", "review_notes", "sequence_available", "build_status",
                    "build_detail", "recorded_json_status", "json_fetch_kind", "json_present", "json_sha256",
                    "recorded_cif_status", "cif_skip_kind", "cif_present", "uniprot_json_present",
                    "annotated_metals_cache", "annotated_metals_recorded_fetch", "annotated_metals_recorded_build",
                    "uniprot_cofactor_names", "recorded_json_candidate_count",
                    "recorded_json_supported_candidate_count", "hits", "qualifying_hits", "elemental_candidates",
                    "annotation_compatible_candidates", "qualifying_compound_ids", "metal_containing_compound_ids"]


def render_report_md(report: dict[str, Any], manifest_note: str) -> str:
    r = report["results"]
    s, f, a, x = r["baseline_support"], r["focal_dependence"], r["focal_attrition"], r["asset_rejections"]
    vocab = list(s["per_class"])
    lines = [f"# EC1 component support and EC{FOCAL_CLASS} attrition", "",
             f"Campaign `{CAMPAIGN_ID}`. CPU evidence and data preparation only: no training, inference, download, "
             "similarity search, feature generation, split writing or held-out access. " + manifest_note, "",
             "## Development support (saved EC1 membership, frozen relationship rule)", "",
             f"- Proteins {s['proteins']}, pockets {s['pockets']}, dependence components {s['components']}; largest "
             f"components {', '.join(map(str, s['largest_component_sizes']))} proteins.",
             f"- Repeated run memberships: {r['membership_check']['ec_run_memberships']} saved EC runs, "
             f"{r['membership_check']['unique_ec_memberships']} unique membership, equal to the baseline (counted once).",
             "", "| Class | Proteins | Pockets | Components |", "|---|---:|---:|---:|"]
    lines += [f"| EC{c} | {s['per_class'][c]['proteins']} | {s['per_class'][c]['pockets']} | "
              f"{s['per_class'][c]['components']} |" for c in vocab]
    lines += ["", "Component counts overlap across classes: one component can hold several classes.", "",
              f"## EC{FOCAL_CLASS} relationships", "",
              f"- {len(f['proteins'])} EC{FOCAL_CLASS} proteins ({', '.join(f['proteins'])}) in {f['components']} "
              f"component(s); other-class members: "
              f"{', '.join(m['protein'] + ' (EC' + m['ec_class'] + ')' for m in f['other_class_members']) or 'none'}.",
              f"- Qualifying unordered EC{FOCAL_CLASS}-EC{FOCAL_CLASS} pairs: {f['qualifying_unordered_focal_pairs']}; "
              f"connected through those links alone: {f['connected_using_only_focal_links']}.",
              f"- Recorded source/coordinate sequence discrepancies among them: "
              f"{', '.join(f['sequence_discrepancies']) or 'none'}.", "", "Interpretation:", ""]
    lines += [f"- {line}" for line in r["interpretation"]]
    lines += ["", f"## EC{FOCAL_CLASS} attrition in the CARE clusterRes30 training source", "",
              f"- Unambiguous EC{FOCAL_CLASS} training proteins {a['focal_unambiguous_training_proteins']}: "
              f"{a['focal_in_baseline']} in the baseline, {a['focal_outside_baseline']} outside "
              f"({a['focal_ambiguous_multi_class_proteins']} further proteins mix EC{FOCAL_CLASS} with another class).",
              "- Recorded build outcomes: " + ", ".join(f"{k} {v}" for k, v in a["build_outcomes"].items()) + ".",
              f"- The {a.get('prepared_structures', 0)} prepared structures supplied {a.get('prepared_candidate_sites', 0)} "
              f"candidate sites; saved final predictions: {a.get('final_catalytic_sites', 0)} catalytic, "
              f"{a.get('final_noncatalytic_sites', 0)} noncatalytic ({a['catalytic_assignment']}).",
              "- Attrition reasons: " + ", ".join(f"{k} {v}" for k, v in a["attrition_reasons"].items()) + "; the "
              f"baseline cohort table is `{r['baseline_cohort_table']}`.",
              "", f"## The {x['allowlist']} missing-asset accessions", "",
              f"- Sequences {x['sequences_available']}, AlphaFill JSON {x['json_present']}, CIF {x['cif_present']}, "
              f"complete pairs {x['complete_json_cif_pairs']}; deliberate CIF skips {x['deliberate_cif_skips']}, "
              f"JSON HTTP 404 {x['json_http_404']} ({', '.join(x['missing_json_accessions'])}).",
              f"- Recorded policy: identity >= {x['policy']['min_alphafill_identity']}, alignment length >= "
              f"{x['policy']['min_alignment_length']}, optional RMSD/PAE/atom-count filters disabled, UniProt policy "
              f"{x['policy']['uniprot_metal_policy']}; annotation-compatible candidates: "
              f"{x['annotation_compatible_candidates']} (elemental candidates {x['elemental_candidates']}).",
              "- Rejection categories: " + ", ".join(f"{k} {v}" for k, v in x["rejection_categories"].items()) + ".",
              f"- Accessions whose qualifying donors carry metal-containing cofactor compounds (FES, SF4, hemes and "
              f"similar; not interpreted by the exact-element filter): {x['accessions_with_metal_containing_compounds']}.",
              "", "## Reproduction check against the handoff", "",
              "All handoff values reproduced." if report["reproduction_check"]["all_match"] else
              "Discrepancies: " + "; ".join(f"{d['item']}: expected {d['expected']}, observed {d['observed']}"
                                            for d in report["reproduction_check"]["discrepancies"]), ""]
    return "\n".join(lines)


def render_decision_md(report: dict[str, Any]) -> str:
    r = report["results"]
    s, f, a, x = r["baseline_support"], r["focal_dependence"], r["focal_attrition"], r["asset_rejections"]
    reasons = a["attrition_reasons"]
    cats = x["rejection_categories"]
    noncat = reasons.get("prepared_noncatalytic_prediction_only", 0)
    return "\n".join([
        "# Next scientific decision: stronger EC evidence", "",
        "Evidence boundary: counts from the saved EC1 development membership and recorded CARE training-side build "
        "tables only. Nothing here trains, promotes, downloads or changes eligibility. No route is claimed to improve "
        "performance; neither route merges new examples into the existing cohort.", "",
        "## What limits EC7 now", "",
        f"- Support: {s['per_class'][FOCAL_CLASS]['proteins']} EC{FOCAL_CLASS} proteins in "
        f"{s['per_class'][FOCAL_CLASS]['components']} dependence component; accession-grouped validation can hold "
        "EC7 only with related training sequences (disclosed).",
        f"- Source exhausted under current eligibility: of {a['focal_unambiguous_training_proteins']} unambiguous EC7 "
        f"training proteins, {a['build_outcomes'].get('NO_UNIPROT_SUPPORTED_TRANSITION_METAL', 0)} lack a "
        f"UniProt-supported transition metal, {a['build_outcomes'].get(MISSING_ASSETS, 0)} lack assets, {noncat} were "
        "prepared but carry only noncatalytic-predicted sites (excluded by the only-catalytic definition; never "
        f"relabel), {a['focal_in_baseline']} are in the baseline.",
        f"- The missing assets are deliberate: {x['deliberate_cif_skips']} CIFs were skipped because the JSON had no "
        f"annotation-compatible elemental candidate (re-derived: {x['annotation_compatible_candidates']}), and "
        f"{x['json_http_404']} JSONs returned HTTP 404. Fetching the skipped CIFs would add no eligible site.",
        f"- Rejections: {cats['elemental_candidates_rejected_by_annotation']} elemental candidates rejected by "
        f"annotation, {cats['compound_ids_outside_exact_element_mapping']} only metal-containing compounds outside the "
        f"exact-element mapping, {cats['no_qualifying_donor_or_transplant']} no qualifying donor/transplant, "
        f"{cats['recorded_missing_json']} recorded missing JSON, {cats['unexpected_change_requiring_review']} "
        "unexpected changes.", "",
        "## Route 1: preserve current eligibility", "",
        "- **Population:** EC depth-1 proteins with an annotation-compatible elemental transition-metal pocket "
        "(Mn, Fe, Co, Ni, Cu, Zn) predicted catalytic, as in the v12 EC1 cohort.",
        "- **Input/eligibility changes:** none.",
        "- **Evidence available:** this report shows the CARE clusterRes30 training source adds no EC7 example under "
        f"this definition (unresolved: the {x['json_http_404']} HTTP-404 JSONs, annotated "
        f"{', '.join(sorted(set(x['missing_json_annotations'])))}).",
        "- **Unknowns:** whether other permitted, non-test sources hold independent EC7 (and EC5/EC6) transition-metal "
        "catalytic pockets; their identity/homology to the development set and to every held-out set; whether the two "
        "404 entries now exist.",
        "- **Code/data preparation:** a CPU inventory of already-local permitted training-side sources with the same "
        "filters; cross-source identity/homology and held-out exclusion certification; any new structures through the "
        "unchanged care_prepare pipeline; a new cohort version identity.",
        "- **Validation and rare classes:** grouped by accession; report EC7 recall with intervals and disclose related "
        "training sequences; component-grouped EC7 assessment needs at least two independent EC7 components.",
        "- **Reuse:** care_prepare, the remoteness tooling, this reporter and the v12 membership.",
        "- **Authorization/choice:** which sources are permitted; any AlphaFill/UniProt/PDB download needs the user's "
        "authorization.", "",
        "## Route 2: separately versioned cofactor-inclusive EC cohort", "",
        "- **Population:** a different one: enzymes whose metal is held in a cofactor (Fe-S clusters, hemes, "
        "cobalamins) as well as elemental sites.",
        "- **Input/eligibility changes:** a reviewed compound-to-metal mapping, a cofactor-pocket centre definition, "
        "annotation matching for cofactors, and a decision whether MAHOMES catalytic prediction applies to cofactor "
        "sites; the metal task stays out of this cohort unless separately defined.",
        f"- **Evidence available:** {x['accessions_with_metal_containing_compounds']} of the {x['allowlist']} "
        "allowlisted EC7 accessions have qualifying AlphaFill donors carrying metal-containing compounds; no CIF was "
        "downloaded for them.",
        "- **Unknowns:** scientific appropriateness (cofactor identity may act as an EC shortcut), independence "
        "(complex subunits, homologs), transplant quality for large cofactors, MAHOMES applicability, and whether "
        "heme-only annotations hid further proteins among those without a supported annotation (not inspected).",
        "- **Code/data preparation:** an additive, versioned compound mapping in care_prepare; CIF downloads for "
        "candidates; a separate work root and cohort; the same support/remoteness analysis; held-out exclusion.",
        "- **Validation and rare classes:** a separate cohort identity, never merged into v12; matched comparison inside "
        "the cohort only; cofactor-stratified metrics as shortcut control.",
        "- **Reuse:** the fetch/build pipeline (with the additive option), this reporter, the remoteness tooling.",
        "- **Authorization/choice:** the scientific decision to define this population, then download "
        "authorization.", "",
        "## Missing interface (documented, not implemented)", "",
        "EC training cannot take an explicit frozen train/validation membership: `--fold-split-source membership` needs "
        "`--source-cohort-csv`, which `src/training/config.py` restricts to `--task metal --metal-example-unit ion`. "
        "The smallest later bridge: accept an EC membership CSV (structure/pocket id, protein group, fold) with its "
        "SHA-256 for `--task ec`, refuse rows outside the cohort, keep `structure_id` EC group weighting, and prove "
        "parity by reproducing the saved v12 fixed split exactly. Its cohort and validation contract depend on the "
        "route chosen above.", "",
        "## Recommendation", "",
        "Route 1 first: keep the v12 population, comparable to the metal task's elemental-site definition, and run a "
        "CPU-only inventory of already-local permitted training-side sources for independent EC7/EC6/EC5 pockets. "
        "Treat Route 2 as a separately versioned exploratory cohort, opened only after an explicit scientific decision "
        "on cofactor pockets. Remaining decision (the user's): which training-side sources are permitted for the "
        "Route 1 inventory, and whether Route 2 should be scoped at all. No download, training or GPU use follows from "
        "this report.", ""])


def run(args: argparse.Namespace) -> dict[str, Any]:
    roots = [args.remoteness_root, args.care_metadata_root, args.care_work_root, args.baseline_split.parent]
    check_output_dir(args.output_dir, roots)
    before = collect_input_records(args)
    report = build_report(args)
    before.update(report["extra_inputs"])
    after = collect_input_records(args)
    after.update({k: file_record(Path(v["path"])) for k, v in report["extra_inputs"].items()})
    require(all(after[k]["sha256"] == v["sha256"] for k, v in before.items()), "An input changed during the run")
    sources = {name: file_record(path) for name, path in (
        ("reporter", Path(__file__).resolve()), ("audit_sequence_remoteness", Path(asr.__file__).resolve()),
        ("care_prepare", Path(cp.__file__).resolve()))}
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    boundary = {"training": False, "inference": False, "download": False, "similarity_search": False,
                "feature_generation": False, "split_writing": False, "held_out_evaluation": False,
                "test_side_rows_used": False}
    support_report = {"schema_version": 1, "report": "ec1_component_support", "campaign_id": CAMPAIGN_ID,
                      "created_at": created, "boundary": boundary,
                      "inputs": {k: before[k] for k in sorted(before)}, "sources": sources,
                      "results": report["results"], "reproduction_check": report["reproduction_check"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir
    t = report["tables"]
    files = {
        "support_report.json": json.dumps(support_report, indent=2, sort_keys=True, default=str) + "\n",
        "proteins.csv": csv_text(t["proteins"], PROTEIN_FIELDS),
        "ec7_links.csv": csv_text(t["links"], LINK_FIELDS),
        "ec7_source_inventory.csv": csv_text(t["inventory"], INVENTORY_FIELDS),
        "ec7_asset_rejections.csv": csv_text(t["rejections"], REJECTION_FIELDS),
        "report.md": render_report_md(report, "Inputs and sources are bound by SHA-256 in artifact_manifest.json."),
        "next_scientific_decision.md": render_decision_md(report),
    }
    for name, content in files.items():
        write_once(out / name, content)
    manifest = {"schema_version": 1, "campaign_id": CAMPAIGN_ID, "created_at": created,
                "command": [sys.executable, *sys.argv], "python": platform.python_version(), "boundary": boundary,
                "inputs_before": {k: before[k] for k in sorted(before)},
                "inputs_unchanged_after_run": True, "sources": sources,
                "outputs": {name: file_record(out / name) for name in sorted(files)},
                "reproduction_all_match": report["reproduction_check"]["all_match"]}
    write_once(out / "artifact_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"output_dir": str(out), "artifact_manifest_sha256": asr.sha256(out / "artifact_manifest.json"),
            "reproduction_all_match": report["reproduction_check"]["all_match"],
            "discrepancies": report["reproduction_check"]["discrepancies"]}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-split", type=Path, required=True, help="saved EC1 expected_split.json")
    parser.add_argument("--remoteness-root", type=Path, required=True, help="frozen remote_homology_v1 folder")
    parser.add_argument("--care-metadata-root", type=Path, required=True,
                        help="CARE_prepare_training_and_test_set/provenance/clusterRes30/metadata")
    parser.add_argument("--care-work-root", type=Path, required=True, help="the CARE task1_30_clusterRes30 work root")
    parser.add_argument("--output-dir", type=Path, required=True, help="a fresh directory outside every input root")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except (ReportError, ValueError, KeyError, OSError) as exc:
        print(f"EC1 component-support report refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
