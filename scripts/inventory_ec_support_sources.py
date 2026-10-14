#!/usr/bin/env python3
"""Read-only inventory of already-local TRAINING sources for additional EC7, EC6 and EC5 support under the current
EC1 eligibility (campaign ec1_independent_support_2026-10, log ec1s-002).

Sources (training side only):
  A. CARE Task-1 full training metadata (protein_train.csv): clusterRes30 NON-representatives. A representative is an
     accession whose clusterRes30 value is itself; non-representatives share their representative's cluster and are
     never treated as independent of it. Existing assets come from the unclustered CARE work root (task1_30): its
     UniProt annotation cache (training rows of allowlisted accessions only), its AlphaFill training JSON/CIF files and
     its training build summary.
  B. The CARE clusterRes30 cohort (representatives): the recorded training build, the predictive final site table and
     the prepared training cohort, through scripts/report_ec1_component_support.py (care_attrition, asset_rejections).
  C. The non-overlap PinMyMetal training side: its catalytic site table, structure manifest, the SIFTS chain table and
     the native-annotation table in the same train/ folder.

Eligibility is the v12 EC1 definition, unchanged: a CARE/UniProt depth-1 single label (src/training/labels.py), a
UniProt-supported transition metal (require_supported), an AlphaFill transplant passing the recorded identity/length
policy with an annotation-compatible elemental candidate, a built structure and a MAHOMES-predicted catalytic site.
Missing evidence is "unknown", never eligible. Relationships use exact sequences, CARE clusterRes30 co-membership and
the saved development sequences only: no alignment search is run. Held-out exclusion is never certified here.

Held-out boundary: no held-out file or directory is opened (input paths are checked, symlinks resolved). Three
mixed-split tables are hashed whole for provenance and their rows are parsed and discarded by a split filter while
streaming (the two UniProt annotation caches and the clusterRes30 fetch summary); no non-training value is used or
written. The tool has no download, training, inference, build, alignment or split-writing mode.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT / "CARE_prepare_training_and_test_set")]

import report_ec1_component_support as rep  # noqa: E402
from training import labels  # noqa: E402

asr, cp = rep.asr, rep.cp
require, file_record, csv_text, write_once, check_output_dir = (rep.require, rep.file_record, rep.csv_text,
                                                                rep.write_once, rep.check_output_dir)
ReportError = rep.ReportError

CAMPAIGN_ID = rep.CAMPAIGN_ID
FOCAL_CLASSES = ("7", "6", "5")  # priority order
SOURCES = ("care_full_train_nonrepresentative", "care_clusterres30_cohort", "pmm_nonoverlap_train")
# A held-out directory anywhere on the given or resolved path, or a held-out-named final file.
HELD_OUT_DIR = re.compile(r"^(tests?|test[_-]?sets?|test[_-]?structures|test[_-]?split|held[_-]?out|hold[_-]?out)$",
                          re.IGNORECASE)
HELD_OUT_FILE = re.compile(r"(^|[_.-])(test|held[_-]?out|hold[_-]?out)([_.0-9-]|$)", re.IGNORECASE)
CARE_FULL_COLUMNS = ("Entry", "Sequence", "EC number", "clusterRes30")
PMM_FILES = {"sites": "final_data_summarazing_table_transition_metals_only_catalytic.csv",
             "manifest": "structure_manifest.csv", "sifts": "cofactor_analysis_by_structure_chain.csv",
             "native": "final_data_summarazing_table_transition_metals_only_catalytic_with_native.csv"}
PREPARED_FILES = {"sites": "final_data_summarazing_table_transition_metals_only_catalytic.csv",
                  "manifest": "structure_manifest.csv"}
UNCLUSTERED_BUILD = {"parameters": Path("mahomes_inputs") / "build_parameters.json",
                     "train_build_summary": Path("mahomes_inputs") / "train" / "build_summary.csv"}
POLICY_KEYS = ("min_alphafill_identity", "min_alignment_length", "site_dedup_distance", "uniprot_metal_policy")
MIXED_SPLIT_INPUTS = ("care_full_annotation_cache", "clusterres30_annotation_cache", "care_metadata/fetch_summary")
NOT_CERTIFIED = "not_certified"
HOMOLOGY_UNKNOWN = "unknown_no_development_vs_candidate_search"
POSITIVE = {"in_development_cohort", "exact_sequence_alias_of_development_protein",
            "same_clusterres30_as_development_protein"}
UNRELATED = {"none_known", "cluster_representative_absent_from_training_entries", "unmapped_chain"}
DOCUMENTED_REJECTIONS = {"no_qualifying_donor_or_transplant", "compound_ids_outside_exact_element_mapping",
                         "elemental_candidates_rejected_by_annotation"}
# Values computed independently during the read-only source mapping (scratch, 2026-10-13); compared, never enforced.
MAPPING_EXPECTATIONS = {
    "care_full_train_rows": 184529, "care_full_train_accessions": 173550, "clusterres30_representatives": 9594,
    "clusterres30_representatives_with_entries": 9466,
    "nonrep_unambiguous": {"7": 6311, "6": 21379, "5": 11351},
    "nonrep_same_cluster_as_development": {"7": 178, "6": 2572, "5": 885},
    "nonrep_representative_absent": {"7": 184, "6": 24, "5": 285},
    "nonrep_supported_metal": {"7": 1156, "6": 4824, "5": 1193},
    "nonrep_supported_json": {"7": 392, "6": 2437, "5": 539},  # refine.out (2,438 = all EC6 JSONs)
    "nonrep_supported_cif": {"7": 97, "6": 1923, "5": 532},
    "cohort_unambiguous": {"7": 179, "6": 388, "5": 459}, "cohort_in_baseline": {"7": 6, "6": 29, "5": 26},
    "cohort_build_outcomes": {
        "7": {"MISSING_ALPHAFILL_JSON_OR_CIF": 24, "NO_UNIPROT_SUPPORTED_TRANSITION_METAL": 141, "OK": 14},
        "6": {"MISSING_ALPHAFILL_JSON_OR_CIF": 37, "NO_UNIPROT_SUPPORTED_TRANSITION_METAL": 313, "OK": 38},
        "5": {"MISSING_ALPHAFILL_JSON_OR_CIF": 15, "NO_UNIPROT_SUPPORTED_TRANSITION_METAL": 410, "OK": 34}},
    "cohort_noncatalytic_only_outside": {"7": 8, "6": 9, "5": 8},
    "prepared_train_proteins": 817, "prepared_absent_from_baseline": 24,
    "prepared_absent_with_focal_class": {"7": 0, "6": 0, "5": 3},
    "pmm_site_rows": 1823, "pmm_chains": 1304, "pmm_pdb_label_chains": {"7": 1, "6": 16, "5": 91},
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def training_path(path: Path) -> Path:
    """Refuse an input whose given or symlink-resolved path names a held-out directory or file."""
    require(path.exists(), f"{path} is missing")
    for candidate in (Path(path), Path(path).resolve()):
        parts = candidate.parts
        bad_dir = [p for p in (parts[:-1] if candidate.is_file() else parts) if HELD_OUT_DIR.match(p)]
        bad_file = candidate.is_file() and HELD_OUT_FILE.search(candidate.name)
        require(not bad_dir and not bad_file,
                f"{path} names a held-out location ({candidate}); this inventory reads training sources only")
    return path


def read_rows(path: Path) -> list[dict[str, str]]:
    with training_path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_sequence(value: str) -> str:
    return "".join(str(value).split()).upper()


def ec_set(value: str) -> set[str]:
    return {ec for ec in labels.normalize_ec_number_list(value).split(";") if ec}


def depth1(ecs: set[str]) -> str | None:
    """The training rule (src/training/labels.py): one depth-1 prefix, else no target (None)."""
    return labels.ec_label_token_from_numbers(tuple(sorted(ecs)), depth=1)


def depth1_classes(ecs: set[str]) -> set[str]:
    return {ec.split(".")[0] for ec in ecs}


def joined(values) -> str:
    return ";".join(sorted({str(v) for v in values if str(v)}))


def listing_digest(names) -> dict[str, Any]:
    ordered = sorted(names)
    return {"names": len(ordered), "sha256": sha256_text("\n".join(ordered))}


# ---------------------------------------------------------------------------
# Shared evidence: the development membership, its sequences and the previous report
# ---------------------------------------------------------------------------

def load_development(baseline_split: Path, remoteness_root: Path, previous_report: Path) -> dict[str, Any]:
    rep.load_baseline(training_path(baseline_split))  # validates the training-side development source
    summary = asr.read_json(baseline_split)
    pockets: Counter = Counter()
    for side in ("train", "validation"):
        for example in summary["retained_split_identity"][side]["examples"]:
            accession = example["structure_id"].split("__")[0].upper()
            require(example["group"] == accession.lower(), f"{example['structure_id']}: group is not the accession")
            pockets[accession] += 1
    manifest = asr.read_json(training_path(remoteness_root / rep.REMOTENESS_FILES["sequence_manifest"]))
    ec_hashes: dict[str, str] = {}
    metal: dict[str, dict[str, str]] = {}
    metal_by_hash: dict[str, set[str]] = defaultdict(set)
    for entry in manifest["sequences"]:
        if entry["task"] == "ec":
            for key in ("sequence_sha256", "care_sequence_sha256"):
                if entry.get(key):
                    ec_hashes[entry[key]] = entry["group_id"].upper()
        elif entry["task"] == "metal":
            metal[entry["structure_id"]] = {"sha256": entry["sequence_sha256"], "group": entry["group_id"]}
            metal_by_hash[entry["sequence_sha256"]].add(entry["structure_id"])
    report_manifest = asr.read_json(training_path(previous_report / "artifact_manifest.json"))
    proteins_csv = previous_report / "proteins.csv"
    require(asr.sha256(proteins_csv) == report_manifest["outputs"]["proteins.csv"]["sha256"],
            "The previous report's proteins.csv differs from its artifact manifest")
    components = {row["accession"].upper(): row for row in read_rows(proteins_csv)}
    require(set(components) == set(pockets), "The previous report's proteins are not the saved development membership")
    return {"pockets": dict(pockets), "accessions": set(pockets), "ec_sequence_hashes": ec_hashes,
            "metal_sequences": metal, "metal_by_hash": metal_by_hash, "components": components,
            "previous_report": {"artifact_manifest_sha256": asr.sha256(previous_report / "artifact_manifest.json"),
                                "path": str(previous_report)}}


def load_care_full_train(path: Path) -> dict[str, dict[str, Any]]:
    """One record per accession: union of its rows' EC numbers, its sequence and clusterRes30 representative."""
    require(training_path(path).name == "protein_train.csv", f"{path} is not the CARE Task-1 training split file")
    proteins: dict[str, dict[str, Any]] = {}
    rows = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(all(c in (reader.fieldnames or []) for c in CARE_FULL_COLUMNS), f"{path.name}: missing columns")
        for row in reader:
            rows += 1
            accession, cluster = row["Entry"].strip(), row["clusterRes30"].strip()
            sequence = clean_sequence(row["Sequence"])
            record = proteins.setdefault(accession, {"ecs": set(), "sequence": sequence, "cluster": cluster, "rows": 0})
            require(record["sequence"] == sequence and record["cluster"] == cluster,
                    f"{accession}: rows disagree on sequence or clusterRes30")
            record["ecs"] |= ec_set(row["EC number"])
            record["rows"] += 1
    for accession, record in proteins.items():
        record["representative"] = record["cluster"] == accession
        record["sha256"] = sha256_text(record["sequence"])
        record["token"] = depth1(record["ecs"])
        record["classes"] = depth1_classes(record["ecs"])
    return {"proteins": proteins, "rows": rows}


def cluster_index(care: dict[str, dict[str, Any]], development: set[str]) -> dict[str, Any]:
    members: dict[str, set[str]] = defaultdict(set)
    by_sequence: dict[str, set[str]] = defaultdict(set)
    for accession, record in care.items():
        members[record["cluster"]].add(accession)
        by_sequence[record["sha256"]].add(accession)
    return {"members": members, "development_clusters": {care[a]["cluster"] for a in development if a in care},
            "duplicates": {h: a for h, a in by_sequence.items() if len(a) > 1},
            "representatives": set(members), "representatives_present": {c for c in members if c in care}}


def recorded_policy_matches(meta: Path, work_root: Path, policy) -> dict[str, Any]:
    """The unclustered work root must have been prepared under the recorded clusterRes30 policy."""
    recorded = asr.read_json(training_path(meta / rep.CARE_METADATA_FILES["build_parameters"]))
    own = asr.read_json(training_path(work_root / UNCLUSTERED_BUILD["parameters"]))
    differing = [k for k in POLICY_KEYS if own.get(k) != recorded.get(k)]
    differing += [] if sorted(own.get("supported_transition_metals", [])) == sorted(
        recorded.get("supported_transition_metals", [])) else ["supported_transition_metals"]
    require(not differing, f"The unclustered work root used another build policy: {differing}")
    return {k: own[k] for k in POLICY_KEYS}


def unclustered_builds(work_root: Path, wanted: set[str]) -> dict[str, dict[str, str]]:
    """Training-only build records of the unclustered work root for the wanted accessions (a smoke build here)."""
    path = work_root / UNCLUSTERED_BUILD["train_build_summary"]
    if not path.is_file():
        return {}
    rows = rep.training_rows(training_path(path), split_columns=("split",))
    return {row["uniprot_id"].strip(): row for row in rows if row["uniprot_id"].strip() in wanted}


# ---------------------------------------------------------------------------
# Source A: CARE full-train non-representatives with the unclustered work root's existing assets
# ---------------------------------------------------------------------------

def relation_for(accession: str, record: dict, clusters: dict, development: dict) -> str:
    if accession in development["accessions"]:
        return "in_development_cohort"
    if record["sha256"] in development["ec_sequence_hashes"]:
        return "exact_sequence_alias_of_development_protein"
    if record["cluster"] in clusters["development_clusters"]:
        return "same_clusterres30_as_development_protein"
    if record["cluster"] not in clusters["representatives_present"]:
        return "cluster_representative_absent_from_training_entries"
    return "none_known"


def nonrepresentative_rows(care: dict[str, dict[str, Any]], clusters: dict[str, Any], development: dict[str, Any],
                           work_root: Path, policy, focal_classes: tuple[str, ...]) -> tuple[list[dict], dict, list]:
    focal = {a: r for a, r in care.items() if not r["representative"] and r["classes"] & set(focal_classes)}
    cache = rep.allowlisted_rows(training_path(work_root / rep.ANNOTATION_CACHE), set(focal))
    builds = unclustered_builds(work_root, set(focal))
    json_dir, cif_dir = (training_path(work_root / "alphafill" / "train" / kind) for kind in ("json", "cif"))
    json_names = {n[:-5] for n in os.listdir(json_dir) if n.endswith(".json")}
    cif_names = {n[:-4] for n in os.listdir(cif_dir) if n.endswith(".cif")}
    rows, json_inputs = [], []
    for accession in sorted(focal):
        record = focal[accession]
        ambiguous = record["token"] is None
        relation = relation_for(accession, record, clusters, development)
        cached = cache.get(accession)
        supported = set(cp.cached_uniprot_supported_metals(
            {("train", accession): cached} if cached else {}, split="train", accession=accession) or [])
        evaluated: dict[str, Any] = {}
        compatible = None
        if not ambiguous and supported and accession in json_names:
            path = json_dir / f"{accession}.json"
            data = asr.read_json(path)
            elemental = cp.extract_candidate_dicts(data, policy)
            compatible = [c for c in elemental if c["metaltype"] in supported]
            hits = rep.qualifying_hits(data, policy)
            compounds = sorted({str(t.get("compound_id") or t.get("analogue_id") or "").strip().upper()
                                for h in hits for t in (h.get("transplants") or [])} - {""})
            json_inputs.append({"accession": accession, **file_record(path), "elemental_candidates": len(elemental),
                                "annotation_compatible": len(compatible)})
            evaluated = {"elemental_candidates": joined(c["metaltype"] for c in elemental),
                         "annotation_compatible_candidates": len(compatible),
                         "donor_pdb_ids": joined(c["alphafill_pdb_id"].upper() for c in compatible),
                         "metal_containing_compound_ids": joined(c for c in compounds
                                                                 if c in rep.METAL_CONTAINING_COMPOUNDS)}
        for focal_class in [c for c in focal_classes if c in record["classes"]]:
            row = {"source": SOURCES[0], "unit_id": accession, "accession": accession, "focal_class": focal_class,
                   "ec_numbers": joined(record["ecs"]), "ec_depth1_classes": joined(record["classes"]),
                   "ec_ambiguous": ambiguous, "source_rows": record["rows"], "clusterres30": record["cluster"],
                   "cluster_representative_present": record["cluster"] in care,
                   "cluster_has_development_protein": record["cluster"] in clusters["development_clusters"],
                   "development_component": development["components"].get(record["cluster"], {}).get(
                       "component_id", ""),
                   "relation": relation, "exact_sequence_duplicates": joined(
                       clusters["duplicates"].get(record["sha256"], set()) - {accession}),
                   "metal_development_exact_alias": joined(development["metal_by_hash"].get(record["sha256"], set())),
                   "annotation_status": cached["status"] if cached else "NO_TRAINING_CACHE_ROW",
                   "supported_metals": joined(supported), "alphafill_json": accession in json_names,
                   "alphafill_cif": accession in cif_names, "build_status": (builds.get(accession) or {}).get("status", ""),
                   "built": (builds.get(accession) or {}).get("status") == "OK", "catalytic_prediction": "none",
                   "final_catalytic_sites": "", "sites": "", "pockets": "", **evaluated}
            row.update(classify_nonrepresentative(row, ambiguous, supported, compatible))
            rows.append(row)
    names = sorted(json_names)
    assets = {"json_listing": listing_digest(json_names), "cif_listing": listing_digest(cif_names),
              "json_name_range": [names[0], names[-1]] if names else [], "annotation_rows_used": len(cache),
              "unclustered_build_records_for_focal_accessions": len(builds),
              "note": ("per-accession AlphaFill fetch outcomes of the unclustered run were not read; its files stop "
                       "at the last listed name (json_name_range), so most later accessions were apparently never "
                       "attempted; its training build summary (a smoke build) is read for the focal accessions")}
    return rows, assets, json_inputs


def classify_nonrepresentative(row: dict, ambiguous: bool, supported: set[str], compatible) -> dict[str, Any]:
    """Evidence level, eligibility and the first missing or failing evidence for one non-representative. A missing or
    NOT_FOUND training annotation is kept 'unknown' (the scripted build would record no supported metal: a deliberate,
    conservative difference)."""
    if ambiguous:
        level, eligibility, reason = "ambiguous_ec_no_target", "ineligible_documented", "depth-1 labels disagree"
    elif row["build_status"]:
        level, eligibility, reason = (f"unclustered_build_{row['build_status'].lower()}", "unknown",
                                      "unclustered smoke-build record; no MAHOMES prediction")
    elif row["annotation_status"] == "NO_TRAINING_CACHE_ROW":
        level, eligibility, reason = "annotation_missing", "unknown", "no training annotation row"
    elif row["annotation_status"] != "OK":
        level, eligibility, reason = "annotation_missing", "unknown", f"annotation {row['annotation_status']}"
    elif not supported:
        level, eligibility, reason = ("no_supported_metal_annotation", "ineligible_documented",
                                      "no UniProt-supported transition metal")
    elif not row["alphafill_json"]:
        level, eligibility, reason = "metadata_only_supported_metal", "unknown", "no local AlphaFill JSON"
    elif compatible is not None and not compatible:
        level, eligibility, reason = ("alphafill_no_annotation_compatible_candidate", "ineligible_documented",
                                      "local AlphaFill JSON has no annotation-compatible elemental candidate")
    elif not row["alphafill_cif"]:
        level, eligibility, reason = "compatible_candidate_cif_missing", "unknown", "AlphaFill CIF absent; not built"
    else:
        level, eligibility, reason = ("compatible_candidate_cif_present_not_built", "unknown",
                                      "not built; no MAHOMES prediction")
    missing = []
    if level == "annotation_missing":
        missing.append("annotation")
    if eligibility == "unknown":
        missing += [m for m, present in (("alphafill_json", row["alphafill_json"]), ("alphafill_cif", row["alphafill_cif"]),
                                         ("build", row["built"])) if not present] + ["catalytic_prediction"]
    return {"evidence_level": level, "eligibility": eligibility, "attrition_reason": reason,
            "missing_evidence": ";".join(missing),
            "homology_status": row["relation"] if row["relation"] in POSITIVE else HOMOLOGY_UNKNOWN,
            "heldout_exclusion": NOT_CERTIFIED}


# ---------------------------------------------------------------------------
# Source B: the CARE clusterRes30 cohort (reused attrition and asset logic) and the prepared training cohort
# ---------------------------------------------------------------------------

def cohort_classification(item: dict[str, Any], category: str) -> tuple[str, str, str]:
    """(evidence level, eligibility, missing evidence) of one clusterRes30 representative."""
    status = item["build_status"]
    if item["in_baseline"]:
        return "in_development_cohort", "eligible_in_development_cohort", ""
    if status == "OK" and item["final_catalytic_sites"] > 0 and not item["sites_without_final_prediction"]:
        return "prepared_catalytic_outside_development", "eligible_outside_cohort", ""
    if status == "OK" and item["attrition_reason"] == "prepared_noncatalytic_prediction_only":
        return "prepared_noncatalytic_prediction_only", "ineligible_documented", ""
    if status == "OK":
        return "prepared_other_review", "unknown", "review"
    if status == "NO_UNIPROT_SUPPORTED_TRANSITION_METAL":
        return "no_uniprot_supported_transition_metal", "ineligible_documented", ""
    if status == rep.MISSING_ASSETS and category in DOCUMENTED_REJECTIONS:
        return "missing_alphafill_json_or_cif", "ineligible_documented", ""
    if status == rep.MISSING_ASSETS and category == "recorded_missing_json":
        return "alphafill_json_download_failed", "unknown", "alphafill_json;alphafill_cif;build;catalytic_prediction"
    return (status.lower() or "no_build_record"), "unknown", "review"


def cohort_rows(meta: Path, clusterres30_work: Path, prepared_dir: Path, development: dict[str, Any],
                care: dict[str, dict[str, Any]], policy, focal_classes: tuple[str, ...]):
    for key, name in rep.CARE_METADATA_FILES.items():
        training_path(meta / name)
    baseline_lower = {a.lower() for a in development["accessions"]}
    rows, summaries, extra_inputs = [], {}, {}
    for focal in focal_classes:
        attrition = rep.care_attrition(meta, baseline_lower, focal)
        rejections, assets = rep.asset_rejections(meta, training_path(clusterres30_work), attrition["allowlist"],
                                                  attrition["proteins"], attrition["build"], policy)
        extra_inputs.update(assets["inputs"])
        rejected = {r["accession"]: r for r in rejections}
        for item in attrition["inventory"]:
            accession = item["accession"]
            category = rejected.get(accession, {}).get("rejection_category", "")
            level, eligibility, missing = cohort_classification(item, category)
            care_record = care.get(accession, {})
            relation = ("in_development_cohort" if item["in_baseline"] else
                        "exact_sequence_alias_of_development_protein"
                        if care_record.get("sha256") in development["ec_sequence_hashes"] else "none_known")
            rows.append({
                "source": SOURCES[1], "unit_id": accession, "accession": accession, "focal_class": focal,
                "ec_numbers": item["ecnumbers"], "ec_depth1_classes": focal, "ec_ambiguous": False,
                "source_rows": care_record.get("rows", ""), "clusterres30": accession,
                "cluster_representative_present": True,
                "cluster_has_development_protein": accession in development["accessions"],
                "development_component": development["components"].get(accession, {}).get("component_id", ""),
                "relation": relation, "exact_sequence_duplicates": "",
                "metal_development_exact_alias": joined(development["metal_by_hash"].get(care_record.get("sha256"),
                                                                                         set())),
                "annotation_status": "recorded_build", "supported_metals": item["uniprot_supported_transition_metals"],
                "build_status": item["build_status"], "built": item["build_status"] == "OK",
                "final_catalytic_sites": item["final_catalytic_sites"] if item["build_status"] == "OK" else "",
                "catalytic_prediction": (f"{item['final_catalytic_sites']} catalytic / "
                                         f"{item['final_noncatalytic_sites']} noncatalytic"
                                         if item["build_status"] == "OK" else "none"),
                "sites": item["candidate_sites"] if item["build_status"] == "OK" else "",
                "pockets": development["pockets"].get(accession, ""),
                "evidence_level": level, "eligibility": eligibility,
                "attrition_reason": category or item["attrition_reason"], "missing_evidence": missing,
                "homology_status": relation if relation in POSITIVE else HOMOLOGY_UNKNOWN,
                "heldout_exclusion": "development_member" if item["in_baseline"] else NOT_CERTIFIED,
                "elemental_candidates": rejected.get(accession, {}).get("elemental_candidates", ""),
                "annotation_compatible_candidates": rejected.get(accession, {}).get("annotation_compatible_candidates", ""),
                "metal_containing_compound_ids": rejected.get(accession, {}).get("metal_containing_compound_ids", "")})
        mixed = sorted(a for a, r in attrition["proteins"].items()
                       if focal in rep.ec_classes(r["ecnumber"]) and rep.ec_classes(r["ecnumber"]) != {focal})
        for accession in mixed:  # no target under the single-label rule: units with ec_ambiguous, as in source A
            build = attrition["build"].get(accession) or {}
            care_record = care.get(accession, {})
            rows.append({
                "source": SOURCES[1], "unit_id": accession, "accession": accession, "focal_class": focal,
                "ec_numbers": cp.normalize_ec_number_list(attrition["proteins"][accession]["ecnumber"]),
                "ec_depth1_classes": joined(rep.ec_classes(attrition["proteins"][accession]["ecnumber"])),
                "ec_ambiguous": True, "source_rows": care_record.get("rows", ""), "clusterres30": accession,
                "cluster_representative_present": True, "cluster_has_development_protein": False,
                "relation": "none_known", "annotation_status": "recorded_build",
                "supported_metals": build.get("uniprot_supported_transition_metals", ""),
                "build_status": build.get("status", "NO_BUILD_RECORD"), "built": build.get("status") == "OK",
                "catalytic_prediction": "not evaluated (no target)", "evidence_level": "ambiguous_ec_no_target",
                "eligibility": "ineligible_documented", "attrition_reason": "depth-1 labels disagree",
                "homology_status": HOMOLOGY_UNKNOWN, "heldout_exclusion": NOT_CERTIFIED})
        build_mixed = Counter((attrition["build"].get(a) or {}).get("status", "NO_BUILD_RECORD") for a in mixed)
        summaries[focal] = {**attrition["summary"], "asset_rejections": assets["summary"],
                            "ambiguous_build_outcomes": dict(sorted(build_mixed.items()))}
    prepared = prepared_reconciliation(prepared_dir, development, meta, focal_classes)
    return rows, summaries, prepared, extra_inputs


def prepared_reconciliation(prepared_dir: Path, development: dict[str, Any], meta: Path,
                            focal_classes: tuple[str, ...]) -> dict[str, Any]:
    """Prepared training cohort proteins absent from the development membership, with the training-rule reason."""
    require(training_path(prepared_dir).name == "train", "The prepared cohort folder must be the train/ side")
    sites = read_rows(prepared_dir / PREPARED_FILES["sites"])
    for row in sites:
        require(row.get("source_split") == "train", "The prepared cohort table holds a non-training row")
    manifest = read_rows(prepared_dir / PREPARED_FILES["manifest"])
    proteins = rep.unique_by_accession(rep.training_rows(meta / rep.CARE_METADATA_FILES["train_proteins"],
                                                         split_columns=("split", "source_split")), "train proteins")
    prepared = sorted({row["uniprot_id"].strip() for row in sites})
    absent = []
    for accession in prepared:
        if accession in development["accessions"]:
            continue
        ecs = ec_set(proteins[accession]["ecnumber"])
        absent.append({"accession": accession, "ec_numbers": joined(ecs), "depth1_classes": joined(depth1_classes(ecs)),
                       "catalytic_site_rows": sum(r["uniprot_id"].strip() == accession for r in sites),
                       "reason": "ambiguous_ec_no_target" if depth1(ecs) is None else "review_required"})
    return {"prepared_proteins": len(prepared), "prepared_site_rows": len(sites), "manifest_structures": len(manifest),
            "absent_from_development": absent,
            "absent_with_focal_class": {c: sorted(r["accession"] for r in absent if c in r["depth1_classes"].split(";"))
                                        for c in focal_classes},
            "absent_reasons": dict(sorted(Counter(r["reason"] for r in absent).items()))}


# ---------------------------------------------------------------------------
# Source C: the non-overlap PinMyMetal training side
# ---------------------------------------------------------------------------

def pmm_rows(pmm_dir: Path, care: dict[str, dict[str, Any]], clusters: dict[str, Any], development: dict[str, Any],
             work_root: Path, focal_classes: tuple[str, ...]) -> tuple[list[dict], dict]:
    require(training_path(pmm_dir).name == "train" and "non_overlapped" in pmm_dir.resolve().parent.name.lower(),
            "The PinMyMetal folder must be the train/ side of the non-overlap root")
    sites = read_rows(pmm_dir / PMM_FILES["sites"])
    manifest = {row["structure_name"]: row for row in read_rows(pmm_dir / PMM_FILES["manifest"])}
    sifts = {(row["structure"], row["chain"]): row["UniProt ID"].strip()
             for row in read_rows(pmm_dir / PMM_FILES["sifts"]) if row["Status"] == "ok"}
    native = {(row["structure"], row["chain_resi"]): row["native"] for row in read_rows(pmm_dir / PMM_FILES["native"])}
    chains: dict[tuple[str, str], dict[str, Any]] = {}
    for row in sites:
        require(row["whether_catalytic"] in {"0", "1"}, "Unexpected whether_catalytic value")
        chain = row["chain_resi"].split("_")[0]
        record = chains.setdefault((row["structure"], chain), {"ecs": set(), "sites": [], "native": []})
        record["ecs"] |= ec_set(row["ecnumber"])
        record["sites"].append(row)
        record["native"].append(native.get((row["structure"], row["chain_resi"]), ""))
    mapped = {key: sifts.get(key, "") for key in chains}
    cache = rep.allowlisted_rows(training_path(work_root / rep.ANNOTATION_CACHE), {a for a in mapped.values() if a})
    rows = []
    for (structure, chain), record in sorted(chains.items()):
        accession = mapped[(structure, chain)]
        care_record = care.get(accession) if accession else None
        pdb_classes = depth1_classes(record["ecs"])
        care_classes = care_record["classes"] if care_record else set()
        effective = care_record["token"] if care_record else depth1(record["ecs"])
        focal = [c for c in focal_classes if c in pdb_classes | care_classes]
        if not focal:
            continue
        names = sorted(n for n in manifest if n.startswith(f"{structure}__chain_{chain}__"))
        require(len(names) == 1, f"{structure}:{chain}: expected one manifest structure, found {names}")
        entry = manifest[names[0]]
        path = (pmm_dir / entry["relative_path"]).resolve()
        present = path.is_file() and path.stat().st_size == int(entry["size_bytes"])
        metal = development["metal_sequences"].get(names[0][:-4])
        cached = cache.get(accession)
        supported = set(cp.cached_uniprot_supported_metals({("train", accession): cached} if cached else {},
                                                           split="train", accession=accession) or [])
        site_metals = {cp.canonicalize_metal(s["metaltype"]) for s in record["sites"]}
        if accession and accession in development["accessions"]:
            relation = "in_development_cohort"
        elif (metal and metal["sha256"] in development["ec_sequence_hashes"]) or (
                care_record and care_record["sha256"] in development["ec_sequence_hashes"]):
            relation = "exact_sequence_alias_of_development_protein"
        elif care_record and care_record["cluster"] in clusters["development_clusters"]:
            relation = "same_clusterres30_as_development_protein"
        elif not accession:
            relation = "unmapped_chain"
        else:
            relation = "none_known"
        conditions = {
            "elemental_catalytic_site": "met (MAHOMES-predicted catalytic rows of the training table)",
            "label_source": "unmet: PDB _entity.pdbx_ec, not the CARE/UniProt label",
            "care_label_agrees": ("unknown: accession not in CARE training data" if care_record is None
                                  else "yes" if care_record["token"] == depth1(record["ecs"]) else "no"),
            "supported_metal_annotation": ("unknown: no CARE training annotation row" if not cached
                                           else "compatible" if site_metals & supported else "incompatible"),
            "structure_basis": "unmet: experimental PDB chain, not an AlphaFill-transplanted model",
            "protein_identity": "mapped (SIFTS)" if accession else "unknown: no SIFTS accession in the training table"}
        for focal_class in focal:
            rows.append({
                "source": SOURCES[2], "unit_id": f"{structure}:{chain}", "accession": accession,
                "focal_class": focal_class, "ec_numbers": joined(record["ecs"]),
                "ec_depth1_classes": joined(pdb_classes), "ec_ambiguous": effective is None,
                "ec_ambiguous_note": (f"PDB label {depth1(record['ecs']) or 'ambiguous'}; CARE label "
                                      f"{(care_record['token'] or 'ambiguous') if care_record else 'none'}"),
                "source_rows": len(record["sites"]), "sites": len(record["sites"]), "pockets": "",
                "clusterres30": care_record["cluster"] if care_record else "",
                "cluster_representative_present": bool(care_record and care_record["cluster"] in care),
                "cluster_has_development_protein": bool(care_record and care_record["cluster"]
                                                        in clusters["development_clusters"]),
                "development_component": (development["components"].get(care_record["cluster"], {})
                                          .get("component_id", "") if care_record else ""),
                "relation": relation, "metal_development_member": names[0][:-4] if metal else "",
                "care_label_agrees": conditions["care_label_agrees"],
                "annotation_compatibility": conditions["supported_metal_annotation"],
                "annotation_status": cached["status"] if cached else "NO_TRAINING_CACHE_ROW",
                "supported_metals": joined(supported), "structure_file_present": present,
                "catalytic_prediction": f"{len(record['sites'])} catalytic (MAHOMES table)",
                "elemental_candidates": joined(site_metals), "native_sites": sum(n == "1" for n in record["native"]),
                "evidence_level": "outside_current_definition", "eligibility": "outside_current_definition",
                "attrition_reason": "; ".join(f"{k}: {v}" for k, v in conditions.items()
                                              if v.startswith(("unmet", "unknown", "no", "incompatible"))),
                "missing_evidence": joined(k for k, v in conditions.items() if v.startswith("unknown")),
                "homology_status": relation if relation in POSITIVE else HOMOLOGY_UNKNOWN,
                "heldout_exclusion": "pdb_id_disjoint_from_pmm_test_by_construction (README); otherwise not_certified",
                "conditions": json.dumps(conditions, sort_keys=True)})
    summary = {"site_rows": len(sites), "chains": len(chains), "pdb_ids": len({s for s, _ in chains}),
               "chains_with_sifts_accession": sum(bool(a) for a in mapped.values()),
               "label_ambiguous_chains": sum(depth1(r["ecs"]) is None for r in chains.values()),
               "pdb_label_chains": {c: sum(depth1(r["ecs"]) == c for r in chains.values()) for c in focal_classes},
               "whether_catalytic_values": dict(sorted(Counter(s["whether_catalytic"] for s in sites).items())),
               "unit_definition": "one row per PDB chain and focal class touched by its PDB or CARE label"}
    return rows, summary


# ---------------------------------------------------------------------------
# Counts, attrition and the requested distinctions
# ---------------------------------------------------------------------------

USER_CATEGORIES = {
    "existing_assets_documented_eligible": lambda r: r["eligibility"] in {"eligible_in_development_cohort",
                                                                          "eligible_outside_cohort"},
    "existing_assets_documented_ineligible": lambda r: r["eligibility"] == "ineligible_documented",
    "metadata_only": lambda r: r["evidence_level"] == "metadata_only_supported_metal",
    "partially_prepared_not_built": lambda r: r["evidence_level"].startswith("compatible_candidate_"),
    "known_duplicate_or_related": lambda r: r["relation"] in POSITIVE or bool(r.get("exact_sequence_duplicates")),
    "cluster_representative_absent_heldout_status_unknown": lambda r: (
        r["relation"] == "cluster_representative_absent_from_training_entries"),
    "missing_annotation_structure_or_prediction": lambda r: bool(r.get("missing_evidence")),
    "unresolved_eligibility_homology_or_heldout": lambda r: (
        r["eligibility"] in {"unknown", "outside_current_definition"} or r["homology_status"] == HOMOLOGY_UNKNOWN
        or r["heldout_exclusion"] != "development_member"),
}
UNIT_DEFINITIONS = {
    SOURCES[0]: "one row per CARE accession and focal class its EC numbers touch (ambiguous ones flagged)",
    SOURCES[1]: "one row per clusterRes30 representative and focal class its EC numbers touch (ambiguous ones flagged)",
    SOURCES[2]: "one row per PDB chain and focal class touched by its PDB or CARE label; clusters only for chains "
                "whose SIFTS accession is in CARE training data",
}


def related_clusters(rows: list[dict[str, Any]], clusters: dict[str, Any]) -> set[str]:
    """Clusters with any positive relation to a development protein (whole-cluster rule)."""
    return set(clusters["development_clusters"]) | {r["clusterres30"] for r in rows
                                                    if r["relation"] in POSITIVE and r["clusterres30"]}


def free_clusters(subset: list[dict[str, Any]], related: set[str]) -> set[str]:
    return {r["clusterres30"] for r in subset if r["relation"] in UNRELATED and r["clusterres30"]} - related


def summarize(rows: list[dict[str, Any]], focal_classes: tuple[str, ...], related: set[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for source in SOURCES:
        for focal in focal_classes:
            subset = [r for r in rows if r["source"] == source and r["focal_class"] == focal]
            if not subset:
                continue
            out[f"{source}/EC{focal}"] = {
                "units": len(subset), "unit_definition": UNIT_DEFINITIONS[source],
                "ambiguous_units": sum(bool(r["ec_ambiguous"]) for r in subset),
                "unique_proteins": len({r["accession"] for r in subset if r["accession"]}),
                "source_rows": sum(int(r["source_rows"] or 0) for r in subset),
                "sites_known": sum(int(r["sites"] or 0) for r in subset),
                "development_pockets": sum(int(r["pockets"] or 0) for r in subset),
                "compatible_candidate_upper_bound": sum(int(r["annotation_compatible_candidates"] or 0) for r in subset),
                "clusterres30_clusters": len({r["clusterres30"] for r in subset if r["clusterres30"]}),
                "clusters_without_known_relation": len(free_clusters(subset, related)),
                "eligibility": dict(sorted(Counter(r["eligibility"] for r in subset).items())),
                "evidence_level": dict(sorted(Counter(r["evidence_level"] for r in subset).items())),
                "relation": dict(sorted(Counter(r["relation"] for r in subset).items())),
                "user_categories": {name: sum(rule(r) for r in subset) for name, rule in USER_CATEGORIES.items()}}
    union = {}
    for focal in focal_classes:
        accessions = {r["accession"] for r in rows if r["focal_class"] == focal and r["accession"]}
        union[f"EC{focal}"] = {"unique_accessions_across_sources": len(accessions),
                               "accessions_in_more_than_one_source": sum(bool(r["cross_source_units"]) for r in rows
                                                                         if r["focal_class"] == focal)}
    out["union_across_sources"] = union
    return out


FUNNEL = {
    SOURCES[0]: [("unambiguous depth-1 label", lambda r: not r["ec_ambiguous"]),
                 ("training annotation row OK", lambda r: r["annotation_status"] == "OK"),
                 ("UniProt-supported transition metal", lambda r: bool(r["supported_metals"])),
                 ("local AlphaFill JSON", lambda r: r["alphafill_json"] is True),
                 ("annotation-compatible elemental candidate",
                  lambda r: int(r.get("annotation_compatible_candidates") or 0) > 0),
                 ("local AlphaFill CIF", lambda r: r["alphafill_cif"] is True),
                 ("built structure with a MAHOMES catalytic prediction",
                  lambda r: r.get("built") is True and int(r.get("final_catalytic_sites") or 0) > 0)],
    SOURCES[1]: [("unambiguous depth-1 label", lambda r: not r["ec_ambiguous"]),
                 ("outside the development membership", lambda r: r["relation"] != "in_development_cohort"),
                 ("recorded build OK", lambda r: r["built"] is True),
                 ("at least one predicted catalytic site", lambda r: int(r.get("final_catalytic_sites") or 0) > 0)],
    SOURCES[2]: [("SIFTS accession", lambda r: bool(r["accession"])),
                 ("CARE training label agrees", lambda r: r["care_label_agrees"] == "yes"),
                 ("compatible CARE training annotation", lambda r: r["annotation_compatibility"] == "compatible"),
                 ("no known development relation", lambda r: r["relation"] in UNRELATED),
                 ("AlphaFill-model structure basis (current definition)", lambda r: False)],
}


def attrition(rows: list[dict[str, Any]], focal_classes: tuple[str, ...], related: set[str]) -> list[dict[str, Any]]:
    out = []
    for source, steps in FUNNEL.items():
        for focal in focal_classes:
            remaining = [r for r in rows if r["source"] == source and r["focal_class"] == focal]
            for index, (condition, keep) in enumerate([("all units", lambda r: True)] + steps):
                remaining = [r for r in remaining if keep(r)]
                out.append({"source": source, "focal_class": focal, "step": index, "condition": condition,
                            "units": len(remaining),
                            "clusters": len({r["clusterres30"] for r in remaining if r["clusterres30"]}),
                            "units_without_known_development_relation": sum(r["relation"] in UNRELATED
                                                                            for r in remaining),
                            "clusters_without_known_relation": len(free_clusters(remaining, related))})
    return out


def observed_for_expectations(care: dict, clusters: dict, rows: list[dict], cohort: dict, prepared: dict,
                              pmm: dict, rows_total: int) -> dict[str, Any]:
    def per(source, predicate):
        return {c: sum(1 for r in rows if r["source"] == source and r["focal_class"] == c and predicate(r))
                for c in FOCAL_CLASSES}
    nonrep = SOURCES[0]
    unambiguous = lambda r: not r["ec_ambiguous"]  # noqa: E731
    return {
        "care_full_train_rows": rows_total, "care_full_train_accessions": len(care),
        "clusterres30_representatives": len(clusters["representatives"]),
        "clusterres30_representatives_with_entries": len(clusters["representatives_present"]),
        "nonrep_unambiguous": per(nonrep, unambiguous),
        "nonrep_same_cluster_as_development": per(nonrep, lambda r: unambiguous(r)
                                                  and r["cluster_has_development_protein"]),
        "nonrep_representative_absent": per(nonrep, lambda r: unambiguous(r) and not r["cluster_representative_present"]),
        "nonrep_supported_metal": per(nonrep, lambda r: unambiguous(r) and bool(r["supported_metals"])),
        "nonrep_supported_json": per(nonrep, lambda r: unambiguous(r) and bool(r["supported_metals"])
                                     and r["alphafill_json"] is True),
        "nonrep_supported_cif": per(nonrep, lambda r: unambiguous(r) and bool(r["supported_metals"])
                                    and r["alphafill_cif"] is True),
        "cohort_unambiguous": {c: cohort[c]["focal_unambiguous_training_proteins"] for c in FOCAL_CLASSES},
        "cohort_in_baseline": {c: cohort[c]["focal_in_baseline"] for c in FOCAL_CLASSES},
        "cohort_build_outcomes": {c: cohort[c]["build_outcomes"] for c in FOCAL_CLASSES},
        "cohort_noncatalytic_only_outside": {c: cohort[c]["attrition_reasons"].get(
            "prepared_noncatalytic_prediction_only", 0) for c in FOCAL_CLASSES},
        "prepared_train_proteins": prepared["prepared_proteins"],
        "prepared_absent_from_baseline": len(prepared["absent_from_development"]),
        "prepared_absent_with_focal_class": {c: len(v) for c, v in prepared["absent_with_focal_class"].items()},
        "pmm_site_rows": pmm["site_rows"], "pmm_chains": pmm["chains"], "pmm_pdb_label_chains": pmm["pdb_label_chains"],
    }


# ---------------------------------------------------------------------------
# Later-step lists: downloads and the build's donor metadata
# ---------------------------------------------------------------------------

def later_step_lists(rows: list[dict[str, Any]], related: set[str], json_last: str,
                     rcsb_cached: dict[str, set[str]]) -> tuple[list[dict], list[dict], dict]:
    downloads = []
    for r in rows:
        if r["source"] == SOURCES[0] and r["evidence_level"] in {"metadata_only_supported_metal",
                                                                 "compatible_candidate_cif_missing"}:
            downloads.append({"focal_class": r["focal_class"], "accession": r["accession"], "clusterres30": r["clusterres30"],
                              "relation": r["relation"],
                              "cluster_without_known_relation": r["relation"] in UNRELATED
                              and r["clusterres30"] not in related,
                              "needs": "alphafill_json" if r["evidence_level"].startswith("metadata") else "alphafill_cif",
                              "after_last_local_json_name": r["accession"] > json_last, "reason": r["attrition_reason"]})
        elif r["evidence_level"] == "alphafill_json_download_failed":
            downloads.append({"focal_class": r["focal_class"], "accession": r["accession"], "clusterres30": r["clusterres30"],
                              "relation": r["relation"], "cluster_without_known_relation": r["relation"] in UNRELATED,
                              "needs": "alphafill_json (recorded HTTP 404; re-fetch)", "after_last_local_json_name": "",
                              "reason": "clusterRes30 cohort"})
    donors: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for r in rows:
        if r["evidence_level"] == "compatible_candidate_cif_present_not_built":
            for pdb in filter(None, str(r.get("donor_pdb_ids", "")).split(";")):
                donors[pdb].add((r["focal_class"], r["accession"]))
    donor_rows = [{"donor_pdb_id": pdb, "candidates": len(users),
                   "focal_classes": joined(c for c, _ in users),
                   "cached_unclustered_root": pdb in rcsb_cached["unclustered"],
                   "cached_clusterres30_root": pdb in rcsb_cached["clusterres30"]} for pdb, users in sorted(donors.items())]
    summary = {"donor_entries": len(donor_rows),
               "cached_in_unclustered_root": sum(d["cached_unclustered_root"] for d in donor_rows),
               "cached_only_in_clusterres30_root": sum(d["cached_clusterres30_root"] and not d["cached_unclustered_root"]
                                                       for d in donor_rows),
               "not_cached_locally": sum(not (d["cached_unclustered_root"] or d["cached_clusterres30_root"])
                                         for d in donor_rows)}
    return sorted(downloads, key=lambda d: (FOCAL_CLASSES.index(d["focal_class"]), d["accession"])), donor_rows, summary


def rcsb_listing(work_root: Path) -> set[str]:
    folder = work_root / "rcsb" / "entry"
    return {n[:-5].upper() for n in os.listdir(training_path(folder)) if n.endswith(".json")} if folder.is_dir() else set()


# ---------------------------------------------------------------------------
# Outputs and the command line
# ---------------------------------------------------------------------------

INVENTORY_FIELDS = ["source", "focal_class", "unit_id", "accession", "ec_numbers", "ec_depth1_classes", "ec_ambiguous",
                    "ec_ambiguous_note", "source_rows", "sites", "pockets", "clusterres30",
                    "cluster_representative_present", "cluster_has_development_protein", "development_component",
                    "relation", "exact_sequence_duplicates", "cross_source_units", "metal_development_exact_alias",
                    "metal_development_member", "care_label_agrees", "annotation_compatibility", "annotation_status",
                    "supported_metals", "alphafill_json", "alphafill_cif", "elemental_candidates",
                    "annotation_compatible_candidates", "donor_pdb_ids", "metal_containing_compound_ids",
                    "native_sites", "structure_file_present", "build_status", "built", "final_catalytic_sites",
                    "catalytic_prediction", "evidence_level", "eligibility", "attrition_reason", "missing_evidence",
                    "homology_status", "heldout_exclusion", "conditions"]
ATTRITION_FIELDS = ["source", "focal_class", "step", "condition", "units", "clusters",
                    "units_without_known_development_relation", "clusters_without_known_relation"]
JSON_INPUT_FIELDS = ["accession", "path", "sha256", "bytes", "elemental_candidates", "annotation_compatible"]
DOWNLOAD_FIELDS = ["focal_class", "accession", "clusterres30", "relation", "cluster_without_known_relation", "needs",
                   "after_last_local_json_name", "reason"]
DONOR_FIELDS = ["donor_pdb_id", "candidates", "focal_classes", "cached_unclustered_root", "cached_clusterres30_root"]
ASSET_FIELDS = ["name", "path", "sha256", "bytes"]


def input_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {"baseline_split": args.baseline_split, "care_full_train": args.care_full_train,
             "care_full_annotation_cache": args.care_full_work_root / rep.ANNOTATION_CACHE,
             "care_full_build_parameters": args.care_full_work_root / UNCLUSTERED_BUILD["parameters"],
             "clusterres30_annotation_cache": args.care_clusterres30_work_root / rep.ANNOTATION_CACHE,
             "remoteness/sequence_manifest": args.remoteness_root / rep.REMOTENESS_FILES["sequence_manifest"],
             "previous_report/artifact_manifest": args.previous_report / "artifact_manifest.json",
             "previous_report/proteins": args.previous_report / "proteins.csv"}
    build_summary = args.care_full_work_root / UNCLUSTERED_BUILD["train_build_summary"]
    if build_summary.is_file():
        paths["care_full_train_build_summary"] = build_summary
    paths.update({f"care_metadata/{k}": args.care_metadata_root / v for k, v in rep.CARE_METADATA_FILES.items()})
    paths.update({f"prepared/{k}": args.care_prepared_train_dir / v for k, v in PREPARED_FILES.items()})
    paths.update({f"pmm/{k}": args.pmm_train_dir / v for k, v in PMM_FILES.items()})
    return paths


def build(args: argparse.Namespace) -> dict[str, Any]:
    focal_classes = tuple(args.focal_classes)
    require(set(focal_classes) <= set(FOCAL_CLASSES) and focal_classes, "Focal classes must be among 7, 6 and 5")
    development = load_development(args.baseline_split, args.remoteness_root, args.previous_report)
    loaded = load_care_full_train(args.care_full_train)
    care = loaded["proteins"]
    clusters = cluster_index(care, development["accessions"])
    policy = rep.alphafill_policy(args.care_metadata_root)
    unclustered_policy = recorded_policy_matches(args.care_metadata_root, args.care_full_work_root, policy)
    a_rows, a_assets, json_inputs = nonrepresentative_rows(care, clusters, development, args.care_full_work_root,
                                                           policy, focal_classes)
    b_rows, cohort, prepared, b_inputs = cohort_rows(args.care_metadata_root, args.care_clusterres30_work_root,
                                                     args.care_prepared_train_dir, development, care, policy,
                                                     focal_classes)
    c_rows, pmm = pmm_rows(args.pmm_train_dir, care, clusters, development, args.care_full_work_root, focal_classes)
    rows = sorted(a_rows + b_rows + c_rows, key=lambda r: (FOCAL_CLASSES.index(r["focal_class"]),
                                                           SOURCES.index(r["source"]), r["unit_id"]))
    sources_by_accession: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for field in INVENTORY_FIELDS:
            row.setdefault(field, "")
        if row["accession"]:
            sources_by_accession[row["accession"]].add(row["source"])
    for row in rows:
        row["cross_source_units"] = joined(sources_by_accession.get(row["accession"], set()) - {row["source"]})
    related = related_clusters(rows, clusters)
    complete = set(focal_classes) == set(FOCAL_CLASSES)
    full = observed_for_expectations(care, clusters, rows, cohort, prepared, pmm, loaded["rows"]) if complete else {}
    discrepancies = [{"item": k, "expected": v, "observed": full.get(k)}
                     for k, v in MAPPING_EXPECTATIONS.items() if complete and full.get(k) != v]
    development_structure = {}
    for focal in focal_classes:
        members = [a for a, row in development["components"].items() if str(row.get("ec_class")) == focal]
        development_structure[focal] = {
            "proteins": len(members), "clusterres30_clusters": len({care[a]["cluster"] for a in members if a in care}),
            "dependence_components": len({development["components"][a]["component_id"] for a in members})}
    rcsb = {"unclustered": rcsb_listing(args.care_full_work_root),
            "clusterres30": rcsb_listing(args.care_clusterres30_work_root)}
    json_last = (a_assets["json_name_range"] or ["", ""])[-1]
    downloads, donors, donor_summary = later_step_lists(rows, related, json_last, rcsb)
    absent_representatives = len(clusters["representatives"] - clusters["representatives_present"])
    seven = development_structure.get("7", {})
    affected = Counter(r["evidence_level"] for r in rows
                       if r["relation"] == "cluster_representative_absent_from_training_entries")
    results = {
        "counts": summarize(rows, focal_classes, related), "cohort_attrition": cohort,
        "prepared_reconciliation": prepared, "nonrepresentative_assets": a_assets, "pmm": pmm,
        "unclustered_policy": unclustered_policy, "development_structure": development_structure,
        "care_full_train": {"rows": loaded["rows"], "accessions": len(care),
                            "clusterres30_representatives": len(clusters["representatives"]),
                            "representatives_with_entries": len(clusters["representatives_present"]),
                            "exact_sequence_duplicate_groups": sorted(sorted(a) for a in
                                                                      clusters["duplicates"].values())},
        "eligible_outside_development": sum(r["eligibility"] == "eligible_outside_cohort" for r in rows),
        "build_donor_metadata": {**donor_summary, "rcsb_listing": {k: listing_digest(v) for k, v in rcsb.items()}},
        "absent_representatives": {"count": absent_representatives,
                                   "record": "CARE_prepare_training_and_test_set/provenance/clusterRes30/metadata/"
                                             "care_task1_30_audit.json lists them (not opened here: it also holds "
                                             "test-side lists)",
                                   "affected_candidates_by_evidence_level": dict(sorted(affected.items()))},
        "user_category_definitions": {
            "existing_assets_documented_eligible": "eligibility established by recorded build/prediction evidence",
            "existing_assets_documented_ineligible": "an existing record (label rule, annotation, AlphaFill JSON, "
                                                     "build or prediction) fails the unchanged definition",
            "metadata_only": "label and supported-metal annotation only; no local AlphaFill asset",
            "partially_prepared_not_built": "local AlphaFill JSON with an annotation-compatible candidate; no build "
                                            "or MAHOMES prediction",
            "known_duplicate_or_related": "development member, exact sequence alias, same CARE clusterRes30 as a "
                                          "development protein, or an exact duplicate of another candidate (positive "
                                          "evidence only)",
            "cluster_representative_absent_heldout_status_unknown": "cluster representative absent from the training "
                                                                    "entries; its held-out status is unknown",
            "missing_annotation_structure_or_prediction": "any of annotation, AlphaFill JSON/CIF, build, catalytic "
                                                          "prediction, accession mapping or CARE label missing",
            "unresolved_eligibility_homology_or_heldout": "eligibility unknown or outside the definition, homology "
                                                          "to the development set unknown, or held-out exclusion "
                                                          "not certified"},
        "categories_overlap": True,
        "relationship_evidence": ("exact sequences, CARE clusterRes30 co-membership and the saved development "
                                  "sequences only; no alignment search. Separate clusters do not prove remoteness: "
                                  f"the {seven.get('proteins')} development EC7 proteins lie in "
                                  f"{seven.get('clusterres30_clusters')} clusterRes30 clusters but "
                                  f"{seven.get('dependence_components')} dependence component(s)"),
        "heldout_evidence": ("none certifies these candidates. CARE's 30%-identity test separation is described "
                             "locally (docs/DATASETS.md) but has no upstream citation and was not verified here; "
                             f"{absent_representatives} clusterRes30 representatives are absent from the training "
                             "entries (listed in the tracked provenance audit; reason not established here); the "
                             "non-overlap PinMyMetal training side is PDB-ID disjoint from its own test by "
                             "construction (README), not by a sequence check"),
    }
    return {"rows": rows, "attrition": attrition(rows, focal_classes, related), "json_inputs": json_inputs,
            "related_clusters": related,
            "extra_inputs": b_inputs, "downloads": downloads, "donors": donors, "results": results,
            "expectation_check": {"source": "independent read-only source mapping (2026-10-13), compared only",
                                  "checked": complete, "expected": MAPPING_EXPECTATIONS, "observed": full,
                                  "discrepancies": discrepancies, "all_match": not discrepancies}}


def class_digest(rows: list[dict[str, Any]], focal: str, related: set[str], json_last: str) -> dict[str, Any]:
    """The numbers the recommendation states for one class (all computed from the inventory rows)."""
    def pick(source, predicate):
        return [r for r in rows if r["source"] == source and r["focal_class"] == focal and predicate(r)]

    def units_clusters(subset):
        free = [r for r in subset if r["relation"] in UNRELATED and r["clusterres30"] not in related]
        return {"units": len(subset), "clusters": len({r["clusterres30"] for r in subset if r["clusterres30"]}),
                "units_without_known_relation": len(free), "clusters_without_known_relation": sorted(
                    {r["clusterres30"] for r in free if r["clusterres30"]}),
                "units_without_known_relation_by_cluster": dict(sorted(Counter(
                    r["clusterres30"] for r in free).items())),
                "units_after_last_local_json_name": sum(r["accession"] > json_last for r in free),
                "compatible_candidate_upper_bound": sum(int(r["annotation_compatible_candidates"] or 0) for r in subset),
                "representative_absent_units": sum(r["relation"] == "cluster_representative_absent_from_training_entries"
                                                   for r in subset)}
    built = pick(SOURCES[1], lambda r: r["built"] is True and not r["ec_ambiguous"])
    catalytic = [r for r in built if int(r["final_catalytic_sites"] or 0) > 0]
    pmm = pick(SOURCES[2], lambda r: True)
    return {
        "development_proteins": len(pick(SOURCES[1], lambda r: r["eligibility"] == "eligible_in_development_cohort")),
        "documented_eligible_outside": len([r for r in rows if r["focal_class"] == focal
                                            and r["eligibility"] == "eligible_outside_cohort"]),
        "partially_prepared": units_clusters(pick(SOURCES[0], lambda r: r["evidence_level"]
                                                  == "compatible_candidate_cif_present_not_built")),
        "cif_missing": units_clusters(pick(SOURCES[0], lambda r: r["evidence_level"] == "compatible_candidate_cif_missing")),
        "metadata_only": units_clusters(pick(SOURCES[0], lambda r: r["evidence_level"] == "metadata_only_supported_metal")),
        "cohort_json_download_failed": sorted(r["accession"] for r in pick(
            SOURCES[1], lambda r: r["evidence_level"] == "alphafill_json_download_failed")),
        "cohort_built": len(built), "cohort_built_catalytic": len(catalytic),
        "cohort_built_catalytic_all_in_development": all(r["relation"] == "in_development_cohort" for r in catalytic),
        "pmm_units": len(pmm), "pmm_related": sum(r["relation"] in POSITIVE for r in pmm),
        "pmm_unknown_annotation": sorted({r["accession"] for r in pmm if r["relation"] == "none_known"
                                          and r["annotation_compatibility"].startswith("unknown")}),
        "pmm_incompatible_annotation": sorted({r["accession"] for r in pmm if r["relation"] == "none_known"
                                               and r["annotation_compatibility"] == "incompatible"}),
        "pmm_compatible_without_relation": sum(r["relation"] == "none_known"
                                               and r["annotation_compatibility"] == "compatible" for r in pmm),
        "pmm_unmapped_chains": sum(r["relation"] == "unmapped_chain" for r in pmm)}


def render_recommendation(report: dict[str, Any], focal_classes) -> str:
    rows, results = report["rows"], report["results"]
    json_last = (results["nonrepresentative_assets"]["json_name_range"] or ["", ""])[-1]
    digests = {c: class_digest(rows, c, report["related_clusters"], json_last) for c in focal_classes}
    prepared = results["prepared_reconciliation"]
    donors = results["build_donor_metadata"]
    lines = ["# Recommendation: EC7, EC6 and EC5 support from local training sources", "",
             "Read-only inventory under the unchanged v12 EC1 eligibility. Units are CARE accessions or PinMyMetal "
             "chains; clusters are CARE clusterRes30 clusters. Catalytic labels are MAHOMES predictions. \"Without a "
             "known relation\" means no member of the cluster has an exact-sequence or same-cluster link to a "
             "development protein; it is not evidence of independence.", "",
             "| Class | Development proteins | Eligible now outside development | Partially prepared, not built: "
             "units / clusters / clusters without a known relation | Supported metal, no local AlphaFill: "
             "units / clusters / clusters without a known relation |", "|---|---:|---:|---|---|"]
    for c, d in digests.items():
        p, m = d["partially_prepared"], d["metadata_only"]
        lines.append(f"| EC{c} | {d['development_proteins']} | {d['documented_eligible_outside']} | "
                     f"{p['units']} / {p['clusters']} / {len(p['clusters_without_known_relation'])} | "
                     f"{m['units']} / {m['clusters']} / {len(m['clusters_without_known_relation'])} |")
    lines += ["", "## Already usable", ""]
    if results["eligible_outside_development"] == 0:
        lines.append("Nothing beyond the current development membership: no EC5, EC6 or EC7 protein outside it has "
                     "documented eligibility in any of the three sources.")
    else:
        lines.append(f"{results['eligible_outside_development']} units outside the development membership have "
                     "documented eligibility (see candidate_inventory.csv, eligibility eligible_outside_cohort).")
    reasons = prepared["absent_reasons"]
    if set(reasons) <= {"ambiguous_ec_no_target"}:
        lines.append(f"The prepared training cohort adds {len(prepared['absent_from_development'])} proteins, all with EC "
                     "numbers spanning several first digits, which the single-label rule gives no target "
                     f"(EC5-containing: {', '.join(prepared['absent_with_focal_class'].get('5', [])) or 'none'}).")
    else:
        lines.append(f"The prepared training cohort adds proteins absent from the membership: {reasons}.")
    lines += ["", "## Uncertain", ""]
    for c, d in digests.items():
        p = d["partially_prepared"]
        clusters = ", ".join(f"{k} ({v})" for k, v in p["units_without_known_relation_by_cluster"].items())
        lines.append(
            f"- **EC{c}.** {p['units']} non-representatives have a local AlphaFill JSON with an annotation-compatible "
            f"elemental candidate and a local CIF, but no build or MAHOMES prediction; {p['units_without_known_relation']} "
            f"of them lie in {len(p['clusters_without_known_relation'])} clusters without a known development relation"
            + (f" ({clusters})" if c == "7" and clusters else "") + ". Candidate sites are unknown until a build "
            f"(at most {p['compatible_candidate_upper_bound']} annotation-compatible transplant candidates before "
            f"1.0 Å de-duplication). In the clusterRes30 cohort {d['cohort_built_catalytic']} of {d['cohort_built']} "
            f"built EC{c} proteins had a predicted catalytic site"
            + (", all of them development members, so this is a property of that selection, not a yield estimate."
               if d["cohort_built_catalytic_all_in_development"] else "."))
    tail = results["relationship_evidence"].split("no alignment search. ", 1)[-1]
    absent = results["absent_representatives"]
    lines += ["- **Homology.** No development-versus-candidate search exists; " + tail[:1].lower() + tail[1:] + ".",
              "- **Held-out exclusion.** " + results["heldout_evidence"][:1].upper() + results["heldout_evidence"][1:]
              + f". Candidates in clusters whose representative is absent: "
              + (", ".join(f"{k} {v}" for k, v in absent["affected_candidates_by_evidence_level"].items()) or "none")
              + ".",
              "- **PinMyMetal.** Outside the current definition (experimental chains, PDB EC labels). "
              + "; ".join(f"EC{c}: {d['pmm_units']} chains, {d['pmm_related']} in or beside the development membership, "
                          f"unknown annotation outside CARE training data "
                          f"{', '.join(d['pmm_unknown_annotation']) or 'none'}, incompatible annotation "
                          f"{', '.join(d['pmm_incompatible_annotation']) or 'none'}, {d['pmm_unmapped_chains']} chains "
                          "without a SIFTS accession" for c, d in digests.items()) + ".",
              "", "## Worthwhile later steps (each needs its own authorization)", "",
              "1. **Training-side relationship search (CPU, no download).** The frozen MMseqs2 protocol, all-vs-all "
              "over the EC development set plus the candidates (partially prepared first, then supported-metal ones), "
              "with components over the union so candidate-candidate dependence is counted. It detects relations "
              "under the frozen rule; no detected relation is not proof of independence. EC7 first.",
              "2. **Label-blind held-out exclusion check.** Identifiers and exact sequences of the surviving candidates "
              "against the CARE test, CLEAN30 folds and PinMyMetal/Zenodo tests, plus the status of the absent "
              "representatives. It reads test data, so it must be authorized and disclosed; running it before builds "
              "and downloads avoids spending them on excluded candidates.",
              "3. **Training-only build and MAHOMES of the existing assets (CPU).** Not download-free as the pipeline "
              "stands: the build has no accession allowlist (a restricted training-only manifest is needed), must run "
              "with `--splits train --no-fetch-missing-uniprot`, and looks up donor PDB resolutions through RCSB entry "
              f"metadata. The partially prepared candidates use {donors['donor_entries']} donor entries: "
              f"{donors['cached_in_unclustered_root']} cached in the unclustered root, "
              f"{donors['cached_only_in_clusterres30_root']} only in the clusterRes30 root (copyable), "
              f"{donors['not_cached_locally']} not cached (an RCSB download) (build_donor_entries.csv).",
              "4. **Targeted AlphaFill downloads.** JSON first, CIF only where a compatible candidate exists, for "
              "supported-metal candidates without a known relation ("
              + "; ".join(f"EC{c}: {d['metadata_only']['units_without_known_relation']} accessions in "
                          f"{len(d['metadata_only']['clusters_without_known_relation'])} clusters, "
                          f"{d['metadata_only']['units_after_last_local_json_name']} after the last local JSON name"
                          for c, d in digests.items())
              + "), and a re-fetch of the recorded HTTP-404 JSONs ("
              + (", ".join(a for d in digests.values() for a in d["cohort_json_download_failed"]) or "none")
              + "). The exact list is download_candidates.csv; the whole supported-metal pool is larger ("
              + ", ".join(f"EC{c} {d['metadata_only']['units']}" for c, d in digests.items()) + ") before the search.",
              "5. **Only then** a separately versioned cohort decision and the EC frozen-membership training interface.",
              "", "PinMyMetal chains are outside the current definition (their EC labels and structure basis differ "
              "from the v12 cohort). "
              + ("None has a compatible CARE training annotation without a known development relation; the remaining "
                 "ones are unresolved (unknown annotation outside CARE training data, an incompatible annotation, or no "
                 "SIFTS accession), so they are not a route for this cohort."
                 if not any(d["pmm_compatible_without_relation"] for d in digests.values()) else
                 "Chains with a compatible CARE training annotation and no known development relation: "
                 + ", ".join(f"EC{c} {d['pmm_compatible_without_relation']}" for c, d in digests.items())
                 + "; they would need a separately versioned cohort decision."), ""]
    return "\n".join(lines)


def render_report(report: dict[str, Any], previous_sha: str) -> str:
    counts = report["results"]["counts"]
    lines = ["# EC7/EC6/EC5 training-source inventory", "",
             f"Campaign `{CAMPAIGN_ID}`; CPU, read-only. Expectation check: "
             f"{'all match' if report['expectation_check']['all_match'] else 'DISCREPANCIES (see inventory.json)'}. "
             f"Documented eligible outside the development membership: "
             f"{report['results']['eligible_outside_development']}.", "",
             "| Source / class | Units (ambiguous) | Proteins | Source rows | Known sites | Development pockets | "
             "Clusters | Clusters without a known relation | Eligibility |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for key, value in counts.items():
        if key == "union_across_sources":
            continue
        lines.append(f"| {key} | {value['units']} ({value['ambiguous_units']}) | {value['unique_proteins']} | "
                     f"{value['source_rows']} | {value['sites_known']} | {value['development_pockets']} | "
                     f"{value['clusterres30_clusters']} | {value['clusters_without_known_relation']} | "
                     f"{', '.join(f'{k} {v}' for k, v in value['eligibility'].items())} |")
    lines += ["", "Unit definitions: " + "; ".join(f"{k}: {v}" for k, v in UNIT_DEFINITIONS.items()) + ".",
              "Across sources: " + "; ".join(f"{k}: {v['unique_accessions_across_sources']} unique accessions, "
                                             f"{v['accessions_in_more_than_one_source']} rows shared with another source"
                                             for k, v in counts["union_across_sources"].items()) + ".", "",
              "| Source / class | " + " | ".join(USER_CATEGORIES) + " |",
              "|---|" + "---:|" * len(USER_CATEGORIES)]
    for key, value in counts.items():
        if key != "union_across_sources":
            lines.append(f"| {key} | " + " | ".join(str(value["user_categories"][n]) for n in USER_CATEGORIES) + " |")
    lines += ["", "Categories overlap (definitions in inventory.json).", "",
              "Attrition (units / clusters / units without a known relation / clusters without a known relation):", ""]
    for row in report["attrition"]:
        lines.append(f"- {row['source']} EC{row['focal_class']} step {row['step']} ({row['condition']}): "
                     f"{row['units']} / {row['clusters']} / {row['units_without_known_development_relation']} / "
                     f"{row['clusters_without_known_relation']}")
    lines += ["", "Relationship evidence: " + report["results"]["relationship_evidence"] + ".",
              "Held-out exclusion: " + report["results"]["heldout_evidence"] + ".", "",
              f"Bound by `artifact_manifest.json`. Previous report manifest: `{previous_sha[:8]}…`.", ""]
    return "\n".join(lines)


def git_state() -> dict[str, Any]:
    def git(*command):
        return subprocess.run(["git", "-C", str(ROOT), *command], capture_output=True, text=True).stdout.strip()
    return {"head": git("rev-parse", "HEAD"),
            "tool_status": git("status", "--porcelain", "--", str(Path(__file__).resolve().relative_to(ROOT)))}


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = input_paths(args)
    for path in paths.values():
        training_path(path)
    roots = [args.care_full_train.parent, args.care_full_work_root, args.care_clusterres30_work_root,
             args.care_metadata_root, args.care_prepared_train_dir, args.pmm_train_dir, args.remoteness_root,
             args.previous_report, args.baseline_split.parent]
    check_output_dir(args.output_dir, roots)
    before = {name: file_record(path) for name, path in sorted(paths.items())}
    report = build(args)
    extra = {f"alphafill_json/{r['accession']}": {k: r[k] for k in ("path", "sha256", "bytes")}
             for r in report["json_inputs"]}
    extra.update(report["extra_inputs"])
    after = {name: file_record(path) for name, path in sorted(paths.items())}
    changed = [n for n in before if before[n]["sha256"] != after[n]["sha256"]]
    changed += [n for n, rec in extra.items() if asr.sha256(Path(rec["path"])) != rec["sha256"]]
    require(not changed, f"Inputs changed during the run: {changed}")
    sources = {name: file_record(path) for name, path in (
        ("inventory_tool", Path(__file__).resolve()), ("report_ec1_component_support", Path(rep.__file__).resolve()),
        ("audit_sequence_remoteness", Path(asr.__file__).resolve()), ("care_prepare", Path(cp.__file__).resolve()),
        ("training_labels", Path(labels.__file__).resolve()))}
    boundary = {"training": False, "inference": False, "download": False, "alignment_search": False,
                "build_or_mahomes": False, "split_writing": False, "cohort_merging": False,
                "eligibility_change": False, "held_out_files_opened": False, "test_side_rows_used": False,
                "mixed_split_files_hashed_whole": [str(paths[n]) for n in MIXED_SPLIT_INPUTS],
                "mixed_split_rows_parsed_and_discarded_by_split_filter": [str(paths[n]) for n in MIXED_SPLIT_INPUTS]}
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    inventory = {"schema_version": 1, "report": "ec_support_source_inventory", "campaign_id": CAMPAIGN_ID,
                 "created_at": created, "focal_classes": list(args.focal_classes), "boundary": boundary,
                 "results": report["results"], "expectation_check": report["expectation_check"],
                 "inputs": before, "alphafill_json_inputs": len(report["json_inputs"]),
                 "reused_asset_inputs": len(report["extra_inputs"]), "sources": sources}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir
    files = {
        "candidate_inventory.csv": csv_text(report["rows"], INVENTORY_FIELDS),
        "attrition.csv": csv_text(report["attrition"], ATTRITION_FIELDS),
        "download_candidates.csv": csv_text(report["downloads"], DOWNLOAD_FIELDS),
        "build_donor_entries.csv": csv_text(report["donors"], DONOR_FIELDS),
        "alphafill_json_inputs.csv": csv_text(sorted(report["json_inputs"], key=lambda r: r["accession"]),
                                              JSON_INPUT_FIELDS),
        "reused_asset_inputs.csv": csv_text([{"name": n, **r} for n, r in sorted(report["extra_inputs"].items())],
                                            ASSET_FIELDS),
        "inventory.json": json.dumps(inventory, indent=2, sort_keys=True, default=sorted) + "\n",
        "report.md": render_report(report, inventory["inputs"]["previous_report/artifact_manifest"]["sha256"]),
        "recommendation.md": render_recommendation(report, tuple(args.focal_classes)),
    }
    for name, content in files.items():
        write_once(out / name, content)
    manifest = {"schema_version": 1, "campaign_id": CAMPAIGN_ID, "created_at": created,
                "command": [sys.executable, *sys.argv], "working_directory": os.getcwd(), "git": git_state(),
                "python": platform.python_version(), "boundary": boundary, "inputs_before": before,
                "inputs_unchanged_after_run": True, "alphafill_json_and_reused_inputs_unchanged": True,
                "listing_digests": {"alphafill_json": report["results"]["nonrepresentative_assets"]["json_listing"],
                                    "alphafill_cif": report["results"]["nonrepresentative_assets"]["cif_listing"],
                                    "rcsb_entries": report["results"]["build_donor_metadata"]["rcsb_listing"]},
                "sources": sources, "outputs": {name: file_record(out / name) for name in sorted(files)},
                "expectation_all_match": report["expectation_check"]["all_match"]}
    write_once(out / "artifact_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"output_dir": str(out), "artifact_manifest_sha256": asr.sha256(out / "artifact_manifest.json"),
            "expectation_all_match": report["expectation_check"]["all_match"],
            "discrepancies": report["expectation_check"]["discrepancies"],
            "eligible_outside_development": report["results"]["eligible_outside_development"]}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-split", type=Path, required=True)
    parser.add_argument("--remoteness-root", type=Path, required=True)
    parser.add_argument("--previous-report", type=Path, required=True, help="the ec1s-001 report folder")
    parser.add_argument("--care-full-train", type=Path, required=True, help="CARE Task-1 protein_train.csv")
    parser.add_argument("--care-full-work-root", type=Path, required=True, help="the unclustered CARE work root")
    parser.add_argument("--care-metadata-root", type=Path, required=True, help="clusterRes30 provenance metadata")
    parser.add_argument("--care-clusterres30-work-root", type=Path, required=True)
    parser.add_argument("--care-prepared-train-dir", type=Path, required=True, help="prepared cohort train/ folder")
    parser.add_argument("--pmm-train-dir", type=Path, required=True, help="non-overlap PinMyMetal train/ folder")
    parser.add_argument("--focal-classes", nargs="+", default=list(FOCAL_CLASSES))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    try:
        print(json.dumps(run(parse_args(argv)), indent=2))
    except (ReportError, ValueError, KeyError, OSError) as exc:
        print(f"EC support-source inventory refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
