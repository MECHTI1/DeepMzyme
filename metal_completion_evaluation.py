"""Prepare a temporal metal-evaluation query offline; never retrieve or score data."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCHEMA = "metal-completion-evaluation-provenance-v1"
REQUIRED_SOURCES = (
    "pinmymetal-source", "pinmymetal-zenodo-exact", "pinmymetal-exact",
    "pinmymetal-nonoverlap", "pinmymetal-harsh", "pinmymetal-common70",
)
# Explicit free-ion CCD codes; cofactors/metal-containing compounds are excluded.
ION_COMPONENTS = {
    "MN": "Mn", "MN3": "Mn", "CU": "Cu", "CU1": "Cu", "CU3": "Cu",
    "ZN": "Zn", "FE": "Fe", "FE2": "Fe", "CO": "Co", "3CO": "Co",
    "NI": "Ni", "3NI": "Ni",
}
OBJECTIVES = ["four_class", "five_class", "six_class"]
COVERAGE_FLAGS = (
    "all_supervised_development_sources_included",
    "all_previously_opened_test_ids_included",
    "all_custom_split_variants_accounted_for",
)


class PreparationError(ValueError):
    """Missing provenance or a conflicting output prevents preparation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreparationError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_value(value: object, label: str) -> str:
    require(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None,
            f"{label}: a recorded SHA-256 is required")
    return value


def iso_date(value: object, label: str) -> date:
    require(isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is not None,
            f"{label}: verified initial release date YYYY-MM-DD is required")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise PreparationError(f"{label}: invalid date") from exc


def pdb_id(value: object) -> str:
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9]{4}|pdb_[A-Za-z0-9]{8}", value) is not None,
            "Invalid PDB ID in metadata-only provenance")
    return value.upper()


def provenance_template() -> dict:
    """Deliberately incomplete; no dates, source-use claims or coverage are invented."""
    return {
        "schema": SCHEMA,
        "metadata_only": True,
        "verification": {
            "status": "unverified", "review_reference": None, "evidence_sha256": None,
            **{flag: False for flag in COVERAGE_FLAGS},
        },
        "sources": [{"source_id": source, "status": "unverified", "use_status": "unknown",
                     "evidence_sha256": None, "pdb_ids": [], "reason": None}
                    for source in REQUIRED_SOURCES],
        "entries": [],
    }


def validate_provenance(manifest: dict, through_date: str) -> tuple[str, list[str]]:
    require(isinstance(manifest, dict) and manifest.get("schema") == SCHEMA,
            "Unsupported provenance schema")
    require(manifest.get("metadata_only") is True, "Only explicitly metadata-only provenance is accepted")
    verification = manifest.get("verification", {})
    require(isinstance(verification, dict) and verification.get("status") == "verified",
            "Provenance must be reviewed and verified; template is intentionally blocked")
    require(isinstance(verification.get("review_reference"), str) and bool(verification["review_reference"].strip()),
            "A provenance review reference is required")
    sha256_value(verification.get("evidence_sha256"), "coverage evidence")
    require(all(verification.get(flag) is True for flag in COVERAGE_FLAGS),
            "All development, opened-test and custom-variant coverage must be verified")

    sources = manifest.get("sources")
    require(isinstance(sources, list) and bool(sources), "Source accounting is required")
    seen_sources: set[str] = set()
    excluded: set[str] = set()
    for source in sources:
        require(isinstance(source, dict), "Malformed source record")
        name = source.get("source_id")
        require(isinstance(name, str) and bool(name.strip()) and name not in seen_sources,
                "Source IDs must be nonempty and unique")
        seen_sources.add(name)
        require(source.get("status") == "verified", f"Unverified source: {name}")
        sha256_value(source.get("evidence_sha256"), f"source {name}")
        ids = source.get("pdb_ids")
        require(isinstance(ids, list), f"source {name}: pdb_ids must be an explicit list")
        if source.get("use_status") == "not_used":
            require(not ids and isinstance(source.get("reason"), str) and bool(source["reason"].strip()),
                    f"source {name}: not_used requires no IDs and an evidence-backed reason")
        else:
            require(source.get("use_status") == "used" and bool(ids), f"source {name}: use is unresolved")
            excluded.update(pdb_id(value) for value in ids)
    require(set(REQUIRED_SOURCES) <= seen_sources, "Required historical source families are missing")
    require(bool(excluded), "No verified previously used PDB IDs were supplied")

    entries = manifest.get("entries")
    require(isinstance(entries, list), "Entry release-date metadata is required")
    releases: dict[str, date] = {}
    for entry in entries:
        require(isinstance(entry, dict) and set(entry) == {"pdb_id", "initial_release_date", "evidence_sha256"},
                "Entry records accept only PDB ID, initial release date and evidence hash; no labels or scores")
        identifier = pdb_id(entry["pdb_id"])
        require(identifier not in releases, f"Duplicate release-date record: {identifier}")
        sha256_value(entry["evidence_sha256"], f"release-date evidence for {identifier}")
        releases[identifier] = iso_date(entry["initial_release_date"], identifier)
    require(set(releases) == excluded, "Release dates must cover exactly every known/excluded PDB ID")
    cutoff = max(releases.values())
    require(cutoff < iso_date(through_date, "query upper date"),
            "Query upper date must be later than every prior source entry; no nonempty temporal window")
    return cutoff.isoformat(), sorted(excluded)


def query_payload(cutoff: str, through_date: str, excluded_ids: list[str]) -> dict:
    def clause(attribute: str, operator: str, value: object, *, negation: bool = False) -> dict:
        return {"type": "terminal", "service": "text", "parameters": {
            "attribute": attribute, "operator": operator, "value": value, "negation": negation}}

    return {
        "query": {"type": "group", "logical_operator": "and", "nodes": [
            clause("rcsb_accession_info.initial_release_date", "greater", cutoff),
            clause("rcsb_accession_info.initial_release_date", "less_or_equal", through_date),
            clause("rcsb_entry_info.structure_determination_methodology", "exact_match", "experimental"),
            clause("rcsb_entry_info.polymer_entity_count_protein", "greater", 0),
            clause("pdbx_entity_nonpoly.comp_id", "in", sorted(ION_COMPONENTS)),
            clause("rcsb_entry_container_identifiers.entry_id", "in", excluded_ids, negation=True),
        ]},
        "return_type": "entry",
        "request_options": {"results_content_type": ["experimental"], "return_all_hits": True},
    }


def prepare(manifest_path: Path, expected_sha256: str, through_date: str, output: Path) -> dict:
    # This is the sole external input read. Evidence hashes are attestations, not paths to follow.
    raw = manifest_path.read_bytes()
    require(digest(raw) == sha256_value(expected_sha256, "manifest pin"), "Provenance manifest SHA-256 mismatch")
    try:
        manifest = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise PreparationError("Invalid provenance JSON") from exc
    cutoff, excluded = validate_provenance(manifest, through_date)
    query = encoded(query_payload(cutoff, through_date, excluded))
    protocol = ROOT / "docs/campaigns/metal_completion_2026-10/evaluation.md"
    receipt = {
        "schema": "metal-completion-evaluation-query-v1", "status": "offline_query_prepared",
        "provenance_sha256": digest(raw), "tool_sha256": digest(Path(__file__).read_bytes()),
        "protocol_sha256": digest(protocol.read_bytes()), "query_sha256": digest(query),
        "cutoff_exclusive": cutoff, "through_date_inclusive": through_date,
        "excluded_pdb_count": len(excluded), "objectives": OBJECTIVES,
        "elemental_components": ION_COMPONENTS, "reporting_classes": ["Mn", "Cu", "Zn", "Fe+Co+Ni"],
        "candidate_query_executed": False, "downloads_authorized": False,
        "heldout_access_authorized": False, "evaluation_authorized": False,
        "provenance_limit": "Hashes bind the supplied review attestations; this tool does not independently certify them.",
        "remaining_gates": ["review source coverage and protocol", "authorize candidate metadata retrieval",
                            "verify current API request limits", "fix quality and ambiguity rules before structures",
                            "certify cross-source sequence exclusions", "preregister support/precision requirements",
                            "freeze model/refit and reporting before scoring"],
    }
    require(not output.exists(), "Output already exists; preserve previous query freezes")
    output.mkdir(parents=True, exist_ok=False)
    (output / "query.json").write_bytes(query)
    (output / "provenance.json").write_bytes(raw)
    (output / "preparation.json").write_bytes(encoded(receipt))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    template = actions.add_parser("template", help="Write an incomplete, deliberately blocked provenance template")
    template.add_argument("--output", type=Path, required=True)
    command = actions.add_parser("prepare", help="Prepare files offline; never submit an RCSB query")
    command.add_argument("--provenance", type=Path, required=True)
    command.add_argument("--provenance-sha256", required=True)
    command.add_argument("--through-date", required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == "template":
            with args.output.open("xb") as handle:
                handle.write(encoded(provenance_template()))
            print("Incomplete template written; verified provenance is required before query preparation.")
        else:
            receipt = prepare(args.provenance, args.provenance_sha256, args.through_date, args.output_dir)
            print(json.dumps({"status": receipt["status"], "cutoff_exclusive": receipt["cutoff_exclusive"]}))
    except (PreparationError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
