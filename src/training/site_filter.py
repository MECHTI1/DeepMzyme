from __future__ import annotations

import csv
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import List, Optional, Tuple

from data_structures import PocketRecord
from training.labels import normalize_ec_number_list, parse_structure_identity

SUMMARY_COLUMN_ALIASES = {
    "pdbid": ("pdbid", "structure"),
    "metal residue number": ("metal residue number", "chain_resi"),
    "EC number": ("EC number", "ecnumber"),
    "metal residue type": ("metal residue type", "metaltype"),
}
CATALYTIC_COLUMN_ALIASES = ("whether_catalytic",)
SiteKey = Tuple[str, str, str]
AllowedSiteMetalLabels = dict[SiteKey, str]


def _resolve_summary_columns(fieldnames: Optional[List[str]], summary_csv: Path) -> dict[str, str]:
    if fieldnames is None:
        raise ValueError(f"Could not read CSV header from: {summary_csv}")

    normalized_fieldnames = {fieldname.strip().lower(): fieldname for fieldname in fieldnames if fieldname}
    resolved_columns: dict[str, str] = {}
    missing_columns: list[str] = []
    for canonical_name, aliases in SUMMARY_COLUMN_ALIASES.items():
        resolved_name = next(
            (normalized_fieldnames[alias.lower()] for alias in aliases if alias.lower() in normalized_fieldnames),
            None,
        )
        if resolved_name is None:
            missing_columns.append(canonical_name)
            continue
        resolved_columns[canonical_name] = resolved_name
    if missing_columns:
        raise ValueError(f"Missing required columns {sorted(missing_columns)} in {summary_csv}")
    return resolved_columns


def _resolve_optional_column(fieldnames: list[str], aliases: tuple[str, ...]) -> str | None:
    normalized_fieldnames = {fieldname.strip().lower(): fieldname for fieldname in fieldnames if fieldname}
    for alias in aliases:
        resolved_name = normalized_fieldnames.get(alias.lower())
        if resolved_name is not None:
            return resolved_name
    return None


def _row_is_catalytic(row: Mapping[str, str], catalytic_column: str | None) -> bool:
    if catalytic_column is None:
        return True
    value = str(row.get(catalytic_column, "")).strip().lower()
    if not value:
        return False
    return value in {"1", "true", "yes", "y", "catalytic"}


def _iter_normalized_summary_rows(summary_csv: Path) -> tuple[list[tuple[SiteKey, dict[str, str]]], dict[str, str]]:
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        resolved_columns = _resolve_summary_columns(reader.fieldnames, summary_csv)
        catalytic_column = _resolve_optional_column(list(reader.fieldnames or []), CATALYTIC_COLUMN_ALIASES)

        normalized_rows: list[tuple[SiteKey, dict[str, str]]] = []
        for row in reader:
            if not _row_is_catalytic(row, catalytic_column):
                continue
            pdbid = row[resolved_columns["pdbid"]].strip().lower()
            ec_number = normalize_ec_number_list(row[resolved_columns["EC number"]])
            metal_residue_number = row[resolved_columns["metal residue number"]].strip()
            if pdbid and ec_number and metal_residue_number:
                normalized_rows.append(((pdbid, ec_number, metal_residue_number), row))
    return normalized_rows, resolved_columns


def load_allowed_site_metal_labels(summary_csv: Path) -> AllowedSiteMetalLabels:
    normalized_rows, resolved_columns = _iter_normalized_summary_rows(summary_csv)
    metal_residue_type_column = resolved_columns["metal residue type"]
    metal_labels: AllowedSiteMetalLabels = {}
    for site_key, row in normalized_rows:
        metal_residue_type = row.get(metal_residue_type_column, "").strip().upper()
        if metal_residue_type:
            metal_labels[site_key] = metal_residue_type
    return metal_labels


def resolve_allowed_site_metal_labels(summary_csv: Path | None) -> AllowedSiteMetalLabels | None:
    if summary_csv is None:
        return None

    summary_path = Path(summary_csv)
    if not summary_path.exists():
        raise FileNotFoundError(f"Training summary file not found: {summary_path}")
    return load_allowed_site_metal_labels(summary_path)


def matched_site_keys_for_pocket(
    pocket: PocketRecord,
    structure_path: Path,
    allowed_site_keys: Collection[SiteKey],
) -> set[SiteKey]:
    matched_site_keys: set[SiteKey] = set()
    pdbid, _chain_id, ec_number = parse_structure_identity(structure_path.stem)
    metal_site_ids = pocket.metadata.get("metal_site_ids", [])
    if not isinstance(metal_site_ids, list):
        return matched_site_keys

    for site_id in metal_site_ids:
        if not isinstance(site_id, tuple) or len(site_id) != 3:
            continue
        chain_id, resseq, _icode = site_id
        try:
            normalized_resseq = int(str(resseq).strip())
        except (TypeError, ValueError):
            continue
        candidate_key = (pdbid, ec_number, f"{str(chain_id).strip()}_{normalized_resseq}")
        if candidate_key in allowed_site_keys:
            matched_site_keys.add(candidate_key)
    return matched_site_keys


def pocket_matches_allowed_sites(
    pocket: PocketRecord,
    structure_path: Path,
    allowed_site_keys: Collection[SiteKey],
) -> bool:
    return bool(matched_site_keys_for_pocket(pocket, structure_path, allowed_site_keys))


def matched_site_metal_types(
    pocket: PocketRecord,
    structure_path: Path,
    allowed_site_metal_labels: Mapping[SiteKey, str],
) -> set[str]:
    return {
        allowed_site_metal_labels[site_key]
        for site_key in matched_site_keys_for_pocket(pocket, structure_path, allowed_site_metal_labels)
    }
