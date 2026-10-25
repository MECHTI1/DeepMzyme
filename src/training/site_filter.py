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


# Selected-site extraction: summary rows name the metal sites that define pockets.
SITE_KEY_MODES = ("accession_ec", "accession")
SelectedSiteId = Tuple[str, int]


def parse_summary_chain_resi(chain_resi: str) -> SelectedSiteId:
    """Parse a summary ``chain_resi`` value such as ``"A_123"`` into ``(chain, resseq)``.

    The value is stripped as the summary reader strips it and splits on its
    last underscore. It must then read back exactly as the key that
    ``matched_site_keys_for_pocket`` builds (``f"{chain}_{resseq}"``): no
    whitespace around the underscore and a plain integer residue number
    without a plus sign or leading zeros. Summary keys carry no insertion
    code, so ``"A_123B"`` or ``"A _ 123"`` raise instead of selecting a site
    whose label would never match.
    """
    text = str(chain_resi).strip()
    parts = text.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Malformed chain_resi {chain_resi!r}: expected '<chain>_<resseq>'.")
    chain_id, resseq_text = parts[0].strip(), parts[1].strip()
    try:
        resseq = int(resseq_text)
    except ValueError:
        resseq = None
    if not chain_id or resseq is None or f"{chain_id}_{resseq}" != text:
        raise ValueError(f"Malformed chain_resi {chain_resi!r}: expected '<chain>_<resseq>'.")
    return chain_id, resseq


def _validate_site_key_mode(key_mode: str) -> str:
    if key_mode not in SITE_KEY_MODES:
        raise ValueError(f"Unsupported site key mode {key_mode!r}; expected one of {SITE_KEY_MODES}.")
    return key_mode


class SiteKeyIndex:
    """Site keys grouped by accession, built once and reused across structures.

    ``selected_site_ids_for_structure`` scans every key when it is given a
    plain collection; given an index, it reads only the structure's own keys.
    """

    def __init__(self, allowed_site_keys: Collection[SiteKey]):
        keys_by_accession: dict[str, list[SiteKey]] = {}
        for site_key in allowed_site_keys:
            if not isinstance(site_key, tuple) or len(site_key) != 3:
                raise ValueError(f"Site keys must be (accession, EC, chain_resi) tuples, got {site_key!r}.")
            keys_by_accession.setdefault(site_key[0], []).append(site_key)
        self._keys_by_accession = {accession: tuple(keys) for accession, keys in keys_by_accession.items()}

    def keys_for_accession(self, accession: str) -> tuple[SiteKey, ...]:
        return self._keys_by_accession.get(accession, ())


def selected_site_ids_for_structure(
    structure_path: Path,
    allowed_site_keys: Collection[SiteKey] | SiteKeyIndex,
    *,
    key_mode: str = "accession_ec",
) -> frozenset[SelectedSiteId]:
    """Return the selected ``(chain, resseq)`` metal sites of one structure.

    A key selects a site exactly when ``matched_site_keys_for_pocket`` would
    match it: its accession and EC must equal the lower-case accession and the
    normalized EC list parsed from the structure file stem, the form that
    ``load_allowed_site_metal_labels`` writes. A key in any other form is
    ignored here as it is by the matcher. ``key_mode="accession"`` ignores the
    EC part, for site tables built without EC labels. Only the keys of this
    structure are parsed, and a malformed ``chain_resi`` among them raises,
    including one with surrounding whitespace, which the matcher never
    matches. To select sites for many structures, build a ``SiteKeyIndex``
    once and pass it instead of the keys.
    """
    _validate_site_key_mode(key_mode)
    accession, _chain_id, ec_number = parse_structure_identity(Path(structure_path).stem)
    if not isinstance(allowed_site_keys, SiteKeyIndex):
        allowed_site_keys = SiteKeyIndex(allowed_site_keys)
    selected: set[SelectedSiteId] = set()
    for _accession, key_ec_number, chain_resi in allowed_site_keys.keys_for_accession(accession):
        if key_mode == "accession_ec" and key_ec_number != ec_number:
            continue
        chain_id, resseq = parse_summary_chain_resi(chain_resi)
        if chain_resi != f"{chain_id}_{resseq}":
            raise ValueError(
                f"Malformed chain_resi {chain_resi!r}: site keys hold the stripped '<chain>_<resseq>' form."
            )
        selected.add((chain_id, resseq))
    return frozenset(selected)


def _count_catalytic_summary_rows(summary_csv: Path) -> tuple[int, int, str | None]:
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        catalytic_column = _resolve_optional_column(list(reader.fieldnames or []), CATALYTIC_COLUMN_ALIASES)
        n_rows = 0
        n_catalytic_rows = 0
        for row in reader:
            n_rows += 1
            n_catalytic_rows += _row_is_catalytic(row, catalytic_column)
    return n_rows, n_catalytic_rows, catalytic_column


def validate_selected_site_summary(
    summary_csv: Path,
    *,
    key_mode: str = "accession_ec",
    max_examples: int = 10,
) -> dict[str, object]:
    """Check a site summary for selected-site extraction and count its contents.

    Rows are selected exactly as ``load_allowed_site_metal_labels`` selects
    them (catalytic, with accession, EC and chain_resi). Where that reader
    keeps the last of duplicate site rows and drops rows with an empty metal
    type, this check raises; it also raises on a malformed ``chain_resi``.
    Duplicates are judged on the ``key_mode`` key. Catalytic rows without an
    accession, EC or chain_resi are counted, not raised on.
    """
    _validate_site_key_mode(key_mode)
    summary_path = Path(summary_csv)
    normalized_rows, resolved_columns = _iter_normalized_summary_rows(summary_path)
    n_rows, n_catalytic_rows, catalytic_column = _count_catalytic_summary_rows(summary_path)
    metal_residue_type_column = resolved_columns["metal residue type"]

    row_counts: dict[tuple, int] = {}
    empty_metal_type_keys: list[SiteKey] = []
    malformed_chain_resi: list[str] = []
    metal_type_counts: dict[str, int] = {}
    structure_keys: set = set()
    accessions: set[str] = set()
    for site_key, row in normalized_rows:
        accession, ec_number, chain_resi = site_key
        try:
            chain_id, resseq = parse_summary_chain_resi(chain_resi)
        except ValueError:
            malformed_chain_resi.append(chain_resi)
            continue
        structure_key = (accession, ec_number) if key_mode == "accession_ec" else (accession,)
        unique_key = (*structure_key, chain_id, resseq)
        row_counts[unique_key] = row_counts.get(unique_key, 0) + 1
        structure_keys.add(structure_key)
        accessions.add(accession)
        metal_type = (row.get(metal_residue_type_column) or "").strip().upper()
        if not metal_type:
            empty_metal_type_keys.append(site_key)
            continue
        metal_type_counts[metal_type] = metal_type_counts.get(metal_type, 0) + 1

    duplicate_keys = sorted(key for key, count in row_counts.items() if count > 1)
    problems = []
    if duplicate_keys:
        problems.append(f"{len(duplicate_keys)} duplicate site keys (e.g. {duplicate_keys[:max_examples]})")
    if empty_metal_type_keys:
        problems.append(
            f"{len(empty_metal_type_keys)} catalytic rows with an empty metaltype "
            f"(e.g. {empty_metal_type_keys[:max_examples]})"
        )
    if malformed_chain_resi:
        problems.append(
            f"{len(malformed_chain_resi)} malformed chain_resi values (e.g. {malformed_chain_resi[:max_examples]})"
        )
    if problems:
        raise ValueError(f"Selected-site summary {summary_path} failed validation: " + "; ".join(problems) + ".")

    return {
        "summary_csv": str(summary_path),
        "key_mode": key_mode,
        "catalytic_column": catalytic_column,
        "n_rows": n_rows,
        "n_catalytic_rows": n_catalytic_rows,
        "n_non_catalytic_rows": n_rows - n_catalytic_rows,
        "n_incomplete_catalytic_rows": n_catalytic_rows - len(normalized_rows),
        "n_site_rows": len(normalized_rows),
        "n_sites": len(row_counts),
        "n_structures": len(structure_keys),
        "n_accessions": len(accessions),
        "metal_type_counts": dict(sorted(metal_type_counts.items())),
    }
