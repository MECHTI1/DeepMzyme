from __future__ import annotations

from pathlib import Path
import re

SUPPORTED_TRANSITION_METALS = frozenset({"MN", "FE", "CO", "NI", "CU", "ZN"})


def build_structure_filename(structure: str, chain_id: str, ecnumber: str) -> str:
    return f"{structure}__chain_{chain_id}__EC_{ecnumber}.pdb"


def normalize_ec_number_list(value: str) -> str:
    values: list[str] = []
    seen: set[str] = set()
    for ec in re.split(r"[;,]", value):
        normalized = ec.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        values.append(normalized)
    return ";".join(values)


def resolve_structure_path(
    structure_dir: Path,
    *,
    structure: str,
    chain_id: str,
    ecnumber: str,
) -> Path:
    structure_dir = Path(structure_dir)
    exact_path = structure_dir / build_structure_filename(structure, chain_id, ecnumber)
    if exact_path.exists():
        return exact_path

    alternate_ecnumber = ecnumber.replace(";", ",")
    alternate_path = structure_dir / build_structure_filename(structure, chain_id, alternate_ecnumber)
    if alternate_path.exists():
        return alternate_path

    for candidate_path in sorted(structure_dir.glob(f"{structure}__chain_{chain_id}__EC_*.pdb")):
        ec_fragment = candidate_path.stem.split("__EC_", 1)[-1]
        if normalize_ec_number_list(ec_fragment) == normalize_ec_number_list(ecnumber):
            return candidate_path

    return exact_path


def _parse_site_from_pdb_record(line: str) -> tuple[str, str] | None:
    if line.startswith("HETATM"):
        resname = line[17:20].strip().upper()
        chain_id = line[21].strip()
        resseq = line[22:26].strip()
    elif line.startswith("HET   "):
        resname = line[7:10].strip().upper()
        chain_id = line[12].strip()
        resseq = line[13:17].strip()
    else:
        return None

    if not resname or not chain_id or not resseq:
        return None
    try:
        normalized_resseq = int(resseq)
    except ValueError:
        return None
    return f"{chain_id}_{normalized_resseq}", resname


def collect_supported_transition_metal_sites(structure_path: Path) -> set[tuple[str, str]]:
    sites: set[tuple[str, str]] = set()
    with Path(structure_path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parsed = _parse_site_from_pdb_record(line)
            if parsed is None:
                continue
            chain_resi, resname = parsed
            if resname not in SUPPORTED_TRANSITION_METALS:
                continue
            sites.add((chain_resi, resname))
    return sites
