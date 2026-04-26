#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable

from project_paths import MAHOMES_SUMMARY_DIR, MAHOMES_TRAIN_SET_DIR
from training.labels import normalize_ec_number_list

JOB_ROOT = MAHOMES_TRAIN_SET_DIR
SUMMARY_DIR = MAHOMES_SUMMARY_DIR
OUTPUT_CSV = SUMMARY_DIR / "data_summarazing_table.csv"
SUPPORTED_METAL_RESIDUE_TYPES = frozenset({"MN", "FE", "CO", "NI", "CU", "ZN"})

STRUCTURE_CHAIN_RE = re.compile(r"__chain_([^_]+)")
STRUCTURE_EC_RE = re.compile(r"__EC_([^_]+)")
SITE_DIR_RE = re.compile(r"^(?P<metal_type>[A-Za-z0-9]+)_(?P<metal_resseq>\d+)__.+$")


def parse_pdbid(structure_dir_name: str) -> str:
    return structure_dir_name.split("__", 1)[0].strip()


def parse_ec_numbers(structure_dir_name: str) -> str:
    return normalize_ec_number_list(";".join(STRUCTURE_EC_RE.findall(structure_dir_name)))


def parse_chain_id(structure_dir_name: str) -> str:
    match = STRUCTURE_CHAIN_RE.search(structure_dir_name)
    if match is None:
        raise ValueError(f"Could not parse chain id from structure directory name: {structure_dir_name}")
    return match.group(1).strip()


def parse_site_dir(site_dir_name: str) -> tuple[str, int] | None:
    match = SITE_DIR_RE.match(site_dir_name)
    if match is None:
        return None
    metal_type = match.group("metal_type")
    metal_resseq = int(match.group("metal_resseq"))
    return metal_type, metal_resseq


def resolve_structure_source_path(job_dir: Path, structure_dir: Path) -> Path | None:
    candidates = (
        job_dir / f"{structure_dir.name}.pdb",
        job_dir / f"{structure_dir.name}.cif",
        job_dir / f"{structure_dir.name}.mmcif",
        structure_dir / f"{structure_dir.name}.pdb",
        structure_dir / f"{structure_dir.name}.cif",
        structure_dir / f"{structure_dir.name}.mmcif",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def iter_supported_metal_sites_from_pdb(structure_path: Path) -> Iterable[tuple[str, int, str]]:
    seen_sites: set[tuple[str, int, str]] = set()

    with structure_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            if not raw_line.startswith("HETATM"):
                continue

            resname = raw_line[17:20].strip().upper()
            if resname not in SUPPORTED_METAL_RESIDUE_TYPES:
                continue

            chain_id = raw_line[21].strip()
            resseq_text = raw_line[22:26].strip()
            if not chain_id or not resseq_text:
                continue

            metal_type = resname
            site_key = (chain_id, int(resseq_text), metal_type)
            if site_key in seen_sites:
                continue

            seen_sites.add(site_key)
            yield site_key


def iter_summary_rows_from_structure_file(
    *,
    job_dir: Path,
    structure_dir: Path,
    pdbid: str,
    ec_number: str,
):
    structure_path = resolve_structure_source_path(job_dir, structure_dir)
    if structure_path is None or structure_path.suffix.lower() != ".pdb":
        return

    for chain_id, metal_resseq, metal_type in iter_supported_metal_sites_from_pdb(structure_path):
        yield {
            "pdbid": pdbid,
            "metal residue number": f"{chain_id}_{metal_resseq}",
            "metal residue type": metal_type,
            "EC number": ec_number,
        }


def iter_summary_rows_from_site_dirs(
    *,
    structure_dir: Path,
    pdbid: str,
    ec_number: str,
):
    for site_dir in sorted(structure_dir.iterdir()):
        if not site_dir.is_dir():
            continue

        parsed = parse_site_dir(site_dir.name)
        if parsed is None:
            continue

        metal_type, metal_resseq = parsed
        metal_type = metal_type.strip().upper()
        if metal_type not in SUPPORTED_METAL_RESIDUE_TYPES:
            continue

        yield {
            "pdbid": pdbid,
            "metal residue number": f"{parse_chain_id(structure_dir.name)}_{metal_resseq}",
            "metal residue type": metal_type,
            "EC number": ec_number,
        }


def iter_summary_rows(job_root: Path):
    for job_dir in sorted(job_root.glob("job_*")):
        if not job_dir.is_dir():
            continue
        for structure_dir in sorted(job_dir.iterdir()):
            if not structure_dir.is_dir():
                continue

            pdbid = parse_pdbid(structure_dir.name)
            ec_number = parse_ec_numbers(structure_dir.name)
            structure_rows = list(
                iter_summary_rows_from_structure_file(
                    job_dir=job_dir,
                    structure_dir=structure_dir,
                    pdbid=pdbid,
                    ec_number=ec_number,
                )
            )
            if structure_rows:
                for row in structure_rows:
                    yield row
                continue

            yield from iter_summary_rows_from_site_dirs(
                structure_dir=structure_dir,
                pdbid=pdbid,
                ec_number=ec_number,
            )


def main() -> None:
    if not JOB_ROOT.exists():
        raise FileNotFoundError(f"Job root not found: {JOB_ROOT}")

    rows = list(iter_summary_rows(JOB_ROOT))
    if not rows:
        raise ValueError(f"No metal-site rows were found under: {JOB_ROOT}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["pdbid", "metal residue number", "metal residue type", "EC number"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
