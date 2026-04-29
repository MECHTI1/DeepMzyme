#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import shutil
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_paths import MEDIA_DATA_ROOT

SOURCE_CSV = (
    MEDIA_DATA_ROOT
    / "pinmymetal_sets"
    / "mahomes"
    / "test_set"
    / "data_summarizing_tables"
    / "final_data_summarazing_table_transition_metals_only_catalytic.csv"
)
SOURCE_PDB_DIR = MEDIA_DATA_ROOT / "pinmymetal_sets" / "test" / "pdb_updatedv2"
DEST_DIR = SRC_DIR.parent / ".data" / "train_and_test_sets_structures_exact_pinmymetal" / "test"
CSV_REQUIRED_COLUMNS = frozenset({"structure", "chain_resi", "ecnumber"})
STRUCTURE_FILE_RE = re.compile(r"^(?P<structure>[^_]+)__chain_(?P<chain>[^_]+)__EC_(?P<ec>.+)\.pdb$")


def require_columns(fieldnames: list[str] | None, required: frozenset[str], csv_path: Path) -> None:
    if fieldnames is None:
        raise ValueError(f"Could not read CSV header from: {csv_path}")
    missing = sorted(required - set(fieldnames))
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")


def parse_chain_id(chain_resi: str) -> str:
    chain_id, separator, _resseq = chain_resi.partition("_")
    if not separator or not chain_id.strip():
        raise ValueError(f"Could not parse chain id from chain_resi value: {chain_resi!r}")
    return chain_id.strip()


def normalize_ec_number_list(value: str) -> str:
    values = []
    seen = set()
    for ec in re.split(r"[;,]", value):
        normalized = ec.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        values.append(normalized)
    return ";".join(values)


def collect_structure_keys(csv_path: Path) -> set[tuple[str, str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader.fieldnames, CSV_REQUIRED_COLUMNS, csv_path)

        structure_keys: set[tuple[str, str, str]] = set()
        for row in reader:
            structure = row["structure"].strip().lower()
            chain_id = parse_chain_id(row["chain_resi"])
            ecnumber = normalize_ec_number_list(row["ecnumber"])
            if not structure or not ecnumber:
                continue
            structure_keys.add((structure, chain_id, ecnumber))

    return structure_keys


def build_source_pdb_index(source_dir: Path) -> dict[tuple[str, str, str], Path]:
    pdb_index: dict[tuple[str, str, str], Path] = {}

    for source_path in sorted(source_dir.glob("*.pdb")):
        match = STRUCTURE_FILE_RE.match(source_path.name)
        if match is None:
            continue

        key = (
            match.group("structure").strip().lower(),
            match.group("chain").strip(),
            normalize_ec_number_list(match.group("ec")),
        )
        pdb_index.setdefault(key, source_path)

    return pdb_index


def copy_file(source_path: Path, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, dest_path)


def main() -> None:
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"Source CSV not found: {SOURCE_CSV}")
    if not SOURCE_PDB_DIR.exists():
        raise FileNotFoundError(f"Source PDB directory not found: {SOURCE_PDB_DIR}")

    structure_keys = collect_structure_keys(SOURCE_CSV)
    if not structure_keys:
        raise ValueError(f"No structure keys were derived from: {SOURCE_CSV}")
    pdb_index = build_source_pdb_index(SOURCE_PDB_DIR)

    missing_paths = []
    source_paths = []
    for structure_key in sorted(structure_keys):
        source_path = pdb_index.get(structure_key)
        if source_path is None:
            structure, chain_id, ecnumber = structure_key
            missing_paths.append(f"{structure}__chain_{chain_id}__EC_{ecnumber}.pdb")
            continue
        source_paths.append(source_path)

    if missing_paths:
        preview = ", ".join(str(path) for path in missing_paths[:10])
        extra_count = len(missing_paths) - min(len(missing_paths), 10)
        suffix = "" if extra_count == 0 else f" ... and {extra_count} more"
        raise FileNotFoundError(f"Missing source PDB files after EC normalization: {preview}{suffix}")

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    copied_structure_count = 0
    for source_path in source_paths:
        copy_file(source_path, DEST_DIR / source_path.name)
        copied_structure_count += 1

    dest_csv_path = DEST_DIR / SOURCE_CSV.name
    copy_file(SOURCE_CSV, dest_csv_path)

    print(f"Copied CSV to {dest_csv_path}")
    print(f"Copied {copied_structure_count} structures to {DEST_DIR}")


if __name__ == "__main__":
    main()
