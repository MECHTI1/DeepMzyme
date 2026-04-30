#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from structure_sync_utils import SUPPORTED_TRANSITION_METALS


TARGET_DATASET_FILENAMES = (
    "final_data_summarazing_table_transition_metals_only_catalytic.csv",
    "final_data_summarazing_table_transition_metals_only_catalytic_with_native.csv",
    "final_data_summarazing_table_transition_metals_only_catalytic_verified_biological_metal.csv",
)
COFACTOR_ANALYSIS_FILENAME = "cofactor_analysis_by_structure_chain.csv"
METALTYPE_COLUMN = "metaltype"
ANNOTATED_COFACTORS_COLUMN = "Annotated Cofactors"
ANNOTATED_METAL_SYMBOLS_COLUMN = "Annotated Metal Symbols"
SUPPORTED_DATASET_METALS = frozenset(SUPPORTED_TRANSITION_METALS)
METAL_NORMALIZATION = {
    "CO": "CO",
    "CU": "CU",
    "FE": "FE",
    "FE2": "FE",
    "FE3": "FE",
    "MG": "MG",
    "MN": "MN",
    "NI": "NI",
    "ZN": "ZN",
}
COFACTOR_SYMBOL_PATTERNS = {
    "CO": [r"^CO(?:\(\d\+\))?$", r"\bCO CATION\b", r"COBALT"],
    "CU": [r"^CU(?:\(\d\+\))?$", r"\bCU CATION\b", r"COPPER"],
    "FE": [r"^FE(?:\(\d\+\))?$", r"\bFE CATION\b", r"IRON"],
    "MG": [r"^MG(?:\(\d\+\))?$", r"\bMG CATION\b", r"MAGNESIUM"],
    "MN": [r"^MN(?:\(\d\+\))?$", r"\bMN CATION\b", r"MANGANESE"],
    "NI": [r"^NI(?:\(\d\+\))?$", r"\bNI CATION\b", r"NICKEL"],
    "ZN": [r"^ZN(?:\(\d\+\))?$", r"\bZN CATION\b", r"ZINC"],
}


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "DeepMzyme_Data").exists() and (candidate / "prepare_training_and_test_set").exists():
            return candidate
    raise FileNotFoundError("Could not locate the project root from the current working directory.")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
DEFAULT_BASE_DIR = PROJECT_ROOT / "DeepMzyme_Data" / "train_and_test_sets_structures_exact_pinmymetal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter exact PinMyMetal tables so they only retain supported transition metals."
    )
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    return parser.parse_args()


def normalize_metal_symbol(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", str(value).upper())
    return METAL_NORMALIZATION.get(cleaned, cleaned)


def split_semicolon_values(value: str) -> list[str]:
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def extract_symbols_from_cofactor_name(value: str) -> set[str]:
    upper_value = str(value).upper().strip()
    matches: set[str] = set()
    for symbol, patterns in COFACTOR_SYMBOL_PATTERNS.items():
        if any(re.search(pattern, upper_value) for pattern in patterns):
            matches.add(symbol)
    return matches


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"Could not read CSV header from: {path}")
        return fieldnames, [dict(row) for row in reader]


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sanitize_dataset_table(path: Path) -> int:
    fieldnames, rows = read_csv_rows(path)
    if METALTYPE_COLUMN not in fieldnames:
        raise ValueError(f"Required column {METALTYPE_COLUMN!r} missing from: {path}")

    filtered_rows: list[dict[str, str]] = []
    removed_rows = 0
    removed_symbols: set[str] = set()
    for row in rows:
        metaltype = normalize_metal_symbol(row.get(METALTYPE_COLUMN, ""))
        row[METALTYPE_COLUMN] = metaltype
        if metaltype not in SUPPORTED_DATASET_METALS:
            removed_rows += 1
            if metaltype:
                removed_symbols.add(metaltype)
            continue
        filtered_rows.append(row)

    write_csv_rows(path, fieldnames, filtered_rows)
    if removed_rows:
        print(f"[INFO] Filtered {removed_rows} unsupported rows from {path.name}: {sorted(removed_symbols)}")
    else:
        print(f"[INFO] No unsupported rows found in {path.name}")
    return removed_rows


def sanitize_cofactor_analysis_table(path: Path) -> tuple[int, int]:
    fieldnames, rows = read_csv_rows(path)
    required = {ANNOTATED_COFACTORS_COLUMN, ANNOTATED_METAL_SYMBOLS_COLUMN}
    missing = sorted(required - set(fieldnames))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    filtered_rows: list[dict[str, str]] = []
    removed_row_count = 0
    removed_cofactor_count = 0

    for row in rows:
        cofactor_names = split_semicolon_values(row.get(ANNOTATED_COFACTORS_COLUMN, ""))
        filtered_cofactors: list[str] = []
        for cofactor_name in cofactor_names:
            matched_symbols = extract_symbols_from_cofactor_name(cofactor_name)
            if matched_symbols and not (matched_symbols & SUPPORTED_DATASET_METALS):
                removed_cofactor_count += 1
                continue
            filtered_cofactors.append(cofactor_name)

        annotated_symbols = [
            normalize_metal_symbol(symbol)
            for symbol in split_semicolon_values(row.get(ANNOTATED_METAL_SYMBOLS_COLUMN, ""))
        ]
        supported_symbols = sorted({symbol for symbol in annotated_symbols if symbol in SUPPORTED_DATASET_METALS})
        row[ANNOTATED_COFACTORS_COLUMN] = "; ".join(filtered_cofactors)
        row[ANNOTATED_METAL_SYMBOLS_COLUMN] = ";".join(supported_symbols)

        if not supported_symbols:
            removed_row_count += 1
            continue
        filtered_rows.append(row)

    write_csv_rows(path, fieldnames, filtered_rows)
    if removed_row_count or removed_cofactor_count:
        print(
            f"[INFO] Sanitized {path.name}: removed {removed_row_count} rows with no supported metals and "
            f"{removed_cofactor_count} unsupported cofactor entries."
        )
    else:
        print(f"[INFO] No unsupported cofactor annotations found in {path.name}")
    return removed_row_count, removed_cofactor_count


def sanitize_split_dir(split_dir: Path) -> None:
    if not split_dir.exists():
        print(f"[INFO] Split directory not found, skipping: {split_dir}")
        return

    print(f"[INFO] Sanitizing {split_dir}")
    for filename in TARGET_DATASET_FILENAMES:
        path = split_dir / filename
        if not path.exists():
            print(f"[INFO] File not found, skipping: {path.name}")
            continue
        sanitize_dataset_table(path)

    cofactor_path = split_dir / COFACTOR_ANALYSIS_FILENAME
    if cofactor_path.exists():
        sanitize_cofactor_analysis_table(cofactor_path)
    else:
        print(f"[INFO] File not found, skipping: {cofactor_path.name}")


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir if args.base_dir.is_absolute() else (PROJECT_ROOT / args.base_dir).resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    sanitize_split_dir(base_dir / "train")
    sanitize_split_dir(base_dir / "test")


if __name__ == "__main__":
    main()
