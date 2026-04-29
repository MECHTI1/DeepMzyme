#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


LIKELY_PDBID_COLUMNS = (
    "pdbid",
    "pdb_id",
    "pdb",
    "structure_id",
    "structure",
    "PDB",
)
PDBID_RE = re.compile(r"(?i)(?<![A-Za-z0-9])([0-9][A-Za-z0-9]{3})(?![A-Za-z0-9])")
UNKNOWN_WARN_COUNT = 10
UNKNOWN_WARN_FRACTION = 0.05


@dataclass(frozen=True)
class ScanResult:
    files: list[Path]
    pdbids: set[str]
    unknown_files: list[Path]
    pdbid_to_files: dict[str, list[Path]]


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / ".data").exists() and (candidate / "prepare_training_and_test_set").exists():
            return candidate
    raise FileNotFoundError("Could not locate the project root from the current working directory.")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
DEFAULT_BASE_DIR = PROJECT_ROOT / ".data" / "train_and_test_sets_structures_exact_pinmymetal"
DEFAULT_TRAIN_DIR = DEFAULT_BASE_DIR / "train"
DEFAULT_TEST_DIR = DEFAULT_BASE_DIR / "test"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".data" / "train_and_test_sets_structures_non_overlapped_pinmymetal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an additional non-overlapping train/test structure split where test PDB IDs have priority."
        )
    )
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-csv", type=Path, default=None, help="Optional specific train CSV to clean and copy.")
    parser.add_argument("--test-csv", type=Path, default=None, help="Optional specific test CSV to copy unchanged.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow removing an existing output directory before recreating it.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def normalize_pdbid(value: str) -> str:
    return str(value).strip().lower()


def extract_pdbid(text: str) -> str | None:
    match = PDBID_RE.search(str(text))
    if match is None:
        return None
    return normalize_pdbid(match.group(1))


def ensure_input_dir(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")


def prepare_output_dir(output_dir: Path, overwrite: bool) -> tuple[Path, Path]:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Re-run with --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    train_out = output_dir / "train"
    test_out = output_dir / "test"
    train_out.mkdir(parents=True, exist_ok=False)
    test_out.mkdir(parents=True, exist_ok=False)
    return train_out, test_out


def scan_structure_dir(directory: Path) -> ScanResult:
    files = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in {".pdb", ".cif"})
    pdbids: set[str] = set()
    unknown_files: list[Path] = []
    pdbid_to_files: dict[str, list[Path]] = {}

    for path in files:
        pdbid = extract_pdbid(path.name)
        if pdbid is None:
            unknown_files.append(path)
            continue
        pdbids.add(pdbid)
        pdbid_to_files.setdefault(pdbid, []).append(path)

    return ScanResult(files=files, pdbids=pdbids, unknown_files=unknown_files, pdbid_to_files=pdbid_to_files)


def warn_if_many_unknowns(unknown_count: int, total_count: int, label: str) -> None:
    if total_count == 0 or unknown_count == 0:
        return
    fraction = unknown_count / total_count
    if unknown_count >= UNKNOWN_WARN_COUNT or fraction >= UNKNOWN_WARN_FRACTION:
        print(
            f"[WARN] {label} has {unknown_count} filenames with unknown PDB IDs "
            f"({fraction:.1%} of {total_count} structure files). They will be skipped."
        )


def copy_structure_files(paths: Iterable[Path], destination_dir: Path) -> int:
    copied = 0
    for source_path in paths:
        shutil.copy2(source_path, destination_dir / source_path.name)
        copied += 1
    return copied


def autodetect_csv_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".csv")


def resolve_csv_inputs(directory: Path, specific_csv: Path | None) -> list[Path]:
    if specific_csv is not None:
        resolved = resolve_path(specific_csv)
        if not resolved.exists():
            raise FileNotFoundError(f"CSV file not found: {resolved}")
        return [resolved]
    return autodetect_csv_files(directory)


def detect_pdbid_column(fieldnames: list[str] | None) -> str | None:
    if fieldnames is None:
        return None

    lowered = {name.strip().lower(): name for name in fieldnames}
    for candidate in LIKELY_PDBID_COLUMNS:
        key = candidate.strip().lower()
        if key in lowered:
            return lowered[key]
    return None


def clean_train_csv(source_csv: Path, dest_csv: Path, test_pdbids: set[str]) -> tuple[int, int, int, str | None]:
    with source_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"Could not read CSV header from: {source_csv}")

        pdbid_column = detect_pdbid_column(fieldnames)
        if pdbid_column is None:
            shutil.copy2(source_csv, dest_csv)
            return 0, 0, 0, None

        kept_rows: list[dict[str, str]] = []
        removed_rows = 0
        unknown_rows = 0

        for row in reader:
            pdbid = extract_pdbid(row.get(pdbid_column, ""))
            if pdbid is None:
                kept_rows.append(row)
                unknown_rows += 1
                continue
            if pdbid in test_pdbids:
                removed_rows += 1
                continue
            kept_rows.append(row)

    with dest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    return len(kept_rows), removed_rows, unknown_rows, pdbid_column


def copy_test_csv(source_csv: Path, dest_csv: Path) -> None:
    shutil.copy2(source_csv, dest_csv)


def verify_final_overlap(train_dir: Path, test_dir: Path) -> tuple[set[str], set[str], set[str]]:
    train_scan = scan_structure_dir(train_dir)
    test_scan = scan_structure_dir(test_dir)
    overlap = train_scan.pdbids & test_scan.pdbids
    return train_scan.pdbids, test_scan.pdbids, overlap


def format_preview(values: Iterable[str], limit: int = 30) -> str:
    items = sorted(set(values))
    if not items:
        return "(none)"
    return ", ".join(items[:limit])


def main() -> None:
    args = parse_args()

    train_dir = resolve_path(args.train_dir)
    test_dir = resolve_path(args.test_dir)
    output_dir = resolve_path(args.output_dir)

    ensure_input_dir(train_dir, "Train structure directory")
    ensure_input_dir(test_dir, "Test structure directory")

    train_scan = scan_structure_dir(train_dir)
    test_scan = scan_structure_dir(test_dir)

    train_pdbids = set(train_scan.pdbids)
    test_pdbids = set(test_scan.pdbids)
    shared_pdbids = train_pdbids & test_pdbids
    clean_train_pdbids = train_pdbids - test_pdbids
    clean_test_pdbids = set(test_pdbids)

    warn_if_many_unknowns(len(train_scan.unknown_files), len(train_scan.files), "Train input")
    warn_if_many_unknowns(len(test_scan.unknown_files), len(test_scan.files), "Test input")

    if clean_test_pdbids != test_pdbids:
        raise RuntimeError("Clean test PDB IDs differ from the original test PDB IDs.")

    train_out, test_out = prepare_output_dir(output_dir, overwrite=args.overwrite)

    copied_test_files = copy_structure_files(
        (path for pdbid, paths in sorted(test_scan.pdbid_to_files.items()) for path in paths if pdbid in clean_test_pdbids),
        test_out,
    )
    copied_train_files = copy_structure_files(
        (path for pdbid, paths in sorted(train_scan.pdbid_to_files.items()) for path in paths if pdbid in clean_train_pdbids),
        train_out,
    )

    train_csvs = resolve_csv_inputs(train_dir, args.train_csv)
    test_csvs = resolve_csv_inputs(test_dir, args.test_csv)

    if train_csvs:
        for csv_path in train_csvs:
            dest_csv = train_out / csv_path.name
            kept_rows, removed_rows, unknown_rows, pdbid_column = clean_train_csv(csv_path, dest_csv, test_pdbids)
            if pdbid_column is None:
                print(f"[WARN] Train CSV {csv_path.name}: no likely PDB ID column found; copied unchanged.")
            else:
                print(
                    f"[INFO] Train CSV {csv_path.name}: kept {kept_rows} rows, removed {removed_rows} rows, "
                    f"unknown PDB ID rows kept {unknown_rows}, using column {pdbid_column!r}."
                )
    else:
        print(f"[INFO] No train CSV files found in {train_dir}")

    if test_csvs:
        for csv_path in test_csvs:
            copy_test_csv(csv_path, test_out / csv_path.name)
            print(f"[INFO] Copied test CSV unchanged: {csv_path.name}")
    else:
        print(f"[INFO] No test CSV files found in {test_dir}")

    final_train_pdbids, final_test_pdbids, final_overlap = verify_final_overlap(train_out, test_out)

    if final_overlap:
        raise RuntimeError(f"Final overlap is not zero: {len(final_overlap)} shared PDB IDs remain.")

    if final_test_pdbids != test_pdbids:
        missing = sorted(test_pdbids - final_test_pdbids)
        raise RuntimeError(
            f"Cleaned test set lost PDB IDs compared with the original test set: {format_preview(missing)}"
        )

    print()
    print("Split report")
    print(f"Original train file count: {len(train_scan.files)}")
    print(f"Original test file count: {len(test_scan.files)}")
    print(f"Original train unique PDB IDs: {len(train_pdbids)}")
    print(f"Original test unique PDB IDs: {len(test_pdbids)}")
    print(f"Number of unknown train filenames: {len(train_scan.unknown_files)}")
    print(f"Number of unknown test filenames: {len(test_scan.unknown_files)}")
    print(f"Number of shared PDB IDs: {len(shared_pdbids)}")
    print(f"First 30 shared PDB IDs: {format_preview(shared_pdbids, limit=30)}")
    print(f"Clean train unique PDB IDs: {len(clean_train_pdbids)}")
    print(f"Clean test unique PDB IDs: {len(clean_test_pdbids)}")
    print(f"Copied clean train structure files: {copied_train_files}")
    print(f"Copied clean test structure files: {copied_test_files}")
    print(f"Final overlap count: {len(final_overlap)}")


if __name__ == "__main__":
    main()
