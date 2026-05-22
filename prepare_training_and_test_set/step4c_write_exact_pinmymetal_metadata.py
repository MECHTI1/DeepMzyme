#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


PDBID_RE = re.compile(r"(?i)(?<![A-Za-z0-9])([0-9][A-Za-z0-9]{3})(?![A-Za-z0-9])")
STRUCTURE_SUFFIXES = {".pdb", ".cif", ".mmcif"}
PRIMARY_CSV_NAME = "final_data_summarazing_table_transition_metals_only_catalytic.csv"


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "DeepMzyme_Data").exists() and (candidate / "prepare_training_and_test_set").exists():
            return candidate
    raise FileNotFoundError("Could not locate the project root from the current working directory.")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
DEFAULT_EXACT_DIR = PROJECT_ROOT / "DeepMzyme_Data" / "train_and_test_sets_structures_exact_pinmymetal"
DEFAULT_SOURCE_TRAIN = PROJECT_ROOT / "prepare_training_and_test_set" / "pinmymetal_files" / "classmodel_train_set"
DEFAULT_SOURCE_TEST = PROJECT_ROOT / "prepare_training_and_test_set" / "pinmymetal_files" / "classmodel_test_set"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write README.md and split_metadata.json for the exact/possibly-overlapped PinMyMetal split."
    )
    parser.add_argument("--exact-dir", type=Path, default=DEFAULT_EXACT_DIR)
    parser.add_argument("--source-train-membership", type=Path, default=DEFAULT_SOURCE_TRAIN)
    parser.add_argument("--source-test-membership", type=Path, default=DEFAULT_SOURCE_TEST)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def normalize_pdbid(value: str) -> str:
    return str(value).strip().lower()


def extract_pdbid(value: str) -> str | None:
    match = PDBID_RE.search(str(value))
    return normalize_pdbid(match.group(1)) if match else None


def read_csv_pdbids(path: Path, column: str) -> tuple[int, set[str]]:
    row_count = 0
    pdbids: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        if column not in fieldnames:
            raise ValueError(f"Required column {column!r} missing from: {path}")
        for row in reader:
            row_count += 1
            pdbid = extract_pdbid(row.get(column, ""))
            if pdbid is not None:
                pdbids.add(pdbid)
    return row_count, pdbids


def scan_structure_pdbids(directory: Path) -> tuple[int, set[str]]:
    file_count = 0
    pdbids: set[str] = set()
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in STRUCTURE_SUFFIXES:
            continue
        file_count += 1
        pdbid = extract_pdbid(path.name)
        if pdbid is not None:
            pdbids.add(pdbid)
    return file_count, pdbids


def require_split_dir(exact_dir: Path) -> None:
    if not exact_dir.is_dir():
        raise FileNotFoundError(f"Exact split directory not found: {exact_dir}")
    for split_name in ("train", "test"):
        split_dir = exact_dir / split_name
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        primary_csv = split_dir / PRIMARY_CSV_NAME
        if not primary_csv.is_file():
            raise FileNotFoundError(f"Missing primary CSV: {primary_csv}")


def write_metadata(exact_dir: Path, metadata: dict[str, object]) -> None:
    (exact_dir / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme_lines = [
        "# Metal Split PinMyMetal",
        "",
        "Available supported-structure projection of the original PinMyMetal train/test membership.",
        "",
        "This split preserves the source train/test side for available PDB IDs and may contain train/test PDB-ID overlap. "
        "The current DeepMzyme summary CSV does not carry the original PinMyMetal `residueid_ion` / `metalid` row "
        "identifiers, so this audit is at the PDB-ID/structure level. It is an exact/possibly-overlapped reference "
        "split, not the trusted final held-out split.",
        "",
        f"Primary CSV: `{PRIMARY_CSV_NAME}`",
        f"Exact train PDB IDs: `{metadata['n_exact_train_pdbids']}`",
        f"Exact test PDB IDs: `{metadata['n_exact_test_pdbids']}`",
        f"Exact train/test PDB-ID overlap: `{metadata['n_exact_overlap_pdbids']}`",
        "",
        "See `split_metadata.json` for the source-membership audit.",
    ]
    (exact_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    exact_dir = resolve_path(args.exact_dir)
    source_train = resolve_path(args.source_train_membership)
    source_test = resolve_path(args.source_test_membership)
    require_split_dir(exact_dir)
    if not source_train.is_file():
        raise FileNotFoundError(f"Source train membership file not found: {source_train}")
    if not source_test.is_file():
        raise FileNotFoundError(f"Source test membership file not found: {source_test}")

    source_train_rows, source_train_pdbids = read_csv_pdbids(source_train, "pdbid")
    source_test_rows, source_test_pdbids = read_csv_pdbids(source_test, "pdbid")
    primary_train_rows, primary_train_pdbids = read_csv_pdbids(exact_dir / "train" / PRIMARY_CSV_NAME, "structure")
    primary_test_rows, primary_test_pdbids = read_csv_pdbids(exact_dir / "test" / PRIMARY_CSV_NAME, "structure")
    exact_train_files, exact_train_pdbids = scan_structure_pdbids(exact_dir / "train")
    exact_test_files, exact_test_pdbids = scan_structure_pdbids(exact_dir / "test")

    metadata: dict[str, object] = {
        "split_name": "Metal Split PinMyMetal",
        "split_type": "metal_split_pinmymetal_possibly_overlapped",
        "split_description": (
            "Available supported-structure projection of the original PinMyMetal train/test membership. "
            "This exact split preserves the source train/test side for available PDB IDs and may contain "
            "train/test PDB-ID overlap."
        ),
        "primary_csv": PRIMARY_CSV_NAME,
        "source_train_membership_file": str(source_train),
        "source_test_membership_file": str(source_test),
        "n_source_train_rows": source_train_rows,
        "n_source_test_rows": source_test_rows,
        "n_source_train_pdbids": len(source_train_pdbids),
        "n_source_test_pdbids": len(source_test_pdbids),
        "n_source_overlap_pdbids": len(source_train_pdbids & source_test_pdbids),
        "n_primary_train_rows": primary_train_rows,
        "n_primary_test_rows": primary_test_rows,
        "n_primary_train_pdbids": len(primary_train_pdbids),
        "n_primary_test_pdbids": len(primary_test_pdbids),
        "n_primary_overlap_pdbids": len(primary_train_pdbids & primary_test_pdbids),
        "n_exact_train_files": exact_train_files,
        "n_exact_test_files": exact_test_files,
        "n_exact_train_pdbids": len(exact_train_pdbids),
        "n_exact_test_pdbids": len(exact_test_pdbids),
        "n_exact_overlap_pdbids": len(exact_train_pdbids & exact_test_pdbids),
        "exact_train_pdbids_absent_from_source_train": sorted(exact_train_pdbids - source_train_pdbids),
        "exact_test_pdbids_absent_from_source_test": sorted(exact_test_pdbids - source_test_pdbids),
        "source_train_pdbids_without_supported_exact_structure_count": len(source_train_pdbids - exact_train_pdbids),
        "source_test_pdbids_without_supported_exact_structure_count": len(source_test_pdbids - exact_test_pdbids),
        "overlap_warning": (
            "This split may contain train/test PDB-ID overlap and should be labeled as exact/possibly-overlapped "
            "in reports."
        ),
        "site_level_caveat": (
            "The DeepMzyme summary CSVs do not retain original PinMyMetal residueid_ion/metalid row identifiers; "
            "this exact split audit verifies available PDB-ID membership, not original site-row reconstruction."
        ),
    }
    write_metadata(exact_dir, metadata)

    print(f"Wrote exact split metadata to {exact_dir / 'split_metadata.json'}")
    print(f"Wrote exact split README to {exact_dir / 'README.md'}")
    print(f"Exact train/test PDB-ID overlap: {metadata['n_exact_overlap_pdbids']}")


if __name__ == "__main__":
    main()
