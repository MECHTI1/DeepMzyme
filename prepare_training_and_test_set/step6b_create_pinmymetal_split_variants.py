#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import filecmp
import json
import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PDBID_RE = re.compile(r"(?i)(?<![A-Za-z0-9])([0-9][A-Za-z0-9]{3})(?![A-Za-z0-9])")
STRUCTURE_SUFFIXES = {".pdb", ".cif", ".mmcif"}
LIKELY_PDBID_COLUMNS = (
    "pdbid",
    "pdb_id",
    "pdb",
    "structure",
    "structure_id",
    "PDB",
)


@dataclass(frozen=True)
class StructureScan:
    files: list[Path]
    pdbids: set[str]
    pdbid_to_files: dict[str, list[Path]]
    unknown_files: list[Path]


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "DeepMzyme_Data").exists() and (candidate / "prepare_training_and_test_set").exists():
            return candidate
    raise FileNotFoundError("Could not locate the project root from the current working directory.")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
DEFAULT_EXACT_DIR = PROJECT_ROOT / "DeepMzyme_Data" / "train_and_test_sets_structures_exact_pinmymetal"
DEFAULT_HARSH_DIR = PROJECT_ROOT / "DeepMzyme_Data" / "train_and_test_sets_structures_harsh_pinmymetal"
DEFAULT_COMMON_70_30_DIR = PROJECT_ROOT / "DeepMzyme_Data" / "train_and_test_sets_structures_common_pdbid_70_30_pinmymetal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create named PinMyMetal split variants from the exact PinMyMetal structure split. "
            "The harsh mode keeps train-only PDB IDs in train and assigns every shared/common "
            "PDB ID, including exact-train structures for that PDB ID, to test. "
            "The common-pdbid-70-30 mode keeps train-only PDB IDs in train and test-only PDB IDs "
            "in test, then assigns shared/common PDB IDs 70% to train and 30% to test."
        )
    )
    parser.add_argument("--exact-dir", type=Path, default=DEFAULT_EXACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=("harsh", "common-pdbid-70-30"),
        default="common-pdbid-70-30",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--test-common-fraction",
        type=float,
        default=0.3,
        help="Fraction of common exact-split PDB IDs assigned to test in common-pdbid-70-30 mode.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="Use hardlinks by default to avoid duplicating large structure files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow removing an existing output directory before recreating it.",
    )
    return parser.parse_args()


def default_output_dir_for_mode(mode: str) -> Path:
    if mode == "harsh":
        return DEFAULT_HARSH_DIR
    if mode == "common-pdbid-70-30":
        return DEFAULT_COMMON_70_30_DIR
    raise AssertionError(f"Unhandled split mode: {mode!r}")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def normalize_pdbid(value: str) -> str:
    return str(value).strip().lower()


def extract_pdbid(text: str) -> str | None:
    match = PDBID_RE.search(str(text))
    return normalize_pdbid(match.group(1)) if match else None


def ensure_split_dir(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")
    for split_name in ("train", "test"):
        split_path = path / split_name
        if not split_path.is_dir():
            raise FileNotFoundError(f"{label} is missing {split_name!r} directory: {split_path}")


def scan_structure_dir(directory: Path) -> StructureScan:
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in STRUCTURE_SUFFIXES
    )
    pdbids: set[str] = set()
    pdbid_to_files: dict[str, list[Path]] = {}
    unknown_files: list[Path] = []
    for path in files:
        pdbid = extract_pdbid(path.name)
        if pdbid is None:
            unknown_files.append(path)
            continue
        pdbids.add(pdbid)
        pdbid_to_files.setdefault(pdbid, []).append(path)
    return StructureScan(files=files, pdbids=pdbids, pdbid_to_files=pdbid_to_files, unknown_files=unknown_files)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> tuple[Path, Path]:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Re-run with --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    train_out = output_dir / "train"
    test_out = output_dir / "test"
    train_out.mkdir(parents=True)
    test_out.mkdir(parents=True)
    return train_out, test_out


def choose_common_pdbids(
    common_pdbids: set[str],
    *,
    seed: int,
    test_fraction: float,
) -> tuple[set[str], set[str]]:
    if not 0.0 <= test_fraction <= 1.0:
        raise ValueError(f"--test-common-fraction must be in [0, 1], got {test_fraction}")
    ordered = sorted(common_pdbids)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n_test = int(round(len(ordered) * test_fraction))
    test_common = set(ordered[:n_test])
    train_common = set(ordered[n_test:])
    return train_common, test_common


def link_or_copy_file(source_path: Path, dest_path: Path, *, link_mode: str) -> bool:
    if dest_path.exists():
        if filecmp.cmp(source_path, dest_path, shallow=False):
            return False
        raise FileExistsError(f"Conflicting files with the same output name: {source_path} -> {dest_path}")

    if link_mode == "hardlink":
        try:
            os.link(source_path, dest_path)
            return True
        except OSError:
            shutil.copy2(source_path, dest_path)
            return True

    shutil.copy2(source_path, dest_path)
    return True


def copy_assigned_structures(
    source_dirs: Iterable[Path],
    *,
    train_pdbids: set[str],
    test_pdbids: set[str],
    train_out: Path,
    test_out: Path,
    link_mode: str,
) -> tuple[int, int, int]:
    copied_train = 0
    copied_test = 0
    skipped_unknown = 0
    for source_dir in source_dirs:
        for source_path in sorted(source_dir.iterdir()):
            if not source_path.is_file() or source_path.suffix.lower() not in STRUCTURE_SUFFIXES:
                continue
            pdbid = extract_pdbid(source_path.name)
            if pdbid is None:
                skipped_unknown += 1
                continue
            if pdbid in train_pdbids:
                copied_train += int(link_or_copy_file(source_path, train_out / source_path.name, link_mode=link_mode))
            elif pdbid in test_pdbids:
                copied_test += int(link_or_copy_file(source_path, test_out / source_path.name, link_mode=link_mode))
            else:
                raise RuntimeError(f"PDB ID {pdbid!r} has no assignment for {source_path}")
    return copied_train, copied_test, skipped_unknown


def autodetect_csv_files(*directories: Path) -> dict[str, list[Path]]:
    files_by_name: dict[str, list[Path]] = {}
    for directory in directories:
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() == ".csv":
                files_by_name.setdefault(path.name, []).append(path)
    return files_by_name


def detect_pdbid_column(fieldnames: list[str] | None) -> str | None:
    if fieldnames is None:
        return None
    lowered = {name.strip().lower(): name for name in fieldnames}
    for candidate in LIKELY_PDBID_COLUMNS:
        key = candidate.strip().lower()
        if key in lowered:
            return lowered[key]
    return None


def merge_assigned_csv(
    source_csvs: list[Path],
    dest_train_csv: Path,
    dest_test_csv: Path,
    *,
    train_pdbids: set[str],
    test_pdbids: set[str],
) -> dict[str, int | str | None]:
    fieldnames: list[str] | None = None
    rows_train: list[dict[str, str]] = []
    rows_test: list[dict[str, str]] = []
    seen_train: set[tuple[str, ...]] = set()
    seen_test: set[tuple[str, ...]] = set()
    skipped_unknown = 0
    skipped_duplicate_train = 0
    skipped_duplicate_test = 0
    pdbid_column: str | None = None

    for source_csv in source_csvs:
        with source_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
                pdbid_column = detect_pdbid_column(fieldnames)
                if pdbid_column is None:
                    raise ValueError(f"Could not detect a PDB ID column in {source_csv}")
            elif list(reader.fieldnames or []) != fieldnames:
                raise ValueError(f"CSV schema mismatch for {source_csv}")

            assert fieldnames is not None
            assert pdbid_column is not None
            for row in reader:
                pdbid = extract_pdbid(row.get(pdbid_column, ""))
                if pdbid is None:
                    skipped_unknown += 1
                    continue
                row_key = tuple(row.get(name, "") for name in fieldnames)
                if pdbid in train_pdbids:
                    if row_key in seen_train:
                        skipped_duplicate_train += 1
                        continue
                    rows_train.append(row)
                    seen_train.add(row_key)
                elif pdbid in test_pdbids:
                    if row_key in seen_test:
                        skipped_duplicate_test += 1
                        continue
                    rows_test.append(row)
                    seen_test.add(row_key)
                else:
                    raise RuntimeError(f"PDB ID {pdbid!r} has no assignment for row in {source_csv}")

    if fieldnames is None:
        raise ValueError("No CSV files were provided.")

    for dest_csv, rows in ((dest_train_csv, rows_train), (dest_test_csv, rows_test)):
        with dest_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return {
        "pdbid_column": pdbid_column,
        "train_rows": len(rows_train),
        "test_rows": len(rows_test),
        "skipped_unknown_rows": skipped_unknown,
        "skipped_duplicate_train_rows": skipped_duplicate_train,
        "skipped_duplicate_test_rows": skipped_duplicate_test,
    }


def write_metadata(output_dir: Path, payload: dict[str, object]) -> None:
    (output_dir / "split_metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme_lines = [
        f"# {payload['split_name']}",
        "",
        "Generated from `train_and_test_sets_structures_exact_pinmymetal`.",
        "",
        str(payload["split_description"]),
        "",
        "This split is a custom comparison split, not the current trusted final held-out",
        "split defined in `Plan.md`.",
        "",
        f"Seed: `{payload['seed']}`",
        f"Assignment scope: `{payload['assignment_scope']}`",
        f"Test common-PDB-ID fraction: `{payload['test_common_fraction']}`",
        f"Final PDB IDs assigned to train: `{payload['n_assigned_pdbids_train']}`",
        f"Final PDB IDs assigned to test: `{payload['n_assigned_pdbids_test']}`",
        f"Common exact-split PDB IDs assigned to train: `{payload['n_common_pdbids_train']}`",
        f"Common exact-split PDB IDs assigned to test: `{payload['n_common_pdbids_test']}`",
        f"Final train/test PDB-ID overlap: `{payload['final_overlap_pdbids']}`",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")


def format_preview(values: Iterable[str], limit: int = 30) -> str:
    items = sorted(set(values))
    if not items:
        return "(none)"
    suffix = "" if len(items) <= limit else f" ... (+{len(items) - limit} more)"
    return ", ".join(items[:limit]) + suffix


def main() -> None:
    args = parse_args()
    exact_dir = resolve_path(args.exact_dir)
    output_dir = resolve_path(args.output_dir or default_output_dir_for_mode(args.mode))
    ensure_split_dir(exact_dir, "Exact PinMyMetal split directory")

    train_dir = exact_dir / "train"
    test_dir = exact_dir / "test"
    train_scan = scan_structure_dir(train_dir)
    test_scan = scan_structure_dir(test_dir)
    common_pdbids = train_scan.pdbids & test_scan.pdbids

    if args.mode == "harsh":
        train_common: set[str] = set()
        test_common = set(common_pdbids)
        split_name = "Harsh Split PinMyMetal"
        split_type = "harsh_pinmymetal"
        assignment_scope = "common_exact_split_pdbids"
        split_description = (
            "Train-only PDB IDs stay in train, test-only PDB IDs stay in test, and every "
            "PDB ID that appears in both exact train and exact test is assigned as a whole "
            "PDB-ID group to test. This moves exact-train structures/rows for shared PDB IDs "
            "into test instead of dropping them."
        )
    elif args.mode == "common-pdbid-70-30":
        train_common, test_common = choose_common_pdbids(
            common_pdbids,
            seed=args.seed,
            test_fraction=args.test_common_fraction,
        )
        split_name = "Common-PDBID 70/30 Split PinMyMetal"
        split_type = "common_pdbid_70_30_pinmymetal"
        assignment_scope = "common_exact_split_pdbids"
        split_description = (
            "Train-only PDB IDs stay in train, test-only PDB IDs stay in test, and PDB IDs "
            "that appear in both exact train and exact test are assigned as whole PDB-ID "
            "groups: 70% of common PDB IDs to train and 30% to test."
        )
    else:
        raise AssertionError(f"Unhandled split mode: {args.mode!r}")

    train_pdbids = (train_scan.pdbids - common_pdbids) | train_common
    test_pdbids = (test_scan.pdbids - common_pdbids) | test_common
    if train_pdbids & test_pdbids:
        raise RuntimeError("Internal split assignment error: train/test PDB IDs overlap.")

    train_out, test_out = prepare_output_dir(output_dir, overwrite=args.overwrite)
    copied_train, copied_test, skipped_unknown_structures = copy_assigned_structures(
        (train_dir, test_dir),
        train_pdbids=train_pdbids,
        test_pdbids=test_pdbids,
        train_out=train_out,
        test_out=test_out,
        link_mode=args.link_mode,
    )

    csv_reports: dict[str, object] = {}
    for filename, source_csvs in autodetect_csv_files(train_dir, test_dir).items():
        csv_reports[filename] = merge_assigned_csv(
            source_csvs,
            train_out / filename,
            test_out / filename,
            train_pdbids=train_pdbids,
            test_pdbids=test_pdbids,
        )

    final_train_scan = scan_structure_dir(train_out)
    final_test_scan = scan_structure_dir(test_out)
    final_overlap = final_train_scan.pdbids & final_test_scan.pdbids
    if final_overlap:
        raise RuntimeError(f"Final train/test PDB-ID overlap is not zero: {format_preview(final_overlap)}")

    metadata: dict[str, object] = {
        "split_name": split_name,
        "split_type": split_type,
        "split_description": split_description,
        "source_split": str(exact_dir),
        "mode": args.mode,
        "assignment_scope": assignment_scope,
        "seed": args.seed,
        "test_common_fraction": args.test_common_fraction,
        "link_mode": args.link_mode,
        "n_exact_train_files": len(train_scan.files),
        "n_exact_test_files": len(test_scan.files),
        "n_exact_train_pdbids": len(train_scan.pdbids),
        "n_exact_test_pdbids": len(test_scan.pdbids),
        "n_common_pdbids": len(common_pdbids),
        "n_common_pdbids_train": len(train_common),
        "n_common_pdbids_test": len(test_common),
        "n_assigned_pdbids_train": len(train_pdbids),
        "n_assigned_pdbids_test": len(test_pdbids),
        "n_final_train_files": len(final_train_scan.files),
        "n_final_test_files": len(final_test_scan.files),
        "n_final_train_pdbids": len(final_train_scan.pdbids),
        "n_final_test_pdbids": len(final_test_scan.pdbids),
        "final_overlap_pdbids": len(final_overlap),
        "copied_or_linked_train_files": copied_train,
        "copied_or_linked_test_files": copied_test,
        "skipped_unknown_structure_files": skipped_unknown_structures,
        "train_common_pdbid_preview": sorted(train_common)[:30],
        "test_common_pdbid_preview": sorted(test_common)[:30],
        "csv_reports": csv_reports,
    }
    write_metadata(output_dir, metadata)

    print(f"Created {split_name}")
    print(f"Output directory: {output_dir}")
    print(f"Assignment scope: {assignment_scope}")
    print(f"Exact train/test PDB-ID overlap: {len(common_pdbids)}")
    print(f"PDB IDs assigned to train: {len(train_pdbids)}")
    print(f"PDB IDs assigned to test: {len(test_pdbids)}")
    print(f"Common PDB IDs assigned to train: {len(train_common)}")
    print(f"Common PDB IDs assigned to test: {len(test_common)}")
    print(f"Final train files: {len(final_train_scan.files)}")
    print(f"Final test files: {len(final_test_scan.files)}")
    print(f"Final train/test PDB-ID overlap: {len(final_overlap)}")


if __name__ == "__main__":
    main()
