#!/usr/bin/env python3
"""Build the primary, PDB-disjoint PinMyMetal split from the exact split.

The exact test side is copied byte-for-byte. Every PDB ID present on that test
side is removed from the training structures and training summary rows. The
result is constructed and verified in a sibling staging directory, then
promoted atomically. Deterministic memberships and hashes make the derivation
auditable without tracking structure files in Git.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


LIKELY_PDBID_COLUMNS = ("pdbid", "pdb_id", "pdb", "structure_id", "structure")
STRUCTURE_SUFFIXES = {".pdb", ".cif", ".mmcif"}
PDBID_RE = re.compile(r"(?i)(?<![A-Za-z0-9])([0-9][A-Za-z0-9]{3})(?![A-Za-z0-9])")
PRIMARY_CSV = "final_data_summarazing_table_transition_metals_only_catalytic.csv"
METADATA_NAME = "split_metadata.json"
MANIFEST_SCHEMA_VERSION = 1

# Guard against silently rebuilding the named benchmark from a different local
# source. Custom inputs can opt out explicitly.
CURRENT_EXACT_PROFILE = {
    "source_train_structure_files": 1483,
    "source_train_pdbids": 1472,
    "source_test_structure_files": 316,
    "source_test_pdbids": 313,
    "source_overlap_pdbids": 177,
    "output_train_structure_files": 1304,
    "output_train_pdbids": 1295,
    "output_test_structure_files": 316,
    "output_test_pdbids": 313,
    "output_train_primary_rows": 1823,
    "output_test_primary_rows": 490,
}
EXPECTED_PRIMARY_METAL_COUNTS = {
    "train": {"CO": 100, "CU": 96, "FE": 374, "MN": 893, "NI": 90, "ZN": 270},
    "test": {"CO": 35, "CU": 42, "FE": 44, "MN": 135, "NI": 22, "ZN": 212},
}


@dataclass(frozen=True)
class ScanResult:
    files: tuple[Path, ...]
    pdbids: frozenset[str]
    pdbid_to_files: Mapping[str, tuple[Path, ...]]


@dataclass(frozen=True)
class CsvFilterResult:
    fieldnames: tuple[str, ...]
    pdbid_column: str
    kept_rows: tuple[dict[str, str], ...]
    removed_rows: int
    source_rows: int
    kept_pdbids: frozenset[str]


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "DeepMzyme_Data").exists() and (candidate / "prepare_training_and_test_set").exists():
            return candidate
    raise FileNotFoundError("Could not locate the project root from this script path.")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
DEFAULT_BASE_DIR = PROJECT_ROOT / "DeepMzyme_Data" / "train_and_test_sets_structures_exact_pinmymetal"
DEFAULT_TRAIN_DIR = DEFAULT_BASE_DIR / "train"
DEFAULT_TEST_DIR = DEFAULT_BASE_DIR / "test"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "DeepMzyme_Data" / "train_and_test_sets_structures_non_overlapped_pinmymetal"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the primary PDB-disjoint PinMyMetal split: preserve the exact test side and remove "
            "every exact-test PDB ID from the exact training side."
        )
    )
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-csv", type=Path, default=None, help="Optional single train CSV; defaults to all train CSVs.")
    parser.add_argument("--test-csv", type=Path, default=None, help="Optional single test CSV; defaults to all test CSVs.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing output only after the staged replacement passes all checks. "
            "The previous output is retained in a sibling '.previous' directory."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and report the derivation without writing files.")
    parser.add_argument(
        "--no-enforce-current-exact-profile",
        action="store_true",
        help="Allow intentional use of a source other than the currently audited exact PinMyMetal projection.",
    )
    return parser.parse_args(argv)


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def normalize_pdbid(value: str) -> str:
    return str(value).strip().lower()


def extract_pdbid(text: str) -> str | None:
    match = PDBID_RE.search(str(text))
    return normalize_pdbid(match.group(1)) if match is not None else None


def ensure_input_dir(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def membership_text(values: Iterable[str]) -> str:
    return "".join(f"{item}\n" for item in sorted({normalize_pdbid(value) for value in values}))


def tree_manifest(directory: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(directory).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(candidate for candidate in directory.rglob("*") if candidate.is_file())
    ]


def tree_sha256(entries: Sequence[Mapping[str, object]]) -> str:
    canonical = json.dumps(list(entries), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return sha256_bytes(canonical)


def scan_structure_dir(directory: Path) -> ScanResult:
    files = tuple(sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in STRUCTURE_SUFFIXES))
    if not files:
        raise ValueError(f"No supported structure files found in {directory}")
    unknown = [path.name for path in files if extract_pdbid(path.name) is None]
    if unknown:
        raise ValueError(
            f"Could not extract a PDB ID from {len(unknown)} structure filename(s) in {directory}: "
            f"{', '.join(unknown[:10])}. Refusing to skip ambiguous structures."
        )
    mutable: dict[str, list[Path]] = {}
    for path in files:
        pdbid = extract_pdbid(path.name)
        assert pdbid is not None
        mutable.setdefault(pdbid, []).append(path)
    mapping = {pdbid: tuple(sorted(paths)) for pdbid, paths in sorted(mutable.items())}
    return ScanResult(files=files, pdbids=frozenset(mapping), pdbid_to_files=mapping)


def autodetect_csv_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".csv")


def resolve_csv_inputs(directory: Path, specific_csv: Path | None) -> list[Path]:
    paths = autodetect_csv_files(directory) if specific_csv is None else [resolve_path(specific_csv)]
    if not paths:
        raise FileNotFoundError(f"No CSV summary files found in {directory}")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"CSV file(s) not found: {', '.join(missing)}")
    return paths


def detect_pdbid_column(fieldnames: Sequence[str] | None) -> str:
    if not fieldnames:
        raise ValueError("CSV has no header.")
    lowered = {name.strip().lower(): name for name in fieldnames}
    for candidate in LIKELY_PDBID_COLUMNS:
        if candidate in lowered:
            return lowered[candidate]
    raise ValueError("CSV has no recognized PDB-ID/structure column. Expected one of: " + ", ".join(LIKELY_PDBID_COLUMNS))


def analyze_train_csv(source_csv: Path, test_pdbids: set[str]) -> CsvFilterResult:
    with source_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        pdbid_column = detect_pdbid_column(fieldnames)
        kept_rows: list[dict[str, str]] = []
        kept_pdbids: set[str] = set()
        removed_rows = 0
        source_rows = 0
        for row_number, row in enumerate(reader, start=2):
            source_rows += 1
            pdbid = extract_pdbid(row.get(pdbid_column, ""))
            if pdbid is None:
                raise ValueError(
                    f"Could not extract a PDB ID from {source_csv}:{row_number} column {pdbid_column!r}; "
                    "refusing to keep an ambiguous row."
                )
            if pdbid in test_pdbids:
                removed_rows += 1
            else:
                kept_rows.append(row)
                kept_pdbids.add(pdbid)
    return CsvFilterResult(
        fieldnames=fieldnames,
        pdbid_column=pdbid_column,
        kept_rows=tuple(kept_rows),
        removed_rows=removed_rows,
        source_rows=source_rows,
        kept_pdbids=frozenset(kept_pdbids),
    )


def analyze_csv_membership(source_csv: Path) -> tuple[int, frozenset[str], str]:
    with source_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        pdbid_column = detect_pdbid_column(tuple(reader.fieldnames or ()))
        pdbids: set[str] = set()
        rows = 0
        for row_number, row in enumerate(reader, start=2):
            rows += 1
            pdbid = extract_pdbid(row.get(pdbid_column, ""))
            if pdbid is None:
                raise ValueError(f"Could not extract a PDB ID from {source_csv}:{row_number}.")
            pdbids.add(pdbid)
    return rows, frozenset(pdbids), pdbid_column


def metal_counts_from_rows(rows: Iterable[Mapping[str, str]], *, source: Path) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row_number, row in enumerate(rows, start=2):
        raw_value = row.get("metaltype")
        if raw_value is None or not raw_value.strip():
            raise ValueError(f"Missing metaltype in canonical CSV {source}:{row_number}")
        counter[raw_value.strip().upper()] += 1
    return dict(sorted(counter.items()))


def read_csv_rows(source_csv: Path) -> tuple[dict[str, str], ...]:
    with source_csv.open("r", encoding="utf-8", newline="") as handle:
        return tuple(csv.DictReader(handle))


def copy_structure_files(paths: Iterable[Path], destination_dir: Path) -> int:
    count = 0
    for source_path in paths:
        shutil.copy2(source_path, destination_dir / source_path.name)
        count += 1
    return count


def write_filtered_csv(result: CsvFilterResult, destination: Path) -> None:
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result.fieldnames)
        writer.writeheader()
        writer.writerows(result.kept_rows)


def verify_profile(actual: Mapping[str, int], *, enforce: bool) -> None:
    mismatches = {
        key: {"expected": expected, "actual": actual.get(key)}
        for key, expected in CURRENT_EXACT_PROFILE.items()
        if actual.get(key) != expected
    }
    if mismatches and enforce:
        raise RuntimeError(
            "The local exact split does not match the audited PinMyMetal profile: "
            + json.dumps(mismatches, sort_keys=True)
            + ". Use --no-enforce-current-exact-profile only for an intentionally different derivation."
        )


def write_text(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8", newline="\n")


def build_readme(metadata: Mapping[str, object]) -> str:
    counts = metadata["counts"]
    hashes = metadata["hashes"]
    assert isinstance(counts, Mapping)
    assert isinstance(hashes, Mapping)
    return f"""# Non-overlapped PinMyMetal split

This is the primary PDB-disjoint metal benchmark derived from the locally
materialized exact PinMyMetal projection. The test directory is a byte-identical
copy of the exact split test directory. Every exact-test PDB ID was removed from
the training structures and from every training summary CSV.

- Train: {counts['output_train_structure_files']} structures, {counts['output_train_pdbids']} PDB IDs, {counts['output_train_primary_rows']} canonical site rows.
- Test: {counts['output_test_structure_files']} structures, {counts['output_test_pdbids']} PDB IDs, {counts['output_test_primary_rows']} canonical site rows.
- Removed from train: {counts['source_overlap_pdbids']} exact-test PDB IDs.
- Final train/test PDB-ID overlap: {counts['output_overlap_pdbids']}.
- Test tree SHA-256: `{hashes['output_test_tree_sha256']}`.

`split_metadata.json` is the machine-readable authority for construction,
membership hashes, class counts, and source/test equivalence. This test
membership is the same as the exact PinMyMetal secondary-reference route; the
two routes are paired views, not independent test sets.
"""


def build_split(
    *,
    train_dir: Path,
    test_dir: Path,
    output_dir: Path,
    train_csv: Path | None = None,
    test_csv: Path | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    enforce_current_exact_profile: bool = True,
) -> dict[str, object]:
    train_dir = resolve_path(train_dir)
    test_dir = resolve_path(test_dir)
    output_dir = resolve_path(output_dir)
    ensure_input_dir(train_dir, "Train structure directory")
    ensure_input_dir(test_dir, "Test structure directory")
    if output_dir in {train_dir, test_dir, train_dir.parent, test_dir.parent}:
        raise ValueError("Output directory must be separate from the source exact split.")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {output_dir}. Refusing to modify it without --overwrite.")

    train_scan = scan_structure_dir(train_dir)
    test_scan = scan_structure_dir(test_dir)
    source_train_pdbids = set(train_scan.pdbids)
    source_test_pdbids = set(test_scan.pdbids)
    overlap_pdbids = source_train_pdbids & source_test_pdbids
    output_train_pdbids = source_train_pdbids - source_test_pdbids

    train_csvs = resolve_csv_inputs(train_dir, train_csv)
    test_csvs = resolve_csv_inputs(test_dir, test_csv)
    train_names = [path.name for path in train_csvs]
    test_names = [path.name for path in test_csvs]
    if len(train_names) != len(set(train_names)) or len(test_names) != len(set(test_names)):
        raise ValueError("CSV basenames must be unique on each split side.")
    if PRIMARY_CSV not in train_names or PRIMARY_CSV not in test_names:
        raise FileNotFoundError(f"Both split sides must contain canonical CSV {PRIMARY_CSV!r}.")

    train_analyses = {path.name: analyze_train_csv(path, source_test_pdbids) for path in train_csvs}
    test_memberships = {path.name: analyze_csv_membership(path) for path in test_csvs}
    for name, analysis in train_analyses.items():
        missing_structures = set(analysis.kept_pdbids) - output_train_pdbids
        if missing_structures:
            raise RuntimeError(f"Filtered train CSV {name} references PDB IDs absent from train structures: {sorted(missing_structures)[:10]}")
    for name, (_rows, pdbids, _column) in test_memberships.items():
        missing_structures = set(pdbids) - source_test_pdbids
        if missing_structures:
            raise RuntimeError(f"Test CSV {name} references PDB IDs absent from test structures: {sorted(missing_structures)[:10]}")

    primary_train_analysis = train_analyses[PRIMARY_CSV]
    primary_test_path = next(path for path in test_csvs if path.name == PRIMARY_CSV)
    primary_test_rows = read_csv_rows(primary_test_path)
    train_metal_counts = metal_counts_from_rows(
        primary_train_analysis.kept_rows,
        source=next(path for path in train_csvs if path.name == PRIMARY_CSV),
    )
    test_metal_counts = metal_counts_from_rows(primary_test_rows, source=primary_test_path)

    selected_train_files = tuple(
        path
        for pdbid, paths in sorted(train_scan.pdbid_to_files.items())
        if pdbid in output_train_pdbids
        for path in paths
    )
    selected_test_files = tuple(path for _pdbid, paths in sorted(test_scan.pdbid_to_files.items()) for path in paths)
    profile = {
        "source_train_structure_files": len(train_scan.files),
        "source_train_pdbids": len(source_train_pdbids),
        "source_test_structure_files": len(test_scan.files),
        "source_test_pdbids": len(source_test_pdbids),
        "source_overlap_pdbids": len(overlap_pdbids),
        "output_train_structure_files": len(selected_train_files),
        "output_train_pdbids": len(output_train_pdbids),
        "output_test_structure_files": len(selected_test_files),
        "output_test_pdbids": len(source_test_pdbids),
        "output_train_primary_rows": len(primary_train_analysis.kept_rows),
        "output_test_primary_rows": len(primary_test_rows),
    }
    verify_profile(profile, enforce=enforce_current_exact_profile)
    if enforce_current_exact_profile:
        actual_metal_counts = {"train": train_metal_counts, "test": test_metal_counts}
        if actual_metal_counts != EXPECTED_PRIMARY_METAL_COUNTS:
            raise RuntimeError(
                "Canonical metal counts differ from the audited profile: "
                + json.dumps({"expected": EXPECTED_PRIMARY_METAL_COUNTS, "actual": actual_metal_counts}, sort_keys=True)
            )

    membership_payloads = {
        "train_pdbids.txt": membership_text(output_train_pdbids),
        "test_pdbids.txt": membership_text(source_test_pdbids),
        "removed_exact_test_pdbids_from_train.txt": membership_text(overlap_pdbids),
        "source_exact_overlap_pdbids.txt": membership_text(overlap_pdbids),
    }
    source_test_manifest = tree_manifest(test_dir)
    source_metadata_path = train_dir.parent / METADATA_NAME
    if not source_metadata_path.is_file():
        raise FileNotFoundError(f"Exact split metadata is required: {source_metadata_path}")

    metadata: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "split_name": "Non-overlapped PinMyMetal",
        "split_type": "non_overlapped_pinmymetal",
        "evaluation_role": "primary_final_report",
        "primary_csv": PRIMARY_CSV,
        "source_split_type": "metal_split_pinmymetal_possibly_overlapped",
        "source_exact_root": project_relative(train_dir.parent),
        "construction_program": project_relative(Path(__file__)),
        "construction_rule": "copy exact test byte-for-byte; remove every exact-test PDB ID from exact train structures and rows",
        "test_membership_relationship": "same PDB-ID membership and byte-identical files as the exact PinMyMetal secondary-reference test",
        "counts": {**profile, "output_overlap_pdbids": 0},
        "canonical_metal_class_counts": {"train": train_metal_counts, "test": test_metal_counts},
        "csvs": {
            "train": {
                name: {
                    "source_rows": analysis.source_rows,
                    "kept_rows": len(analysis.kept_rows),
                    "removed_rows": analysis.removed_rows,
                    "pdbid_column": analysis.pdbid_column,
                }
                for name, analysis in sorted(train_analyses.items())
            },
            "test": {
                name: {"rows": rows, "pdbid_column": column, "copied_byte_identically": True}
                for name, (rows, _pdbids, column) in sorted(test_memberships.items())
            },
        },
        "membership_files": {
            name: {"count": len(payload.splitlines()), "sha256": sha256_bytes(payload.encode("utf-8"))}
            for name, payload in sorted(membership_payloads.items())
        },
        "hashes": {
            "source_exact_split_metadata_sha256": sha256_file(source_metadata_path),
            "source_exact_test_tree_sha256": tree_sha256(source_test_manifest),
            "output_test_tree_sha256": tree_sha256(source_test_manifest),
            "source_exact_test_tree_manifest": source_test_manifest,
        },
        "validation": {
            "current_exact_profile_enforced": enforce_current_exact_profile,
            "final_pdbid_overlap_is_zero": True,
            "test_tree_byte_identical_to_exact": True,
            "unknown_structure_filenames": 0,
            "unknown_csv_rows": 0,
        },
    }

    if dry_run:
        return metadata

    staging_dir = output_dir.with_name(f".{output_dir.name}.staging")
    previous_dir = output_dir.with_name(f"{output_dir.name}.previous")
    if staging_dir.exists():
        raise FileExistsError(f"Staging directory already exists; inspect and remove it before retrying: {staging_dir}")
    if overwrite and output_dir.exists() and previous_dir.exists():
        raise FileExistsError(f"Recovery directory already exists; refusing to overwrite it: {previous_dir}")

    try:
        train_out = staging_dir / "train"
        test_out = staging_dir / "test"
        train_out.mkdir(parents=True, exist_ok=False)
        test_out.mkdir(parents=True, exist_ok=False)
        copy_structure_files(selected_train_files, train_out)
        copy_structure_files(selected_test_files, test_out)
        for source in train_csvs:
            write_filtered_csv(train_analyses[source.name], train_out / source.name)
        for source in test_csvs:
            shutil.copy2(source, test_out / source.name)
        for name, payload in membership_payloads.items():
            write_text(staging_dir / name, payload)

        final_train_scan = scan_structure_dir(train_out)
        final_test_scan = scan_structure_dir(test_out)
        final_overlap = set(final_train_scan.pdbids) & set(final_test_scan.pdbids)
        if final_overlap:
            raise RuntimeError(f"Staged split still has {len(final_overlap)} shared PDB IDs.")
        if set(final_train_scan.pdbids) != output_train_pdbids:
            raise RuntimeError("Staged train membership differs from the computed membership.")
        if set(final_test_scan.pdbids) != source_test_pdbids:
            raise RuntimeError("Staged test membership differs from the exact test membership.")
        output_test_manifest = tree_manifest(test_out)
        if output_test_manifest != source_test_manifest:
            raise RuntimeError("Staged test directory is not byte-identical to the exact test directory.")
        metadata["hashes"]["output_test_tree_sha256"] = tree_sha256(output_test_manifest)  # type: ignore[index]
        write_text(staging_dir / METADATA_NAME, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        write_text(staging_dir / "README.md", build_readme(metadata))

        if output_dir.exists():
            output_dir.rename(previous_dir)
        try:
            staging_dir.rename(output_dir)
        except BaseException:
            if previous_dir.exists() and not output_dir.exists():
                previous_dir.rename(output_dir)
            raise
    except BaseException:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise

    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    metadata = build_split(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        enforce_current_exact_profile=not args.no_enforce_current_exact_profile,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    if args.dry_run:
        print("[DRY RUN] No files were written.")
    else:
        print(f"Created verified non-overlap split: {resolve_path(args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
